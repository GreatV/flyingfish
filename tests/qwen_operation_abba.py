"""Frozen 36-cell gate; patch span is projection+bias+position after packing, 2026-10-20."""
from pathlib import Path
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys

T95_DF15 = 2.131449545559323
LIMIT = 1.02
GDN = [('recurrent', 1, 1), ('legacy', 2, 2), ('legacy', 5, 5),
       ('verify', 2, 2), ('verify', 5, 5), ('verify', 5, 2),
       ('legacy', 64, 64), ('legacy', 512, 512), ('legacy', 512, 509),
       ('legacy', 4096, 4096), ('p2', 64, 64), ('p2', 512, 512),
       ('p2', 512, 509), ('p2', 4096, 4096)]
EXPECTED = {(op, mode, rows, live, graph)
            for op, cells in [('gdn', GDN), ('patch', [('projection_bias_position', n, n)
                                                     for n in [1, 37, 1120, 7104]])]
            for mode, rows, live in cells for graph in [False, True]}
TESTS = {'gdn': 'backend::cuda::gdn::tests::gdn_operation_timing_gpu',
         'patch': 'backend::cuda::engine::vision_runtime::tests::vision_patch_operation_timing_gpu'}


def interval(values):
    assert len(values) == 16 and all(math.isfinite(x) for x in values)
    mean = statistics.mean(values)
    width = T95_DF15 * statistics.stdev(values) / math.sqrt(16)
    lo, hi = math.exp(mean - width), math.exp(mean + width)
    return dict(ratio=math.exp(mean), ci95=[lo, hi],
                verdict='PASS' if hi <= LIMIT else 'DEFECT' if lo > LIMIT else 'INCONCLUSIVE')


def controls():
    assert interval([0.0] * 16)['verdict'] == 'PASS'
    assert interval([math.log(1.05)] * 16)['verdict'] == 'DEFECT'
    assert interval([math.log(0.90), math.log(1.10)] * 8)['verdict'] == 'INCONCLUSIVE'
    assert len(EXPECTED) == 36
    print('T20_OPERATOR_SCORER_CONTROLS PASS equal=PASS slow=DEFECT noisy=INCONCLUSIVE cells=36', flush=True)


def read_arm(directory, arm):
    records = {}
    for label in TESTS:
        test = directory / label
        assert (test / 'rc').read_text().strip() == '0', (test, 'runner rc')
        raw = (test / 'raw.log').read_text()
        assert 'test result: ok. 1 passed; 0 failed; 0 ignored;' in raw, (test, 'actual test count')
        for line in raw.splitlines():
            if 'T20_OP_SAMPLE ' not in line:
                continue
            row = json.loads(line.split('T20_OP_SAMPLE ', 1)[1])
            key = tuple(row[k] for k in ['operation', 'mode', 'rows', 'valid', 'graph'])
            assert key in EXPECTED and key not in records, (test, key, 'cell inventory')
            assert row['warm'] == 16 and row['calls'] == 32 and row['native_weights'] is True, row
            value = row['mean_ms']
            assert math.isfinite(value) and value > 0, row
            if label == 'patch':
                assert row['bias_mode'] == ('BeforeOutputRound' if arm == 'A' else 'AfterOutputRound'), row
            records[key] = value
    assert set(records) == EXPECTED, (directory, 'missing cells')
    return records


def score(root):
    by_cell = {cell: [] for cell in EXPECTED}
    raw = []
    for block in range(1, 18):
        samples = [read_arm(root / f'block-{block:02d}' / f'{i}-{arm}', arm)
                   for i, arm in enumerate('ABBA', 1)]
        for cell in sorted(EXPECTED):
            a1, b1, b2, a2 = (sample[cell] for sample in samples)
            ratio = (math.log(b1) + math.log(b2) - math.log(a1) - math.log(a2)) / 2
            raw.append(dict(block=block, cell=cell, A1_ms=a1, B1_ms=b1, B2_ms=b2,
                            A2_ms=a2, log_ratio=ratio, retained=block != 1))
            if block != 1:
                by_cell[cell].append(ratio)
    rows = [dict(cell=cell, **interval(by_cell[cell]), retained_blocks=16) for cell in sorted(EXPECTED)]
    for row in rows:
        print('T20_OPERATOR_ABBA_CELL ' + json.dumps(row), flush=True)
    counts = {verdict: sum(row['verdict'] == verdict for row in rows)
              for verdict in ['PASS', 'DEFECT', 'INCONCLUSIVE']}
    final = 'PASS' if counts['PASS'] == 36 else 'DEFECT' if counts['DEFECT'] else 'INCONCLUSIVE'
    (root / 'RESULTS.json').write_text(json.dumps(dict(verdict=final, counts=counts, cells=rows), indent=2) + '\n')
    (root / 'raw-samples.json').write_text(json.dumps(raw, indent=2) + '\n')
    print(f'T20_OPERATOR_ABBA {final} cells=36 counts={json.dumps(counts, sort_keys=True)} '
          'blocks=17 retained=16 raw_samples=2448 no_extension=true no_sample_removal=true upper_ratio_limit=1.02', flush=True)
    return final, 0 if final == 'PASS' else 101 if final == 'DEFECT' else 1


def now():
    return subprocess.check_output(['date', '-u', '+%FT%TZ'], text=True).strip()


def free(ledger):
    events = [line for line in ledger.read_text().splitlines()
              if re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (START|CLOSED) ', line)]
    assert not events or ' CLOSED ' in events[-1], 'local ledger is owned'


def metadata(paths):
    return {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}


def run(a, b, model, root, runtime, ledger, owner):
    assert root.resolve() != runtime.resolve() and not root.exists(), 'fresh output directory required'
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', owner), 'invalid ledger owner'
    root.mkdir(parents=True)
    (root / 'COMMAND').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')
    free(ledger)
    with (root / 'SMI-BEFORE').open('w') as log:
        subprocess.run(['nvidia-smi'], stdout=log, check=True)
    query = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']
    apps = subprocess.check_output(query)
    (root / 'APPS-BEFORE').write_bytes(apps)
    assert not apps.strip(), 'GPU has a compute process'
    free(ledger)
    with ledger.open('a') as log:
        log.write(f'{now()} START {owner} T20-cost-r2-ABBA HOST_EXCLUSIVE cap 30 min MemoryMax12G\n')
    status, result = 'FAIL', 2
    try:
        files = [a, b, *[p for p in model.rglob('*') if p.is_file()]]
        before = metadata(files)
        (root / 'files-before.json').write_text(json.dumps(before, indent=2) + '\n')
        for block in range(1, 18):
            for position, arm in enumerate('ABBA', 1):
                directory = root / f'block-{block:02d}' / f'{position}-{arm}'
                arm_runtime = runtime / arm
                vision = arm_runtime / 'vision'
                vision.mkdir(parents=True, exist_ok=True)
                env = {k: os.environ[k] for k in ['HOME', 'XDG_RUNTIME_DIR', 'PATH', 'LD_LIBRARY_PATH', 'LANG'] if k in os.environ}
                env.update(CUDA_VISIBLE_DEVICES='0', FF_Q38_GPU_WINDOW='granted',
                           FF_Q38_VISION_WINDOW='granted', FF_Q38_MODEL=str(model),
                           FF_Q38_VISION_OUTPUT_DIR=str(vision), FLYINGFISH_RUNTIME_DIR=str(arm_runtime))
                for label, test in TESTS.items():
                    output = directory / label
                    output.mkdir(parents=True)
                    command = ['timeout', '--signal=TERM', '--kill-after=10s', '360',
                               'systemd-run', '--user', '--scope', '-p', 'MemoryMax=12G', '--',
                               str(a if arm == 'A' else b), test,
                               '--exact', '--include-ignored', '--test-threads=1', '--nocapture']
                    (output / 'COMMAND').write_text(shlex.join(command) + '\n')
                    (output / 'ENVIRONMENT.json').write_text(json.dumps(env, indent=2) + '\n')
                    with (output / 'raw.log').open('w') as log:
                        child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    (output / 'rc').write_text(str(child.returncode) + '\n')
                    assert child.returncode == 0, (output, 'test runner failure')
                read_arm(directory, arm)
            print(f'T20_COST_ABBA block={block} complete', flush=True)
        assert before == metadata(files), 'measured binary or published model changed'
        (root / 'files-after.json').write_text(json.dumps(metadata(files), indent=2) + '\n')
        assert not list(runtime.rglob('*.json')), 'fixed operator path wrote calibration entries'
        status, result = score(root)
    finally:
        with (root / 'SMI-AFTER').open('w') as log:
            subprocess.run(['nvidia-smi'], stdout=log, check=True)
        apps = subprocess.check_output(query)
        (root / 'APPS-AFTER').write_bytes(apps)
        if apps.strip():
            status, result = 'FAIL', 3
        with ledger.open('a') as log:
            log.write(f'{now()} CLOSED {owner} T20-cost-r2-ABBA HOST_EXCLUSIVE {status} {root}\n')
        (root / 'rc').write_text(str(result) + '\n')
    return result


if __name__ == '__main__':
    controls()
    if sys.argv[1:] != ['--controls-only']:
        if len(sys.argv) != 8:
            raise SystemExit('usage: qwen_operation_abba.py A B MODEL OUTPUT RUNTIME LEDGER OWNER')
        raise SystemExit(run(*[Path(arg).resolve() for arg in sys.argv[1:7]], sys.argv[7]))
