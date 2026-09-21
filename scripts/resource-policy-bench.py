#!/usr/bin/env python3
"""Retain alternating H3/GLM command pairs and build verified resource evidence.

No fabricated cold label: cold plans invoke the explicit verified Linux control.
Raw stdout/stderr, telemetry, outputs and trial observations remain in the archive.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

COPY_BUFFER_BYTES = 1024 * 1024


def outputs_equal(left, right):
    if file_bytes(left) != file_bytes(right):
        return False
    with Path(left).open('rb') as a, Path(right).open('rb') as b:
        while True:
            block = a.read(COPY_BUFFER_BYTES)
            if block != b.read(COPY_BUFFER_BYTES):
                return False
            if not block:
                return True


def append_signal_file(out, name, path):
    """Frame each decoded file with its name and length, then copy its bytes."""
    encoded = name.encode('utf-8')
    with Path(path).open('rb') as source:
        size = os.fstat(source.fileno()).st_size
        out.write(struct.pack('<Q', len(encoded)))
        out.write(encoded)
        out.write(struct.pack('<Q', size))
        remaining = size
        while remaining:
            block = source.read(min(COPY_BUFFER_BYTES, remaining))
            if not block:
                raise ValueError('truncated decoded output')
            out.write(block)
            remaining -= len(block)
        if source.read(1):
            raise ValueError('decoded output grew while copying')


def file_bytes(path):
    return Path(path).stat().st_size


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f'refusing to overwrite {path}')
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n')


def status(root, value):
    temp = root / 'status.tmp'
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temp.replace(root / 'status.json')


def safetensors(path):
    with Path(path).open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        if length > 100_000_000:
            raise ValueError('oversize safetensors header')
        header = json.loads(f.read(length))
    return header, 8 + length


def tensor_bytes(path, header, base, name):
    item = header[name]
    lo, hi = item['data_offsets']
    with Path(path).open('rb') as f:
        f.seek(base + lo)
        value = f.read(hi - lo)
    if len(value) != hi-lo:
        raise ValueError('truncated tensor')
    return value


def h3_signal(path, destination):
    header, base = safetensors(path)
    with destination.open('xb') as out:
        for name in ['video_latents', 'audio_latents']:
            item = header[name]
            layout = json.dumps([name, item['dtype'], item['shape']], separators=(',', ':')).encode()
            out.write(struct.pack('<Q', len(layout)))
            out.write(layout)
            lo, hi = item['data_offsets']
            if item['dtype'] != 'F32' or lo < 0 or hi < lo or (hi-lo) % 4:
                raise ValueError('invalid output latent layout or dtype')
            with Path(path).open('rb') as source:
                source.seek(base + lo)
                remaining = hi - lo
                while remaining:
                    count = min(COPY_BUFFER_BYTES, remaining)
                    payload = source.read(count)
                    if len(payload) != count:
                        raise ValueError('truncated tensor')
                    if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)):
                        raise ValueError('non-finite output latent')
                    out.write(payload)
                    remaining -= count
    policy = json.loads(tensor_bytes(path, header, base, 'ff_policy_history_json'))['segments'][-1]['policy']
    return policy


def file_ref(root, path):
    return {'file': str(path.relative_to(root)), 'bytes': file_bytes(path)}


def collect_sample(pid):
    try:
        # /usr/bin/time is the direct child; sample its ff child, not time's RSS.
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        if not children:
            return None
        child = children[0]
        fields = {}
        for line in Path(f'/proc/{child}/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                k, v, _ = line.split(); fields[k[:-1]] = int(v) * 1024
        for line in Path(f'/proc/{child}/io').read_text().splitlines():
            k, v = line.split(':'); fields[k] = int(v)
        return fields
    except (OSError, ValueError):
        return None


def run_trial(root, binary, plan, pair, role, *, resource_policy='conservative', evidence=None):
    if resource_policy not in ['conservative', 'performance']:
        raise ValueError('invalid resource policy mode')
    case = root / f'pair-{pair:02d}-{role}'
    case.mkdir()
    glm = plan['family'] == 'glm'
    whole_h3 = not glm and plan['common_args'][:2] == ['video', 'generate']
    result_path = case / ('result.json' if glm else 'latents.safetensors')
    if whole_h3:
        result_path = case/'run'/'denoised-latents.safetensors'
    sidecar = case/'selection.json' if glm else (case/'run'/'resource-selection.json' if whole_h3 else result_path.with_suffix('.resource-selection.json'))
    telemetry = case / 'telemetry.json'
    args = [str(binary), *plan['common_args'], *plan[f'{role}_args'],
            '--resource-policy', resource_policy,
            '--telemetry-json', str(telemetry)]
    if evidence is not None:
        args += ['--resource-evidence', str(evidence)]
    args += ['--output-dir', str(case/'run')] if whole_h3 else ['--output', str(result_path)]
    if plan['cache_state'] == 'cold':
        args += ['--resource-cold-cache']
    if glm:
        args += ['--json', '--resource-selection', str(sidecar),
                 '--execution-manifest', str(case / 'execution.json'),
                 '--routing-trace', str(case / 'routing.json')]
    write_json(case / 'command.json', {'argv': args, 'cwd': os.getcwd()})
    print(f'START pair {pair} {role}', flush=True)
    started = time.monotonic_ns()
    status(root, {'state': 'running', 'pair': pair, 'role': role, 'case': str(case)})
    samples = []
    with (case/'stdout.log').open('wb') as stdout, (case/'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(case/'time.txt'), *args], stdout=stdout, stderr=stderr)
        while process.poll() is None:
            sample = collect_sample(process.pid)
            if sample:
                samples.append({'elapsed_us': (time.monotonic_ns()-started)//1000, **sample})
            time.sleep(.1)
    wall = (time.monotonic_ns()-started)//1000
    write_json(case/'process-samples.json', samples)
    if process.returncode:
        write_json(case/'failure.json', {'exit_code': process.returncode, 'wall_us': wall})
        print(f'FAIL pair {pair} {role}: exit {process.returncode}', flush=True)
        return None
    selection = json.loads(sidecar.read_text())
    report = json.loads(telemetry.read_text())
    if report['process_sampling_errors'] or report['device_sampling_errors']:
        raise RuntimeError('telemetry sampling errors prevent qualification')
    signal = case/'output-signal.bin'
    if glm:
        output = json.loads(result_path.read_text())
        policy = output['execution_policy']
        signal.write_bytes(json.dumps({'token_ids': output['generated_token_ids'], 'text': output['text']},
                                      sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
        context = output['evidence_context']
        phase = {'prefill': output['prefill_elapsed_ms']*1000, 'decode': output['decode_elapsed_ms']*1000}
        counters = {f'expert_{key}': value for key, value in output['expert_cache'].items() if isinstance(value, int)}
        if plan['cache_state'] == 'cold' and output['cache_preparation']['resident_pages'] != 0:
            raise RuntimeError('cold state was not verified')
    else:
        policy = h3_signal(result_path, signal)
        if whole_h3:
            # Compare decoded artifacts as well, excluding policy/timing-bearing
            # completion records. The signal file carries their bytes, so the
            # paired outputs are compared by content rather than by identity.
            frames = sorted((case/'run'/'frames').glob('*.png'))
            if not frames:
                raise RuntimeError('whole generation has no decoded frames')
            with signal.open('ab') as f:
                append_signal_file(f, 'generated.wav', case/'run'/'generated.wav')
                for frame in frames:
                    append_signal_file(f, 'frames/' + frame.name, frame)
        context = {'model': selection['model'],
                   'request': selection['request'], 'hardware': selection['hardware'],
                   'executable_metadata': selection.get('executable'),
                   'environment': selection['environment'],
                   'cache_state': 'cold' if plan['cache_state'] == 'cold' else 'uncontrolled'}
        phase = {'model_execution_and_publication': report['elapsed_ms']*1000}
        counters = {'telemetry_samples': report['samples']}
        if plan['cache_state'] == 'cold' and selection['workload'].get('cold_cache_resident_pages') != 0:
            raise RuntimeError('cold state was not verified')
    metadata = context.get('executable_metadata')
    if metadata is not None:
        stat = binary.stat()
        if (metadata['bytes'], metadata['modified_unix_ns']) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError('running executable identity changed')
    phase['other_command_work'] = max(0, wall-sum(phase.values()))
    for key in ['read_bytes', 'write_bytes', 'rchar', 'wchar']:
        if samples:
            counters[key] = max(s.get(key, 0) for s in samples)
    rss = max(report.get('peak_process_rss_bytes') or 0, report.get('process_high_watermark_bytes') or 0)
    match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', (case/'time.txt').read_text())
    if match:
        rss = max(rss, int(match[1])*1024)
    baseline_rss = next((s['VmRSS'] for s in samples if s.get('VmRSS', 0)>0), 0)
    counters['rss_baseline_is_startup_sample'] = int(baseline_rss > 0)
    trial = {'schema_version': 1, 'context': context,
             'policy': {'family': plan['family'], 'policy': policy},
             'wall_us': wall, 'output_bytes': file_bytes(signal), 'output_file': str(signal.relative_to(root)),
             'phase_us': phase, 'peak_process_rss_bytes': rss, 'baseline_process_rss_bytes': min(rss, baseline_rss),
             'peak_device_bytes': report['cuda_peak_used_bytes'], 'baseline_device_bytes': report['cuda_baseline_used_bytes'],
             'counters': counters}
    write_json(case/'trial.json', trial)
    write_json(case/'policy.json', policy)
    print(f'DONE pair {pair} {role}: {wall/1e6:.2f}s', flush=True)
    return {'trial': trial, 'case': case, 'policy': policy}


def run(args):
    plan = json.loads(args.plan.read_text())
    required = {'schema_version', 'family', 'common_args', 'baseline_args', 'candidate_args', 'pairs', 'minimum_improvement_basis_points', 'cache_state'}
    if set(plan) != required or plan['schema_version'] != 1 or plan['family'] not in ['h3', 'glm']:
        raise ValueError('invalid benchmark plan')
    if plan['pairs'] < 3 or plan['cache_state'] not in ['cold', 'uncontrolled']:
        raise ValueError('at least three pairs and an explicit cache state are required')
    for key in ['common_args', 'baseline_args', 'candidate_args']:
        if not isinstance(plan[key], list) or any(not isinstance(x, str) for x in plan[key]):
            raise ValueError('arguments must be string arrays')
    expected = [['text','generate']] if plan['family']=='glm' else [['video','denoise'],['video','denoise-conditioned'],['video','generate']]
    if plan['common_args'][:2] not in expected:
        raise ValueError('unsupported benchmark command for this family')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = root/'ff'
    shutil.copy2(args.binary, binary)
    binary.chmod(0o555)
    write_json(root/'plan.json', plan)
    files = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z']).decode().split('\0')
    source = {p: file_bytes(p) for p in sorted(set(files)) if p and Path(p).is_file()}
    head = subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'HEAD'], capture_output=True, text=True)
    if head.returncode not in (0, 1):
        head.check_returncode()
    write_json(root/'source.json', {'head': head.stdout.strip() if head.returncode == 0 else None,
                                  'files':source, 'binary':file_bytes(binary)})
    paired = []
    for pair in range(plan['pairs']):
        roles = ['baseline','candidate'] if pair%2 == 0 else ['candidate','baseline']
        outcomes = {role:run_trial(root,binary,plan,pair,role) for role in roles}
        paired.append(outcomes)
    good = [p for p in paired if p['baseline'] and p['candidate']]
    if len(good) != plan['pairs']:
        status(root, {'state':'finished_incomplete', 'successful_pairs':len(good), 'requested_pairs':plan['pairs']})
        return 1
    context = good[0]['baseline']['trial']['context']
    base_policy = good[0]['baseline']['trial']['policy']
    candidate_policy = good[0]['candidate']['trial']['policy']
    for pair in good:
        for role in ['baseline','candidate']:
            if pair[role]['trial']['context'] != context:
                raise RuntimeError('evidence context changed across paired trials')
        if pair['baseline']['trial']['policy'] != base_policy or pair['candidate']['trial']['policy'] != candidate_policy:
            raise RuntimeError('execution policy changed across repeated trials')
    candidate = {'baseline_policy':base_policy,
                 'candidate':{'family':plan['family'],'policy':good[0]['candidate']['policy']},
                 'minimum_improvement_basis_points':plan['minimum_improvement_basis_points'],
                 'routing_trace':None,'routing_replay':None,'pairs':[]}
    for pair in good:
        b,c=pair['baseline'],pair['candidate']
        candidate['pairs'].append({'baseline_wall_us':b['trial']['wall_us'],'candidate_wall_us':c['trial']['wall_us'],
                                  'baseline_record':file_ref(root,b['case']/'trial.json'),
                                  'candidate_record':file_ref(root,c['case']/'trial.json')})
    if plan['family']=='glm' and good[0]['candidate']['policy']['expert_cache']['maximum_bound_bytes']:
        trace=good[0]['candidate']['case']/'routing.json'
        replay=root/'routing-replay.json'
        budget=good[0]['candidate']['policy']['expert_cache']['maximum_bound_bytes']
        if budget%(1024*1024):
            raise RuntimeError('CLI replay requires an integral MiB budget')
        subprocess.run([str(binary),'text','replay-routing','--adapter','glm','--trace',str(trace),'--output',str(replay),
                        '--segment-lengths','4,16','--cache-mib',str(budget//(1024*1024))],check=True,
                       stdout=(root/'replay.log').open('w'),stderr=subprocess.STDOUT)
        candidate['routing_trace']=file_ref(root,trace)
        candidate['routing_replay']=file_ref(root,replay)
    write_json(root/'evidence.json',{'schema_version':1,'context':context,'candidates':[candidate]})
    with (root/'verification.json').open('w') as out, (root/'verification.log').open('w') as err:
        subprocess.run([str(binary),'verify-resource-evidence','--input',str(root/'evidence.json')],check=True,stdout=out,stderr=err)
    status(root,{'state':'finished','pairs':len(good),
                 'output_equal':all(outputs_equal(root/p['baseline']['trial']['output_file'],
                                                  root/p['candidate']['trial']['output_file']) for p in good),
                 'baseline_wall_us':[p['baseline']['trial']['wall_us'] for p in good],
                 'candidate_wall_us':[p['candidate']['trial']['wall_us'] for p in good]})
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as error:
        if args.output.exists():
            status(args.output,{'state':'failed','error':str(error)})
        raise
