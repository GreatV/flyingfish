"""Independent prefill softmax references and cross-device rounding floors."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import save_file


def pattern(shape, seed):
    count = 1
    for value in shape:
        count *= value
    result = torch.empty(count, dtype=torch.bfloat16)
    for start in range(0, count, 1048576):
        index = torch.arange(start, min(start + 1048576, count), dtype=torch.int64)
        values = ((index * 17 + seed * 31) % 257 - 128).float() / 64
        values += ((index * 29 + seed * 13) % 263 - 131).float() / 128
        result[start:start + len(index)] = values.bfloat16()
    return result.reshape(shape)


def fixtures():
    cases = []
    families = [(256, 24, 4, 'causal'), (256, 16, 2, 'causal'),
                (256, 8, 2, 'causal'), (128, 16, 2, 'causal'),
                (128, 16, 8, 'bidirectional'), (128, 12, 12, 'prefix'),
                (128, 32, 4, 'window'), (128, 32, 4, 'causal')]
    for d, hq, hkv, mode in families:
        for seed in (31, 47, 71):
            for rows, start in ((1, 0), (3, 17), (16, 31), (17, 127), (33, 128),
                                (65, 257), (129, 33), (257, 0)):
                keys = start + rows
                if mode == 'bidirectional':
                    keys += 17
                extents = (0, keys // 2, keys) if mode == 'prefix' else ((0, 511) if mode == 'window' else (0,))
                for extent in extents:
                    cases.append(dict(d=d, hq=hq, hkv=hkv, mode=mode, seed=seed,
                                      rows=rows, start=start, keys=keys, extent=extent))
        for rows in (16384, 65536):
            cases.append(dict(d=d, hq=hq, hkv=hkv, mode=mode, seed=71, rows=rows,
                              start=0, keys=rows, extent=rows // 2 if mode == 'prefix' else 511))
    for index, case in enumerate(cases):
        case.update(name=f'cell{index:04}', capacity=case['keys'] + 23,
                    pitch=case['hq'] * case['d'] + 128)
    return cases


def reference(case):
    d, hq, hkv = (case[k] for k in ('d', 'hq', 'hkv'))
    rows, start, keys = (case[k] for k in ('rows', 'start', 'keys'))
    selected = list(range(rows)) if rows <= 257 else sorted(set(
        [0, 1, 15, 16, 31, 32, 63, 64, rows // 2 - 1, rows // 2, rows - 2, rows - 1]))
    q = pattern((rows, hq, d), case['seed'])[selected].transpose(0, 1).contiguous()
    k = pattern((hkv, case['capacity'], d), case['seed'] + 7)[:, :keys].contiguous()
    v = pattern((hkv, case['capacity'], d), case['seed'] + 19)[:, :keys].contiguous()
    q = q.cuda(); k = k.cuda(); v = v.cuda()
    query = torch.tensor(selected, device='cuda') + start
    key = torch.arange(keys, device='cuda')
    mask = key[None, :] <= query[:, None]
    if case['mode'] == 'bidirectional':
        mask = torch.ones_like(mask)
    elif case['mode'] == 'prefix':
        mask = key[None, :] < torch.maximum(query[:, None] + 1, torch.tensor(case['extent'], device='cuda'))
    elif case['mode'] == 'window':
        mask &= key[None, :] >= query[:, None] - case['extent']
    outputs, lses, baseline, baseline_lse, variations = [], [], [], [], []
    for head in range(hq):
        kv = head // (hq // hkv)
        scores = q[head].double() @ k[kv].double().T / d**0.5
        scores.masked_fill_(~mask, -torch.inf)
        lse = scores.logsumexp(-1)
        expected = scores.softmax(-1) @ v[kv].double()
        score = q[head].float() @ k[kv].float().T / d**0.5
        score.masked_fill_(~mask, -torch.inf)
        actual = (score.softmax(-1) @ v[kv].float()).bfloat16().float()
        one = torch.cat([(q[head, row:row + 17].float() @ k[kv].float().T / d**0.5)
                         .masked_fill(~mask[row:row + 17], -torch.inf).softmax(-1)
                         @ v[kv].float() for row in range(0, len(selected), 17)]).bfloat16().float()
        for _ in range(2):
            again = q[head].float() @ k[kv].float().T / d**0.5
            again.masked_fill_(~mask, -torch.inf)
            assert torch.equal((again.softmax(-1) @ v[kv].float()).bfloat16().float(), actual)
        outputs.append(expected.cpu()); lses.append(lse.cpu())
        baseline.append(actual.cpu()); baseline_lse.append(score.logsumexp(-1).cpu())
        variations.append(one.cpu())
    return dict(output=torch.stack(outputs).transpose(0, 1).contiguous(),
                lse=torch.stack(lses).T.contiguous(),
                baseline=torch.stack(baseline).transpose(0, 1).contiguous(),
                baseline_lse=torch.stack(baseline_lse).T.contiguous(),
                single=torch.stack(variations).transpose(0, 1).contiguous(),
                selected=torch.tensor(selected))


def extended_cases():
    cases = {}

    def add(d, hq, hkv, rows, start, scale, window, padded):
        case = dict(d=d, hq=hq, hkv=hkv, rows=rows, start=start, keys=start + rows,
                    capacity=start + rows + 64, pitch=hq*d + (512 if padded else 0),
                    scale=scale, mode='window' if window is not None else 'causal',
                    extent=window if window is not None else 0)
        cases.setdefault(tuple(case.values()), case)

    lengths = (1, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257)
    for group, (hq, hkv) in enumerate(((8, 1), (8, 2), (16, 8), (32, 16))):
        for window in (512, 1024):
            starts = (0, 1, 31, 63, window-1, window, window+1, 4095)
            for scale in (1.0, 0.0625):
                for i, rows in enumerate(lengths):
                    for padded in (False, True):
                        add(256, hq, hkv, rows, starts[(i+group) % len(starts)], scale, window-1, padded)
                for start in (window-1, window, window+1):
                    for rows in (1, 5, 65):
                        add(256, hq, hkv, rows, start, scale, window-1, False)
                for rows in (16384, 65536):
                    for padded in (False, True):
                        add(256, hq, hkv, rows, 0, scale, window-1, padded)
        for rows in (1, 5, 65):
            for scale in (1.0, 0.0625):
                for window in (None, 512):
                    add(256, hq, hkv, rows, 4095, scale, window, True)
    for group, (hq, hkv, limit) in enumerate(((8, 1, 131072), (8, 2, 131072), (16, 1, 262144), (32, 4, 262144))):
        for i, rows in enumerate(lengths):
            for padded in (False, True):
                add(512, hq, hkv, rows, (0, 1, 31, 63, 4095)[(i+group) % 5], 1.0, None, padded)
        for rows in (16384, 65536):
            for padded in (False, True):
                add(512, hq, hkv, rows, 0, 1.0, None, padded)
        for rows in (1, 5, 16):
            for padded in (False, True):
                add(512, hq, hkv, rows, limit-rows, 1.0, None, padded)
    assert len(cases) == 764
    return [dict(case, name=f'e-cell{i}-seed{seed}', seed=seed)
            for seed in (47, 71) for i, case in enumerate(cases.values())]


def extended_pattern(positions, heads, dim, seed, kind, head_offset=0):
    row = np.asarray(positions, dtype=np.int64)[None, :, None]
    head = np.arange(head_offset, head_offset+heads)[:, None, None]
    channel = np.arange(dim)[None, None, :]
    if kind == 'q':
        value = ((row*37 + head*53 + channel*seed + 11) % 1009 - 504) / 512.
    elif kind == 'k':
        value = ((row*17 + head*41 + channel*(seed+2) + 23) % 1013 - 506) / 512.
    else:
        value = ((row*29 + head*67 + channel*(seed+4) + 7) % 1009 - 504) / 512.
    bits = value.astype(np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.view(np.float32).astype(np.float64)


def extended_reference(case):
    rows, d, hq, hkv = (case[k] for k in ('rows', 'd', 'hq', 'hkv'))
    selected = np.arange(rows) if rows <= 257 else np.unique([0, 1, 15, 16, 31, 63, 64, rows//2, rows-2, rows-1])
    query = selected + case['start']
    first = max(0, int(query.min()) - case['extent']) if case['mode'] == 'window' else 0
    keys = np.arange(first, int(query.max())+1)
    legal = keys[None, :] <= query[:, None]
    if case['mode'] == 'window':
        legal &= keys[None, :] >= np.maximum(0, query[:, None] - case['extent'])
    q = extended_pattern(query, hq, d, case['seed'], 'q')
    output = np.empty((len(selected), hq, d), np.float64)
    lse = np.empty((len(selected), hq), np.float64)
    for kv in range(hkv):
        k = extended_pattern(keys, 1, d, case['seed'], 'k', kv)[0]
        v = extended_pattern(keys, 1, d, case['seed'], 'v', kv)[0]
        for head in range(kv*(hq//hkv), (kv+1)*(hq//hkv)):
            scores = np.where(legal, (q[head] @ k.T)*case['scale'], -np.inf)
            peak = scores.max(1)
            probability = np.exp(scores-peak[:, None])
            total = probability.sum(1)
            output[:, head] = probability @ v / total[:, None]
            lse[:, head] = peak + np.log(total)
    return {name: torch.from_numpy(value) for name, value in
            dict(output=output, lse=lse, selected=selected.astype(np.int64)).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=('native', 'extended'), default='native')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = extended_cases() if args.family == 'extended' else fixtures()
    (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    for case in cases:
        data = extended_reference(case) if args.family == 'extended' else reference(case)
        assert all(torch.isfinite(value).all() for value in data.values())
        save_file(data, args.output / (case['name'] + '.safetensors'))
        print('ATTENTION_REFERENCE', case['name'], case['d'], case['mode'], case['rows'], flush=True)
    (args.output / 'report.json').write_text(json.dumps(dict(cells=len(cases), torch=torch.__version__,
        device='CPU' if args.family == 'extended' else torch.cuda.get_device_name(),
        reference='independent FP64 softmax', timing=False)) + '\n')


if __name__ == '__main__':
    main()
