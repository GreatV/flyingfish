"""Independent prefill softmax references and cross-device rounding floors."""
import argparse
import json
from pathlib import Path

import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = fixtures()
    (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    for case in cases:
        data = reference(case)
        assert all(torch.isfinite(value).all() for value in data.values())
        save_file(data, args.output / (case['name'] + '.safetensors'))
        print('ATTENTION_REFERENCE', case['name'], case['d'], case['mode'], case['rows'], flush=True)
    (args.output / 'report.json').write_text(json.dumps(dict(cells=len(cases), torch=torch.__version__,
        device=torch.cuda.get_device_name(), reference='independent FP64 softmax', timing=False)) + '\n')


if __name__ == '__main__':
    main()
