#!/usr/bin/env python3
"""Reference greedy decode for Qwen3.8-27B via HF transformers (bf16, CPU).

Emits the prompt ids and the greedy token ids — the acceptance fixture for
the ff-qwen35 int4 path. Needs transformers with qwen3_5 support and CPU
torch; ~52 GiB of bf16 weights, so run memory-scoped (e.g. systemd-run
--user --scope -p MemoryMax=...).

Usage: qwen35_reference.py [n_tokens] [prompt...]
"""

import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "models/Qwen/Qwen3.8-27B"

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    prompts = sys.argv[2:] or ["Explain paging to a systems programmer."]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    for prompt in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=n,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        prompt_ids = ids["input_ids"][0].tolist()
        gen_ids = out[0][len(prompt_ids):].tolist()
        print(json.dumps({"prompt": prompt, "prompt_ids": prompt_ids, "gen_ids": gen_ids}))

if __name__ == "__main__":
    main()
