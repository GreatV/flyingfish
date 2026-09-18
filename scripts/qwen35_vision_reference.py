#!/usr/bin/env python3
"""HF bf16 reference for the qwen35 vision gates. MemoryMax-scope the run
(~52 GiB weights for tower/e2e modes; processor/rope modes are light).

Usage:
  qwen35_vision_reference.py make-image <out.png>          # deterministic test PNG
  qwen35_vision_reference.py processor <image.png> <out.json>   # grid + pixel_values fixture
  qwen35_vision_reference.py rope <image.png> <out.json>        # get_rope_index positions + delta
  qwen35_vision_reference.py tower <image.png> <out.json>       # model.visual pooler_output
  qwen35_vision_reference.py e2e <image.png> <out.json> [n] [prompt...]  # greedy ids + margins
"""

import json
import struct
import sys
import zlib

MODEL = "models/Qwen/Qwen3.8-27B"
MERGE = 2


def make_image(path):
    """Deterministic 96x80 RGB8 PNG (gradient + channel offset) — no deps."""
    w, h = 96, 80

    def row(y):
        out = bytearray([0])  # filter: none
        for x in range(w):
            out += bytes(((x * 3 + y) % 256, (x + y * 2) % 256, (x * x + y) % 256))
        return bytes(out)

    raw = b"".join(row(y) for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)
    print(f"wrote {path} ({w}x{h})", file=sys.stderr)


def load_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL)


def processor_dump(image_path, out_path):
    import torch
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    image = load_rgb(image_path)
    result = proc.image_processor(images=[image], return_tensors="pt")
    pixel_values = result["pixel_values"]
    grid = result["image_grid_thw"][0].tolist()
    with open(out_path, "w") as f:
        json.dump(
            {
                "grid_thw": grid,
                "shape": list(pixel_values.shape),
                "pixel_values": pixel_values.flatten().tolist(),
            },
            f,
        )
    print(f"grid {grid}, patches {pixel_values.shape}", file=sys.stderr)


def load_rgb(image_path):
    """Decode the RGB8 PNG without PIL (the venv may lack it)."""
    import numpy as np

    with open(image_path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = height = None
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, ctype, _, _, _ = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and ctype == 2, "RGB8 only"
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        line = bytearray(raw[y * stride + 1 : (y + 1) * stride])
        filt = raw[y * stride]
        assert filt == 0, f"filter {filt} unsupported (our encoder writes none)"
        out[y] = np.frombuffer(bytes(line), dtype=np.uint8).reshape(width, 3)
    from PIL import Image

    return Image.fromarray(out, "RGB")


def rope_dump(image_path, out_path):
    """mrope positions + delta for one image + a fixed short prompt."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    proc = AutoProcessor.from_pretrained(MODEL)
    image = load_rgb(image_path)
    prompt = "Describe this image."
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = proc(text=[text], images=[image], return_tensors="pt")
    model = load_model()
    position_ids, rope_delta = model.model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        inputs.get("image_grid_thw"),
        None,
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "input_ids": inputs["input_ids"][0].tolist(),
                "grid_thw": inputs["image_grid_thw"][0].tolist(),
                "position_ids": position_ids[:, 0].tolist(),
                "rope_delta": float(rope_delta[0].item()),
            },
            f,
        )
    print(
        f"seq {inputs['input_ids'].shape}, delta {rope_delta[0].item()}",
        file=sys.stderr,
    )


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def tower_dump(image_path, out_path):
    """Vision-module-only fixture, in f32: the bf16 tower is measurably
    ill-conditioned in late blocks (bf16-vs-f32 HF diverges by hundreds of
    units before the merger), so the gate references the f32 run — the
    merger's LN makes the f32 output well-conditioned."""
    import glob

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    import safetensors.torch as st

    proc = load_processor()
    image = load_rgb(image_path)
    result = proc.image_processor(images=[image], return_tensors="pt")
    cfg = AutoConfig.from_pretrained(MODEL)
    visual = Qwen3_5VisionModel(cfg.vision_config)
    sd = {}
    for shard in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
        for k, v in st.load_file(shard).items():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = v
    visual.load_state_dict(sd, strict=True)
    visual = visual.to(torch.float32).eval()
    with torch.no_grad():
        out = visual(result["pixel_values"].float(), result["image_grid_thw"])
    pooler = out.pooler_output if hasattr(out, "pooler_output") else out
    with open(out_path, "w") as f:
        json.dump(
            {
                "shape": list(pooler.shape),
                "values": pooler.float().flatten().tolist(),
            },
            f,
        )
    print(f"tower output {list(pooler.shape)}", file=sys.stderr)


def e2e(image_path, out_path, n, prompts):
    import torch

    proc = load_processor()
    image = load_rgb(image_path)
    model = load_model()
    results = []
    for prompt in prompts:
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = proc(text=[text], images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=n, do_sample=False, return_dict_in_generate=True,
                output_logits=True,
            )
        ids = out.sequences[0][inputs["input_ids"].shape[1] :].tolist()
        margins = []
        for logits in out.logits:
            top2 = torch.topk(logits[0].float(), 2).values
            margins.append((top2[0] - top2[1]).item())
        results.append({"prompt": prompt, "gen_ids": ids, "margins": margins})
    with open(out_path, "w") as f:
        json.dump(results, f)
    for r in results:
        print(f"e2e ids: {r['gen_ids']}", file=sys.stderr)


def main():
    mode = sys.argv[1]
    if mode == "make-image":
        make_image(sys.argv[2])
    elif mode == "processor":
        processor_dump(sys.argv[2], sys.argv[3])
    elif mode == "rope":
        rope_dump(sys.argv[2], sys.argv[3])
    elif mode == "tower":
        tower_dump(sys.argv[2], sys.argv[3])
    elif mode == "e2e":
        e2e(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16, sys.argv[5:] or ["Describe this image."])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
