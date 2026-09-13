#!/usr/bin/env python3
"""Reproduce the checked-in Qwen GELU PTX with pinned NVRTC 13.0."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path

from cuda.bindings import nvrtc


CUDA_BINDINGS_VERSION = "13.3.1"
NVRTC_VERSION = (13, 0)
NVRTC_BYTES = 109_338_064
NVRTC_BUILTINS_BYTES = 4_380_616
OPTIONS = (
    b"--std=c++17",
    b"--gpu-architecture=compute_80",
    b"--fmad=true",
    b"--prec-div=true",
    b"--prec-sqrt=true",
    b"--ftz=false",
)


def check(result: tuple[object, ...]) -> object:
    if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC call failed: {result[0]}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def mapped_nvrtc_paths() -> set[Path]:
    paths = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5]
        path = Path(raw.removesuffix(" (deleted)"))
        if path.name.startswith(("libnvrtc.so.", "libnvrtc-builtins.so.")):
            if raw.endswith(" (deleted)"):
                raise SystemExit(f"mapped NVRTC binary was deleted: {raw}")
            paths.add(path.resolve(strict=True))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nvrtc-library", type=Path, required=True)
    parser.add_argument("--nvrtc-builtins", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="the checked-in PTX this compilation must reproduce byte for byte",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve(strict=False)
    library = args.nvrtc_library.resolve(strict=True)
    builtins = args.nvrtc_builtins.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite output: {output}")
    if not output.parent.is_dir():
        raise SystemExit("output parent does not exist")
    if importlib.metadata.version("cuda-bindings") != CUDA_BINDINGS_VERSION:
        raise SystemExit("unexpected cuda-bindings package version")
    # Size and the version NVRTC reports, which is what can be established
    # without reading these binaries. It is not a claim about their contents.
    for path, size in ((library, NVRTC_BYTES), (builtins, NVRTC_BUILTINS_BYTES)):
        if path.stat().st_size != size:
            raise SystemExit(f"unexpected NVRTC binary size: {path}")
    runtime_handles = [ctypes.CDLL(str(library)), ctypes.CDLL(str(builtins))]
    status, major, minor = nvrtc.nvrtcVersion()
    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS or (major, minor) != NVRTC_VERSION:
        raise SystemExit(f"unexpected NVRTC version/status: {(status, major, minor)}")
    expected_mapped = {library, builtins}
    observed_mapped = mapped_nvrtc_paths()
    if observed_mapped != expected_mapped:
        raise SystemExit(
            f"mapped NVRTC binaries differ: observed={observed_mapped}, expected={expected_mapped}"
        )

    source_bytes = source.read_bytes()
    program = check(
        nvrtc.nvrtcCreateProgram(source_bytes, source.name.encode(), 0, [], [])
    )
    try:
        compiled = nvrtc.nvrtcCompileProgram(program, len(OPTIONS), list(OPTIONS))
        if compiled[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log_size = check(nvrtc.nvrtcGetProgramLogSize(program))
            log = b" " * int(log_size)
            check(nvrtc.nvrtcGetProgramLog(program, log))
            raise SystemExit(log.rstrip(b"\0").decode(errors="replace"))
        ptx_size = check(nvrtc.nvrtcGetPTXSize(program))
        ptx = b" " * int(ptx_size)
        check(nvrtc.nvrtcGetPTX(program, ptx))
    finally:
        check(nvrtc.nvrtcDestroyProgram(program))
    if mapped_nvrtc_paths() != expected_mapped:
        raise SystemExit("mapped NVRTC binaries changed during compilation")
    # NVRTC returns a trailing NUL and blank line. Canonical repository PTX
    # has exactly one final newline so byte identity is platform-independent.
    ptx = ptx.rstrip(b"\0\n") + b"\n"
    # The point of this script is that the checked-in PTX is reproducible, and
    # the checked-in PTX is right here to compare against.
    committed = args.reference.read_bytes() if args.reference else None
    if committed is not None and ptx != committed:
        raise SystemExit(
            f"compiled PTX differs from {args.reference} "
            f"({len(ptx)} bytes compiled, {len(committed)} committed)"
        )
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(ptx)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "source": str(source),
                "ptx_bytes": len(ptx),
                "reproduced_reference": committed is not None,
                "options": [option.decode() for option in OPTIONS],
            },
            sort_keys=True,
        )
    )
    _ = runtime_handles


if __name__ == "__main__":
    main()
