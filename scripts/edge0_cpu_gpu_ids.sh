#!/usr/bin/env bash
# Standing acceptance: CPU and GPU builds must emit identical token ids.
# Run at every close-out; catches GPU-side omissions that neither error nor
# slow down (the missing shared_expert LoRA class of bug).
set -euo pipefail
cd "$(dirname "$0")/.."
# One cuda build serves both runs: EDGE0_GPU unset = CPU path,
# EDGE0_GPU=full = GPU full-resident (the generate example declares
# required-features = ["cuda"] — a non-cuda build no longer exists).
cargo build --release --example generate -p ff-edge0 --features cuda
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
./target/release/examples/generate | grep -o 'id [0-9]*' | awk '{print $2}' > "$tmp/ids_cpu"
EDGE0_GPU=full ./target/release/examples/generate | grep -o 'id [0-9]*' | awk '{print $2}' > "$tmp/ids_gpu"
if diff -q "$tmp/ids_cpu" "$tmp/ids_gpu" > /dev/null; then
    echo "CPU/GPU ids identical ($(wc -l < "$tmp/ids_cpu") tokens)"
else
    echo "MISMATCH:" && diff "$tmp/ids_cpu" "$tmp/ids_gpu" && exit 1
fi
