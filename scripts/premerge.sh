#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
    printf '\n=== %s ===\n' "$*"
    "$@"
}

run cargo fmt --all -- --check
run cargo clippy --locked --workspace --all-targets
run cargo test --locked --workspace --all-targets
RUSTDOCFLAGS="-D warnings" run cargo doc --locked --workspace --no-deps
run cargo build --locked --release
