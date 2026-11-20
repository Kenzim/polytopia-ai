#!/bin/bash
# usage: tools/pgo_build.sh    Profile-guided build of the polyai_engine extension. Opt-in: normal builds are untouched.
#   1. builds polyai-core's bench_decision example with -Cprofile-generate (target dir target/pgo-gen);
#   2. trains on it: retail AI games with the collector's per-decision work (legal list, observation, action
#      encoding, features, truth labels) done before every command, which are the engine and polyai-core paths the
#      trainer spends its time in;
#   3. merges the profile with rustup's llvm-profdata (`rustup component add llvm-tools-preview`; the system one is
#      an older LLVM than rustc's) and builds the extension with -Cprofile-use, fat LTO and one codegen unit.
# Output: with OUT=<dir>, <dir>/polyai_engine.so (put the directory first on PYTHONPATH to use it without touching the
# installed module); without OUT, `maturin develop --release` installs it into the active virtualenv.
# Env: TRAIN_GAMES (default 16), TRAIN_TURNS (default 30), TRAIN_PLAYERS (default 2:55,3:25,4:20),
#      RUSTFLAGS_EXTRA (added to both builds), TARGET (cargo target dir, default target).
# Training uses one thread: the profile counters are shared, and several threads fighting over them slow training a lot.
# PGO changes code layout and inlining only: the script stops if the instrumented and the optimised bench_decision
# print different checksums.
set -e
AI=$(cd "$(dirname "$0")/.." && pwd)
cd "$AI"
PD=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata
[ -x "$PD" ] || { echo "llvm-profdata missing: rustup component add llvm-tools-preview"; exit 1; }
TARGET=${TARGET:-$AI/target}
DATA=$TARGET/pgo-data
rm -rf "$DATA"; mkdir -p "$DATA/raw"
train=("${TRAIN_GAMES:-16}" "${TRAIN_TURNS:-30}" "${TRAIN_PLAYERS:-2:55,3:25,4:20}")
export CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

CARGO_TARGET_DIR=$TARGET/pgo-gen RUSTFLAGS="-Cprofile-generate=$DATA/raw $RUSTFLAGS_EXTRA" \
  cargo build --release -p polyai-core --example bench_decision --features polytopia-sim/mimalloc
"$TARGET/pgo-gen/release/examples/bench_decision" "${train[@]}" | tee "$DATA/gen.txt"
"$PD" merge -o "$DATA/merged.profdata" "$DATA/raw"

export RUSTFLAGS="-Cprofile-use=$DATA/merged.profdata $RUSTFLAGS_EXTRA"
export CARGO_TARGET_DIR=$TARGET/pgo
cargo build --release -p polyai-core --example bench_decision --features polytopia-sim/mimalloc
"$TARGET/pgo/release/examples/bench_decision" "${train[@]}" | tee "$DATA/use.txt"
[ "$(grep '^checksums' "$DATA/gen.txt")" = "$(grep '^checksums' "$DATA/use.txt")" ] || { echo "checksums differ between the two builds"; exit 1; }
if [ -n "$OUT" ]; then
  cargo build --release -p polyai-py
  mkdir -p "$OUT"
  cp "$TARGET/pgo/release/libpolyai_engine.so" "$OUT/polyai_engine.so"
  echo "built: $OUT/polyai_engine.so"
else
  (cd crates/polyai-py && maturin develop --release)
fi
