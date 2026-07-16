#!/usr/bin/env bash
# Decisive Wall Chess search gate.
#
# Runs candidate search env knobs against the stripped deployed champion across
# multiple opening lengths and seeds. Do not stop on a favorable partial read.
#
# Usage:
#   core/scripts/search_gate.sh [games] [depth] [node_budget]
#
# Defaults: games=300 depth=12 node_budget=600000
# Override matrix:
#   SEEDS="42 99 123" OPENS="6 8" CANDIDATE_ENV="WC_PVS=1 WC_ASP=1 WC_NULLMOVE=1" core/scripts/search_gate.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-300}"
DEPTH="${2:-12}"
NODES="${3:-600000}"
SEEDS="${SEEDS:-42 99 123}"
OPENS="${OPENS:-6 8}"
CANDIDATE_ENV="${CANDIDATE_ENV:-WC_PVS=1 WC_ASP=1 WC_NULLMOVE=1}"

printf 'building release bestmove + xmatch ...\n'
cargo build --release --bin bestmove --bin xmatch >/dev/null 2>&1
BM="$(pwd)/target/release/bestmove"
XM="$(pwd)/target/release/xmatch"

# Champion = deployed default, with every search experiment knob stripped.
CHAMP="env -u WC_PRESET -u WC_PVS -u WC_ASP -u WC_NULLMOVE -u WC_RFP -u WC_FUTILITY -u WC_LMP -u WC_QUIESCENCE $BM $DEPTH $NODES"
read -r -a ENV_PARTS <<< "$CANDIDATE_ENV"

printf 'candidate env: %s\n' "$CANDIDATE_ENV"
printf 'champion: %s\n' "$CHAMP"
printf 'games=%s depth=%s nodes=%s opens=(%s) seeds=(%s)\n' "$GAMES" "$DEPTH" "$NODES" "$OPENS" "$SEEDS"

for open in $OPENS; do
  for seed in $SEEDS; do
    printf '\n==================================================================\n'
    printf 'open=%s seed=%s games=%s\n' "$open" "$seed" "$GAMES"
    env "${ENV_PARTS[@]}" "$XM" "$BM $DEPTH $NODES" "$CHAMP" "$GAMES" "$open" "$seed" 2>/dev/null | tail -1
  done
done

printf '\ndone. A = candidate, B = stripped deployed champion. Commit only a pre-declared, repeated edge.\n'
