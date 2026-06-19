#!/usr/bin/env bash
# Pruning tradeoff harness.
#
# Compares pruning configs against the deployed-default "champion" via the
# xmatch referee. Two regimes:
#   equal-depth  — same fixed depth for both engines. Answers "does pruning lose
#                  strength at the SAME depth?" (pure accuracy cost of pruning).
#   equal-nodes  — same node budget, high depth cap. The pruned engine reaches a
#                  greater depth in the same budget. Answers "is pruning a net
#                  win when the saved nodes are spent going deeper?"
#
# Pruning config is selected per-engine through WC_* env vars. The default
# (champion) engine runs with WC_PRESET stripped, so it is exactly the deployed
# search regardless of the caller's environment.
#
# Usage:
#   core/scripts/prune_tune.sh [games] [depth] [node_budget] [seed]
# Defaults: games=300 depth=4 node_budget=300000 seed=1
#
# Run a single config instead of the full sweep:
#   CONFIGS="WC_PRESET=aggressive" core/scripts/prune_tune.sh 300
#
# NOTE: run the FULL game count. Partial favorable reads are sampling bias.

set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-300}"
DEPTH="${2:-4}"
NODES="${3:-300000}"
SEED="${4:-1}"

echo "building release bestmove + xmatch ..."
cargo build --release --bin bestmove --bin xmatch >/dev/null 2>&1
BM="$(pwd)/target/release/bestmove"
XM="$(pwd)/target/release/xmatch"

# Champion = deployed default, env stripped so it never inherits a preset.
CHAMP_DEPTH="env -u WC_PRESET $BM $DEPTH"
CHAMP_NODES="env -u WC_PRESET $BM 12 $NODES"

# Configs under test. Each entry is the WC_* env prefix applied to engine A.
# Override with CONFIGS="..." (newline- or semicolon-separated).
DEFAULT_CONFIGS='WC_PRESET=safe
WC_PRESET=aggressive
WC_PVS=1 WC_ASP=1
WC_PVS=1 WC_ASP=1 WC_NULLMOVE=1
WC_PVS=1 WC_ASP=1 WC_NULLMOVE=1 WC_RFP=1
WC_PVS=1 WC_ASP=1 WC_NULLMOVE=1 WC_RFP=1 WC_FUTILITY=1 WC_LMP=1'
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"

run_match() {
  local label="$1" envprefix="$2" champ="$3" extra="$4"
  echo "=================================================================="
  echo "CONFIG: $envprefix    ($label)"
  # A = config under test (envprefix), B = champion.
  env $envprefix "$XM" "$BM $extra" "$champ" "$GAMES" 6 "$SEED" 2>/dev/null | tail -1
}

echo "########## EQUAL-DEPTH (d$DEPTH, $GAMES games, seed $SEED) ##########"
echo "$CONFIGS" | tr ';' '\n' | while IFS= read -r cfg; do
  [ -z "$cfg" ] && continue
  run_match "equal-depth d$DEPTH" "$cfg" "$CHAMP_DEPTH" "$DEPTH"
done

echo
echo "########## EQUAL-NODES (budget=$NODES, cap d12, $GAMES games) ##########"
echo "$CONFIGS" | tr ';' '\n' | while IFS= read -r cfg; do
  [ -z "$cfg" ] && continue
  run_match "equal-nodes $NODES" "$cfg" "$CHAMP_NODES" "12 $NODES"
done

echo
echo "done. A = config-under-test, B = deployed champion. A wins > B wins = improvement."
