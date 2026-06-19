#!/usr/bin/env bash
# Eval A/B harness — Gen-2 eval vs the deployed-default "previous generation".
#
# The new eval is selected entirely through WC_EVAL* env vars (read by
# Heuristic::from_env in bestmove). The champion runs with every WC_EVAL* var
# stripped, so it is exactly the deployed eval (w_path=50, w_wall=100, no
# endgame resolution) regardless of the caller's environment.
#
# A = candidate eval (the env prefix under test). B = champion. Both engines
# run at the SAME depth, so a win is attributable to the EVAL, not to depth.
# Sides alternate inside xmatch; we additionally sweep several seeds and sum,
# because equal-strength engines are noisy over a single seed (d4-vs-d4 came
# out 18-12 on one seed). Trust only the multi-seed aggregate.
#
# Usage:
#   core/scripts/eval_tune.sh [games_per_seed] [depth] [seeds...]
# Defaults: games_per_seed=50 depth=4 seeds="1 2 3 4"  (=> 200 games/config)
#
# Override the config list (newline- or semicolon-separated env prefixes):
#   CONFIGS="WC_EVAL=v2; WC_EVAL=v2 WC_EVAL_ENDGAME=0" core/scripts/eval_tune.sh
#
# NOTE: always run the FULL game count. Partial favorable reads are sampling
# bias. The gate is >=70% over >=100 games.

set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-50}"
DEPTH="${2:-4}"
shift || true; shift || true
SEEDS="${*:-1 2 3 4}"

# Strip every WC_EVAL_* var for the champion so it can never inherit a candidate
# eval. With all stripped, bestmove's Heuristic::from_env() == the deployed default.
CLEAN="env -u WC_EVAL_W_PATH -u WC_EVAL_W_WALL -u WC_EVAL_W_LEAD_QUANT \
  -u WC_EVAL_EXACT_ENDGAME -u WC_EVAL_ENDGAME_MARGIN"

echo "building release bestmove + xmatch ..."
cargo build --release --bin bestmove --bin xmatch >/dev/null 2>&1
BM="$(pwd)/target/release/bestmove"
XM="$(pwd)/target/release/xmatch"

CHAMP="$CLEAN $BM $DEPTH"

# Tuning grid (most-promising first). #1 isolates the centerpiece; layer extras
# only if it clears the gate. Each line is the WC_EVAL_* env prefix for engine A.
DEFAULT_CONFIGS='WC_EVAL_EXACT_ENDGAME=1
WC_EVAL_EXACT_ENDGAME=1 WC_EVAL_ENDGAME_MARGIN=1
WC_EVAL_EXACT_ENDGAME=1 WC_EVAL_W_LEAD_QUANT=20
WC_EVAL_EXACT_ENDGAME=1 WC_EVAL_W_LEAD_QUANT=30
WC_EVAL_EXACT_ENDGAME=1 WC_EVAL_W_WALL=120
WC_EVAL_EXACT_ENDGAME=1 WC_EVAL_W_PATH=60 WC_EVAL_W_WALL=120'
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"

echo "########## EVAL A/B  d$DEPTH  games/seed=$GAMES  seeds=[$SEEDS] ##########"
echo "champion (B) = deployed default eval (w_path=50 w_wall=100, no endgame)"
echo

printf '%-60s %6s %6s %6s %7s\n' "CONFIG (A)" "A" "B" "draw" "A%"
echo "------------------------------------------------------------------------------------"

echo "$CONFIGS" | tr ';' '\n' | while IFS= read -r cfg; do
  [ -z "$cfg" ] && continue
  cfg="$(echo "$cfg" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -z "$cfg" ] && continue
  aw=0; bw=0; dr=0
  for s in $SEEDS; do
    line="$(env $cfg "$XM" "$BM $DEPTH" "$CHAMP" "$GAMES" 6 "$s" 2>/dev/null | tail -1)"
    # TOTAL A [cmdA] <aw> - <bw> B [cmdB]  draws=<dr> ...   (cmdA has no ']')
    a="$(echo "$line" | sed -n 's/.*A \[[^]]*\] \([0-9]*\) - [0-9]* B .*/\1/p')"
    b="$(echo "$line" | sed -n 's/.*A \[[^]]*\] [0-9]* - \([0-9]*\) B .*/\1/p')"
    d="$(echo "$line" | sed -n 's/.*draws=\([0-9]*\).*/\1/p')"
    aw=$((aw + ${a:-0})); bw=$((bw + ${b:-0})); dr=$((dr + ${d:-0}))
  done
  tot=$((aw + bw))
  pct=0; [ "$tot" -gt 0 ] && pct=$(( aw * 100 / tot ))
  printf '%-60s %6s %6s %6s %6s%%\n' "$cfg" "$aw" "$bw" "$dr" "$pct"
done

echo
echo "A = candidate eval, B = deployed champion (same depth). A% = decisive-game win rate."
echo "GATE: A% >= 70 over >= 100 decisive games to qualify as Gen-2."
