#!/usr/bin/env bash
set -euo pipefail

SWEEP_JSON="${1:-sweep.json}"
PY="./src/main.py"
MAX_PARALLEL="${MAX_PARALLEL:-1}"   # override: MAX_PARALLEL=3 ./sweep.sh
DRY_RUN="${DRY_RUN:-0}"             # override: DRY_RUN=1 ./sweep.sh

# --- checks ---
command -v jq >/dev/null || { echo "Please install jq"; exit 1; }
[ -f "$SWEEP_JSON" ] || { echo "No $SWEEP_JSON"; exit 1; }
BASE_CFG="$(jq -r '.base_config' "$SWEEP_JSON")"
[ -f "$BASE_CFG" ] || { echo "Base config not found: $BASE_CFG"; exit 1; }
[ -f "$PY" ] || { echo "main.py not found at $PY"; exit 1; }

# --- jq helpers: deep-merge and param tag (to match main.py’s tag) ---
JQ_DEEP_MERGE='
def rmerge($a; $b):
  reduce ($b | keys_unsorted[]) as $k
    ($a;
      if ( $a[$k]|type ) == "object" and ( $b[$k]|type ) == "object"
      then .[$k] = rmerge($a[$k]; $b[$k])
      else .[$k] = $b[$k]
      end
    );
'

JQ_PARAM_TAG='
def s($k; $v):
  if $v == null then empty else ($k + ($v|tostring)) end;
def make_tag:
  .mcts_params as $m |
  if $m == null or $m == {} then "mcts_default"
  else
    ("mcts_" + ([ s("pd"; $m.planning_depth),
                  s("ni"; $m.num_iterations),
                  s("uc"; $m.ucb1_c),
                  s("df"; $m.discount_factor),
                  s("to"; $m.timeout),
                  s("pa"; $m.parallel) ]
       | map(select(length>0)) | join("_")))
  end;
'

# concurrency helper
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 0.2; done
}

run_one() {
  local idx="$1"

  # 1) pull experiment name + overrides
  local NAME OVERRIDES
  NAME="$(jq -r ".experiments[$idx].name // (\"exp_\" + ($idx|tostring))" "$SWEEP_JSON")"
  OVERRIDES="$(jq ".experiments[$idx].overrides" "$SWEEP_JSON")"

  # 2) deep-merge base + overrides -> merged json (tmp file)
  local MERGED
  MERGED="$(mktemp)"
  jq -n --slurpfile base "$BASE_CFG" --argjson over "$OVERRIDES" \
     "$JQ_DEEP_MERGE rmerge(\$base[0]; \$over)" > "$MERGED"

  # 3) compute the results root exactly like main.py
  # results_root = project_path/trials/<field_type_lower>_<start_position>[__mcts_tag]
  local PROJECT FIELD START STRAT PPATH
  PROJECT="$(jq -r '.project_path' "$MERGED")"; PROJECT="${PROJECT%/}"
  FIELD="$(jq -r '.field_type' "$MERGED" | tr '[:upper:]' '[:lower:]')"
  START="$(jq -r '.start_position' "$MERGED")"
  STRAT="$(jq -r '.action_strategy' "$MERGED")"
  PPATH="$(jq -r '.params_in_path // true' "$MERGED")"

  local RUN_BASE TAG RESULTS_ROOT
  RUN_BASE="${FIELD}_${START}"
  if [[ "$STRAT" == "mcts" && "$PPATH" == "true" ]]; then
    TAG="$(jq -r "$JQ_PARAM_TAG make_tag" "$MERGED")"
    RUN_BASE="${RUN_BASE}__${TAG}"
  fi
  RESULTS_ROOT="${PROJECT}/trials/${RUN_BASE}"

  # 4) place a copy of the merged config INSIDE the results root
  mkdir -p "$RESULTS_ROOT"
  local CFG_OUT LOG_OUT
  CFG_OUT="${RESULTS_ROOT}/config_${NAME}.json"
  LOG_OUT="${RESULTS_ROOT}/run_${NAME}.log"
  cp "$MERGED" "$CFG_OUT"

  echo "[$(date +%H:%M:%S)] ▶ ${NAME}"
  echo "  config → $CFG_OUT"
  echo "  logs   → $LOG_OUT"
  echo "  folder → $RESULTS_ROOT"

  # 5) run main with that config
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry-run) python $PY --config \"$CFG_OUT\""
  else
    python "$PY" --config "$CFG_OUT" >"$LOG_OUT" 2>&1 &
  fi

  rm -f "$MERGED"
}

COUNT="$(jq '.experiments | length' "$SWEEP_JSON")"
if [[ "$COUNT" -eq 0 ]]; then
  echo "No experiments in $SWEEP_JSON (.experiments is empty)"; exit 1
fi

echo "Found $COUNT experiments. Parallel jobs: $MAX_PARALLEL"
for i in $(seq 0 $((COUNT-1))); do
  wait_for_slot
  run_one "$i"
done

# wait for all background jobs if not dry-run
if [[ "$DRY_RUN" != "1" ]]; then
  wait
  echo "All experiments finished."
fi
