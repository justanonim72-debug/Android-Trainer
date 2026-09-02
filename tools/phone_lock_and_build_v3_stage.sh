#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PILOT="/storage/emulated/0/Download/model0001-v3-lr-pilot-report.json"
RECIPE="/storage/emulated/0/Download/model0001-v3-production-recipe.json"
STAGE="/storage/emulated/0/Download/model0001-foundation-v3.atstage"
AUDIT="/storage/emulated/0/Download/model0001-transition-audit.json"
DATASET="$PROJECT/artifacts/model0001_dataset_v3/DATASET_V3_REPORT.json"
PY="/root/mobilellm-ref/.venv/bin/python"

run_inside() {
  local exe="$1"
  "$exe" "$SELF/lock_model0001_v3_production_recipe.py"     --pilot-report "$PILOT"     --output "$RECIPE"

  "$exe" "$SELF/build_model0001_stage_package.py"     --audit "$AUDIT"     --dataset-report "$DATASET"     --recipe "$RECIPE"     --project "$PROJECT"     --output "$STAGE"
}

if [ -x "$PY" ]; then
  run_inside "$PY"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/lock_model0001_v3_production_recipe.py'       --pilot-report '$PILOT'       --output '$RECIPE'
    \"\$PY\" '$SELF/build_model0001_stage_package.py'       --audit '$AUDIT'       --dataset-report '$DATASET'       --recipe '$RECIPE'       --project '$PROJECT'       --output '$STAGE'
  "
else
  echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
  exit 1
fi

echo
echo "FOUNDATION-v3 PRODUCTION READY:"
ls -lh "$RECIPE" "$STAGE"
echo
echo "Import this file with button 7:"
echo "$STAGE"
