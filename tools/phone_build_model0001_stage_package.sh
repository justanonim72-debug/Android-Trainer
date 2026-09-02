#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
AUDIT="/storage/emulated/0/Download/model0001-transition-audit.json"
DATASET="$PROJECT/artifacts/model0001_dataset_v3/DATASET_V3_REPORT.json"
RECIPE="/storage/emulated/0/Download/model0001-v3-production-recipe.json"
OUT="/storage/emulated/0/Download/model0001-foundation-v3.atstage"

run_cmd() {
  "$1" "$SELF/build_model0001_stage_package.py" \
    --audit "$AUDIT" \
    --dataset-report "$DATASET" \
    --recipe "$RECIPE" \
    --project "$PROJECT" \
    --output "$OUT"
}

if [ -x "$PY" ]; then
  run_cmd "$PY"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/build_model0001_stage_package.py' \
      --audit '$AUDIT' \
      --dataset-report '$DATASET' \
      --recipe '$RECIPE' \
      --project '$PROJECT' \
      --output '$OUT'
  "
else
  echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
  exit 1
fi

echo
echo "FOUNDATION-v3 PRODUCTION PACKAGE READY:"
ls -lh "$OUT"
echo "$OUT"
