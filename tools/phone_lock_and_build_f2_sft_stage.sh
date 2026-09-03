#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PILOT="/storage/emulated/0/Download/model0001-f2-sft-lr-pilot-report.json"
RECIPE="/storage/emulated/0/Download/model0001-f2-sft-production-recipe.json"
STAGE="/storage/emulated/0/Download/model0001-f2-sft.atsftstage"
SOURCE_AUDIT="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json"
PY="/root/mobilellm-ref/.venv/bin/python"

run_inside() {
  local exe="$1"
  "$exe" "$SELF/lock_model0001_f2_sft_production_recipe.py" \
    --pilot-report "$PILOT" \
    --project "$PROJECT" \
    --output "$RECIPE"

  "$exe" "$SELF/build_model0001_f2_sft_stage_package.py" \
    --project "$PROJECT" \
    --source-audit "$SOURCE_AUDIT" \
    --recipe "$RECIPE" \
    --output "$STAGE"
}

if [ -x "$PY" ]; then
  run_inside "$PY"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/lock_model0001_f2_sft_production_recipe.py' \
      --pilot-report '$PILOT' \
      --project '$PROJECT' \
      --output '$RECIPE'
    \"\$PY\" '$SELF/build_model0001_f2_sft_stage_package.py' \
      --project '$PROJECT' \
      --source-audit '$SOURCE_AUDIT' \
      --recipe '$RECIPE' \
      --output '$STAGE'
  "
else
  echo "STOP: existing Ubuntu/Python environment unavailable" >&2
  exit 1
fi

echo
echo "F2 SFT PRODUCTION READY ✅"
ls -lh "$RECIPE" "$STAGE"
echo
echo "Import with button 12:"
echo "$STAGE"
