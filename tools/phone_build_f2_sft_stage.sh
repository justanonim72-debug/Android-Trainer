#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
RECIPE="/storage/emulated/0/Download/model0001-f2-sft-production-recipe.json"
SOURCE_AUDIT="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json"
OUT="/storage/emulated/0/Download/model0001-f2-sft.atsftstage"

run_builder() {
  "$1" "$SELF/build_model0001_f2_sft_stage_package.py" \
    --project "$PROJECT" \
    --source-audit "$SOURCE_AUDIT" \
    --recipe "$RECIPE" \
    --output "$OUT"
}

if [ -x "$PY" ]; then
  run_builder "$PY"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/build_model0001_f2_sft_stage_package.py' \
      --project '$PROJECT' \
      --source-audit '$SOURCE_AUDIT' \
      --recipe '$RECIPE' \
      --output '$OUT'
  "
else
  echo "STOP: existing Ubuntu Python environment unavailable" >&2
  exit 1
fi

echo
echo "F2 SFT PRODUCTION PACKAGE READY:"
ls -lh "$RECIPE" "$OUT"
echo "$OUT"
