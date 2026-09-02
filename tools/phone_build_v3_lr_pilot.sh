#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
OUT="/storage/emulated/0/Download/model0001-v3-lr-pilot.atpilot"

if [ -x "$PY" ]; then
  "$PY" "$SELF/build_model0001_v3_lr_pilot.py" --project "$PROJECT" --output "$OUT"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/build_model0001_v3_lr_pilot.py' --project '$PROJECT' --output '$OUT'
  "
else
  echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
  exit 1
fi

echo
echo "FOUNDATION-v3 LR PILOT PACKAGE READY:"
ls -lh "$OUT"
echo "$OUT"
