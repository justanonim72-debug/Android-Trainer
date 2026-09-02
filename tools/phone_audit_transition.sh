#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
OUT="/storage/emulated/0/Download/model0001-transition-audit.json"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"

if [ -x "$PY" ]; then
  "$PY" "$SELF/audit_model0001_transition.py" --project "$PROJECT" --output "$OUT"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/audit_model0001_transition.py' --project '$PROJECT' --output '$OUT'
  "
else
  echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
  exit 1
fi

echo
echo "TRANSITION AUDIT READY:"
ls -lh "$OUT"
