#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"

run_it() {
  "$1" "$SELF/promote_model0001_foundation_v3_checkpoint.py"
}

if [ -x "$PY" ]; then
  run_it "$PY"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/promote_model0001_foundation_v3_checkpoint.py'
  "
else
  echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
  exit 1
fi

echo
echo "FOUNDATION-v3 F2 SOURCE READY:"
ls -lh   /storage/emulated/0/Download/model0001-foundation-v3-source.atb   /storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json
