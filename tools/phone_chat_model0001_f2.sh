#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"

if [ -x "$PY" ]; then
  exec "$PY" "$SELF/chat_model0001_f2.py"
elif command -v proot-distro >/dev/null 2>&1; then
  exec proot-distro login ubuntu -- bash -lc "
    cd '$SELF'
    exec /root/mobilellm-ref/.venv/bin/python '$SELF/chat_model0001_f2.py'
  "
else
  echo "STOP: existing Ubuntu/Python environment unavailable" >&2
  exit 1
fi
