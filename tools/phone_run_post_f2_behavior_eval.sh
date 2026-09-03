#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
OUT="/storage/emulated/0/Download/model0001-post-f2-behavior-eval.json"

run_eval() {
  "$1" "$SELF/run_model0001_post_f2_behavior_eval.py"
}

if [ -x "$PY" ]; then
  run_eval "$PY"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    cd '$SELF'
    \"\$PY\" '$SELF/run_model0001_post_f2_behavior_eval.py'
  "
else
  echo "STOP: existing Ubuntu/Python environment unavailable" >&2
  exit 1
fi

echo
echo "POST-F2 BEHAVIOR EVAL READY:"
ls -lh "$OUT"
echo
echo "Kirim file ini ke project Training:"
echo "$OUT"
