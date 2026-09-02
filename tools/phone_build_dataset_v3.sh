#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
OUT="$PROJECT/artifacts/model0001_dataset_v3/DATASET_V3_REPORT.json"

if [ -x "$PY" ]; then
  "$PY" "$SELF/build_model0001_dataset_v3.py" --project "$PROJECT"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/build_model0001_dataset_v3.py' --project '$PROJECT'
  "
else
  echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
  exit 1
fi

echo
echo "DATASET-v3 BUILD READY:"
ls -lh "$OUT"
echo
echo "Kirim file ini ke project Training:"
echo "$OUT"
