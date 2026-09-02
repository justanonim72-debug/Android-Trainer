#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"

run_py() {
  local script="$1"
  if [ -x "$PY" ]; then
    "$PY" "$script" --project "$PROJECT"
  elif command -v proot-distro >/dev/null 2>&1; then
    proot-distro login ubuntu -- bash -lc "
      set -euo pipefail
      PY=/root/mobilellm-ref/.venv/bin/python
      test -x \"\$PY\"
      \"\$PY\" '$script' --project '$PROJECT'
    "
  else
    echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
    exit 1
  fi
}

echo "=== MODEL #0001 DATASET-v3 SOURCE ACQUISITION ==="
run_py "$SELF/acquire_model0001_v3_sources.py"

echo
echo "=== NORMALIZE + DEDUPE + FROZEN TOKENIZER COUNT ==="
run_py "$SELF/prepare_model0001_v3_pool.py"

OUT="$PROJECT/data/corpus_v3_candidates/CANDIDATE_POOL_AUDIT.json"

echo
echo "V3 CANDIDATE POOL AUDIT READY:"
ls -lh "$OUT"
echo
echo "Kirim JSON ini balik ke project Training:"
echo "$OUT"
