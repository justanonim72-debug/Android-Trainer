#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
SRC="$PROJECT/data/f2_sft/friend_f2_sft_source.jsonl"

run_py() {
  local script="$1"; shift
  if [ -x "$PY" ]; then
    "$PY" "$script" "$@"
  elif command -v proot-distro >/dev/null 2>&1; then
    local quoted=""
    for arg in "$@"; do quoted="$quoted $(printf '%q' "$arg")"; done
    proot-distro login ubuntu -- bash -lc "
      set -euo pipefail
      PY=/root/mobilellm-ref/.venv/bin/python
      test -x \"\$PY\"
      \"\$PY\" '$script' $quoted
    "
  else
    echo "STOP: existing Ubuntu/PyTorch environment not available" >&2
    exit 1
  fi
}

echo "=== F2 SFT SOURCE ACQUISITION ==="
run_py "$SELF/acquire_model0001_f2_sft_sources.py" --project "$PROJECT"

echo
echo "=== F2 SFT SOURCE BUILD ==="
run_py "$SELF/build_model0001_f2_sft_source.py" --project "$PROJECT"

echo
echo "=== STRICT F2 SOURCE VALIDATION ==="
run_py "$SELF/validate_friend_f2_sft_jsonl.py" "$SRC"

echo
echo "=== ASSISTANT-ONLY SFT PACK ==="
run_py "$SELF/pack_model0001_f2_sft.py" --project "$PROJECT"

echo
echo "F2 SFT PACK READY:"
ls -lh "$PROJECT/artifacts/model0001_f2_sft/F2_SFT_PACK_REPORT.json"
echo "$PROJECT/artifacts/model0001_f2_sft/F2_SFT_PACK_REPORT.json"
