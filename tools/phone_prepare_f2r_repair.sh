#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
BEHAVIOR="/storage/emulated/0/Download/model0001-post-f2-behavior-eval.json"
AUDIT="/storage/emulated/0/Download/model0001-f2-collapse-audit.json"

run_py() {
  local script="$1"; shift
  if [ -x "$PY" ]; then
    "$PY" "$script" "$@"
  elif command -v proot-distro >/dev/null 2>&1; then
    local qargs=""
    for a in "$@"; do qargs="$qargs $(printf '%q' "$a")"; done
    proot-distro login ubuntu -- bash -lc "
      set -euo pipefail
      PY=/root/mobilellm-ref/.venv/bin/python
      test -x \"\$PY\"
      \"\$PY\" '$script' $qargs
    "
  else
    echo "STOP: existing Ubuntu/Python environment unavailable" >&2
    exit 1
  fi
}

echo "=== 1/4 EMPIRICAL F2 COLLAPSE AUDIT ==="
run_py "$SELF/audit_model0001_f2_behavior_collapse.py"   --project "$PROJECT"   --behavior-report "$BEHAVIOR"   --output "$AUDIT"

echo
echo "=== 2/4 BUILD F2R REPLACEMENT SOURCE ==="
run_py "$SELF/build_model0001_f2r_repair_source.py"   --project "$PROJECT"   --collapse-audit "$AUDIT"

echo
echo "=== 3/4 STRICT SOURCE VALIDATION ==="
SRC="$PROJECT/data/f2r_repair/friend_f2r_repair_source.jsonl"
run_py "$SELF/validate_friend_f2_sft_jsonl.py" "$SRC"

echo
echo "=== 4/4 RECORD-ISOLATED ASSISTANT-ONLY PACK ==="
run_py "$SELF/pack_model0001_f2r_repair.py"   --project "$PROJECT"

echo
echo "F2R REPAIR DATA READY ✅"
ls -lh   "$AUDIT"   "$PROJECT/data/f2r_repair/F2R_SOURCE_REPORT.json"   "$PROJECT/artifacts/model0001_f2r_repair/F2R_PACK_REPORT.json"
echo
echo "SEND THESE REPORTS:"
echo "  $AUDIT"
echo "  $PROJECT/data/f2r_repair/F2R_SOURCE_REPORT.json"
echo "  $PROJECT/artifacts/model0001_f2r_repair/F2R_PACK_REPORT.json"
