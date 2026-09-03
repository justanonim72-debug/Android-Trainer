#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
PY="/root/mobilellm-ref/.venv/bin/python"
SOURCE_AUDIT="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json"
OUT="/storage/emulated/0/Download/model0001-f2r-lr-pilot.atsftpilot"
REPORT="/storage/emulated/0/Download/model0001-f2r-lr-pilot-package-report.json"

if [ -x "$PY" ]; then
  "$PY" "$SELF/build_model0001_f2r_lr_pilot.py"     --project "$PROJECT" --source-audit "$SOURCE_AUDIT"     --output "$OUT" --report-output "$REPORT"
elif command -v proot-distro >/dev/null 2>&1; then
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\"
    \"\$PY\" '$SELF/build_model0001_f2r_lr_pilot.py'       --project '$PROJECT' --source-audit '$SOURCE_AUDIT'       --output '$OUT' --report-output '$REPORT'
  "
else
  echo "STOP: existing Ubuntu/Python environment unavailable" >&2
  exit 1
fi

echo
echo "F2R LR PILOT READY ✅"
ls -lh "$OUT" "$REPORT"
echo "Import with trainer button 10, then run button 11."
