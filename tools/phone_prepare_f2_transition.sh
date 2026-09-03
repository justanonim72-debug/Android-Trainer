#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
SELF="$(cd "$(dirname "$0")" && pwd)"
OLD_BUNDLE="/storage/emulated/0/Download/model0001-gpu-gate.atb"
V3_CKPT="/storage/emulated/0/Download/model0001-foundation-v3-final.atnckpt"
V3_REPORT="/storage/emulated/0/Download/model0001-foundation-v3-stage-report.json"
F2_SOURCE="/storage/emulated/0/Download/model0001-foundation-v3-source.atb"
F2_AUDIT="/storage/emulated/0/Download/model0001-foundation-v3-source-bundle-audit.json"
F2_PILOT="/storage/emulated/0/Download/model0001-f2-sft-lr-pilot.atsftpilot"
PY="/root/mobilellm-ref/.venv/bin/python"

if [ ! -s "$OLD_BUNDLE" ]; then
  echo "Canonical CPT-v2 .atb missing; rebuilding local template..."
  bash "$SELF/phone_export_bundle.sh"
fi

for p in "$OLD_BUNDLE" "$V3_CKPT" "$V3_REPORT"; do
  test -s "$p" || { echo "STOP: required transition input missing: $p" >&2; exit 1; }
done

run_py_host_or_ubuntu() {
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
    echo "STOP: existing Ubuntu Python environment unavailable" >&2
    exit 1
  fi
}

echo "=== 1/5 PROMOTE FOUNDATION-v3 CHECKPOINT -> F2 SOURCE ==="
run_py_host_or_ubuntu   "$SELF/promote_model0001_foundation_v3_checkpoint.py"   --template-bundle "$OLD_BUNDLE"   --checkpoint "$V3_CKPT"   --stage-report "$V3_REPORT"   --output "$F2_SOURCE"   --audit-output "$F2_AUDIT"

echo
echo "=== 2/5 ACQUIRE HUMAN-FIRST F2 SOURCES ==="
run_py_host_or_ubuntu   "$SELF/acquire_model0001_f2_sft_sources.py"   --project "$PROJECT"

echo
echo "=== 3/5 BUILD + STRICT-VALIDATE F2 SOURCE ==="
run_py_host_or_ubuntu   "$SELF/build_model0001_f2_sft_source.py"   --project "$PROJECT"
SRC="$PROJECT/data/f2_sft/friend_f2_sft_source.jsonl"
run_py_host_or_ubuntu   "$SELF/validate_friend_f2_sft_jsonl.py" "$SRC"

echo
echo "=== 4/5 PACK ASSISTANT-ONLY F2 WINDOWS ==="
run_py_host_or_ubuntu   "$SELF/pack_model0001_f2_sft.py"   --project "$PROJECT"

echo
echo "=== 5/5 BUILD PHYSICAL F2 LR PILOT PACKAGE ==="
run_py_host_or_ubuntu   "$SELF/build_model0001_f2_sft_lr_pilot.py"   --project "$PROJECT"   --source-audit "$F2_AUDIT"   --output "$F2_PILOT"

echo
echo "F2 TRANSITION READY ✅"
ls -lh   "$F2_SOURCE"   "$F2_AUDIT"   "$PROJECT/artifacts/model0001_f2_sft/F2_SFT_PACK_REPORT.json"   "$F2_PILOT"
echo
echo "Files for Android:"
echo "  source: $F2_SOURCE"
echo "  pilot : $F2_PILOT"
