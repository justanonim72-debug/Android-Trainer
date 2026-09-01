#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
OUT="/storage/emulated/0/Download/model0001-gpu-gate.atb"
CHECKPOINT="$PROJECT/artifacts/model0001_runs/model0001_cpt_v2_epoch1/latest.pt"
EXPECTED_MODEL_SHA="047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d"

SELF="$(cd "$(dirname "$0")" && pwd)/export_model0001_bundle.py"

if [ ! -f "$SELF" ]; then
  echo "STOP: export_model0001_bundle.py must be beside this script." >&2
  exit 1
fi
if [ ! -d "$PROJECT" ]; then
  echo "STOP: Friend-Core project not found: $PROJECT" >&2
  exit 1
fi

echo "Android-Trainer Model #0001 CPT-v2 bundle export"
echo "Expected final model SHA: $EXPECTED_MODEL_SHA"
echo "Project: $PROJECT"
echo "Output : $OUT"
echo "Checkpoint: $CHECKPOINT"
test -f "$CHECKPOINT" || { echo "STOP: final CPT-v2 latest.pt missing: $CHECKPOINT" >&2; exit 1; }

run_inside_ubuntu() {
  local exporter="$1"
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\" || { echo 'STOP: Ubuntu PyTorch venv missing: /root/mobilellm-ref/.venv/bin/python' >&2; exit 1; }
    \"\$PY\" '$exporter' --project '$PROJECT' --checkpoint '$CHECKPOINT' --output '$OUT'
  "
}

if [ -x /root/mobilellm-ref/.venv/bin/python ]; then
  /root/mobilellm-ref/.venv/bin/python "$SELF" --project "$PROJECT" --checkpoint "$CHECKPOINT" --output "$OUT"
elif command -v proot-distro >/dev/null 2>&1; then
  run_inside_ubuntu "$SELF"
else
  echo "STOP: run this from the Ubuntu environment, or install/use the existing proot-distro Ubuntu launcher." >&2
  exit 1
fi

test -s "$OUT"
echo
echo "BUNDLE READY ✅"
ls -lh "$OUT"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT"
fi
