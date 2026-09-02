#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)"
PROJECT="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1"
OUT="/storage/emulated/0/Download/model0001-cpu-reference-benchmark.json"

run_bench() {
  local script="$1"
  proot-distro login ubuntu -- bash -lc "
    set -euo pipefail
    PY=/root/mobilellm-ref/.venv/bin/python
    test -x \"\$PY\" || { echo 'STOP: Ubuntu PyTorch venv missing' >&2; exit 1; }
    \"\$PY\" '$script' --project '$PROJECT' --output '$OUT' --warmup 1 --steps 20
  "
}

if [ -x /root/mobilellm-ref/.venv/bin/python ]; then
  /root/mobilellm-ref/.venv/bin/python "$SELF/benchmark_model0001_cpu_reference.py"     --project "$PROJECT" --output "$OUT" --warmup 1 --steps 20
elif command -v proot-distro >/dev/null 2>&1; then
  run_bench "$SELF/benchmark_model0001_cpu_reference.py"
else
  echo "STOP: existing Ubuntu/PyTorch environment unavailable" >&2
  exit 1
fi

echo
echo "CPU REFERENCE BENCHMARK READY:"
cat "$OUT"
