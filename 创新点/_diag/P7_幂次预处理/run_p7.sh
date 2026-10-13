#!/bin/bash
# R0 only on corel5k. Rγ is run locally by the user and is not started here.
# Hyperparameters stay at main-final.py defaults.
set -u
BASE="/workspace/创新点/_diag/P7_幂次预处理"
WORKDIR="$BASE/workdir_rank"
LOG="$BASE/run_log.md"
mkdir -p "$BASE/logs" "$BASE/records_R0" "$BASE/records_Rgamma"
cd "$WORKDIR"
export PYTHONUNBUFFERED=1
# Do not export PYTHONHASHSEED / torch / numpy seeds. main-final.py does not set one.
export RANK_MAX_WALL_SEC="${RANK_MAX_WALL_SEC:-43200}"

python3 - << 'PY'
import os, platform, shutil, datetime
import torch
base = "/workspace/创新点/_diag/P7_幂次预处理"
path = os.path.join(base, "run_log.md")
gpu = "none"
cuda = torch.cuda.is_available()
if cuda:
    gpu = torch.cuda.get_device_name(0)
text = """# P7 run log

- started_utc: {ts}
- python: {py}
- torch: {torch}
- cuda_available: {cuda}
- gpu: {gpu}
- nvidia-smi: {smi}
- RANK tree: copy of https://github.com/justsmart/RANK commit 5635dceb8bb66d19b017a438489a4d5b29eb9a2f
- data mats used (official repo blobs, not the README Google Drive link):
  - {base}/workdir_rank/data/corel5k/corel5k_six_view.mat
  - {base}/workdir_rank/data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat
- hyperparameters: main-final.py corel5k lists and argparse defaults, not overridden
  - lr_list = [1e-1]
  - alpha_list = [1e0]
  - beta_list = [0.97]
  - gamma_list = [1e-1]
  - sigma_list = [0]
  - epochs = 250
  - folds_num = 10
  - batch_size = 128
  - n_z = 512
  - n_stacks = 4
  - SGD momentum = 0.9
  - mask_view_ratio = 0.5
  - mask_label_ratio = 0.5
  - training_sample_ratio = 0.7
  - datasets = ['corel5k']
- seed: argparse default exists (1) but main never calls a seed setter; this run does not set one
- device: cuda:0 if torch.cuda.is_available() else cpu
- RANK_MAX_WALL_SEC: {wall}
- records_R0: {base}/records_R0
- records_Rgamma: {base}/records_Rgamma
- metrics_R0: {base}/metrics_R0.csv
- metrics_Rgamma: {base}/metrics_Rgamma.csv

""".format(
    ts=datetime.datetime.utcnow().isoformat() + "Z",
    py=platform.python_version(),
    torch=torch.__version__,
    cuda=cuda,
    gpu=gpu,
    smi=shutil.which("nvidia-smi") or "absent",
    base=base,
    wall=os.environ.get("RANK_MAX_WALL_SEC", ""),
)
with open(path, "w") as handle:
    handle.write(text)
print(text)
PY

echo "## preflight" >> "$LOG"
if ! python3 "$BASE/preflight_preprocess.py" >> "$LOG" 2>&1; then
  echo "- status: blocked_preflight" >> "$LOG"
  python3 "$BASE/summarize_p7.py" >> "$LOG" 2>&1 || true
  exit 1
fi

run_arm() {
  local name="$1"
  local mode="$2"
  export RANK_PREPROCESS="$mode"
  export RANK_FOLD_METRICS_CSV="$BASE/metrics_${name}.csv"
  : > "$RANK_FOLD_METRICS_CSV"
  echo "## arm ${name} RANK_PREPROCESS=${mode}" >> "$LOG"
  echo "- start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  local rc=0
  python3 main-final.py --records-dir "$BASE/records_${name}" > "$BASE/logs/${name}.stdout" 2>&1 || rc=$?
  echo "- exit_code: ${rc}" >> "$LOG"
  echo "- end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  echo "- stdout: ${BASE}/logs/${name}.stdout" >> "$LOG"
  return "$rc"
}

rc_all=0
run_arm R0 r0 || rc_all=$?
r0_rows=0
if [[ -f "$BASE/metrics_R0.csv" ]]; then
  r0_rows=$(python3 - << 'PY'
import os
p="/workspace/创新点/_diag/P7_幂次预处理/metrics_R0.csv"
n=0
if os.path.exists(p):
    with open(p) as h:
        n=max(0, sum(1 for _ in h)-1)
print(n)
PY
)
fi
echo "- completed_folds_R0: ${r0_rows}" >> "$LOG"
echo "- Rgamma: run locally, pending" >> "$LOG"
python3 "$BASE/summarize_p7.py" >> "$LOG" 2>&1 || true
echo "- run_p7_exit: ${rc_all}" >> "$LOG"
exit "$rc_all"
