#!/bin/bash
set -euo pipefail

# user_bashrc="/mnt/pfs/pg4hw0/niantian/.bashrc"
code_dir="/mnt/pfs/pg4hw0/niantian/wall-x"
venv_dir="${code_dir}/.venv"
config_path="${code_dir}/workspace/lerobot_example"

# if [[ -f "${user_bashrc}" ]]; then
#     source "${user_bashrc}"
# fi

if [[ -f "${venv_dir}/bin/activate" ]]; then
    source "${venv_dir}/bin/activate"
else
    echo "Missing virtual environment: ${venv_dir}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# print current time
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

# Use a fixed port instead of a random one
export PORT=$((21000 + $RANDOM % 30000))

MASTER_PORT=10239 # use 5 digits ports

export LAUNCHER="accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT"

export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}/config_qact.yml --seed $MASTER_PORT"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"

cd "${code_dir}"
$LAUNCHER $SCRIPT $SCRIPT_ARGS
