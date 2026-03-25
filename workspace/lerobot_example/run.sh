#!/bin/bash
set -euo pipefail

user_bashrc="/mnt/pfs/pg4hw0/niantian/.bashrc"
code_dir="/mnt/pfs/pg4hw0/niantian/wall-x_bak"
venv_dir="${code_dir}/.venv"
python_bin="${venv_dir}/bin/python"
config_path="${code_dir}/workspace/lerobot_example"

if [[ -f "${user_bashrc}" ]]; then
    set +u
    source "${user_bashrc}"
    set -u
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Missing virtual environment: ${venv_dir}" >&2
    exit 1
fi

torch_lib_dir=$(echo "${venv_dir}"/lib/python*/site-packages/torch/lib)
if [[ -d "${torch_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${torch_lib_dir}:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${code_dir}/lerobot/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# print current time
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

# Use a fixed port instead of a random one
export PORT=$((21000 + $RANDOM % 30000))

MASTER_PORT=10239 # use 5 digits ports

export LAUNCHER="${python_bin} -m accelerate.commands.launch --num_processes=$NUM_GPUS --main_process_port=$PORT"

export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}/config_qact.yml --seed $MASTER_PORT"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"

cd "${code_dir}"
$LAUNCHER $SCRIPT $SCRIPT_ARGS
