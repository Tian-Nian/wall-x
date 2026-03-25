# WALL-OSS-FLOW Finetune

## Environment

```bash
cd /mnt/pfs/pg4hw0/niantian/wall-x
bash -ic 'source /mnt/pfs/pg4hw0/niantian/.bashrc; enableProxy; source .venv/bin/activate; python -V'
```

## 1. Download pretrained weights

Model path:

```text
/mnt/pfs/pg4hw0/niantian/model_weights/wall-oss-flow
```

If you need to resume or re-download:

```bash
cd /mnt/pfs/pg4hw0/niantian/wall-x
bash -ic 'source /mnt/pfs/pg4hw0/niantian/.bashrc; enableProxy; source .venv/bin/activate; python - <<'"'"'PY'"'"'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="x-square-robot/wall-oss-flow",
    local_dir="/mnt/pfs/pg4hw0/niantian/model_weights/wall-oss-flow",
)
PY'
```

## 2. Prepare norm stats

Training requires a dataset stats JSON. For the default dataset:

```bash
cd /mnt/pfs/pg4hw0/niantian/wall-x
bash -ic 'source /mnt/pfs/pg4hw0/niantian/.bashrc; enableProxy; source .venv/bin/activate; python scripts/compute_norm_stats.py \
  --repo-id lerobot/aloha_mobile_cabinet \
  --root /mnt/pfs/pg4hw0/niantian/lerobot \
  --output /mnt/pfs/pg4hw0/niantian/wall-x/workspace/lerobot_example/stats/aloha_mobile_cabinet_norm_stats.json'
```

If you use another LeRobot dataset, replace `--repo-id` and output path accordingly.

## 3. Verify training config

`workspace/lerobot_example/config_qact.yml` is set for FLOW finetuning:

- `pretrained_wallx_path=/mnt/pfs/pg4hw0/niantian/model_weights/wall-oss-flow`
- `save_path=/mnt/pfs/pg4hw0/niantian/wall-x/workspace/outputs/wall_oss_flow_ft`
- `use_fast_tokenizer=False`
- `norm_stats_path=/mnt/pfs/pg4hw0/niantian/wall-x/workspace/lerobot_example/stats/aloha_mobile_cabinet_norm_stats.json`

If your robot DoF or dataset mapping differs from `lerobot/aloha_mobile_cabinet`, update:

- `dof_config`
- `agent_pos_config`
- `customized_robot_config`
- `data.lerobot_config.repo_id`

## 4. Start finetuning

```bash
cd /mnt/pfs/pg4hw0/niantian/wall-x
bash -ic 'source /mnt/pfs/pg4hw0/niantian/.bashrc; source .venv/bin/activate; bash ./workspace/lerobot_example/run.sh'
```

## 5. Notes

- `workspace/lerobot_example/run.sh` already points to the local repo path.
- The default script uses `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
- FLOW finetuning does not need the FAST tokenizer repo.
- If you change datasets, regenerate `norm_stats.json` before training.
