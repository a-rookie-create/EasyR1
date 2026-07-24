# EasyR1 UI-S1 操作命令

本页只保留日常操作所需命令。训练逻辑、参数说明、日志字段和显存含义见 [EasyR1/docs/ui_s1_training.md](EasyR1/docs/ui_s1_training.md)。

## 1. 一次性环境准备

宿主机确认 GPU：

```bash
nvidia-smi
```

首次在新服务器配置运行时路径：

```bash
cd /data5/zst/biye210
cp EasyR1/examples/ui_s1/runtime.env.example EasyR1/examples/ui_s1/runtime.env
# 编辑 EasyR1/examples/ui_s1/runtime.env，填写 PROJECT_ROOT、OUTPUT_ROOT、RAY 等服务器路径。
```

容器不存在时创建；已存在时只需 `docker restart easyr1`：

```bash
docker run -d \
  --name easyr1 \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -p 8265:8265 \
  -v /data5/zst/biye210:/data5/zst/biye210 \
  -w /data5/zst/biye210/EasyR1 \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 \
  sleep infinity

docker exec easyr1 python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

预期最后一条输出为 `True 4`。

## 2. 生成 5% RL 数据（首次或数据变更后）

```bash
cd /data5/zst/biye210/EasyR1
set -a; source examples/ui_s1/runtime.env; set +a

python3 -B examples/ui_s1/prepare_ui_s1_android_control_rl_data.py \
  --android-control-dir /data5/zst/biye210/datasets/android_control \
  --image-dir /data5/zst/biye210/llamafactory/data/ui_s1_android_control_sft/images \
  --output-dir /data5/zst/biye210/EasyR1/datasets/android_control_5pct \
  --sample-ratio 0.05 --dataset-name android_control_5pct --seed 42

python3 -B examples/ui_s1/prepare_ui_s1_amex_rl_data.py \
  --amex-dir /data5/zst/biye210/datasets/amex \
  --output-dir /data5/zst/biye210/EasyR1/datasets/amex_5pct \
  --sample-ratio 0.05 --dataset-name amex_5pct --seed 42
```

需要重新生成已有数据时，在对应命令末尾加 `--overwrite`。

## 3. 训练前检查

```bash
docker exec -it -w /data5/zst/biye210/EasyR1 easyr1 bash -lc '
python3 -m pytest -q \
  tests/test_ui_s1_reward.py \
  tests/test_ui_s1_advantage.py \
  tests/test_ui_s1_rollout_support.py \
  tests/test_ui_s1_gpu_monitor.py
'
```

## 4. AndroidControl 全数据集：1 epoch、4 GPU

```bash
docker exec -it -w /data5/zst/biye210/EasyR1 easyr1 bash -lc '
GPU_IDS=0,1,2,3 \
MODEL_PATH=/data5/zst/biye210/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/data5/zst/biye210/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
DATASET=android_control \
DATA_DIR=/data5/zst/biye210/EasyR1/datasets/android_control \
TRAIN_FILE=/data5/zst/biye210/EasyR1/datasets/android_control/android_control_train.jsonl \
VAL_FILE=/data5/zst/biye210/EasyR1/datasets/android_control/android_control_val.jsonl \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
MAX_ROLLOUTS_PER_TASK=8 \
DIVERSITY_REFILL_BATCH_SIZE=4 \
GENERATION_MICRO_BATCH_SIZE=4 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
VLLM_GPU_MEMORY_UTILIZATION=0.60 \
VLLM_ENFORCE_EAGER=true \
VLLM_ENABLE_SLEEP_MODE=true \
GPU_MEMORY_MONITOR_INTERVAL_SECONDS=1 \
VALIDATION_PROGRESS_INTERVAL=25 \
SAVE_INTERVAL_SECONDS=7200 \
SAVE_LIMIT=-1 \
RUN_NAME=ui_s1_qwen25vl_3b_android_control_4gpu_1epoch_v1 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

## 5. AMEX 全数据集：1 epoch、4 GPU

```bash
docker exec -it -w /data5/zst/biye210/EasyR1 easyr1 bash -lc '
GPU_IDS=0,1,2,3 \
MODEL_PATH=/data5/zst/biye210/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/data5/zst/biye210/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
DATASET=amex \
DATA_DIR=/data5/zst/biye210/EasyR1/datasets/amex \
TRAIN_FILE=/data5/zst/biye210/EasyR1/datasets/amex/amex_train.jsonl \
VAL_FILE=/data5/zst/biye210/EasyR1/datasets/amex/amex_val.jsonl \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
MAX_ROLLOUTS_PER_TASK=8 \
DIVERSITY_REFILL_BATCH_SIZE=4 \
GENERATION_MICRO_BATCH_SIZE=4 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
VLLM_GPU_MEMORY_UTILIZATION=0.60 \
VLLM_ENFORCE_EAGER=true \
VLLM_ENABLE_SLEEP_MODE=true \
GPU_MEMORY_MONITOR_INTERVAL_SECONDS=1 \
VALIDATION_PROGRESS_INTERVAL=25 \
SAVE_INTERVAL_SECONDS=7200 \
SAVE_LIMIT=-1 \
RUN_NAME=ui_s1_qwen25vl_3b_amex_4gpu_1epoch_v1 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

开始一项新训练时必须使用未存在的 `RUN_NAME`，例如将末尾 `v1` 改为 `v2`；训练脚本会拒绝覆盖已有输出目录。意外中断后，保留原 `RUN_NAME` 并增加 `RESUME=true`，即可从最新 checkpoint 继续。

## 6. 查看运行状态与结果

```bash
RUN_DIR=/data5/zst/biye210/EasyR1/output/<RUN_NAME>

tail -f "$RUN_DIR/training_progress.log"
tail -f "$RUN_DIR/semi_online_rollouts.jsonl"
cat "$RUN_DIR/gpu_memory_peak.json"
cat "$RUN_DIR/experiment_log.jsonl"
```

训练完成后，`global_step_<n>/` 为 checkpoint；`training_progress.log` 用于查看阶段和耗时，`gpu_memory_peak.json` 用于查看 GPU 显存峰值。验证使用 `*_val.jsonl`；`*_test.jsonl` 不参与训练。
