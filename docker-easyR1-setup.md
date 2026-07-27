# EasyR1 UI-S1 操作命令

本页只保留日常操作所需命令。训练逻辑、参数说明、日志字段和显存含义见 [EasyR1/docs/ui_s1_training.md](EasyR1/docs/ui_s1_training.md)。

## 1. 一次性环境准备

宿主机确认 GPU：

```bash
nvidia-smi
```

首次在新服务器配置运行时路径：

```bash
cd /home/zst/biye215
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
  -v /home/zst/biye215:/home/zst/biye215 \
  -w /home/zst/biye215/EasyR1 \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 \
  sleep infinity

docker exec easyr1 python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

预期第一项为 `True`，且设备数不少于 2；正式训练仍由 `GPU_IDS=0,1` 只选择两张卡。

## 2. 生成 5% RL 数据（首次或数据变更后）

```bash
cd /home/zst/biye215/EasyR1
set -a; source examples/ui_s1/runtime.env; set +a

python3 -B examples/ui_s1/prepare_ui_s1_android_control_rl_data.py \
  --android-control-dir /home/zst/biye215/datasets/android_control \
  --image-dir /home/zst/biye215/llamafactory/data/ui_s1_android_control_sft/images \
  --output-dir /home/zst/biye215/EasyR1/datasets/android_control_5pct \
  --sample-ratio 0.05 --dataset-name android_control_5pct --seed 42

python3 -B examples/ui_s1/prepare_ui_s1_amex_rl_data.py \
  --amex-dir /home/zst/biye215/datasets/amex \
  --output-dir /home/zst/biye215/EasyR1/datasets/amex_5pct \
  --sample-ratio 0.05 --dataset-name amex_5pct --seed 42
```

需要重新生成已有数据时，在对应命令末尾加 `--overwrite`。

## 3. 训练前检查

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
python3 -m pytest -q \
  tests/test_patch_imitation_config.py \
  tests/test_patch_imitation_actor.py \
  tests/test_patch_imitation_resume.py \
  tests/test_ui_s1_patch_imitation.py \
  tests/test_ui_s1_reward.py \
  tests/test_ui_s1_advantage.py \
  tests/test_ui_s1_rollout_support.py \
  tests/test_ui_s1_gpu_monitor.py
'
```

## 4. AndroidControl 全数据集：1 epoch、2 GPU

下面命令默认是 `PATCH_IMITATION_ENABLED=false` 的纯 GRPO 基线。运行 GRPO+Patch 主实验时，还需显式加入 `PATCH_IMITATION_ENABLED=true`、正的 `PATCH_IMITATION_LAMBDA_INITIAL`，以及选定的 decay/min；这里不替你虚构尚未确认的 lambda 数值。

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
GPU_IDS=0,1 \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
DATASET=android_control \
DATA_DIR=/home/zst/biye215/EasyR1/datasets/android_control \
TRAIN_FILE=/home/zst/biye215/EasyR1/datasets/android_control/android_control_train.jsonl \
VAL_FILE=/home/zst/biye215/EasyR1/datasets/android_control/android_control_val.jsonl \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
MAX_ROLLOUTS_PER_TASK=8 \
DIVERSITY_REFILL_BATCH_SIZE=4 \
GENERATION_MICRO_BATCH_SIZE=2 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
VLLM_GPU_MEMORY_UTILIZATION=0.60 \
VLLM_ENFORCE_EAGER=true \
VLLM_ENABLE_SLEEP_MODE=true \
GPU_MEMORY_MONITOR_INTERVAL_SECONDS=1 \
VALIDATION_PROGRESS_INTERVAL=25 \
SAVE_INTERVAL_SECONDS=7200 \
SAVE_LIMIT=-1 \
RESUME=false \
RUN_NAME=ui_s1_qwen25vl_3b_android_control_2gpu_1epoch_v1 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

## 5. AMEX 全数据集：Patch 模仿学习、1 epoch、2 GPU

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
# 当前 v3 的实际续训命令。仅在该 run 已停止时执行；运行中不要重复执行。
GPU_IDS=0,2 \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
DATASET=amex \
DATA_DIR=/home/zst/biye215/EasyR1/datasets/amex \
TRAIN_FILE=/home/zst/biye215/EasyR1/datasets/amex/amex_train.jsonl \
VAL_FILE=/home/zst/biye215/EasyR1/datasets/amex/amex_val.jsonl \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
MAX_ROLLOUTS_PER_TASK=8 \
DIVERSITY_REFILL_BATCH_SIZE=4 \
GENERATION_MICRO_BATCH_SIZE=2 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
VLLM_GPU_MEMORY_UTILIZATION=0.60 \
VLLM_ENFORCE_EAGER=true \
VLLM_ENABLE_SLEEP_MODE=true \
GPU_MEMORY_MONITOR_INTERVAL_SECONDS=1 \
VALIDATION_PROGRESS_INTERVAL=25 \
PATCH_IMITATION_ENABLED=true \
PATCH_IMITATION_LAMBDA_INITIAL=1.0 \
PATCH_IMITATION_LAMBDA_DECAY=0.994 \
PATCH_IMITATION_LAMBDA_MIN=0.0 \
PATCH_IMITATION_TARGET_MODE=action_only \
PATCH_HISTORY_MODE=keep_model_thinking \
SAVE_EVERY_N_EPOCHS=1 \
SAVE_INTERVAL_SECONDS=18000 \
SAVE_LIMIT=-1 \
RESUME=true \
RUN_NAME=ui_s1_qwen25vl_3b_amex_2gpu_patch_1epoch_v3 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

该命令从 `checkpoint_tracker.json` 所指向的最新完整 checkpoint 恢复；当前设置将时间保存间隔从 2 小时改为 5 小时。停机等待时间不计入保存间隔，恢复完成后重新计时。开始一项新训练时必须使用未存在的 `RUN_NAME`、设置 `RESUME=false`，并保留同样的 Patch 模仿参数。

### 断点续训：参数变更边界

下表中的“可修改”指在不改变已保存模型/优化器状态的前提下可以修改；“不可修改”是代码会明确拒绝，或会使续训不再是同一训练实验。

| 分类 | 参数 | 续训时是否可修改 | 依据与限制 |
|---|---|---|---|
| 原地续训时保持 | `RUN_NAME`、`OUTPUT_ROOT` | 否 | 要续写原 run 的日志与 checkpoint，必须定位到原 run 目录及其 `checkpoint_tracker.json`；改路径并显式给出 `RESUME_CHECKPOINT_PATH` 属于从该 checkpoint 派生新 run。 |
| 必须保持 | `RESUME` | 必须为 `true` | 现有输出目录配合 `RESUME=false` 会被启动脚本拒绝。 |
| 代码明确校验 | GPU 数量 / FSDP world size | 否 | 必须与 checkpoint 的 actor 分片 world size 相同；当前为 2。缺少 `model`、`optim`、`extra_state` 分片或 `dataloader.pt` 也会拒绝恢复。 |
| 可修改 | `GPU_IDS` 的具体编号 | 是 | 可由 `0,1` 改为当前的 `0,2`；前提是设备数量仍为 2、CUDA 环境兼容且显存足够。 |
| 代码明确校验 | `PATCH_IMITATION_ENABLED` | 否 | 必须与 checkpoint 保存的开关一致。 |
| 代码明确校验 | `PATCH_IMITATION_LAMBDA_INITIAL`、`LAMBDA_DECAY`、`LAMBDA_MIN` | 否 | 必须与 checkpoint 一致，保证后续 \(\lambda_t\) 连续。 |
| 代码明确校验 | `PATCH_IMITATION_TARGET_MODE`、`PATCH_HISTORY_MODE` | 否 | 必须与 checkpoint 一致。 |
| 应保持不变 | 模型/Tokenizer 路径、LoRA rank/alpha/target modules、FSDP 结构 | 否 | 虽非每项都单独比较，但 checkpoint 的模型与优化器状态依赖相同参数拓扑；改动可能加载失败或破坏实验连续性。 |
| 应保持不变 | 数据集文件、shuffle/seed、batch size、`ROLLOUT_N`、数据长度 | 否 | checkpoint 保存并恢复 dataloader cursor；改动数据或批处理结构会使后续样本序列和训练语义不连续。 |
| 应保持不变 | reward、GRPO/UI-S1 参数、KL、actor LR、rollout 温度/采样参数、`PATCH_THRESHOLD` | 否 | 代码不会逐项阻止，但它们会改变后续优化目标、奖励、轨迹分布或更新尺度；应新建 run 做消融。 |
| 可修改 | `SAVE_INTERVAL_SECONDS` | 是 | 只控制恢复后下一次时间保存；当前从 `7200` 改为 `18000` 秒。 |
| 可修改 | `SAVE_LIMIT`、`SAVE_EVERY_N_EPOCHS` | 是 | 只影响后续 checkpoint 保留与保存频率。 |
| 可修改 | `GPU_MEMORY_MONITOR_INTERVAL_SECONDS`、`VALIDATION_PROGRESS_INTERVAL`、logger、Ray dashboard | 是 | 只影响监控或日志。 |
| 条件可修改 | `EPOCHS` / `MAX_STEPS` | 可以延长，不应缩短到当前 step 以下 | 用于明确延长训练总步数；会形成新的训练阶段，应在实验记录中注明。 |
| 条件可修改 | `RESUME_CHECKPOINT_PATH` | 是 | 可指定同一兼容 run 的较早完整 `global_step_<n>` checkpoint，用于回退；仍受 world size、Patch 配置和完整性校验。 |

时间 checkpoint 只会在一个完整 actor update 后检查。当前 `save_freq=-1`，因此没有“每 N step 保存”；另有 `SAVE_EVERY_N_EPOCHS=1` 的 epoch 末保存和训练正常结束时的兜底保存。

## 6. 查看运行状态与结果

```bash
RUN_DIR=/home/zst/biye215/EasyR1/output/<RUN_NAME>

tail -f "$RUN_DIR/training_progress.log"
tail -f "$RUN_DIR/semi_online_rollouts.jsonl"
cat "$RUN_DIR/gpu_memory_peak.json"
cat "$RUN_DIR/experiment_log.jsonl"
```

训练完成后，`global_step_<n>/` 为 checkpoint；`training_progress.log` 用于查看阶段和耗时，`gpu_memory_peak.json` 用于查看 GPU 显存峰值。验证使用 `*_val.jsonl`；`*_test.jsonl` 不参与训练。
