# EasyR1 UI-S1 操作命令

本页只保留日常操作所需命令。训练逻辑、参数说明、日志字段和显存含义见 [docs/ui_s1_training.md](docs/ui_s1_training.md)。

## 1. 环境准备

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

## 3. AMEX 训练模式验证命令

以下六组验证命令覆盖普通训练、严格续训、LoRA 热启动以及 Patch 模仿学习开关的有效组合。启用模拟学习的命令统一使用 80-step 截止调度。

| 训练入口 | 模拟学习关闭 | 模拟学习开启 |
|---|---:|---:|
| 普通新训练 | 试验 1 | 试验 3 |
| 严格继续训练 | 试验 2 | 试验 5 |
| LoRA 热启动 | 试验 6 | 试验 4 |

两条依赖链的执行顺序为：

```text
试验 1 → 试验 2 → 试验 4
试验 3 → 试验 5 → 试验 6
```

公共约定：

- 基础模型为 `/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct`。
- `DATASET=amex` 自动使用 `/home/zst/biye215/EasyR1/datasets/amex` 中的训练集和验证集。
- 每条命令新增 2 个 actor update，每步处理 4 条指令。
- `VAL_AFTER_TRAIN=false` 关闭训练结束后的验证集推理，不影响训练和 checkpoint 保存。
- 新训练和热启动必须使用尚不存在的 `RUN_NAME`。

### 试验 1：普通新训练，关闭模拟学习

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
DATASET=amex \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
RUN_NAME=verify_matrix_nopatch_v2 \
RESUME=false \
MAX_STEPS=2 \
GPU_IDS=0,1 \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ACTOR_LR=1.0e-5 \
VAL_AFTER_TRAIN=false \
SAVE_INTERVAL_SECONDS=18000 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

预期生成 step 1–2，创建全新的 LoRA、optimizer 和 scheduler，并保存 `global_step_2`。

### 试验 2：严格继续训练，关闭模拟学习

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
DATASET=amex \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
RUN_NAME=verify_matrix_nopatch_v2 \
RESUME=true \
MAX_STEPS=4 \
GPU_IDS=0,1 \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ACTOR_LR=1.0e-5 \
VAL_AFTER_TRAIN=false \
SAVE_INTERVAL_SECONDS=18000 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

预期从 tracker 指向的 `global_step_2` 恢复完整训练状态，新增 step 3–4，并保存 `global_step_4`。

### 试验 3：普通新训练，开启模拟学习

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
DATASET=amex \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
RUN_NAME=verify_matrix_patch_v3 \
RESUME=false \
MAX_STEPS=2 \
GPU_IDS=0,1 \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
PATCH_IMITATION_ENABLED=true \
PATCH_IMITATION_LAMBDA_INITIAL=1.0 \
PATCH_IMITATION_LAMBDA_DECAY=0.9433732216 \
PATCH_IMITATION_LAMBDA_MIN=0.0 \
PATCH_IMITATION_LAMBDA_CUTOFF_STEP=80 \
PATCH_IMITATION_TARGET_MODE=action_only \
PATCH_HISTORY_MODE=keep_model_thinking \
VAL_AFTER_TRAIN=false \
SAVE_INTERVAL_SECONDS=18000 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

预期生成 step 1–2；对应的 imitation lambda 依次为 `1.0`、`0.9433732216`，并保存 `global_step_2`。

### 试验 4：从无模拟学习 checkpoint 热启动，并开启模拟学习

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
DATASET=amex \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
RUN_NAME=verify_matrix_warm_patch_from_nopatch_v3 \
RESUME=false \
WARM_START_CHECKPOINT_PATH=/home/zst/biye215/EasyR1/output/verify_matrix_nopatch_v2/global_step_4 \
WARM_START_DATALOADER=inherit \
MAX_STEPS=2 \
GPU_IDS=0,1 \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
PATCH_IMITATION_ENABLED=true \
PATCH_IMITATION_LAMBDA_INITIAL=1.0 \
PATCH_IMITATION_LAMBDA_DECAY=0.9433732216 \
PATCH_IMITATION_LAMBDA_MIN=0.0 \
PATCH_IMITATION_LAMBDA_CUTOFF_STEP=80 \
PATCH_IMITATION_TARGET_MODE=action_only \
PATCH_HISTORY_MODE=keep_model_thinking \
VAL_AFTER_TRAIN=false \
SAVE_INTERVAL_SECONDS=18000 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

预期复制源 run 的 step 1–4 历史，只继承 `global_step_4` 的 LoRA，并用新的 optimizer 和 scheduler 生成 step 5–6。新阶段的 imitation lambda 从 `1.0`、`0.9433732216` 重新计数。

### 试验 5：严格继续训练，开启模拟学习

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
DATASET=amex \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
RUN_NAME=verify_matrix_patch_v3 \
RESUME=true \
MAX_STEPS=4 \
GPU_IDS=0,1 \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ACTOR_LR=1.0e-5 \
PATCH_THRESHOLD=3 \
PATCH_IMITATION_ENABLED=true \
PATCH_IMITATION_LAMBDA_INITIAL=1.0 \
PATCH_IMITATION_LAMBDA_DECAY=0.9433732216 \
PATCH_IMITATION_LAMBDA_MIN=0.0 \
PATCH_IMITATION_LAMBDA_CUTOFF_STEP=80 \
PATCH_IMITATION_TARGET_MODE=action_only \
PATCH_HISTORY_MODE=keep_model_thinking \
VAL_AFTER_TRAIN=false \
SAVE_INTERVAL_SECONDS=18000 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

预期从 `global_step_2` 恢复完整状态并新增 step 3–4；imitation lambda 连续衰减为约 `0.889953035232`、`0.839557861919`。

### 试验 6：从模拟学习 checkpoint 热启动，并关闭模拟学习

```bash
docker exec -it -w /home/zst/biye215/EasyR1 easyr1 bash -lc '
DATASET=amex \
MODEL_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
TOKENIZER_PATH=/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct \
RUN_NAME=verify_matrix_warm_nopatch_from_patch_v3 \
RESUME=false \
WARM_START_CHECKPOINT_PATH=/home/zst/biye215/EasyR1/output/verify_matrix_patch_v3/global_step_4 \
WARM_START_DATALOADER=inherit \
MAX_STEPS=2 \
GPU_IDS=0,1 \
EPOCHS=1 \
ROLLOUT_BATCH_SIZE=4 \
ACTOR_GLOBAL_BATCH_SIZE=4 \
ROLLOUT_N=4 \
ACTOR_LR=1.0e-5 \
VAL_AFTER_TRAIN=false \
SAVE_INTERVAL_SECONDS=18000 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
'
```

预期复制源 run 的 step 1–4 历史，只继承 `global_step_4` 的 LoRA，并用新的 optimizer 和 scheduler 生成 step 5–6；源历史中的 imitation 指标保留，新步骤不再产生 imitation loss。

## 4. 训练模式边界

| 模式 | 入口参数 | 输出目录 | 加载状态 | 数据位置 | 训练配置 |
|---|---|---|---|---|---|
| 普通新训练 | `RESUME=false`，不设置热启动路径 | 必须是新目录 | 基座模型 + 新 LoRA、新 optimizer、新 scheduler | 从头开始 | 可自由设置 |
| 严格续训 | `RESUME=true` | 使用原 `RUN_NAME` | 模型、optimizer、scheduler、RNG、dataloader | 从 checkpoint 继续 | Patch 配置和训练拓扑必须兼容 |
| LoRA 热启动 | `RESUME=false` + `WARM_START_CHECKPOINT_PATH` | 必须是新目录 | 只继承 LoRA，其余状态重新创建 | 由 `WARM_START_DATALOADER` 决定 | 可自由修改，包括启停模拟学习 |

严格续训没有设置 `RESUME_CHECKPOINT_PATH` 时，从 `checkpoint_tracker.json` 的 `last_global_step` 指向的最新完整 checkpoint 恢复。`MAX_STEPS` 是当前训练阶段的累计目标步数：从 phase step 2 再训练两步时，应设置为 4。

热启动的 `WARM_START_DATALOADER=inherit` 会从源 checkpoint 后的下一个 batch 继续，要求训练文件、shuffle seed 和 rollout batch size 兼容；使用 `reset` 则保留 LoRA，但从新数据顺序开头开始。新目录会复制截至源 step 的结构化历史，并建立同编号的初始 checkpoint，使可视化曲线保持连续。

不要同时设置 `RESUME=true` 和 `WARM_START_CHECKPOINT_PATH`。

### 严格续训的日志回退

显式从同一 run 的较早 `RESUME_CHECKPOINT_PATH` 恢复时，启动脚本会在训练前：

- 将 `training_progress.log`、`experiment_log.jsonl` 和 `semi_online_rollouts.jsonl` 截断到恢复 step。
- 删除恢复 step 之后的 checkpoint 目录。
- 将不能安全逐行截断的原始日志移入 `rollback_archive/before_global_step_<N>/`。
- 将回滚详情写入 `rollback_archive/before_global_step_<N>/rollback.json`。

因此，从 step 99 恢复一个已有 step 104 日志的 run 时，旧 step 100–104 会先被移除，重新训练后不会出现重复 step。所有 checkpoint 完整性、GPU world size 和训练参数检查通过后才执行回滚。

## 5. 查看运行状态与结果

```bash
RUN_DIR=/home/zst/biye215/EasyR1/output/<RUN_NAME>

tail -f "$RUN_DIR/training_progress.log"
tail -f "$RUN_DIR/semi_online_rollouts.jsonl"
cat "$RUN_DIR/gpu_memory_peak.json"
cat "$RUN_DIR/experiment_log.jsonl"
```

训练完成后，`global_step_<n>/` 为完整 checkpoint；其中 `actor/lora_adapter/` 可用于 vLLM 部署。`training_progress.log` 用于查看阶段和耗时，`experiment_log.jsonl` 用于可视化训练指标，`gpu_memory_peak.json` 用于查看 GPU 显存峰值。

设置 `VAL_AFTER_TRAIN=false` 时不会执行训练结束后的验证集推理；设为 `true` 或不填写时使用 `*_val.jsonl` 验证。`*_test.jsonl` 不参与训练。
