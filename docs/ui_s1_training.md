# UI-S1 半在线 GRPO 训练说明

本文记录当前 EasyR1 中 UI-S1 Android Control 的训练设置、rollout 调度单位，以及 `training_progress.log` 的阅读方法。

## 当前推荐设置

当前实验使用 Qwen2.5-VL-3B-Instruct、4 张 GPU，并以 LoRA 进行半在线 GRPO 训练。启动脚本是 `examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh`；默认训练参数记录在 `examples/ui_s1/train.env`。

| 设置 | 当前值 | 含义 |
| --- | ---: | --- |
| `GPU_IDS` | `0,1,2,3` | 4 张可见 GPU。 |
| `worker.rollout.tensor_parallel_size` | `1` | 每张 GPU 上各有一个独立的 TP=1 vLLM worker；它们不是一个四卡张量并行 engine。 |
| `ROLLOUT_BATCH_SIZE` | `4` | 每次模型更新采样 4 个**任务**。 |
| `ACTOR_GLOBAL_BATCH_SIZE` | `4` | actor 更新的全局 batch 大小。 |
| `ROLLOUT_N` | `4` | 每个任务初始生成并最终希望保留的 rollout 数量。 |
| `MAX_ROLLOUTS_PER_TASK` | `8` | 某任务未达到 advantage diversity 要求时，候选 rollout 的上限。 |
| `GENERATION_MICRO_BATCH_SIZE` | `4` | 每个 rollout wave 最多并行调度 4 个不同的活跃轨迹 step；对 4 个 TP=1 worker 是合适的起点。 |
| `UIS1_ADVANTAGE_STD_THRESHOLD` | `0.3` | 每个任务候选 advantage 的标准差阈值；不足时触发补充 rollout。 |
| `PATCH_THRESHOLD` | `1` | 一条 UI 轨迹最多允许一次 patch 后继续。 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.50` | vLLM 可使用的显存比例。 |
| `VLLM_ENFORCE_EAGER` | `true` | 当前实验使用 eager 模式。 |

`MAX_STEPS` 仅用于 smoke test。每次启动后，应以输出目录的 `experiment_config.json` 中实际写入的 `trainer.max_steps` 为准。例如 `ui_s1_qwen25vl_3b_android_control_rl_4gpu_fast_smoke_v2` 的实际值为 `2`，因此它会执行两个更新 step，而不是一个。

## 必须区分的三个层级

```text
一次训练更新（STEP）
├── 4 个任务（ROLLOUT_BATCH_SIZE=4）
│   └── 每个任务的 4 条候选 rollout（ROLLOUT_N=4）
│       └── 每条 rollout 的多个 UI 动作 step（数量不固定）
```

- **任务（task）**：一条 UI 目标，例如进入某个页面并完成一个操作。
- **rollout**：同一个任务的一条候选完整轨迹，包含从初始截图到结束的若干动作。
- **轨迹 step**：一条 rollout 内的一次“当前图像/历史 → 模型生成一个 UI 动作 → 计算该动作奖励”。不同 rollout 的 step 数不同。

因此，`GENERATION_MICRO_BATCH_SIZE=4` 调度的是“当前待生成的轨迹 step”，而不是“一次只生成四条完整 rollout”。同一条 rollout 的 step 0、step 1、step 2 必须顺序生成；不同 rollout 当前所处的 step 可以并行。

当一个 wave 的活跃轨迹少于 4 时，部分 GPU 没有可用的不同请求。例如，已经完成的 rollout 不能再为 GPU 提供工作。这是轨迹长度不同导致的尾部空闲，不表示 micro batch 设置失效。

## rollout 日志：`semi_online_rollouts.jsonl`

此文件同时包含三种记录。它们不是三次独立模型生成。

| `record_type` | 写入时机 | 是否包含真实模型生成 | 作用 |
| --- | --- | --- | --- |
| `rollout_progress` | 一条候选 rollout 完成时 | 是 | 原始候选的完成快照：逐 step 的 `model_response`、reward、patch、结束原因等。 |
| `rollout` | 候选池完成 diversity 筛选后 | 不会重新生成；复用前者内容 | 记录被选入训练 batch 的轨迹，并加入 `candidate_pool_size`、`diversity_std`、`selection_advantage`、`selected_for_update` 等筛选结果。 |
| `actor_update` | actor 更新完成后 | 否 | 记录本次实际参与权重更新的 trajectory ID。 |

例如，若本次有 4 个任务，每任务初始生成 4 条且无需补充，则会有 16 条 `rollout_progress` 和 16 条 `rollout`。这仍然只代表 16 条候选轨迹生成；`rollout` 不是第二次生成。

若某任务的候选 advantage 标准差低于 0.3，框架会继续生成补充候选，直到满足阈值或达到 `MAX_ROLLOUTS_PER_TASK=8`。补充候选会增加 `rollout_progress` 数量；最终 `rollout` 仅保存被保留用于更新的候选。

## `training_progress.log` 阶段说明

日志每行格式为：

```text
时间 | [STEP n |] 阶段 | 状态 | 关键字段
```

时间使用 UTC；北京时间为日志时间加 8 小时。`START` / `END` 成对出现的阶段，`END` 中的 `elapsed_s` 是该阶段墙钟耗时。

### 初始化阶段

| 阶段 | 含义 |
| --- | --- |
| `RUN` | 本次实验启动，记录实验名和 GPU / 节点数量。 |
| `MODEL_PROCESSOR` | 加载 tokenizer 与视觉 processor；不是加载完整训练模型权重。 |
| `DATASET` | 加载并构建 train/validation dataloader；`train_batches` 和 `val_batches` 是各自 batch 数。 |
| `WORKERS` | 创建 Ray worker，并初始化 actor、reference、vLLM rollout engine、reward 等角色；实际模型权重初始化主要发生在此阶段。 |
| `TRAINING_LOOP` | 训练主循环开始或结束。`planned_steps` 是本次有效的更新次数。 |
| `CHECKPOINT_LOAD` | 尝试恢复 checkpoint。`SKIP` 表示本次没有要求恢复。 |

### 每个训练更新（`STEP n`）

| 阶段 | 含义 |
| --- | --- |
| `STEP \| START` | 第 `n` 次权重更新开始；`tasks_per_update` 是任务数，`rollout_n` 是每任务初始候选数。 |
| `ROLLOUT_ENGINE_SYNC` | 将当前 actor 的参数同步到 vLLM rollout engine。它是内存中的 actor → vLLM 同步，不是从磁盘重新加载模型。下一次更新后的新权重，会在下一次该阶段同步给 vLLM。 |
| `ROLLOUT` | 本更新所有任务的候选轨迹采样与筛选的总阶段。`selected_rollouts` 是最终用于训练的轨迹数量。 |
| `ROLLOUT_WAVE` | 一个生成轮次：调度所有仍在进行的 rollout 的**当前轨迹 step**。`active_rollout_steps` 是该轮待生成的 step 数；不是任务数。 |
| `REWARD \| SUMMARY` | 对刚完成 wave 中 UI 动作计算即时 reward 后的均值。`overall_mean` 是总奖励，另三个字段分别是格式、工具类型和动作参数准确性奖励。 |
| `DIVERSITY \| READY` | 各任务候选 advantage 的分布已满足阈值，可开始选择训练轨迹。`task_ids`、`candidate_counts`、`diversity_std` 三个列表按相同位置对齐。 |
| `DIVERSITY \| RETRY` | 仍有任务的 advantage 标准差不足，需要补充候选。`task_ids[i] → candidate_counts[i] → diversity_std[i]` 表示第 `i` 个任务的 ID、当前候选数与自身标准差；是否重试由每个任务自身的标准差决定，不由列表均值决定。 |
| `ROLLOUT_ENGINE_RELEASE` | 让 vLLM sleep/offload，释放部分显存供 actor / reference 计算使用。 |
| `OLD_LOG_PROBS` | 用当前更新前的 actor 计算采样 token 的概率。这里的“old”是 PPO 更新前的策略，不是磁盘里的旧 checkpoint。 |
| `REF_LOG_PROBS` | 使用冻结 reference policy 计算概率，用于 KL 约束。 |
| `ADVANTAGE` | 根据 step / episode reward 计算并规范化 advantage；通常耗时很短。 |
| `ACTOR_UPDATE` | 使用 rollout、old log probability、reference log probability 和 advantage 反向传播，更新 LoRA / actor 权重。`padding` 表示为均衡各 GPU token 数而加入的比例。 |
| `CHECKPOINT_SAVE` | 保存 checkpoint；可能在常规保存点或训练结束时出现。 |
| `STEP \| END` | 本次更新结束。`generation_s` 包含权重同步、rollout 和 rollout engine release；`old_log_probs_s`、`ref_log_probs_s`、`actor_update_s` 分别是后续三个主要模型计算阶段。 |

## 如何理解 `throughput`

`STEP | END` 中的 `throughput` 是**每张 GPU 每秒处理的训练 token 数**，不是每秒 rollout 数，也不是仅生成 token 的速度。

```text
throughput = total_num_tokens / step_elapsed_s / GPU 数
```

对于 fast smoke v2 的 step 1：

```text
145,305 / 227.7330 / 4 = 159.5124 tokens/s/GPU
```

长 UI prompt、视觉输入、rollout 内多步动作、actor/reference 的完整前向计算和 actor 反向传播都会计入 step 时间。输出动作虽然通常较短，但平均 prompt 长度约 4,471 tokens，因此不能只根据 response 长度判断耗时。

## 当前日志的已知边界

目前 `training_progress.log` 已覆盖训练初始化、rollout、奖励、筛选、概率计算和 actor 更新。框架在训练结束后仍可能执行最终 validation；当前该 validation 的开始/结束没有单独写入 `training_progress.log`，应同时查看 `train.log` 中的 `Start validation...`。验证固定使用 `data.val_batch_size=1`，在验证集较大时可能占用明显时间。
