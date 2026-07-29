# UI-S1 半在线 GRPO 训练说明

本文记录当前 EasyR1 中 UI-S1 AndroidControl / AMEX 的训练设置、rollout 调度单位，以及 `training_progress.log` 的阅读方法。面向人工操作的简洁命令见仓库外层的 `docker-easyR1-setup.md`。

## Checkpoint 谱系：严格续训与 LoRA 热启动

严格续训使用 `RESUME=true`，恢复完整 FSDP 模型、optimizer、scheduler、RNG 和 dataloader 状态；它仍是同一实验，因此 Patch imitation 配置必须一致。启动前会将结构化日志和 checkpoint 回退到实际恢复点，避免旧分支混入后续可视化。

LoRA 热启动使用新的 `RUN_NAME`、`RESUME=false` 和 `WARM_START_CHECKPOINT_PATH=.../global_step_<N>`。它只加载该 checkpoint 的 `actor/lora_adapter`，创建新的 optimizer/scheduler，且可自由修改 RL 与 Patch 配置。`WARM_START_DATALOADER=inherit` 恢复数据游标，`reset` 则重置数据顺序。

热启动的新输出目录会复制源 run 截止 `N` 的结构化历史，并从全局 step `N+1` 继续记录。内部的 `phase_step` 从 0 开始，因此新 optimizer 的 schedule 和 Patch lambda 属于新阶段。新 run 会立即写入新的 `global_step_N` checkpoint，从而可独立使用 `RESUME=true`。

## 当前推荐设置

当前实验使用 Qwen2.5-VL-3B-Instruct、2 张 RTX 3090 24GB，并以 LoRA 进行半在线 GRPO 训练。启动脚本是 `examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh`；默认训练参数记录在 `examples/ui_s1/train.env`。

| 设置 | 当前值 | 含义 |
| --- | ---: | --- |
| `GPU_IDS` | `0,1` | 本次训练使用 2 张可见 GPU。 |
| `EPOCHS` | `1` | 默认训练完整的 1 个 epoch；设置 `MAX_STEPS` 时忽略该值。 |
| `worker.rollout.tensor_parallel_size` | `1` | 每张 GPU 上各有一个独立的 TP=1 vLLM worker；它们不是一个两卡张量并行 engine。 |
| `ROLLOUT_BATCH_SIZE` | `4` | 每次模型更新采样 4 个**任务**。 |
| `ACTOR_GLOBAL_BATCH_SIZE` | `4` | actor 更新的任务级全局 batch 大小；结合 `ROLLOUT_N=4` 后 actor 每个 mini-batch 为 16 个 rollout 样本。 |
| `ROLLOUT_N` | `4` | 每个任务初始生成并最终希望保留的 rollout 数量。 |
| `MAX_ROLLOUTS_PER_TASK` | `8` | 某任务未达到 advantage diversity 要求时，候选 rollout 的上限。 |
| `DIVERSITY_REFILL_BATCH_SIZE` | `4` | 某任务未达 diversity 阈值时一次新增的候选数。默认从 4 条初始候选直接补到 8 条。 |
| `GENERATION_MICRO_BATCH_SIZE` | `2` | 每个 rollout wave 并行调度 2 个不同的活跃轨迹 step，与 2 个 TP=1 worker 对齐。 |
| `UIS1_ADVANTAGE_STD_THRESHOLD` | `0.3` | 每个任务候选 advantage 的标准差阈值；不足时触发补充 rollout。 |
| `PATCH_THRESHOLD` | `1` | 一条 UI 轨迹最多允许一次 patch 后继续。 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.60` | 24GB 3090 的保守起点；正式长跑前仍需先做两卡 smoke test。 |
| `GPU_MEMORY_MONITOR_INTERVAL_SECONDS` | `1` | GPU 显存高水位监控的采样间隔（秒）。 |
| `VALIDATION_PROGRESS_INTERVAL` | `25` | 每完成 N 个 validation batch 写一条简洁进度记录。 |
| `VLLM_ENFORCE_EAGER` | `true` | 当前实验使用 eager 模式。 |

`MAX_STEPS` 仅用于 smoke test。每次启动后，应以输出目录的 `experiment_config.json` 中实际写入的 `trainer.max_steps` 为准。例如 `ui_s1_qwen25vl_3b_android_control_rl_2gpu_fast_smoke_v1` 的实际值为 `2`，因此它会执行两个更新 step，而不是一个。

## Patch 动作模拟学习

Patch 模拟学习是独立、默认关闭的辅助目标。`PATCH_THRESHOLD` 仍控制 rollout 中最多允许多少次专家动作 patch；`PATCH_IMITATION_ENABLED` 则单独决定这些 patch 是否产生专家动作模拟梯度。因此保持相同的 `PATCH_THRESHOLD` 并切换 `PATCH_IMITATION_ENABLED`，即可进行严格的 GRPO / GRPO+模拟学习消融。

代码支持的主方案直接从原始 `Qwen2.5-VL-3B-Instruct` 开始 RL，不加载动作 SFT checkpoint，也不额外执行预训练 SFT；启用 Patch 模拟学习后，训练前期的专家引导由按需触发的模拟目标提供，后期可通过 lambda 衰减逐步转向 GRPO。仓库默认仍保持关闭，避免在尚未确定实验 lambda 时静默采用任意权重；主实验必须显式设置 `PATCH_IMITATION_ENABLED=true` 和正的 `PATCH_IMITATION_LAMBDA_INITIAL`。

\[
J_{\text{total}}=J_{\text{GRPO}}+\lambda_{\text{patch}}J_{\text{patch}},
\qquad
g_{\text{update}}=g_{\text{GRPO}}+\lambda_{\text{patch}}g_{\text{patch}}
\]

实现中两项先后反向并累积到同一组 LoRA 参数，在 EasyR1 原有的同一个 Adam 更新边界内统一裁剪和 `optimizer.step()`；没有 patch 时该边界仍是原来的纯 GRPO，功能关闭时不会构造 patch 张量或增加额外前反向。

| 环境变量 | 配置键 | 默认值 | 当前含义 |
| --- | --- | ---: | --- |
| `PATCH_IMITATION_ENABLED` | `algorithm.patch_imitation.enabled` | `false` | 是否开启专家动作模拟学习。关闭时不应构造模拟样本或执行额外前反向。 |
| `PATCH_IMITATION_LAMBDA_INITIAL` | `algorithm.patch_imitation.lambda_initial` | `0.0` | 第一次 trainer update 的模拟目标权重。开启功能时必须显式设为正数。 |
| `PATCH_IMITATION_LAMBDA_DECAY` | `algorithm.patch_imitation.lambda_decay` | `1.0` | 每完成一次 trainer update 后的乘法衰减，取值范围为 `(0, 1]`。 |
| `PATCH_IMITATION_LAMBDA_MIN` | `algorithm.patch_imitation.lambda_min` | `0.0` | 模拟目标权重下限，不能超过初始值。 |
| `PATCH_IMITATION_TARGET_MODE` | `algorithm.patch_imitation.target_mode` | `action_only` | 当前只监督专家 `<tool_call>` 动作 token。 |
| `PATCH_HISTORY_MODE` | `algorithm.patch_imitation.history_mode` | `keep_model_thinking` | rollout 历史继续保留模型自己的 thinking，不用专家内容替换。 |

第 \(s\) 次 trainer update 使用的权重为：

\[
\lambda_{\text{patch}}(s)
=
\max\left(
\lambda_{\min},
\lambda_{\text{initial}}\lambda_{\text{decay}}^{s-1}
\right),
\qquad s\ge 1
\]

这里的 `global_step` 从 1 开始，因此 step 1 恰好使用 `lambda_initial`。衰减单位是完整 trainer update，而不是 patch 样本数；某一步没有可用 patch 时，调度仍由全局 step 唯一确定。

当前模拟样本复用 rollout 当步的提示词与图片，并保留模型生成的 thinking 作为模拟目标权重为零的前缀；只有转换到模型坐标系后的专家动作 token 参与模拟学习。模型 thinking 在原 rollout 中保持不变，也不会被专家 thinking 替换。`thinking_and_action` 与 `replace_with_expert_thinking` 仅是后续扩展方向，目前若传入这些未实现模式，启动脚本和 dataclass 都会立即报错。

Patch 目标只从 diversity 筛选后真正进入 actor 更新的 rollout 构造。同一任务同一步若有 \(M\) 条 rollout 同时发生 patch，每条样本只占该步的 \(1/M\)；再对不同被 patch 的任务-step 等权平均，并在每条样本内部对专家 `<tool_call>` token 求平均。因此“一处 patch”和“同一步重复出现多处相同 patch”的总专家权重一致，actor padding 复制出的行权重固定为零。

## 定时 checkpoint 与断点续训

checkpoint 是完整训练状态：actor 的 FSDP/LoRA 权重、优化器、学习率调度器、各 rank 的随机数状态及 stateful dataloader 状态都会写入 `RUN_DIR/global_step_<N>/`。因此恢复不会只加载 actor 权重，也不会重新从数据集开头开始。

在启动命令中设置 `SAVE_INTERVAL_SECONDS=X`（单位：秒）即可开启定时保存，例如 `SAVE_INTERVAL_SECONDS=1800` 表示每 30 分钟保存一次。计时到期后，框架会等当前 update 的 actor 更新完成才保存，所以 checkpoint 始终对应一个完整的最新 actor step；训练结束时也会额外保存最后一步。`SAVE_LIMIT=-1` 默认保留全部断点；设为正整数可限制保留数量。

要从最新断点继续，使用原来的 `RUN_NAME` 并增加 `RESUME=true`。未设置 `RESUME_CHECKPOINT_PATH` 时，启动脚本默认读取 `RUN_DIR/checkpoint_tracker.json` 的 `last_global_step`，并恢复其指向的最后一个完整 `global_step_*`；它不会仅按目录名猜测恢复点：

`EPOCHS` / `MAX_STEPS` 要设为希望达到的总更新量，而不是“额外训练量”。例如：

```bash
RUN_NAME=ui_s1_qwen25vl_3b_amex_5pct_2gpu_1epoch_v1 \
RESUME=true \
SAVE_INTERVAL_SECONDS=1800 \
bash examples/ui_s1/run_qwen2_5_vl_3b_ui_s1_semionline_grpo_lora.sh
```

要回退到同一 run 中较早的保存点，可额外指定 `RESUME_CHECKPOINT_PATH=/absolute/path/to/RUN_NAME/global_step_N`。严格续训不接受其他 run 的 checkpoint；跨 run 只继承 LoRA 应使用 `WARM_START_CHECKPOINT_PATH`。恢复时不应使用 `trainer.save_model_only=true`，否则优化器等训练状态不存在；本启动脚本固定使用完整 checkpoint。

FSDP 断点只能用与保存时相同的 GPU 数恢复。当前复制到本机的历史断点是 `world_size=4` 分片，不能直接在本次 `GPU_IDS=0,1` 的两卡训练中续跑；本机新建的两卡 run 可以正常两卡续训。启动脚本会在初始化模型前检查分片数量与完整性，并对四卡断点给出明确错误。若要迁移旧权重，需要先单独合并/重分片并作为新实验启动，这不等同于恢复原优化器和 dataloader 状态。

恢复前会先把 `training_progress.log`、`experiment_log.jsonl` 和 `semi_online_rollouts.jsonl` 截断到实际选中的 checkpoint。比如日志已经写到 104、但 tracker 的最新完整保存点是 99，则删除 100–104 的结构化日志后，再从 step 100 追加。无可靠 step 字段的 `train.log`、`generations.log`、GPU 峰值和奖励图片会移入 `rollback_archive/before_global_step_99/`，新活跃文件重新写入。进度日志随后以 `RUN | RESUME` 开始新的一段，并在 `CHECKPOINT_LOAD | END` 后从 checkpoint step 加一继续。

Patch 模拟目标权重没有独立的可变计数器，而是由恢复后的 `global_step` 和上述 lambda 配置直接计算。例如 `global_step_N` 已包含第 N 次 actor 更新，恢复后第一次更新是 N+1，使用的指数为 N。每个 checkpoint 内的 `trainer_state.json` 会记录 step、actor world size、完整 Patch 配置和当步 lambda，并在恢复前执行强一致性校验；显式 checkpoint 路径不存在、tracker 指向的目录缺失、lambda 不一致或训练分片不完整都会直接终止，而不会静默从 step 0 开始。`experiment_config.json` 和新写入的 `experiment_config.resume.json` 仅用于人工查看与审计。

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

因此，`GENERATION_MICRO_BATCH_SIZE=2` 调度的是“当前待生成的轨迹 step”，而不是“一次生成两条完整 rollout”。同一条 rollout 的 step 0、step 1、step 2 必须顺序生成；不同 rollout 当前所处的 step 可以并行。

当一个 wave 的活跃轨迹少于 2 时，部分 GPU 没有可用的不同请求。例如，已经完成的 rollout 不能再为 GPU 提供工作。这是轨迹长度不同导致的尾部空闲，不表示 micro batch 设置失效。

## rollout 日志：`semi_online_rollouts.jsonl`

此文件同时包含三种记录。它们不是三次独立模型生成。

| `record_type` | 写入时机 | 是否包含真实模型生成 | 作用 |
| --- | --- | --- | --- |
| `rollout_progress` | 一条候选 rollout 完成时 | 是 | 原始候选的完成快照：逐 step 的 `model_response`、reward、patch、结束原因等。 |
| `rollout` | 候选池完成 diversity 筛选后 | 不会重新生成；复用前者内容 | 记录被选入训练 batch 的轨迹，并加入 `candidate_pool_size`、`diversity_std`、`selection_advantage`、`selected_for_update` 等筛选结果。 |
| `actor_update` | actor 更新完成后 | 否 | 记录本次实际参与权重更新的 trajectory ID。 |

例如，若本次有 4 个任务，每任务初始生成 4 条且无需补充，则会有 16 条 `rollout_progress` 和 16 条 `rollout`。这仍然只代表 16 条候选轨迹生成；`rollout` 不是第二次生成。

若某任务的候选 advantage 标准差低于 0.3，框架会一次生成最多 `DIVERSITY_REFILL_BATCH_SIZE=4` 条补充候选，直到满足阈值或达到 `MAX_ROLLOUTS_PER_TASK=8`。默认配置下，失败任务会从 4 条初始候选直接扩展至 8 条。候选全部完成后，框架枚举 8 选 4 的组合并保留 advantage 标准差最大的 4 条；补充候选会增加 `rollout_progress` 数量，但最终 `rollout` 仍只保存 4 条用于更新的轨迹。

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
| `VALIDATION` | 验证开始、进度或结束。`START` 记录总 batch 数和生成数；`PROGRESS` 每 25 个 batch 记录一次已完成数量、当前 overall reward 均值和耗时；`END` 记录总耗时与核心验证 reward。 |
| `VALIDATION_ENGINE_SYNC` | 验证前将 actor 权重同步到 vLLM。 |
| `VALIDATION_ENGINE_RELEASE` | 验证结束后让 vLLM sleep/offload。 |

### 每个训练更新（`STEP n`）

| 阶段 | 含义 |
| --- | --- |
| `STEP \| START` | 第 `n` 次权重更新开始；`tasks_per_update` 是任务数，`rollout_n` 是每任务初始候选数。 |
| `ROLLOUT_ENGINE_SYNC` | 将当前 actor 的参数同步到 vLLM rollout engine。它是内存中的 actor → vLLM 同步，不是从磁盘重新加载模型。下一次更新后的新权重，会在下一次该阶段同步给 vLLM。 |
| `ROLLOUT` | 本更新所有任务的候选轨迹采样与筛选的总阶段。`selected_rollouts` 是最终用于训练的轨迹数量；`selected_rollout_step_counts=[[...],[...],[...],[...]]` 按任务顺序列出最终选中 rollout 的模型动作 step 数，每个内层列表有 `ROLLOUT_N` 个值。 |
| `ROLLOUT_WAVE` | 一个生成轮次：调度所有仍在进行的 rollout 的**当前轨迹 step**。`active_rollout_steps` 是该轮待生成的 step 数；不是任务数。 |
| `REWARD \| SUMMARY` | 对刚完成 wave 中 UI 动作计算即时 reward 后的均值。`overall_mean` 是总奖励，另三个字段分别是格式、工具类型和动作参数准确性奖励。 |
| `DIVERSITY \| READY` | 各任务候选 advantage 的分布已满足阈值，可开始选择训练轨迹。`task_ids`、`candidate_counts`、`diversity_std` 三个列表按相同位置对齐。 |
| `DIVERSITY \| RETRY` | 仍有任务的 advantage 标准差不足，需要补充候选。`task_ids[i] → candidate_counts[i] → diversity_std[i] → refill_counts[i] → next_candidate_counts[i]` 表示第 `i` 个任务的 ID、参与本次标准差计算的候选数、自身标准差、本轮新增候选数和补充后的候选数；是否重试由每个任务自身的标准差决定，不由列表均值决定。 |
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

## GPU 显存高水位：`gpu_memory_peak.json`

启动脚本会在每个输出目录启动独立的 `monitor_gpu_memory.py`，每秒原子更新一次 `gpu_memory_peak.json`。该文件会在训练中持续记录每张 GPU 的当前显存、历史峰值、峰值时间和总设备峰值，训练结束时写入 `finished_at`。

`vllm_memory_budget_mib_per_gpu` 等于 GPU 总显存乘以 `VLLM_GPU_MEMORY_UTILIZATION`。例如 RTX 3090 的 24,576 MiB 与 `0.60` 对应 14,746 MiB（14.4 GiB）的 vLLM 总预算。这个数字不是纯 KV cache：模型权重、CUDA graph / runtime buffer 等也占用这部分预算；KV cache 是扣除这些部分后的剩余空间。

监控的是设备级总显存，因此包含 vLLM、actor、reference 等所有训练阶段的显存。它适合寻找不会 OOM 的安全上限；不能单独作为 vLLM 纯 KV cache 的容量指标。

## 当前日志的已知边界

目前 `training_progress.log` 已覆盖训练初始化、rollout、奖励、筛选、概率计算、actor 更新和 validation。验证固定使用 `data.val_batch_size=1`，在验证集较大时可能占用明显时间；因此只记录开始、每 25 batch 的进度、结束和 vLLM 切换，不写逐样本日志。
