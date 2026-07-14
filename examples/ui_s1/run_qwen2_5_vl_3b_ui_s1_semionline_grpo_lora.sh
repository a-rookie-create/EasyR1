#!/bin/bash

# Source-agnostic formal UI-S1 semi-online GRPO LoRA runner.
# Select AndroidControl or AMEX through DATA_DIR.
set -euo pipefail
shopt -s nullglob

MODEL_PATH=${MODEL_PATH:-/home/zst/biye215/llamafactory/output/ui_s1_qwen25vl_3b_android_control_amex_lora_full_3gpu_merged}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct}
DATA_DIR=${DATA_DIR:-/home/zst/biye215/EasyR1/datasets/ui_s1_android_control_rl}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/zst/biye215/EasyR1/output}
DATASET_LABEL=${DATASET_LABEL:-$(basename "${DATA_DIR}")}
RUN_NAME=${RUN_NAME:-ui_s1_qwen25vl_3b_${DATASET_LABEL}_semionline_grpo_lora_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
# GPU 1 is currently occupied by another workload. This remains overridable.
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,2}}

# Paper-aligned values are retained where the two available 24 GB GPUs permit them.
EPOCHS=${EPOCHS:-5}
# Optional smoke-test cap. Leave unset for the formal five-epoch run.
MAX_STEPS=${MAX_STEPS:-}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-3}
ACTOR_GLOBAL_BATCH_SIZE=${ACTOR_GLOBAL_BATCH_SIZE:-3}
ROLLOUT_N=${ROLLOUT_N:-8}
# Keep one UI trajectory step per backward pass on two 24 GB cards. Dynamic
# token batching can otherwise combine several shorter multimodal steps.
ACTOR_DYNAMIC_BATCHING=${ACTOR_DYNAMIC_BATCHING:-false}
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-false}
# Generate one task group at a time on the two-card setup; ROLLOUT_N stays 8.
GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-1}
MAX_ROLLOUTS_PER_TASK=${MAX_ROLLOUTS_PER_TASK:-20}
HISTORY_IMAGE_LIMIT=${HISTORY_IMAGE_LIMIT:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-12288}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
# 2 x RTX 3090: FSDP shards are larger than in the three-GPU run.
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.62}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-12800}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-24576}
# vLLM supports LoRA updates for the language model, not Qwen2.5-VL's visual tower.
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
PATCH_THRESHOLD=${PATCH_THRESHOLD:-1}
UIS1_GAMMA=${UIS1_GAMMA:-0.5}
UIS1_STEP_ADVANTAGE_WEIGHT=${UIS1_STEP_ADVANTAGE_WEIGHT:-1.0}
UIS1_EPISODE_ADVANTAGE_WEIGHT=${UIS1_EPISODE_ADVANTAGE_WEIGHT:-1.0}
UIS1_ADVANTAGE_STD_THRESHOLD=${UIS1_ADVANTAGE_STD_THRESHOLD:-0.3}
SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS:-1}

TRAINER_MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
    TRAINER_MAX_STEPS_ARG=("trainer.max_steps=${MAX_STEPS}")
fi

if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
    echo "Set RUN_NAME to a new value so checkpoints and rollout logs cannot be mixed." >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"
RUN_LOG=${RUN_LOG:-${RUN_DIR}/train.log}
exec > >(tee -a "${RUN_LOG}") 2>&1
set -x

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export RAY_DASHBOARD_HOST=${RAY_DASHBOARD_HOST:-0.0.0.0}
IFS=',' read -ra GPU_ID_ARRAY <<< "${GPU_IDS}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${#GPU_ID_ARRAY[@]}}

cd /home/zst/biye215/EasyR1

if [[ -z "${TRAIN_FILE:-}" ]]; then
    train_candidates=("${DATA_DIR}"/*_train.jsonl)
    [[ ${#train_candidates[@]} -eq 1 ]] || { echo "Expected one *_train.jsonl in ${DATA_DIR}" >&2; exit 2; }
    TRAIN_FILE=${train_candidates[0]}
fi
if [[ -z "${VAL_FILE:-}" ]]; then
    val_candidates=("${DATA_DIR}"/*_val.jsonl)
    [[ ${#val_candidates[@]} -eq 1 ]] || { echo "Expected one *_val.jsonl in ${DATA_DIR}" >&2; exit 2; }
    VAL_FILE=${val_candidates[0]}
fi

python3 -B -c "import json, pathlib; data=pathlib.Path('${TRAIN_FILE}'); assert data.exists(), f'missing train file: {data}'; first=json.loads(data.read_text(encoding='utf-8').splitlines()[0]); image=pathlib.Path(first['trajectory_steps'][0]['image']); assert image.exists(), f'missing first screenshot: {image}'"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.image_dir=null \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=1 \
    data.format_prompt=examples/ui_s1/format_prompt/ui_s1_android.jinja \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo \
    algorithm.semi_online=true \
    algorithm.patch_threshold=${PATCH_THRESHOLD} \
    algorithm.semi_online_gamma=${UIS1_GAMMA} \
    algorithm.semi_online_step_advantage_weight=${UIS1_STEP_ADVANTAGE_WEIGHT} \
    algorithm.semi_online_episode_advantage_weight=${UIS1_EPISODE_ADVANTAGE_WEIGHT} \
    algorithm.semi_online_normalize_by_std=true \
    algorithm.semi_online_advantage_std_threshold=${UIS1_ADVANTAGE_STD_THRESHOLD} \
    algorithm.semi_online_image_limit=${HISTORY_IMAGE_LIMIT} \
    algorithm.semi_online_generation_micro_batch_size=${GENERATION_MICRO_BATCH_SIZE} \
    algorithm.semi_online_max_rollouts_per_task=${MAX_ROLLOUTS_PER_TASK} \
    algorithm.use_kl_loss=true \
    algorithm.kl_coef=1.0e-4 \
    worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.dynamic_batching=${ACTOR_DYNAMIC_BATCHING} \
    worker.actor.use_torch_compile=${ACTOR_USE_TORCH_COMPILE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.tokenizer_path=${TOKENIZER_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.model.lora.rank=64 \
    worker.actor.model.lora.alpha=32 \
    worker.actor.model.lora.target_modules=${LORA_TARGET_MODULES} \
    worker.actor.optim.lr=1.0e-6 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=0.9 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=${HISTORY_IMAGE_LIMIT} \
    worker.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=${VLLM_MAX_MODEL_LEN} \
    worker.rollout.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} \
    worker.reward.reward_function=examples/ui_s1/reward_ui_s1_step.py:compute_score \
    trainer.total_epochs=${EPOCHS} \
    "${TRAINER_MAX_STEPS_ARG[@]}" \
    trainer.project_name=easy_r1_ui_s1 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.logger='["console","file"]' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_every_n_epochs=${SAVE_EVERY_N_EPOCHS} \
    trainer.save_limit=-1 \
    trainer.save_model_only=false \
    trainer.save_checkpoint_path=${RUN_DIR} \
    trainer.rollout_log_path=${RUN_DIR}/semi_online_rollouts.jsonl \
    trainer.find_last_checkpoint=false
