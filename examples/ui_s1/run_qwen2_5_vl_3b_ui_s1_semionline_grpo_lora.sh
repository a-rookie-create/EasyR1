#!/bin/bash

# Source-agnostic formal UI-S1 semi-online GRPO LoRA runner.
# Select AndroidControl or AMEX through DATASET.
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EASYR1_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUNTIME_ENV=${RUNTIME_ENV:-${SCRIPT_DIR}/runtime.env}
TRAIN_ENV=${TRAIN_ENV:-${SCRIPT_DIR}/train.env}

for env_file in "${RUNTIME_ENV}" "${TRAIN_ENV}"; do
    if [[ ! -r "${env_file}" ]]; then
        echo "Missing configuration file: ${env_file}" >&2
        echo "Copy runtime.env.example to runtime.env and set this server's paths and GPUs." >&2
        exit 2
    fi
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
done

case "${DATASET}" in
    android_control)
        DATA_DIR=${DATA_DIR:-${ANDROID_CONTROL_OUTPUT_DIR:?Set ANDROID_CONTROL_OUTPUT_DIR in runtime.env}}
        ;;
    amex)
        DATA_DIR=${DATA_DIR:-${AMEX_OUTPUT_DIR:?Set AMEX_OUTPUT_DIR in runtime.env}}
        ;;
    *) echo "Unsupported DATASET=${DATASET}; expected android_control or amex" >&2; exit 2 ;;
esac

required_runtime_vars=(
    MODEL_PATH
    TOKENIZER_PATH
    DATA_DIR
    OUTPUT_ROOT
    GPU_IDS
    RAY_DASHBOARD_HOST
    VLLM_GPU_MEMORY_UTILIZATION
    PYTORCH_CUDA_ALLOC_CONF
)
for required_var in "${required_runtime_vars[@]}"; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "Missing required runtime setting: ${required_var}" >&2
        exit 2
    fi
done

DATASET_LABEL=${DATASET_LABEL:-$(basename "${DATA_DIR}")}
RUN_NAME=${RUN_NAME:-ui_s1_qwen25vl_3b_${DATASET_LABEL}_semionline_grpo_lora_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}

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
export RAY_DASHBOARD_HOST
# Avoid allocator fragmentation when UI steps have substantially different
# image resolutions and therefore different activation sizes.
export PYTORCH_CUDA_ALLOC_CONF
IFS=',' read -ra GPU_ID_ARRAY <<< "${GPU_IDS}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${#GPU_ID_ARRAY[@]}}

cd "${EASYR1_ROOT}"

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
    data.max_pixels=${MAX_IMAGE_PIXELS} \
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
    worker.actor.micro_batch_size_per_device_for_experience=${EXPERIENCE_MICRO_BATCH_SIZE} \
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
