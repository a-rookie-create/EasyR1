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
    VLLM_ENFORCE_EAGER
    PYTORCH_CUDA_ALLOC_CONF
)
for required_var in "${required_runtime_vars[@]}"; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "Missing required runtime setting: ${required_var}" >&2
        exit 2
    fi
done

DATASET_LABEL=${DATASET_LABEL:-${DATASET}}
RUN_NAME=${RUN_NAME:-ui_s1_qwen25vl_3b_${DATASET_LABEL}_semionline_grpo_lora_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
RESUME=${RESUME:-false}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
WARM_START_CHECKPOINT_PATH=${WARM_START_CHECKPOINT_PATH:-}
WARM_START_DATALOADER=${WARM_START_DATALOADER:-inherit}
VAL_AFTER_TRAIN=${VAL_AFTER_TRAIN:-true}

TRAINER_MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
    TRAINER_MAX_STEPS_ARG=("trainer.max_steps=${MAX_STEPS}")
fi

case "${RESUME}" in
    true|false) ;;
    *) echo "RESUME must be true or false, got ${RESUME}." >&2; exit 2 ;;
esac
case "${WARM_START_DATALOADER}" in
    inherit|reset) ;;
    *) echo "WARM_START_DATALOADER must be inherit or reset, got ${WARM_START_DATALOADER}." >&2; exit 2 ;;
esac
case "${VAL_AFTER_TRAIN}" in
    true|false) ;;
    *) echo "VAL_AFTER_TRAIN must be true or false, got ${VAL_AFTER_TRAIN}." >&2; exit 2 ;;
esac
case "${VLLM_ENFORCE_EAGER}" in
    true|false) ;;
    *) echo "VLLM_ENFORCE_EAGER must be true or false, got ${VLLM_ENFORCE_EAGER}." >&2; exit 2 ;;
esac

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

TRAINER_RESUME_ARGS=()
TRAINER_WARM_START_ARGS=()
WARM_START_ADAPTER_PATH=
WARM_START_STEP=0
if [[ -n "${WARM_START_CHECKPOINT_PATH}" ]]; then
    if [[ "${RESUME}" == "true" ]]; then
        echo "RESUME=true and WARM_START_CHECKPOINT_PATH are mutually exclusive." >&2
        exit 2
    fi
    if [[ -e "${RUN_DIR}" ]]; then
        echo "Warm start requires a new RUN_NAME; output directory already exists: ${RUN_DIR}" >&2
        exit 2
    fi
    if [[ ! -d "${WARM_START_CHECKPOINT_PATH}" ]]; then
        echo "Warm-start checkpoint directory does not exist: ${WARM_START_CHECKPOINT_PATH}" >&2
        exit 2
    fi
    RESOLVED_WARM_START_CHECKPOINT_PATH=$(cd -- "${WARM_START_CHECKPOINT_PATH}" && pwd)
    if [[ "$(basename -- "${RESOLVED_WARM_START_CHECKPOINT_PATH}")" =~ ^global_step_([1-9][0-9]*)$ ]]; then
        WARM_START_STEP=${BASH_REMATCH[1]}
    else
        echo "Warm-start checkpoint must end with global_step_<positive integer>: ${RESOLVED_WARM_START_CHECKPOINT_PATH}" >&2
        exit 2
    fi
    WARM_START_ADAPTER_PATH="${RESOLVED_WARM_START_CHECKPOINT_PATH}/actor/lora_adapter"
    if ! python3 -c 'import json, sys
from pathlib import Path
adapter = Path(sys.argv[1])
expected_rank, expected_alpha = map(int, sys.argv[2:4])
expected_modules = {item.strip() for item in sys.argv[4].split(",") if item.strip()}
config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
actual_modules = set(config.get("target_modules", []))
actual_rank = config.get("r")
actual_alpha = config.get("lora_alpha")
if actual_rank != expected_rank or actual_alpha != expected_alpha or actual_modules != expected_modules:
    raise SystemExit("LoRA structure mismatch: adapter rank/alpha/modules=%r/%r/%r, expected=%r/%r/%r" % (actual_rank, actual_alpha, sorted(actual_modules), expected_rank, expected_alpha, sorted(expected_modules)))
' "${WARM_START_ADAPTER_PATH}" 64 32 "${LORA_TARGET_MODULES}"; then
        echo "Warm-start LoRA adapter is incompatible with this UI-S1 LoRA layout." >&2
        exit 2
    fi
    EXPECTED_DATA_JSON=$(python3 -c 'import json, sys
print(json.dumps({"train_files": sys.argv[1], "shuffle": True, "seed": 1, "rollout_batch_size": int(sys.argv[2]), "mini_rollout_batch_size": None}))
' "${TRAIN_FILE}" "${ROLLOUT_BATCH_SIZE}")
    python3 -B examples/ui_s1/checkpoint_lineage.py prepare-warm-start \
        --source-checkpoint "${RESOLVED_WARM_START_CHECKPOINT_PATH}" \
        --run-dir "${RUN_DIR}" \
        --dataloader-mode "${WARM_START_DATALOADER}" \
        --expected-data-json "${EXPECTED_DATA_JSON}"
    TRAINER_WARM_START_ARGS=(
        "trainer.warm_start_checkpoint_path=${RESOLVED_WARM_START_CHECKPOINT_PATH}"
        "trainer.warm_start_dataloader=${WARM_START_DATALOADER}"
        "trainer.warm_start_global_step=${WARM_START_STEP}"
        "trainer.append_existing_history=true"
        "worker.actor.model.lora_adapter_path=${WARM_START_ADAPTER_PATH}"
    )
fi

if [[ -e "${RUN_DIR}" ]]; then
    if [[ "${RESUME}" == "true" ]]; then
        if [[ ! -d "${RUN_DIR}" ]]; then
            echo "RUN_DIR exists but is not a directory: ${RUN_DIR}" >&2
            exit 2
        fi
        if [[ -z "${RESUME_CHECKPOINT_PATH}" && ! -f "${RUN_DIR}/checkpoint_tracker.json" ]]; then
            echo "Cannot resume: ${RUN_DIR}/checkpoint_tracker.json is missing." >&2
            echo "Set RESUME_CHECKPOINT_PATH to a complete global_step_* checkpoint if it lives elsewhere." >&2
            exit 2
        fi
    else
        if [[ -n "${WARM_START_CHECKPOINT_PATH}" ]]; then
            : # The lineage tool has just created this new warm-start directory.
        else
            echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
            echo "Set RESUME=true to restore its latest checkpoint, or set RUN_NAME to a new value." >&2
            exit 2
        fi
    fi
elif [[ "${RESUME}" == "true" && -z "${RESUME_CHECKPOINT_PATH}" ]]; then
    echo "Cannot resume: run directory does not exist: ${RUN_DIR}" >&2
    exit 2
fi

RESOLVED_RESUME_CHECKPOINT_PATH=
if [[ "${RESUME}" == "true" ]]; then
    if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
        if [[ ! -d "${RESUME_CHECKPOINT_PATH}" ]]; then
            echo "Cannot resume: checkpoint directory does not exist: ${RESUME_CHECKPOINT_PATH}" >&2
            exit 2
        fi
        RESOLVED_RESUME_CHECKPOINT_PATH=$(cd -- "${RESUME_CHECKPOINT_PATH}" && pwd)
    else
        if ! checkpoint_step=$(python3 -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    step = json.load(stream).get("last_global_step")
if type(step) is not int or step < 1:
    raise SystemExit("last_global_step must be a positive integer")
print(step)' "${RUN_DIR}/checkpoint_tracker.json"); then
            echo "Cannot resume: invalid checkpoint tracker: ${RUN_DIR}/checkpoint_tracker.json" >&2
            exit 2
        fi
        RESOLVED_RESUME_CHECKPOINT_PATH="${RUN_DIR}/global_step_${checkpoint_step}"
    fi
    if [[ ! "$(basename -- "${RESOLVED_RESUME_CHECKPOINT_PATH}")" =~ ^global_step_[1-9][0-9]*$ ]]; then
        echo "Cannot resume: checkpoint must end with global_step_<positive integer>: ${RESOLVED_RESUME_CHECKPOINT_PATH}" >&2
        exit 2
    fi
    if [[ ! -d "${RESOLVED_RESUME_CHECKPOINT_PATH}/actor" ]]; then
        echo "Cannot resume: actor checkpoint directory is missing: ${RESOLVED_RESUME_CHECKPOINT_PATH}/actor" >&2
        exit 2
    fi
    if [[ ! -f "${RESOLVED_RESUME_CHECKPOINT_PATH}/dataloader.pt" ]]; then
        echo "Cannot resume: dataloader state is missing: ${RESOLVED_RESUME_CHECKPOINT_PATH}/dataloader.pt" >&2
        exit 2
    fi
    if [[ "$(dirname -- "${RESOLVED_RESUME_CHECKPOINT_PATH}")" != "$(cd -- "${RUN_DIR}" && pwd)" ]]; then
        echo "Strict RESUME checkpoint must be inside RUN_DIR; use WARM_START_CHECKPOINT_PATH for a checkpoint from another run." >&2
        exit 2
    fi
    TRAINER_RESUME_ARGS=("trainer.find_last_checkpoint=true")
    if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
        TRAINER_RESUME_ARGS+=("trainer.load_checkpoint_path=${RESOLVED_RESUME_CHECKPOINT_PATH}")
    fi
else
    TRAINER_RESUME_ARGS=("trainer.find_last_checkpoint=false")
fi
mkdir -p "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export RAY_DASHBOARD_HOST
# Avoid allocator fragmentation when UI steps have substantially different
# image resolutions and therefore different activation sizes.
export PYTORCH_CUDA_ALLOC_CONF
IFS=',' read -ra GPU_ID_ARRAY <<< "${GPU_IDS}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${#GPU_ID_ARRAY[@]}}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_nonnegative_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

is_positive_number() {
    is_nonnegative_number "$1" && awk -v value="$1" 'BEGIN { exit !((value + 0) > 0) }'
}

is_unit_interval_number() {
    is_positive_number "$1" && awk -v value="$1" 'BEGIN { exit !((value + 0) <= 1) }'
}

is_number_less_than_or_equal() {
    awk -v left="$1" -v right="$2" 'BEGIN { exit !((left + 0) <= (right + 0)) }'
}

is_positive_integer "${N_GPUS_PER_NODE}" || {
    echo "N_GPUS_PER_NODE must be a positive integer, got ${N_GPUS_PER_NODE}." >&2
    exit 2
}
if (( N_GPUS_PER_NODE != ${#GPU_ID_ARRAY[@]} )); then
    echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) must match the number of GPU_IDS entries (${#GPU_ID_ARRAY[@]})." >&2
    exit 2
fi
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "GPU_IDS entries must be non-negative integer device indices, got ${gpu_id@Q}." >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
        echo "GPU_IDS must not contain duplicate device indices, got ${GPU_IDS}." >&2
        exit 2
    fi
    SEEN_GPU_IDS["${gpu_id}"]=1
done
is_positive_integer "${ROLLOUT_BATCH_SIZE}" || {
    echo "ROLLOUT_BATCH_SIZE must be a positive integer, got ${ROLLOUT_BATCH_SIZE}." >&2
    exit 2
}
is_positive_integer "${ACTOR_GLOBAL_BATCH_SIZE}" || {
    echo "ACTOR_GLOBAL_BATCH_SIZE must be a positive integer, got ${ACTOR_GLOBAL_BATCH_SIZE}." >&2
    exit 2
}
is_positive_integer "${ROLLOUT_N}" || {
    echo "ROLLOUT_N must be a positive integer, got ${ROLLOUT_N}." >&2
    exit 2
}
if (( ROLLOUT_N < 2 )); then
    echo "ROLLOUT_N must be at least 2 for GRPO, got ${ROLLOUT_N}." >&2
    exit 2
fi
is_positive_integer "${EXPERIENCE_MICRO_BATCH_SIZE}" || {
    echo "EXPERIENCE_MICRO_BATCH_SIZE must be a positive integer, got ${EXPERIENCE_MICRO_BATCH_SIZE}." >&2
    exit 2
}
is_positive_integer "${MAX_ROLLOUTS_PER_TASK}" || {
    echo "MAX_ROLLOUTS_PER_TASK must be a positive integer, got ${MAX_ROLLOUTS_PER_TASK}." >&2
    exit 2
}
is_positive_integer "${DIVERSITY_REFILL_BATCH_SIZE}" || {
    echo "DIVERSITY_REFILL_BATCH_SIZE must be a positive integer, got ${DIVERSITY_REFILL_BATCH_SIZE}." >&2
    exit 2
}
is_positive_integer "${VALIDATION_PROGRESS_INTERVAL}" || {
    echo "VALIDATION_PROGRESS_INTERVAL must be a positive integer, got ${VALIDATION_PROGRESS_INTERVAL}." >&2
    exit 2
}
is_nonnegative_number "${SAVE_INTERVAL_SECONDS}" || {
    echo "SAVE_INTERVAL_SECONDS must be a non-negative number, got ${SAVE_INTERVAL_SECONDS}." >&2
    exit 2
}
case "${PATCH_IMITATION_ENABLED}" in
    true|false) ;;
    *) echo "PATCH_IMITATION_ENABLED must be true or false, got ${PATCH_IMITATION_ENABLED}." >&2; exit 2 ;;
esac
is_nonnegative_number "${PATCH_IMITATION_LAMBDA_INITIAL}" || {
    echo "PATCH_IMITATION_LAMBDA_INITIAL must be a non-negative number, got ${PATCH_IMITATION_LAMBDA_INITIAL}." >&2
    exit 2
}
is_unit_interval_number "${PATCH_IMITATION_LAMBDA_DECAY}" || {
    echo "PATCH_IMITATION_LAMBDA_DECAY must be in the interval (0, 1], got ${PATCH_IMITATION_LAMBDA_DECAY}." >&2
    exit 2
}
is_nonnegative_number "${PATCH_IMITATION_LAMBDA_MIN}" || {
    echo "PATCH_IMITATION_LAMBDA_MIN must be a non-negative number, got ${PATCH_IMITATION_LAMBDA_MIN}." >&2
    exit 2
}
is_number_less_than_or_equal "${PATCH_IMITATION_LAMBDA_MIN}" "${PATCH_IMITATION_LAMBDA_INITIAL}" || {
    echo "PATCH_IMITATION_LAMBDA_MIN (${PATCH_IMITATION_LAMBDA_MIN}) must not exceed PATCH_IMITATION_LAMBDA_INITIAL (${PATCH_IMITATION_LAMBDA_INITIAL})." >&2
    exit 2
}
if [[ "${PATCH_IMITATION_ENABLED}" == "true" ]] && ! is_positive_number "${PATCH_IMITATION_LAMBDA_INITIAL}"; then
    echo "PATCH_IMITATION_LAMBDA_INITIAL must be positive when PATCH_IMITATION_ENABLED=true." >&2
    exit 2
fi
case "${PATCH_IMITATION_TARGET_MODE}" in
    action_only) ;;
    *) echo "PATCH_IMITATION_TARGET_MODE currently supports only action_only, got ${PATCH_IMITATION_TARGET_MODE}." >&2; exit 2 ;;
esac
case "${PATCH_HISTORY_MODE}" in
    keep_model_thinking) ;;
    *) echo "PATCH_HISTORY_MODE currently supports only keep_model_thinking, got ${PATCH_HISTORY_MODE}." >&2; exit 2 ;;
esac
if (( MAX_ROLLOUTS_PER_TASK < ROLLOUT_N )); then
    echo "MAX_ROLLOUTS_PER_TASK (${MAX_ROLLOUTS_PER_TASK}) must be at least ROLLOUT_N (${ROLLOUT_N})." >&2
    exit 2
fi
if (( ROLLOUT_BATCH_SIZE % ACTOR_GLOBAL_BATCH_SIZE != 0 )); then
    echo "ROLLOUT_BATCH_SIZE (${ROLLOUT_BATCH_SIZE}) must be divisible by ACTOR_GLOBAL_BATCH_SIZE (${ACTOR_GLOBAL_BATCH_SIZE})." >&2
    exit 2
fi
ACTOR_EFFECTIVE_GLOBAL_BATCH_SIZE=$((ACTOR_GLOBAL_BATCH_SIZE * ROLLOUT_N))
if (( ACTOR_EFFECTIVE_GLOBAL_BATCH_SIZE % N_GPUS_PER_NODE != 0 )); then
    echo "ACTOR_GLOBAL_BATCH_SIZE * ROLLOUT_N (${ACTOR_EFFECTIVE_GLOBAL_BATCH_SIZE}) must be divisible by N_GPUS_PER_NODE (${N_GPUS_PER_NODE})." >&2
    exit 2
fi
if [[ "${RESUME}" == "true" ]]; then
    checkpoint_model_shards=("${RESOLVED_RESUME_CHECKPOINT_PATH}/actor"/model_world_size_*_rank_*.pt)
    if (( ${#checkpoint_model_shards[@]} == 0 )); then
        echo "Cannot resume: actor model shards are missing in ${RESOLVED_RESUME_CHECKPOINT_PATH}/actor." >&2
        exit 2
    fi
    SAVED_ACTOR_WORLD_SIZE=
    for checkpoint_shard in "${checkpoint_model_shards[@]}"; do
        checkpoint_shard_name=$(basename -- "${checkpoint_shard}")
        if [[ ! "${checkpoint_shard_name}" =~ ^model_world_size_([1-9][0-9]*)_rank_[0-9]+[.]pt$ ]]; then
            continue
        fi
        shard_world_size=${BASH_REMATCH[1]}
        if [[ -z "${SAVED_ACTOR_WORLD_SIZE}" ]]; then
            SAVED_ACTOR_WORLD_SIZE=${shard_world_size}
        elif [[ "${SAVED_ACTOR_WORLD_SIZE}" != "${shard_world_size}" ]]; then
            echo "Cannot resume: checkpoint mixes actor shards from multiple world sizes." >&2
            exit 2
        fi
    done
    if [[ -z "${SAVED_ACTOR_WORLD_SIZE}" ]]; then
        echo "Cannot resume: actor shard filenames are invalid in ${RESOLVED_RESUME_CHECKPOINT_PATH}/actor." >&2
        exit 2
    fi
    if (( SAVED_ACTOR_WORLD_SIZE != N_GPUS_PER_NODE )); then
        echo "Cannot resume ${SAVED_ACTOR_WORLD_SIZE}-GPU FSDP shards with ${N_GPUS_PER_NODE} GPUs; direct cross-world-size resume is unsupported." >&2
        exit 2
    fi
    for ((rank = 0; rank < N_GPUS_PER_NODE; rank++)); do
        for shard_prefix in model optim extra_state; do
            expected_shard="${RESOLVED_RESUME_CHECKPOINT_PATH}/actor/${shard_prefix}_world_size_${N_GPUS_PER_NODE}_rank_${rank}.pt"
            if [[ ! -f "${expected_shard}" ]]; then
                echo "Cannot resume: checkpoint shard is missing: ${expected_shard}" >&2
                exit 2
            fi
        done
    done
fi

# A one-request micro batch causes the semi-online driver to make one vLLM
# call per trajectory. train.env records the two-GPU default; this fallback
# only applies when a custom TRAIN_ENV leaves the value empty.
GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE:-${N_GPUS_PER_NODE}}
is_nonnegative_integer "${GENERATION_MICRO_BATCH_SIZE}" || {
    echo "GENERATION_MICRO_BATCH_SIZE must be a non-negative integer, got ${GENERATION_MICRO_BATCH_SIZE}." >&2
    exit 2
}
if (( GENERATION_MICRO_BATCH_SIZE > 0 && GENERATION_MICRO_BATCH_SIZE < N_GPUS_PER_NODE )); then
    echo "WARNING: GENERATION_MICRO_BATCH_SIZE=${GENERATION_MICRO_BATCH_SIZE} is below the ${N_GPUS_PER_NODE} visible GPUs; padded duplicate requests cannot increase useful rollout concurrency." >&2
fi

echo "UI-S1 effective rollout configuration: tasks_per_update=${ROLLOUT_BATCH_SIZE}, selected_rollouts_per_task=${ROLLOUT_N}, actor_effective_global_batch=${ACTOR_EFFECTIVE_GLOBAL_BATCH_SIZE}, max_candidates_per_task=${MAX_ROLLOUTS_PER_TASK}, diversity_refill_batch=${DIVERSITY_REFILL_BATCH_SIZE}, generation_micro_batch=${GENERATION_MICRO_BATCH_SIZE}, visible_gpus=${N_GPUS_PER_NODE}."

cd "${EASYR1_ROOT}"

# Roll back the output lineage only after every mode, checkpoint, world-size,
# and training-parameter check has passed, but before any log writer, monitor,
# Ray process, or trainer starts.  This makes an explicit older checkpoint a
# real branch reset: all structured records and checkpoints after that step
# are removed before new records are appended.
if [[ "${RESUME}" == "true" ]]; then
    python3 -B examples/ui_s1/checkpoint_lineage.py prepare-resume-rollback \
        --run-dir "${RUN_DIR}" --checkpoint "${RESOLVED_RESUME_CHECKPOINT_PATH}"
fi

# All mode and parameter validation has completed.  Only now begin the raw
# stdout log, GPU monitor, and Ray training process.
RUN_LOG=${RUN_LOG:-${RUN_DIR}/train.log}
exec > >(tee -a "${RUN_LOG}") 2>&1
set -x

GPU_MEMORY_MONITOR_PATH="${RUN_DIR}/gpu_memory_peak.json"
python3 -B examples/ui_s1/monitor_gpu_memory.py \
    --gpu-ids "${GPU_IDS}" \
    --output "${GPU_MEMORY_MONITOR_PATH}" \
    --interval-seconds "${GPU_MEMORY_MONITOR_INTERVAL_SECONDS}" \
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" &
GPU_MEMORY_MONITOR_PID=$!
stop_gpu_memory_monitor() {
    if kill -0 "${GPU_MEMORY_MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${GPU_MEMORY_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MEMORY_MONITOR_PID}" || true
    fi
}
trap stop_gpu_memory_monitor EXIT
echo "GPU memory monitor: ${GPU_MEMORY_MONITOR_PATH} (interval=${GPU_MEMORY_MONITOR_INTERVAL_SECONDS}s)"

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
    algorithm.semi_online_diversity_refill_batch_size=${DIVERSITY_REFILL_BATCH_SIZE} \
    algorithm.patch_imitation.enabled=${PATCH_IMITATION_ENABLED} \
    algorithm.patch_imitation.lambda_initial=${PATCH_IMITATION_LAMBDA_INITIAL} \
    algorithm.patch_imitation.lambda_decay=${PATCH_IMITATION_LAMBDA_DECAY} \
    algorithm.patch_imitation.lambda_min=${PATCH_IMITATION_LAMBDA_MIN} \
    algorithm.patch_imitation.target_mode=${PATCH_IMITATION_TARGET_MODE} \
    algorithm.patch_imitation.history_mode=${PATCH_HISTORY_MODE} \
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
    "${TRAINER_WARM_START_ARGS[@]}" \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=0.9 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=${HISTORY_IMAGE_LIMIT} \
    worker.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
    worker.rollout.enable_sleep_mode=${VLLM_ENABLE_SLEEP_MODE} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enforce_eager=${VLLM_ENFORCE_EAGER} \
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
    trainer.val_after_train=${VAL_AFTER_TRAIN} \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_every_n_epochs=${SAVE_EVERY_N_EPOCHS} \
    trainer.save_interval_seconds=${SAVE_INTERVAL_SECONDS} \
    trainer.save_limit=${SAVE_LIMIT} \
    trainer.save_model_only=false \
    trainer.save_checkpoint_path=${RUN_DIR} \
    trainer.rollout_log_path=${RUN_DIR}/semi_online_rollouts.jsonl \
    trainer.progress_log_path=${RUN_DIR}/training_progress.log \
    trainer.progress_validation_interval=${VALIDATION_PROGRESS_INTERVAL} \
    "${TRAINER_RESUME_ARGS[@]}"
