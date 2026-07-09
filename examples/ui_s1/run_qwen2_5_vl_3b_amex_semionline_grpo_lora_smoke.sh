#!/bin/bash

set -e

MODEL_PATH=${MODEL_PATH:-/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct}
DATA_DIR=${DATA_DIR:-/home/zst/biye215/datasets/ui_s1_easy_r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_amex_semionline_grpo_lora_smoke}
RUN_LOG_DIR=${RUN_LOG_DIR:-/home/zst/biye215/EasyR1/logs/ui_s1}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2}}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-3}
ACTOR_GLOBAL_BATCH_SIZE=${ACTOR_GLOBAL_BATCH_SIZE:-3}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.75}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-12288}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-24576}

mkdir -p "${RUN_LOG_DIR}"
RUN_LOG=${RUN_LOG:-${RUN_LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "${RUN_LOG}") 2>&1
set -x
echo "Writing run log to ${RUN_LOG}"

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
IFS=',' read -ra GPU_ID_ARRAY <<< "${GPU_IDS}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${#GPU_ID_ARRAY[@]}}

cd /home/zst/biye215/EasyR1

python3 -B -c "import json, pathlib; data=pathlib.Path('${DATA_DIR}/amex_train_trajectories.jsonl'); assert data.exists(), f'missing train file: {data}'; first=json.loads(data.read_text(encoding='utf-8').splitlines()[0]); image=pathlib.Path(first['trajectory_steps'][0]['image']); assert image.exists(), f'missing first screenshot: {image}'"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/amex_train_trajectories.jsonl \
    data.val_files=${DATA_DIR}/amex_val_trajectories.jsonl \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.image_dir=null \
    data.max_prompt_length=8192 \
    data.max_response_length=256 \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=1 \
    data.format_prompt=examples/ui_s1/format_prompt/ui_s1_android.jinja \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo \
    algorithm.semi_online=true \
    algorithm.patch_threshold=2 \
    algorithm.use_kl_loss=true \
    algorithm.kl_coef=1.0e-4 \
    worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.model.lora.rank=64 \
    worker.actor.model.lora.alpha=32 \
    worker.actor.model.lora.target_modules=all-linear \
    worker.actor.model.lora.exclude_modules='.*visual.*' \
    worker.actor.optim.lr=1.0e-6 \
    worker.rollout.n=2 \
    worker.rollout.temperature=0.9 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=1 \
    worker.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=${VLLM_MAX_MODEL_LEN} \
    worker.rollout.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} \
    worker.reward.reward_function=examples/ui_s1/reward_ui_s1_step.py:compute_score \
    trainer.max_steps=1 \
    trainer.total_epochs=1 \
    trainer.project_name=easy_r1_ui_s1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console","file"]' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1
