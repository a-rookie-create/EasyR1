#!/bin/bash

set -e
set -x

MODEL_PATH=${MODEL_PATH:-/home/zst/biye215/models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct}
DATA_DIR=${DATA_DIR:-/home/zst/biye215/datasets/ui_s1_easy_r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_ui_s1_grpo_lora_smoke}

cd /home/zst/biye215/EasyR1

python3 -B -c "import json, pathlib, sys; data=pathlib.Path('${DATA_DIR}/android_control_train_steps.jsonl'); assert data.exists(), f'missing train file: {data}'; first=json.loads(data.read_text(encoding='utf-8').splitlines()[0]); image=pathlib.Path(first['images'][0]); assert image.exists(), f'missing first screenshot: {image}. Regenerate data with --image-prefix-from/--image-prefix-to or mount the image directory into the container.'"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/android_control_train_steps.jsonl \
    data.val_files=${DATA_DIR}/android_control_val_steps.jsonl \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.image_dir=null \
    data.max_prompt_length=12288 \
    data.max_response_length=512 \
    data.rollout_batch_size=2 \
    data.val_batch_size=2 \
    data.format_prompt=examples/ui_s1/format_prompt/ui_s1_android.jinja \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_loss=true \
    algorithm.kl_coef=1.0e-4 \
    worker.actor.global_batch_size=2 \
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
    worker.rollout.gpu_memory_utilization=0.65 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=examples/ui_s1/reward_ui_s1_step.py:compute_score \
    trainer.max_steps=1 \
    trainer.total_epochs=1 \
    trainer.project_name=easy_r1_ui_s1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1
