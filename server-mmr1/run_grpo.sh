#!/bin/bash
set -x
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

cd /mnt/data/ericdu/mmr1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/ericdu/conda_envs/verl-mmr1

export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TENSORBOARD_DIR=/mnt/data/ericdu/output/mmr1_3b_rl/tensorboard

JOB_NAME=mmr1_3b_rl bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    trainer.total_episodes=1 \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.fsdp_size=8 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    data.rollout_batch_size=128 \
    data.curriculum_rollout_batch_size=128 \
    data.max_response_length=4096 \
    worker.rollout.enforce_eager=true \
    worker.rollout.gpu_memory_utilization=0.75 \
    worker.rollout.max_num_batched_tokens=65536
