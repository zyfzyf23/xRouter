#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# =================== Frequently Used Variables ===================
export STEM_LLM_JUDGE_URL="http://10.3.91.185:8900"  # Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain

# We'll track PIDs so we can wait on them and detect errors
head_node_ip="127.0.0.1"
port=6395
address_head="${head_node_ip}:${port}"

export HYDRA_FULL_ERROR=1

# =================== SandboxFusion Configuration ===================
export SANDBOX_FUSION_SERVERS="ip-10-3-91-185"

# =================== Data Mixture ===================
# Note that the "/" must be at the end of the path!!
TRAIN_DATA_DIR=./data/train_filtered/
ONLINE_EVAL_DATA_DIR=./data/offline_eval/

export train_files="[$(
  ls "${TRAIN_DATA_DIR}/"/*.parquet \
    | xargs -n1 basename \
    | sed "s|^|'${TRAIN_DATA_DIR}|;s|$|'|" \
    | paste -sd, -
)]"
echo "train_files = $train_files"

export test_files="[$(
  ls "${ONLINE_EVAL_DATA_DIR}/"/*.parquet \
    | xargs -n1 basename \
    | sed "s|^|'${ONLINE_EVAL_DATA_DIR}|;s|$|'|" \
    | paste -sd, -
)]"
echo "test_files = $test_files"

# =================== Model ===================
BASE_MODEL=/fsx/home/cqian/projects/model/Qwen2.5-0.5B-Instruct

# =================== Logging ===================
WANDB_PROJECT=Router-Tool-RL
WANDB_EXPERIMENT_NAME="test_1003-${BASE_MODEL##*/}-01"

export WANDB_DISABLED=true
export RAY_TMPDIR=~/tmp

# # =================== Ray start ===================
# # ray stop at all nodes
# echo "Stopping any existing Ray cluster…"
# ray stop || true
# rm -rf /tmp/ray/ray_current_cluster

echo "Starting Ray head at ${address_head} ..."
# Start Ray head node  
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --include-dashboard=False --block &

sleep 5

# =================== RL Config ===================
# Note, we borrowed the config format from DAPO while here disabled all DAPO features to run the naive RL baseline.

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28 # default 0.2, higher as suggested in https://arxiv.org/pdf/2503.14476

max_prompt_length=$((1024 * 16))
max_response_length=$((1024 * 8))
max_model_length=$((1024 * 32))  # Set to desired value, e.g., 32K tokens
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=4  # Reduced for faster rollout
train_prompt_mini_bsz=128  # model grad update batchsize

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
sp_size=1
gen_tp=1
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length)))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length)))  # increase this to speed up modelforward, but note memory overflow
offload=True

# Tool Config
tool_config_path="./examples/sglang_multiturn/config/tool_config/router_tool_config.yaml"
max_turns=3
reward_lambda=3.0
reward_K=1.0
cost_max=0.8

# =================== Start RL training ===================
PYTHONUNBUFFERED=1 python3 -m verl.recipe.dapo.src.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_model_len=${max_model_length} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    actor_rollout_ref.rollout.multi_turn.cost_max=${cost_max} \
    actor_rollout_ref.rollout.multi_turn.reward_lambda=${reward_lambda} \
    actor_rollout_ref.rollout.multi_turn.reward_K=${reward_K} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.resume_mode=auto