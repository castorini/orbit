#!/bin/bash
set -x

unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3 # We ran this on 4xH100 GPUs

RETRIEVER_PORT=8280
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and data
model_name="${1:-Qwen/Qwen3-4B}"
dataset_name="${2:-mix-nq-hotpotqa-infoseek-ratio-1-1-1-modified}"
train_data="${SCRIPT_DIR}/data/${dataset_name}/train.parquet"
val_data="${SCRIPT_DIR}/data/${dataset_name}/test.parquet"

# export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/n3thakur/scratch/triton_cache}
# export HF_HOME=/home/n3thakur/scratch/cache
# export DATASETS_HF_HOME=/home/n3thakur/scratch/cache

# Retriever
retriever_url="http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
retriever_topk=5
action_stop_tokens="</search>,</answer>"

# Training hyperparameters
rl_alg=grpo
n_gpus_per_node=4
n_nodes=1
n=8
batch_size=256
val_batch_size=256
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
max_prompt_length=2048
max_response_length=8192
max_action_length=2048
max_obs_length=1024
max_turns=5
max_concurrent_trajectories=32
temperature=1.0
top_p=1.0
strategy="fsdp"
lr=1e-6
kl_loss_coef=0.0
kl_coef=0
kl_loss_type=low_var_kl
entropy_coeff=0
gpu_memory_utilization=0.6
do_offload=True
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
tensor_model_parallel_size=1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=False
reward_manager=search_r1_qa_em
rollout_mode=async
total_epochs=10
total_training_steps=1005

# Run name
model_pretty_name=$(echo "$model_name" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_]+/-/g' | cut -d'-' -f1-4)
dataset_pretty_name=$(echo "$dataset_name" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_]+/-/g' | cut -d'-' -f1-4)
run_name="${model_pretty_name}-${dataset_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-search_r1-5turns-5docs"

export VERL_RUN_ID=$run_name
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARNING
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RETRIEVER_PORT
export RETRIEVER_URL=$retriever_url
export RETRIEVER_TOPK=$retriever_topk
export RETRIEVER_TIMEOUT=120
export DEFAULT_TIMEOUT=120

# Start retrieval server
bash "${SCRIPT_DIR}/start_retriever.sh" "$RETRIEVER_PORT" &
retriever_pid=$!

echo "Waiting for retrieval server (PID=$retriever_pid) on port $RETRIEVER_PORT..."
until curl -sf -X POST http://127.0.0.1:${RETRIEVER_PORT}/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries":["test"],"topk":1,"return_scores":false}' > /dev/null 2>&1; do
    sleep 3
done
echo "Retrieval server is ready."

# Action stop tokens file
mkdir -p "${SCRIPT_DIR}/tmp"
action_stop_tokens_file="${SCRIPT_DIR}/tmp/action_stop_tokens.txt"
echo -n "$action_stop_tokens" > "$action_stop_tokens_file"

# Start tool server
tool_host="0.0.0.0"
tool_port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$tool_host:$tool_port/get_observation

python -m verl_tool.servers.serve \
    --host $tool_host \
    --port $tool_port \
    --tool_type "search_retrieval" \
    --workers_per_tool 4 \
    --request_timeout $DEFAULT_TIMEOUT &
tool_server_pid=$!

echo "Waiting for tool server (PID=$tool_server_pid) at $tool_server_url..."
until curl -sf http://$tool_host:$tool_port/health > /dev/null 2>&1; do
    sleep 2
done
echo "Tool server is ready."

# Run training
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    data.seed=42 \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent_trajectories \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.agent.default_agent_loop=verltool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.rollout_data_dir=${SCRIPT_DIR}/verl_step_records/$run_name \
    trainer.validation_data_dir=${SCRIPT_DIR}/verl_step_records/$run_name-val \
    trainer.default_local_dir=/home/n3thakur/scratch/browsecomp/models/$run_name \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps

# Cleanup
echo "Training completed. Cleaning up..."
kill -9 $tool_server_pid 2>/dev/null || true
kill -9 $retriever_pid 2>/dev/null || true
echo "Done."
