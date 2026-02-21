#!/usr/bin/env bash
# train_granite_grpo.sh
#
# GRPO training of ibm-granite/granite-4.0-h-tiny with:
#   • Arithmetic tool server
#   • BM25 answer reward
#   • LoRA rank-32 on attention + Mamba projection layers
#
# Prerequisites:
#   bash setup.sh          (copies tool + reward manager, builds dataset)
#
# Adjust GPU_IDS, n_gpus_per_node, and batch sizes for your cluster.

set -xeuo pipefail

# ── GPU / environment ────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/parquet"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

train_data="$DATA_DIR/train.parquet"
val_data="[$DATA_DIR/test.parquet]"

# ── Model ────────────────────────────────────────────────────────────────
model_name="ibm-granite/granite-4.0-h-tiny"

# ── LoRA ─────────────────────────────────────────────────────────────────
lora_rank=32
lora_alpha=32
# Note: module names with dots (shared_mlp.*) are passed as plain strings
# inside the Hydra list — Hydra treats them as string values, not key paths.
target_modules="[q_proj,k_proj,v_proj,o_proj,shared_mlp.input_linear,shared_mlp.output_linear,mamba.in_proj]"

# ── Algorithm ────────────────────────────────────────────────────────────
rl_alg=grpo
n=8                          # rollout samples per prompt (GRPO group size)
batch_size=16
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=4

# ── Sequence lengths ─────────────────────────────────────────────────────
max_prompt_length=512
max_response_length=1024
max_obs_length=256
max_action_length=512
max_turns=5

# ── Rollout / sampling ───────────────────────────────────────────────────
temperature=1.0
top_p=1.0
gpu_memory_utilization=0.5   # leave headroom for training
tensor_model_parallel_size=1

# ── Training ─────────────────────────────────────────────────────────────
lr=1e-5
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
do_offload=True
use_dynamic_bsz=True
strategy=fsdp
fsdp_size=-1
ulysses_sequence_parallel_size=1

# ── Reward / agent ───────────────────────────────────────────────────────
reward_manager=bm25
enable_agent=True
mask_observations=True
enable_mtrl=False
rollout_mode=async

# EOS token IDs for Granite-4.0.
# Run: python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-h-tiny', trust_remote_code=True); print(t.eos_token_id)"
# and set the value here.  Use [] to rely solely on the model's default EOS.
additional_eos_token_ids=[]

n_gpus_per_node=1
n_nodes=1

# ── Action stop token ────────────────────────────────────────────────────
# The model stops after closing the tool call tag.  The server then returns
# a full <tool_response>...</tool_response> block as the observation.
# Written to a temp file because verl cannot receive angle-bracket strings
# as direct CLI arguments.
action_stop_tokens='</tool_call>'
action_stop_tokens_file="$(mktemp)"
printf '%s' "$action_stop_tokens" > "$action_stop_tokens_file"
echo "action_stop_tokens_file=$action_stop_tokens_file"

# ── Run name ─────────────────────────────────────────────────────────────
model_pretty_name=$(echo "$model_name" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="bm25-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-lr${lr}-lora${lora_rank}"
export VERL_RUN_ID="$run_name"

# ── Tool server ──────────────────────────────────────────────────────────
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url="http://$host:$port/get_observation"

uv run python -m verl_tool.servers.serve \
    --host "$host" \
    --port "$port" \
    --tool_type "arithmetic" \
    --workers_per_tool 4 \
    --use_ray=True \
    > "$LOG_DIR/tool_server.log" 2>&1 &
server_pid=$!
echo "Tool server (pid=$server_pid) started at $tool_server_url"

# Give the server a moment to initialise
sleep 5

# ── Training ─────────────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 uv run python -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=1024 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=$lora_rank \
    actor_rollout_ref.model.lora_alpha=$lora_alpha \
    "actor_rollout_ref.model.target_modules=$target_modules" \
    \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url="$tool_server_url" \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens="$action_stop_tokens_file" \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    +actor_rollout_ref.agent.retokenization=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path="$model_name" \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    \
    trainer.logger=['console'] \
    trainer.experiment_name="$run_name" \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.rollout_data_dir="$SCRIPT_DIR/verl_step_records/$run_name" \
    trainer.validation_data_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5

# ── Cleanup ──────────────────────────────────────────────────────────────
pkill -P "$server_pid" 2>/dev/null || true
kill -9 "$server_pid" 2>/dev/null || true
rm -f "$action_stop_tokens_file"
