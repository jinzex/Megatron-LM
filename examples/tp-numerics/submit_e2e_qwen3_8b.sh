#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --job-name=tp-inv-8b
#SBATCH --output=slurm_%j.log
#SBATCH --exclusive

# ==============================================================================
# E2E TP-Invariant Validation — Qwen3-8B Dense (1 node)
#
# Validates that TP-invariant mode produces identical loss curves across TP
# degrees for a dense transformer model.
#
# Usage:
#   TP_SIZE=1 sbatch submit_e2e_qwen3_8b.sh    # baseline
#   TP_SIZE=2 sbatch submit_e2e_qwen3_8b.sh
#   TP_SIZE=4 sbatch submit_e2e_qwen3_8b.sh
#   TP_SIZE=8 sbatch submit_e2e_qwen3_8b.sh
# ==============================================================================

set -ex

TP_SIZE=${TP_SIZE:-1}
TRAIN_ITERS=${TRAIN_ITERS:-100}
BIK=${BIK:-0}

WORKSPACE="/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/jinzex/pre-training"
MEGATRON_BRIDGE_PATH="${WORKSPACE}/third-party/Megatron-Bridge"
MEGATRON_LM_PATH="${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM"
CONTAINER_IMAGE="${WORKSPACE}/containers/nemo-25.11.01.sqsh"
CONTAINER_MOUNTS="/lustre/:/lustre/"
PATCHES_DIR="${WORKSPACE}/projects/Numerics/tp-numerics/patches"

WANDB_EXP_NAME="tp-inv-qwen3-8b-tp${TP_SIZE}"
OUTPUT_DIR="${WORKSPACE}/projects/Numerics/tp-numerics/output/${WANDB_EXP_NAME}"

source "${WORKSPACE}/.env"

###############################################################################
# Environment — TP-invariant mode
###############################################################################

export NVTE_TP_INVARIANT_MODE=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_ALGO=^NVLS

mkdir -p ${OUTPUT_DIR}

###############################################################################
# Training script (embedded Python)
###############################################################################

TEMP_SCRIPT="${OUTPUT_DIR}/${SLURM_JOB_ID}_train.py"

cat > ${TEMP_SCRIPT} << 'PYTHON_EOF'
import os, inspect

USE_BIK = os.environ.get("BIK", "0") == "1"
if USE_BIK:
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
    )
    enable_batch_invariant_mode()
    print("BIK: aten/TE kernel patches ENABLED")

from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

tp_size = int(os.environ["TP_SIZE"])
train_iters = int(os.environ["TRAIN_ITERS"])

config = qwen3_8b_pretrain_config(
    mock=True,
    use_null_tokenizer=True,
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=1,
    sequence_parallel=(tp_size > 1),
    global_batch_size=8,
    micro_batch_size=1,
    seq_length=2048,
    train_iters=train_iters,
    lr_warmup_iters=0,
    lr_decay_iters=train_iters,
)

config.model.use_cpu_initialization = True
config.model.deterministic_mode = True
config.model.cross_entropy_loss_fusion = False
config.model.recompute_granularity = "selective"
config.model.recompute_modules = ["core_attn"]

if USE_BIK:
    config.model.batch_invariant_mode = True
    print(f"BIK: config.batch_invariant_mode=True, attention_backend={config.model.attention_backend} (num_splits=1)")

config.model.cuda_graph_impl = "none"
config.model.cuda_graph_scope = []

if config.comm_overlap is not None:
    config.comm_overlap.tp_comm_overlap = False
    config.comm_overlap.delay_wgrad_compute = False

config.ddp.data_parallel_sharding_strategy = "no_shard"

config.logger.tensorboard_dir = os.environ["OUTPUT_DIR"] + "/tensorboard"
config.logger.wandb_project = os.environ.get("WANDB_PROJECT", "")
config.logger.wandb_exp_name = os.environ["WANDB_EXP_NAME"]
config.logger.wandb_entity = os.environ.get("WANDB_ENTITY", "")
config.logger.wandb_save_dir = "/nemo_run/wandb"
config.logger.log_interval = 1

config.checkpoint.save_interval = train_iters + 1
config.checkpoint.save = None
config.checkpoint.load = None
config.train.eval_interval = train_iters + 1

world_size = int(os.environ.get("WORLD_SIZE", 1))
tp = config.model.tensor_model_parallel_size
pp = config.model.pipeline_model_parallel_size
dp = world_size // (tp * pp)
gbs = config.train.global_batch_size
mbs = config.train.micro_batch_size
ga = gbs // (mbs * dp) if dp > 0 else 0

# Verify TE patch
import transformer_engine.pytorch.module.linear as _te_linear
src = inspect.getsource(_te_linear._Linear.forward)
patched = "tp_invariant" in src.lower() or "NVTE_TP_INVARIANT_MODE" in src
print(f"TE TP-invariant patch: {'ACTIVE' if patched else 'NOT FOUND'}")

print(f"""
======== TP-Invariant E2E: Qwen3-8B Dense ========
GPUs: {world_size} | TP: {tp} | PP: {pp} | DP: {dp} | GA: {ga}
GBS: {gbs} | MBS: {mbs} | SeqLen: {config.model.seq_length}
Iters: {train_iters} | SP: {config.model.sequence_parallel}
Deterministic: {config.model.deterministic_mode}
NVTE_TP_INVARIANT_MODE: {os.environ.get('NVTE_TP_INVARIANT_MODE', '0')}
BIK: {USE_BIK}
===================================================
""")

pretrain(config=config, forward_step_func=forward_step)
PYTHON_EOF

###############################################################################
# Launch
###############################################################################

echo "======================================"
echo "TP-Invariant E2E: Qwen3-8B Dense"
echo "TP=${TP_SIZE} | Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "======================================"

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
NNODES=${SLURM_JOB_NUM_NODES:-1}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
TE_MOD="/opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/module"

srun \
    --no-container-mount-home \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${MEGATRON_BRIDGE_PATH} \
    bash -c "
        # Install TE module patches (TP-invariant linear layers)
        cp ${PATCHES_DIR}/layernorm_linear.py ${TE_MOD}/layernorm_linear.py
        cp ${PATCHES_DIR}/linear.py           ${TE_MOD}/linear.py

        # Install TE attention patches (FA3 num_splits passthrough for TE 2.9)
        TE_DPA="/opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/attention/dot_product_attention"
        cp ${PATCHES_DIR}/backends.py              \${TE_DPA}/backends.py
        cp ${PATCHES_DIR}/dot_product_attention.py \${TE_DPA}/dot_product_attention.py

        # Install Megatron-Core patches (cross-entropy + output projection)
        MCORE_TP=${MEGATRON_LM_PATH}/megatron/core/tensor_parallel
        cp ${PATCHES_DIR}/cross_entropy.py \${MCORE_TP}/cross_entropy.py
        cp ${PATCHES_DIR}/layers.py        \${MCORE_TP}/layers.py

        export PYTHONPATH=${MEGATRON_LM_PATH}:${MEGATRON_BRIDGE_PATH}/src:\${PYTHONPATH:-}
        export MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH}

        export MASTER_ADDR=${MASTER_ADDR}
        export MASTER_PORT=29500
        export WORLD_SIZE=${WORLD_SIZE}
        export RANK=\${SLURM_PROCID}
        export LOCAL_RANK=\${SLURM_LOCALID}

        export TP_SIZE=${TP_SIZE}
        export TRAIN_ITERS=${TRAIN_ITERS}
        export BIK=${BIK}
        export OUTPUT_DIR=${OUTPUT_DIR}
        export WANDB_EXP_NAME=${WANDB_EXP_NAME}

        python ${TEMP_SCRIPT}
    " 2>&1 | tee ${OUTPUT_DIR}/${SLURM_JOB_ID}.log

echo "Job completed: ${WANDB_EXP_NAME}"
