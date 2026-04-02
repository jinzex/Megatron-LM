"""E2E TP-invariance validation: Toy MoE (Qwen3-30B-A3B architecture, 4 layers, 8 experts).

Validates that loss and grad_norm are bitwise identical across TP degrees.
Uses nproc_per_node=TP_SIZE so DP=1, PP=1, EP=1 — only TP varies.
Toy MoE (~0.5B params) fits on 1 GPU.

Usage:
    PROJ=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/jinzex/pre-training
    BRIDGE=$PROJ/third-party/Megatron-Bridge
    SCRIPT=$PROJ/projects/Numerics/tp-numerics/validate_e2e_moe_qwen3_toy.py

    # Common env vars for all runs:
    export PYTHONPATH=$BRIDGE/3rdparty/Megatron-LM:$BRIDGE/src
    export NVTE_TP_INVARIANT_MODE=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NCCL_ALGO=^NVLS NCCL_NVLS_ENABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export BIK=1 ATTN_BACKEND=unfused

    # Run for each TP degree (nproc = TP_SIZE):
    for TP in 1 2 4; do
        TP_SIZE=$TP TRAIN_ITERS=10 \
        torchrun --nproc_per_node=$TP $SCRIPT 2>&1 | tee /tmp/moe_tp${TP}.log
    done

    # Compare: grep "lm loss" /tmp/moe_tp*.log

Environment variables:
    TP_SIZE        - Tensor parallel degree (default: 1)
    TRAIN_ITERS    - Number of training iterations (default: 10)
    BIK            - Enable Batch Invariant Kernels (0/1, MANDATORY for MoE)
    ATTN_BACKEND   - Attention backend: unfused/flash/auto (default: model default)
    NO_CLIP        - Disable gradient clipping (0/1, default: 0)
"""
import os, inspect

os.environ.setdefault("TP_SIZE", "1")
os.environ.setdefault("TRAIN_ITERS", "10")
USE_BIK = os.environ.get("BIK", "0") == "1"

if not USE_BIK:
    print("WARNING: BIK=1 is MANDATORY for MoE TP-invariance (router GEMM has variable M-dim)")
    print("         Set BIK=1 to enable. Proceeding without BIK — results will NOT be TP-invariant.")

if USE_BIK:
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
    )
    enable_batch_invariant_mode()
    print("BIK: aten/TE kernel patches ENABLED")

from megatron.core.transformer.enums import AttnBackend
from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

tp_size = int(os.environ["TP_SIZE"])
train_iters = int(os.environ["TRAIN_ITERS"])

config = qwen3_30b_a3b_pretrain_config(
    mock=True,
    use_null_tokenizer=True,
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=1,
    sequence_parallel=(tp_size > 1),
    global_batch_size=1,
    micro_batch_size=1,
    seq_length=2048,
    train_iters=train_iters,
    lr_warmup_iters=0,
    lr_decay_iters=train_iters,
)

# Override to toy size
config.model.num_layers = 4
config.model.num_moe_experts = 8
config.model.moe_router_topk = 2
config.model.moe_token_dispatcher_type = "alltoall"

# CRITICAL: disable grouped GEMM — GroupedLinear bypasses BIK-patched TE code
config.model.moe_grouped_gemm = False

config.model.use_cpu_initialization = True
config.model.deterministic_mode = True
config.model.cross_entropy_loss_fusion = False

# Isolation test: disable aux loss to check if it's the source of backward divergence
if os.environ.get("NO_AUX_LOSS", "0") == "1":
    config.model.moe_aux_loss_coeff = 0.0
    print("MoE auxiliary loss DISABLED (moe_aux_loss_coeff=0)")

# Attention backend
_attn_backend = os.environ.get("ATTN_BACKEND", "").lower()
if _attn_backend:
    config.model.attention_backend = getattr(AttnBackend, _attn_backend)

if USE_BIK:
    if config.model.attention_backend == AttnBackend.unfused:
        config.model.batch_invariant_mode = False
    else:
        config.model.batch_invariant_mode = True
    print(f"BIK: config.batch_invariant_mode={config.model.batch_invariant_mode}, attention_backend={config.model.attention_backend}")

config.model.cuda_graph_impl = "none"
config.model.cuda_graph_scope = []

if config.comm_overlap is not None:
    config.comm_overlap.tp_comm_overlap = False
    config.comm_overlap.delay_wgrad_compute = False

config.ddp.data_parallel_sharding_strategy = "no_shard"

if os.environ.get("NO_CLIP", "0") == "1":
    config.optimizer.clip_grad = float('inf')
    print("Gradient clipping DISABLED (clip_grad=inf)")

config.logger.tensorboard_dir = "/tmp/tp-inv-moe-toy/tensorboard"
config.logger.log_interval = 1
config.logger.wandb_project = ""
config.logger.wandb_exp_name = ""

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

assert tp == tp_size, f"TP mismatch: config={tp}, env={tp_size}"
assert dp == world_size // tp, f"DP mismatch: {dp} != {world_size}//{tp}"
if tp > 1:
    assert config.model.sequence_parallel, "SP must be True when TP > 1"

import transformer_engine.pytorch.module.linear as _te_linear
src = inspect.getsource(_te_linear._Linear.forward)
patched = "tp_invariant" in src.lower() or "NVTE_TP_INVARIANT_MODE" in src
print(f"TE TP-invariant patch: {'ACTIVE' if patched else 'NOT FOUND'}")

print(f"""
======== TP-Invariant E2E: Toy MoE (4L, 8E, top-2) ========
GPUs: {world_size} | TP: {tp} | PP: {pp} | DP: {dp} | GA: {ga}
GBS: {gbs} | MBS: {mbs} | SeqLen: {config.model.seq_length}
Layers: {config.model.num_layers} | Experts: {config.model.num_moe_experts} | TopK: {config.model.moe_router_topk}
GroupedGEMM: {config.model.moe_grouped_gemm} | Dispatcher: {config.model.moe_token_dispatcher_type}
Iters: {train_iters} | SP: {config.model.sequence_parallel}
Deterministic: {config.model.deterministic_mode}
NVTE_TP_INVARIANT_MODE: {os.environ.get('NVTE_TP_INVARIANT_MODE', '0')}
BIK: {USE_BIK}
============================================================
""")

# Dump parallel group info for debugging
if os.environ.get("DUMP_GROUPS", "0") == "1":
    import megatron.core.parallel_state as _ps
    def _dump_groups_hook(config_unused):
        if _ps.get_tensor_model_parallel_rank() == 0:
            print(f"\n=== PARALLEL GROUPS (rank 0) ===")
            print(f"  TP size: {_ps.get_tensor_model_parallel_world_size()}")
            print(f"  DP size: {_ps.get_data_parallel_world_size()}")
            try:
                print(f"  EP size: {_ps.get_expert_model_parallel_world_size()}")
            except: pass
            try:
                edp = _ps.get_data_parallel_group(with_expert_parallel=True)
                print(f"  Expert DP group size: {edp.size()}")
            except Exception as e:
                print(f"  Expert DP group: {e}")
            # Check what group expert weights allreduce over
            from megatron.core.distributed import distributed_data_parallel as _ddp_mod
            print(f"=== END GROUPS ===\n")
    import megatron.training.training as _train_mod
    _orig_setup = getattr(_train_mod, 'setup_model_and_optimizer', None)

if os.environ.get("DUMP_GRADS", "0") == "1":
    import torch, hashlib
    from megatron.core import parallel_state as ps
    import megatron.core.optimizer.clip_grads as _cg
    import megatron.core.optimizer.optimizer as _opt_mod

    _orig_get_norm = _cg.get_grad_norm_fp32
    _dump_file = os.environ.get("DUMP_GRADS_FILE", "/tmp/moe_grads.txt")

    def _patched_get_norm(grads_for_norm, *a, **kw):
        result = _orig_get_norm(grads_for_norm, *a, **kw)
        if ps.get_tensor_model_parallel_rank() == 0:
            with open(_dump_file, "w") as f:
                for i, g in enumerate(grads_for_norm):
                    gf = g.float()
                    h = hashlib.md5(gf.cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
                    f.write(f"{i:3d} shape={str(list(g.shape)):30s} norm={gf.norm().item():.10f} hash={h}\n")
                f.write(f"total_norm={result:.10f}\n")
        return result

    # Patch at both module level AND the imported reference in optimizer.py
    _cg.get_grad_norm_fp32 = _patched_get_norm
    _opt_mod.get_grad_norm_fp32 = _patched_get_norm

pretrain(config=config, forward_step_func=forward_step)
