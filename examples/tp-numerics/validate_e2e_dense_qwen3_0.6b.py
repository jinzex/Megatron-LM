"""E2E TP-invariance validation: Qwen3-0.6B Dense (28 layers).

Validates that loss and grad_norm are bitwise identical across TP degrees.
Uses nproc_per_node=TP_SIZE so DP=1, PP=1 — only TP varies.
Qwen3-0.6B (28 layers, H=1024, FFN=3072, 16 heads) fits on 1 GPU.

Usage:
    PROJ=<path-to-pre-training-repo>
    BRIDGE=$PROJ/third-party/Megatron-Bridge
    SCRIPT=$PROJ/projects/Numerics/tp-numerics/validate_e2e_dense_qwen3_0.6b.py

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
        torchrun --nproc_per_node=$TP $SCRIPT 2>&1 | tee /tmp/tp${TP}.log
    done

    # Compare: grep "lm loss" /tmp/tp*.log

Environment variables:
    TP_SIZE        - Tensor parallel degree (default: 1)
    TRAIN_ITERS    - Number of training iterations (default: 10)
    BIK            - Enable Batch Invariant Kernels (0/1, default: 0)
    ATTN_BACKEND   - Attention backend: unfused/flash/auto (default: model default)
    NO_CLIP        - Disable gradient clipping (0/1, default: 0)
"""
import os, inspect

os.environ.setdefault("TP_SIZE", "1")
os.environ.setdefault("TRAIN_ITERS", "10")
USE_BIK = os.environ.get("BIK", "0") == "1"

if USE_BIK:
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
    )
    enable_batch_invariant_mode()
    print("BIK: aten/TE kernel patches ENABLED")

from megatron.core.transformer.enums import AttnBackend
from megatron.bridge.recipes.qwen import qwen3_600m_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

# DUMP_DIR=<path> HW_TAG=<b300|h100> hooks each TransformerLayer's output, model logits,
# and loss on iter 1, saves to {DUMP_DIR}/e2e_dump_{HW_TAG}.pt, then continues.
import torch as _torch
from megatron.bridge.training import gpt_step as _gpt_step_mod

_DUMP_DIR = os.environ.get("DUMP_DIR")
if _DUMP_DIR:
    _HW_TAG = os.environ.get("HW_TAG", "unknown")
    _DUMPS = {}
    _DONE = [False]
    _HOOKED = [False]
    _orig_fs = _gpt_step_mod.forward_step
    _orig_fsc = _gpt_step_mod._forward_step_common

    def _layer_hook(name):
        def hk(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            _DUMPS[name] = t.detach().to("cpu", _torch.float32).clone()
        return hk

    def _hooked_fsc(state, data_iterator, model, return_schedule_plan=False):
        if not _HOOKED[0]:
            from megatron.core.utils import unwrap_model
            base = unwrap_model(model)
            if isinstance(base, list):
                base = base[0]
            # Snapshot a few key weight tensors to verify init determinism
            sd = base.state_dict()
            for k in list(sd.keys())[:3] + [k for k in sd.keys() if "layers.0." in k][:3]:
                if isinstance(sd[k], _torch.Tensor):
                    _DUMPS[f"weight::{k}"] = sd[k].detach().to("cpu", _torch.float32).clone()
            for i, layer in enumerate(base.decoder.layers):
                layer.register_forward_hook(_layer_hook(f"layer_{i:02d}_out"))
            _HOOKED[0] = True
            print(f"[DUMP] hooks on {len(base.decoder.layers)} layers, weights: {sum(1 for k in _DUMPS if k.startswith('weight::'))}", flush=True)
        output, loss_mask = _orig_fsc(state, data_iterator, model, return_schedule_plan)
        if not _DONE[0]:
            _DUMPS["model_output_per_token_loss"] = output.detach().to("cpu", _torch.float32).clone()
        return output, loss_mask

    def _hooked_fs(state, data_iterator, model, return_schedule_plan=False):
        output, loss_function = _orig_fs(state, data_iterator, model, return_schedule_plan)
        if not _DONE[0]:
            _orig_lf = loss_function
            def _wrapped_loss(output_tensor):
                lo = _orig_lf(output_tensor)
                if not _DONE[0]:
                    if isinstance(lo, tuple):
                        _DUMPS["loss"] = lo[0].detach().to("cpu", _torch.float32).clone()
                        if len(lo) >= 3 and isinstance(lo[2], dict) and "lm loss" in lo[2]:
                            _DUMPS["lm_loss_reporting"] = lo[2]["lm loss"].detach().to("cpu", _torch.float32).clone()
                    else:
                        _DUMPS["loss"] = lo.detach().to("cpu", _torch.float32).clone()
                    path = f"{_DUMP_DIR}/e2e_dump_{_HW_TAG}.pt"
                    _torch.save(_DUMPS, path)
                    print(f"[DUMP] saved {path} ({len(_DUMPS)} keys)", flush=True)
                    _DONE[0] = True
                return lo
            return output, _wrapped_loss
        return output, loss_function

    _gpt_step_mod._forward_step_common = _hooked_fsc
    _gpt_step_mod.forward_step = _hooked_fs
    forward_step = _hooked_fs  # rebind script-local symbol passed to pretrain()
    print(f"[DUMP] forward_step wrapped, will dump to {_DUMP_DIR}/e2e_dump_{_HW_TAG}.pt", flush=True)

tp_size = int(os.environ["TP_SIZE"])
train_iters = int(os.environ["TRAIN_ITERS"])
world_size_env = int(os.environ.get("WORLD_SIZE", 1))
dp_size = world_size_env // tp_size
# GBS: use env var if set (for fixed-GBS DP-invariance tests), else DP (1 sample/rank)
gbs = int(os.environ.get("GBS", dp_size)) if dp_size > 0 else 1

config = qwen3_600m_pretrain_config(
    mock=True,
    use_null_tokenizer=True,
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=1,
    sequence_parallel=(tp_size > 1),
    global_batch_size=gbs,
    micro_batch_size=1,
    seq_length=2048,
    train_iters=train_iters,
    lr_warmup_iters=0,
    lr_decay_iters=train_iters,
)

config.model.use_cpu_initialization = True
config.model.deterministic_mode = True
config.model.cross_entropy_loss_fusion = False

# Override attention backend for determinism testing
_attn_backend = os.environ.get("ATTN_BACKEND", "").lower()
if _attn_backend:
    config.model.attention_backend = getattr(AttnBackend, _attn_backend)

if USE_BIK:
    # batch_invariant_mode enables num_splits=1 for attention — only works with flash/auto
    # For unfused backend, disable it (BIK aten patches still active from enable_batch_invariant_mode)
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

# Disable gradient clipping for TP-invariant mode (validation)
if os.environ.get("NO_CLIP", "0") == "1":
    config.optimizer.clip_grad = float('inf')
    print("Gradient clipping DISABLED (clip_grad=inf)")

config.logger.tensorboard_dir = "/tmp/tp-inv-600m/tensorboard"
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

import transformer_engine.pytorch.module.linear as _te_linear
src = inspect.getsource(_te_linear._Linear.forward)
patched = "tp_invariant" in src.lower() or "NVTE_TP_INVARIANT_MODE" in src
print(f"TE TP-invariant patch: {'ACTIVE' if patched else 'NOT FOUND'}")

assert tp == tp_size, f"TP mismatch: config={tp}, env={tp_size}"
assert dp == world_size // tp, f"DP mismatch: {dp} != {world_size}//{tp}"
if tp > 1:
    assert config.model.sequence_parallel, "SP must be True when TP > 1"

print(f"""
======== TP-Invariant E2E: Qwen3-0.6B (28 layers) ========
GPUs: {world_size} | TP: {tp} | PP: {pp} | DP: {dp} | GA: {ga}
GBS: {gbs} | MBS: {mbs} | SeqLen: {config.model.seq_length}
Iters: {train_iters} | SP: {config.model.sequence_parallel}
Deterministic: {config.model.deterministic_mode}
NVTE_TP_INVARIANT_MODE: {os.environ.get('NVTE_TP_INVARIANT_MODE', '0')}
BIK: {USE_BIK}
===========================================================
""")

pretrain(config=config, forward_step_func=forward_step)
