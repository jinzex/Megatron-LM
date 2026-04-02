"""
TP vs DP numerical equivalence test for Qwen3-8B (dense) and MoE architectures.

Compares TransformerBlock forward (and optionally backward) outputs across
TP=1/2/4/8 using random init (use_cpu_initialization=True).

Usage (8 GPUs, inside container):
    cd <MEGATRON_LM_PATH>
    PYTHONPATH=. CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    [NVTE_TP_INVARIANT_MODE=1] [NVTE_FP32_TP_REDUCE=1] \
    [SEQUENCE_PARALLEL=1] [MOE=1] [TEST_BACKWARD=1] \
    torchrun --nproc_per_node=8 /path/to/test_tp_numerics.py

Environment variables:
    QWEN_NUM_LAYERS       - number of layers (default: 1)
    QWEN_SEQ_LENGTH       - sequence length (default: 4096)
    QWEN_MICRO_BATCH_SIZE - micro batch size (default: 2)
    NVTE_FP32_TP_REDUCE   - fp32 GEMM output for TP reduction
    NVTE_TP_INVARIANT_MODE - full GEMM in row-parallel fwd / column-parallel bwd
    SEQUENCE_PARALLEL     - enable SP (default: 0)
    MOE                   - enable MoE architecture (default: 0)
    TEST_BACKWARD         - also compare input gradients (default: 0)
"""

import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

import megatron.core.parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
try:
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
    )
except ModuleNotFoundError:
    enable_batch_invariant_mode = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

NUM_LAYERS = int(os.environ.get("QWEN_NUM_LAYERS", "1"))
SEQ_LENGTH = int(os.environ.get("QWEN_SEQ_LENGTH", "4096"))
MICRO_BATCH_SIZE = int(os.environ.get("QWEN_MICRO_BATCH_SIZE", "2"))
SEQUENCE_PARALLEL = os.environ.get("SEQUENCE_PARALLEL", "0") == "1"
USE_MOE = os.environ.get("MOE", "0") == "1"
TEST_BACKWARD = os.environ.get("TEST_BACKWARD", "0") == "1"
USE_BIK = os.environ.get("BIK", "0") == "1"
INIT_SEED = 12345

HIDDEN_SIZE = 4096
NUM_ATTENTION_HEADS = 32
NUM_QUERY_GROUPS = 8
FFN_HIDDEN_SIZE = 12288
KV_CHANNELS = 128

MOE_NUM_EXPERTS = 8
MOE_TOP_K = 2
MOE_FFN_HIDDEN_SIZE = 1536


def _build_config(tp_size: int) -> TransformerConfig:
    sp = SEQUENCE_PARALLEL and tp_size > 1
    kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_query_groups=NUM_QUERY_GROUPS,
        ffn_hidden_size=MOE_FFN_HIDDEN_SIZE if USE_MOE else FFN_HIDDEN_SIZE,
        kv_channels=KV_CHANNELS,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        use_cpu_initialization=True,
        perform_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        qk_layernorm=True,
        no_rope_freq=[1] * NUM_LAYERS,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        sequence_parallel=sp,
        attention_backend=AttnBackend.unfused if USE_BIK else None,
    )
    if USE_MOE:
        kwargs.update(
            num_moe_experts=MOE_NUM_EXPERTS,
            moe_router_topk=MOE_TOP_K,
            moe_token_dispatcher_type="alltoall",
        )
    return TransformerConfig(**kwargs)


DIAG = os.environ.get("DIAG", "0") == "1"


def _gather_sp(tensor, tp_size, tp_group):
    """All-gather SP-sharded tensor back to full sequence."""
    if tp_size <= 1 or not SEQUENCE_PARALLEL:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered, tensor.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=0)


def _install_moe_hooks(block, tp_size, captures):
    """Install hooks on MoE submodules to capture intermediates."""
    tp_group = ps.get_tensor_model_parallel_group() if tp_size > 1 else None

    for layer in block.layers:
        moe = getattr(layer, "mlp", None)
        if moe is None:
            continue

        def moe_pre_hook(module, args, *, _tp=tp_size, _g=tp_group):
            hs = args[0].detach().clone()
            captures["moe_input"] = _gather_sp(hs, _tp, _g)

        moe.register_forward_pre_hook(moe_pre_hook)

        router = getattr(moe, "router", None)
        if router is not None:
            orig_gating = router.gating

            def patched_gating(inp, *, _orig=orig_gating, _tp=tp_size, _g=tp_group):
                logits = _orig(inp)
                captures["router_logits"] = _gather_sp(
                    logits.detach().clone(), _tp, _g
                )
                return logits

            router.gating = patched_gating

        def moe_post_hook(module, args, output, *, _tp=tp_size, _g=tp_group):
            out = output[0] if isinstance(output, tuple) else output
            captures["moe_output"] = _gather_sp(out.detach().clone(), _tp, _g)

        moe.register_forward_hook(moe_post_hook)


def _gather_inner(tensor, tp_size, tp_group):
    """All-gather tensor along last dim (for column-parallel sharded activations)."""
    if tp_size <= 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered, tensor.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=-1)


def _install_mlp_interior_hooks(block, tp_size, captures):
    """Hook on intermediate MLP tensors to isolate backward diff source."""
    tp_group = ps.get_tensor_model_parallel_group() if tp_size > 1 else None

    for layer in block.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        linear_fc2 = getattr(mlp, "linear_fc2", None)
        if linear_fc2 is not None:
            orig_fc2_forward = linear_fc2.forward

            def patched_fc2_forward(hidden_states, *args, _orig=orig_fc2_forward, _tp=tp_size, _g=tp_group, **kwargs):
                if hidden_states.requires_grad:
                    def grab_grad(grad, *, _c=captures, _tp2=_tp, _g2=_g):
                        # Linear2's input is [seq, batch, ffn/TP] — full sequence (NOT SP-sharded)
                        # Only need _gather_inner to reconstruct ffn dimension, NOT _gather_sp
                        _c["bwd_fc2_input_grad"] = _gather_inner(grad.detach().clone(), _tp2, _g2)
                    hidden_states.register_hook(grab_grad)
                return _orig(hidden_states, *args, **kwargs)

            linear_fc2.forward = patched_fc2_forward

        linear_fc1 = getattr(mlp, "linear_fc1", None)
        if linear_fc1 is not None:
            orig_fc1_forward = linear_fc1.forward

            def patched_fc1_forward(hidden_states, *args, _orig=orig_fc1_forward, _tp=tp_size, _g=tp_group, **kwargs):
                if hidden_states.requires_grad:
                    def grab_grad(grad, *, _c=captures, _tp2=_tp, _g2=_g):
                        _c["bwd_fc1_input_grad"] = _gather_sp(grad.detach().clone(), _tp2, _g2)
                    hidden_states.register_hook(grab_grad)
                result = _orig(hidden_states, *args, **kwargs)
                out_tensor = result[0] if isinstance(result, tuple) else result
                if out_tensor.requires_grad:
                    def grab_out_grad(grad, *, _c=captures, _tp2=_tp, _g2=_g):
                        # Gradient of Linear1's output = SwiGLU backward output
                        _c["bwd_fc1_output_grad"] = _gather_inner(grad.detach().clone(), _tp2, _g2)
                    out_tensor.register_hook(grab_out_grad)
                return result

            linear_fc1.forward = patched_fc1_forward


def _install_backward_hooks(block, tp_size, captures):
    """Install backward hooks to capture gradient flow at key module boundaries."""
    tp_group = ps.get_tensor_model_parallel_group() if tp_size > 1 else None

    for layer in block.layers:
        attn = getattr(layer, "self_attention", None)
        mlp = getattr(layer, "mlp", None)
        pre_norm = getattr(layer, "input_layernorm", None)
        post_norm = getattr(layer, "pre_mlp_layernorm", None)

        if attn is not None:
            def attn_bwd_hook(module, grad_input, grad_output, *, _tp=tp_size, _g=tp_group):
                for i, gi in enumerate(grad_input):
                    if gi is not None and gi.shape[-1] == HIDDEN_SIZE:
                        captures["bwd_attn_grad_input"] = _gather_sp(
                            gi.detach().clone(), _tp, _g
                        )
                        break
                if grad_output and grad_output[0] is not None:
                    go = grad_output[0]
                    if go.shape[-1] == HIDDEN_SIZE:
                        captures["bwd_attn_grad_output"] = _gather_sp(
                            go.detach().clone(), _tp, _g
                        )
            attn.register_full_backward_hook(attn_bwd_hook)

        if mlp is not None:
            def mlp_bwd_hook(module, grad_input, grad_output, *, _tp=tp_size, _g=tp_group):
                for i, gi in enumerate(grad_input):
                    if gi is not None and gi.shape[-1] == HIDDEN_SIZE:
                        captures["bwd_mlp_grad_input"] = _gather_sp(
                            gi.detach().clone(), _tp, _g
                        )
                        break
                if grad_output and grad_output[0] is not None:
                    go = grad_output[0]
                    if go.shape[-1] == HIDDEN_SIZE:
                        captures["bwd_mlp_grad_output"] = _gather_sp(
                            go.detach().clone(), _tp, _g
                        )
            mlp.register_full_backward_hook(mlp_bwd_hook)

        if pre_norm is not None:
            def pre_norm_bwd_hook(module, grad_input, grad_output, *, _tp=tp_size, _g=tp_group):
                for i, gi in enumerate(grad_input):
                    if gi is not None and gi.shape[-1] == HIDDEN_SIZE:
                        captures["bwd_pre_norm_grad_input"] = _gather_sp(
                            gi.detach().clone(), _tp, _g
                        )
                        break
            pre_norm.register_full_backward_hook(pre_norm_bwd_hook)

        if post_norm is not None:
            def post_norm_bwd_hook(module, grad_input, grad_output, *, _tp=tp_size, _g=tp_group):
                for i, gi in enumerate(grad_input):
                    if gi is not None and gi.shape[-1] == HIDDEN_SIZE:
                        captures["bwd_post_norm_grad_input"] = _gather_sp(
                            gi.detach().clone(), _tp, _g
                        )
                        break
            post_norm.register_full_backward_hook(post_norm_bwd_hook)


def _run_scenario(
    tp_size: int,
    dp_size: int,
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rank = torch.distributed.get_rank()

    ps.destroy_model_parallel()
    if USE_MOE:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
    else:
        ps.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    model_parallel_cuda_manual_seed(INIT_SEED, force_reset_rng=True)

    torch.manual_seed(INIT_SEED)
    torch.cuda.manual_seed(INIT_SEED)

    config = _build_config(tp_size)
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        qk_layernorm=True,
        num_experts=MOE_NUM_EXPERTS if USE_MOE else None,
        moe_grouped_gemm=False,
    )
    block = TransformerBlock(config, layer_spec).cuda().bfloat16()

    captures = {}
    if DIAG and USE_MOE:
        _install_moe_hooks(block, tp_size, captures)
    if DIAG and TEST_BACKWARD:
        _install_backward_hooks(block, tp_size, captures)
        _install_mlp_interior_hooks(block, tp_size, captures)

    logger.info(
        f"[Rank {rank}] TP={tp_size}/DP={dp_size}: "
        f"params={sum(p.numel() for p in block.parameters()):,}"
    )

    sp_enabled = config.sequence_parallel and tp_size > 1
    input_hs = hidden_states.clone().detach()
    if sp_enabled:
        seq_len = input_hs.shape[0]
        chunk_size = seq_len // tp_size
        tp_rank = ps.get_tensor_model_parallel_rank()
        input_hs = input_hs[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]

    if TEST_BACKWARD:
        input_hs = input_hs.requires_grad_(True)

    if TEST_BACKWARD:
        output = block(hidden_states=input_hs, attention_mask=None)
        loss = output.float().sum()
        loss.backward()
        grad = input_hs.grad.clone().detach()
    else:
        with torch.no_grad():
            output = block(hidden_states=input_hs, attention_mask=None)
        grad = torch.zeros(1, device="cuda")

    output = output.detach()
    if sp_enabled:
        gathered = [torch.empty_like(output) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered, output, group=ps.get_tensor_model_parallel_group())
        output = torch.cat(gathered, dim=0)
        if TEST_BACKWARD:
            grad_gathered = [torch.empty_like(grad) for _ in range(tp_size)]
            torch.distributed.all_gather(grad_gathered, grad, group=ps.get_tensor_model_parallel_group())
            grad = torch.cat(grad_gathered, dim=0)

    logger.info(
        f"[Rank {rank}] TP={tp_size} fwd mean={output.float().mean():.6f}"
        + (f" grad mean={grad.float().mean():.6f}" if TEST_BACKWARD else "")
    )

    del block
    torch.cuda.empty_cache()
    return output, grad, captures


def _log_comparison(baseline, candidate, label) -> dict:
    diff = (baseline.float() - candidate.float())
    abs_diff = diff.abs()
    total = abs_diff.numel()
    stats = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "pct_nonzero": 100.0 * (abs_diff > 0).sum().item() / total,
    }
    logger.info(f"\n{'='*70}")
    logger.info(f"{label}")
    logger.info(f"  Max: {stats['max_abs_diff']:.8f}  Mean: {stats['mean_abs_diff']:.8f}  "
                f"Nonzero: {stats['pct_nonzero']:.2f}%")
    logger.info(f"{'='*70}")
    return stats


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size == 8, f"Test requires 8 GPUs, got {world_size}"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if USE_BIK:
        enable_batch_invariant_mode()
        logger.info(f"[Rank {rank}] batch_invariant_mode ENABLED")
        import transformer_engine.pytorch.module.linear as _te_lin
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            _te_general_gemm_patched,
        )
        _bik_ok = getattr(_te_lin, "general_gemm", None) is _te_general_gemm_patched
        logger.info(f"[Rank {rank}] BIK intercepts TE linear.general_gemm: {_bik_ok}")

        _bik_call_count = [0]
        _bik_gemm_log = []  # [(call_idx, A_hash, B_hash, result_hash, A_shape, B_shape)]
        _real_patched = _te_lin.general_gemm
        def _counting_gemm(*a, **kw):
            _bik_call_count[0] += 1
            result = _real_patched(*a, **kw)
            if DIAG and TEST_BACKWARD:
                A_t, B_t = a[0], a[1]
                res_t = result[0] if isinstance(result, (list, tuple)) else result
                import hashlib
                def _thash(t):
                    raw = t.detach().cpu().contiguous().view(torch.uint8)
                    return hashlib.md5(raw.numpy().tobytes()).hexdigest()[:12]
                _bik_gemm_log.append((
                    _bik_call_count[0],
                    _thash(A_t), _thash(B_t), _thash(res_t),
                    tuple(A_t.shape), tuple(B_t.shape), tuple(res_t.shape),
                ))
            return result
        _te_lin.general_gemm = _counting_gemm

    tp_scenarios = [(1, 8), (2, 4), (4, 2), (8, 1)]

    logger.info(f"\n[Rank {rank}] layers={NUM_LAYERS} seq={SEQ_LENGTH} mbs={MICRO_BATCH_SIZE}")
    logger.info(f"[Rank {rank}] SP={SEQUENCE_PARALLEL} MOE={USE_MOE} BACKWARD={TEST_BACKWARD}")
    logger.info(f"[Rank {rank}] NVTE_FP32_TP_REDUCE={os.environ.get('NVTE_FP32_TP_REDUCE','0')}")
    logger.info(f"[Rank {rank}] NVTE_TP_INVARIANT_MODE={os.environ.get('NVTE_TP_INVARIANT_MODE','0')}")

    torch.manual_seed(42)
    hidden_states = torch.randn(
        (SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.distributed.broadcast(hidden_states, src=0)

    results: Dict[int, tuple] = {}
    for tp_size, dp_size in tp_scenarios:
        logger.info(f"\n[Rank {rank}] ===== TP={tp_size} / DP={dp_size} =====")
        if USE_BIK:
            _bik_call_count[0] = 0
        results[tp_size] = _run_scenario(tp_size, dp_size, hidden_states)
        if USE_BIK:
            logger.info(f"[Rank {rank}] BIK general_gemm calls for TP={tp_size}: {_bik_call_count[0]}")
            if rank == 0 and DIAG and TEST_BACKWARD:
                for entry in _bik_gemm_log:
                    idx, ah, bh, rh, ash, bsh, rsh = entry
                    logger.info(f"  GEMM#{idx}: A={ah} {ash}  B={bh} {bsh}  R={rh} {rsh}")
                _bik_gemm_log.clear()
        torch.distributed.barrier()

    if rank == 0:
        fwd_baseline, grad_baseline, caps_baseline = results[1]
        all_fwd_match = True
        all_grad_match = True
        for tp_size, _ in tp_scenarios[1:]:
            fwd_out, grad_out, caps = results[tp_size]
            s = _log_comparison(fwd_baseline, fwd_out, f"FWD: TP=1 vs TP={tp_size}")
            if s["max_abs_diff"] > 0:
                all_fwd_match = False
            if TEST_BACKWARD:
                s = _log_comparison(grad_baseline, grad_out, f"GRAD: TP=1 vs TP={tp_size}")
                if s["max_abs_diff"] > 0:
                    all_grad_match = False
            if DIAG and USE_MOE:
                for key in ["moe_input", "router_logits", "moe_output"]:
                    if key in caps_baseline and key in caps:
                        _log_comparison(caps_baseline[key], caps[key],
                                        f"DIAG {key}: TP=1 vs TP={tp_size}")
            if DIAG and TEST_BACKWARD:
                bwd_keys = [
                    "bwd_mlp_grad_output",
                    "bwd_fc2_input_grad",
                    "bwd_fc1_output_grad",
                    "bwd_fc1_input_grad",
                    "bwd_mlp_grad_input",
                    "bwd_post_norm_grad_input",
                    "bwd_attn_grad_output", "bwd_attn_grad_input",
                    "bwd_pre_norm_grad_input",
                ]
                for key in bwd_keys:
                    if key in caps_baseline and key in caps:
                        _log_comparison(caps_baseline[key], caps[key],
                                        f"DIAG {key}: TP=1 vs TP={tp_size}")
                    elif key in caps_baseline or key in caps:
                        logger.info(f"DIAG {key}: MISSING in {'baseline' if key not in caps_baseline else 'candidate'}")

        if all_fwd_match:
            logger.info(f"RESULT-FWD: All TP configs bitwise identical to TP=1")
        else:
            logger.info(f"RESULT-FWD: Numerical differences detected")
        if TEST_BACKWARD:
            if all_grad_match:
                logger.info(f"RESULT-GRAD: All TP configs bitwise identical to TP=1")
            else:
                logger.info(f"RESULT-GRAD: Numerical differences detected")

    ps.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
