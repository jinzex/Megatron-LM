# TP-Invariant Numerics: Bitwise Identical Forward & Backward Across TP Degrees

Achieve **bitwise identical** forward and backward passes for Megatron-Core TransformerBlocks
regardless of Tensor Parallelism (TP) degree — TP=1, 2, 4, 8 all produce the same result.

**Source branches** (view diffs directly):
- [Megatron-LM](https://github.com/jinzex/Megatron-LM/tree/jinzex/tp-invariant-numerics) — MCore patches (clip_grads, batch_invariant_kernels, etc.)
- [TransformerEngine](https://github.com/jinzex/TransformerEngine/tree/jinzex/tp-invariant-numerics) — TE patches (layernorm_linear, linear, etc.)

## Status

| Model | Unit Test | E2E Training |
|-------|-----------|--------------|
| **Dense** | Bitwise identical fwd+bwd, TP=1/2/4/8 | TP=1/2/4 bitwise identical loss, **ALL 100 iters** |
| **MoE** | Bitwise identical fwd+bwd, TP=1/2/4/8 (BIK) | Pending |

Config: `BIK=1 NVTE_TP_INVARIANT_MODE=1` | TE 2.9 | Qwen3-0.6B / 8B

## Baseline vs TP-Invariant (Qwen3-0.6B)

Without TP-invariant mode, loss diverges from iteration 1:

| Iter | Baseline TP=1 | Baseline TP=2 | Baseline TP=4 | TP-Inv (TP=1/2/4) |
|------|--------------|--------------|--------------|-----------------|
| 1 | **1.213320E+01** | 1.213298E+01 | 1.213321E+01 | **1.213320E+01** |
| 2 | **1.187939E+01** | 1.188345E+01 | 1.188395E+01 | **1.187872E+01** |
| 5 | **1.252811E+01** | 1.253201E+01 | 1.253240E+01 | **1.257282E+01** |
| 10 | **1.165137E+01** | 1.164608E+01 | 1.164373E+01 | **1.143649E+01** |
| 100 | **8.277067E+00** | - | - | **8.277067E+00** |

Baseline: `NVTE_TP_INVARIANT_MODE=0` | TP-Inv: `NVTE_TP_INVARIANT_MODE=1 BIK=1`

## Necessary Components

Achieving bitwise identical results requires all of these fixes working together.
Listed in the same order as the [Patch Inventory](#patch-inventory).

| # | Component | Fix | Patch |
|---|-----------|-----|-------|
| 1 | **TP-Invariant GEMM** (fwd+bwd) | All-gather sharded weight, full-K GEMM | `layernorm_linear.py`, `linear.py` |
| 2 | **Gated deinterleave** (bwd) | Reorder via `partition_stride` after all-gather | `layernorm_linear.py` |
| 3 | **Cross-entropy** (fwd) | All-gather exp_logits, local sum | `cross_entropy.py` |
| 4 | **Output projection** (bwd) | All-gather weight+grad, full dgrad GEMM | `layers.py` |
| 5 | **FA3 num_splits** (fwd) | Cherry-pick passthrough for TE 2.9 | `backends.py`, `dot_product_attention.py` |
| 6 | **Gradient clipping** (optimizer) | Float64 norm + pow2 clip_coeff | `clip_grads.py` |
| 7 | **RMSNorm dgamma** (bwd) | All-gather + rank-0-only reduction | `batch_invariant_kernels.py` |
| 8 | **BIK M-invariant GEMM** (fwd+bwd) | Fixed-tile Triton matmul_persistent | `batch_invariant_kernels.py` |
| 9 | **Config: unfused + BIK** | Allow unfused attention with batch_invariant_mode | `transformer_config.py` |
| 10 | **TE 2.9 compat** | num_splits warning for TE < 2.10 | `transformer_engine.py` |

### Gradient Clipping Fix Details

Root cause: `multi_tensor_l2norm` (Apex CUDA) reduces over different-shaped TP shards.
Float32 reduction for 3M vs 1.5M elements gives different results → different clip_factor.

Fix in `clip_grads.py`:
1. `get_grad_norm_fp32()`: compute in float64
2. `clip_grad_by_total_norm_fp32()`: round clip_coeff to nearest power of 2

```
TP-mismatch analysis (1M test points, norm=[0.01, 316]):
  Float32 (no round): 40.3% mismatch
  Pow2 (no mantissa):  0.000% mismatch  <-- chosen
```

## Prerequisites

| Requirement | Version |
|-------------|---------|
| TransformerEngine | **2.9** (`2.9.0+70f53666`) |
| Megatron-Core | From workspace MLM (has BIK + `partition_stride`) |
| PyTorch | 2.9+ |
| GPUs | 8x H100 80GB |

**PYTHONPATH must point to workspace Megatron-LM** — the container's `/opt/megatron-lm/`
lacks BIK and `partition_stride`. Without the correct PYTHONPATH, tests silently fall back
and show ~99% nonzero diffs.

## Quick Start

### 1. Install patches

```bash
PROJ=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/jinzex/pre-training
PATCHES=$PROJ/projects/Numerics/tp-numerics/patches
BRIDGE=$PROJ/third-party/Megatron-Bridge
MCORE=$BRIDGE/3rdparty/Megatron-LM/megatron/core
TE_MOD=/opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/module
TE_DPA=/opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/attention/dot_product_attention

# TE patches (must copy — TE is a site-package, not overridable via PYTHONPATH)
cp $PATCHES/layernorm_linear.py     $TE_MOD/layernorm_linear.py
cp $PATCHES/linear.py               $TE_MOD/linear.py
cp $PATCHES/backends.py             $TE_DPA/backends.py
cp $PATCHES/dot_product_attention.py $TE_DPA/dot_product_attention.py

# MCore patches (copy into MLM loaded via PYTHONPATH)
cp $PATCHES/cross_entropy.py        $MCORE/tensor_parallel/cross_entropy.py
cp $PATCHES/layers.py               $MCORE/tensor_parallel/layers.py

# Verify:
grep -c "NVTE_TP_INVARIANT_MODE" $TE_MOD/layernorm_linear.py  # must be >= 4
```

> MCore patches in `patches/megatron-core/` (clip_grads, batch_invariant_kernels, etc.)
> are already committed to the workspace MLM. Copy them only if using a different MLM checkout.

### 2. Run unit tests

```bash
cd $BRIDGE/3rdparty/Megatron-LM

# Dense (8 GPUs, single fwd+bwd)
PYTHONPATH=. NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_TP_INVARIANT_MODE=1 \
SEQUENCE_PARALLEL=1 TEST_BACKWARD=1 DIAG=1 \
torchrun --nproc_per_node=8 $PROJ/projects/Numerics/tp-numerics/test_tp_numerics.py

# MoE (8 GPUs, single fwd+bwd, requires BIK=1)
PYTHONPATH=. NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_TP_INVARIANT_MODE=1 \
SEQUENCE_PARALLEL=1 TEST_BACKWARD=1 MOE=1 BIK=1 DIAG=1 \
torchrun --nproc_per_node=8 $PROJ/projects/Numerics/tp-numerics/test_tp_numerics.py
```

### 3. Run E2E validation (Dense, 100 iters)

```bash
SCRIPT=$PROJ/projects/Numerics/tp-numerics/validate_e2e_dense_qwen3_0.6b.py

export PYTHONPATH=$BRIDGE/3rdparty/Megatron-LM:$BRIDGE/src
export NVTE_TP_INVARIANT_MODE=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=^NVLS NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BIK=1 ATTN_BACKEND=unfused

for TP in 1 2 4; do
    TP_SIZE=$TP TRAIN_ITERS=100 torchrun --nproc_per_node=$TP $SCRIPT \
      2>&1 | grep "iteration.*lm loss" | tee /tmp/tp${TP}.log
done

# Compare (should be empty = bitwise identical):
diff <(grep -oP "lm loss: \S+" /tmp/tp1.log) <(grep -oP "lm loss: \S+" /tmp/tp2.log)
diff <(grep -oP "lm loss: \S+" /tmp/tp1.log) <(grep -oP "lm loss: \S+" /tmp/tp4.log)
```

## Patches

Patches are split into two directories:

- **`patches/`** — TE and MCore tensor_parallel patches. Must be **copied into the container**
  at runtime because TE is a site-package (not overridable via PYTHONPATH). Requires **TE 2.9**.

- **`patches/megatron-core/`** — MCore optimizer/transformer patches. Already committed to the
  workspace MLM and loaded via PYTHONPATH. Copies kept here for reference and for use with other
  MLM checkouts.

### Patch Inventory

| # | Patch | Target | Purpose |
|---|-------|--------|---------|
| 1 | `patches/layernorm_linear.py` | TE `_LayerNormLinear` | BWD: TP-invariant dgrad GEMM + gated deinterleave |
| 2 | `patches/linear.py` | TE `_Linear` | FWD+BWD: TP-invariant GEMM for row-parallel |
| 3 | `patches/cross_entropy.py` | MCore `cross_entropy` | TP-invariant sum_exp_logits |
| 4 | `patches/layers.py` | MCore `ColumnParallelLinear` | TP-invariant output projection BWD |
| 5 | `patches/backends.py` | TE DPA backends | FA3 num_splits passthrough (TE 2.9) |
| 6 | `patches/dot_product_attention.py` | TE DPA | FA3 num_splits passthrough (TE 2.9) |
| 7 | `patches/megatron-core/clip_grads.py` | MCore optimizer | Float64 grad norm + pow2 clip_coeff |
| 8 | `patches/megatron-core/batch_invariant_kernels.py` | MCore BIK | TP-invariant RMSNorm dgamma + BIK fixes |
| 9 | `patches/megatron-core/transformer_config.py` | MCore config | Allow unfused attention with BIK |
| 10 | `patches/megatron-core/transformer_engine.py` | MCore TE ext | TE 2.9 num_splits warning |

Original (unpatched) files are in `patches/*.orig` for diffing.

## Extended Validation Results

All logs in `results/` (naming: `{model}_{backend}_tp{N}_{iters}.log`).

| Model | Backend | TP Degrees | Iterations | Result |
|-------|---------|------------|------------|--------|
| Qwen3-0.6B | unfused | TP=1/2/4 | 100 | Bitwise identical |
| Qwen3-0.6B | auto (cuDNN) | TP=1/2 | 100 | Bitwise identical |
| Qwen3-0.6B | flash (FA3) | TP=1/2 | 100 | Bitwise identical |
| Qwen3-0.6B | baseline (OFF) | TP=1/2/4 | 10 | Diverges from iter 1 |
| Qwen3-8B | unfused | TP=4/8 | 10 | Bitwise identical |
| Qwen3-8B | auto (cuDNN) | TP=4/8 | 100 | Bitwise identical |
| Qwen3-8B | — | TP=1/2 | — | OOM (all-gather peak memory) |

Recommended backend: **unfused** (no extra installation needed).
auto/flash also work but FA3 requires cherry-picked TE patches for `num_splits` on TE 2.9.

### Dense E2E: Full Fix Stack

TP=1 = TP=2 = TP=4 bitwise identical for ALL 100 iterations.
Config: `BIK=1 ATTN_BACKEND=unfused NVTE_TP_INVARIANT_MODE=1`

| Iter | Loss (all TP) | Grad Norm (all TP) |
|------|--------------|-------------------|
| 1 | 1.213320E+01 | 18.259 |
| 2 | 1.187872E+01 | 57.587 |
| 5 | 1.253663E+01 | 1.300 |
| 10 | 1.159266E+01 | 1.463 |
| 50 | 8.068619E+00 | 0.623 |
| 100 | 8.277067E+00 | 0.542 |

### Unit Test Results

**Dense** — All 0.00% nonzero (bitwise identical), TP=2/4/8 vs TP=1:

| Metric | TP=2 | TP=4 | TP=8 |
|--------|------|------|------|
| FWD | 0.00% | 0.00% | 0.00% |
| GRAD | 0.00% | 0.00% | 0.00% |

**MoE** — All 0.00% nonzero (bitwise identical), TP=2/4/8 vs TP=1, requires `BIK=1`:

| Metric | TP=2 | TP=4 | TP=8 |
|--------|------|------|------|
| FWD | 0.00% | 0.00% | 0.00% |
| GRAD | 0.00% | 0.00% | 0.00% |

## MoE E2E

MoE unit tests achieve **bitwise identical** fwd+bwd across TP=1/2/4/8 (with BIK=1).
MoE E2E training is pending.

**Forward pass**: Bitwise identical at iter 1 (loss 1.232552E+01 at TP=1 = TP=2).

**Backward pass (open)**: Expert wgrad accumulates over different token subsets per TP rank
(partial-K). DP=1: no sync. DP>1: synced but sum(partials) ≠ full due to FP32 accumulation
order. This is the same class of issue as Dense (partial-K GEMM), but harder to fix — expert
weights are replicated (not TP-sharded), so the Dense all-gather-weight approach doesn't apply.

## Known Limitations

**Pipeline Parallelism** changes weight initialization (out of scope). Megatron seeds each
PP stage differently. All E2E validation uses PP=1.

**DP > 1**: Different DP means different gradient accumulation order (GA local sum vs NCCL
all-reduce). DP-invariance is a separate problem not addressed here.

## FAQ

**Q: Why do all three attention backends (unfused/auto/flash) achieve TP-invariance?**

Attention is intra-rank — no cross-rank reduction. The TP-invariant fixes target
cross-rank operations (GEMM, cross-entropy, dgamma, grad norm), which are independent
of the attention kernel.

**Q: Why does Qwen3-8B OOM at TP=1 and TP=2?**

TP-invariant GEMM all-gathers the full weight, temporarily doubling peak memory.
At TP=2 for 8B, each rank needs ~75GB on 80GB H100. Validation-mode tradeoff.

## File Organization

```
tp-numerics/
  README.md
  test_tp_numerics.py                       # Unit test (8-GPU, TP=1/2/4/8, fwd+bwd)
  validate_e2e_dense_qwen3_0.6b.py         # E2E: Qwen3-0.6B Dense
  validate_e2e_dense_qwen3_8b.py           # E2E: Qwen3-8B Dense
  validate_e2e_moe_qwen3_toy.py            # E2E: Toy MoE (4L, 8E)
  submit_e2e_*.sh                           # Slurm sbatch scripts
  patches/                                  # TE patches (copy into container, TE 2.9)
    layernorm_linear.py, linear.py          #   TP-invariant GEMM + deinterleave
    backends.py, dot_product_attention.py   #   FA3 num_splits (TE 2.9)
    cross_entropy.py, layers.py             #   MCore tensor_parallel patches
    megatron-core/                          # MCore patches (in workspace MLM)
      clip_grads.py                         #   Float64 grad norm + pow2 clip
      batch_invariant_kernels.py            #   RMSNorm dgamma + BIK fixes
      transformer_config.py, transformer_engine.py
    *.orig                                  #   Originals for diffing
  results/                                  # Validation log files
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NVTE_TP_INVARIANT_MODE` | `1` | Enable TP-invariant GEMM paths in TE |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | Force deterministic algorithms |
| `BIK` | `1` | Enable Batch Invariant Kernels (required for MoE) |
| `ATTN_BACKEND` | `unfused` | Attention backend (unfused/auto/flash) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Required for deterministic cuBLAS |
| `NCCL_ALGO` | `^NVLS` | Required for deterministic NCCL |
| `FIXED_GRAD_NORM` | `<value>` | Debug: bypass grad norm computation |
