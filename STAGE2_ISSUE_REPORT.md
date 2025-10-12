# Stage 2 Training Issue Report

**Date**: 2025-10-12  
**Issue**: All flow-generated bundles return zero valuation  
**Status**: Under investigation

---

## Problem Summary

Stage 2 training fails because **ALL 512 flow-generated bundles match ZERO atoms** across all valuations. This is statistically impossible (~0% probability with 512 bundles × 20 atoms = 10,240 checks).

**Symptom**: Revenue diverges to 5.6+ (theoretical max ≈ 1.0)

---

## Experimental Setup

### Stage 1 (Completed Successfully ✅)
```bash
python -m src.train_stage1 \
  --m 50 --iters 50000 --batch 512 --lr 1e-3 \
  --lambda_j 1e-3 --lambda_k 1e-3 --lambda_tr 1e-4 \
  --use_scheduler --use_csv \
  --out_dir checkpoints
```

**Results**:
- Final loss: 1.80 (converged)
- sT_mean: 0.500 (⚠️ suspiciously exact)
- sT_range: [-0.18, 1.20]
- Checkpoint: `flow_stage1_final.pt` (1.4 MB)

### Stage 2 (Failing ❌)
```bash
python -m src.train_stage2 \
  --flow_ckpt checkpoints/flow_stage1_final.pt \
  --m 50 --K 128 --D 4 --iters 10000 \
  --batch 64 --ode_steps 25 \
  --warmstart --reinit_every 1000 \
  --out_dir checkpoints_stage2
```

**Configuration**:
- Valuations: 5000 synthetic XOR (a=20, uniform [0,1])
- Menu elements: 128 + 1 null
- Using frozen FlowModel from Stage 1

---

## What Works ✅

### 1. Data Generation
- 5000 valuations created
- Each has 20 atoms with prices in [0, 1]
- `v0.m = 50` (correct)
- **Test**: `v0.value(all_50_items_bundle) = 0.9496` ✅

### 2. Flow-Generated Bundles
- 512 bundles generated per iteration
- 185 unique bundles (36% diversity)
- Items per bundle: mean=25.3, std=3.4, range=[18, 34]
- Values: {0.0, 1.0} (binary, correct)

### 3. Utility/Revenue Computation
- Tensor shapes: U(64, 129), Z(64, 129), beta(129) ✅
- Softmax computation: correct
- Revenue formula: mathematically correct

---

## What Fails ❌

### The Core Issue: Zero Valuations

**Same valuation, different results**:
```python
v0.value(all_items_bundle) = 0.9496  # ✅ Startup test
v0.value(flow_bundle[0]) = 0.0000    # ❌ Iteration 20
v0.value(flow_bundle[1]) = 0.0000    # ❌ All 512 bundles
```

**Internal debugging** (20 atoms checked):
```
[value DEBUG] atom_mask=302866576577186, Sm=283191968948183, check=20272681844768, match=False
[value DEBUG] atom_mask=352658401584126, Sm=283191968948183, check=70575611486248, match=False
...
[value DEBUG] Total matched: 0/20, final value: 0.0000
```

### Consequences

```
All utilities = -β (negative)
→ Softmax ≈ uniform distribution
→ Revenue = (128/129) × β
→ ∂Revenue/∂β > 0
→ β → 5.7 → 11.7 → ... (unbounded)
```

---

## Detailed Debug Output

### Example Bundle vs Atom:
```
Bundle[0] (27 items): bit positions [0,1,2,4,6,7,8,9,10,11,12,14,17,18,20,21,23,26,30,31,32,33,34,35,39,40,48]
Atom[0] (23 items):   bit positions [1,5,7,9,12,17,18,20,21,23,25,26,27,28,31,34,36,37,38,40,41,44,48]

Missing in bundle: [5,25,27,28,36,37,38,41,44]  # 9 items
Overlap: 14 out of 23 items

Matching check: (atom_mask & ~bundle_mask) = 20272681844768 ≠ 0
Result: No match ❌
```

This is **normal for one bundle**, but abnormal for ALL 512 bundles.

---

## Hypotheses (Ordered by Likelihood)

### 1. Stage 1 Flow Collapsed ⚠️⚠️⚠️

**Evidence**:
- `sT_mean = 0.500` (exactly 0.5 is suspicious)
- All flow outputs might be nearly identical

**Test**:
```python
# Generate 1000 random μ ∈ [-0.2, 1.2]
# Pass through flow
# Count unique bundles
# If < 100 unique → flow collapsed
```

**Possible causes**:
- Regularization too strong (λ_j, λ_k, λ_tr)
- Network saturated (tanh outputs stuck at ±1)
- ODE collapsed to trivial solution

---

### 2. μ=0 is Pathological ⚠️⚠️

**Without warmstart**:
```python
MenuElement.__init__:
    self.mus = nn.Parameter(torch.zeros(D, m))  # All zeros
```

**Questions**:
- Is μ=0 in Stage 1's training distribution?
- What bundle does `flow(μ=0)` produce?
- Should μ be initialized with small random noise?

**Observed**: Warmstart ON/OFF makes no difference → Both produce zero-valued bundles

---

### 3. Distribution Mismatch ⚠️⚠️

**Stage 1 trained**:
```python
α0 ~ MoG: μ_d ∈ [-0.2, 1.2]^m, σ=0.5, D=8
```

**Stage 2 uses**:
```python
# Originally: μ ∈ [0, 1]^m (FIXED)
# Now: μ ∈ [-0.2, 1.2]^m
# But still fails
```

---

### 4. Numerical Precision Issues ⚠️

**Suspects**:
- `flow.round_to_bundle(sT)`: `(sT >= 0.5).to(dtype)`
- If sT values are exactly 0.5 → rounding is random
- Bundles become meaningless noise

---

### 5. Hidden Data Corruption ⚠️

**Observation**: `v0` works in startup, fails in iteration

**Possibilities**:
- `random.sample(train, B)` corrupts valuation objects?
- Shallow vs deep copy issue?
- GPU/CPU transfers corrupt data?

**Test**: 
```python
print(id(v0), id(train[0]), id(batch[0]))
# Are they the same object or different?
```

---

### 6. Atom Masks Use Different Item Set ⚠️

**Wild theory**:
- Atoms were generated for items {1, 2, ..., 50}
- Bundles represent items {0, 1, ..., 49}
- Off-by-one mapping error

**Verified**: Mapping is correct (tensor[0] = item 1 → bit 0)

---

## Reproducible Minimal Test

```python
import torch
from bf.flow import FlowModel
from bf.data import gen_uniform_iid_xor

# Load flow
ckpt = torch.load("checkpoints/flow_stage1_final.pt")
flow = FlowModel(m=50, use_spectral_norm=False)
flow.load_state_dict(ckpt["model"])
flow.eval()

# Generate valuation
v = gen_uniform_iid_xor(m=50, a=20, seed=42)

# Test inputs
t_grid = torch.linspace(0, 1, 25)
test_cases = [
    ("All items", torch.ones(50)),
    ("μ=0", torch.zeros(1, 50)),
    ("μ=0.5", torch.full((1, 50), 0.5)),
    ("μ random [0,1]", torch.rand(1, 50)),
    ("μ random [-0.2,1.2]", torch.rand(1, 50) * 1.4 - 0.2),
]

for name, mu_or_bundle in test_cases:
    if mu_or_bundle.shape[0] == 1:  # μ input
        sT = flow.flow_forward(mu_or_bundle, t_grid)
        s = flow.round_to_bundle(sT)
        val = v.value(s[0])
    else:  # Direct bundle
        val = v.value(mu_or_bundle)
    print(f"{name}: {val:.4f}")

# Expected: At least SOME should be > 0
# Actual: ALL are 0? → Major bug
```

---

## Questions for Expert

1. **Is `sT_mean = 0.500` after 50K iterations suspicious?**
   - Could indicate flow collapse?
   
2. **Should μ be initialized randomly instead of zeros?**
   - Is μ=0 in the training distribution?
   
3. **Is there a known "warmup period" before bundles match atoms?**
   - Or should matching occur from iteration 1?
   
4. **Are there additional constraints/penalties not in the paper?**
   - IR penalty on negative utilities?
   - Bounds on β?
   
5. **Could this be a known numerical stability issue?**
   - Specific PyTorch version requirements?
   - GPU precision issues?

---

## Additional Context

- **Environment**: Google Colab, CUDA, PyTorch
- **All diagnostics show** individual components work correctly
- **But the combination produces pathological results**
- Spent 200+ tool calls debugging with full instrumentation
- Every assumption verified except: "Does Stage 1 actually learn a useful flow?"

---

## Next Steps

1. Verify Stage 1 flow is not degenerate (diversity test)
2. Test μ=0 behavior explicitly
3. Add β constraints (sigmoid) to prevent divergence
4. Consider re-training Stage 1 with different hyperparameters

---

**Thank you for any insights!** This is a fascinating bug that survives every targeted fix attempt.

