# Gumbel-Softmax + Straight-Through EstimatorによるBundleFlow修正案

## 1. 現在の問題の再確認

### Stage 2の病理

**Forward時（テスト）**:
```python
k* = argmax(u)  # Hard選択
revenue = β[k*] if u[k*] >= 0 else 0
```

**Backward時（トレーニング）**:
```python
Z = softmax(u / λ)  # Soft選択
revenue = Σ Z_k × β_k  # 期待値
```

**問題**: Forward と Backward が異なる関数 → 乖離が発生

---

## 2. Gumbel-Softmax + STE とは

### 2.1 Gumbel-Softmax (Jang et al. 2017)

離散的なカテゴリ分布からのサンプリングを微分可能にする手法。

**標準的なサンプリング（微分不可能）**:
```python
k ~ Categorical(p)  # 離散サンプル → 勾配が流れない
```

**Gumbel-Softmax（微分可能）**:
```python
g_k ~ Gumbel(0, 1)  # ノイズを追加
z_k = softmax((log(p_k) + g_k) / τ)  # 連続緩和
```

τ → 0 のとき、one-hot に近づく。

### 2.2 Straight-Through Estimator (Bengio et al. 2013)

**Forward**: 離散的な操作を実行  
**Backward**: 連続的な代理勾配を使用

```python
# Forward
y = hard_function(x)  # 例: argmax, round

# Backward  
∂L/∂x ≈ ∂L/∂y  # 勾配を直接通す（straight-through）
```

### 2.3 組み合わせ: Gumbel-Softmax + STE

```python
# Forward (テストと同じ)
k* = argmax(u + g)  # Hard選択（Gumbelノイズ付き）
revenue = β[k*] if u[k*] >= 0 else 0

# Backward (微分可能)
z = softmax((u + g) / τ)  # Soft勾配
∂revenue/∂β ≈ ∂(Σ z_k × β_k)/∂β  # Gumbel-Softmaxの勾配
```

**利点**: Forward と Backward のギャップが小さくなる！

---

## 3. BundleFlowへの適用

### 3.1 Stage 2: メニュー選択の修正

#### 現在の実装 (`bf/menu.py`)

```python
def revenue_loss(flow, V, menu, t_grid, lam=0.1):
    U = utilities_matrix(flow, V, menu, t_grid)  # (B, K+1)
    Z = torch.softmax(U / lam, dim=1)  # Softmax選択
    beta = torch.cat([elem.beta for elem in menu] + [torch.zeros(1)])
    revenue = (Z * beta.unsqueeze(0)).sum(dim=1).mean()
    return -revenue
```

#### 提案: Gumbel-Softmax + STE実装

```python
def revenue_loss_gumbel(flow, V, menu, t_grid, tau=0.1, hard=True):
    """
    Gumbel-Softmax + Straight-Through Estimatorによる収益計算
    
    Args:
        tau: Gumbel-Softmax温度（低いほどhard argmaxに近い）
        hard: Trueならforward時にhard argmax、Falseなら soft
    """
    U = utilities_matrix(flow, V, menu, t_grid)  # (B, K+1)
    beta = torch.cat([elem.beta for elem in menu] + [torch.zeros(1)])
    
    # Gumbelノイズを追加
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(U) + 1e-10) + 1e-10)
    logits = (U + gumbel_noise) / tau
    
    if hard:
        # Forward: Hard argmax（テストと同じ）
        k_hard = torch.argmax(logits, dim=1)  # (B,)
        y_hard = torch.nn.functional.one_hot(k_hard, num_classes=U.size(1))  # (B, K+1)
        
        # Backward: Softmaxの勾配を使用（STE）
        y_soft = torch.softmax(logits, dim=1)
        y = y_hard.float() - y_soft.detach() + y_soft  # Straight-through trick
    else:
        # Soft選択（デバッグ用）
        y = torch.softmax(logits, dim=1)
    
    # 収益計算
    revenue = (y * beta.unsqueeze(0)).sum(dim=1).mean()
    return -revenue
```

**重要なトリック**:
```python
y = y_hard - y_soft.detach() + y_soft
#   ^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^
#   Forward     勾配をブロック    Backward
```

- Forward時: `y = y_hard` （hard argmax）
- Backward時: `∂y/∂logits = ∂y_soft/∂logits` （softmaxの勾配）

### 3.2 温度パラメータ τ のアニーリング

```python
# 学習の進行に応じてτを減少させる
tau_start = 1.0  # 初期は soft
tau_end = 0.01   # 最終的にはほぼhard
tau = tau_start * (tau_end / tau_start) ** (t / T)
```

---

## 4. Stage 1 (Rectified Flow) への適用可能性

### 4.1 現在のバンドル離散化

```python
def round_to_bundle(sT: torch.Tensor) -> torch.Tensor:
    """連続値 sT ∈ [0,1]^m を離散バンドル {0,1}^m に変換"""
    return (sT > 0.5).float()  # Hard thresholding
```

**問題**: `sT > 0.5` は微分不可能（正確には勾配=0）

### 4.2 Straight-Through Estimator の適用

```python
def round_to_bundle_ste(sT: torch.Tensor, tau=0.1):
    """
    STE版のrounding: Forward時はhard, Backward時はsigmoidの勾配
    """
    # Forward: Hard threshold
    s_hard = (sT > 0.5).float()
    
    # Backward: Sigmoidの勾配を使用
    s_soft = torch.sigmoid((sT - 0.5) / tau)
    
    # Straight-through
    s = s_hard - s_soft.detach() + s_soft
    return s
```

**利点**: 
- `sT` が0.5付近の値に対して勾配が流れる
- よりスムーズな学習

### 4.3 Gumbel-Softmaxによるカテゴリカルバンドル

もっと洗練された方法として、各アイテムを独立にGumbel-Softmaxでサンプル：

```python
def sample_bundle_gumbel(mu: torch.Tensor, tau=0.1):
    """
    各アイテムをGumbel-Softmaxで{0,1}にサンプル
    
    mu: (m,) 各アイテムの確率（logit）
    """
    # 各アイテムについて [含まない, 含む] の2値選択
    logits = torch.stack([torch.zeros_like(mu), mu], dim=-1)  # (m, 2)
    
    # Gumbel-Softmax
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = torch.softmax((logits + gumbel) / tau, dim=-1)
    
    # Hard selection (forward)
    y_hard = torch.nn.functional.one_hot(
        torch.argmax(y_soft, dim=-1), num_classes=2
    ).float()
    
    # Straight-through
    y = y_hard - y_soft.detach() + y_soft
    
    # "含む"を選択 → bundle
    bundle = y[..., 1]  # (m,)
    return bundle
```

---

## 5. 実装比較

### 5.1 既存のSoftmax実装

| 側面 | 評価 |
|------|------|
| Forward-Backward一致 | ❌ 完全に異なる |
| IR制約の尊重 | ❌ 無視される |
| 学習安定性 | ⚠️ τ次第 |
| 計算コスト | ✅ 低い |

### 5.2 Gumbel-Softmax + STE

| 側面 | 評価 |
|------|------|
| Forward-Backward一致 | ✅ ほぼ一致（τ小で） |
| IR制約の尊重 | ✅ Forward時に厳密に適用 |
| 学習安定性 | ✅ より安定 |
| 計算コスト | ⚠️ やや高い（Gumbel sampling） |

### 5.3 期待される効果

#### `a=2` の場合

**現在**:
```
Training: Revenue = 6.12 (幻想)
Test:     Revenue = 0.00 (現実)
Gap:      6.12 (破綻！)
```

**Gumbel-Softmax + STE導入後**:
```
Training: Revenue ≈ 0.05 (τ=0.01)
Test:     Revenue ≈ 0.00
Gap:      0.05 (許容範囲)
```

価格βも自動的に下方修正される（Forward時の実際の需要を反映）

---

## 6. 実装ステップ

### ステップ1: Gumbel-Softmax utility関数を追加

```python
# bf/utils.py に追加
def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Gumbel-Softmax sampling
    
    Args:
        logits: (*, num_classes) 未正規化ログ確率
        tau: 温度パラメータ
        hard: Trueならhard one-hot、Falseならsoft
        dim: softmax次元
    
    Returns:
        sampled: (*, num_classes) one-hot（hard時）またはソフト確率
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    gumbels = (logits + gumbels) / tau
    y_soft = torch.softmax(gumbels, dim=dim)
    
    if hard:
        # Straight-through estimator
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    
    return ret
```

### ステップ2: `revenue_loss` を修正

```python
# bf/menu.py を修正
from bf.utils import gumbel_softmax

def revenue_loss(flow, V, menu, t_grid, tau=0.1, use_gumbel=True, **kwargs):
    U = utilities_matrix(flow, V, menu, t_grid, **kwargs)  # (B, K+1)
    beta = torch.cat([elem.beta for elem in menu] + [torch.zeros(1, device=U.device)])
    
    if use_gumbel:
        # Gumbel-Softmax + STE
        y = gumbel_softmax(U, tau=tau, hard=True, dim=1)  # (B, K+1)
    else:
        # 従来のSoftmax
        lam = kwargs.get('lam', 0.1)
        y = torch.softmax(U / lam, dim=1)
    
    # IR制約の明示的チェック（hard選択時のみ）
    if use_gumbel:
        # どのメニューが選ばれたか
        selected = torch.argmax(y, dim=1)  # (B,)
        # そのメニューの効用
        selected_utility = torch.gather(U, 1, selected.unsqueeze(1)).squeeze(1)  # (B,)
        # IR制約: u < 0 なら収益ゼロ
        ir_mask = (selected_utility >= 0).float()  # (B,)
        revenue = (y * beta.unsqueeze(0)).sum(dim=1) * ir_mask
    else:
        revenue = (y * beta.unsqueeze(0)).sum(dim=1)
    
    return -revenue.mean()
```

### ステップ3: トレーニングループで τ をアニーリング

```python
# src/train_stage2.py を修正
for it in range(1, args.iters + 1):
    # τアニーリング
    tau = args.tau_start * (args.tau_end / args.tau_start) ** ((it - 1) / args.iters)
    
    # 損失計算
    loss = revenue_loss(
        flow, batch, menu, t_grid,
        tau=tau, use_gumbel=True
    )
```

### ステップ4: コマンドライン引数を追加

```python
# src/train_stage2.py
ap.add_argument("--use_gumbel", action="store_true", help="Use Gumbel-Softmax + STE")
ap.add_argument("--tau_start", type=float, default=1.0, help="Initial temperature")
ap.add_argument("--tau_end", type=float, default=0.01, help="Final temperature")
```

---

## 7. 期待される効果と理論的根拠

### 7.1 なぜGumbel-Softmaxが機能するか

1. **Forwardの正確性**: Hard argmaxを使用 → テスト時と同じ挙動
2. **Backwardの情報**: Softmaxの勾配 → 滑らかな最適化
3. **Gumbelノイズ**: 探索を促進 → 局所最適回避

### 7.2 Rectified Flowとの相性

Rectified Flowは連続→離散の変換を含む：

```
μ (連続) → sT (連続) → s (離散)
```

STEを使えば、この離散化の勾配を改善できる：

```
∂L/∂sT ← STE ← ∂L/∂s
```

現在は `round` 関数で勾配が消失している可能性。

---

## 8. 実験計画

### Phase 1: Stage 2のみでテスト

```bash
# 既存のflow checkpointを使用
python -m src.train_stage2 \
  --flow_ckpt checkpoints/flow_stage1_final.pt \
  --m 2 --K 4 --a 2 --iters 1000 \
  --use_gumbel --tau_start 1.0 --tau_end 0.01 \
  --out_dir checkpoints_gumbel
```

**期待される結果**:
- Training revenue ≈ Test revenue
- β が評価値分布に適応（上限に張り付かない）

### Phase 2: Stage 1でもSTEを適用

```bash
# round_to_bundle_ste を実装後
python -m src.train_stage1 \
  --m 2 --iters 10000 \
  --use_ste \
  --out_dir checkpoints_ste
```

**期待される効果**:
- より正確なバンドル生成
- 0.5付近の確率値に対する学習改善

---

## 9. 参考文献

1. **Jang, E., Gu, S., & Poole, B. (2017).** Categorical Reparameterization with Gumbel-Softmax. *ICLR 2017*.
   - Gumbel-Softmaxの原論文
   - カテゴリカル分布の再パラメータ化

2. **Bengio, Y., Léonard, N., & Courville, A. (2013).** Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. *arXiv:1308.3432*.
   - Straight-Through Estimatorの提案
   - 離散ニューロンの勾配伝播

3. **Maddison, C. J., Mnih, A., & Teh, Y. W. (2017).** The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. *ICLR 2017*.
   - Gumbel-Softmaxの独立発見（Concrete Distribution）
   - 理論的な背景

4. **Huijben, I. A., et al. (2022).** A Review of the Gumbel-max Trick and its Extensions for Discrete Stochasticity in Machine Learning. *IEEE TPAMI*.
   - Gumbel-max trickの包括的レビュー
   - 様々な拡張と応用

---

## 10. まとめ

### 現状の問題
- Softmax relaxation が Forward（test）と Backward（train）で異なる関数を使用
- IR制約が実質的に無視される → 価格暴走

### Gumbel-Softmax + STEの解決策
- ✅ Forward時にhard argmaxを使用（テストと一致）
- ✅ Backward時にsoftmaxの勾配を使用（微分可能）
- ✅ IR制約を明示的に適用可能
- ✅ Training-Test gapを大幅に削減

### 次のステップ
1. `bf/utils.py` に `gumbel_softmax` 関数を実装
2. `bf/menu.py` の `revenue_loss` を修正
3. `src/train_stage2.py` でτアニーリングを追加
4. `m=2, a=2` で実験して効果を検証

**この手法は理論的にも実装的にも非常に promising です！**

