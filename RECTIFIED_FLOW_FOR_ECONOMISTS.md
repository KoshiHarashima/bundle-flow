# Rectified Flow入門：経済学者のための技術解説

**対象読者**: 統計学には精通しているが、機械学習には不慣れな経済学研究者

---

## Executive Summary

Rectified Flowは、**連続分布から離散的な組み合わせ構造へのサンプリング**を可能にする深層学習手法である。BundleFlowでは、この技術を用いて「無限に近い数のバンドル候補」から「買い手にとって魅力的なバンドル」を効率的に生成する。本稿では、Rectified Flowの数学的基礎を、経済学者に馴染みのある確率論・最適化理論の観点から解説する。

---

## 1. 問題設定：組み合わせ爆発との戦い

### 1.1 バンドルオークションの課題

`m` 個の商品があるとき、可能なバンドルの数は：

$$
|\mathcal{S}| = 2^m
$$

**具体例**:
- `m = 10`: 1,024通り
- `m = 50`: 1,125,899,906,842,624通り（1000兆超）
- `m = 100`: 10³⁰ 通り（宇宙の原子の数より多い）

**問題**: セラーはどのバンドルを提供すべきか？

### 1.2 従来のアプローチ

#### (A) 全探索

すべてのバンドルについて収益を計算：

$$
\max_{S_1, \ldots, S_K \in \{0,1\}^m} \mathbb{E}_{v \sim \mathcal{V}}[\text{Revenue}(S_1, \ldots, S_K; v)]
$$

**問題**: `m` が大きいと計算不可能（NP困難）

#### (B) 貪欲法・ヒューリスティック

- VCG メカニズムの近似
- 線形緩和 + 整数計画法
- シミュレーテッドアニーリング

**問題**: 
- 最適性の保証がない
- バイヤーの評価値分布に依存

### 1.3 BundleFlowの提案

**アイデア**: 「良いバンドル」を生成する確率分布を**学習**する

$$
\mu \xrightarrow{\text{学習済み変換}} S \in \{0,1\}^m
$$

where `μ` は連続分布（扱いやすい）、`S` は離散バンドル（実際に必要）

**利点**:
- サンプリングで多様なバンドルを生成
- 評価値分布に適応
- 計算可能

---

## 2. Rectified Flow: 連続→離散の橋渡し

### 2.1 基本概念

**目標**: 単純な連続分布 `μ` を、離散バンドル分布に変換する写像を学習

#### ステップ1: 初期分布 `μ₀`

`D` 個の支持点を持つGaussian混合分布：

$$
\mu_0 = \sum_{d=1}^D w_d \cdot \mathcal{N}(\mu_d, \sigma_d^2 I)
$$

where:
- `w_d`: 混合重み（`Σ w_d = 1`）
- `μ_d ∈ ℝ^m`: 各成分の中心
- `σ_d > 0`: 分散

**直感**: `μ₀` は `m` 次元空間内の「確率的な点」を表す

#### ステップ2: 終点分布 `s`

離散的な0/1バンドル：

$$
s \in \{0,1\}^m
$$

#### ステップ3: Rectified Flow

時間 `t ∈ [0,1]` で連続的に変換：

$$
\frac{ds_t}{dt} = \phi(t, s_t, s_0)
$$

初期条件: `s_0 ~ μ₀`, 終点: `s_1 ≈ s` （離散バンドル）

### 2.2 経済学的解釈

#### 類似概念: 均衡への調整過程

経済学における**tatonnement process**（模索過程）と類似：

```
初期価格 p₀ → 市場メカニズム → 均衡価格 p*
```

Rectified Flowでは：

```
連続分布 μ → Flow変換 φ → 離散バンドル s
```

両方とも「初期状態から望ましい終状態への連続的な移行」を記述。

#### 確率的な経路

時刻 `t` での状態は**確率的**：

$$
s_t \sim \text{Law}(s_t | s_0)
$$

これは確率微分方程式（SDE）ではなく、常微分方程式（ODE）の解の分布。

---

## 3. 数学的定式化

### 3.1 ベクトル場 `φ`

**定義**: `φ: [0,1] × ℝ^m × ℝ^m → ℝ^m` は以下の形：

$$
\phi(t, s_t, s_0) = \eta(t) \cdot Q(s_0) \cdot s_t
$$

where:
- `η(t)`: スカラー関数（時間依存の「速度」）
- `Q(s_0) ∈ ℝ^{m×m}`: 行列関数（初期状態依存の「方向」）

**直感**:
- `η(t)`: 時刻 `t` でどれくらい速く変化するか
- `Q(s_0)`: どの方向に変化するか（初期値に依存）

### 3.2 ODEの積分

初期値問題：

$$
\begin{cases}
\frac{ds_t}{dt} = \phi(t, s_t, s_0) \\
s_t|_{t=0} = s_0
\end{cases}
$$

**数値解法**（Euler法）:

$$
s_{t+\Delta t} = s_t + \Delta t \cdot \phi(t, s_t, s_0)
$$

時間グリッド `0 = t₀ < t₁ < ... < t_N = 1` で反復：

$$
s_{t_{i+1}} = s_{t_i} + (t_{i+1} - t_i) \cdot \phi(t_i, s_{t_i}, s_0)
$$

**統計的解釈**: これは確率変数の決定論的変換（`s_0` がランダムなので `s_1` もランダム）

### 3.3 離散化（Rounding）

終点 `s_T ∈ ℝ^m` は連続値なので、離散バンドルに変換：

$$
s_i = \mathbb{1}\{s_{T,i} \geq 0.5\}, \quad i = 1, \ldots, m
$$

**経済学的類推**: 閾値ルール（例：投票での過半数、信用スコアでの承認基準）

---

## 4. 学習目的関数

### 4.1 Rectification Loss

**目標**: Flow `φ` を学習して、最短経路で `s_0 → s_T` を実現

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{s_0, s_T, t} \left[ \| (s_T - s_0) - \phi(t, s_t, s_0) \|^2 \right]
$$

where:
- `s_0 ~ μ₀` （初期分布からサンプル）
- `s = round(s_0)` （離散化）
- `s_T ~ N(s, σ_z² I)` （ノイズ付加、Eq. 14）
- `t ~ Uniform[0,1]` （ランダムな時刻）
- `s_t = t · s_T + (1-t) · s_0` （線形補間、Eq. 16）

### 4.2 直感的理解

#### (1) ターゲット: `s_T - s_0`

もし `φ` が完璧なら、

$$
s_T = s_0 + \int_0^1 \phi(t, s_t, s_0) dt
$$

微分すると、

$$
\phi(t, s_t, s_0) = \frac{d s_t}{dt} = s_T - s_0 \quad (\text{定数})
$$

つまり、理想的な Flow は**直線的な経路**（Rectified = 矯正された）

#### (2) 学習の仕組み

ランダムな時刻 `t` で、`φ` の予測と実際のターゲットを比較：

```
予測:   φ(t, s_t, s_0)
正解:   s_T - s_0
損失:   || 予測 - 正解 ||²
```

これは**regression problem**（回帰問題）！

### 4.3 なぜノイズ `σ_z` が必要か

`s` は離散的（`{0,1}^m`）なので、そのままでは学習が難しい。

**Solution**: 離散点の周辺に Gaussian ノイズを追加：

$$
s_T \sim \mathcal{N}(s, \sigma_z^2 I)
$$

**効果**:
- 連続的な分布を作る → 勾配が流れやすい
- 同じバンドル `s` でも複数の `s_T` が生成 → データ拡張
- `σ_z` が小さい → `s_T ≈ s` （離散に近い）

**経済学的類推**: Logit shock in discrete choice models

### 4.4 密度重み項

Flowの変換で確率密度がどう変わるかを追跡：

$$
\log p(s_T) = \log p(s_0) - \int_0^T \text{Tr}[Q(s_0)] \cdot \eta(t) \, dt
$$

**Jacobian の行列式**:

$$
\frac{\partial \phi}{\partial s_t} = \eta(t) \cdot Q(s_0)
$$

トレース（対角和）:

$$
\text{div} \, \phi = \text{Tr}[Q(s_0)] \cdot \eta(t)
$$

**経済学的解釈**: 
- Demand system での価格変化に対する需要の変化（Slutsky方程式）
- こここでは「状態空間の体積の変化」を追跡

---

## 5. ニューラルネットワークによる近似

### 5.1 パラメトリックモデル

`φ` を深層ニューラルネットワークで近似：

$$
\phi(t, s_t, s_0; \theta) = \eta(t; \theta_\eta) \cdot Q(s_0; \theta_Q) \cdot s_t
$$

#### (A) `Q(s_0; θ_Q)`: 行列値ニューラルネット

```
s_0 ∈ ℝ^m → [Linear → Tanh]×3 → Linear → Q ∈ ℝ^{m×m}
```

**入力**: 初期状態 `s_0`  
**出力**: `m × m` 行列

**役割**: `s_0` に応じて「どの方向に進むか」を決定

#### (B) `η(t; θ_η)`: スカラー値ニューラルネット

```
t ∈ [0,1] → [Linear → Tanh]×2 → Linear → Tanh → η ∈ [-c, c]
```

**入力**: 時刻 `t`  
**出力**: スカラー

**役割**: 時間経過に応じて「どれくらい速く進むか」を調整

### 5.2 なぜこの構造？

#### 線形性の重要性

`φ` が `s_t` に関して線形：

$$
\phi(t, \alpha s_t, s_0) = \alpha \cdot \phi(t, s_t, s_0)
$$

**利点**:
- ODEの解が安定
- 理論的な保証（Lipschitz連続性）

**経済学的類推**: 線形需要システム（AIDS modelなど）

#### 初期状態依存

`Q` が `s_0` に依存：

$$
Q(s_0) \neq Q(s_0')
$$

**直感**: 
- 異なる初期分布から異なるバンドルを生成したい
- `s_0 = [0.8, 0.2, 0.1, ...]` → 「最初の商品を含むバンドル」に向かう
- `s_0 = [0.1, 0.9, 0.8, ...]` → 「2-3番目の商品を含むバンドル」に向かう

**経済学的類推**: State-dependent transition probabilities（Markov過程）

---

## 6. 学習アルゴリズム

### 6.1 データ生成プロセス

各イテレーションで：

1. **初期サンプリング**: `s_0 ~ Σ w_d N(μ_d, σ_d² I)`
2. **離散化**: `s = round(s_0)` （要素ごとに `s_i = 𝟙{s_{0,i} ≥ 0.5}`）
3. **ノイズ付加**: `s_T ~ N(s, σ_z² I)`
4. **時刻サンプリング**: `t ~ Uniform[0,1]`
5. **線形補間**: `s_t = t · s_T + (1-t) · s_0`

**統計的解釈**: 
- `s_0, s_T, t` は i.i.d. サンプル（Monte Carlo法）
- `s_t` は conditional distribution `p(s_t | s_0, s_T, t)`

### 6.2 勾配降下法

損失関数：

$$
\mathcal{L}(\theta) = \mathbb{E}_{s_0, s_T, t} \left[ \| (s_T - s_0) - \phi(t, s_t, s_0; \theta) \|^2 \right]
$$

勾配：

$$
\nabla_\theta \mathcal{L} = \mathbb{E} \left[ -2(s_T - s_0 - \phi) \cdot \nabla_\theta \phi \right]
$$

**アルゴリズム**（SGD）:

```
for iteration = 1 to N:
    # サンプリング
    s_0 ~ μ₀
    s = round(s_0)
    s_T ~ N(s, σ_z² I)
    t ~ U[0,1]
    s_t = t·s_T + (1-t)·s_0
    
    # 損失計算
    target = s_T - s_0
    pred = φ(t, s_t, s_0; θ)
    loss = ||target - pred||²
    
    # パラメータ更新
    θ ← θ - η ∇_θ loss
```

**経済学的類推**: 
- Simulated Method of Moments (SMM)
- Neural network = flexible functional form

### 6.3 正則化項

過学習を防ぐための追加項：

#### (A) Jacobian Penalty（軌道短縮）

$$
\lambda_j \cdot \mathbb{E} \left[ \left\| \frac{\partial \phi}{\partial s_t} \right\|_F^2 \right]
$$

**直感**: `φ` の変化を滑らかにする（過度に複雑な軌道を防ぐ）

#### (B) Kinetic Energy（ベクトル場制御）

$$
\lambda_k \cdot \mathbb{E} \left[ \| \phi(t, s_t, s_0) \|^2 \right]
$$

**直感**: Flow の「速度」を制限（発散を防ぐ）

#### (C) Trace Penalty（発散抑制）

$$
\lambda_{tr} \cdot \mathbb{E} \left[ (\text{Tr}[Q(s_0)])^2 \right]
$$

**直感**: 密度重みの変化を抑制（数値的安定性）

---

## 7. 推論（バンドル生成）

### 7.1 学習後の使用方法

パラメータ `θ` が学習できたら、バンドルを生成：

```python
# 1. 初期分布からサンプル
s_0 ~ Σ w_d N(μ_d, σ_d² I)

# 2. ODEを積分
for i = 0 to N-1:
    s_{t_{i+1}} = s_{t_i} + Δt · φ(t_i, s_{t_i}, s_0; θ)

# 3. 離散化
s_i = 𝟙{s_{T,i} ≥ 0.5}, i = 1, ..., m
```

**結果**: `s ∈ {0,1}^m` （離散バンドル）

### 7.2 多様性の確保

異なる `s_0` から異なる `s` が生成される：

```
s_0^(1) = [0.9, 0.1, 0.2, ...] → s^(1) = [1, 0, 0, ...]
s_0^(2) = [0.1, 0.8, 0.9, ...] → s^(2) = [0, 1, 1, ...]
s_0^(3) = [0.5, 0.5, 0.1, ...] → s^(3) = [?, ?, 0, ...]  # 確率的
```

**経済学的意義**: 
- 多様なバンドルを提供 → 異質な買い手に対応
- メカニズムデザインの「メニュー」概念と一致

---

## 8. BundleFlowにおける2段階学習

### 8.1 Stage 1: Rectified Flow の学習

**入力データ**: なし（教師なし学習）

**学習対象**: 
- `μ_d` （各成分の中心）
- `w_d` （混合重み）
- `θ_Q, θ_η` （ニューラルネットのパラメータ）

**目的**: 多様なバンドルを生成できる分布を学習

$$
\min_{\mu, w, \theta} \mathbb{E}_{s_0 \sim \mu_0} \left[ \| (s_T - s_0) - \phi(t, s_t, s_0; \theta) \|^2 \right]
$$

**経済学的解釈**: 
- 「市場に出回る商品の種類」を決める
- 買い手の嗜好を**まだ考慮していない**（純粋に技術的な問題）

### 8.2 Stage 2: メニュー最適化

**入力データ**: 買い手の評価値分布 `V = {v₁, ..., v_n}`

**学習対象**: 
- `K` 個のメニュー要素、各々が：
  - `β_k`: 価格
  - `μ_d^(k)`: 初期分布（バンドル生成用）
  - `w_d^(k)`: 混合重み

**目的**: 収益を最大化

$$
\max_{\beta, \mu, w} \mathbb{E}_{v \sim \mathcal{V}} \left[ \sum_{k=1}^K Z_k(v) \cdot \beta_k \right]
$$

where `Z_k(v)` は買い手 `v` がメニュー `k` を選ぶ確率（softmax or Gumbel-Softmax）

**経済学的解釈**:
- Myerson の最適オークション設計
- Price discrimination（価格差別）
- Menu design（Rochet & Choné 1998）

### 8.3 なぜ2段階？

#### 技術的理由

1. **探索空間の削減**: Stage 1で「合理的なバンドル空間」を学習
2. **初期化**: Stage 2で良い初期値を提供
3. **安定性**: 2つの目的を分離 → 学習が安定

#### 経済学的類推

**2段階推定** (Two-step estimation):
1. **第1段階**: 構造的パラメータを推定
2. **第2段階**: 経済的パラメータを推定（第1段階を固定）

例: Heckman selection model, Control function approach

---

## 9. 理論的正当化

### 9.1 なぜ直線的な経路？

**Rectified Flow の主張**: 
> 最適な輸送は直線的

これは**Optimal Transport理論**に基づく：

$$
\min_{\gamma} \mathbb{E}_{(s_0, s_T) \sim \gamma} \left[ \| s_T - s_0 \|^2 \right]
$$

subject to: `γ` の周辺分布が `μ₀` と `s` の分布に一致

**Monge-Kantorovich問題**の特殊ケース（`L²` コスト）

#### 経済学での類似概念

- **Walrasian equilibrium**: 最も効率的な資源配分
- **Lindahl prices**: 公共財の最適価格
- **Matching theory**: 安定マッチング（Gale-Shapley）

### 9.2 Flow Matching の背景

Rectified Flowは**Flow Matching**の一種：

1. **Normalizing Flows** (Rezende & Mohamed 2015): 可逆変換で複雑な分布を生成
2. **Continuous Normalizing Flows** (Chen et al. 2018): ODEで表現
3. **Flow Matching** (Lipman et al. 2023): 直接ODEを学習
4. **Rectified Flow** (Liu et al. 2022): 直線経路に矯正

**統計学での類似**: 
- Normalizing constants in Bayesian inference
- Importance sampling の reweighting

---

## 10. 実装上の技術的詳細

### 10.1 Spectral Normalization

ニューラルネット `Q` の重みに制約：

$$
\| W \|_2 \leq 1 \quad (\text{最大特異値を1に制限})
$$

**目的**: Lipschitz 連続性を保証

$$
\| Q(s_0) - Q(s_0') \| \leq L \| s_0 - s_0' \|
$$

**経済学的類推**: 
- Smoothness restriction in nonparametric estimation
- Bounded rationality （限定合理性）

### 10.2 学習スケジュール

#### Cosine Annealing (学習率調整)

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

**直感**: 学習の終盤で小さい学習率 → 精密な調整

**経済学での類似**: Sequential testing with declining significance levels

### 10.3 数値的安定性

#### Log-Sum-Exp Trick

重み付き和を計算する際：

$$
\sum_{d=1}^D w_d \cdot \exp(\text{log\_density}_d) \cdot v_d
$$

数値的にオーバーフローを防ぐ：

$$
M = \max_d \text{log\_weight}_d
$$

$$
\text{sum} = \exp(M) \cdot \sum_d \exp(\text{log\_weight}_d - M) \cdot v_d
$$

**統計学での類似**: Logsumexp in softmax regression, log-likelihood computation

---

## 11. Straight-Through Estimator (STE)

### 11.1 離散化の勾配問題

**問題**: `round` 関数は微分不可能

$$
s = \mathbb{1}\{s_T \geq 0.5\}
$$

勾配:

$$
\frac{\partial s}{\partial s_T} = 0 \quad (\text{ほぼすべての点で})
$$

**結果**: 学習が進まない！

### 11.2 STEの解決策

**Forward時**: 離散的な操作を実行

$$
s^{\text{forward}} = \mathbb{1}\{s_T \geq 0.5\}
$$

**Backward時**: 連続的な近似の勾配を使用

$$
\frac{\partial s}{\partial s_T}\Big|_{\text{backward}} = \frac{\partial}{\partial s_T} \sigma\left(\frac{s_T - 0.5}{\tau}\right)
$$

where `σ(x) = 1/(1 + e^{-x})` はsigmoid関数。

**PyTorch実装**:

```python
s_hard = (s_T >= 0.5).float()
s_soft = torch.sigmoid((s_T - 0.5) / tau)
s = s_hard - s_soft.detach() + s_soft
#   ^^^^^^   ^^^^^^^^^^^^^^^^   ^^^^^^
#   使う値    勾配をブロック    勾配の元
```

### 11.3 バイアスと実務的成功

#### 理論的問題

STEは**バイアスのある推定量**：

$$
\mathbb{E}[\nabla_\theta^{\text{STE}}] \neq \nabla_\theta \mathbb{E}[\mathcal{L}]
$$

真の勾配と異なる！

#### なぜ機能するか

1. **低分散**: Policy Gradientより分散が小さい
2. **局所的近似**: `f_hard ≈ f_soft` when `τ` is small
3. **大数の法則**: バッチサイズが大きいとバイアスが平均化
4. **経験的成功**: VQ-VAE, Binarized NNsなどで実証済み

**計量経済学での類推**:
- Quasi-maximum likelihood: 誤った尤度関数でも一致推定量
- GMM: モーメント条件の近似

---

## 12. Stage 2でのGumbel-Softmax

### 12.1 メニュー選択の問題

買い手 `v` は効用最大化：

$$
k^* = \arg\max_{k \in [K]} u^{(k)}(v)
$$

where `u^(k) = v(S_k) - β_k`.

**問題**: `argmax` は微分不可能

### 12.2 Gumbel-Max Trick

**定理**: カテゴリカル分布は以下と等価：

$$
k \sim \text{Categorical}(p) \iff k = \arg\max_i (\log p_i + G_i)
$$

where `G_i ~ Gumbel(0,1)` は独立。

**Gumbel分布**:

$$
P(G \leq g) = \exp(-\exp(-g))
$$

**経済学的解釈**: 
- Logit model の確率的効用理論
- Random utility maximization (McFadden 1974)

### 12.3 Gumbel-Softmax

`argmax` を `softmax` で連続緩和：

$$
y_k = \frac{\exp\left( (u^{(k)} + G_k) / \tau \right)}{\sum_j \exp\left( (u^{(j)} + G_j) / \tau \right)}
$$

**温度 `τ`**:
- `τ → 0`: one-hot（離散）
- `τ → ∞`: 一様分布（連続）

**+ STE**:

```python
# Forward: hard選択（テストと同じ）
k* = argmax(u + G)
revenue = β[k*] if u[k*] >= 0 else 0

# Backward: softの勾配（学習可能）
y_soft = softmax((u + G) / τ)
gradient uses: ∂y_soft/∂u
```

### 12.4 なぜSoftmax relaxationは失敗したか

#### 従来（Softmax relaxation）

```
Forward:  Z = softmax(u / λ)  # 連続
Backward: Z = softmax(u / λ)  # 連続
Test:     k* = argmax(u)      # 離散 ← 不一致！
```

**問題**: Training objective ≠ Test objective

#### Gumbel + STE

```
Forward:  k* = argmax(u + G)  # 離散（テストと同じ）
Backward: y = softmax(...)     # 連続（勾配計算）
Test:     k* = argmax(u)       # 離散 ✓
```

**利点**: Training ≈ Test

**経済学的教訓**:
> Estimating equation should match the economic model

- Simulated MLE: Simulated distribution ≈ True distribution
- Indirect inference: Auxiliary model ≈ Structural model

---

## 13. 計算複雑度

### 13.1 Stage 1

各イテレーション:

1. **サンプリング**: `O(B · m)`
2. **ODE積分**: `O(B · m² · N_steps)` （行列積 `Q · s_t`）
3. **勾配計算**: `O(B · m² · N_params)`

**Total**: `O(B · m² · N_steps)` per iteration

`m=50, B=512, N_steps=25` の場合: 約3200万演算/iteration

### 13.2 Stage 2

各イテレーション:

1. **バンドル生成**: `K · D` 回のODE積分 → `O(K · D · m² · N_steps)`
2. **効用計算**: `B` 個の買い手 × `K` 個のメニュー → `O(B · K · D)`
3. **Softmax/Gumbel**: `O(B · K)`

**Total**: `O(K · D · m² · N_steps + B · K · D)`

`m=50, K=512, D=8, B=128, N_steps=25` の場合: 約50億演算/iteration

**GPUの重要性**: バッチ処理で並列化 → 実際には数秒

---

## 14. 理論的限界と拡張可能性

### 14.1 現在の制約

#### (A) 離散化の損失

`round(s_T)` で情報が失われる：

```
s_T = [0.49, 0.51] → s = [0, 1]  # 微妙な差が消える
s_T = [0.01, 0.99] → s = [0, 1]  # 同じ結果
```

**解決策（将来）**: Soft rounding の改善、STEの精緻化

#### (B) 初期分布の設計

`μ₀` の選択に恣意性：

- Gaussian混合を仮定
- 支持サイズ `D` は固定

**解決策**: データ駆動の初期分布設計

### 14.2 経済学への応用可能性

#### (A) 多商品オークション以外

- **Matching markets**: 医師-病院、学校選択
- **Voting systems**: 投票ルール設計
- **Contract design**: 複雑な契約の最適化

#### (B) 評価値推定との統合

現在は `v` が既知と仮定。実務では：

```
データ → 評価値推定 → メカニズム設計
```

**提案**: End-to-end学習

---

## 15. まとめ

### Rectified Flowの3つの柱

1. **ODE-based generation**: 連続分布から離散構造へ
2. **Learnable transformation**: 深層学習で柔軟な写像を獲得
3. **Straight-through estimation**: 離散操作の微分可能近似

### 経済学との接点

| 概念 | Rectified Flow | 経済学 |
|------|---------------|--------|
| 状態遷移 | `s_0 → s_T` via ODE | Tatonnement, adjustment process |
| 確率的選択 | Gumbel-Max Trick | Random utility model (Logit) |
| 最適化 | SGD on neural net | Simulated methods (SMM, MSM) |
| 正則化 | Jacobian/Kinetic penalty | Smoothness restriction |
| 2段階学習 | Stage 1 + Stage 2 | Two-step estimation |

### 実装のポイント

1. **確率分布の正規化**: すべて `softmax` で保証 ✓
2. **数値安定性**: Log-sum-exp, clipping ✓
3. **勾配の流れ**: STE, Gumbel-Softmaxで確保 ✓

### 今後の研究方向

1. **理論的保証**: STEのバイアス評価、収束条件
2. **実データ適用**: CATS, FCC spectrum auction データ
3. **他手法との比較**: RegretNet, RochetNet, VCG
4. **計算効率化**: 近似アルゴリズム、分散学習

---

## 16. 参考文献

### Rectified Flow関連

1. **Liu, X., Gong, C., & Liu, Q. (2022).** Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *arXiv:2209.03003*.
   - Rectified Flowの原論文

2. **Lipman, Y., Chen, R. T. Q., et al. (2023).** Flow Matching for Generative Modeling. *ICLR 2023*.
   - Flow Matchingの一般的枠組み

3. **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).** Neural Ordinary Differential Equations. *NeurIPS 2018*.
   - Neural ODEの基礎

### Gumbel-Softmax & STE

4. **Jang, E., Gu, S., & Poole, B. (2017).** Categorical Reparameterization with Gumbel-Softmax. *ICLR 2017*.
   - Gumbel-Softmaxの提案

5. **Bengio, Y., Léonard, N., & Courville, A. (2013).** Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. *arXiv:1308.3432*.
   - Straight-Through Estimatorの提案

### 経済学・最適化理論

6. **Villani, C. (2009).** Optimal Transport: Old and New. *Springer*.
   - Optimal Transport理論の教科書

7. **McFadden, D. (1974).** Conditional Logit Analysis of Qualitative Choice Behavior. *Frontiers in Econometrics*.
   - Random utility modelの基礎

8. **Myerson, R. B. (1981).** Optimal Auction Design. *Mathematics of Operations Research*, 6(1), 58-73.
   - 最適メカニズム設計

9. **Rochet, J. C., & Choné, P. (1998).** Ironing, Sweeping, and Multidimensional Screening. *Econometrica*, 66(4), 783-826.
   - 多次元スクリーニング

### BundleFlow

10. **Wang, T., Jiang, Y., & Parkes, D. C. (2025).** BundleFlow: Deep Menus for Combinatorial Auctions by Diffusion-Based Optimization. *arXiv:2502.15283*.
    - BundleFlowの原論文

---

## 付録A: 主要な数式一覧

### Stage 1: Rectified Flow学習

**初期分布**:

$$
s_0 \sim \mu_0 = \sum_{d=1}^D w_d \cdot \mathcal{N}(\mu_d, \sigma_d^2 I)
$$

**ベクトル場**:

$$
\phi(t, s_t, s_0) = \eta(t) \cdot Q(s_0) \cdot s_t
$$

**損失関数**:

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{s_0, t} \left[ \| (s_T - s_0) - \phi(t, s_t, s_0) \|^2 \right]
$$

where:
- `s = round(s_0)`
- `s_T ~ N(s, σ_z² I)`
- `s_t = t · s_T + (1-t) · s_0`

**正則化**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_j \mathcal{R}_{\text{jacobian}} + \lambda_k \mathcal{R}_{\text{kinetic}} + \lambda_{tr} \mathcal{R}_{\text{trace}}
$$

### Stage 2: メニュー最適化

**効用関数**:

$$
u^{(k)}(v) = \sum_{d=1}^D w_d^{(k)} \cdot v(s_d^{(k)}) \cdot \rho_d^{(k)} - \beta^{(k)}
$$

where:
- `s_d^(k)` = Flow で生成されたバンドル（`μ_d^(k)` から）
- `ρ_d^(k)` = 密度重み `exp(-Tr[Q(μ_d^(k))] ∫ η)`
- `β^(k)` = 価格

**収益（Gumbel-Softmax版）**:

$$
\text{Revenue} = \mathbb{E}_{v, G} \left[ \beta_{k^*} \cdot \mathbb{1}\{u^{(k^*)} \geq 0\} \right]
$$

where `k* = argmax_k (u^(k) + G_k)`.

**損失関数**:

$$
\mathcal{L}_{\text{revenue}} = -\text{Revenue}
$$

---

## 付録B: 用語対照表

| 技術用語 | 経済学的解釈 |
|---------|------------|
| Neural ODE | Continuous-time dynamic system |
| Flow Matching | Optimal transport problem |
| Rectified Flow | Monge transportation |
| Gumbel-Softmax | Multinomial logit with error |
| Straight-Through | Quasi-likelihood approximation |
| Spectral Norm | Lipschitz constraint |
| Log-sum-exp | Numerical stable logsumexp |
| Batch normalization | Cross-sectional standardization |
| Gradient clipping | Bounded learning step |
| Cosine annealing | Adaptive step size |
| Mixture of Gaussians | Latent class model |
| Density weight | Jacobian determinant |
| Softmax temperature | Choice precision (McFadden) |
| Menu element | Contract offering (Rochet-Choné) |
| Utility | Indirect utility function |
| IR constraint | Participation constraint |
| Revenue | Expected payment |

---

**Document Author**: Technical explanation for economists  
**Date**: October 12, 2025  
**Target Audience**: Economists familiar with statistics but new to machine learning  
**Prerequisites**: Probability theory, optimization, basic linear algebra

