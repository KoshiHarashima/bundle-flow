# BundleFlow: 包括的ガイド

**🎯 Rectified Flow–based menus for combinatorial auctions**

このドキュメントは、BundleFlowプロジェクトの完全なガイドです。セットアップから高度な設定、問題解決まで、すべての情報を一箇所にまとめています。

---

## 📋 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [クイックスタート](#2-クイックスタート)
3. [モデル理論](#3-モデル理論)
4. [セットアップガイド](#4-セットアップガイド)
5. [設定とパラメータ](#5-設定とパラメータ)
6. [問題解決とトラブルシューティング](#6-問題解決とトラブルシューティング)
7. [経済学的評価](#7-経済学的評価)
8. [API リファレンス](#8-api-リファレンス)
9. [開発者向け情報](#9-開発者向け情報)

---

## 1. プロジェクト概要

### 1.1 BundleFlowとは

BundleFlowは、Rectified Flowモデルを使用した組み合わせオークションのための革新的なメニュー最適化システムです。

**主な特徴:**
- **Stage 1**: Flow初期化による束生成
- **Stage 2**: メニュー最適化による収入最大化
- **新しいAPI構造**: 明確な関心の分離
- **型安全性**: 完全な型注釈と経済記号のドキュメント
- **数値安定性**: Log-sum-exp、softplus制約、ウォームアップスケジューリング
- **GPU加速**: CUDA/MPSサポートと自動最適化
- **再現性**: 決定論的アルゴリズムと包括的な環境チェック

### 1.2 プロジェクト構造

```
bundle-flow/
├─ bundleflow/          # メインパッケージ
│  ├─ models/           # BundleFlow, MenuElement, Mechanism
│  ├─ valuation/        # XORValuation
│  ├─ train/            # Stage1, Stage2学習スクリプト
│  ├─ data.py           # データローダー
│  └─ utils.py          # ユーティリティ
├─ conf/                # 設定ファイル
├─ tests/               # テスト
├─ tools/               # 環境チェックツール
├─ checkpoints/         # チェックポイント
├─ BundleFlow.ipynb     # デモノートブック
├─ CITATION.cff         # 引用情報
├─ LICENSE              # ライセンス
├─ Makefile             # ビルドツール
└─ pyproject.toml       # パッケージ設定
```

### 1.3 クイックコマンド

```bash
make env        # 環境構築
make test       # テスト実行
make format     # コードフォーマット
make lint       # リントチェック
make reproduce  # 5分で再現（小規模）
```

---

## 2. クイックスタート

### 2.1 環境構築

```bash
# 1. パッケージのインストール
pip install -e .

# 2. 基本動作確認
python -c "from bundleflow.models.flow import BundleFlow; print('✅ インストール成功')"

# 3. テスト実行
pytest tests/ -v
```

### 2.2 最小実行例

```python
from bundleflow.models.flow import BundleFlow
from bundleflow.models.menu import MenuElement, Mechanism
from bundleflow.valuation.valuation import XORValuation
import torch

# 1. 速度場の初期化
m = 10  # 商品数
flow = BundleFlow(m=m)

# 2. メニュー要素の作成
K = 5   # メニュー要素数
D = 8   # 初期分布の混合成分数
menu = [MenuElement(m=m, D=D) for _ in range(K)]

# 3. 評価関数の作成
atoms = [([1, 2, 3], 5.0), ([4, 5], 3.0), ([6, 7, 8, 9], 8.0)]
valuation = XORValuation.from_bundle_list(m, atoms)

# 4. メカニズムの作成
mechanism = Mechanism(flow, menu)

# 5. 期待収入の計算
revenue = mechanism.expected_revenue([valuation])
print(f"期待収入: {revenue.item():.4f}")
```

### 2.3 Stage1/2学習

```bash
# Stage1学習（Flow初期化）
bundleflow-stage1 --cfg conf/stage1.yaml

# Stage2学習（Menu最適化）
bundleflow-stage2 --cfg conf/stage2.yaml
```

---

## 3. モデル理論

### 3.1 基本記号

- **m**: 商品数
- **b ∈ {0,1}^m**: 束（離散）
- **x ∈ ℝ^m**: 連続束変数（ODE上での状態）
- **K**: メニュー要素数
- **D**: 初期分布の混合成分数

### 3.2 三層モデル構造

#### 3.2.1 速度場 v_θ: ℝ^m × [0,1] → ℝ^m
- **目的**: 連続変数xをdx/dt=v_θ(x,t)で輸送し、x(1)の支持を{0,1}^mに寄せる
- **実装**: `BundleFlow`クラス（QNet + EtaNet）
- **記号**: φ(t,s_t) = η(t)·Q(s0)·s_t （Eq.9）

#### 3.2.2 初期分布 p_0(φ_k)
- **目的**: 各メニュー要素kごとに異なる初期分布パラメータφ_kを学習
- **実装**: `MenuElement`クラスの`mus`パラメータ
- **記号**: s0 ~ p_0(φ_k) = Σ_d w_d^(k) δ(μ_d^(k))

#### 3.2.3 メニュー価格 p_k
- **目的**: 各メニュー要素kの価格を学習
- **実装**: `MenuElement`クラスの`beta`パラメータ
- **記号**: p_k = softplus(β_k) ≥ 0

### 3.3 生成過程

1. **初期化**: z ~ p_0(φ_k)
2. **ODE積分**: dx/dt = v_θ(x,t), x(0) = z
3. **離散化**: b = I(x(1) ≥ 0.5) （評価専用）

### 3.4 目的関数

- **効用**: U(b;v) = v(b) - p_k
- **収入**: R = Σ_k z_k(v) · p_k
- **厚生**: W = Σ_k z_k(v) · v(b_k)

### 3.5 2段階学習

- **Stage 1**: 混合Gaussian（固定φ）でv_θの可行域学習
- **Stage 2**: メニューごとにsupportと重み（混合Dirac）＋価格を学習

### 3.6 制約と保証

- **DSIC**: メニューの自己依存排除・代理人最適化で担保（Rochet系）
- **IR制約**: U(b;v) ≥ 0 （Individual Rationality）
- **価格制約**: p_k ≥ 0 （softplusで保証）

### 3.7 Liouville方程式

密度重み: exp(-Tr[Q(s0)] ∫_0^T η(t)dt) （Eq.12）

### 3.8 数値安定化

- **log-sum-exp**: 効用計算の数値安定化
- **softplus**: 価格の非負制約
- **勾配クリッピング**: 学習安定化
- **スペクトル正規化**: Lipschitz制御

---

## 4. セットアップガイド

### 4.1 GPU環境確認

#### Colab環境でのGPU確認

1. **Runtime設定**
   ```
   Runtime → Change runtime type → GPU（A100/T4）
   ```

2. **GPU確認**
   ```bash
   !nvidia-smi
   ```

### 4.2 CUDA版Torchのインストール

```bash
# 1. CUDA版PyTorchを明示的にインストール
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "torchvision" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "torchaudio" --index-url https://download.pytorch.org/whl/cu121

# 2. 環境確認
python tools/envcheck.py
```

### 4.3 パッケージインストール

```bash
# BundleFlowをインストール（相対パスimportの崩れを根絶）
python -m pip install -e .
```

**これで以下が利用可能:**
- `bundleflow-stage1` - Stage1学習（Flow初期化）
- `bundleflow-stage2` - Stage2学習（Menu最適化）

### 4.4 最小の動作確認

```bash
# 1. 基本動作確認
python - <<'PY'
from bundleflow.valuation.valuation import XORValuation
import torch
s = torch.zeros(6); s[0] = 1
v = XORValuation.from_bundle_list(6, [([1], 1.0)])
print("✅ OK:", isinstance(v.value(s), float))
PY

# 2. 新しいAPI構造のテスト
python - <<'PY'
from bundleflow.models.flow import BundleFlow
from bundleflow.models.menu import MenuElement, Mechanism
from bundleflow.valuation.valuation import XORValuation
import torch

# 速度場のテスト
flow = BundleFlow(m=5)
x = torch.randn(3, 5)
t = torch.rand(3)
v = flow.velocity(x, t)
print("✅ BundleFlow velocity shape:", v.shape)

# メニュー要素のテスト
elem = MenuElement(m=5, D=3)
price = elem.price()
print("✅ MenuElement price:", price.item())

# 評価関数のテスト
val = XORValuation.from_bundle_list(5, [([1, 2], 1.0)])
bundle = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
value = val.value(bundle)
print("✅ XORValuation value:", value)

print("🎉 新しいAPI構造が正常に動作しています！")
PY
```

### 4.5 よくある落とし穴

#### 1. ノートブック／端末のPython実体が異なる

**問題:** `torch.cuda.is_available()==False` になる

**解決:**
```bash
# ❌ 間違い
pip install torch

# ✅ 正しい（必ず python -m pip を使用）
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
```

#### 2. 相対パスimportの崩れ

**問題:** `ModuleNotFoundError: No module named 'bundleflow'`

**解決:** 必ず `pip install -e .` を実行

#### 3. 決定論的設定の不備

**問題:** 再現性が保証されない

**解決:** 環境チェックツールで確認
```bash
python tools/envcheck.py
```

---

## 5. 設定とパラメータ

### 5.1 Stage1設定（Flow初期化）

```yaml
# conf/stage1.yaml
m: 50
D: 8
iters: 60000
batch: 1024
lr: 5e-3
sigma_z: 0.05
ode_steps: 25
seed: 42
device: "auto"
out_dir: "checkpoints"
use_scheduler: false
warmup_ratio: 0.05
weight_decay: 0.0
trace_penalty: 0.0
lambda_j: 0.0
lambda_k: 0.0
lambda_tr: 0.0
grad_clip: 1.0
ckpt_every: 5000
log_every: 200
probe: 0
cpu: false
```

### 5.2 Stage2設定（Menu最適化）

#### 5.2.1 現実的なStage2設定

```yaml
# conf/stage2.yaml
m: 50
K: 32           # メニュー要素数（中規模）
D: 8            # 初期分布の混合成分数（より多様）
iters: 2000     # イテレーション数（十分な学習）
batch: 64       # バッチサイズ（安定性重視）
lr: 5e-3        # 学習率（安定した学習）
lam_start: 0.1  # SoftMax温度の開始値（柔軟な選択）
lam_end: 0.5    # SoftMax温度の終了値（適度な硬直化）
ode_steps: 50   # ODE積分ステップ数（高精度）
n_val: 500      # 評価関数の数（統計的信頼性）
a: 64           # XOR原子数（現実的な複雑さ）
seed: 42
```

#### 5.2.2 設定変更の根拠

**1. メニュー要素数: 16 → 32**
- より多様な価格帯を提供
- 価格差別戦略の実現
- 顧客セグメントの細分化

**2. 混合成分数: 4 → 8**
- より多様な初期分布
- 束生成の多様性向上
- 局所最適解の回避

**3. 学習率: 3e-1 → 5e-3**
- 価格パラメータの安定した学習
- 発散の防止
- 細かい価格調整の実現

**4. 温度スケジューリング: 1e-3-0.1 → 0.1-0.5**
- 柔軟なメニュー選択から硬直的な選択へ
- 学習初期の探索と後期の活用のバランス
- 収入最適化の改善

### 5.3 小規模テスト用設定

```yaml
# conf/stage2_small_test.yaml
m: 20           # 小規模
K: 64           # 小規模
D: 4
iters: 1000     # 短縮
batch: 32
lr: 3e-1
ode_steps: 25
lam_start: 1e-3
lam_end: 2e-1
flow_ckpt: "checkpoints/flow_stage1_final.pt"
seed: 123
device: "auto"
a: 10           # 小規模
n_val: 1000     # 小規模
warmup_iters: 100
match_rate_threshold: 0.01
reinit_on_failure: 50
freeze_beta_iters: 200
auto_optimize: true
use_gumbel: false
warmstart: true
warmstart_grid: 50
reinit_every: 500
reinit_threshold: 0.01
grad_clip: 1.0
ckpt_every: 500
log_every: 50
eval_n: 100
atom_size_mode: "small"
cpu: false
```

### 5.4 期待される改善結果

#### 1. 価格分布の改善
```python
# 期待される結果
価格の統計: min=0.1234, max=4.5678, mean=1.2345
```
- 多様な価格帯の実現
- 価格差別戦略の成功
- 顧客セグメントへの対応

#### 2. 束生成の改善
```python
# 期待される結果
束サイズの統計: min=1.0, max=6.0, mean=3.2
多様性の統計: min=0.400, max=0.800, mean=0.600
```
- 空でない束の生成
- より多様な束の生成
- 効率的な資源配分

#### 3. 収入性能の改善
```python
# 期待される結果
期待収入: 15.2345
ハード割当収入: 12.3456
平均効用: 8.7654
IR制約満足率: 0.8500
```
- 期待収入とハード割当収入の乖離縮小
- 実際の取引の発生
- 適切なIR制約満足率

---

## 6. 問題解決とトラブルシューティング

### 6.1 重要な問題の分析

#### 6.1.1 Bundle Generation Failure（平均束サイズ=0）

**症状:**
```
平均束サイズ= 0.0±0.0 | 多様性=0.001
重み分布: [0.125 0.125 0.125 0.125 0.125]...
```

**解決方針:**

**1. Flow の"単体"テスト（RF/CFMとして成立しているか）**

```python
# sanity_flow.py
import torch
from bundleflow.models.flow import BundleFlow
T, m = 1.0, 10
flow = BundleFlow(m).eval()
t_grid = torch.linspace(0, T, 21)
# μを3点: all-zero / 0.5 / U[-0.2,1.2]
mus = torch.stack([torch.zeros(m), torch.full((m,),0.5), torch.rand(m)*1.4-0.2])
with torch.no_grad():
    sT = flow.flow_forward(mus, t_grid)    # (*,m)
    s  = (sT >= 0.5)
print("bundle sizes:", s.sum(dim=1).tolist())   # 例: [0, ~m/2, ~m/2] を期待
```

**2. 数値の初期化と丸めの問題を切る**

- **μ=0 病**: MenuElement.mus を全 0 で始めるのは最悪。mu ∈ [−0.2,1.2] の小ノイズ、もしくは代表束（全1 / 少数1）近傍でランダム初期化。
- **丸め**: (sT >= 0.5) は仕様に合うが、sT の平均が0.5付近だと少数誤差で全0化しやすい。t_grid を float64 に、sT も float64 で丸め直し→閾値越えがあるかをログ。

**3. 解析的な健全性チェック（線形系の簡易解）**

```python
# 行列指数近似での比較
torch.linalg.matrix_exp( integ_eta * Q(s0) ) @ s0
```

#### 6.1.2 Abnormal Learning Dynamics（末期の収益崩壊）

**観察されたパターン:**
```
Phase 1 (0-850):   Revenue rapid increase (0.18 → 6.37) ✅
Phase 2 (850-1375): Revenue gradual decline (6.37 → 4.87) ⚠️
Phase 3 (1375-2000): Revenue sharp decline (4.87 → 3.69) ❌
```

**解決方針:**

**1. Stage 2 を "RochetNet の数値安定版" に揃える**

```python
# Before
- weight = torch.exp(-trQ * integ)               # (D,)
- u = (vals * w * weight).sum() - elem.beta

# After
+ log_w = torch.log_softmax(elem.logits.float(), dim=0) - trQ.float() * integ.float()
+ M = log_w.max()
+ u = torch.exp(M) * torch.sum(torch.exp(log_w - M) * vals.float())
+ beta = F.softplus(elem.beta_raw)
+ u = u - beta
```

**2. "一致率=0%"を可視化して再初期化**

- match_rate＝(atom_mask & ~bundle_mask)==0 の割合を毎 iter ログ
- 連続 N iter で 0% なら、μ を再初期化（代表束近傍へ）／D を2以上に
- 合成XORの原子サイズがm/2だと一致確率は `0.5^25` レベルで実質0。平均5–8の"現実的サイズ"に変えるだけで学習信号が出る

**3. 最低限の訓練安定トリック**

- **Optim**: AdamW, lr を2–5×小さく、grad_clip=0.5
- 学習前半は β を凍結（1–2k iter、μ,w だけ更新）→一致が出てから β を解放
- **監視**: rev(hard-argmax), beta_med, z_entropy（要素の利用分散）

### 6.2 トラブルシューティング

#### 6.2.1 GPU未使用の対策

1. **CUDA wheel再インストール**
   ```bash
   python -m pip uninstall torch torchvision torchaudio -y
   python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Python実体確認**
   ```bash
   which python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **環境チェック**
   ```bash
   python tools/envcheck.py > envcheck.log 2>&1
   cat envcheck.log
   ```

#### 6.2.2 Stage2の数値安定

1. **β≥0制約**
   - `softplus`関数で実装済み
   - 暴走抑止効果

2. **指数重みの安定化**
   - `log-sum-exp`で実装済み
   - オーバーフロー防止

3. **ウォームアップ**
   - 最初のN千iterはβ固定
   - λ（SoftMax）を1e-3→0.2へ

4. **マッチ率監視**
   - 連続0%ならμ再初期化
   - 合成XORの原子サイズを小さめ（期待5-8）に設定

#### 6.2.3 学習信号の確保

1. **合成XORの原子サイズ**
   - 小さめ（期待5-8）に変更済み
   - ランダム半分原子の一致確率向上

2. **マッチ率目標**
   - >1%を目安
   - 統計設定で詰むのを回避

### 6.3 監視すべきメトリクス

#### Stage1（Flow初期化）
- **Coverage probe**: 被覆率が徐々に上昇（最終的に >0.1）
- **Loss**: RectifiedFlow損失が収束
- **Bundle diversity**: 生成されるバンドルの多様性

#### Stage2（Menu最適化）
- **match_rate**: (atom_mask & ~bundle_mask)==0 の割合（>1% を目標）
- **rev, rev@hard**: SoftMax 収益と argmax 収益
- **β 統計**: min/median/max（ウォームアップ後に上がり過ぎないか）
- **z 分布**: 要素利用率の偏り（死に要素の検知）

---

## 7. 経済学的評価

### 7.1 経済学的意義

#### 7.1.1 理論的基盤

**Rectified Flow による束生成**
- **連続最適化**: 離散的な束空間 `{0,1}^m` を連続空間 `ℝ^m` で近似
- **効率性**: ODE積分による滑らかな束生成で局所最適解を回避
- **スケーラビリティ**: 商品数 `m` が増加しても計算量が指数的に増加しない

**2段階学習アプローチ**
- **Stage 1**: 速度場 `v_θ` の学習（技術的制約の解決）
- **Stage 2**: メニュー最適化（経済的目標の達成）
- **分離原理**: 技術的制約と経済的目標を分離して最適化

#### 7.1.2 メカニズムデザインの観点

**Individual Rationality (IR) 制約**
```python
ir_mask = (selected_utility >= 0.0).float()  # IR制約の実装
```
- **理論的保証**: 代理人は非負の効用を得る場合のみ参加
- **実装の堅牢性**: `make_null_element()` によるフォールバック

**Dominant Strategy Incentive Compatibility (DSIC)**
- **理論的根拠**: メニュー形式により自己依存排除を実現
- **実装**: 各代理人は自分の真の評価関数に基づいて最適選択

**収入最大化**
```python
revenue = (Z * prices.unsqueeze(0)).sum(dim=1).mean()  # 期待収入
```
- **Myersonの原理**: 仮想評価関数に基づく最適価格設定
- **実装**: Softmax温度による滑らかな割当

### 7.2 実験結果の経済学的解釈

#### 7.2.1 Stage1: 束生成の多様性

**経済学的意味**
- **市場の厚み**: 多様な束が生成されることで、異なる選好を持つ代理人に対応
- **価格差別の可能性**: 束の多様性が価格差別戦略の基盤
- **効率性**: 連続最適化により、離散最適化では見つからない束も生成

#### 7.2.2 Stage2: メニュー最適化

**価格分布の分析**
```python
prices = [elem.price().detach().item() for elem in menu[:-1]]
print(f"価格の統計: min={min(prices):.4f}, max={max(prices):.4f}, mean={np.mean(prices):.4f}")
```

**経済学的解釈**
- **価格差別**: 価格の分散が大きいほど、異なる支払い意欲に対応
- **市場細分化**: 複数の価格帯で異なる顧客セグメントをターゲット
- **収入最適化**: 価格の分布が期待収入最大化に寄与

### 7.3 経済学的指標

#### 7.3.1 効率性指標

**Allocative Efficiency (AE)**
```python
# 総厚生の最大化
total_welfare = sum(valuation.value(allocated_bundle) for valuation, allocated_bundle in allocations)
max_possible_welfare = sum(max(valuation.value(bundle) for bundle in all_bundles) for valuation in valuations)
allocative_efficiency = total_welfare / max_possible_welfare
```

**Revenue Efficiency (RE)**
```python
# 収入の効率性
actual_revenue = mechanism.expected_revenue(valuations)
max_possible_revenue = sum(max(valuation.value(bundle) for bundle in all_bundles) for valuation in valuations)
revenue_efficiency = actual_revenue / max_possible_revenue
```

#### 7.3.2 公平性指標

**Individual Rationality Rate**
```python
ir_rate = result['ir_satisfied'].mean()  # IR制約満足率
```

**Price Discrimination Index**
```python
# 価格差別の程度
price_variance = np.var(prices)
price_discrimination_index = price_variance / np.mean(prices)
```

### 7.4 経済学的含意

#### 7.4.1 理論的貢献

**新しいアプローチ**
- **連続最適化**: 離散問題の連続緩和による効率的解法
- **生成モデル**: 束の生成と価格設定の統合的最適化
- **学習ベース**: データ駆動型のメカニズムデザイン

**実用的意義**
- **スケーラビリティ**: 大規模組み合わせオークションへの適用可能性
- **適応性**: 評価関数分布の変化への動的対応
- **計算効率**: リアルタイムオークションへの適用

#### 7.4.2 政策含意

**規制当局への示唆**
- **市場設計**: 効率的なオークション設計の指針
- **競争政策**: 価格差別と競争のバランス
- **消費者保護**: IR制約による消費者利益の保護

**実務者への示唆**
- **オークション設計**: 実際のオークションでの適用方法
- **価格戦略**: 動的価格設定の最適化
- **リスク管理**: 不確実性下での意思決定

---

## 8. API リファレンス

### 8.1 新しいAPI構造

#### 8.1.1 BundleFlow（速度場）

```python
from bundleflow.models.flow import BundleFlow

# 初期化
flow = BundleFlow(m=10)

# 速度計算
x = torch.randn(5, 10)  # (batch_size, m)
t = torch.rand(5)       # (batch_size,)
velocity = flow.velocity(x, t)

# 前進フロー
z = torch.randn(5, 10)  # 初期状態
t_grid = torch.linspace(0, 1, 21)
x_T = flow.flow_forward(z, t_grid)

# 束への丸め
bundles = flow.round_to_bundle(x_T)
```

#### 8.1.2 MenuElement（メニュー要素）

```python
from bundleflow.models.menu import MenuElement

# 初期化
elem = MenuElement(m=10, D=8)

# 価格取得
price = elem.price()

# 重み取得
weights = elem.weights

# 初期分布サンプル
samples = elem.sample_init(n=100)
```

#### 8.1.3 Mechanism（メカニズム）

```python
from bundleflow.models.menu import Mechanism

# 初期化
mechanism = Mechanism(flow, menu)

# 期待収入計算
revenue = mechanism.expected_revenue(valuations)

# ハード割当
result = mechanism.argmax_menu(valuations)
print(f"選択: {result['assignments']}")
print(f"効用: {result['utilities']}")
print(f"価格: {result['prices']}")
print(f"収入: {result['revenue']}")
```

#### 8.1.4 XORValuation（評価関数）

```python
from bundleflow.valuation.valuation import XORValuation

# 束リストから作成
atoms = [([1, 2, 3], 5.0), ([4, 5], 3.0)]
valuation = XORValuation.from_bundle_list(m=10, atoms)

# 価値計算
bundle = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
value = valuation.value(bundle)

# バッチ価値計算
bundles = torch.rand(5, 10)
values = valuation.batch_value(bundles)
```

### 8.2 後方互換性

既存のコードも引き続き動作します：

```python
# 旧API（後方互換性あり）
from bundleflow import FlowModel, MenuElement, XORValuation

# 新API（推奨）
from bundleflow.models import BundleFlow, MenuElement, Mechanism
from bundleflow.valuation import XORValuation
```

---

## 9. 開発者向け情報

### 9.1 開発環境セットアップ

```bash
# 開発用依存関係のインストール
pip install -e ".[dev]"

# テスト実行
pytest tests/ -v

# コードフォーマット
black .
ruff --fix .

# リントチェック
ruff check .
black --check .
```

### 9.2 テスト

#### 9.2.1 ユニットテスト

```bash
# 全テスト実行
pytest tests/ -v

# 特定のテスト実行
pytest tests/test_models.py::TestBundleFlow::test_velocity_shape -v

# カバレッジ付きテスト
pytest tests/ --cov=bundleflow --cov-report=html
```

#### 9.2.2 統合テスト

```bash
# 小規模統合テスト
python -c "
from bundleflow.models.flow import BundleFlow
from bundleflow.models.menu import MenuElement, Mechanism
from bundleflow.valuation.valuation import XORValuation
import torch

# 統合テスト
m = 5
flow = BundleFlow(m=m)
menu = [MenuElement(m=m, D=3) for _ in range(2)]
mechanism = Mechanism(flow, menu)

# 簡単な評価関数
atoms = [([1, 2], 1.0), ([3, 4], 2.0)]
valuation = XORValuation.from_bundle_list(m, atoms)

# 期待収入計算
revenue = mechanism.expected_revenue([valuation])
print(f'統合テスト成功: 期待収入 = {revenue.item():.4f}')
"
```

### 9.3 チェックポイント管理

#### 9.3.1 チェックポイントの保存と読み込み

```python
# Stage1チェックポイントの保存
torch.save({
    "model": flow.state_dict(),
    "m": args.m,
    "config": args
}, "checkpoints/flow_stage1_final.pt")

# Stage1チェックポイントの読み込み
checkpoint = torch.load("checkpoints/flow_stage1_final.pt")
flow = BundleFlow(m=checkpoint["m"])
flow.load_state_dict(checkpoint["model"])
```

#### 9.3.2 チェックポイントディレクトリ

```
checkpoints/
├─ flow_stage1_final.pt      # Stage1最終モデル
├─ flow_stage1_step10000.pt  # Stage1中間チェックポイント
├─ menu_stage2_final.pt      # Stage2最終モデル
└─ README.md                 # チェックポイント説明
```

### 9.4 パフォーマンス最適化

#### 9.4.1 GPU最適化

```python
# CUDA使用の確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# 混合精度学習（オプション）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 9.4.2 メモリ最適化

```python
# 勾配チェックポイント（オプション）
from torch.utils.checkpoint import checkpoint

# バッチサイズの調整
# GPU メモリに応じて batch_size を調整
batch_size = 1024 if torch.cuda.get_device_properties(0).total_memory > 8e9 else 512
```

### 9.5 デバッグとプロファイリング

#### 9.5.1 デバッグツール

```python
# 勾配の監視
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")

# 損失の監視
if iteration % 100 == 0:
    print(f"Iteration {iteration}: loss = {loss.item():.6f}")
```

#### 9.5.2 プロファイリング

```python
# PyTorch プロファイラー
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_forward"):
        output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 9.6 貢献ガイドライン

#### 9.6.1 コードスタイル

- **フォーマット**: Black を使用
- **リント**: Ruff を使用
- **型注釈**: 可能な限り型注釈を追加
- **ドキュメント**: 関数とクラスにdocstringを追加

#### 9.6.2 テスト要件

- **新機能**: 対応するテストを追加
- **バグ修正**: 回帰テストを追加
- **カバレッジ**: 新規コードのカバレッジを維持

#### 9.6.3 プルリクエスト

1. 機能ブランチを作成
2. テストを追加・実行
3. コードフォーマットを適用
4. プルリクエストを作成
5. レビューを受けてマージ

---

## 📚 参考文献

- Myerson, R. B. (1981). Optimal auction design. Mathematics of operations research, 6(1), 58-73.
- Cramton, P., Shoham, Y., & Steinberg, R. (2006). Combinatorial auctions. MIT press.
- Dütting, P., et al. (2019). Optimal auctions through deep learning. ICML.
- Rahme, J., et al. (2021). A differentiable economics approach to mechanism design. ICLR.

---

## 📄 ライセンス

MIT License - see [LICENSE](LICENSE) for details.

## 📝 引用

If you use this software, please cite it as described in [CITATION.cff](CITATION.cff).

---

**🎉 これで BundleFlow の完全な理解と使用が可能です！**
