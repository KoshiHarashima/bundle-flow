# BundleFlow Critical Issues — 解決方針（論文根拠つき）

## 0) 状況の"意味"を整理

### 観察された異常パターンの解釈

**平均束サイズ=0.0±0.0 & 多様性≈0** は、`s_T ∈ R^m` が全次元で 0.5 未満になって `round_to_bundle(s_T>=0.5)` が全0を返していることを示唆。

**RF/CFM の観点では**、`φ(t,s)=η(t)Q(s₀)s` を積分しても `∫₀ᵀ η(t) dt ≈ 0`、あるいは `Q ≈ 0` なら s は動かず、初期 `s₀` が 0 近傍なら最終も 0 に張り付くのは自然。

**RFの目的（式(6)）**は直線経路に合わせるMSE回帰、Flow 表現（式(7)）と Liouville（式(12)）から、密度重みは `exp{−TrQ⋅∫η}` で効く。∴ η と Tr(Q) が過小だと束は動かない。

**「前半◯→後半×」の収益崩壊**は、一致率=0%（全ユーティリティが −β）→SoftMaxが均等→∂Rev/∂β>0で β が上がり続ける → 学習信号がさらに失われる、という 式(22)(23) の副作用。

## 1) Bundle Generation Failure（平均束サイズ=0）を止める

### 1-1. Flow の"単体"テスト（RF/CFMとして成立しているか）

**目的**：ODE が"前に進む"ことを、直線経路回帰（式(6)(15–17)）として白黒判定。

**テスト（10行）**：
```python
# sanity_flow.py
import torch
from bf.flow import FlowModel
T, m = 1.0, 10
flow = FlowModel(m).eval()
t_grid = torch.linspace(0, T, 21)
# μを3点: all-zero / 0.5 / U[-0.2,1.2]
mus = torch.stack([torch.zeros(m), torch.full((m,),0.5), torch.rand(m)*1.4-0.2])
with torch.no_grad():
    sT = flow.flow_forward(mus, t_grid)    # (*,m)
    s  = (sT >= 0.5)
print("bundle sizes:", s.sum(dim=1).tolist())   # 例: [0, ~m/2, ~m/2] を期待
```

**期待**：少なくとも mu=0.5／U[-0.2,1.2] のケースで 0以外の束サイズ。

**ダメなら**：∫η や Q 出力がゼロ付近。η の 温度tanh・Q の スペクトル正規化を緩める/外す（やり過ぎの正則化で"動かない流れ"を作っていないか）。（RFは直線回帰なので、まず "動く" のが正しい）

### 1-2. 数値の初期化と丸めの問題を切る

**μ=0 病**：MenuElement.mus を全 0 で始めるのは最悪。mu ∈ [−0.2,1.2] の小ノイズ、もしくは代表束（全1 / 少数1）近傍でランダム初期化。

**丸め**：(sT >= 0.5) は仕様に合うが、sT の平均が0.5付近だと少数誤差で全0化しやすい。t_grid を float64 に、sT も float64 で丸め直し→閾値越えがあるかをログ。

**ODE**：flow_forward のEuler積分で s = s + dt * φ(t,s,s0) と逐次更新しているか、dt・t の型/デバイスが一致しているかをチェック。B×m vs (B,) のブロードキャスト不整合で φ=0 になる事故も疑う。

### 1-3. 解析的な健全性チェック（線形系の簡易解）

本モデルの φ は `φ(t,s)=η(t)Q(s₀)s`。Q が s₀ の関数だが t に依存しないので、

`s(T) ≈ exp((∫₀ᵀ η(t)dt)Q(s₀)) s(0)`

の行列指数近似が**単調に"動く"**ことの目安になります（大域最適でなくてよい）。

`torch.linalg.matrix_exp( integ_eta * Q(s0) ) @ s0` を1ステップの比較値として出し、Euler解と方向一致を見る。

方向すら一致しないなら Q/η の符号やスケールが正則化で潰れている。λ_J/λ_K/λ_tr を一旦 0 に近づけ、再学習で「まず動く」を確認。
（根拠：Rectified Flow の"直線"はx₁−x₀の方向合わせで、強い正則化は"動かない"を好むことがある）

## 2) Abnormal Learning Dynamics（末期の収益崩壊）を止める

### 2-1. Stage 2 を "RochetNet の数値安定版" に揃える

**β≥0**：beta = F.softplus(beta_raw)（IRと一致、β暴走を強制的に止める）。

**log-sum-exp**：式(21)の重み `exp{−TrQ⋅∫η}` は log-sum-exp 実装に変更（float64 & M = max(log_w) シフト）。

**SoftMax温度 λ**：0.001→0.2 に緩やか上げ（式(23)）。最初は均等探索、後半で決定化。

**ヌル要素**（ゼロ配分・ゼロ価格）は必須（IRの安全弁）。

**最小差分（概略）**：
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

### 2-2. "一致率=0%"を可視化して再初期化

match_rate＝(atom_mask & ~bundle_mask)==0 の割合を毎 iter ログ。

連続 N iter で 0% なら、μ を再初期化（代表束近傍へ）／D を2以上に（Table 2 の退化回避）。

合成XORの原子サイズがm/2だと一致確率は `0.5^25` レベルで実質0。平均5–8の"現実的サイズ"に変えるだけで学習信号が出る。

### 2-3. 最低限の訓練安定トリック

**Optim**：AdamW, lr を2–5×小さく、grad_clip=0.5。

学習前半は β を凍結（1–2k iter、μ,w だけ更新）→一致が出てから β を解放。

**監視**：rev(hard-argmax), beta_med, z_entropy（要素の利用分散）。

## 3) コード断面（優先度の高い見直し点）

### 3-1. ODE / 丸め

**flow_forward**：`for i in range(len(t_grid)-1): dt = t[i+1]-t[i]; s = s + dt*phi(t[i], s, s0)` になっているか、s0 と s の混線がないか、t_grid dtype/device が一致しているか。

**round_to_bundle**：float64 で (sT>=0.5)、sum(sT>=0.5) が 0 ばかりかをログ。

### 3-2. Menu 初期化/更新

**mus**：ゼロ初期化を止める。U[-0.2,1.2] 小ノイズ、または代表束（全1/小k）近傍。

**weights(logits)**：logit=0 は均等でOK。ただし学習が進んでいるか（torch.allclose(logits.grad,0)で死んでないか）。

**beta**：softplus＋ウォームアップ凍結。

### 3-3. 収益損失

数式は (21)(22)(23) に忠実に。指数は log-sum-exp、SoftMax は λ スケジュール。

**device/shape**：U(B,K), Z(B,K), beta(K) のブロードキャストが意図通りか、dtype=float64で一度通す。

## 4) テストとデバッグの粒度

### 4-1. ユニット

**flow_smoke**：mu in {0, 0.5, U[-0.2,1.2]} で束サイズ>0 を確認。

**menu_value**：固定 vals=[1,2], log_w=[0,0] に対してlog-sum-exp の値が理論通り。

**round_threshold**：sT=0.4999/0.5000/0.5001 で丸め境界の挙動を検証。

### 4-2. 統合

match_rate と rev(hard/soft) が 0→上昇→安定の形に乗るか（小規模 m=10, K=64, D=2, iters=2k）。

βウォームアップ有無の A/B 試験。

## 5) 文献 ⇔ 方針の対応表

| 問題 | 方針 | 根拠 |
|------|------|------|
| 平均束サイズ=0 | η と Q が過小 → RF/CFM の回帰が"動かない"学習になっている。η温度tanh、正則化の緩和、行列指数の単体検証で"まず動く"。 | RFの式(6) は直線経路回帰、Flow表現(7)、密度(12) |
| 末期の収益崩壊 | 一致率=0%→SoftMax均等→β暴走。log-sum-exp、β≥0、ウォームアップ、合成XOR原子サイズの現実化。 | RochetNet型のStage2 (19–23)、IRはヌル要素で担保 |
| 数値不安定 | float64・log-sum-exp・grad clip・AdamW、device/shape 整合。 | 生成流/拡散での安定実装の常識、式(12)(21)の指数安定化 |
| メニュー学習が進まない | μ のゼロ初期化禁止、代表束近傍/小ノイズ、D≥2。 | Table 2 の D=1 退化、Fig.3 の挙動 |
| データ側の無信号 | 合成XORの原子サイズ（期待5–8）に変更、CATS/Regions等に寄せる。 | CATS系の慣行・JELサーベイの実務文脈 |

---

**Note**: この解決方針は論文の理論的根拠に基づいており、実装の修正により期待される改善が得られるはずです。
