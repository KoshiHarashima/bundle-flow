# bundle-flow
I'm an exchange student at Northwestern University.  

This code was created for experiments in the field of economics, especially in multi-product auction theory.

## 🆕 Recent Improvements (2025-10-10)

**Stage1 の学習安定化と詳細ログ機能を追加しました！**

- ✅ ネットワーク初期化とスケール制御の改善
- ✅ Cosine annealing 学習率スケジューラ
- ✅ 正則化（Weight decay + Tr(Q)² 罰則）
- ✅ 詳細な統計ログ（||φ||₂, Tr(Q), η, 勾配ノルムなど）
- ✅ TensorBoard / CSV 出力対応

**詳細はこちら**:
- [IMPROVEMENTS_SUMMARY.md](./IMPROVEMENTS_SUMMARY.md) - 改善内容の概要
- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) - 詳細な使用ガイド
- [test_improvements.py](./test_improvements.py) - 動作確認テスト

**クイックスタート**（推奨設定）:
```bash
python -m src.train_stage1 \
  --use_scheduler --use_csv --use_tensorboard \
  --lr 1e-3 --trace_penalty 1e-4 --weight_decay 1e-5 \
  --sigma_z 0.03 --mu_min 0.0 --mu_max 1.0 \
  --log_every 50 --out_dir checkpoints_stable
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This work is based on the folloing paper;   

Wang, Tonghan, Yanchen Jiang, and David C. Parkes. 2025. “BundleFlow: Deep Menus for Combinatorial Auctions by Diffusion-Based Optimization.” arXiv [Cs.GT]. arXiv. http://arxiv.org/abs/2502.15283.

Dütting, Paul, Zhe Feng, H. Narasimhan, and D. Parkes. 2017. “Optimal Auctions through Deep Learning.” International Conference on Machine Learning 64 (June): 109–16.

Jang, Eric, Shixiang Gu, and Ben Poole. 2016. “Categorical Reparameterization with Gumbel-Softmax.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1611.01144.

=======

```
bundle-flow/
├── data/ 
├── bf/ 
│   ├── data.py
│   ├── flow.py
│   ├── menu.py
│   ├── utils.py
│   ├── valuation.py
├── brn/                        # ← 追加: Bundle-RochetNet 用
│   ├── menu_itemwise.py        # RochetNet風メニュー（α∈[0,1]^m, β）
│   ├── gumbel.py               # Gumbel-Softmax/ST サンプラ
│   ├── train_brn.py            # 学習スクリプト（1段学習）
│   ├── utils.py                
├── src/ 
│   ├── train_stage1
│   ├── train_stage2
│── README.md
├── requirements.txt         
└── .devcontainer         


Below are example code, simulating an economic environment with 50 items, each represented by an 8-dimensional feature vector (\(D=8\)). Agents iteratively update their strategies for 60{,}000 iterations (\(\text{iters}=60000\)) to approximate a Nash equilibrium. Each iteration uses a batch of 1,024 samples, with learning rate \(5\times10^{-3}\) and noise scale \(\sigma_z=0.05\). This setup corresponds to a multi-good auction or market environment in which agents learn optimal bidding or pricing strategies through repeated interaction.

First,  

pip install -r requirements.txt

Second, run src.train_stage1. 

python -m src.train_stage1 \
  --m 50 \
  --D 8 \
  --iters 60000 \
  --batch 1024 \
  --lr 5e-3 \
  --sigma_z 0.05 \
  --out_dir checkpoints

python -m src.train_stage1 --m 50 --D 8 --iters 60000 --batch 1024 --lr 5e-3 --sigma_z 0.05 --out_dir checkpoints

Third, run src.train_stage2.  

python -m src.train_stage2 \
  --flow_ckpt checkpoints/flow_stage1_final.pt \
  --m 50 \
  --K 512 \
  --D 8 \
  --iters 20000 \
  --batch 128 \
  --lr 0.3 \
  --a 20 \
  --n_val 5000 \
  --out_dir checkpoints


python -m src.train_stage2  --flow_ckpt checkpoints/flow_stage1_final.pt  --m 50  --K 512  --D 8  --iters 20000  --batch 128  --lr 0.3  --a 20  --n_val 5000  --out_dir checkpoints

If you want to use CATS, run this!  

python -m scripts.train_stage2 \
  --flow_ckpt checkpoints/flow_stage1_final.pt \
  --cats_glob "cats_out/*.txt" \
  --m 50 \
  --K 512 \
  --D 8 \
  --iters 20000

```