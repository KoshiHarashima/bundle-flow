# bundle-flow
I'm an exchange student at Northwestern University.  

This code was created for experiments in the field of economics, especially in multi-product auction theory.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This work is based on the folloing paper;   

Wang, Tonghan, Yanchen Jiang, and David C. Parkes. 2025. “BundleFlow: Deep Menus for Combinatorial Auctions by Diffusion-Based Optimization.” arXiv [Cs.GT]. arXiv. http://arxiv.org/abs/2502.15283.

<<<<<<< HEAD
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
├── src/ 
│   ├── train_stage1
│   ├── train_stage2
│── README.md
├── requirements.txt         
└── .devcontainer         


First,  

pip install requirements.txt  

Second, run scripts.train_stage1  

python -m scripts.train_stage1 \
  --m 50 \
  --D 8 \
  --iters 60000 \
  --batch 1024 \
  --lr 5e-3 \
  --sigma_z 0.05 \
  --out_dir checkpoints


Third, run scripts.train_stage2  

python -m scripts.train_stage2 \
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


If you want to use CATS, run this!  

python -m scripts.train_stage2 \
  --flow_ckpt checkpoints/flow_stage1_final.pt \
  --cats_glob "cats_out/*.txt" \
  --m 50 \
  --K 512 \
  --D 8 \
  --iters 20000

```