"""
Stage1実装ロジック
既存のsrc/train_stage1.pyから分離
"""

import torch
import torch.optim as optim
from omegaconf import DictConfig
from bundleflow.flow import FlowModel
from bundleflow.data import load_cats_dir, train_test_split, gen_uniform_iid_xor
from bundleflow.utils import seed_all

def run(cfg: DictConfig, device: torch.device):
    """Stage1学習の実行"""
    print(f"[Stage1] Starting training with config: {cfg}")
    
    # データセット作成
    if cfg.get('cats_glob'):
        V = load_cats_dir(cfg.cats_glob, m=cfg.m, keep_dummy=None, 
                         max_files=cfg.get('max_files'), shuffle=True)
    else:
        V = [gen_uniform_iid_xor(cfg.m, a=cfg.get('a', 20), low=0.0, high=1.0, 
                                seed=cfg.seed + i, atom_size_mode="small") 
             for i in range(cfg.get('n_val', 5000))]
    
    train, test = train_test_split(V, train_ratio=0.95, seed=cfg.seed)
    
    # モデル作成
    flow = FlowModel(m=cfg.m, D=cfg.D, use_spectral_norm=False).to(device)
    opt = optim.Adam(flow.parameters(), lr=cfg.lr)
    
    # 学習ループ（簡略版）
    for it in range(1, cfg.iters + 1):
        # バッチ作成
        batch = train[:cfg.batch] if len(train) >= cfg.batch else train
        
        # 損失計算（簡略版）
        loss = torch.tensor(0.0, device=device)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if it % 100 == 0:
            print(f"[{it}/{cfg.iters}] loss={loss.item():.4f}")
    
    # チェックポイント保存
    torch.save({
        'model': flow.state_dict(),
        'm': cfg.m,
        'D': cfg.D,
    }, f"{cfg.out_dir}/flow_stage1_final.pt")
    
    print(f"Saved final: {cfg.out_dir}/flow_stage1_final.pt")
