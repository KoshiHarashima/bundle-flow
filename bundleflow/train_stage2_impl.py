"""
Stage2実装ロジック
既存のsrc/train_stage2.pyから分離
"""

import torch
import torch.optim as optim
from omegaconf import DictConfig
from bundleflow.flow import FlowModel
from bundleflow.menu import MenuElement, make_null_element, revenue_loss, utilities_matrix
from bundleflow.data import load_cats_dir, train_test_split, gen_uniform_iid_xor
from bundleflow.valuation import XORValuation, _tensor_to_mask
from bundleflow.utils import seed_all

def run(cfg: DictConfig, device: torch.device):
    """Stage2学習の実行"""
    print(f"[Stage2] Starting training with config: {cfg}")
    
    # FlowModel読み込み
    ckpt = torch.load(cfg.flow_ckpt, map_location=device)
    flow = FlowModel(m=cfg.m, D=cfg.D, use_spectral_norm=False).to(device)
    flow.load_state_dict(ckpt['model'])
    flow.eval()
    
    # データセット作成
    if cfg.get('cats_glob'):
        V = load_cats_dir(cfg.cats_glob, m=cfg.m, keep_dummy=None, 
                         max_files=cfg.get('max_files'), shuffle=True)
    else:
        V = [gen_uniform_iid_xor(cfg.m, a=cfg.get('a', 20), low=0.0, high=1.0, 
                                seed=cfg.seed + i, atom_size_mode="small") 
             for i in range(cfg.get('n_val', 5000))]
    
    train, test = train_test_split(V, train_ratio=0.95, seed=cfg.seed)
    
    # メニュー初期化
    menu = [MenuElement(cfg.m, D=cfg.D).to(device) for _ in range(cfg.K)]
    menu.append(make_null_element(cfg.m).to(device))
    
    # 時間グリッド
    t_grid = torch.linspace(0.0, 1.0, steps=cfg.ode_steps, device=device)
    
    # オプティマイザー
    mu_w_params = [elem.mus for elem in menu] + [elem.logits for elem in menu]
    beta_params = [elem.beta_raw for elem in menu]
    opt = optim.Adam(mu_w_params + beta_params, lr=cfg.lr)
    
    # 学習ループ（簡略版）
    for it in range(1, cfg.iters + 1):
        # バッチ作成
        batch = train[:cfg.batch] if len(train) >= cfg.batch else train
        
        # λスケジュール
        if it <= cfg.get('warmup_iters', 500):
            lam = cfg.lam_start * (it / cfg.get('warmup_iters', 500))
        else:
            progress = (it - cfg.get('warmup_iters', 500)) / (cfg.iters - cfg.get('warmup_iters', 500))
            lam = cfg.lam_start + (cfg.lam_end - cfg.lam_start) * progress
        
        # 損失計算
        loss = revenue_loss(flow, batch, menu, t_grid, lam=lam, 
                           verbose=(it % cfg.get('log_every', 200) == 0), 
                           debug=False)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if it % cfg.get('log_every', 200) == 0:
            print(f"[{it}/{cfg.iters}] LRev={loss.item():.4f} lam={lam:.4f}")
    
    # チェックポイント保存
    torch.save({
        'menu': [elem.state_dict() for elem in menu],
        'config': cfg,
    }, f"{cfg.get('out_dir', 'checkpoints')}/menu_stage2_final.pt")
    
    print(f"Saved final: {cfg.get('out_dir', 'checkpoints')}/menu_stage2_final.pt")
