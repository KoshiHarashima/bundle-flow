# bundleflow/train/stage1.py
"""
BundleFlow Stage 1: Flow Initialization
参照: Eq.(13)-(17), 丸めと近傍ノイズ Eq.(14), ODE前進 Eq.(20)（プローブ用）.
"""

import os, time, argparse, random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from bundleflow.models.flow import BundleFlow  # φ(t,s_t)=η(t)Q(s0)s_t, RectifiedFlow損失など実装済

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# ---------- utils ----------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_alpha0_mog(m: int, D: int, device, mu_range=(-0.2, 1.2), sigma_val=0.5):
    """
    初期分布 α0: Mixture-of-Gaussian (Eq.(13))
      s0 ~ Σ_d w_d N(μ_d, σ_d^2 I_m)
    ・μ_d ~ U[mu_range]^m, σ_d=sigma_val, w_d=1/D（固定: Stage 1はα0固定）
    """
    mus = torch.empty(D, m, device=device).uniform_(*mu_range)
    sigmas = torch.full((D,), sigma_val, device=device)
    weights = torch.full((D,), 1.0 / D, device=device)
    return mus, sigmas, weights

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine annealing学習率スケジューラ"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------- train ----------
def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seed_all(args.seed)

    # TensorBoard と CSV の準備
    writer = None
    if args.use_tensorboard and HAS_TENSORBOARD:
        log_dir = os.path.join(args.out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    
    csv_path = None
    if args.use_csv:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, "train_log.csv")
        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                'iteration', 'loss', 'ema_loss', 'lr', 'grad_norm', 
                'phi_norm', 'trace_q_mean', 'trace_q_std', 
                'eta_mean', 'eta_std', 's0_min', 's0_max', 's0_mean',
                'sT_min', 'sT_max', 'sT_mean', 'Q_norm', 'time'
            ])
        print(f"CSV logging to: {csv_path}")

    # φ(t,·)=η(t)Q(s0)·s_t（Eq.(9)）を学習
    flow = BundleFlow(
        m=args.m, 
        use_spectral_norm=args.use_spectral_norm,
        q_mode=args.q_mode,
        c_eta=args.c_eta,
        eta_init_scale=args.eta_init_scale,
        eta_temperature=args.eta_temperature,
        use_eta_layernorm=args.use_eta_layernorm,
        eta_integral_clip=args.eta_integral_clip
    ).to(device)
    opt = optim.Adam(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Setup: LR=5e-3
    
    # Cosine annealing スケジューラ
    scheduler = None
    if args.use_scheduler:
        warmup_steps = int(args.iters * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, args.iters, args.min_lr_ratio)
        print(f"Using cosine scheduler: warmup={warmup_steps}, min_lr_ratio={args.min_lr_ratio}")

    # 初期分布 α0 を固定（MoG; Eq.(13)）
    mus, sigmas, weights = init_alpha0_mog(
        args.m, args.D, device, 
        mu_range=(args.mu_min, args.mu_max), 
        sigma_val=args.mog_sigma
    )

    ema = None
    t0 = time.time()
    flow.train()
    for it in range(1, args.iters + 1):
        # Q mode切り替え（対角ウォームスタート → フルモード）
        if args.q_mode_switch_iter > 0 and it == args.q_mode_switch_iter:
            flow.Q.switch_to_full_mode()
        
        # Rectified Flow 損失（Eq.(15)-(17)）:
        #  s_T ~ N( s , σ_z^2 I ) with s = I(s0≥0.5)（Eq.(14), Sec.3.2 の近傍化）
        result = flow.rectified_flow_loss(
            B=args.batch, mus=mus, sigmas=sigmas, weights=weights, 
            sigma_z=args.sigma_z,
            trace_penalty_weight=args.trace_penalty,
            lambda_j=args.lambda_j,
            lambda_k=args.lambda_k,
            lambda_tr=args.lambda_tr,
            return_stats=True,
            use_ste=args.use_ste,
            ste_tau=args.ste_tau
        )
        loss, stats = result
        
        opt.zero_grad()
        loss.backward()
        
        # 勾配ノルムを計算（クリップ前）
        grad_norm = 0.0
        for p in flow.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if args.grad_clip is not None and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
        opt.step()
        
        if scheduler is not None:
            scheduler.step()

        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        current_lr = opt.param_groups[0]['lr']
        
        if it % args.log_every == 0:
            dt = time.time() - t0
            log_msg = (f"[{it}/{args.iters}] loss={loss.item():.6f} ema={ema:.6f} lr={current_lr:.6f} "
                      f"grad_norm={grad_norm:.4f} phi_norm={stats['phi_norm']:.4f} "
                      f"trace_q={stats['trace_q_mean']:.4f}±{stats['trace_q_std']:.4f} "
                      f"eta={stats['eta_mean']:.4f}±{stats['eta_std']:.4f}")
            
            # 正則化項の情報を追加（有効な場合のみ）
            if 'jacobian_penalty' in stats:
                log_msg += (f" J={stats['jacobian_penalty']:.4f} "
                           f"K={stats['kinetic_penalty']:.4f} "
                           f"Tr={stats['trace_penalty']:.4f}")
            
            log_msg += f" time={dt:.1f}s"
            print(log_msg, flush=True)
            
            # TensorBoard への記録
            if writer is not None:
                writer.add_scalar('loss/train', loss.item(), it)
                writer.add_scalar('loss/ema', ema, it)
                writer.add_scalar('lr', current_lr, it)
                writer.add_scalar('grad_norm', grad_norm, it)
                for key, val in stats.items():
                    writer.add_scalar(f'stats/{key}', val, it)
            
            # CSV への記録
            if csv_path is not None:
                with open(csv_path, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        it, loss.item(), ema, current_lr, grad_norm,
                        stats['phi_norm'], stats['trace_q_mean'], stats['trace_q_std'],
                        stats['eta_mean'], stats['eta_std'], 
                        stats['s0_min'], stats['s0_max'], stats['s0_mean'],
                        stats['sT_min'], stats['sT_max'], stats['sT_mean'],
                        stats['Q_norm'], dt
                    ])

        if args.ckpt_every > 0 and it % args.ckpt_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            path = os.path.join(args.out_dir, f"flow_stage1_step{it}.pt")
            torch.save({"model": flow.state_dict(), "m": args.m}, path)
            print(f"Saved checkpoint: {path}")

    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "flow_stage1_final.pt")
    torch.save({"model": flow.state_dict(), "m": args.m}, final_path)
    print(f"Saved final: {final_path}")
    
    if writer is not None:
        writer.close()

    # （任意）被覆率プローブ: 小mで φ 前進 (Eq.(20))→丸め（Eq.(19のI(·)相当）で支配的束のカバレッジを見る
    if args.probe > 0 and args.m <= 18:
        with torch.no_grad():
            s0 = flow.sample_mog(args.probe, mus, sigmas, weights)
            t_grid = torch.linspace(0.0, 1.0, steps=args.ode_steps, device=device)
            sT = flow.flow_forward(s0, t_grid)   # Eq.(20)
            s = flow.round_to_bundle(sT)         # 丸め I(·)
            # ビット被覆率
            uniq = set()
            for b in range(s.shape[0]):
                mask = 0
                for i, bit in enumerate((s[b] > 0.5).tolist()):
                    if bit:
                        mask |= (1 << i)
                uniq.add(mask)
            cov = len(uniq) / (2 ** args.m)
            print(f"Coverage probe: {len(uniq)}/{2**args.m} = {cov:.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=None)
    args = ap.parse_args()
    
    if args.cfg:
        # YAML設定を使用
        cfg = OmegaConf.load(args.cfg)
        # 既存の学習関数を呼び出し
        train_stage1_from_config(cfg)
    else:
        # 従来のCLI引数を使用
        train_stage1_cli()

def train_stage1_from_config(cfg):
    """YAML設定から学習を実行"""
    # 設定をargparse形式に変換
    import sys
    sys.argv = ['train_stage1.py']
    for key, value in cfg.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append(f'--{key}')
        else:
            sys.argv.extend([f'--{key}', str(value)])
    train_stage1_cli()

def train_stage1_cli():
    """従来のCLI引数で学習を実行"""
    ap = argparse.ArgumentParser()
    # 基本パラメータ
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--D", type=int, default=8)               # Setup既定の小さな支持サイズ
    ap.add_argument("--iters", type=int, default=60000)       # Fig.2: 60K iter のデモ
    ap.add_argument("--batch", type=int, default=1024)
    
    # 学習率制御
    ap.add_argument("--lr", type=float, default=5e-3)         # Setup
    ap.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing scheduler")
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for scheduler")
    ap.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR ratio for cosine decay")
    
    # ネットワーク安定化
    ap.add_argument("--use_spectral_norm", action="store_true", help="Apply spectral normalization to QNet")
    ap.add_argument("--q_mode", type=str, default="full", choices=["diag", "full"], 
                    help="Q matrix mode: 'diag' for diagonal warmstart, 'full' for full matrix")
    ap.add_argument("--q_mode_switch_iter", type=int, default=0, 
                    help="Iteration to switch from 'diag' to 'full' mode (0=no switch)")
    ap.add_argument("--c_eta", type=float, default=2.0, 
                    help="Scale factor for eta(t) range: [-c_eta, c_eta]. Increase if eta saturates (e.g., 5.0-10.0)")
    ap.add_argument("--eta_init_scale", type=float, default=0.01,
                    help="Initialization scale for EtaNet weights (smaller = less saturation)")
    ap.add_argument("--eta_temperature", type=float, default=1.0,
                    help="Temperature for tanh activation (5-10 to prevent saturation)")
    ap.add_argument("--use_eta_layernorm", action="store_true",
                    help="Use LayerNorm in EtaNet for stability")
    ap.add_argument("--eta_integral_clip", type=float, default=10.0,
                    help="Clip eta integral to prevent explosion (0 = no clip)")
    
    # 正則化
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    ap.add_argument("--trace_penalty", type=float, default=0.0, help="Tr(Q)^2 penalty weight (legacy)")
    ap.add_argument("--lambda_j", type=float, default=0.0, help="Jacobian penalty weight (軌道短縮)")
    ap.add_argument("--lambda_k", type=float, default=0.0, help="Kinetic penalty weight (ベクトル場制御)")
    ap.add_argument("--lambda_tr", type=float, default=0.0, help="Trace penalty weight (発散抑制)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    
    # 分布パラメータ
    ap.add_argument("--sigma_z", type=float, default=0.05)    # Eq.(14) の σ_z
    ap.add_argument("--mog_sigma", type=float, default=0.5, help="MoG component sigma")
    ap.add_argument("--mu_min", type=float, default=-0.2, help="MoG mu lower bound")
    ap.add_argument("--mu_max", type=float, default=1.2, help="MoG mu upper bound")
    
    # ログとチェックポイント
    ap.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    ap.add_argument("--use_csv", action="store_true", help="Enable CSV logging")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)
    
    # Straight-Through Estimator（STEによる離散化改善）
    ap.add_argument("--use_ste", action="store_true", help="Use STE for bundle rounding (improves gradients)")
    ap.add_argument("--ste_tau", type=float, default=0.1, help="STE sigmoid temperature (lower = sharper)")
    
    # その他
    ap.add_argument("--ode_steps", type=int, default=25)
    ap.add_argument("--probe", type=int, default=0)           # 小mでの被覆率チェック用
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--device", type=str, default="auto", help="Device to use: cuda, cpu, mps, or auto")
    
    args = ap.parse_args()
    train_stage1(args)

if __name__ == "__main__":
    main()
