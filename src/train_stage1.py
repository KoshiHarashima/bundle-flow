# scripts/train_stage1.py
# BundleFlow Stage 1: Flow Initialization
# 参照: Eq.(13)-(17), 丸めと近傍ノイズ Eq.(14), ODE前進 Eq.(20)（プローブ用）.  :contentReference[oaicite:2]{index=2}

import os, time, argparse, random
import torch
import torch.nn as nn
import torch.optim as optim

from bf.flow import FlowModel  # φ(t,s_t)=η(t)Q(s0)s_t, RectifiedFlow損失など実装済

# ---------- utils ----------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_alpha0_mog(m: int, D: int, device):
    """
    初期分布 α0: Mixture-of-Gaussian (Eq.(13))  :contentReference[oaicite:3]{index=3}
      s0 ~ Σ_d w_d N(μ_d, σ_d^2 I_m)
    ・μ_d ~ U[-0.2, 1.2]^m, σ_d=0.5, w_d=1/D（固定: Stage 1はα0固定）
    """
    mus = torch.empty(D, m, device=device).uniform_(-0.2, 1.2)
    sigmas = torch.full((D,), 0.5, device=device)
    weights = torch.full((D,), 1.0 / D, device=device)
    return mus, sigmas, weights

# ---------- train ----------
def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seed_all(args.seed)

    # φ(t,·)=η(t)Q(s0)·s_t（Eq.(9)）を学習  :contentReference[oaicite:4]{index=4}
    flow = FlowModel(m=args.m).to(device)
    opt = optim.Adam(flow.parameters(), lr=args.lr)  # Setup: LR=5e-3  :contentReference[oaicite:5]{index=5}

    # 初期分布 α0 を固定（MoG; Eq.(13)）  :contentReference[oaicite:6]{index=6}
    mus, sigmas, weights = init_alpha0_mog(args.m, args.D, device)

    ema = None
    t0 = time.time()
    flow.train()
    for it in range(1, args.iters + 1):
        # Rectified Flow 損失（Eq.(15)-(17)）:
        #  s_T ~ N( s , σ_z^2 I ) with s = I(s0≥0.5)（Eq.(14), Sec.3.2 の近傍化）  :contentReference[oaicite:7]{index=7}
        loss = flow.rectified_flow_loss(
            B=args.batch, mus=mus, sigmas=sigmas, weights=weights, sigma_z=args.sigma_z
        )
        opt.zero_grad()
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
        opt.step()

        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        if it % args.log_every == 0:
            dt = time.time() - t0
            print(f"[{it}/{args.iters}] loss={loss.item():.6f} ema={ema:.6f} time={dt:.1f}s", flush=True)

        if args.ckpt_every > 0 and it % args.ckpt_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            path = os.path.join(args.out_dir, f"flow_stage1_step{it}.pt")
            torch.save({"model": flow.state_dict(), "m": args.m}, path)
            print(f"Saved checkpoint: {path}")

    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "flow_stage1_final.pt")
    torch.save({"model": flow.state_dict(), "m": args.m}, final_path)
    print(f"Saved final: {final_path}")

    # （任意）被覆率プローブ: 小mで φ 前進 (Eq.(20))→丸め（Eq.(19のI(·)相当）で支配的束のカバレッジを見る  :contentReference[oaicite:8]{index=8}
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--D", type=int, default=8)               # Setup既定の小さな支持サイズ  :contentReference[oaicite:9]{index=9}
    ap.add_argument("--iters", type=int, default=60000)       # Fig.2: 60K iter のデモ  :contentReference[oaicite:10]{index=10}
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-3)         # Setup  :contentReference[oaicite:11]{index=11}
    ap.add_argument("--sigma_z", type=float, default=0.05)    # Eq.(14) の σ_z
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ode_steps", type=int, default=25)
    ap.add_argument("--probe", type=int, default=0)           # 小mでの被覆率チェック用
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train_stage1(args)
