# scripts/train_stage2.py
# BundleFlow Stage 2: Menu Optimization
# 参照: 期待効用の等価表現 Eq.(19), ODE解 Eq.(20), 厳密効用の有限支持版 Eq.(21),
#       収益最大化損失 Eq.(22), SoftMax 割当 Eq.(23) 。:contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

import os, time, argparse, random
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim

from bf.flow import FlowModel
from bf.menu import MenuElement, make_null_element, revenue_loss, utilities_matrix
from bf.data import load_cats_dir, train_test_split, gen_uniform_iid_xor
from bf.valuation import XORValuation

# ---------- utils（最小限） ----------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def lambda_schedule(step: int, total: int, start: float = 1e-3, end: float = 0.2) -> float:
    # SoftMax温度 λ を線形で Eq.(23) に投入（Setup参照）。:contentReference[oaicite:4]{index=4}
    alpha = min(max(step / max(total, 1), 0.0), 1.0)
    return (1 - alpha) * start + alpha * end

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()

# ---------- menu 構築 ----------
def build_menu(m: int, K: int, D: int) -> List[MenuElement]:
    menu = [MenuElement(m, D) for _ in range(K)]
    # IR用ヌル要素（ゼロ配分・ゼロ価格）
    menu.append(make_null_element(m))
    return menu

# ---------- データ ----------
def make_dataset(args) -> List[XORValuation]:
    if args.cats_glob:
        V = load_cats_dir(args.cats_glob, m=args.m, keep_dummy=None, max_files=args.max_files, shuffle=True)
    else:
        # 合成XOR（Table 4 の原子数 a を模倣）を N 本生成。:contentReference[oaicite:5]{index=5}
        V = [gen_uniform_iid_xor(args.m, a=args.a, low=0.0, high=1.0, seed=1337 + i) for i in range(args.n_val)]
    return V

# ---------- 学習本体（Eq.(21)→(22) 最適化） ----------
def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seed_all(args.seed)

    # Stage 1 のフロー読込（φ 固定；Eq.(20)で使用） :contentReference[oaicite:6]{index=6}
    ckpt = torch.load(args.flow_ckpt, map_location=device)
    flow = FlowModel(m=ckpt["m"]).to(device)
    flow.load_state_dict(ckpt["model"])
    freeze_module(flow)  # φ を固定（Stage 2）

    # メニュー初期化（各要素は価格 β と Dirac混合 α0；Eq.(21)） :contentReference[oaicite:7]{index=7}
    menu = build_menu(args.m, args.K, args.D)
    params = []
    for elem in menu:
        for p in elem.parameters():
            if p.requires_grad:
                params.append(p)
    opt = optim.Adam(params, lr=args.lr)  # Setup: lr≈0.3 推奨。:contentReference[oaicite:8]{index=8}

    # 価値関数データ（V を train/test に分割） :contentReference[oaicite:9]{index=9}
    V_all = make_dataset(args)
    train, test = train_test_split(V_all, train_ratio=0.95, seed=args.seed)

    # 時間グリッド（Eq.(12),(20) の離散化）
    t_grid = torch.linspace(0.0, 1.0, steps=args.ode_steps, device=device)

    # ループ
    ema = None
    t0 = time.time()
    for it in range(1, args.iters + 1):
        # バッチをサンプル
        B = min(args.batch, len(train))
        batch = random.sample(train, B)

        # λ（Eq.(23)）をスケジュール
        lam = lambda_schedule(it, args.iters, start=args.lam_start, end=args.lam_end)

        # 収益損失（Eq.(22)）を最小化。内部で効用 u^(k)(v) を Eq.(21) で厳密計算。
        loss = revenue_loss(flow, batch, menu, t_grid, lam=lam)

        opt.zero_grad()
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        if it % args.log_every == 0:
            dt = time.time() - t0
            print(f"[{it}/{args.iters}] LRev={loss.item():.6f} ema={ema:.6f} lam={lam:.4f} time={dt:.1f}s", flush=True)

        # 途中保存
        if args.ckpt_every > 0 and it % args.ckpt_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "menu": [ (e.beta.detach().cpu(), e.logits.detach().cpu(), e.mus.detach().cpu()) for e in menu ],
                "m": args.m, "K": args.K, "D": args.D,
                "flow_ckpt": args.flow_ckpt
            }, os.path.join(args.out_dir, f"menu_stage2_step{it}.pt"))

    # 最終保存
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "menu_stage2_final.pt")
    torch.save({
        "menu": [ (e.beta.detach().cpu(), e.logits.detach().cpu(), e.mus.detach().cpu()) for e in menu ],
        "m": args.m, "K": args.K, "D": args.D,
        "flow_ckpt": args.flow_ckpt
    }, final_path)
    print(f"Saved final: {final_path}")

    # （任意）テスト収益（ハードargmax；本文での推論手順） :contentReference[oaicite:10]{index=10}
    rev = eval_hard_revenue(flow, test[:args.eval_n], menu, t_grid)
    print(f"[TEST] hard-argmax revenue on {min(args.eval_n, len(test))} vals = {rev:.4f}")

# ---------- 評価（テスト時はハードargmax；Sec.3.3） ----------
@torch.no_grad()
def eval_hard_revenue(flow: FlowModel, V: List[XORValuation], menu: List[MenuElement], t_grid: torch.Tensor) -> float:
    device = t_grid.device
    total = 0.0
    for v in V:
        U = utilities_matrix(flow, [v], menu, t_grid)[0]   # (K+1,)
        k = int(torch.argmax(U).item())
        u_star = float(U[k].item())
        beta_k = float(menu[k].beta.item())
        total += beta_k if u_star >= 0.0 else 0.0
    return total / max(1, len(V))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", type=str, default="checkpoints/flow_stage1_final.pt")
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--K", type=int, default=512)      # 実験既定は m≤100で5k/それ以上で20k。:contentReference[oaicite:11]{index=11}
    ap.add_argument("--D", type=int, default=8)        # 有限支持サイズ（Dirac混合の個数）。:contentReference[oaicite:12]{index=12}
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-1)  # Setup: 0.3 を推奨。:contentReference[oaicite:13]{index=13}
    ap.add_argument("--lam_start", type=float, default=1e-3)
    ap.add_argument("--lam_end", type=float, default=2e-1)
    ap.add_argument("--ode_steps", type=int, default=25)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    # データ（CATS or 合成）
    ap.add_argument("--cats_glob", type=str, default="")  # 例: "cats_out/*.txt"
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--n_val", type=int, default=5000)    # 合成の本数
    ap.add_argument("--a", type=int, default=20)          # 合成XORの原子数（Table 4 を模倣）:contentReference[oaicite:14]{index=14}
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    train_stage2(args)
