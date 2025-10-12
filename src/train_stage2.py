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

# ---------- μのウォームスタート ----------
@torch.no_grad()
def warmstart_mus(flow: FlowModel, menu: List[MenuElement], t_grid: torch.Tensor, 
                  n_grid: int = 100, seed: int = 42):
    """
    フローで代表束を生成し、μを初期化（ウォームスタート）
    効果：異なる束領域にμを散らし、初期収束を加速
    """
    device = t_grid.device
    m = flow.m
    torch.manual_seed(seed)
    
    # ランダムなμグリッドを生成（Stage1と同じ範囲: [-0.2, 1.2]）
    mu_grid = torch.rand(n_grid, m, device=device) * 1.4 - 0.2  # [0,1] → [-0.2, 1.2]
    
    # flow_forward → round して代表束を得る
    sT_grid = flow.flow_forward(mu_grid, t_grid)  # (n_grid, m)
    bundles = flow.round_to_bundle(sT_grid)       # (n_grid, m)
    
    # 重複を削除（ユニークな束のみ）
    bundles_unique = torch.unique(bundles, dim=0)
    print(f"[WarmStart] Generated {len(bundles_unique)} unique bundles from {n_grid} samples")
    
    # 各メニュー要素のμを代表束から初期化
    for k, elem in enumerate(menu[:-1]):  # 最後のnull要素を除く
        D = elem.mus.shape[0]
        # ランダムに代表束を選択してμとして配置
        if len(bundles_unique) >= D:
            idx = torch.randperm(len(bundles_unique))[:D]
            elem.mus.data.copy_(bundles_unique[idx])
        else:
            # 代表束が足りない場合は繰り返し使用
            idx = torch.randint(0, len(bundles_unique), (D,))
            elem.mus.data.copy_(bundles_unique[idx])
        
        # 小さなノイズを追加（探索のため）
        elem.mus.data += 0.05 * torch.randn_like(elem.mus.data)
        elem.mus.data.clamp_(-0.2, 1.2)  # Stage1と同じ範囲
    
    print(f"[WarmStart] Initialized μ for {len(menu)-1} menu elements")

# ---------- 弱い再初期化 ----------
@torch.no_grad()
def reinit_unused_elements(flow: FlowModel, menu: List[MenuElement], Z: torch.Tensor,
                           t_grid: torch.Tensor, threshold: float = 0.01):
    """
    選択確率が低い要素のμを再初期化（探索の質を維持）
    Z: (B, K+1) の選択確率行列
    threshold: この値未満の平均選択確率を持つ要素を再初期化
    """
    device = t_grid.device
    m = flow.m
    
    # 各要素の平均選択確率を計算
    z_mean = Z[:, :-1].mean(dim=0)  # (K,) 最後のnull要素を除く
    
    # 閾値未満の要素を特定
    unused_mask = z_mean < threshold
    n_unused = unused_mask.sum().item()
    
    if n_unused > 0:
        print(f"[ReInit] Found {n_unused} unused elements (z_mean < {threshold}), reinitializing...")
        
        # 代表束を生成（Stage1と同じ範囲）
        mu_grid = torch.rand(50, m, device=device) * 1.4 - 0.2  # [-0.2, 1.2]
        sT_grid = flow.flow_forward(mu_grid, t_grid)
        bundles = flow.round_to_bundle(sT_grid)
        bundles_unique = torch.unique(bundles, dim=0)
        
        # 未使用要素のμを再初期化
        for k, elem in enumerate(menu[:-1]):
            if unused_mask[k]:
                D = elem.mus.shape[0]
                if len(bundles_unique) >= D:
                    idx = torch.randperm(len(bundles_unique))[:D]
                    elem.mus.data.copy_(bundles_unique[idx])
                else:
                    idx = torch.randint(0, len(bundles_unique), (D,))
                    elem.mus.data.copy_(bundles_unique[idx])
                
                # ノイズ追加
                elem.mus.data += 0.05 * torch.randn_like(elem.mus.data)
                elem.mus.data.clamp_(-0.2, 1.2)  # Stage1と同じ範囲
                
                # 価格βも軽く再初期化
                elem.beta.data.fill_(0.1 * torch.randn(1).item())
    
    return n_unused

# ---------- データ ----------
def make_dataset(args) -> List[XORValuation]:
    if args.cats_glob:
        V = load_cats_dir(args.cats_glob, m=args.m, keep_dummy=None, max_files=args.max_files, shuffle=True)
    else:
        # 合成XOR（Table 4 の原子数 a を模倣）を N 本生成。:contentReference[oaicite:5]{index=5}
        V = [gen_uniform_iid_xor(args.m, a=args.a, low=0.0, high=1.0, seed=1337 + i, 
                                 atom_size_mode=args.atom_size_mode) for i in range(args.n_val)]
    return V

# ---------- 学習本体（Eq.(21)→(22) 最適化） ----------
def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Stage2] Using device: {device}", flush=True)
    seed_all(args.seed)

    # Stage 1 のフロー読込（φ 固定；Eq.(20)で使用） :contentReference[oaicite:6]{index=6}
    print(f"[Stage2] Loading checkpoint: {args.flow_ckpt}", flush=True)
    if os.path.exists(args.flow_ckpt):
        file_size = os.path.getsize(args.flow_ckpt) / (1024 * 1024)  # MB
        print(f"[Stage2] Checkpoint file size: {file_size:.2f} MB", flush=True)
    else:
        print(f"[Stage2] WARNING: Checkpoint file not found: {args.flow_ckpt}", flush=True)
    
    print(f"[Stage2] Starting torch.load()...", flush=True)
    ckpt = torch.load(args.flow_ckpt, map_location=device)
    print(f"[Stage2] Checkpoint loaded successfully", flush=True)
    print(f"[Stage2] Checkpoint keys: {list(ckpt.keys())}", flush=True)
    
    # スペクトル正規化の有無を判定
    print(f"[Stage2] Checking for spectral normalization...", flush=True)
    use_spectral_norm = any('weight_orig' in k for k in ckpt["model"].keys())
    print(f"[Stage2] use_spectral_norm = {use_spectral_norm}", flush=True)
    
    # チェックポイントと同じ設定でモデル作成
    print(f"[Stage2] Creating FlowModel with m={ckpt['m']}, use_spectral_norm={use_spectral_norm}", flush=True)
    flow = FlowModel(m=ckpt["m"], use_spectral_norm=use_spectral_norm).to(device)
    print(f"[Stage2] Loading model state_dict...", flush=True)
    flow.load_state_dict(ckpt["model"])
    print(f"[Stage2] Freezing FlowModel parameters...", flush=True)
    freeze_module(flow)  # φ を固定（Stage 2）
    print(f"[Stage2] FlowModel ready (frozen)", flush=True)

    # メニュー初期化（各要素は価格 β と Dirac混合 α0；Eq.(21)） :contentReference[oaicite:7]{index=7}
    # チェックポイントから再開する場合
    start_iter = 1
    loaded_ema = None
    if args.resume_ckpt:
        print(f"[Stage2] Resuming from checkpoint: {args.resume_ckpt}", flush=True)
        resume_data = torch.load(args.resume_ckpt, map_location=device)
        start_iter = resume_data.get("iteration", 0) + 1
        loaded_ema = resume_data.get("ema", None)
        print(f"[Stage2] Resuming from iteration {start_iter}, ema={loaded_ema}", flush=True)
    
    menu = build_menu(args.m, args.K, args.D)
    # メニュー要素をdeviceに移動
    for elem in menu:
        elem.to(device)
    
    # チェックポイントからパラメータを復元
    if args.resume_ckpt:
        for i, elem in enumerate(menu):
            if i < len(resume_data["menu"]):
                beta_raw, logits, mus = resume_data["menu"][i]
                elem.beta_raw.data.copy_(beta_raw.to(device))
                elem.logits.data.copy_(logits.to(device))
                elem.mus.data.copy_(mus.to(device))
        print(f"[Stage2] Restored menu parameters from checkpoint", flush=True)
    
    # パラメータをμ/w とβに分離
    mu_w_params = []
    beta_params = []
    for elem in menu:
        for name, p in elem.named_parameters():
            if p.requires_grad:
                if 'beta' in name:
                    beta_params.append(p)
                else:  # mus, logits
                    mu_w_params.append(p)
    
    # 最初はμ/wのみ学習（βは凍結）
    opt = optim.Adam(mu_w_params, lr=args.lr)  # Setup: lr≈0.3 推奨。:contentReference[oaicite:8]{index=8}
    
    print(f"[Stage2] Warmup phase: Optimizing {len(mu_w_params)} μ/w parameters, {len(beta_params)} β parameters frozen", flush=True)

    # 価値関数データ（V を train/test に分割） :contentReference[oaicite:9]{index=9}
    V_all = make_dataset(args)
    
    # デバッグ：最初のvaluationを詳しく確認
    if len(V_all) > 0:
        print(f"[Stage2] Dataset check: {len(V_all)} valuations created", flush=True)
        v0 = V_all[0]
        print(f"[Stage2] First valuation: v0.m = {v0.m}", flush=True)
        print(f"[Stage2] First valuation has {len(v0.atoms)} atoms", flush=True)
        if len(v0.atoms) > 0:
            atom0_mask, atom0_price = v0.atoms[0]
            atom0_bits = bin(atom0_mask).count('1')
            atom0_highest = atom0_mask.bit_length() - 1
            print(f"[Stage2] First atom: {atom0_bits} items, price={atom0_price:.4f}, mask={atom0_mask}, highest_bit={atom0_highest}", flush=True)
            # テスト：全てのアイテムを持つbundleの価値
            all_items_bundle = torch.ones(args.m, device=device)
            test_val = v0.value(all_items_bundle)
            print(f"[Stage2] Value of bundle with ALL {args.m} items: {test_val:.4f}", flush=True)
            if test_val == 0.0:
                print(f"[Stage2] WARNING: Even with all items, value is 0! This suggests a bug.", flush=True)
            
            # さらに：v0.m とargs.mの比較
            if hasattr(v0, 'm') and v0.m != args.m:
                print(f"[Stage2] ERROR: Valuation.m ({v0.m}) != args.m ({args.m})!", flush=True)
    
    train, test = train_test_split(V_all, train_ratio=0.95, seed=args.seed)
    
    # デバッグ用：v0を保存（起動時テストで成功したvaluation）
    v0_for_debug = V_all[0]

    # 時間グリッド（Eq.(12),(20) の離散化）
    t_grid = torch.linspace(0.0, 1.0, steps=args.ode_steps, device=device)
    
    # μのウォームスタート（代表束から初期化）
    if args.warmstart:
        warmstart_mus(flow, menu, t_grid, n_grid=args.warmstart_grid, seed=args.seed)

    # ループ
    ema = loaded_ema if loaded_ema is not None else None
    t0 = time.time()
    beta_unfrozen = False  # βの凍結状態を管理
    
    # チェックポイントから再開する場合、凍結状態も復元
    if args.resume_ckpt and start_iter > args.freeze_beta_iters:
        beta_unfrozen = True
        all_params = mu_w_params + beta_params
        opt = optim.Adam(all_params, lr=args.lr)
        if "optimizer_state" in resume_data:
            opt.load_state_dict(resume_data["optimizer_state"])
        print(f"[Stage2] Resumed with β unfrozen", flush=True)
    
    for it in range(start_iter, args.iters + 1):
        # βの凍結解除（warmup期間後）
        if not beta_unfrozen and it > args.freeze_beta_iters:
            print(f"\n[Iteration {it}] Unfreezing β parameters...", flush=True)
            all_params = mu_w_params + beta_params
            opt = optim.Adam(all_params, lr=args.lr)
            beta_unfrozen = True
            print(f"[Iteration {it}] Now optimizing {len(all_params)} parameters (μ/w/β)", flush=True)
        # バッチをサンプル
        B = min(args.batch, len(train))
        batch = random.sample(train, B)

        # λ（Eq.(23)）をスケジュール
        lam = lambda_schedule(it, args.iters, start=args.lam_start, end=args.lam_end)
        
        # Gumbel-Softmax温度をアニーリング（use_gumbelの時のみ使用）
        if args.use_gumbel:
            tau = args.tau_start * (args.tau_end / args.tau_start) ** ((it - 1) / args.iters)
        else:
            tau = 0.1  # ダミー（使用されない）

        # 収益損失（Eq.(22)）を最小化。内部で効用 u^(k)(v) を Eq.(21) で厳密計算。
        # ログ出力時にデバッグ情報を表示
        is_log_iter = (it % args.log_every == 0)
        verbose = (it == 1)
        if verbose:
            method_str = "Gumbel-Softmax+STE" if args.use_gumbel else "Softmax"
            print(f"\n[Iteration {it}] Starting forward pass ({method_str}, tau={tau:.4f})...", flush=True)
        
        # メニュー可視化（最初のiterationと定期的に）
        if it == 1 or (is_log_iter and it % (args.log_every * 5) == 0):
            from bf.menu import visualize_menu
            visualize_menu(flow, menu, t_grid, max_items=8, device=device)
        loss = revenue_loss(flow, batch, menu, t_grid, lam=lam, verbose=verbose, debug=is_log_iter, 
                           v0_test=v0_for_debug, use_gumbel=args.use_gumbel, tau=tau)

        opt.zero_grad()
        loss.backward()
        if verbose:
            print(f"  Backward pass complete. Clipping gradients and updating parameters...", flush=True)
        if args.grad_clip and args.grad_clip > 0:
            # 現在最適化中のパラメータに対してclip
            current_params = opt.param_groups[0]['params']
            nn.utils.clip_grad_norm_(current_params, args.grad_clip)
        opt.step()
        if verbose:
            print(f"  Iteration {it} complete!\n", flush=True)

        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        
        # 簡易進捗（毎10 iterationごと）
        if it % 10 == 0 and it % args.log_every != 0:
            print(f"  [{it}/{args.iters}] ...", end='\r', flush=True)
        
        # 詳細ログ
        if it % args.log_every == 0:
            dt = time.time() - t0
            iter_per_sec = it / dt if dt > 0 else 0
            eta_sec = (args.iters - it) / iter_per_sec if iter_per_sec > 0 else 0
            eta_min = eta_sec / 60
            temp_str = f"tau={tau:.4f}" if args.use_gumbel else f"lam={lam:.4f}"
            print(f"[{it}/{args.iters}] LRev={loss.item():.6f} ema={ema:.6f} {temp_str} time={dt:.1f}s speed={iter_per_sec:.2f}it/s ETA={eta_min:.1f}min", flush=True)
        
        # 弱い再初期化（未使用要素の探索維持）
        if args.reinit_every > 0 and it % args.reinit_every == 0:
            # 現在のバッチで選択確率を計算
            with torch.no_grad():
                U_reinit = utilities_matrix(flow, batch, menu, t_grid)
                Z_reinit = torch.softmax(lam * U_reinit, dim=1)
                n_reinit = reinit_unused_elements(flow, menu, Z_reinit, t_grid, 
                                                  threshold=args.reinit_threshold)

        # 途中保存
        if args.ckpt_every > 0 and it % args.ckpt_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "menu": [ (e.beta_raw.detach().cpu(), e.logits.detach().cpu(), e.mus.detach().cpu()) for e in menu ],
                "iteration": it,
                "m": args.m, "K": args.K, "D": args.D,
                "flow_ckpt": args.flow_ckpt,
                "optimizer_state": opt.state_dict(),
                "ema": ema
            }, os.path.join(args.out_dir, f"menu_stage2_step{it}.pt"))
            print(f"Saved checkpoint: menu_stage2_step{it}.pt")

    # 最終保存
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "menu_stage2_final.pt")
    torch.save({
        "menu": [ (e.beta_raw.detach().cpu(), e.logits.detach().cpu(), e.mus.detach().cpu()) for e in menu ],
        "iteration": args.iters,
        "m": args.m, "K": args.K, "D": args.D,
        "flow_ckpt": args.flow_ckpt,
        "optimizer_state": opt.state_dict(),
        "ema": ema
    }, final_path)
    print(f"Saved final: {final_path}")

    # （任意）テスト収益（ハードargmax；本文での推論手順） :contentReference[oaicite:10]{index=10}
    rev = eval_hard_revenue(flow, test[:args.eval_n], menu, t_grid)
    print(f"[TEST] hard-argmax revenue on {min(args.eval_n, len(test))} vals = {rev:.4f}")
    
    # 最終メニュー可視化
    print(f"\n[FINAL] Final menu visualization:", flush=True)
    from bf.menu import visualize_menu
    visualize_menu(flow, menu, t_grid, max_items=10, device=device)

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
    ap.add_argument("--resume_ckpt", type=str, default="", help="Resume from Stage2 checkpoint")
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--K", type=int, default=512)      # 実験既定は m≤100で5k/それ以上で20k。
    ap.add_argument("--D", type=int, default=8)        # 有限支持サイズ（Dirac混合の個数）。
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-1)  # Setup: 0.3 を推奨。
    ap.add_argument("--lam_start", type=float, default=1e-3)
    ap.add_argument("--lam_end", type=float, default=1e-1)  # 0.2 → 0.1（速めに上げる）
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
    ap.add_argument("--a", type=int, default=20)          # 合成XORの原子数（Table 4 を模倣）
    ap.add_argument("--atom_size_mode", type=str, default="small", 
                    choices=["small", "medium", "large", "uniform_3_8"],
                    help="Atom size distribution: small(~5 items), medium(~10), large(~25), uniform_3_8(3-8)")
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")
    
    # Gumbel-Softmax + STE（Stage 2の破綻を修正）
    ap.add_argument("--use_gumbel", action="store_true", help="Use Gumbel-Softmax + STE (fixes training-test gap)")
    ap.add_argument("--tau_start", type=float, default=1.0, help="Initial Gumbel-Softmax temperature")
    ap.add_argument("--tau_end", type=float, default=0.01, help="Final Gumbel-Softmax temperature")
    
    # μのウォームスタート＋再初期化
    ap.add_argument("--warmstart", action="store_true", help="Enable μ warmstart from representative bundles")
    ap.add_argument("--warmstart_grid", type=int, default=200, help="Number of random μ samples for warmstart")
    ap.add_argument("--reinit_every", type=int, default=2000, help="Reinitialize unused elements every N steps (0=disable)")
    ap.add_argument("--reinit_threshold", type=float, default=0.01, help="Reinit elements with z_mean < threshold")
    ap.add_argument("--freeze_beta_iters", type=int, default=1000, help="Freeze β for first N iterations (warmup)")

    args = ap.parse_args()
    train_stage2(args)
