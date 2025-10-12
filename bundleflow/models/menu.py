# bundleflow/models/menu.py
"""
MenuElement: 一要素 k の (φ_k, p_k)
Mechanism: 全メニュー（共通 v_θ と K個の(φ_k,p_k)）

目的: 共通のv_θと各要素(φ_k,p_k)からメニューMを構成し期待収入E[R]を最大化.
記号: K=メニュー要素数. 価値関数v(b)は外生.
API: expected_revenue(valuation_batch), argmax_menu(valuation_batch)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# v は XOR 評価器などで、v.value(s_bool: Tensor[m]) -> float を想定（推論時は厳密計算）.

class MenuElement(nn.Module):
    """
    一要素 k の (φ_k, p_k)。p_k≥0はsoftplus等で保証。
    
    目的: 各メニュー要素kの初期分布パラメータφ_kと価格p_kを学習
    記号: φ_k = {μ_d^(k), w_d^(k)}_d, p_k = softplus(β_k)
    """
    
    def __init__(self, m: int, D: int):
        super().__init__()
        self.m = m
        self.D = D
        
        # βの初期化を多様化（-3.0から-1.0の範囲でランダム）
        beta_init = torch.randn(1) * 0.5 - 2.0  # mean=-2.0, std=0.5
        self.beta_raw = nn.Parameter(beta_init)
        self.logits = nn.Parameter(torch.zeros(D))        # 混合重みのロジット
        self.mus = nn.Parameter(torch.zeros(D, m))        # 初期分布の支持 μ_d^(k)

    @property
    def beta(self) -> torch.Tensor:
        """
        価格 p_k = softplus(β_k) ≥ 0
        
        Returns:
            p_k: 価格 (scalar)
        """
        # softplusで β ≥ 0 を保証（論文の p ≥ 0 と整合）
        # log_densityクリップ後は、IR制約が自然にβを制限するため、
        # 上限は念のための安全装置として緩く設定（10.0）
        beta_unbounded = torch.nn.functional.softplus(self.beta_raw)
        return torch.clamp(beta_unbounded, 0.0, 10.0)  # 緩い上限
    
    @property
    def weights(self) -> torch.Tensor:
        """
        混合重み w_d^(k)
        
        Returns:
            w: 重み (D,) simplex
        """
        return torch.softmax(self.logits, dim=0)       # simplex
    
    def sample_init(self, n: int) -> torch.Tensor:
        """
        初期分布からサンプル z ~ p_0(φ_k)
        
        Args:
            n: サンプル数
            
        Returns:
            z: サンプル (n, m)
        """
        # 混合Dirac分布からサンプル
        device = self.mus.device
        cat = torch.distributions.Categorical(probs=self.weights)
        idx = cat.sample((n,))  # (n,)
        return self.mus[idx]  # (n, m)
    
    def price(self) -> torch.Tensor:
        """
        価格 p_k
        
        Returns:
            p_k: 価格 (scalar)
        """
        return self.beta
    
    def to(self, device):
        """デバイス移動メソッド"""
        super().to(device)
        return self


def make_null_element(m: int) -> MenuElement:
    """
    IRのためにNull Menuを用意
    
    Args:
        m: 商品数
        
    Returns:
        null_elem: ヌル要素（価格0、効用0）
    """
    elem = MenuElement(m, D=1)
    with torch.no_grad():
        elem.beta_raw.fill_(float('-inf'))  # softplus(-inf) ≈ 0.0
        elem.logits.fill_(0.0)
        elem.mus.zero_()
    for p in elem.parameters():
        p.requires_grad_(False)
    return elem


class Mechanism(nn.Module):
    """
    全メニュー（共通 v_θ と K個の(φ_k,p_k)）
    
    目的: 共通のv_θと各要素(φ_k,p_k)からメニューMを構成し期待収入E[R]を最大化.
    記号: K=メニュー要素数. 価値関数v(b)は外生.
    API: expected_revenue(valuation_batch), argmax_menu(valuation_batch)
    """
    
    def __init__(self, flow, menu: List[MenuElement]):
        super().__init__()
        self.flow = flow
        self.menu = menu
        self.K = len(menu)
    
    def expected_revenue(self, valuation_batch) -> torch.Tensor:
        """
        期待収入を計算
        
        Args:
            valuation_batch: 評価関数のリスト
            
        Returns:
            revenue: 期待収入
        """
        # 効用行列を計算
        t_grid = torch.linspace(0.0, 1.0, steps=25, device=next(self.menu[0].parameters()).device)
        U = utilities_matrix(self.flow, valuation_batch, self.menu, t_grid)  # (B, K)
        
        # 価格を取得
        beta = torch.stack([elem.beta for elem in self.menu]).squeeze()  # (K,)
        
        # ソフト割当
        lam = 0.1  # 温度パラメータ
        Z = torch.softmax(lam * U, dim=1)  # (B, K)
        
        # 期待収入
        revenue = (Z * beta.unsqueeze(0)).sum(dim=1).mean()
        
        return revenue
    
    def argmax_menu(self, valuation_batch) -> Dict[str, Any]:
        """
        ハード割当でのメニュー選択結果
        
        Args:
            valuation_batch: 評価関数のリスト
            
        Returns:
            result: 割当/価格/期待厚生などの辞書
        """
        # 効用行列を計算
        t_grid = torch.linspace(0.0, 1.0, steps=25, device=next(self.menu[0].parameters()).device)
        U = utilities_matrix(self.flow, valuation_batch, self.menu, t_grid)  # (B, K)
        
        # ハード割当
        selected_idx = torch.argmax(U, dim=1)  # (B,)
        selected_utility = torch.gather(U, 1, selected_idx.unsqueeze(1)).squeeze(1)  # (B,)
        
        # 価格を取得
        beta = torch.stack([elem.beta for elem in self.menu]).squeeze()  # (K,)
        selected_prices = beta[selected_idx]  # (B,)
        
        # IR制約チェック
        ir_mask = (selected_utility >= 0.0).float()  # (B,)
        
        # 収入計算
        revenue = (selected_prices * ir_mask).mean()
        
        # 厚生計算（簡略化）
        welfare = selected_utility.mean()
        
        return {
            'assignments': selected_idx,
            'utilities': selected_utility,
            'prices': selected_prices,
            'revenue': revenue,
            'welfare': welfare,
            'ir_satisfied': ir_mask.mean()
        }


# ---- Stage 2: 効用・損失 ------------------------------------------------

# メニューlに関して、効用を計算している。
# u^(k)(v) = Σ_d v(s(μ_d^(k))) · w_d^(k) · exp{-Tr[Q(μ_d^(k))]∫η} − β^(k)  （Eq.(21)）
def utility_element(flow, v, elem: MenuElement, t_grid: torch.Tensor) -> torch.Tensor:
    """
    単一メニュー要素の効用を計算
    
    Args:
        flow: BundleFlow
        v: 評価関数
        elem: MenuElement
        t_grid: 時間グリッド
        
    Returns:
        utility: 効用
    """
    sT = flow.flow_forward(elem.mus, t_grid)     # (D,m)  （Eq.(20)）
    s = flow.round_to_bundle(sT)                 # (D,m)  （Eq.(21)の s(μ)）
    vals = torch.tensor([float(v.value(s[d])) for d in range(s.shape[0])],
                        device=s.device, dtype=torch.float64)  # 安定化: float64
    
    # 数値安定化されたlog-sum-exp実装
    trQ = flow.Q(elem.mus).diagonal(dim1=-2, dim2=-1).sum(-1).to(torch.float64)   # (D,)
    integ = flow.eta_integral(t_grid).to(torch.float64)                           # ()
    log_w = torch.log_softmax(elem.logits.to(torch.float64), dim=0) - trQ * integ # (D,)
    
    # log-sum-exp: M = max(log_w), u = exp(M) * Σ_d exp(log_w - M) * v_d
    M = log_w.max()
    u = torch.exp(M) * torch.sum(torch.exp(log_w - M) * vals)                     # Σ_d e^{log_w-M} v_d
    
    # β≥0: softplusで正の値に制限（IRと整合）
    beta = torch.nn.functional.softplus(elem.beta_raw)                            # β≥0
    
    return (u.to(s.dtype) - beta)


# バッチ処理版：全menu elementsを一度に処理
# U[b,k] = u^(k)(v_b)  （Eq.(21) を全組で）
def utilities_matrix_batched(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor, verbose: bool = False, debug: bool = False, v0_test=None) -> torch.Tensor:
    """
    高速バッチ処理版：全menu elementsのmusを統合して一度に処理
    速度: O(B * K * D) → O(B * 1)のflow_forward呼び出し
    
    Args:
        flow: BundleFlow
        V: 評価関数のリスト
        menu: MenuElementのリスト
        t_grid: 時間グリッド
        verbose: 詳細ログ
        debug: デバッグログ
        v0_test: テスト用評価関数
        
    Returns:
        U: 効用行列 (B, K)
    """
    K = len(menu)
    B = len(V)
    device = t_grid.device
    
    if verbose:
        print(f"  [Batched] Collecting all mus from {K} menu elements...", flush=True)
    
    # null要素（最後の要素、D=1）を分離
    menu_main = menu[:-1]  # 通常のメニュー要素（D=8など）
    null_elem = menu[-1]   # null要素（D=1）
    K_main = len(menu_main)
    
    if verbose:
        print(f"  [Batched] K={K}, K_main={K_main}, menu length={len(menu)}", flush=True)
        print(f"  [Batched] Splitting: menu_main has {K_main} elements, null is 1 element", flush=True)
    
    # 通常のメニュー要素のmusを統合: [(D, m)] * K_main -> (K_main, D, m)
    all_mus = torch.stack([elem.mus for elem in menu_main])  # (K_main, D, m)
    all_weights = torch.stack([elem.weights for elem in menu_main])  # (K_main, D)
    all_betas = torch.stack([elem.beta for elem in menu_main]).squeeze()  # (K_main,)
    
    K_main, D, m = all_mus.shape
    
    if verbose:
        print(f"  [Batched] Running flow_forward on {K_main*D} bundles at once...", flush=True)
        # μの範囲を確認
        print(f"  [Batched] all_mus range: [{all_mus.min().item():.4f}, {all_mus.max().item():.4f}], mean={all_mus.mean().item():.4f}", flush=True)
    
    # 全てのμをバッチ処理: (K_main, D, m) -> (K_main*D, m)
    mus_flat = all_mus.view(K_main * D, m)
    sT_flat = flow.flow_forward(mus_flat, t_grid)  # (K_main*D, m) - 一度の呼び出し！
    
    if verbose:
        print(f"  [Batched] sT_flat range BEFORE rounding: [{sT_flat.min().item():.4f}, {sT_flat.max().item():.4f}], mean={sT_flat.mean().item():.4f}", flush=True)
    
    s_flat = flow.round_to_bundle(sT_flat)  # (K_main*D, m)
    
    if verbose:
        print(f"  [Batched] s_flat range AFTER rounding: [{s_flat.min().item():.4f}, {s_flat.max().item():.4f}], mean={s_flat.mean().item():.4f}", flush=True)
    
    # log_density_weightもバッチ処理
    log_density_flat = flow.log_density_weight(mus_flat, t_grid)  # (K_main*D,)
    # 安定化：log_densityをクリップ（exp爆発を防止）
    log_density_flat = torch.clamp(log_density_flat, -10.0, 0.0)  # 密度重み ≤ 1.0
    log_density = log_density_flat.view(K_main, D)  # (K_main, D)
    
    # null要素も一度に処理
    null_sT = flow.flow_forward(null_elem.mus, t_grid)  # (1, m)
    null_s = flow.round_to_bundle(null_sT)  # (1, m)
    null_log_density = flow.log_density_weight(null_elem.mus, t_grid)  # (1,)
    null_log_density = torch.clamp(null_log_density, -10.0, 0.0)  # 安定化
    null_weight = null_elem.weights  # (1,)
    null_beta = null_elem.beta  # scalar
    
    # 各valuationについて効用を計算
    # 注意：menuは K_main + 1 個（null含む）
    U = torch.zeros(B, K, device=device, dtype=torch.float32)  # K = K_main + 1
    
    # s_flatをCPUに一度だけ転送（全valuationで共有）
    s_flat_cpu = s_flat.cpu()
    
    for i, v in enumerate(V):
        if verbose and i % 10 == 0:
            print(f"  [Batched] Processing valuation {i+1}/{B}...", flush=True)
        
        # 全bundlesの価値を一度に計算
        if hasattr(v, 'batch_value'):
            vals_flat = v.batch_value(s_flat_cpu)  # (K_main*D,)
            
            # デバッグ：複数のvaluationをチェック
            if debug and i < 3:  # 最初の3個のvaluationをチェック
                non_zero_count = (vals_flat > 0).sum().item()
                print(f"  [DEBUG] Valuation {i}: batch_value min={vals_flat.min().item():.4f}, max={vals_flat.max().item():.4f}, non-zero={non_zero_count}/{len(vals_flat)}", flush=True)
                print(f"  [DEBUG] Valuation {i}: num_atoms={len(v.atoms)}, max_price={max(p for _, p in v.atoms):.4f}", flush=True)
            
            vals_flat = vals_flat.to(device)  # GPUに戻す
        else:
            # fallback: 逐次処理
            vals_flat = torch.tensor([v.value(s_flat_cpu[j]) for j in range(K_main*D)],
                                    device=device, dtype=torch.float32)
        
        vals = vals_flat.view(K_main, D)  # (K_main, D)
        
        # 各menu elementの効用を計算（log-sum-exp）
        log_w = torch.log(all_weights + 1e-10)  # (K_main, D)
        log_weights = log_w + log_density  # (K_main, D)
        
        # デバッグ: 最初のvaluationで詳細を出力（debugフラグが立っているとき）
        if __debug__ and debug and i == 0:
            # 簡潔なデバッグ情報のみ
            print(f"  [DEBUG] s_flat shape: {s_flat.shape}, unique bundles: {len(torch.unique(s_flat, dim=0))}", flush=True)
            print(f"  [DEBUG] Number of valuation atoms: {len(v.atoms)}", flush=True)
            print(f"  [DEBUG] vals range: [{vals.min().item():.4f}, {vals.max().item():.4f}]", flush=True)
        
        # log-sum-exp per menu element
        M = torch.max(log_weights, dim=1, keepdim=True)[0]  # (K_main, 1)
        weighted_sum = (torch.exp(log_weights - M) * vals).sum(dim=1)  # (K_main,)
        u_main = torch.exp(M.squeeze(1)) * weighted_sum - all_betas  # (K_main,)
        
        if __debug__ and debug and i == 0:
            print(f"  [DEBUG] u_main range: [{u_main.min().item():.4f}, {u_main.max().item():.4f}]", flush=True)
        
        # null要素の効用を計算（事前計算済みのデータを使用）
        if hasattr(v, 'value'):
            null_val = v.value(null_s[0])
        else:
            null_val = 0.0
        log_w_null = torch.log(null_weight + 1e-10)
        log_weight_null = log_w_null + null_log_density
        u_null = torch.exp(log_weight_null[0]) * null_val - null_beta
        
        # 結合: [u_main, u_null]
        if debug and i == 0:
            print(f"  [DEBUG] u_main shape: {u_main.shape}, u_null shape: {u_null.shape if isinstance(u_null, torch.Tensor) else 'scalar'}", flush=True)
            print(f"  [DEBUG] U shape: {U.shape}, U[i, :-1] will be shape {U[i, :-1].shape}", flush=True)
            print(f"  [DEBUG] Assigning u_main (len={len(u_main)}) to U[{i}, :-1] (len={len(U[i, :-1])})", flush=True)
        
        U[i, :-1] = u_main
        U[i, -1] = u_null
        
        if debug and i == 0:
            print(f"  [DEBUG] After assignment: U[{i}] range = [{U[i].min().item():.4f}, {U[i].max().item():.4f}]", flush=True)
            print(f"  [DEBUG] U[{i}, -1] (null) = {U[i, -1].item():.4f}", flush=True)
    
    if verbose:
        print(f"  [Batched] Utilities computed: {U.shape}", flush=True)
    
    return U  # (B, K)


# バリュー集合Vと、全てのメニューlに関して、効用を行列表示している。
# U[b,k] = u^(k)(v_b)  （Eq.(21) を全組で）
def utilities_matrix(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor, verbose: bool = False, debug: bool = False, v0_test=None) -> torch.Tensor:
    """
    効用行列を計算
    
    Args:
        flow: BundleFlow
        V: 評価関数のリスト
        menu: MenuElementのリスト
        t_grid: 時間グリッド
        verbose: 詳細ログ
        debug: デバッグログ
        v0_test: テスト用評価関数
        
    Returns:
        U: 効用行列 (B, K)
    """
    # バッチ処理版を使用
    return utilities_matrix_batched(flow, V, menu, t_grid, verbose=verbose, debug=debug, v0_test=v0_test)


# 学習時の連続割り当てを返す。
# z^(k)(v) = SoftMax_k( λ · u^(k)(v) )  （Eq.(23)）
def soft_assignment(U: torch.Tensor, lam: float) -> torch.Tensor:
    """
    ソフト割当を計算
    
    Args:
        U: 効用行列 (B, K)
        lam: 温度パラメータ
        
    Returns:
        Z: ソフト割当 (B, K)
    """
    return torch.softmax(lam * U, dim=1)


# 期待損益の負の値を最小化している。
def revenue_loss(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor, lam: float = 0.1, 
                 verbose: bool = False, debug: bool = False, v0_test=None,
                 use_gumbel: bool = False, tau: float = 0.1) -> torch.Tensor:
    """
    Stage 2の収益損失関数
    
    Args:
        flow: FlowModel
        V: Valuations
        menu: List[MenuElement]
        t_grid: 時間グリッド
        lam: Softmax温度（use_gumbel=Falseの時のみ使用）
        verbose: 詳細ログ
        debug: デバッグログ
        v0_test: テスト用valuation
        use_gumbel: TrueならGumbel-Softmax + STE、FalseならSoftmax relaxation
        tau: Gumbel-Softmax温度（use_gumbel=Trueの時のみ使用）
        
    Returns:
        -revenue (損失)
    """
    # LRev = -(1/|V|) Σ_v Σ_k z_k(v) β_k  （Eq.(22)）
    if verbose:
        print(f"  Computing utilities matrix ({len(V)} valuations × {len(menu)} menu elements)...", flush=True)
    U = utilities_matrix(flow, V, menu, t_grid, verbose=verbose, debug=debug, v0_test=v0_test)  # (B,K)
    beta = torch.stack([elem.beta for elem in menu]).squeeze()     # (K,)
    
    if use_gumbel:
        # Gumbel-Softmax + STE: Forward時にhard argmax、Backward時にsoft勾配
        if verbose:
            print(f"  Computing Gumbel-Softmax assignment (tau={tau}, hard=True)...", flush=True)
        
        from bf.utils import gumbel_softmax
        if debug:
            print(f"  [DEBUG-GUMBEL] U shape: {U.shape}, U range: [{U.min().item():.4f}, {U.max().item():.4f}]", flush=True)
        Z = gumbel_softmax(U, tau=tau, hard=True, dim=1)  # (B, K) one-hot
        if debug:
            print(f"  [DEBUG-GUMBEL] Z shape: {Z.shape}, Z sum per row: {Z.sum(dim=1)[:5].tolist()}", flush=True)
        
        # IR制約の明示的適用: 選択されたメニューの効用が負なら収益ゼロ
        selected_idx = torch.argmax(Z, dim=1)  # (B,)
        selected_utility = torch.gather(U, 1, selected_idx.unsqueeze(1)).squeeze(1)  # (B,)
        ir_mask = (selected_utility >= 0.0).float()  # (B,)
        
        # 収益計算: IR制約を満たさない場合は0
        if debug:
            print(f"  [DEBUG-GUMBEL] beta shape: {beta.shape}, beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]", flush=True)
        rev_per_buyer = (Z * beta.unsqueeze(0)).sum(dim=1) * ir_mask  # (B,)
        rev = rev_per_buyer.mean()
        
        if debug:
            print(f"  [DEBUG-REV-GUMBEL] U range: [{U.min().item():.4f}, {U.max().item():.4f}]", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] U mean: {U.mean().item():.4f}", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] beta mean: {beta.mean().item():.4f}", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] Z shape: {Z.shape}, is one-hot: {(Z.sum(dim=1) == 1.0).all().item()}", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] Selected indices: {selected_idx[:5].tolist()}", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] Selected utilities: {selected_utility[:5].tolist()}", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] IR constraint satisfied: {ir_mask.sum().item()}/{len(ir_mask)}", flush=True)
            print(f"  [DEBUG-REV-GUMBEL] Revenue per buyer (first 5): {rev_per_buyer[:5].tolist()}", flush=True)
    else:
        # 従来のSoftmax relaxation
        if verbose:
            print(f"  Computing soft assignment (lambda={lam})...", flush=True)
        Z = soft_assignment(U, lam)  # (B,K)
        rev = (Z * beta.unsqueeze(0)).sum(dim=1).mean()
        
        if debug:
            print(f"  [DEBUG-REV] Z shape: {Z.shape}, beta shape: {beta.shape}", flush=True)
            print(f"  [DEBUG-REV] Z[0] (first valuation selection probs): min={Z[0].min().item():.6f}, max={Z[0].max().item():.6f}", flush=True)
            print(f"  [DEBUG-REV] Z[:, -1] (null selection): min={Z[:, -1].min().item():.6f}, max={Z[:, -1].max().item():.6f}, mean={Z[:, -1].mean().item():.6f}", flush=True)
            print(f"  [DEBUG-REV] beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]", flush=True)
            print(f"  [DEBUG-REV] beta[-1] (null price): {beta[-1].item():.6f}", flush=True)
            print(f"  [DEBUG-REV] (Z * beta) per valuation: {(Z * beta.unsqueeze(0)).sum(dim=1)[:5].tolist()}", flush=True)
    
    if verbose:
        print(f"  Revenue: {rev.item():.6f} ({'Gumbel+STE' if use_gumbel else 'Softmax'})", flush=True)
    return -rev


# テスト時
@torch.no_grad()
def visualize_menu(flow, menu: List[MenuElement], t_grid: torch.Tensor, 
                   max_items: int = 10, device: torch.device = None) -> None:
    """
    メニューの内容を可視化（どの商品の組み合わせが提供されているか）
    
    Args:
        flow: BundleFlow
        menu: MenuElementのリスト
        t_grid: 時間グリッド
        max_items: 表示する最大アイテム数
        device: デバイス
    """
    if device is None:
        device = next(menu[0].parameters()).device
    
    print(f"\n{'='*60}")
    print(f"📋 MENU VISUALIZATION (showing first {max_items} items)")
    print(f"{'='*60}")
    
    for k, elem in enumerate(menu[:max_items]):
        # バンドル生成（μから）
        with torch.no_grad():
            s_T = flow.flow_forward(elem.mus, t_grid)  # (D, m)
            s = flow.round_to_bundle(s_T)  # (D, m) -> discrete bundles
            
        # 各バンドルの内容を表示
        beta_val = elem.beta.item()
        print(f"\n🍽️  Menu Item {k+1}: Price = {beta_val:.4f}")
        
        # ユニークなバンドルを抽出
        unique_bundles = []
        for d in range(s.shape[0]):
            bundle = s[d].cpu().numpy()
            bundle_tuple = tuple(bundle.astype(int))
            if bundle_tuple not in unique_bundles:
                unique_bundles.append(bundle_tuple)
        
        print(f"   📦 Generated bundles ({len(unique_bundles)} unique):")
        for i, bundle in enumerate(unique_bundles[:5]):  # 最大5個表示
            items = [j for j, val in enumerate(bundle) if val == 1]
            if items:
                items_str = ", ".join([f"Item_{j}" for j in items])
                print(f"      {i+1}. [{items_str}]")
            else:
                print(f"      {i+1}. [Empty bundle]")
        
        if len(unique_bundles) > 5:
            print(f"      ... and {len(unique_bundles)-5} more bundles")
    
    if len(menu) > max_items:
        print(f"\n... and {len(menu)-max_items} more menu items")
    
    print(f"{'='*60}\n")


# テスト時はSoftMaxではなく、argmaxを使っている。
def infer_choice(flow, v, menu: List[MenuElement], t_grid: torch.Tensor) -> int:
    """
    推論時の選択を計算
    
    Args:
        flow: BundleFlow
        v: 評価関数
        menu: MenuElementのリスト
        t_grid: 時間グリッド
        
    Returns:
        choice: 選択されたメニュー要素のインデックス
    """
    # 推論時は argmax（本文 Sec.3.3 / Eq.(23) のハード版）
    U = torch.stack([utility_element(flow, v, elem, t_grid) for elem in menu])  # (K,)
    return int(torch.argmax(U).item())
