# bf/menu.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

# v は XOR 評価器などで、v.value(s_bool: Tensor[m]) -> float を想定（推論時は厳密計算）.

class MenuElement(nn.Module):
    def __init__(self, m: int, D: int):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.0))    # 価格 β^(k)
        self.logits = nn.Parameter(torch.zeros(D))     # 混合重みのロジット
        self.mus = nn.Parameter(torch.zeros(D, m))     # 初期分布の支持 μ_d^(k)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)       # simplex

# IRのためにNull Menuを用意している。
def make_null_element(m: int) -> MenuElement:
    elem = MenuElement(m, D=1)
    with torch.no_grad():
        elem.beta.fill_(0.0)
        elem.logits.fill_(0.0)
        elem.mus.zero_()
    for p in elem.parameters():
        p.requires_grad_(False)
    return elem

# ---- Stage 2: 効用・損失 ------------------------------------------------

# メニューlに関して、効用を計算している。
# u^(k)(v) = Σ_d v(s(μ_d^(k))) · w_d^(k) · exp{-Tr[Q(μ_d^(k))]∫η} − β^(k)  （Eq.(21)）
def utility_element(flow, v, elem: MenuElement, t_grid: torch.Tensor) -> torch.Tensor:
    w = elem.weights                             # (D,)
    sT = flow.flow_forward(elem.mus, t_grid)     # (D,m)  （Eq.(20)）
    s = flow.round_to_bundle(sT)                 # (D,m)  （Eq.(21)の s(μ)）
    with torch.no_grad():
        vals = torch.tensor([float(v.value(s[d])) for d in range(s.shape[0])],
                            device=s.device, dtype=s.dtype)  # v(s)
    
    # log-sum-exp トリックで数値安定化（式(21)の指数重み）
    log_w = torch.log(w + 1e-10)                              # log(w_d)
    log_density = flow.log_density_weight(elem.mus, t_grid)   # (D,) = -Tr[Q]∫η
    log_weights = log_w + log_density                         # (D,) = log(w_d) + log_density
    
    # log-sum-exp: u = exp(M) * Σ_d exp(log_w_d - M) * v_d - β
    M = torch.max(log_weights)
    weighted_sum = (torch.exp(log_weights - M) * vals).sum()
    u = torch.exp(M) * weighted_sum - elem.beta
    return u

# バッチ処理版：全menu elementsを一度に処理
# U[b,k] = u^(k)(v_b)  （Eq.(21) を全組で）
def utilities_matrix_batched(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor, verbose: bool = False, debug: bool = False, v0_test=None) -> torch.Tensor:
    """
    高速バッチ処理版：全menu elementsのmusを統合して一度に処理
    速度: O(B * K * D) → O(B * 1)のflow_forward呼び出し
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
    all_betas = torch.stack([elem.beta for elem in menu_main])  # (K_main,)
    
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
    log_density = log_density_flat.view(K_main, D)  # (K_main, D)
    
    # null要素も一度に処理
    null_sT = flow.flow_forward(null_elem.mus, t_grid)  # (1, m)
    null_s = flow.round_to_bundle(null_sT)  # (1, m)
    null_log_density = flow.log_density_weight(null_elem.mus, t_grid)  # (1,)
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
                print(f"  [DEBUG] Valuation {i}: batch_value min={vals_flat.min().item():.4f}, max={vals_flat.max().item():.4f}, non-zero={non_zero_count}/512", flush=True)
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
        if debug and i == 0:
            # bundleの中身を確認
            print(f"  [DEBUG] s_flat shape: {s_flat.shape}, unique values: {torch.unique(s_flat).tolist()}", flush=True)
            
            # bundleの多様性を確認
            unique_bundles = torch.unique(s_flat, dim=0)
            print(f"  [DEBUG] Number of UNIQUE bundles: {len(unique_bundles)} out of {len(s_flat)}", flush=True)
            
            # 各bundleのビット数（アイテム数）の分布
            num_items = s_flat.sum(dim=1)  # 各bundleのアイテム数
            print(f"  [DEBUG] Items per bundle: min={num_items.min().item():.0f}, max={num_items.max().item():.0f}, mean={num_items.mean().item():.1f}, std={num_items.std().item():.1f}", flush=True)
            
            print(f"  [DEBUG] s_flat[0:5]: {s_flat[0:5].tolist()}", flush=True)
            print(f"  [DEBUG] Number of valuation atoms: {len(v.atoms)}", flush=True)
            if len(v.atoms) > 0:
                print(f"  [DEBUG] First atom: mask={v.atoms[0][0]}, price={v.atoms[0][1]:.4f}", flush=True)
                # このvaluationの最大可能価値（最も高いatomの価格）
                max_possible_value = max(price for _, price in v.atoms)
                print(f"  [DEBUG] Max possible value from this valuation: {max_possible_value:.4f}", flush=True)
            else:
                print(f"  [DEBUG] WARNING: This valuation has NO atoms!", flush=True)
            
            # value()メソッドで直接テスト（CPUに転送してから）
            print(f"  [DEBUG] Calling v.value(s_flat_cpu[0]) with debug=True:", flush=True)
            test_val_direct = v.value(s_flat_cpu[0], debug=True)
            print(f"  [DEBUG] v.value(s_flat_cpu[0]) = {test_val_direct:.4f}", flush=True)
            
            # 起動時に成功したv0でもテスト
            if v0_test is not None:
                print(f"  [DEBUG] Testing with v0 (startup valuation) that worked before:", flush=True)
                v0_test_result = v0_test.value(s_flat_cpu[0])
                print(f"  [DEBUG] v0.value(s_flat_cpu[0]) = {v0_test_result:.4f}", flush=True)
                all_items_cpu = torch.ones(50, device='cpu')
                v0_all_items = v0_test.value(all_items_cpu)
                print(f"  [DEBUG] v0.value(all_items) = {v0_all_items:.4f} (should be 0.9496)", flush=True)
            
            # マスクの詳細を確認
            from bf.valuation import _tensor_to_mask
            test_mask = _tensor_to_mask(s_flat_cpu[0])
            print(f"  [DEBUG] _tensor_to_mask(s_flat_cpu[0]) = {test_mask}", flush=True)
            
            # 比較：GPU tensorで試すとどうなるか
            try:
                test_mask_gpu = _tensor_to_mask(s_flat[0])
                print(f"  [DEBUG] GPU tensor mask: {test_mask_gpu}, CPU tensor mask: {test_mask}, equal? {test_mask_gpu == test_mask}", flush=True)
            except Exception as e:
                print(f"  [DEBUG] GPU tensor failed: {e}", flush=True)
            print(f"  [DEBUG] bin(test_mask) = {bin(test_mask)}", flush=True)
            print(f"  [DEBUG] First atom mask = {v.atoms[0][0]}, bin = {bin(v.atoms[0][0])}", flush=True)
            
            # ビット位置を直接比較
            bundle_bits = [i for i in range(50) if (test_mask & (1 << i)) != 0]
            atom0_bits = [i for i in range(50) if (v.atoms[0][0] & (1 << i)) != 0]
            print(f"  [DEBUG] Bundle[0] items (bit positions): {bundle_bits}", flush=True)
            print(f"  [DEBUG] Atom[0] items (bit positions): {atom0_bits}", flush=True)
            
            # どのアイテムがAtomにあるがBundleにない？
            missing_in_bundle = [i for i in atom0_bits if i not in bundle_bits]
            print(f"  [DEBUG] Items in Atom[0] but NOT in Bundle[0]: {missing_in_bundle}", flush=True)
            
            # 逆：Bundleにあるがatomにない
            extra_in_bundle = [i for i in bundle_bits if i not in atom0_bits]
            print(f"  [DEBUG] Items in Bundle[0] but NOT in Atom[0]: {extra_in_bundle} (count: {len(extra_in_bundle)})", flush=True)
            
            # 手動でT⊆Sをチェック
            atom_mask = v.atoms[0][0]
            test_subset = (atom_mask & (~test_mask)) == 0
            print(f"  [DEBUG] Is atom[0] ⊆ s_flat[0]? {test_subset} (atom_mask & ~test_mask = {atom_mask & (~test_mask)})", flush=True)
            
            # 全てのatomをチェック
            matching_atoms = []
            print(f"  [DEBUG] Checking all {len(v.atoms)} atoms...", flush=True)
            for idx, (atom_mask, price) in enumerate(v.atoms[:5]):  # 最初の5個だけ
                check_result = (atom_mask & (~test_mask))
                is_subset = check_result == 0
                print(f"  [DEBUG]   Atom {idx}: mask={atom_mask}, price={price:.4f}, atom&~test={check_result}, is_subset={is_subset}", flush=True)
                if is_subset:
                    matching_atoms.append((atom_mask, price))
            
            # 全atomをチェック
            for atom_mask, price in v.atoms:
                if (atom_mask & (~test_mask)) == 0:
                    matching_atoms.append((atom_mask, price))
            
            print(f"  [DEBUG] Number of matching atoms: {len(matching_atoms)}", flush=True)
            if len(matching_atoms) > 0:
                print(f"  [DEBUG] Max matching price: {max(p for _, p in matching_atoms):.4f}", flush=True)
            
            # このvaluationで全ONのbundleを評価（起動時テストと同じ）
            all_items_cpu = torch.ones(50, device='cpu')
            all_items_val = v.value(all_items_cpu)
            print(f"  [DEBUG] THIS valuation's value for ALL items: {all_items_val:.4f}", flush=True)
            
            # 逆に、test_maskがどのatomのスーパーセットか確認
            print(f"  [DEBUG] Checking if test_mask is a superset of any atom...", flush=True)
            for idx, (atom_mask, price) in enumerate(v.atoms[:3]):
                # T⊆Sの判定が正しいか確認
                subset_check = (atom_mask & (~test_mask)) == 0
                print(f"  [DEBUG]   Atom {idx}: T⊆S? {subset_check}", flush=True)
                # ビット数を確認
                atom_bits = bin(atom_mask).count('1')
                test_bits = bin(test_mask).count('1')
                print(f"  [DEBUG]     Atom has {atom_bits} bits set, test has {test_bits} bits set", flush=True)
                
                # atomの最上位ビットを確認
                highest_bit = atom_mask.bit_length() - 1  # 最上位ビットの位置
                print(f"  [DEBUG]     Highest bit position: {highest_bit} (should be < 50)", flush=True)
            
            # batch_value()の結果
            print(f"  [DEBUG] vals range: [{vals.min().item():.4f}, {vals.max().item():.4f}]", flush=True)
            print(f"  [DEBUG] vals[0:5]: {vals[0, :].tolist()}", flush=True)
            
            # vals_flatの最初の10個
            print(f"  [DEBUG] vals_flat[0:10]: {vals_flat[0:10].tolist()}", flush=True)
            
            print(f"  [DEBUG] log_w range: [{log_w.min().item():.4f}, {log_w.max().item():.4f}]", flush=True)
            print(f"  [DEBUG] log_density range: [{log_density.min().item():.4f}, {log_density.max().item():.4f}]", flush=True)
            print(f"  [DEBUG] log_weights range: [{log_weights.min().item():.4f}, {log_weights.max().item():.4f}]", flush=True)
        
        # log-sum-exp per menu element
        M = torch.max(log_weights, dim=1, keepdim=True)[0]  # (K_main, 1)
        weighted_sum = (torch.exp(log_weights - M) * vals).sum(dim=1)  # (K_main,)
        u_main = torch.exp(M.squeeze(1)) * weighted_sum - all_betas  # (K_main,)
        
        if debug and i == 0:
            print(f"  [DEBUG] M range: [{M.min().item():.4f}, {M.max().item():.4f}]", flush=True)
            print(f"  [DEBUG] exp(M) range: [{torch.exp(M).min().item():.4e}, {torch.exp(M).max().item():.4e}]", flush=True)
            print(f"  [DEBUG] weighted_sum range: [{weighted_sum.min().item():.4f}, {weighted_sum.max().item():.4f}]", flush=True)
            print(f"  [DEBUG] u_main range: [{u_main.min().item():.4f}, {u_main.max().item():.4f}]", flush=True)
            print(f"  [DEBUG] all_betas range: [{all_betas.min().item():.4f}, {all_betas.max().item():.4f}]", flush=True)
            # βの分布を確認
            unique_betas = torch.unique(all_betas)
            print(f"  [DEBUG] Number of unique beta values: {len(unique_betas)}, values: {unique_betas.tolist()}", flush=True)
        
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
    # バッチ処理版を使用
    return utilities_matrix_batched(flow, V, menu, t_grid, verbose=verbose, debug=debug, v0_test=v0_test)

# 学習時の連続割り当てを返す。
# z^(k)(v) = SoftMax_k( λ · u^(k)(v) )  （Eq.(23)）
def soft_assignment(U: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.softmax(lam * U, dim=1)

# 期待損益の負の値を最小化している。
def revenue_loss(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor, lam: float, verbose: bool = False, debug: bool = False, v0_test=None) -> torch.Tensor:
    # LRev = -(1/|V|) Σ_v Σ_k z_k(v) β_k  （Eq.(22)）
    if verbose:
        print(f"  Computing utilities matrix ({len(V)} valuations × {len(menu)} menu elements)...", flush=True)
    U = utilities_matrix(flow, V, menu, t_grid, verbose=verbose, debug=debug, v0_test=v0_test)          # (B,K)
    if verbose:
        print(f"  Computing soft assignment...", flush=True)
    Z = soft_assignment(U, lam)                          # (B,K)
    beta = torch.stack([elem.beta for elem in menu])     # (K,)
    rev = (Z * beta.unsqueeze(0)).sum(dim=1).mean()
    
    if debug:
        print(f"  [DEBUG-REV] Z shape: {Z.shape}, beta shape: {beta.shape}", flush=True)
        print(f"  [DEBUG-REV] Z[0] (first valuation selection probs): min={Z[0].min().item():.6f}, max={Z[0].max().item():.6f}", flush=True)
        print(f"  [DEBUG-REV] Z[:, -1] (null selection): min={Z[:, -1].min().item():.6f}, max={Z[:, -1].max().item():.6f}, mean={Z[:, -1].mean().item():.6f}", flush=True)
        print(f"  [DEBUG-REV] beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]", flush=True)
        print(f"  [DEBUG-REV] beta[-1] (null price): {beta[-1].item():.6f}", flush=True)
        print(f"  [DEBUG-REV] (Z * beta) per valuation: {(Z * beta.unsqueeze(0)).sum(dim=1)[:5].tolist()}", flush=True)
    
    if verbose:
        print(f"  Revenue: {-rev.item():.6f}", flush=True)
    return -rev

# テスト時
@torch.no_grad()

# テスト時はSoftMaxではなく、argmaxを使っている。
def infer_choice(flow, v, menu: List[MenuElement], t_grid: torch.Tensor) -> int:
    # 推論時は argmax（本文 Sec.3.3 / Eq.(23) のハード版）
    U = torch.stack([utility_element(flow, v, elem, t_grid) for elem in menu])  # (K,)
    return int(torch.argmax(U).item())
