# bundleflow/valuation/valuation.py
"""
XORValuation: 評価関数 v(b) やCATSなどデータ分布の実装（外生）

XOR言語の意味: v(S) は XOR原子 (S', bid) のうち S'⊆S を満たす価格の最大値（Sec.2）.
本v(S)は効用 u^(k)(v) の項 v(s(μ_d))（Eq.21）や 期待効用（Eq.19）に直接用いる.
"""

from typing import Iterable, List, Sequence, Tuple
import torch

def _set_to_mask(S: Iterable[int], m: int) -> int:
    """
    集合をビットマスクに変換
    
    Args:
        S: 集合（アイテムのインデックス）
        m: 商品数
        
    Returns:
        mask: ビットマスク
    """
    mask = 0
    for i in S:
        if 1 <= i <= m:
            mask |= (1 << (i - 1))
    return mask

def _tensor_to_mask(s_bool: torch.Tensor) -> int:
    """
    テンソルをビットマスクに変換
    
    Args:
        s_bool: 束テンソル ∈ {0,1}^m（最終束は I(φ(T,·)≥0.5)；Eq.19, Eq.20 の後に丸め）
        
    Returns:
        mask: ビットマスク
    """
    m = s_bool.numel()
    mask = 0
    sb = (s_bool > 0.5).to(torch.int32).view(-1)
    for i in range(m):
        if int(sb[i].item()) == 1:
            mask |= (1 << i)
    return mask

class XORValuation:
    """
    XOR評価関数
    
    目的: 束bに対する価値v(b)を計算
    記号: v(b) = max_{(T, bid)} { bid : T ⊆ b }
    """
    
    def __init__(self, m: int, atoms_mask_price: List[Tuple[int, float]]):
        """
        XOR評価関数を初期化
        
        Args:
            m: 商品数
            atoms_mask_price: [(mask(S_j), price_j)] のリスト
        """
        self.m = m
        # 同じマスクの重複は高値のみ保持
        best = {}
        for mk, p in atoms_mask_price:
            best[mk] = max(p, best.get(mk, float('-inf')))
        # 降順にしておく（わずかな早期打ち切りの助け）
        self.atoms = sorted(best.items(), key=lambda x: -x[1])

    @classmethod
    def from_bundle_list(cls, m: int, atoms: List[Tuple[List[int], float]]) -> "XORValuation":
        """
        束リストからXOR評価関数を作成
        
        Args:
            m: 商品数
            atoms: [(S_j, price_j)] のリスト
            
        Returns:
            valuation: XORValuationインスタンス
        """
        arr = [(_set_to_mask(S, m), float(price)) for S, price in atoms]
        return cls(m, arr)

    # ---- 値関数 v(S) ----------------------------------------------------------
    
    def value(self, s_bool: Sequence[float] | torch.Tensor, debug: bool = False) -> float:
        """
        束の価値を計算
        
        v(S) = max_{(T, bid)} { bid : T ⊆ S }  （XORの意味；Sec.2）
        ※ 本メソッドは Eq.(21) の v(s(μ_d))、および Eq.(19) の v(s) に相当
        
        Args:
            s_bool: 束（テンソルまたはリスト）
            debug: デバッグ情報を出力するかどうか
            
        Returns:
            value: 束の価値
        """
        if isinstance(s_bool, torch.Tensor):
            Sm = _tensor_to_mask(s_bool)
        else:
            # list/np.ndarray想定
            t = torch.tensor(s_bool, dtype=torch.float32)
            Sm = _tensor_to_mask(t)
        
        val = 0.0
        matched_count = 0
        for mk, p in self.atoms:
            check = mk & (~Sm)
            is_match = (check == 0)
            if debug:
                print(f"    [value DEBUG] atom_mask={mk}, Sm={Sm}, check={check}, match={is_match}, price={p:.4f}")
            if is_match:  # T⊆S 判定（ビット包含）
                matched_count += 1
                if p > val:
                    val = p
        
        if debug:
            print(f"    [value DEBUG] Total matched: {matched_count}/{len(self.atoms)}, final value: {val:.4f}")
        
        return float(val)

    def batch_value(self, S_bool: torch.Tensor) -> torch.Tensor:
        """
        バッチ版 v(S_b)（Eq.(21), Eq.(19) の高速化）
        
        Args:
            S_bool: 束のバッチ (B, m) ∈ {0,1}^m
            
        Returns:
            values: 価値のバッチ (B,)
        """
        B, m = S_bool.shape
        device = S_bool.device
        
        # 各bundleについてvalue()を直接呼ぶ
        out = torch.zeros(B, dtype=torch.float32, device=device)
        for b in range(B):
            out[b] = self.value(S_bool[b])
        
        return out
