# bf/valuation.py
# XOR言語の意味: v(S) は XOR原子 (S', bid) のうち S'⊆S を満たす価格の最大値（Sec.2）。:contentReference[oaicite:8]{index=8}
# 本v(S)は効用 u^(k)(v) の項 v(s(μ_d))（Eq.21）や 期待効用（Eq.19）に直接用いる。:contentReference[oaicite:9]{index=9}

from typing import Iterable, List, Sequence, Tuple
import torch

def _set_to_mask(S: Iterable[int], m: int) -> int:
    mask = 0
    for i in S:
        if 1 <= i <= m:
            mask |= (1 << (i - 1))
    return mask

def _tensor_to_mask(s_bool: torch.Tensor) -> int:
    # s_bool ∈ {0,1}^m（最終束は I(φ(T,·)≥0.5)；Eq.19, Eq.20 の後に丸め）:contentReference[oaicite:10]{index=10}
    m = s_bool.numel()
    mask = 0
    sb = (s_bool > 0.5).to(torch.int32).view(-1)
    for i in range(m):
        if int(sb[i].item()) == 1:
            mask |= (1 << i)
    return mask

class XORValuation:
    def __init__(self, m: int, atoms_mask_price: List[Tuple[int, float]]):
        """
        atoms_mask_price: [(mask(S_j), price_j)]
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
        arr = [(_set_to_mask(S, m), float(price)) for S, price in atoms]
        return cls(m, arr)

    # ---- 値関数 v(S) ----------------------------------------------------------
    
    def value(self, s_bool: Sequence[float] | torch.Tensor) -> float:
        """
        v(S) = max_{(T, bid)} { bid : T ⊆ S }  （XORの意味；Sec.2）
        ※ 本メソッドは Eq.(21) の v(s(μ_d))、および Eq.(19) の v(s) に相当
        """
        if isinstance(s_bool, torch.Tensor):
            Sm = _tensor_to_mask(s_bool)
        else:
            # list/np.ndarray想定
            t = torch.tensor(s_bool, dtype=torch.float32)
            Sm = _tensor_to_mask(t)
        val = 0.0
        for mk, p in self.atoms:
            if mk & (~Sm) == 0:  # T⊆S 判定（ビット包含）
                if p > val:
                    val = p
        return float(val)

    def batch_value(self, S_bool: torch.Tensor) -> torch.Tensor:
        """
        バッチ版 v(S_b)（Eq.(21), Eq.(19) の高速化）。S_bool: (B,m) ∈ {0,1}^m
        シンプルなループ実装（確実に動作する）
        """
        B, m = S_bool.shape
        device = S_bool.device
        
        # 各行をマスクに変換
        S_cpu = S_bool.cpu()
        masks = []
        for b in range(B):
            masks.append(_tensor_to_mask(S_cpu[b]))
        
        # 出力を初期化
        out = torch.zeros(B, dtype=torch.float32, device=device)
        
        # 各バッチについて、value()と同じロジックを実行
        for b in range(B):
            val = 0.0
            for atom_mask, price in self.atoms:
                # T⊆S の判定: (atom_mask & ~Sm) == 0
                if (atom_mask & (~masks[b])) == 0:
                    if price > val:
                        val = price
            out[b] = val
        
        return out
