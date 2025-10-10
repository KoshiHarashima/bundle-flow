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

def make_null_element(m: int) -> MenuElement:
    # ゼロ配分・ゼロ価格（IR 用のヌル要素）
    elem = MenuElement(m, D=1)
    with torch.no_grad():
        elem.beta.fill_(0.0)
        elem.logits.fill_(0.0)
        elem.mus.zero_()
    for p in elem.parameters():
        p.requires_grad_(False)
    return elem

# ---- Stage 2: 効用・損失 ------------------------------------------------

def utility_element(flow, v, elem: MenuElement, t_grid: torch.Tensor) -> torch.Tensor:
    # u^(k)(v) = Σ_d v(s(μ_d^(k))) · w_d^(k) · exp{-Tr[Q(μ_d^(k))]∫η} − β^(k)  （Eq.(21)）
    w = elem.weights                             # (D,)
    sT = flow.flow_forward(elem.mus, t_grid)     # (D,m)  （Eq.(20)）
    s = flow.round_to_bundle(sT)                 # (D,m)  （Eq.(21)の s(μ)）
    with torch.no_grad():
        vals = torch.tensor([float(v.value(s[d])) for d in range(s.shape[0])],
                            device=s.device, dtype=s.dtype)  # v(s)
    weight = flow.density_weight(elem.mus, t_grid)           # (D,)  （Eq.(12)指数）
    return (vals * w * weight).sum() - elem.beta

def utilities_matrix(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor) -> torch.Tensor:
    # U[b,k] = u^(k)(v_b)  （Eq.(21) を全組で）
    out = []
    for v in V:
        row = [utility_element(flow, v, elem, t_grid) for elem in menu]
        out.append(torch.stack(row))
    return torch.stack(out)                      # (B,K)

def soft_assignment(U: torch.Tensor, lam: float) -> torch.Tensor:
    # z^(k)(v) = SoftMax_k( λ · u^(k)(v) )  （Eq.(23)）
    return torch.softmax(lam * U, dim=1)

def revenue_loss(flow, V: List, menu: List[MenuElement], t_grid: torch.Tensor, lam: float) -> torch.Tensor:
    # LRev = -(1/|V|) Σ_v Σ_k z_k(v) β_k  （Eq.(22)）
    U = utilities_matrix(flow, V, menu, t_grid)          # (B,K)
    Z = soft_assignment(U, lam)                          # (B,K)
    beta = torch.stack([elem.beta for elem in menu])     # (K,)
    rev = (Z * beta.unsqueeze(0)).sum(dim=1).mean()
    return -rev

@torch.no_grad()
def infer_choice(flow, v, menu: List[MenuElement], t_grid: torch.Tensor) -> int:
    # 推論時は argmax（本文 Sec.3.3 / Eq.(23) のハード版）
    U = torch.stack([utility_element(flow, v, elem, t_grid) for elem in menu])  # (K,)
    return int(torch.argmax(U).item())
