# bf/flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Nets --------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, m: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers, in_dim = [], m
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, m * m)]
        self.net = nn.Sequential(*layers)
        self.m = m

    def forward(self, s0: torch.Tensor) -> torch.Tensor:
        # Q(s0) ∈ R^{m×m}
        B = s0.shape[0]
        return self.net(s0).view(B, self.m, self.m)


class EtaNet(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers, in_dim = [], 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # η(t)（スカラー出力）
        if t.dim() == 0:
            t = t[None]
        return self.net(t.unsqueeze(-1)).squeeze(-1)


# ---- Flow core ---------------------------------------------------------

class FlowModel(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.m = m
        self.Q = QNet(m)
        self.eta = EtaNet()

    @torch.no_grad()
    def round_to_bundle(self, s_T: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        # s = I(φ(T,s0) ≥ 0.5)  （Eq.(19), (21)）
        return (s_T >= tau).to(s_T.dtype)

    def phi(self, t: torch.Tensor, s_t: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        # φ(t,s_t) = η(t)·Q(s0)·s_t  （Eq.(9)）
        Q = self.Q(s0)                             # (B,m,m)
        eta = self.eta(t).view(-1, 1, 1)           # (B,1,1)
        return torch.bmm(eta * Q, s_t.unsqueeze(-1)).squeeze(-1)  # (B,m)

    def divergence(self, t: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        # ∇·φ = η(t)·Tr(Q(s0))  （Eq.(10)–(12)）
        Q = self.Q(s0)
        trQ = Q.diagonal(dim1=-2, dim2=-1).sum(-1)        # (B,)
        return self.eta(t) * trQ                           # (B,)

    def eta_integral(self, t_grid: torch.Tensor) -> torch.Tensor:
        # ∫_0^T η(t) dt （台形則で近似；Eq.(12) のスカラー積分）
        eta_vals = self.eta(t_grid)                        # (T,)
        return torch.trapz(eta_vals, t_grid)               # ()

    def density_weight(self, s0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        # exp( - Tr(Q(s0)) ∫ η )  （Eq.(12) の指数形）
        Q = self.Q(s0)
        trQ = Q.diagonal(dim1=-2, dim2=-1).sum(-1)        # (B,)
        integ = self.eta_integral(t_grid)                  # ()
        return torch.exp(-trQ * integ)                     # (B,)

    def flow_forward(self, s0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        # φ を Euler で前進解法  （Eq.(20) の数値化）
        s = s0.clone()
        for i in range(len(t_grid) - 1):
            t = t_grid[i].expand(s0.shape[0])
            dt = (t_grid[i + 1] - t_grid[i])
            s = s + dt * self.phi(t, s, s0)
        return s

    # ---- Stage 1: Rectified Flow 損失 -----------------------------------
    @staticmethod
    def sample_mog(B: int, mus: torch.Tensor, sigmas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # s0 ~ Σ_d w_d N(μ_d, σ_d^2 I)  （Eq.(13)）
        device, D, m = mus.device, mus.shape[0], mus.shape[1]
        cat = torch.distributions.Categorical(logits=torch.log_softmax(weights, dim=0))
        idx = cat.sample((B,))                              # (B,)
        eps = torch.randn(B, m, device=device)
        s0 = mus[idx] + sigmas[idx].unsqueeze(-1) * eps
        return s0

    def rectified_flow_loss(
        self,
        B: int,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
        weights: torch.Tensor,
        sigma_z: float = 0.05,
    ) -> torch.Tensor:
        # LFlow = E || (s_T - s_0) - φ(t,s_t) ||^2  （Eq.(15)）
        # s_t = t s_T + (1-t) s_0                     （Eq.(16)）
        # s_T ~ N( round(s_0), σ_z^2 I )              （Eq.(14)）
        s0 = self.sample_mog(B, mus, sigmas, weights)      # (B,m)
        s = self.round_to_bundle(s0)                       # (B,m)
        sT = s + sigma_z * torch.randn_like(s0)            # (B,m)
        t = torch.rand(B, device=s0.device)                # t ~ U[0,1]
        s_t = t.unsqueeze(-1) * sT + (1.0 - t).unsqueeze(-1) * s0
        target = sT - s0
        pred = self.phi(t, s_t, s0)                        # （Eq.(17)）
        return (target - pred).pow(2).sum(-1).mean()
