# bf/flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Nets --------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, m: int, hidden: int = 128, depth: int = 3, init_scale: float = 0.01, 
                 use_spectral_norm: bool = True, q_mode: str = 'full'):
        super().__init__()
        self.m = m
        self.init_scale = init_scale
        self.q_mode = q_mode  # 'diag' または 'full'
        
        # ネットワーク構築
        layers, in_dim = [], m
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden)
            # スペクトル正規化を適用（Lipschitz制御）
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear, n_power_iterations=1)
            layers += [linear, nn.Tanh()]
            in_dim = hidden
        
        # 出力層：diagモードなら m 次元、fullモードなら m*m 次元
        out_dim = m if q_mode == 'diag' else m * m
        final_linear = nn.Linear(in_dim, out_dim)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear, n_power_iterations=1)
        layers += [final_linear]
        
        self.net = nn.Sequential(*layers)
        
        # 小さい初期値で初期化（スケール制御）
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, s0: torch.Tensor) -> torch.Tensor:
        # Q(s0) ∈ R^{m×m}
        B = s0.shape[0]
        out = self.net(s0)
        
        if self.q_mode == 'diag':
            # 対角モード：d(s0) → diag(d)
            d = 0.1 * torch.tanh(out)  # (B, m)
            Q = torch.diag_embed(d)    # (B, m, m)
        else:
            # フルモード：全行列を出力
            Q = 0.1 * torch.tanh(out.view(B, self.m, self.m))
        
        return Q
    
    def switch_to_full_mode(self):
        """対角モードからフルモードへ切り替え（ウォームスタート後に呼び出す）"""
        if self.q_mode == 'diag':
            self.q_mode = 'full'
            # 出力層を再構築（m → m*m）
            old_net = list(self.net.children())
            # 最後の層を取得
            old_final = old_net[-1]
            
            # 新しい出力層を作成
            use_sn = hasattr(old_final, 'weight_orig')  # spectral_normが適用されているか
            new_final = nn.Linear(old_final.in_features, self.m * self.m)
            
            # 重みを初期化
            nn.init.normal_(new_final.weight, std=self.init_scale)
            if new_final.bias is not None:
                nn.init.zeros_(new_final.bias)
            
            if use_sn:
                new_final = nn.utils.spectral_norm(new_final, n_power_iterations=1)
            
            # ネットワークを再構築
            old_net[-1] = new_final
            self.net = nn.Sequential(*old_net)
            
            print(f"[QNet] Switched from 'diag' to 'full' mode")


class EtaNet(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 2, init_scale: float = 0.01, 
                 c_eta: float = 2.0, temperature: float = 1.0, use_layernorm: bool = False):
        super().__init__()
        layers, in_dim = [], 1
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden)]
            if use_layernorm:
                layers += [nn.LayerNorm(hidden)]  # 安定化
            layers += [nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.init_scale = init_scale
        self.c_eta = c_eta  # tanhゲートのスケール係数
        self.temperature = temperature  # 温度付きtanh
        
        # 小さい初期値で初期化（スケール制御）
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # η(t) = c_eta * tanh(eta_raw / τ)（温度付きtanhで線形領域を拡大）
        if t.dim() == 0:
            t = t[None]
        raw = self.net(t.unsqueeze(-1)).squeeze(-1)
        # 温度で除算してtanhの線形領域を広げる
        return self.c_eta * torch.tanh(raw / self.temperature)


# ---- Flow core ---------------------------------------------------------

class FlowModel(nn.Module):
    def __init__(self, m: int, use_spectral_norm: bool = True, q_mode: str = 'full', 
                 c_eta: float = 2.0, eta_init_scale: float = 0.01,
                 eta_temperature: float = 1.0, use_eta_layernorm: bool = False,
                 eta_integral_clip: float = 10.0):
        super().__init__()
        self.m = m
        self.Q = QNet(m, use_spectral_norm=use_spectral_norm, q_mode=q_mode)
        self.eta = EtaNet(init_scale=eta_init_scale, c_eta=c_eta, 
                         temperature=eta_temperature, use_layernorm=use_eta_layernorm)
        self.eta_integral_clip = eta_integral_clip  # ∫ηの上限クリップ

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
        integral = torch.trapz(eta_vals, t_grid)           # ()
        # 物理的にクリップして発散を防ぐ
        if self.eta_integral_clip > 0:
            integral = torch.clamp(integral, -self.eta_integral_clip, self.eta_integral_clip)
        return integral

    def log_density_weight(self, s0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        # log( exp( - Tr(Q(s0)) ∫ η ) ) = -Tr(Q(s0)) * ∫ η  （Eq.(12) のlog形、数値安定）
        Q = self.Q(s0)
        trQ = Q.diagonal(dim1=-2, dim2=-1).sum(-1)        # (B,)
        integ = self.eta_integral(t_grid)                  # ()
        return -trQ * integ                                # (B,)

    def density_weight(self, s0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        # exp( - Tr(Q(s0)) ∫ η )  （Eq.(12) の指数形）
        # 注：数値安定性のため、内部では log_density_weight を使用
        return torch.exp(self.log_density_weight(s0, t_grid))  # (B,)

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

    def compute_trace_q_penalty(self, s0: torch.Tensor) -> torch.Tensor:
        # Tr(Q(s0))² の罰則（発散の制御）
        Q = self.Q(s0)
        trQ = Q.diagonal(dim1=-2, dim2=-1).sum(-1)        # (B,)
        return (trQ ** 2).mean()
    
    def compute_jacobian_penalty(self, s0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Jacobian罰則：λ_J * E[(η(t))^2 * ||Q(s0)||_F^2]
        軌道短縮・ベクトル場の滑らかさを促進"""
        Q = self.Q(s0)                                     # (B, m, m)
        eta = self.eta(t)                                  # (B,)
        Q_norm_sq = (Q ** 2).sum(dim=(1, 2))              # (B,) ||Q||_F^2
        return (eta ** 2 * Q_norm_sq).mean()
    
    def compute_kinetic_penalty(self, s0: torch.Tensor, s_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Kinetic罰則：λ_K * E[||η(t) * Q(s0) * s_t||^2]
        ベクトル場の大きさを制御し、密度重みの暴騰を抑制"""
        phi = self.phi(t, s_t, s0)                         # (B, m)
        return (phi ** 2).sum(dim=-1).mean()
    
    def compute_stability_regularization(
        self, 
        s0: torch.Tensor, 
        s_t: torch.Tensor, 
        t: torch.Tensor,
        lambda_j: float = 1e-3,
        lambda_k: float = 1e-3,
        lambda_tr: float = 1e-4
    ) -> dict:
        """Stage-1安定化正則化の全項を計算"""
        jac_penalty = self.compute_jacobian_penalty(s0, t)
        kin_penalty = self.compute_kinetic_penalty(s0, s_t, t)
        tr_penalty = self.compute_trace_q_penalty(s0)
        
        total_reg = lambda_j * jac_penalty + lambda_k * kin_penalty + lambda_tr * tr_penalty
        
        return {
            'total_reg': total_reg,
            'jacobian_penalty': jac_penalty.item(),
            'kinetic_penalty': kin_penalty.item(),
            'trace_penalty': tr_penalty.item()
        }
    
    def compute_statistics(
        self,
        s0: torch.Tensor,
        sT: torch.Tensor,
        t: torch.Tensor,
        s_t: torch.Tensor,
        pred: torch.Tensor
    ) -> dict:
        # 学習モニタリング用の統計量を計算
        with torch.no_grad():
            Q = self.Q(s0)
            trQ = Q.diagonal(dim1=-2, dim2=-1).sum(-1)
            eta_vals = self.eta(t)
            
            stats = {
                'phi_norm': pred.norm(dim=-1).mean().item(),
                'trace_q_mean': trQ.mean().item(),
                'trace_q_std': trQ.std().item(),
                'eta_mean': eta_vals.mean().item(),
                'eta_std': eta_vals.std().item(),
                's0_min': s0.min().item(),
                's0_max': s0.max().item(),
                's0_mean': s0.mean().item(),
                'sT_min': sT.min().item(),
                'sT_max': sT.max().item(),
                'sT_mean': sT.mean().item(),
                'Q_norm': Q.norm(dim=(1,2)).mean().item(),
            }
        return stats

    def rectified_flow_loss(
        self,
        B: int,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
        weights: torch.Tensor,
        sigma_z: float = 0.05,
        trace_penalty_weight: float = 0.0,
        lambda_j: float = 0.0,
        lambda_k: float = 0.0,
        lambda_tr: float = 0.0,
        return_stats: bool = False,
    ):
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
        
        loss = (target - pred).pow(2).sum(-1).mean()
        
        # 安定化正則化（新しい方式）
        if lambda_j > 0 or lambda_k > 0 or lambda_tr > 0:
            reg_dict = self.compute_stability_regularization(
                s0, s_t, t, lambda_j=lambda_j, lambda_k=lambda_k, lambda_tr=lambda_tr
            )
            loss = loss + reg_dict['total_reg']
        
        # 正則化項を追加（後方互換性：旧trace_penalty_weight）
        elif trace_penalty_weight > 0:
            trace_penalty = self.compute_trace_q_penalty(s0)
            loss = loss + trace_penalty_weight * trace_penalty
        
        if return_stats:
            stats = self.compute_statistics(s0, sT, t, s_t, pred)
            # 正則化項の統計を追加
            if lambda_j > 0 or lambda_k > 0 or lambda_tr > 0:
                stats.update({
                    'jacobian_penalty': reg_dict['jacobian_penalty'],
                    'kinetic_penalty': reg_dict['kinetic_penalty'],
                    'trace_penalty': reg_dict['trace_penalty']
                })
            return loss, stats
        return loss
