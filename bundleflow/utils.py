# bf/utils.py
# Utility helpers for BundleFlow / Bundle-RochetNet
# - 時間グリッド: Eq.(12),(20)
# - SoftMax温度(λ)スケジュール: Eq.(23)
# - 再現性シード
# - チェックポイント I/O
# - デバイス移動
# - ハードargmax評価: Eq.(21)→(23)

from __future__ import annotations
import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---- 時間グリッド -----------------------------------------------------------
def make_t_grid(T: float = 1.0, steps: int = 50, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Eq.(12),(20) の数値積分/前進解法で用いる時間グリッド t ∈ [0,T]。
    """
    return torch.linspace(0.0, float(T), steps=max(2, int(steps)), device=device)

# ---- SoftMax 温度(λ)スケジュール -------------------------------------------
def lambda_schedule(step: int,
                    total: int,
                    start: float = 1e-3,
                    end: float = 2e-1,
                    mode: str = "linear") -> float:
    """
    学習時の SoftMax パラメータ λ をスケジュール（Eq.(23)）。
    mode: "linear" | "cosine"
    """
    if total <= 0:
        return end
    x = max(0.0, min(1.0, step / float(total)))
    if mode == "cosine":
        # 余弦ウォームアップ: start → end
        k = 0.5 * (1.0 - math.cos(math.pi * x))
        return (1 - k) * start + k * end
    # 既定: 線形
    return (1 - x) * start + x * end

# ---- 再現性 -----------------------------------------------------------------
def seed_all(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    乱数シードの固定（Python/NumPy/PyTorch）。
    再現性を保証するための包括的なシード設定。
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    
    # 決定論的設定（デフォルトON）
    if deterministic_cudnn:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---- チェックポイント I/O ---------------------------------------------------
def save_ckpt(path: str,
              flow: Optional[nn.Module] = None,
              menu: Optional[List[nn.Module]] = None,
              extra: Optional[Dict[str, Any]] = None) -> None:
    """
    flow: FlowModel（Stage 1; Eq.(9),(15)-(17),(20)）
    menu: List[MenuElement]（Stage 2; Eq.(21)-(23)）
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {"extra": extra or {}}
    if flow is not None:
        payload["flow_state_dict"] = flow.state_dict()
        # 便宜的にアイテム数 m を含めたい場合は extra に入れる想定
    if menu is not None:
        # MenuElement は (beta, logits, mus) を保存（学習/推論に十分）
        payload["menu"] = [
            {
                "beta": float(elem.beta.detach().cpu().item()),
                "logits": elem.logits.detach().cpu(),
                "mus": elem.mus.detach().cpu(),
            }
            for elem in menu
        ]
    torch.save(payload, path)

def load_ckpt(path: str,
              flow: Optional[nn.Module] = None,
              menu: Optional[List[nn.Module]] = None,
              map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    保存形式に応じて flow と menu を復元。戻り値は extra を含む辞書。
    """
    ckpt = torch.load(path, map_location=map_location or "cpu")
    if flow is not None and "flow_state_dict" in ckpt:
        flow.load_state_dict(ckpt["flow_state_dict"])
    if menu is not None and "menu" in ckpt:
        assert len(menu) == len(ckpt["menu"]), "menu 長が一致しません。"
        for elem, blob in zip(menu, ckpt["menu"]):
            with torch.no_grad():
                elem.beta.copy_(torch.tensor(blob["beta"], dtype=elem.beta.dtype))
                elem.logits.copy_(blob["logits"].to(elem.logits.device, dtype=elem.logits.dtype))
                elem.mus.copy_(blob["mus"].to(elem.mus.device, dtype=elem.mus.dtype))
    return ckpt.get("extra", {})

# ---- デバイス移動 -----------------------------------------------------------
def to_device(obj: Any, device: torch.device) -> Any:
    """
    Tensor/Module/リスト/タプル/辞書を再帰的に device へ移動。
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        c = [to_device(v, device) for v in obj]
        return type(obj)(c) if isinstance(obj, tuple) else c
    return obj

# ---- Gumbel-Softmax + Straight-Through Estimator ------------------------------
def gumbel_softmax(logits: torch.Tensor, 
                   tau: float = 1.0, 
                   hard: bool = False, 
                   dim: int = -1,
                   eps: float = 1e-10) -> torch.Tensor:
    """
    Gumbel-Softmax sampling (Jang et al. 2017, Maddison et al. 2017)
    
    カテゴリカル分布からのサンプリングを微分可能にする手法。
    
    Args:
        logits: (*, num_classes) 未正規化ログ確率
        tau: 温度パラメータ（低いほどhard argmaxに近づく）
        hard: Trueならforward時にhard one-hot、Falseならsoft確率
        dim: softmax次元
        eps: 数値安定性のための小さい値
    
    Returns:
        sampled: (*, num_classes) one-hot（hard時）またはソフト確率
        
    Example:
        >>> logits = torch.tensor([[1.0, 2.0, 0.5]])
        >>> y = gumbel_softmax(logits, tau=0.1, hard=True)
        >>> y  # tensor([[0., 1., 0.]])  ← argmax=1
    """
    # Gumbelノイズを追加
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    gumbels = (logits + gumbels) / tau
    y_soft = torch.softmax(gumbels, dim=dim)
    
    if hard:
        # Straight-through estimator
        # Forward: hard one-hot
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        
        # Backward: softmaxの勾配を使用
        # この"マジック"により、forward=hard, backward=softが実現される
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    
    return ret


def sample_gumbel(shape: Tuple[int, ...], 
                  device: Optional[torch.device] = None,
                  eps: float = 1e-10) -> torch.Tensor:
    """
    Gumbel(0, 1)分布からのサンプリング
    
    Gumbel分布: F(x) = exp(-exp(-x))
    
    Args:
        shape: 出力のshape
        device: デバイス
        eps: 数値安定性
        
    Returns:
        Gumbel samples
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


# ---- Hard argmax 収益（推論） ----------------------------------------------
@torch.no_grad()
def eval_revenue_argmax(V: List[Any],
                        flow: nn.Module,
                        menu: List[nn.Module],
                        t_grid: torch.Tensor) -> float:
    """
    テスト時の平均収益を計算（Hard argmax）。
    - 各 v について u^{(k)}(v)（Eq.(21)）を計算 → argmax（Eq.(23) の極限）で選択。
    - 収益は選択要素の価格 β。IR を満たすヌル要素が含まれていれば負収益は自然に回避。
    """
    from bf.menu import utility_element  # 循環 import 回避のためローカル import
    total = 0.0
    for v in V:
        U = torch.stack([utility_element(flow, v, elem, t_grid) for elem in menu])  # (K,)
        k = int(torch.argmax(U).item())
        beta_k = float(menu[k].beta.item())
        # 念のため負効用ならゼロ収益にクリップ（ヌル要素があれば常に選ばれるはず）
        total += beta_k if float(U[k].item()) >= 0.0 else 0.0
    return total / max(1, len(V))
