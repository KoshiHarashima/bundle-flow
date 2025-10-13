# bf/data.py
# 参照: 初期分布MoG (Eq.13), 束近傍のガウス (Eq.14), 効用の厳密評価に向けた有限支持 (Eq.21)。
# CATSのXOR出力は「ダミー項目」で単一入札者を識別（Sec.5.1）し、95/5でtrain/testを分割（p.15）。:contentReference[oaicite:2]{index=2}

import os, glob, re, random
from typing import Dict, List, Tuple, Optional
import torch
from .valuation import XORValuation

# ---------- CATS parser -------------------------------------------------------

def _parse_line_to_bundle_and_price(line: str, m: int) -> List[Tuple[str, List[int], float]]:
    """
    CATS出力の1行から [(dummy_id, items<=m, price)] を抽出。
    - バンドル: { ... } / [ ... ] 内のトークンから item と dummy を抽出
    - 価格: 行内の数値のうち末尾の浮動小数を採用（慣例的に価格が末尾に来るケースを想定）
    注: CATSのフォーマット差異に対して正規表現でロバストに抽出する。
    """
    # 角/波括弧で囲まれたグループを取得
    groups = re.findall(r'[\{\[]([^}\]]+)[}\]]', line)
    out = []
    # 行の数値（小数優先）を抽出し最後を価格とみなす
    nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)
    price = float(nums[-1]) if nums else 0.0
    for g in groups:
        tokens = re.findall(r'[A-Za-z]\w+|\d+', g)
        items = []
        dummy = None
        for t in tokens:
            if t.isdigit():
                it = int(t)
                if 1 <= it <= m:
                    items.append(it)
                else:
                    # mを超える整数をダミーと見做す場合もある
                    dummy = dummy or f'd{it}'
            else:
                # 文字から始まるトークンはダミー識別子候補（例: d123）
                if dummy is None and re.match(r'^[A-Za-z]\w*$', t):
                    dummy = t
        if items:
            out.append((dummy or 'd0', sorted(set(items)), price))
    return out

def parse_cats_output_file(path: str, m: int) -> Dict[str, XORValuation]:
    """
    1ファイル → {dummy_id: XORValuation}
    CATSは「bundle-bidのペアを出力し、同一ダミーが同一入札者のXORを構成」（Sec.5.1）。
    """
    atoms_by_dummy: Dict[str, List[Tuple[List[int], float]]] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for dummy, items, price in _parse_line_to_bundle_and_price(line, m):
                atoms_by_dummy.setdefault(dummy, []).append((items, float(price)))
    vals: Dict[str, XORValuation] = {}
    for dummy, atoms in atoms_by_dummy.items():
        vals[dummy] = XORValuation.from_bundle_list(m, atoms)
    return vals

def load_cats_dir(dir_or_glob: str, m: int, keep_dummy: Optional[str] = None,
                  max_files: Optional[int] = None, shuffle: bool = True) -> List[XORValuation]:
    """
    ディレクトリ/グロブからCATSファイル群を読み込み、単一入札者XORの配列を返す（Sec.5.1）。:contentReference[oaicite:4]{index=4}
    keep_dummy を指定すれば、そのダミーIDに一致する評価のみ抽出（論文の「一貫したダミーIDで抽出」に対応）。
    """
    paths = sorted(glob.glob(dir_or_glob))
    if shuffle:
        random.shuffle(paths)
    if max_files is not None:
        paths = paths[:max_files]

    V: List[XORValuation] = []
    for p in paths:
        vs = parse_cats_output_file(p, m)
        if keep_dummy is not None:
            if keep_dummy in vs:
                V.append(vs[keep_dummy])
        else:
            V.extend(vs.values())
    return V

def train_test_split(V: List[XORValuation], train_ratio: float = 0.95, seed: int = 0):
    """
    95/5 分割（Sec.5.1の設定）。
    """
    rng = random.Random(seed)
    idx = list(range(len(V)))
    rng.shuffle(idx)
    k = int(len(V) * train_ratio)
    train = [V[i] for i in idx[:k]]
    test  = [V[i] for i in idx[k:]]
    return train, test

# ---------- Synthetic XOR generators  ------------------------

def gen_uniform_iid_xor(m: int, a: int, low: float = 0.0, high: float = 1.0, seed: Optional[int] = None, 
                        atom_size_mode: str = "small") -> XORValuation:
    """
    [0.0, 1.0]のi.i.d.でvaluationが出る場合についてのデータを作成する
    
    atom_size_mode:
        "small": p=0.1で各アイテム採択（平均5個、CATS風）
        "medium": p=0.2で各アイテム採択（平均10個）
        "large": p=0.5で各アイテム採択（平均m/2個、元の実装）
        "uniform_3_8": サイズを3-8の一様分布で選択
    """
    rng = random.Random(seed)
    atoms: List[Tuple[List[int], float]] = []
    seen = set()
    
    while len(atoms) < a:
        # 原子サイズの生成方法を選択（学習信号を出すための修正）
        if atom_size_mode == "small":
            # 小さい原子を作る: 期待サイズ 5–8
            import numpy as np
            k = max(1, int(np.random.geometric(p=0.2)))  # 期待 ~5
            items = rng.sample(range(1, m + 1), k=min(k, m))
            S = sorted(items)
        elif atom_size_mode == "medium":
            # 中程度の原子: 期待サイズ 8–15
            import numpy as np
            k = max(1, int(np.random.geometric(p=0.15)))  # 期待 ~6.7
            items = rng.sample(range(1, m + 1), k=min(k, m))
            S = sorted(items)
        elif atom_size_mode == "uniform_3_8":
            # サイズを3-8の一様分布で選択
            k = rng.randint(3, 8)
            S = rng.sample(range(1, m + 1), k)
        else:  # "large" (元の実装)
            # p=0.5で各アイテム採択（平均m/2個）
            S = [i for i in range(1, m + 1) if rng.random() < 0.5]
        
        if not S:
            continue
        key = tuple(sorted(S))
        if key in seen:
            continue
        seen.add(key)
        price = rng.uniform(low, high)
        atoms.append((list(key), price))
    
    return XORValuation.from_bundle_list(m, atoms)

# ---------- 小道具（学習補助） ---------------------------------------------------

def gaussian_around_bundle(s_bool: torch.Tensor, sigma_z: float) -> torch.Tensor:
    """
    束ベクトル s∈{0,1}^m の近傍ガウス N(s, σ_z^2 I_m) をサンプル（Eq.14）。:contentReference[oaicite:7]{index=7}
    """
    return s_bool + sigma_z * torch.randn_like(s_bool)
