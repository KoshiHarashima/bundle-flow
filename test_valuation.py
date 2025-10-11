#!/usr/bin/env python3
"""Valuationのマッチングをテスト"""

import torch
from bf.data import gen_uniform_iid_xor
from bf.valuation import _tensor_to_mask

# 小さいデータセットでテスト
m = 10  # 10個のアイテム
a = 5   # 5個のatom

print("=" * 60)
print("XOR Valuation Matching Test")
print("=" * 60)

# Valuationを生成
v = gen_uniform_iid_xor(m, a=a, seed=42)

print(f"\nGenerated {len(v.atoms)} atoms:")
for i, (mask, price) in enumerate(v.atoms):
    items = [j for j in range(m) if (mask & (1 << j)) != 0]
    print(f"  Atom {i}: items={items}, price={price:.4f}, mask={mask}, bin={bin(mask)}")

# テストbundleを作成
print(f"\nTest bundles:")

# Bundle 1: 全て1
bundle1 = torch.ones(m)
mask1 = _tensor_to_mask(bundle1)
val1 = v.value(bundle1)
items1 = [j for j in range(m) if (mask1 & (1 << j)) != 0]
print(f"  Bundle 1 (all ones): items={items1}, mask={mask1}, value={val1:.4f}")

# Bundle 2: 交互に1
bundle2 = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(m)])
mask2 = _tensor_to_mask(bundle2)
val2 = v.value(bundle2)
items2 = [j for j in range(m) if (mask2 & (1 << j)) != 0]
print(f"  Bundle 2 (alternating): items={items2}, mask={mask2}, value={val2:.4f}")

# Bundle 3: 最初の5個
bundle3 = torch.tensor([1.0 if i < 5 else 0.0 for i in range(m)])
mask3 = _tensor_to_mask(bundle3)
val3 = v.value(bundle3)
items3 = [j for j in range(m) if (mask3 & (1 << j)) != 0]
print(f"  Bundle 3 (first 5): items={items3}, mask={mask3}, value={val3:.4f}")

# 各atomがどれかのbundleにマッチするか確認
print(f"\nMatching check:")
for i, (atom_mask, price) in enumerate(v.atoms):
    match1 = (atom_mask & (~mask1)) == 0
    match2 = (atom_mask & (~mask2)) == 0
    match3 = (atom_mask & (~mask3)) == 0
    atom_items = [j for j in range(m) if (atom_mask & (1 << j)) != 0]
    print(f"  Atom {i} (items={atom_items}): match_all={match1}, match_alt={match2}, match_first5={match3}")

print("\n" + "=" * 60)
print("If ALL atoms show False for all bundles, there's a bug!")
print("=" * 60)

