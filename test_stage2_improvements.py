#!/usr/bin/env python3
"""
Stage 2 改善のテストスクリプト
- log_density_weight の数値安定性をテスト
- ウォームスタートと再初期化の動作を確認
"""

import torch
import numpy as np
from bf.flow import FlowModel
from bf.menu import MenuElement, utility_element
from bf.valuation import XORValuation
from bf.data import gen_uniform_iid_xor

def test_log_density_weight():
    """log_density_weightとdensity_weightの一貫性をテスト"""
    print("\n" + "="*60)
    print("Test 1: log_density_weight の数値安定性")
    print("="*60)
    
    m = 10
    flow = FlowModel(m=m)
    t_grid = torch.linspace(0.0, 1.0, steps=25)
    
    # テストケース1: 通常の値
    s0_normal = torch.rand(5, m)
    log_weight = flow.log_density_weight(s0_normal, t_grid)
    weight = flow.density_weight(s0_normal, t_grid)
    weight_from_log = torch.exp(log_weight)
    
    diff = torch.abs(weight - weight_from_log).max().item()
    print(f"通常の値での誤差: {diff:.2e}")
    assert diff < 1e-6, f"誤差が大きすぎます: {diff}"
    print("✓ 通常の値でのテスト成功")
    
    # テストケース2: 極端な値（大きなTr[Q]）
    # Qネットワークを強制的に大きな値を出すように設定
    with torch.no_grad():
        # s0を極端な値に設定してTr[Q]を大きくする
        s0_extreme = torch.ones(5, m) * 10.0  # 極端な入力
    
    log_weight_extreme = flow.log_density_weight(s0_extreme, t_grid)
    weight_extreme = flow.density_weight(s0_extreme, t_grid)
    weight_extreme_from_log = torch.exp(log_weight_extreme)
    
    diff_extreme = torch.abs(weight_extreme - weight_extreme_from_log).max().item()
    print(f"極端な値での誤差: {diff_extreme:.2e}")
    
    # inf/nanのチェック
    assert not torch.isnan(log_weight_extreme).any(), "log_weight に NaN が含まれています"
    assert not torch.isinf(log_weight_extreme).any(), "log_weight に Inf が含まれています"
    assert not torch.isnan(weight_extreme).any(), "weight に NaN が含まれています"
    print("✓ 極端な値でのテスト成功（NaN/Inf なし）")
    
    print(f"\n統計:")
    print(f"  通常の log_weight 範囲: [{log_weight.min().item():.3f}, {log_weight.max().item():.3f}]")
    print(f"  極端な log_weight 範囲: [{log_weight_extreme.min().item():.3f}, {log_weight_extreme.max().item():.3f}]")
    print(f"  通常の weight 範囲: [{weight.min().item():.3e}, {weight.max().item():.3e}]")
    print(f"  極端な weight 範囲: [{weight_extreme.min().item():.3e}, {weight_extreme.max().item():.3e}]")

def test_utility_element_stability():
    """utility_element のlog-sum-exp実装をテスト"""
    print("\n" + "="*60)
    print("Test 2: utility_element の数値安定性")
    print("="*60)
    
    m = 10
    D = 4
    flow = FlowModel(m=m)
    t_grid = torch.linspace(0.0, 1.0, steps=25)
    elem = MenuElement(m, D)
    
    # テスト用の価値関数（XOR）
    v = gen_uniform_iid_xor(m, a=5, low=0.0, high=1.0, seed=42)
    
    # 効用を計算
    try:
        u = utility_element(flow, v, elem, t_grid)
        print(f"効用値: {u.item():.4f}")
        
        # NaN/Infチェック
        assert not torch.isnan(u), "効用が NaN です"
        assert not torch.isinf(u), "効用が Inf です"
        print("✓ 効用計算成功（NaN/Inf なし）")
        
        # 勾配のテスト
        u.backward()
        grad_norm = sum(p.grad.norm().item() for p in elem.parameters() if p.grad is not None)
        print(f"勾配ノルム: {grad_norm:.4f}")
        assert not np.isnan(grad_norm), "勾配が NaN です"
        print("✓ 勾配計算成功")
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        raise

def test_warmstart():
    """ウォームスタート機能をテスト"""
    print("\n" + "="*60)
    print("Test 3: μのウォームスタート")
    print("="*60)
    
    # Stage2のtrain関数からwarmstart_musをインポート
    import sys
    sys.path.insert(0, 'src')
    from train_stage2 import warmstart_mus, build_menu
    
    m = 20
    K = 10
    D = 4
    flow = FlowModel(m=m)
    t_grid = torch.linspace(0.0, 1.0, steps=25)
    menu = build_menu(m, K, D)
    
    # ウォームスタート前のμを保存
    mus_before = [elem.mus.clone() for elem in menu[:-1]]
    
    # ウォームスタート実行
    warmstart_mus(flow, menu, t_grid, n_grid=50, seed=42)
    
    # ウォームスタート後のμをチェック
    mus_after = [elem.mus for elem in menu[:-1]]
    
    # μが変化したことを確認
    changed = False
    for before, after in zip(mus_before, mus_after):
        if not torch.allclose(before, after):
            changed = True
            break
    
    assert changed, "μが変化していません"
    print("✓ ウォームスタートによりμが初期化されました")
    
    # μの範囲チェック
    for k, elem in enumerate(menu[:-1]):
        assert elem.mus.min() >= -0.1, f"要素{k}のμが範囲外（最小値: {elem.mus.min()}）"
        assert elem.mus.max() <= 1.1, f"要素{k}のμが範囲外（最大値: {elem.mus.max()}）"
    print("✓ μの範囲が適切です [0, 1]")
    
    # 多様性のチェック（異なる束を持っているか）
    all_mus = torch.cat([elem.mus for elem in menu[:-1]], dim=0)  # (K*D, m)
    print(f"全μの統計:")
    print(f"  平均: {all_mus.mean().item():.3f}")
    print(f"  標準偏差: {all_mus.std().item():.3f}")
    print(f"  範囲: [{all_mus.min().item():.3f}, {all_mus.max().item():.3f}]")

def test_reinit():
    """再初期化機能をテスト"""
    print("\n" + "="*60)
    print("Test 4: 弱い再初期化")
    print("="*60)
    
    # Stage2のtrain関数からreinit_unused_elementsをインポート
    import sys
    sys.path.insert(0, 'src')
    from train_stage2 import reinit_unused_elements, build_menu
    
    m = 20
    K = 10
    D = 4
    flow = FlowModel(m=m)
    t_grid = torch.linspace(0.0, 1.0, steps=25)
    menu = build_menu(m, K, D)
    
    # 模擬選択確率行列（いくつかの要素はほとんど選ばれない）
    B = 50
    Z = torch.rand(B, K+1)
    Z[:, :3] = 0.001  # 最初の3要素はほとんど選ばれない
    Z = Z / Z.sum(dim=1, keepdim=True)  # 正規化
    
    print(f"選択確率の平均（最初の5要素）: {Z[:, :5].mean(dim=0).tolist()}")
    
    # 再初期化実行
    n_reinit = reinit_unused_elements(flow, menu, Z, t_grid, threshold=0.01)
    
    assert n_reinit >= 3, f"少なくとも3要素が再初期化されるべきですが、{n_reinit}個でした"
    print(f"✓ {n_reinit}個の未使用要素が再初期化されました")

def test_numerical_stability_extreme():
    """極端な条件下での数値安定性をテスト"""
    print("\n" + "="*60)
    print("Test 5: 極端な条件下での数値安定性")
    print("="*60)
    
    m = 10
    D = 8
    flow = FlowModel(m=m)
    t_grid = torch.linspace(0.0, 1.0, steps=25)
    
    # 極端な重みを持つ要素
    elem = MenuElement(m, D)
    with torch.no_grad():
        # ロジットを極端な値に設定
        elem.logits[0] = 100.0   # 非常に大きい
        elem.logits[1] = -100.0  # 非常に小さい
        elem.logits[2:] = 0.0
    
    v = gen_uniform_iid_xor(m, a=5, low=0.0, high=1.0, seed=42)
    
    try:
        u = utility_element(flow, v, elem, t_grid)
        print(f"極端な重みでの効用値: {u.item():.4f}")
        
        assert not torch.isnan(u), "効用が NaN です"
        assert not torch.isinf(u), "効用が Inf です"
        print("✓ 極端な重みでも数値的に安定")
        
        # 重みを確認
        weights = elem.weights
        print(f"重み（最初の4つ）: {weights[:4].tolist()}")
        assert torch.isfinite(weights).all(), "重みに非有限値が含まれています"
        print("✓ 重みの計算も安定")
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        raise

def main():
    """全テストを実行"""
    print("\n" + "="*60)
    print("Stage 2 改善のテスト開始")
    print("="*60)
    
    try:
        test_log_density_weight()
        test_utility_element_stability()
        test_warmstart()
        test_reinit()
        test_numerical_stability_extreme()
        
        print("\n" + "="*60)
        print("✓ 全テスト成功！")
        print("="*60)
        print("\n改善が正しく実装されています：")
        print("1. log-sum-exp トリックによる数値安定性")
        print("2. μのウォームスタート機能")
        print("3. 弱い再初期化機能")
        print("\nこれらの改善により、Stage 2の学習がより安定かつ効率的になります。")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ テスト失敗: {e}")
        print("="*60)
        raise

if __name__ == "__main__":
    main()

