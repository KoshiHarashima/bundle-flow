#!/usr/bin/env python3
"""
初期化の改善テスト - μ=0病の解消と適切な初期分布設定

目的: MenuElement の初期化が適切に行われているかを検証
"""

import torch
import sys
import os
import numpy as np

# BundleFlow のインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bundleflow.models.menu import MenuElement, make_null_element
from bundleflow.models.flow import BundleFlow

def test_menu_element_initialization():
    """MenuElement の初期化テスト"""
    print("=== MenuElement 初期化テスト ===")
    
    m, D = 10, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MenuElement の作成
    elem = MenuElement(m, D).to(device)
    
    print(f"商品数 m: {m}, 混合成分数 D: {D}")
    print(f"デバイス: {device}")
    
    # 初期値の確認
    print(f"\n初期値の確認:")
    print(f"  mus: {elem.mus.shape}, 範囲: [{elem.mus.min().item():.3f}, {elem.mus.max().item():.3f}]")
    print(f"  logits: {elem.logits.shape}, 範囲: [{elem.logits.min().item():.3f}, {elem.logits.max().item():.3f}]")
    print(f"  beta_raw: {elem.beta_raw.shape}, 値: {elem.beta_raw.item():.3f}")
    
    # 重みの確認
    weights = elem.weights
    print(f"  weights: {weights.shape}, 範囲: [{weights.min().item():.3f}, {weights.max().item():.3f}]")
    print(f"  weights sum: {weights.sum().item():.3f}")
    
    # 価格の確認
    price = elem.price()
    print(f"  price: {price.item():.3f}")
    
    # サンプリングのテスト
    print(f"\nサンプリングテスト:")
    n_samples = 100
    samples = elem.sample_init(n_samples)
    print(f"  サンプル数: {n_samples}")
    print(f"  サンプル形状: {samples.shape}")
    print(f"  サンプル範囲: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    print(f"  サンプル平均: {samples.mean().item():.3f}")
    
    # μ=0病のチェック
    zero_count = (elem.mus == 0.0).all(dim=1).sum().item()
    print(f"\nμ=0病チェック:")
    print(f"  全成分が0の数: {zero_count}/{D}")
    
    if zero_count == D:
        print("  ❌ μ=0病が発生しています！")
        return False
    else:
        print("  ✅ μ=0病は発生していません")
        return True

def test_improved_initialization():
    """改善された初期化のテスト"""
    print("\n=== 改善された初期化テスト ===")
    
    m, D = 10, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 改善された初期化
    elem = MenuElement(m, D).to(device)
    
    # μの改善された初期化
    with torch.no_grad():
        # 方法1: 代表束近傍での初期化
        # 全1束 (1,1,1,...,1)
        elem.mus[0] = torch.ones(m, device=device)
        
        # 少数1束 (1,0,0,...,0), (0,1,0,...,0), etc.
        for d in range(1, min(D, m+1)):
            elem.mus[d] = torch.zeros(m, device=device)
            elem.mus[d][d-1] = 1.0
        
        # 残りはランダム初期化
        for d in range(min(D, m+1), D):
            elem.mus[d] = torch.rand(m, device=device) * 1.4 - 0.2  # U[-0.2, 1.2]
        
        # 重みの初期化（均等）
        elem.logits.zero_()
        
        # 価格の初期化（多様化）
        elem.beta_raw.data = torch.randn(1, device=device) * 0.5 - 2.0
    
    print(f"改善された初期値:")
    print(f"  mus: {elem.mus.shape}, 範囲: [{elem.mus.min().item():.3f}, {elem.mus.max().item():.3f}]")
    print(f"  mus[0] (全1): {elem.mus[0].tolist()}")
    print(f"  mus[1] (少数1): {elem.mus[1].tolist()}")
    print(f"  mus[2] (少数1): {elem.mus[2].tolist()}")
    
    # サンプリングのテスト
    n_samples = 100
    samples = elem.sample_init(n_samples)
    print(f"\n改善されたサンプリング:")
    print(f"  サンプル範囲: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    print(f"  サンプル平均: {samples.mean().item():.3f}")
    
    # 束生成のテスト
    flow = BundleFlow(m).to(device)
    flow.eval()
    
    t_grid = torch.linspace(0.0, 1.0, steps=25, device=device)
    
    with torch.no_grad():
        sT = flow.flow_forward(elem.mus, t_grid)
        s = flow.round_to_bundle(sT)
        
    print(f"\n束生成テスト:")
    print(f"  sT範囲: [{sT.min().item():.3f}, {sT.max().item():.3f}]")
    print(f"  s範囲: [{s.min().item():.3f}, {s.max().item():.3f}]")
    
    # 束サイズの計算
    bundle_sizes = s.sum(dim=1).tolist()
    print(f"  束サイズ: {bundle_sizes}")
    
    # 非ゼロ束の数
    non_zero_bundles = sum(1 for size in bundle_sizes if size > 0)
    print(f"  非ゼロ束の数: {non_zero_bundles}/{D}")
    
    if non_zero_bundles > 0:
        print("  ✅ 改善された初期化で束生成が成功")
        return True
    else:
        print("  ❌ 改善された初期化でも束生成が失敗")
        return False

def test_flow_with_improved_initialization():
    """改善された初期化でのFlow統合テスト"""
    print("\n=== Flow統合テスト ===")
    
    m, D = 10, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Flow と MenuElement の作成
    flow = BundleFlow(m).to(device)
    flow.eval()
    
    elem = MenuElement(m, D).to(device)
    
    # 改善された初期化
    with torch.no_grad():
        # 代表束近傍での初期化
        elem.mus[0] = torch.ones(m, device=device)  # 全1束
        elem.mus[1] = torch.zeros(m, device=device)
        elem.mus[1][0] = 1.0  # 少数1束
        
        # 残りはランダム
        for d in range(2, D):
            elem.mus[d] = torch.rand(m, device=device) * 1.4 - 0.2
        
        elem.logits.zero_()
        elem.beta_raw.data = torch.randn(1, device=device) * 0.5 - 2.0
    
    # 時間グリッド
    t_grid = torch.linspace(0.0, 1.0, steps=25, device=device)
    
    # 束生成のテスト
    with torch.no_grad():
        sT = flow.flow_forward(elem.mus, t_grid)
        s = flow.round_to_bundle(sT)
        
    print(f"統合テスト結果:")
    print(f"  sT範囲: [{sT.min().item():.3f}, {sT.max().item():.3f}]")
    print(f"  s範囲: [{s.min().item():.3f}, {s.max().item():.3f}]")
    
    # 束サイズの計算
    bundle_sizes = s.sum(dim=1).tolist()
    print(f"  束サイズ: {bundle_sizes}")
    
    # 多様性の計算
    unique_bundles = len(torch.unique(s, dim=0))
    print(f"  ユニーク束数: {unique_bundles}/{D}")
    
    # 結果の評価
    non_zero_bundles = sum(1 for size in bundle_sizes if size > 0)
    diversity = unique_bundles / D
    
    print(f"\n評価結果:")
    print(f"  非ゼロ束: {non_zero_bundles}/{D}")
    print(f"  多様性: {diversity:.3f}")
    
    if non_zero_bundles > 0 and diversity > 0.1:
        print("  ✅ 統合テスト成功")
        return True
    else:
        print("  ❌ 統合テスト失敗")
        return False

def main():
    """メインテスト"""
    print("初期化の改善テスト開始")
    print("=" * 50)
    
    results = []
    
    # 各テストの実行
    results.append(("MenuElement初期化", test_menu_element_initialization()))
    results.append(("改善された初期化", test_improved_initialization()))
    results.append(("Flow統合テスト", test_flow_with_improved_initialization()))
    
    # 結果の集計
    print("\n" + "=" * 50)
    print("テスト結果:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    # 全体の結果
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 全てのテストが成功しました！")
        print("   → 初期化の改善が有効です。")
    else:
        print("\n🚨 一部のテストが失敗しました。")
        print("   → 初期化の改善が必要です。")
    
    return all_passed

if __name__ == "__main__":
    main()
