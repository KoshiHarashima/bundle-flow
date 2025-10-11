#!/usr/bin/env python3
"""
η(t) 飽和修正のテスト
"""

import torch
from bf.flow import FlowModel

def test_eta_ranges():
    """異なるc_eta設定でηの範囲をテスト"""
    print("="*60)
    print("η(t) 範囲テスト")
    print("="*60)
    
    m = 10
    t_grid = torch.linspace(0.0, 1.0, steps=100)
    
    configs = [
        {"c_eta": 2.0, "eta_init_scale": 0.01, "desc": "デフォルト（飽和しやすい）"},
        {"c_eta": 5.0, "eta_init_scale": 0.01, "desc": "c_eta=5.0（改善版）"},
        {"c_eta": 10.0, "eta_init_scale": 0.01, "desc": "c_eta=10.0（推奨）"},
        {"c_eta": 10.0, "eta_init_scale": 0.001, "desc": "c_eta=10.0 + 小初期化"},
    ]
    
    for config in configs:
        print(f"\n【{config['desc']}】")
        print(f"  c_eta={config['c_eta']}, eta_init_scale={config['eta_init_scale']}")
        
        # モデル作成
        flow = FlowModel(m=m, c_eta=config['c_eta'], eta_init_scale=config['eta_init_scale'])
        flow.eval()
        
        with torch.no_grad():
            # η(t)を計算
            eta_vals = flow.eta(t_grid).numpy()
            
            # EtaNetの内部rawを取得
            raw_vals = []
            for t in t_grid:
                t_input = t.unsqueeze(0) if t.dim() == 0 else t
                raw = flow.eta.net(t_input.unsqueeze(-1)).squeeze(-1)
                raw_vals.append(raw.item())
            import numpy as np
            raw_vals = np.array(raw_vals)
            tanh_vals = np.tanh(raw_vals)
            tanh_grad = 1 - tanh_vals**2
        
        print(f"  η(t): [{eta_vals.min():.3f}, {eta_vals.max():.3f}], "
              f"平均={eta_vals.mean():.3f}±{eta_vals.std():.3f}")
        print(f"  raw: [{raw_vals.min():.3f}, {raw_vals.max():.3f}]")
        print(f"  tanh(raw): [{tanh_vals.min():.3f}, {tanh_vals.max():.3f}]")
        print(f"  飽和度(|tanh|>0.99): {(np.abs(tanh_vals) > 0.99).mean()*100:.1f}%")
        print(f"  勾配感度: {tanh_grad.mean():.3e}")
        
        # 判定
        saturation_ratio = (np.abs(tanh_vals) > 0.99).mean()
        if saturation_ratio > 0.5:
            print(f"  → ⚠️  飽和リスクあり")
        else:
            print(f"  → ✓ 健全")

def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n" + "="*60)
    print("後方互換性テスト")
    print("="*60)
    
    m = 10
    
    # デフォルト引数でモデル作成可能か
    try:
        flow1 = FlowModel(m=m)
        print("✓ デフォルト引数で作成可能")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return
    
    # 明示的に引数指定
    try:
        flow2 = FlowModel(m=m, c_eta=10.0, eta_init_scale=0.001)
        print("✓ 明示的な引数指定で作成可能")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return
    
    # 既存チェックポイントの読み込みシミュレーション
    try:
        # チェックポイントのシミュレーション（c_etaなし）
        ckpt = {
            'm': m,
            'model': flow1.state_dict()
        }
        
        # 新しいモデルで読み込み
        flow3 = FlowModel(m=m, c_eta=5.0)  # 異なるc_etaでも読み込める
        flow3.load_state_dict(ckpt['model'])
        print("✓ 旧チェックポイントを読み込み可能（c_eta は設定で変更）")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return
    
    print("\n全ての後方互換性テストに合格！")

if __name__ == "__main__":
    test_eta_ranges()
    test_backward_compatibility()
    
    print("\n" + "="*60)
    print("テスト完了")
    print("="*60)
    print("\n推奨ワークフロー:")
    print("  1. まずデフォルト設定（c_eta=2.0）で学習")
    print("     python -m src.train_stage1 ...")
    print("\n  2. 飽和診断を実行")
    print("     python diagnose_eta_saturation.py <checkpoint>")
    print("\n  3. 飽和が確認されたら c_eta を増やして再学習")
    print("     python -m src.train_stage1 --c_eta 5.0 --eta_init_scale 0.001 ...")

