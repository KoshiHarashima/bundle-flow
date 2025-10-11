#!/usr/bin/env python3
"""
改善されたコードの動作確認用テストスクリプト
"""

import torch
import sys
sys.path.insert(0, '.')

from bf.flow import FlowModel

def test_qnet_initialization():
    """QNet の初期化とスケール制限をテスト"""
    print("Testing QNet initialization and output scale...")
    m = 10
    model = FlowModel(m=m)
    
    # ランダム入力
    s0 = torch.randn(5, m)
    Q = model.Q(s0)
    
    print(f"  Q shape: {Q.shape}")
    print(f"  Q min/max: {Q.min().item():.4f} / {Q.max().item():.4f}")
    print(f"  Q mean: {Q.mean().item():.4f}")
    
    # 出力スケールが [-0.1, 0.1] に制限されているか確認
    assert Q.min() >= -0.11 and Q.max() <= 0.11, "Q output scale is not properly constrained"
    print("  ✓ QNet output scale is properly constrained to [-0.1, 0.1]")

def test_etanet_initialization():
    """EtaNet の初期化とスケール制限をテスト（tanhゲート版）"""
    print("\nTesting EtaNet initialization and output scale...")
    c_eta = 2.0
    model = FlowModel(m=10, c_eta=c_eta)
    
    # t ∈ [0, 1] の範囲でテスト
    t = torch.linspace(0, 1, 20)
    eta = model.eta(t)
    
    print(f"  eta shape: {eta.shape}")
    print(f"  eta min/max: {eta.min().item():.4f} / {eta.max().item():.4f}")
    print(f"  eta mean: {eta.mean().item():.4f}")
    
    # 出力スケールが [-c_eta, c_eta] に制限されているか確認（tanh出力）
    assert eta.min() >= -c_eta - 0.1 and eta.max() <= c_eta + 0.1, "Eta output scale is not properly constrained"
    print(f"  ✓ EtaNet output scale is properly constrained to [-{c_eta}, {c_eta}] (tanh gate)")

def test_statistics_computation():
    """統計量計算のテスト"""
    print("\nTesting statistics computation...")
    m = 10
    B = 5
    model = FlowModel(m=m)
    
    # ダミーデータ
    s0 = torch.randn(B, m)
    sT = torch.randn(B, m)
    t = torch.rand(B)
    s_t = 0.5 * s0 + 0.5 * sT
    pred = model.phi(t, s_t, s0)
    
    stats = model.compute_statistics(s0, sT, t, s_t, pred)
    
    print("  Computed statistics:")
    for key, val in stats.items():
        print(f"    {key}: {val:.4f}")
    
    # 必須のキーがすべて存在するか確認
    required_keys = ['phi_norm', 'trace_q_mean', 'trace_q_std', 'eta_mean', 
                     'eta_std', 's0_min', 's0_max', 's0_mean', 'Q_norm']
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    print("  ✓ All required statistics are computed")

def test_trace_penalty():
    """Tr(Q)² 罰則のテスト"""
    print("\nTesting trace penalty computation...")
    m = 10
    B = 5
    model = FlowModel(m=m)
    
    s0 = torch.randn(B, m)
    penalty = model.compute_trace_q_penalty(s0)
    
    print(f"  Trace penalty: {penalty.item():.6f}")
    assert penalty.item() >= 0, "Trace penalty should be non-negative"
    print("  ✓ Trace penalty is properly computed")

def test_loss_with_regularization():
    """正則化付き損失のテスト"""
    print("\nTesting loss computation with regularization...")
    m = 10
    D = 4
    model = FlowModel(m=m)
    
    # ダミー MoG パラメータ
    device = torch.device('cpu')
    mus = torch.randn(D, m, device=device)
    sigmas = torch.full((D,), 0.5, device=device)
    weights = torch.full((D,), 1.0 / D, device=device)
    
    # return_statsのテスト - タプルが返るか
    result = model.rectified_flow_loss(
        B=8, mus=mus, sigmas=sigmas, weights=weights,
        trace_penalty_weight=0.0, return_stats=True
    )
    assert isinstance(result, tuple) and len(result) == 2, "return_stats=True should return (loss, stats)"
    loss, stats = result
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Stats keys: {list(stats.keys())}")
    
    # return_stats=False のテスト
    loss_only = model.rectified_flow_loss(
        B=8, mus=mus, sigmas=sigmas, weights=weights,
        trace_penalty_weight=0.0, return_stats=False
    )
    assert isinstance(loss_only, torch.Tensor), "return_stats=False should return only loss"
    print(f"  Loss only: {loss_only.item():.6f}")
    
    print("  ✓ Regularization interface is working correctly")

def test_spectral_norm():
    """スペクトル正規化のテスト"""
    print("\nTesting spectral normalization...")
    m = 10
    
    # スペクトル正規化あり
    model_sn = FlowModel(m=m, use_spectral_norm=True)
    # スペクトル正規化なし
    model_no_sn = FlowModel(m=m, use_spectral_norm=False)
    
    # スペクトル正規化が適用されているか確認
    has_sn = False
    for name, module in model_sn.Q.net.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_orig'):
                has_sn = True
                print(f"  ✓ Spectral norm applied to: {name}")
                break
    
    assert has_sn, "Spectral normalization not applied"
    print("  ✓ Spectral normalization is working correctly")

def test_q_mode_diag():
    """対角モードのテスト"""
    print("\nTesting Q diagonal mode...")
    m = 10
    B = 5
    model = FlowModel(m=m, q_mode='diag')
    
    s0 = torch.randn(B, m)
    Q = model.Q(s0)
    
    print(f"  Q shape: {Q.shape}")
    
    # 対角行列であることを確認
    for i in range(B):
        off_diag = Q[i].clone()
        off_diag.fill_diagonal_(0.0)
        assert off_diag.abs().max() < 1e-6, f"Q[{i}] is not diagonal"
    
    print("  ✓ Q is diagonal in 'diag' mode")

def test_q_mode_switch():
    """対角→フルモード切り替えのテスト"""
    print("\nTesting Q mode switch (diag -> full)...")
    m = 10
    B = 5
    model = FlowModel(m=m, q_mode='diag')
    
    s0 = torch.randn(B, m)
    
    # 対角モード
    Q_diag = model.Q(s0)
    assert model.Q.q_mode == 'diag', "Initial mode should be 'diag'"
    
    # フルモードへ切り替え
    model.Q.switch_to_full_mode()
    assert model.Q.q_mode == 'full', "Mode should be 'full' after switch"
    
    # フルモードで行列を生成
    Q_full = model.Q(s0)
    print(f"  Q shape after switch: {Q_full.shape}")
    print("  ✓ Q mode switch works correctly")

def test_new_regularization():
    """新しい正則化項（Jacobian, Kinetic, Trace）のテスト"""
    print("\nTesting new regularization penalties...")
    m = 10
    D = 4
    model = FlowModel(m=m)
    
    device = torch.device('cpu')
    mus = torch.randn(D, m, device=device)
    sigmas = torch.full((D,), 0.5, device=device)
    weights = torch.full((D,), 1.0 / D, device=device)
    
    # 新しい正則化項を使用
    loss, stats = model.rectified_flow_loss(
        B=8, mus=mus, sigmas=sigmas, weights=weights,
        lambda_j=1e-3, lambda_k=1e-3, lambda_tr=1e-4,
        return_stats=True
    )
    
    print(f"  Loss: {loss.item():.6f}")
    
    # 正則化項の統計が含まれているか確認
    assert 'jacobian_penalty' in stats, "Jacobian penalty not in stats"
    assert 'kinetic_penalty' in stats, "Kinetic penalty not in stats"
    assert 'trace_penalty' in stats, "Trace penalty not in stats"
    
    print(f"  Jacobian penalty: {stats['jacobian_penalty']:.6f}")
    print(f"  Kinetic penalty: {stats['kinetic_penalty']:.6f}")
    print(f"  Trace penalty: {stats['trace_penalty']:.6f}")
    print("  ✓ New regularization penalties are working correctly")

def test_gradient_flow():
    """勾配が正しく流れるかテスト"""
    print("\nTesting gradient flow...")
    m = 10
    D = 4
    model = FlowModel(m=m)
    
    device = torch.device('cpu')
    mus = torch.randn(D, m, device=device)
    sigmas = torch.full((D,), 0.5, device=device)
    weights = torch.full((D,), 1.0 / D, device=device)
    
    loss = model.rectified_flow_loss(
        B=8, mus=mus, sigmas=sigmas, weights=weights,
        trace_penalty_weight=1e-3
    )
    
    loss.backward()
    
    # 勾配が計算されているか確認
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"  ✓ Gradient computed for: {name}")
            break
    
    assert has_grad, "No gradients computed"
    print("  ✓ Gradients are flowing correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing BundleFlow Improvements")
    print("=" * 60)
    
    try:
        # 基本機能テスト
        test_qnet_initialization()
        test_etanet_initialization()
        test_statistics_computation()
        test_trace_penalty()
        test_loss_with_regularization()
        
        # 新機能テスト
        test_spectral_norm()
        test_q_mode_diag()
        test_q_mode_switch()
        test_new_regularization()
        
        # 勾配フローテスト
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

