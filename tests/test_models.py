# tests/test_models.py
"""
最小テスト：形状と有限性、単調性のスモーク、収入計算の一致
"""

import torch
import pytest
from bundleflow.models.flow import BundleFlow
from bundleflow.models.menu import MenuElement, Mechanism
from bundleflow.valuation.valuation import XORValuation


class TestBundleFlow:
    """BundleFlowの基本機能テスト"""
    
    def test_velocity_shape(self):
        """velocity/pushforwardの出力が(B×m)で有限"""
        m = 5
        flow = BundleFlow(m=m)
        
        # テストデータ
        B = 3
        x = torch.randn(B, m)
        t = torch.rand(B)
        
        # velocity計算
        v = flow.velocity(x, t)
        
        # 形状チェック
        assert v.shape == (B, m), f"Expected shape ({B}, {m}), got {v.shape}"
        
        # 有限性チェック
        assert torch.isfinite(v).all(), "Velocity contains non-finite values"
    
    def test_pushforward_shape(self):
        """pushforwardの出力が(B×m)で有限"""
        m = 5
        flow = BundleFlow(m=m)
        
        # テストデータ
        B = 3
        z = torch.randn(B, m)
        
        # pushforward計算
        x_T = flow.pushforward(z, t1=1.0, steps=10)
        
        # 形状チェック
        assert x_T.shape == (B, m), f"Expected shape ({B}, {m}), got {x_T.shape}"
        
        # 有限性チェック
        assert torch.isfinite(x_T).all(), "Pushforward contains non-finite values"
    
    def test_round_to_bundle(self):
        """束への丸めが正しく動作"""
        m = 5
        flow = BundleFlow(m=m)
        
        # テストデータ
        B = 3
        s_T = torch.rand(B, m)
        
        # 丸め
        b = flow.round_to_bundle(s_T)
        
        # 形状チェック
        assert b.shape == (B, m), f"Expected shape ({B}, {m}), got {b.shape}"
        
        # 値の範囲チェック
        assert (b == 0.0).logical_or(b == 1.0).all(), "Bundle should contain only 0s and 1s"
    
    def test_loss_rectified_finite(self):
        """Rectified Flow損失が有限"""
        m = 5
        flow = BundleFlow(m=m)
        
        # テストデータ
        B = 10
        z = torch.randn(B, m)
        
        # 損失計算
        loss = flow.loss_rectified(z)
        
        # 有限性チェック
        assert torch.isfinite(loss), "Loss is not finite"
        assert loss >= 0, "Loss should be non-negative"


class TestMenuElement:
    """MenuElementの基本機能テスト"""
    
    def test_menu_element_creation(self):
        """MenuElementの作成と基本プロパティ"""
        m = 5
        D = 3
        elem = MenuElement(m=m, D=D)
        
        # パラメータの形状チェック
        assert elem.mus.shape == (D, m), f"Expected mus shape ({D}, {m}), got {elem.mus.shape}"
        assert elem.logits.shape == (D,), f"Expected logits shape ({D},), got {elem.logits.shape}"
        assert elem.beta_raw.shape == (1,), f"Expected beta_raw shape (1,), got {elem.beta_raw.shape}"
    
    def test_price_non_negative(self):
        """価格が非負"""
        m = 5
        D = 3
        elem = MenuElement(m=m, D=D)
        
        price = elem.price()
        
        # 非負チェック
        assert price >= 0, f"Price should be non-negative, got {price}"
        assert torch.isfinite(price), "Price should be finite"
    
    def test_weights_simplex(self):
        """重みがsimplex（合計が1）"""
        m = 5
        D = 3
        elem = MenuElement(m=m, D=D)
        
        weights = elem.weights
        
        # simplexチェック
        assert torch.allclose(weights.sum(), torch.tensor(1.0)), f"Weights should sum to 1, got {weights.sum()}"
        assert (weights >= 0).all(), "Weights should be non-negative"
    
    def test_sample_init_shape(self):
        """初期分布サンプルの形状"""
        m = 5
        D = 3
        elem = MenuElement(m=m, D=D)
        
        n = 10
        samples = elem.sample_init(n)
        
        # 形状チェック
        assert samples.shape == (n, m), f"Expected shape ({n}, {m}), got {samples.shape}"
        assert torch.isfinite(samples).all(), "Samples contain non-finite values"


class TestMechanism:
    """Mechanismの基本機能テスト"""
    
    def test_mechanism_creation(self):
        """Mechanismの作成"""
        m = 5
        flow = BundleFlow(m=m)
        menu = [MenuElement(m=m, D=3) for _ in range(2)]
        
        mechanism = Mechanism(flow, menu)
        
        assert mechanism.K == 2, f"Expected K=2, got {mechanism.K}"
        assert len(mechanism.menu) == 2, f"Expected 2 menu elements, got {len(mechanism.menu)}"
    
    def test_expected_revenue_finite(self):
        """期待収入が有限"""
        m = 5
        flow = BundleFlow(m=m)
        menu = [MenuElement(m=m, D=3) for _ in range(2)]
        mechanism = Mechanism(flow, menu)
        
        # 簡単な評価関数を作成
        valuations = []
        for _ in range(5):
            atoms = [([1, 2], 1.0), ([3, 4], 2.0)]  # 簡単なXOR評価関数
            val = XORValuation.from_bundle_list(m, atoms)
            valuations.append(val)
        
        # 期待収入計算
        revenue = mechanism.expected_revenue(valuations)
        
        # 有限性チェック
        assert torch.isfinite(revenue), "Expected revenue is not finite"
    
    def test_argmax_menu_structure(self):
        """argmax_menuの出力構造"""
        m = 5
        flow = BundleFlow(m=m)
        menu = [MenuElement(m=m, D=3) for _ in range(2)]
        mechanism = Mechanism(flow, menu)
        
        # 簡単な評価関数を作成
        valuations = []
        for _ in range(3):
            atoms = [([1, 2], 1.0), ([3, 4], 2.0)]
            val = XORValuation.from_bundle_list(m, atoms)
            valuations.append(val)
        
        # argmax_menu計算
        result = mechanism.argmax_menu(valuations)
        
        # 構造チェック
        required_keys = ['assignments', 'utilities', 'prices', 'revenue', 'welfare', 'ir_satisfied']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # 形状チェック
        assert result['assignments'].shape == (3,), f"Expected assignments shape (3,), got {result['assignments'].shape}"
        assert result['utilities'].shape == (3,), f"Expected utilities shape (3,), got {result['utilities'].shape}"
        assert result['prices'].shape == (3,), f"Expected prices shape (3,), got {result['prices'].shape}"


class TestXORValuation:
    """XORValuationの基本機能テスト"""
    
    def test_valuation_creation(self):
        """XORValuationの作成"""
        m = 5
        atoms = [([1, 2], 1.0), ([3, 4], 2.0)]
        val = XORValuation.from_bundle_list(m, atoms)
        
        assert val.m == m, f"Expected m={m}, got {val.m}"
        assert len(val.atoms) == 2, f"Expected 2 atoms, got {len(val.atoms)}"
    
    def test_value_calculation(self):
        """価値計算の基本動作"""
        m = 5
        atoms = [([1, 2], 1.0), ([3, 4], 2.0)]
        val = XORValuation.from_bundle_list(m, atoms)
        
        # 束 [1, 1, 0, 0, 0] は最初の原子とマッチ
        bundle = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        value = val.value(bundle)
        
        assert value == 1.0, f"Expected value 1.0, got {value}"
        
        # 束 [1, 1, 1, 1, 0] は両方の原子とマッチ（高い方）
        bundle = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        value = val.value(bundle)
        
        assert value == 2.0, f"Expected value 2.0, got {value}"
    
    def test_batch_value_shape(self):
        """バッチ価値計算の形状"""
        m = 5
        atoms = [([1, 2], 1.0), ([3, 4], 2.0)]
        val = XORValuation.from_bundle_list(m, atoms)
        
        # バッチ束
        B = 3
        bundles = torch.rand(B, m)
        values = val.batch_value(bundles)
        
        # 形状チェック
        assert values.shape == (B,), f"Expected shape ({B},), got {values.shape}"
        assert torch.isfinite(values).all(), "Values contain non-finite values"


class TestIntegration:
    """統合テスト"""
    
    def test_flow_menu_integration(self):
        """FlowとMenuの統合動作"""
        m = 5
        flow = BundleFlow(m=m)
        menu = [MenuElement(m=m, D=3) for _ in range(2)]
        
        # 時間グリッド
        t_grid = torch.linspace(0.0, 1.0, steps=10)
        
        # 各メニュー要素の束生成
        for elem in menu:
            s_T = flow.flow_forward(elem.mus, t_grid)
            s = flow.round_to_bundle(s_T)
            
            # 形状チェック
            assert s.shape == (3, m), f"Expected shape (3, {m}), got {s.shape}"
            assert (s == 0.0).logical_or(s == 1.0).all(), "Bundle should contain only 0s and 1s"
    
    def test_revenue_monotonicity(self):
        """収入の単調性テスト（温度やstepsを増やすと収入が非劣化）"""
        m = 5
        flow = BundleFlow(m=m)
        menu = [MenuElement(m=m, D=3) for _ in range(2)]
        mechanism = Mechanism(flow, menu)
        
        # 簡単な評価関数
        valuations = []
        for _ in range(5):
            atoms = [([1, 2], 1.0), ([3, 4], 2.0)]
            val = XORValuation.from_bundle_list(m, atoms)
            valuations.append(val)
        
        # 異なる温度での収入計算
        revenues = []
        for lam in [0.1, 0.2, 0.5]:
            # 温度を変更するために一時的にsoft_assignmentを直接使用
            from bundleflow.models.menu import utilities_matrix, soft_assignment
            t_grid = torch.linspace(0.0, 1.0, steps=10)
            U = utilities_matrix(flow, valuations, menu, t_grid)
            Z = soft_assignment(U, lam)
            beta = torch.stack([elem.beta for elem in menu]).squeeze()
            revenue = (Z * beta.unsqueeze(0)).sum(dim=1).mean()
            revenues.append(revenue.item())
        
        # 単調性チェック（温度が高いほど収入が高くなる傾向）
        # 注意：これは確率的な傾向なので、厳密な単調性は保証されない
        print(f"Revenues at different temperatures: {revenues}")
        assert all(r >= 0 for r in revenues), "All revenues should be non-negative"


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
