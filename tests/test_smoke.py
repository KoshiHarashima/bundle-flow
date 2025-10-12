import torch
from bundleflow.valuation import XORValuation
from bundleflow.data import gen_uniform_iid_xor
from bundleflow.flow import FlowModel

def test_xor_small():
    """XOR評価の基本動作確認"""
    v = gen_uniform_iid_xor(m=6, a=4, seed=1)
    # 単品束で値が0/非0のどちらかにはなる
    for i in range(6):
        s = torch.zeros(6); s[i]=1
        assert isinstance(v.value(s), float)

def test_flow_model():
    """FlowModelの基本動作確認"""
    flow = FlowModel(m=6, use_spectral_norm=False)
    s = torch.randn(8, 6)  # 8個のバンドル
    sT = flow.flow_forward(s, torch.linspace(0, 1, 10))
    s_rounded = flow.round_to_bundle(sT)
    
    # フロー前→丸め後のユニーク束数>0を確認
    unique_bundles = len(torch.unique(s_rounded, dim=0))
    assert unique_bundles > 0
    assert s_rounded.shape == s.shape
