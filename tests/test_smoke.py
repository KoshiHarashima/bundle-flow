import torch
from bundleflow.valuation import XORValuation
from bundleflow.data import gen_uniform_iid_xor

def test_xor_small():
    v = gen_uniform_iid_xor(m=6, a=4, seed=1)
    # 単品束で値が0/非0のどちらかにはなる
    for i in range(6):
        s = torch.zeros(6); s[i]=1
        assert isinstance(v.value(s), float)
