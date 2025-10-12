# bf package
"""
BundleFlow core library

Modules:
- models: BundleFlow, MenuElement, Mechanism
- valuation: XORValuation and related classes
- data: Data loading and generation
- utils: Utility functions
- train: Training scripts for Stage1 and Stage2
"""

__version__ = "1.0.0"

# 後方互換性のためのインポート
from .models.flow import BundleFlow, FlowModel
from .models.menu import MenuElement, Mechanism
from .valuation.valuation import XORValuation
from . import data
from . import utils

__all__ = [
    'BundleFlow', 'FlowModel',  # 後方互換性
    'MenuElement', 'Mechanism',
    'XORValuation',
    'data', 'utils'
]

