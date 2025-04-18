# dual_momentum/__init__.py
from .core import extra  # extra.pyを先に実行してimport群を登録
from .core.model import DualMomentumModel
from .ui.dashboard import create_dual_momentum_ui  # UIコンポーネントを追加

__all__ = [
    "DualMomentumModel",  # コアモデル
    "create_dual_momentum_ui"  # UIコンポーネント
]
