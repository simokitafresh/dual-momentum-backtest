# dual_momentum/__init__.py
from .core import extra  # ← 追加：extra.py を先に実行して import 群を登録
from .core.model import DualMomentumModel

__all__ = ["DualMomentumModel"]

