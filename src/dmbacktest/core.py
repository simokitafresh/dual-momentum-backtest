"""
デュアル・モメンタム戦略のバックテスト本体
（ノートブックのロジックをここへ移植）
"""
import pandas as pd
import numpy as np

def run_backtest(price_df,
                 lookbacks=(2, 6, 12),
                 weights  =(0.2,0.2,0.6),
                 abs_asset="LQD",
                 out_assets=("XLU","GLD"),
                 hold_top=1):
    # ▼ ノートブックの計算式をそのまま移してくれば OK
    #   ここではダミーで累積リターンを返すだけ
    cumret = price_df.groupby("Ticker")["price"].pct_change().add(1).cumprod()
    return cumret
