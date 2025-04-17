"""
データ取得・前処理を担当
"""
import yfinance as yf
import pandas as pd

def fetch_data(tickers, start, end):
    """
    tickers: list[str]
    start, end: 'YYYY-MM-DD'
    return: pd.DataFrame  (MultiIndex: Date x Ticker)
    """
    price = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    return (price
            .stack()              # Date,Ticker の MultiIndex
            .rename("price")      # Series → DataFrame 用
            .to_frame()
           )
