import numpy as np

import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as mticker

import seaborn as sns

from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta

import calendar

import ipywidgets as widgets

from IPython.display import display, clear_output, HTML

import json

import gc

from tqdm import tqdm

import warnings

import concurrent.futures

import logging

import os

import types

import pandas_market_calendars as mcal

logging.getLogger('DualMomentumModel').setLevel(logging.WARNING)

try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception as e:
    pass

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('DualMomentumModel')

def display_performance_summary(model, display_summary=True):
    """
    DualMomentumModel クラスの display_performance_summary メソッドを呼び出すラッパー関数。
    既存の依存関係を維持するために用意されています。

    Parameters:
    model (DualMomentumModel): 表示対象のモデルインスタンス
    display_summary (bool): サマリーを表示するかどうか
    """
    # クラスメソッドを呼び出し
    model.display_performance_summary(display_summary=display_summary)

def display_all_signals_for_patterns(model):
    patterns = [
        {"title": "24-month return", "lookback_period": 24, "lookback_unit": "Months", "performance_periods": "Single Period"},
        {"title": "15-day return",   "lookback_period": 15, "lookback_unit": "Days", "performance_periods": "Single Period"},
        {"title": "1-month return",  "lookback_period": 1,  "lookback_unit": "Months", "performance_periods": "Single Period"}
    ]
    for pat in patterns:
        model.performance_periods = pat["performance_periods"]
        model.lookback_period = pat["lookback_period"]
        model.lookback_unit = pat["lookback_unit"]
        model.display_model_signals_dynamic()
        display(HTML("<hr>"))

def display_performance_summary_ui(model):
    display_performance_summary(model)

def display_model_signals_dynamic_ui(model):
    model.display_model_signals_dynamic()

try:
    import google.colab
    print("Google Colab environment detected.")
    print("必要なパッケージをインストールしました（openpyxlを含む）")
except Exception as e:
    print("Running in local environment.")
    # ローカル環境でもopenpyxlが必要
    try:
        import openpyxl
    except ImportError:
        pass
#        print("openpyxlパッケージが必要です。pip install openpyxlでインストールしてください。")

#model = create_dual_momentum_ui()

