# ===== 共通インポート =====
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import calendar
import logging
import concurrent.futures
import yfinance as yf              # ★ 新規追加：価格データ取得
from tqdm import tqdm              # ★ 新規追加：進捗バー
import pandas_market_calendars as mcal  # ★ 新規追加：取引日カレンダー
import gc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


from IPython.display import display, HTML
from ..utils.logger import logger
from ..validators.input import InputValidator

class DualMomentumModel:
    def __init__(self):
        today = datetime.now()
        # 初期値は後でUIから上書きされる
        self.start_year = 2010
        self.start_month = 1
        self.end_year = today.year
        self.end_month = today.month
        self.tickers = ["TQQQ", "TECL"]
        self.single_absolute_momentum = "Yes"
        self.absolute_momentum_asset = "LQD"
        self.negative_relative_momentum = "No"
        self.out_of_market_assets = ["XLU"]
        self.out_of_market_strategy = "Equal Weight"   # 退避先資産の選択戦略 ("Equal Weight" または "Top 1")
        self.performance_periods = "Multiple Periods"  # "Single Period"も選択可能
        self.lookback_period = 12
        self.lookback_unit = "Months"  # "Days"も選択可能
        self.multiple_periods = [
            {"length": 2, "unit": "Months", "weight": 20},
            {"length": 6, "unit": "Months", "weight": 20},
            {"length": 12, "unit": "Months", "weight": 60},
            {"length": None, "unit": None, "weight": 0},
            {"length": None, "unit": None, "weight": 0}
        ]
        self.multiple_periods_count = 3
        self.weighting_method = "Weight Performance"
        self.assets_to_hold = 1

        self.trading_frequency = "Monthly"  # "Monthly", "Bimonthly (hold: 1,3,5,7,9,11)",
                                            # "Bimonthly (hold: 2,4,6,8,10,12)",
                                            # "Quarterly (hold: 1,4,7,10)", "Quarterly (hold: 2,5,8,11)",
                                            # "Quarterly (hold: 3,6,9,12)"
                                            # Note: For options with "hold:", rebalancing occurs at the end of the month prior to holding

        self.trade_execution = "Trade at next open price"  # または "Trade at end of month price"
        self.benchmark_ticker = "SPY"
        self.price_data = None
        self.monthly_data = None
        self.results = None
        self.rfr_data = None
        self.rfr_data_daily = None  # 日次リスクフリーレート用
        self.absolute_momentum_custom_period = False
        self.absolute_momentum_period = 12
        self.momentum_cache = {}
        self._cache_expiry = 7  # キャッシュ有効期間（日）
        self._last_data_fetch = None
        self.valid_period_start = None
        self.valid_period_end = None
        self.momentum_results = None
        self.data_quality_info = None
        self.validation_errors = []
        self.validation_warnings = []

    def get_exact_period_dates(self, end_date, months):

        """
        正確な計算期間の開始日と終了日を取得する
        """
        # 終了日の調整（データ最終日を超えないように）
        if self.price_data is not None and not self.price_data.empty:
            available_dates = self.price_data.index[self.price_data.index <= end_date]
            if not available_dates.empty:
                end_date = available_dates[-1]

        # 正確に N ヶ月前の日付を計算
        start_date = end_date - relativedelta(months=months)

        # 開始日の調整（データが存在する最も近い日に）
        if self.price_data is not None and not self.price_data.empty:
            available_dates = self.price_data.index[self.price_data.index <= start_date]
            if not available_dates.empty:
                start_date = available_dates[-1]

        return start_date, end_date

    # ----------------------
    # キャッシュ管理メソッド
    def clear_cache(self):
        """Clear the momentum cache and reset cache timestamps"""
        self.momentum_cache = {}
        self._last_data_fetch = None
        logger.info("Cache cleared")

    def _save_to_cache(self, key, data):
        """Save calculated momentum data to cache with timestamp"""
        self.momentum_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        logger.debug(f"Cache saved for key: {key}")

    def _get_from_cache(self, key):
        """Retrieve momentum data from cache if it exists and is not expired"""
        if key not in self.momentum_cache:
            return None
        cache_entry = self.momentum_cache[key]
        cache_age = (datetime.now() - cache_entry['timestamp']).days
        if cache_age > self._cache_expiry:
            logger.debug(f"Cache entry expired for key {key} (age: {cache_age} days)")
            return None
        return cache_entry['data']

    def diagnose_cache(self):
        """Provide diagnostic information about the cache state"""
        if not self.momentum_cache:
            return {
                "status": "empty",
                "message": "Cache is empty",
                "entries": 0
            }
        entries = len(self.momentum_cache)
        oldest_entry = min([entry['timestamp'] for entry in self.momentum_cache.values()])
        newest_entry = max([entry['timestamp'] for entry in self.momentum_cache.values()])
        oldest_age = (datetime.now() - oldest_entry).days
        if oldest_age > self._cache_expiry:
            status = "stale"
            message = f"Cache contains stale entries (oldest: {oldest_age} days, expiry: {self._cache_expiry} days)"
        else:
            status = "ok"
            message = f"Cache contains {entries} valid entries"
        return {
            "status": status,
            "message": message,
            "entries": entries,
            "oldest_entry": oldest_entry,
            "newest_entry": newest_entry,
            "oldest_age_days": oldest_age,
            "expiry_days": self._cache_expiry
        }

    def clear_results(self):
        """すべての結果関連インスタンス変数をクリアする"""
        self.results = None
        self.positions = []
        self.monthly_returns_data = {}
        self.pivot_monthly_returns = None
        self.momentum_results = None
        self.metrics = None
        self.clear_cache()
        logger.info("全ての結果データがクリアされました")

    # ----------------------
    def validate_parameters(self):
        errors = []
        warnings_list = []
        valid, message = InputValidator.validate_date_range(
            self.start_year, self.start_month, self.end_year, self.end_month
        )
        if not valid:
            errors.append(message)
        valid, message = InputValidator.validate_ticker_symbols(self.tickers)
        if not valid:
            errors.append(message)
        if self.performance_periods == "Single Period":
            valid, message = InputValidator.validate_lookback_period(
                self.lookback_period, self.lookback_unit
            )
            if not valid:
                errors.append(message)
            if self.absolute_momentum_custom_period:
                valid, message = InputValidator.validate_lookback_period(
                    self.absolute_momentum_period, self.lookback_unit
                )
                if not valid:
                    errors.append(f"絶対モメンタム期間のエラー: {message}")
        else:
            period_weights = []
            for i, period in enumerate(self.multiple_periods):
                length = period.get("length")
                unit = period.get("unit")
                weight = period.get("weight", 0)
                if length is not None and weight > 0:
                    valid, message = InputValidator.validate_lookback_period(length, unit)
                    if not valid:
                        errors.append(f"期間 #{i+1} のエラー: {message}")
                    period_weights.append(weight)
            if period_weights:
                valid, message = InputValidator.validate_weights(period_weights)
                if not valid:
                    warnings_list.append(message)
                    logger.warning(message)
                    total = sum(period_weights)
                    if total > 0:
                        for i, period in enumerate(self.multiple_periods):
                            if period.get("weight", 0) > 0:
                                period["weight"] = round(period["weight"] * 100 / total)
                        adjusted_weights = [p["weight"] for p in self.multiple_periods if p.get("weight", 0) > 0]
                        adjusted_total = sum(adjusted_weights)
                        if adjusted_total != 100 and adjusted_weights:
                            diff = 100 - adjusted_total
                            max_idx = adjusted_weights.index(max(adjusted_weights))
                            count = 0
                            for i, period in enumerate(self.multiple_periods):
                                if period.get("weight", 0) > 0:
                                    if count == max_idx:
                                        period["weight"] += diff
                                    count += 1
                            logger.info(f"重みが自動調整されました: {[p['weight'] for p in self.multiple_periods if p.get('weight', 0) > 0]}")
            else:
                errors.append("複数期間モードでは、少なくとも1つの期間に正の重みを設定する必要があります。")
        if self.assets_to_hold < 1:
            errors.append(f"保有資産数は1以上である必要があります: {self.assets_to_hold}")
        if not self.out_of_market_assets:
            warnings_list.append("退避先資産が指定されていません。市場退出時の代替資産がありません。")
        return len(errors) == 0, errors, warnings_list

    def check_data_quality(self, max_consecutive_na_threshold=20):
        quality_warnings = []
        if self.price_data is None or self.price_data.empty:
            quality_warnings.append("価格データが空です。")
            return False, quality_warnings
        data_period_days = (self.price_data.index[-1] - self.price_data.index[0]).days
        data_period_years = data_period_days / 365.25
        logger.info(f"データ全体の期間: {self.price_data.index[0].strftime('%Y-%m-%d')} から {self.price_data.index[-1].strftime('%Y-%m-%d')} ({data_period_days}日間, 約{data_period_years:.1f}年)")
        assets_info = {}
        for column in self.price_data.columns:
            valid_count = self.price_data[column].count()
            total_count = len(self.price_data)
            missing_count = total_count - valid_count
            missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
            max_consecutive_na = 0
            current_consecutive_na = 0
            for val in self.price_data[column]:
                if pd.isna(val):
                    current_consecutive_na += 1
                    max_consecutive_na = max(max_consecutive_na, current_consecutive_na)
                else:
                    current_consecutive_na = 0
            zero_count = len(self.price_data[self.price_data[column] == 0])
            negative_count = len(self.price_data[self.price_data[column] < 0])
            asset_data = self.price_data[column].dropna()
            first_date = asset_data.index[0] if not asset_data.empty else None
            last_date = asset_data.index[-1] if not asset_data.empty else None
            assets_info[column] = {
                "valid_count": valid_count,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "max_consecutive_na": max_consecutive_na,
                "zero_count": zero_count,
                "negative_count": negative_count,
                "first_date": first_date,
                "last_date": last_date
            }
            if max_consecutive_na >= max_consecutive_na_threshold:
                quality_warnings.append(f"資産 {column} に {max_consecutive_na} 日連続の欠損データがあります。（閾値: {max_consecutive_na_threshold}日）")
            if zero_count > 0:
                quality_warnings.append(f"資産 {column} に {zero_count} 件のゼロ値があります。")
            if negative_count > 0:
                quality_warnings.append(f"資産 {column} に {negative_count} 件の負の値があります。これは通常、価格データでは想定されません。")
            if missing_percentage > 10:
                quality_warnings.append(f"資産 {column} のデータ欠損率が高いです: {missing_percentage:.1f}%")
        valid_starts = [info["first_date"] for _, info in assets_info.items() if info["first_date"] is not None]
        valid_ends = [info["last_date"] for _, info in assets_info.items() if info["last_date"] is not None]
        if valid_starts and valid_ends:
            common_start = max(valid_starts)
            common_end = min(valid_ends)
            if common_start <= common_end:
                common_period_days = (common_end - common_start).days
                common_period_years = common_period_days / 365.25
                logger.info(f"全対象資産共通の有効期間: {common_start.strftime('%Y-%m-%d')} から {common_end.strftime('%Y-%m-%d')} ({common_period_days}日間, 約{common_period_years:.1f}年)")
                if common_period_days < 365:
                    quality_warnings.append(f"共通有効期間が短いです: {common_period_days}日（約{common_period_years:.1f}年）。より長い期間でのバックテストをお勧めします。")
                self.valid_period_start = common_start
                self.valid_period_end = common_end
            else:
                quality_warnings.append(f"全対象資産に共通する有効期間がありません。最長開始日: {common_start.strftime('%Y-%m-%d')}, 最短終了日: {common_end.strftime('%Y-%m-%d')}")
        else:
            quality_warnings.append("有効な日付情報がない資産があります。")
        self.data_quality_info = {
            "assets_info": assets_info,
            "warnings": quality_warnings,
            "check_timestamp": datetime.now()
        }
        return len(quality_warnings) == 0, quality_warnings

    def display_data_quality_info(self):
        if not hasattr(self, 'data_quality_info') or self.data_quality_info is None:
            print("データ品質情報がありません。check_data_quality()を実行してください。")
            return
        check_time = self.data_quality_info["check_timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        html_output = f"""
        <h3>データ品質チェック結果 ({check_time})</h3>
        """
        warnings_list = self.data_quality_info["warnings"]
        if warnings_list:
            html_output += "<div style='color: #c00; margin-bottom: 10px;'><p><strong>⚠️ 警告:</strong></p><ul>"
            for warning in warnings_list:
                html_output += f"<li>{warning}</li>"
            html_output += "</ul></div>"
        else:
            html_output += "<p style='color: #0c0;'><strong>✅ データ品質に問題は見つかりませんでした。</strong></p>"
        assets_info = self.data_quality_info["assets_info"]
        html_output += """
        <table style="border-collapse: collapse; width: 100%; margin-top: 15px;">
        <tr style="background-color: #f2f2f2;">
          <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">資産</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">有効開始日</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">有効終了日</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">欠損率</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">最大連続欠損</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">ゼロ値</th>
          <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">負の値</th>
        </tr>
        """
        for asset, info in assets_info.items():
            first_date_str = info["first_date"].strftime("%Y-%m-%d") if info["first_date"] is not None else "N/A"
            last_date_str = info["last_date"].strftime("%Y-%m-%d") if info["last_date"] is not None else "N/A"
            missing_color = "#0c0"
            if info["missing_percentage"] > 5:
                missing_color = "#fc0"
            if info["missing_percentage"] > 10:
                missing_color = "#c00"
            consecutive_color = "#0c0"
            if info["max_consecutive_na"] > 5:
                consecutive_color = "#fc0"
            if info["max_consecutive_na"] > 20:
                consecutive_color = "#c00"
            zeros_color = "#0c0" if info["zero_count"] == 0 else "#c00"
            negatives_color = "#0c0" if info["negative_count"] == 0 else "#c00"
            html_output += f"""
            <tr>
              <td style="border: 1px solid #ddd; padding: 8px;">{asset}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{first_date_str}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{last_date_str}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {missing_color};">{info["missing_percentage"]:.2f}%</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {consecutive_color};">{info["max_consecutive_na"]}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {zeros_color};">{info["zero_count"]}</td>
              <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {negatives_color};">{info["negative_count"]}</td>
            </tr>
            """
        html_output += "</table>"
        if hasattr(self, 'valid_period_start') and self.valid_period_start is not None:
            common_period_days = (self.valid_period_end - self.valid_period_start).days
            common_period_years = common_period_days / 365.25
            html_output += f"""
            <div style="margin-top: 15px;">
              <p><strong>共通有効期間:</strong> {self.valid_period_start.strftime("%Y-%m-%d")} から {self.valid_period_end.strftime("%Y-%m-%d")}</p>
              <p><strong>期間長:</strong> {common_period_days}日間 (約{common_period_years:.1f}年)</p>
            </div>
            """
        display(HTML(html_output))

    def display_fetch_summary_text(self):
        if self.price_data is None or self.price_data.empty:
            print("=========================================")
            print("❌ データ取得に失敗しました")
            print("=========================================")
            print("価格データが取得できませんでした。設定を見直してから再試行してください。")
            return

        assets_info = []
        for asset in self.price_data.columns:
            asset_data = self.price_data[asset].dropna()
            if not asset_data.empty:
                first_date = asset_data.index[0]
                last_date = asset_data.index[-1]
                days = len(asset_data)
                years = round(days / 252, 1)
                assets_info.append({
                    "asset": asset,
                    "start_date": first_date.strftime('%Y-%m-%d'),
                    "end_date": last_date.strftime('%Y-%m-%d'),
                    "years": years
                })

        print("=========================================")
        print("✅ データ取得完了")
        print("=========================================")
        print()
        print("【取得資産】")
        print(", ".join([info["asset"] for info in assets_info]))
        print()
        if hasattr(self, 'valid_period_start') and self.valid_period_start is not None:
            common_days = (self.valid_period_end - self.valid_period_start).days
            common_years = round(common_days / 365.25, 1)
            print("【共通データ期間】")
            print(f"開始日: {self.valid_period_start.strftime('%Y-%m-%d')}")
            print(f"終了日: {self.valid_period_end.strftime('%Y-%m-%d')}")
            print(f"期間長: {common_days}日間 (約{common_years}年)")
            print()
        if self.performance_periods == "Single Period":
            lookback_info = f"{self.lookback_period}{'ヶ月' if self.lookback_unit == 'Months' else '日間'}"
            if self.lookback_unit == 'Months' and self.lookback_period >= 12:
                years_val = self.lookback_period // 12
                months_val = self.lookback_period % 12
                lookback_info += f"（{years_val}年"
                if months_val > 0:
                    lookback_info += f"{months_val}ヶ月"
                lookback_info += "）"
            print("【設定ルックバック期間】")
            print(lookback_info)
            print()
        else:
            print("【ルックバック期間設定（複数期間使用）】")
            max_lookback = 0
            max_unit = "Months"
            for period in self.multiple_periods:
                if period.get("length") is not None and period.get("weight", 0) > 0:
                    length = period["length"]
                    unit = period["unit"]
                    weight = period["weight"]
                    if unit == "Months" and length > max_lookback:
                        max_lookback = length
                        max_unit = "Months"
                    elif unit == "Days" and (max_unit == "Days" or length > max_lookback * 30):
                        max_lookback = length
                        max_unit = "Days"
                    period_info = f"{length}{'ヶ月' if unit == 'Months' else '日間'}"
                    if unit == 'Months' and length >= 12:
                        years_val = length // 12
                        months_val = length % 12
                        period_info += f"（{years_val}年"
                        if months_val > 0:
                            period_info += f"{months_val}ヶ月"
                        period_info += "）"
                    print(f"- {period_info}: {weight}%")
            print()
        if self.performance_periods == "Single Period":
            if self.lookback_unit == "Months":
                effective_start = self.valid_period_start + relativedelta(months=self.lookback_period)
            else:
                effective_start = self.valid_period_start + timedelta(days=self.lookback_period)
        else:
            if max_unit == "Months":
                effective_start = self.valid_period_start + relativedelta(months=max_lookback)
            else:
                effective_start = self.valid_period_start + timedelta(days=max_lookback)
        if effective_start <= self.valid_period_end:
            effective_days = (self.valid_period_end - effective_start).days
            effective_years = round(effective_days / 365.25, 1)
            print("【実行可能バックテスト期間】")
            print(f"開始日: {effective_start.strftime('%Y-%m-%d')} (ルックバック期間適用後)")
            print(f"終了日: {self.valid_period_end.strftime('%Y-%m-%d')}")
            print(f"期間長: {effective_days}日間 (約{effective_years}年)")
            print()
        print("-----------------------------------------")
        print("詳細資産情報:")
        print("-----------------------------------------")
        print("資産    開始日        終了日        データ期間")
        for info in assets_info:
            print(f"{info['asset']:<8}{info['start_date']:<14}{info['end_date']:<14}{info['years']}年")
        print()
        print("=========================================")
        print("「Run Backtest」ボタンをクリックして")
        print("バックテストを実行できます。")
        print("=========================================")

    def fetch_data(self):
        self.clear_cache()
        valid, errors, warnings_list = self.validate_parameters()
        if not valid:
            logger.error("パラメータ検証に失敗しました:")
            for error in errors:
                logger.error(f"- {error}")
            return False
        if warnings_list:
            logger.warning("検証で警告が発生しました:")
            for warning in warnings_list:
                logger.warning(f"- {warning}")
        start_date = f"{self.start_year-3}-{self.start_month:02d}-01"
        _, last_day = calendar.monthrange(self.end_year, self.end_month)
        end_date = f"{self.end_year}-{self.end_month:02d}-{last_day}"
        # --- ここを書き換える ----------------------------------------
        from itertools import chain

        all_assets = list(chain(
            self.tickers,                       # 投資対象
            [self.absolute_momentum_asset],     # 絶対モメンタム資産
            self.out_of_market_assets,          # 退避先資産（リストをそのまま展開）
            [self.benchmark_ticker]             # ベンチマーク
        ))
        all_assets = [
            asset for asset in all_assets
            if asset                       # None でない
            and asset.lower() not in ("none", "cash")   # "None" や "cash" を除外
        ]
        # --------------------------------------------------------------

        if not all_assets:
            logger.error("有効な資産がリストにありません。")
            return False
        logger.info(f"データ取得期間: {start_date} から {end_date}")
        logger.info(f"対象資産数: {len(all_assets)} - {', '.join(all_assets)}")
        batch_size = 10
        price_data_batches = []
        batches = [all_assets[i:i+batch_size] for i in range(0, len(all_assets), batch_size)]

        def download_batch(batch):
            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False
                )
                # 終値と始値の両方を取得
                close_data = data['Close']
                open_data = data['Open']
                # 列名をOpen_とClose_のプレフィックスを付けて区別
                open_data.columns = [f"Open_{col}" for col in open_data.columns]
                # 横方向に結合
                combined_data = pd.concat([close_data, open_data], axis=1)
                return combined_data if not combined_data.empty else None
            except Exception as e:
                logger.error(f"バッチ {batch} のデータ取得に失敗: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_batch, batch) for batch in batches]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(batches),
                desc="データ取得中",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            ):
                batch_data = future.result()
                if batch_data is not None:
                    price_data_batches.append(batch_data)
        if not price_data_batches:
            logger.error("全てのバッチでデータ取得に失敗しました。")
            return False
        try:
            self.price_data = pd.concat(price_data_batches, axis=1)
            self.price_data = self.price_data.loc[:, ~self.price_data.columns.duplicated()]
            self.price_data = self.price_data.astype('float32')
            logger.info(f"データ取得完了: {len(self.price_data)} 日分, {len(self.price_data.columns)} 銘柄")
            self.monthly_data = self.price_data.resample('ME').last()
            self._fetch_risk_free_rate(start_date, end_date)
            self._validate_data_periods(all_assets)
            self._last_data_fetch = datetime.now()
            quality_ok, quality_warnings = self.check_data_quality()
            if quality_warnings:
                logger.warning("データ品質チェックで警告が発生しました:")
                for warning in quality_warnings:
                    logger.warning(f"- {warning}")
            # --- fetch_data() の最後、display_fetch_summary_text() の直前に -------------
            #   Multiple Periods 用の厳密な effective_start を計算
            if self.performance_periods == "Single Period":
                if self.lookback_unit == "Months":
                    effective_start = self.valid_period_start + relativedelta(months=self.lookback_period)
                else:
                    effective_start = self.valid_period_start + timedelta(days=self.lookback_period)
            else:
                # ① 重みが正 (>0) の期間だけ対象
                candidate_starts = []
                for p in self.multiple_periods:
                    length = p.get("length")
                    unit   = p.get("unit")
                    weight = p.get("weight", 0)
                    if length is None or length <= 0 or weight <= 0:
                        continue

                    if unit == "Months":
                        cand = self.valid_period_start + relativedelta(months=length)
                    else:                       # "Days"
                        cand = self.valid_period_start + timedelta(days=length)

                    candidate_starts.append(cand)

                # ② 期間が 1 つも無効ならフォールバック
                if candidate_starts:
                    effective_start = max(candidate_starts)   # ←最も遅い日
                else:
                    effective_start = self.valid_period_start
            # ------------------------------------------------------------------------------

            self.bt_start_date = effective_start
            self.bt_end_date   = self.valid_period_end


            self.display_fetch_summary_text()
            
            return True
        except Exception as e:
            logger.error(f"データ結合中にエラーが発生しました: {e}")
            return False

    def _fetch_risk_free_rate(self, start_date, end_date):
        """リスクフリーレートを取得するメソッド（FRED API DTB3を優先、失敗時はIRXにフォールバック）"""

        # DTB3データをFRED APIから取得を試みる
        try:
            # fredapiパッケージを使用
            from fredapi import Fred

            # APIキーを設定（実際のAPIキーに置き換えてください）
            fred = Fred(api_key='a8d44f5fee887e9c844a783374065be4')

            # DTB3データを取得
            logger.info(f"FRED APIからDTB3データを取得中... ({start_date} から {end_date})")
            dtb3_data = fred.get_series('DTB3', observation_start=start_date, observation_end=end_date)

            # データが取得できたかチェック
            if dtb3_data.empty:
                logger.warning("DTB3データが空です。IRXデータにフォールバックします。")
                return self._fetch_risk_free_rate_irx(start_date, end_date)

            # IRXと同様の計算方法で年率を月次・日次レートに変換
            logger.info("DTB3データからリスクフリーレートを計算中...")

            # 月次レート計算（年率→月率）
            rfr_data = ((1 + dtb3_data / 100) ** (1/12)) - 1
            self.rfr_data = rfr_data.resample('ME').last()

            # 日次レート計算（年率→日率）
            rfr_data_daily = ((1 + dtb3_data / 100) ** (1/252)) - 1
            self.rfr_data_daily = rfr_data_daily

            # データフレーム形式の場合はシリーズに変換
            if isinstance(self.rfr_data, pd.DataFrame):
                self.rfr_data = self.rfr_data.iloc[:, 0] if not self.rfr_data.empty else pd.Series(0.001, index=self.monthly_data.index)
            if isinstance(self.rfr_data_daily, pd.DataFrame):
                self.rfr_data_daily = self.rfr_data_daily.iloc[:, 0]

            # データソース情報を保存（オプション）
            self._risk_free_rate_source = "DTB3 (FRED API)"

            logger.info("DTB3データを使用したリスクフリーレート設定完了（複利換算式を使用）")
            return True

        except ImportError as e:
            logger.warning(f"fredapiのインポートに失敗: {e} - IRXデータにフォールバック")
            return self._fetch_risk_free_rate_irx(start_date, end_date)

        except Exception as e:
            logger.warning(f"DTB3データ取得中にエラー発生: {e} - IRXデータにフォールバック")
            return self._fetch_risk_free_rate_irx(start_date, end_date)

    def _fetch_risk_free_rate_irx(self, start_date, end_date):
        """IRXデータを使用したリスクフリーレート取得（フォールバック方法）"""
        try:
            logger.info(f"yfinanceからIRXデータを取得中... ({start_date} から {end_date})")
            irx_data = yf.download("^IRX", start=start_date, end=end_date, auto_adjust=True)['Close']

            # データが空の場合はデフォルト値を使用
            if irx_data.empty:
                logger.warning("IRXデータが空です。デフォルト値を使用します。")
                self.rfr_data = pd.Series(0.001, index=self.monthly_data.index)
                self.rfr_data_daily = pd.Series(0.001/252, index=self.price_data.index)

                # データソース情報を保存（オプション）
                self._risk_free_rate_source = "デフォルト値"
                return False

            # 月次レート計算（年率→月率）
            rfr_data = ((1 + irx_data / 100) ** (1/12)) - 1
            self.rfr_data = rfr_data.resample('ME').last()

            # 日次レート計算（年率→日率）
            rfr_data_daily = ((1 + irx_data / 100) ** (1/252)) - 1
            self.rfr_data_daily = rfr_data_daily

            # データフレーム形式の場合はシリーズに変換
            if isinstance(self.rfr_data, pd.DataFrame):
                self.rfr_data = self.rfr_data.iloc[:, 0] if not self.rfr_data.empty else pd.Series(0.001, index=self.monthly_data.index)
            if isinstance(self.rfr_data_daily, pd.DataFrame):
                self.rfr_data_daily = self.rfr_data_daily.iloc[:, 0]

            # データソース情報を保存（オプション）
            self._risk_free_rate_source = "IRX (Yahoo Finance)"

            logger.info("IRXデータを使用したリスクフリーレート設定完了（複利換算式を使用）")
            return True

        except Exception as e:
            logger.warning(f"IRXデータ取得中にエラー発生: {e} - デフォルト値を使用します")
            self.rfr_data = pd.Series(0.001, index=self.monthly_data.index)
            self.rfr_data_daily = pd.Series(0.001/252, index=self.price_data.index)

            # データソース情報を保存（オプション）
            self._risk_free_rate_source = "デフォルト値"
            return False

    def get_risk_free_rate_source(self):
        """現在使用中のリスクフリーレートのデータソースを返す"""
        if hasattr(self, '_risk_free_rate_source'):
            return self._risk_free_rate_source
        else:
            return "未設定（データ取得前）"

    def display_trade_history(self, display_table=True):
        """
        取引履歴テーブルを表示する関数

        Args:
            display_table: HTMLテーブルを表示するかどうか (デフォルト: True)

        Returns:
            pd.DataFrame: 取引履歴のデータフレーム
        """
        if not hasattr(self, 'positions') or not self.positions:
            if display_table:
                print("取引履歴がありません。まずバックテストを実行してください。")
            return None

        # サマリーデータの生成
        summary = []
        for position in self.positions:
            signal_date = position.get("signal_date")
            start_date = position.get("start_date")
            end_date = position.get("end_date")
            assets = position.get("assets", [])
            ret = position.get("return")
            message = position.get("message", "")
            abs_return = position.get("abs_return")
            rfr_return = position.get("rfr_return")

            summary.append({
                "シグナル判定日": signal_date.date() if signal_date else None,
                "保有開始日": start_date.date() if start_date else None,
                "保有終了日": end_date.date() if end_date else None,
                "保有資産": ', '.join(assets),
                "保有期間リターン": f"{ret*100:.2f}%" if ret is not None else "N/A",
                "モメンタム判定結果": message,
                "絶対モメンタムリターン": f"{abs_return*100:.2f}%" if abs_return is not None else "N/A",
                "リスクフリーレート": f"{rfr_return*100:.2f}%" if rfr_return is not None else "N/A"
            })

        # データフレーム作成
        if summary:
            summary_df = pd.DataFrame(summary)
            columns = ["シグナル判定日", "保有開始日", "保有終了日", "保有資産", "保有期間リターン",
                    "モメンタム判定結果", "絶対モメンタムリターン", "リスクフリーレート"]

            # 列が存在することを確認してから列順序を設定
            avail_columns = [col for col in columns if col in summary_df.columns]
            summary_df = summary_df[avail_columns]

            # 表示が要求された場合のみ表示
            if display_table:
                display(HTML("""
                <h2 style="color:#3367d6;">取引履歴</h2>
                """ + summary_df.to_html(index=False, classes='table table-striped')))

            return summary_df

        return None

    def _create_holdings_from_assets(self, selected_assets):
        """資産リストから保有比率を作成するヘルパーメソッド"""
        holdings = {}
        if selected_assets:
            weight_per_asset = 1.0 / len(selected_assets)
            for asset in selected_assets:
                if asset.lower() == 'cash':
                    holdings['Cash'] = weight_per_asset
                elif asset in self.price_data.columns:
                    holdings[asset] = weight_per_asset
                else:
                    logger.warning(f"警告: 選択資産 {asset} がデータに存在しません")
        return holdings

    def _validate_data_periods(self, all_assets):
        data_availability = {}
        valid_period_start = {}
        valid_period_end = {}
        relevant_assets = set(self.tickers + [self.absolute_momentum_asset] +
                              self.out_of_market_assets + [self.benchmark_ticker])
        relevant_assets = {asset for asset in relevant_assets if asset != 'None' and asset.lower() != 'cash'}
        for asset in all_assets:
            if asset in self.price_data.columns:
                asset_data = self.price_data[asset].dropna()
                if len(asset_data) > 0:
                    first_date = asset_data.index[0]
                    last_date = asset_data.index[-1]
                    data_availability[asset] = {
                        'start_date': first_date.strftime('%Y-%m-%d'),
                        'end_date': last_date.strftime('%Y-%m-%d'),
                        'days': len(asset_data),
                        'years': round(len(asset_data) / 252, 1)
                    }
                    if asset in relevant_assets:
                        valid_period_start[asset] = first_date
                        valid_period_end[asset] = last_date
        if valid_period_start and valid_period_end:
            common_start = max(valid_period_start.values())
            common_end = min(valid_period_end.values())
            if common_start <= common_end:
                logger.info(f"\n全対象資産共通の有効期間: {common_start.strftime('%Y-%m-%d')} から {common_end.strftime('%Y-%m-%d')}")
                logger.info(f"推奨バックテスト期間: {common_start.year}/{common_start.month} - {common_end.year}/{common_end.month}")
                self.valid_period_start = common_start
                self.valid_period_end = common_end
            else:
                logger.warning("\n警告: 全対象資産に共通する有効期間がありません。")
        # 標準出力は削除済み

    def _calculate_single_asset_return(self, data, asset, start_date, end_date):
        """特定の2日付間の正確なリターンを計算"""
        try:
            # 日付を標準化
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # 対象資産のチェック
            if asset not in data.columns:
                logger.debug(f"資産 {asset} はデータに存在しません")
                return None

            # 日付存在チェック（重要）
            if start_date not in data.index:
                logger.warning(f"開始日 {start_date.strftime('%Y-%m-%d')} のデータがありません: {asset}")
                return None

            if end_date not in data.index:
                logger.warning(f"終了日 {end_date.strftime('%Y-%m-%d')} のデータがありません: {asset}")
                return None

            # データ取得と有効性チェック
            start_price = data.loc[start_date, asset]
            end_price = data.loc[end_date, asset]

            if pd.isna(start_price) or pd.isna(end_price):
                logger.warning(f"資産 {asset} のデータに欠損があります")
                return None

            # リターン計算
            if start_price <= 0:
                logger.warning(f"⚠️ 資産 {asset} の開始価格が0以下です: {start_price}")
                return None

            ret = (end_price / start_price) - 1

            # 極端なリターンをチェック（警告のみ）
            period_days = (end_date - start_date).days
            if abs(ret) > 1.0 and period_days < 365:  # 100%以上の変動かつ1年未満
                logger.warning(f"異常リターン: {asset} が {period_days} 日間で {ret*100:.1f}%")

            return ret

        except Exception as e:
            logger.error(f"リターン計算エラー ({asset}): {e}")
            return None

    def calculate_monthly_momentum(self, asset, current_date, lookback_months):
        """厳密なルールに基づく月次モメンタム計算
        ルール：前月の最終取引日の終値～当月の最終取引日の終値
        月中実行時は、当月の最新データを使用
        """
        # 日付のパース
        current_date = pd.to_datetime(current_date)

        # この日付までのデータに制限
        available_price_data = self.price_data[self.price_data.index <= current_date]

        if available_price_data.empty:
            logger.warning(f"{current_date.strftime('%Y-%m-%d')}以前のデータがありません")
            return None

        # 計算に使う年月を特定
        current_year = current_date.year
        current_month = current_date.month

        # 当月の取引日を全て取得
        current_month_dates = available_price_data.index[
            (available_price_data.index.year == current_year) &
            (available_price_data.index.month == current_month)
        ]

        # データチェック
        if current_month_dates.empty:
            logger.warning(f"{current_year}年{current_month}月のデータがありません")
            return None

        # 当月の最終取引日
        current_month_last_trading_day = current_month_dates[-1]

        # 終値の日付を決定
        end_trading_day = current_month_last_trading_day

        # 前月の計算（厳密に月数で遡る）
        target_month = current_month
        target_year = current_year

        # lookback_months分だけ月を遡る
        for _ in range(lookback_months):
            if target_month == 1:
                target_month = 12
                target_year -= 1
            else:
                target_month -= 1

        # 遡った月の取引日を取得
        prior_month_dates = available_price_data.index[
            (available_price_data.index.year == target_year) &
            (available_price_data.index.month == target_month)
        ]

        if prior_month_dates.empty:
            logger.warning(f"{target_year}年{target_month}月のデータがありません")
            return None

        # 前月の最終取引日
        start_trading_day = prior_month_dates[-1]

        # 計算に使用する日付をログ出力
        logger.info(f"モメンタム計算: {asset}, {start_trading_day.strftime('%Y-%m-%d')} から {end_trading_day.strftime('%Y-%m-%d')}")

        # 以下リターン計算...（既存のコード）

        # リターン計算
        if asset not in self.price_data.columns:
            logger.warning(f"資産 {asset} はデータに存在しません")
            return None

        try:
            # 直接価格を取得（_calculate_single_asset_returnではなく）
            if start_trading_day not in self.price_data.index or end_trading_day not in self.price_data.index:
                logger.warning(f"計算に必要な日付のデータがありません: {start_trading_day} - {end_trading_day}")
                return None

            start_price = self.price_data.loc[start_trading_day, asset]
            end_price = self.price_data.loc[end_trading_day, asset]

            if pd.isna(start_price) or pd.isna(end_price):
                logger.warning(f"資産 {asset} のデータに欠損があります")
                return None

            if start_price <= 0:
                logger.warning(f"資産 {asset} の開始価格が無効です: {start_price}")
                return None

            return (end_price / start_price) - 1
        except Exception as e:
            logger.error(f"モメンタム計算エラー ({asset}): {e}")
            return None

    def calculate_daily_momentum(self, asset, current_date, lookback_days):
        """厳密な日数に基づくモメンタム計算
        N日前の取引日から現在日までのリターンを計算
        """
        # 日付のパース
        current_date = pd.to_datetime(current_date)

        # この日付までのデータに制限
        available_price_data = self.price_data[self.price_data.index <= current_date]

        if available_price_data.empty:
            logger.warning(f"{current_date.strftime('%Y-%m-%d')}以前のデータがありません")
            return None

        # 当日の取引日を特定
        current_dates = available_price_data.index[available_price_data.index <= current_date]
        if current_dates.empty:
            logger.warning(f"{current_date.strftime('%Y-%m-%d')}のデータがありません")
            return None

        # 当日の終値の日付を決定
        end_trading_day = current_dates[-1]

        # N日前の日付を計算
        target_date = end_trading_day - pd.Timedelta(days=lookback_days)

        # N日前に最も近い取引日を取得（指定日以前の最終取引日）
        prior_dates = available_price_data.index[available_price_data.index <= target_date]
        if prior_dates.empty:
            logger.warning(f"{target_date.strftime('%Y-%m-%d')}以前のデータがありません")
            return None

        # N日前の取引日
        start_trading_day = prior_dates[-1]

        # 計算に使用する日付をログ出力
        logger.info(f"日次モメンタム計算: {asset}, {start_trading_day.strftime('%Y-%m-%d')} から {end_trading_day.strftime('%Y-%m-%d')}")

        # リターン計算
        if asset not in self.price_data.columns:
            logger.warning(f"資産 {asset} はデータに存在しません")
            return None

        try:
            # 直接価格を取得
            if start_trading_day not in self.price_data.index or end_trading_day not in self.price_data.index:
                logger.warning(f"計算に必要な日付のデータがありません: {start_trading_day} - {end_trading_day}")
                return None

            start_price = self.price_data.loc[start_trading_day, asset]
            end_price = self.price_data.loc[end_trading_day, asset]

            if pd.isna(start_price) or pd.isna(end_price):
                logger.warning(f"資産 {asset} のデータに欠損があります")
                return None

            if start_price <= 0:
                logger.warning(f"資産 {asset} の開始価格が無効です: {start_price}")
                return None

            return (end_price / start_price) - 1
        except Exception as e:
            logger.error(f"日次モメンタム計算エラー ({asset}): {e}")
            return None

    def _calculate_asset_returns(self, data, assets, start_date, end_date):
        returns = {}
        for asset in assets:
            returns[asset] = self._calculate_single_asset_return(data, asset, start_date, end_date)
        return returns

    def _calculate_rfr_return(self, decision_date, default=0.01):
        """
        リスクフリーレートを取得する
        新しい月次モメンタム計算に合わせて修正
        """
        decision_date = pd.to_datetime(decision_date)

        if self.rfr_data is None or self.rfr_data.empty:
            return default

        # 指定日付以前の最新のRFRデータを取得
        available = self.rfr_data[self.rfr_data.index <= decision_date]
        if len(available) > 0:
            # 最新の月次RFRを取得
            return available.iloc[-1]
        else:
            return default

    def _evaluate_out_of_market_assets(self, as_of_date):
        """
        退避先資産のモメンタムを評価し、戦略に応じて資産を選択する

        Parameters:
        as_of_date (datetime): 評価日

        Returns:
        list: 選択された退避先資産のリスト
        """
        # 退避先資産が1つ以下の場合は、そのまま返す
        if len(self.out_of_market_assets) <= 1:
            return self.out_of_market_assets

        # 「等ウェイト」モードの場合は、全ての退避先資産を返す
        if self.out_of_market_strategy == "Equal Weight":
            logger.info(f"退避先戦略: 等ウェイト - {self.out_of_market_assets}")
            return self.out_of_market_assets

        # 以下は「Top 1」モードの処理
        # 退避先資産のうち、実際にデータに存在する資産のみを対象とする
        target_assets = [asset for asset in self.out_of_market_assets
                        if asset in self.price_data.columns]

        if not target_assets:
            logger.warning("退避先資産がデータに存在しません。元のリストを使用します。")
            return self.out_of_market_assets

        # キャッシュキーの生成（通常のモメンタム計算と区別するために接頭辞をつける）
        cache_key = "safe_" + self._generate_cache_key(as_of_date)
        cached_results = self._get_from_cache(cache_key)

        if cached_results is not None:
            logger.debug(f"退避先資産評価: キャッシュヒット {cache_key}")
            sorted_assets = cached_results.get("sorted_assets", [])
        else:
            logger.debug(f"退避先資産評価: キャッシュミス {cache_key}")

            # シングル期間モードの処理
            if self.performance_periods == "Single Period":
                # 各資産のモメンタム計算
                returns = {}
                for asset in target_assets:
                    # 単位に応じた適切なメソッド使用
                    if self.lookback_unit == "Months":
                        ret = self.calculate_monthly_momentum(asset, as_of_date, self.lookback_period)
                    else:  # Days
                        ret = self.calculate_daily_momentum(asset, as_of_date, self.lookback_period)

                    if ret is not None:
                        returns[asset] = ret
                    else:
                        logger.warning(f"退避先資産 {asset} のモメンタム計算に失敗")

                # リターンでソート
                sorted_assets = sorted(returns.items(), key=lambda x: x[1], reverse=True)

            # 複数期間モードの処理
            else:
                # 既存の複数期間計算メソッドを再利用
                period_returns = self._calculate_multiple_period_returns_unified(as_of_date, target_assets)

                if self.weighting_method == "Weight Performance":
                    weighted_returns = self._calculate_weighted_performance(period_returns, target_assets)
                    sorted_assets = sorted(weighted_returns.items(), key=lambda x: x[1], reverse=True)
                else:
                    weighted_ranks = self._calculate_weighted_ranks(period_returns, target_assets)
                    sorted_assets = sorted(weighted_ranks.items(), key=lambda x: x[1], reverse=True)

            # 結果をキャッシュに保存
            self._save_to_cache(cache_key, {"sorted_assets": sorted_assets})

        # 上位1銘柄を選択
        if sorted_assets:
            top_asset = sorted_assets[0][0]
            top_value = sorted_assets[0][1]

            # 他の資産の結果も詳細ログに出力
            detail_str = ", ".join([f"{a}:{v:.2%}" if isinstance(v, float) else f"{a}:{v:.2f}"
                                   for a, v in sorted_assets])

            # 情報ログに選択結果を出力
            logger.info(f"退避先戦略: Top 1 - 選択資産 {top_asset} (値: {top_value:.4f})")
            logger.debug(f"退避先資産の全評価結果: {detail_str}")

            return [top_asset]

        # 計算に失敗した場合は元のリストを返す
        logger.warning("退避先資産の評価に失敗しました。元のリストを使用します。")
        return self.out_of_market_assets

    def calculate_cumulative_rfr_return(self, end_date, lookback_months):
        """期間に応じた累積リスクフリーレートを計算"""
        end_date = pd.to_datetime(end_date)

        # 開始日の計算
        start_date = end_date - relativedelta(months=lookback_months)

        # 期間内のリスクフリーレートを取得（end_dateまでのデータのみ）
        if self.rfr_data is None or self.rfr_data.empty:
            logger.warning("リスクフリーレートデータがないため、デフォルト値を使用")
            return 0.01 * (lookback_months/12)  # 年率1%の月割り

        # 該当期間のリスクフリーレートを抽出（end_date以前のデータのみ）
        available_rfr = self.rfr_data[self.rfr_data.index <= end_date]
        period_rfr = available_rfr[(available_rfr.index >= start_date) &
                                (available_rfr.index <= end_date)]

        if period_rfr.empty:
            logger.warning(f"期間 {start_date} - {end_date} のリスクフリーレートデータがありません")
            return 0.01 * (lookback_months/12)  # 年率1%の月割り

        # 複利計算で累積リターンを計算
        cumulative_rfr = (1 + period_rfr).prod() - 1

        logger.info(f"期間 {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        # 月数の表示を修正
        month_difference = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        logger.info(f"累積リスクフリーレート: {cumulative_rfr:.4f} ({month_difference}ヶ月分)")

        return cumulative_rfr

    def calculate_cumulative_rfr_return_days(self, end_date, lookback_days):
        """日数に応じた累積リスクフリーレートを計算"""
        end_date = pd.to_datetime(end_date)

        # 開始日の計算
        start_date = end_date - pd.Timedelta(days=lookback_days)

        # 期間内のリスクフリーレートを取得（end_dateまでのデータのみ）
        if self.rfr_data_daily is None or self.rfr_data_daily.empty:
            logger.warning("日次リスクフリーレートデータがないため、デフォルト値を使用")
            return 0.01 * (lookback_days/365)  # 年率1%の日割り

        # 該当期間のリスクフリーレートを抽出（end_date以前のデータのみ）
        available_rfr = self.rfr_data_daily[self.rfr_data_daily.index <= end_date]
        period_rfr = available_rfr[(available_rfr.index >= start_date) &
                                (available_rfr.index <= end_date)]

        if period_rfr.empty:
            logger.warning(f"期間 {start_date} - {end_date} の日次リスクフリーレートデータがありません")
            return 0.01 * (lookback_days/365)  # 年率1%の日割り

        # 複利計算で累積リターンを計算
        cumulative_rfr = (1 + period_rfr).prod() - 1

        logger.info(f"日次期間 {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"累積日次リスクフリーレート: {cumulative_rfr:.4f} ({lookback_days}日分)")

        return cumulative_rfr

    def _evaluate_absolute_momentum(self, data, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if self.absolute_momentum_asset not in data.columns:
            logger.warning(f"警告: 絶対モメンタム資産 {self.absolute_momentum_asset} が存在しません")
            return True, 0.0
        lqd_return = self._calculate_single_asset_return(data, self.absolute_momentum_asset, start_date, end_date)
        rfr_series = self.rfr_data[self.rfr_data.index >= start_date]
        rfr_series = rfr_series[rfr_series.index <= end_date]
        if rfr_series.empty:
            rfr_total = 0.01
        else:
            compounded = (1 + rfr_series).prod()
            rfr_total = compounded - 1
        excess_return = lqd_return - rfr_total
        logger.info(f"評価期間 {start_date.date()} ~ {end_date.date()} で、LQDリターン: {lqd_return:.2%}, RFR累積リターン: {rfr_total:.2%}, 超過リターン: {excess_return:.2%}")
        return absolute_momentum_pass, excess_return



    def _calculate_multiple_period_returns_unified(self, end_date, assets):
        """シングルピリオドと同一の計算法を使用した複数期間リターン計算"""
        period_returns = {}

        # 計算日をログ出力
        logger.info(f"計算日: {end_date.strftime('%Y-%m-%d')}")

        for period_idx in range(self.multiple_periods_count):
            period = self.multiple_periods[period_idx]
            length, unit = period.get("length"), period.get("unit")

            if length is None or length <= 0:
                continue

            # 各資産のリターンを計算
            period_returns[period_idx] = {}

            for asset in assets:
                # 単位に応じて適切なメソッドを使用
                if unit == "Months":
                    # 月単位の計算
                    asset_return = self.calculate_monthly_momentum(asset, end_date, length)
                else:
                    # 日数単位の計算 - 新しいメソッドを使用
                    asset_return = self.calculate_daily_momentum(asset, end_date, length)

                if asset_return is not None:
                    period_returns[period_idx][asset] = asset_return
                    logger.debug(f"期間 {length} {unit}, 資産 {asset}: リターン {asset_return:.2%}")
                else:
                    logger.warning(f"期間 {length} {unit}, 資産 {asset} のリターン計算ができませんでした。")

        return period_returns

    def _validate_and_normalize_weights(self, weights):
        valid_weights = [w for w in weights if w is not None and w > 0]
        if not valid_weights:
            logger.warning("有効な重みがありません。デフォルト値として均等配分を使用します。")
            return [1.0 / len(weights)] * len(weights)
        total_weight = sum(valid_weights)
        if abs(total_weight - 100) <= 0.001:
            return valid_weights
        logger.info(f"重みの合計が100%ではありません ({total_weight:.2f}%)。正規化を実行します。")
        normalized_weights = [w * (100 / total_weight) for w in valid_weights]
        return normalized_weights

    def _calculate_weighted_performance(self, period_returns, assets):
        weighted_returns = {}
        for asset in assets:
            weighted_return = 0.0
            total_weight = 0.0
            weights = []
            returns = []
            for period_idx in range(self.multiple_periods_count):
                if period_idx in period_returns and asset in period_returns[period_idx]:
                    weight = self.multiple_periods[period_idx]["weight"] / 100.0
                    weights.append(weight)
                    returns.append(period_returns[period_idx][asset])
                    total_weight += weight
            if total_weight > 0:
                normalized_weights = self._validate_and_normalize_weights([w * 100 for w in weights])
                normalized_weights = [w / 100 for w in normalized_weights]
                for i, weight in enumerate(normalized_weights):
                    if returns[i] is not None: weighted_return += weight * returns[i]
                weighted_returns[asset] = weighted_return
            else:
                weighted_returns[asset] = 0.0
        return weighted_returns

    def _calculate_weighted_ranks(self, period_returns, assets):
        period_ranks = {}
        for period_idx in period_returns:
            sorted_period_assets = sorted(period_returns[period_idx].items(), key=lambda x: x[1], reverse=True)
            rank_scores = {}
            for rank, (asset, _) in enumerate(sorted_period_assets):
                rank_scores[asset] = len(sorted_period_assets) - rank
            period_ranks[period_idx] = rank_scores
        weighted_ranks = {}
        for asset in assets:
            weighted_rank = 0.0
            total_weight = 0.0
            weights = []
            ranks = []
            for period_idx in period_ranks:
                if asset in period_ranks[period_idx]:
                    weight = self.multiple_periods[period_idx]["weight"] / 100.0
                    weights.append(weight)
                    ranks.append(period_ranks[period_idx][asset])
                    total_weight += weight
            if total_weight > 0:
                normalized_weights = self._validate_and_normalize_weights([w * 100 for w in weights])
                normalized_weights = [w / 100 for w in normalized_weights]
                for i, weight in enumerate(normalized_weights):
                    if ranks[i] is not None: weighted_rank += weight * ranks[i]
                weighted_ranks[asset] = weighted_rank
            else:
                weighted_ranks[asset] = 0.0
        return weighted_ranks

    def _calculate_weighted_rfr_return(self, end_date):
        """
        複数期間の重み付きリスクフリーレートを計算する（修正版）
        """
        rfr_weighted_return = 0.0
        total_weight = 0.0
        weights = []
        rfr_returns = []

        for period_idx in range(self.multiple_periods_count):
            period = self.multiple_periods[period_idx]
            length, unit, weight_pct = period.get("length"), period.get("unit"), period.get("weight", 0)

            if length is not None and length > 0 and weight_pct > 0:
                weight = weight_pct / 100.0

                # 期間に応じたRFRリターン計算（ここを修正）
                if unit == "Months":
                    # 月単位の場合
                    period_rfr_return = self.calculate_cumulative_rfr_return(end_date, length)
                else:
                    # 日数単位の場合は日次計算メソッドを使用
                    period_rfr_return = self.calculate_cumulative_rfr_return_days(end_date, length)

                # None値チェック
                if period_rfr_return is None:
                    logger.warning(f"期間 {length} {unit} のRFR計算ができませんでした。デフォルト値を使用します。")
                    period_rfr_return = 0.001  # デフォルト値

                weights.append(weight_pct)
                rfr_returns.append(period_rfr_return)
                total_weight += weight
                logger.info(f"期間 {length} {unit}: RFRリターン {period_rfr_return:.4f}")

        if total_weight > 0 and rfr_returns:  # 空でないことを確認
            normalized_weights = self._validate_and_normalize_weights(weights)
            normalized_weights = [w / 100 for w in normalized_weights]

            for i, weight in enumerate(normalized_weights):
                rfr_weighted_return += weight * rfr_returns[i]

            logger.info(f"リスクフリーレート重み付けリターン: {rfr_weighted_return:.4f}")
            return rfr_weighted_return
        else:
            return 0.01  # デフォルト値

    def _calculate_weighted_absolute_momentum_unified(self, end_date):
        """シングルピリオドと同一の計算法を使用した重み付き絶対モメンタム計算"""
        abs_mom_weighted_return = 0.0
        total_weight = 0.0
        weights = []
        abs_returns = []
        successful_periods = []

        for period_idx in range(self.multiple_periods_count):
            period = self.multiple_periods[period_idx]
            length, unit, weight_pct = period.get("length"), period.get("unit"), period.get("weight", 0)

            if length is None or length <= 0 or weight_pct <= 0:
                continue

            # 単位に応じた適切なメソッドを使用
            if unit == "Months":
                # 月単位の計算
                period_return = self.calculate_monthly_momentum(
                    self.absolute_momentum_asset,
                    end_date,
                    length
                )
            else:
                # 日数単位の計算 - 新しいメソッドを使用
                period_return = self.calculate_daily_momentum(
                    self.absolute_momentum_asset,
                    end_date,
                    length
                )

            # 成功した計算のみ使用
            if period_return is not None:
                weights.append(weight_pct)
                abs_returns.append(period_return)
                total_weight += weight_pct
                successful_periods.append(f"{length} {unit}")
                logger.info(f"期間 {length} {unit}: リターン {period_return:.2%}")
            else:
                logger.warning(f"期間 {length} {unit} の絶対モメンタム計算ができませんでした。この期間はスキップします。")

        # 計算成功率とログ出力
        if successful_periods:
            success_rate = len(successful_periods) / len([p for p in self.multiple_periods if p.get("weight", 0) > 0])
            logger.info(f"絶対モメンタム計算: {len(successful_periods)} 期間成功 (成功率 {success_rate:.0%})")
            logger.info(f"計算成功期間: {', '.join(successful_periods)}")

        # 重み付け計算
        if total_weight > 0:
            # 重みの正規化
            normalized_weights = self._validate_and_normalize_weights(weights)
            normalized_weights = [w / 100 for w in normalized_weights]

            # 各期間の重み付けリターンを計算
            for i, weight in enumerate(normalized_weights):
                abs_mom_weighted_return += weight * abs_returns[i]

            logger.info(f"絶対モメンタム重み付けリターン: {abs_mom_weighted_return:.4f}")
            return abs_mom_weighted_return
        else:
            logger.warning("有効な絶対モメンタム計算期間がありませんでした。デフォルト値0.0を使用します。")
            return 0.0

    def _generate_cache_key(self, decision_date):
        """
        キャッシュキーを生成する（日付だけでなく設定情報も含める）
        """
        base_key = decision_date.strftime("%Y-%m-%d")

        # 設定情報をキーに含める
        config_hash = f"P{self.performance_periods}_L{self.lookback_period}_{self.lookback_unit}"

        # マルチ期間の設定をハッシュに含める
        if self.performance_periods == "Multiple Periods":
            periods_hash = "_".join([
                f"{p.get('length')}_{p.get('unit')}_{p.get('weight')}"
                for p in self.multiple_periods
                if p.get('length') is not None and p.get('weight', 0) > 0
            ])
            config_hash += f"_M{periods_hash}"

        return f"{base_key}_{config_hash}"

    def get_latest_rebalance_date(self, calc_date):
        year = calc_date.year
        month = calc_date.month
        return self._get_last_trading_day(year, month)

    def get_monthly_next_rebalance_candidate(self, calc_date):
        year = calc_date.year
        month = calc_date.month
        last_td = self._get_last_trading_day(year, month)
        return last_td

    def get_bimonthly_next_rebalance_candidate(self, calc_date):
        next_month_date = calc_date + relativedelta(months=1)
        return self._get_last_trading_day(next_month_date.year, next_month_date.month)

    def get_quarterly_next_rebalance_candidate(self, calc_date):
        quarter = ((calc_date.month - 1) // 3) + 1
        end_month = quarter * 3
        return self._get_last_trading_day(calc_date.year, end_month)

    def _get_last_trading_day(self, year, month):
        start_date = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        if schedule.empty:
            return pd.Timestamp(end_date)

    def get_last_trading_day_of_month(self, year, month):
        """指定された年月の最終取引日を取得（実際のデータに基づく）"""
        # 月末のカレンダー日を取得
        _, last_day = calendar.monthrange(year, month)
        month_end = pd.Timestamp(year=year, month=month, day=last_day)

        # 価格データから月内の全取引日を抽出
        dates_in_month = self.price_data.index[
            (self.price_data.index.year == year) &
            (self.price_data.index.month == month)
        ]

        if dates_in_month.empty:
            return None  # データなし

        # 月内の最後の取引日を返す
        return dates_in_month[-1]

    def get_latest_valid_rebalance_date(self, current_date):
        """
        Trading Frequency設定に基づいて、直近の有効なリバランス日（月末）を取得する

        Parameters:
        current_date (datetime): 現在の日付

        Returns:
        datetime: 直近の有効なリバランス日、または None
        """
        if not isinstance(current_date, pd.Timestamp):
            current_date = pd.to_datetime(current_date)

        current_year = current_date.year
        current_month = current_date.month

        # 月のリストを過去に向かって生成（当月を含む過去12ヶ月分）
        past_months = []
        for i in range(12):  # 最大12ヶ月遡る
            month = current_month - i
            year = current_year
            while month <= 0:
                month += 12
                year -= 1
            past_months.append((year, month))

        # Trading Frequencyに基づいて有効なリバランス月をフィルタリング
        valid_months = []
        if self.trading_frequency == "Monthly":
            valid_months = past_months
        elif self.trading_frequency == "Bimonthly (hold: 1,3,5,7,9,11)":
            valid_months = [(y, m) for y, m in past_months if m in [12, 2, 4, 6, 8, 10]]
        elif self.trading_frequency == "Bimonthly (hold: 2,4,6,8,10,12)":
            valid_months = [(y, m) for y, m in past_months if m in [1, 3, 5, 7, 9, 11]]
        elif self.trading_frequency == "Quarterly (hold: 1,4,7,10)":
            valid_months = [(y, m) for y, m in past_months if m in [12, 3, 6, 9]]
        elif self.trading_frequency == "Quarterly (hold: 2,5,8,11)":
            valid_months = [(y, m) for y, m in past_months if m in [1, 4, 7, 10]]
        elif self.trading_frequency == "Quarterly (hold: 3,6,9,12)":
            valid_months = [(y, m) for y, m in past_months if m in [2, 5, 8, 11]]

        # 最新の有効なリバランス月を取得（現在の月を含む）
        for year, month in valid_months:
            last_trading_day = self.get_last_trading_day_of_month(year, month)
            if last_trading_day is not None:
                # 月末が現在の日付より前であることを確認
                if last_trading_day <= current_date:
                    return last_trading_day

        # 有効なリバランス日が見つからない場合はNoneを返す
        return None

    def _get_first_trading_day(self, year, month):
        start_date = f"{year}-{month:02d}-01"
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=(pd.Timestamp(start_date) + pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
        if schedule.empty:
            return pd.Timestamp(start_date)
        return schedule.index[0]

    def calculate_momentum_ranks(self, as_of_date=None):
        """モメンタムランク計算 (指定日付まで)"""
        # キャッシュクリア
        self.clear_cache()

        # 日付指定がない場合は最新の取引日を使用
        if as_of_date is None:
            as_of_date = self.price_data.index[-1]
        else:
            as_of_date = pd.to_datetime(as_of_date)

        # 対象資産の確認
        target_assets = [ticker for ticker in self.tickers if ticker in self.price_data.columns]
        if not target_assets:
            logger.warning("対象資産がデータに存在しません。")
            return {"sorted_assets": [], "selected_assets": self.out_of_market_assets, "message": "対象資産がデータに存在しません"}

        # シングル期間モードの場合
        if self.performance_periods == "Single Period":
            # 各資産のモメンタム計算
            returns = {}
            for asset in target_assets:
                # 単位に応じた適切なメソッドを使用
                if self.lookback_unit == "Months":
                    ret = self.calculate_monthly_momentum(asset, as_of_date, self.lookback_period)
                else:  # Days
                    ret = self.calculate_daily_momentum(asset, as_of_date, self.lookback_period)

                if ret is not None:
                    returns[asset] = ret
                else:
                    logger.warning(f"資産 {asset} のモメンタム計算ができませんでした")

            sorted_assets = sorted(returns.items(), key=lambda x: x[1], reverse=True)

            # 絶対モメンタム評価
            if self.single_absolute_momentum == "Yes":
                abs_lookback = self.absolute_momentum_period if self.absolute_momentum_custom_period else self.lookback_period

                # 絶対モメンタム資産のリターン計算
                if self.lookback_unit == "Months":
                    abs_ret = self.calculate_monthly_momentum(self.absolute_momentum_asset, as_of_date, abs_lookback)
                    # 同期間の累積リスクフリーレート計算
                    rfr_return = self.calculate_cumulative_rfr_return(as_of_date, abs_lookback)
                else:  # Days
                    abs_ret = self.calculate_daily_momentum(self.absolute_momentum_asset, as_of_date, abs_lookback)
                    # 同期間の累積リスクフリーレート計算
                    rfr_return = self.calculate_cumulative_rfr_return_days(as_of_date, abs_lookback)

                # 判定
                if abs_ret is None:
                    absolute_momentum_pass = False
                    logger.warning(f"絶対モメンタム資産 {self.absolute_momentum_asset} のリターンが計算不能")
                else:
                    absolute_momentum_pass = abs_ret > rfr_return

                # 詳細なログ出力
                logger.info(f"絶対モメンタム評価 ({as_of_date.strftime('%Y-%m-%d')}):")
                logger.info(f"- 資産: {self.absolute_momentum_asset}, 期間: {abs_lookback}{self.lookback_unit.lower()}")
                logger.info(f"- リターン: {abs_ret:.4f} vs リスクフリーレート: {rfr_return:.4f}")
                logger.info(f"- 判定結果: {'合格' if absolute_momentum_pass else '失格'}")

                if absolute_momentum_pass:
                    selected_assets = [asset for asset, _ in sorted_assets[:self.assets_to_hold]]

                    # 追加: Negative relative momentumオプションがYes & トップ銘柄が負(<0)なら退避先へ
                    if self.negative_relative_momentum == "Yes" and len(sorted_assets) > 0:
                        top_asset, top_return = sorted_assets[0]
                        if top_return < 0:
                            # 修正: 新しいメソッドを使って退避先を選択
                            selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                            result_text = "Absolute: Passed but top RM < 0 -> Out of Market"
                        else:
                            result_text = "Absolute: Passed"
                    else:
                        result_text = "Absolute: Passed"

                else:
                    # 修正: 新しいメソッドを使って退避先を選択
                    selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                    result_text = "Absolute: Failed"

                logger.info(f"{result_text}: {selected_assets} を選択")

            else:
                selected_assets = [asset for asset, ret in sorted_assets[:self.assets_to_hold] if ret is not None and ret > 0]
                if not selected_assets:
                    selected_assets = self.out_of_market_assets
                result_text = "Relative Only"

        # 複数期間モードの場合
        else:
            # 統一された計算方法を使用
            period_returns = self._calculate_multiple_period_returns_unified(as_of_date, target_assets)

            if self.weighting_method == "Weight Performance":
                weighted_returns = self._calculate_weighted_performance(period_returns, target_assets)
                sorted_assets = sorted(weighted_returns.items(), key=lambda x: x[1], reverse=True)
            else:
                weighted_ranks = self._calculate_weighted_ranks(period_returns, target_assets)
                sorted_assets = sorted(weighted_ranks.items(), key=lambda x: x[1], reverse=True)

            if self.single_absolute_momentum == "Yes":
                # 統一された絶対モメンタム計算を使用
                rfr_weighted_return = self._calculate_weighted_rfr_return(as_of_date)
                abs_mom_weighted_return = self._calculate_weighted_absolute_momentum_unified(as_of_date)

                # 判定ロジックは従来通り（仕様どおり）
                absolute_momentum_pass = abs_mom_weighted_return > rfr_weighted_return

                if absolute_momentum_pass:
                    selected_assets = [asset for asset, _ in sorted_assets[:self.assets_to_hold]]

                    # 追加: Negative relative momentumオプション
                    if self.negative_relative_momentum == "Yes" and len(sorted_assets) > 0:
                        top_asset, top_return = sorted_assets[0]
                        if top_return < 0:
                            # 修正: 新しいメソッドを使って退避先を選択
                            selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                            result_text = "Absolute: Passed (Multiple) but top RM < 0 -> Out of Market"
                        else:
                            result_text = "Absolute: Passed (Multiple)"
                    else:
                        result_text = "Absolute: Passed (Multiple)"

                else:
                    # 修正: 新しいメソッドを使って退避先を選択
                    selected_assets = self._evaluate_out_of_market_assets(as_of_date)
                    result_text = "Absolute: Failed (Multiple)"

                logger.info(f"{result_text}: {selected_assets} を選択（重み付け絶対モメンタム: {abs_mom_weighted_return:.2%} vs {rfr_weighted_return:.2%}）")

            else:
                selected_assets = [asset for asset, _ in sorted_assets[:self.assets_to_hold]]
                result_text = "Relative Only (Multiple)"

                if not selected_assets:
                    selected_assets = self.out_of_market_assets
                    logger.info(f"選択可能な資産がないため {self.out_of_market_assets} を選択")

        # 絶対モメンタム情報を保存するための変数
        abs_momentum_info = None

        # 絶対モメンタムを使用している場合のみ情報を保存
        if self.single_absolute_momentum == "Yes":
            if self.performance_periods == "Single Period":
                # シングル期間モードの場合
                abs_momentum_info = {
                    "absolute_return": abs_ret,
                    "risk_free_rate": rfr_return
                }
            else:
                # 複数期間モードの場合
                abs_momentum_info = {
                    "absolute_return": abs_mom_weighted_return,
                    "risk_free_rate": rfr_weighted_return
                }

        # 結果オブジェクトに絶対モメンタム情報を含めて保存
        self.momentum_results = {
            "sorted_assets": sorted_assets,
            "selected_assets": selected_assets,
            "message": result_text,
            "abs_momentum_info": abs_momentum_info  # 追加
        }
        self._save_to_cache(self._generate_cache_key(as_of_date), self.momentum_results)
        return self.momentum_results

    # --- ここから差し替え ----------------------------
    #--------------------------------------------------
    #  run_backtest
    #   ルックバック期間を差し引いた「実際に使える期間」を
    #   デフォルト引数として採用できるように改造
    #--------------------------------------------------
    def run_backtest(self,
                     start_date: str | None = None,
                     end_date  : str | None = None):
        # 新しい実行の前に過去の結果をクリア
        self.clear_results()

        valid, errors, warnings_list = self.validate_parameters()
        if not valid:
            logger.error("バックテスト実行前のパラメータ検証に失敗しました:")
            for error in errors:
                logger.error(f"- {error}")
            return None
        if warnings_list:
            logger.warning("検証で警告が発生しました:")
            for warning in warnings_list:
                logger.warning(f"- {warning}")

        # ユーザーが日付を渡していなければ「実行可能期間」をデフォルトにする
        _, last_day = calendar.monthrange(self.end_year, self.end_month)

        if start_date is None:
            if hasattr(self, "bt_start_date"):
                start_date = self.bt_start_date.strftime("%Y-%m-%d")
            else:
                start_date = f"{self.start_year}-{self.start_month:02d}-01"

        if end_date is None:
            if hasattr(self, "bt_end_date"):
                end_date = self.bt_end_date.strftime("%Y-%m-%d")
            else:
                end_date = f"{self.end_year}-{self.end_month:02d}-{last_day}"

        logger.info(f"バックテスト実行: {start_date} から {end_date}")
        return self._run_backtest_next_close(start_date, end_date)
# --- ここまで差し替え ----------------------------


    def _run_backtest_next_close(self, start_date, end_date):
        """正確な日付でのバックテスト実行（修正版）"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # ポジション情報を初期化（追加）
        self.positions = []

        # 前月末日の計算
        if start_date.month == 1:
            prev_month_year = start_date.year - 1
            prev_month = 12
        else:
            prev_month_year = start_date.year
            prev_month = start_date.month - 1

        prev_month_end = self.get_last_trading_day_of_month(prev_month_year, prev_month)
        has_prev_month_data = (prev_month_end is not None and
                            prev_month_end in self.price_data.index)

        # ユーザー指定期間の日次データ
        daily = self.price_data.loc[start_date:end_date].copy()
        if daily.empty:
            logger.error("指定された期間に日次データがありません。")
            return None

        # 初期設定
        initial_investment = 100000
        portfolio = pd.DataFrame(index=daily.index, columns=["Portfolio_Value", "Benchmark_Value"])
        portfolio.iloc[0, :] = initial_investment

        summary = []
        positions = []  # ポジション情報を追跡するリスト

        # 初期ポジション情報を保存（終了日は後で設定）
        has_initial_position = False
        holdings = {}  # 初期ホールディングを空のディクショナリで初期化

        if has_prev_month_data:
            # 前月末のモメンタム計算を実行して初期ポジションを決定
            if self.lookback_unit == "Days":
                past_data = self.price_data.loc[:prev_month_end].copy()
                temp_data = self.price_data
                self.price_data = past_data
                initial_momentum_results = self.calculate_momentum_ranks(prev_month_end)
                self.price_data = temp_data
            else:
                past_monthly = self.monthly_data.loc[:prev_month_end].copy()
                temp_monthly = self.monthly_data
                self.monthly_data = past_monthly
                initial_momentum_results = self.calculate_momentum_ranks(prev_month_end)
                self.monthly_data = temp_monthly

            # 初期選択資産を取得
            initial_selected_assets = initial_momentum_results["selected_assets"]
            # 初期ホールディングを設定
            holdings = self._create_holdings_from_assets(initial_selected_assets)

            if daily.index[0] != start_date:
                first_valid_date = daily.index[0]
                logger.warning(f"指定開始日 {start_date.strftime('%Y-%m-%d')} のデータがありません。最初の有効日 {first_valid_date.strftime('%Y-%m-%d')} を使用します。")
            else:
                first_valid_date = start_date



            logger.info(f"初期ポジション: {holdings}（判定基準日: {prev_month_end.strftime('%Y-%m-%d')}）")
        else:
            # 前月末データがない場合は開始日を基準に判断（従来ロジック）
            logger.warning(f"前月末データがありません。開始日 {start_date.strftime('%Y-%m-%d')} を基準に初期判断を行います。")

            # 開始日でのモメンタム計算を実行
            initial_momentum_results = self.calculate_momentum_ranks(start_date)
            initial_selected_assets = initial_momentum_results["selected_assets"]
            holdings = self._create_holdings_from_assets(initial_selected_assets)
            logger.info(f"開始日判断による初期ポジション: {holdings}")

        # 全ての月末取引日を計算
        rebalance_dates = []
        current_date = pd.Timestamp(start_date.year, start_date.month, 1)

        while current_date <= end_date:
            last_td = self.get_last_trading_day_of_month(current_date.year, current_date.month)
            if last_td is not None and last_td >= start_date and last_td <= end_date:
                rebalance_dates.append(last_td)
            current_date += relativedelta(months=1)

        # リバランス頻度に応じたフィルタリング
        if self.trading_frequency == "Bimonthly (hold: 1,3,5,7,9,11)":
            # 奇数月に保有を開始するためには前月末にリバランス
            rebalance_months = [12, 2, 4, 6, 8, 10]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Bimonthly (hold: 2,4,6,8,10,12)":
            # 偶数月に保有を開始するためには前月末にリバランス
            rebalance_months = [1, 3, 5, 7, 9, 11]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Quarterly (hold: 1,4,7,10)":
            # 1,4,7,10月から保有するためには前月末にリバランス
            rebalance_months = [12, 3, 6, 9]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Quarterly (hold: 2,5,8,11)":
            # 2,5,8,11月から保有するためには前月末にリバランス
            rebalance_months = [1, 4, 7, 10]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]
        elif self.trading_frequency == "Quarterly (hold: 3,6,9,12)":
            # 3,6,9,12月から保有するためには前月末にリバランス
            rebalance_months = [2, 5, 8, 11]  # 保有月の前月
            rebalance_dates = [d for d in rebalance_dates if d.month in rebalance_months]

        # 判断日と実行日のマッピング
        decision_dates = rebalance_dates
        logger.info(f"リバランス日数: {len(decision_dates)}")

        execution_map = {}
        daily_dates = daily.index
        for dec_date in decision_dates:
            if self.trade_execution == "Trade at end of month price":
                execution_map[dec_date] = dec_date
            elif self.trade_execution == "Trade at next open price" or self.trade_execution == "Trade at next close price":
                next_dates = daily_dates[daily_dates > dec_date]
                execution_map[dec_date] = next_dates[0] if not next_dates.empty else dec_date

        # 初期ポジションの終了日を決定
        current_position_end_date = None
        if decision_dates and execution_map:
            first_dec_date = decision_dates[0]
            first_execution_date = execution_map[first_dec_date]
            current_position_end_date = first_execution_date
        else:
            current_position_end_date = daily.index[-1]

        # 絶対モメンタム情報を取得
        initial_abs_return = None
        initial_rfr_return = None
        if "abs_momentum_info" in initial_momentum_results and initial_momentum_results["abs_momentum_info"]:
            initial_abs_info = initial_momentum_results["abs_momentum_info"]
            initial_abs_return = initial_abs_info.get("absolute_return")
            initial_rfr_return = initial_abs_info.get("risk_free_rate")

        # 絶対モメンタムが無効でも、リスクフリーレートを計算
        if initial_rfr_return is None:
            if self.performance_periods == "Single Period":
                lookback = self.lookback_period
                if self.absolute_momentum_custom_period:
                    lookback = self.absolute_momentum_period
                # 信号日に基づくリスクフリーレート計算
                calc_date = prev_month_end if has_prev_month_data else start_date
                initial_rfr_return = self.calculate_cumulative_rfr_return(calc_date, lookback)
            else:
                # 複数期間の場合は重み付きリスクフリーレート
                calc_date = prev_month_end if has_prev_month_data else start_date
                initial_rfr_return = self._calculate_weighted_rfr_return(calc_date)

        # 初期ポジション情報を記録
        positions.append({
            "signal_date": prev_month_end if has_prev_month_data else start_date,
            "start_date": first_valid_date if 'first_valid_date' in locals() else daily.index[0],
            "end_date": current_position_end_date,
            "assets": initial_selected_assets,
            "message": initial_momentum_results.get("message", ""),
            "abs_return": initial_abs_return,  # 追加
            "rfr_return": initial_rfr_return   # 追加
        })

        # 最後のリバランス日を初期化（開始日として使用）
        last_rebalance_date = daily.index[0]
        prev_date = daily.index[0]
        cache_hits = 0
        cache_misses = 0



        # 各日の価値を計算
        for current_date in daily.index[1:]:
            # ベンチマークリターン計算
            if self.benchmark_ticker in daily.columns:
                try:
                    # NaN値チェック
                    if pd.isna(daily[self.benchmark_ticker].loc[current_date]) or pd.isna(daily[self.benchmark_ticker].loc[prev_date]):
                        # 価格データがない場合は変化なし
                        portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]
                        logger.debug(f"日付 {current_date.strftime('%Y-%m-%d')} のベンチマークデータが不完全です")
                    elif daily[self.benchmark_ticker].loc[prev_date] <= 0:
                        # ゼロ以下の価格は異常値
                        portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]
                        logger.warning(f"ベンチマーク {self.benchmark_ticker} の価格が異常です: {daily[self.benchmark_ticker].loc[prev_date]}")
                    else:
                        bench_ret = (daily[self.benchmark_ticker].loc[current_date] / daily[self.benchmark_ticker].loc[prev_date]) - 1
                        portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"] * (1 + bench_ret)
                except Exception as e:
                    # エラー時は変化なし
                    portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]
                    logger.error(f"ベンチマーク計算エラー ({current_date.strftime('%Y-%m-%d')}): {e}")
            else:
                portfolio.loc[current_date, "Benchmark_Value"] = portfolio.loc[prev_date, "Benchmark_Value"]

            # ポートフォリオリターン計算
            if holdings:
                daily_ret = 0
                valid_calculations = 0

                for asset, weight in holdings.items():
                    # 元の資産名を取得（Open_プレフィックスを処理するため）
                    original_asset = asset

                    # リバランス実行日かどうかをチェック
                    is_rebalance_day = False
                    for dec_date, exec_date in execution_map.items():
                        if current_date == exec_date:
                            is_rebalance_day = True
                            break

                    # Trade at next open priceの場合はOpen価格を使用
                    if is_rebalance_day and self.trade_execution == "Trade at next open price":
                        open_asset = f"Open_{original_asset}"

                        # Open価格データが存在するか確認
                        if open_asset in daily.columns:
                            asset_column = open_asset
                        else:
                            # Open価格がない場合は通常のClose価格を使用
                            asset_column = original_asset
                            logger.warning(f"資産 {original_asset} のOpen価格データがないため、Close価格を使用します")
                    else:
                        # 通常はClose価格を使用
                        asset_column = original_asset

                    # 元の資産名でデータチェック
                    if original_asset in daily.columns:
                        try:
                            # 使用する価格カラムがデータフレームに存在するか確認
                            if asset_column not in daily.columns:
                                asset_column = original_asset  # フォールバック

                            # NaN値チェック
                            if pd.isna(daily[asset_column].loc[current_date]) or pd.isna(daily[original_asset].loc[prev_date]):
                                # 欠損データがある場合はリスクフリーレート相当のリターンとする
                                asset_ret = 0.001 / 252  # 日次リスクフリーレート相当
                                logger.debug(f"資産 {original_asset} の日付 {current_date.strftime('%Y-%m-%d')} のデータが不完全です")
                            elif daily[original_asset].loc[prev_date] <= 0:
                                # ゼロ以下の価格は異常値
                                asset_ret = 0
                                logger.warning(f"資産 {original_asset} の価格が異常です: {daily[original_asset].loc[prev_date]}")
                            else:
                                # 今日の価格は選択されたタイプ（OpenまたはClose）
                                # 前日の価格は常にClose
                                asset_ret = (daily[asset_column].loc[current_date] / daily[original_asset].loc[prev_date]) - 1
                                valid_calculations += 1
                                if is_rebalance_day and self.trade_execution == "Trade at next open price":
                                    logger.debug(f"リバランス日 {current_date.strftime('%Y-%m-%d')} の資産 {original_asset} は始値 {daily[asset_column].loc[current_date]} で取引")

                            daily_ret += weight * asset_ret
                        except Exception as e:
                            # エラー時は日次リスクフリーレート相当
                            logger.error(f"資産 {original_asset} のリターン計算エラー ({current_date.strftime('%Y-%m-%d')}): {e}")
                            daily_ret += weight * (0.001 / 252)
                    else:
                        # データがない資産は現金と同等と見なす
                        daily_ret += weight * (0.001 / 252)

                # データ品質ログ
                if valid_calculations == 0 and len(holdings) > 0:
                    logger.warning(f"日付 {current_date.strftime('%Y-%m-%d')} - 全ての保有資産でデータ不完全")
            else:
                daily_ret = 0

            # 最終的にポートフォリオ価値を更新
            portfolio.loc[current_date, "Portfolio_Value"] = portfolio.loc[prev_date, "Portfolio_Value"] * (1 + daily_ret)

            # リバランス処理
            for dec_date, exec_date in execution_map.items():
                if current_date == exec_date:
                    # この部分を修正:
                    if self.lookback_unit == "Days":
                        past_data = self.price_data.loc[:dec_date].copy()
                        temp_data = self.price_data
                        self.price_data = past_data
                        cache_key = self._generate_cache_key(dec_date)
                        momentum_results = self._get_from_cache(cache_key)
                        if momentum_results is not None:
                            cache_hits += 1
                            logger.debug(f"キャッシュヒット: {cache_key}")
                        else:
                            cache_misses += 1
                            # dec_dateを引数として渡す
                            momentum_results = self.calculate_momentum_ranks(dec_date)
                            logger.debug(f"キャッシュミス: {cache_key} - 新規計算実行")
                        self.price_data = temp_data
                    else:
                        past_monthly = self.monthly_data.loc[:dec_date].copy()
                        temp_monthly = self.monthly_data
                        self.monthly_data = past_monthly
                        cache_key = self._generate_cache_key(dec_date)
                        momentum_results = self._get_from_cache(cache_key)
                        if momentum_results is not None:
                            cache_hits += 1
                            logger.debug(f"キャッシュヒット: {cache_key}")
                        else:
                            cache_misses += 1
                            # dec_dateを引数として渡す
                            momentum_results = self.calculate_momentum_ranks(dec_date)
                            logger.debug(f"キャッシュミス: {cache_key} - 新規計算実行")
                        self.monthly_data = temp_monthly

                    selected_assets = momentum_results["selected_assets"]

                    # ポジション変更前のポートフォリオ価値を記録
                    start_val = portfolio.loc[last_rebalance_date, "Portfolio_Value"]
                    end_val = portfolio.loc[current_date, "Portfolio_Value"]
                    ret = (end_val / start_val) - 1

                    # 次のリバランス実行日を見つける（保有終了日として設定）
                    end_date_for_period = daily.index[-1]  # デフォルトは取引最終日

                    # 現在の判断日（dec_date）が何番目かを特定
                    if dec_date in decision_dates:
                        current_idx = decision_dates.index(dec_date)
                        # 次の判断日とその実行日が存在するか確認
                        if current_idx + 1 < len(decision_dates):
                            next_dec_date = decision_dates[current_idx + 1]
                            if next_dec_date in execution_map:
                                end_date_for_period = execution_map[next_dec_date]

                    # 絶対モメンタム情報を取得
                    abs_return = None
                    rfr_return = None
                    if "abs_momentum_info" in momentum_results and momentum_results["abs_momentum_info"]:
                        abs_info = momentum_results["abs_momentum_info"]
                        abs_return = abs_info.get("absolute_return")
                        rfr_return = abs_info.get("risk_free_rate")

                    # 絶対モメンタムが無効でも、リスクフリーレートを計算
                    if rfr_return is None:
                        if self.performance_periods == "Single Period":
                            lookback = self.lookback_period
                            if self.absolute_momentum_custom_period:
                                lookback = self.absolute_momentum_period
                            # dec_date（シグナル判定日）に基づく計算
                            rfr_return = self.calculate_cumulative_rfr_return(dec_date, lookback)
                        else:
                            # 複数期間の場合は重み付きリスクフリーレート
                            rfr_return = self._calculate_weighted_rfr_return(dec_date)

                    # 新しいポジション情報を記録
                    positions.append({
                        "signal_date": dec_date,
                        "start_date": current_date,
                        "end_date": end_date_for_period,
                        "assets": selected_assets,
                        "message": momentum_results.get("message", ""),
                        "abs_return": abs_return,  # 追加
                        "rfr_return": rfr_return   # 追加
                    })

                    # 新しいポジションを設定
                    new_holdings = self._create_holdings_from_assets(selected_assets)
                    holdings = new_holdings
                    logger.info(f"{current_date.strftime('%Y-%m-%d')}: リバランス実行 - {holdings}（{self.trade_execution}）")

                    # 次のリバランスのための基準日を更新
                    last_rebalance_date = current_date

            # 最終日チェックと処理
            if current_date == daily.index[-1]:
                # 最終日時点でのモメンタム計算
                final_momentum_results = self.calculate_momentum_ranks(current_date)
                selected_assets = final_momentum_results.get("selected_assets", [])
                message = final_momentum_results.get("message", "")

                # 絶対モメンタム情報の取得
                abs_info = final_momentum_results.get("abs_momentum_info", {})
                abs_return = abs_info.get("absolute_return")
                rfr_return = abs_info.get("risk_free_rate")

                # 同じポジションが既に記録されていないか確認
                is_duplicate = False
                if positions and positions[-1].get("signal_date") == current_date:
                    is_duplicate = True

                # 重複していない場合のみ最終日のポジション情報を記録
                if not is_duplicate:
                    positions.append({
                        "signal_date": current_date,
                        "start_date": current_date,
                        "end_date": current_date,
                        "assets": selected_assets,
                        "return": 0.0,  # 同日なのでリターンは0
                        "message": message,
                        "abs_return": abs_return,
                        "rfr_return": rfr_return
                    })
                    logger.info(f"{current_date.strftime('%Y-%m-%d')}: 最終日ポジション記録 - {selected_assets}")

            # 次のループのために現在の日付を保存
            prev_date = current_date

        # キャッシュ統計の出力
        logger.info(f"キャッシュ統計: ヒット {cache_hits}回, ミス {cache_misses}回")
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            logger.info(f"キャッシュヒット率: {hit_rate:.2f}%")

        # 1) まず 日次リターンを計算しておく
        portfolio["Portfolio_Return"] = portfolio["Portfolio_Value"].pct_change()
        portfolio["Benchmark_Return"] = portfolio["Benchmark_Value"].pct_change()

        # 2) self.results_daily にコピー
        self.results_daily = portfolio.copy()

        # 月次結果を計算
        monthly_result = portfolio.resample('ME').last()
        self._calculate_portfolio_metrics(monthly_result)

        # 全てのポジションの保有期間リターンを計算
        for i, position in enumerate(positions):
            start_date = position["start_date"]
            end_date = position["end_date"]

            if start_date in portfolio.index and end_date in portfolio.index:
                start_value = portfolio.loc[start_date, "Portfolio_Value"]
                end_value = portfolio.loc[end_date, "Portfolio_Value"]
                ret = (end_value / start_value) - 1
                ret_str = f"{ret:.2%}"

                # ポジションオブジェクトにリターン情報を追加保存
                position["return"] = ret
                position["portfolio_start"] = start_value
                position["portfolio_end"] = end_value
            else:
                ret_str = "N/A"
                position["return"] = None
                position["portfolio_start"] = None
                position["portfolio_end"] = None

            # サマリーテーブル用にデータを整形
            summary.append({
                "シグナル判定日": position["signal_date"].date(),
                "保有開始日": start_date.date(),
                "保有終了日": end_date.date(),
                "保有資産": ', '.join(position["assets"]),
                "保有期間リターン": ret_str,
                "モメンタム判定結果": position["message"],
                "絶対モメンタムリターン": f"{position.get('abs_return')*100:.2f}%" if position.get('abs_return') is not None else "N/A",
                "リスクフリーレート": f"{position.get('rfr_return')*100:.2f}%" if position.get('rfr_return') is not None else "N/A"
            })


        # メモリ解放
        try:
            del daily
        except NameError:
            pass
        gc.collect()

        # ポジション情報をクラス変数として保存
        self.positions = positions

        return monthly_result


    def _calculate_portfolio_metrics(self, portfolio):
        portfolio = portfolio.sort_index().ffill()
        portfolio["Portfolio_Return"] = portfolio["Portfolio_Value"].pct_change().astype(float)
        portfolio["Benchmark_Return"] = portfolio["Benchmark_Value"].pct_change().astype(float)
        portfolio = portfolio.infer_objects(copy=False)
        portfolio["Portfolio_Cumulative"] = (1 + portfolio["Portfolio_Return"]).cumprod()
        portfolio["Benchmark_Cumulative"] = (1 + portfolio["Benchmark_Return"]).cumprod()
        portfolio["Portfolio_Peak"] = portfolio["Portfolio_Value"].cummax()
        portfolio["Portfolio_Drawdown"] = (portfolio["Portfolio_Value"] / portfolio["Portfolio_Peak"]) - 1
        portfolio["Benchmark_Peak"] = portfolio["Benchmark_Value"].cummax()
        portfolio["Benchmark_Drawdown"] = (portfolio["Benchmark_Value"] / portfolio["Benchmark_Peak"]) - 1
        self.results = portfolio

    def calculate_performance_metrics(self):
        if self.results is None:
            logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return None
        years = (self.results.index[-1] - self.results.index[0]).days / 365.25
        if "Portfolio_Cumulative" in self.results.columns:
            cumulative_return_portfolio = self.results["Portfolio_Cumulative"].iloc[-1] - 1
        else:
            cumulative_return_portfolio = self.results["Portfolio_Value"].iloc[-1] / self.results["Portfolio_Value"].iloc[0] - 1
        if "Benchmark_Cumulative" in self.results.columns:
            cumulative_return_benchmark = self.results["Benchmark_Cumulative"].iloc[-1] - 1
        else:
            cumulative_return_benchmark = self.results["Benchmark_Value"].iloc[-1] / self.results["Benchmark_Value"].iloc[0] - 1

        # 初期値を$100,000として計算する
        initial_investment = 100000.0
        portfolio_total_return = self.results["Portfolio_Value"].iloc[-1] / initial_investment - 1
        benchmark_total_return = self.results["Benchmark_Value"].iloc[-1] / initial_investment - 1

        portfolio_cagr = (1 + portfolio_total_return) ** (1 / years) - 1
        benchmark_cagr = (1 + benchmark_total_return) ** (1 / years) - 1
        portfolio_vol = self.results["Portfolio_Return"].std() * np.sqrt(12)
        benchmark_vol = self.results["Benchmark_Return"].std() * np.sqrt(12)
        portfolio_max_dd = self.results["Portfolio_Drawdown"].min()
        benchmark_max_dd = self.results["Benchmark_Drawdown"].min()
        portfolio_sharpe = portfolio_cagr / portfolio_vol if portfolio_vol != 0 else np.nan
        benchmark_sharpe = benchmark_cagr / benchmark_vol if benchmark_vol != 0 else np.nan
        monthly_returns_portfolio = self.results["Portfolio_Return"].dropna()
        downside_returns_portfolio = monthly_returns_portfolio[monthly_returns_portfolio < 0]
        downside_std_portfolio = downside_returns_portfolio.std() * np.sqrt(12) if len(downside_returns_portfolio) > 0 else np.nan
        portfolio_sortino = portfolio_cagr / downside_std_portfolio if (downside_std_portfolio is not None and downside_std_portfolio != 0) else np.nan
        monthly_returns_benchmark = self.results["Benchmark_Return"].dropna()
        downside_returns_benchmark = monthly_returns_benchmark[monthly_returns_benchmark < 0]
        downside_std_benchmark = downside_returns_benchmark.std() * np.sqrt(12) if len(downside_returns_benchmark) > 0 else np.nan
        benchmark_sortino = benchmark_cagr / downside_std_benchmark if (downside_std_benchmark is not None and downside_std_benchmark != 0) else np.nan
        portfolio_mar = portfolio_cagr / abs(portfolio_max_dd) if (portfolio_max_dd is not None and portfolio_max_dd != 0) else np.nan
        benchmark_mar = benchmark_cagr / abs(benchmark_max_dd) if (benchmark_max_dd is not None and benchmark_max_dd != 0) else np.nan
        self.metrics = {
            "Cumulative Return": {"Portfolio": cumulative_return_portfolio, "Benchmark": cumulative_return_benchmark},
            "CAGR": {"Portfolio": portfolio_cagr, "Benchmark": benchmark_cagr},
            "Volatility": {"Portfolio": portfolio_vol, "Benchmark": benchmark_vol},
            "Max Drawdown": {"Portfolio": portfolio_max_dd, "Benchmark": benchmark_max_dd},
            "Sharpe Ratio": {"Portfolio": portfolio_sharpe, "Benchmark": benchmark_sharpe},
            "Sortino Ratio": {"Portfolio": portfolio_sortino, "Benchmark": benchmark_sortino},
            "MAR Ratio": {"Portfolio": portfolio_mar, "Benchmark": benchmark_mar}
        }
        return self.metrics

    def plot_performance(self, display_plot=True):
        """
        パフォーマンスグラフを表示

        Args:
            display_plot (bool): グラフを表示するかどうか。False の場合、グラフは生成されません。
                            デフォルトは True。

        Returns:
            None
        """
        # 結果がない場合は早期リターン
        if self.results is None:
            if display_plot:  # 表示モードの場合のみエラーメッセージを表示
                logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return

        # 表示フラグがFalseなら何もせずに終了
        if not display_plot:
            return

        if not hasattr(self, 'metrics') or self.metrics is None:
            self.calculate_performance_metrics()
        period_str = f"{self.start_year}/{self.start_month:02d} - {self.end_year}/{self.end_month:02d}"
        fig_norm, ax_norm = plt.subplots(figsize=(9, 6))
        ax_norm.plot(self.results.index, self.results["Portfolio_Value"], label="Dual Momentum Portfolio", color='navy')
        ax_norm.plot(self.results.index, self.results["Benchmark_Value"], label=f"Benchmark ({self.benchmark_ticker})", color='darkorange')
        ax_norm.set_title(f"Portfolio Performance (Normal Scale) | Test Period: {period_str}", fontsize=14)
        ax_norm.set_ylabel("Value ($)")
        ax_norm.legend()
        ax_norm.grid(True, linestyle='-', linewidth=1, color='gray')
        ax_norm.xaxis.set_major_locator(mdates.YearLocator())
        ax_norm.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()
        fig_log, ax_log = plt.subplots(figsize=(9, 6))
        ax_log.plot(self.results.index, self.results["Portfolio_Value"], label="Dual Momentum Portfolio", color='navy')
        ax_log.plot(self.results.index, self.results["Benchmark_Value"], label=f"Benchmark ({self.benchmark_ticker})", color='darkorange')
        ax_log.set_yscale("log")
        ax_log.set_title(f"Portfolio Performance (Log Scale) | Test Period: {period_str}", fontsize=14)
        ax_log.set_ylabel("Value ($)")
        ax_log.legend()
        major_locator = mticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        ax_log.yaxis.set_major_locator(major_locator)
        ax_log.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"$10^{{{int(np.log10(y))}}}$" if y > 0 else ""))
        minor_locator = mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10)
        ax_log.yaxis.set_minor_locator(minor_locator)
        ax_log.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax_log.grid(True, which='major', linestyle='-', linewidth=1, color='gray')
        ax_log.grid(True, which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax_log.xaxis.set_major_locator(mdates.YearLocator())
        ax_log.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()
        fig_dd, ax_dd = plt.subplots(figsize=(9, 3))
        ax_dd.plot(self.results.index, self.results["Portfolio_Drawdown"], label="Portfolio Drawdown", color='navy')
        ax_dd.plot(self.results.index, self.results["Benchmark_Drawdown"], label="Benchmark Drawdown", color='darkorange')
        ax_dd.set_title("Drawdown", fontsize=14)
        ax_dd.set_ylabel("Drawdown (%)")
        min_dd = min(self.results["Portfolio_Drawdown"].min(), self.results["Benchmark_Drawdown"].min())
        ax_dd.set_ylim(min_dd * 1.1, 0.05)
        ax_dd.legend()
        ax_dd.grid(True, linestyle='-', linewidth=1, color='gray')
        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.show()

    def generate_annual_returns_table(self, display_table=True):
        """
        実際の保有期間リターンに基づいた年次リターンテーブルを生成

        Args:
            display_table (bool): テーブルをHTML形式で表示するかどうか。
                                False の場合、計算は行われますがHTMLテーブルは表示されません。
                                データ分析のみ必要な場合に便利です。
                                デフォルトは True。

        Returns:
            pd.DataFrame: 年次リターンのデータフレーム。
                        表示フラグに関わらず、データフレームは常に返されます。
        """
        # 結果がない場合の早期リターン
        if self.results is None:
            if display_table:  # 表示モードの場合のみエラーメッセージを表示
                logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return None

        # まず月次リターンテーブルをクリアして強制的に再生成
        if hasattr(self, 'pivot_monthly_returns'):
            delattr(self, 'pivot_monthly_returns')

        # 月次リターンテーブルを生成（表示しない）
        self.generate_monthly_returns_table(display_table=False)

        # 月次リターンから年次リターンを抽出
        portfolio_annual_returns = {}
        if hasattr(self, 'pivot_monthly_returns'):
            for year in self.pivot_monthly_returns.index:
                if pd.notnull(self.pivot_monthly_returns.loc[year, 'Annual']):
                    portfolio_annual_returns[year] = self.pivot_monthly_returns.loc[year, 'Annual']

        # ベンチマークの年次リターンは既存の計算通り
        benchmark_annual_returns = {}
        for year in range(self.start_year, self.end_year + 1):
            year_data = self.results[self.results.index.year == year]
            if not year_data.empty:
                b_first_value = year_data["Benchmark_Value"].iloc[0]
                b_last_value = year_data["Benchmark_Value"].iloc[-1]
                benchmark_annual_returns[year] = (b_last_value / b_first_value) - 1

        # 結果をテーブルにまとめる
        all_years = sorted(set(list(portfolio_annual_returns.keys()) + list(benchmark_annual_returns.keys())))
        annual_data = {
            "Year": all_years,
            "Dual Momentum Portfolio": [f"{portfolio_annual_returns.get(y, 0):.2%}" for y in all_years],
            f"Benchmark ({self.benchmark_ticker})": [f"{benchmark_annual_returns.get(y, 0):.2%}" for y in all_years]
        }

        annual_df = pd.DataFrame(annual_data)

        if display_table:
            display(HTML("""
            <h2 style="color:#3367d6;">Annual Returns</h2>
            """ + annual_df.to_html(index=False, classes='table table-striped')))

        return annual_df

    def generate_monthly_returns_table(self, display_table=True):
        """実際の保有期間リターンに基づいた月次リターンテーブルを生成

        Args:
            display_table: HTMLテーブルを表示するかどうか (デフォルト: True)
        """

        # 月次リターンデータを初期化（追加）
        self.monthly_returns_data = {}
        self.pivot_monthly_returns = None

        if self.results is None:
            if display_table:
                logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return

        # positionsが存在しない場合のチェック
        if not hasattr(self, 'positions') or not self.positions:
            if display_table:
                logger.warning("保有期間データがありません。従来の月次リターン計算を使用します。")

            # 従来のコードを実行（省略）- 元のコードを残す場合はここに記述
            monthly_returns = self.results["Portfolio_Return"].copy()
            # 以下省略...

            return None

        # 保有期間からの月次リターン計算
        monthly_returns = {}

        # 各月の日数を取得するヘルパー関数
        def get_month_days(year, month):
            return calendar.monthrange(year, month)[1]

        # 各ポジションのリターンを日割りで各月に配分
        for position in self.positions:
            if position.get("return") is None:
                continue

            start_date = position["start_date"]
            end_date = position["end_date"]
            position_return = position["return"]

            # 全期間の日数
            total_days = (end_date - start_date).days + 1
            if total_days <= 0:
                logger.warning(f"無効な保有期間: {start_date} - {end_date}")
                continue

            # 開始月と終了月
            start_year, start_month = start_date.year, start_date.month
            end_year, end_month = end_date.year, end_date.month

            # 同じ月内の場合
            if start_year == end_year and start_month == end_month:
                month_key = (start_year, start_month)
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(position_return)
                continue

            # 複数月にまたがる場合
            current_year, current_month = start_year, start_month
            while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
                month_key = (current_year, current_month)
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []

                # 当月の日数計算
                if current_year == start_year and current_month == start_month:
                    # 開始月
                    month_days = get_month_days(current_year, current_month)
                    days_in_month = month_days - start_date.day + 1
                    month_weight = days_in_month / total_days
                    monthly_returns[month_key].append(position_return * month_weight)
                elif current_year == end_year and current_month == end_month:
                    # 終了月
                    days_in_month = end_date.day
                    month_weight = days_in_month / total_days
                    monthly_returns[month_key].append(position_return * month_weight)
                else:
                    # 間の月
                    month_days = get_month_days(current_year, current_month)
                    month_weight = month_days / total_days
                    monthly_returns[month_key].append(position_return * month_weight)

                # 次の月へ
                if current_month == 12:
                    current_year += 1
                    current_month = 1
                else:
                    current_month += 1

        # 月次リターンの集計
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        # データ範囲内の全ての年月を抽出
        all_years = sorted(set(year for year, _ in monthly_returns.keys()))
        all_months = list(range(1, 13))

        # 空のデータフレームを作成
        pivot_table = pd.DataFrame(index=all_years, columns=[month_names[m] for m in all_months] + ['Annual'])

        # 各月のリターンを計算
        for (year, month), returns in monthly_returns.items():
            monthly_return = sum(returns)  # 各ポジションから配分されたリターンの合計
            pivot_table.loc[year, month_names[month]] = monthly_return

        # 年間リターンを計算
        for year in all_years:
            year_returns = [pivot_table.loc[year, month_names[m]] for m in all_months if pd.notnull(pivot_table.loc[year, month_names[m]])]
            if year_returns:
                annual_return = ((1 + pd.Series(year_returns)).prod() - 1)
                pivot_table.loc[year, 'Annual'] = annual_return

        # 表示用にフォーマット
        formatted_table = pivot_table.map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")

        # HTML表示（条件付き）
        if display_table:
            display(HTML("""
        <h2 style="color:#3367d6;">Monthly Returns (Dual Momentum Portfolio)</h2>
        """ + formatted_table.to_html(classes='table table-striped')))

        # クラス変数として保存（他のメソッドで使用可能に）
        self.monthly_returns_data = monthly_returns
        self.pivot_monthly_returns = pivot_table

        return pivot_table

    # display_model_signals_dynamic メソッド内を修正
    # 以下は、Single PeriodとMultiple Periods両方でIRXを表示するための変更です

    def display_model_signals_dynamic(self, dummy=None):
        """
        モデルシグナルを動的に表示する関数。
        バックテスト結果がある場合はその最終ポジションを使用し、
        ない場合は現在の設定に基づいて予測を行います。

        Parameters:
            dummy: ダミーパラメータ（UI連携用）
        """


        # キャッシュ強制クリア前の状態を保存（結果には影響しない一時的なキャッシュクリア）
        original_cache = self.momentum_cache.copy() if hasattr(self, 'momentum_cache') else {}
        original_momentum_results = self.momentum_results

        # キャッシュを強制的にクリア（一時的）
        self.clear_cache()

        # リスクフリーレートのソース情報を取得
        rfr_source = self.get_risk_free_rate_source()
        rfr_source_short = rfr_source.split(' ')[0] if ' ' in rfr_source else rfr_source  # "DTB3"または"IRX"部分のみを取得

        # バックテスト結果が存在するかチェック
        use_backtest_result = False
        backtest_signal_date = None
        backtest_assets = []
        backtest_message = ""
        abs_momentum_asset_return = None
        risk_free_rate = None
        excess_return = None

        if hasattr(self, 'positions') and self.positions:
            # 最終ポジション情報を使用
            last_position = self.positions[-1]
            backtest_signal_date = last_position.get("signal_date")
            backtest_assets = last_position.get("assets", [])
            backtest_message = last_position.get("message", "")
            use_backtest_result = True

            # リスクフリーレート情報を計算（絶対モメンタムの設定に関わらず）
            if backtest_signal_date is not None:
                try:
                    if self.performance_periods == "Single Period":
                        # シングル期間モードの場合
                        lookback = self.lookback_period
                        if self.absolute_momentum_custom_period:
                            lookback = self.absolute_momentum_period

                        # リスクフリーレートは常に計算する
                        risk_free_rate = self.calculate_cumulative_rfr_return(
                            backtest_signal_date,
                            lookback
                        )

                        # 絶対モメンタムが有効な場合のみ、資産のリターンを計算
                        if self.single_absolute_momentum == "Yes" and self.absolute_momentum_asset is not None:
                            abs_momentum_asset_return = self.calculate_monthly_momentum(
                                self.absolute_momentum_asset,
                                backtest_signal_date,
                                lookback
                            )

                            if abs_momentum_asset_return is not None and risk_free_rate is not None:
                                excess_return = abs_momentum_asset_return - risk_free_rate

                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate

                    else:
                        # 複数期間の場合
                        # 各期間の詳細情報を格納する配列
                        period_returns = []
                        period_weights = []
                        period_rfr_returns = []

                        # 各期間のモメンタム計算
                        for period_idx, period in enumerate(self.multiple_periods):
                            length, unit, weight = period.get("length"), period.get("unit"), period.get("weight", 0)

                            if length is None or weight <= 0:
                                continue

                            # 期間の重みを記録
                            period_weights.append(weight / 100.0)  # パーセントから小数に変換

                            # リターン計算 (統一された計算メソッドを使用)
                            if unit == "Months":
                                period_return = self.calculate_monthly_momentum(
                                    self.absolute_momentum_asset,
                                    backtest_signal_date,
                                    length
                                )
                            else:  # Days
                                # 日数を月数に近似
                                approx_months = max(1, round(length / 30))
                                period_return = self.calculate_monthly_momentum(
                                    self.absolute_momentum_asset,
                                    backtest_signal_date,
                                    approx_months
                                )

                            # リスクフリーレート計算
                            if unit == "Months":
                                period_rfr = self.calculate_cumulative_rfr_return(
                                    backtest_signal_date,
                                    length
                                )
                            else:  # Days
                                # 日数を月数に近似
                                approx_months = max(1, round(length / 30))
                                period_rfr = self.calculate_cumulative_rfr_return(
                                    backtest_signal_date,
                                    approx_months
                                )

                            # 結果を記録
                            if period_return is not None and period_rfr is not None:
                                period_returns.append(period_return)
                                period_rfr_returns.append(period_rfr)

                        # 重み付き平均を計算
                        if period_returns and period_weights and period_rfr_returns:
                            # 重みの正規化
                            total_weight = sum(period_weights)
                            if total_weight > 0:
                                normalized_weights = [w/total_weight for w in period_weights]

                                # 重み付きリターンとRFRを計算
                                abs_momentum_asset_return = sum(r * w for r, w in zip(period_returns, normalized_weights))
                                risk_free_rate = sum(rfr * w for rfr, w in zip(period_rfr_returns, normalized_weights))
                                excess_return = abs_momentum_asset_return - risk_free_rate

                except Exception as e:
                    logger.error(f"絶対モメンタム情報の計算中にエラー: {e}")
                    abs_momentum_asset_return = None
                    risk_free_rate = None
                    excess_return = None

        # 置換後のコード：バックテスト結果またはデータの最新日を使用
        if not use_backtest_result:
            # 常に最新の利用可能なデータ日を使用
            if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty:
                signal_date = self.price_data.index[-1]
            else:
                # データがない場合はフォールバック
                import calendar
                end_year_val = self.end_year
                end_month_val = self.end_month
                _, last_day = calendar.monthrange(end_year_val, end_month_val)
                signal_date = pd.to_datetime(f"{end_year_val}-{end_month_val}-{last_day}")
        else:
            signal_date = backtest_signal_date

        # MM/DD/YYYY形式の日付に変換
        signal_date_str = signal_date.strftime("%m/%d/%Y")

        # モメンタム計算 (バックテスト結果がない場合のみ)
        if not use_backtest_result:
            # 計算日と前月末日を取得（前月末データが必要な場合）
            if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty:
                calculation_date = self.price_data.index[-1]

                # 置換後のコード：常に最新日のデータを使用してシグナル計算
                logger.info(f"最新日 {calculation_date.strftime('%Y-%m-%d')} に基づくシグナル計算")
                momentum_results = self.calculate_momentum_ranks(calculation_date)

            else:
                # データがない場合は通常計算
                momentum_results = self.calculate_momentum_ranks()

            selected_assets = momentum_results.get("selected_assets", [])
            message = momentum_results.get("message", "")

            # リスクフリーレート情報の計算（予測用・絶対モメンタムの設定に関わらず）
            try:
                calculation_date = self.price_data.index[-1] if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty else pd.to_datetime("today")

                if self.performance_periods == "Single Period":
                    # 単一期間の場合
                    lookback = self.lookback_period
                    if self.absolute_momentum_custom_period:
                        lookback = self.absolute_momentum_period

                    # リスクフリーレートは常に計算
                    risk_free_rate = self.calculate_cumulative_rfr_return(
                        calculation_date,
                        lookback
                    )

                    # 絶対モメンタムが有効な場合のみ、資産のリターンを計算
                    if self.single_absolute_momentum == "Yes":
                        abs_momentum_asset_return = self.calculate_monthly_momentum(
                            self.absolute_momentum_asset,
                            calculation_date,
                            lookback
                        )

                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate
                else:
                    # 複数期間の場合も同様（リスクフリーレートは常に計算）
                    risk_free_rate = self._calculate_weighted_rfr_return(calculation_date)

                    # 絶対モメンタムが有効な場合のみ
                    if self.single_absolute_momentum == "Yes":
                        abs_momentum_asset_return = self._calculate_weighted_absolute_momentum_unified(calculation_date)

                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate

                    if self.performance_periods == "Single Period":
                        # 単一期間の場合
                        lookback = self.lookback_period
                        if self.absolute_momentum_custom_period:
                            lookback = self.absolute_momentum_period

                        abs_momentum_asset_return = self.calculate_monthly_momentum(
                            self.absolute_momentum_asset,
                            calculation_date,
                            lookback
                        )
                        risk_free_rate = self.calculate_cumulative_rfr_return(
                            calculation_date,
                            lookback
                        )
                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate
                    else:
                        # 複数期間の場合（上記のバックテストと同様の処理）
                        abs_momentum_asset_return = self._calculate_weighted_absolute_momentum_unified(calculation_date)
                        risk_free_rate = self._calculate_weighted_rfr_return(calculation_date)
                        if abs_momentum_asset_return is not None and risk_free_rate is not None:
                            excess_return = abs_momentum_asset_return - risk_free_rate
            except Exception as e:
                    logger.error(f"予測モードでの絶対モメンタム情報の計算中にエラー: {e}")
                    abs_momentum_asset_return = None
                    risk_free_rate = None
                    excess_return = None
        else:
            # バックテスト結果を使用
            selected_assets = backtest_assets
            message = backtest_message

        # 判定結果を英語に変換
        english_result = message

        # アセット文字列の生成
        if len(selected_assets) > 0:
            # 退避先資産かどうかを判断（メッセージに "Out of Market" が含まれているか）
            is_out_of_market = any(s in message for s in ["Out of Market", "Failed"])

            if is_out_of_market and self.out_of_market_strategy == "Top 1" and len(selected_assets) == 1:
                # Top 1戦略の場合は100%表示
                assets_str_list = [f"100.00% {selected_assets[0]}"]
            else:
                # 通常の等分割表示
                alloc_pct = 1.0 / len(selected_assets)
                assets_str_list = [f"{alloc_pct*100:.2f}% {asset}" for asset in selected_assets]

            final_assets_str = ", ".join(assets_str_list)
        else:
            final_assets_str = "None"

        # 詳細テーブルの生成
        relevant_assets = set(self.tickers + [self.absolute_momentum_asset] + self.out_of_market_assets)
        relevant_assets = [a for a in relevant_assets if a and a.lower() != "cash"]

        rows = []

        if self.performance_periods == "Single Period":
            lookback_period = self.lookback_period
            lookback_unit = self.lookback_unit

            # リターン計算の対象日（バックテスト結果があればその日付、なければ最新日）
            calculation_date = signal_date if use_backtest_result else (
                self.price_data.index[-1] if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty
                else pd.to_datetime("today")
            )

            # 各資産のリターン計算
            returns_map = {}
            for asset in relevant_assets:
                ret = self.calculate_monthly_momentum(asset, calculation_date, lookback_period)
                returns_map[asset] = ret

            # リスクフリーレートを事前計算（新規追加）
            rfr_return = self.calculate_cumulative_rfr_return(calculation_date, lookback_period)

            # テーブル行の作成
            for asset in relevant_assets:
                r = returns_map.get(asset)
                formatted_return = f"{r*100:.2f}%" if r is not None else "N/A"

                row = {
                    "Asset": asset,
                    f"{lookback_period}-{lookback_unit.lower()} return": formatted_return,
                    "Score": formatted_return
                }
                rows.append(row)

            # RFRをテーブルに直接追加（新規追加）
            formatted_rfr = f"{rfr_return*100:.2f}%" if rfr_return is not None else "N/A"
            rows.append({
                "Asset": f"RFR ({rfr_source_short})",
                f"{lookback_period}-{lookback_unit.lower()} return": formatted_rfr,
                "Score": formatted_rfr
            })

            columns = ["Asset", f"{lookback_period}-{lookback_unit.lower()} return", "Score"]

        else:  # Multiple Periods
            # リターン計算の対象日
            calculation_date = signal_date if use_backtest_result else (
                self.price_data.index[-1] if hasattr(self, 'price_data') and self.price_data is not None and not self.price_data.empty
                else pd.to_datetime("today")
            )

            # 期間ごとのカラム名と期間情報を準備
            period_details = []
            for idx, p in enumerate(self.multiple_periods):
                length = p.get("length", None)
                unit = p.get("unit", None)
                weight = p.get("weight", 0)

                if length is None or length <= 0 or weight <= 0:
                    continue

                # ここで各期間の開始・終了日付を特定（表示用）
                if unit == "Months":
                    # 月数に基づき計算
                    target_month = calculation_date.month
                    target_year = calculation_date.year

                    # 指定月数分遡る
                    for _ in range(length):
                        if target_month == 1:
                            target_month = 12
                            target_year -= 1
                        else:
                            target_month -= 1

                    # おおよその日付範囲（表示用）
                    start_date_approx = pd.Timestamp(year=target_year, month=target_month, day=1)
                    date_range_str = f"{start_date_approx.strftime('%Y/%m')}～{calculation_date.strftime('%Y/%m')}"
                else:  # Days
                    # 日数に基づき計算
                    start_date_approx = calculation_date - timedelta(days=length)
                    date_range_str = f"{start_date_approx.strftime('%Y/%m/%d')}～{calculation_date.strftime('%Y/%m/%d')}"

                colname = f"{length}-{unit.lower()} return\n({date_range_str})"

                period_details.append({
                    "idx": idx,
                    "length": length,
                    "unit": unit,
                    "weight": weight,
                    "colname": colname
                })

            # 各期間・各資産のリターンを計算
            period_returns = {}
            for period in period_details:
                idx = period["idx"]
                length = period["length"]

                # 単位を揃える（新メソッドは月単位のみ対応）
                if period["unit"] == "Days":
                    # 日数を月数に近似変換（30日≒1ヶ月）
                    months_approx = max(1, round(length / 30))
                    logger.info(f"{length}日間を約{months_approx}ヶ月として計算")

                    period_returns[idx] = {}
                    for asset in relevant_assets:
                        ret = self.calculate_monthly_momentum(asset, calculation_date, months_approx)
                        period_returns[idx][asset] = ret

                else:
                    period_returns[idx] = {}
                    for asset in relevant_assets:
                        ret = self.calculate_monthly_momentum(asset, calculation_date, length)
                        period_returns[idx][asset] = ret

            # 重み付け結果の計算
            if self.weighting_method == "Weight Performance":
                weighted_result = self._calculate_weighted_performance(period_returns, relevant_assets)
            else:  # "Weight Rank Orders"
                weighted_result = self._calculate_weighted_ranks(period_returns, relevant_assets)

            # 表示用のカラムを準備
            period_columns = [p["colname"] for p in period_details]

            # 各資産の結果をテーブルに追加
            for asset in relevant_assets:
                row_data = {"Asset": asset}

                # 各期間のリターンを追加
                for period in period_details:
                    idx = period["idx"]
                    colname = period["colname"]

                    if idx in period_returns and asset in period_returns[idx]:
                        val = period_returns[idx][asset]
                        row_data[colname] = f"{val*100:.2f}%" if val is not None else "N/A"
                    else:
                        row_data[colname] = "N/A"

                # 重み付け結果
                w_val = weighted_result.get(asset)
                row_data["Weighted"] = f"{w_val*100:.2f}%" if w_val is not None else "N/A"
                row_data["Score"] = f"{w_val*100:.2f}%" if w_val is not None else "N/A"

                rows.append(row_data)

            # IRXを計算して追加（複数期間用に追加）
            # 各期間のリスクフリーレートを計算
            rfr_row = {"Asset": f"RFR ({rfr_source_short})"}

            # 各期間のRFRを計算
            for period in period_details:
                idx = period["idx"]
                colname = period["colname"]
                length = period["length"]
                unit = period["unit"]

                # 期間に応じたRFRを計算
                if unit == "Months":
                    period_rfr = self.calculate_cumulative_rfr_return(calculation_date, length)
                else:  # Days
                    # 日数を月数に近似変換
                    months_approx = max(1, round(length / 30))
                    period_rfr = self.calculate_cumulative_rfr_return(calculation_date, months_approx)

                # 表示用フォーマット
                rfr_row[colname] = f"{period_rfr*100:.2f}%" if period_rfr is not None else "N/A"

            # 重み付きRFR
            rfr_weighted = self._calculate_weighted_rfr_return(calculation_date)
            rfr_row["Weighted"] = f"{rfr_weighted*100:.2f}%" if rfr_weighted is not None else "N/A"
            rfr_row["Score"] = f"{rfr_weighted*100:.2f}%" if rfr_weighted is not None else "N/A"

            # リスクフリーレート行を追加
            rows.append(rfr_row)

            columns = ["Asset"] + period_columns + ["Weighted", "Score"]

        # 詳細テーブルの作成
        df_details = pd.DataFrame(rows)
        if columns:
            df_details = df_details[columns]

        # HTMLの生成
        html = f"""
        <h2 style="color:#3367d6;">Model Signals</h2>
        <table style="border-collapse: collapse; width:600px;">
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Signal Date</b></td>
            <td style="padding:4px; border:1px solid #ccc;">{signal_date_str}</td>
        </tr>
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Assets</b></td>
            <td style="padding:4px; border:1px solid #ccc;">{final_assets_str}</td>
        </tr>
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Details</b></td>
            <td style="padding:4px; border:1px solid #ccc;">
            {df_details.to_html(index=False, classes='table table-striped')}
            </td>
        </tr>
        """

        # リスクフリーレートは常に表示、絶対モメンタム情報は条件付きで表示
        if risk_free_rate is not None:
            html += f"""
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>{"Absolute Momentum" if self.single_absolute_momentum == "Yes" else "Risk-Free Rate"}</b></td>
            <td style="padding:4px; border:1px solid #ccc;">
            <table style="width:100%; border-collapse: collapse;">
            """

            # 絶対モメンタムが有効で資産リターンがある場合
            if self.single_absolute_momentum == "Yes" and abs_momentum_asset_return is not None:
                html += f"""
                <tr>
                <td style="padding: 4px; width: 150px;">Absolute({self.absolute_momentum_asset}):</td>
                <td style="padding: 4px;">{abs_momentum_asset_return:.2%}</td>
                </tr>
                """

            # リスクフリーレートは常に表示
            html += f"""
                <tr>
                <td style="padding: 4px;">Risk-Free Rate ({rfr_source_short}):</td>
                <td style="padding: 4px;">{risk_free_rate:.2%}</td>
                </tr>
            """

            # 超過リターンも条件付きで表示
            if self.single_absolute_momentum == "Yes" and abs_momentum_asset_return is not None and excess_return is not None:
                html += f"""
                <tr>
                <td style="padding: 4px;">Excess Return:</td>
                <td style="padding: 4px;">{excess_return:.2%}</td>
                </tr>
                """

            html += """
            </table>
            </td>
        </tr>
    """

        # 判定結果
        html += f"""
        <tr>
            <td style="padding:4px; border:1px solid #ccc;"><b>Decision Result</b></td>
            <td style="padding:4px; border:1px solid #ccc;">{english_result}</td>
        </tr>
        </table>
        """

        # 元のキャッシュと結果を復元
        if hasattr(self, 'momentum_cache'):
            self.momentum_cache = original_cache
        self.momentum_results = original_momentum_results

        display(HTML(html))

    def display_performance_summary(self, display_summary=True):
        """
        バックテストのパフォーマンスサマリーを表示するメソッド。
        display_summary=False の場合は出力を抑制し、内部計算のみ行うなどの拡張も可能。
        """
        if self.results is None:
            if display_summary:
                print("バックテスト結果がありません。run_backtest()を実行してください。")
            return

        # 表示フラグがFalseなら計算と表示をスキップ
        if not display_summary:
            # メトリクスが既に計算されていれば返す、なければ計算して返す
            if hasattr(self, 'metrics') and self.metrics is not None:
                return self.metrics
            else:
                return self.calculate_performance_metrics()

        # 既存の年次リターンテーブルを強制的にクリアして再生成
        if hasattr(self, 'pivot_monthly_returns'):
            delattr(self, 'pivot_monthly_returns')

        # 修正：先に月次リターンテーブルを生成（未生成の場合）、表示しない
        self.generate_monthly_returns_table(display_table=False)

        # バックテストの実際の開始日を使用
        if hasattr(self, 'positions') and self.positions:
            # 最初のポジションの開始日を取得
            start_date = self.positions[0]['start_date']
        else:
            # フォールバックとして結果の最初のインデックスを使用
            start_date = self.results.index[0]

        if self.price_data is not None and not self.price_data.empty:
            end_date = self.price_data.index[-1]
        else:
            end_date = self.results.index[-1]

        metrics = self.calculate_performance_metrics()

        # 修正：先に月次リターンテーブルを生成（未生成の場合）、表示しない
        if not hasattr(self, 'pivot_monthly_returns'):
            self.generate_monthly_returns_table(display_table=False)

        # 修正：保有期間ベースの年次リターンを使用
        annual_returns = {}
        if hasattr(self, 'pivot_monthly_returns'):
            for year in self.pivot_monthly_returns.index:
                if pd.notnull(self.pivot_monthly_returns.loc[year, 'Annual']):
                    annual_returns[year] = self.pivot_monthly_returns.loc[year, 'Annual']

        # ベンチマークは従来通り
        benchmark_annual_returns = {}
        for year in range(self.start_year, self.end_year + 1):
            year_data = self.results[self.results.index.year == year]
            if not year_data.empty:
                b_first_value = year_data["Benchmark_Value"].iloc[0]
                b_last_value = year_data["Benchmark_Value"].iloc[-1]
                benchmark_annual_returns[year] = (b_last_value / b_first_value) - 1

        best_year = max(annual_returns.items(), key=lambda x: x[1]) if annual_returns else ("N/A", np.nan)
        worst_year = min(annual_returns.items(), key=lambda x: x[1]) if annual_returns else ("N/A", np.nan)
        best_year_benchmark = max(benchmark_annual_returns.items(), key=lambda x: x[1]) if benchmark_annual_returns else ("N/A", np.nan)
        worst_year_benchmark = min(benchmark_annual_returns.items(), key=lambda x: x[1]) if benchmark_annual_returns else ("N/A", np.nan)
        if "Portfolio_Return" in self.results.columns and "Benchmark_Return" in self.results.columns:
            benchmark_corr = self.results["Portfolio_Return"].corr(self.results["Benchmark_Return"])
        else:
            benchmark_corr = np.nan
        summary_data = {
        "Metric": ["Start Balance", "End Balance", "Annualized Return (CAGR)", "Standard Deviation",
                "Best Year", "Worst Year", "Maximum Drawdown", "Sharpe Ratio", "Sortino Ratio", "MAR Ratio",
                "Benchmark Correlation", "退避先資産戦略"],  # 追加
        "Dual Momentum Model": [
            "$100,000.00",
            f"${self.results['Portfolio_Value'].iloc[-1]:,.2f}",
            f"{metrics['CAGR']['Portfolio']*100:.2f}%",
            f"{metrics['Volatility']['Portfolio']*100:.2f}%",
            f"{best_year[0]}: {best_year[1]*100:.2f}%" if best_year[0] != "N/A" else "N/A",
            f"{worst_year[0]}: {worst_year[1]*100:.2f}%" if worst_year[0] != "N/A" else "N/A",
            f"{metrics['Max Drawdown']['Portfolio']*100:.2f}%",
            f"{metrics['Sharpe Ratio']['Portfolio']:.2f}",
            f"{metrics['Sortino Ratio']['Portfolio']:.2f}",
            f"{metrics['MAR Ratio']['Portfolio']:.2f}",
            f"{benchmark_corr:.2f}",
            f"{self.out_of_market_strategy}"  # 追加
        ],
            "Benchmark (" + self.benchmark_ticker + ")": [
                "$100,000.00",
                f"${self.results['Benchmark_Value'].iloc[-1]:,.2f}",
                f"{metrics['CAGR']['Benchmark']*100:.2f}%",
                f"{metrics['Volatility']['Benchmark']*100:.2f}%",
                f"{best_year_benchmark[0]}: {best_year_benchmark[1]*100:.2f}%" if best_year_benchmark[0] != "N/A" else "N/A",
                f"{worst_year_benchmark[0]}: {worst_year_benchmark[1]*100:.2f}%" if worst_year_benchmark[0] != "N/A" else "N/A",
                f"{metrics['Max Drawdown']['Benchmark']*100:.2f}%",
                f"{metrics['Sharpe Ratio']['Benchmark']:.2f}",
                f"{metrics['Sortino Ratio']['Benchmark']:.2f}",
                f"{metrics['MAR Ratio']['Benchmark']:.2f}",
                "1.00",
                "N/A"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        html = f"""
        <h2 style="color:#3367d6;">Performance Summary</h2>
        <p><strong>({start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')})</strong></p>
        """ + summary_df.to_html(index=False, classes='table table-striped')
        display(HTML(html))

        return metrics

    def display_model_signals_dynamic_ui(self):
        """
        UI用のシグナル表示関数（一時的にキャッシュをクリアしてシグナルを計算）
        """
        # 一時的にキャッシュをクリアしてシグナルを表示（元のキャッシュは自動的に復元される）
        self.display_model_signals_dynamic()

    def export_to_excel(self, filename=None, auto_download=False):
        """
        バックテスト結果をエクセルファイルに出力する

        Parameters:
        filename (str, optional): 出力ファイル名。指定がない場合は自動生成
        auto_download (bool): Colabの場合に自動ダウンロードするかどうか

        Returns:
        dict または None: 成功した場合は情報辞書、失敗の場合はNone
        """
        import pandas as pd
        from datetime import datetime
        import json
        import os

        # ファイル名が指定されていない場合は自動生成
        if filename is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"DM_{now}.xlsx"

        # 拡張子の確認と追加
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'

        # バックテスト結果が存在するか確認
        if self.results is None:
            logger.error("バックテスト結果がありません。run_backtest()を実行してください。")
            return None

        try:
            # 日付範囲の取得
            start_date = self.results.index[0]
            end_date = self.results.index[-1]

            # Excel Writerの作成
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                #------------------------------------------------------
                # 1. 設定シート (Settings)
                #------------------------------------------------------
                settings_data = []

                # バックテスト期間情報
                settings_data.append(["バックテスト期間情報", ""])
                settings_data.append(["設定開始日", f"{self.start_year}/{self.start_month:02d}/01"])

                # 終了日は月の最終日を取得
                import calendar
                _, last_day = calendar.monthrange(self.end_year, self.end_month)
                settings_data.append(["設定終了日", f"{self.end_year}/{self.end_month:02d}/{last_day}"])

                # 実際のバックテスト期間
                settings_data.append(["実際のバックテスト開始日", start_date.strftime('%Y/%m/%d')])
                settings_data.append(["実際のバックテスト終了日", end_date.strftime('%Y/%m/%d')])

                # 期間計算
                days_diff = (end_date - start_date).days
                years = days_diff // 365
                months = (days_diff % 365) // 30
                settings_data.append(["有効バックテスト期間", f"{years}年{months}ヶ月"])

                # ルックバック期間
                if self.performance_periods == "Single Period":
                    lb_info = f"{self.lookback_period} {self.lookback_unit}"
                else:
                    # 最長のルックバック期間を表示
                    max_lb = 0
                    max_unit = ""
                    for period in self.multiple_periods:
                        if period.get("length") and period.get("weight", 0) > 0:
                            if period.get("length") > max_lb:
                                max_lb = period.get("length")
                                max_unit = period.get("unit")
                    lb_info = f"{max_lb} {max_unit}"
                settings_data.append(["ルックバック期間", lb_info])

                # 資産設定
                settings_data.append(["", ""])
                settings_data.append(["資産設定", ""])
                settings_data.append(["投資対象銘柄", ", ".join(self.tickers)])
                settings_data.append(["絶対モメンタム", self.single_absolute_momentum])
                settings_data.append(["絶対モメンタム資産", self.absolute_momentum_asset])
                settings_data.append(["市場退避先資産", ", ".join(self.out_of_market_assets)])

                # モメンタム設定
                settings_data.append(["", ""])
                settings_data.append(["モメンタム設定", ""])
                settings_data.append(["パフォーマンス期間", self.performance_periods])

                if self.performance_periods == "Single Period":
                    settings_data.append(["ルックバック期間", f"{self.lookback_period} {self.lookback_unit}"])
                    if self.absolute_momentum_custom_period:
                        settings_data.append(["絶対モメンタム期間", f"{self.absolute_momentum_period} {self.lookback_unit}"])
                else:
                    # 複数期間の設定
                    for idx, period in enumerate(self.multiple_periods, start=1):
                        if period.get("length") and period.get("weight", 0) > 0:
                            settings_data.append([f"期間{idx}",
                                                f"{period.get('length')} {period.get('unit')} ({period.get('weight')}%)"])
                    settings_data.append(["重み付け方法", self.weighting_method])

                settings_data.append(["保有資産数", self.assets_to_hold])

                # 取引設定
                settings_data.append(["", ""])
                settings_data.append(["取引設定", ""])
                settings_data.append(["取引頻度", self.trading_frequency])
                settings_data.append(["取引実行", self.trade_execution])
                settings_data.append(["ベンチマーク", self.benchmark_ticker])

                # データフレームに変換して出力
                settings_df = pd.DataFrame(settings_data, columns=["パラメータ", "値"])

                # 1. Settingsシート
                if self.excel_sheets_to_export.get("settings", True):
                    settings_df.to_excel(writer, sheet_name="Settings", index=False)

                #------------------------------------------------------
                # 2. パフォーマンスシート (Performance)
                #------------------------------------------------------
                # メトリクスが計算されていなければ計算
                if not hasattr(self, 'metrics') or self.metrics is None:
                    self.calculate_performance_metrics()

                perf_data = []

                # バックテスト期間情報
                perf_data.append(["バックテスト期間", f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}", ""])
                perf_data.append(["", "", ""])

                # パフォーマンス指標
                perf_data.append(["指標", "Dual Momentum Portfolio", f"Benchmark ({self.benchmark_ticker})"])

                # 基本指標
                initial_investment = 100000.0
                perf_data.append(["開始金額", f"${initial_investment:,.2f}", f"${initial_investment:,.2f}"])
                perf_data.append(["終了金額",
                                f"${self.results['Portfolio_Value'].iloc[-1]:,.2f}",
                                f"${self.results['Benchmark_Value'].iloc[-1]:,.2f}"])

                # その他の指標
                for metric_name, metric_values in self.metrics.items():
                    # パーセント表示が必要な指標
                    if metric_name in ["Cumulative Return", "CAGR", "Volatility", "Max Drawdown"]:
                        perf_data.append([metric_name,
                                        f"{metric_values['Portfolio']*100:.2f}%",
                                        f"{metric_values['Benchmark']*100:.2f}%"])
                    else:
                        perf_data.append([metric_name,
                                        f"{metric_values['Portfolio']:.2f}",
                                        f"{metric_values['Benchmark']:.2f}"])

                # 年次リターン情報
                if hasattr(self, 'pivot_monthly_returns'):
                    best_year_portfolio = None
                    worst_year_portfolio = None
                    best_return = -float('inf')
                    worst_return = float('inf')

                    for year in self.pivot_monthly_returns.index:
                        if 'Annual' in self.pivot_monthly_returns.columns and pd.notnull(self.pivot_monthly_returns.loc[year, 'Annual']):
                            annual_return = self.pivot_monthly_returns.loc[year, 'Annual']
                            if annual_return > best_return:
                                best_return = annual_return
                                best_year_portfolio = year
                            if annual_return < worst_return:
                                worst_return = annual_return
                                worst_year_portfolio = year

                    if best_year_portfolio is not None:
                        perf_data.append(["最良年 (Portfolio)",
                                        f"{best_year_portfolio}: {best_return*100:.2f}%",
                                        ""])

                    if worst_year_portfolio is not None:
                        perf_data.append(["最悪年 (Portfolio)",
                                        f"{worst_year_portfolio}: {worst_return*100:.2f}%",
                                        ""])

                # ベンチマークの年次リターン
                benchmark_annual_returns = {}
                for year in range(start_date.year, end_date.year + 1):
                    year_data = self.results[self.results.index.year == year]
                    if not year_data.empty:
                        b_first_value = year_data["Benchmark_Value"].iloc[0]
                        b_last_value = year_data["Benchmark_Value"].iloc[-1]
                        benchmark_annual_returns[year] = (b_last_value / b_first_value) - 1

                if benchmark_annual_returns:
                    best_year_bench = max(benchmark_annual_returns.items(), key=lambda x: x[1])
                    worst_year_bench = min(benchmark_annual_returns.items(), key=lambda x: x[1])

                    perf_data.append(["最良年 (Benchmark)",
                                    "",
                                    f"{best_year_bench[0]}: {best_year_bench[1]*100:.2f}%"])

                    perf_data.append(["最悪年 (Benchmark)",
                                    "",
                                    f"{worst_year_bench[0]}: {worst_year_bench[1]*100:.2f}%"])

                # 相関係数
                if "Portfolio_Return" in self.results.columns and "Benchmark_Return" in self.results.columns:
                    benchmark_corr = self.results["Portfolio_Return"].corr(self.results["Benchmark_Return"])
                    perf_data.append(["ベンチマーク相関", f"{benchmark_corr:.2f}", "1.00"])

                # データフレームに変換して出力
                perf_df = pd.DataFrame(perf_data)

                # 2. Performanceシート
                if self.excel_sheets_to_export.get("performance", True):
                    perf_df.to_excel(writer, sheet_name="Performance", index=False, header=False)

                #------------------------------------------------------
                # 3. 簡易日次データシート (Daily Returns Simple)
                #------------------------------------------------------
                # 必要なのは日付とリターンのみ
                # リターンは小数表示 (+3.57% -> 1.0357, -15.7% -> 0.843)

                # 3. 簡易日次データシート (Daily Returns Simple)
                daily_simple_data = []

                daily_simple_data.append(["日付", "ポートフォリオリターン", "ベンチマークリターン"])

                # ★ここで self.results_daily に切り替える
                if not hasattr(self, 'results_daily') or self.results_daily is None:
                    # results_daily がまだ存在しない場合の警告
                    logger.warning("results_daily が存在しないため、Daily Returns Simpleを出力できません。")
                else:
                    for idx, date in enumerate(self.results_daily.index):
                        if idx == 0:
                            continue  # 最初の行はリターンなし

                        if ("Portfolio_Return" in self.results_daily.columns and
                            "Benchmark_Return" in self.results_daily.columns):
                            port_ret = self.results_daily["Portfolio_Return"].iloc[idx]
                            bench_ret = self.results_daily["Benchmark_Return"].iloc[idx]

                            # 欠損値処理など挿入する場合はここ

                            daily_simple_data.append([
                                date.strftime('%Y/%m/%d'),
                                port_ret,
                                bench_ret
                            ])

                # データフレームに変換して出力
                daily_simple_df = pd.DataFrame(daily_simple_data[1:], columns=daily_simple_data[0])
                # 3. Daily Returns Simple
                if self.excel_sheets_to_export.get("daily_simple", True):
                    daily_simple_df.to_excel(writer, sheet_name="Daily Returns Simple", index=False)

                #------------------------------------------------------
                # 4. JSON設定シート (JSON Config)
                #------------------------------------------------------
                json_data = []

                # モデル設定をJSONに変換
                config = {
                    "time": {
                        "start_year": self.start_year,
                        "start_month": self.start_month,
                        "end_year": self.end_year,
                        "end_month": self.end_month
                    },
                    "assets": {
                        "tickers": list(tickers.value),  # タプルをリストに変換
                        "specify_tickers": specify_tickers.value,  # 新しいフィールドを追加
                        "single_absolute_momentum": single_absolute_momentum.value,
                        "absolute_momentum_asset": absolute_momentum_asset.value,
                        "specify_absolute_momentum_asset": specify_absolute_momentum_asset.value,
                        "out_of_market_assets": list(out_of_market_assets.value),
                        "specify_out_of_market_asset": specify_out_of_market_asset.value,
                        "out_of_market_strategy": out_of_market_strategy.value
                    },
                    "performance": {
                        "performance_periods": self.performance_periods,
                        "lookback_period": self.lookback_period,
                        "lookback_unit": self.lookback_unit,
                        "multiple_periods": [
                            {
                                "length": p.get("length"),
                                "unit": p.get("unit"),
                                "weight": p.get("weight")
                            } for p in self.multiple_periods if p.get("length") is not None
                        ],
                        "weighting_method": self.weighting_method,
                        "assets_to_hold": self.assets_to_hold
                    },
                    "trade": {
                        "trading_frequency": self.trading_frequency,
                        "trade_execution": self.trade_execution,
                        "benchmark_ticker": self.benchmark_ticker
                    },
                    "absolute_momentum": {
                        "custom_period": self.absolute_momentum_custom_period,
                        "period": self.absolute_momentum_period
                    }
                }

                # 生のJSON文字列
                json_str = json.dumps(config, indent=2, ensure_ascii=False)
                json_data.append(["生のJSON設定:"])
                json_data.append([json_str])
                json_data.append([""])
                json_data.append([""])

                # フラット化したJSON設定
                json_data.append(["フラット化した設定情報:"])
                json_data.append(["パラメータ", "値"])

                # 再帰的にJSONをフラット化する関数
                def flatten_json(json_obj, prefix=""):
                    items = []
                    for key, value in json_obj.items():
                        new_key = f"{prefix}{key}" if prefix else key
                        if isinstance(value, dict):
                            items.extend(flatten_json(value, f"{new_key}."))
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    items.extend(flatten_json(item, f"{new_key}[{i}]."))
                                else:
                                    items.append((f"{new_key}[{i}]", item))
                        else:
                            items.append((new_key, value))
                    return items

                # フラット化したJSON設定を追加
                for key, value in flatten_json(config):
                    json_data.append([key, value])

                # データフレームに変換して出力
                json_df = pd.DataFrame(json_data)
                # 4. JSON Config
                if self.excel_sheets_to_export.get("json_config", True):
                    json_df.to_excel(writer, sheet_name="JSON Config", index=False, header=False)

                #------------------------------------------------------
                # 5. 月次リターンシート (Monthly Returns)
                #------------------------------------------------------
                # 5. 月次リターンシート (Monthly Returns)
                # もし UI 側で monthly_returns がオフならシートを作らない
                if self.excel_sheets_to_export.get("monthly_returns", True):

                    # 月次リターンテーブルをまだ生成していない場合、生成を試みる
                    if not hasattr(self, 'pivot_monthly_returns') or self.pivot_monthly_returns is None:
                        self.generate_monthly_returns_table(display_table=False)

                    if hasattr(self, 'pivot_monthly_returns') and self.pivot_monthly_returns is not None:
                        # 月次リターンをパーセント表示形式でコピー
                        monthly_returns_df = self.pivot_monthly_returns.copy()

                        # データフレームを出力
                        monthly_returns_df.to_excel(writer, sheet_name="Monthly Returns")
                    else:
                        # 月次リターンが生成できない場合は空のシートを作成
                        pd.DataFrame().to_excel(writer, sheet_name="Monthly Returns")
                # else:
                #   何もしない（シートを出力しない）

                #------------------------------------------------------
                # 6. 詳細な日次データシート (Daily Returns Detailed)
                #------------------------------------------------------
                # すべての日次データを含む
                daily_detailed_df = self.results.copy()

                # カラム名を日本語に変更
                column_mapping = {
                    "Portfolio_Value": "ポートフォリオ値",
                    "Benchmark_Value": "ベンチマーク値",
                    "Portfolio_Return": "ポートフォリオリターン",
                    "Benchmark_Return": "ベンチマークリターン",
                    "Portfolio_Cumulative": "ポートフォリオ累積",
                    "Benchmark_Cumulative": "ベンチマーク累積",
                    "Portfolio_Drawdown": "ポートフォリオドローダウン",
                    "Benchmark_Drawdown": "ベンチマークドローダウン",
                    "Portfolio_Peak": "ポートフォリオピーク",
                    "Benchmark_Peak": "ベンチマークピーク"
                }

                daily_detailed_df = daily_detailed_df.rename(columns=column_mapping)

                # 相対リターンを追加
                if "ポートフォリオリターン" in daily_detailed_df.columns and "ベンチマークリターン" in daily_detailed_df.columns:
                    daily_detailed_df["相対リターン"] = daily_detailed_df["ポートフォリオリターン"] - daily_detailed_df["ベンチマークリターン"]

                # データフレームを出力
                # 6. Daily Returns Detailed
                if self.excel_sheets_to_export.get("daily_detailed", True):
                    daily_detailed_df.to_excel(writer, sheet_name="Daily Returns Detailed")

                #------------------------------------------------------
                # 7. 取引シート (Trades)
                #------------------------------------------------------
                if hasattr(self, 'positions') and self.positions:
                    trades_data = []

                    # ヘッダー行
                    trades_data.append([
                        "シグナル判定日", "保有開始日", "保有終了日", "保有資産",
                        "保有期間リターン", "モメンタム判定結果",
                        "絶対モメンタムリターン", "リスクフリーレート"
                    ])

                    # データ行
                    for position in self.positions:
                        signal_date = position.get("signal_date").strftime('%Y/%m/%d') if position.get("signal_date") else ""
                        start_date = position.get("start_date").strftime('%Y/%m/%d') if position.get("start_date") else ""
                        end_date = position.get("end_date").strftime('%Y/%m/%d') if position.get("end_date") else ""
                        assets = ", ".join(position.get("assets", []))
                        ret = f"{position.get('return')*100:.2f}%" if position.get("return") is not None else "N/A"
                        message = position.get("message", "")
                        abs_return = f"{position.get('abs_return')*100:.2f}%" if position.get("abs_return") is not None else "N/A"
                        rfr_return = f"{position.get('rfr_return')*100:.2f}%" if position.get("rfr_return") is not None else "N/A"

                        trades_data.append([
                            signal_date, start_date, end_date, assets,
                            ret, message, abs_return, rfr_return
                        ])

                    # データフレームに変換して出力
                    trades_df = pd.DataFrame(trades_data[1:], columns=trades_data[0])
                    # 7. Trades
                    if self.excel_sheets_to_export.get("trades", True):
                        trades_df.to_excel(writer, sheet_name="Trades", index=False)

                else:
                    # 取引情報がない場合は空のシートを作成
                    pd.DataFrame().to_excel(writer, sheet_name="Trades")

            logger.info(f"エクセルファイルを出力しました: {filename}")

            # ここが重要な変更部分：自動ダウンロード処理を削除し、代わりに情報を返す
            return {"filename": filename, "should_download": auto_download}

        except Exception as e:
            logger.error(f"エクセルファイル出力中にエラーが発生しました: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

