# デュアル・モメンタム・バックテスト・システム

Gary Antonacci 氏の **Dual Momentum** 手法を Python で完全実装し、
*絶対モメンタム判定*・*複数期間ウェイト*・*退避先資産戦略*・*対話型 UI* などを拡張したオールインワン・バックテスト環境です。

> **💡 GitHub の README は UTF‑8 なので日本語も問題なく表示できます**。
>
> 国際的なプロジェクトの場合は `README_EN.md` を別途用意すると親切ですが、
> 日本語 README だけでも構いません。

---
## ✨ 主な特長

| 区分 | ハイライト |
|------|------------|
| **データ** | • **yfinance** 経由で日次 OHLCV を取得  <br>• *FRED* から 3 ヶ月 T‑Bill (DTB3) を取得し、失敗時は IRX に自動フォールバック |
| **戦略ロジック** | • 相対モメンタム（単一 or 複数ルックバック窓）  <br>• リスクフリーレートと比較する絶対モメンタム  <br>• ネガティブ相対モメンタム時の安全装置 |
| **ポートフォリオ構築** | • 上位 *N* 資産を均等保持  <br>• 退避先資産を **等ウェイト** または **Top 1** で保有 |
| **リバランス** | • 月次／隔月／四半期  <br>• 月末終値または翌取引日の寄付きで約定 |
| **分析指標** | • CAGR・シャープ／ソルティノ・MAR・ドローダウン・相関  <br>• 線形／対数リターン曲線・月次／年次リターン表 |
| **ユーティリティ** | • パラメータ検証とデータ品質診断  <br>• モメンタム計算キャッシュ（期限付き）  <br>• ワンクリックで多シート構成の Excel を出力 |
| **UI** | • `ipywidgets` ベースの GUI（年月ピッカー、ラジオボタン、テーブル）  <br>• 動的シグナル・ダッシュボード |

---
## 📦 インストール

### Google Colab（推奨）
  1. `dual-momentum-backtest.ipynb` を開き、先頭セルを実行すると依存パッケージが自動インストールされます。

### ローカル環境
```bash
git clone https://github.com/yourname/dual-momentum-backtest.git
cd dual-momentum-backtest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # import から自動生成した一覧
jupyter notebook dual-momentum-backtest.ipynb
```
*Python 3.9 以上* を推奨（動作確認は 3.11）。

---
## 🚀 クイックスタート

```python
from dual_momentum import DualMomentumModel
model = DualMomentumModel()
model.tickers = ["TQQQ", "TECL", "XLU"]
model.fetch_data()
model.run_backtest()
model.plot_performance()
model.display_performance_summary()
```

1. UI で開始／終了年月と資産を選択
2. **Fetch Data** で価格取得 & 品質チェック
3. **Run Backtest** をクリック
4. グラフ・テーブル・シグナルを確認
5. 必要に応じて `export_to_excel()` でレポート保存

---
## 🔧 主要パラメータ

| 属性 | デフォルト | 説明 |
|------|-----------|------|
| `tickers` | `["TQQQ", "TECL"]` | リスクオン候補 ETF |
| `single_absolute_momentum` | `"Yes"` | 絶対モメンタム判定を有効化 |
| `absolute_momentum_asset` | `"LQD"` | 絶対モメンタム用安全資産 |
| `out_of_market_assets` | `["XLU"]` | 市場退出時の退避先資産 |
| `out_of_market_strategy` | `"Equal Weight" / "Top 1"` | 退避先資産の配分方法 |
| `performance_periods` | `"Single Period" / "Multiple Periods"` | 1 期間 or 最大 5 期間ウェイト |
| `lookback_period` / `lookback_unit` | `12`, `"Months"` | 単一期間モードのルックバック |
| `multiple_periods` | （コード参照） | 長さ／単位／重みのタプル |
| `trading_frequency` | `"Monthly"` | リバランス頻度 |
| `trade_execution` | `"Trade at next open price"` | 執行タイミング |
| `assets_to_hold` | `1` | 保有する上位銘柄数 |

全パラメータは UI でも Python でも変更可能です。

---
## 📈 出力物

* **エクイティカーブ**（線形 & 対数）
* **ドローダウン推移**
* **月次／年次リターン表**（ポートフォリオ & ベンチマーク）
* **取引ログ**（シグナル日・保有期間・リターン・RFR 等）
* **Excel レポート**（Settings / Performance / Daily Simple / JSON Config / Monthly Returns / Daily Detailed / Trades の 7 シート）

---
## 🤔 トラブルシューティング

| 症状 | 典型原因 | 対処 |
|------|----------|------|
| *Parameter validation failed* | ルックバック範囲外、重み合計 ≠ 100 % | エラーメッセージに従い値を修正 |
| 価格データが欠落 | 入力ティッカーの打ち間違い・上場廃止 | Yahoo Finance で確認し差し替え |
| FRED API エラー | `FRED_API_KEY` 未設定 or 不正 | 有効キーをセット or IRX フォールバックを利用 |
| NaN 警告が多数 | データ期間が短い / 取引停止銘柄 | バックテスト期間を短縮 or 資産を変更 |

---
## 📝 ライセンス

本リポジトリは **MIT License** で公開されています。詳細は `LICENSE` をご覧ください。

---
## 🙏 謝辞

* Gary Antonacci – Dual Momentum 理論
* OSS コミュニティ: `yfinance`, `fredapi`, `pandas_market_calendars` など
* 2025 年 4 月、Google Colab にて ❤️ を込めて開発

