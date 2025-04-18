# dual_momentum/ui/dashboard.py の先頭に追加
from dual_momentum.core.model import DualMomentumModel
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import calendar
import json
import os
from dual_momentum.validators.input import InputValidator

# 既存のコード...

def create_dual_momentum_ui():
    model = DualMomentumModel()
    today = datetime.now()
    # 時間設定：スライダーから年月セレクターに変更
    start_picker = create_year_month_picker(2010, 1, 'Start Year')
    end_picker = create_year_month_picker(today.year, today.month, 'End Year')

    tickers = widgets.SelectMultiple(
        options=['TQQQ', 'TECL', 'XLU', 'SPXL', 'QQQ'],
        value=('TQQQ', 'TECL'),
        description='Tickers:',
        style={'description_width': 'initial'}
    )

    specify_tickers = widgets.Text(
        value='',
        description='Specify Tickers:',
        placeholder='例: TQQQ,TECL,UPRO',
        style={'description_width': 'initial'}
    )
    single_absolute_momentum = widgets.RadioButtons(
        options=['Yes', 'No'],
        value='Yes',
        description='Single absolute momentum:',
        style={'description_width': 'initial'}
    )
    negative_relative_momentum = widgets.RadioButtons(
        options=['Yes', 'No'],
        value='No',
        description='Negative relative momentum:',
        style={'description_width': 'initial'}
    )

    absolute_momentum_asset = widgets.Dropdown(
        options=['LQD', '^VIX', 'TMF'],
        value='LQD',
        description='Absolute momentum asset:',
        style={'description_width': 'initial'}
    )

    specify_absolute_momentum_asset = widgets.Text(
        value='',
        description='Specify absolute momentum asset:',
        placeholder='例: TLT',
        style={'description_width': 'initial'}
    )

    out_of_market_assets = widgets.SelectMultiple(
        options=['XLU', 'GLD', 'SHY' ,'TMV' ,'TQQQ'],
        value=("XLU",),
        description='Out of Market Assets:',
        style={'description_width': 'initial'}
    )

    # 退避先資産の選択戦略（新規追加）
    out_of_market_strategy = widgets.RadioButtons(
        options=['Equal Weight', 'Top 1'],
        value='Equal Weight',
        description='退避先資産の選択:',
        style={'description_width': 'initial'}
    )
    specify_out_of_market_asset = widgets.Text(
        value='',
        description='Specify out of market asset:',
        placeholder='例: TQQQ,IEF',
        style={'description_width': 'initial'}
    )
    performance_periods = widgets.RadioButtons(
        options=['Single Period', 'Multiple Periods'],
        value='Multiple Periods',
        description='Performance Periods:',
        style={'description_width': 'initial'}
    )
    lookback_period = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=12,
        description='Lookback period:',
        style={'description_width': 'initial'}
    )
    lookback_unit = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit:',
        disabled=False
    )
    absolute_momentum_custom_period_checkbox = widgets.Checkbox(
        value=False,
        description='絶対モメンタムの期間をカスタマイズ',
        style={'description_width': 'initial'}
    )
    absolute_momentum_period = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=12,
        description='Absolute Momentum period:',
        style={'description_width': 'initial'},
        disabled=True
    )
    lookback_period1 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=2,
        description='Length #1:',
        style={'description_width': 'initial'}
    )
    lookback_unit1 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #1:',
        disabled=False
    )
    weight1 = widgets.IntSlider(
        value=20,
        min=0,
        max=100,
        step=5,
        description='Weight #1 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period2 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=6,
        description='Length #2:',
        style={'description_width': 'initial'}
    )
    lookback_unit2 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #2:',
        disabled=False
    )
    weight2 = widgets.IntSlider(
        value=20,
        min=0,
        max=100,
        step=5,
        description='Weight #2 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period3 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=12,
        description='Length #3:',
        style={'description_width': 'initial'}
    )
    lookback_unit3 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #3:',
        disabled=False
    )
    weight3 = widgets.IntSlider(
        value=60,
        min=0,
        max=100,
        step=5,
        description='Weight #3 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period4 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=3,
        description='Length #4:',
        style={'description_width': 'initial'}
    )
    lookback_unit4 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #4:',
        disabled=False
    )
    weight4 = widgets.IntSlider(
        value=0,
        min=0,
        max=100,
        step=5,
        description='Weight #4 (%):',
        style={'description_width': 'initial'}
    )
    lookback_period5 = widgets.Dropdown(
        options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
        value=3,
        description='Length #5:',
        style={'description_width': 'initial'}
    )
    lookback_unit5 = widgets.RadioButtons(
        options=['Months', 'Days'],
        value='Months',
        description='Unit #5:',
        disabled=False
    )
    weight5 = widgets.IntSlider(
        value=0,
        min=0,
        max=100,
        step=5,
        description='Weight #5 (%):',
        style={'description_width': 'initial'}
    )
    assets_to_hold = widgets.Dropdown(
        options=[1,2,3,4,5,6],
        value=1,
        description='Assets to hold:',
        style={'description_width': 'initial'}
    )

    excel_export_checkbox = widgets.Checkbox(
        value=False,
        description='バックテスト後にエクセル出力',
        style={'description_width': 'initial'}
    )

    # Excel出力用のチェックボックス群
    excel_label = widgets.HTML(value="<b>Excel Output Sheets:</b>")

    excel_cb_settings = widgets.Checkbox(value=True, description="Settingsシート", layout=widgets.Layout(width='250px'))
    excel_cb_performance = widgets.Checkbox(value=True, description="Performanceシート", layout=widgets.Layout(width='250px'))
    excel_cb_daily_simple = widgets.Checkbox(value=True, description="Daily Returns Simple", layout=widgets.Layout(width='250px'))
    excel_cb_json_config = widgets.Checkbox(value=True, description="JSON Config", layout=widgets.Layout(width='250px'))
    excel_cb_monthly_returns = widgets.Checkbox(value=True, description="Monthly Returns", layout=widgets.Layout(width='250px'))
    excel_cb_daily_detailed = widgets.Checkbox(value=True, description="Daily Returns Detailed", layout=widgets.Layout(width='250px'))
    excel_cb_trades = widgets.Checkbox(value=True, description="Trades", layout=widgets.Layout(width='250px'))

    excel_sheets_vbox = widgets.VBox([
        excel_label,
        excel_cb_settings,
        excel_cb_performance,
        excel_cb_daily_simple,
        excel_cb_json_config,
        excel_cb_monthly_returns,
        excel_cb_daily_detailed,
        excel_cb_trades
    ])

    output_options = widgets.VBox([
        widgets.HTML(value="<b>出力オプション:</b>"),
        widgets.Checkbox(value=True, description='パフォーマンスグラフ', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='年次リターンテーブル', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='月次リターンテーブル', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='モデルシグナル表示', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='パフォーマンスサマリー', layout=widgets.Layout(width='250px')),
        widgets.Checkbox(value=True, description='取引履歴テーブル', layout=widgets.Layout(width='250px')),
        excel_sheets_vbox  # 追加
    ])

    trading_frequency = widgets.Dropdown(
        options=[
            'Monthly',
            'Bimonthly (hold: 1,3,5,7,9,11)',
            'Bimonthly (hold: 2,4,6,8,10,12)',
            'Quarterly (hold: 1,4,7,10)',
            'Quarterly (hold: 2,5,8,11)',
            'Quarterly (hold: 3,6,9,12)'
        ],
        value='Monthly',
        description='Trading Frequency:',
        style={'description_width': 'initial'}
    )


    trade_execution_label = widgets.HTML(value='<p style="font-weight: bold;">Trade Execution:</p>')
    trade_execution_at_end = widgets.Checkbox(value=False, description='Trade at end of month price')
    trade_execution_at_next = widgets.Checkbox(value=False, description='Trade at next close price')
    trade_execution_at_next_open = widgets.Checkbox(value=True, description='Trade at next open price')

    def update_trade_execution(change):
        if change['owner'] == trade_execution_at_end and change['new']:
            trade_execution_at_next.value = False
            trade_execution_at_next_open.value = False
        elif change['owner'] == trade_execution_at_next and change['new']:
            trade_execution_at_end.value = False
            trade_execution_at_next_open.value = False
        elif change['owner'] == trade_execution_at_next_open and change['new']:
            trade_execution_at_end.value = False
            trade_execution_at_next.value = False
        # いずれかが選択されていることを確認
        if not (trade_execution_at_end.value or trade_execution_at_next.value or trade_execution_at_next_open.value):
            change['owner'].value = True
    trade_execution_at_end.observe(update_trade_execution, names='value')
    trade_execution_at_next.observe(update_trade_execution, names='value')
    trade_execution_at_next_open.observe(update_trade_execution, names='value')

    def get_trade_execution():
        if trade_execution_at_end.value:
            return 'Trade at end of month price'
        elif trade_execution_at_next_open.value:
            return 'Trade at next open price'
        else:
            return 'Trade at next close price'

    benchmark_ticker = widgets.Text(
        value='SPY',
        description='Benchmark Ticker:',
        style={'description_width': 'initial'}
    )



    config_textarea = widgets.Textarea(
        value="",
        description="Config JSON:",
        layout=widgets.Layout(width="100%", height="150px")
    )
    config_textarea.disabled = True
    save_button = widgets.Button(
        description="Save Settings",
        button_style="info",
        icon="save"
    )
    load_button = widgets.Button(
        description="Load Settings",
        button_style="warning",
        icon="upload"
    )
    file_upload = widgets.FileUpload(
        accept=".json",
        multiple=False
    )
    uploaded_portfolio_names = set()
    portfolio_list_label = widgets.HTML(value="<b>Uploaded Portfolios:</b><br>None")
    def update_portfolio_list_display():
        if uploaded_portfolio_names:
            portfolio_list_label.value = (
                "<b>Uploaded Portfolios:</b><br>" +
                "<br>".join(sorted(uploaded_portfolio_names))
            )
        else:
            portfolio_list_label.value = "<b>Uploaded Portfolios:</b><br>None"
    fetch_button = widgets.Button(
        description='Fetch Data',
        button_style='primary',
        icon='download'
    )
    run_button = widgets.Button(
        description='Run Backtest',
        button_style='success',
        icon='play'
    )
    output = widgets.Output()
    def update_absolute_momentum_period(change):
        absolute_momentum_period.disabled = not change['new']
        model.absolute_momentum_custom_period = change['new']
    absolute_momentum_custom_period_checkbox.observe(update_absolute_momentum_period, names='value')
    validation_state = {
        'start_year': True,
        'start_month': True,
        'end_year': True,
        'end_month': True,
        'tickers': True,
        'single_absolute_momentum': True,
        'absolute_momentum_asset': True,
        'out_of_market_assets': True,
        'lookback_period': True,
        'lookback_unit': True,
        'absolute_momentum_period': True,
        'lookback_period1': True,
        'lookback_unit1': True,
        'weight1': True,
        'lookback_period2': True,
        'lookback_unit2': True,
        'weight2': True,
        'lookback_period3': True,
        'lookback_unit3': True,
        'weight3': True,
        'lookback_period4': True,
        'lookback_unit4': True,
        'weight4': True,
        'lookback_period5': True,
        'lookback_unit5': True,
        'weight5': True,
        'benchmark_ticker': True,
    }
    validation_message = widgets.HTML(
        value="",
        description="",
        style={'description_width': 'initial'}
    )
    def update_validation_message():
        error_messages = []
        warning_messages = []
        valid, message = InputValidator.validate_date_range(
            start_picker.children[1].children[0].value, start_picker.children[1].children[1].value,
            end_picker.children[1].children[0].value, end_picker.children[1].children[1].value
        )
        if not valid:
            error_messages.append(f"📅 {message}")
            validation_state['start_year'] = False
            validation_state['start_month'] = False
            validation_state['end_year'] = False
            validation_state['end_month'] = False
        else:
            validation_state['start_year'] = True
            validation_state['start_month'] = True
            validation_state['end_year'] = True
            validation_state['end_month'] = True
        if specify_tickers.value.strip():
            ticker_list = [t.strip() for t in specify_tickers.value.split(',') if t.strip()]
        else:
            ticker_list = list(tickers.value)
        valid, message = InputValidator.validate_ticker_symbols(ticker_list)
        if not valid:
            error_messages.append(f"🏷️ {message}")
            validation_state['tickers'] = False
        else:
            validation_state['tickers'] = True
        if single_absolute_momentum.value == 'Yes':
            valid, message = InputValidator.validate_absolute_momentum_asset(absolute_momentum_asset.value)
            if not valid:
                error_messages.append(f"🔄 {message}")
                validation_state['absolute_momentum_asset'] = False
            else:
                validation_state['absolute_momentum_asset'] = True
        else:
            validation_state['absolute_momentum_asset'] = True
        out_assets = list(out_of_market_assets.value)
        if specify_out_of_market_asset.value.strip():
            out_assets = [s.strip() for s in specify_out_of_market_asset.value.split(',') if s.strip()]
        valid, message = InputValidator.validate_out_of_market_assets(out_assets)
        if not valid:
            warning_messages.append(f"⚠️ {message}")
            validation_state['out_of_market_assets'] = False
        else:
            validation_state['out_of_market_assets'] = True
        if performance_periods.value == 'Single Period':
            valid, message = InputValidator.validate_lookback_period(
                lookback_period.value, lookback_unit.value
            )
            if not valid:
                error_messages.append(f"📊 {message}")
                validation_state['lookback_period'] = False
            else:
                validation_state['lookback_period'] = True
            if absolute_momentum_custom_period_checkbox.value:
                valid, message = InputValidator.validate_lookback_period(
                    absolute_momentum_period.value, lookback_unit.value
                )
                if not valid:
                    error_messages.append(f"🔄 絶対モメンタム期間: {message}")
                    validation_state['absolute_momentum_period'] = False
                else:
                    validation_state['absolute_momentum_period'] = True
        else:
            period_widgets = [
                (lookback_period1, lookback_unit1, weight1, 'lookback_period1', 'weight1'),
                (lookback_period2, lookback_unit2, weight2, 'lookback_period2', 'weight2'),
                (lookback_period3, lookback_unit3, weight3, 'lookback_period3', 'weight3'),
                (lookback_period4, lookback_unit4, weight4, 'lookback_period4', 'weight4'),
                (lookback_period5, lookback_unit5, weight5, 'lookback_period5', 'weight5')
            ]
            period_weights = []
            for i, (period, unit, weight, period_key, weight_key) in enumerate(period_widgets):
                if weight.value > 0:
                    valid, message = InputValidator.validate_lookback_period(period.value, unit.value)
                    if not valid:
                        error_messages.append(f"📊 期間 #{i+1}: {message}")
                        validation_state[period_key] = False
                    else:
                        validation_state[period_key] = True
                    period_weights.append(weight.value)
                    validation_state[weight_key] = True
                else:
                    validation_state[period_key] = True
                    validation_state[weight_key] = True
            if period_weights:
                valid, message = InputValidator.validate_weights(period_weights)
                if not valid:
                    warning_messages.append(f"⚠️ {message}")
                    for _, _, _, _, weight_key in period_widgets:
                        validation_state[weight_key] = False
            else:
                error_messages.append("📊 複数期間モードでは、少なくとも1つの期間に正の重みを設定する必要があります。")
                for _, _, _, _, weight_key in period_widgets:
                    validation_state[weight_key] = False
        valid, message = InputValidator.validate_benchmark_ticker(benchmark_ticker.value)
        if not valid:
            error_messages.append(f"📈 {message}")
            validation_state['benchmark_ticker'] = False
        else:
            validation_state['benchmark_ticker'] = True
        update_widget_styles()
        if error_messages:
            error_html = "<div style='color: red; margin-bottom: 10px;'><strong>⛔ エラー:</strong><ul>"
            for msg in error_messages:
                error_html += f"<li>{msg}</li>"
            error_html += "</ul></div>"
            if warning_messages:
                error_html += "<div style='color: orange; margin-bottom: 10px;'><strong>⚠️ 警告:</strong><ul>"
                for msg in warning_messages:
                    error_html += f"<li>{msg}</li>"
                error_html += "</ul></div>"
            validation_message.value = error_html
        elif warning_messages:
            warning_html = "<div style='color: orange; margin-bottom: 10px;'><strong>⚠️ 警告:</strong><ul>"
            for msg in warning_messages:
                warning_html += f"<li>{msg}</li>"
            warning_html += "</ul></div>"
            validation_message.value = warning_html
        else:
            validation_message.value = "<div style='color: green; margin-bottom: 10px;'><strong>✅ 全ての入力が有効です</strong></div>"
    error_style = {'description_width': 'initial', 'border': '1px solid red'}
    normal_style = {'description_width': 'initial'}
    def update_widget_styles():
        # 時間設定：start_pickerとend_pickerの子ウィジェットのスタイル更新
        start_year_widget = start_picker.children[1].children[0]
        start_month_widget = start_picker.children[1].children[1]
        end_year_widget = end_picker.children[1].children[0]
        end_month_widget = end_picker.children[1].children[1]
        start_year_widget.style = error_style if not validation_state['start_year'] else normal_style
        start_year_widget.description = '❌ Year:' if not validation_state['start_year'] else 'Year:'
        start_month_widget.style = error_style if not validation_state['start_month'] else normal_style
        start_month_widget.description = '❌ Month:' if not validation_state['start_month'] else 'Month:'
        end_year_widget.style = error_style if not validation_state['end_year'] else normal_style
        end_year_widget.description = '❌ Year:' if not validation_state['end_year'] else 'Year:'
        end_month_widget.style = error_style if not validation_state['end_month'] else normal_style
        end_month_widget.description = '❌ Mon:' if not validation_state['end_month'] else 'Month:'
        tickers.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['tickers'] else {'description_width': 'initial'}
        tickers.description = '❌ Tickers:' if not validation_state['tickers'] else 'Tickers:'
        if single_absolute_momentum.value == 'Yes':
            absolute_momentum_asset.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['absolute_momentum_asset'] else {'description_width': 'initial'}
            absolute_momentum_asset.description = '❌ Absolute momentum asset:' if not validation_state['absolute_momentum_asset'] else 'Absolute momentum asset:'
        else:
            absolute_momentum_asset.style = normal_style
        benchmark_ticker.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['benchmark_ticker'] else {'description_width': 'initial'}
        benchmark_ticker.description = '❌ Benchmark Ticker:' if not validation_state['benchmark_ticker'] else 'Benchmark Ticker:'
        if performance_periods.value == 'Single Period':
            lookback_period.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['lookback_period'] else {'description_width': 'initial'}
            lookback_period.description = '❌ Lookback period:' if not validation_state['lookback_period'] else 'Lookback period:'
            if absolute_momentum_custom_period_checkbox.value:
                absolute_momentum_period.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state['absolute_momentum_period'] else {'description_width': 'initial'}
                absolute_momentum_period.description = '❌ Absolute Momentum period:' if not validation_state['absolute_momentum_period'] else 'Absolute Momentum period:'
        else:
            period_widgets = [
                (lookback_period1, 'lookback_period1', 'Length #1:'),
                (lookback_period2, 'lookback_period2', 'Length #2:'),
                (lookback_period3, 'lookback_period3', 'Length #3:'),
                (lookback_period4, 'lookback_period4', 'Length #4:'),
                (lookback_period5, 'lookback_period5', 'Length #5:')
            ]
            for widget, key, desc in period_widgets:
                widget.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state[key] else {'description_width': 'initial'}
                widget.description = f'❌ {desc.replace("❌ ", "")}' if not validation_state[key] else desc
            weight_widgets = [
                (weight1, 'weight1', 'Weight #1 (%):'),
                (weight2, 'weight2', 'Weight #2 (%):'),
                (weight3, 'weight3', 'Weight #3 (%):'),
                (weight4, 'weight4', 'Weight #4 (%):'),
                (weight5, 'weight5', 'Weight #5 (%):')
            ]
            for widget, key, desc in weight_widgets:
                widget.style = {'description_width': 'initial', 'border': '1px solid red'} if not validation_state[key] else {'description_width': 'initial'}
                widget.description = f'❌ {desc.replace("❌ ", "")}' if not validation_state[key] else desc
    def connect_validation_callbacks():
        start_picker.children[1].children[0].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        start_picker.children[1].children[1].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        end_picker.children[1].children[0].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        end_picker.children[1].children[1].observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        tickers.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        single_absolute_momentum.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        absolute_momentum_asset.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        out_of_market_assets.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        specify_out_of_market_asset.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        performance_periods.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        lookback_period.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        lookback_unit.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        absolute_momentum_custom_period_checkbox.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        absolute_momentum_period.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        period_widgets = [
            (lookback_period1, lookback_unit1, weight1),
            (lookback_period2, lookback_unit2, weight2),
            (lookback_period3, lookback_unit3, weight3),
            (lookback_period4, lookback_unit4, weight4),
            (lookback_period5, lookback_unit5, weight5)
        ]
        for period, unit, weight in period_widgets:
            period.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
            unit.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
            weight.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
        benchmark_ticker.observe(lambda change: update_validation_message() if change['name'] == 'value' else None, names='value')
    connect_validation_callbacks()
    update_validation_message()

    def on_save_clicked(b):
        config = {
            "time": {
                "start_year": start_picker.children[1].children[0].value,
                "start_month": start_picker.children[1].children[1].value,
                "end_year": end_picker.children[1].children[0].value,
                "end_month": end_picker.children[1].children[1].value
            },
            "assets": {
                "tickers": tickers.value,
                "single_absolute_momentum": single_absolute_momentum.value,
                "absolute_momentum_asset": absolute_momentum_asset.value,
                "out_of_market_assets": list(out_of_market_assets.value),
                "specify_out_of_market_asset": specify_out_of_market_asset.value,
                "out_of_market_strategy": out_of_market_strategy.value
            },
            "performance": {
                "performance_periods": performance_periods.value,
                "lookback_period": lookback_period.value,
                "lookback_unit": lookback_unit.value,
                "multiple_periods": {
                    "period1": {"lookback_period": lookback_period1.value, "lookback_unit": lookback_unit1.value, "weight": weight1.value},
                    "period2": {"lookback_period": lookback_period2.value, "lookback_unit": lookback_unit2.value, "weight": weight2.value},
                    "period3": {"lookback_period": lookback_period3.value, "lookback_unit": lookback_unit3.value, "weight": weight3.value},
                    "period4": {"lookback_period": lookback_period4.value, "lookback_unit": lookback_unit4.value, "weight": weight4.value},
                    "period5": {"lookback_period": lookback_period5.value, "lookback_unit": lookback_unit5.value, "weight": weight5.value}
                },
                "weighting_method": performance_periods.value == 'Single Period' and lookback_period.value or model.weighting_method,
                "assets_to_hold": assets_to_hold.value
            },
            "trade": {
                "trading_frequency": trading_frequency.value,
                "trade_execution": get_trade_execution(),
                "benchmark_ticker": benchmark_ticker.value
            },
            "absolute_momentum": {
                "custom_period": model.absolute_momentum_custom_period,
                "period": absolute_momentum_period.value
            }
        }
        json_str = json.dumps(config, ensure_ascii=False, indent=2)
        config_textarea.value = json_str
        with output:
            clear_output()
            print("✅ Settings saved in JSON format.")
    save_button.on_click(on_save_clicked)
    def on_load_clicked(b):
        nonlocal file_upload
        if len(file_upload.value) == 0:
            with output:
                clear_output()
                print("❌ Please upload a settings file first.")
            return
        filename = list(file_upload.value.keys())[0]
        uploaded_file = file_upload.value[filename]
        portfolio_name = os.path.splitext(filename.strip().lower())[0]
        if portfolio_name in uploaded_portfolio_names:
            with output:
                clear_output()
                print("⚠️ This portfolio has already been uploaded.")
            return
        uploaded_portfolio_names.add(portfolio_name)
        try:
            config = json.loads(uploaded_file['content'].decode("utf-8"))
        except Exception as e:
            with output:
                clear_output()
                print(f"❌ Failed to load settings file: {e}")
            return
        apply_config_to_ui(config)
        update_portfolio_list_display()
        with output:
            clear_output()
            print("✅ Settings loaded successfully.")
        new_file_upload = widgets.FileUpload(accept=".json", multiple=False)
        config_buttons.children = [save_button, load_button, new_file_upload]
        file_upload = new_file_upload
    load_button.on_click(on_load_clicked)

    def apply_config_to_ui(config):
        if "time" in config:
            start_picker.children[1].children[0].value = config["time"].get("start_year", start_picker.children[1].children[0].value)
            start_picker.children[1].children[1].value = config["time"].get("start_month", start_picker.children[1].children[1].value)
            end_picker.children[1].children[0].value = config["time"].get("end_year", end_picker.children[1].children[0].value)
            end_picker.children[1].children[1].value = config["time"].get("end_month", end_picker.children[1].children[1].value)

        if "assets" in config:
            # タプルに変換して設定（SelectMultipleはタプルを期待）
            tickers_list = config["assets"].get("tickers", [])
            # リストでない場合は変換
            if not isinstance(tickers_list, list):
                tickers_list = [tickers_list] if tickers_list else []
            tickers.value = tuple(tickers_list)

            # Specify Tickersの設定
            specify_tickers.value = config["assets"].get("specify_tickers", "")

            specify_tickers.value = config["assets"].get("specify_tickers", specify_tickers.value)

            single_absolute_momentum.value = config["assets"].get("single_absolute_momentum", single_absolute_momentum.value)

            # シンプルな実装
            absolute_momentum_asset.value = config["assets"].get("absolute_momentum_asset", absolute_momentum_asset.value)
            specify_absolute_momentum_asset.value = config["assets"].get("specify_absolute_momentum_asset", specify_absolute_momentum_asset.value)

            out_of_market_assets.value = tuple(config["assets"].get("out_of_market_assets", list(out_of_market_assets.value)))
            specify_out_of_market_asset.value = config["assets"].get("specify_out_of_market_asset", specify_out_of_market_asset.value)
            out_of_market_strategy.value = config["assets"].get("out_of_market_strategy", out_of_market_strategy.value)
        if "performance" in config:
            performance_periods.value = config["performance"].get("performance_periods", performance_periods.value)
            lookback_period.value = config["performance"].get("lookback_period", lookback_period.value)
            lookback_unit.value = config["performance"].get("lookback_unit", lookback_unit.value)
            if "multiple_periods" in config["performance"]:
                mp = config["performance"]["multiple_periods"]
                period1 = mp.get("period1", {})
                lookback_period1.value = period1.get("lookback_period", lookback_period1.value)
                lookback_unit1.value = period1.get("lookback_unit", lookback_unit1.value)
                weight1.value = period1.get("weight", weight1.value)
                period2 = mp.get("period2", {})
                lookback_period2.value = period2.get("lookback_period", lookback_period2.value)
                lookback_unit2.value = period2.get("lookback_unit", lookback_unit2.value)
                weight2.value = period2.get("weight", weight2.value)
                period3 = mp.get("period3", {})
                lookback_period3.value = period3.get("lookback_period", lookback_period3.value)
                lookback_unit3.value = period3.get("lookback_unit", lookback_unit3.value)
                weight3.value = period3.get("weight", weight3.value)
                period4 = mp.get("period4", {})
                lookback_period4.value = period4.get("lookback_period", lookback_period4.value)
                lookback_unit4.value = period4.get("lookback_unit", lookback_unit4.value)
                weight4.value = period4.get("weight", weight4.value)
                period5 = mp.get("period5", {})
                lookback_period5.value = period5.get("lookback_period", lookback_period5.value)
                lookback_unit5.value = period5.get("lookback_unit", lookback_unit5.value)
                weight5.value = period5.get("weight", weight5.value)

        if "trade" in config:
            trading_frequency.value = config["trade"].get("trading_frequency", trading_frequency.value)
            trade_exec = config["trade"].get("trade_execution", "Trade at end of month price")

            # すべてのチェックボックスをリセット
            trade_execution_at_end.value = False
            trade_execution_at_next.value = False
            trade_execution_at_next_open.value = False

            # 該当するオプションを選択
            if trade_exec == "Trade at end of month price":
                trade_execution_at_end.value = True
            elif trade_exec == "Trade at next open price":
                trade_execution_at_next_open.value = True
            else:  # "Trade at next close price"がデフォルト
                trade_execution_at_next.value = True

            benchmark_ticker.value = config["trade"].get("benchmark_ticker", benchmark_ticker.value)

        if "absolute_momentum" in config:
            abs_config = config["absolute_momentum"]
            absolute_momentum_custom_period_checkbox.value = abs_config.get("custom_period", model.absolute_momentum_custom_period)
            absolute_momentum_period.value = abs_config.get("period", absolute_momentum_period.value)
            model.absolute_momentum_custom_period = absolute_momentum_custom_period_checkbox.value
            model.absolute_momentum_period = absolute_momentum_period.value
    def on_fetch_clicked(b):
        with output:
            clear_output()
            update_validation_message()
            if any(not state for state in validation_state.values()):
                print("⛔ 入力エラーがあります。エラーを修正してから再試行してください。")
                display(validation_message)
                return
            model.momentum_cache = {}
            # 時間設定の取得：カスタム年月セレクターから値を取得
            model.start_year = start_picker.children[1].children[0].value
            model.start_month = start_picker.children[1].children[1].value
            model.end_year = end_picker.children[1].children[0].value
            model.end_month = end_picker.children[1].children[1].value
            if specify_tickers.value.strip():
                model.tickers = [t.strip() for t in specify_tickers.value.split(',') if t.strip()]
            else:
                model.tickers = list(tickers.value)

            model.single_absolute_momentum = single_absolute_momentum.value
            # 絶対モメンタム資産の設定（Out of Market Assetsと同じロジック）
            if specify_absolute_momentum_asset.value.strip():
                model.absolute_momentum_asset = specify_absolute_momentum_asset.value.strip()
            else:
                model.absolute_momentum_asset = absolute_momentum_asset.value
            model.negative_relative_momentum = negative_relative_momentum.value
            if specify_out_of_market_asset.value.strip():
                model.out_of_market_assets = [s.strip() for s in specify_out_of_market_asset.value.split(',') if s.strip()]
            else:
                model.out_of_market_assets = list(out_of_market_assets.value)

            model.out_of_market_strategy = out_of_market_strategy.value
            model.performance_periods = performance_periods.value
            if model.performance_periods == 'Single Period':
                model.lookback_period = lookback_period.value
                model.lookback_unit = lookback_unit.value
            else:
                # 複数期間設定はcreate_multiple_periods_tableウィジェットのコールバックでmodel.multiple_periodsが更新されているため追加処理不要
                model.multiple_periods_count = sum(1 for p in model.multiple_periods if p.get("weight", 0) > 0)
            model.assets_to_hold = assets_to_hold.value
            model.trading_frequency = trading_frequency.value
            model.trade_execution = get_trade_execution()
            model.benchmark_ticker = benchmark_ticker.value
            model.absolute_momentum_custom_period = absolute_momentum_custom_period_checkbox.value
            model.absolute_momentum_period = absolute_momentum_period.value
            valid, errors, warnings_list = model.validate_parameters()
            if not valid:
                print("⚠️ Parameter validation failed. Please correct the following errors:")
                for error in errors:
                    print(f"  ❌ {error}")
                return
            if warnings_list:
                print("⚠️ Warnings:")
                for warning in warnings_list:
                    print(f"  ⚠️ {warning}")
                print("")
            print("🔄 Fetching data...")
            success = model.fetch_data()
            if not success:
                print("❌ Data fetch failed. Please review your settings.")
                return
            cache_info = model.diagnose_cache()
            if cache_info["status"] != "ok" and cache_info["status"] != "empty":
                print(f"\n⚠️ キャッシュ警告: {cache_info['message']}")
    fetch_button.on_click(on_fetch_clicked)

    def on_run_clicked(b):
        with output:
            clear_output()
            update_validation_message()
            if any(not state for state in validation_state.values()):
                print("⛔ 入力エラーがあります。エラーを修正してから再試行してください。")
                display(validation_message)
                return

            # (★) ここで全結果をまとめてクリア
            model.clear_results()

            print("🧹 前回の結果データを完全にクリアしています...")

            # ---- 以下は各種パラメータを model に設定する流れ ----
            model.start_year = start_picker.children[1].children[0].value
            model.start_month = start_picker.children[1].children[1].value
            model.end_year = end_picker.children[1].children[0].value
            model.end_month = end_picker.children[1].children[1].value

            if specify_tickers.value.strip():
                model.tickers = [t.strip() for t in specify_tickers.value.split(',') if t.strip()]
            else:
                model.tickers = list(tickers.value)

            model.single_absolute_momentum = single_absolute_momentum.value
            # 絶対モメンタム資産の設定（Out of Market Assetsと同じロジック）
            if specify_absolute_momentum_asset.value.strip():
                model.absolute_momentum_asset = specify_absolute_momentum_asset.value.strip()
            else:
                model.absolute_momentum_asset = absolute_momentum_asset.value

            if specify_out_of_market_asset.value.strip():
                model.out_of_market_assets = [
                    s.strip() for s in specify_out_of_market_asset.value.split(',')
                    if s.strip()
                ]
            else:
                model.out_of_market_assets = list(out_of_market_assets.value)
            model.out_of_market_strategy = out_of_market_strategy.value
            model.performance_periods = performance_periods.value
            if model.performance_periods == 'Single Period':
                model.lookback_period = lookback_period.value
                model.lookback_unit = lookback_unit.value
            else:
                model.multiple_periods_count = sum(
                    1 for p in model.multiple_periods if p.get("weight", 0) > 0
                )

            model.assets_to_hold = assets_to_hold.value
            model.trading_frequency = trading_frequency.value
            model.trade_execution = get_trade_execution()
            model.benchmark_ticker = benchmark_ticker.value

            model.absolute_momentum_custom_period = absolute_momentum_custom_period_checkbox.value
            model.absolute_momentum_period = absolute_momentum_period.value

            summary_lines = []
            summary_lines.append("--- Running Backtest ---")
            summary_lines.append(f"Period: {start_picker.children[1].children[0].value}/{start_picker.children[1].children[1].value} - {end_picker.children[1].children[0].value}/{end_picker.children[1].children[1].value}")
            summary_lines.append(f"Tickers: {model.tickers}")
            summary_lines.append(f"Single absolute momentum: {model.single_absolute_momentum}")
            summary_lines.append(f"Absolute momentum asset: {model.absolute_momentum_asset}")
            summary_lines.append(f"Out of market assets: {model.out_of_market_assets}")
            summary_lines.append(f"Performance periods: {model.performance_periods}")
            if model.performance_periods == "Multiple Periods":
                summary_lines.append("Multiple period evaluation:")
                for idx, period in enumerate(model.multiple_periods, start=1):
                    if period["length"] is not None and period["weight"] > 0:
                        summary_lines.append(f"  Period #{idx}: {period['length']} {period['unit']} (Weight: {period['weight']}%)")
            else:
                summary_lines.append(f"Lookback period: {model.lookback_period} {model.lookback_unit}")
                if model.absolute_momentum_custom_period:
                    summary_lines.append(f"Absolute momentum period: {model.absolute_momentum_period} {model.lookback_unit}")

            summary_lines.append(f"Weighting method: {model.weighting_method}")
            summary_lines.append(f"Assets to hold: {model.assets_to_hold}")
            summary_lines.append(f"Trading frequency: {model.trading_frequency}")
            summary_lines.append(f"Trade execution: {model.trade_execution}")
            summary_lines.append(f"Benchmark: {model.benchmark_ticker}")

            user_start = datetime(model.start_year, model.start_month, 1)
            _, last_day = calendar.monthrange(model.end_year, model.end_month)
            user_end = datetime(model.end_year, model.end_month, last_day)

            if model.valid_period_start is not None:
                if model.performance_periods == "Single Period" and model.lookback_unit == "Months":
                    effective_start = model.valid_period_start + relativedelta(months=model.lookback_period)
                elif model.performance_periods == "Multiple Periods":
                    candidates = []
                    for period in model.multiple_periods:
                        if period["length"] is not None and period["weight"] > 0:
                            if period["unit"] == "Months":
                                candidate = model.valid_period_start + relativedelta(months=period["length"])
                            else:
                                candidate = model.valid_period_start + timedelta(days=period["length"])
                            candidates.append(candidate)
                    effective_start = max(candidates) if candidates else model.valid_period_start
                else:
                    effective_start = model.valid_period_start

                if user_start < effective_start:
                    summary_lines.append(f"\nWarning: The user-specified start date {user_start.strftime('%Y-%m-%d')} is")
                    if model.performance_periods == "Single Period":
                        summary_lines.append(f"earlier than required for the lookback period ({model.lookback_period} months).")
                    else:
                        summary_lines.append(f"earlier than required for the longest lookback period.")
                    summary_lines.append(f"Calculations will start from {effective_start.strftime('%Y-%m-%d')}.")
                    user_start = effective_start
                    model.start_year = user_start.year
                    model.start_month = user_start.month

                if model.valid_period_end is not None and user_end > model.valid_period_end:
                    summary_lines.append(f"\nWarning: The user-specified end date {user_end.strftime('%Y-%m-%d')} is")
                    summary_lines.append(f"later than the common data end date {model.valid_period_end.strftime('%Y-%m-%d')}.")
            print("\n".join(summary_lines))
            print("--- Running Backtest ---")

            results = model._run_backtest_next_close(user_start.strftime("%Y-%m-%d"), user_end.strftime("%Y-%m-%d"))

            if results is not None:
                # チェックボックスの状態を取得
                checkboxes = output_options.children[1:]  # 最初のHTML要素をスキップ

                if checkboxes[0].value:  # パフォーマンスグラフ
                    model.plot_performance(display_plot=True)
                if checkboxes[1].value:  # 年次リターンテーブル
                    model.generate_annual_returns_table(display_table=True)
                if checkboxes[2].value:  # 月次リターンテーブル
                    model.generate_monthly_returns_table(display_table=True)
                if checkboxes[3].value:  # モデルシグナル表示
                    model.display_model_signals_dynamic_ui()
                if checkboxes[4].value:  # パフォーマンスサマリー
                    if hasattr(model, 'display_performance_summary_ui'):
                        model.display_performance_summary_ui()
                    else:
                        model.display_performance_summary(display_summary=True)
                if checkboxes[5].value:  # 取引履歴テーブル
                    model.display_trade_history(display_table=True)

                model.excel_sheets_to_export = {
                    "settings": excel_cb_settings.value,
                    "performance": excel_cb_performance.value,
                    "daily_simple": excel_cb_daily_simple.value,
                    "json_config": excel_cb_json_config.value,
                    "monthly_returns": excel_cb_monthly_returns.value,
                    "daily_detailed": excel_cb_daily_detailed.value,
                    "trades": excel_cb_trades.value
                }

                if excel_export_checkbox.value:
                    try:
                        print("\n---エクセルファイルを出力中...---")
                        result = model.export_to_excel(auto_download=False)
                        if result and "filename" in result:
                            print(f"✅ エクセルファイルが正常に出力されました: {result['filename']}")
                            try:
                                from google.colab import files
                                print(f"🔄 ファイルをダウンロードしています...")
                                files.download(result['filename'])
                            except ImportError:
                                pass
                        else:
                            print("❌ エクセルファイルの出力に失敗しました")
                    except Exception as e:
                        print(f"❌ エクセル出力中にエラーが発生しました: {e}")

            else:
                print("❌ Backtest failed. Please check your data period and ticker settings.")


    run_button.on_click(on_run_clicked)
    def update_ui_visibility():
        if performance_periods.value == 'Single Period':
            single_period_settings.layout.display = 'block'
            multiple_periods_settings.layout.display = 'none'
        else:
            single_period_settings.layout.display = 'none'
            multiple_periods_settings.layout.display = 'block'
    performance_periods.observe(lambda change: update_ui_visibility() if change['name'] == 'value' else None, names='value')

    # タブの構成
    time_tab = widgets.VBox([start_picker, end_picker])
    assets_tab = widgets.VBox([
        tickers,
        specify_tickers,
        single_absolute_momentum,
        negative_relative_momentum,
        absolute_momentum_asset,
        specify_absolute_momentum_asset,
        out_of_market_assets,
        specify_out_of_market_asset,
        out_of_market_strategy  # 新しいウィジェットを追加
    ])

    # 単一期間設定はそのまま
    single_period_settings = widgets.VBox([lookback_period, lookback_unit, widgets.HBox([absolute_momentum_custom_period_checkbox]), widgets.HBox([absolute_momentum_period])])

    # 複数期間設定をテーブル形式に変更
    multiple_periods_settings = create_multiple_periods_table(model)
    performance_tab = widgets.VBox([performance_periods, single_period_settings, multiple_periods_settings, assets_to_hold])

    # 取引設定タブ
    trade_tab = widgets.VBox([trading_frequency, trade_execution_label, trade_execution_at_end, trade_execution_at_next, trade_execution_at_next_open, benchmark_ticker])

    # 出力設定タブ - 既存のoutput_optionsを使用
    # output_optionsはすでに定義されています

    # タブウィジェットを作成し、すべてのタブを含めるように設定
    tab = widgets.Tab()
    # これが重要: 5つのタブをすべて明示的に含める
    tab.children = [time_tab, assets_tab, performance_tab, trade_tab, output_options]
    tab.set_title(0, 'Time Period')
    tab.set_title(1, 'Assets')
    tab.set_title(2, 'Performance Period')
    tab.set_title(3, 'Trading Settings')
    tab.set_title(4, 'Output Settings')  # 5つ目のタブには出力設定

    # UIの可視性を更新
    update_ui_visibility()

    # 設定ボタン
    config_buttons = widgets.HBox([save_button, load_button, file_upload])

    # メインレイアウト - output_optionsは含めない
    main_layout = widgets.VBox([
        tab,
        validation_message,
        widgets.HBox([fetch_button, run_button]),
        excel_export_checkbox,
        config_buttons,
        config_textarea,
        portfolio_list_label,
        output
    ])

    display(main_layout)
    with output:
        print("After configuring settings, click 'Fetch Data' to download price data.")
    return model

