def create_year_month_picker(year_value, month_value, description):
    """年と月を選択するカスタムウィジェットを作成"""
    today = datetime.now()
    years = list(range(1990, today.year + 1))
    months = list(range(1, 13))

    year_dropdown = widgets.Dropdown(
        options=years,
        value=year_value,
        description='Year:',
        style={'description_width': 'initial'}
    )

    month_dropdown = widgets.Dropdown(
        options=months,
        value=month_value,
        description='Month:',
        style={'description_width': 'initial'}
    )

    label = widgets.HTML(value=f"<b>{description}</b>")
    return widgets.VBox([label, widgets.HBox([year_dropdown, month_dropdown])])

def create_multiple_periods_table(model):
    """複数期間設定をテーブル形式で表示するウィジェットを作成"""
    # テーブルのスタイル定義 (変更なし)
    table_style = """
    <style>
    .periods-table {
        border-collapse: collapse;
        width: 100%;
    }
    .periods-table th, .periods-table td {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    </style>
    """

    # テーブルヘッダー (変更なし)
    table_header = """
    <div style="overflow-x: auto;">
      <table class="periods-table">
        <thead>
          <tr>
            <th>Period</th>
            <th>Length</th>
            <th>Unit</th>
            <th>Weights (%)</th>
          </tr>
        </thead>
      </table>
    </div>
    """

    header_html = widgets.HTML(value=table_style + table_header)

    # 行ウィジェットを作成 (変更なし)
    rows = []
    periods_count = min(5, len(model.multiple_periods))

    for i in range(periods_count):
        # 各期間の現在の設定値を取得
        period = model.multiple_periods[i] if i < len(model.multiple_periods) else {"length": 3, "unit": "Months", "weight": 0}
        length_val = period.get("length", 3)
        unit_val = period.get("unit", "Months")
        weight_val = period.get("weight", 0)

        # 期間番号
        period_num = widgets.HTML(value=f"#{i+1}")

        # 期間長（ドロップダウン）
        length = widgets.Dropdown(
            options=[1,2,3,4,5,6,7,8,9,10,11,12,15,18,24,30,36],
            value=length_val,
            layout=widgets.Layout(width='100px')
        )

        # 単位（ドロップダウン）
        unit = widgets.Dropdown(
            options=['Months', 'Days'],
            value=unit_val,
            layout=widgets.Layout(width='100px')
        )

        # 重み（数値入力）
        weight = widgets.IntText(
            value=weight_val,
            min=0,
            max=100,
            step=5,
            layout=widgets.Layout(width='80px')
        )

        # 値変更時のコールバック (変更なし)
        def create_callback(idx, length_w, unit_w, weight_w):
            def callback(change):
                if idx >= len(model.multiple_periods):
                    # 配列の拡張が必要な場合
                    while len(model.multiple_periods) <= idx:
                        model.multiple_periods.append({"length": None, "unit": None, "weight": 0})
                model.multiple_periods[idx] = {
                    "length": length_w.value,
                    "unit": unit_w.value,
                    "weight": weight_w.value
                }
            return callback

        callback_fn = create_callback(i, length, unit, weight)
        length.observe(callback_fn, names='value')
        unit.observe(callback_fn, names='value')
        weight.observe(callback_fn, names='value')

        # 行を作成
        row = widgets.HBox(
            [period_num, length, unit, weight],
            layout=widgets.Layout(
                border_bottom='1px solid #ddd',
                padding='8px',
                align_items='center'
            )
        )
        rows.append(row)

    # その他の設定項目
    weighting_method_label = widgets.HTML(value="<div style='margin-top: 20px'><b>Period Weighting:</b></div>")
    weighting_method = widgets.Dropdown(
        options=['Weight Performance', 'Weight Rank Orders'],
        value=model.weighting_method,
        layout=widgets.Layout(width='200px')
    )

    def update_weighting_method(change):
        model.weighting_method = change['new']

    weighting_method.observe(update_weighting_method, names='value')

    # すべてを組み合わせる
    return widgets.VBox(
        [header_html] + rows + [
            weighting_method_label,
            weighting_method
        ]
    )

