class InputValidator:
    @staticmethod
    def validate_lookback_period(value, unit):
        if unit == "Days":
            if value < 15 or value > 90:
                return False, f"Days の有効範囲は 15-90 です。入力値 {value} は範囲外です。"
        elif unit == "Months":
            if value < 1 or value > 36:
                return False, f"Months の有効範囲は 1-36 です。入力値 {value} は範囲外です。"
        return True, ""

    @staticmethod
    def validate_weights(weights):
        valid_weights = [w for w in weights if w is not None and w > 0]
        if not valid_weights:
            return False, "有効な重みがありません。少なくとも1つの期間に正の重みを設定してください。"
        total_weight = sum(valid_weights)
        if abs(total_weight - 100) > 0.1:
            return False, f"重みの合計が100%ではありません。現在の合計: {total_weight:.2f}%"
        return True, ""

    @staticmethod
    def validate_ticker_symbols(tickers):
        if not tickers:
            return False, "少なくとも1つのティッカーシンボルを指定してください。"
        invalid_tickers = []
        for ticker in tickers:
            if not ticker or not ticker.strip() or any(c in ticker for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
                invalid_tickers.append(ticker)
        if invalid_tickers:
            return False, f"無効なティッカーシンボル: {', '.join(invalid_tickers)}"
        return True, ""

    @staticmethod
    def validate_date_range(start_year, start_month, end_year, end_month):
        if start_month < 1 or start_month > 12:
            return False, f"開始月が無効です: {start_month}。1-12の範囲で指定してください。"
        if end_month < 1 or end_month > 12:
            return False, f"終了月が無効です: {end_month}。1-12の範囲で指定してください。"
        if start_year < 1990:
            return False, f"開始年が無効です: {start_year}。1990年以降を指定してください。"
        if end_year < start_year or (end_year == start_year and end_month < start_month):
            return False, f"終了日（{end_year}/{end_month}）は開始日（{start_year}/{start_month}）より後でなければなりません。"
        return True, ""

    @staticmethod
    def validate_benchmark_ticker(ticker):
        if not ticker or not ticker.strip():
            return False, "ベンチマークティッカーを指定してください。"
        if any(c in ticker for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
            return False, f"無効なベンチマークティッカー: {ticker}"
        return True, ""

    @staticmethod
    def validate_absolute_momentum_asset(ticker):
        if not ticker or not ticker.strip():
            return False, "絶対モメンタム資産を指定してください。"
        if any(c in ticker for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
            return False, f"無効な絶対モメンタム資産: {ticker}"
        return True, ""

    @staticmethod
    def validate_out_of_market_assets(assets):
        if not assets:
            return False, "少なくとも1つの退避先資産を指定してください。"
        invalid_assets = []
        for asset in assets:
            if not asset or not asset.strip() or any(c in asset for c in " !@#$%&*()+={}[]|\\/;:'\",<>?"):
                invalid_assets.append(asset)
        if invalid_assets:
            return False, f"無効な退避先資産: {', '.join(invalid_assets)}"
        return True, ""

