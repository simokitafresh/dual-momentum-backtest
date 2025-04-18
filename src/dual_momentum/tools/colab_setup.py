# dual_momentum/tools/colab_setup.py
def setup_for_colab():
    """
    Google Colab環境でデュアルモメンタムUIを使用するための設定を行います。
    """
    import sys
    
    # 必要なライブラリをインストール
    try:
        import IPython
        IPython.get_ipython().system('pip install -q numpy pandas matplotlib seaborn ipywidgets yfinance pandas_market_calendars tqdm openpyxl')
    except:
        print("自動インストールに失敗しました。手動で必要なパッケージをインストールしてください。")
    
    # Colabでのウィジェット対応
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
        print("✅ Google Colabウィジェットマネージャーを有効化しました")
    except ImportError:
        print("⚠️ Google Colab環境ではありません")
    except Exception as e:
        print(f"⚠️ ウィジェットマネージャーの有効化に失敗しました: {e}")
    
    print("✅ デュアルモメンタムパッケージのColab設定が完了しました")
    print("以下のコードでUIを起動できます:")
    print("from dual_momentum import create_dual_momentum_ui")
    print("model = create_dual_momentum_ui()")
