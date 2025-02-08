import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

# 1. Nasdaq-100-Ticker von Wikipedia abrufen
def get_nasdaq100_tickers():
    """
    Liest die Nasdaq‑100-Ticker von Wikipedia.
    Wir nehmen an, dass die Tabelle mit Index 4 die Spalte "Symbol" enthält.
    """
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    nasdaq_table = tables[4]  # Annahme: Tabelle 4 enthält die Nasdaq-100-Komponenten
    tickers = nasdaq_table["Symbol"].tolist()
    return tickers

# 2. Daten über Alphavantage herunterladen
def download_data_alphavantage(tickers, start_date, end_date, pause=12):
    """
    Lädt wöchentliche Daten über Alphavantage für die angegebenen Ticker.
    Verwendet die Funktion get_weekly_adjusted, um den "adjusted close" abzurufen.
    Zwischen den Requests wird eine Pause eingelegt, um API-Limits zu berücksichtigen.
    """
    api_key = "BU79RQUIGTPHCH8W"
    ts = TimeSeries(key=api_key, output_format='pandas')
    data_list = []
    for ticker in tickers:
        print(f"Lade Daten für: {ticker}")
        try:
            # Abruf der wöchentlichen adjustierten Daten
            data, meta_data = ts.get_weekly_adjusted(symbol=ticker)
            data.sort_index(inplace=True)
            # Konvertiere den Index in datetime
            data.index = pd.to_datetime(data.index)
            # Filtere den Zeitraum
            data = data.loc[start_date:end_date]
            # Verwende die Spalte "5. adjusted close"
            if '5. adjusted close' in data.columns:
                data = data[['5. adjusted close']].rename(columns={'5. adjusted close': ticker})
            else:
                raise KeyError(f"Ticker {ticker}: '5. adjusted close' nicht gefunden.")
            data_list.append(data)
        except Exception as e:
            print(f"Fehler beim Laden von {ticker}: {e}")
        time.sleep(pause)
    if not data_list:
        raise ValueError("Keine Daten konnten heruntergeladen werden.")
    df = pd.concat(data_list, axis=1)
    df.sort_index(inplace=True)
    return df

# 3. RSL (Relative Stärke nach Levy) berechnen
def compute_rsl(prices, lookback=26):
    """
    Berechnet den RSL-Wert:
      RSL = aktueller Preis / (Gleitender Durchschnitt der letzten 'lookback'-Perioden)
    """
    rolling_mean = prices.rolling(window=lookback, min_periods=lookback).mean()
    rsl = prices / rolling_mean
    return rsl

# 4. Backtest der RSL-Strategie
def backtest_rsl_strategy(start_date, end_date, lookback=26):
    """
    Führt den Backtest der RSL-Strategie auf Wochenbasis durch:
      - Es werden die Nasdaq‑100-Komponenten (von Wikipedia) und drei Benchmarks abgerufen.
      - Für jeden Ticker werden wöchentliche Daten (adjusted close) von Alphavantage geladen.
      - Für die Nasdaq‑100-Aktien wird der RSL-Wert über einen Lookback von 'lookback' Wochen berechnet.
      - An jedem Rebalancing-Tag (nach dem Lookback) werden die 10 Aktien mit dem höchsten RSL ausgewählt.
      - Das Portfolio (gleichgewichtet) wird wöchentlich neu berechnet.
      - Parallel dazu wird die Performance der Benchmarks (Nasdaq Index, MSCI World, MSCI World Momentum) ermittelt.
    """
    # Ticker definieren
    nasdaq_stocks = get_nasdaq100_tickers()
    benchmark_tickers = {
        'Nasdaq Index': '^NDX',  # Hinweis: Alphavantage liefert möglicherweise keine Indexdaten für "^NDX".
                                # Falls Probleme auftreten, könntest du alternativ einen ETF wie "QQQ" verwenden.
        'MSCI World': 'URTH',
        'MSCI World Momentum': 'IWMO'
    }
    
    # Gesamtliste: Nasdaq-Stocks + Benchmarks
    all_tickers = nasdaq_stocks + list(benchmark_tickers.values())
    print("Lade historische Daten über Alphavantage für alle Ticker...")
    data = download_data_alphavantage(all_tickers, start_date, end_date, pause=12)
    
    data.sort_index(inplace=True)
    data.dropna(how='all', inplace=True)
    
    # Berechne wöchentliche Renditen
    returns = data.pct_change()
    
    # RSL nur für die Nasdaq-Stocks berechnen
    nasdaq_data = data[nasdaq_stocks]
    rsl = compute_rsl(nasdaq_data, lookback=lookback)
    
    portfolio_returns = []
    portfolio_dates = []
    
    # Backtest-Schleife: Beginn ab dem Zeitpunkt, an dem 'lookback'-Wochen vorhanden sind
    for t in range(lookback, len(data) - 1):
        current_date = data.index[t]
        next_date = data.index[t + 1]
        
        # RSL-Werte der Nasdaq-Stocks zum aktuellen Datum
        current_rsl = rsl.loc[current_date].dropna()
        if current_rsl.empty:
            continue
        
        # Auswahl der 10 Aktien mit dem höchsten RSL
        top10 = current_rsl.sort_values(ascending=False).head(10)
        selected_stocks = top10.index.tolist()
        print(f"{current_date.date()}: Top 10 Aktien: {selected_stocks}")
        
        # Gleichgewichtete Gewichtung
        weight = 1 / len(selected_stocks)
        
        # Berechnung der wöchentlichen Portfolio-Rendite
        ret = returns.loc[next_date, selected_stocks].fillna(0)
        port_return = np.dot(np.repeat(weight, len(selected_stocks)), ret)
        portfolio_returns.append(port_return)
        portfolio_dates.append(next_date)
    
    portfolio_df = pd.DataFrame({'Portfolio Return': portfolio_returns}, index=portfolio_dates)
    portfolio_df['Portfolio Cumulative'] = (1 + portfolio_df['Portfolio Return']).cumprod()
    
    # Benchmark-Performance berechnen
    benchmark_results = {}
    for name, ticker in benchmark_tickers.items():
        if ticker in returns.columns:
            bm_returns = returns[ticker].loc[portfolio_df.index].fillna(0)
            bm_cumulative = (1 + bm_returns).cumprod()
            benchmark_results[name] = bm_cumulative
        else:
            print(f"Warnung: Für Benchmark {name} ({ticker}) sind keine Daten vorhanden.")
    
    benchmarks_df = pd.DataFrame(benchmark_results)
    
    result_df = pd.concat([portfolio_df, benchmarks_df], axis=1)
    return result_df

# 5. Hauptprogramm
if __name__ == "__main__":
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    lookback_weeks = 26  # ca. 6 Monate Lookback
    
    result_df = backtest_rsl_strategy(start_date, end_date, lookback=lookback_weeks)
    
    print("Backtest-Ergebnisse (letzte Zeilen):")
    print(result_df.tail())
    
    print("\nFinaler kumulativer Portfolio Return:", result_df['Portfolio Cumulative'].iloc[-1])
    for benchmark in ['Nasdaq Index', 'MSCI World', 'MSCI World Momentum']:
        if benchmark in result_df.columns:
            print(f"Finaler kumulativer {benchmark} Return:", result_df[benchmark].iloc[-1])
    
    # Plot der kumulativen Performance
    plt.figure(figsize=(12, 6))
    plt.plot(result_df.index, result_df['Portfolio Cumulative'], label='Portfolio (Top 10 Nasdaq Stocks)')
    for benchmark in ['Nasdaq Index', 'MSCI World', 'MSCI World Momentum']:
        if benchmark in result_df.columns:
            plt.plot(result_df.index, result_df[benchmark], label=benchmark)
    plt.xlabel('Datum')
    plt.ylabel('Kumulative Performance')
    plt.title('Backtest der RSL-Strategie vs. Benchmarks (Alphavantage-Daten)')
    plt.legend()
    plt.grid(True)
    plt.show()
