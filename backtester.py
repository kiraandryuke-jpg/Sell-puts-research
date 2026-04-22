import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import yaml

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def black_scholes_put(S, K, T, r, sigma):
    # Avoid division by zero if sigma is 0
    if sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def run_backtest():
    # 1. Get ALL data once
    all_tickers = config['tickers'] + [config['market_indicator']]
    print(f"Downloading data for: {all_tickers}")
    df_data = yf.download(all_tickers, period="2y")
    
    # Check if data loaded correctly
    if df_data.empty:
        raise ValueError("No data downloaded. Check your ticker symbols.")

    # Flatten column structure if yfinance returns multi-index
    if isinstance(df_data.columns, pd.MultiIndex):
        df_data.columns = df_data.columns.get_level_values(1)

    # 2. Setup tracking
    active_positions = [] 
    results = []
    
    # 3. Main Loop
    # Iterate through the dates where we have SPY data
    spy_data = df_data[[config['market_indicator']]]
    
    for i in range(config['strategy']['volatility_window'], len(df_data)):
        current_date = df_data.index[i]
        
        # Get market data from pre-downloaded dataframe
        spy_open = df_data[config['market_indicator']]['Open'].iloc[i]
        spy_close = df_data[config['market_indicator']]['Close'].iloc[i]
        
        # Rule: SPY Red Day
        if spy_close < spy_open:
            for ticker in config['tickers']:
                S = df_data[ticker]['Close'].iloc[i]
                K = S * config['strategy']['strike_offset']
                T = config['strategy']['dte'] / 252 
                
                # Calculate rolling vol
                sigma = df_data[ticker]['Close'].iloc[i-20:i].pct_change().std() * np.sqrt(252)
                
                premium = black_scholes_put(S, K, T, config['strategy']['risk_free_rate'], sigma)
                
                active_positions.append({
                    'ticker': ticker,
                    'entry_price': premium,
                    'entry_date': current_date,
                    'exit_target': premium * config['strategy']['profit_target_pct']
                })
        
        # 4. Update/Exit Check
        for pos in active_positions[:]: 
            S_curr = df_data[pos['ticker']]['Close'].iloc[i]
            # Use same sigma for simplicity
            current_val = black_scholes_put(S_curr, K, T, config['strategy']['risk_free_rate'], sigma)
            
            if current_val <= pos['exit_target']:
                results.append({'ticker': pos['ticker'], 'profit': pos['entry_price'] - current_val})
                active_positions.remove(pos)

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_backtest()
    print(f"Total trades completed: {len(df)}")
    if not df.empty:
        print(f"Total PnL: {df['profit'].sum():.2f}")
        
