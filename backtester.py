import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import yaml

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def black_scholes_put(S, K, T, r, sigma):
    """Calculates theoretical price of a European Put."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def run_backtest():
    # 1. Get Data
    all_tickers = config['tickers'] + [config['market_indicator']]
    data = yf.download(all_tickers, period="2y")['Close']
    
    # 2. Setup tracking
    active_positions = [] # List of dicts: {'ticker':..., 'entry_premium':..., 'expiry_date':...}
    results = []
    
    # 3. Main Loop
    for i in range(config['strategy']['volatility_window'], len(data)):
        current_date = data.index[i]
        spy_open = yf.download(config['market_indicator'], start=current_date, end=current_date, progress=False)['Open'].values[0]
        spy_close = data[config['market_indicator']].iloc[i]
        
        # Rule: SPY Red Day
        if spy_close < spy_open:
            for ticker in config['tickers']:
                S = data[ticker].iloc[i]
                K = S * config['strategy']['strike_offset']
                T = config['strategy']['dte'] / 252 # Time in years
                
                # Calculate realized vol as proxy for sigma
                sigma = data[ticker].iloc[i-20:i].pct_change().std() * np.sqrt(252)
                
                premium = black_scholes_put(S, K, T, config['strategy']['risk_free_rate'], sigma)
                
                active_positions.append({
                    'ticker': ticker,
                    'entry_price': premium,
                    'entry_date': current_date,
                    'exit_target': premium * config['strategy']['profit_target_pct']
                })
        
        # 4. Update/Exit Check
        for pos in active_positions[:]: # Copy list to iterate safely
            # Theoretical price update
            S_curr = data[pos['ticker']].iloc[i]
            # (Simplification: using same T and sigma for daily update)
            current_val = black_scholes_put(S_curr, K, T, config['strategy']['risk_free_rate'], sigma)
            
            if current_val <= pos['exit_target']:
                results.append({'ticker': pos['ticker'], 'profit': pos['entry_price'] - current_val})
                active_positions.remove(pos)

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_backtest()
    print(f"Total trades completed: {len(df)}")
    print(f"Total PnL approximation: {df['profit'].sum():.2f}")
