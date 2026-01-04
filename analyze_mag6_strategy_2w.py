import pandas as pd
import os
import datetime

# Define MAG6 Symbols (Mag7 minus GOOG/GOOGL)
SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA']

def load_data(symbol, data_dir='data'):
    files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_") and f.endswith('.csv')]
    if not files:
        print(f"Warning: No data found for {symbol}")
        return None
    
    # Pick the first matching file (usually best match from previous steps)
    target_file = files[0]
    file_path = os.path.join(data_dir, target_file)
    # print(f"Loading {symbol} from {target_file}...")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    
    try:
        df = df.tz_convert('US/Eastern')
    except TypeError:
        df = df.tz_localize('UTC').tz_convert('US/Eastern')
        
    return df

def get_prices_at_time(df, target_time_str='15:50'):
    # Get closing price of the 15:50 bar (or nearest)
    return df.at_time(target_time_str)['close']

def get_session_close_prices(df):
    # Get close of daily session (approx 16:00)
    rth_data = df.between_time('09:30', '16:00')
    daily_closes = rth_data.resample('D')['close'].last().dropna()
    return daily_closes


def find_future_date_idx(current_date, days_delta, available_dates):
    """
    Finds the first available trading date >= current_date + days_delta.
    """
    target_date = current_date + pd.DateOffset(days=days_delta)
    
    # Use searchsorted to find the index of the first date >= target_date
    idx = available_dates.searchsorted(target_date)
    
    if idx < len(available_dates):
        return available_dates[idx]
    return None

def analyze_mag6_2week():
    data_map = {}
    for sym in SYMBOLS:
        df = load_data(sym)
        if df is not None:
            data_map[sym] = df
            
    if not data_map:
        print("No data loaded.")
        return

    ref_symbol = list(data_map.keys())[0]
    ref_df = data_map[ref_symbol]
    
    # Unified DataFrame
    combined_data = pd.DataFrame(index=ref_df.index.normalize().unique().sort_values())
    combined_data.index.name = 'date'
    
    # Populate Data
    for sym, df in data_map.items():
        d_closes = get_session_close_prices(df)
        prev_closes = d_closes.shift(1)
        prices_1550 = get_prices_at_time(df, '15:50')
        prices_1550_daily = prices_1550.copy()
        prices_1550_daily.index = prices_1550_daily.index.normalize()
        prices_1550_daily = prices_1550_daily[~prices_1550_daily.index.duplicated(keep='first')]
        
        combined_data = combined_data.join(prev_closes.rename(f"{sym}_prev_close"), how='left')
        combined_data = combined_data.join(prices_1550_daily.rename(f"{sym}_price_1550"), how='left')

    combined_data.dropna(how='all', inplace=True)
    all_dates = combined_data.index
    
    results = []

    for date_idx, row in combined_data.iterrows():
        # Identify Worst Performer
        worst_perf = 99999.0
        worst_sym = None
        current_price = None
        
        valid_day = False
        for sym in SYMBOLS:
            p_close = row[f"{sym}_prev_close"]
            curr = row[f"{sym}_price_1550"]
            
            if pd.notna(p_close) and pd.notna(curr):
                pct_change = (curr - p_close) / p_close
                if pct_change < worst_perf:
                    worst_perf = pct_change
                    worst_sym = sym
                    current_price = curr
                valid_day = True
        
        # Filter: Must be a loss (< 0) - REMOVED per user instruction
        if not valid_day or worst_sym is None:
            continue
            
        # Calculate Returns: 2 Weeks (14 Days)
        ret_2w = None
        
        # Find sell date (>= 14 days later)
        sell_date = find_future_date_idx(date_idx, 14, all_dates)
        
        if sell_date is not None:
            sell_price = combined_data.loc[sell_date, f"{worst_sym}_price_1550"]
            if pd.notna(sell_price) and pd.notna(current_price):
                ret_2w = (sell_price - current_price) / current_price
                
        results.append({
            'date': date_idx,
            'weekday': date_idx.day_name(),
            'loser': worst_sym,
            'drop_pct': worst_perf,
            'return_2week': ret_2w,
            'sell_date': sell_date
        })
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No trades found.")
        return

    # Filter for Monday and Tuesday
    mon_tue_df = res_df[res_df['weekday'].isin(['Monday', 'Tuesday'])].copy()
    mon_tue_df = mon_tue_df.dropna(subset=['return_2week'])
    
    print("-" * 80)
    print("Strategy: Buy Mag6 Lowest Return Stock on Monday & Tuesday (Always Buy)")
    print("Hold Period: 2 Weeks (Sold at 15:50 on 10th trading day approx)")
    print("-" * 80)
    
    if len(mon_tue_df) > 0:
        mean_ret = mon_tue_df['return_2week'].mean()
        std_ret = mon_tue_df['return_2week'].std()
        win_rate = (mon_tue_df['return_2week'] > 0).mean()
        
        # Annualized Return Calculation (Theoretical)
        # Trades happen 2 times a week = 104 trades/year.
        # Hold is 2 weeks = 4 capital slots.
        # Annual Return = Mean * (104/4) = Mean * 26
        
        annual_ret = mean_ret * 26
        
        print(f"Total Trades: {len(mon_tue_df)}")
        print(f"Mean 2-Week Return: {mean_ret*100:.2f}%")
        print(f"Standard Deviation: {std_ret*100:.2f}%")
        print(f"Win Rate:             {win_rate*100:.2f}%")
        print("-" * 40)
        print(f"Theoretical Annual Return Ratio (Mean * 26): {annual_ret*100:.2f}%")
        print("-" * 80)
        
        # Breakdown by Day
        print("Breakdown by Entry Day:")
        for day in ['Monday', 'Tuesday']:
            subset = mon_tue_df[mon_tue_df['weekday'] == day]
            if not subset.empty:
                m = subset['return_2week'].mean()
                c = len(subset)
                print(f"{day:<10}: {m*100:6.2f}% (n={c})")
    
    else:
        print("No valid Mon/Tue trades found with 2-week history.")

if __name__ == "__main__":
    analyze_mag6_2week()
