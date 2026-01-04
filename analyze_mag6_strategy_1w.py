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
    # Use DateOffset to ensure we land on the same wall-clock time (midnight),
    # avoiding DST shifts that push Timedelta into 1AM or 11PM.
    target_date = current_date + pd.DateOffset(days=days_delta)
    
    # Use searchsorted to find the index of the first date >= target_date
    idx = available_dates.searchsorted(target_date)
    
    if idx < len(available_dates):
        return available_dates[idx]
    return None

def analyze_mag6_1week():
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
        
        # Filter: Must be a loss (< 0)

        # Filter: Must be a loss (< 0) - REMOVED per user instruction
        if not valid_day or worst_sym is None:
            continue
            
        # Calculate Returns: 1 Week (7 Days)
        ret_1w = None
        
        # Find sell date (>= 7 days later)
        sell_date = find_future_date_idx(date_idx, 7, all_dates)
        
        if sell_date is not None:
            sell_price = combined_data.loc[sell_date, f"{worst_sym}_price_1550"]
            if pd.notna(sell_price) and pd.notna(current_price):
                ret_1w = (sell_price - current_price) / current_price
                
        results.append({
            'date': date_idx,
            'weekday': date_idx.day_name(),
            'loser': worst_sym,
            'drop_pct': worst_perf,
            'return_1week': ret_1w,
            'sell_date': sell_date
        })
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No trades found.")
        return

    # Filter for Monday and Tuesday
    mon_tue_df = res_df[res_df['weekday'].isin(['Monday', 'Tuesday'])].copy()
    mon_tue_df = mon_tue_df.dropna(subset=['return_1week'])
    
    print("-" * 80)
    print("Strategy: Buy Mag6 Lowest Return Stock on Monday & Tuesday (Always Buy)")
    print("Hold Period: 1 Week (Sold at 15:50 on 5th trading day approx)")
    print("-" * 80)
    
    if len(mon_tue_df) > 0:
        mean_ret = mon_tue_df['return_1week'].mean()
        std_ret = mon_tue_df['return_1week'].std()
        win_rate = (mon_tue_df['return_1week'] > 0).mean()
        
        # Theoretical 1-Week Annualized 
        # Mean * 52
        theoretical_annual_ret = mean_ret * 52
        
        print(f"Total Trades: {len(mon_tue_df)}")
        print(f"Mean 1-Week Return: {mean_ret*100:.2f}%")
        print(f"Standard Deviation: {std_ret*100:.2f}%")
        print(f"Win Rate:             {win_rate*100:.2f}%")

        print("-" * 40)
        print(f"Theoretical Annual Return Ratio (Mean * 52): {theoretical_annual_ret*100:.2f}%")
        print("-" * 80)
        
        # --- Consistency Analysis ---
        mon_tue_df['year'] = mon_tue_df['date'].dt.year
        years = mon_tue_df['year'].unique()
        years.sort()
        
        print(f"\nYear-by-Year Consistency:")
        print(f"{'Year':<6} | {'Count':<5} | {'Mean Return':<12} | {'Win Rate':<10} | {'Sum of Returns':<15}")
        print("-" * 70)
        
        for year in years:
            y_df = mon_tue_df[mon_tue_df['year'] == year]
            count = len(y_df)
            mean_r = y_df['return_1week'].mean()
            win_r = (y_df['return_1week'] > 0).mean()
            sum_r = y_df['return_1week'].sum()
            
            # Theoretical Annual for this specific year (extrapolated or just sum?)
            # The user asked about "contribution". Sum of returns is a good proxy for total contribution.
            print(f"{year:<6} | {count:<5} | {mean_r*100:6.2f}%      | {win_r*100:6.2f}%    | {sum_r*100:6.2f}%")
            
        print("-" * 70)
        
        # Quarter/Month Checks for "lumpy" returns
        mon_tue_df['month_yr'] = mon_tue_df['date'].dt.to_period('M')
        monthly_returns = mon_tue_df.groupby('month_yr')['return_1week'].sum()
        
        best_month = monthly_returns.idxmax()
        best_month_val = monthly_returns.max()
        worst_month = monthly_returns.idxmin()
        worst_month_val = monthly_returns.min()
        
        print(f"\nDistribution of Monthly Returns (Sum of %):")
        print(f"Best Month:  {best_month} ({best_month_val*100:.2f}%)")
        print(f"Worst Month: {worst_month} ({worst_month_val*100:.2f}%)")
        print(f"Positive Months: {(monthly_returns > 0).sum()} / {len(monthly_returns)} ({((monthly_returns > 0).mean()*100):.1f}%)")

        # Rolling Average to check for dead zones
        # 3-Month Rolling Average of Mean Return per trade
        # Resample to monthly means first
        monthly_means = mon_tue_df.set_index('date').resample('M')['return_1week'].mean()
        rolling_3m = monthly_means.rolling(3).mean()
        
        print("\nRolling 3-Month Average Return per Trade (Trends):")
        # Identify periods where rolling avg was negative
        neg_periods = rolling_3m[rolling_3m < 0]
        if not neg_periods.empty:
            print(f"  Periods with negative expectancy (3-month rolling): {len(neg_periods)} months")
            # print(neg_periods.head())
        else:
            print("  No 3-month periods with negative expectancy.")
            
    else:
        print("No valid Mon/Tue trades found.")

if __name__ == "__main__":
    analyze_mag6_1week()
