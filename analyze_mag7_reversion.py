import pandas as pd
import os
import datetime

# Define MAG7 Symbols
# Define MAG7 Symbols (Excluding GOOGL)
SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA']

def load_data(symbol, data_dir='data'):
    # Look for file pattern
    files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_") and f.endswith('.csv')]
    if not files:
        print(f"Warning: No data found for {symbol}")
        return None
    
    # Heuristic: Pick the one with 'STK' and largest file size or matching recent fetch logic
    # Assuming standard naming from previous steps
    target_file = files[0]
    file_path = os.path.join(data_dir, target_file)
    print(f"Loading {symbol} from {target_file}...")
    
    df = pd.read_csv(file_path)
    # Parse dates with UTC=True
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    
    # Convert to US/Eastern
    try:
        df = df.tz_convert('US/Eastern')
    except TypeError:
        df = df.tz_localize('UTC').tz_convert('US/Eastern')
        
    return df

def get_session_close_prices(df):
    """
    Get the last price of the trading day (approx 16:00 ET).
    """
    rth_data = df.between_time('09:30', '16:00')
    daily_closes = rth_data.resample('D')['close'].last().dropna()
    return daily_closes

def get_prices_at_time(df, target_time_str='15:50'):
    """
    Get the price at a specific time of day for each day.
    """
    # Filter for exact time or nearest bar?
    # using between_time with include_start=True, include_end=True might be loose
    # We want exact bar at 15:50.
    
    # 15:50 bar means the snapshot at 15:50 (could be bar ending 15:50 or starting).
    # Assuming 'bar timestamp' usually denotes start of bar. 
    # If 5 min bars: 15:50 bar covers 15:50-15:55.
    
    # Let's extract rows where time matches
    target_prices = df.at_time(target_time_str)
    
    # We only want the 'close' of that bar? Or open? Close is usually safest 'current price' metric
    return target_prices['close']

def analyze_reversion():
    data_map = {}
    
    # Load all data
    for sym in SYMBOLS:
        df = load_data(sym)
        if df is not None:
            data_map[sym] = df
            
    if not data_map:
        print("No data loaded.")
        return

    # Use one symbol to drive the dates
    ref_symbol = list(data_map.keys())[0]
    ref_df = data_map[ref_symbol]
    
    # We need a unified DataFrame to iterate day by day
    # Let's build a DF containing 'Prev_Close' and 'Price_1550' for each symbol
    
    combined_data = pd.DataFrame(index=ref_df.index.normalize().unique().sort_values())
    combined_data.index.name = 'date'
    
    # Step 1: Populate with Prev Close and 15:50 Price for each symbol
    for sym, df in data_map.items():
        # Daily Closes
        d_closes = get_session_close_prices(df)
        # Prev Closes (Shift 1)
        prev_closes = d_closes.shift(1)
        
        # 15:50 Prices
        prices_1550 = get_prices_at_time(df, '15:50')
        # Reindex to dates (normalize index to join)
        prices_1550_daily = prices_1550.copy()
        prices_1550_daily.index = prices_1550_daily.index.normalize()
        # Handle duplicates if any (shouldn't be for daily normalized if ONE 15:50 bar exists)
        prices_1550_daily = prices_1550_daily[~prices_1550_daily.index.duplicated(keep='first')]
        
        # Merge columns
        # Rename strictly
        col_prev = f"{sym}_prev_close"
        col_curr = f"{sym}_price_1550"
        
        # Join to combined
        combined_data = combined_data.join(prev_closes.rename(col_prev), how='left')
        combined_data = combined_data.join(prices_1550_daily.rename(col_curr), how='left')

    # Drop days where we don't have good data (e.g. weekends/holidays that crept in, or first day)
    combined_data.dropna(how='all', inplace=True)
    
    results = []

    # Step 2: Iterate through days
    for date_idx, row in combined_data.iterrows():
        # Find the worst performing stock for THIS day
        
        worst_perf = 99999.0
        worst_sym = None
        current_price_of_loser = None
        
        # Check each symbol
        valid_day = False
        for sym in SYMBOLS:
            col_prev = f"{sym}_prev_close"
            col_curr = f"{sym}_price_1550"
            
            p_close = row[col_prev]
            curr = row[col_curr]
            
            if pd.notna(p_close) and pd.notna(curr):
                # Compare Today's 15:50 price vs Yesterday's 16:00 Close
                pct_change = (curr - p_close) / p_close
                if pct_change < worst_perf:
                    worst_perf = pct_change
                    worst_sym = sym
                    current_price_of_loser = curr
                valid_day = True
        
        if not valid_day or worst_sym is None:
            continue
            
        # Is it a "Loss"? (perf < 0)
        # Assuming we only care if it dropped.
        if worst_perf >= 0:
            continue
            
        # --- Analyze Outcomes ---
        
        # 1. Next Trading Day Outcome
        # Find price at 15:50 on next available row in combined_data
        
        # We want to calculate return for the "Loser" AND the "Average of All" (Random Stock Benchmark)
        
        # Loser Return
        loser_ret_next = None
        
        # Benchmark Return (Average of all available stocks)
        bench_ret_next = []
        
        try:
            curr_loc = combined_data.index.get_loc(date_idx)
            if curr_loc + 1 < len(combined_data):
                next_date = combined_data.index[curr_loc + 1]
                
                # Check Loser
                next_price_loser = combined_data.loc[next_date, f"{worst_sym}_price_1550"]
                if pd.notna(next_price_loser) and pd.notna(current_price_of_loser):
                     loser_ret_next = (next_price_loser - current_price_of_loser) / current_price_of_loser
                     
                # Check All (Benchmark)
                for s in SYMBOLS:
                    curr_p = row[f"{s}_price_1550"]
                    next_p = combined_data.loc[next_date, f"{s}_price_1550"]
                    if pd.notna(curr_p) and pd.notna(next_p):
                        r = (next_p - curr_p) / curr_p
                        bench_ret_next.append(r)

            else:
                pass # End of data
        except Exception as e:
            # print(f"Next day calc error: {e}")
            pass

        avg_bench_ret_next = sum(bench_ret_next) / len(bench_ret_next) if bench_ret_next else None


        # 2. Next Week (7 Days later) Outcome
        loser_ret_7d = None
        bench_ret_7d = []
        
        target_date_7d = date_idx + pd.Timedelta(days=7)
        
        # Check if this date exists in index
        if target_date_7d in combined_data.index:
            # Loser
            price_7d_loser = combined_data.loc[target_date_7d, f"{worst_sym}_price_1550"]
            if pd.notna(price_7d_loser) and pd.notna(current_price_of_loser):
                loser_ret_7d = (price_7d_loser - current_price_of_loser) / current_price_of_loser
            
            # Benchmark
            for s in SYMBOLS:
                curr_p = row[f"{s}_price_1550"]
                price_7d = combined_data.loc[target_date_7d, f"{s}_price_1550"]
                if pd.notna(curr_p) and pd.notna(price_7d):
                    r = (price_7d - curr_p) / curr_p
                    bench_ret_7d.append(r)
        
        avg_bench_ret_7d = sum(bench_ret_7d) / len(bench_ret_7d) if bench_ret_7d else None

                
        results.append({
            'date': date_idx,
            'loser': worst_sym,
            'drop_pct': worst_perf,
            'return_next_day': loser_ret_next,
            'return_next_week': loser_ret_7d,
            'bench_next_day': avg_bench_ret_next,
            'bench_next_week': avg_bench_ret_7d
        })

    # Summary Statistics
    if not results:
        print("No valid events found.")
        return
        
    res_df = pd.DataFrame(results)
    
    # Add Day of Week (0=Monday, 6=Sunday)
    # Ensure 'date' is datetime
    res_df['date'] = pd.to_datetime(res_df['date'])
    res_df['weekday'] = res_df['date'].dt.day_name()
    res_df['weekday_idx'] = res_df['date'].dt.dayofweek
    
    total_events = len(res_df)
    
    print("-" * 80)
    print(f"Analysis: Buying the biggest MAG6 loser of the day (at 3:50 PM ET)")
    print(f"Comparison: vs Buying an Equal-Weight Portfolio of All (Random Pick Proxy)")
    print(f"Condition: Trade triggers only if worst performer < 0% (vs Prev Close 16:00 ET)")
    print("-" * 120)
    
    # Define Weekday Order
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    print(f"{'Weekday':<12} | {'Count':<5} | {'Next Day Edge':<15} | {'Next Week Edge':<15} | {'Strat 1W Returns':<15}")
    print("-" * 80)
    
    plot_data = {}
    
    for day in days:
        day_data = res_df[res_df['weekday'] == day]
        count = len(day_data)
        
        if count == 0:
            print(f"{day:<12} | {0:<5} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
            continue
            
        # Next Day
        valid_next = day_data.dropna(subset=['return_next_day', 'bench_next_day'])
        if len(valid_next) > 0:
            mean_strat = valid_next['return_next_day'].mean()
            mean_bench = valid_next['bench_next_day'].mean()
            edge_next = mean_strat - mean_bench
        else:
            edge_next = float('nan')
            
        # 3-Sigma Filtering for Next Week Returns
        valid_week = day_data.dropna(subset=['return_next_week', 'bench_next_week']).copy()
        
        if len(valid_week) > 0:
            # Calculate stats for filtering on STRATEGY returns
            mu = valid_week['return_next_week'].mean()
            sigma = valid_week['return_next_week'].std()
            
            # Filter
            mask = (valid_week['return_next_week'] >= mu - 3*sigma) & \
                   (valid_week['return_next_week'] <= mu + 3*sigma)
            
            filtered_week = valid_week[mask]
            
            # Recalculate Edges on filtered data
            mean_strat_w = filtered_week['return_next_week'].mean()
            mean_bench_w = filtered_week['bench_next_week'].mean() # Benchmark on same set of valid trades
            edge_week = mean_strat_w - mean_bench_w
            
            # Formatting for Histogram
            plot_data[day] = {
                'strat': filtered_week['return_next_week'] * 100,
                'bench': filtered_week['bench_next_week'] * 100
            }
        else:
            edge_week = float('nan')
            mean_strat_w = float('nan')

        print(f"{day:<12} | {count:<5} | {edge_next*100:6.2f}%         | {edge_week*100:6.2f}%         | {mean_strat_w*100:6.2f}%")

    print("-" * 80)
    print(f"Overall Total Trade Opportunities: {total_events}")
    
    # Frequency of Losers
    print("-" * 80)
    print("Frequency of Being the Biggest Loser:")
    loser_counts = res_df['loser'].value_counts()
    for sym, freq in loser_counts.items():
        print(f"  {sym:<6}: {freq}")
    print("-" * 80)

    # Edge by Stock
    print("Edge by Individual Stock:")
    print(f"{'Stock':<8} | {'Count':<5} | {'1W Edge Mean':<12} | {'1W Edge Std':<12} | {'Strat 1W Mean':<12} | {'Strat 1W Std':<12}")
    print("-" * 90)
    
    sorted_losers = loser_counts.index.tolist()
    
    # Prepare Plot Data
    stock_plot_data = {}
    
    # Calculate Overall Stats first for comparison
    valid_week_all = res_df.dropna(subset=['return_next_week', 'bench_next_week']).copy()
    if len(valid_week_all) > 0:
        edge_all = valid_week_all['return_next_week'] - valid_week_all['bench_next_week']
        print(f"{'OVERALL':<8} | {len(valid_week_all):<5} | {edge_all.mean()*100:6.2f}%      | {edge_all.std()*100:6.2f}%      | {valid_week_all['return_next_week'].mean()*100:6.2f}%      | {valid_week_all['return_next_week'].std()*100:6.2f}%")
        stock_plot_data['OVERALL'] = {
            'strat': valid_week_all['return_next_week'] * 100,
            'bench': valid_week_all['bench_next_week'] * 100
        }
    print("-" * 90)

    for stock in sorted_losers:
        stock_data = res_df[res_df['loser'] == stock]
        
        valid_week = stock_data.dropna(subset=['return_next_week', 'bench_next_week']).copy()
        
        if len(valid_week) > 0:
            edge_series = valid_week['return_next_week'] - valid_week['bench_next_week']
            mean_edge = edge_series.mean()
            std_edge = edge_series.std()
            mean_strat = valid_week['return_next_week'].mean()
            std_strat = valid_week['return_next_week'].std()
            
            stock_plot_data[stock] = {
                'strat': valid_week['return_next_week'] * 100,
                'bench': valid_week['bench_next_week'] * 100
            }
        else:
            mean_edge = float('nan')
            std_edge = float('nan')
            mean_strat = float('nan')
            std_strat = float('nan')
            
        print(f"{stock:<8} | {len(valid_week):<5} | {mean_edge*100:6.2f}%      | {std_edge*100:6.2f}%      | {mean_strat*100:6.2f}%      | {std_strat*100:6.2f}%")
        
    print("-" * 90)
    
    # Plot Histograms (Overall + Each Stock)
    if stock_plot_data:
        try:
            import matplotlib.pyplot as plt
            import math
            
            # Determine grid size
            num_plots = len(stock_plot_data)
            cols = 4
            rows = math.ceil(num_plots / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
            axes = axes.flatten() # Flatten to 1D array for easy iteration
            
            # Plot Overall first, then stocks
            plot_keys = ['OVERALL'] + sorted_losers
            
            for i, key in enumerate(plot_keys):
                if i >= len(axes): break
                ax = axes[i]
                
                if key in stock_plot_data:
                    data = stock_plot_data[key]
                    ax.hist(data['strat'], bins=20, alpha=0.6, label='Strategy (Loser)', color='blue', edgecolor='k')
                    ax.hist(data['bench'], bins=20, alpha=0.4, label='Benchmark', color='gray', edgecolor='k')
                    
                    stats_text = f"Mean Edge: {(data['strat'].mean() - data['bench'].mean()):.2f}%"
                    ax.set_title(f"{key} (n={len(data['strat'])})\n{stats_text}")
                    ax.legend(fontsize='x-small')
                    ax.grid(True, alpha=0.3)
                    
            # Hide unused axes
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout()
            stock_img = 'results/mag7_stock_histograms.png'
            plt.savefig(stock_img)
            print(f"Stock Histograms saved to {stock_img}")
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Error plotting stocks: {e}")
    
            
        except ImportError:
            pass
        except Exception as e:
            print(f"Error plotting stocks: {e}")
            
    # Breakdown by Stock AND Weekday
    print("-" * 120)
    print("Edge by Stock AND Weekday (Mean ± Std of 1- Week Edge):")
    print(f"{'Stock':<8} | {'Mon':<18} | {'Tue':<18} | {'Wed':<18} | {'Thu':<18} | {'Fri':<18}")
    print("-" * 120)
    
    breakdown_data = [] # For saving to separate CSV
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    for stock in sorted_losers:
        row_str = f"{stock:<8} |"
        stock_breakdown_entry = {'Stock': stock}
        
        for day in weekdays:
            # Filter for this stock AND this day
            subset = res_df[(res_df['loser'] == stock) & (res_df['weekday'] == day)]
            valid_subset = subset.dropna(subset=['return_next_week', 'bench_next_week'])
            
            if len(valid_subset) > 0:
                edge_series = valid_subset['return_next_week'] - valid_subset['bench_next_week']
                edge_mean = edge_series.mean()
                edge_std = edge_series.std()
                count = len(valid_subset)
                
                # Formatted string: "Mean% ±Std% (n)"
                # "2.73% ±6.5% (31)"
                # Space is tight, let's try compact
                cell_str = f"{edge_mean*100:5.2f} ±{edge_std*100:4.1f}% ({count})"
                row_str += f" {cell_str:<18} |"
                
                stock_breakdown_entry[f"{day}_Mean"] = edge_mean
                stock_breakdown_entry[f"{day}_Std"] = edge_std
                stock_breakdown_entry[f"{day}_Count"] = count
            else:
                row_str += f" {'N/A':<18} |"
                stock_breakdown_entry[f"{day}_Mean"] = None
                stock_breakdown_entry[f"{day}_Std"] = None
                stock_breakdown_entry[f"{day}_Count"] = 0
        
        print(row_str)
        breakdown_data.append(stock_breakdown_entry)
        
    print("-" * 120)

    # Save breakdown
    breakdown_df = pd.DataFrame(breakdown_data)
    breakdown_csv = 'results/mag7_edge_breakdown.csv'
    breakdown_df.to_csv(breakdown_csv, index=False)
    print(f"Stock/Weekday breakdown (with StdDev) saved to {breakdown_csv}")

    # Aggregated Analysis: Mon + Tue Combined
    print("-" * 120)
    print("Strategy Check: Buy Biggest Loser on Mon OR Tue (Any Symbol):")
    
    # Filter for Mon or Tue
    mon_tue_df = res_df[res_df['weekday'].isin(['Monday', 'Tuesday'])].dropna(subset=['return_next_week', 'bench_next_week'])
    
    if len(mon_tue_df) > 0:
        edge_series = mon_tue_df['return_next_week'] - mon_tue_df['bench_next_week']
        mean_edge = edge_series.mean()
        std_edge = edge_series.std()
        win_rate = (edge_series > 0).mean()
        
        strat_return = mon_tue_df['return_next_week'].mean()
        
        print(f"Total Trades: {len(mon_tue_df)}")
        print(f"Mean 1-Week Return: {strat_return*100:.2f}%")
        print(f"Mean Edge vs Benchmark: {mean_edge*100:.2f}%")
        print(f"Edge Std Dev: {std_edge*100:.2f}%")
        print(f"Win Rate (Edge > 0): {win_rate*100:.2f}%")
        
        # Check if excluding AAPL/AMZN improves it? (Optional, but user asked regardless of symbol)
        print("\nNote: This aggregate includes the 'bad' performers (AAPL, AMZN, etc.)")
    else:
        print("No valid Mon/Tue trades found.")

    print("-" * 120)



    # Aggregated Analysis: Mon + Tue (Excluding TSLA & NVDA)
    print("Strategy Check: Buy Biggest Loser on Mon OR Tue (Excluding TSLA & NVDA):")
    
    # Filter for Mon or Tue AND NOT TSLA/NVDA
    safe_df = res_df[
        (res_df['weekday'].isin(['Monday', 'Tuesday'])) & 
        (~res_df['loser'].isin(['TSLA', 'NVDA']))
    ].dropna(subset=['return_next_week', 'bench_next_week'])
    
    if len(safe_df) > 0:
        edge_series = safe_df['return_next_week'] - safe_df['bench_next_week']
        mean_edge = edge_series.mean()
        std_edge = edge_series.std()
        win_rate = (edge_series > 0).mean()
        
        strat_return = safe_df['return_next_week'].mean()
        annualized_return = strat_return * 52 # Simple extrapolation (Weekly Mean * 52), valid if 1 trade/week per unit
        
        print(f"Total Trades: {len(safe_df)}")
        print(f"Mean 1-Week Return: {strat_return*100:.2f}%")
        print(f"Mean Edge vs Benchmark: {mean_edge*100:.2f}%")
        print(f"Edge Std Dev: {std_edge*100:.2f}%")
        print(f"Win Rate (Edge > 0): {win_rate*100:.2f}%")
        
        # Estimate Annualized Return
        # Assumption: 2 Capital Units required (1 for Mon, 1 for Tue)
        # However, we might skip weeks if only TSLA/NVDA were the losers.
        # But IF we trade, this is the expected return.
        # Let's verify trade frequency. 
        # 3 years = ~156 weeks.
        # Total Trades = len(safe_df). If len(safe_df) < 2 * 156, we are sitting in cash sometimes.
        # So "Annualized on Deployed Capital" vs "Annualized on Portfolio".
        # Let's print Annualized Return (assuming constant deployment) for simplicity.
        print(f"Simple Annualized Return per Active Capital Unit: {annualized_return*100:.2f}%")

    else:
        print("No valid trades found (after excluding TSLA/NVDA).")

    print("-" * 120)
    
    # Buy & Hold Comparison
    print("Buy & Hold Performance (3-Year Annualized):")
    print(f"{'Stock':<8} | {'Total Return':<15} | {'Annualized (CAGR)':<20}")
    print("-" * 120)
    
    # We need to reload or re-use the data to get start/end prices
    # Since we have data_map in analyze_reversion, we can pass it or reload. 
    # analyze_reversion doesn't expose data_map easily unless we refactor. 
    # It's safer to just reload quickly or assume we can calculate from the files we know exist.
    
    for sym in SYMBOLS:
        try:
            # Reconstruct filename or just use load_data logic
            # Since load_data logic is simple (starts with sym_), we reuse it
            df = load_data(sym)
            if df is not None and not df.empty:
                # Get first and last close
                start_price = df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                
                # Time delta in years
                start_date = df.index[0]
                end_date = df.index[-1]
                days = (end_date - start_date).days
                years = days / 365.25
                
                if years > 0:
                    total_ret = (end_price - start_price) / start_price
                    cagr = (end_price / start_price) ** (1 / years) - 1
                    
                    print(f"{sym:<8} | {total_ret*100:6.2f}%         | {cagr*100:6.2f}%")
        except Exception as e:
            print(f"{sym:<8} | Error: {e}")

    print("-" * 120)
    
    # Year-over-Year Consistency Check
    print("Year-over-Year Consistency (Mon/Tue Strategy):")
    print(f"{'Year':<6} | {'Count':<5} | {'Mean Return':<12} | {'Annualized':<12} | {'Win Rate':<10}")
    print("-" * 120)
    
    # Ensure index is datetime to extract year
    if not pd.api.types.is_datetime64_any_dtype(res_df['date']):
         res_df['date'] = pd.to_datetime(res_df['date'])
         
    years = res_df['date'].dt.year.unique()
    years.sort()
    
    for year in years:
        year_df = res_df[
            (res_df['date'].dt.year == year) &
            (res_df['weekday'].isin(['Monday', 'Tuesday']))
        ].dropna(subset=['return_next_week'])
        
        if len(year_df) > 0:
            mean_ret = year_df['return_next_week'].mean()
            annual_ret = mean_ret * 52
            win_rate = (year_df['return_next_week'] > 0).mean() # Check absolute win rate (return > 0)
            print(f"{year:<6} | {len(year_df):<5} | {mean_ret*100:6.2f}%      | {annual_ret*100:6.2f}%      | {win_rate*100:6.2f}%")
        else:
            print(f"{year:<6} | 0     | N/A          | N/A          | N/A")
            
    print("-" * 120)

    # Save results
    if not os.path.exists('results'):
        os.makedirs('results')
    res_df.to_csv('results/mag7_loser_strategy.csv', index=False)
    print("Detailed logs saved to results/mag7_loser_strategy.csv")

if __name__ == "__main__":
    analyze_reversion()
