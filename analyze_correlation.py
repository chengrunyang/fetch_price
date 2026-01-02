import pandas as pd
import numpy as np
import os
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_data(symbol_pattern, data_dir='data'):
    files = [f for f in os.listdir(data_dir) if symbol_pattern in f and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No file found for {symbol_pattern} in {data_dir}")
    # Pick the first match (or the most recent/specific one if needed)
    file_path = os.path.join(data_dir, files[0])
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    # Parse with utc=True to handle potential mixed offsets (DST)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    
    # Check if we need to convert to Eastern here or later. 
    # It's safer to work in Eastern for the whole script since we care about 16:00 ET.
    try:
        df = df.tz_convert('US/Eastern')
    except TypeError:
        # If naive (shouldn't be with utc=True), localize
        df = df.tz_localize('UTC').tz_convert('US/Eastern')
        
    return df

def get_market_closes(df):
    """
    Extracts the closing price for each day at 16:00 ET.
    Assumes index is timezone aware or in ET.
    """
    # Filter for RTH close approx (16:00 ET)
    # Since we have 5 min bars, we look for the bar at 15:55 or 16:00.
    # A safer way to find 'Market Close' regardless of exact timestamps is to 
    # take the last bar of the regular session for each day.
    
    # We can filter for times between 09:30 and 16:00
    # Then take the last one for each day.
    
    # Check timezone of index
    if df.index.tz is None:
        # Assuming UTC or ET? The fetch script likely saved them with offsets.
        pass
    
    # Convert to Eastern for RTH logic if needed, but if the data has offsets, 
    # we just need to know what 16:00 looks like.
    # Usually easier to just resample to Day and take the last price of the "Session".
    # But since we have extended hours, 'last' would be 20:00.
    
    # Let's filter strictly for <= 16:00
    # Extract entries where time is <= 16:00
    # actually, simple comparison on time component requires conversion to local/ET
    
    # Let's assume the data is properly localized.
    # df_rth = df.between_time('09:30', '16:00') # This works if index is DatetimeIndex
    
    # But if timezone is mixed or UTC, this might be tricky.
    # Let's inspect one date first? No, let's write robust code.
    
    # We'll just take the close of the last bar before or at 16:00 for each day.
    # Create a column for 'Date' (day only)
    df['day_date'] = df.index.date
    
    # Filter for rows where time <= 16:00:00 (approx)
    # We need to be careful with timezones. 
    # If the CSV saved as string with offset, pandas converts it to "mixed" or awareness.
    # Let's convert to 'US/Eastern' to be safe for RTH filtering.
    
    try:
        df_et = df.tz_convert('US/Eastern')
    except TypeError:
        # If it's already naive (unlikely if fetched from IB), localize?
        # If it's naive, assume it's ET based on script defaults.
        df_et = df.tz_localize('US/Eastern')
        
    # Filter: Time <= 16:00
    rth_data = df_et.between_time('09:30', '16:00')
    
    # Group by date and take the last close
    # Using resample to get DatetimeIndex for easier period manipulation
    # Resample 'D' produces daily bins.
    daily_closes = rth_data.resample('D')['close'].last().dropna()
    
    return daily_closes

def analyze():
    # Load TQQQ
    tqqq = load_data('TQQQ')
    # Load VIX
    vix = load_data('VIX')
    
    # Align data
    tqqq_et = tqqq.tz_convert('US/Eastern')
    vix_et = vix.tz_convert('US/Eastern')
    
    # Get TQQQ Daily Closes (DatetimeIndex)
    tqqq_daily_closes = get_market_closes(tqqq)
    
    # Get VIX Daily Closes (DatetimeIndex)
    vix_daily_closes = get_market_closes(vix)
    
    # --- Defined Reference Prices (TQQQ and VIX at start of period) ---
    
    # 1. Yesterday's Close
    tqqq_yesterday = tqqq_daily_closes.shift(1)
    vix_yesterday = vix_daily_closes.shift(1)
    
    # 2. 7 Days Ago Close (Fixed 7 Days)
    dates_7d_ago = tqqq_daily_closes.index - pd.Timedelta(days=7)
    
    # TQQQ 7d
    vals_7d_tqqq = tqqq_daily_closes.asof(dates_7d_ago)
    tqqq_7d = pd.Series(vals_7d_tqqq.values, index=tqqq_daily_closes.index)
    
    # VIX 7d
    # Note: Using TQQQ dates to drive the lookup ensures alignment, 
    # but VIX might have diff holidays? usually same.
    # Safer to look up VIX using the same dates_7d_ago against VIX index
    vals_7d_vix = vix_daily_closes.asof(dates_7d_ago)
    vix_7d = pd.Series(vals_7d_vix.values, index=tqqq_daily_closes.index)
    
    # 3. 30 Days Ago Close (Fixed 30 Days)
    dates_30d_ago = tqqq_daily_closes.index - pd.Timedelta(days=30)
    
    # TQQQ 30d
    vals_30d_tqqq = tqqq_daily_closes.asof(dates_30d_ago)
    tqqq_30d = pd.Series(vals_30d_tqqq.values, index=tqqq_daily_closes.index)
    
    # VIX 30d
    vals_30d_vix = vix_daily_closes.asof(dates_30d_ago)
    vix_30d = pd.Series(vals_30d_vix.values, index=tqqq_daily_closes.index)

    # --- Merge definitions into intraday dataframe ---
    
    tqqq_et['day_date'] = tqqq_et.index.normalize() # Normalize to midnight for joining
    
    # Prepare ref dataframe with both TQQQ and VIX start values
    refs = pd.DataFrame({
        'tqqq_yesterday': tqqq_yesterday,
        'tqqq_7d': tqqq_7d,
        'tqqq_30d': tqqq_30d,
        'vix_yesterday': vix_yesterday,
        'vix_7d': vix_7d,
        'vix_30d': vix_30d
    })
    
    merged_tqqq = tqqq_et.merge(refs, left_on='day_date', right_index=True, how='left')
    
    # Metrics Configuration: (Label, TQQQ Ref Col, VIX Ref Col)
    metrics = [
        ('Daily Change (vs Yesterday)', 'tqqq_yesterday', 'vix_yesterday'),
        ('Weekly Change (vs 7 Days Ago)', 'tqqq_7d', 'vix_7d'),
        ('Monthly Change (vs 30 Days Ago)', 'tqqq_30d', 'vix_30d')
    ]
    
    # Join Current VIX (though we might not need it for correlation anymore, useful for debug or checking)
    # We actually need 'vix_yesterday', etc. which are already in merged_tqqq
    final_df = merged_tqqq.copy()
    
    print(f"Analysis Period: {final_df.index.min()} to {final_df.index.max()}")
    print("-" * 100)
    print(f"{'Metric':<35} | {'Regime':<10} | {'VIX Range':<12} | {'Count':<8} | {'Correlation':<10}")
    print("-" * 100)
    
    # Define Regimes
    # You can adjust these thresholds
    regimes = {
        'Low':    (0, 15),
        'Medium': (15, 25),
        'High':   (25, 999)
    }
    
    plot_data = []

    for name, tqqq_ref, vix_ref in metrics:
        # Calculate pct change for TQQQ
        col_pct_change = f'pct_chg_{tqqq_ref}'
        final_df[col_pct_change] = (final_df['close'] - final_df[tqqq_ref]) / final_df[tqqq_ref]
        
        # Base valid data
        base_valid = final_df.dropna(subset=[col_pct_change, vix_ref])
        
        # 1. Overall Correlation (for reference)
        overall_corr = base_valid[col_pct_change].corr(base_valid[vix_ref])
        print(f"{name:<35} | {'Overall':<10} | {'All':<12} | {len(base_valid):<8} | {overall_corr:.4f}")
        
        # 2. Regime specific
        for regime_name, (low, high) in regimes.items():
            # Filter by VIX at START of period
            mask = (base_valid[vix_ref] >= low) & (base_valid[vix_ref] < high)
            regime_data = base_valid[mask]
            
            count = len(regime_data)
            if count > 10: # Need some data points
                corr = regime_data[col_pct_change].corr(regime_data[vix_ref])
                print(f"{'':<35} | {regime_name:<10} | {low}-{high:<9} | {count:<8} | {corr:.4f}")
            else:
                print(f"{'':<35} | {regime_name:<10} | {low}-{high:<9} | {count:<8} | N/A")
                
        print("-" * 100)
        
        # Collect for plotting (Overall)
        # We could do fancy plotting per regime but simple scatter is usually enough with color coding?
        # Let's keep the simple scatter but maybe color code points in the future.
        # For now, let's stick to the previous plot style for the File Output but maybe assume the console table is the primary result.
        if len(base_valid) > 0:
            plot_data.append((name, col_pct_change, vix_ref, overall_corr))

    # Plot (Same as before, coloring by VIX regime would be cool but simple first)
    if HAS_MATPLOTLIB and plot_data:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            if len(plot_data) < 3:
                pass 
                
            for i, (name, col_change, col_vix, corr) in enumerate(plot_data):
                ax = axes[i] if len(plot_data) > 1 else axes
                valid_data = final_df.dropna(subset=[col_change, col_vix])
                
                if len(valid_data) > 10000:
                    sample = valid_data.sample(10000)
                else:
                    sample = valid_data
                
                # Color code
                # Low VIX < 15 (Green), Med 15-25 (Blue), High > 25 (Red)
                colors = []
                for v in sample[col_vix]:
                    if v < 15: colors.append('green')
                    elif v < 25: colors.append('blue')
                    else: colors.append('red')
                    
                ax.scatter(sample[col_vix], sample[col_change], alpha=0.1, s=1, c=colors)
                ax.set_title(f"{name}\n(Overall Corr: {corr:.4f})")
                ax.set_xlabel(f"Start VIX Level")
                if i == 0:
                    ax.set_ylabel("TQQQ % Change")
                ax.grid(True, alpha=0.3)
                
                # Legend
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=5),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5)]
                ax.legend(custom_lines, ['Low (<15)', 'Med (15-25)', 'High (>25)'])
                
            
            # Save to results folder
            results_dir = 'results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            save_path = os.path.join(results_dir, 'correlation_analysis_regimes.png')
            plt.savefig(save_path)
            print(f"Regime-colored scatter plots saved to '{save_path}'")
        except Exception as e:
            print(f"Error plotting: {e}")

if __name__ == "__main__":
    analyze()
