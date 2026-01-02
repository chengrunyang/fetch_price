"""
Script to fetch historical data for a specified stock symbol from IBKR.
Includes both Regular Trading Hours (RTH) and outside RTH data.
Default fetches 1 year (365 days) backwards from now, but duration is configurable.

Usage Examples:
    # Fetch 1 year of data for default symbol (TQQQ) at 5 mins interval
    python fetch_single_symbol.py

    # Fetch data for a specific symbol (e.g., SOXL)
    python fetch_single_symbol.py --symbol SOXL

    # Fetch 3 years (1095 days) of data
    python fetch_single_symbol.py --symbol AAPL --duration-days 1095

    # Fetch data for VIX Index (CBOE)
    python fetch_single_symbol.py --symbol VIX --sec-type IND --exchange CBOE --primary-exchange ""

    # Fetch 1-minute bars
    python fetch_single_symbol.py --interval "1 min"

    # Fetch 1-hour bars
    python fetch_single_symbol.py --interval "1 hour"
"""

from ib_insync import *
import pandas as pd
import datetime
import nest_asyncio
import argparse
import sys
import random

# Apply nest_asyncio to allow running in environments with existing loops
nest_asyncio.apply()

def fetch_historical_data(symbol='TQQQ', sec_type='STK', exchange='SMART', primary_exchange='NASDAQ', currency='USD', interval='5 mins', duration_days=365):
    ib = IB()
    
    # Try to connect with a random client ID to avoid conflicts
    # Retry a few times if connection fails
    connected = False
    # Try a default range or random IDs
    # We loop a few times to find an available ID
    for attempt in range(5):
        if attempt == 0:
             # First attempt could be a fixed ID or random. 
             # To be safe and avoid conflicts with other tools, start random.
             client_id = random.randint(1, 9999)
        else:
             client_id = random.randint(1, 9999)

        try:
            # Connect to IB Gateway or TWS
            ib.connect('127.0.0.1', 4002, clientId=client_id)
            connected = True
            break
        except Exception as e:
            print(f"Connection attempt {attempt+1} failed with clientId={client_id}: {e}")
            
    if not connected:
        print("Could not connect to IBKR after multiple attempts.")
        sys.exit(1)

    print('Connected:', ib.isConnected())
    
    # Define Contract
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    if primary_exchange:
        contract.primaryExchange = primary_exchange

    try:
        ib.qualifyContracts(contract)
        print(f"Qualified Contract: {contract}")
    except Exception as e:
        print(f"Error qualifying contract for {symbol}: {e}")
        ib.disconnect()
        sys.exit(1)

    # Settings for historical data
    end_time = '' # Empty string means 'now'
    all_bars = []
    
    # Calculate target start date (based on duration_days)
    # Using naive datetime for simplicity, assuming local system time calls are roughly aligned
    target_date = datetime.datetime.now() - datetime.timedelta(days=duration_days)
    
    print(f"Starting fetch loop (Target: ~{target_date.date()} for {duration_days} days)...")
    
    while True:
        # Fetch in chunks. 20 Days allows for faster fetching while respecting typical limits
        # 'useRTH=False' ensures outside trading hours are included
        print(f"Fetching chunk ending {end_time if end_time != '' else 'NOW'}")
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_time,
            durationStr='20 D',
            barSizeSetting=interval,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=False 
        )
        
        if not bars:
            print("No bars returned for this chunk.")
            break
            
        all_bars.extend(bars)
        
        earliest_bar = bars[0]
        earliest_time = earliest_bar.date
        
        # Log progress
        print(f"Fetched {len(bars)} bars. Earliest: {earliest_time}")
        
        # Handle timezone comparison
        # if earliest_time is timezone aware, we need to handle target_date
        if earliest_time.tzinfo is not None and target_date.tzinfo is None:
             target_date = target_date.replace(tzinfo=earliest_time.tzinfo)

        if earliest_time < target_date:
            print(f"Reached {duration_days} days limit.")
            break
            
        # Set end_time for the next chunk to the earliest time of current chunk
        # Subtract a small epsilon or just use the time. 
        # reqHistoricalData is exclusive of endDateTime usually if it's precise, but for dates it might vary.
        # Safest is to use the earliest time.
        end_time = earliest_time
        
        # Sleep to be nice to the API
        ib.sleep(0.5)

    ib.disconnect()

    # Process data
    if all_bars:
        df = util.df(all_bars)
        # Convert date column to datetime
        # utc=True handles mixed offsets if any, usually IB returns consistent though
        df['date'] = pd.to_datetime(df['date'], utc=True) 
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'])
        
        # Sort by date
        df = df.sort_values(by='date')
        
        # Filter strictly for the last 'duration_days'
        df = df[df['date'] >= target_date]
        
        print(f"\nTotal unique bars fetched: {len(df)}")
        print("First 5 rows:")
        print(df.head())
        print("Last 5 rows:")
        print(df.tail())
        
        return df
    else:
        print("No data fetched.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch historical data for a symbol from IBKR.')
    parser.add_argument('--symbol', type=str, default='TQQQ', help='Symbol to fetch (default: TQQQ)')
    parser.add_argument('--sec-type', type=str, default='STK', help='Security type (STK, IND, etc. default: STK)')
    parser.add_argument('--exchange', type=str, default='SMART', help='Exchange (SMART, CBOE, etc. default: SMART)')
    parser.add_argument('--primary-exchange', type=str, default='NASDAQ', help='Primary Exchange (default: NASDAQ). Set to empty string for Indices like VIX.')
    parser.add_argument("--currency", type=str, default="USD", help="Currency (default: USD)")
    parser.add_argument("--interval", type=str, default="5 mins", help="Bar size (e.g. '1 min', '5 mins', '1 hour', '1 day')")
    parser.add_argument("--duration-days", type=int, default=365, help="Number of days of history to fetch (default: 365)")

    args = parser.parse_args()
    
    # Handle primary_exchange being an empty string for indices
    p_exchange = args.primary_exchange if args.primary_exchange != '""' else ''

    df = fetch_historical_data(args.symbol, args.sec_type, args.exchange, p_exchange, args.currency, args.interval, args.duration_days)

    if df is not None:
        # Save to CSV
        import os
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Filename includes metadata to differentiate
        # Sanitize interval string for filename (replace spaces with underscores)
        safe_interval = args.interval.replace(' ', '_')
        
        # Get start and end dates for filename
        # df['date'] is datetime objects
        
        start_date_str = df['date'].min().strftime('%Y%m%d')
        end_date_str = df['date'].max().strftime('%Y%m%d')
        
        csv_filename = os.path.join(data_dir, f"{args.symbol}_{args.sec_type}_{safe_interval}_{start_date_str}_{end_date_str}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"\nData saved to {csv_filename}")
    else:
        print("No data fetched.")