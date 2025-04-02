import os
import time
import requests
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Set parameters



def create_binance_client():
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            client = Client(API_KEY, API_SECRET)
            client.ping()
            return client
        except requests.exceptions.ConnectionError as e:
            print(f"Connection failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError("Failed to establish connection after multiple attempts") from e

client = create_binance_client()

def fetch_historical_data(symbol, interval, start_date, end_date=None):
    """Fetch historical candlestick data between start_date and end_date"""
    if end_date is None:
        end_date = datetime.now()
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    data = []
    
    while True:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            limit=1000
        )
        
        if not klines:
            break
        
        data.extend(klines)
        last_ts = klines[-1][0]
        
        if last_ts >= end_ts:
            break
        
        start_ts = last_ts + 1
        time.sleep(0.1)
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df['timestamp'] <= end_date]
    
    return df

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
days=3

end_date = datetime.now()
start_date = end_date - timedelta(days)
print(f"Fetching historical data from {start_date} to {end_date}")
df = fetch_historical_data(SYMBOL, INTERVAL, start_date, end_date)
print(f"Fetched {len(df)} candles")

path = f"data/{SYMBOL}{days}_{INTERVAL}.csv"
df.to_csv(path)
print(f"Saved data to {path}")