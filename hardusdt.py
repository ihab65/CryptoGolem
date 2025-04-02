from binance.client import Client
import requests, os, time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Set parameters
INTERVAL = "5m"
SYMBOL = "HARDUSDT"
SPIKE_THRESHOLD = 0.08  # 8% price increase
WINDOW_4H = 48
WINDOW_24H = 288

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

def add_technical_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
    """Add technical indicators to DataFrame"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # RSI Calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    
    # Volume Analysis
    df['volume_20MA'] = df['volume'].rolling(20).mean()
    
    return df.set_index('timestamp').sort_index().dropna()

def detect_price_spikes(df, 
                       min_spike=SPIKE_THRESHOLD,  # Reduced threshold
                       window_range=(WINDOW_4H, WINDOW_24H),  # 4h-24h window
                       volatility_factor=0.5):
    
    """Improved spike detection with adaptive thresholds"""
    # Calculate volatility-adjusted thresholds
    df['volatility'] = df['close'].rolling(24).std() / df['close'].rolling(24).mean()
    dynamic_threshold = min_spike * (1 + df['volatility'] * volatility_factor)
    
    spike_indices = []
    
    for i in range(len(df) - window_range[1]):
        window = df.iloc[i+window_range[0]:i+window_range[1]]
        if window.empty:
            continue
            
        peak_val = window['close'].max()
        base_val = df['close'].iloc[i]
        
        # Multi-factor confirmation
        vol_ratio = window['volume'].mean() / df['volume_20MA'].iloc[i]
        rsi_condition = df['RSI'].iloc[i] < 30
        macd_cross = (df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]) and \
                    (df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1])
        
        if (peak_val/base_val - 1 >= dynamic_threshold.iloc[i] and
            vol_ratio > 1.2 and
            (rsi_condition or macd_cross)):
            spike_indices.append(i)
            
    return spike_indices

def group_spike_events(spike_indices, group_window=96):
    """Group nearby spike events"""
    if not spike_indices:
        return []
    
    grouped = []
    current_group = [spike_indices[0]]
    
    for idx in spike_indices[1:]:
        if idx - current_group[-1] <= group_window:
            current_group.append(idx)
        else:
            grouped.append(current_group)
            current_group = [idx]
    
    grouped.append(current_group)
    return [min(group) for group in grouped]

def analyze_indicators(df, spike_indices):
    """Analyze technical indicators at spike events"""
    analysis = []
    
    for idx in spike_indices:
        try:
            if idx + WINDOW_24H >= len(df):
                continue
                
            window = df.iloc[idx:idx+WINDOW_24H]
            peak_idx = window['close'].idxmax()
            
            if peak_idx not in df.index:
                continue
                
            analysis.append({
                'event_time': df.index[idx],
                'peak_time': peak_idx,
                'start_price': df['close'].iloc[idx],
                'peak_price': df.loc[peak_idx, 'close'],
                'price_increase': (df.loc[peak_idx, 'close']/df['close'].iloc[idx] - 1) * 100,
                'RSI': df['RSI'].iloc[idx],
                'MACD_bullish': df['MACD'].iloc[idx] > df['MACD_signal'].iloc[idx],
                'volume_spike': df['volume'].iloc[idx] > df['volume_20MA'].iloc[idx] * 1.5,
                'volume_ratio': df['volume'].iloc[idx]/df['volume_20MA'].iloc[idx]
            })
        except (IndexError, KeyError):
            continue
    
    return pd.DataFrame(analysis)

def plot_results(df, analysis_df, start_date='2025-02-15', end_date='today'):
    """Visualize results for specific time period with enhanced spike detection"""
    # Filter data for specified date range
    date_mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    filtered_df = df[date_mask]
    filtered_analysis = analysis_df[analysis_df['event_time'].between(
        pd.to_datetime(start_date), pd.to_datetime(end_date))]

    plt.figure(figsize=(15, 12))
    
    # Price Plot
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(filtered_df.index, filtered_df['close'], label='Price')
    
    if not filtered_analysis.empty:
        valid_peaks = filtered_analysis[filtered_analysis['peak_time'].isin(filtered_df.index)]
        plt.scatter(
            valid_peaks['peak_time'],
            valid_peaks['peak_price'],
            color='red', zorder=5, label='Peak Price'
        )
    
    plt.title(f'Price Chart with Detected Spikes (n={len(filtered_analysis)})')
    plt.ylabel('Price')
    plt.legend()

    # Technical Indicators
    for i, (col, title) in enumerate(zip(
        ['RSI', 'MACD', 'volume'],
        ['RSI', 'MACD', 'Volume']
    ), 2):
        ax = plt.subplot(4, 1, i, sharex=ax1)
        
        if col == 'RSI':
            plt.plot(filtered_df.index, filtered_df['RSI'], label='RSI')
            plt.axhline(70, color='grey', linestyle='--')
            plt.axhline(30, color='grey', linestyle='--')
        elif col == 'MACD':
            plt.plot(filtered_df.index, filtered_df['MACD'], label='MACD')
            plt.plot(filtered_df.index, filtered_df['MACD_signal'], label='Signal')
        elif col == 'volume':
            plt.bar(filtered_df.index, filtered_df['volume'], alpha=0.5, label='Volume')
            plt.plot(filtered_df.index, filtered_df['volume_20MA'], label='20MA', color='orange')
            
        plt.ylabel(title)
        plt.legend()

    # Add event markers
    for ax in [ax1, *plt.gcf().axes[1:]]:
        for t in filtered_analysis['event_time']:
            if t in filtered_df.index:
                ax.axvline(t, color='red', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

# Main Execution Flow
if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    print(f"Fetching historical data from {start_date} to {end_date}")
    df = fetch_historical_data(SYMBOL, INTERVAL, start_date, end_date)
    print(f"Fetched {len(df)} candles")
    # Load and preprocess data

    df = add_technical_indicators(df)
    df.to_csv("/home/ihab65/Projects/CryptoBot/Outputs/hardusdt.csv")
    
    # Detect and analyze spikes
    spike_indices = detect_price_spikes(df)
    grouped_spikes = group_spike_events(spike_indices)
    analysis_df = analyze_indicators(df, grouped_spikes)
    

    # Generate output
    plot_results(df, analysis_df)
    
    print("Technical Analysis Report:")
    print(f"Detected spikes: {len(analysis_df)}")
    print(f"Average price increase: {analysis_df['price_increase'].mean():.1f}%")
    print(f"RSI > 70 ratio: {analysis_df['RSI'].gt(70).mean():.1%}")
    print(f"MACD bullish ratio: {analysis_df['MACD_bullish'].mean():.1%}")
    print(f"Volume spikes ratio: {analysis_df['volume_spike'].mean():.1%}")