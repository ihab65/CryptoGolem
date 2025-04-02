import time
import pandas as pd
import numpy as np
import ta
from binance.client import Client
import os
from datetime import datetime
import colorama
from colorama import Fore, Style
import math
import requests
import sys
import csv

# Initialize colorama
colorama.init(autoreset=True)

# 🔹 Binance API Keys (Read-Only for Safety)
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Initialize client with retry logic
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

# 🔹 Trading Configuration
INTERVAL = "5m"
LIMIT = 300  # Increased for EMA 200 calculation
SYMBOL = "ETHUSDT"
LEVERAGE = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Strategy Parameters
EMA_SHORT_PERIOD = 50
EMA_LONG_PERIOD = 200
PSAR_STEP = 0.02
PSAR_MAX_STEP = 0.2

# Logging Setup
LOG_FILE = "trading_recommendations.csv"

# Track open recommendations
open_recommendations = []

# ======================
# INDICATOR CALCULATIONS
# ======================

def get_historical_data(symbol, interval, limit=LIMIT):
    """Fetch historical candlestick data"""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    return df

def calculate_indicators(df):
    """Calculate technical indicators"""
    # Trend indicators
    df['EMA_50'] = df['close'].ewm(span=EMA_SHORT_PERIOD, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=EMA_LONG_PERIOD, adjust=False).mean()
    
    # Parabolic SAR
    psar = ta.trend.PSARIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        step=PSAR_STEP,
        max_step=PSAR_MAX_STEP
    )
    df['PSAR'] = psar.psar()
    
    # Momentum indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    macd = ta.trend.MACD(df['close'], 12, 26, 9)
    df['MACD_Hist'] = macd.macd_diff()
    
    # Volatility indicator
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['ATR'] = atr.average_true_range()
    
    return df

# ======================
# SIGNAL GENERATION
# ======================

def generate_recommendation(df):
    """Generate trading recommendations based on strategy"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['close']
    
    # Trend analysis
    trend = "bullish" if latest['EMA_50'] > latest['EMA_200'] else "bearish"
    
    # Parabolic SAR signal
    sar_bullish = (prev['PSAR'] > prev['close']) and (latest['PSAR'] < latest['close'])
    sar_bearish = (prev['PSAR'] < prev['close']) and (latest['PSAR'] > latest['close'])
    
    # Momentum filters
    rsi_valid = (latest['RSI'] < RSI_OVERBOUGHT) if trend == "bullish" else (latest['RSI'] > RSI_OVERSOLD)
    macd_valid = (latest['MACD_Hist'] > 0) if trend == "bullish" else (latest['MACD_Hist'] < 0)
    
    recommendation = None
    
    # Generate long recommendation
    if trend == "bullish" and sar_bullish and rsi_valid and macd_valid:
        sl = price - latest['ATR'] * ATR_MULTIPLIER_SL
        tp = price + latest['ATR'] * ATR_MULTIPLIER_TP
        recommendation = {
            'timestamp': datetime.now(),
            'action': 'long',
            'entry': price,
            'stop_loss': sl,
            'take_profit': tp,
            'status': 'open'
        }
    
    # Generate short recommendation
    elif trend == "bearish" and sar_bearish and rsi_valid and macd_valid:
        sl = price + latest['ATR'] * ATR_MULTIPLIER_SL
        tp = price - latest['ATR'] * ATR_MULTIPLIER_TP
        recommendation = {
            'timestamp': datetime.now(),
            'action': 'short',
            'entry': price,
            'stop_loss': sl,
            'take_profit': tp,
            'status': 'open'
        }
    
    return recommendation

# ======================
# LOGGING MECHANISM
# ======================

def log_recommendation(recommendation, outcome=None):
    """Log recommendations to CSV file"""
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'action', 'entry', 'stop_loss', 'take_profit',
                'leverage', 'status', 'outcome', 'exit_price'
            ])
        
        writer.writerow([
            recommendation['timestamp'],
            recommendation['action'],
            recommendation['entry'],
            recommendation['stop_loss'],
            recommendation['take_profit'],
            LEVERAGE,
            recommendation['status'],
            outcome,
            recommendation.get('exit_price', '')
        ])

# ======================
# MAIN TRADING LOGIC
# ======================

def check_recommendations(current_price):
    """Check open recommendations for exit conditions"""
    global open_recommendations
    
    for rec in open_recommendations:
        if rec['status'] == 'closed':
            continue
        
        exit_condition = False
        outcome = None
        
        # Check long position exits
        if rec['action'] == 'long':
            if current_price <= rec['stop_loss']:
                exit_condition = True
                outcome = 'stop_loss'
            elif current_price >= rec['take_profit']:
                exit_condition = True
                outcome = 'take_profit'
        
        # Check short position exits
        elif rec['action'] == 'short':
            if current_price >= rec['stop_loss']:
                exit_condition = True
                outcome = 'stop_loss'
            elif current_price <= rec['take_profit']:
                exit_condition = True
                outcome = 'take_profit'
        
        if exit_condition:
            rec['status'] = 'closed'
            rec['outcome'] = outcome
            rec['exit_price'] = current_price
            log_recommendation(rec, outcome)
            
    # Remove closed recommendations
    open_recommendations = [rec for rec in open_recommendations if rec['status'] == 'open']

def main():
    try:
        while True:
            # Fetch and process data
            df = get_historical_data(SYMBOL, INTERVAL)
            df = calculate_indicators(df)
            current_price = df.iloc[-1]['close']
            
            # Generate new recommendation
            recommendation = generate_recommendation(df)
            
            # Check exit conditions for open recommendations
            check_recommendations(current_price)
            
            # Process new recommendation
            if recommendation:
                open_recommendations.append(recommendation)
                log_recommendation(recommendation)
                
                # Print recommendation details
                print(f"\n{Fore.CYAN}{'='*40}")
                print(f"📈 New Recommendation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
                print(f"Action: {Fore.GREEN if recommendation['action'] == 'long' else Fore.RED}{recommendation['action'].upper()}")
                print(f"Entry Price: {current_price:.2f}")
                print(f"Stop Loss: {recommendation['stop_loss']:.2f}")
                print(f"Take Profit: {recommendation['take_profit']:.2f}")
                print(f"Leverage: {LEVERAGE}x")
                print(f"ATR: {df.iloc[-1]['ATR']:.2f}")
                print(f"RSI: {df.iloc[-1]['RSI']:.2f}")
                print(f"MACD Hist: {df.iloc[-1]['MACD_Hist']:.4f}")
                print(f"EMA 50/200: {df.iloc[-1]['EMA_50']:.2f}/{df.iloc[-1]['EMA_200']:.2f}")
            
            time.sleep(60)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Session terminated. Final recommendations:")
        for rec in open_recommendations:
            status = f"{Fore.GREEN}open" if rec['status'] == 'open' else f"{Fore.RED}closed"
            print(f" - {rec['action'].upper()} @ {rec['entry']:.2f} | {status}")

if __name__ == "__main__":
    main()