# Developing a method to programmatically detect divergences similar to those observed 
# in the ETHUSDT chart over the past two days (31st and 30th March 2025)

import numpy as np
import ta
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

def plot(df: DataFrame) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['ma200'], label='MA200', color='orange')

    ax1.fill_between(df.index, df['close'], df['ma200'], where=(df['close'] > df['ma200']), 
                     color='green', alpha=0.3, label='Above MA200')
    ax1.fill_between(df.index, df['close'], df['ma200'], where=(df['close'] < df['ma200']),
                     color='red', alpha=0.3, label='Below MA200')
    ax1.set_title('ETHUSDT Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax2.plot(df.index, df['stoch_k'], label='Stochastic RSI K', color='green')
    ax2.plot(df.index, df['stoch_d'], label='Stochastic RSI D', color='red')
    ax2.axhline(y=0.8, color='gray', linestyle='--')
    ax2.axhline(y=0.2, color='gray', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stochastic RSI')
    ax3.plot(df.index, df['macd'], label='MACD', color='blue')
    ax3.plot(df.index, df['macd_signal'], label='MACD Signal', color='red')
    ax3.axhline(y=0, color='gray', linestyle='--')

    plt.show()

def get_divergences(df: DataFrame, window: int) -> None:
    df['price_lh'] = df['close'] < df['close'].shift(window)  # Lower high in price
    df['stoch_hh'] = df['stoch_k'] > df['stoch_k'].shift(window)  # Higher high in Stoch RSI

    df['hidden_bearish'] = (df['price_lh'] & df['stoch_hh'])  # Price LH + Stoch HH
    return df

def plot_hidden_bearish_divergences(df):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # 📌 Price Chart
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['ma200'], label='MA200', color='orange')
    ax1.plot(df.index, df['ma200']*1.01, label='MA200+1%', color='green')
    ax1.plot(df.index, df['ma200']*0.99, label='MA200-1%', color='red')

    ax1.fill_between(df.index, df['close'], df['ma200'], where=(df['close'] > df['ma200']), color='green', alpha=0.3, label='Above MA200')
    ax1.fill_between(df.index, df['close'], df['ma200'], where=(df['close'] < df['ma200']), color='red', alpha=0.3, label='Below MA200')

    # Mark Hidden Bearish Divergences
    bearish_divs = df[df['hidden_bearish']]
    ax1.scatter(bearish_divs.index, bearish_divs['close'], color='red', marker='v', s=50, label='Hidden Bearish Divergence')

    ax1.set_title('ETHUSDT Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    # 📌 Stochastic RSI
    ax2.plot(df.index, df['rsi'], label='RSI', color='green')
    
    ax2.axhline(y=70, color='gray', linestyle='--')
    ax2.axhline(y=50, color='gray', linestyle='--')
    ax2.axhline(y=30, color='gray', linestyle='--')

    # Mark divergence points on Stochastic RSI
    ax2.scatter(bearish_divs.index, bearish_divs['rsi'], color='red', marker='v', s=50)

    # 📌 MACD
    ax3.plot(df.index, df['macd'], label='MACD', color='blue')
    ax3.plot(df.index, df['macd_signal'], label='MACD Signal', color='red')
    
    ax3.axhline(y=0, color='gray', linestyle='--')

    # Mark divergence points on MACD
    ax3.scatter(bearish_divs.index, bearish_divs['macd'], color='red', marker='v', s=50)

    ax3.set_xlabel('Date')
    ax3.set_ylabel('MACD')
    ax3.legend()

    plt.show()


MACD_SLOW = 26
MACD_FAST = 12
MACD_SIGNAL = 9

STOCH_RSI_PERIOD = 14
STOCH_RSI_SMOOTH = 3

df = pd.read_csv('data/ETHUSDT3_5m.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

# MA200 Calculation
df['ma200'] = df['close'].rolling(window=200).mean()

# MACD Calculation
macd = ta.trend.MACD(
    close=df['close'],
    window_slow=MACD_SLOW,
    window_fast=MACD_FAST,
    window_sign=MACD_SIGNAL
)
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_hsitogram'] = macd.macd_diff()
    
# RSI Calculation
df['rsi'] = ta.momentum.RSIIndicator(
    close=df['close'],
    window=14
).rsi()

# Stochastic RSI Calculation
stoch_rsi = ta.momentum.StochRSIIndicator(
    close=df['close'],
    window=STOCH_RSI_PERIOD,
    smooth1=STOCH_RSI_SMOOTH,
    smooth2=STOCH_RSI_SMOOTH
)
df['stoch_k'] = stoch_rsi.stochrsi_k()
df['stoch_d'] = stoch_rsi.stochrsi_d()


df = get_divergences(df, window=30)
df = df.dropna()
plot_hidden_bearish_divergences(df)

# # PRICE< MA200, STOCH RSI and MACD
# df = df.dropna()
# print(df.head()) 
# plot(df)              

# # CALCULATE PRICE INCREASE

# time = df.index.to_numpy()
# prices = df['close'].to_numpy()
# min_prices = np.minimum.accumulate(prices)
# max_prices = np.maximum.accumulate(prices)

# price_increase = (max_prices[-1] - min_prices[-1])/min_prices[-1] * 100
# print(f"The biggest price increase was from {min_prices[-1]} to {max_prices[-1]}: {price_increase:.2f}%")
# time_elapsed: np.timedelta64 = time[-1] - time[0]
# print(f"This happend during {time_elapsed.astype('timedelta64[h]')}")

# plt.plot(time, prices, label='Price', color='blue')
# plt.plot(time, min_prices, label='Min Price', color='orange')
# plt.plot(time, max_prices, label='Max Price', color='green')
# plt.show()