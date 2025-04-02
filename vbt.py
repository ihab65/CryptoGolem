import vectorbt as vbt
import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt

# Configuration
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
INITIAL_CASH = 10000  # USDT
COMMISSION = 0.0004    # Taker fee
SLIPPAGE = 0.0005      # 0.05%
RISK_PER_TRADE = 0.01  # 2% of capital per trade

# Strategy Parameters
VWAP_WINDOW = 20
MACD_FAST = 8
MACD_SLOW = 21
MACD_SIGNAL = 9
STOCH_RSI_PERIOD = 10
STOCH_RSI_SMOOTH = 3
STOCH_OVERBOUGHT = 0.8     # Stricter entry filter
STOCH_OVERSOLD = 0.2
EMA_TREND_PERIOD = 200  
ATR_PERIOD = 14
TP_MULTIPLIER = 2.0
SL_MULTIPLIER = 1.2

# Fetch data using correct method
futures_data = vbt.BinanceData.download(
    SYMBOL,
    interval=TIMEFRAME,
    start=START_DATE,
    end=END_DATE
)

# Convert to pandas DataFrame with proper column names
ohlcv_df = pd.DataFrame({
    'Open': futures_data.get('Open'),
    'High': futures_data.get('High'),
    'Low': futures_data.get('Low'),
    'Close': futures_data.get('Close'),
    'Volume': futures_data.get('Volume')
})

# Calculate indicators
def calculate_indicators(df):
    # VWAP Calculation
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=VWAP_WINDOW
    ).volume_weighted_average_price()
    
    # MACD Calculation
    macd = ta.trend.MACD(
        close=df['Close'],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Stochastic RSI Calculation
    stoch_rsi = ta.momentum.StochRSIIndicator(
        close=df['Close'],
        window=STOCH_RSI_PERIOD,
        smooth1=STOCH_RSI_SMOOTH,
        smooth2=STOCH_RSI_SMOOTH
    )
    df['stoch_k'] = stoch_rsi.stochrsi_k()
    df['stoch_d'] = stoch_rsi.stochrsi_d()
    
    # ATR Calculation
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=ATR_PERIOD
    ).average_true_range()

    # Add Trend Filter
    df['ema_trend'] = ta.trend.EMAIndicator(
        close=df['Close'], window=EMA_TREND_PERIOD
    ).ema_indicator()
    
    # Add Volume Confirmation
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_spike'] = df['Volume'] > df['volume_ma'] * 1.5
    
    return df.bfill()

# Calculate indicators
indicators = calculate_indicators(ohlcv_df)

# Generate entry signals
# Enhanced Entry Conditions
entries = (
    (indicators['Close'] > indicators['ema_trend']) &          
    (indicators['Close'] > indicators['vwap']) &               
    (indicators['macd'] > indicators['macd_signal']) &         
    (indicators['stoch_k'] > STOCH_OVERSOLD) &         
    (indicators['stoch_k'] < STOCH_OVERBOUGHT) &       
    (indicators['stoch_k'] > indicators['stoch_d']) &          
    (indicators['volume_spike'])                       
).shift(1)

# Calculate dynamic position sizing
def calculate_position_size(close, atr):
    risk_amount = INITIAL_CASH * RISK_PER_TRADE
    price_risk = atr * SL_MULTIPLIER
    return risk_amount / price_risk

position_size = calculate_position_size(indicators['Close'], indicators['atr'])

# Generate exit signals with ATR-based SL/TP
exits = pd.Series(False, index=indicators.index)
tp_levels = pd.Series(np.nan, index=indicators.index)
sl_levels = pd.Series(np.nan, index=indicators.index)

in_position = False
for i in range(1, len(indicators)):
    if entries.iloc[i] and not in_position:
        entry_price = indicators['Close'].iloc[i]
        current_atr = indicators['atr'].iloc[i]
        
        # Dynamic TP/SL based on volatility regime
        volatility_factor = indicators['atr'].iloc[i] / indicators['atr'].rolling(50).mean().iloc[i]
        tp_level = entry_price + current_atr * TP_MULTIPLIER * volatility_factor
        sl_level = entry_price - current_atr * SL_MULTIPLIER * volatility_factor
        
        # Store initial levels
        best_price = entry_price
        in_position = True
    
    elif in_position:
        current_close = indicators['Close'].iloc[i]
        current_high = indicators['High'].iloc[i]
        current_low = indicators['Low'].iloc[i]
        
        # Update best price for trailing
        if current_close > best_price:
            best_price = current_close
            sl_level = best_price - (current_atr * SL_MULTIPLIER * volatility_factor)
        
        # Check exits
        if current_high >= tp_level or current_low <= sl_level:
            exits.iloc[i] = True
            in_position = False

# Ensure entries and exits are boolean type
entries = entries.astype(bool)
exits = exits.astype(bool)

# Ensure position size is numeric and fills NaNs with zero
position_size = position_size.fillna(0).astype(float)

# Reindex to match the indicators' index
entries = entries.reindex(indicators.index, fill_value=False)
exits = exits.reindex(indicators.index, fill_value=False)

# Execute backtest
portfolio = vbt.Portfolio.from_signals(
    close=indicators['Close'],
    entries=entries,
    exits=exits,
    init_cash=INITIAL_CASH,
    fees=COMMISSION,
    slippage=SLIPPAGE,
    size=position_size,
    freq=TIMEFRAME
)

# Performance analysis
print("Strategy Performance:")
print(portfolio.stats())

# Visual improvements
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[:2])
ax2 = fig.add_subplot(gs[2])

# Price and indicators plot
ax1.set_title("Price Action with Entries/Exits")
ax1.plot(indicators['Close'], label='Price', color='purple', alpha=0.8)
ax1.plot(indicators['vwap'], label='VWAP', color='orange')
ax1.scatter(indicators.index[entries], indicators['Close'][entries], 
           marker='^', color='green', s=100, label='Entry')
ax1.scatter(indicators.index[exits], indicators['Close'][exits],
           marker='v', color='red', s=100, label='Exit')
ax1.plot(tp_levels, linestyle='--', color='blue', alpha=0.7, label='TP Level')
ax1.plot(sl_levels, linestyle='--', color='brown', alpha=0.7, label='SL Level')
ax1.legend()

# Equity curve
equity = portfolio.value()
ax2.set_title("Portfolio Equity Curve")
ax2.plot(equity, label='Equity', color='navy')
ax2.fill_between(equity.index, equity, color='navy', alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# Enhanced metrics
print("\nAdvanced Statistics:")
print(f"Average Holding Period: {portfolio.trades.duration.mean():.2f} bars")
print(f"Profit Factor: {portfolio.trades.profit_factor():.2f}")
print(f"Max Drawdown: {portfolio.stats('max_dd') * 100:.2f}%")
print(f"Sharpe Ratio: {portfolio.stats('sharpe_ratio'):.2f}")
print(f"Calmar Ratio: {portfolio.stats('calmar_ratio'):.2f}")

# Trade analysis
trades = portfolio.trades.records_readable
trades['return_pct'] = trades['PnL'] / trades['Size'] * 100
print("\nTrade Performance Summary:")
print(trades[['Entry Date', 'Exit Date', 'PnL', 'return_pct', 'Duration']].describe())

# Risk metrics
print("\nRisk Analysis:")
print(f"Winning Trade Avg: ${trades[trades['PnL'] > 0]['PnL'].mean():.2f}")
print(f"Losing Trade Avg: ${trades[trades['PnL'] < 0]['PnL'].mean():.2f}")
print(f"Best Trade: ${trades['PnL'].max():.2f}")
print(f"Worst Trade: ${trades['PnL'].min():.2f}")