import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def add_technical_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
    """Add technical indicators to DataFrame"""
    # Ensure we're working with a copy and preserve timestamp column
    df = df.reset_index().sort_values('timestamp').copy()
    
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

# Modified loading sequence
df = pd.read_csv('hardusdt.csv', parse_dates=['timestamp'])
df = add_technical_indicators(df)  # Let the function handle timestamp
df = df.sort_index()  # Final sorting

class CryptoStrategyBacktester:
    """
    Backtesting engine for lower peak entry strategy
    Includes spike detection, trough identification, and performance analysis
    """
    
    def __init__(self, df, params):
        self.df = df.sort_index()
        self.params = params
        self.trades = []
        
        # Default parameters
        self.default_params = {
            'spike_threshold': 0.10,
            'trough_window': 24,  # 2 hours for 5m intervals
            'min_peak_distance': 12,  # 1 hour
            'take_profit': 0.12,
            'stop_loss': 0.05,
            'max_hold_period': 288  # 24 hours
        }
        
        # Merge user parameters
        self.default_params.update(params)
        self.params = self.default_params

    def _find_local_minima(self, series, window=24):
        """Identify local minima using rolling window comparison"""
        minima = argrelextrema(series.values, np.less_equal, 
                              order=window)[0]
        return series.iloc[minima]

    def detect_entry_points(self):
        """
        Find potential entry points that precede price spikes
        Returns DataFrame with entry candidates and their features
        """
        # Find local minima using price and volume confirmation
        price_minima = self._find_local_minima(self.df['low'], 
                                             window=self.params['trough_window'])
        
        entry_candidates = []
        
        for idx, row in self.df.iterrows():
            # Check if we're in a local minimum
            if idx not in price_minima.index:
                continue
                
            # Technical confirmation criteria
            rsi_ok = row['RSI'] < 65
            volume_ok = row['volume'] > row['volume_20MA']
            macd_bullish = row['MACD'] > row['MACD_signal']
            
            if rsi_ok and volume_ok and macd_bullish:
                entry_candidates.append({
                    'timestamp': idx,
                    'price': row['close'],
                    'RSI': row['RSI'],
                    'MACD_diff': row['MACD'] - row['MACD_signal'],
                    'volume_ratio': row['volume'] / row['volume_20MA']
                })
                
        return pd.DataFrame(entry_candidates).set_index('timestamp')

    def backtest_strategy(self, entry_points):
        """
        Simulate trades based on entry points and exit rules
        """
        for entry_time, entry_row in entry_points.iterrows():
            # Find subsequent price action
            future_data = self.df.loc[entry_time:].iloc[1:self.params['max_hold_period']]
            
            if future_data.empty:
                continue
                
            # Exit conditions
            take_profit_level = entry_row['price'] * (1 + self.params['take_profit'])
            stop_loss_level = entry_row['price'] * (1 - self.params['stop_loss'])
            
            # Check for TP/SL hits
            tp_hit = future_data['high'] >= take_profit_level
            sl_hit = future_data['low'] <= stop_loss_level
            
            # Find exit time
            exit_condition = tp_hit | sl_hit
            if exit_condition.any():
                exit_idx = exit_condition.idxmax()
                exit_type = 'TP' if tp_hit.loc[exit_idx] else 'SL'
            else:
                exit_idx = future_data.index[-1]
                exit_type = 'Timeout'
                
            # Record trade
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': exit_idx,
                'entry_price': entry_row['price'],
                'exit_price': self.df.loc[exit_idx, 'close'],
                'exit_type': exit_type,
                'hold_period': (exit_idx - entry_time).total_seconds()/3600
            })
            
        return pd.DataFrame(self.trades)

    def calculate_metrics(self, trades):
        """Calculate strategy performance metrics"""
        trades = trades.copy()
        trades['roi'] = (trades['exit_price'] / trades['entry_price'] - 1) * 100
        
        # Win/loss classification
        trades['success'] = trades['roi'] > 0
        
        metrics = {
            'total_trades': len(trades),
            'win_rate': trades['success'].mean(),
            'avg_roi': trades['roi'].mean(),
            'max_drawdown': (trades['exit_price'] / trades['entry_price'] - 1).min() * 100,
            'profit_factor': trades[trades['roi'] > 0]['roi'].sum() / 
                            abs(trades[trades['roi'] < 0]['roi'].sum()),
            'avg_hold_time': trades['hold_period'].mean()
        }
        
        return metrics

# Usage Example ###############################################################

# Load and preprocess data (from previous implementation)
df = pd.read_csv('hardusdt.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()

# Add technical indicators
df = add_technical_indicators(df)

# Initialize backtester with strategy parameters
params = {
    'take_profit':     0.10,  # Align with observed average spike
    'stop_loss':       0.03,
    'max_hold_period': 288  # 24 hours in 5m intervals
}

backtester = CryptoStrategyBacktester(df, params)

# Step 1: Detect potential entry points
entry_points = backtester.detect_entry_points()

# Step 2: Backtest strategy
trades_df = backtester.backtest_strategy(entry_points)

# Step 3: Calculate performance metrics
metrics = backtester.calculate_metrics(trades_df)

# Output results
print("\nStrategy Performance Metrics:")
print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Average ROI: {metrics['avg_roi']:.1f}%")
print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Average Hold Time: {metrics['avg_hold_time']:.1f} hours")

# Visualize trades
plt.figure(figsize=(15,6))
plt.plot(df['close'], label='Price')
plt.scatter(entry_points.index, entry_points['price'], 
           color='green', marker='^', label='Entries')
plt.scatter(trades_df['exit_time'], trades_df['exit_price'],
           color='red', marker='v', label='Exits')
plt.title('Strategy Backtest Visualization')
plt.legend()
plt.show()