import pandas as pd
import ta
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from typing import Dict, Any
from datetime import timedelta

# Initialize precision
getcontext().prec = 8

class FullLeverageBacktester:
    def __init__(self, csv_path: str, symbol: str = "SOLUSDT", initial_balance: Decimal = Decimal("5")):
        getcontext().prec = 8
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = 20  # Fixed 20x leverage
        self.position_size_usdt = Decimal("100")  # $5 * 20x
        self.taker_fee = Decimal("0.0004")
        self.maker_fee = Decimal("0.0002")
        
        # ATR parameters for dynamic stops
        self.atr_period = 14
        self.atr_multiplier_sl = Decimal("1.5")  # 1.5x ATR for stop loss
        self.atr_multiplier_tp = Decimal("3.0")  # 3x ATR for take profit (1:2 RR)
        
        # Indicator parameters
        self.vwap_window = 20
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_rsi_period = 14
        self.stoch_rsi_smooth = 3
        
        # Trade execution control
        self.active_trade = None
        self.last_trade_time = None
        self.trade_cooldown = timedelta(hours=1)
        
        # Load and process data
        self.data = self.load_data(csv_path)
        self.trades = []
        self.equity_curve = [{'timestamp': self.data.iloc[0]['timestamp'], 
                             'balance': float(initial_balance)}]

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare OHLCV data with ATR"""
        df = pd.read_csv(csv_path)
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in CSV")
            
        # Convert and sort
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return self.calculate_indicators(df)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # Volatility indicator
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], 
                close=df['close'], window=self.atr_period
            ).average_true_range()
            
            # Trend indicators
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], 
                close=df['close'], volume=df['volume'],
                window=self.vwap_window
            ).volume_weighted_average_price()
            
            macd = ta.trend.MACD(
                close=df['close'],
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # Momentum indicators
            stoch_rsi = ta.momentum.StochRSIIndicator(
                close=df['close'],
                window=self.stoch_rsi_period,
                smooth1=self.stoch_rsi_smooth
            )
            df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
            
            return df.dropna()
        except Exception as e:
            print(f"Indicator error: {str(e)}")
            return pd.DataFrame()

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Generate trading signals with ATR-based stops"""
        if idx < self.atr_period or self.active_trade:
            return None
            
        current = df.iloc[idx]
        atr = current['atr']
        
        signal = {
            "direction": None,
            "entry_price": float(current['close']),
            "atr": float(atr),
            "timestamp": current['timestamp']
        }
        
        # Check indicator validity
        if any(pd.isna([current['macd'], current['macd_signal'], 
                       current['stoch_rsi_k'], current['vwap'], atr])):
            return None
        
        # Long conditions
        long_cond = (
            current['close'] > current['vwap'] and
            current['macd'] > current['macd_signal'] and
            current['macd_hist'] > 0 and
            current['stoch_rsi_k'] < 0.2 and
            current['stoch_rsi_k'] > current['stoch_rsi_d']
        )
        
        # Short conditions
        short_cond = (
            current['close'] < current['vwap'] and
            current['macd'] < current['macd_signal'] and
            current['macd_hist'] < 0 and
            current['stoch_rsi_k'] > 0.8 and
            current['stoch_rsi_k'] < current['stoch_rsi_d']
        )
        
        if long_cond:
            signal["direction"] = "long"
            signal["stop_loss"] = float(current['close'] - float(self.atr_multiplier_sl) * atr)
            signal["take_profit"] = float(current['close'] + float(self.atr_multiplier_tp) * atr)
            
        elif short_cond:
            signal["direction"] = "short"
            signal["stop_loss"] = float(current['close'] + float(self.atr_multiplier_sl) * atr)
            signal["take_profit"] = float(current['close'] - float(self.atr_multiplier_tp) * atr)
            
        return signal if signal["direction"] else None

    def calculate_position_size(self, entry_price: float) -> Decimal:
        """Always use full $5 with 20x leverage ($100 position)"""
        return round(self.position_size_usdt / Decimal(str(entry_price)), 3)

    def execute_trade(self, signal: Dict, idx: int) -> Dict:
        """Execute trade with full leverage and ATR stops"""
        if not signal:
            return {"status": "no_signal"}
            
        current_time = pd.to_datetime(signal['timestamp'])
        
        # Check cooldown period
        if (self.last_trade_time and 
            (current_time - self.last_trade_time) < self.trade_cooldown):
            return {"status": "cooldown_active"}
            
        # Calculate position size (full leverage)
        quantity = self.calculate_position_size(signal["entry_price"])
        if quantity <= 0:
            return {"status": "invalid_size"}
            
        # Calculate fees and PnL
        entry_fee = Decimal(str(signal["entry_price"])) * quantity * self.taker_fee
        exit_fee = Decimal(str(signal["take_profit"])) * quantity * self.maker_fee
        
        if signal["direction"] == "long":
            profit = (Decimal(str(signal["take_profit"])) - Decimal(str(signal["entry_price"]))) * quantity
        else:
            profit = (Decimal(str(signal["entry_price"])) - Decimal(str(signal["take_profit"]))) * quantity
            
        net_profit = profit - entry_fee - exit_fee
        
        # Create trade record
        trade = {
            "status": "executed",
            "direction": signal["direction"],
            "entry_price": signal["entry_price"],
            "stop_loss": signal["stop_loss"],
            "take_profit": signal["take_profit"],
            "atr": signal["atr"],
            "quantity": float(quantity),
            "gross_profit": float(profit),
            "net_profit": float(net_profit),
            "fees": float(entry_fee + exit_fee),
            "leverage": self.leverage,
            "timestamp": signal["timestamp"],
            "balance": float(self.balance + net_profit)
        }
        
        # Update state
        self.balance += net_profit
        self.equity_curve.append({
            "timestamp": signal["timestamp"],
            "balance": float(self.balance)
        })
        self.trades.append(trade)
        self.active_trade = trade
        self.last_trade_time = current_time
        
        return trade

    def run_backtest(self):
        """Run the complete backtest with full leverage"""
        print(f"Starting backtest for {self.symbol}")
        print(f"Initial balance: ${float(self.initial_balance)}")
        print(f"Position size: ${float(self.position_size_usdt)} (20x leverage)")
        print(f"ATR Stops: {float(self.atr_multiplier_sl)}x/TP {float(self.atr_multiplier_tp)}x")
        
        for idx in range(self.atr_period, len(self.data)):
            # Check for active trade closure
            if self.active_trade:
                current_price = self.data.iloc[idx]['close']
                direction = self.active_trade["direction"]
                
                if ((direction == "long" and 
                     (current_price >= self.active_trade["take_profit"] or 
                      current_price <= self.active_trade["stop_loss"])) or
                    (direction == "short" and 
                     (current_price <= self.active_trade["take_profit"] or 
                      current_price >= self.active_trade["stop_loss"]))):
                    
                    self.active_trade = None
            
            # Generate and execute new signal
            if not self.active_trade:
                signal = self.generate_signal(self.data, idx)
                if signal:
                    result = self.execute_trade(signal, idx)
                    if result["status"] == "executed":
                        print(f"\nTrade {len(self.trades)}: {result['direction']} at {result['timestamp']}")
                        print(f"Price: {result['entry_price']:.4f} | ATR: {result['atr']:.4f}")
                        print(f"SL: {result['stop_loss']:.4f} | TP: {result['take_profit']:.4f}")
                        print(f"Profit: {result['net_profit']:.2f} | Balance: {result['balance']:.2f}")
        
        # Final account status
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive performance analysis"""
        if not self.trades:
            print("\nNo trades executed")
            return
            
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate metrics
        total_net = trades_df['net_profit'].sum()
        total_fees = trades_df['fees'].sum()
        roi = (total_net / float(self.initial_balance)) * 100
        win_rate = (len(trades_df[trades_df['net_profit'] > 0]) / len(trades_df)) * 100
        avg_profit = trades_df['net_profit'].mean()
        max_dd = self.calculate_max_drawdown(equity_df)
        
        print("\n=== Backtest Results ===")
        print(f"Initial Balance: ${float(self.initial_balance):.2f}")
        print(f"Final Balance: ${float(self.balance):.2f}")
        print(f"Total Net Profit: ${total_net:.2f} (ROI: {roi:.2f}%)")
        print(f"Win Rate: {win_rate:.2f}% | Avg Profit: ${avg_profit:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}% | Trades: {len(trades_df)}")
        
        # Additional stats
        if not trades_df.empty:
            avg_atr = trades_df['atr'].mean()
            print(f"\nRisk Statistics:")
            print(f"Avg ATR: {avg_atr:.4f}")
            print(f"Largest Loss: ${trades_df['net_profit'].min():.2f}")
            print(f"Largest Win: ${trades_df['net_profit'].max():.2f}")
        
        # Plotting
        self.plot_results(trades_df, equity_df)

    def calculate_max_drawdown(self, equity_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        equity = equity_df['balance']
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        return drawdown.max() * 100

    def plot_results(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame):
        """Generate performance visualizations"""
        plt.figure(figsize=(15, 10))
        
        # Equity Curve
        plt.subplot(2, 2, 1)
        plt.plot(pd.to_datetime(equity_df['timestamp']), equity_df['balance'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (USD)')
        plt.grid(True)
        
        # Profit Distribution
        plt.subplot(2, 2, 2)
        plt.hist(trades_df['net_profit'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Profit Distribution')
        plt.xlabel('Profit (USD)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Cumulative Profit
        plt.subplot(2, 2, 3)
        plt.plot(pd.to_datetime(trades_df['timestamp']), trades_df['net_profit'].cumsum())
        plt.title('Cumulative Profit')
        plt.xlabel('Date')
        plt.ylabel('Profit (USD)')
        plt.grid(True)
        
        # Trade Directions
        plt.subplot(2, 2, 4)
        dir_counts = trades_df['direction'].value_counts()
        colors = ['green' if d == 'long' else 'red' for d in dir_counts.index]
        plt.bar(dir_counts.index, dir_counts.values, color=colors)
        plt.title('Trade Direction')
        plt.xlabel('Direction')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    csv_path = "data/solusdt_1h.csv"  # Replace with your data file
    try:
        bot = FullLeverageBacktester(csv_path)
        bot.run_backtest()
    except Exception as e:
        print(f"Backtest failed: {str(e)}")