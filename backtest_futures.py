import pandas as pd
import numpy as np
import ta, os
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from typing import Dict, Any, Optional
from datetime import timedelta, datetime
import random
from collections import deque
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Import your liquidation calculator components here
from liquidation import SymbolData, LiquidationCalculator, TradeSimulator

getcontext().prec = 8

class ProfessionalBacktester:
    def __init__(self, csv_path: str, symbol: str = "SOLUSDT",
                 initial_balance: Decimal = Decimal("1000"),
                 risk_per_trade: Decimal = Decimal("0.02"),
                 max_leverage: int = 5):
        
        # ----------
        # Core Configuration
        # ----------
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.trade_history = deque(maxlen=1000)  # Rolling window for MC simulations
        
        # ----------
        # Market Structure
        # ----------
        self.symbol_info = self.load_symbol_data(symbol)
        self.price_precision = self.symbol_info.price_precision
        self.quantity_precision = self.symbol_info.quantity_precision
        
        # ----------
        # Execution Parameters
        # ----------
        self.taker_fee = Decimal("0.0006")
        self.maker_fee = Decimal("0.0002")
        self.slippage_model = {
            'base': Decimal("0.0005"),
            'volatility_multiplier': Decimal("0.25")
        }
        self.min_liquidity_ratio = Decimal("0.05")  # 5% of candle volume
        
        # ----------
        # Strategy Parameters
        # ----------
        self.vwap_window = 20
        self.macd_params = (12, 26, 9)
        self.stoch_rsi_params = (14, 3)
        self.atr_period = 14
        self.atr_multiplier = Decimal("1.5")
        self.rr_ratio = Decimal("2.0")
        
        # ----------
        # Risk Management
        # ----------
        self.max_daily_trades = 3
        self.volatility_filter = Decimal("0.25")  # 25% of 30d ATR
        self.portfolio_beta = Decimal("1.0")  # Placeholder for multi-asset
        
        # ----------
        # Data Handling
        # ----------
        self.full_data = self.load_and_preprocess_data(csv_path)
        self.train_data, self.test_data = self.time_split_data(0.7)
        self.current_data = None
        
        # ----------
        # State Tracking
        # ----------
        self.active_positions = []
        self.equity_curve = []
        self.trade_log = []
        self.market_regime = None

    def load_symbol_data(self, symbol: str) -> SymbolData:
        """Load symbol-specific parameters from exchange"""
        try:
            client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
            info = client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    return SymbolData(s)
            return self.get_hardcoded_symbol_data(symbol)
        except BinanceAPIException as e:
            print(f"API Error: {e}. Using fallback data.")
            return self.get_hardcoded_symbol_data(symbol)
        except Exception as e:
            print(f"General error: {e}. Using fallback data.")
            return self.get_hardcoded_symbol_data(symbol)

    def load_and_preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate historical data with outlier detection"""
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        
        # Validate completeness
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.validate_dataset(df, required_cols)
        
        # Clean and sort
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = self.handle_missing_data(df)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Add market regime labels
        df = self.calculate_market_regimes(df)
        
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with look-ahead prevention"""
        # Volatility
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'],
            window=self.atr_period
        ).average_true_range().shift(1)
        
        # Trend
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'],
            volume=df['volume'], window=self.vwap_window
        ).volume_weighted_average_price().shift(1)
        
        # Momentum
        macd = ta.trend.MACD(
            close=df['close'], window_slow=self.macd_params[1],
            window_fast=self.macd_params[0], window_sign=self.macd_params[2]
        )
        df['macd'] = macd.macd().shift(1)
        df['macd_signal'] = macd.macd_signal().shift(1)
        
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close=df['close'], window=self.stoch_rsi_params[0],
            smooth1=self.stoch_rsi_params[1]
        )
        df['stoch_k'] = stoch_rsi.stochrsi_k().shift(1)
        df['stoch_d'] = stoch_rsi.stochrsi_d().shift(1)
        
        # Liquidity measure
        df['volume_z'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
        
        return df.dropna()

    def generate_signal(self, idx: int) -> Optional[Dict]:
        """Generate signals with multiple validation layers"""
        if idx < max(self.vwap_window, self.atr_period) + 1:
            return None
            
        current = self.current_data.iloc[idx]
        prev = self.current_data.iloc[idx-1]
        
        # ----------
        # Market Condition Filters
        # ----------
        if not self.validate_market_conditions(current):
            return None
            
        # ----------
        # Core Strategy Logic
        # ----------
        signal = {
            'direction': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'liquidation_price': None,
            'atr': current['atr'],
            'timestamp': current['timestamp']
        }
        
        # Long Conditions
        long_cond = (
            current['close'] > current['vwap'] and
            current['macd'] > current['macd_signal'] and
            current['stoch_k'] < 0.25 and
            current['stoch_k'] > current['stoch_d']
        )
        
        # Short Conditions  
        short_cond = (
            current['close'] < current['vwap'] and
            current['macd'] < current['macd_signal'] and
            current['stoch_k'] > 0.75 and
            current['stoch_k'] < current['stoch_d']
        )
        
        # ----------
        # Position Sizing & Risk Management
        # ----------
        if long_cond or short_cond:
            direction = 'long' if long_cond else 'short'
            entry_price = self.calculate_entry_price(idx, direction)
            stop_loss = self.calculate_stop_loss(current, direction)
            position_size = self.calculate_position_size(entry_price, stop_loss)
            
            if not self.validate_liquidity(current, position_size):
                return None
                
            # Liquidation Check
            liq_price = self.calculate_liquidation_price(
                direction, entry_price, position_size
            )
            
            signal.update({
                'direction': direction,
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(self.calculate_take_profit(
                    entry_price, stop_loss, direction
                )),
                'liquidation_price': float(liq_price),
                'position_size': float(position_size)
            })
            
            return signal
            
        return None

    def calculate_position_size(self, entry_price: Decimal, stop_loss: Decimal) -> Decimal:
        """ATR-based position sizing with risk constraints"""
        risk_amount = self.balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return Decimal('0')
            
        raw_size = risk_amount / price_risk
        return self.apply_size_constraints(raw_size)

    def apply_size_constraints(self, size: Decimal) -> Decimal:
        """Apply exchange and portfolio-level constraints"""
        # Leverage constraint
        max_size = self.balance * self.max_leverage
        size = min(size, max_size)
        
        # Lot size constraints
        step_size = self.symbol_info.get_lot_size()['step_size']
        size = size.quantize(step_size, rounding='ROUND_DOWN')
        
        return size

    def calculate_liquidation_price(self, direction: str, 
                                   entry_price: Decimal,
                                   size: Decimal) -> Decimal:
        """Integrated liquidation price calculation"""
        return LiquidationCalculator.calculate(
            self.symbol_info,
            direction,
            entry_price,
            size,
            self.max_leverage
        )[0]

    def execute_trade(self, signal: Dict) -> Dict:
        """Execute trade with realistic market impact"""
        # ----------
        # Slippage Simulation
        # ----------
        slippage = self.calculate_slippage(signal['atr'])
        execution_price = (
            Decimal(signal['entry_price']) * 
            (Decimal('1') + slippage if signal['direction'] == 'long' 
             else Decimal('1') - slippage)
        )
        
        # ----------
        # Fee Calculation
        # ----------
        fee = execution_price * Decimal(signal['position_size']) * self.taker_fee
        
        # ----------
        # Position Tracking
        # ----------
        position = {
            'entry_time': signal['timestamp'],
            'direction': signal['direction'],
            'entry_price': float(execution_price),
            'size': signal['position_size'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'liquidation_price': signal['liquidation_price'],
            'fee': float(fee)
        }
        
        self.balance -= fee
        self.active_positions.append(position)
        
        return position

    def manage_risk(self, idx: int):
        """Continuous risk management checks"""
        current_candle = self.current_data.iloc[idx]
        
        for position in self.active_positions:
            # Check Liquidation
            if self.check_liquidation(position, current_candle):
                self.close_position(position, current_candle, 'liquidation')
                continue
                
            # Check Stop Loss/Take Profit
            if self.check_exit_conditions(position, current_candle):
                self.close_position(position, current_candle, 'exit_signal')

    def check_liquidation(self, position: Dict, candle: pd.Series) -> bool:
        """Check if price has hit liquidation level"""
        if position['direction'] == 'long':
            return candle['low'] <= position['liquidation_price']
        else:
            return candle['high'] >= position['liquidation_price']

    def run_walk_forward_test(self, num_windows: int = 5):
        """Professional walk-forward validation"""
        window_size = len(self.full_data) // (num_windows + 1)
        
        for i in range(num_windows):
            train_start = i * window_size
            train_end = (i+1) * window_size
            test_start = train_end
            test_end = train_end + window_size
            
            print(f"\n=== Walk-Forward Iteration {i+1} ===")
            print(f"Training: {self.full_data.iloc[train_start]['timestamp']} to {self.full_data.iloc[train_end]['timestamp']}")
            print(f"Testing: {self.full_data.iloc[test_start]['timestamp']} to {self.full_data.iloc[test_end]['timestamp']}")
            
            # Train and optimize
            self.current_data = self.full_data.iloc[train_start:train_end]
            self.run_backtest(phase='train')
            
            # Test
            self.current_data = self.full_data.iloc[test_start:test_end]
            self.balance = self.initial_balance
            self.active_positions = []
            self.run_backtest(phase='test')
            
            # Generate report
            self.analyze_performance()

    def analyze_performance(self):
        """Advanced performance analytics"""
        # Calculate standard metrics
        trades = pd.DataFrame(self.trade_log)
        returns = trades['net_profit'].sum() / self.initial_balance
        
        # Risk-adjusted metrics
        sharpe = self.calculate_sharpe(trades)
        sortino = self.calculate_sortino(trades)
        max_dd = self.calculate_max_drawdown()
        
        # Monte Carlo analysis
        mc_results = self.monte_carlo_simulation(1000)
        
        # Regime analysis
        regime_stats = self.analyze_regime_performance()
        
        print(f"\n=== Advanced Performance Report ===")
        print(f"Sharpe Ratio: {sharpe:.2f} | Sortino Ratio: {sortino:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}% | Profit Factor: {self.calculate_profit_factor():.2f}")
        print(f"Monte Carlo Success Rate: {mc_results['success_rate']:.1%}")
        print(f"Regime Performance:\n{regime_stats}")

    def monte_carlo_simulation(self, iterations: int) -> Dict:
        """Run Monte Carlo simulations on trade sequence"""
        results = []
        for _ in range(iterations):
            random.shuffle(self.trade_history)
            equity = self.simulate_trade_sequence(self.trade_history)
            results.append(equity)
            
        success_rate = sum(1 for x in results if x[-1] > self.initial_balance) / iterations
        return {'success_rate': success_rate}
    
    def validate_dataset(self, df: pd.DataFrame, required_cols: list):
        """Validate that the dataset contains all required columns."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data by forward-filling and dropping remaining NaNs."""
        df = df.fillna(method='ffill').dropna()
        return df

    def calculate_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label market regimes based on volatility or trend."""
        df['regime'] = np.where(
            df['atr'] > df['atr'].rolling(30).mean() * self.volatility_filter,
            'volatile', 'stable'
        )
        return df

    def time_split_data(self, train_ratio: float) -> tuple:
        """Split the data into training and testing sets."""
        split_idx = int(len(self.full_data) * train_ratio)
        train_data = self.full_data.iloc[:split_idx]
        test_data = self.full_data.iloc[split_idx:]
        return train_data, test_data

    def calculate_entry_price(self, idx: int, direction: str) -> Decimal:
        """Calculate the entry price for a trade."""
        current_price = Decimal(self.current_data.iloc[idx]['close'])
        slippage = self.calculate_slippage(self.current_data.iloc[idx]['atr'])
        if direction == 'long':
            return current_price * (Decimal('1') + slippage)
        else:
            return current_price * (Decimal('1') - slippage)

    def calculate_stop_loss(self, current: pd.Series, direction: str) -> Decimal:
        """Calculate the stop loss price based on ATR."""
        atr = Decimal(current['atr'])
        close_price = Decimal(current['close'])
        if direction == 'long':
            return close_price - (self.atr_multiplier * atr)
        else:
            return close_price + (self.atr_multiplier * atr)

    def calculate_take_profit(self, entry_price: Decimal, stop_loss: Decimal, direction: str) -> Decimal:
        """Calculate the take profit price based on risk-reward ratio."""
        risk = abs(entry_price - stop_loss)
        if direction == 'long':
            return entry_price + (self.rr_ratio * risk)
        else:
            return entry_price - (self.rr_ratio * risk)

    def validate_market_conditions(self, current: pd.Series) -> bool:
        """Validate if the market conditions are suitable for trading."""
        return current['regime'] == 'stable' and current['volume_z'] > self.min_liquidity_ratio

    def validate_liquidity(self, current: pd.Series, position_size: Decimal) -> bool:
        """Check if the position size is within acceptable liquidity limits."""
        candle_volume = Decimal(current['volume'])
        return position_size <= candle_volume * self.min_liquidity_ratio

    def calculate_slippage(self, atr: float) -> Decimal:
        """Calculate slippage based on ATR and a volatility multiplier."""
        return self.slippage_model['base'] + (Decimal(atr) * self.slippage_model['volatility_multiplier'])

    def calculate_sharpe(self, trades: pd.DataFrame) -> float:
        """Calculate the Sharpe ratio."""
        returns = trades['net_profit'] / self.initial_balance
        return returns.mean() / returns.std() if returns.std() != 0 else 0

    def calculate_sortino(self, trades: pd.DataFrame) -> float:
        """Calculate the Sortino ratio."""
        returns = trades['net_profit'] / self.initial_balance
        downside_risk = returns[returns < 0].std()
        return returns.mean() / downside_risk if downside_risk != 0 else 0

    def calculate_max_drawdown(self) -> float:
        """Calculate the maximum drawdown."""
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve['peak'] = equity_curve['balance'].cummax()
        equity_curve['drawdown'] = equity_curve['balance'] / equity_curve['peak'] - 1
        return equity_curve['drawdown'].min() * 100

    def calculate_profit_factor(self) -> float:
        """Calculate the profit factor."""
        trades = pd.DataFrame(self.trade_log)
        gross_profit = trades[trades['net_profit'] > 0]['net_profit'].sum()
        gross_loss = abs(trades[trades['net_profit'] < 0]['net_profit'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')

    def run_stress_tests(self):
        """Run stress tests on the strategy."""
        print("\n=== Stress Testing ===")
        for leverage in [5, 10, 20]:
            self.max_leverage = leverage
            self.balance = self.initial_balance
            self.active_positions = []
            self.run_backtest(phase='stress_test')
            print(f"Stress Test with {leverage}x Leverage:")
            self.analyze_performance()

    def simulate_trade_sequence(self, trades: deque) -> list:
        """Simulate a sequence of trades for Monte Carlo analysis."""
        equity = [self.initial_balance]
        for trade in trades:
            equity.append(equity[-1] + trade['net_profit'])
        return equity

    def analyze_regime_performance(self) -> pd.DataFrame:
        """Analyze performance by market regime."""
        trades = pd.DataFrame(self.trade_log)
        trades['regime'] = trades['timestamp'].apply(
            lambda ts: self.full_data.loc[self.full_data['timestamp'] == ts, 'regime'].values[0]
        )
        return trades.groupby('regime')['net_profit'].sum()

# ----------
# Main Execution
# ----------
if __name__ == "__main__":
    backtester = ProfessionalBacktester(
        csv_path="data/SOLUSDT30_1h.csv",
        initial_balance=Decimal("250"),
        risk_per_trade=Decimal("0.02"),
        max_leverage=20
    )
    
    # Run complete validation suite
    backtester.run_walk_forward_test()
    backtester.analyze_performance()
    
    # Generate stress tests
    backtester.run_stress_tests()