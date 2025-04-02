import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
from decimal import Decimal, getcontext
from typing import Literal, Optional, Tuple, Dict, Any

# ---------------------------
# 1. Initialize the Binance Futures Client
# ---------------------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Set decimal precision for financial calculations
getcontext().prec = 8

# Create a client (mainnet by default)
client = Client(API_KEY, API_SECRET)

# ---------------------------
# 2. Enhanced Symbol Information Processor
# ---------------------------
class SymbolData:
    """Process and store all relevant symbol information."""
    
    def __init__(self, symbol_info: dict):
        self.raw_data = symbol_info
        self.symbol = symbol_info['symbol']
        self.base_asset = symbol_info['baseAsset']
        self.quote_asset = symbol_info['quoteAsset']
        self.margin_asset = symbol_info['marginAsset']
        self.price_precision = int(symbol_info['pricePrecision'])
        self.quantity_precision = int(symbol_info['quantityPrecision'])
        self.maint_margin_percent = Decimal(symbol_info['maintMarginPercent']) / Decimal('100')
        self.required_margin_percent = Decimal(symbol_info['requiredMarginPercent']) / Decimal('100')
        self.liquidation_fee = Decimal(symbol_info['liquidationFee']) / Decimal('100')
        
        # Process filters
        self.filters = {f['filterType']: f for f in symbol_info['filters']}
        
        # Leverage filter (might not exist for all symbols)
        self.max_leverage = self._get_max_leverage()
        
    def _get_max_leverage(self) -> int:
        """Extract max leverage from filters with fallback."""
        if 'LEVERAGE' in self.filters:
            return int(self.filters['LEVERAGE']['maxLeverage'])
        return 75  # Conservative default for SOL
    
    def get_price_filter(self) -> Dict[str, Decimal]:
        """Get price filter parameters."""
        pf = self.filters.get('PRICE_FILTER', {})
        return {
            'min_price': Decimal(pf.get('minPrice', '0')),
            'max_price': Decimal(pf.get('maxPrice', '999999')),
            'tick_size': Decimal(pf.get('tickSize', '0.0001'))
        }
    
    def get_lot_size(self) -> Dict[str, Decimal]:
        """Get lot size parameters."""
        ls = self.filters.get('LOT_SIZE', {})
        return {
            'min_qty': Decimal(ls.get('minQty', '1')),
            'max_qty': Decimal(ls.get('maxQty', '1000000')),
            'step_size': Decimal(ls.get('stepSize', '1'))
        }

def get_symbol_data(symbol: str) -> Optional[SymbolData]:
    """Fetch and process symbol information."""
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                return SymbolData(s)
        return None
    except BinanceAPIException as e:
        print(f"Error fetching exchange info for {symbol}: {e}")
        return None

# ---------------------------
# 3. Advanced Liquidation Price Calculator
# ---------------------------
class LiquidationCalculator:
    """Handle all liquidation price calculations with proper rounding."""
    
    @staticmethod
    def calculate(
        symbol_data: SymbolData,
        position_side: Literal['long', 'short'],
        entry_price: Decimal,
        quantity: Decimal,
        leverage: int,
        is_isolated: bool = True
    ) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """
        Calculate liquidation price with all Binance-specific parameters.
        
        Returns:
            (liquidation_price, metadata)
        """
        try:
            # Validate inputs
            if leverage < 1 or leverage > symbol_data.max_leverage:
                raise ValueError(f"Leverage must be between 1 and {symbol_data.max_leverage}")
                
            if entry_price <= 0 or quantity <= 0:
                raise ValueError("Price and quantity must be positive")
            
            # Get required parameters
            mmr = symbol_data.maint_margin_percent
            liquidation_fee = symbol_data.liquidation_fee
            price_filter = symbol_data.get_price_filter()
            
            # Calculate position values
            position_value = entry_price * quantity
            initial_margin = position_value / Decimal(leverage)
            
            # Calculate liquidation price
            if position_side.lower() == 'long':
                numerator = position_value * (Decimal('1') + liquidation_fee) - initial_margin
                denominator = quantity * (Decimal('1') - mmr + liquidation_fee)
                liq_price = numerator / denominator
            else:
                numerator = position_value * (Decimal('1') - liquidation_fee) + initial_margin
                denominator = quantity * (Decimal('1') + mmr - liquidation_fee)
                liq_price = numerator / denominator
            
            # Apply price filter constraints
            liq_price = max(min(liq_price, price_filter['max_price']), price_filter['min_price'])
            
            # Round to appropriate precision
            liq_price = liq_price.quantize(Decimal('1').scaleb(-symbol_data.price_precision))
            
            # Prepare metadata
            metadata = {
                'position_value': float(position_value),
                'initial_margin': float(initial_margin),
                'maintenance_margin_rate': float(mmr),
                'liquidation_fee_rate': float(liquidation_fee),
                'leverage': leverage,
                'is_isolated': is_isolated,
                'price_precision': symbol_data.price_precision,
                'calculation_method': 'binance_advanced'
            }
            
            return liq_price, metadata
            
        except Exception as e:
            print(f"Liquidation calculation error: {str(e)}")
            return None, {'error': str(e)}

# ---------------------------
# 4. Comprehensive Trade Simulator
# ---------------------------
class TradeSimulator:
    """Handle complete trade simulations with all Binance parameters."""
    
    @staticmethod
    def simulate(
        symbol_data: SymbolData,
        position_side: Literal['long', 'short'],
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        leverage: int,
        funding_rate: Decimal = Decimal('0.0001'),
        funding_periods: int = 1,
        taker_fee: Decimal = Decimal('0.0004'),
        maker_fee: Decimal = Decimal('0.0002')
    ) -> Dict[str, Any]:
        """Run complete trade simulation with all fees and constraints."""
        result = {
            'success': False,
            'error': None,
            'gross_pnl': Decimal('0'),
            'net_pnl': Decimal('0'),
            'total_fees': Decimal('0'),
            'funding_cost': Decimal('0'),
            'roi': Decimal('0'),
            'liquidation_price': None,
            'metadata': {}
        }
        
        try:
            # Validate inputs
            if entry_price <= 0 or exit_price <= 0 or quantity <= 0:
                raise ValueError("Prices and quantity must be positive")
                
            if leverage < 1 or leverage > symbol_data.max_leverage:
                raise ValueError(f"Leverage must be between 1 and {symbol_data.max_leverage}")
            
            # Verify quantity meets lot size requirements
            lot_size = symbol_data.get_lot_size()
            if quantity < lot_size['min_qty'] or quantity > lot_size['max_qty']:
                raise ValueError(f"Quantity must be between {lot_size['min_qty']} and {lot_size['max_qty']}")
            
            # Calculate position values
            position_value = entry_price * quantity
            initial_margin = position_value / Decimal(leverage)
            
            # Calculate PnL
            if position_side.lower() == 'long':
                gross_pnl = (exit_price - entry_price) * quantity
            else:
                gross_pnl = (entry_price - exit_price) * quantity
            
            # Calculate fees (taker on entry, maker on exit)
            entry_fee = position_value * taker_fee
            exit_fee = exit_price * quantity * maker_fee
            total_fees = entry_fee + exit_fee
            
            # Calculate funding costs
            funding_cost = position_value * funding_rate * Decimal(funding_periods)
            
            # Calculate net PnL and ROI
            net_pnl = gross_pnl - total_fees - funding_cost
            roi = (net_pnl / initial_margin) * Decimal('100')
            
            # Calculate liquidation price
            liq_price, liq_metadata = LiquidationCalculator.calculate(
                symbol_data, position_side, entry_price, quantity, leverage
            )
            
            # Prepare result
            result.update({
                'success': True,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'total_fees': total_fees,
                'funding_cost': funding_cost,
                'roi': roi,
                'liquidation_price': liq_price,
                'metadata': {
                    'position_value': position_value,
                    'initial_margin': initial_margin,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'leverage': leverage,
                    'symbol': symbol_data.symbol,
                    'liquidation_metadata': liq_metadata
                }
            })
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

# ---------------------------
# 5. Example Usage with SOLUSDT
# ---------------------------
if __name__ == "__main__":
    symbol = "SOLUSDT"
    
    try:
        # Get and process symbol data
        sol_data = SymbolData({
            'symbol': 'SOLUSDT',
            'pair': 'SOLUSDT',
            'contractType': 'PERPETUAL',
            'deliveryDate': 4133404800000,
            'onboardDate': 1569398400000,
            'status': 'TRADING',
            'maintMarginPercent': '2.5000',
            'requiredMarginPercent': '5.0000',
            'baseAsset': 'SOL',
            'quoteAsset': 'USDT',
            'marginAsset': 'USDT',
            'pricePrecision': 4,
            'quantityPrecision': 0,
            'baseAssetPrecision': 8,
            'quotePrecision': 8,
            'underlyingType': 'COIN',
            'underlyingSubType': ['Layer-1'],
            'triggerProtect': '0.0500',
            'liquidationFee': '0.015000',
            'marketTakeBound': '0.05',
            'maxMoveOrderLimit': 10000,
            'filters': [
                {'tickSize': '0.0100', 'filterType': 'PRICE_FILTER', 'minPrice': '0.4200', 'maxPrice': '6857'},
                {'maxQty': '1000000', 'filterType': 'LOT_SIZE', 'minQty': '1', 'stepSize': '1'},
                {'minQty': '1', 'maxQty': '5000', 'filterType': 'MARKET_LOT_SIZE', 'stepSize': '1'},
                {'limit': 200, 'filterType': 'MAX_NUM_ORDERS'},
                {'limit': 10, 'filterType': 'MAX_NUM_ALGO_ORDERS'},
                {'notional': '5', 'filterType': 'MIN_NOTIONAL'},
                {'multiplierDecimal': '4', 'filterType': 'PERCENT_PRICE', 'multiplierUp': '1.0500', 'multiplierDown': '0.9500'}
            ],
            'orderTypes': ['LIMIT', 'MARKET', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET'],
            'timeInForce': ['GTC', 'IOC', 'FOK', 'GTX', 'GTD']
        })
        
        print(f"\nSymbol Analysis for {sol_data.symbol}:")
        print(f"- Base Asset: {sol_data.base_asset}")
        print(f"- Quote Asset: {sol_data.quote_asset}")
        print(f"- Max Leverage: {sol_data.max_leverage}x")
        print(f"- Maintenance Margin: {sol_data.maint_margin_percent * 100:.2f}%")
        print(f"- Liquidation Fee: {sol_data.liquidation_fee * 100:.3f}%")
        
        # Example trade parameters
        position_side = 'long'
        entry_price = Decimal('150.50')
        exit_price = Decimal('155.25')
        quantity = Decimal('10')  # SOL (must be integer for SOLUSDT)
        leverage = min(20, sol_data.max_leverage)
        
        # Calculate liquidation price
        liq_price, liq_metadata = LiquidationCalculator.calculate(
            sol_data, position_side, entry_price, quantity, leverage
        )
        
        print(f"\nLiquidation Price Calculation:")
        print(f"- Position: {position_side.upper()} {quantity} {sol_data.base_asset} at {entry_price}")
        print(f"- Leverage: {leverage}x")
        if liq_price is not None:
            print(f"- Estimated Liquidation Price: {liq_price:.4f} {sol_data.quote_asset}")
            print("- Calculation Details:")
            for k, v in liq_metadata.items():
                if isinstance(v, float):
                    print(f"  - {k}: {v:.6f}")
                else:
                    print(f"  - {k}: {v}")
        else:
            print("- Could not calculate liquidation price")
            if 'error' in liq_metadata:
                print(f"- Error: {liq_metadata['error']}")
        
        # Simulate trade
        trade_result = TradeSimulator.simulate(
            sol_data, position_side, entry_price, exit_price, quantity, leverage
        )
        
        print("\nTrade Simulation Results:")
        if trade_result['success']:
            print(f"- Gross PnL: {trade_result['gross_pnl']:.4f} {sol_data.quote_asset}")
            print(f"- Net PnL: {trade_result['net_pnl']:.4f} {sol_data.quote_asset}")
            print(f"- ROI: {trade_result['roi']:.2f}%")
            print(f"- Total Fees: {trade_result['total_fees']:.4f} {sol_data.quote_asset}")
            print(f"- Funding Costs: {trade_result['funding_cost']:.4f} {sol_data.quote_asset}")
            print(f"- Liquidation Price: {trade_result['liquidation_price'] or 'N/A'}")
        else:
            print(f"- Simulation failed: {trade_result['error']}")
            
    except Exception as e:
        print(f"\nFatal error in main execution: {str(e)}")