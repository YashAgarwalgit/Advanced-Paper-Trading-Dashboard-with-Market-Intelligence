"""
Pre-trade risk validation and checks
"""
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from config import Config
from utils.helpers import safe_float

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_order_value: float = Config.MAX_ORDER_VALUE
    max_daily_turnover: float = Config.MAX_DAILY_TURNOVER
    max_position_weight: float = Config.MAX_POSITION_WEIGHT
    max_sector_weight: float = Config.MAX_SECTOR_WEIGHT
    max_leverage: float = 2.0
    max_concentration: float = 0.2  # 20% max in single position
    var_limit: Optional[float] = None
    tracking_error_limit: Optional[float] = None

class TradingRiskValidator:
    """
    Comprehensive pre-trade risk validation system
    Features: Position limits, concentration checks, leverage monitoring
    """
    
    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(__name__)
        
        # Sector mappings for concentration checks
        self.sector_mappings = self._load_sector_mappings()
    
    def validate_trade(
        self, 
        portfolio_data: Dict[str, Any],
        ticker: str,
        action: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET"
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Comprehensive pre-trade risk validation
        
        Args:
            portfolio_data: Portfolio state
            ticker: Asset ticker
            action: BUY or SELL
            quantity: Trade quantity
            price: Trade price
            order_type: Order type
            
        Returns:
            (is_valid, risk_warnings, risk_metrics)
        """
        
        try:
            risk_warnings = []
            risk_metrics = {}
            
            # Basic parameter validation
            basic_check = self._validate_basic_parameters(ticker, action, quantity, price)
            if not basic_check[0]:
                return False, [basic_check[1]], {}
            
            # Order size limits
            size_check = self._check_order_size_limits(quantity, price)
            if not size_check[0]:
                return False, [size_check[1]], {}
            if size_check[1]:
                risk_warnings.append(size_check[1])
            
            # Portfolio structure validation
            structure_check = self._validate_portfolio_structure(portfolio_data)
            if not structure_check[0]:
                return False, [structure_check[1]], {}
            
            # Cash and margin checks
            funding_check = self._check_funding_requirements(portfolio_data, action, quantity, price)
            if not funding_check[0]:
                return False, [funding_check[1]], {}
            if funding_check[1]:
                risk_warnings.append(funding_check[1])
            
            # Position size and concentration checks
            concentration_check = self._check_concentration_limits(
                portfolio_data, ticker, action, quantity, price
            )
            if not concentration_check[0]:
                return False, [concentration_check[1]], {}
            if concentration_check[1]:
                risk_warnings.extend(concentration_check[1])
            
            # Sector concentration checks
            sector_check = self._check_sector_concentration(
                portfolio_data, ticker, action, quantity, price
            )
            if not sector_check[0]:
                return False, [sector_check[1]], {}
            if sector_check[1]:
                risk_warnings.extend(sector_check[1])
            
            # Leverage checks
            leverage_check = self._check_leverage_limits(portfolio_data, action, quantity, price)
            if not leverage_check[0]:
                return False, [leverage_check[1]], {}
            if leverage_check[1]:
                risk_warnings.append(leverage_check[1])
            
            # Daily turnover limits
            turnover_check = self._check_daily_turnover(portfolio_data, quantity, price)
            if not turnover_check[0]:
                return False, [turnover_check[1]], {}
            if turnover_check[1]:
                risk_warnings.append(turnover_check[1])
            
            # Calculate risk metrics
            risk_metrics = self._calculate_post_trade_risk_metrics(
                portfolio_data, ticker, action, quantity, price
            )
            
            # Advanced risk checks
            advanced_warnings = self._run_advanced_risk_checks(portfolio_data, risk_metrics)
            risk_warnings.extend(advanced_warnings)
            
            return True, risk_warnings, risk_metrics
            
        except Exception as e:
            self.logger.error(f"Risk validation failed: {e}")
            return False, [f"Risk validation error: {e}"], {}
    
    def _validate_basic_parameters(self, ticker: str, action: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Basic parameter validation"""
        
        if not ticker or not ticker.strip():
            return False, "Ticker cannot be empty"
        
        if action not in ["BUY", "SELL"]:
            return False, "Action must be BUY or SELL"
        
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        return True, ""
    
    def _check_order_size_limits(self, quantity: float, price: float) -> Tuple[bool, Optional[str]]:
        """Check order size against limits"""
        
        order_value = quantity * price
        
        if order_value > self.risk_limits.max_order_value:
            return False, f"Order value {order_value:,.0f} exceeds maximum {self.risk_limits.max_order_value:,.0f}"
        
        # Warning for large orders
        if order_value > self.risk_limits.max_order_value * 0.5:
            return True, f"Large order warning: {order_value:,.0f} (50%+ of limit)"
        
        return True, None
    
    def _validate_portfolio_structure(self, portfolio_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate portfolio data structure"""
        
        required_fields = ['metadata', 'balances', 'positions']
        
        for field in required_fields:
            if field not in portfolio_data:
                return False, f"Missing portfolio field: {field}"
        
        # Validate balances structure
        balances = portfolio_data['balances']
        required_balance_fields = ['cash', 'total_value', 'market_value']
        
        for field in required_balance_fields:
            if field not in balances:
                return False, f"Missing balance field: {field}"
            
            if not isinstance(balances[field], (int, float)):
                return False, f"Invalid balance type for {field}"
        
        return True, ""
    
    def _check_funding_requirements(
        self, 
        portfolio_data: Dict[str, Any], 
        action: str, 
        quantity: float, 
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """Check cash and margin requirements"""
        
        if action != "BUY":
            return True, None  # No funding check needed for sales
        
        balances = portfolio_data['balances']
        trade_value = quantity * price
        
        # Estimate total cost with commission and slippage
        estimated_commission = trade_value * 0.0005  # 0.05%
        estimated_slippage = trade_value * 0.001    # 0.1%
        total_cost = trade_value + estimated_commission + estimated_slippage
        
        available_cash = safe_float(balances.get('cash', 0))
        available_margin = safe_float(balances.get('available_margin', 0))
        total_available = available_cash + available_margin
        
        if total_cost > total_available:
            return False, f"Insufficient funds: need {total_cost:,.0f}, have {total_available:,.0f}"
        
        # Warning if using significant margin
        if total_cost > available_cash and available_margin > 0:
            margin_usage = (total_cost - available_cash) / available_margin * 100
            if margin_usage > 80:
                return True, f"High margin usage warning: {margin_usage:.1f}%"
        
        return True, None
    
    def _check_concentration_limits(
        self, 
        portfolio_data: Dict[str, Any], 
        ticker: str, 
        action: str, 
        quantity: float, 
        price: float
    ) -> Tuple[bool, List[str]]:
        """Check position concentration limits"""
        
        balances = portfolio_data['balances']
        positions = portfolio_data.get('positions', {})
        total_value = safe_float(balances.get('total_value', 0))
        
        if total_value <= 0:
            return True, []  # Cannot calculate concentration for zero portfolio
        
        warnings = []
        
        # Calculate post-trade position value
        current_position = positions.get(ticker, {})
        current_quantity = safe_float(current_position.get('quantity', 0))
        
        if action == "BUY":
            new_quantity = current_quantity + quantity
            new_position_value = new_quantity * price
        else:  # SELL
            new_quantity = current_quantity - quantity
            new_position_value = max(0, new_quantity * price)
        
        # Calculate concentration percentage
        concentration_pct = new_position_value / total_value
        
        # Check against limits
        if concentration_pct > self.risk_limits.max_concentration:
            return False, f"Position concentration {concentration_pct:.1%} exceeds limit {self.risk_limits.max_concentration:.1%}"
        
        # Warning for high concentration
        if concentration_pct > self.risk_limits.max_concentration * 0.8:
            warnings.append(f"High concentration warning: {concentration_pct:.1%} in {ticker}")
        
        return True, warnings
    
    def _check_sector_concentration(
        self, 
        portfolio_data: Dict[str, Any], 
        ticker: str, 
        action: str, 
        quantity: float, 
        price: float
    ) -> Tuple[bool, List[str]]:
        """Check sector concentration limits"""
        
        sector = self.sector_mappings.get(ticker.replace('.NS', ''), 'Other')
        balances = portfolio_data['balances']
        positions = portfolio_data.get('positions', {})
        total_value = safe_float(balances.get('total_value', 0))
        
        if total_value <= 0:
            return True, []
        
        warnings = []
        
        # Calculate current sector exposure
        sector_value = 0
        for pos_ticker, position in positions.items():
            pos_sector = self.sector_mappings.get(pos_ticker.replace('.NS', ''), 'Other')
            if pos_sector == sector:
                pos_value = safe_float(position.get('market_value', 0))
                if pos_value <= 0:  # Fallback calculation
                    pos_value = safe_float(position.get('quantity', 0)) * safe_float(position.get('last_price', 0))
                sector_value += pos_value
        
        # Add trade impact
        trade_value = quantity * price
        if action == "BUY":
            sector_value += trade_value
        else:  # SELL
            sector_value = max(0, sector_value - trade_value)
        
        sector_concentration = sector_value / total_value
        
        # Check against limits
        if sector_concentration > self.risk_limits.max_sector_weight:
            return False, f"Sector {sector} concentration {sector_concentration:.1%} exceeds limit {self.risk_limits.max_sector_weight:.1%}"
        
        # Warning for high sector concentration
        if sector_concentration > self.risk_limits.max_sector_weight * 0.8:
            warnings.append(f"High sector concentration: {sector_concentration:.1%} in {sector}")
        
        return True, warnings
    
    def _check_leverage_limits(
        self, 
        portfolio_data: Dict[str, Any], 
        action: str, 
        quantity: float, 
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """Check leverage limits"""
        
        balances = portfolio_data['balances']
        cash = safe_float(balances.get('cash', 0))
        market_value = safe_float(balances.get('market_value', 0))
        used_margin = safe_float(balances.get('used_margin', 0))
        
        # Calculate post-trade values
        trade_value = quantity * price
        
        if action == "BUY":
            new_market_value = market_value + trade_value
            # Assume margin is used if insufficient cash
            if trade_value > cash:
                new_used_margin = used_margin + (trade_value - cash)
                new_cash = 0
            else:
                new_cash = cash - trade_value
                new_used_margin = used_margin
        else:  # SELL
            new_market_value = max(0, market_value - trade_value)
            new_cash = cash + trade_value
            new_used_margin = used_margin
        
        # Calculate leverage (total exposure / equity)
        equity = new_cash + new_market_value
        if equity <= 0:
            return False, "Negative equity detected"
        
        leverage = (new_market_value + new_used_margin) / equity
        
        if leverage > self.risk_limits.max_leverage:
            return False, f"Leverage {leverage:.2f}x exceeds limit {self.risk_limits.max_leverage:.2f}x"
        
        # Warning for high leverage
        if leverage > self.risk_limits.max_leverage * 0.8:
            return True, f"High leverage warning: {leverage:.2f}x"
        
        return True, None
    
    def _check_daily_turnover(
        self, 
        portfolio_data: Dict[str, Any], 
        quantity: float, 
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """Check daily turnover limits"""
        
        trade_value = quantity * price
        
        # Calculate today's turnover from transactions
        today = datetime.now().date()
        transactions = portfolio_data.get('transactions', [])
        
        daily_turnover = 0
        for transaction in transactions:
            try:
                trans_date = datetime.fromisoformat(transaction['timestamp']).date()
                if trans_date == today:
                    daily_turnover += safe_float(transaction.get('trade_value', 0))
            except (ValueError, KeyError):
                continue
        
        new_daily_turnover = daily_turnover + trade_value
        
        if new_daily_turnover > self.risk_limits.max_daily_turnover:
            return False, f"Daily turnover {new_daily_turnover:,.0f} exceeds limit {self.risk_limits.max_daily_turnover:,.0f}"
        
        # Warning for high turnover
        if new_daily_turnover > self.risk_limits.max_daily_turnover * 0.8:
            return True, f"High daily turnover: {new_daily_turnover:,.0f} (80%+ of limit)"
        
        return True, None
    
    def _calculate_post_trade_risk_metrics(
        self, 
        portfolio_data: Dict[str, Any], 
        ticker: str, 
        action: str, 
        quantity: float, 
        price: float
    ) -> Dict[str, Any]:
        """Calculate risk metrics after hypothetical trade"""
        
        try:
            balances = portfolio_data['balances']
            positions = portfolio_data.get('positions', {}).copy()
            
            # Simulate trade impact on positions
            trade_value = quantity * price
            
            if action == "BUY":
                if ticker in positions:
                    pos = positions[ticker]
                    total_cost = (pos['avg_price'] * pos['quantity']) + trade_value
                    new_quantity = pos['quantity'] + quantity
                    positions[ticker] = {
                        **pos,
                        'quantity': new_quantity,
                        'avg_price': total_cost / new_quantity,
                        'market_value': new_quantity * price
                    }
                else:
                    positions[ticker] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'market_value': trade_value
                    }
            else:  # SELL
                if ticker in positions:
                    pos = positions[ticker]
                    new_quantity = pos['quantity'] - quantity
                    if new_quantity <= 0:
                        del positions[ticker]
                    else:
                        positions[ticker] = {
                            **pos,
                            'quantity': new_quantity,
                            'market_value': new_quantity * price
                        }
            
            # Calculate risk metrics
            total_market_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            # Position weights
            position_weights = {}
            for pos_ticker, pos in positions.items():
                weight = pos.get('market_value', 0) / total_market_value if total_market_value > 0 else 0
                position_weights[pos_ticker] = weight
            
            # Concentration measures
            max_weight = max(position_weights.values()) if position_weights else 0
            herfindahl_index = sum(w**2 for w in position_weights.values())
            effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
            
            # Sector diversification
            sector_weights = {}
            for pos_ticker, weight in position_weights.items():
                sector = self.sector_mappings.get(pos_ticker.replace('.NS', ''), 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            max_sector_weight = max(sector_weights.values()) if sector_weights else 0
            
            return {
                'total_positions': len(positions),
                'max_position_weight': max_weight,
                'max_sector_weight': max_sector_weight,
                'herfindahl_index': herfindahl_index,
                'effective_positions': effective_positions,
                'sector_count': len(sector_weights),
                'position_weights': position_weights,
                'sector_weights': sector_weights
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _run_advanced_risk_checks(
        self, 
        portfolio_data: Dict[str, Any], 
        risk_metrics: Dict[str, Any]
    ) -> List[str]:
        """Run advanced risk checks"""
        
        warnings = []
        
        try:
            # Diversification warnings
            effective_positions = risk_metrics.get('effective_positions', 0)
            if effective_positions < 5:
                warnings.append(f"Low diversification: {effective_positions:.1f} effective positions")
            
            # Sector diversification
            sector_count = risk_metrics.get('sector_count', 0)
            if sector_count < 3:
                warnings.append(f"Low sector diversification: {sector_count} sectors")
            
            # Check for correlations (simplified)
            position_weights = risk_metrics.get('position_weights', {})
            it_exposure = sum(w for ticker, w in position_weights.items() 
                             if self.sector_mappings.get(ticker.replace('.NS', ''), '') == 'IT')
            
            if it_exposure > 0.5:
                warnings.append(f"High IT sector exposure: {it_exposure:.1%}")
            
        except Exception as e:
            self.logger.warning(f"Advanced risk check error: {e}")
        
        return warnings
    
    def _load_sector_mappings(self) -> Dict[str, str]:
        """Load sector mappings for tickers"""
        
        return {
            # IT Sector
            "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
            
            # Banking Sector  
            "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", 
            "KOTAKBANK": "Banking", "AXISBANK": "Banking",
            
            # Energy & Oil
            "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
            
            # Automotive
            "MARUTI": "Auto", "TATAMOTORS": "Auto", "BAJAJ-AUTO": "Auto",
            
            # FMCG
            "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
            
            # Pharmaceuticals
            "SUNPHARMA": "Pharma", "DRREDDY": "Pharma",
            
            # Metals & Mining
            "TATASTEEL": "Metals", "HINDALCO": "Metals"
        }
    
    def get_risk_summary(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive risk summary for portfolio"""
        
        try:
            balances = portfolio_data['balances']
            positions = portfolio_data.get('positions', {})
            
            total_value = safe_float(balances.get('total_value', 0))
            cash = safe_float(balances.get('cash', 0))
            market_value = safe_float(balances.get('market_value', 0))
            used_margin = safe_float(balances.get('used_margin', 0))
            
            # Basic metrics
            cash_allocation = cash / total_value if total_value > 0 else 0
            equity_allocation = market_value / total_value if total_value > 0 else 0
            leverage = (market_value + used_margin) / (cash + market_value) if (cash + market_value) > 0 else 0
            
            # Position analysis
            position_weights = {}
            sector_weights = {}
            
            for ticker, position in positions.items():
                pos_value = safe_float(position.get('market_value', 0))
                if pos_value > 0 and total_value > 0:
                    weight = pos_value / total_value
                    position_weights[ticker] = weight
                    
                    sector = self.sector_mappings.get(ticker.replace('.NS', ''), 'Other')
                    sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            # Concentration measures
            max_position = max(position_weights.values()) if position_weights else 0
            max_sector = max(sector_weights.values()) if sector_weights else 0
            
            # Risk status
            risk_flags = []
            if max_position > self.risk_limits.max_concentration:
                risk_flags.append("Position concentration exceeded")
            if max_sector > self.risk_limits.max_sector_weight:
                risk_flags.append("Sector concentration exceeded")
            if leverage > self.risk_limits.max_leverage:
                risk_flags.append("Leverage limit exceeded")
            
            return {
                'total_value': total_value,
                'cash_allocation': cash_allocation,
                'equity_allocation': equity_allocation,
                'leverage': leverage,
                'max_position_weight': max_position,
                'max_sector_weight': max_sector,
                'number_of_positions': len(positions),
                'number_of_sectors': len(sector_weights),
                'position_weights': position_weights,
                'sector_weights': sector_weights,
                'risk_flags': risk_flags,
                'risk_score': len(risk_flags)  # Simple risk scoring
            }
            
        except Exception as e:
            self.logger.error(f"Risk summary calculation failed: {e}")
            return {"error": str(e)}
