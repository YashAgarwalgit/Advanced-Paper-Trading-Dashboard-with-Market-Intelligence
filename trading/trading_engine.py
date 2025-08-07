"""
Advanced trading execution engine with realistic modeling
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple
import logging
from config import Config
from utils.helpers import get_timestamp_iso
from portfolio.transaction_manager import PortfolioTransactionManager

class AdvancedTradingEngine:
    """
    Professional trading execution engine
    Features: Realistic slippage modeling, commission calculations, atomic execution
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transaction_manager = PortfolioTransactionManager()
        
        # Default execution parameters
        self.default_slippage_pct = Config.DEFAULT_SLIPPAGE_PCT
        self.default_commission_bps = Config.DEFAULT_COMMISSION_BPS
        
        # Market impact models
        self.base_slippage_bps = 5  # Base slippage in basis points
        self.quantity_impact_factor = 0.0001  # Quantity impact factor
        
    async def execute_trade_async(
        self, 
        portfolio: Dict[str, Any], 
        ticker: str, 
        action: str, 
        quantity: float, 
        market_price: float,
        max_slippage_pct: float = None
    ) -> Dict[str, Any]:
        """
        Async trade execution with comprehensive modeling
        
        Args:
            portfolio: Portfolio data
            ticker: Asset ticker
            action: BUY or SELL
            quantity: Trade quantity
            market_price: Current market price
            max_slippage_pct: Maximum allowed slippage
            
        Returns:
            Execution result with details
        """
        
        try:
            # Use async execution in thread pool for atomic operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.execute_trade,
                portfolio, ticker, action, quantity, market_price, max_slippage_pct
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Async trade execution failed: {e}")
            return {"success": False, "message": f"Async execution error: {e}"}
    
    def execute_trade(
        self,
        portfolio: Dict[str, Any],
        ticker: str,
        action: str,
        quantity: float,
        market_price: float,
        max_slippage_pct: float = None,
        commission_bps: float = None
    ) -> Dict[str, Any]:
        """
        Execute trade with atomic transactions and realistic modeling
        
        Args:
            portfolio: Portfolio data or portfolio name
            ticker: Asset ticker
            action: BUY or SELL
            quantity: Trade quantity
            market_price: Current market price
            max_slippage_pct: Maximum allowed slippage percentage
            commission_bps: Commission in basis points
            
        Returns:
            Detailed execution result
        """
        
        # Handle portfolio name vs portfolio data
        if isinstance(portfolio, str):
            portfolio_name = portfolio
            # Would load portfolio from storage
        else:
            portfolio_name = portfolio.get('metadata', {}).get('portfolio_name')
        
        if not portfolio_name:
            return {"success": False, "message": "Invalid portfolio reference"}
        
        try:
            with self.transaction_manager.atomic_portfolio_update(portfolio_name) as portfolio_data:
                
                # Validate inputs
                validation_result = self._validate_trade_inputs(
                    portfolio_data, ticker, action, quantity, market_price
                )
                
                if not validation_result["valid"]:
                    return {"success": False, "message": validation_result["error"]}
                
                # Calculate execution price with slippage
                execution_price = self._calculate_execution_price(
                    market_price, quantity, action, max_slippage_pct or self.default_slippage_pct
                )
                
                # Calculate commission
                trade_value = quantity * execution_price
                commission = self._calculate_commission(trade_value, commission_bps)
                
                # Execute trade in portfolio
                execution_result = self._execute_in_portfolio(
                    portfolio_data, ticker, action, quantity, execution_price, commission
                )
                
                if execution_result["success"]:
                    # Add to transaction history
                    transaction = {
                        "timestamp": get_timestamp_iso(),
                        "ticker": ticker,
                        "action": action,
                        "quantity": quantity,
                        "price": execution_price,
                        "commission": commission,
                        "order_type": "MARKET",
                        "trade_value": trade_value,
                        "slippage": abs(execution_price - market_price),
                        "slippage_pct": abs(execution_price - market_price) / market_price * 100
                    }
                    
                    portfolio_data.setdefault('transactions', []).insert(0, transaction)
                    
                    # Update equity history
                    self._update_equity_history(portfolio_data)
                    
                    return {
                        "success": True,
                        "message": f"Successfully {action.lower()}ed {quantity} shares of {ticker}",
                        "execution_price": execution_price,
                        "commission": commission,
                        "slippage": abs(execution_price - market_price),
                        "slippage_pct": abs(execution_price - market_price) / market_price * 100,
                        "trade_value": trade_value,
                        "total_cost": trade_value + commission if action == "BUY" else trade_value - commission
                    }
                else:
                    return execution_result
                    
        except Exception as e:
            self.logger.error(f"Trade execution failed for {ticker}: {e}")
            return {"success": False, "message": f"Execution error: {e}"}
    
    def _validate_trade_inputs(
        self, 
        portfolio_data: Dict[str, Any], 
        ticker: str, 
        action: str, 
        quantity: float, 
        market_price: float
    ) -> Dict[str, Any]:
        """Comprehensive trade input validation"""
        
        # Basic parameter validation
        if quantity <= 0:
            return {"valid": False, "error": "Quantity must be positive"}
        
        if market_price <= 0:
            return {"valid": False, "error": "Market price must be positive"}
        
        if action not in ["BUY", "SELL"]:
            return {"valid": False, "error": "Action must be BUY or SELL"}
        
        # Portfolio validation
        if 'balances' not in portfolio_data:
            return {"valid": False, "error": "Invalid portfolio structure"}
        
        balances = portfolio_data['balances']
        positions = portfolio_data.get('positions', {})
        
        # Trade size limits
        trade_value = quantity * market_price
        if trade_value > Config.MAX_ORDER_VALUE:
            return {
                "valid": False, 
                "error": f"Trade value {trade_value:,.0f} exceeds maximum {Config.MAX_ORDER_VALUE:,.0f}"
            }
        
        # Position and cash validation
        if action == "BUY":
            # Estimate total cost including commission and slippage
            estimated_cost = trade_value * 1.015  # Add 1.5% buffer for costs
            available_cash = balances.get('cash', 0)
            
            if estimated_cost > available_cash:
                # Check available margin
                available_margin = balances.get('available_margin', 0)
                if estimated_cost > available_cash + available_margin:
                    return {
                        "valid": False,
                        "error": f"Insufficient funds: need {estimated_cost:,.0f}, have {available_cash:,.0f} cash + {available_margin:,.0f} margin"
                    }
        
        elif action == "SELL":
            current_position = positions.get(ticker, {})
            available_quantity = current_position.get('quantity', 0)
            
            if quantity > available_quantity:
                return {
                    "valid": False,
                    "error": f"Insufficient position: have {available_quantity}, trying to sell {quantity}"
                }
        
        return {"valid": True}
    
    def _calculate_execution_price(
        self, 
        market_price: float, 
        quantity: float, 
        action: str, 
        max_slippage_pct: float
    ) -> float:
        """
        Calculate execution price with sophisticated slippage modeling
        
        Model considers:
        - Base market spread
        - Quantity impact
        - Market conditions
        - Maximum slippage limits
        """
        
        # Base slippage (market spread)
        base_slippage_pct = self.base_slippage_bps / 10000
        
        # Quantity impact (larger orders have more impact)
        quantity_impact_pct = min(quantity * self.quantity_impact_factor, 0.01)  # Cap at 1%
        
        # Market condition impact (could be enhanced with volatility data)
        market_impact_pct = 0.0001  # Minimal base market impact
        
        # Total slippage
        total_slippage_pct = base_slippage_pct + quantity_impact_pct + market_impact_pct
        
        # Apply maximum slippage limit
        total_slippage_pct = min(total_slippage_pct, max_slippage_pct / 100)
        
        # Apply slippage based on trade direction
        if action == "BUY":
            execution_price = market_price * (1 + total_slippage_pct)
        else:  # SELL
            execution_price = market_price * (1 - total_slippage_pct)
        
        return round(execution_price, 2)
    
    def _calculate_commission(self, trade_value: float, commission_bps: float = None) -> float:
        """
        Calculate trading commission with institutional rates
        
        Uses tiered commission structure:
        - Base rate for smaller trades
        - Reduced rates for larger trades
        - Minimum commission floor
        """
        
        if commission_bps is None:
            commission_bps = self.default_commission_bps
        
        # Tiered commission structure
        if trade_value > 10_000_000:  # ₹1 crore+
            effective_bps = commission_bps * 0.5  # 50% discount
        elif trade_value > 1_000_000:  # ₹10 lakh+
            effective_bps = commission_bps * 0.75  # 25% discount
        else:
            effective_bps = commission_bps
        
        # Calculate commission
        commission = trade_value * (effective_bps / 10000)
        
        # Apply minimum commission
        min_commission = 20.0  # ₹20 minimum
        commission = max(commission, min_commission)
        
        return round(commission, 2)
    
    def _execute_in_portfolio(
        self, 
        portfolio_data: Dict[str, Any], 
        ticker: str, 
        action: str,
        quantity: float, 
        execution_price: float, 
        commission: float
    ) -> Dict[str, Any]:
        """
        Execute trade in portfolio with atomic updates
        """
        
        try:
            balances = portfolio_data['balances']
            positions = portfolio_data.get('positions', {})
            
            trade_value = quantity * execution_price
            
            if action == "BUY":
                total_cost = trade_value + commission
                
                # Check and update cash/margin
                if balances['cash'] >= total_cost:
                    balances['cash'] -= total_cost
                else:
                    # Use margin
                    cash_used = balances['cash']
                    margin_used = total_cost - cash_used
                    
                    balances['cash'] = 0
                    balances['used_margin'] = balances.get('used_margin', 0) + margin_used
                    balances['available_margin'] -= margin_used
                
                # Update position
                if ticker in positions:
                    pos = positions[ticker]
                    total_cost_basis = (pos['avg_price'] * pos['quantity']) + trade_value
                    pos['quantity'] += quantity
                    pos['avg_price'] = total_cost_basis / pos['quantity']
                    pos['last_price'] = execution_price
                    pos['market_value'] = pos['quantity'] * execution_price
                else:
                    positions[ticker] = {
                        "quantity": quantity,
                        "avg_price": execution_price,
                        "last_price": execution_price,
                        "market_value": trade_value,
                        "unrealized_pnl": 0.0,
                        "entry_date": get_timestamp_iso()
                    }
            
            elif action == "SELL":
                # Update cash
                net_proceeds = trade_value - commission
                balances['cash'] += net_proceeds
                
                # Update position
                pos = positions[ticker]
                
                # Calculate realized P&L
                realized_pnl = (execution_price - pos['avg_price']) * quantity
                
                pos['quantity'] -= quantity
                
                if pos['quantity'] < 1e-6:  # Position closed
                    del positions[ticker]
                else:
                    pos['market_value'] = pos['quantity'] * execution_price
                    pos['last_price'] = execution_price
                
                # Update realized P&L in balances
                balances['realized_pnl'] = balances.get('realized_pnl', 0) + realized_pnl
            
            # Recalculate portfolio totals
            self._recalculate_portfolio_totals(portfolio_data)
            
            return {"success": True, "message": "Trade executed successfully"}
            
        except Exception as e:
            self.logger.error(f"Portfolio execution failed: {e}")
            return {"success": False, "message": f"Portfolio execution error: {e}"}
    
    def _recalculate_portfolio_totals(self, portfolio_data: Dict[str, Any]):
        """Recalculate portfolio total values after trade"""
        
        balances = portfolio_data['balances']
        positions = portfolio_data.get('positions', {})
        
        # Calculate total market value
        total_market_value = 0
        total_unrealized_pnl = 0
        
        for pos in positions.values():
            market_value = pos.get('market_value', pos['quantity'] * pos.get('last_price', pos['avg_price']))
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            
            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl
        
        # Update balances
        balances['market_value'] = total_market_value
        balances['unrealized_pnl'] = total_unrealized_pnl
        balances['total_value'] = balances['cash'] + total_market_value
    
    def _update_equity_history(self, portfolio_data: Dict[str, Any]):
        """Add new equity history point"""
        
        balances = portfolio_data['balances']
        
        equity_point = {
            "timestamp": get_timestamp_iso(),
            "total_value": balances['total_value'],
            "cash": balances['cash'],
            "market_value": balances['market_value'],
            "unrealized_pnl": balances.get('unrealized_pnl', 0)
        }
        
        equity_history = portfolio_data.setdefault('equity_history', [])
        equity_history.append(equity_point)
        
        # Keep only last 1000 points for performance
        if len(equity_history) > 1000:
            portfolio_data['equity_history'] = equity_history[-1000:]
    
    def get_execution_statistics(self, portfolio_name: str) -> Dict[str, Any]:
        """Get execution statistics for a portfolio"""
        
        try:
            # Load portfolio to get transaction history
            from ..portfolio.manager import EnhancedPortfolioManager
            portfolio_data = EnhancedPortfolioManager.load_portfolio_safe(portfolio_name)
            
            if not portfolio_data:
                return {"error": "Portfolio not found"}
            
            transactions = portfolio_data.get('transactions', [])
            
            if not transactions:
                return {"message": "No transactions found"}
            
            # Calculate statistics
            total_trades = len(transactions)
            total_volume = sum(t.get('trade_value', 0) for t in transactions)
            total_commission = sum(t.get('commission', 0) for t in transactions)
            
            # Slippage analysis
            slippages = [t.get('slippage_pct', 0) for t in transactions if 'slippage_pct' in t]
            avg_slippage = sum(slippages) / len(slippages) if slippages else 0
            
            # Trade size analysis
            trade_values = [t.get('trade_value', 0) for t in transactions]
            avg_trade_size = sum(trade_values) / len(trade_values) if trade_values else 0
            
            return {
                "total_trades": total_trades,
                "total_volume": total_volume,
                "total_commission": total_commission,
                "commission_rate_bps": (total_commission / total_volume * 10000) if total_volume > 0 else 0,
                "average_slippage_pct": avg_slippage,
                "average_trade_size": avg_trade_size,
                "commission_per_trade": total_commission / total_trades if total_trades > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get execution statistics: {e}")
            return {"error": str(e)}
