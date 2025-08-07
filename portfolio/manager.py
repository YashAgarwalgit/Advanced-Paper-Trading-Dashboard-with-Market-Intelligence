"""
Enhanced portfolio management with atomic transactions and validation
"""
import os
import json
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import sys
from config import Config
from utils.helpers import save_json_atomic, load_json_safe, get_timestamp_iso
from data.async_market_data import AsyncMarketDataManager
from .transaction_manager import PortfolioTransactionManager
from .analytics import PortfolioAnalytics

class EnhancedPortfolioManager:
    """Advanced portfolio management with atomic transactions and institutional features"""
    
    def __init__(self, portfolios_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("EnhancedPortfolioManager.__init__ ENTRY")
        self.portfolios_dir = portfolios_dir or Config.PORTFOLIOS_DIR
        self.transaction_manager = PortfolioTransactionManager(self.portfolios_dir)
        self.analytics = PortfolioAnalytics()
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        os.makedirs(self.portfolios_dir, exist_ok=True)
        self.logger.info("EnhancedPortfolioManager.__init__ EXIT")
    
    def get_portfolio_path(self, portfolio_name: str) -> str:
        self.logger.info(f"EnhancedPortfolioManager.get_portfolio_path ENTRY: {portfolio_name}")
        """Get full path to portfolio file"""
        result = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
        self.logger.info(f"EnhancedPortfolioManager.get_portfolio_path EXIT: {result}")
        return result
    
    def get_available_portfolios(self) -> List[str]:
        self.logger.info("EnhancedPortfolioManager.get_available_portfolios ENTRY")
        """Get list of available portfolios"""
        if not os.path.exists(self.portfolios_dir):
            return []
        
        files = [f for f in os.listdir(self.portfolios_dir) 
                if f.endswith('.json') and not f.endswith('.tmp')]
        result = [os.path.splitext(f)[0] for f in files]
        self.logger.info(f"EnhancedPortfolioManager.get_available_portfolios EXIT: {result}")
        return result
    
    def create_enhanced_portfolio(self,
        portfolio_name: str, 
        initial_capital: float,
        asset_allocation: Dict[str, float] = None, 
        benchmark: str = "^NSEI",
        risk_profile: str = "Moderate"
    ) -> bool:
        """Create enhanced portfolio with comprehensive initialization"""
        
        self.logger.info(f"EnhancedPortfolioManager.create_enhanced_portfolio ENTRY: {portfolio_name}")
        if not portfolio_name.strip():
            raise ValueError("Portfolio name cannot be empty")
        
        portfolio_path = self.get_portfolio_path(portfolio_name)
        
        if os.path.exists(portfolio_path):
            raise ValueError(f"Portfolio '{portfolio_name}' already exists")
        
        try:
            # Get benchmark price for initialization
            from ..data import AdvancedMarketData
            success, benchmark_prices = AdvancedMarketData.get_live_prices([benchmark])
            initial_benchmark_value = benchmark_prices.get(benchmark, 0) if success else 0
            
            # Default asset allocation
            if asset_allocation is None:
                asset_allocation = {"Equities": 0.8, "Cash": 0.2}
            
            # Create portfolio structure
            portfolio_data = self._create_portfolio_structure(
                portfolio_name, initial_capital, asset_allocation, 
                benchmark, initial_benchmark_value, risk_profile
            )
            
            # Validate portfolio data
            self._validate_portfolio_structure(portfolio_data)
            
            # Save using atomic write
            success = save_json_atomic(portfolio_data, portfolio_path)
            
            if success:
                self.logger.info(f"Portfolio '{portfolio_name}' created successfully")
                self.logger.info(f"EnhancedPortfolioManager.create_enhanced_portfolio EXIT: {portfolio_name} (success)")
                return True
            else:
                self.logger.info(f"EnhancedPortfolioManager.create_enhanced_portfolio EXIT: {portfolio_name} (save failed)")
                raise Exception("Failed to save portfolio data")
                
        except Exception as e:
            self.logger.error(f"Failed to create portfolio '{portfolio_name}': {e}")
            self.logger.info(f"EnhancedPortfolioManager.create_enhanced_portfolio EXIT: {portfolio_name} (error)")
            raise
    
    def _create_portfolio_structure(
        self, 
        name: str, 
        capital: float,
        allocation: Dict[str, float], 
        benchmark: str,
        benchmark_value: float,
        risk_profile: str
    ) -> Dict[str, Any]:
        """Create standardized portfolio data structure"""
        
        return {
            "metadata": {
                "portfolio_name": name,
                "created_utc": get_timestamp_iso(),
                "benchmark": benchmark,
                "asset_allocation_target": allocation,
                "risk_profile": risk_profile,
                "strategy_type": "Multi-Asset",
                "version": "2.0"
            },
            "balances": {
                "initial_capital": capital,
                "cash": capital,
                "market_value": 0.0,
                "total_value": capital,
                "available_margin": capital * 2.0,
                "used_margin": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            },
            "positions": {},
            "pending_orders": {},
            "transactions": [],
            "equity_history": [{
                "timestamp": get_timestamp_iso(),
                "total_value": capital,
                "benchmark_value": benchmark_value,
                "cash": capital,
                "market_value": 0.0
            }],
            "risk_metrics": {
                "var_95": 0.0,
                "var_99": 0.0,
                "expected_shortfall": 0.0,
                "beta": 0.0,
                "alpha": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "volatility": 0.0
            },
            "analytics": {
                "sector_exposure": {},
                "factor_exposure": {},
                "correlation_matrix": {},
                "performance_attribution": {},
                "optimization_history": []
            }
        }
    
    def _validate_portfolio_structure(self, data: Dict[str, Any]) -> None:
        """Validate portfolio data structure and business rules"""
        
        required_sections = ['metadata', 'balances', 'positions', 'transactions', 'equity_history']
        
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate metadata
        metadata = data['metadata']
        required_metadata = ['portfolio_name', 'created_utc', 'benchmark']
        
        for field in required_metadata:
            if field not in metadata:
                raise ValueError(f"Missing metadata field: {field}")
        
        # Validate balances
        balances = data['balances']
        required_balances = ['initial_capital', 'cash', 'market_value', 'total_value']
        
        for field in required_balances:
            if field not in balances:
                raise ValueError(f"Missing balance field: {field}")
            
            if not isinstance(balances[field], (int, float)):
                raise ValueError(f"Invalid balance type for {field}")
            
            if balances[field] < 0 and field not in ['unrealized_pnl', 'realized_pnl']:
                raise ValueError(f"Negative balance not allowed for {field}")
        
        # Validate asset allocation
        if 'asset_allocation_target' in metadata:
            allocation = metadata['asset_allocation_target']
            if isinstance(allocation, dict):
                total_allocation = sum(allocation.values())
                if abs(total_allocation - 1.0) > 0.01:
                    raise ValueError(f"Asset allocation must sum to 100%, got {total_allocation*100:.1f}%")
    
    def load_portfolio_safe(self, portfolio_name: str) -> Optional[Dict[str, Any]]:
        """Safely load portfolio with error handling"""
        
        portfolio_path = self.get_portfolio_path(portfolio_name)
        
        if not os.path.exists(portfolio_path):
            self.logger.warning(f"Portfolio '{portfolio_name}' not found")
            return None
        
        try:
            portfolio_data = load_json_safe(portfolio_path)
            
            if portfolio_data is None:
                self.logger.error(f"Failed to load portfolio '{portfolio_name}'")
                return None
            
            # Validate loaded data
            self._validate_portfolio_structure(portfolio_data)
            
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio '{portfolio_name}': {e}")
            return None
    
    def update_portfolio_prices(self, portfolio_name: str, market_data_manager: AsyncMarketDataManager = None) -> bool:
        """Update portfolio with latest market prices"""
        
        try:
            with self.transaction_manager.atomic_portfolio_update(portfolio_name) as portfolio_data:
                positions = portfolio_data.get('positions', {})
                
                if not positions:
                    return True  # No positions to update
                
                # Get current prices
                tickers = list(positions.keys())
                
                if market_data_manager:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        success, prices = loop.run_until_complete(
                            market_data_manager.get_live_prices_async(tickers)
                        )
                    finally:
                        loop.close()
                else:
                    from ..data import AdvancedMarketData
                    success, prices = AdvancedMarketData.get_live_prices(tickers)
                
                if not success:
                    self.logger.warning(f"Could not fetch prices for portfolio '{portfolio_name}'")
                    return False
                
                # Update positions
                total_market_value = 0
                total_unrealized_pnl = 0
                
                for ticker, position in positions.items():
                    if ticker in prices:
                        current_price = prices[ticker]
                        quantity = position['quantity']
                        avg_price = position['avg_price']
                        
                        # Update position values
                        position['last_price'] = current_price
                        position['market_value'] = quantity * current_price
                        position['unrealized_pnl'] = (current_price - avg_price) * quantity
                        
                        total_market_value += position['market_value']
                        total_unrealized_pnl += position['unrealized_pnl']
                
                # Update balances
                balances = portfolio_data['balances']
                balances['market_value'] = total_market_value
                balances['unrealized_pnl'] = total_unrealized_pnl
                balances['total_value'] = balances['cash'] + total_market_value
                
                # Add equity history point
                portfolio_data['equity_history'].append({
                    "timestamp": get_timestamp_iso(),
                    "total_value": balances['total_value'],
                    "cash": balances['cash'],
                    "market_value": total_market_value,
                    "benchmark_value": 0  # Would fetch benchmark price here
                })
                
                # Keep only last 1000 history points
                if len(portfolio_data['equity_history']) > 1000:
                    portfolio_data['equity_history'] = portfolio_data['equity_history'][-1000:]
                
                self.logger.info(f"Updated prices for portfolio '{portfolio_name}'")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update prices for portfolio '{portfolio_name}': {e}")
            return False
    
    def calculate_portfolio_metrics(self, portfolio_name: str) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive portfolio performance metrics"""
        
        try:
            portfolio_data = self.load_portfolio_safe(portfolio_name)
            
            if not portfolio_data:
                return None
            
            # Calculate risk and performance metrics
            equity_history = portfolio_data.get('equity_history', [])
            positions = portfolio_data.get('positions', {})
            
            metrics = self.analytics.calculate_comprehensive_metrics(
                equity_history, positions, portfolio_data.get('balances', {})
            )
            
            # Update portfolio with calculated metrics
            with self.transaction_manager.atomic_portfolio_update(portfolio_name) as portfolio_data:
                portfolio_data['risk_metrics'].update(metrics.get('risk_metrics', {}))
                portfolio_data['analytics'].update(metrics.get('analytics', {}))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics for portfolio '{portfolio_name}': {e}")
            return None
    
    def delete_portfolio(self, portfolio_name: str) -> bool:
        """Safely delete portfolio with backup"""
        
        try:
            # Create backup before deletion
            backup_path = self.transaction_manager.create_portfolio_backup(portfolio_name)
            self.logger.info(f"Created backup before deletion: {backup_path}")
            
            # Delete portfolio file
            portfolio_path = self.get_portfolio_path(portfolio_name)
            
            if os.path.exists(portfolio_path):
                os.remove(portfolio_path)
                self.logger.info(f"Portfolio '{portfolio_name}' deleted successfully")
                return True
            else:
                self.logger.warning(f"Portfolio '{portfolio_name}' not found for deletion")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete portfolio '{portfolio_name}': {e}")
            return False
    
    def rename_portfolio(self, old_name: str, new_name: str) -> bool:
        """Rename portfolio with atomic operation"""
        
        if not new_name.strip():
            raise ValueError("New portfolio name cannot be empty")
        
        if old_name == new_name:
            return True  # No change needed
        
        try:
            # Check if new name already exists
            new_path = self.get_portfolio_path(new_name)
            if os.path.exists(new_path):
                raise ValueError(f"Portfolio '{new_name}' already exists")
            
            # Update portfolio metadata and move file
            with self.transaction_manager.atomic_portfolio_update(old_name) as portfolio_data:
                portfolio_data['metadata']['portfolio_name'] = new_name
                
                # The atomic update will save to old location
                # Now we need to move to new location
                old_path = self.get_portfolio_path(old_name)
                shutil.move(old_path, new_path)
            
            self.logger.info(f"Portfolio renamed from '{old_name}' to '{new_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rename portfolio from '{old_name}' to '{new_name}': {e}")
            raise
    
    def duplicate_portfolio(self, source_name: str, target_name: str) -> bool:
        """Create a copy of existing portfolio"""
        
        if not target_name.strip():
            raise ValueError("Target portfolio name cannot be empty")
        
        try:
            # Load source portfolio
            source_data = self.load_portfolio_safe(source_name)
            if not source_data:
                raise ValueError(f"Source portfolio '{source_name}' not found")
            
            # Create copy with new metadata
            target_data = source_data.copy()
            target_data['metadata']['portfolio_name'] = target_name
            target_data['metadata']['created_utc'] = get_timestamp_iso()
            
            # Save as new portfolio
            target_path = self.get_portfolio_path(target_name)
            success = save_json_atomic(target_data, target_path)
            
            if success:
                self.logger.info(f"Portfolio duplicated from '{source_name}' to '{target_name}'")
                return True
            else:
                raise Exception("Failed to save duplicated portfolio")
                
        except Exception as e:
            self.logger.error(f"Failed to duplicate portfolio: {e}")
            raise
    
    def export_portfolio(self, portfolio_name: str) -> Optional[str]:
        """Export portfolio to JSON string"""
        
        try:
            portfolio_data = self.load_portfolio_safe(portfolio_name)
            if portfolio_data:
                return json.dumps(portfolio_data, indent=2, default=str)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to export portfolio '{portfolio_name}': {e}")
            return None
    
    def import_portfolio(self, portfolio_name: str, json_data: str) -> bool:
        """Import portfolio from JSON string"""
        
        try:
            # Parse JSON data
            portfolio_data = json.loads(json_data)
            
            # Validate structure
            self._validate_portfolio_structure(portfolio_data)
            
            # Update metadata
            portfolio_data['metadata']['portfolio_name'] = portfolio_name
            portfolio_data['metadata']['imported_utc'] = get_timestamp_iso()
            
            # Save portfolio
            portfolio_path = self.get_portfolio_path(portfolio_name)
            success = save_json_atomic(portfolio_data, portfolio_path)
            
            if success:
                self.logger.info(f"Portfolio '{portfolio_name}' imported successfully")
                return True
            else:
                raise Exception("Failed to save imported portfolio")
                
        except Exception as e:
            self.logger.error(f"Failed to import portfolio '{portfolio_name}': {e}")
            raise
