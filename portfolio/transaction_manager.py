"""
Atomic portfolio transaction manager with rollback capability
"""
import os
import json
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Any
from contextlib import contextmanager
import logging
import sys
from config import Config
from utils.helpers import get_timestamp_iso

class PortfolioTransactionManager:
    """
    Atomic portfolio operations with rollback capability
    Ensures data consistency during all portfolio modifications
    """
    
    def __init__(self, portfolios_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("PortfolioTransactionManager.__init__ ENTRY")
        self.portfolios_dir = portfolios_dir or Config.PORTFOLIOS_DIR
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.portfolios_dir, exist_ok=True)
        self.logger.info("PortfolioTransactionManager.__init__ EXIT")
    
    @contextmanager
    def atomic_portfolio_update(self, portfolio_name: str):
        self.logger.info(f"PortfolioTransactionManager.atomic_portfolio_update ENTRY: {portfolio_name}")
        """
        Context manager for atomic portfolio updates with rollback
        
        Usage:
            with transaction_manager.atomic_portfolio_update("MyPortfolio") as portfolio:
                portfolio['balances']['cash'] -= 1000
                # Any exception here will rollback changes
        """
        
        portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
        backup_path = None
        temp_path = None
        
        try:
            # Create backup of current state
            if os.path.exists(portfolio_path):
                backup_path = f"{portfolio_path}.backup_{int(datetime.now().timestamp())}"
                shutil.copy2(portfolio_path, backup_path)
                self.logger.debug(f"Created backup: {backup_path}")
            else:
                raise FileNotFoundError(f"Portfolio '{portfolio_name}' not found")
            
            # Load current portfolio data
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            
            # Validate data integrity before modification
            self._validate_portfolio_data(portfolio_data)
            
            # Yield portfolio data for modification
            yield portfolio_data
            
            # Validate data after modification
            self._validate_portfolio_data(portfolio_data)
            self._validate_business_rules(portfolio_data)
            
            # Atomic write using temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.portfolios_dir, 
                delete=False, 
                suffix='.tmp',
                prefix=f"{portfolio_name}_"
            ) as temp_file:
                json.dump(portfolio_data, temp_file, indent=4, default=str)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
                temp_path = temp_file.name
            
            # Atomic move to final location
            shutil.move(temp_path, portfolio_path)
            
            # Clean up backup after successful operation
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)
                self.logger.debug(f"Cleaned up backup: {backup_path}")
            
            self.logger.info(f"Successfully updated portfolio: {portfolio_name}")
            self.logger.info(f"PortfolioTransactionManager.atomic_portfolio_update EXIT: {portfolio_name} (success)")
            
        except Exception as e:
            self.logger.error(f"Portfolio update failed for '{portfolio_name}': {e}")
            self.logger.info(f"PortfolioTransactionManager.atomic_portfolio_update EXIT: {portfolio_name} (error)")
            
            # Rollback from backup if available
            if backup_path and os.path.exists(backup_path):
                try:
                    if os.path.exists(portfolio_path):
                        os.remove(portfolio_path)
                    shutil.move(backup_path, portfolio_path)
                    self.logger.info(f"Successfully rolled back from backup: {backup_path}")
                    self.logger.info(f"PortfolioTransactionManager.atomic_portfolio_update EXIT: {portfolio_name} (rollback)")
                except Exception as rollback_error:
                    self.logger.critical(f"Rollback failed: {rollback_error}")
            
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
            # Re-raise the original exception
            raise
    
    def _validate_portfolio_data(self, data: Dict[str, Any]) -> None:
        """Validate portfolio data structure and types"""
        
        required_fields = ['metadata', 'balances', 'positions', 'transactions', 'equity_history']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate metadata
        metadata = data['metadata']
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        required_metadata_fields = ['portfolio_name', 'created_utc']
        for field in required_metadata_fields:
            if field not in metadata:
                raise ValueError(f"Missing metadata field: {field}")
        
        # Validate balances
        balances = data['balances']
        if not isinstance(balances, dict):
            raise ValueError("Balances must be a dictionary")
        
        required_balance_fields = ['initial_capital', 'cash', 'market_value', 'total_value']
        for field in required_balance_fields:
            if field not in balances:
                raise ValueError(f"Missing balance field: {field}")
            
            if not isinstance(balances[field], (int, float)):
                raise ValueError(f"Balance field '{field}' must be numeric, got {type(balances[field])}")
            
            # Check for negative values (except P&L fields)
            if balances[field] < 0 and field not in ['unrealized_pnl', 'realized_pnl']:
                raise ValueError(f"Negative balance not allowed for '{field}': {balances[field]}")
        
        # Validate positions
        positions = data['positions']
        if not isinstance(positions, dict):
            raise ValueError("Positions must be a dictionary")
        
        for ticker, position in positions.items():
            if not isinstance(position, dict):
                raise ValueError(f"Position for '{ticker}' must be a dictionary")
            
            required_pos_fields = ['quantity', 'avg_price']
            for field in required_pos_fields:
                if field not in position:
                    raise ValueError(f"Missing position field '{field}' for ticker '{ticker}'")
                
                if not isinstance(position[field], (int, float)):
                    raise ValueError(f"Position field '{field}' for '{ticker}' must be numeric")
                
                if position[field] < 0 and field == 'quantity':
                    raise ValueError(f"Negative quantity not allowed for '{ticker}': {position[field]}")
        
        # Validate transactions
        transactions = data['transactions']
        if not isinstance(transactions, list):
            raise ValueError("Transactions must be a list")
        
        for i, transaction in enumerate(transactions):
            if not isinstance(transaction, dict):
                raise ValueError(f"Transaction {i} must be a dictionary")
            
            required_txn_fields = ['timestamp', 'ticker', 'action', 'quantity', 'price']
            for field in required_txn_fields:
                if field not in transaction:
                    raise ValueError(f"Missing transaction field '{field}' in transaction {i}")
        
        # Validate equity history
        equity_history = data['equity_history']
        if not isinstance(equity_history, list):
            raise ValueError("Equity history must be a list")
    
    def _validate_business_rules(self, data: Dict[str, Any]) -> None:
        """Validate business logic and consistency rules"""
        
        balances = data['balances']
        positions = data['positions']
        
        # Calculate total position value
        total_position_value = 0
        for ticker, position in positions.items():
            if 'market_value' in position:
                total_position_value += position['market_value']
            else:
                # Fallback calculation
                quantity = position.get('quantity', 0)
                last_price = position.get('last_price', position.get('avg_price', 0))
                total_position_value += quantity * last_price
        
        # Validate total value consistency
        expected_total = balances['cash'] + total_position_value
        actual_total = balances['total_value']
        
        # Allow for small rounding differences
        tolerance = max(abs(expected_total) * 0.01, 1.0)
        
        if abs(expected_total - actual_total) > tolerance:
            self.logger.warning(
                f"Total value mismatch detected - Expected: {expected_total:.2f}, "
                f"Actual: {actual_total:.2f}, Difference: {abs(expected_total - actual_total):.2f}"
            )
            
            # Auto-correct minor discrepancies
            balances['total_value'] = expected_total
            balances['market_value'] = total_position_value
            
            self.logger.info(f"Auto-corrected total value to {expected_total:.2f}")
        
        # Validate margin usage
        if 'available_margin' in balances and 'used_margin' in balances:
            available_margin = balances['available_margin']
            used_margin = balances['used_margin']
            
            if used_margin < 0:
                raise ValueError(f"Used margin cannot be negative: {used_margin}")
            
            if available_margin < 0:
                self.logger.warning(f"Available margin is negative: {available_margin}")
        
        # Validate position consistency
        for ticker, position in positions.items():
            quantity = position.get('quantity', 0)
            
            if quantity <= 0:
                self.logger.warning(f"Position '{ticker}' has zero or negative quantity: {quantity}")
            
            # Check if all required position fields are present
            if 'avg_price' not in position:
                raise ValueError(f"Position '{ticker}' missing average price")
            
            avg_price = position['avg_price']
            if avg_price <= 0:
                raise ValueError(f"Position '{ticker}' has invalid average price: {avg_price}")
    
    def create_portfolio_backup(self, portfolio_name: str) -> str:
        """Create a manual backup of portfolio with timestamp"""
        
        portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
        
        if not os.path.exists(portfolio_path):
            raise FileNotFoundError(f"Portfolio '{portfolio_name}' not found")
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{portfolio_name}_backup_{timestamp}.json"
        backup_path = os.path.join(self.portfolios_dir, backup_filename)
        
        try:
            shutil.copy2(portfolio_path, backup_path)
            self.logger.info(f"Manual backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for '{portfolio_name}': {e}")
            raise
    
    def restore_portfolio_from_backup(self, portfolio_name: str, backup_path: str) -> bool:
        """Restore portfolio from backup file"""
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
        
        try:
            # Validate backup data before restore
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            self._validate_portfolio_data(backup_data)
            self._validate_business_rules(backup_data)
            
            # Create backup of current version before restore
            if os.path.exists(portfolio_path):
                current_backup = self.create_portfolio_backup(portfolio_name)
                self.logger.info(f"Current version backed up to: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_path, portfolio_path)
            
            self.logger.info(f"Portfolio '{portfolio_name}' restored from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore portfolio '{portfolio_name}' from backup: {e}")
            raise
    
    def cleanup_old_backups(self, portfolio_name: str, keep_count: int = 5) -> int:
        """Clean up old backup files, keeping only the most recent ones"""
        
        try:
            # Find all backup files for this portfolio
            backup_pattern = f"{portfolio_name}_backup_"
            backup_files = []
            
            for filename in os.listdir(self.portfolios_dir):
                if filename.startswith(backup_pattern) and filename.endswith('.json'):
                    file_path = os.path.join(self.portfolios_dir, filename)
                    backup_files.append((file_path, os.path.getmtime(file_path)))
            
            if len(backup_files) <= keep_count:
                return 0  # Nothing to clean up
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            cleaned_count = 0
            for file_path, _ in backup_files[keep_count:]:
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                    self.logger.debug(f"Removed old backup: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not remove backup {file_path}: {e}")
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old backup files for '{portfolio_name}'")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up backups for '{portfolio_name}': {e}")
            return 0
    
    def get_backup_history(self, portfolio_name: str) -> List[Dict[str, Any]]:
        """Get list of available backups for a portfolio"""
        
        try:
            backup_pattern = f"{portfolio_name}_backup_"
            backups = []
            
            for filename in os.listdir(self.portfolios_dir):
                if filename.startswith(backup_pattern) and filename.endswith('.json'):
                    file_path = os.path.join(self.portfolios_dir, filename)
                    stat = os.stat(file_path)
                    
                    # Extract timestamp from filename
                    timestamp_str = filename.replace(backup_pattern, '').replace('.json', '')
                    
                    backups.append({
                        'filename': filename,
                        'path': file_path,
                        'timestamp_str': timestamp_str,
                        'created_at': datetime.fromtimestamp(stat.st_ctime),
                        'size_bytes': stat.st_size
                    })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            
            return backups
            
        except Exception as e:
            self.logger.error(f"Error getting backup history for '{portfolio_name}': {e}")
            return []
    
    def verify_portfolio_integrity(self, portfolio_name: str) -> Dict[str, Any]:
        """Comprehensive portfolio data integrity check"""
        
        portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
        
        if not os.path.exists(portfolio_path):
            return {
                'valid': False,
                'errors': [f"Portfolio file not found: {portfolio_path}"],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        try:
            # Load and validate portfolio data
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            
            try:
                self._validate_portfolio_data(portfolio_data)
            except ValueError as e:
                errors.append(f"Structure validation: {e}")
            
            try:
                self._validate_business_rules(portfolio_data)
            except ValueError as e:
                errors.append(f"Business rule validation: {e}")
            
            # Additional integrity checks
            balances = portfolio_data.get('balances', {})
            positions = portfolio_data.get('positions', {})
            
            # Check for orphaned data
            if len(positions) == 0 and balances.get('market_value', 0) > 0:
                warnings.append("Market value exists but no positions found")
            
            # Check transaction history consistency
            transactions = portfolio_data.get('transactions', [])
            if len(transactions) > 0:
                # Verify transactions are sorted by timestamp
                timestamps = [t.get('timestamp', '') for t in transactions[:10]]  # Check first 10
                if timestamps != sorted(timestamps, reverse=True):
                    warnings.append("Transaction history may not be properly sorted")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'checks_performed': [
                    'Structure validation',
                    'Business rules validation', 
                    'Data consistency checks',
                    'Orphaned data detection'
                ]
            }
            
        except Exception as e:
            errors.append(f"File integrity error: {e}")
            return {
                'valid': False,
                'errors': errors,
                'warnings': warnings
            }
