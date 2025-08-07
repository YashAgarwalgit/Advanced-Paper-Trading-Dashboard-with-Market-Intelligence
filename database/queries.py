"""
Database query operations for portfolio and trading data
"""
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import sys
from .connection import DatabaseConnection

class DatabaseQueries:
    """Centralized database query operations"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatabaseQueries.__init__ ENTRY [LOG_DB]")
        self.logger.info(f"Initialized DatabaseQueries with connection: {db_connection} [LOG_DB]")
        self.logger.info("DatabaseQueries.__init__ EXIT [LOG_DB]")
    
    # Portfolio Operations
    def create_portfolio(self, portfolio_data: Dict[str, Any]) -> Optional[int]:
        """Create new portfolio and return portfolio ID"""
        self.logger.info("DatabaseQueries.create_portfolio ENTRY [LOG_DB]")
        self.logger.debug(f"Creating portfolio with data: {portfolio_data.get('metadata', {}).get('portfolio_name', 'Unknown')} [LOG_DB]")
        
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO portfolios (name, metadata, balances, risk_metrics, analytics)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    portfolio_data['metadata']['portfolio_name'],
                    json.dumps(portfolio_data['metadata']),
                    json.dumps(portfolio_data['balances']),
                    json.dumps(portfolio_data.get('risk_metrics', {})),
                    json.dumps(portfolio_data.get('analytics', {}))
                ))
                
                portfolio_id = cursor.lastrowid
                
                # Insert initial equity history
                if portfolio_data.get('equity_history'):
                    history = portfolio_data['equity_history'][0]
                    cursor.execute("""
                        INSERT INTO equity_history (portfolio_id, total_value, cash, market_value, benchmark_value)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        portfolio_id,
                        history.get('total_value', 0),
                        history.get('cash', 0), 
                        history.get('market_value', 0),
                        history.get('benchmark_value', 0)
                    ))
                
                self.logger.info(f"Successfully created portfolio with ID: {portfolio_id} [LOG_DB]")
                self.logger.info("DatabaseQueries.create_portfolio EXIT [LOG_DB]")
                return portfolio_id
                
        except Exception as e:
            self.logger.error(f"Failed to create portfolio: {e} [LOG_DB]", exc_info=True)
            self.logger.info("DatabaseQueries.create_portfolio EXIT (ERROR) [LOG_DB]")
            return None
    
    def get_portfolio(self, portfolio_name: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data by name"""
        self.logger.info("DatabaseQueries.get_portfolio ENTRY [LOG_DB]")
        self.logger.debug(f"Fetching portfolio: {portfolio_name} [LOG_DB]")
        
        try:
            query = """
                SELECT id, metadata, balances, risk_metrics, analytics, created_at, updated_at
                FROM portfolios 
                WHERE name = ? AND is_active = 1
            """
            
            result = self.db.execute_query(query, (portfolio_name,))
            
            if not result:
                self.logger.warning(f"Portfolio not found: {portfolio_name} [LOG_DB]")
                self.logger.info("DatabaseQueries.get_portfolio EXIT (NOT FOUND) [LOG_DB]")
                return None
            
            row = result[0]
            portfolio_id = row['id']
            
            # Build portfolio data structure
            portfolio_data = {
                'metadata': json.loads(row['metadata']),
                'balances': json.loads(row['balances']),
                'risk_metrics': json.loads(row['risk_metrics'] or '{}'),
                'analytics': json.loads(row['analytics'] or '{}'),
                'positions': self.get_positions(portfolio_id),
                'transactions': self.get_transactions(portfolio_id, limit=100),
                'equity_history': self.get_equity_history(portfolio_id, limit=1000),
                'pending_orders': self.get_pending_orders(portfolio_id)
            }
            
            self.logger.info(f"Successfully retrieved portfolio: {portfolio_name} [LOG_DB]")
            self.logger.info("DatabaseQueries.get_portfolio EXIT [LOG_DB]")
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio '{portfolio_name}': {e} [LOG_DB]", exc_info=True)
            self.logger.info("DatabaseQueries.get_portfolio EXIT (ERROR) [LOG_DB]")
            return None
    
    def update_portfolio(self, portfolio_name: str, portfolio_data: Dict[str, Any]) -> bool:
        """Update portfolio data"""
        self.logger.info("DatabaseQueries.update_portfolio ENTRY [LOG_DB]")
        self.logger.debug(f"Updating portfolio: {portfolio_name} [LOG_DB]")
        
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                
                # Update main portfolio record
                cursor.execute("""
                    UPDATE portfolios 
                    SET metadata = ?, balances = ?, risk_metrics = ?, analytics = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (
                    json.dumps(portfolio_data['metadata']),
                    json.dumps(portfolio_data['balances']),
                    json.dumps(portfolio_data.get('risk_metrics', {})),
                    json.dumps(portfolio_data.get('analytics', {})),
                    portfolio_name
                ))
                
                portfolio_id = self.db.execute_scalar(
                    "SELECT id FROM portfolios WHERE name = ?", (portfolio_name,)
                )
                
                if not portfolio_id:
                    return False
                
                # Update positions
                self._sync_positions(cursor, portfolio_id, portfolio_data.get('positions', {}))
                
                # Add equity history point if provided
                if portfolio_data.get('equity_history'):
                    latest_history = portfolio_data['equity_history'][-1]
                    cursor.execute("""
                        INSERT INTO equity_history (portfolio_id, total_value, cash, market_value, benchmark_value)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        portfolio_id,
                        latest_history.get('total_value', 0),
                        latest_history.get('cash', 0),
                        latest_history.get('market_value', 0),
                        latest_history.get('benchmark_value', 0)
                    ))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update portfolio '{portfolio_name}': {e}")
            return False
    
    def _sync_positions(self, cursor: sqlite3.Cursor, portfolio_id: int, positions: Dict[str, Any]):
        """Synchronize positions with database"""
        
        # Get current positions
        cursor.execute(
            "SELECT ticker FROM positions WHERE portfolio_id = ?", (portfolio_id,)
        )
        current_tickers = set(row[0] for row in cursor.fetchall())
        new_tickers = set(positions.keys())
        
        # Remove positions that no longer exist
        for ticker in current_tickers - new_tickers:
            cursor.execute(
                "DELETE FROM positions WHERE portfolio_id = ? AND ticker = ?",
                (portfolio_id, ticker)
            )
        
        # Insert or update positions
        for ticker, position in positions.items():
            cursor.execute("""
                INSERT OR REPLACE INTO positions 
                (portfolio_id, ticker, quantity, avg_price, last_price, market_value, 
                 unrealized_pnl, entry_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                portfolio_id, ticker,
                position.get('quantity', 0),
                position.get('avg_price', 0),
                position.get('last_price', 0),
                position.get('market_value', 0),
                position.get('unrealized_pnl', 0),
                position.get('entry_date')
            ))
    
    def get_positions(self, portfolio_id: int) -> Dict[str, Any]:
        """Get all positions for a portfolio"""
        
        try:
            query = """
                SELECT ticker, quantity, avg_price, last_price, market_value, 
                       unrealized_pnl, entry_date, updated_at
                FROM positions 
                WHERE portfolio_id = ? AND quantity > 0
                ORDER BY market_value DESC
            """
            
            results = self.db.execute_query(query, (portfolio_id,))
            
            positions = {}
            for row in results:
                positions[row['ticker']] = {
                    'quantity': row['quantity'],
                    'avg_price': row['avg_price'],
                    'last_price': row['last_price'],
                    'market_value': row['market_value'],
                    'unrealized_pnl': row['unrealized_pnl'],
                    'entry_date': row['entry_date']
                }
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions for portfolio {portfolio_id}: {e}")
            return {}
    
    def add_transaction(self, portfolio_id: int, transaction: Dict[str, Any]) -> bool:
        """Add transaction to database"""
        
        try:
            query = """
                INSERT INTO transactions 
                (portfolio_id, order_id, ticker, action, quantity, price, commission, 
                 trade_value, order_type, timestamp, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                portfolio_id,
                transaction.get('order_id'),
                transaction['ticker'],
                transaction['action'],
                transaction['quantity'],
                transaction['price'],
                transaction.get('commission', 0),
                transaction.get('trade_value', transaction['quantity'] * transaction['price']),
                transaction.get('order_type', 'MARKET'),
                transaction.get('timestamp', datetime.utcnow().isoformat()),
                transaction.get('notes')
            )
            
            rows_affected = self.db.execute_non_query(query, params)
            return rows_affected > 0
            
        except Exception as e:
            self.logger.error(f"Failed to add transaction: {e}")
            return False
    
    def get_transactions(self, portfolio_id: int, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get transaction history for portfolio"""
        
        try:
            query = """
                SELECT order_id, ticker, action, quantity, price, commission, 
                       trade_value, order_type, timestamp, notes
                FROM transactions 
                WHERE portfolio_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            
            results = self.db.execute_query(query, (portfolio_id, limit, offset))
            
            transactions = []
            for row in results:
                transactions.append({
                    'order_id': row['order_id'],
                    'ticker': row['ticker'],
                    'action': row['action'],
                    'quantity': row['quantity'],
                    'price': row['price'],
                    'commission': row['commission'],
                    'trade_value': row['trade_value'],
                    'order_type': row['order_type'],
                    'timestamp': row['timestamp'],
                    'notes': row['notes']
                })
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Failed to get transactions for portfolio {portfolio_id}: {e}")
            return []
    
    def get_equity_history(self, portfolio_id: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get equity history for portfolio"""
        
        try:
            query = """
                SELECT total_value, cash, market_value, unrealized_pnl, benchmark_value, timestamp
                FROM equity_history 
                WHERE portfolio_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            results = self.db.execute_query(query, (portfolio_id, limit))
            
            history = []
            for row in results:
                history.append({
                    'total_value': row['total_value'],
                    'cash': row['cash'],
                    'market_value': row['market_value'],
                    'unrealized_pnl': row['unrealized_pnl'],
                    'benchmark_value': row['benchmark_value'],
                    'timestamp': row['timestamp']
                })
            
            # Return in chronological order
            return list(reversed(history))
            
        except Exception as e:
            self.logger.error(f"Failed to get equity history for portfolio {portfolio_id}: {e}")
            return []
    
    def get_pending_orders(self, portfolio_id: int) -> Dict[str, Any]:
        """Get pending orders for portfolio"""
        
        try:
            query = """
                SELECT order_id, ticker, side, order_type, quantity, price, stop_price,
                       status, filled_quantity, remaining_quantity, created_at
                FROM orders 
                WHERE portfolio_id = ? AND status IN ('PENDING', 'PARTIAL_FILLED')
                ORDER BY created_at DESC
            """
            
            results = self.db.execute_query(query, (portfolio_id,))
            
            orders = {}
            for row in results:
                orders[row['order_id']] = {
                    'ticker': row['ticker'],
                    'side': row['side'],
                    'order_type': row['order_type'],
                    'quantity': row['quantity'],
                    'price': row['price'],
                    'stop_price': row['stop_price'],
                    'status': row['status'],
                    'filled_quantity': row['filled_quantity'],
                    'remaining_quantity': row['remaining_quantity'],
                    'timestamp': row['created_at']
                }
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get pending orders for portfolio {portfolio_id}: {e}")
            return {}
    
    def get_portfolio_performance(self, portfolio_id: int, days: int = 30) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        
        try:
            # Get equity history for the period
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            query = """
                SELECT total_value, timestamp
                FROM equity_history 
                WHERE portfolio_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """
            
            results = self.db.execute_query(query, (portfolio_id, cutoff_date))
            
            if len(results) < 2:
                return {'period_return': 0.0, 'data_points': len(results)}
            
            start_value = results[0]['total_value']
            end_value = results[-1]['total_value']
            
            period_return = (end_value / start_value - 1) * 100 if start_value > 0 else 0
            
            return {
                'period_return': period_return,
                'start_value': start_value,
                'end_value': end_value,
                'data_points': len(results),
                'period_days': days
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance for portfolio {portfolio_id}: {e}")
            return {'period_return': 0.0, 'error': str(e)}
    
    def get_portfolio_list(self) -> List[Dict[str, Any]]:
        """Get list of all active portfolios"""
        
        try:
            query = """
                SELECT name, metadata, balances, created_at, updated_at
                FROM portfolios 
                WHERE is_active = 1
                ORDER BY updated_at DESC
            """
            
            results = self.db.execute_query(query)
            
            portfolios = []
            for row in results:
                metadata = json.loads(row['metadata'])
                balances = json.loads(row['balances'])
                
                portfolios.append({
                    'name': row['name'],
                    'display_name': metadata.get('portfolio_name', row['name']),
                    'strategy': metadata.get('strategy_type', 'Multi-Asset'),
                    'total_value': balances.get('total_value', 0),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })
            
            return portfolios
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio list: {e}")
            return []
    
    def delete_portfolio(self, portfolio_name: str) -> bool:
        """Soft delete portfolio (mark as inactive)"""
        
        try:
            rows_affected = self.db.execute_non_query(
                "UPDATE portfolios SET is_active = 0 WHERE name = ?",
                (portfolio_name,)
            )
            
            if rows_affected > 0:
                self.logger.info(f"Portfolio '{portfolio_name}' marked as inactive")
                return True
            else:
                self.logger.warning(f"Portfolio '{portfolio_name}' not found for deletion")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete portfolio '{portfolio_name}': {e}")
            return False
    
    def cleanup_old_data(self, days: int = 365) -> int:
        """Clean up old data to manage database size"""
        
        cleanup_count = 0
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                
                # Clean up old equity history (keep recent data)
                cursor.execute("""
                    DELETE FROM equity_history 
                    WHERE timestamp < ? 
                    AND id NOT IN (
                        SELECT id FROM equity_history 
                        WHERE portfolio_id IN (SELECT DISTINCT portfolio_id FROM equity_history)
                        ORDER BY timestamp DESC 
                        LIMIT 100
                    )
                """, (cutoff_date,))
                
                cleanup_count += cursor.rowcount
                
                # Clean up expired cache entries
                cursor.execute(
                    "DELETE FROM market_data_cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),)
                )
                
                cleanup_count += cursor.rowcount
                
            self.logger.info(f"Cleaned up {cleanup_count} old records")
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            return 0
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database usage statistics"""
        
        try:
            stats = {}
            
            # Table row counts
            tables = ['portfolios', 'positions', 'transactions', 'equity_history', 'orders']
            
            for table in tables:
                count = self.db.execute_scalar(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = count or 0
            
            # Database size
            stats['database_size_mb'] = self.db.get_database_size() / (1024 * 1024)
            
            # Recent activity
            stats['transactions_last_30_days'] = self.db.execute_scalar("""
                SELECT COUNT(*) FROM transactions 
                WHERE timestamp >= date('now', '-30 days')
            """) or 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database statistics: {e}")
            return {}
