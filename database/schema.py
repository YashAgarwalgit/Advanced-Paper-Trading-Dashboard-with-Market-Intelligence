"""
Database schema definitions and migrations
"""
import sqlite3
from typing import List, Dict, Any
import logging
import sys
from datetime import datetime
from .connection import DatabaseConnection

class DatabaseSchema:
    """Database schema management and migrations"""
    
    CURRENT_VERSION = "2.0"
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatabaseSchema.__init__ ENTRY [LOG_DB]")
        self.logger.info(f"Initialized DatabaseSchema with connection: {db_connection} [LOG_DB]")
        self.logger.info("DatabaseSchema.__init__ EXIT [LOG_DB]")
    
    def initialize_schema(self):
        """Initialize database schema with all tables"""
        self.logger.info("DatabaseSchema.initialize_schema ENTRY [LOG_DB]")
        self.logger.info(f"Initializing database schema to version {self.CURRENT_VERSION} [LOG_DB]")
        
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                
                # Create all tables
                self._create_portfolios_table(cursor)
                self._create_positions_table(cursor)
                self._create_transactions_table(cursor)
                self._create_equity_history_table(cursor)
                self._create_orders_table(cursor)
                self._create_order_fills_table(cursor)
                self._create_market_data_cache_table(cursor)
                self._create_system_metadata_table(cursor)
                
                # Create indexes
                self._create_indexes(cursor)
                
                # Set schema version
                cursor.execute("""
                    INSERT OR REPLACE INTO system_metadata (key, value)
                    VALUES ('schema_version', ?)
                """, (self.CURRENT_VERSION,))
                
                self.logger.info(f"Successfully initialized database schema (version {self.CURRENT_VERSION}) [LOG_DB]")
                self.logger.info("DatabaseSchema.initialize_schema EXIT [LOG_DB]")
                
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {e} [LOG_DB]", exc_info=True)
            self.logger.info("DatabaseSchema.initialize_schema EXIT (ERROR) [LOG_DB]")
            raise
    
    def _create_portfolios_table(self, cursor: sqlite3.Cursor):
        """Create portfolios table"""
        self.logger.debug("Creating 'portfolios' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    metadata TEXT NOT NULL,
                    balances TEXT NOT NULL,
                    risk_metrics TEXT,
                    analytics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version TEXT DEFAULT '2.0',
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            self.logger.debug("Successfully created 'portfolios' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'portfolios' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_positions_table(self, cursor: sqlite3.Cursor):
        """Create positions table"""
        self.logger.debug("Creating 'positions' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    last_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL DEFAULT 0,
                    entry_date TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE,
                    UNIQUE(portfolio_id, ticker)
                )
            """)
            self.logger.debug("Successfully created 'positions' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'positions' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_transactions_table(self, cursor: sqlite3.Cursor):
        """Create transactions table"""
        self.logger.debug("Creating 'transactions' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    order_id TEXT,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL')),
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL NOT NULL DEFAULT 0,
                    trade_value REAL NOT NULL,
                    order_type TEXT DEFAULT 'MARKET',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE
                )
            """)
            self.logger.debug("Successfully created 'transactions' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'transactions' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_equity_history_table(self, cursor: sqlite3.Cursor):
        """Create equity history table"""
        self.logger.debug("Creating 'equity_history' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    benchmark_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE
                )
            """)
            self.logger.debug("Successfully created 'equity_history' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'equity_history' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_orders_table(self, cursor: sqlite3.Cursor):
        """Create orders table"""
        self.logger.debug("Creating 'orders' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    order_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    stop_price REAL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    filled_quantity REAL DEFAULT 0,
                    remaining_quantity REAL,
                    average_fill_price REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    time_in_force TEXT DEFAULT 'DAY',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE
                )
            """)
            self.logger.debug("Successfully created 'orders' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'orders' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_order_fills_table(self, cursor: sqlite3.Cursor):
        """Create order fills table"""
        self.logger.debug("Creating 'order_fills' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    fill_id TEXT UNIQUE NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id) ON DELETE CASCADE
                )
            """)
            self.logger.debug("Successfully created 'order_fills' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'order_fills' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_market_data_cache_table(self, cursor: sqlite3.Cursor):
        """Create market data cache table"""
        self.logger.debug("Creating 'market_data_cache' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.logger.debug("Successfully created 'market_data_cache' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'market_data_cache' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_system_metadata_table(self, cursor: sqlite3.Cursor):
        """Create system metadata table"""
        self.logger.debug("Creating 'system_metadata' table [LOG_DB]")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.logger.debug("Successfully created 'system_metadata' table [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Failed to create 'system_metadata' table: {e} [LOG_DB]", exc_info=True)
            raise
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes"""
        self.logger.debug("Creating database indexes [LOG_DB]")
        
        try:
            # Portfolio indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_name ON portfolios(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_active ON portfolios(is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_updated ON portfolios(updated_at DESC)")
            
            # Position indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_portfolio_ticker ON positions(portfolio_id, ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions(updated_at DESC)")
            
            # Transaction indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_portfolio ON transactions(portfolio_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_timestamp ON transactions(portfolio_id, timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_action ON transactions(action)")
            
            # Equity history indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_history_portfolio ON equity_history(portfolio_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_history_portfolio_timestamp ON equity_history(portfolio_id, timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_history_timestamp ON equity_history(timestamp DESC)")
            
            # Order indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_portfolio ON orders(portfolio_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_portfolio_status ON orders(portfolio_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at DESC)")
            
            # Order fills indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_fills_order_id ON order_fills(order_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_fills_timestamp ON order_fills(timestamp DESC)")
            
            # Cache indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_cache_key ON market_data_cache(cache_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_cache_expires ON market_data_cache(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_cache_accessed ON market_data_cache(last_accessed)")
            
            self.logger.debug("Successfully created all database indexes [LOG_DB]")
            
        except Exception as e:
            self.logger.error(f"Failed to create database indexes: {e} [LOG_DB]", exc_info=True)
            raise
            # Remove duplicate index definitions that were incorrectly pasted
        
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                self.logger.warning(f"Index creation warning: {e}")
    
    def get_schema_version(self) -> str:
        """Get current schema version"""
        
        try:
            if not self.db.check_table_exists('system_metadata'):
                return "1.0"  # Legacy version
            
            version = self.db.execute_scalar(
                "SELECT value FROM system_metadata WHERE key = 'schema_version'"
            )
            return version or "1.0"
            
        except Exception:
            return "1.0"
    
    def migrate_schema(self, target_version: str = None) -> bool:
        """Migrate schema to target version"""
        
        current_version = self.get_schema_version()
        target_version = target_version or self.CURRENT_VERSION
        
        if current_version == target_version:
            self.logger.info(f"Schema already at version {current_version}")
            return True
        
        try:
            self.logger.info(f"Migrating schema from {current_version} to {target_version}")
            
            # Version-specific migrations
            if current_version == "1.0" and target_version == "2.0":
                self._migrate_1_0_to_2_0()
            
            # Update version
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
                    VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
                """, (target_version,))
            
            self.logger.info(f"Schema migration completed: {current_version} -> {target_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Schema migration failed: {e}")
            return False
    
    def _migrate_1_0_to_2_0(self):
        """Migrate from version 1.0 to 2.0"""
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Add new columns to existing tables
            migrations = [
                "ALTER TABLE portfolios ADD COLUMN is_active BOOLEAN DEFAULT 1",
                "ALTER TABLE positions ADD COLUMN realized_pnl REAL DEFAULT 0",
                "ALTER TABLE equity_history ADD COLUMN unrealized_pnl REAL DEFAULT 0",
                "ALTER TABLE orders ADD COLUMN remaining_quantity REAL",
                "ALTER TABLE orders ADD COLUMN time_in_force TEXT DEFAULT 'DAY'",
                "ALTER TABLE orders ADD COLUMN expires_at TIMESTAMP",
                "ALTER TABLE order_fills ADD COLUMN fill_id TEXT",
                "ALTER TABLE market_data_cache ADD COLUMN access_count INTEGER DEFAULT 0",
                "ALTER TABLE market_data_cache ADD COLUMN last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "ALTER TABLE transactions ADD COLUMN notes TEXT"
            ]
            
            for migration in migrations:
                try:
                    cursor.execute(migration)
                except sqlite3.Error as e:
                    # Column might already exist
                    if "duplicate column name" not in str(e).lower():
                        raise
            
            # Update remaining_quantity for existing orders
            cursor.execute("""
                UPDATE orders 
                SET remaining_quantity = quantity - filled_quantity 
                WHERE remaining_quantity IS NULL
            """)
    
    def create_backup_schema(self, backup_db_path: str) -> bool:
        """Create a backup of the current schema"""
        
        try:
            # Create backup connection
            backup_conn = sqlite3.connect(backup_db_path)
            
            with self.db.get_connection() as source_conn:
                source_conn.backup(backup_conn)
            
            backup_conn.close()
            
            self.logger.info(f"Schema backup created: {backup_db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Schema backup failed: {e}")
            return False
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate current schema integrity"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'tables_checked': []
        }
        
        required_tables = [
            'portfolios', 'positions', 'transactions', 'equity_history',
            'orders', 'order_fills', 'market_data_cache', 'system_metadata'
        ]
        
        try:
            for table in required_tables:
                if self.db.check_table_exists(table):
                    validation_result['tables_checked'].append(table)
                    
                    # Check table structure
                    table_info = self.db.get_table_info(table)
                    if not table_info:
                        validation_result['errors'].append(f"Table {table} has no columns")
                        validation_result['valid'] = False
                else:
                    validation_result['errors'].append(f"Missing required table: {table}")
                    validation_result['valid'] = False
            
            # Check foreign key constraints
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                
                if fk_violations:
                    validation_result['errors'].extend([
                        f"Foreign key violation: {violation}" for violation in fk_violations
                    ])
                    validation_result['valid'] = False
            
        except Exception as e:
            validation_result['errors'].append(f"Schema validation error: {e}")
            validation_result['valid'] = False
        
        return validation_result
