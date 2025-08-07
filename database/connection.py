"""
Database connection management with pooling and transaction safety
"""
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Generator
import logging
import sys
import os
from pathlib import Path


class DatabaseConnection:
    """
    Production-grade database connection manager
    Features:
    - Connection pooling
    - Transaction safety  
    - WAL mode for concurrency
    - Automatic reconnection
    """
    
    def __init__(self, db_path=None, pool_size=10, enable_wal=True, timeout=30.0, check_same_thread=False):
        """
        Initialize database connection with explicit parameters
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum connections in pool
            enable_wal: Enable WAL mode for better concurrency
            timeout: Connection timeout in seconds
            check_same_thread: SQLite threading check
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatabaseConnection.__init__ ENTRY [LOG_DB]")
        
        # Handle missing db_path
        if db_path is None:
            try:
                from config import Config
                db_path = Config.DATABASE_PATH
                self.logger.info(f"Using database path from config: {db_path} [LOG_DB]")
            except ImportError:
                db_path = "institutional_trading.db"
                self.logger.warning("Config module not found, using default database path [LOG_DB]")
        
        self.db_path = db_path
        self.pool_size = pool_size
        self.enable_wal = enable_wal
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._initialized = False
        
        # Ensure database directory exists
        db_file = Path(self.db_path)
        if db_file.parent and not db_file.parent.exists():
            os.makedirs(db_file.parent, exist_ok=True)
            self.logger.info(f"Created database directory: {db_file.parent} [LOG_DB]")
        
        # Initialize database
        self.logger.info("Initializing database... [LOG_DB]")
        self._initialize_database()
        self.logger.info("DatabaseConnection.__init__ EXIT [LOG_DB]")
    
    def _initialize_database(self):
        """Initialize database with proper settings"""
        self.logger.info("DatabaseConnection._initialize_database ENTRY [LOG_DB]")
        
        if self._initialized:
            self.logger.info("Database already initialized, skipping [LOG_DB]")
            self.logger.info("DatabaseConnection._initialize_database EXIT [LOG_DB]")
            return
        
        try:
            self.logger.info("Acquiring database connection for initialization [LOG_DB]")
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency
                if self.enable_wal:
                    self.logger.info("Enabling WAL journal mode [LOG_DB]")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    wal_result = cursor.fetchone()
                    self.logger.info(f"WAL mode set to: {wal_result[0]} [LOG_DB]")
                
                # Performance settings
                self.logger.info("Configuring database performance settings [LOG_DB]")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.execute("PRAGMA foreign_keys=ON")
                
                conn.commit()
                
            self._initialized = True
            self.logger.info(f"Database initialized successfully: {self.db_path} [LOG_DB]")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.info("DatabaseConnection._initialize_database EXIT [LOG_DB]")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection from pool with automatic cleanup"""
        self.logger.debug("DatabaseConnection.get_connection ENTRY [LOG_DB]")
        
        conn = None
        try:
            # Try to get connection from pool
            with self._pool_lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
                    self.logger.debug(f"Reusing connection from pool. Pool size: {len(self._connection_pool)} [LOG_DB]")
                
            # Create new connection if pool is empty
            if conn is None:
                self.logger.debug("Creating new database connection [LOG_DB]")
                conn = self._create_connection()
                self.logger.debug("New database connection created [LOG_DB]")
            
            # Test connection
            conn.execute("SELECT 1")
            
            yield conn
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            # Don't return broken connection to pool
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            raise
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise
            
        finally:
            # Return connection to pool or close if pool is full
            if conn:
                try:
                    # Reset connection state
                    conn.rollback()
                    
                    # Return to pool if there's room
                    with self._pool_lock:
                        if len(self._connection_pool) < self.pool_size:
                            self._connection_pool.append(conn)
                            self.logger.debug(f"Returned connection to pool. Pool size: {len(self._connection_pool)} [LOG_DB]")
                        else:
                            conn.close()
                            self.logger.debug("Closed connection (pool full) [LOG_DB]")
                except Exception as e:
                    self.logger.error(f"Error returning connection to pool: {e} [LOG_DB]")
                    try:
                        conn.close()
                        self.logger.debug("Closed connection after error [LOG_DB]")
                    except Exception as close_error:
                        self.logger.error(f"Error closing connection: {close_error} [LOG_DB]", exc_info=True)
        
        self.logger.debug("DatabaseConnection.get_connection EXIT [LOG_DB]")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper settings"""
        self.logger.debug("DatabaseConnection._create_connection ENTRY [LOG_DB]")
        
        try:
            self.logger.debug(f"Connecting to database: {self.db_path} [LOG_DB]")
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=self.check_same_thread,
                isolation_level=None  # Use explicit transactions
            )
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Set row factory for named tuples
            conn.row_factory = sqlite3.Row
            
            self.logger.debug("Database connection created successfully [LOG_DB]")
            return conn
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create database connection: {e} [LOG_DB]")
            raise
        finally:
            self.logger.debug("DatabaseConnection._create_connection EXIT [LOG_DB]")
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Database transaction context manager with rollback"""
        self.logger.debug("DatabaseConnection.transaction ENTRY [LOG_DB]")
        
        # Get connection and start transaction
        conn = self.get_connection()
        self.logger.debug("Starting database transaction [LOG_DB]")
        conn.execute("BEGIN IMMEDIATE")
        
        try:
            self.logger.debug("Yielding connection for transaction [LOG_DB]")
            yield conn
            conn.commit()
            self.logger.debug("Transaction committed successfully [LOG_DB]")
        except Exception as e:
            self.logger.error(f"Transaction error, rolling back: {e} [LOG_DB]")
            conn.rollback()
            self.logger.error(f"Transaction rolled back due to error [LOG_DB]")
            raise
        finally:
            self.logger.debug("Transaction context manager cleanup [LOG_DB]")
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute SELECT query and return results"""
        self.logger.debug(f"DatabaseConnection.execute_query ENTRY [LOG_DB] Query: {query[:100]}...")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Query execution failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.debug("DatabaseConnection.execute_query EXIT [LOG_DB]")
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update/insert/delete query and return rowcount"""
        self.logger.debug(f"DatabaseConnection.execute_update ENTRY [LOG_DB] Query: {query[:100]}...")
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                self.logger.debug(f"Executing update with params: {params} [LOG_DB]")
                cursor.execute(query, params)
                rowcount = cursor.rowcount
                conn.commit()
                self.logger.debug(f"Update executed successfully, affected {rowcount} rows [LOG_DB]")
                return rowcount
        except Exception as e:
            self.logger.error(f"Update execution failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.debug("DatabaseConnection.execute_update EXIT [LOG_DB]")
    
    def execute_scalar(self, query: str, params: tuple = ()) -> Optional[any]:
        """Execute query and return single value"""
        self.logger.debug(f"DatabaseConnection.execute_scalar ENTRY [LOG_DB] Query: {query[:100]}...")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            self.logger.error(f"Scalar execution failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.debug("DatabaseConnection.execute_scalar EXIT [LOG_DB]")
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        self.logger.debug(f"DatabaseConnection.check_table_exists ENTRY [LOG_DB] Table: {table_name}")
        
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        
        try:
            result = self.execute_scalar(query, (table_name,))
            return result is not None
        except Exception as e:
            self.logger.error(f"Table existence check failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.debug("DatabaseConnection.check_table_exists EXIT [LOG_DB]")
    
    def get_table_info(self, table_name: str) -> list:
        """Get table schema information"""
        self.logger.debug(f"DatabaseConnection.get_table_info ENTRY [LOG_DB] Table: {table_name}")
        
        query = f"PRAGMA table_info({table_name})"
        
        try:
            return self.execute_query(query)
        except Exception as e:
            self.logger.error(f"Table info retrieval failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.debug("DatabaseConnection.get_table_info EXIT [LOG_DB]")
    
    def vacuum_database(self):
        """Vacuum database to optimize storage"""
        self.logger.info("DatabaseConnection.vacuum_database ENTRY [LOG_DB]")
        
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            self.logger.info("Database vacuum completed")
        except Exception as e:
            self.logger.error(f"Database vacuum failed: {e} [LOG_DB]")
            raise
        finally:
            self.logger.info("DatabaseConnection.vacuum_database EXIT [LOG_DB]")
    
    def get_database_size(self) -> int:
        """Get database file size in bytes"""
        self.logger.debug("DatabaseConnection.get_database_size ENTRY [LOG_DB]")
        
        try:
            return os.path.getsize(self.db_path)
        except OSError:
            return 0
        finally:
            self.logger.debug("DatabaseConnection.get_database_size EXIT [LOG_DB]")
    
    def close_all(self):
        """Close all connections in the pool"""
        self.logger.info("DatabaseConnection.close_all ENTRY [LOG_DB]")
        try:
            with self._pool_lock:
                pool_size = len(self._connection_pool)
                self.logger.info(f"Closing {pool_size} database connections [LOG_DB]")
                
                for i, conn in enumerate(self._connection_pool, 1):
                    try:
                        conn.close()
                        self.logger.debug(f"Closed connection {i}/{pool_size} [LOG_DB]")
                    except Exception as e:
                        self.logger.error(f"Error closing connection {i}: {e} [LOG_DB]")
                
                self._connection_pool.clear()
                self._initialized = False
                self.logger.info("All database connections closed [LOG_DB]")
                
        except Exception as e:
            self.logger.error(f"Error in close_all: {e} [LOG_DB]")
            raise
        finally:
            self.logger.info("DatabaseConnection.close_all EXIT [LOG_DB]")
    
    # Async compatibility methods for your platform
    async def initialize(self):
        """Async initialization (compatibility method)"""
        self._initialize_database()
    
    async def close(self):
        """Async cleanup (compatibility method)"""
        self.close_all_connections()
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.close_all_connections()
        except:
            pass
