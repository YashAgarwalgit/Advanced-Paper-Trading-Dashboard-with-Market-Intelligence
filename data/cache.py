"""
Async data cache with TTL support
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

class AsyncDataCache:
    """Async data cache with TTL support"""
    
    def __init__(self, default_ttl: float = 300):
        self.logger = logging.getLogger(__name__)
        self.logger.info("AsyncDataCache.__init__ ENTRY")
        self.default_ttl = default_ttl
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttls: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.logger.info("AsyncDataCache.__init__ EXIT")
    
    async def get(self, key: str) -> Optional[Any]:
        self.logger.info(f"AsyncDataCache.get ENTRY: {key}")
        """Get cached value if not expired"""
        async with self._lock:
            if key in self.cache and key in self.timestamps:
                self.logger.info(f"AsyncDataCache.get cache HIT: {key}")
                ttl = self.ttls.get(key, self.default_ttl)
                age = time.time() - self.timestamps[key]
                
                if age < ttl:
                    return self.cache[key]
                else:
                    # Expired - remove from cache
                    await self._remove_key(key)
                    self.logger.info(f"AsyncDataCache.get cache EXPIRED: {key}")
            
            self.logger.info(f"AsyncDataCache.get EXIT: {key} (miss or expired)")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        self.logger.info(f"AsyncDataCache.set ENTRY: {key}")
        """Set cached value with optional custom TTL"""
        async with self._lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.ttls[key] = ttl if ttl is not None else self.default_ttl
    
    async def delete(self, key: str) -> bool:
        self.logger.info(f"AsyncDataCache.delete ENTRY: {key}")
        """Delete specific key from cache"""
        async with self._lock:
            return await self._remove_key(key)
    
    async def _remove_key(self, key: str) -> bool:
        self.logger.info(f"AsyncDataCache._remove_key ENTRY: {key}")
        """Remove key from all cache dictionaries"""
        removed = False
        
        if key in self.cache:
            del self.cache[key]
            removed = True
        
        if key in self.timestamps:
            del self.timestamps[key]
        
        if key in self.ttls:
            del self.ttls[key]
        
        return removed
    
    async def cleanup(self):
        self.logger.info("AsyncDataCache.cleanup ENTRY")
        """Clean up expired entries"""
        async with self._lock:
            now = time.time()
            expired_keys = []
            
            for key, timestamp in self.timestamps.items():
                ttl = self.ttls.get(key, self.default_ttl)
                if now - timestamp > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._remove_key(key)
    
    async def clear(self):
        self.logger.info("AsyncDataCache.clear ENTRY")
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
    
    async def size(self) -> int:
        self.logger.info("AsyncDataCache.size ENTRY")
        """Get current cache size"""
        async with self._lock:
            return len(self.cache)
    
    async def keys(self) -> list:
        self.logger.info("AsyncDataCache.keys ENTRY")
        """Get all cache keys"""
        async with self._lock:
            return list(self.cache.keys())
