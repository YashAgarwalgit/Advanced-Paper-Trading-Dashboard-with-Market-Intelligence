"""
Async rate limiter for API calls
"""
import logging
import asyncio
import time
from typing import List

# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

class AsyncRateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, calls: int, period: float):
        self.logger = logging.getLogger(__name__)
        self.logger.info("AsyncRateLimiter.__init__ ENTRY")
        self.calls = calls
        self.period = period
        self.call_times: List[float] = []
        self._lock = asyncio.Lock()
        self.logger.info("AsyncRateLimiter.__init__ EXIT")
    
    async def acquire(self):
        self.logger.info("AsyncRateLimiter.acquire ENTRY")
        """Acquire permission to make an API call"""
        async with self._lock:
            now = time.time()
            
            # Remove old call times
            self.call_times = [t for t in self.call_times if now - t < self.period]
            
            if len(self.call_times) >= self.calls:
                sleep_time = self.period - (now - self.call_times[0])
                if sleep_time > 0:
                    self.logger.info(f"AsyncRateLimiter.acquire sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            self.call_times.append(now)
            self.logger.info("AsyncRateLimiter.acquire EXIT (permit granted)")
    
    async def get_status(self) -> dict:
        self.logger.info("AsyncRateLimiter.get_status ENTRY")
        """Get current rate limiter status"""
        async with self._lock:
            now = time.time()
            # Remove old call times
            self.call_times = [t for t in self.call_times if now - t < self.period]
            
            return {
                "calls_made": len(self.call_times),
                "calls_limit": self.calls,
                "period": self.period,
                "calls_remaining": max(0, self.calls - len(self.call_times)),
                "reset_time": self.call_times[0] + self.period if self.call_times else now
            }
