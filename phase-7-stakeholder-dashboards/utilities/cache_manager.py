"""
Cache Manager for Dashboard API

Handles Redis caching with fallback to in-memory caching.
"""

import json
from typing import Optional, Any
import redis
from functools import lru_cache


class CacheManager:
    """
    Manages caching for API responses.
    Uses Redis if available, falls back to in-memory LRU cache.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            print(f"Connected to Redis at {redis_url}")
        except Exception as e:
            print(f"Redis not available, using in-memory cache: {e}")
            self._memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                print(f"Error getting from Redis: {e}")
        else:
            return self._memory_cache.get(key)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default 1 hour)
        """
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
            except Exception as e:
                print(f"Error setting in Redis: {e}")
        else:
            self._memory_cache[key] = value
    
    def delete(self, key: str):
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        """
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"Error deleting from Redis: {e}")
        else:
            self._memory_cache.pop(key, None)
    
    def clear_all(self):
        """Clear all cached values."""
        if self.redis_client:
            try:
                self.redis_client.flushall()
            except Exception as e:
                print(f"Error clearing Redis: {e}")
        else:
            self._memory_cache.clear()
