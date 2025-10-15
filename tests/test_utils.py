"""
Tests for the utils module of PyNexus.
"""

import time
import pynexus as nx
from pynexus.utils.cache import memoize, LRUCache
from pynexus.utils.lazy import LazyLoader, lazy_import
from pynexus.utils.profile import timer, profiler, profile


def test_memoize():
    """Test the memoize decorator."""
    call_count = 0
    
    @memoize
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call with same argument (should use cache)
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Should not increment


def test_lru_cache():
    """Test the LRUCache class."""
    cache = LRUCache(maxsize=2)
    
    # Add items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Check items
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    
    # Add third item (should evict key1)
    cache.put("key3", "value3")
    
    # key1 should be evicted, key2 and key3 should remain
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_lazy_loader():
    """Test the LazyLoader class."""
    # Create a lazy loader for a standard library module
    lazy_json = LazyLoader('json')
    
    # This should not raise an exception (module not loaded yet)
    assert lazy_json is not None
    
    # Accessing an attribute should load the module
    dumps_func = lazy_json.dumps
    assert dumps_func is not None


def test_timer_decorator():
    """Test the timer decorator."""
    @timer
    def slow_function():
        time.sleep(0.01)  # Sleep for 10ms
        return "done"
    
    # This should execute without error and print timing info
    result = slow_function()
    assert result == "done"


def test_profiler():
    """Test the Profiler class."""
    # Start timing
    profiler.start("test_operation")
    
    # Simulate some work
    time.sleep(0.01)
    
    # Stop timing
    elapsed = profiler.stop("test_operation")
    
    # Should be greater than 0
    assert elapsed > 0


if __name__ == "__main__":
    test_memoize()
    test_lru_cache()
    test_lazy_loader()
    test_timer_decorator()
    test_profiler()
    print("All utils tests passed!")