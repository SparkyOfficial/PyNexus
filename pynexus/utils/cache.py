"""
caching utilities for PyNexus.

цей модуль надає функції для кешування результатів обчислень.
автор: Андрій Будильников
"""

import functools
import hashlib
import pickle
from typing import Any, Callable


def memoize(func: Callable) -> Callable:
    """
    декоратор для кешування результатів функції.
    
    args:
        func: функція для кешування
        
    returns:
        callable: функція з кешуванням
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # create a key from args and kwargs
        key = _make_hashable((args, kwargs))
        
        # check if result is in cache
        if key in cache:
            return cache[key]
        
        # compute and cache result
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper


def _make_hashable(obj: Any) -> Any:
    """
    робить об'єкт придатним для хешування.
    
    args:
        obj: об'єкт для хешування
        
    returns:
        any: хешований об'єкт
    """
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((_make_hashable(k), _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # for other objects, use pickle
        return hashlib.md5(pickle.dumps(obj)).hexdigest()


class LRUCache:
    """
    lru (least recently used) кеш.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        ініціалізація lru кешу.
        
        args:
            maxsize: максимальний розмір кешу
        """
        self.maxsize = maxsize
        self.cache = {}
        self.order = []
    
    def get(self, key: Any) -> Any:
        """
        отримати значення з кешу.
        
        args:
            key: ключ
            
        returns:
            any: значення або none якщо не знайдено
        """
        if key in self.cache:
            # move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """
        додати значення до кешу.
        
        args:
            key: ключ
            value: значення
        """
        if key in self.cache:
            # update existing
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # add new
            if len(self.cache) >= self.maxsize:
                # remove least recently used
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.order.append(key)
    
    def clear(self) -> None:
        """очистити кеш."""
        self.cache.clear()
        self.order.clear()
    
    def info(self) -> dict:
        """інформація про кеш."""
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'keys': list(self.cache.keys())
        }