"""
profiling utilities for PyNexus.

цей модуль надає функції для профілювання коду.
автор: Андрій Будильников
"""

import time
import functools
from typing import Callable, Any
from contextlib import contextmanager


def timer(func: Callable) -> Callable:
    """
    декоратор для вимірювання часу виконання функції.
    
    args:
        func: функція для вимірювання
        
    returns:
        callable: обгорнута функція
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@contextmanager
def timer_context(name: str = "operation"):
    """
    контекстний менеджер для вимірювання часу.
    
    args:
        name: ім'я операції
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{name} executed in {end_time - start_time:.4f} seconds")


class Profiler:
    """
    простий профайлер для вимірювання продуктивності.
    """
    
    def __init__(self):
        """ініціалізація профайлера."""
        self.times = {}
    
    def start(self, name: str) -> None:
        """
        почати вимірювання.
        
        args:
            name: ім'я операції
        """
        self.times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """
        зупинити вимірювання.
        
        args:
            name: ім'я операції
            
        returns:
            float: час виконання
        """
        if name in self.times:
            elapsed = time.time() - self.times[name]
            print(f"{name} executed in {elapsed:.4f} seconds")
            return elapsed
        return 0.0
    
    def report(self) -> dict:
        """
        отримати звіт про вимірювання.
        
        returns:
            dict: звіт про час виконання
        """
        return self.times.copy()


# global profiler instance
profiler = Profiler()


def profile(func: Callable) -> Callable:
    """
    декоратор для профілювання функції.
    
    args:
        func: функція для профілювання
        
    returns:
        callable: обгорнута функція
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler.start(func.__name__)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.stop(func.__name__)
    return wrapper