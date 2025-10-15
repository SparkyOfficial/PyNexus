"""
benchmark module for PyNexus.

цей модуль надає функції для тестування продуктивності.
автор: Андрій Будильников
"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Any, List, Tuple
from pynexus.utils.profile import timer_context


def benchmark_function(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """
    виміряти продуктивність функції.
    
    args:
        func: функція для тестування
        *args: аргументи функції
        **kwargs: ключові аргументи функції
        
    returns:
        tuple: (час виконання, результат функції)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def compare_functions(funcs: List[Callable], *args, **kwargs) -> dict:
    """
    порівняти продуктивність кількох функцій.
    
    args:
        funcs: список функцій для порівняння
        *args: аргументи для всіх функцій
        **kwargs: ключові аргументи для всіх функцій
        
    returns:
        dict: результати порівняння
    """
    results = {}
    
    for i, func in enumerate(funcs):
        exec_time, result = benchmark_function(func, *args, **kwargs)
        results[f"function_{i+1}"] = {
            'time': exec_time,
            'result': result
        }
    
    return results


def benchmark_array_operations(size: int = 1000000) -> dict:
    """
    тестування операцій з масивами.
    
    args:
        size: розмір масиву
        
    returns:
        dict: результати тестування
    """
    # create test arrays
    arr1 = np.random.random(size)
    arr2 = np.random.random(size)
    
    results = {}
    
    # test addition
    with timer_context("array_addition") as t:
        result = arr1 + arr2
    results['addition'] = t
    
    # test multiplication
    with timer_context("array_multiplication") as t:
        result = arr1 * arr2
    results['multiplication'] = t
    
    # test dot product
    with timer_context("array_dot_product") as t:
        result = np.dot(arr1, arr2)
    results['dot_product'] = t
    
    return results


def benchmark_dataframe_operations(rows: int = 100000, cols: int = 10) -> dict:
    """
    тестування операцій з dataframe.
    
    args:
        rows: кількість рядків
        cols: кількість стовпців
        
    returns:
        dict: результати тестування
    """
    # create test dataframe
    data = {f'col_{i}': np.random.random(rows) for i in range(cols)}
    df = pd.DataFrame(data)
    
    results = {}
    
    # test groupby operation
    df['category'] = np.random.choice(['a', 'b', 'c'], rows)
    
    with timer_context("dataframe_groupby") as t:
        result = df.groupby('category').mean()
    results['groupby'] = t
    
    # test filter operation
    with timer_context("dataframe_filter") as t:
        result = df[df['col_0'] > 0.5]
    results['filter'] = t
    
    # test aggregation
    with timer_context("dataframe_aggregation") as t:
        result = df.mean()
    results['aggregation'] = t
    
    return results


def run_benchmarks() -> dict:
    """
    запустити всі тести продуктивності.
    
    returns:
        dict: зведені результати
    """
    results = {}
    
    print("Running PyNexus benchmarks...")
    
    # array operations benchmark
    print("\n1. Array operations benchmark:")
    array_results = benchmark_array_operations()
    results['array_operations'] = array_results
    
    # dataframe operations benchmark
    print("\n2. DataFrame operations benchmark:")
    df_results = benchmark_dataframe_operations()
    results['dataframe_operations'] = df_results
    
    print("\nBenchmarks completed!")
    return results