"""
core module for PyNexus.

цей модуль надає обгортки для основних структур даних.
автор: Андрій Будильников
"""

import numpy as np
import pandas as pd


def array(data, *args, **kwargs):
    """
    create a pynexus array (numpy wrapper).
    
    args:
        data: array-like data
        *args: additional arguments to pass to np.array
        **kwargs: additional keyword arguments to pass to np.array
        
    returns:
        numpy.ndarray: a numpy array
        
    example:
        >>> arr = array([1, 2, 3])
        >>> print(arr)
        [1 2 3]
    """
    return np.array(data, *args, **kwargs)


def table(data=None, *args, **kwargs):
    """
    create a pynexus table (pandas dataframe wrapper).
    
    args:
        data: data to create dataframe from
        *args: additional arguments to pass to pd.dataframe
        **kwargs: additional keyword arguments to pass to pd.dataframe
        
    returns:
        pandas.dataframe: a pandas dataframe
        
    example:
        >>> df = table({'a': [1, 2], 'b': [3, 4]})
        >>> print(df)
           a  b
        0  1  3
        1  2  4
    """
    return pd.DataFrame(data, *args, **kwargs)


def matrix(data, *args, **kwargs):
    """
    create a pynexus matrix (numpy matrix wrapper).
    
    args:
        data: array-like data
        *args: additional arguments to pass to np.matrix
        **kwargs: additional keyword arguments to pass to np.matrix
        
    returns:
        numpy.matrix: a numpy matrix
        
    example:
        >>> mat = matrix([[1, 2], [3, 4]])
        >>> print(mat)
        [[1 2]
         [3 4]]
    """
    return np.matrix(data, *args, **kwargs)


def series(data, *args, **kwargs):
    """
    create a pynexus series (pandas series wrapper).
    
    args:
        data: array-like data
        *args: additional arguments to pass to pd.series
        **kwargs: additional keyword arguments to pass to pd.series
        
    returns:
        pandas.series: a pandas series
        
    example:
        >>> s = series([1, 2, 3, 4])
        >>> print(s)
        0    1
        1    2
        2    3
        3    4
        dtype: int64
    """
    return pd.Series(data, *args, **kwargs)


# extended array operations
def zeros(shape, dtype=float, order='C'):
    """
    створити масив заповнений нулями.
    
    args:
        shape: форма масиву (int або tuple of ints)
        dtype: тип даних (за замовчуванням float)
        order: порядок зберігання ('c' або 'f')
        
    returns:
        numpy.ndarray: масив з нулями
        
    example:
        >>> arr = zeros(5)
        >>> print(arr)
        [0. 0. 0. 0. 0.]
        
        >>> arr = zeros((2, 3))
        >>> print(arr)
        [[0. 0. 0.]
         [0. 0. 0.]]
    """
    return np.zeros(shape, dtype=dtype, order=order)


def ones(shape, dtype=float, order='C'):
    """
    створити масив заповнений одиницями.
    
    args:
        shape: форма масиву (int або tuple of ints)
        dtype: тип даних (за замовчуванням float)
        order: порядок зберігання ('c' або 'f')
        
    returns:
        numpy.ndarray: масив з одиницями
        
    example:
        >>> arr = ones(5)
        >>> print(arr)
        [1. 1. 1. 1. 1.]
        
        >>> arr = ones((2, 3))
        >>> print(arr)
        [[1. 1. 1.]
         [1. 1. 1.]]
    """
    return np.ones(shape, dtype=dtype, order=order)


def full(shape, fill_value, dtype=None, order='C'):
    """
    створити масив заповнений певним значенням.
    
    args:
        shape: форма масиву (int або tuple of ints)
        fill_value: значення для заповнення
        dtype: тип даних (за замовчуванням визначається з fill_value)
        order: порядок зберігання ('c' або 'f')
        
    returns:
        numpy.ndarray: масив з певним значенням
        
    example:
        >>> arr = full(5, 3.14)
        >>> print(arr)
        [3.14 3.14 3.14 3.14 3.14]
        
        >>> arr = full((2, 3), 7)
        >>> print(arr)
        [[7 7 7]
         [7 7 7]]
    """
    return np.full(shape, fill_value, dtype=dtype, order=order)


def eye(N, M=None, k=0, dtype=float, order='C'):
    """
    створити двовимірний масив з одиницями на діагоналі.
    
    args:
        n: кількість рядків
        m: кількість стовпців (за замовчуванням n)
        k: індекс діагоналі (0 - головна діагональ)
        dtype: тип даних (за замовчуванням float)
        order: порядок зберігання ('c' або 'f')
        
    returns:
        numpy.ndarray: одинична матриця
        
    example:
        >>> arr = eye(3)
        >>> print(arr)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
         
        >>> arr = eye(3, 4, k=1)
        >>> print(arr)
        [[0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]
    """
    return np.eye(N, M=M, k=k, dtype=dtype, order=order)


def identity(n, dtype=float):
    """
    створити квадратну одиничну матрицю.
    
    args:
        n: розмір матриці
        dtype: тип даних (за замовчуванням float)
        
    returns:
        numpy.ndarray: квадратна одинична матриця
        
    example:
        >>> arr = identity(3)
        >>> print(arr)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    return np.identity(n, dtype=dtype)


def empty(shape, dtype=float, order='C'):
    """
    створити порожній масив (не ініціалізований).
    
    args:
        shape: форма масиву (int або tuple of ints)
        dtype: тип даних (за замовчуванням float)
        order: порядок зберігання ('c' або 'f')
        
    returns:
        numpy.ndarray: порожній масив
        
    example:
        >>> arr = empty(5)
        >>> print(arr.shape)
        (5,)
        
        >>> arr = empty((2, 3))
        >>> print(arr.shape)
        (2, 3)
    """
    return np.empty(shape, dtype=dtype, order=order)


def arange(start, stop=None, step=1, dtype=None):
    """
    створити масив з рівномірно розподілених значень.
    
    args:
        start: початкове значення (якщо stop не задано, то це stop)
        stop: кінцеве значення (не включається)
        step: крок (за замовчуванням 1)
        dtype: тип даних (за замовчуванням визначається автоматично)
        
    returns:
        numpy.ndarray: масив з рівномірно розподілених значень
        
    example:
        >>> arr = arange(5)
        >>> print(arr)
        [0 1 2 3 4]
        
        >>> arr = arange(2, 10, 2)
        >>> print(arr)
        [2 4 6 8]
    """
    return np.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """
    створити масив з рівномірно розподілених чисел у вказаному діапазоні.
    
    args:
        start: початкове значення
        stop: кінцеве значення
        num: кількість елементів (за замовчуванням 50)
        endpoint: чи включати кінцеве значення (за замовчуванням true)
        retstep: чи повертати крок (за замовчуванням false)
        dtype: тип даних (за замовчуванням визначається автоматично)
        axis: вісь для вставки нових елементів
        
    returns:
        numpy.ndarray: масив з рівномірно розподілених чисел
        або tuple (масив, крок) якщо retstep=true
        
    example:
        >>> arr = linspace(0, 10, 5)
        >>> print(arr)
        [ 0.   2.5  5.   7.5 10. ]
        
        >>> arr, step = linspace(0, 10, 5, retstep=true)
        >>> print(step)
        2.5
    """
    return np.linspace(start, stop, num, endpoint, retstep, dtype, axis)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """
    створити масив з логарифмічно розподілених чисел.
    
    args:
        start: початкове значення (base ** start)
        stop: кінцеве значення (base ** stop)
        num: кількість елементів (за замовчуванням 50)
        endpoint: чи включати кінцеве значення (за замовчуванням true)
        base: основа логарифма (за замовчуванням 10.0)
        dtype: тип даних (за замовчуванням визначається автоматично)
        axis: вісь для вставки нових елементів
        
    returns:
        numpy.ndarray: масив з логарифмічно розподілених чисел
        
    example:
        >>> arr = logspace(0, 2, 5)
        >>> print(arr)
        [  1.   3.16 10.  31.62 100. ]
    """
    return np.logspace(start, stop, num, endpoint, base, dtype, axis)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    створити масив з геометрично розподілених чисел.
    
    args:
        start: початкове значення
        stop: кінцеве значення
        num: кількість елементів (за замовчуванням 50)
        endpoint: чи включати кінцеве значення (за замовчуванням true)
        dtype: тип даних (за замовчуванням визначається автоматично)
        axis: вісь для вставки нових елементів
        
    returns:
        numpy.ndarray: масив з геометрично розподілених чисел
        
    example:
        >>> arr = geomspace(1, 1000, 4)
        >>> print(arr)
        [   1.   10.  100. 1000.]
    """
    return np.geomspace(start, stop, num, endpoint, dtype, axis)


def random_uniform(low=0.0, high=1.0, size=None):
    """
    створити масив з випадковими числами з рівномірного розподілу.
    
    args:
        low: нижня межа (за замовчуванням 0.0)
        high: верхня межа (за замовчуванням 1.0)
        size: форма масиву (за замовчуванням одна випадкова величина)
        
    returns:
        numpy.ndarray: масив з випадковими числами
        
    example:
        >>> arr = random_uniform(0, 10, 5)
        >>> print(arr)
        [3.2 7.8 1.5 9.1 4.6]  # приклад, значення будуть різні
    """
    return np.random.uniform(low, high, size)


def random_normal(loc=0.0, scale=1.0, size=None):
    """
    створити масив з випадковими числами з нормального розподілу.
    
    args:
        loc: середнє значення (за замовчуванням 0.0)
        scale: стандартне відхилення (за замовчуванням 1.0)
        size: форма масиву (за замовчуванням одна випадкова величина)
        
    returns:
        numpy.ndarray: масив з випадковими числами
        
    example:
        >>> arr = random_normal(0, 1, 5)
        >>> print(arr)
        [-0.5  1.2  0.3 -1.1  0.8]  # приклад, значення будуть різні
    """
    return np.random.normal(loc, scale, size)


def random_integers(low, high=None, size=None):
    """
    створити масив з випадковими цілими числами.
    
    args:
        low: нижня межа (або верхня, якщо high не задано)
        high: верхня межа (за замовчуванням none)
        size: форма масиву (за замовчуванням одна випадкова величина)
        
    returns:
        numpy.ndarray: масив з випадковими цілими числами
        
    example:
        >>> arr = random_integers(1, 10, 5)
        >>> print(arr)
        [3 7 1 9 4]  # приклад, значення будуть різні
    """
    if high is None:
        return np.random.randint(low, size=size)
    else:
        return np.random.randint(low, high, size=size)


def reshape(a, newshape, order='C'):
    """
    змінити форму масиву.
    
    args:
        a: вхідний масив
        newshape: нова форма (int або tuple of ints)
        order: порядок читання елементів ('c', 'f', 'a')
        
    returns:
        numpy.ndarray: масив з новою формою
        
    example:
        >>> arr = array([1, 2, 3, 4, 5, 6])
        >>> reshaped = reshape(arr, (2, 3))
        >>> print(reshaped)
        [[1 2 3]
         [4 5 6]]
    """
    return np.reshape(a, newshape, order)


def flatten(a, order='C'):
    """
    перетворити масив в одновимірний.
    
    args:
        a: вхідний масив
        order: порядок читання елементів ('c', 'f', 'a')
        
    returns:
        numpy.ndarray: одновимірний масив
        
    example:
        >>> arr = array([[1, 2], [3, 4]])
        >>> flat = flatten(arr)
        >>> print(flat)
        [1 2 3 4]
    """
    return np.ravel(a, order=order)


def transpose(a, axes=None):
    """
    транспонувати масив.
    
    args:
        a: вхідний масив
        axes: послідовність осей (за замовчуванням зворотній порядок)
        
    returns:
        numpy.ndarray: транспонований масив
        
    example:
        >>> arr = array([[1, 2, 3], [4, 5, 6]])
        >>> transposed = transpose(arr)
        >>> print(transposed)
        [[1 4]
         [2 5]
         [3 6]]
    """
    return np.transpose(a, axes)


def concatenate(arrays, axis=0, out=None):
    """
    об'єднати масиви вздовж вказаної осі.
    
    args:
        arrays: послідовність масивів
        axis: вісь для об'єднання (за замовчуванням 0)
        out: вихідний масив (за замовчуванням none)
        
    returns:
        numpy.ndarray: об'єднаний масив
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5, 6])
        >>> concatenated = concatenate([arr1, arr2])
        >>> print(concatenated)
        [1 2 3 4 5 6]
        
        >>> arr1 = array([[1, 2], [3, 4]])
        >>> arr2 = array([[5, 6]])
        >>> concatenated = concatenate([arr1, arr2], axis=0)
        >>> print(concatenated)
        [[1 2]
         [3 4]
         [5 6]]
    """
    return np.concatenate(arrays, axis=axis, out=out)


def stack(arrays, axis=0, out=None):
    """
    об'єднати масиви вздовж нової осі.
    
    args:
        arrays: послідовність масивів
        axis: вісь для створення (за замовчуванням 0)
        out: вихідний масив (за замовчуванням none)
        
    returns:
        numpy.ndarray: масив з новою віссю
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5, 6])
        >>> stacked = stack([arr1, arr2])
        >>> print(stacked)
        [[1 2 3]
         [4 5 6]]
    """
    return np.stack(arrays, axis=axis, out=out)


def vstack(tup):
    """
    об'єднати масиви вертикально (по рядках).
    
    args:
        tup: послідовність масивів
        
    returns:
        numpy.ndarray: вертикально об'єднаний масив
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5, 6])
        >>> stacked = vstack([arr1, arr2])
        >>> print(stacked)
        [[1 2 3]
         [4 5 6]]
    """
    return np.vstack(tup)


def hstack(tup):
    """
    об'єднати масиви горизонтально (по стовпцях).
    
    args:
        tup: послідовність масивів
        
    returns:
        numpy.ndarray: горизонтально об'єднаний масив
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5, 6])
        >>> stacked = hstack([arr1, arr2])
        >>> print(stacked)
        [1 2 3 4 5 6]
        
        >>> arr1 = array([[1], [2], [3]])
        >>> arr2 = array([[4], [5], [6]])
        >>> stacked = hstack([arr1, arr2])
        >>> print(stacked)
        [[1 4]
         [2 5]
         [3 6]]
    """
    return np.hstack(tup)


def split(ary, indices_or_sections, axis=0):
    """
    розділити масив на кілька підмасивів.
    
    args:
        ary: вхідний масив
        indices_or_sections: індекси або кількість розділів
        axis: вісь для розділення (за замовчуванням 0)
        
    returns:
        list of numpy.ndarray: список підмасивів
        
    example:
        >>> arr = array([1, 2, 3, 4, 5, 6])
        >>> subarrays = split(arr, 3)
        >>> print([sub.tolist() for sub in subarrays])
        [[1, 2], [3, 4], [5, 6]]
        
        >>> arr = array([[1, 2, 3], [4, 5, 6]])
        >>> subarrays = split(arr, 2, axis=1)
        >>> print([sub.tolist() for sub in subarrays])
        [[[1, 2], [4, 5]], [[3], [6]]]
    """
    return np.split(ary, indices_or_sections, axis=axis)


def array_sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """
    обчислити суму елементів масиву.
    
    args:
        a: вхідний масив
        axis: вісь для сумування (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: сума елементів
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_sum(arr)
        >>> print(result)
        10
        
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_sum(arr, axis=0)
        >>> print(result)
        [4 6]
    """
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def array_mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """
    обчислити середнє значення елементів масиву.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення середнього (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: середнє значення елементів
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_mean(arr)
        >>> print(result)
        2.5
        
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_mean(arr, axis=0)
        >>> print(result)
        [2. 3.]
    """
    return np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def array_std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    """
    обчислити стандартне відхилення елементів масиву.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення стандартного відхилення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        ddof: степені свободи (за замовчуванням 0)
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: стандартне відхилення елементів
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_std(arr)
        >>> print(result)
        1.118033988749895
        
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_std(arr, axis=0)
        >>> print(result)
        [1. 1.]
    """
    return np.std(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def array_var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    """
    обчислити дисперсію елементів масиву.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення дисперсії (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        ddof: степені свободи (за замовчуванням 0)
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: дисперсія елементів
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_var(arr)
        >>> print(result)
        1.25
        
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_var(arr, axis=0)
        >>> print(result)
        [1. 1.]
    """
    return np.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def array_min(a, axis=None, out=None, keepdims=np._NoValue):
    """
    знайти мінімальне значення в масиві.
    
    args:
        a: вхідний масив
        axis: вісь для пошуку мінімуму (за замовчуванням всі елементи)
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: мінімальне значення
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_min(arr)
        >>> print(result)
        1
        
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_min(arr, axis=0)
        >>> print(result)
        [1 2]
    """
    return np.min(a, axis=axis, out=out, keepdims=keepdims)


def array_max(a, axis=None, out=None, keepdims=np._NoValue):
    """
    знайти максимальне значення в масиві.
    
    args:
        a: вхідний масив
        axis: вісь для пошуку максимуму (за замовчуванням всі елементи)
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: максимальне значення
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_max(arr)
        >>> print(result)
        4
        
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_max(arr, axis=0)
        >>> print(result)
        [3 4]
    """
    return np.max(a, axis=axis, out=out, keepdims=keepdims)


def array_argmin(a, axis=None, out=None):
    """
    знайти індекс мінімального значення в масиві.
    
    args:
        a: вхідний масив
        axis: вісь для пошуку мінімуму (за замовчуванням всі елементи)
        out: вихідний масив
        
    returns:
        numpy.ndarray: індекс мінімального значення
        
    example:
        >>> arr = array([3, 1, 4, 1, 5])
        >>> result = array_argmin(arr)
        >>> print(result)
        1
        
        >>> arr = array([[3, 1], [4, 1]])
        >>> result = array_argmin(arr, axis=0)
        >>> print(result)
        [1 1]
    """
    return np.argmin(a, axis=axis, out=out)


def array_argmax(a, axis=None, out=None):
    """
    знайти індекс максимального значення в масиві.
    
    args:
        a: вхідний масив
        axis: вісь для пошуку максимуму (за замовчуванням всі елементи)
        out: вихідний масив
        
    returns:
        numpy.ndarray: індекс максимального значення
        
    example:
        >>> arr = array([3, 1, 4, 1, 5])
        >>> result = array_argmax(arr)
        >>> print(result)
        4
        
        >>> arr = array([[3, 1], [4, 1]])
        >>> result = array_argmax(arr, axis=0)
        >>> print(result)
        [1 0]
    """
    return np.argmax(a, axis=axis, out=out)


def array_sort(a, axis=-1, kind=None, order=None):
    """
    відсортувати масив.
    
    args:
        a: вхідний масив
        axis: вісь для сортування (за замовчуванням остання)
        kind: алгоритм сортування
        order: порядок сортування для структурованих масивів
        
    returns:
        numpy.ndarray: відсортований масив
        
    example:
        >>> arr = array([3, 1, 4, 1, 5])
        >>> result = array_sort(arr)
        >>> print(result)
        [1 1 3 4 5]
        
        >>> arr = array([[3, 1], [4, 1]])
        >>> result = array_sort(arr, axis=0)
        >>> print(result)
        [[3 1]
         [4 1]]
    """
    return np.sort(a, axis=axis, kind=kind, order=order)


def array_argsort(a, axis=-1, kind=None, order=None):
    """
    отримати індекси для сортування масиву.
    
    args:
        a: вхідний масив
        axis: вісь для сортування (за замовчуванням остання)
        kind: алгоритм сортування
        order: порядок сортування для структурованих масивів
        
    returns:
        numpy.ndarray: індекси для сортування
        
    example:
        >>> arr = array([3, 1, 4, 1, 5])
        >>> result = array_argsort(arr)
        >>> print(result)
        [1 3 0 2 4]
    """
    return np.argsort(a, axis=axis, kind=kind, order=order)


def array_unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """
    знайти унікальні значення в масиві.
    
    args:
        a: вхідний масив
        return_index: чи повертати індекси перших входжень (за замовчуванням false)
        return_inverse: чи повертати індекси для відновлення оригіналу (за замовчуванням false)
        return_counts: чи повертати кількість входжень (за замовчуванням false)
        axis: вісь для пошуку унікальних значень (за замовчуванням всі елементи)
        
    returns:
        numpy.ndarray: унікальні значення
        або tuple з додатковими масивами якщо return_* параметри true
        
    example:
        >>> arr = array([1, 1, 2, 2, 3, 3])
        >>> result = array_unique(arr)
        >>> print(result)
        [1 2 3]
        
        >>> arr = array([1, 1, 2, 2, 3, 3])
        >>> unique_vals, indices, inverse, counts = array_unique(arr, return_index=true, return_inverse=true, return_counts=true)
        >>> print(f"unique: {unique_vals}, indices: {indices}, inverse: {inverse}, counts: {counts}")
        unique: [1 2 3], indices: [0 2 4], inverse: [0 0 1 1 2 2], counts: [2 2 2]
    """
    return np.unique(a, return_index=return_index, return_inverse=return_inverse, 
                     return_counts=return_counts, axis=axis)


def array_where(condition, x=None, y=None):
    """
    повернути елементи з x або y в залежності від умови.
    
    args:
        condition: умова (булевий масив)
        x: значення для true елементів (за замовчуванням none)
        y: значення для false елементів (за замовчуванням none)
        
    returns:
        numpy.ndarray: масив з вибраними елементами
        або tuple of arrays якщо x і y не задані
        
    example:
        >>> condition = array([true, false, true, false])
        >>> x = array([1, 2, 3, 4])
        >>> y = array([10, 20, 30, 40])
        >>> result = array_where(condition, x, y)
        >>> print(result)
        [ 1 20  3 40]
        
        >>> arr = array([1, 2, 3, 4])
        >>> indices = array_where(arr > 2)
        >>> print(indices)
        (array([2, 3]),)
    """
    return np.where(condition, x, y)


def array_clip(a, a_min, a_max, out=None):
    """
    обмежити значення масиву в заданому діапазоні.
    
    args:
        a: вхідний масив
        a_min: мінімальне значення
        a_max: максимальне значення
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з обмеженими значеннями
        
    example:
        >>> arr = array([1, 2, 3, 4, 5])
        >>> result = array_clip(arr, 2, 4)
        >>> print(result)
        [2 2 3 4 4]
    """
    return np.clip(a, a_min, a_max, out=out)


def array_abs(x, out=None):
    """
    обчислити абсолютне значення елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з абсолютними значеннями
        
    example:
        >>> arr = array([-1, -2, 3, -4])
        >>> result = array_abs(arr)
        >>> print(result)
        [1 2 3 4]
    """
    return np.abs(x, out=out)


def array_sqrt(x, out=None):
    """
    обчислити квадратний корінь елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з квадратними коренями
        
    example:
        >>> arr = array([1, 4, 9, 16])
        >>> result = array_sqrt(arr)
        >>> print(result)
        [1. 2. 3. 4.]
    """
    return np.sqrt(x, out=out)


def array_exp(x, out=None):
    """
    обчислити експоненту елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з експонентами
        
    example:
        >>> arr = array([0, 1, 2])
        >>> result = array_exp(arr)
        >>> print(result)
        [1.         2.71828183 7.3890561 ]
    """
    return np.exp(x, out=out)


def array_log(x, out=None):
    """
    обчислити натуральний логарифм елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з логарифмами
        
    example:
        >>> arr = array([1, np.e, np.e**2])
        >>> result = array_log(arr)
        >>> print(result)
        [0. 1. 2.]
    """
    return np.log(x, out=out)


def array_sin(x, out=None):
    """
    обчислити синус елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з синусами
        
    example:
        >>> arr = array([0, np.pi/2, np.pi])
        >>> result = array_sin(arr)
        >>> print(result)
        [0.0000000e+00 1.0000000e+00 1.2246468e-16]
    """
    return np.sin(x, out=out)


def array_cos(x, out=None):
    """
    обчислити косинус елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з косинусами
        
    example:
        >>> arr = array([0, np.pi/2, np.pi])
        >>> result = array_cos(arr)
        >>> print(result)
        [ 1.000000e+00  6.123234e-17 -1.000000e+00]
    """
    return np.cos(x, out=out)


def array_tan(x, out=None):
    """
    обчислити тангенс елементів масиву.
    
    args:
        x: вхідний масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: масив з тангенсами
        
    example:
        >>> arr = array([0, np.pi/4, np.pi/2])
        >>> result = array_tan(arr)
        >>> print(result)
        [ 0.00000000e+00  1.00000000e+00  1.63312394e+16]
    """
    return np.tan(x, out=out)


def array_dot(a, b, out=None):
    """
    обчислити скалярний добуток двох масивів.
    
    args:
        a: перший масив
        b: другий масив
        out: вихідний масив
        
    returns:
        numpy.ndarray: скалярний добуток
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5, 6])
        >>> result = array_dot(arr1, arr2)
        >>> print(result)
        32
        
        >>> arr1 = array([[1, 2], [3, 4]])
        >>> arr2 = array([[5, 6], [7, 8]])
        >>> result = array_dot(arr1, arr2)
        >>> print(result)
        [[19 22]
         [43 50]]
    """
    return np.dot(a, b, out=out)


def array_cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    обчислити векторний добуток двох масивів.
    
    args:
        a: перший масив
        b: другий масив
        axisa: вісь першого масиву (за замовчуванням -1)
        axisb: вісь другого масиву (за замовчуванням -1)
        axisc: вісь результату (за замовчуванням -1)
        axis: вісь для векторного добутку (за замовчуванням none)
        
    returns:
        numpy.ndarray: векторний добуток
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5, 6])
        >>> result = array_cross(arr1, arr2)
        >>> print(result)
        [-3  6 -3]
    """
    return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


def array_outer(a, b, out=None):
    """
    обчислити зовнішній добуток двох векторів.
    
    args:
        a: перший вектор
        b: другий вектор
        out: вихідний масив
        
    returns:
        numpy.ndarray: зовнішній добуток
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([4, 5])
        >>> result = array_outer(arr1, arr2)
        >>> print(result)
        [[ 4  5]
         [ 8 10]
         [12 15]]
    """
    return np.outer(a, b, out=out)


def array_kron(a, b):
    """
    обчислити добуток кронекера двох масивів.
    
    args:
        a: перший масив
        b: другий масив
        
    returns:
        numpy.ndarray: добуток кронекера
        
    example:
        >>> arr1 = array([[1, 2], [3, 4]])
        >>> arr2 = array([[0, 1], [1, 0]])
        >>> result = array_kron(arr1, arr2)
        >>> print(result)
        [[0 1 0 2]
         [1 0 2 0]
         [0 3 0 4]
         [3 0 4 0]]
    """
    return np.kron(a, b)


def array_solve(a, b):
    """
    розв'язати систему лінійних рівнянь ax = b.
    
    args:
        a: коефіцієнти матриці
        b: вектор правих частин
        
    returns:
        numpy.ndarray: розв'язок системи
        
    example:
        >>> a = array([[3, 1], [1, 2]])
        >>> b = array([9, 8])
        >>> result = array_solve(a, b)
        >>> print(result)
        [2. 3.]
    """
    return np.linalg.solve(a, b)


def array_inv(a):
    """
    обчислити обернену матрицю.
    
    args:
        a: вхідна матриця
        
    returns:
        numpy.ndarray: обернена матриця
        
    example:
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_inv(arr)
        >>> print(result)
        [[-2.   1. ]
         [ 1.5 -0.5]]
    """
    return np.linalg.inv(a)


def array_det(a):
    """
    обчислити визначник матриці.
    
    args:
        a: вхідна матриця
        
    returns:
        float: визначник матриці
        
    example:
        >>> arr = array([[1, 2], [3, 4]])
        >>> result = array_det(arr)
        >>> print(result)
        -2.0
    """
    return np.linalg.det(a)


def array_eig(a):
    """
    обчислити власні значення та власні вектори матриці.
    
    args:
        a: вхідна матриця
        
    returns:
        tuple: (власні значення, власні вектори)
        
    example:
        >>> arr = array([[1, 2], [2, 1]])
        >>> eigenvalues, eigenvectors = array_eig(arr)
        >>> print(eigenvalues)
        [ 3. -1.]
        >>> print(eigenvectors)
        [[ 0.70710678 -0.70710678]
         [ 0.70710678  0.70710678]]
    """
    return np.linalg.eig(a)


def array_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    обчислити сингулярне розкладання матриці.
    
    args:
        a: вхідна матриця
        full_matrices: чи повертати повні матриці (за замовчуванням true)
        compute_uv: чи обчислювати u та vh (за замовчуванням true)
        hermitian: чи матриця ермітова (за замовчуванням false)
        
    returns:
        tuple: (u, s, vh) або тільки s якщо compute_uv=false
        
    example:
        >>> arr = array([[1, 2], [3, 4], [5, 6]])
        >>> u, s, vh = array_svd(arr)
        >>> print(s)
        [9.52551809 0.51430058]
    """
    return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)


def array_fft(a, n=None, axis=-1, norm=None):
    """
    обчислити дискретне перетворення фур'є.
    
    args:
        a: вхідний масив
        n: кількість точок (за замовчуванням довжина вхідного масиву)
        axis: вісь для перетворення (за замовчуванням -1)
        norm: нормалізація ('backward', 'ortho', 'forward')
        
    returns:
        numpy.ndarray: перетворення фур'є
        
    example:
        >>> arr = array([0, 1, 2, 3])
        >>> result = array_fft(arr)
        >>> print(result)
        [ 6.+0.j -2.+2.j -2.+0.j -2.-2.j]
    """
    return np.fft.fft(a, n=n, axis=axis, norm=norm)


def array_ifft(a, n=None, axis=-1, norm=None):
    """
    обчислити обернене дискретне перетворення фур'є.
    
    args:
        a: вхідний масив
        n: кількість точок (за замовчуванням довжина вхідного масиву)
        axis: вісь для перетворення (за замовчуванням -1)
        norm: нормалізація ('backward', 'ortho', 'forward')
        
    returns:
        numpy.ndarray: обернене перетворення фур'є
        
    example:
        >>> arr = array([6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
        >>> result = array_ifft(arr)
        >>> print(result)
        [0.+0.j 1.+0.j 2.+0.j 3.+0.j]
    """
    return np.fft.ifft(a, n=n, axis=axis, norm=norm)


def array_convolve(a, v, mode='full'):
    """
    обчислити згортку двох одновимірних масивів.
    
    args:
        a: перший масив
        v: другий масив
        mode: режим ('full', 'valid', 'same')
        
    returns:
        numpy.ndarray: згортка масивів
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([0, 1, 0.5])
        >>> result = array_convolve(arr1, arr2)
        >>> print(result)
        [0.  1.  2.5 4.  1.5]
    """
    return np.convolve(a, v, mode=mode)


def array_correlate(a, v, mode='valid'):
    """
    обчислити кореляцію двох одновимірних масивів.
    
    args:
        a: перший масив
        v: другий масив
        mode: режим ('full', 'valid', 'same')
        
    returns:
        numpy.ndarray: кореляція масивів
        
    example:
        >>> arr1 = array([1, 2, 3])
        >>> arr2 = array([0, 1, 0.5])
        >>> result = array_correlate(arr1, arr2)
        >>> print(result)
        [2.  3.5]
    """
    return np.correlate(a, v, mode=mode)


def array_histogram(a, bins=10, range=None, normed=None, weights=None, density=None):
    """
    обчислити гістограму масиву.
    
    args:
        a: вхідний масив
        bins: кількість інтервалів або їх межі
        range: діапазон значень (за замовчуванням [min, max])
        normed: чи нормувати (застарілий параметр)
        weights: ваги елементів
        density: чи повертати щільність (за замовчуванням none)
        
    returns:
        tuple: (гістограма, межі інтервалів)
        
    example:
        >>> arr = array([1, 2, 1, 3, 2, 1, 4, 3, 2, 1])
        >>> hist, bin_edges = array_histogram(arr, bins=4)
        >>> print(hist)
        [4 3 2 1]
        >>> print(bin_edges)
        [1.   1.75 2.5  3.25 4.  ]
    """
    return np.histogram(a, bins=bins, range=range, normed=normed, weights=weights, density=density)


def array_percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False):
    """
    обчислити процентилі масиву.
    
    args:
        a: вхідний масив
        q: процентилі (0-100)
        axis: вісь для обчислення (за замовчуванням всі елементи)
        out: вихідний масив
        overwrite_input: чи можна змінювати вхідний масив (за замовчуванням false)
        method: метод інтерполяції
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: процентилі масиву
        
    example:
        >>> arr = array([1, 2, 3, 4, 5])
        >>> result = array_percentile(arr, 50)
        >>> print(result)
        3.0
        
        >>> result = array_percentile(arr, [25, 50, 75])
        >>> print(result)
        [2. 3. 4.]
    """
    return np.percentile(a, q, axis=axis, out=out, overwrite_input=overwrite_input, method=method, keepdims=keepdims)


def array_nanmean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """
    обчислити середнє значення масиву, ігноруючи nan.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: середнє значення без nan
        
    example:
        >>> arr = array([1, 2, np.nan, 4, 5])
        >>> result = array_nanmean(arr)
        >>> print(result)
        3.0
    """
    return np.nanmean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def array_nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    """
    обчислити стандартне відхилення масиву, ігноруючи nan.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        ddof: степені свободи (за замовчуванням 0)
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: стандартне відхилення без nan
        
    example:
        >>> arr = array([1, 2, np.nan, 4, 5])
        >>> result = array_nanstd(arr)
        >>> print(result)
        1.5811388300841898
    """
    return np.nanstd(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def array_nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    """
    обчислити дисперсію масиву, ігноруючи nan.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        ddof: степені свободи (за замовчуванням 0)
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: дисперсія без nan
        
    example:
        >>> arr = array([1, 2, np.nan, 4, 5])
        >>> result = array_nanvar(arr)
        >>> print(result)
        2.5
    """
    return np.nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def array_nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """
    обчислити суму масиву, ігноруючи nan.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: сума без nan
        
    example:
        >>> arr = array([1, 2, np.nan, 4, 5])
        >>> result = array_nansum(arr)
        >>> print(result)
        12.0
    """
    return np.nansum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def array_nanmin(a, axis=None, out=None, keepdims=np._NoValue):
    """
    знайти мінімальне значення масиву, ігноруючи nan.
    
    args:
        a: вхідний масив
        axis: вісь для пошуку (за замовчуванням всі елементи)
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: мінімальне значення без nan
        
    example:
        >>> arr = array([1, 2, np.nan, 4, 5])
        >>> result = array_nanmin(arr)
        >>> print(result)
        1.0
    """
    return np.nanmin(a, axis=axis, out=out, keepdims=keepdims)


def array_nanmax(a, axis=None, out=None, keepdims=np._NoValue):
    """
    знайти максимальне значення масиву, ігноруючи nan.
    
    args:
        a: вхідний масив
        axis: вісь для пошуку (за замовчуванням всі елементи)
        out: вихідний масив
        keepdims: чи зберігати розмірності (за замовчуванням false)
        
    returns:
        numpy.ndarray: максимальне значення без nan
        
    example:
        >>> arr = array([1, 2, np.nan, 4, 5])
        >>> result = array_nanmax(arr)
        >>> print(result)
        5.0
    """
    return np.nanmax(a, axis=axis, out=out, keepdims=keepdims)


def array_interp(x, xp, fp, left=None, right=None, period=None):
    """
    одновимірна інтерполяція.
    
    args:
        x: точки для інтерполяції
        xp: вузли інтерполяції (повинні бути впорядковані)
        fp: значення в вузлах
        left: значення для x < xp[0] (за замовчуванням fp[0])
        right: значення для x > xp[-1] (за замовчуванням fp[-1])
        period: період для періодичної інтерполяції
        
    returns:
        numpy.ndarray: інтерпольовані значення
        
    example:
        >>> xp = array([1, 2, 3])
        >>> fp = array([3, 2, 0])
        >>> x = array([0, 1, 1.5, 2.5, 4])
        >>> result = array_interp(x, xp, fp)
        >>> print(result)
        [3.  3.  2.5 1.  0. ]
    """
    return np.interp(x, xp, fp, left=left, right=right, period=period)


def array_trapz(y, x=None, dx=1.0, axis=-1):
    """
    обчислити інтеграл методом трапецій.
    
    args:
        y: значення функції
        x: точки (за замовчуванням рівномірна сітка з кроком dx)
        dx: крок сітки (використовується якщо x не задано)
        axis: вісь для інтегрування (за замовчуванням -1)
        
    returns:
        float or numpy.ndarray: значення інтеграла
        
    example:
        >>> y = array([1, 2, 3, 4])
        >>> result = array_trapz(y)
        >>> print(result)
        7.5
        
        >>> x = array([0, 1, 2, 3])
        >>> y = array([1, 2, 3, 4])
        >>> result = array_trapz(y, x)
        >>> print(result)
        7.5
    """
    return np.trapz(y, x=x, dx=dx, axis=axis)


def array_gradient(f, *varargs, axis=None, edge_order=1):
    """
    обчислити градієнт масиву.
    
    args:
        f: вхідний масив
        *varargs: кроки сітки для кожної осі
        axis: осі для обчислення градієнта (за замовчуванням всі)
        edge_order: порядок апроксимації на краях (1 або 2)
        
    returns:
        list of numpy.ndarray: градієнти для кожної осі
        або numpy.ndarray якщо axis задано
        
    example:
        >>> arr = array([[1, 2, 6], [3, 4, 5]])
        >>> result = array_gradient(arr)
        >>> print([r.tolist() for r in result])
        [[array([2., 2., -1.]), array([2., 2., -1.])], [array([1. , 2.5, -1. ]), array([1., 1., 1.])]]
    """
    return np.gradient(f, *varargs, axis=axis, edge_order=edge_order)


def array_diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    обчислити n-ну різницю масиву.
    
    args:
        a: вхідний масив
        n: порядок різниці (за замовчуванням 1)
        axis: вісь для обчислення (за замовчуванням -1)
        prepend: значення для додавання на початку
        append: значення для додавання в кінці
        
    returns:
        numpy.ndarray: масив різниць
        
    example:
        >>> arr = array([1, 2, 4, 7, 0])
        >>> result = array_diff(arr)
        >>> print(result)
        [ 1  2  3 -7]
        
        >>> result = array_diff(arr, n=2)
        >>> print(result)
        [ 1  1 -10]
    """
    return np.diff(a, n=n, axis=axis, prepend=prepend, append=append)


def array_ediff1d(ary, to_end=None, to_begin=None):
    """
    обчислити різниці між сусідніми елементами масиву.
    
    args:
        ary: вхідний масив
        to_end: значення для додавання в кінці
        to_begin: значення для додавання на початку
        
    returns:
        numpy.ndarray: масив різниць
        
    example:
        >>> arr = array([1, 2, 4, 7, 0])
        >>> result = array_ediff1d(arr)
        >>> print(result)
        [ 1  2  3 -7]
    """
    return np.ediff1d(ary, to_end=to_end, to_begin=to_begin)


def array_cumsum(a, axis=None, dtype=None, out=None):
    """
    обчислити кумулятивну суму елементів масиву.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        
    returns:
        numpy.ndarray: кумулятивна сума
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_cumsum(arr)
        >>> print(result)
        [ 1  3  6 10]
    """
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)


def array_cumprod(a, axis=None, dtype=None, out=None):
    """
    обчислити кумулятивний добуток елементів масиву.
    
    args:
        a: вхідний масив
        axis: вісь для обчислення (за замовчуванням всі елементи)
        dtype: тип даних результату
        out: вихідний масив
        
    returns:
        numpy.ndarray: кумулятивний добуток
        
    example:
        >>> arr = array([1, 2, 3, 4])
        >>> result = array_cumprod(arr)
        >>> print(result)
        [ 1  2  6 24]
    """
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)


def array_searchsorted(a, v, side='left', sorter=None):
    """
    знайти індекси для вставки елементів у впорядкований масив.
    
    args:
        a: впорядкований масив
        v: значення для вставки
        side: сторона для вставки ('left' або 'right')
        sorter: індекси для сортування масиву a
        
    returns:
        numpy.ndarray: індекси для вставки
        
    example:
        >>> arr = array([1, 2, 3, 4, 5])
        >>> values = array([1.5, 3.5, 5.5])
        >>> result = array_searchsorted(arr, values)
        >>> print(result)
        [1 3 5]
    """
    return np.searchsorted(a, v, side=side, sorter=sorter)


def array_digitize(x, bins, right=False):
    """
    визначити індекси інтервалів для елементів масиву.
    
    args:
        x: вхідний масив
        bins: межі інтервалів (повинні бути впорядковані)
        right: чи використовувати праві межі (за замовчуванням false)
        
    returns:
        numpy.ndarray: індекси інтервалів
        
    example:
        >>> arr = array([0.2, 6.4, 3.0, 1.6])
        >>> bins = array([0.0, 1.0, 2.5, 4.0, 10.0])
        >>> result = array_digitize(arr, bins)
        >>> print(result)
        [1 4 3 2]
    """
    return np.digitize(x, bins, right=right)


def array_histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    """
    обчислити двовимірну гістограму.
    
    args:
        x: значення по осі x
        y: значення по осі y
        bins: кількість інтервалів або їх межі
        range: діапазони значень (за замовчуванням визначаються автоматично)
        normed: чи нормувати (застарілий параметр)
        weights: ваги елементів
        density: чи повертати щільність (за замовчуванням none)
        
    returns:
        tuple: (гістограма, межі інтервалів по x, межі інтервалів по y)
        
    example:
        >>> x = array([1, 2, 3, 4, 5])
        >>> y = array([2, 3, 4, 5, 6])
        >>> hist, xedges, yedges = array_histogram2d(x, y, bins=3)
        >>> print(hist)
        [[1. 0. 0.]
         [1. 0. 0.]
         [1. 1. 1.]]
        >>> print(xedges)
        [1. 2.33333333 3.66666667 5. ]
        >>> print(yedges)
        [2. 3.33333333 4.66666667 6. ]
    """
    return np.histogram2d(x, y, bins=bins, range=range, normed=normed, weights=weights, density=density)