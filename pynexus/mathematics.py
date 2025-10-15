"""
Модуль для обчислювальної математики в PyNexus.
Містить функції для чисельного аналізу, теорії чисел, алгебри та інших розділів математики.
"""

import math
import cmath
import random
from typing import List, Tuple, Union, Callable, Optional
import numpy as np
from scipy import special, integrate, optimize
from scipy.linalg import solve, eig
import matplotlib.pyplot as plt

# Константи для обчислювальної математики
EULER_GAMMA = 0.5772156649015329  # Гамма-константа Ейлера
GOLDEN_RATIO = 1.618033988749895  # Золотий перетин
FEIGENBAUM_CONSTANT = 4.669201609102990  # Константа Фейгенбаума
PLANCK_CONSTANT = 6.62607015e-34  # Постійна Планка (Дж·с)
SPEED_OF_LIGHT = 299792458  # Швидкість світла у вакуумі (м/с)

def numerical_derivative(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Обчислити чисельну похідну функції в точці за формулою центральної різниці.
    
    Параметри:
        func: Функція, похідну якої потрібно обчислити
        x: Точка, в якій обчислюється похідна
        h: Крок дискретизації, за замовчуванням 1e-7
    
    Повертає:
        Значення похідної в точці x
    """
    if h <= 0:
        raise ValueError("Крок дискретизації повинен бути додатнім")
    
    return (func(x + h) - func(x - h)) / (2 * h)

def numerical_integral(func: Callable[[float], float], a: float, b: float, 
                      method: str = "simpson", n: int = 1000) -> float:
    """
    Обчислити чисельний інтеграл функції на інтервалі [a, b].
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        method: Метод інтегрування ("rectangle", "trapezoid", "simpson"), за замовчуванням "simpson"
        n: Кількість інтервалів, за замовчуванням 1000
    
    Повертає:
        Значення інтегралу
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if n <= 0:
        raise ValueError("Кількість інтервалів повинна бути додатньою")
    
    dx = (b - a) / n
    
    if method == "rectangle":
        # Метод прямокутників
        integral = sum(func(a + (i + 0.5) * dx) for i in range(n)) * dx
    elif method == "trapezoid":
        # Метод трапецій
        integral = (func(a) + func(b)) / 2
        integral += sum(func(a + i * dx) for i in range(1, n))
        integral *= dx
    elif method == "simpson":
        # Метод Сімпсона (парабол)
        if n % 2 != 0:
            n += 1  # n повинно бути парним для методу Сімпсона
            dx = (b - a) / n
        
        integral = func(a) + func(b)
        for i in range(1, n):
            if i % 2 == 0:
                integral += 2 * func(a + i * dx)
            else:
                integral += 4 * func(a + i * dx)
        integral *= dx / 3
    else:
        raise ValueError("Невідомий метод інтегрування")
    
    return integral

def monte_carlo_integration(func: Callable[[float], float], a: float, b: float, 
                           n_samples: int = 1000000) -> float:
    """
    Обчислити інтеграл методом Монте-Карло.
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        n_samples: Кількість випадкових точок, за замовчуванням 1000000
    
    Повертає:
        Значення інтегралу
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if n_samples <= 0:
        raise ValueError("Кількість точок повинна бути додатньою")
    
    # Генеруємо випадкові точки
    x_points = [random.uniform(a, b) for _ in range(n_samples)]
    
    # Обчислюємо значення функції в цих точках
    y_values = [func(x) for x in x_points]
    
    # Обчислюємо інтеграл
    integral = (b - a) * sum(y_values) / n_samples
    
    return integral

def newton_raphson(func: Callable[[float], float], 
                  derivative: Callable[[float], float], 
                  x0: float, 
                  tol: float = 1e-10, 
                  max_iter: int = 100) -> Tuple[float, int]:
    """
    Знайти корінь функції методом Ньютона-Рафсона.
    
    Параметри:
        func: Функція, корінь якої потрібно знайти
        derivative: Похідна функції
        x0: Початкове наближення
        tol: Точність, за замовчуванням 1e-10
        max_iter: Максимальна кількість ітерацій, за замовчуванням 100
    
    Повертає:
        Кортеж (корінь, кількість ітерацій)
    """
    if tol <= 0:
        raise ValueError("Точність повинна бути додатньою")
    if max_iter <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    x = x0
    for i in range(max_iter):
        fx = func(x)
        if abs(fx) < tol:
            return x, i + 1
        
        dfx = derivative(x)
        if abs(dfx) < tol:
            raise ValueError("Похідна близька до нуля, метод не збігається")
        
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, i + 1
        
        x = x_new
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def bisection_method(func: Callable[[float], float], 
                    a: float, 
                    b: float, 
                    tol: float = 1e-10, 
                    max_iter: int = 100) -> Tuple[float, int]:
    """
    Знайти корінь функції методом бісекції.
    
    Параметри:
        func: Функція, корінь якої потрібно знайти
        a: Початок інтервалу
        b: Кінець інтервалу
        tol: Точність, за замовчуванням 1e-10
        max_iter: Максимальна кількість ітерацій, за замовчуванням 100
    
    Повертає:
        Кортеж (корінь, кількість ітерацій)
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if tol <= 0:
        raise ValueError("Точність повинна бути додатньою")
    if max_iter <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    fa = func(a)
    fb = func(b)
    
    if fa * fb >= 0:
        raise ValueError("Функція повинна мати різні знаки на кінцях інтервалу")
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = func(c)
        
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, i + 1
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def fibonacci_sequence(n: int) -> List[int]:
    """
    Згенерувати послідовність Фібоначчі.
    
    Параметри:
        n: Кількість чисел у послідовності
    
    Повертає:
        Список з n перших чисел Фібоначчі
    """
    if n <= 0:
        raise ValueError("Кількість чисел повинна бути додатньою")
    
    if n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

def prime_sieve(limit: int) -> List[int]:
    """
    Знайти всі прості числа до заданого ліміту за допомогою решета Ератосфена.
    
    Параметри:
        limit: Верхня межа пошуку простих чисел
    
    Повертає:
        Список простих чисел
    """
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def gcd(a: int, b: int) -> int:
    """
    Обчислити найбільший спільний дільник двох чисел за алгоритмом Евкліда.
    
    Параметри:
        a: Перше число
        b: Друге число
    
    Повертає:
        Найбільший спільний дільник
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """
    Обчислити найменше спільне кратне двох чисел.
    
    Параметри:
        a: Перше число
        b: Друге число
    
    Повертає:
        Найменше спільне кратне
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def matrix_determinant(matrix: List[List[float]]) -> float:
    """
    Обчислити визначник матриці методом розкладання за рядком.
    
    Параметри:
        matrix: Квадратна матриця у вигляді списку списків
    
    Повертає:
        Визначник матриці
    """
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(n):
        # Створюємо мінор
        minor = []
        for i in range(1, n):
            row = []
            for j in range(n):
                if j != col:
                    row.append(matrix[i][j])
            minor.append(row)
        
        # Рекурсивно обчислюємо визначник мінора
        sign = (-1) ** col
        det += sign * matrix[0][col] * matrix_determinant(minor)
    
    return det

def matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
    """
    Обчислити обернену матрицю методом приєднаної матриці.
    
    Параметри:
        matrix: Квадратна матриця у вигляді списку списків
    
    Повертає:
        Обернена матриця
    """
    n = len(matrix)
    det = matrix_determinant(matrix)
    
    if abs(det) < 1e-10:
        raise ValueError("Матриця сингулярна, обернена матриця не існує")
    
    # Створюємо приєднану матрицю
    adjoint = []
    for i in range(n):
        row = []
        for j in range(n):
            # Створюємо мінор
            minor = []
            for k in range(n):
                if k != i:
                    minor_row = []
                    for l in range(n):
                        if l != j:
                            minor_row.append(matrix[k][l])
                    minor.append(minor_row)
            
            # Обчислюємо алгебраїчне доповнення
            cofactor = ((-1) ** (i + j)) * matrix_determinant(minor)
            row.append(cofactor)
        adjoint.append(row)
    
    # Транспонуємо приєднану матрицю
    adjoint_transpose = [[adjoint[j][i] for j in range(n)] for i in range(n)]
    
    # Ділимо на визначник
    inverse = [[adjoint_transpose[i][j] / det for j in range(n)] for i in range(n)]
    
    return inverse

def eigenvalues(matrix: List[List[float]]) -> List[complex]:
    """
    Обчислити власні значення матриці.
    
    Параметри:
        matrix: Квадратна матриця у вигляді списку списків
    
    Повертає:
        Список власних значень (можуть бути комплексними)
    """
    try:
        # Використовуємо NumPy для обчислення власних значень
        np_matrix = np.array(matrix)
        eigenvals = np.linalg.eigvals(np_matrix)
        return eigenvals.tolist()
    except Exception as e:
        raise ValueError(f"Не вдалося обчислити власні значення: {str(e)}")

def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Розв'язати систему лінійних рівнянь Ax = b.
    
    Параметри:
        A: Матриця коефіцієнтів
        b: Вектор правої частини
    
    Повертає:
        Вектор розв'язку
    """
    try:
        # Використовуємо NumPy для розв'язання системи
        np_A = np.array(A)
        np_b = np.array(b)
        solution = np.linalg.solve(np_A, np_b)
        return solution.tolist()
    except Exception as e:
        raise ValueError(f"Не вдалося розв'язати систему: {str(e)}")

def fourier_transform(signal: List[float]) -> List[complex]:
    """
    Обчислити дискретне перетворення Фур'є сигналу.
    
    Параметри:
        signal: Вхідний сигнал у вигляді списку значень
    
    Повертає:
        Спектр сигналу у вигляді комплексних чисел
    """
    N = len(signal)
    if N == 0:
        return []
    
    spectrum = []
    for k in range(N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            real_part += signal[n] * math.cos(angle)
            imag_part += signal[n] * math.sin(angle)
        spectrum.append(complex(real_part, imag_part))
    
    return spectrum

def inverse_fourier_transform(spectrum: List[complex]) -> List[float]:
    """
    Обчислити обернене дискретне перетворення Фур'є.
    
    Параметри:
        spectrum: Спектр сигналу у вигляді комплексних чисел
    
    Повертає:
        Відновлений сигнал у вигляді дійсних чисел
    """
    N = len(spectrum)
    if N == 0:
        return []
    
    signal = []
    for n in range(N):
        real_part = 0
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            real_part += spectrum[k].real * math.cos(angle) - spectrum[k].imag * math.sin(angle)
        signal.append(real_part / N)
    
    return signal

def interpolate_lagrange(x_points: List[float], 
                        y_points: List[float], 
                        x: float) -> float:
    """
    Інтерполювати значення функції за допомогою полінома Лагранжа.
    
    Параметри:
        x_points: Точки x
        y_points: Точки y
        x: Точка, в якій потрібно інтерполювати
    
    Повертає:
        Інтерпольоване значення
    """
    if len(x_points) != len(y_points):
        raise ValueError("Кількість точок x та y повинна бути однаковою")
    if len(x_points) == 0:
        raise ValueError("Потрібно хоча б одну точку для інтерполяції")
    
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                if x_points[i] == x_points[j]:
                    raise ValueError("Точки x не повинні повторюватися")
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    
    return result

def numerical_gradient(func: Callable[[List[float]], float], 
                      point: List[float], 
                      h: float = 1e-7) -> List[float]:
    """
    Обчислити чисельний градієнт функції багатьох змінних.
    
    Параметри:
        func: Функція багатьох змінних
        point: Точка, в якій обчислюється градієнт
        h: Крок дискретизації, за замовчуванням 1e-7
    
    Повертає:
        Градієнт у вигляді списку частинних похідних
    """
    if h <= 0:
        raise ValueError("Крок дискретизації повинен бути додатнім")
    
    n = len(point)
    gradient = []
    
    for i in range(n):
        # Створюємо точки зі зміненими координатами
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        
        # Обчислюємо частинну похідну
        partial_derivative = (func(point_plus) - func(point_minus)) / (2 * h)
        gradient.append(partial_derivative)
    
    return gradient

def runge_kutta_4(func: Callable[[float, float], float], 
                 x0: float, 
                 y0: float, 
                 x_end: float, 
                 n_steps: int) -> Tuple[List[float], List[float]]:
    """
    Розв'язати звичайне диференціальне рівняння першого порядку методом Рунге-Кутта 4-го порядку.
    
    Параметри:
        func: Функція f(x, y) у рівнянні dy/dx = f(x, y)
        x0: Початкове значення x
        y0: Початкове значення y
        x_end: Кінцеве значення x
        n_steps: Кількість кроків
    
    Повертає:
        Кортеж (список x, список y) з розв'язком
    """
    if n_steps <= 0:
        raise ValueError("Кількість кроків повинна бути додатньою")
    
    h = (x_end - x0) / n_steps
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    for _ in range(n_steps):
        k1 = h * func(x, y)
        k2 = h * func(x + h/2, y + k1/2)
        k3 = h * func(x + h/2, y + k2/2)
        k4 = h * func(x + h, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        x_values.append(x)
        y_values.append(y)
    
    return x_values, y_values

def finite_difference_second_derivative(func: Callable[[float], float], 
                                      x: float, 
                                      h: float = 1e-5) -> float:
    """
    Обчислити другу похідну функції за допомогою скінченної різниці.
    
    Параметри:
        func: Функція, другу похідну якої потрібно обчислити
        x: Точка, в якій обчислюється друга похідна
        h: Крок дискретизації, за замовчуванням 1e-5
    
    Повертає:
        Значення другої похідної в точці x
    """
    if h <= 0:
        raise ValueError("Крок дискретизації повинен бути додатнім")
    
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h * h)

def romberg_integration(func: Callable[[float], float], 
                       a: float, 
                       b: float, 
                       max_iterations: int = 10) -> float:
    """
    Обчислити інтеграл методом Ромберга.
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        max_iterations: Максимальна кількість ітерацій, за замовчуванням 10
    
    Повертає:
        Значення інтегралу
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if max_iterations <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    # Створюємо таблицю Ромберга
    R = [[0.0 for _ in range(max_iterations)] for _ in range(max_iterations)]
    
    # Перший стовпець - метод трапецій з різною кількістю інтервалів
    for i in range(max_iterations):
        n = 2 ** i
        h = (b - a) / n
        
        # Метод трапецій
        if i == 0:
            R[i][0] = (func(a) + func(b)) * h / 2
        else:
            sum_terms = sum(func(a + (j + 0.5) * h) for j in range(n))
            R[i][0] = (R[i-1][0] + h * sum_terms) / 2
    
    # Екстраполяція Річардсона
    for j in range(1, max_iterations):
        for i in range(j, max_iterations):
            R[i][j] = (4**j * R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
    
    return R[max_iterations-1][max_iterations-1]

def adaptive_integration(func: Callable[[float], float], 
                        a: float, 
                        b: float, 
                        tol: float = 1e-10) -> float:
    """
    Обчислити інтеграл адаптивним методом.
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        tol: Точність, за замовчуванням 1e-10
    
    Повертає:
        Значення інтегралу
    """
    def integrate_recursive(a, b, fa, fb, fc, S):
        c = (a + b) / 2
        h = b - a
        d = (a + c) / 2
        e = (c + b) / 2
        fd = func(d)
        fe = func(e)
        Sleft = h * (fa + 4 * fd + fc) / 12
        Sright = h * (fc + 4 * fe + fb) / 12
        S2 = Sleft + Sright
        
        if abs(S2 - S) <= 15 * tol:
            return S2 + (S2 - S) / 15
        else:
            return (integrate_recursive(a, c, fa, fc, fd, Sleft) + 
                   integrate_recursive(c, b, fc, fb, fe, Sright))
    
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if tol <= 0:
        raise ValueError("Точність повинна бути додатньою")
    
    c = (a + b) / 2
    fa = func(a)
    fb = func(b)
    fc = func(c)
    S = (b - a) * (fa + 4 * fc + fb) / 6
    
    return integrate_recursive(a, b, fa, fb, fc, S)

def legendre_polynomial(n: int, x: float) -> float:
    """
    Обчислити поліном Лежандра порядку n в точці x.
    
    Параметри:
        n: Порядок полінома
        x: Точка, в якій обчислюється поліном
    
    Повертає:
        Значення полінома Лежандра
    """
    if n < 0:
        raise ValueError("Порядок полінома повинен бути невід'ємним")
    
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        # Рекурентне співвідношення Бонне
        P0 = 1.0
        P1 = x
        for i in range(2, n + 1):
            P2 = ((2 * i - 1) * x * P1 - (i - 1) * P0) / i
            P0, P1 = P1, P2
        return P1

def hermite_polynomial(n: int, x: float) -> float:
    """
    Обчислити поліном Ерміта порядку n в точці x.
    
    Параметри:
        n: Порядок полінома
        x: Точка, в якій обчислюється поліном
    
    Повертає:
        Значення полінома Ерміта
    """
    if n < 0:
        raise ValueError("Порядок полінома повинен бути невід'ємним")
    
    if n == 0:
        return 1.0
    elif n == 1:
        return 2 * x
    else:
        # Рекурентне співвідношення
        H0 = 1.0
        H1 = 2 * x
        for i in range(2, n + 1):
            H2 = 2 * x * H1 - 2 * (i - 1) * H0
            H0, H1 = H1, H2
        return H1

def chebyshev_polynomial(n: int, x: float) -> float:
    """
    Обчислити поліном Чебишова першого роду порядку n в точці x.
    
    Параметри:
        n: Порядок полінома
        x: Точка, в якій обчислюється поліном
    
    Повертає:
        Значення полінома Чебишова
    """
    if n < 0:
        raise ValueError("Порядок полінома повинен бути невід'ємним")
    
    if abs(x) > 1:
        raise ValueError("Аргумент повинен бути в діапазоні [-1, 1]")
    
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        # Рекурентне співвідношення
        T0 = 1.0
        T1 = x
        for i in range(2, n + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T1

def bessel_function(n: int, x: float) -> float:
    """
    Обчислити функцію Бесселя першого роду порядку n в точці x.
    
    Параметри:
        n: Порядок функції
        x: Точка, в якій обчислюється функція
    
    Повертає:
        Значення функції Бесселя
    """
    if n < 0:
        raise ValueError("Порядок функції повинен бути невід'ємним")
    
    if x == 0:
        return 1.0 if n == 0 else 0.0
    
    # Ряд для функції Бесселя
    result = 0.0
    factorial = 1.0
    power_x = (x / 2) ** n
    
    for k in range(50):  # Достатньо для збіжності
        if k > 0:
            factorial *= k
        gamma = math.factorial(n + k) if n + k <= 170 else float('inf')  # Обмеження для великих значень
        if gamma == float('inf'):
            break
        term = ((-1) ** k) * power_x / (factorial * gamma)
        result += term
        power_x *= (x / 2) ** 2
    
    return result

def gamma_function(x: float) -> float:
    """
    Обчислити гамма-функцію в точці x.
    
    Параметри:
        x: Точка, в якій обчислюється гамма-функція
    
    Повертає:
        Значення гамма-функції
    """
    if x <= 0 and x == int(x):
        raise ValueError("Гамма-функція не визначена для невід'ємних цілих чисел")
    
    # Використовуємо апроксимацію Ланцоша
    g = 7
    coefficients = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    
    if x < 0.5:
        # Використовуємо рефлексійну формулу
        return math.pi / (math.sin(math.pi * x) * gamma_function(1 - x))
    else:
        x -= 1
        tmp = coefficients[0]
        for i in range(1, g + 2):
            tmp += coefficients[i] / (x + i)
        t = x + g + 0.5
        return math.sqrt(2 * math.pi) * (t ** (x + 0.5)) * math.exp(-t) * tmp

def riemann_zeta(s: float, terms: int = 1000) -> float:
    """
    Обчислити дзета-функцію Рімана в точці s.
    
    Параметри:
        s: Точка, в якій обчислюється дзета-функція
        terms: Кількість членів ряду, за замовчуванням 1000
    
    Повертає:
        Значення дзета-функції
    """
    if s <= 1:
        raise ValueError("Дзета-функція Рімана збігається лише для s > 1")
    if terms <= 0:
        raise ValueError("Кількість членів повинна бути додатньою")
    
    # Прямий обчислення ряду Діріхле
    result = sum(1 / (n ** s) for n in range(1, terms + 1))
    return result

def elliptic_integral_first_kind(phi: float, k: float) -> float:
    """
    Обчислити еліптичний інтеграл першого роду.
    
    Параметри:
        phi: Амплітуда (радіани)
        k: Модуль (0 ≤ k < 1)
    
    Повертає:
        Значення еліптичного інтеграла
    """
    if k < 0 or k >= 1:
        raise ValueError("Модуль повинен бути в діапазоні [0, 1)")
    if phi < 0:
        raise ValueError("Амплітуда повинна бути невід'ємною")
    
    # Чисельне інтегрування
    def integrand(theta):
        return 1 / math.sqrt(1 - k**2 * math.sin(theta)**2)
    
    return numerical_integral(integrand, 0, phi, method="simpson", n=1000)

def elliptic_integral_second_kind(phi: float, k: float) -> float:
    """
    Обчислити еліптичний інтеграл другого роду.
    
    Параметри:
        phi: Амплітуда (радіани)
        k: Модуль (0 ≤ k < 1)
    
    Повертає:
        Значення еліптичного інтеграла
    """
    if k < 0 or k >= 1:
        raise ValueError("Модуль повинен бути в діапазоні [0, 1)")
    if phi < 0:
        raise ValueError("Амплітуда повинна бути невід'ємною")
    
    # Чисельне інтегрування
    def integrand(theta):
        return math.sqrt(1 - k**2 * math.sin(theta)**2)
    
    return numerical_integral(integrand, 0, phi, method="simpson", n=1000)

def beta_function(x: float, y: float) -> float:
    """
    Обчислити бета-функцію.
    
    Параметри:
        x: Перший параметр
        y: Другий параметр
    
    Повертає:
        Значення бета-функції
    """
    if x <= 0 or y <= 0:
        raise ValueError("Параметри повинні бути додатніми")
    
    # Бета-функція виражається через гамма-функції
    return (gamma_function(x) * gamma_function(y)) / gamma_function(x + y)

def error_function(x: float) -> float:
    """
    Обчислити функцію помилок (інтеграл від гауссіана).
    
    Параметри:
        x: Точка, в якій обчислюється функція
    
    Повертає:
        Значення функції помилок
    """
    # Використовуємо апроксимацію
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return sign * y

def complementary_error_function(x: float) -> float:
    """
    Обчислити доповнюючу функцію помилок.
    
    Параметри:
        x: Точка, в якій обчислюється функція
    
    Повертає:
        Значення доповнюючої функції помилок
    """
    return 1.0 - error_function(x)

def logistic_map(r: float, x0: float, n_iterations: int) -> List[float]:
    """
    Згенерувати послідовність логістичного відображення.
    
    Параметри:
        r: Параметр управління
        x0: Початкове значення
        n_iterations: Кількість ітерацій
    
    Повертає:
        Список значень послідовності
    """
    if n_iterations <= 0:
        raise ValueError("Кількість ітерацій повинна бути додатньою")
    if r < 0:
        raise ValueError("Параметр управління повинен бути невід'ємним")
    if x0 < 0 or x0 > 1:
        raise ValueError("Початкове значення повинно бути в діапазоні [0, 1]")
    
    sequence = [x0]
    x = x0
    
    for _ in range(n_iterations - 1):
        x = r * x * (1 - x)
        sequence.append(x)
    
    return sequence

def mandelbrot_set(width: int, height: int, 
                  x_min: float = -2.0, x_max: float = 1.0,
                  y_min: float = -1.5, y_max: float = 1.5,
                  max_iterations: int = 100) -> List[List[int]]:
    """
    Згенерувати множину Мандельброта.
    
    Параметри:
        width: Ширина зображення
        height: Висота зображення
        x_min: Мінімальне значення x
        x_max: Максимальне значення x
        y_min: Мінімальне значення y
        y_max: Максимальне значення y
        max_iterations: Максимальна кількість ітерацій
    
    Повертає:
        Матриця значень ітерацій для кожної точки
    """
    if width <= 0 or height <= 0:
        raise ValueError("Розміри зображення повинні бути додатніми")
    if max_iterations <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    result = []
    
    for y in range(height):
        row = []
        for x in range(width):
            # Перетворюємо координати пікселя в комплексне число
            c_real = x_min + (x / (width - 1)) * (x_max - x_min)
            c_imag = y_min + (y / (height - 1)) * (y_max - y_min)
            c = complex(c_real, c_imag)
            
            # Ітеруємо z = z^2 + c
            z = complex(0, 0)
            for i in range(max_iterations):
                z = z * z + c
                if abs(z) > 2:
                    row.append(i)
                    break
            else:
                row.append(max_iterations)
        
        result.append(row)
    
    return result

def julia_set(c_real: float, c_imag: float,
             width: int, height: int,
             x_min: float = -2.0, x_max: float = 2.0,
             y_min: float = -2.0, y_max: float = 2.0,
             max_iterations: int = 100) -> List[List[int]]:
    """
    Згенерувати множину Жюліа.
    
    Параметри:
        c_real: Дійсна частина константи c
        c_imag: Уявна частина константи c
        width: Ширина зображення
        height: Висота зображення
        x_min: Мінімальне значення x
        x_max: Максимальне значення x
        y_min: Мінімальне значення y
        y_max: Максимальне значення y
        max_iterations: Максимальна кількість ітерацій
    
    Повертає:
        Матриця значень ітерацій для кожної точки
    """
    if width <= 0 or height <= 0:
        raise ValueError("Розміри зображення повинні бути додатніми")
    if max_iterations <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    c = complex(c_real, c_imag)
    result = []
    
    for y in range(height):
        row = []
        for x in range(width):
            # Перетворюємо координати пікселя в комплексне число
            z_real = x_min + (x / (width - 1)) * (x_max - x_min)
            z_imag = y_min + (y / (height - 1)) * (y_max - y_min)
            z = complex(z_real, z_imag)
            
            # Ітеруємо z = z^2 + c
            for i in range(max_iterations):
                z = z * z + c
                if abs(z) > 2:
                    row.append(i)
                    break
            else:
                row.append(max_iterations)
        
        result.append(row)
    
    return result

def fft(signal: List[complex]) -> List[complex]:
    """
    Обчислити швидке перетворення Фур'є за допомогою алгоритму Куля-Тьюкі.
    
    Параметри:
        signal: Вхідний сигнал (комплексні числа)
    
    Повертає:
        Спектр сигналу
    """
    N = len(signal)
    if N <= 1:
        return signal
    
    if N & (N - 1) != 0:
        # Доповнюємо до степеня 2
        next_power_of_2 = 1
        while next_power_of_2 < N:
            next_power_of_2 <<= 1
        signal += [0] * (next_power_of_2 - N)
        N = next_power_of_2
    
    # Рекурсивна реалізація FFT
    if N == 1:
        return signal
    
    # Парні та непарні елементи
    even = fft(signal[0::2])
    odd = fft(signal[1::2])
    
    # Комбінуємо результати
    result = [0] * N
    for k in range(N // 2):
        t = cmath.exp(-2j * cmath.pi * k / N) * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    
    return result

def ifft(spectrum: List[complex]) -> List[complex]:
    """
    Обчислити обернене швидке перетворення Фур'є.
    
    Параметри:
        spectrum: Спектр сигналу
    
    Повертає:
        Відновлений сигнал
    """
    N = len(spectrum)
    if N <= 1:
        return spectrum
    
    # Комплексне спряження
    conjugate_spectrum = [x.conjugate() for x in spectrum]
    
    # Застосовуємо FFT до спряженого спектра
    conjugate_signal = fft(conjugate_spectrum)
    
    # Комплексне спряження результату та нормалізація
    signal = [x.conjugate() / N for x in conjugate_signal]
    
    return signal

def convolution(signal1: List[float], signal2: List[float]) -> List[float]:
    """
    Обчислити згортку двох сигналів.
    
    Параметри:
        signal1: Перший сигнал
        signal2: Другий сигнал
    
    Повертає:
        Результат згортки
    """
    N1 = len(signal1)
    N2 = len(signal2)
    
    if N1 == 0 or N2 == 0:
        return []
    
    # Довжина результату
    result_length = N1 + N2 - 1
    result = [0.0] * result_length
    
    # Обчислюємо згортку
    for n in range(result_length):
        for k in range(max(0, n - N2 + 1), min(N1, n + 1)):
            result[n] += signal1[k] * signal2[n - k]
    
    return result

def correlation(signal1: List[float], signal2: List[float]) -> List[float]:
    """
    Обчислити кореляцію двох сигналів.
    
    Параметри:
        signal1: Перший сигнал
        signal2: Другий сигнал
    
    Повертає:
        Результат кореляції
    """
    # Кореляція - це згортка одного сигналу з оберненим іншим
    signal2_reversed = signal2[::-1]
    return convolution(signal1, signal2_reversed)

def solve_differential_equation_2nd_order(a: float, b: float, c: float,
                                        y0: float, y1: float, x0: float, x1: float,
                                        n_points: int) -> Tuple[List[float], List[float]]:
    """
    Розв'язати лінійне диференціальне рівняння другого порядку:
    a*y'' + b*y' + c*y = 0
    
    Параметри:
        a, b, c: Коефіцієнти рівняння
        y0, y1: Початкові умови (y(x0), y'(x0))
        x0, x1: Початкове та кінцеве значення x
        n_points: Кількість точок для обчислення
    
    Повертає:
        Кортеж (список x, список y) з розв'язком
    """
    if a == 0:
        raise ValueError("Коефіцієнт a не може бути нульовим")
    if n_points <= 1:
        raise ValueError("Кількість точок повинна бути більшою за 1")
    
    # Перетворюємо в систему першого порядку
    # y' = v
    # v' = (-b*v - c*y) / a
    
    def system(x, yv):
        y, v = yv
        dydx = v
        dvdx = (-b * v - c * y) / a
        return [dydx, dvdx]
    
    # Використовуємо метод Рунге-Кутта
    h = (x1 - x0) / (n_points - 1)
    x_values = [x0 + i * h for i in range(n_points)]
    y_values = [y0]
    v_values = [y1]  # y'(x0) = y1
    
    y = y0
    v = y1
    
    for i in range(1, n_points):
        # Метод Рунге-Кутта 4-го порядку
        k1_y, k1_v = system(x_values[i-1], [y, v])
        k2_y, k2_v = system(x_values[i-1] + h/2, [y + h*k1_y/2, v + h*k1_v/2])
        k3_y, k3_v = system(x_values[i-1] + h/2, [y + h*k2_y/2, v + h*k2_v/2])
        k4_y, k4_v = system(x_values[i-1] + h, [y + h*k3_y, v + h*k3_v])
        
        y += h * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        v += h * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        
        y_values.append(y)
    
    return x_values, y_values

def newton_interpolation(x_points: List[float], 
                        y_points: List[float], 
                        x: float) -> float:
    """
    Інтерполювати значення функції за допомогою інтерполяційної формули Ньютона.
    
    Параметри:
        x_points: Точки x
        y_points: Точки y
        x: Точка, в якій потрібно інтерполювати
    
    Повертає:
        Інтерпольоване значення
    """
    if len(x_points) != len(y_points):
        raise ValueError("Кількість точок x та y повинна бути однаковою")
    if len(x_points) == 0:
        raise ValueError("Потрібно хоча б одну точку для інтерполяції")
    
    n = len(x_points)
    
    # Створюємо таблицю розділених різниць
    divided_diff = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Заповнюємо перший стовпець
    for i in range(n):
        divided_diff[i][0] = y_points[i]
    
    # Обчислюємо розділені різниці
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i+1][j-1] - divided_diff[i][j-1]) / (x_points[i+j] - x_points[i])
    
    # Обчислюємо інтерпольоване значення
    result = divided_diff[0][0]
    product = 1.0
    
    for i in range(1, n):
        product *= (x - x_points[i-1])
        result += divided_diff[0][i] * product
    
    return result

def cubic_spline_interpolation(x_points: List[float], 
                              y_points: List[float], 
                              x: float) -> float:
    """
    Інтерполювати значення функції за допомогою кубічного сплайну.
    
    Параметри:
        x_points: Точки x (повинні бути впорядковані)
        y_points: Точки y
        x: Точка, в якій потрібно інтерполювати
    
    Повертає:
        Інтерпольоване значення
    """
    if len(x_points) != len(y_points):
        raise ValueError("Кількість точок x та y повинна бути однаковою")
    if len(x_points) < 2:
        raise ValueError("Потрібно хоча б дві точки для сплайн-інтерполяції")
    
    n = len(x_points)
    
    # Перевіряємо впорядкування точок x
    for i in range(1, n):
        if x_points[i] <= x_points[i-1]:
            raise ValueError("Точки x повинні бути строго зростаючими")
    
    # Знаходимо інтервал, до якого належить x
    if x < x_points[0] or x > x_points[-1]:
        raise ValueError("Точка інтерполяції повинна бути в межах заданих точок")
    
    # Знаходимо індекс інтервалу
    interval = 0
    for i in range(n - 1):
        if x_points[i] <= x <= x_points[i+1]:
            interval = i
            break
    
    # Обчислюємо коефіцієнти кубічного сплайну
    # Розв'язуємо систему для других похідних
    h = [x_points[i+1] - x_points[i] for i in range(n-1)]
    
    # Матриця для системи лінійних рівнянь
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    b = [0.0] * n
    
    # Внутрішні точки
    for i in range(1, n-1):
        A[i][i-1] = h[i-1]
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i+1] = h[i]
        b[i] = 6 * ((y_points[i+1] - y_points[i]) / h[i] - (y_points[i] - y_points[i-1]) / h[i-1])
    
    # Граничні умови (натуральний сплайн)
    A[0][0] = 2
    A[n-1][n-1] = 2
    
    # Розв'язуємо систему
    M = solve_linear_system(A, b)  # Другі похідні
    
    # Обчислюємо значення сплайну
    i = interval
    xi = x_points[i]
    xi1 = x_points[i+1]
    yi = y_points[i]
    yi1 = y_points[i+1]
    Mi = M[i]
    Mi1 = M[i+1]
    
    # Коефіцієнти кубічного полінома
    h_i = xi1 - xi
    a = (xi1 - x) / h_i
    b_coeff = (x - xi) / h_i
    c = (1/6) * (a**3 - a) * h_i**2
    d = (1/6) * (b_coeff**3 - b_coeff) * h_i**2
    
    # Значення сплайну
    result = a * yi + b_coeff * yi1 + c * Mi + d * Mi1
    
    return result

def simpson_3_8_rule(func: Callable[[float], float], 
                    a: float, 
                    b: float, 
                    n: int) -> float:
    """
    Обчислити інтеграл за правилом Сімпсона 3/8.
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        n: Кількість інтервалів (повинно бути кратним 3)
    
    Повертає:
        Значення інтегралу
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if n <= 0:
        raise ValueError("Кількість інтервалів повинна бути додатньою")
    if n % 3 != 0:
        raise ValueError("Кількість інтервалів повинна бути кратною 3")
    
    h = (b - a) / n
    integral = func(a) + func(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 3 == 0:
            integral += 2 * func(x)
        else:
            integral += 3 * func(x)
    
    integral *= 3 * h / 8
    return integral

def boole_rule(func: Callable[[float], float], 
              a: float, 
              b: float, 
              n: int) -> float:
    """
    Обчислити інтеграл за правилом Буля (правило Кіддера).
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        n: Кількість інтервалів (повинно бути кратним 4)
    
    Повертає:
        Значення інтегралу
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if n <= 0:
        raise ValueError("Кількість інтервалів повинна бути додатньою")
    if n % 4 != 0:
        raise ValueError("Кількість інтервалів повинна бути кратною 4")
    
    h = (b - a) / n
    integral = 7 * (func(a) + func(b))
    
    for i in range(1, n):
        x = a + i * h
        if i % 4 == 0:
            integral += 14 * func(x)
        elif i % 2 == 0:
            integral += 12 * func(x)
        else:
            integral += 32 * func(x)
    
    integral *= 2 * h / 45
    return integral

def gaussian_quadrature(func: Callable[[float], float], 
                       a: float, 
                       b: float, 
                       n_points: int = 5) -> float:
    """
    Обчислити інтеграл методом гауссової квадратури.
    
    Параметри:
        func: Функція для інтегрування
        a: Початок інтервалу
        b: Кінець інтервалу
        n_points: Кількість точок квадратури, за замовчуванням 5
    
    Повертає:
        Значення інтегралу
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if n_points <= 0:
        raise ValueError("Кількість точок повинна бути додатньою")
    
    # Точки та ваги для гауссової квадратури Лежандра
    if n_points == 2:
        points = [-0.5773502692, 0.5773502692]
        weights = [1.0, 1.0]
    elif n_points == 3:
        points = [-0.7745966692, 0.0, 0.7745966692]
        weights = [0.5555555556, 0.8888888889, 0.5555555556]
    elif n_points == 4:
        points = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
        weights = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]
    elif n_points == 5:
        points = [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
        weights = [0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]
    else:
        raise ValueError("Підтримуються лише n_points = 2, 3, 4, 5")
    
    # Перетворюємо інтервал [-1, 1] в [a, b]
    integral = 0.0
    for i in range(n_points):
        x_transformed = ((b - a) * points[i] + (a + b)) / 2
        integral += weights[i] * func(x_transformed)
    
    integral *= (b - a) / 2
    return integral

def secant_method(func: Callable[[float], float], 
                 x0: float, 
                 x1: float, 
                 tol: float = 1e-10, 
                 max_iter: int = 100) -> Tuple[float, int]:
    """
    Знайти корінь функції методом секущих.
    
    Параметри:
        func: Функція, корінь якої потрібно знайти
        x0: Перше початкове наближення
        x1: Друге початкове наближення
        tol: Точність, за замовчуванням 1e-10
        max_iter: Максимальна кількість ітерацій, за замовчуванням 100
    
    Повертає:
        Кортеж (корінь, кількість ітерацій)
    """
    if tol <= 0:
        raise ValueError("Точність повинна бути додатньою")
    if max_iter <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    f0 = func(x0)
    f1 = func(x1)
    
    for i in range(max_iter):
        if abs(f1 - f0) < tol:
            raise ValueError("Функції в точках дуже близькі, метод не збігається")
        
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        if abs(x_new - x1) < tol:
            return x_new, i + 1
        
        x0, x1 = x1, x_new
        f0, f1 = f1, func(x1)
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def fixed_point_iteration(func: Callable[[float], float], 
                        x0: float, 
                        tol: float = 1e-10, 
                        max_iter: int = 100) -> Tuple[float, int]:
    """
    Знайти нерухому точку функції методом ітерацій.
    
    Параметри:
        func: Функція, нерухому точку якої потрібно знайти
        x0: Початкове наближення
        tol: Точність, за замовчуванням 1e-10
        max_iter: Максимальна кількість ітерацій, за замовчуванням 100
    
    Повертає:
        Кортеж (нерухома точка, кількість ітерацій)
    """
    if tol <= 0:
        raise ValueError("Точність повинна бути додатньою")
    if max_iter <= 0:
        raise ValueError("Максимальна кількість ітерацій повинна бути додатньою")
    
    x = x0
    
    for i in range(max_iter):
        x_new = func(x)
        
        if abs(x_new - x) < tol:
            return x_new, i + 1
        
        x = x_new
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def lu_decomposition(matrix: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Виконати LU-розклад матриці.
    
    Параметри:
        matrix: Квадратна матриця
    
    Повертає:
        Кортеж (L, U) - нижня та верхня трикутні матриці
    """
    n = len(matrix)
    if n == 0:
        return [[1]], [[0]]
    
    # Ініціалізуємо L та U
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # LU-розклад
    for i in range(n):
        # Верхня трикутна матриця U
        for k in range(i, n):
            sum_val = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - sum_val
        
        # Нижня трикутна матриця L
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                sum_val = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (matrix[k][i] - sum_val) / U[i][i]
    
    return L, U

def cholesky_decomposition(matrix: List[List[float]]) -> List[List[float]]:
    """
    Виконати розклад Холецького для симетричної додатно визначеної матриці.
    
    Параметри:
        matrix: Симетрична додатно визначена матриця
    
    Повертає:
        Нижня трикутна матриця L така, що A = L * L^T
    """
    n = len(matrix)
    if n == 0:
        return []
    
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1):
            sum_val = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j:
                L[i][j] = math.sqrt(matrix[i][i] - sum_val)
            else:
                L[i][j] = (matrix[i][j] - sum_val) / L[j][j]
    
    return L

def qr_decomposition(matrix: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Виконати QR-розклад матриці методом Грама-Шмідта.
    
    Параметри:
        matrix: Матриця розміром m x n
    
    Повертає:
        Кортеж (Q, R) - ортогональна та верхня трикутна матриці
    """
    m = len(matrix)
    if m == 0:
        return [], []
    n = len(matrix[0])
    if n == 0:
        return [[] for _ in range(m)], []
    
    # Транспонуємо матрицю для зручності роботи зі стовпцями
    A_transpose = [[matrix[i][j] for i in range(m)] for j in range(n)]
    
    # Ортогоналізуємо стовпці методом Грама-Шмідта
    Q_transpose = []
    
    for i in range(n):
        # Беремо i-й стовпець
        v = A_transpose[i][:]
        
        # Віднімаємо проекції на попередні ортогональні вектори
        for j in range(i):
            q = Q_transpose[j]
            proj = sum(v[k] * q[k] for k in range(m))  # <v, q>
            norm_q_sq = sum(q[k] * q[k] for k in range(m))  # <q, q>
            for k in range(m):
                v[k] -= proj * q[k] / norm_q_sq
        
        # Нормалізуємо вектор
        norm = math.sqrt(sum(v[k] * v[k] for k in range(m)))
        if norm > 1e-10:
            q = [v[k] / norm for k in range(m)]
            Q_transpose.append(q)
        else:
            # Лінійно залежний вектор
            break
    
    # Транспонуємо Q_transpose, щоб отримати Q
    Q = [[Q_transpose[j][i] for j in range(len(Q_transpose))] for i in range(m)]
    
    # Обчислюємо R = Q^T * A
    R = [[0.0 for _ in range(n)] for _ in range(len(Q_transpose))]
    for i in range(len(Q_transpose)):
        for j in range(n):
            R[i][j] = sum(Q_transpose[i][k] * matrix[k][j] for k in range(m))
    
    return Q, R

def power_iteration(matrix: List[List[float]], 
                   max_iter: int = 1000, 
                   tol: float = 1e-10) -> Tuple[float, List[float]]:
    """
    Знайти найбільше за модулем власне значення та відповідний власний вектор методом степеневої ітерації.
    
    Параметри:
        matrix: Квадратна матриця
        max_iter: Максимальна кількість ітерацій, за замовчуванням 1000
        tol: Точність, за замовчуванням 1e-10
    
    Повертає:
        Кортеж (власне значення, власний вектор)
    """
    n = len(matrix)
    if n == 0:
        return 0.0, []
    
    # Початковий вектор (одиничний)
    x = [1.0 / math.sqrt(n) for _ in range(n)]
    
    eigenvalue = 0.0
    
    for _ in range(max_iter):
        # Множимо матрицю на вектор
        y = [sum(matrix[i][j] * x[j] for j in range(n)) for i in range(n)]
        
        # Нормалізуємо вектор
        norm = math.sqrt(sum(y[i] * y[i] for i in range(n)))
        if norm < tol:
            raise ValueError("Вектор близький до нульового")
        
        y = [y[i] / norm for i in range(n)]
        
        # Обчислюємо наближене власне значення
        new_eigenvalue = sum(y[i] * sum(matrix[i][j] * y[j] for j in range(n)) for i in range(n))
        
        # Перевіряємо збіжність
        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, y
        
        eigenvalue = new_eigenvalue
        x = y
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def inverse_iteration(matrix: List[List[float]], 
                     eigenvalue_estimate: float,
                     max_iter: int = 1000, 
                     tol: float = 1e-10) -> Tuple[float, List[float]]:
    """
    Знайти власне значення та відповідний власний вектор методом оберненої ітерації.
    
    Параметри:
        matrix: Квадратна матриця
        eigenvalue_estimate: Оцінка власного значення
        max_iter: Максимальна кількість ітерацій, за замовчуванням 1000
        tol: Точність, за замовчуванням 1e-10
    
    Повертає:
        Кортеж (власне значення, власний вектор)
    """
    n = len(matrix)
    if n == 0:
        return 0.0, []
    
    # Створюємо матрицю (A - λI)
    shifted_matrix = [[matrix[i][j] - (eigenvalue_estimate if i == j else 0) for j in range(n)] for i in range(n)]
    
    # Початковий вектор (одиничний)
    x = [1.0 / math.sqrt(n) for _ in range(n)]
    
    eigenvalue = 0.0
    
    for iteration in range(max_iter):
        # Розв'язуємо систему (A - λI) * y = x
        try:
            y = solve_linear_system(shifted_matrix, x)
        except:
            raise ValueError("Не вдалося розв'язати систему на ітерації {}".format(iteration))
        
        # Нормалізуємо вектор
        norm = math.sqrt(sum(y[i] * y[i] for i in range(n)))
        if norm < tol:
            raise ValueError("Вектор близький до нульового")
        
        y = [y[i] / norm for i in range(n)]
        
        # Обчислюємо наближене власне значення
        rayleigh_quotient = sum(y[i] * sum(matrix[i][j] * y[j] for j in range(n)) for i in range(n))
        rayleigh_denominator = sum(y[i] * y[i] for i in range(n))
        new_eigenvalue = rayleigh_quotient / rayleigh_denominator
        
        # Перевіряємо збіжність
        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, y
        
        eigenvalue = new_eigenvalue
        x = y
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def conjugate_gradient(A: List[List[float]], 
                      b: List[float], 
                      x0: Optional[List[float]] = None,
                      max_iter: int = 1000, 
                      tol: float = 1e-10) -> List[float]:
    """
    Розв'язати систему лінійних рівнянь Ax = b методом спряжених градієнтів.
    
    Параметри:
        A: Симетрична додатно визначена матриця
        b: Вектор правої частини
        x0: Початкове наближення (за замовчуванням - нульовий вектор)
        max_iter: Максимальна кількість ітерацій, за замовчуванням 1000
        tol: Точність, за замовчуванням 1e-10
    
    Повертає:
        Вектор розв'язку
    """
    n = len(A)
    if n == 0:
        return []
    if len(b) != n:
        raise ValueError("Розмірність матриці та вектора не узгоджені")
    
    # Початкове наближення
    x = x0[:] if x0 is not None else [0.0 for _ in range(n)]
    
    # Обчислюємо початковий залишок
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    
    # Початковий напрямок пошуку
    p = r[:]
    
    rsold = sum(r[i] * r[i] for i in range(n))
    
    for iteration in range(max_iter):
        # Обчислюємо Ap
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
        
        # Обчислюємо крок
        alpha = rsold / sum(p[i] * Ap[i] for i in range(n))
        
        # Оновлюємо розв'язок
        for i in range(n):
            x[i] += alpha * p[i]
        
        # Оновлюємо залишок
        for i in range(n):
            r[i] -= alpha * Ap[i]
        
        # Перевіряємо збіжність
        rsnew = sum(r[i] * r[i] for i in range(n))
        if math.sqrt(rsnew) < tol:
            return x
        
        # Обчислюємо новий напрямок
        beta = rsnew / rsold
        for i in range(n):
            p[i] = r[i] + beta * p[i]
        
        rsold = rsnew
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def newton_method_system(funcs: List[Callable[[List[float]], float]], 
                        jacobian: List[List[Callable[[List[float]], float]]],
                        x0: List[float],
                        max_iter: int = 100, 
                        tol: float = 1e-10) -> List[float]:
    """
    Розв'язати систему нелінійних рівнянь методом Ньютона.
    
    Параметри:
        funcs: Список функцій системи
        jacobian: Якобіан (матриця частинних похідних)
        x0: Початкове наближення
        max_iter: Максимальна кількість ітерацій, за замовчуванням 100
        tol: Точність, за замовчуванням 1e-10
    
    Повертає:
        Вектор розв'язку
    """
    n = len(funcs)
    if n == 0:
        return []
    if len(x0) != n:
        raise ValueError("Розмірність початкового наближення не узгоджена з кількістю функцій")
    if len(jacobian) != n or any(len(row) != n for row in jacobian):
        raise ValueError("Якобіан повинен бути квадратною матрицею відповідного розміру")
    
    x = x0[:]
    
    for iteration in range(max_iter):
        # Обчислюємо значення функцій
        F = [func(x) for func in funcs]
        
        # Перевіряємо збіжність
        norm_F = math.sqrt(sum(f * f for f in F))
        if norm_F < tol:
            return x
        
        # Обчислюємо якобіан
        J = [[jacobian[i][j](x) for j in range(n)] for i in range(n)]
        
        # Розв'язуємо систему J * delta = -F
        try:
            delta = solve_linear_system(J, [-f for f in F])
        except:
            raise ValueError("Не вдалося розв'язати систему на ітерації {}".format(iteration))
        
        # Оновлюємо x
        for i in range(n):
            x[i] += delta[i]
    
    raise ValueError("Метод не збігся за вказану кількість ітерацій")

def finite_element_1d(func: Callable[[float], float], 
                     a: float, 
                     b: float, 
                     n_elements: int) -> Tuple[List[float], List[float]]:
    """
    Розв'язати крайову задачу методом скінченних елементів в 1D.
    
    Параметри:
        func: Функція правої частини диференціального рівняння
        a: Початок інтервалу
        b: Кінець інтервалу
        n_elements: Кількість елементів
    
    Повертає:
        Кортеж (точки, значення) з розв'язком
    """
    if a >= b:
        raise ValueError("Початок інтервалу повинен бути меншим за кінець")
    if n_elements <= 0:
        raise ValueError("Кількість елементів повинна бути додатньою")
    
    # Створюємо сітку
    h = (b - a) / n_elements
    nodes = [a + i * h for i in range(n_elements + 1)]
    
    # Матриця жорсткості та вектор навантаження
    A = [[0.0 for _ in range(n_elements + 1)] for _ in range(n_elements + 1)]
    b_vec = [0.0 for _ in range(n_elements + 1)]
    
    # Заповнюємо матрицю та вектор для кожного елемента
    for i in range(n_elements):
        # Локальна матриця жорсткості для лінійного елемента
        local_A = [[1/h, -1/h], [-1/h, 1/h]]
        # Локальний вектор навантаження
        mid_point = (nodes[i] + nodes[i+1]) / 2
        local_b = [func(mid_point) * h / 2, func(mid_point) * h / 2]
        
        # Додаємо до глобальної матриці
        for j in range(2):
            for k in range(2):
                A[i+j][i+k] += local_A[j][k]
            b_vec[i+j] += local_b[j]
    
    # Граничні умови (нульові на кінцях)
    A[0][0] = 1e30  # Дуже велике число для фіксації значення
    A[-1][-1] = 1e30
    b_vec[0] = 0
    b_vec[-1] = 0
    
    # Розв'язуємо систему
    try:
        solution = solve_linear_system(A, b_vec)
        return nodes, solution
    except:
        raise ValueError("Не вдалося розв'язати систему скінченних елементів")

def monte_carlo_pi(n_samples: int = 1000000) -> float:
    """
    Обчислити число π методом Монте-Карло.
    
    Параметри:
        n_samples: Кількість випадкових точок, за замовчуванням 1000000
    
    Повертає:
        Наближене значення π
    """
    if n_samples <= 0:
        raise ValueError("Кількість точок повинна бути додатньою")
    
    inside_circle = 0
    
    for _ in range(n_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    
    return 4 * inside_circle / n_samples

def bootstrap_statistics(data: List[float], 
                        statistic_func: Callable[[List[float]], float],
                        n_bootstrap: int = 1000) -> Tuple[float, float, List[float]]:
    """
    Обчислити статистику методом бутстрепу.
    
    Параметри:
        data: Вхідні дані
        statistic_func: Функція для обчислення статистики
        n_bootstrap: Кількість бутстреп-вибірок, за замовчуванням 1000
    
    Повертає:
        Кортеж (середнє, стандартне відхилення, список значень статистики)
    """
    if len(data) == 0:
        raise ValueError("Дані не можуть бути порожніми")
    if n_bootstrap <= 0:
        raise ValueError("Кількість бутстреп-вибірок повинна бути додатньою")
    
    n = len(data)
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Генеруємо бутстреп-вибірку
        bootstrap_sample = [data[random.randint(0, n-1)] for _ in range(n)]
        # Обчислюємо статистику
        stat_value = statistic_func(bootstrap_sample)
        bootstrap_values.append(stat_value)
    
    # Обчислюємо середнє та стандартне відхилення
    mean_stat = sum(bootstrap_values) / n_bootstrap
    variance = sum((x - mean_stat) ** 2 for x in bootstrap_values) / (n_bootstrap - 1)
    std_stat = math.sqrt(variance)
    
    return mean_stat, std_stat, bootstrap_values

def markov_chain_simulation(transition_matrix: List[List[float]], 
                          initial_state: int,
                          n_steps: int) -> List[int]:
    """
    Симулювати марківський ланцюг.
    
    Параметри:
        transition_matrix: Матриця переходів
        initial_state: Початковий стан
        n_steps: Кількість кроків
    
    Повертає:
        Список станів протягом симуляції
    """
    n_states = len(transition_matrix)
    if n_states == 0:
        return []
    if any(len(row) != n_states for row in transition_matrix):
        raise ValueError("Матриця переходів повинна бути квадратною")
    if initial_state < 0 or initial_state >= n_states:
        raise ValueError("Невірний початковий стан")
    if n_steps <= 0:
        raise ValueError("Кількість кроків повинна бути додатньою")
    
    # Перевіряємо, що всі рядки матриці переходів сумуються до 1
    for i, row in enumerate(transition_matrix):
        row_sum = sum(row)
        if abs(row_sum - 1.0) > 1e-10:
            raise ValueError(f"Рядок {i} матриці переходів не сумується до 1")
    
    states = [initial_state]
    current_state = initial_state
    
    for _ in range(n_steps - 1):
        # Генеруємо наступний стан
        cumulative_prob = 0.0
        rand_val = random.random()
        
        for next_state, prob in enumerate(transition_matrix[current_state]):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                current_state = next_state
                break
        
        states.append(current_state)
    
    return states

def linear_programming_simplex(c: List[float], 
                              A: List[List[float]], 
                              b: List[float],
                              max_iter: int = 1000) -> Tuple[float, List[float]]:
    """
    Розв'язати задачу лінійного програмування симплекс-методом.
    
    Параметри:
        c: Коефіцієнти цільової функції
        A: Матриця обмежень
        b: Вектор обмежень
        max_iter: Максимальна кількість ітерацій, за замовчуванням 1000
    
    Повертає:
        Кортеж (оптимальне значення, оптимальний розв'язок)
    """
    m = len(A)  # Кількість обмежень
    n = len(c)  # Кількість змінних
    
    if m == 0 or n == 0:
        return 0.0, [0.0] * n
    if len(b) != m:
        raise ValueError("Розмірність вектора обмежень не узгоджена")
    if any(len(row) != n for row in A):
        raise ValueError("Розмірність матриці обмежень не узгоджена")
    
    # Створюємо розширену матрицю
    tableau = []
    
    # Додаємо обмеження
    for i in range(m):
        row = A[i][:] + [0] * m + [b[i]]
        row[n + i] = 1  # Додаємо слак-змінні
        tableau.append(row)
    
    # Додаємо цільову функцію
    obj_row = [-x for x in c] + [0] * (m + 1)
    tableau.append(obj_row)
    
    # Симплекс-метод
    for iteration in range(max_iter):
        # Знаходимо стовпець з найменшим значенням в останньому рядку
        pivot_col = min(range(n + m), key=lambda i: tableau[-1][i])
        
        # Перевіряємо оптимальність
        if tableau[-1][pivot_col] >= -1e-10:
            # Оптимальний розв'язок знайдено
            solution = [0.0] * n
            for i in range(m):
                # Знаходимо базисні змінні
                basic_var = -1
                for j in range(n + m):
                    if abs(tableau[i][j] - 1.0) < 1e-10 and all(abs(tableau[k][j]) < 1e-10 for k in range(m) if k != i):
                        basic_var = j
                        break
                if 0 <= basic_var < n:
                    solution[basic_var] = tableau[i][-1]
            
            return -tableau[-1][-1], solution
        
        # Знаходимо рядок з мінімальним відношенням
        ratios = []
        for i in range(m):
            if tableau[i][pivot_col] > 1e-10:
                ratios.append(tableau[i][-1] / tableau[i][pivot_col])
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            raise ValueError("Задача необмежена")
        
        pivot_row = min(range(m), key=lambda i: ratios[i])
        
        # Виконуємо елементарні перетворення
        pivot_element = tableau[pivot_row][pivot_col]
        
        # Нормалізуємо провідний рядок
        for j in range(n + m + 1):
            tableau[pivot_row][j] /= pivot_element
        
        # Обнуляємо провідний стовпець
        for i in range(m + 1):
            if i != pivot_row and abs(tableau[i][pivot_col]) > 1e-10:
                factor = tableau[i][pivot_col]
                for j in range(n + m + 1):
                    tableau[i][j] -= factor * tableau[pivot_row][j]
    
    raise ValueError("Симплекс-метод не збігся за вказану кількість ітерацій")