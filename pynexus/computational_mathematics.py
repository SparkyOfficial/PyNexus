"""
Модуль для обчислювальної математики в PyNexus.
Включає функції для чисельного аналізу, розв'язання диференціальних рівнянь,
інтерполяції, апроксимації та інших обчислювальних методів.
"""

import math
import numpy as np
from typing import List, Tuple, Callable, Union, Optional
from scipy import integrate, interpolate, optimize
from scipy.linalg import solve, eig
import matplotlib.pyplot as plt

# Чисельний аналіз
def numerical_differentiation(f: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Чисельне диференціювання функції методом центральної різниці.
    
    Параметри:
        f: Функція для диференціювання
        x: Точка, в якій обчислюється похідна
        h: Крок диференціювання
    
    Повертає:
        Значення похідної в точці x
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_integration(f: Callable[[float], float], a: float, b: float, 
                         method: str = 'simpson', n: int = 1000) -> float:
    """
    Чисельне інтегрування функції різними методами.
    
    Параметри:
        f: Функція для інтегрування
        a: Нижня межа інтегрування
        b: Верхня межа інтегрування
        method: Метод інтегрування ('rectangle', 'trapezoid', 'simpson')
        n: Кількість інтервалів
    
    Повертає:
        Значення інтеграла
    """
    if method == 'rectangle':
        # Метод прямокутників
        dx = (b - a) / n
        return sum(f(a + i * dx) for i in range(n)) * dx
    elif method == 'trapezoid':
        # Метод трапецій
        dx = (b - a) / n
        return (f(a) + 2 * sum(f(a + i * dx) for i in range(1, n)) + f(b)) * dx / 2
    elif method == 'simpson':
        # Метод Сімпсона
        if n % 2 == 1:
            n += 1  # n має бути парним
        dx = (b - a) / n
        return (f(a) + 4 * sum(f(a + i * dx) for i in range(1, n, 2)) + 
                2 * sum(f(a + i * dx) for i in range(2, n, 2)) + f(b)) * dx / 3
    else:
        raise ValueError("Невідомий метод інтегрування")

def monte_carlo_integration(f: Callable[[List[float]], float], 
                          bounds: List[Tuple[float, float]], 
                          n_samples: int = 1000000) -> float:
    """
    Інтегрування методом Монте-Карло.
    
    Параметри:
        f: Функція для інтегрування
        bounds: Межі інтегрування для кожної змінної [(min1, max1), (min2, max2), ...]
        n_samples: Кількість випадкових точок
    
    Повертає:
        Значення інтеграла
    """
    dim = len(bounds)
    volume = 1.0
    for min_val, max_val in bounds:
        volume *= (max_val - min_val)
    
    # Генерація випадкових точок
    points = []
    for _ in range(n_samples):
        point = [np.random.uniform(min_val, max_val) for min_val, max_val in bounds]
        points.append(point)
    
    # Обчислення значень функції
    values = [f(point) for point in points]
    
    # Середнє значення
    mean_value = sum(values) / n_samples
    
    return mean_value * volume

def adaptive_integration(f: Callable[[float], float], a: float, b: float, 
                       tol: float = 1e-6) -> float:
    """
    Адаптивне інтегрування методом Ромберга.
    
    Параметри:
        f: Функція для інтегрування
        a: Нижня межа інтегрування
        b: Верхня межа інтегрування
        tol: Точність
    
    Повертає:
        Значення інтеграла
    """
    # Метод трапецій з подвоєнням кількості інтервалів
    def trapezoid_method(n):
        h = (b - a) / n
        return (f(a) + 2 * sum(f(a + i * h) for i in range(1, n)) + f(b)) * h / 2
    
    # Початкові значення
    n = 1
    old_result = trapezoid_method(n)
    n *= 2
    new_result = trapezoid_method(n)
    
    # Екстраполяція Річардсона
    while abs(new_result - old_result) > tol:
        n *= 2
        old_result = new_result
        new_result = trapezoid_method(n)
        # Екстраполяція
        new_result = (4 * new_result - old_result) / 3
    
    return new_result

# Інтерполяція та апроксимація
def polynomial_interpolation(x_points: List[float], y_points: List[float], 
                           x: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    Поліноміальна інтерполяція методом Лагранжа.
    
    Параметри:
        x_points: Точки x
        y_points: Точки y
        x: Точка(и) для інтерполяції
    
    Повертає:
        Значення інтерполяції в точці(ах) x
    """
    def lagrange_basis(i, x_val):
        """Базисний поліном Лагранжа"""
        result = 1.0
        for j in range(len(x_points)):
            if i != j:
                result *= (x_val - x_points[j]) / (x_points[i] - x_points[j])
        return result
    
    def interpolate_single(x_val):
        """Інтерполяція в одній точці"""
        result = 0.0
        for i in range(len(x_points)):
            result += y_points[i] * lagrange_basis(i, x_val)
        return result
    
    if isinstance(x, (int, float)):
        return interpolate_single(x)
    else:
        return [interpolate_single(xi) for xi in x]

def spline_interpolation(x_points: List[float], y_points: List[float], 
                        x: Union[float, List[float]], 
                        degree: int = 3) -> Union[float, List[float]]:
    """
    Сплайн-інтерполяція.
    
    Параметри:
        x_points: Точки x
        y_points: Точки y
        x: Точка(и) для інтерполяції
        degree: Степінь сплайна (1 - лінійний, 3 - кубічний)
    
    Повертає:
        Значення інтерполяції в точці(ах) x
    """
    # Створення сплайна
    if degree == 1:
        spl = interpolate.interp1d(x_points, y_points, kind='linear')
    elif degree == 3:
        spl = interpolate.interp1d(x_points, y_points, kind='cubic')
    else:
        raise ValueError("Підтримуються лише лінійна (1) та кубічна (3) інтерполяція")
    
    if isinstance(x, (int, float)):
        return float(spl(x))
    else:
        return [float(val) for val in spl(x)]

def least_squares_approximation(x_points: List[float], y_points: List[float], 
                               degree: int = 1) -> List[float]:
    """
    Апроксимація методом найменших квадратів.
    
    Параметри:
        x_points: Точки x
        y_points: Точки y
        degree: Степінь полінома
    
    Повертає:
        Коефіцієнти полінома [a0, a1, a2, ...] для a0 + a1*x + a2*x^2 + ...
    """
    # Створення матриці Вандермонда
    A = []
    for xi in x_points:
        row = [xi ** i for i in range(degree + 1)]
        A.append(row)
    
    A = np.array(A)
    y = np.array(y_points)
    
    # Розв'язання системи нормальних рівнянь: A^T * A * x = A^T * y
    AtA = np.dot(A.T, A)
    Aty = np.dot(A.T, y)
    
    # Розв'язання системи лінійних рівнянь
    coefficients = solve(AtA, Aty)
    
    return coefficients.tolist()

def fourier_approximation(x_points: List[float], y_points: List[float], 
                         n_terms: int = 10) -> Tuple[List[float], List[float], float]:
    """
    Апроксимація рядом Фур'є.
    
    Параметри:
        x_points: Точки x
        y_points: Точки y
        n_terms: Кількість членів ряду
    
    Повертає:
        Кортеж (a_coefficients, b_coefficients, a0) з коефіцієнтами ряду Фур'є
    """
    # Період
    T = x_points[-1] - x_points[0]
    
    # Постійний член
    a0 = 2 * sum(y_points) / len(y_points)
    
    # Коефіцієнти an
    a_coeffs = []
    for n in range(1, n_terms + 1):
        an = 0
        for i in range(len(x_points)):
            an += y_points[i] * math.cos(2 * math.pi * n * x_points[i] / T)
        an *= 2 / len(y_points)
        a_coeffs.append(an)
    
    # Коефіцієнти bn
    b_coeffs = []
    for n in range(1, n_terms + 1):
        bn = 0
        for i in range(len(x_points)):
            bn += y_points[i] * math.sin(2 * math.pi * n * x_points[i] / T)
        bn *= 2 / len(y_points)
        b_coeffs.append(bn)
    
    return (a_coeffs, b_coeffs, a0)

# Розв'язання диференціальних рівнянь
def euler_method(f: Callable[[float, float], float], x0: float, y0: float, 
                x_end: float, n_steps: int) -> Tuple[List[float], List[float]]:
    """
    Метод Ейлера для розв'язання звичайного диференціального рівняння.
    
    Параметри:
        f: Функція dy/dx = f(x, y)
        x0: Початкове значення x
        y0: Початкове значення y
        x_end: Кінцеве значення x
        n_steps: Кількість кроків
    
    Повертає:
        Кортеж (x_values, y_values) з розв'язками
    """
    h = (x_end - x0) / n_steps
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    for _ in range(n_steps):
        y += h * f(x, y)
        x += h
        x_values.append(x)
        y_values.append(y)
    
    return (x_values, y_values)

def runge_kutta_method(f: Callable[[float, float], float], x0: float, y0: float, 
                      x_end: float, n_steps: int) -> Tuple[List[float], List[float]]:
    """
    Метод Рунге-Кутта 4-го порядку для розв'язання звичайного диференціального рівняння.
    
    Параметри:
        f: Функція dy/dx = f(x, y)
        x0: Початкове значення x
        y0: Початкове значення y
        x_end: Кінцеве значення x
        n_steps: Кількість кроків
    
    Повертає:
        Кортеж (x_values, y_values) з розв'язками
    """
    h = (x_end - x0) / n_steps
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    for _ in range(n_steps):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        x_values.append(x)
        y_values.append(y)
    
    return (x_values, y_values)

def system_runge_kutta_method(f_system: List[Callable], initial_conditions: List[float], 
                             x0: float, x_end: float, n_steps: int) -> Tuple[List[float], List[List[float]]]:
    """
    Метод Рунге-Кутта 4-го порядку для системи диференціальних рівнянь.
    
    Параметри:
        f_system: Список функцій [f1, f2, ...] де dyi/dx = fi(x, y1, y2, ...)
        initial_conditions: Початкові умови [y1_0, y2_0, ...]
        x0: Початкове значення x
        x_end: Кінцеве значення x
        n_steps: Кількість кроків
    
    Повертає:
        Кортеж (x_values, y_system_values) де y_system_values - список [y1_values, y2_values, ...]
    """
    h = (x_end - x0) / n_steps
    n_equations = len(initial_conditions)
    
    x_values = [x0]
    y_system_values = [[y0] for y0 in initial_conditions]
    
    x = x0
    y = initial_conditions.copy()
    
    for _ in range(n_steps):
        # k1
        k1 = [h * f(x, *y) for f in f_system]
        
        # k2
        y_temp = [y[i] + k1[i]/2 for i in range(n_equations)]
        k2 = [h * f(x + h/2, *y_temp) for f in f_system]
        
        # k3
        y_temp = [y[i] + k2[i]/2 for i in range(n_equations)]
        k3 = [h * f(x + h/2, *y_temp) for f in f_system]
        
        # k4
        y_temp = [y[i] + k3[i] for i in range(n_equations)]
        k4 = [h * f(x + h, *y_temp) for f in f_system]
        
        # Оновлення значень
        for i in range(n_equations):
            y[i] += (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
            y_system_values[i].append(y[i])
        
        x += h
        x_values.append(x)
    
    return (x_values, y_system_values)

# Лінійна алгебра
def eigenvalue_decomposition(matrix: List[List[float]]) -> Tuple[List[complex], List[List[complex]]]:
    """
    Власні значення та власні вектори матриці.
    
    Параметри:
        matrix: Квадратна матриця
    
    Повертає:
        Кортеж (eigenvalues, eigenvectors) з власними значеннями та власними векторами
    """
    # Конвертація до numpy масиву
    A = np.array(matrix)
    
    # Обчислення власних значень та векторів
    eigenvalues, eigenvectors = eig(A)
    
    return (eigenvalues.tolist(), eigenvectors.tolist())

def singular_value_decomposition(matrix: List[List[float]]) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Сингулярне розкладання матриці.
    
    Параметри:
        matrix: Прямокутна матриця
    
    Повертає:
        Кортеж (U, singular_values, Vt) з компонентами SVD
    """
    # Конвертація до numpy масиву
    A = np.array(matrix)
    
    # SVD розкладання
    U, s, Vt = np.linalg.svd(A)
    
    return (U.tolist(), s.tolist(), Vt.tolist())

def matrix_exponential(matrix: List[List[float]]) -> List[List[float]]:
    """
    Експонента матриці.
    
    Параметри:
        matrix: Квадратна матриця
    
    Повертає:
        Експонента матриці
    """
    # Конвертація до numpy масиву
    A = np.array(matrix)
    
    # Обчислення експоненти матриці
    exp_A = np.exp(A)
    
    return exp_A.tolist()

def condition_number(matrix: List[List[float]]) -> float:
    """
    Число обумовленості матриці.
    
    Параметри:
        matrix: Квадратна матриця
    
    Повертає:
        Число обумовленості
    """
    # Конвертація до numpy масиву
    A = np.array(matrix)
    
    # Обчислення числа обумовленості
    cond_num = np.linalg.cond(A)
    
    return float(cond_num)

# Оптимізація
def gradient_descent(f: Callable[[List[float]], float], 
                    grad_f: Callable[[List[float]], List[float]], 
                    initial_point: List[float], 
                    learning_rate: float = 0.01, 
                    max_iterations: int = 1000, 
                    tolerance: float = 1e-6) -> Tuple[List[float], float]:
    """
    Градієнтний спуск для мінімізації функції.
    
    Параметри:
        f: Функція для мінімізації
        grad_f: Градієнт функції
        initial_point: Початкова точка
        learning_rate: Крок навчання
        max_iterations: Максимальна кількість ітерацій
        tolerance: Точність
    
    Повертає:
        Кортеж (optimal_point, optimal_value) з оптимальною точкою та значенням функції
    """
    x = initial_point.copy()
    
    for i in range(max_iterations):
        # Обчислення градієнта
        gradient = grad_f(x)
        
        # Оновлення точки
        x_new = [x[j] - learning_rate * gradient[j] for j in range(len(x))]
        
        # Перевірка збіжності
        diff = sum((x_new[j] - x[j])**2 for j in range(len(x)))**0.5
        if diff < tolerance:
            break
            
        x = x_new
    
    return (x, f(x))

def newton_method(f: Callable[[float], float], 
                 df: Callable[[float], float], 
                 d2f: Callable[[float], float], 
                 initial_guess: float, 
                 max_iterations: int = 100, 
                 tolerance: float = 1e-6) -> Tuple[float, float]:
    """
    Метод Ньютона для мінімізації функції однієї змінної.
    
    Параметри:
        f: Функція для мінімізації
        df: Перша похідна функції
        d2f: Друга похідна функції
        initial_guess: Початкове наближення
        max_iterations: Максимальна кількість ітерацій
        tolerance: Точність
    
    Повертає:
        Кортеж (optimal_point, optimal_value) з оптимальною точкою та значенням функції
    """
    x = initial_guess
    
    for i in range(max_iterations):
        # Обчислення першої та другої похідних
        first_derivative = df(x)
        second_derivative = d2f(x)
        
        # Умова зупинки
        if abs(first_derivative) < tolerance:
            break
            
        # Оновлення точки
        if abs(second_derivative) < tolerance:
            # Уникаємо ділення на нуль
            x -= first_derivative * 0.1
        else:
            x -= first_derivative / second_derivative
    
    return (x, f(x))

def constrained_optimization(f: Callable[[List[float]], float], 
                           constraints: List[Callable[[List[float]], float]], 
                           initial_point: List[float], 
                           method: str = 'penalty') -> Tuple[List[float], float]:
    """
    Оптимізація з обмеженнями.
    
    Параметри:
        f: Цільова функція
        constraints: Список функцій обмежень (повинні дорівнювати 0)
        initial_point: Початкова точка
        method: Метод ('penalty' або 'lagrange')
    
    Повертає:
        Кортеж (optimal_point, optimal_value) з оптимальною точкою та значенням функції
    """
    if method == 'penalty':
        # Метод штрафних функцій
        def penalty_function(x, penalty_coeff=1000):
            objective = f(x)
            penalty = 0
            for constraint in constraints:
                penalty += penalty_coeff * constraint(x)**2
            return objective + penalty
        
        # Використання scipy.optimize для мінімізації
        result = optimize.minimize(penalty_function, initial_point, method='BFGS')
        return (result.x.tolist(), result.fun)
    elif method == 'lagrange':
        # Метод множників Лагранжа
        # Це спрощена реалізація, для складних випадків використовуйте спеціалізовані бібліотеки
        raise NotImplementedError("Метод множників Лагранжа ще не реалізовано")
    else:
        raise ValueError("Невідомий метод оптимізації")

# Статистика та теорія ймовірностей
def monte_carlo_simulation(n_simulations: int, 
                          random_process: Callable[[], float]) -> List[float]:
    """
    Монте-Карло симуляція.
    
    Параметри:
        n_simulations: Кількість симуляцій
        random_process: Функція, що моделює випадковий процес
    
    Повертає:
        Результати симуляцій
    """
    results = []
    for _ in range(n_simulations):
        results.append(random_process())
    return results

def bootstrap_sampling(data: List[float], 
                      n_samples: int = 1000, 
                      sample_size: Optional[int] = None) -> List[List[float]]:
    """
    Bootstrap вибірка.
    
    Параметри:
        data: Вихідні дані
        n_samples: Кількість bootstrap вибірок
        sample_size: Розмір кожної вибірки (за замовчуванням - розмір вихідних даних)
    
    Повертає:
        Список bootstrap вибірок
    """
    if sample_size is None:
        sample_size = len(data)
    
    samples = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=sample_size, replace=True).tolist()
        samples.append(sample)
    
    return samples

def markov_chain_simulation(transition_matrix: List[List[float]], 
                           initial_state: int, 
                           n_steps: int) -> List[int]:
    """
    Симуляція марковського ланцюга.
    
    Параметри:
        transition_matrix: Матриця переходів
        initial_state: Початковий стан
        n_steps: Кількість кроків
    
    Повертає:
        Послідовність станів
    """
    states = [initial_state]
    current_state = initial_state
    
    for _ in range(n_steps):
        # Вибір наступного стану згідно з ймовірностями переходу
        probabilities = transition_matrix[current_state]
        next_state = np.random.choice(len(probabilities), p=probabilities)
        states.append(next_state)
        current_state = next_state
    
    return states

def bayesian_inference(prior: Callable[[float], float], 
                      likelihood: Callable[[float, float], float], 
                      data: List[float], 
                      n_samples: int = 10000) -> List[float]:
    """
    Байєсівський висновок методом Монте-Карло.
    
    Параметри:
        prior: Апріорний розподіл
        likelihood: Функція правдоподібності
        data: Спостережувані дані
        n_samples: Кількість зразків
    
    Повертає:
        Апостеріорний розподіл
    """
    samples = []
    
    # Генерація зразків з апріорного розподілу
    theta_samples = [np.random.uniform(-10, 10) for _ in range(n_samples)]
    
    # Обчислення правдоподібності для кожного зразка
    likelihoods = []
    for theta in theta_samples:
        likelihood_val = 1.0
        for x in data:
            likelihood_val *= likelihood(theta, x)
        likelihoods.append(likelihood_val)
    
    # Обчислення апостеріорної ймовірності
    posterior = [prior(theta) * likelihoods[i] for i, theta in enumerate(theta_samples)]
    
    # Нормалізація
    total = sum(posterior)
    if total > 0:
        posterior = [p / total for p in posterior]
    
    # Вибірка з апостеріорного розподілу
    selected_indices = np.random.choice(n_samples, size=n_samples, p=posterior)
    samples = [theta_samples[i] for i in selected_indices]
    
    return samples

# Чисельні методи для розв'язання систем рівнянь
def newton_raphson_system(f_system: List[Callable], 
                         jacobian: List[List[Callable]], 
                         initial_guess: List[float], 
                         max_iterations: int = 100, 
                         tolerance: float = 1e-6) -> Tuple[List[float], List[float]]:
    """
    Метод Ньютона-Рафсона для системи нелінійних рівнянь.
    
    Параметри:
        f_system: Список функцій системи
        jacobian: Якобіан системи (матриця часткових похідних)
        initial_guess: Початкове наближення
        max_iterations: Максимальна кількість ітерацій
        tolerance: Точність
    
    Повертає:
        Кортеж (solution, residuals) з розв'язком та залишками
    """
    x = initial_guess.copy()
    
    for iteration in range(max_iterations):
        # Обчислення значень функцій
        f_values = [f(*x) for f in f_system]
        
        # Перевірка збіжності
        residual = sum(val**2 for val in f_values)**0.5
        if residual < tolerance:
            break
        
        # Обчислення якобіана
        J = []
        for i in range(len(jacobian)):
            row = []
            for j in range(len(jacobian[i])):
                row.append(jacobian[i][j](*x))
            J.append(row)
        
        # Розв'язання системи J * dx = -f для знаходження кроку
        try:
            dx = solve(np.array(J), -np.array(f_values))
            # Оновлення x
            x = [x[i] + dx[i] for i in range(len(x))]
        except np.linalg.LinAlgError:
            # Якщо матриця сингулярна, використовуємо простий градієнтний крок
            x = [x[i] - 0.1 * f_values[i] for i in range(len(x))]
    
    # Обчислення фінальних залишків
    final_residuals = [f(*x) for f in f_system]
    
    return (x, final_residuals)

def fixed_point_iteration(g: Callable[[float], float], 
                         initial_guess: float, 
                         max_iterations: int = 100, 
                         tolerance: float = 1e-6) -> Tuple[float, int]:
    """
    Метод простої ітерації для розв'язання рівняння x = g(x).
    
    Параметри:
        g: Функція ітерації
        initial_guess: Початкове наближення
        max_iterations: Максимальна кількість ітерацій
        tolerance: Точність
    
    Повертає:
        Кортеж (solution, iterations) з розв'язком та кількістю ітерацій
    """
    x = initial_guess
    
    for i in range(max_iterations):
        x_new = g(x)
        
        # Перевірка збіжності
        if abs(x_new - x) < tolerance:
            return (x_new, i + 1)
        
        x = x_new
    
    return (x, max_iterations)

# Спеціальні функції
def gamma_function(x: float) -> float:
    """
    Гамма-функція.
    
    Параметри:
        x: Аргумент функції
    
    Повертає:
        Значення гамма-функції
    """
    return math.gamma(x)

def beta_function(a: float, b: float) -> float:
    """
    Бета-функція.
    
    Параметри:
        a, b: Параметри функції
    
    Повертає:
        Значення бета-функції
    """
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

def error_function(x: float) -> float:
    """
    Функція помилок (ерф).
    
    Параметри:
        x: Аргумент функції
    
    Повертає:
        Значення функції помилок
    """
    return math.erf(x)

def bessel_function(n: int, x: float) -> float:
    """
    Функція Бесселя першого роду.
    
    Параметри:
        n: Порядок функції
        x: Аргумент функції
    
    Повертає:
        Значення функції Бесселя
    """
    from scipy.special import jv
    return float(jv(n, x))

# Інтегральні перетворення
def fourier_transform(signal: List[float], 
                     sampling_rate: float = 1.0) -> Tuple[List[float], List[complex]]:
    """
    Дискретне перетворення Фур'є.
    
    Параметри:
        signal: Вхідний сигнал
        sampling_rate: Частота дискретизації
    
    Повертає:
        Кортеж (frequencies, spectrum) з частотами та спектром
    """
    # Обчислення FFT
    spectrum = np.fft.fft(signal)
    
    # Обчислення частот
    n = len(signal)
    frequencies = np.fft.fftfreq(n, 1/sampling_rate)
    
    return (frequencies.tolist(), spectrum.tolist())

def inverse_fourier_transform(spectrum: List[complex], 
                            frequencies: List[float]) -> List[float]:
    """
    Обернене перетворення Фур'є.
    
    Параметри:
        spectrum: Спектр сигналу
        frequencies: Частоти
    
    Повертає:
        Відновлений сигнал
    """
    # Обчислення оберненого FFT
    signal = np.fft.ifft(spectrum)
    
    return signal.real.tolist()

def laplace_transform(f: Callable[[float], float], 
                     s_values: List[float], 
                     t_max: float = 10.0, 
                     n_points: int = 1000) -> List[complex]:
    """
    Чисельне перетворення Лапласа.
    
    Параметри:
        f: Функція для перетворення
        s_values: Значення комплексної змінної s
        t_max: Максимальне значення часу
        n_points: Кількість точок для інтегрування
    
    Повертає:
        Значення перетворення Лапласа для кожного s
    """
    results = []
    
    # Точки для інтегрування
    t_points = np.linspace(0, t_max, n_points)
    dt = t_max / (n_points - 1)
    
    for s in s_values:
        # Обчислення інтеграла ∫₀^∞ f(t) * e^(-st) dt
        integral = 0
        for i in range(n_points):
            t = t_points[i]
            integral += f(t) * np.exp(-s * t) * dt
        results.append(complex(integral))
    
    return results

# Чисельні методи для розв'язання інтегральних рівнянь
def fredholm_equation_first_kind(kernel: Callable[[float, float], float], 
                                g: Callable[[float], float], 
                                a: float, b: float, 
                                n_points: int = 100) -> Tuple[List[float], List[float]]:
    """
    Розв'язання інтегрального рівняння Фредгольма першого роду методом квадратур.
    
    ∫ₐᵇ K(x,t) * f(t) dt = g(x)
    
    Параметри:
        kernel: Ядро інтегрального рівняння K(x,t)
        g: Права частина рівняння
        a, b: Межі інтегрування
        n_points: Кількість точок дискретизації
    
    Повертає:
        Кортеж (t_points, f_values) з розв'язком
    """
    # Точки дискретизації
    t_points = np.linspace(a, b, n_points)
    x_points = np.linspace(a, b, n_points)
    dt = (b - a) / (n_points - 1)
    
    # Матриця системи
    A = []
    for x in x_points:
        row = []
        for t in t_points:
            row.append(kernel(x, t) * dt)
        A.append(row)
    
    # Права частина
    b_vector = [g(x) for x in x_points]
    
    # Розв'язання системи лінійних рівнянь
    f_values = solve(np.array(A), np.array(b_vector))
    
    return (t_points.tolist(), f_values.tolist())

def fredholm_equation_second_kind(kernel: Callable[[float, float], float], 
                                 g: Callable[[float], float], 
                                 lambda_param: float, 
                                 a: float, b: float, 
                                 n_points: int = 100) -> Tuple[List[float], List[float]]:
    """
    Розв'язання інтегрального рівняння Фредгольма другого роду методом Ністрема.
    
    f(x) - λ ∫ₐᵇ K(x,t) * f(t) dt = g(x)
    
    Параметри:
        kernel: Ядро інтегрального рівняння K(x,t)
        g: Права частина рівняння
        lambda_param: Параметр λ
        a, b: Межі інтегрування
        n_points: Кількість точок дискретизації
    
    Повертає:
        Кортеж (x_points, f_values) з розв'язком
    """
    # Точки дискретизації
    x_points = np.linspace(a, b, n_points)
    dt = (b - a) / (n_points - 1)
    
    # Матриця системи (I - λ*K)
    A = []
    for i, x in enumerate(x_points):
        row = []
        for j, t in enumerate(x_points):
            if i == j:
                row.append(1.0 - lambda_param * kernel(x, t) * dt)
            else:
                row.append(-lambda_param * kernel(x, t) * dt)
        A.append(row)
    
    # Права частина
    b_vector = [g(x) for x in x_points]
    
    # Розв'язання системи лінійних рівнянь
    f_values = solve(np.array(A), np.array(b_vector))
    
    return (x_points.tolist(), f_values.tolist())

# Методи для розв'язання крайових задач
def finite_difference_method(second_derivative_coeff: Callable[[float], float], 
                           first_derivative_coeff: Callable[[float], float], 
                           zero_derivative_coeff: Callable[[float], float], 
                           source_term: Callable[[float], float], 
                           boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]], 
                           a: float, b: float, 
                           n_points: int = 100) -> Tuple[List[float], List[float]]:
    """
    Метод скінченних різниць для розв'язання крайової задачі.
    
    a(x) * u''(x) + b(x) * u'(x) + c(x) * u(x) = f(x)
    
    Параметри:
        second_derivative_coeff: Коефіцієнт a(x) при другій похідній
        first_derivative_coeff: Коефіцієнт b(x) при першій похідній
        zero_derivative_coeff: Коефіцієнт c(x) при функції
        source_term: Права частина f(x)
        boundary_conditions: Граничні умови ((x0, u0), (xn, un))
        a, b: Межі відрізка
        n_points: Кількість точок
    
    Повертає:
        Кортеж (x_points, u_values) з розв'язком
    """
    # Точки дискретизації
    x_points = np.linspace(a, b, n_points)
    dx = (b - a) / (n_points - 1)
    
    # Матриця системи
    A = []
    b_vector = []
    
    for i in range(n_points):
        x = x_points[i]
        row = [0.0] * n_points
        
        if i == 0:
            # Ліва гранична умова
            row[0] = 1.0
            b_vector.append(boundary_conditions[0][1])
        elif i == n_points - 1:
            # Права гранична умова
            row[n_points - 1] = 1.0
            b_vector.append(boundary_conditions[1][1])
        else:
            # Внутрішні точки
            # Апроксимація другої похідної: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
            # Апроксимація першої похідної: (u[i+1] - u[i-1]) / (2*dx)
            
            # Коефіцієнти
            a_coeff = second_derivative_coeff(x)
            b_coeff = first_derivative_coeff(x)
            c_coeff = zero_derivative_coeff(x)
            f_val = source_term(x)
            
            # Заповнення матриці
            row[i-1] = a_coeff / (dx*dx) - b_coeff / (2*dx)
            row[i] = -2 * a_coeff / (dx*dx) + c_coeff
            row[i+1] = a_coeff / (dx*dx) + b_coeff / (2*dx)
            
            b_vector.append(f_val)
        
        A.append(row)
    
    # Розв'язання системи лінійних рівнянь
    u_values = solve(np.array(A), np.array(b_vector))
    
    return (x_points.tolist(), u_values.tolist())

# Функції для роботи з комплексними числами
def complex_analysis_functions(z: complex) -> dict:
    """
    Аналітичні функції комплексної змінної.
    
    Параметри:
        z: Комплексне число
    
    Повертає:
        Словник зі значеннями різних функцій
    """
    results = {
        'exp': np.exp(z),
        'log': np.log(z) if z != 0 else complex(float('inf')),
        'sin': np.sin(z),
        'cos': np.cos(z),
        'tan': np.tan(z),
        'sinh': np.sinh(z),
        'cosh': np.cosh(z),
        'tanh': np.tanh(z),
        'sqrt': np.sqrt(z),
        'abs': abs(z),
        'arg': np.angle(z),
        'conj': np.conj(z)
    }
    
    return results

def residue_theorem(poles: List[complex], 
                   residues: List[complex], 
                   contour_integral: float = 0) -> complex:
    """
    Теорема про лишки в комплексному аналізі.
    
    Параметри:
        poles: Полюси функції всередині контуру
        residues: Лишки в полюсах
        contour_integral: Інтеграл по контуру
    
    Повертає:
        Значення інтеграла
    """
    # Сума лишків
    residue_sum = sum(residues)
    
    # За теоремою про лишки: ∮ f(z) dz = 2πi * Σ(лишки)
    integral_value = 2j * np.pi * residue_sum + contour_integral
    
    return integral_value

# Методи для розв'язання рівнянь в частинних похідних
def finite_element_method_1d(coefficients: Callable[[float], Tuple[float, float, float]], 
                            source_term: Callable[[float], float], 
                            boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]], 
                            a: float, b: float, 
                            n_elements: int = 100) -> Tuple[List[float], List[float]]:
    """
    Метод скінченних елементів для розв'язання рівняння в частинних похідних 1D.
    
    -(a(x) * u'(x))' + b(x) * u'(x) + c(x) * u(x) = f(x)
    
    Параметри:
        coefficients: Функція, що повертає (a(x), b(x), c(x))
        source_term: Права частина f(x)
        boundary_conditions: Граничні умови ((x0, u0), (xn, un))
        a, b: Межі відрізка
        n_elements: Кількість елементів
    
    Повертає:
        Кортеж (x_points, u_values) з розв'язком
    """
    # Точки дискретизації
    x_points = np.linspace(a, b, n_elements + 1)
    dx = (b - a) / n_elements
    
    # Глобальна матриця жорсткості та вектор навантаження
    global_matrix = np.zeros((n_elements + 1, n_elements + 1))
    global_vector = np.zeros(n_elements + 1)
    
    # Локальні матриці для кожного елемента
    for i in range(n_elements):
        x_left = x_points[i]
        x_right = x_points[i + 1]
        x_mid = (x_left + x_right) / 2
        
        # Оцінка коефіцієнтів у середині елемента
        a_coeff, b_coeff, c_coeff = coefficients(x_mid)
        
        # Локальна матриця жорсткості (спрощена для лінійних елементів)
        local_matrix = np.array([
            [a_coeff/dx + b_coeff/2 + c_coeff*dx/3, -a_coeff/dx + b_coeff/2 - c_coeff*dx/6],
            [-a_coeff/dx - b_coeff/2 - c_coeff*dx/6, a_coeff/dx - b_coeff/2 + c_coeff*dx/3]
        ])
        
        # Локальний вектор навантаження
        f_mid = source_term(x_mid)
        local_vector = np.array([f_mid * dx/2, f_mid * dx/2])
        
        # Додавання до глобальної матриці
        global_matrix[i:i+2, i:i+2] += local_matrix
        global_vector[i:i+2] += local_vector
    
    # Застосування граничних умов
    # Ліва умова
    global_matrix[0, :] = 0
    global_matrix[0, 0] = 1
    global_vector[0] = boundary_conditions[0][1]
    
    # Права умова
    global_matrix[-1, :] = 0
    global_matrix[-1, -1] = 1
    global_vector[-1] = boundary_conditions[1][1]
    
    # Розв'язання системи
    u_values = solve(global_matrix, global_vector)
    
    return (x_points.tolist(), u_values.tolist())

# Спеціалізовані чисельні методи
def spectral_method_basis_functions(n_points: int, 
                                   domain: Tuple[float, float] = (-1, 1)) -> List[Callable]:
    """
    Генерація базисних функцій для спектрального методу.
    
    Параметри:
        n_points: Кількість точок
        domain: Область визначення
    
    Повертає:
        Список базисних функцій
    """
    a, b = domain
    basis_functions = []
    
    for k in range(n_points):
        def basis_func(x, k=k):
            # Нормалізовані поліноми Лежандра
            return np.polynomial.legendre.legval((2*x - (a+b))/(b-a), [0]*k + [1])
        basis_functions.append(basis_func)
    
    return basis_functions

def multigrid_method(initial_guess: List[List[float]], 
                    residual_func: Callable, 
                    smoother: Callable, 
                    n_cycles: int = 5) -> List[List[float]]:
    """
    Метод багаторівневої ітерації (Multigrid).
    
    Параметри:
        initial_guess: Початкове наближення
        residual_func: Функція обчислення залишків
        smoother: Функція згладжування
        n_cycles: Кількість циклів
    
    Повертає:
        Розв'язок
    """
    solution = [row[:] for row in initial_guess]  # Глибока копія
    
    for cycle in range(n_cycles):
        # Згладжування
        solution = smoother(solution)
        
        # Обчислення залишків
        residuals = residual_func(solution)
        
        # Рестрикція (перехід на грубішу сітку)
        # (спрощена реалізація)
        
        # Рекурсивне розв'язання на грубій сітці
        # (спрощена реалізція)
        
        # Інтерполяція (перехід на тоншу сітку)
        # (спрощена реалізація)
        
        # Додаткове згладжування
    
    return solution

# Адаптивні методи
def adaptive_step_size_integration(f: Callable[[float, float], float], 
                                  x0: float, y0: float, 
                                  x_end: float, 
                                  initial_step: float = 0.1, 
                                  tolerance: float = 1e-6) -> Tuple[List[float], List[float]]:
    """
    Адаптивний метод інтегрування зі змінним кроком.
    
    Параметри:
        f: Функція dy/dx = f(x, y)
        x0: Початкове значення x
        y0: Початкове значення y
        x_end: Кінцеве значення x
        initial_step: Початковий крок
        tolerance: Точність
    
    Повертає:
        Кортеж (x_values, y_values) з розв'язками
    """
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    h = initial_step
    
    while x < x_end:
        # Метод Рунге-Кутта 4-го порядку
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y_rk4 = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Метод Рунге-Кутта 2-го порядку (для оцінки похибки)
        k1_low = h * f(x, y)
        k2_low = h * f(x + h, y + k1_low)
        y_rk2 = y + (k1_low + k2_low) / 2
        
        # Оцінка похибки
        error = abs(y_rk4 - y_rk2)
        
        # Адаптація кроку
        if error <= tolerance:
            # Прийняти крок
            x += h
            y = y_rk4
            x_values.append(x)
            y_values.append(y)
            
            # Збільшити крок якщо похибка маленька
            if error < tolerance / 4:
                h *= 1.5
        else:
            # Зменшити крок
            h *= 0.5
    
    return (x_values, y_values)

# Функції для роботи з розрідженими матрицями
def sparse_matrix_operations(matrix_dict: dict, 
                           operation: str = 'multiply') -> dict:
    """
    Операції з розрідженими матрицями.
    
    Параметри:
        matrix_dict: Словник {(i, j): value} з ненульовими елементами
        operation: Тип операції
    
    Повертає:
        Результат операції
    """
    if operation == 'multiply':
        # Множення розріджених матриць (спрощена реалізація)
        result = {}
        # Для кожної ненульової пари елементів
        for (i1, j1), val1 in matrix_dict.items():
            for (i2, j2), val2 in matrix_dict.items():
                if j1 == i2:  # Умова множення матриць
                    key = (i1, j2)
                    if key in result:
                        result[key] += val1 * val2
                    else:
                        result[key] = val1 * val2
        return result
    elif operation == 'transpose':
        # Транспонування
        result = {}
        for (i, j), val in matrix_dict.items():
            result[(j, i)] = val
        return result
    else:
        raise ValueError("Невідома операція")

# Ітераційні методи для великих систем
def conjugate_gradient_method(A_dict: dict, 
                            b: List[float], 
                            x0: List[float], 
                            max_iterations: int = 1000, 
                            tolerance: float = 1e-6) -> Tuple[List[float], int]:
    """
    Метод спряжених градієнтів для розв'язання системи Ax = b.
    
    Параметри:
        A_dict: Розріджена матриця A у форматі {(i, j): value}
        b: Вектор правої частини
        x0: Початкове наближення
        max_iterations: Максимальна кількість ітерацій
        tolerance: Точність
    
    Повертає:
        Кортеж (solution, iterations) з розв'язком та кількістю ітерацій
    """
    x = x0.copy()
    r = [b[i] - sum(A_dict.get((i, j), 0) * x[j] for j in range(len(x))) for i in range(len(b))]
    p = r.copy()
    
    rsold = sum(ri**2 for ri in r)
    
    for i in range(max_iterations):
        Ap = [sum(A_dict.get((i, j), 0) * p[j] for j in range(len(p))) for i in range(len(p))]
        alpha = rsold / sum(pi * Api for pi, Api in zip(p, Ap))
        
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * Api for ri, Api in zip(r, Ap)]
        
        rsnew = sum(ri**2 for ri in r)
        
        if math.sqrt(rsnew) < tolerance:
            return (x, i + 1)
        
        beta = rsnew / rsold
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rsold = rsnew
    
    return (x, max_iterations)

# Статистичні методи для великих даних
def online_statistics(data_stream: List[float]) -> dict:
    """
    Онлайн обчислення статистик (для великих потоків даних).
    
    Параметри:
        data_stream: Потік даних
    
    Повертає:
        Словник зі статистиками
    """
    n = 0
    mean = 0.0
    m2 = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    
    for x in data_stream:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
        
        min_val = min(min_val, x)
        max_val = max(max_val, x)
    
    if n < 2:
        variance = 0.0
    else:
        variance = m2 / (n - 1)
    
    std_dev = math.sqrt(variance) if variance >= 0 else 0.0
    
    return {
        'count': n,
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': min_val,
        'max': max_val
    }

# Методи для роботи з випадковими процесами
def wiener_process_simulation(n_steps: int, 
                             dt: float = 0.01) -> Tuple[List[float], List[float]]:
    """
    Симуляція вінерівського процесу (бровіанського руху).
    
    Параметри:
        n_steps: Кількість кроків
        dt: Крок часу
    
    Повертає:
        Кортеж (time_points, values) з траєкторією процесу
    """
    time_points = [i * dt for i in range(n_steps + 1)]
    values = [0.0]  # Початкове значення
    
    for i in range(n_steps):
        # Нормальний розподіл з середнім 0 і дисперсією dt
        increment = np.random.normal(0, math.sqrt(dt))
        values.append(values[-1] + increment)
    
    return (time_points, values)

def ornstein_uhlenbeck_process(theta: float, 
                              mu: float, 
                              sigma: float, 
                              x0: float, 
                              n_steps: int, 
                              dt: float = 0.01) -> Tuple[List[float], List[float]]:
    """
    Симуляція процесу Орнштейна-Уленбека.
    
    Параметри:
        theta: Швидкість повернення до середнього
        mu: Середнє значення
        sigma: Волатильність
        x0: Початкове значення
        n_steps: Кількість кроків
        dt: Крок часу
    
    Повертає:
        Кортеж (time_points, values) з траєкторією процесу
    """
    time_points = [i * dt for i in range(n_steps + 1)]
    values = [x0]
    
    for i in range(n_steps):
        # dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        drift = theta * (mu - values[-1]) * dt
        diffusion = sigma * np.random.normal(0, math.sqrt(dt))
        values.append(values[-1] + drift + diffusion)
    
    return (time_points, values)

# Методи для розв'язання стохастичних диференціальних рівнянь
def euler_maruyama_method(drift: Callable[[float, float], float], 
                         diffusion: Callable[[float, float], float], 
                         x0: float, 
                         t0: float, 
                         t_end: float, 
                         n_steps: int) -> Tuple[List[float], List[float]]:
    """
    Метод Ейлера-Маруйами для стохастичних диференціальних рівнянь.
    
    dX_t = drift(t, X_t) * dt + diffusion(t, X_t) * dW_t
    
    Параметри:
        drift: Функція дрейфу
        diffusion: Функція дифузії
        x0: Початкове значення
        t0: Початковий час
        t_end: Кінцевий час
        n_steps: Кількість кроків
    
    Повертає:
        Кортеж (time_points, values) з розв'язком
    """
    dt = (t_end - t0) / n_steps
    time_points = [t0 + i * dt for i in range(n_steps + 1)]
    values = [x0]
    
    for i in range(n_steps):
        t = time_points[i]
        x = values[i]
        
        # Детермінована частина
        drift_term = drift(t, x) * dt
        
        # Стохастична частина
        diffusion_term = diffusion(t, x) * np.random.normal(0, math.sqrt(dt))
        
        values.append(x + drift_term + diffusion_term)
    
    return (time_points, values)

# Методи для роботи з фракталами
def mandelbrot_set(max_iterations: int = 100, 
                  x_range: Tuple[float, float] = (-2, 1), 
                  y_range: Tuple[float, float] = (-1.5, 1.5), 
                  resolution: int = 500) -> List[List[int]]:
    """
    Генерація множини Мандельброта.
    
    Параметри:
        max_iterations: Максимальна кількість ітерацій
        x_range: Діапазон по x
        y_range: Діапазон по y
        resolution: Роздільна здатність
    
    Повертає:
        Матриця з кількістю ітерацій для кожної точки
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Створення сітки точок
    x_points = np.linspace(x_min, x_max, resolution)
    y_points = np.linspace(y_min, y_max, resolution)
    
    # Матриця результатів
    result = []
    
    for y in y_points:
        row = []
        for x in x_points:
            # Комплексне число c = x + iy
            c = complex(x, y)
            z = 0j
            
            # Ітерації z_{n+1} = z_n^2 + c
            for i in range(max_iterations):
                z = z*z + c
                if abs(z) > 2:
                    row.append(i)
                    break
            else:
                row.append(max_iterations)
        
        result.append(row)
    
    return result

def julia_set(c: complex, 
             max_iterations: int = 100, 
             x_range: Tuple[float, float] = (-2, 2), 
             y_range: Tuple[float, float] = (-2, 2), 
             resolution: int = 500) -> List[List[int]]:
    """
    Генерація множини Жюліа.
    
    Параметри:
        c: Комплексний параметр
        max_iterations: Максимальна кількість ітерацій
        x_range: Діапазон по x
        y_range: Діапазон по y
        resolution: Роздільна здатність
    
    Повертає:
        Матриця з кількістю ітерацій для кожної точки
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Створення сітки точок
    x_points = np.linspace(x_min, x_max, resolution)
    y_points = np.linspace(y_min, y_max, resolution)
    
    # Матриця результатів
    result = []
    
    for y in y_points:
        row = []
        for x in x_points:
            # Початкове значення z = x + iy
            z = complex(x, y)
            
            # Ітерації z_{n+1} = z_n^2 + c
            for i in range(max_iterations):
                z = z*z + c
                if abs(z) > 2:
                    row.append(i)
                    break
            else:
                row.append(max_iterations)
        
        result.append(row)
    
    return result

# Методи для роботи з хвильовими процесами
def wave_equation_numerical(initial_displacement: List[float], 
                           initial_velocity: List[float], 
                           c: float, 
                           dx: float, 
                           dt: float, 
                           n_steps: int) -> List[List[float]]:
    """
    Чисельне розв'язання хвильового рівняння методом скінченних різниць.
    
    ∂²u/∂t² = c² * ∂²u/∂x²
    
    Параметри:
        initial_displacement: Початкове зміщення
        initial_velocity: Початкова швидкість
        c: Швидкість хвилі
        dx: Крок по x
        dt: Крок по t
        n_steps: Кількість часових кроків
    
    Повертає:
        Список матриць з розв'язками для кожного кроку
    """
    nx = len(initial_displacement)
    
    # Ініціалізація розв'язків
    u_prev = initial_displacement.copy()
    
    # Перший крок з використанням початкової швидкості
    u_curr = [0.0] * nx
    for i in range(1, nx - 1):
        u_curr[i] = u_prev[i] + dt * initial_velocity[i] + \
                   0.5 * (c*dt/dx)**2 * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
    
    # Граничні умови (фіксовані кінці)
    u_curr[0] = 0.0
    u_curr[-1] = 0.0
    
    solutions = [u_prev, u_curr]
    
    # Основний цикл
    for step in range(2, n_steps + 1):
        u_next = [0.0] * nx
        
        # Внутрішні точки
        for i in range(1, nx - 1):
            u_next[i] = 2*u_curr[i] - u_prev[i] + \
                       (c*dt/dx)**2 * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
        
        # Граничні умови
        u_next[0] = 0.0
        u_next[-1] = 0.0
        
        solutions.append(u_next)
        u_prev = u_curr
        u_curr = u_next
    
    return solutions

# Методи для роботи з дифузійними процесами
def heat_equation_numerical(initial_temperature: List[float], 
                           alpha: float, 
                           dx: float, 
                           dt: float, 
                           n_steps: int) -> List[List[float]]:
    """
    Чисельне розв'язання рівняння теплопровідності методом скінченних різниць.
    
    ∂u/∂t = α * ∂²u/∂x²
    
    Параметри:
        initial_temperature: Початковий розподіл температури
        alpha: Коефіцієнт теплопровідності
        dx: Крок по x
        dt: Крок по t
        n_steps: Кількість часових кроків
    
    Повертає:
        Список матриць з розв'язками для кожного кроку
    """
    nx = len(initial_temperature)
    
    # Перевірка стійкості (умова Куранта-Фрідріхса-Леві)
