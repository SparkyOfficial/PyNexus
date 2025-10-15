"""
Модуль для обчислювальної науки в PyNexus.
Містить функції для чисельного моделювання, обчислювальних експериментів та наукових симуляцій.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from collections import Counter, defaultdict
import numpy as np

# Константи для обчислювальної науки
PLANCK_CONSTANT = 6.62607015e-34  # Постійна Планка (Дж·с)
SPEED_OF_LIGHT = 299792458  # Швидкість світла у вакуумі (м/с)
BOLTZMANN_CONSTANT = 1.380649e-23  # Постійна Больцмана (Дж/К)
AVOGADRO_CONSTANT = 6.02214076e23  # Число Авогадро (1/моль)
GRAVITATIONAL_CONSTANT = 6.67430e-11  # Гравітаційна стала (м³/(кг·с²))
ELEMENTARY_CHARGE = 1.602176634e-19  # Елементарний заряд (Кл)

def numerical_methods_error_analysis(true_values: List[float], 
                                   approximated_values: List[float]) -> Dict[str, float]:
    """
    Аналіз похибок чисельних методів.
    
    Параметри:
        true_values: Точні значення
        approximated_values: Наближені значення
    
    Повертає:
        Словник з метриками похибок
    """
    if len(true_values) != len(approximated_values):
        raise ValueError("Списки повинні мати однакову довжину")
    
    if not true_values:
        return {
            'absolute_error': 0.0,
            'relative_error': 0.0,
            'mean_squared_error': 0.0,
            'root_mean_square_error': 0.0
        }
    
    n = len(true_values)
    
    # Абсолютна похибка
    absolute_errors = [abs(true_val - approx_val) for true_val, approx_val in zip(true_values, approximated_values)]
    absolute_error = sum(absolute_errors) / n
    
    # Відносна похибка
    relative_errors = []
    for true_val, approx_val in zip(true_values, approximated_values):
        if true_val != 0:
            relative_errors.append(abs((true_val - approx_val) / true_val))
        else:
            relative_errors.append(0.0 if approx_val == 0 else float('inf'))
    
    # Фільтруємо нескінченні значення
    finite_relative_errors = [err for err in relative_errors if not math.isinf(err)]
    relative_error = sum(finite_relative_errors) / len(finite_relative_errors) if finite_relative_errors else 0.0
    
    # Середньоквадратична похибка
    squared_errors = [(true_val - approx_val) ** 2 for true_val, approx_val in zip(true_values, approximated_values)]
    mean_squared_error = sum(squared_errors) / n
    
    # Корінь з середньоквадратичної похибки
    root_mean_square_error = math.sqrt(mean_squared_error)
    
    return {
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'mean_squared_error': mean_squared_error,
        'root_mean_square_error': root_mean_square_error
    }

def monte_carlo_integration(func: Callable[[List[float]], float], 
                          bounds: List[Tuple[float, float]], 
                          num_samples: int = 1000000) -> Tuple[float, float]:
    """
    Інтегрування методом Монте-Карло.
    
    Параметри:
        func: Функція для інтегрування
        bounds: Межі інтегрування [(min1, max1), (min2, max2), ...]
        num_samples: Кількість випадкових точок
    
    Повертає:
        Кортеж (значення інтегралу, оцінка похибки)
    """
    if not bounds:
        raise ValueError("Межі інтегрування не можуть бути порожніми")
    
    if num_samples <= 0:
        raise ValueError("Кількість точок повинна бути додатньою")
    
    dimensions = len(bounds)
    
    # Обчислюємо об'єм області інтегрування
    volume = 1.0
    for min_bound, max_bound in bounds:
        volume *= (max_bound - min_bound)
    
    # Генеруємо випадкові точки та обчислюємо значення функції
    function_values = []
    for _ in range(num_samples):
        # Генеруємо точку в області інтегрування
        point = [random.uniform(min_bound, max_bound) for min_bound, max_bound in bounds]
        function_values.append(func(point))
    
    # Обчислюємо інтеграл
    integral_estimate = volume * sum(function_values) / num_samples
    
    # Оцінка похибки (стандартне відхилення)
    mean_value = sum(function_values) / num_samples
    variance = sum((val - mean_value) ** 2 for val in function_values) / (num_samples - 1)
    error_estimate = volume * math.sqrt(variance / num_samples)
    
    return integral_estimate, error_estimate

def finite_element_method_stiffness_matrix(nodes: List[Tuple[float, float]], 
                                         elements: List[List[int]],
                                         material_properties: Dict[str, float]) -> List[List[float]]:
    """
    Метод скінченних елементів: матриця жорсткості.
    
    Параметри:
        nodes: Список вузлів [(x, y), ...]
        elements: Список елементів [[вузол1, вузол2, вузол3], ...]
        material_properties: Властивості матеріалу
    
    Повертає:
        Матриця жорсткості
    """
    if not nodes or not elements:
        return []
    
    num_nodes = len(nodes)
    
    # Ініціалізуємо матрицю жорсткості
    stiffness_matrix = [[0.0 for _ in range(2 * num_nodes)] for _ in range(2 * num_nodes)]
    
    # Модуль Юнга та коефіцієнт Пуассона
    young_modulus = material_properties.get('young_modulus', 200e9)  # За замовчуванням сталь
    poisson_ratio = material_properties.get('poisson_ratio', 0.3)
    
    # Матриця пружності
    d_matrix = [
        [1, poisson_ratio, 0],
        [poisson_ratio, 1, 0],
        [0, 0, (1 - poisson_ratio) / 2]
    ]
    
    # Множник для матриці пружності
    multiplier = young_modulus / (1 - poisson_ratio ** 2)
    for i in range(3):
        for j in range(3):
            d_matrix[i][j] *= multiplier
    
    # Обчислюємо внесок кожного елемента
    for element in elements:
        if len(element) != 3:  # Трикутний елемент
            continue
            
        # Координати вузлів елемента
        x1, y1 = nodes[element[0]]
        x2, y2 = nodes[element[1]]
        x3, y3 = nodes[element[2]]
        
        # Площа трикутника
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        if area == 0:
            continue
        
        # Похідні формових функцій
        b_matrix = [
            [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
            [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
            [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
        ]
        
        # Нормалізуємо матрицю B
        for i in range(3):
            for j in range(6):
                b_matrix[i][j] /= (2 * area)
        
        # Локальна матриця жорсткості
        local_stiffness = [[0.0 for _ in range(6)] for _ in range(6)]
        
        # K_local = B^T * D * B * area
        for i in range(6):
            for j in range(6):
                for k in range(3):
                    for l in range(3):
                        local_stiffness[i][j] += b_matrix[k][i] * d_matrix[k][l] * b_matrix[l][j]
                local_stiffness[i][j] *= area
        
        # Додаємо локальну матрицю до глобальної
        for i in range(3):
            for j in range(3):
                global_i = 2 * element[i]
                global_j = 2 * element[j]
                
                # Додаємо внесок для x-компоненти
                stiffness_matrix[global_i][global_j] += local_stiffness[2*i][2*j]
                stiffness_matrix[global_i][global_j + 1] += local_stiffness[2*i][2*j + 1]
                stiffness_matrix[global_i + 1][global_j] += local_stiffness[2*i + 1][2*j]
                stiffness_matrix[global_i + 1][global_j + 1] += local_stiffness[2*i + 1][2*j + 1]
    
    return stiffness_matrix

def computational_fluid_dynamics_navier_stokes(velocity_field: List[List[float]], 
                                             pressure_field: List[float],
                                             viscosity: float,
                                             density: float,
                                             dt: float,
                                             dx: float) -> Tuple[List[List[float]], List[float]]:
    """
    Чисельне розв'язання рівнянь Нав'є-Стокса.
    
    Параметри:
        velocity_field: Поле швидкості [[u, v], ...]
        pressure_field: Поле тиску
        viscosity: Кінематична в'язкість
        density: Густина
        dt: Крок часу
        dx: Просторовий крок
    
    Повертає:
        Кортеж (нове поле швидкості, нове поле тиску)
    """
    if not velocity_field or not pressure_field:
        return velocity_field, pressure_field
    
    if len(velocity_field) != len(pressure_field):
        raise ValueError("Поля швидкості та тиску повинні мати однакову довжину")
    
    if dt <= 0 or dx <= 0:
        raise ValueError("Кроки часу та простору повинні бути додатніми")
    
    n_points = len(velocity_field)
    new_velocity = [vel[:] for vel in velocity_field]  # Копія
    new_pressure = pressure_field[:]  # Копія
    
    # Параметри для чисельного розв'язку
    reynolds_number = density * dx * max(max(abs(u), abs(v)) for u, v in velocity_field) / viscosity if viscosity > 0 else float('inf')
    
    # Ітераційне оновлення полів
    for i in range(n_points):
        u, v = velocity_field[i]
        
        # Похідні для рівняння Нав'є-Стокса
        # du/dt + u*du/dx + v*du/dy = -dp/dx + ν*(d²u/dx² + d²u/dy²)
        # dv/dt + u*dv/dx + v*dv/dy = -dp/dy + ν*(d²v/dx² + d²v/dy²)
        
        # Апроксимація похідних (центральні різниці)
        if 0 < i < n_points - 1:
            # Похідні по x
            dudx = (velocity_field[i+1][0] - velocity_field[i-1][0]) / (2 * dx)
            dvdx = (velocity_field[i+1][1] - velocity_field[i-1][1]) / (2 * dx)
            dpdx = (pressure_field[i+1] - pressure_field[i-1]) / (2 * dx)
            
            # Похідні по y (припускаємо одновимірний випадок для спрощення)
            dudy = 0.0
            dvdy = 0.0
            dpdy = 0.0
            
            # Другі похідні
            d2udx2 = (velocity_field[i+1][0] - 2*u + velocity_field[i-1][0]) / (dx ** 2)
            d2udy2 = 0.0  # Спрощення
            d2vdx2 = (velocity_field[i+1][1] - 2*v + velocity_field[i-1][1]) / (dx ** 2)
            d2vdy2 = 0.0  # Спрощення
            
            # Оновлення швидкості
            du_dt = -u * dudx - v * dudy - dpdx / density + viscosity * (d2udx2 + d2udy2)
            dv_dt = -u * dvdx - v * dvdy - dpdy / density + viscosity * (d2vdx2 + d2vdy2)
            
            new_velocity[i][0] = u + du_dt * dt
            new_velocity[i][1] = v + dv_dt * dt
            
            # Оновлення тиску (рівняння неперервності)
            div_u = dudx + dvdy
            new_pressure[i] = pressure_field[i] - dt * density * div_u
    
    return new_velocity, new_pressure

def molecular_dynamics_simulation(particles: List[Dict[str, Any]], 
                                time_steps: int,
                                dt: float,
                                box_size: float) -> List[Dict[str, Any]]:
    """
    Молекулярна динаміка.
    
    Параметри:
        particles: Список частинок з атрибутами
        time_steps: Кількість часових кроків
        dt: Крок часу
        box_size: Розмір симуляційної коробки
    
    Повертає:
        Список станів частинок
    """
    if not particles or time_steps <= 0 or dt <= 0:
        return particles
    
    # Копія частинок для симуляції
    sim_particles = [particle.copy() for particle in particles]
    
    # Історія станів
    trajectory = []
    
    for step in range(time_steps):
        # Обчислюємо сили для кожної частинки
        for i, particle in enumerate(sim_particles):
            fx, fy, fz = 0.0, 0.0, 0.0
            
            # Взаємодія з іншими частинками
            for j, other in enumerate(sim_particles):
                if i != j:
                    # Відстань між частинками
                    dx = particle['x'] - other['x']
                    dy = particle['y'] - other['y']
                    dz = particle['z'] - other['z']
                    
                    # Періодичні граничні умови
                    dx = dx - box_size * round(dx / box_size)
                    dy = dy - box_size * round(dy / box_size)
                    dz = dz - box_size * round(dz / box_size)
                    
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if distance > 0 and distance < 2.5:  # Потенціал Леннард-Джонса
                        # Потенціал Леннард-Джонса
                        sigma = 1.0  # Параметр
                        epsilon = 1.0  # Параметр
                        
                        # Похідна потенціалу (сила)
                        force_mag = 48 * epsilon * ((sigma**12)/(distance**13) - (sigma**6)/(distance**7))
                        
                        # Компоненти сили
                        fx += force_mag * dx / distance
                        fy += force_mag * dy / distance
                        fz += force_mag * dz / distance
            
            # Оновлюємо прискорення
            mass = particle.get('mass', 1.0)
            particle['ax'] = fx / mass
            particle['ay'] = fy / mass
            particle['az'] = fz / mass
        
        # Оновлюємо швидкість та положення
        for particle in sim_particles:
            # Оновлення швидкості
            particle['vx'] += particle['ax'] * dt
            particle['vy'] += particle['ay'] * dt
            particle['vz'] += particle['az'] * dt
            
            # Оновлення положення
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['z'] += particle['vz'] * dt
            
            # Періодичні граничні умови
            particle['x'] = particle['x'] % box_size
            particle['y'] = particle['y'] % box_size
            particle['z'] = particle['z'] % box_size
        
        # Зберігаємо стан на цьому кроці
        if step % max(1, time_steps // 100) == 0:  # Зберігаємо кожен 1% кроків
            trajectory.append([particle.copy() for particle in sim_particles])
    
    return trajectory

def quantum_mechanics_schrodinger_solver(potential: List[float], 
                                       mass: float,
                                       dx: float,
                                       energy_guess: float) -> Dict[str, Union[List[float], float]]:
    """
    Чисельний розв'язок рівняння Шредінгера.
    
    Параметри:
        potential: Потенціал V(x)
        mass: Маса частинки
        dx: Просторовий крок
        energy_guess: Припущення для енергії
    
    Повертає:
        Словник з хвильовою функцією та власною енергією
    """
    if not potential or dx <= 0:
        return {'wave_function': [], 'energy': 0.0, 'norm': 0.0}
    
    n_points = len(potential)
    
    # Параметри
    hbar = PLANCK_CONSTANT / (2 * math.pi)
    # Коефіцієнт у рівнянні Шредінгера
    coefficient = hbar ** 2 / (2 * mass * dx ** 2)
    
    # Метод стрільби для знаходження власних значень
    # Початкові умови
    psi = [0.0] * n_points
    psi[0] = 0.0
    psi[1] = 0.001  # Невелике ненульове значення
    
    # Інтегруємо рівняння Шредінгера
    for i in range(1, n_points - 1):
        # Рівняння Шредінгера: d²ψ/dx² = (2m/ħ²)(V(x) - E)ψ
        second_derivative = (2 * mass / hbar ** 2) * (potential[i] - energy_guess) * psi[i]
        
        # Апроксимація другої похідної: ψ[i+1] = 2ψ[i] - ψ[i-1] + dx² * d²ψ/dx²
        psi[i + 1] = 2 * psi[i] - psi[i - 1] + dx ** 2 * second_derivative
    
    # Нормалізація хвильової функції
    norm_squared = sum(p ** 2 for p in psi) * dx
    norm = math.sqrt(norm_squared) if norm_squared > 0 else 1.0
    
    if norm > 0:
        normalized_psi = [p / norm for p in psi]
    else:
        normalized_psi = psi[:]
    
    # Обчислюємо точну енергію з умови нормалізації
    # Спрощений підхід: використовуємо припущення для енергії
    energy = energy_guess
    
    return {
        'wave_function': normalized_psi,
        'energy': energy,
        'norm': norm
    }

def computational_thermodynamics(partition_function: Callable[[float], float], 
                               temperature: float) -> Dict[str, float]:
    """
    Обчислювальна термодинаміка.
    
    Параметри:
        partition_function: Функція розподілу Z(T)
        temperature: Температура
    
    Повертає:
        Словник з термодинамічними величинами
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    kb = BOLTZMANN_CONSTANT
    
    # Обчислюємо функцію розподілу
    Z = partition_function(temperature)
    
    if Z <= 0:
        return {
            'free_energy': 0.0,
            'entropy': 0.0,
            'internal_energy': 0.0,
            'heat_capacity': 0.0
        }
    
    # Вільна енергія Гельмгольца: F = -kT ln(Z)
    free_energy = -kb * temperature * math.log(Z)
    
    # Ентропія: S = k(ln(Z) + T * d(ln(Z))/dT)
    # Чисельна похідна
    dT = temperature * 1e-6  # Мале зміщення
    Z_plus = partition_function(temperature + dT)
    dlnZ_dT = (math.log(Z_plus) - math.log(Z)) / dT if Z > 0 and Z_plus > 0 else 0
    
    entropy = kb * (math.log(Z) + temperature * dlnZ_dT)
    
    # Внутрішня енергія: U = -d(ln(Z))/dβ, де β = 1/(kT)
    beta = 1.0 / (kb * temperature)
    dbeta = beta * 1e-6
    Z_beta_plus = partition_function(1.0 / (kb * (beta + dbeta)))
    dlnZ_dbeta = (math.log(Z_beta_plus) - math.log(Z)) / dbeta if Z > 0 and Z_beta_plus > 0 else 0
    
    internal_energy = -dlnZ_dbeta
    
    # Теплоємність: C = dU/dT
    T_plus = temperature + 1.0  # Зміна температури на 1 К
    Z_T_plus = partition_function(T_plus)
    beta_plus = 1.0 / (kb * T_plus)
    dlnZ_dbeta_plus = (math.log(Z_T_plus) - math.log(Z)) / (beta_plus - beta) if Z > 0 and Z_T_plus > 0 and beta_plus != beta else 0
    U_plus = -dlnZ_dbeta_plus
    heat_capacity = (U_plus - internal_energy) / (T_plus - temperature) if T_plus != temperature else 0
    
    return {
        'free_energy': free_energy,
        'entropy': entropy,
        'internal_energy': internal_energy,
        'heat_capacity': heat_capacity,
        'partition_function': Z
    }

def computational_electromagnetism(maxwell_equations: Dict[str, Callable], 
                                 boundary_conditions: Dict[str, Any],
                                 time_steps: int) -> Dict[str, List[List[float]]]:
    """
    Чисельне розв'язання рівнянь Максвелла.
    
    Параметри:
        maxwell_equations: Рівняння Максвелла
        boundary_conditions: Граничні умови
        time_steps: Кількість часових кроків
    
    Повертає:
        Словник з полями E та B
    """
    # Ініціалізація полів
    electric_field = []
    magnetic_field = []
    
    # Граничні умови
    boundary_type = boundary_conditions.get('type', 'pec')  # Perfect Electric Conductor
    
    # Часова еволюція полів (спрощена FDTD схема)
    for step in range(time_steps):
        # Оновлення магнітного поля
        # ∇ × E = -∂B/∂t
        # B^{n+1/2} = B^{n-1/2} - Δt * ∇ × E^n
        
        # Оновлення електричного поля
        # ∇ × B = μ₀ε₀ * ∂E/∂t
        # E^{n+1} = E^n + Δt/(μ₀ε₀) * ∇ × B^{n+1/2}
        
        # Для спрощення припускаємо одновимірний випадок
        if step == 0:
            # Початкові умови
            electric_field.append([0.0] * 100)  # 100 точок
            magnetic_field.append([0.0] * 100)
        else:
            # Копіюємо попередній стан
            electric_field.append(electric_field[-1][:])
            magnetic_field.append(magnetic_field[-1][:])
            
            # Оновлюємо поля (спрощена модель)
            for i in range(1, 99):  # Уникаємо границь
                # Оновлення магнітного поля
                curl_E = electric_field[-2][i+1] - electric_field[-2][i-1]
                magnetic_field[-1][i] = magnetic_field[-2][i] - 0.5 * curl_E
                
                # Оновлення електричного поля
                curl_B = magnetic_field[-1][i+1] - magnetic_field[-1][i-1]
                electric_field[-1][i] = electric_field[-2][i] + 0.5 * curl_B
    
    return {
        'electric_field': electric_field,
        'magnetic_field': magnetic_field
    }

def computational_optics(wave_function: List[complex], 
                       optical_elements: List[Dict[str, Any]],
                       propagation_distance: float) -> List[complex]:
    """
    Обчислювальна оптика (дифракція, інтерференція).
    
    Параметри:
        wave_function: Хвильова функція (комплексна амплітуда)
        optical_elements: Оптичні елементи
        propagation_distance: Відстань поширення
    
    Повертає:
        Результуюча хвильова функція
    """
    if not wave_function:
        return wave_function
    
    # Копія хвильової функції
    result_wave = wave_function[:]
    
    # Поширення хвилі (рівняння Френеля)
    wavelength = 500e-9  # 500 нм (видиме світло)
    k = 2 * math.pi / wavelength  # Хвильове число
    
    n_points = len(result_wave)
    
    # Дискретне перетворення Френеля
    for i in range(n_points):
        # Обчислюємо внесок від кожної точки
        new_amplitude = 0j
        for j in range(n_points):
            # Відстань між точками
            x1 = i * 1e-6  # 1 мкм крок
            x2 = j * 1e-6
            distance = abs(x1 - x2)
            
            # Фазовий множник
            phase = k * distance - 1j * k * distance**2 / (2 * propagation_distance) if propagation_distance > 0 else 0
            
            # Внесок від точки j до точки i
            new_amplitude += result_wave[j] * cmath.exp(1j * phase) / (1j * wavelength * propagation_distance)
        
        result_wave[i] = new_amplitude
    
    # Взаємодія з оптичними елементами
    for element in optical_elements:
        element_type = element.get('type', 'none')
        element_position = element.get('position', 0)
        element_size = element.get('size', 1)
        
        if element_type == 'lens':
            # Лінза - фазовий множник
            focal_length = element.get('focal_length', 0.1)
            for i in range(max(0, element_position - element_size//2), 
                          min(n_points, element_position + element_size//2)):
                x = (i - element_position) * 1e-6
                # Квадратична фаза
                phase_shift = -k * x**2 / (2 * focal_length)
                result_wave[i] *= cmath.exp(1j * phase_shift)
        
        elif element_type == 'aperture':
            # Діафрагма - обнулення поза апертурою
            for i in range(0, max(0, element_position - element_size//2)):
                result_wave[i] = 0j
            for i in range(min(n_points, element_position + element_size//2), n_points):
                result_wave[i] = 0j
    
    return result_wave

def computational_materials_science(crystal_structure: List[Tuple[float, float, float]], 
                                  atomic_properties: Dict[str, float],
                                  temperature: float) -> Dict[str, Union[float, List[float]]]:
    """
    Обчислювальна наука про матеріали.
    
    Параметри:
        crystal_structure: Кристалічна структура [(x, y, z), ...]
        atomic_properties: Властивості атомів
        temperature: Температура
    
    Повертає:
        Словник з матеріальними властивостями
    """
    if not crystal_structure:
        return {
            'lattice_parameter': 0.0,
            'density': 0.0,
            'young_modulus': 0.0,
            'thermal_conductivity': 0.0
        }
    
    # Параметри кристалічної ґратки
    num_atoms = len(crystal_structure)
    
    # Обчислюємо параметр ґратки (спрощений підхід)
    if num_atoms > 1:
        # Середня відстань між атомами
        distances = []
        for i in range(min(10, num_atoms)):  # Обмежуємо для ефективності
            for j in range(i+1, min(20, num_atoms)):
                x1, y1, z1 = crystal_structure[i]
                x2, y2, z2 = crystal_structure[j]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                distances.append(dist)
        
        lattice_parameter = sum(distances) / len(distances) if distances else 0.0
    else:
        lattice_parameter = 0.0
    
    # Густина
    atomic_mass = atomic_properties.get('atomic_mass', 12.0)  # За замовчуванням вуглець
    volume_per_atom = lattice_parameter ** 3 if lattice_parameter > 0 else 1.0
    density = atomic_mass / (volume_per_atom * AVOGADRO_CONSTANT) if volume_per_atom > 0 else 0.0
    
    # Модуль Юнга (емпірична формула)
    atomic_radius = atomic_properties.get('atomic_radius', 0.77e-10)  # За замовчуванням вуглець
    young_modulus = 1e11 * (atomic_radius / 1e-10) ** -3  # Проста залежність
    
    # Теплопровідність (залежить від температури)
    base_thermal_conductivity = atomic_properties.get('thermal_conductivity', 100.0)
    # Зменшення теплопровідності з температурою
    thermal_conductivity = base_thermal_conductivity * math.exp(-temperature / 1000.0)
    
    return {
        'lattice_parameter': lattice_parameter,
        'density': density,
        'young_modulus': young_modulus,
        'thermal_conductivity': thermal_conductivity,
        'num_atoms': num_atoms
    }

def computational_astrophysics(n_body_system: List[Dict[str, float]], 
                             time_steps: int,
                             dt: float) -> List[List[Dict[str, float]]]:
    """
    Обчислювальна астрофізика (задача N тіл).
    
    Параметри:
        n_body_system: Система тіл з масами та координатами
        time_steps: Кількість часових кроків
        dt: Крок часу
    
    Повертає:
        Траєкторії тіл
    """
    if not n_body_system or time_steps <= 0 or dt <= 0:
        return [n_body_system] if n_body_system else []
    
    # Копія системи для симуляції
    bodies = [body.copy() for body in n_body_system]
    trajectories = [[body.copy() for body in bodies]]
    
    # Гравітаційна константа
    G = GRAVITATIONAL_CONSTANT
    
    for step in range(time_steps):
        # Обчислюємо сили для кожного тіла
        for i, body in enumerate(bodies):
            fx, fy, fz = 0.0, 0.0, 0.0
            
            # Взаємодія з іншими тілами
            for j, other in enumerate(bodies):
                if i != j:
                    # Відстань між тілами
                    dx = other['x'] - body['x']
                    dy = other['y'] - body['y']
                    dz = other['z'] - body['z']
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if distance > 0:
                        # Закон всесвітнього тяжіння
                        force_magnitude = G * body['mass'] * other['mass'] / (distance ** 2)
                        
                        # Компоненти сили
                        fx += force_magnitude * dx / distance
                        fy += force_magnitude * dy / distance
                        fz += force_magnitude * dz / distance
            
            # Оновлюємо прискорення
            body['ax'] = fx / body['mass']
            body['ay'] = fy / body['mass']
            body['az'] = fz / body['mass']
        
        # Оновлюємо швидкість та положення
        for body in bodies:
            # Оновлення швидкості
            body['vx'] += body['ax'] * dt
            body['vy'] += body['ay'] * dt
            body['vz'] += body['az'] * dt
            
            # Оновлення положення
            body['x'] += body['vx'] * dt
            body['y'] += body['vy'] * dt
            body['z'] += body['vz'] * dt
        
        # Зберігаємо стан системи
        trajectories.append([body.copy() for body in bodies])
    
    return trajectories

def computational_climate_model(temperature_field: List[float], 
                              greenhouse_gases: List[float],
                              solar_radiation: float,
                              time_steps: int) -> List[float]:
    """
    Обчислювальна кліматологія.
    
    Параметри:
        temperature_field: Температурне поле
        greenhouse_gases: Концентрація парникових газів
        solar_radiation: Сонячна радіація
        time_steps: Кількість часових кроків
    
    Повертає:
        Еволюція температурного поля
    """
    if not temperature_field or time_steps <= 0:
        return temperature_field
    
    # Копія температурного поля
    temp_field = temperature_field[:]
    evolution = [temp_field[:]]
    
    # Параметри моделі
    heat_capacity = 4.2e6  # Теплоємність океану (Дж/м³·К)
    thermal_conductivity = 0.6  # Теплопровідність (Вт/м·К)
    stefan_boltzmann = 5.67e-8  # Константа Стефана-Больцмана
    
    # Середні значення
    avg_greenhouse = sum(greenhouse_gases) / len(greenhouse_gases) if greenhouse_gases else 0.0
    
    n_points = len(temp_field)
    
    for step in range(time_steps):
        new_temp = [0.0] * n_points
        
        for i in range(n_points):
            # Тепловий потік від Сонця
            solar_input = solar_radiation * (1 - 0.3)  # Альбедо Землі ~ 0.3
            
            # Парниковий ефект
            greenhouse_effect = avg_greenhouse * 100  # Проста модель
            
            # Випромінювання в космос (закон Стефана-Больцмана)
            radiation_loss = stefan_boltzmann * temp_field[i] ** 4
            
            # Теплопровідність (обмін з сусідніми точками)
            heat_conduction = 0.0
            if i > 0:
                heat_conduction += thermal_conductivity * (temp_field[i-1] - temp_field[i])
            if i < n_points - 1:
                heat_conduction += thermal_conductivity * (temp_field[i+1] - temp_field[i])
            
            # Зміна температури
            dt_temp = (solar_input + greenhouse_effect - radiation_loss + heat_conduction) / heat_capacity
            
            # Оновлення температури
            new_temp[i] = temp_field[i] + dt_temp
            
            # Обмеження температури реалістичними межами
            new_temp[i] = max(180, min(320, new_temp[i]))  # 180-320 К
        
        temp_field = new_temp[:]
        evolution.append(temp_field[:])
    
    return temp_field

def computational_biology_model(population_dynamics: List[Dict[str, float]], 
                              interaction_matrix: List[List[float]],
                              environmental_factors: List[float]) -> List[Dict[str, float]]:
    """
    Обчислювальна біологія (динаміка популяцій).
    
    Параметри:
        population_dynamics: Параметри популяцій
        interaction_matrix: Матриця взаємодій між видами
        environmental_factors: Фактори середовища
    
    Повертає:
        Еволюція популяцій
    """
    if not population_dynamics:
        return population_dynamics
    
    # Копія популяцій
    populations = [pop.copy() for pop in population_dynamics]
    
    n_species = len(populations)
    
    # Середні фактори середовища
    avg_environment = sum(environmental_factors) / len(environmental_factors) if environmental_factors else 1.0
    
    # Модель Лотки-Вольтерра для взаємодій
    new_populations = []
    
    for i in range(n_species):
        species = populations[i]
        current_population = species.get('population', 0)
        growth_rate = species.get('growth_rate', 0.1)
        
        # Вплив взаємодій з іншими видами
        interaction_effect = 0.0
        for j in range(n_species):
            if i != j:
                interaction_effect += interaction_matrix[i][j] * populations[j].get('population', 0)
        
        # Вплив середовища
        environmental_effect = avg_environment - 1.0  # Нормалізовано навколо 1.0
        
        # Зміна популяції
        dN_dt = growth_rate * current_population * (1 - current_population / 1000.0) - interaction_effect + environmental_effect * current_population
        
        # Оновлення популяції
        new_population = max(0, current_population + dN_dt)
        
        new_species = species.copy()
        new_species['population'] = new_population
        new_populations.append(new_species)
    
    return new_populations

def computational_chemistry_model(molecular_structure: List[Dict[str, Any]], 
                                reaction_rates: List[float],
                                temperature: float) -> Dict[str, Union[float, List[float]]]:
    """
    Обчислювальна хімія.
    
    Параметри:
        molecular_structure: Структура молекул
        reaction_rates: Швидкості реакцій
        temperature: Температура
    
    Повертає:
        Словник з хімічними властивостями
    """
    if not molecular_structure:
        return {
            'reaction_rate': 0.0,
            'activation_energy': 0.0,
            'equilibrium_constant': 0.0,
            'concentration_profile': []
        }
    
    # Кількість молекул
    num_molecules = len(molecular_structure)
    
    # Середня швидкість реакцій
    avg_reaction_rate = sum(reaction_rates) / len(reaction_rates) if reaction_rates else 0.0
    
    # Енергія активації (емпірична формула)
    # E_a = E₀ * exp(-T/T₀)
    E0 = 50000.0  # Базова енергія активації (Дж/моль)
    T0 = 300.0    # Характерна температура (К)
    activation_energy = E0 * math.exp(-temperature / T0)
    
    # Константа рівноваги (рівняння Вант-Гоффа)
    # ln(K) = -ΔH/RT + ΔS/R
    delta_H = -50000.0  # Ентальпія (Дж/моль)
    delta_S = 100.0     # Ентропія (Дж/моль·К)
    R = 8.314           # Універсальна газова стала
    
    ln_K = -delta_H / (R * temperature) + delta_S / R
    equilibrium_constant = math.exp(ln_K)
    
    # Профіль концентрації (спрощена кінетична модель)
    # dC/dt = k * C * (1 - C/C₀)
    initial_concentration = 1.0
    concentration_profile = [initial_concentration]
    
    current_concentration = initial_concentration
    for _ in range(100):  # 100 точок профілю
        dC_dt = avg_reaction_rate * current_concentration * (1 - current_concentration / initial_concentration)
        current_concentration += dC_dt * 0.01  # Маленький крок
        concentration_profile.append(max(0, current_concentration))
    
    return {
        'reaction_rate': avg_reaction_rate,
        'activation_energy': activation_energy,
        'equilibrium_constant': equilibrium_constant,
        'concentration_profile': concentration_profile,
        'num_molecules': num_molecules
    }

def computational_geophysics(seismic_data: List[float], 
                           earth_model: Dict[str, List[float]],
                           time_steps: int) -> List[float]:
    """
    Обчислювальна геофізика (сеїсмічне моделювання).
    
    Параметри:
        seismic_data: Сеїсмічні дані
        earth_model: Модель будови Землі
        time_steps: Кількість часових кроків
    
    Повертає:
        Синтетичні сеїсмограми
    """
    if not seismic_data or time_steps <= 0:
        return seismic_data
    
    # Копія сеїсмічних даних
    synthetic_seismogram = seismic_data[:]
    
    # Параметри земної моделі
    velocities = earth_model.get('velocities', [5000.0] * len(seismic_data))  # Швидкості хвиль
    densities = earth_model.get('densities', [2700.0] * len(seismic_data))    # Густини
    
    n_points = len(synthetic_seismogram)
    
    # Моделювання поширення сейсмічних хвиль
    for step in range(time_steps):
        new_seismogram = [0.0] * n_points
        
        # Фінітні різниці для хвильового рівняння
        for i in range(1, n_points - 1):
            # Хвильове рівняння: ∂²u/∂t² = v² * ∂²u/∂x²
            dt = 0.001  # Крок часу
            dx = 10.0   # Просторовий крок
            
            # Швидкість хвилі в цій точці
            velocity = velocities[i] if i < len(velocities) else 5000.0
            
            # Апроксимація другої похідної по простору
            d2u_dx2 = (synthetic_seismogram[i+1] - 2 * synthetic_seismogram[i] + synthetic_seismogram[i-1]) / (dx ** 2)
            
            # Оновлення сейсмограми
            new_seismogram[i] = 2 * synthetic_seismogram[i] - synthetic_seismogram[i] + (velocity * dt / dx) ** 2 * d2u_dx2
        
        synthetic_seismogram = new_seismogram[:]
    
    return synthetic_seismogram

def computational_neuroscience_model(neural_network: List[Dict[str, Any]], 
                                   input_stimuli: List[float],
                                   simulation_time: float) -> List[Dict[str, List[float]]]:
    """
    Обчислювальна нейронаука.
    
    Параметри:
        neural_network: Нейронна мережа
        input_stimuli: Вхідні стимули
        simulation_time: Час симуляції
    
    Повертає:
        Активність нейронів у часі
    """
    if not neural_network or not input_stimuli:
        return []
    
    # Копія нейронної мережі
    neurons = [neuron.copy() for neuron in neural_network]
    
    # Історія активності
    activity_history = []
    
    # Параметри моделі нейрона (модель Ходжкіна-Хакслі спрощена)
    dt = 0.01  # Крок часу (мс)
    steps = int(simulation_time / dt)
    
    for step in range(steps):
        # Часова мітка
        time = step * dt
        
        # Вхідний стимул (може змінюватися з часом)
        input_current = input_stimuli[min(step, len(input_stimuli) - 1)] if step < len(input_stimuli) else 0.0
        
        # Активність кожного нейрона
        step_activity = []
        
        for neuron in neurons:
            # Потенціал мембрани
            v = neuron.get('membrane_potential', -70.0)  # Спочатку спочинний потенціал
            
            # Струми
            leak_current = neuron.get('leak_conductance', 0.3) * (v - neuron.get('leak_reversal', -70.0))
            spike_current = neuron.get('spike_conductance', 0.0) * (v - neuron.get('spike_reversal', 50.0))
            
            # Зовнішній струм
            external_current = input_current * neuron.get('input_weight', 1.0)
            
            # Зміна потенціалу
            dv_dt = (-leak_current - spike_current + external_current) / neuron.get('membrane_capacitance', 1.0)
            new_v = v + dv_dt * dt
            
            # Генерація спайку
            spike_threshold = neuron.get('spike_threshold', -55.0)
            is_spiking = new_v >= spike_threshold
            
            # Скидання потенціалу після спайку
            if is_spiking:
                new_v = neuron.get('reset_potential', -70.0)
                # Збільшення провідності після спайку
                neuron['spike_conductance'] = neuron.get('spike_conductance', 0.0) + 0.1
            
            # Оновлення потенціалу
            neuron['membrane_potential'] = new_v
            
            # Зменшення спайкової провідності з часом
            neuron['spike_conductance'] = max(0, neuron.get('spike_conductance', 0.0) - 0.01)
            
            step_activity.append({
                'potential': new_v,
                'spiking': is_spiking,
                'time': time
            })
        
        activity_history.append(step_activity)
    
    return activity_history

def computational_economics_model(market_data: List[Dict[str, float]], 
                                agent_behaviors: List[str],
                                time_horizon: int) -> Dict[str, Union[float, List[float]]]:
    """
    Обчислювальна економіка (агентне моделювання ринку).
    
    Параметри:
        market_data: Ринкові дані
        agent_behaviors: Поведінка агентів
        time_horizon: Горизонт моделювання
    
    Повертає:
        Словник з прогнозами ринку
    """
    if not market_data or time_horizon <= 0:
        return {
            'price_trajectory': [],
            'volatility': 0.0,
            'market_efficiency': 0.0,
            'crisis_probability': 0.0
        }
    
    # Початкові ринкові показники
    initial_price = market_data[-1].get('price', 100.0) if market_data else 100.0
    initial_volume = market_data[-1].get('volume', 1000.0) if market_data else 1000.0
    
    # Траєкторія цін (модель випадкового блукання з тенденцією)
    price_trajectory = [initial_price]
    current_price = initial_price
    
    # Волатильність
    base_volatility = 0.02  # 2% базова волатильність
    
    # Ефективність ринку (0-1, де 1 - повна ефективність)
    market_efficiency = len(set(agent_behaviors)) / len(agent_behaviors) if agent_behaviors else 0.5
    
    # Імовірність кризи
    crisis_probability = 0.0
    
    for t in range(time_horizon):
        # Випадкова зміна ціни
        random_shock = random.normalvariate(0, base_volatility)
        
        # Тренд (може бути позитивним або негативним)
        trend = 0.0001 * (t - time_horizon / 2)  # Параболічний тренд
        
        # Вплив поведінки агентів
        behavioral_impact = 0.0
        for behavior in agent_behaviors:
            if behavior == 'rational':
                behavioral_impact += 0.0001
            elif behavior == 'herding':
                behavioral_impact += 0.0002 * math.sin(t * 0.1)  # Циклічна поведінка
            elif behavior == 'noise':
                behavioral_impact += random.uniform(-0.001, 0.001)
        
        # Зміна ціни
        price_change = current_price * (trend + random_shock + behavioral_impact)
        current_price += price_change
        
        # Обмеження ціни позитивними значеннями
        current_price = max(0.01, current_price)
        
        price_trajectory.append(current_price)
        
        # Оновлення імовірності кризи
        if current_price < initial_price * 0.8:  # Криза при падінні > 20%
            crisis_probability = min(1.0, crisis_probability + 0.05)
        elif current_price > initial_price * 1.2:  # Бульовий ринок при зростанні > 20%
            crisis_probability = max(0.0, crisis_probability - 0.02)
    
    # Обчислюємо фінальну волатильність
    if len(price_trajectory) > 1:
        returns = [math.log(price_trajectory[i] / price_trajectory[i-1]) 
                  for i in range(1, len(price_trajectory))]
        volatility = math.sqrt(sum(r**2 for r in returns) / len(returns)) if returns else 0.0
    else:
        volatility = base_volatility
    
    return {
        'price_trajectory': price_trajectory,
        'volatility': volatility,
        'market_efficiency': market_efficiency,
        'crisis_probability': crisis_probability,
        'final_price': current_price,
        'price_change': (current_price - initial_price) / initial_price * 100
    }