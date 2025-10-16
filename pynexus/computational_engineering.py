"""
Модуль для обчислювальної інженерії в PyNexus.
Включає функції для інженерних розрахунків, аналізу конструкцій, 
гідравліки, термодинаміки, електротехніки та інших інженерних дисциплін.

Автор: Андрій Будильников
"""

# Спроба імпорту бібліотек
NUMPY_AVAILABLE = False
SCIPY_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

# Умовний імпорт з обгорткою для уникнення помилок лінтера
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Створення заглушок для numpy
    class NumpyStub:
        def __getattr__(self, name):
            return None
    np = NumpyStub()

try:
    from scipy import constants, optimize, linalg, integrate
    SCIPY_AVAILABLE = True
except ImportError:
    # Створення заглушок для scipy
    class ConstantsStub:
        g = 9.81
        pi = 3.141592653589793
    
    class ScipyStub:
        def __getattr__(self, name):
            return None
    
    constants = ConstantsStub()
    optimize = ScipyStub()
    linalg = ScipyStub()
    integrate = ScipyStub()

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    # Створення заглушки для matplotlib
    class MatplotlibStub:
        def __getattr__(self, name):
            return None
    plt = MatplotlibStub()

from typing import List, Tuple, Callable, Union, Optional, Dict, Any
import math

# Допоміжні функції для роботи без numpy
def create_array(size, value=0):
    """Створення масиву без numpy"""
    if NUMPY_AVAILABLE:
        try:
            return np.full(size, value)
        except:
            return [value] * size
    return [value] * size

def create_zeros_array(size):
    """Створення масиву з нулями без numpy"""
    if NUMPY_AVAILABLE:
        try:
            return np.zeros(size)
        except:
            return [0.0] * size
    return [0.0] * size

def create_2d_array(rows, cols, value=0):
    """Створення 2D масиву без numpy"""
    if NUMPY_AVAILABLE:
        try:
            return np.full((rows, cols), value)
        except:
            return [[value for _ in range(cols)] for _ in range(rows)]
    return [[value for _ in range(cols)] for _ in range(rows)]

def create_zeros_2d_array(rows, cols):
    """Створення 2D масиву з нулями без numpy"""
    if NUMPY_AVAILABLE:
        try:
            return np.zeros((rows, cols))
        except:
            return [[0.0 for _ in range(cols)] for _ in range(rows)]
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def my_linspace(start, stop, num):
    """Створення рівномірно розподілених значень без numpy"""
    if NUMPY_AVAILABLE:
        try:
            return np.linspace(start, stop, num)
        except:
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1) if num > 1 else 0
            return [start + i * step for i in range(num)]
    
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1) if num > 1 else 0
    return [start + i * step for i in range(num)]

def safe_get_item(arr, index, default=0):
    """Безпечне отримання елемента масиву"""
    try:
        if isinstance(arr, list) and 0 <= index < len(arr):
            return arr[index]
        elif hasattr(arr, '__getitem__') and hasattr(arr, '__len__'):
            # Для numpy масивів та інших послідовностей
            if 0 <= index < len(arr):
                return arr[index]
        return default
    except:
        return default

def array_dot(a, b):
    """Скалярний добуток векторів"""
    if NUMPY_AVAILABLE:
        try:
            return np.dot(a, b)
        except:
            return sum(safe_get_item(a, i, 0) * safe_get_item(b, i, 0) for i in range(min(len(a) if hasattr(a, '__len__') else 0, len(b) if hasattr(b, '__len__') else 0)))
    
    return sum(safe_get_item(a, i, 0) * safe_get_item(b, i, 0) for i in range(min(len(a) if hasattr(a, '__len__') else 0, len(b) if hasattr(b, '__len__') else 0)))

def my_solve(a, b):
    """Розв'язання системи лінійних рівнянь"""
    if SCIPY_AVAILABLE and linalg is not None:
        try:
            return linalg.solve(a, b)
        except:
            pass
    
    # Проста реалізація методом Гаусса для невеликих систем 2x2
    if isinstance(a, list) and len(a) == 2 and isinstance(a[0], list) and len(a[0]) == 2:
        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        if det != 0:
            x = (b[0] * a[1][1] - b[1] * a[0][1]) / det
            y = (b[1] * a[0][0] - b[0] * a[1][0]) / det
            return [x, y]
    
    # Заглушка для інших випадків
    return [0] * (len(b) if hasattr(b, '__len__') else 0)

def my_array(data):
    """Створення масиву"""
    if NUMPY_AVAILABLE:
        try:
            return np.array(data)
        except:
            return data
    return data

def my_pinv(matrix):
    """Псевдообернення матриці"""
    if NUMPY_AVAILABLE:
        try:
            return np.linalg.pinv(matrix)
        except:
            return matrix
    return matrix

def my_dot(a, b):
    """Добуток матриць"""
    if NUMPY_AVAILABLE:
        try:
            return np.dot(a, b)
        except:
            pass
    
    # Проста реалізація для 2D випадку
    if (isinstance(a, list) and len(a) > 0 and isinstance(a[0], list) and 
        isinstance(b, list) and len(b) > 0 and isinstance(b[0], list)):
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                val = sum(safe_get_item(safe_get_item(a, i, []), k, 0) * safe_get_item(safe_get_item(b, k, []), j, 0) for k in range(len(b)))
                row.append(val)
            result.append(row)
        return result
    
    # Для 1D випадку
    return sum(safe_get_item(a, i, 0) * safe_get_item(b, i, 0) for i in range(min(len(a) if hasattr(a, '__len__') else 0, len(b) if hasattr(b, '__len__') else 0)))

# Статика та опір матеріалів
def beam_deflection_analysis(beam_length: float, 
                           load_distribution: Callable[[float], float], 
                           young_modulus: float, 
                           moment_of_inertia: float, 
                           boundary_conditions: str = "simply_supported") -> Dict[str, Any]:
    """
    Аналіз прогину балки.
    
    Параметри:
        beam_length: Довжина балки
        load_distribution: Функція розподілу навантаження q(x)
        young_modulus: Модуль Юнга
        moment_of_inertia: Момент інерції перерізу
        boundary_conditions: Граничні умови ("simply_supported", "cantilever", "fixed")
    
    Повертає:
        Словник з результатами аналізу
    """
    # Жорсткість на згин
    flexural_rigidity = young_modulus * moment_of_inertia
    
    # Диференціальне рівняння прогину: E*I * d⁴y/dx⁴ = q(x)
    # Інтегрування для отримання згинальних моментів, поперечних сил та прогинів
    
    # Спрощений чисельний підхід методом скінченних різниць
    n_points = 100
    x_points = my_linspace(0, beam_length, n_points)
    dx = beam_length / (n_points - 1)
    
    # Навантаження в точках
    loads = [load_distribution(x) for x in x_points]
    
    # Матриця жорсткості для прогину (спрощена)
    stiffness_matrix = create_zeros_2d_array(n_points, n_points)
    
    # Внутрішні точки (центральні різниці для d⁴y/dx⁴)
    for i in range(2, n_points - 2):
        stiffness_matrix[i][i-2] = 1
        stiffness_matrix[i][i-1] = -4
        stiffness_matrix[i][i] = 6
        stiffness_matrix[i][i+1] = -4
        stiffness_matrix[i][i+2] = 1
    
    # Граничні умови
    if boundary_conditions == "simply_supported":
        # y(0) = y(L) = 0, M(0) = M(L) = 0
        stiffness_matrix[0][0] = 1
        stiffness_matrix[1][1] = 1
        stiffness_matrix[-1][-1] = 1
        stiffness_matrix[-2][-2] = 1
        loads[0] = 0
        loads[1] = 0
        loads[-1] = 0
        loads[-2] = 0
    elif boundary_conditions == "cantilever":
        # y(0) = 0, dy/dx(0) = 0, M(L) = 0, V(L) = 0
        stiffness_matrix[0][0] = 1
        stiffness_matrix[1][0] = -1
        stiffness_matrix[1][1] = 1
        stiffness_matrix[-1][-1] = 1
        stiffness_matrix[-2][-2] = 2
        stiffness_matrix[-2][-1] = -1
        loads[0] = 0
        loads[1] = 0
        loads[-1] = 0
        loads[-2] = 0
    elif boundary_conditions == "fixed":
        # y(0) = y(L) = 0, dy/dx(0) = dy/dx(L) = 0
        stiffness_matrix[0][0] = 1
        stiffness_matrix[1][0] = -1
        stiffness_matrix[1][1] = 1
        stiffness_matrix[-1][-1] = 1
        stiffness_matrix[-2][-1] = -1
        stiffness_matrix[-2][-2] = 1
        loads[0] = 0
        loads[1] = 0
        loads[-1] = 0
        loads[-2] = 0
    
    # Розв'язання системи
    try:
        loads_array = my_array(loads)
        adjusted_loads = [load_val * dx**4 / flexural_rigidity for load_val in loads_array]
        deflections = my_solve(stiffness_matrix, adjusted_loads)
    except:
        # Якщо матриця сингулярна, використовуємо псевдообернення
        loads_array = my_array(loads)
        adjusted_loads = [load_val * dx**4 / flexural_rigidity for load_val in loads_array]
        stiffness_pinv = my_pinv(stiffness_matrix)
        deflections = my_dot(stiffness_pinv, adjusted_loads)
    
    # Обчислення згинальних моментів
    # M(x) = -E*I * d²y/dx²
    bending_moments = create_zeros_array(n_points)
    for i in range(1, n_points - 1):
        # Безпечне отримання значень з масиву
        prev_deflection = safe_get_item(deflections, i-1, 0)
        curr_deflection = safe_get_item(deflections, i, 0)
        next_deflection = safe_get_item(deflections, i+1, 0)
        d2y_dx2 = (prev_deflection - 2*curr_deflection + next_deflection) / dx**2
        bending_moments[i] = -flexural_rigidity * d2y_dx2
    
    # Обчислення поперечних сил
    # V(x) = -E*I * d³y/dx³
    shear_forces = create_zeros_array(n_points)
    dx3 = 2*dx**3
    for i in range(2, n_points - 2):
        # Безпечне отримання значень з масиву
        deflection_i_minus_2 = safe_get_item(deflections, i-2, 0)
        deflection_i_minus_1 = safe_get_item(deflections, i-1, 0)
        deflection_i_plus_1 = safe_get_item(deflections, i+1, 0)
        deflection_i_plus_2 = safe_get_item(deflections, i+2, 0)
        d3y_dx3 = (deflection_i_minus_2 - 2*deflection_i_minus_1 + 2*deflection_i_plus_1 - deflection_i_plus_2) / dx3
        shear_forces[i] = -flexural_rigidity * d3y_dx3
    
    # Максимальний прогин
    max_deflection = max(abs(d) for d in deflections) if hasattr(deflections, '__iter__') else 0
    max_deflection_index = 0
    if hasattr(deflections, '__iter__'):
        for i, d in enumerate(deflections):
            if abs(d) > abs(safe_get_item(deflections, max_deflection_index, 0)):
                max_deflection_index = i
    max_deflection_location = safe_get_item(x_points, max_deflection_index, 0) if hasattr(x_points, '__iter__') else 0
    
    # Максимальний згинальний момент
    max_moment = max(abs(m) for m in bending_moments) if hasattr(bending_moments, '__iter__') else 0
    max_moment_index = 0
    if hasattr(bending_moments, '__iter__'):
        for i, m in enumerate(bending_moments):
            if abs(m) > abs(safe_get_item(bending_moments, max_moment_index, 0)):
                max_moment_index = i
    max_moment_location = safe_get_item(x_points, max_moment_index, 0) if hasattr(x_points, '__iter__') else 0
    
    # Конвертація результатів до списків якщо потрібно
    deflections_list = list(deflections) if hasattr(deflections, '__iter__') else [0]
    bending_moments_list = list(bending_moments) if hasattr(bending_moments, '__iter__') else [0] * n_points
    shear_forces_list = list(shear_forces) if hasattr(shear_forces, '__iter__') else [0] * n_points
    x_points_list = list(x_points) if hasattr(x_points, '__iter__') else [0]
    
    return {
        'x_coordinates': x_points_list,
        'deflections': deflections_list,
        'bending_moments': bending_moments_list,
        'shear_forces': shear_forces_list,
        'max_deflection': float(max_deflection),
        'max_deflection_location': float(max_deflection_location),
        'max_bending_moment': float(max_moment),
        'max_moment_location': float(max_moment_location),
        'boundary_conditions': boundary_conditions
    }

def stress_analysis(normal_forces: List[float], 
                   shear_forces: List[float], 
                   cross_sectional_area: float, 
                   moment_of_inertia: float, 
                   distances_from_neutral_axis: List[float]) -> Dict[str, Any]:
    """
    Аналіз напружень у перерізі.
    
    Параметри:
        normal_forces: Нормальні сили в різних точках
        shear_forces: Поперечні сили в різних точках
        cross_sectional_area: Площа поперечного перерізу
        moment_of_inertia: Момент інерції
        distances_from_neutral_axis: Відстані від нейтральної осі
    
    Повертає:
        Словник з аналізом напружень
    """
    n_points = len(normal_forces)
    
    # Нормальні напруження
    normal_stresses = []
    for i in range(n_points):
        # σ = N/A + M*y/I
        # Припускаємо, що M=0 для спрощення
        stress = normal_forces[i] / cross_sectional_area
        normal_stresses.append(stress)
    
    # Дотичні напруження
    shear_stresses = []
    for i in range(n_points):
        # τ = V*Q/(I*b) - формула Журавського
        # Спрощено: τ = V/A (середнє дотичне напруження)
        stress = shear_forces[i] / cross_sectional_area
        shear_stresses.append(stress)
    
    # Головні напруження (теорія максимальних напружень)
    principal_stresses = []
    max_shear_stresses = []
    
    for i in range(n_points):
        sigma_x = normal_stresses[i]
        sigma_y = 0  # припускаємо плоский напружений стан
        tau_xy = shear_stresses[i]
        
        # Головні напруження
        sigma_1 = (sigma_x + sigma_y) / 2 + math.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        sigma_2 = (sigma_x + sigma_y) / 2 - math.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        
        principal_stresses.append([sigma_1, sigma_2])
        
        # Максимальне дотичне напруження
        max_shear = math.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        max_shear_stresses.append(max_shear)
    
    # Еквівалентне напруження (теорія енергії формозміни)
    equivalent_stresses = []
    for i in range(n_points):
        sigma_1, sigma_2 = principal_stresses[i]
        # σ_eq = sqrt(sigma_1² - sigma_1*sigma_2 + sigma_2²)
        eq_stress = math.sqrt(sigma_1**2 - sigma_1*sigma_2 + sigma_2**2)
        equivalent_stresses.append(eq_stress)
    
    return {
        'normal_stresses': normal_stresses,
        'shear_stresses': shear_stresses,
        'principal_stresses': principal_stresses,
        'max_shear_stresses': max_shear_stresses,
        'equivalent_stresses': equivalent_stresses,
        'max_normal_stress': max(normal_stresses),
        'max_shear_stress': max(max_shear_stresses),
        'max_equivalent_stress': max(equivalent_stresses)
    }

# Гідравліка та гідрологія
def fluid_flow_analysis(pipe_diameter: float, 
                       pipe_length: float, 
                       fluid_density: float, 
                       fluid_viscosity: float, 
                       pressure_drop: float, 
                       pipe_roughness: float = 0.0) -> Dict[str, Any]:
    """
    Аналіз течії рідини в трубопроводі.
    
    Параметри:
        pipe_diameter: Діаметр труби
        pipe_length: Довжина труби
        fluid_density: Густина рідини
        fluid_viscosity: В'язкість рідини
        pressure_drop: Перепад тиску
        pipe_roughness: Шорсткість труби
    
    Повертає:
        Словник з результатами аналізу течії
    """
    # Площа поперечного перерізу
    cross_area = math.pi * (pipe_diameter / 2)**2
    
    # Визначення режиму течії (число Рейнольдса)
    def reynolds_number(velocity):
        return fluid_density * velocity * pipe_diameter / fluid_viscosity
    
    # Коефіцієнт гідравлічного опору (формула Колбрука-Уайта)
    def friction_factor(reynolds, relative_roughness):
        if reynolds < 2300:
            # Ламінарна течія
            return 64 / reynolds
        else:
            # Турбулентна течія
            # Приблизне рішення
            relative_rough = relative_roughness / pipe_diameter if pipe_diameter > 0 else 0
            if relative_rough > 0:
                # Формула Хааланда
                return (-1.8 * math.log10((relative_rough/3.7)**1.11 + 6.9/reynolds))**(-2)
            else:
                # Гладка труба
                return 0.0055 * (1 + (2*10**4 * relative_rough + 10**6/reynolds)**(1/3))
    
    # Рівняння Дарсі-Вейсбаха: Δp = f * (L/D) * (ρ*v²/2)
    def pressure_drop_equation(velocity):
        re = reynolds_number(velocity)
        f = friction_factor(re, pipe_roughness)
        return f * (pipe_length / pipe_diameter) * (fluid_density * velocity**2 / 2)
    
    # Знаходження швидкості течії
    try:
        def objective(velocity):
            return abs(pressure_drop_equation(velocity) - pressure_drop)
        
        velocity_result = optimize.minimize_scalar(objective, bounds=(0, 100), method='bounded')
        flow_velocity = velocity_result.x
    except:
        # Спрощене рішення для ламінарної течії
        flow_velocity = math.sqrt(pressure_drop * pipe_diameter**4 / (128 * fluid_viscosity * pipe_length))
    
    # Число Рейнольдса
    reynolds = reynolds_number(flow_velocity)
    
    # Режим течії
    if reynolds < 2300:
        flow_regime = "laminar"
    elif reynolds < 4000:
        flow_regime = "transitional"
    else:
        flow_regime = "turbulent"
    
    # Витрата
    flow_rate = cross_area * flow_velocity
    
    # Напірна втрата
    head_loss = pressure_drop / (fluid_density * constants.g)
    
    # Коефіцієнт гідравлічного опору
    friction_coeff = friction_factor(reynolds, pipe_roughness)
    
    return {
        'flow_velocity': flow_velocity,
        'flow_rate': flow_rate,
        'reynolds_number': reynolds,
        'flow_regime': flow_regime,
        'head_loss': head_loss,
        'friction_coefficient': friction_coeff,
        'cross_sectional_area': cross_area,
        'pressure_drop': pressure_drop
    }

def open_channel_flow(channel_width: float, 
                     channel_depth: float, 
                     channel_slope: float, 
                     manning_roughness: float, 
                     flow_rate: float) -> Dict[str, Any]:
    """
    Аналіз течії в відкритих руслах (формула Маннінга).
    
    Параметри:
        channel_width: Ширина каналу
        channel_depth: Глибина потоку
        channel_slope: Ухил дна
        manning_roughness: Коефіцієнт шорсткості Маннінга
        flow_rate: Витрата
    
    Повертає:
        Словник з результатами аналізу
    """
    # Геометричні характеристики
    area = channel_width * channel_depth  # площа живого перерізу
    wetted_perimeter = channel_width + 2 * channel_depth  # змочений периметр
    hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0  # гідравлічний радіус
    
    # Формула Маннінга: Q = (1/n) * A * R^(2/3) * S^(1/2)
    def manning_equation(depth):
        a = channel_width * depth
        p = channel_width + 2 * depth
        r = a / p if p > 0 else 0
        return (1/manning_roughness) * a * (r**(2/3)) * (channel_slope**0.5)
    
    # Знаходження нормальної глибини
    try:
        def objective(depth):
            return abs(manning_equation(depth) - flow_rate)
        
        normal_depth_result = optimize.minimize_scalar(objective, bounds=(0.01, 10*channel_depth), method='bounded')
        normal_depth = normal_depth_result.x
    except:
        normal_depth = channel_depth  # за замовчуванням
    
    # Критична глибина
    # Для прямокутного каналу: yc = (Q²/(g*b²))^(1/3)
    if channel_width > 0:
        critical_depth = (flow_rate**2 / (constants.g * channel_width**2))**(1/3)
    else:
        critical_depth = 0
    
    # Швидкість потоку
    velocity = flow_rate / area if area > 0 else 0
    
    # Число Фруда
    froude_number = velocity / math.sqrt(constants.g * channel_depth) if channel_depth > 0 else 0
    
    # Режим потоку
    if froude_number < 1:
        flow_type = "subcritical"  # спокійний потік
    elif froude_number > 1:
        flow_type = "supercritical"  # бурхливий потік
    else:
        flow_type = "critical"  # критичний потік
    
    # Енергія питома
    specific_energy = channel_depth + velocity**2 / (2 * constants.g) if constants.g > 0 else 0
    
    return {
        'normal_depth': normal_depth,
        'critical_depth': critical_depth,
        'flow_velocity': velocity,
        'froude_number': froude_number,
        'flow_type': flow_type,
        'specific_energy': specific_energy,
        'hydraulic_radius': hydraulic_radius,
        'wetted_perimeter': wetted_perimeter,
        'cross_sectional_area': area
    }

# Теплотехніка
def heat_transfer_analysis(thermal_conductivity: float, 
                         surface_area: float, 
                         temperature_difference: float, 
                         convection_coefficient: float = 0.0, 
                         thickness: float = 1.0) -> Dict[str, Any]:
    """
    Аналіз теплопередачі.
    
    Параметри:
        thermal_conductivity: Коефіцієнт теплопровідності
        surface_area: Площа поверхні
        temperature_difference: Різниця температур
        convection_coefficient: Коефіцієнт тепловіддачі
        thickness: Товщина стінки
    
    Повертає:
        Словник з результатами аналізу теплопередачі
    """
    # Теплопровідність (закон Фур'є)
    conduction_heat_transfer = thermal_conductivity * surface_area * temperature_difference / thickness
    
    # Тепловіддача (закон Ньютона-Ріхмана)
    convection_heat_transfer = convection_coefficient * surface_area * temperature_difference
    
    # Загальна теплопередача
    total_heat_transfer = conduction_heat_transfer + convection_heat_transfer
    
    # Термічний опір
    conduction_resistance = thickness / (thermal_conductivity * surface_area) if thermal_conductivity * surface_area > 0 else float('inf')
    convection_resistance = 1 / (convection_coefficient * surface_area) if convection_coefficient * surface_area > 0 else float('inf')
    total_resistance = conduction_resistance + convection_resistance
    
    # Коефіцієнт теплопередачі
    overall_heat_transfer_coeff = 1 / total_resistance if total_resistance > 0 else 0
    
    return {
        'conduction_heat_transfer': conduction_heat_transfer,
        'convection_heat_transfer': convection_heat_transfer,
        'total_heat_transfer': total_heat_transfer,
        'conduction_resistance': conduction_resistance,
        'convection_resistance': convection_resistance,
        'total_thermal_resistance': total_resistance,
        'overall_heat_transfer_coefficient': overall_heat_transfer_coeff
    }

def thermodynamic_cycle_analysis(temperatures: List[float], 
                               pressures: List[float], 
                               volumes: List[float], 
                               heat_added: List[float], 
                               heat_removed: List[float]) -> Dict[str, Any]:
    """
    Аналіз термодинамічного циклу.
    
    Параметри:
        temperatures: Температури в точках циклу
        pressures: Тиски в точках циклу
        volumes: Об'єми в точках циклу
        heat_added: Додана теплота в процесах
        heat_removed: Відведена теплота в процесах
    
    Повертає:
        Словник з результатами аналізу циклу
    """
    n_points = len(temperatures)
    
    # Робота циклу (площа під кривою в p-V діаграмі)
    work_done = 0
    for i in range(n_points - 1):
        # Трапецієподібне інтегрування
        work_done += (pressures[i] + pressures[i+1]) * (volumes[i+1] - volumes[i]) / 2
    # Замикання циклу
    work_done += (pressures[-1] + pressures[0]) * (volumes[0] - volumes[-1]) / 2
    
    # Теплота
    total_heat_added = sum(heat_added)
    total_heat_removed = sum(heat_removed)
    
    # ККД циклу
    thermal_efficiency = work_done / total_heat_added if total_heat_added > 0 else 0
    
    # Теоретичний ККД (цикл Карно)
    if max(temperatures) > 0:
        carnot_efficiency = 1 - min(temperatures) / max(temperatures)
    else:
        carnot_efficiency = 0
    
    # Середній тиск
    mean_effective_pressure = work_done / (max(volumes) - min(volumes)) if (max(volumes) - min(volumes)) > 0 else 0
    
    # Зміна ентропії (спрощено)
    entropy_changes = []
    for i in range(n_points - 1):
        if temperatures[i] > 0:
            delta_s = heat_added[i] / temperatures[i] if heat_added[i] > 0 else heat_removed[i] / temperatures[i]
            entropy_changes.append(delta_s)
        else:
            entropy_changes.append(0)
    
    return {
        'work_done': work_done,
        'heat_added': total_heat_added,
        'heat_removed': total_heat_removed,
        'thermal_efficiency': thermal_efficiency,
        'carnot_efficiency': carnot_efficiency,
        'mean_effective_pressure': mean_effective_pressure,
        'entropy_changes': entropy_changes,
        'n_cycle_points': n_points
    }

# Електротехніка
def electrical_circuit_analysis(resistances: List[float], 
                              voltages: List[float], 
                              circuit_topology: str = "series") -> Dict[str, Any]:
    """
    Аналіз електричного кола.
    
    Параметри:
        resistances: Опори елементів
        voltages: Напруги елементів
        circuit_topology: Топологія кола ("series", "parallel")
    
    Повертає:
        Словник з результатами аналізу
    """
    if circuit_topology == "series":
        # Послідовне з'єднання
        total_resistance = sum(resistances)
        total_voltage = sum(voltages)
        
        # Струм в колі (закон Ома)
        if total_resistance > 0:
            current = total_voltage / total_resistance
        else:
            current = 0
        
        # Напруги на елементах
        element_voltages = [current * r for r in resistances]
        
        # Потужності
        powers = [current**2 * r for r in resistances]
        total_power = sum(powers)
        
    elif circuit_topology == "parallel":
        # Паралельне з'єднання
        # Загальний опір
        if all(r > 0 for r in resistances):
            total_resistance = 1 / sum(1/r for r in resistances)
        else:
            total_resistance = 0
        
        # Загальна напруга (припускаємо, що всі елементи мають однакову напругу)
        total_voltage = voltages[0] if voltages else 0
        
        # Струми в гілках
        branch_currents = [total_voltage / r if r > 0 else 0 for r in resistances]
        total_current = sum(branch_currents)
        
        # Потужності
        powers = [total_voltage**2 / r if r > 0 else 0 for r in resistances]
        total_power = sum(powers)
        
        current = total_current
        element_voltages = [total_voltage] * len(resistances)
    else:
        # Складна топологія - спрощений аналіз
        total_resistance = sum(resistances) / len(resistances) if resistances else 0
        total_voltage = sum(voltages) / len(voltages) if voltages else 0
        current = total_voltage / total_resistance if total_resistance > 0 else 0
        element_voltages = [total_voltage / len(voltages)] * len(voltages) if voltages else []
        branch_currents = [current / len(resistances)] * len(resistances) if resistances else []
        powers = [v**2 / r if r > 0 else 0 for v, r in zip(element_voltages, resistances)]
        total_power = sum(powers)
    
    # Енергія
    energy_dissipated = total_power  # потужність = енергія/час, приймаємо час = 1с
    
    return {
        'total_resistance': total_resistance,
        'total_voltage': total_voltage,
        'total_current': current,
        'element_voltages': element_voltages,
        'branch_currents': branch_currents if circuit_topology == "parallel" else [current] * len(resistances),
        'powers': powers,
        'total_power': total_power,
        'energy_dissipated': energy_dissipated,
        'circuit_topology': circuit_topology
    }

def ac_circuit_analysis(resistance: float, 
                      inductance: float, 
                      capacitance: float, 
                      frequency: float, 
                      voltage_amplitude: float) -> Dict[str, Any]:
    """
    Аналіз кола змінного струму.
    
    Параметри:
        resistance: Активний опір (Ом)
        inductance: Індуктивність (Гн)
        capacitance: Ємність (Ф)
        frequency: Частота (Гц)
        voltage_amplitude: Амплітуда напруги (В)
    
    Повертає:
        Словник з результатами аналізу
    """
    # Кутова частота
    omega = 2 * math.pi * frequency
    
    # Реактивні опори
    inductive_reactance = omega * inductance
    capacitive_reactance = 1 / (omega * capacitance) if omega * capacitance > 0 else float('inf')
    
    # Повний опір (імпеданс)
    impedance_real = resistance
    impedance_imag = inductive_reactance - capacitive_reactance
    impedance_magnitude = math.sqrt(impedance_real**2 + impedance_imag**2)
    impedance_phase = math.atan2(impedance_imag, impedance_real)
    
    # Струм
    current_amplitude = voltage_amplitude / impedance_magnitude if impedance_magnitude > 0 else 0
    current_phase = -impedance_phase  # струм відстає від напруги на кут φ
    
    # Напруги на елементах
    voltage_r = current_amplitude * resistance  # активна напруга
    voltage_l = current_amplitude * inductive_reactance  # індуктивна напруга
    voltage_c = current_amplitude * capacitive_reactance if capacitive_reactance != float('inf') else 0  # ємнісна напруга
    
    # Потужності
    apparent_power = voltage_amplitude * current_amplitude  # повна потужність
    real_power = voltage_amplitude * current_amplitude * math.cos(impedance_phase)  # активна потужність
    reactive_power = voltage_amplitude * current_amplitude * math.sin(impedance_phase)  # реактивна потужність
    
    # Коефіцієнт потужності
    power_factor = math.cos(impedance_phase)
    
    # Резонансна частота
    resonant_frequency = 1 / (2 * math.pi * math.sqrt(inductance * capacitance)) if inductance * capacitance > 0 else 0
    
    # Характер навантаження
    if impedance_imag > 0:
        load_character = "inductive"
    elif impedance_imag < 0:
        load_character = "capacitive"
    else:
        load_character = "resistive"
    
    return {
        'impedance_magnitude': impedance_magnitude,
        'impedance_phase': impedance_phase,
        'current_amplitude': current_amplitude,
        'current_phase': current_phase,
        'voltage_r': voltage_r,
        'voltage_l': voltage_l,
        'voltage_c': voltage_c,
        'apparent_power': apparent_power,
        'real_power': real_power,
        'reactive_power': reactive_power,
        'power_factor': power_factor,
        'resonant_frequency': resonant_frequency,
        'load_character': load_character,
        'inductive_reactance': inductive_reactance,
        'capacitive_reactance': capacitive_reactance
    }

# Механіка рідин та газів
def fluid_dynamics_analysis(velocity_field: List[List[float]], 
                          density: float, 
                          viscosity: float, 
                          characteristic_length: float) -> Dict[str, Any]:
    """
    Аналіз динаміки рідини.
    
    Параметри:
        velocity_field: Поле швидкостей [[u,v,w], ...]
        density: Густина рідини
        viscosity: Динамічна в'язкість
        characteristic_length: Характерна довжина
    
    Повертає:
        Словник з результатами аналізу
    """
    # Число Рейнольдса
    if len(velocity_field) > 0:
        velocity_magnitude = math.sqrt(sum(v**2 for v in velocity_field[0]))
    else:
        velocity_magnitude = 0
    
    reynolds_number = density * velocity_magnitude * characteristic_length / viscosity if viscosity > 0 else 0
    
    # Режим течії
    if reynolds_number < 2300:
        flow_regime = "laminar"
    elif reynolds_number < 4000:
        flow_regime = "transitional"
    else:
        flow_regime = "turbulent"
    
    # Дивергенція поля швидкостей (для перевірки збереження маси)
    # Спрощено для 2D випадку
    divergence = 0
    if len(velocity_field) >= 4:
        # Приблизна оцінка дивергенції
        dx = 1.0  # крок сітки
        dy = 1.0
        # du/dx + dv/dy
        du_dx = (velocity_field[1][0] - velocity_field[0][0]) / dx
        dv_dy = (velocity_field[2][1] - velocity_field[0][1]) / dy
        divergence = du_dx + dv_dy
    
    # Вихор (ротор) поля швидкостей
    # Спрощено для 2D випадку: ω = dv/dx - du/dy
    vorticity = 0
    if len(velocity_field) >= 4:
        dv_dx = (velocity_field[1][1] - velocity_field[0][1]) / dx
        du_dy = (velocity_field[2][0] - velocity_field[0][0]) / dy
        vorticity = dv_dx - du_dy
    
    # Кінетична енергія
    kinetic_energy = 0
    for velocity in velocity_field:
        kinetic_energy += 0.5 * density * sum(v**2 for v in velocity)
    
    # Напруження в'язкості (спрощено)
    # τ = μ * du/dy
    shear_stress = 0
    if len(velocity_field) >= 2 and viscosity > 0:
        du_dy = (velocity_field[1][0] - velocity_field[0][0]) / dy if dy > 0 else 0
        shear_stress = viscosity * du_dy
    
    return {
        'reynolds_number': reynolds_number,
        'flow_regime': flow_regime,
        'velocity_magnitude': velocity_magnitude,
        'divergence': divergence,
        'vorticity': vorticity,
        'kinetic_energy': kinetic_energy,
        'shear_stress': shear_stress,
        'n_velocity_points': len(velocity_field)
    }

def bernoulli_equation_analysis(pressures: List[float], 
                               velocities: List[float], 
                               heights: List[float], 
                               fluid_density: float) -> Dict[str, Any]:
    """
    Аналіз рівняння Бернуллі.
    
    Параметри:
        pressures: Тиски в різних точках
        velocities: Швидкості в різних точках
        heights: Висоти в різних точках
        fluid_density: Густина рідини
    
    Повертає:
        Словник з результатами аналізу
    """
    n_points = len(pressures)
    
    # Повний напір в кожній точці
    total_heads = []
    for i in range(n_points):
        # H = p/(ρg) + v²/(2g) + z
        pressure_head = pressures[i] / (fluid_density * constants.g) if constants.g > 0 else 0
        velocity_head = velocities[i]**2 / (2 * constants.g) if constants.g > 0 else 0
        elevation_head = heights[i]
        total_head = pressure_head + velocity_head + elevation_head
        total_heads.append(total_head)
    
    # Втрати напору
    head_losses = []
    for i in range(n_points - 1):
        loss = total_heads[i] - total_heads[i+1]
        head_losses.append(loss)
    
    # Коефіцієнт втрат
    if len(head_losses) > 0 and total_heads[0] != 0:
        loss_coefficient = sum(head_losses) / total_heads[0]
    else:
        loss_coefficient = 0
    
    # Максимальна швидкість
    max_velocity = max(velocities) if velocities else 0
    
    # Максимальний тиск
    max_pressure = max(pressures) if pressures else 0
    
    # Мінімальний тиск (може бути вакуумом)
    min_pressure = min(pressures) if pressures else 0
    
    # Кавітація
    vapor_pressure = 2338  # приблизне значення для води при 20°C
    cavitation_risk = min_pressure < vapor_pressure
    
    return {
        'total_heads': total_heads,
        'head_losses': head_losses,
        'loss_coefficient': loss_coefficient,
        'max_velocity': max_velocity,
        'max_pressure': max_pressure,
        'min_pressure': min_pressure,
        'cavitation_risk': cavitation_risk,
        'vapor_pressure': vapor_pressure,
        'n_points': n_points
    }

# Механічні коливання та хвилі
def vibration_analysis(mass: float, 
                     stiffness: float, 
                     damping: float, 
                     initial_displacement: float = 0, 
                     initial_velocity: float = 0, 
                     time_span: Tuple[float, float] = (0, 10)) -> Dict[str, Any]:
    """
    Аналіз механічних коливань.
    
    Параметри:
        mass: Маса системи
        stiffness: Жорсткість
        damping: Демпфування
        initial_displacement: Початкове зміщення
        initial_velocity: Початкова швидкість
        time_span: Інтервал часу
    
    Повертає:
        Словник з результатами аналізу
    """
    # Природна частота
    natural_frequency = math.sqrt(stiffness / mass) if mass > 0 else 0
    
    # Частота з демпфуванням
    damped_frequency = math.sqrt(natural_frequency**2 - (damping/(2*mass))**2) if mass > 0 else 0
    
    # Коефіцієнт демпфування
    damping_ratio = damping / (2 * math.sqrt(mass * stiffness)) if mass * stiffness > 0 else 0
    
    # Тип демпфування
    if damping_ratio < 1:
        damping_type = "underdamped"
    elif damping_ratio == 1:
        damping_type = "critically_damped"
    else:
        damping_type = "overdamped"
    
    # Розв'язок диференціального рівняння
    # m*x'' + c*x' + k*x = 0
    dt = 0.01
    time_points = np.arange(time_span[0], time_span[1], dt)
    displacements = []
    velocities = []
    accelerations = []
    
    # Початкові умови
    x = initial_displacement
    v = initial_velocity
    
    for t in time_points:
        # Прискорення: x'' = -(c*v + k*x)/m
        if mass > 0:
            a = -(damping * v + stiffness * x) / mass
        else:
            a = 0
        
        displacements.append(x)
        velocities.append(v)
        accelerations.append(a)
        
        # Оновлення швидкості та зміщення (метод Ейлера)
        v += a * dt
        x += v * dt
    
    # Амплітуда коливань
    amplitude = max(abs(d) for d in displacements) if displacements else 0
    
    # Період коливань
    period = 2 * math.pi / damped_frequency if damped_frequency > 0 else float('inf')
    
    # Частота (Гц)
    frequency_hz = damped_frequency / (2 * math.pi) if damped_frequency > 0 else 0
    
    # Енергія системи
    kinetic_energy = [0.5 * mass * v**2 for v in velocities]
    potential_energy = [0.5 * stiffness * x**2 for x in displacements]
    total_energy = [ke + pe for ke, pe in zip(kinetic_energy, potential_energy)]
    
    return {
        'natural_frequency': natural_frequency,
        'damped_frequency': damped_frequency,
        'damping_ratio': damping_ratio,
        'damping_type': damping_type,
        'displacements': displacements,
        'velocities': velocities,
        'accelerations': accelerations,
        'time_points': time_points.tolist(),
        'amplitude': amplitude,
        'period': period,
        'frequency_hz': frequency_hz,
        'kinetic_energy': kinetic_energy,
        'potential_energy': potential_energy,
        'total_energy': total_energy
    }

def wave_equation_analysis(wave_speed: float, 
                         wavelength: float, 
                         amplitude: float, 
                         time_points: List[float], 
                         position_points: List[float]) -> Dict[str, Any]:
    """
    Аналіз хвильового рівняння.
    
    Параметри:
        wave_speed: Швидкість поширення хвилі
        wavelength: Довжина хвилі
        amplitude: Амплітуда хвилі
        time_points: Точки часу
        position_points: Точки простору
    
    Повертає:
        Словник з результатами аналізу
    """
    # Частота
    frequency = wave_speed / wavelength if wavelength > 0 else 0
    
    # Кутова частота
    angular_frequency = 2 * math.pi * frequency
    
    # Хвильове число
    wave_number = 2 * math.pi / wavelength if wavelength > 0 else 0
    
    # Рівняння біжучої хвилі: y(x,t) = A * sin(k*x - ω*t)
    wave_field = []
    for t in time_points:
        wave_row = []
        for x in position_points:
            y = amplitude * math.sin(wave_number * x - angular_frequency * t)
            wave_row.append(y)
        wave_field.append(wave_row)
    
    # Енергія хвилі (на одиницю довжини)
    # E = 0.5 * μ * ω² * A²
    # де μ - лінійна густина
    linear_density = 1.0  # приблизне значення
    wave_energy = 0.5 * linear_density * angular_frequency**2 * amplitude**2
    
    # Імпульс хвилі
    wave_momentum = wave_energy / wave_speed if wave_speed > 0 else 0
    
    # Інтенсивність хвилі
    intensity = 0.5 * linear_density * wave_speed * angular_frequency**2 * amplitude**2
    
    # Стояча хвиля (сума прямої та відбитої хвиль)
    standing_wave = []
    for t in time_points:
        wave_row = []
        for x in position_points:
            # y = 2*A * sin(k*x) * cos(ω*t)
            y = 2 * amplitude * math.sin(wave_number * x) * math.cos(angular_frequency * t)
            wave_row.append(y)
        standing_wave.append(wave_row)
    
    # Вузли стоячої хвилі
    nodes = []
    for i in range(int(wavelength) + 1):
        node_position = i * wavelength / 2
        nodes.append(node_position)
    
    # Пучності стоячої хвилі
    antinodes = []
    for i in range(int(wavelength) + 1):
        antinode_position = (i + 0.5) * wavelength / 2
        antinodes.append(antinode_position)
    
    return {
        'frequency': frequency,
        'angular_frequency': angular_frequency,
        'wave_number': wave_number,
        'wave_field': wave_field,
        'wave_energy': wave_energy,
        'wave_momentum': wave_momentum,
        'intensity': intensity,
        'standing_wave': standing_wave,
        'nodes': nodes,
        'antinodes': antinodes,
        'wave_speed': wave_speed,
        'wavelength': wavelength
    }

if __name__ == "__main__":
    # Тестування функцій модуля
    print("Тестування модуля обчислювальної інженерії PyNexus")
    
    # Тест аналізу прогину балки
    def uniform_load(x):
        return 1000  # Н/м
    
    beam_result = beam_deflection_analysis(
        beam_length=10.0,
        load_distribution=uniform_load,
        young_modulus=2e11,  # Сталь
        moment_of_inertia=1e-4,
        boundary_conditions="simply_supported"
    )
    print(f"Максимальний прогин балки: {beam_result['max_deflection']:.6f} м")
    
    # Тест аналізу напружень
    stress_result = stress_analysis(
        normal_forces=[10000, 15000, 12000],
        shear_forces=[5000, 3000, 4000],
        cross_sectional_area=0.01,
        moment_of_inertia=1e-5,
        distances_from_neutral_axis=[0.05, 0.03, 0.04]
    )
    print(f"Максимальне еквівалентне напруження: {stress_result['max_equivalent_stress']:.0f} Па")
    
    # Тест аналізу течії рідини
    fluid_result = fluid_flow_analysis(
        pipe_diameter=0.1,
        pipe_length=100,
        fluid_density=1000,
        fluid_viscosity=0.001,
        pressure_drop=50000
    )
    print(f"Швидкість течії: {fluid_result['flow_velocity']:.2f} м/с")
    print(f"Режим течії: {fluid_result['flow_regime']}")
    
    # Тест теплопередачі
    heat_result = heat_transfer_analysis(
        thermal_conductivity=401,  # Мідь
        surface_area=1.0,
        temperature_difference=50,
        convection_coefficient=10,
        thickness=0.01
    )
    print(f"Загальна теплопередача: {heat_result['total_heat_transfer']:.0f} Вт")
    
    # Тест електричного кола
    circuit_result = electrical_circuit_analysis(
        resistances=[10, 20, 30],
        voltages=[12, 0, 0],
        circuit_topology="series"
    )
    print(f"Струм в послідовному колі: {circuit_result['total_current']:.2f} А")
    
    # Тест коливань
    vibration_result = vibration_analysis(
        mass=1.0,
        stiffness=100,
        damping=5,
        initial_displacement=0.1
    )
    print(f"Природна частота: {vibration_result['natural_frequency']:.2f} рад/с")
    print(f"Тип демпфування: {vibration_result['damping_type']}")
    
    print("Тестування завершено!")