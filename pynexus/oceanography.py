"""
Модуль для обчислювальної океанології
Computational Oceanography Module
"""
import numpy as np
from typing import Union, Tuple, List, Optional, Dict, Any
import math

# Константи для океанологічних обчислень
# Oceanographic constants
EARTH_RADIUS = 6371000  # Радіус Землі в метрах
GRAVITY = 9.80665  # Прискорення вільного падіння (м/с²)
CORIOLIS_PARAMETER_F0 = 1.45842e-4  # Параметр Коріоліса на екваторі (с⁻¹)
EARTH_ROTATION_RATE = 7.292115e-5  # Кутова швидкість обертання Землі (рад/с)
SEAWATER_DENSITY = 1025  # Середня густина морської води (кг/м³)
FRESHWATER_DENSITY = 1000  # Густина прісної води (кг/м³)
SEAWATER_SPECIFIC_HEAT = 3990  # Питома теплоємність морської води (Дж/(кг·К))
REFERENCE_SALINITY = 35  # Референсна солоність (‰)
REFERENCE_TEMPERATURE = 15  # Референсна температура (°C)
UNIVERSAL_GAS_CONSTANT = 8.31446261815324  # Універсальна газова стала (Дж/(моль·К))
OXYGEN_SOLUBILITY_CONSTANT = 1.0  # Константа розчинності кисню

def seawater_density(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити густину морської води за формулою UNESCO 1983.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Абсолютна солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Густина морської води (кг/м³)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Формула UNESCO 1983 для густини морської води
    # Спочатку обчислюємо густину при атмосферному тиску
    rho_0 = (
        999.842594 + 6.793952e-2 * temperature - 9.095290e-3 * temperature**2 +
        1.001685e-4 * temperature**3 - 1.120083e-6 * temperature**4 + 
        6.536332e-9 * temperature**5
    )
    
    # Додаткові члени для солоності
    A = (
        8.24493e-1 - 4.0899e-3 * temperature + 7.6438e-5 * temperature**2 -
        8.2467e-7 * temperature**3 + 5.3875e-9 * temperature**4
    )
    
    B = (
        -5.72466e-3 + 1.0227e-4 * temperature - 1.6546e-6 * temperature**2
    )
    
    C = 4.8314e-4
    
    rho_stp = rho_0 + A * salinity + B * salinity**(3/2) + C * salinity**2
    
    # Корекція на тиск
    if pressure > 0:
        # Коефіцієнти для тискової корекції
        K0 = (
            19652.21 + 148.4206 * temperature - 2.327105 * temperature**2 +
            1.360477e-2 * temperature**3 - 5.155288e-5 * temperature**4
        )
        
        K1 = (
            3.239908 + 1.43713e-3 * temperature + 1.16092e-4 * temperature**2 -
            5.77905e-7 * temperature**3
        )
        
        K2 = (
            8.50935e-5 - 6.12293e-6 * temperature + 5.2787e-8 * temperature**2
        )
        
        # Часткова похідна густини по тиску
        d_rho_dp = (K0 + K1 * salinity + K2 * salinity**(3/2)) / 10000
        
        # Корекція густини на тиск
        rho = rho_stp / (1 - pressure / d_rho_dp)
    else:
        rho = rho_stp
    
    return rho

def seawater_specific_volume(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити питомий об'єм морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Абсолютна солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Питомий об'єм морської води (м³/кг)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Питомий об'єм = 1 / густина
    density = seawater_density(temperature, salinity, pressure)
    specific_volume = 1 / density
    return specific_volume

def seawater_compressibility(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити стисливість морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Абсолютна солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Стисливість морської води (1/Па)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Обчислюємо густину при поточному тиску
    rho_p = seawater_density(temperature, salinity, pressure)
    
    # Обчислюємо густину при тиску + 1 дбар
    rho_p1 = seawater_density(temperature, salinity, pressure + 1)
    
    # Стисливість = (1/ρ) * (dρ/dp)
    # Приблизно: (1/ρ) * (Δρ/Δp)
    # Δp = 1 дбар = 10000 Па
    compressibility = (1/rho_p) * ((rho_p1 - rho_p) / 10000)
    return compressibility

def seawater_speed_of_sound(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити швидкість звуку в морській воді за формулою Макензі.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Швидкість звуку в морській воді (м/с)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Формула Макензі для швидкості звуку в морській воді
    c = (
        1448.96 + 4.591 * temperature - 5.304e-2 * temperature**2 + 
        2.374e-4 * temperature**3 + 1.340 * (salinity - 35) +
        1.630e-2 * pressure + 1.675e-7 * pressure**2 -
        1.025e-2 * temperature * (salinity - 35) -
        7.139e-13 * temperature * pressure**3
    )
    return c

def seawater_thermal_expansion(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити коефіцієнт термічного розширення морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Абсолютна солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Коефіцієнт термічного розширення (1/К)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Обчислюємо густину при поточній температурі
    rho_t = seawater_density(temperature, salinity, pressure)
    
    # Обчислюємо густину при температурі + 1°C
    rho_t1 = seawater_density(temperature + 1, salinity, pressure)
    
    # Коефіцієнт термічного розширення = -(1/ρ) * (dρ/dT)
    # Приблизно: -(1/ρ) * (Δρ/ΔT)
    # ΔT = 1°C
    thermal_expansion = -(1/rho_t) * ((rho_t1 - rho_t) / 1)
    return thermal_expansion

def seawater_salinity_conductivity(conductivity: float, temperature: float) -> float:
    """
    Обчислити солоність з провідності за формулою UNESCO 1983.
    
    Параметри:
        conductivity: Провідність (мСм/см)
        temperature: Температура води (°C)
    
    Повертає:
        Практична солоність (‰)
    """
    if conductivity < 0:
        raise ValueError("Провідність повинна бути невід'ємною")
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    
    # Нормалізована провідність при 15°C
    R15 = conductivity / 42.914
    
    # Температурна корекція
    Rt = (
        R15 / (1 + 0.008 * (temperature - 15) - 0.0001 * (temperature - 15)**2)
    )
    
    # Формула UNESCO для солоності
    salinity = (
        0.008 + 0.0005 * Rt**(1/2) - 0.00001 * Rt + 0.0000005 * Rt**(3/2)
    ) * 1000
    return salinity

def seawater_freezing_point(salinity: float, pressure: float = 0) -> float:
    """
    Обчислити температуру замерзання морської води.
    
    Параметри:
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Температура замерзання (°C)
    """
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Формула для температури замерзання морської води
    Tf = -0.0575 * salinity + 1.710523e-3 * salinity**(3/2) - 2.154996e-4 * salinity**2
    
    # Корекція на тиск
    if pressure > 0:
        Tf = Tf - 7.53e-4 * pressure
    
    return Tf

def seawater_heat_capacity(temperature: float, salinity: float) -> float:
    """
    Обчислити питому теплоємність морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
    
    Повертає:
        Питома теплоємність (Дж/(кг·К))
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    
    # Емпірична формула для питомої теплоємності морської води
    Cp = (
        SEAWATER_SPECIFIC_HEAT + 
        0.5 * salinity - 
        0.01 * temperature * salinity +
        1e-4 * temperature**2 * salinity
    )
    return Cp

def seawater_viscosity(temperature: float, salinity: float) -> float:
    """
    Обчислити динамічну в'язкість морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
    
    Повертає:
        Динамічна в'язкість (Па·с)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    
    # В'язкість прісної води при заданій температурі
    # Формула для в'язкості прісної води
    mu_fresh = (
        1.7916e-3 - 5.575e-5 * temperature + 1.0847e-6 * temperature**2 -
        1.4318e-8 * temperature**3
    )
    
    # Корекція на солоність
    mu_sea = mu_fresh * (1 + 0.0018 * salinity)
    return mu_sea

def seawater_kinematic_viscosity(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити кінематичну в'язкість морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Кінематична в'язкість (м²/с)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Динамічна в'язкість
    mu = seawater_viscosity(temperature, salinity)
    
    # Густина
    rho = seawater_density(temperature, salinity, pressure)
    
    # Кінематична в'язкість = динамічна в'язкість / густина
    nu = mu / rho
    return nu

def seawater_prandtl_number(temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити число Прандтля для морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Число Прандтля
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Теплопровідність морської води (приблизна)
    k = 0.555 + 0.0022 * temperature
    
    # Динамічна в'язкість
    mu = seawater_viscosity(temperature, salinity)
    
    # Питома теплоємність
    Cp = seawater_heat_capacity(temperature, salinity)
    
    # Густина
    rho = seawater_density(temperature, salinity, pressure)
    
    # Число Прандтля = (mu * Cp) / k
    Pr = (mu * Cp) / k
    return Pr

def seawater_reynolds_number(velocity: float, length: float, 
                           temperature: float, salinity: float, pressure: float = 0) -> float:
    """
    Обчислити число Рейнольдса для потоку морської води.
    
    Параметри:
        velocity: Швидкість потоку (м/с)
        length: Характерна довжина (м)
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Число Рейнольдса
    """
    if velocity < 0:
        raise ValueError("Швидкість повинна бути невід'ємною")
    if length <= 0:
        raise ValueError("Довжина повинна бути додатньою")
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Кінематична в'язкість
    nu = seawater_kinematic_viscosity(temperature, salinity, pressure)
    
    # Число Рейнольдса = (швидкість * довжина) / кінематична в'язкість
    Re = (velocity * length) / nu
    return Re

def seawater_grashof_number(temperature_surface: float, temperature_bulk: float,
                          length: float, temperature: float, salinity: float, 
                          pressure: float = 0) -> float:
    """
    Обчислити число Грасгофа для конвективного потоку морської води.
    
    Параметри:
        temperature_surface: Температура поверхні (°C)
        temperature_bulk: Температура основної маси води (°C)
        length: Характерна довжина (м)
        temperature: Середня температура води (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Число Грасгофа
    """
    if length <= 0:
        raise ValueError("Довжина повинна бути додатньою")
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Коефіцієнт термічного розширення
    beta = seawater_thermal_expansion(temperature, salinity, pressure)
    
    # Кінематична в'язкість
    nu = seawater_kinematic_viscosity(temperature, salinity, pressure)
    
    # Різниця температур
    delta_T = abs(temperature_surface - temperature_bulk)
    
    # Число Грасгофа = (g * beta * delta_T * L^3) / nu^2
    Gr = (GRAVITY * beta * delta_T * length**3) / nu**2
    return Gr

def seawater_nusselt_number(grashof_number: float, prandtl_number: float) -> float:
    """
    Обчислити число Нуссельта для конвективного теплообміну.
    
    Параметри:
        grashof_number: Число Грасгофа
        prandtl_number: Число Прандтля
    
    Повертає:
        Число Нуссельта
    """
    if grashof_number < 0:
        raise ValueError("Число Грасгофа повинно бути невід'ємним")
    if prandtl_number < 0:
        raise ValueError("Число Прандтля повинно бути невід'ємним")
    
    # Для ламінарної конвекції
    if grashof_number * prandtl_number < 1e9:
        Nu = 0.59 * (grashof_number * prandtl_number)**0.25
    # Для турбулентної конвекції
    else:
        Nu = 0.1 * (grashof_number * prandtl_number)**0.33
    return Nu

def seawater_heat_transfer_coefficient(nusselt_number: float, thermal_conductivity: float, 
                                     characteristic_length: float) -> float:
    """
    Обчислити коефіцієнт тепловіддачі.
    
    Параметри:
        nusselt_number: Число Нуссельта
        thermal_conductivity: Теплопровідність (Вт/(м·К))
        characteristic_length: Характерна довжина (м)
    
    Повертає:
        Коефіцієнт тепловіддачі (Вт/(м²·К))
    """
    if nusselt_number < 0:
        raise ValueError("Число Нуссельта повинно бути невід'ємним")
    if thermal_conductivity <= 0:
        raise ValueError("Теплопровідність повинна бути додатньою")
    if characteristic_length <= 0:
        raise ValueError("Характерна довжина повинна бути додатньою")
    
    # Коефіцієнт тепловіддачі = (Nu * k) / L
    h = (nusselt_number * thermal_conductivity) / characteristic_length
    return h

def seawater_coriolis_frequency(latitude: float) -> float:
    """
    Обчислити частоту Коріоліса.
    
    Параметри:
        latitude: Географічна широта (градуси)
    
    Повертає:
        Частота Коріоліса (с⁻¹)
    """
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    
    # f = 2 * Ω * sin(φ)
    f = 2 * EARTH_ROTATION_RATE * math.sin(math.radians(latitude))
    return f

def seawater_rossby_radius(temperature: float, salinity: float, 
                         latitude: float, depth: float = 1000) -> float:
    """
    Обчислити радіус Россбі для внутрішніх хвиль.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        latitude: Географіча широта (градуси)
        depth: Глибина (м), за замовчуванням 1000 м
    
    Повертає:
        Радіус Россбі (м)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    if depth <= 0:
        raise ValueError("Глибина повинна бути додатньою")
    
    # Частота Коріоліса
    f = seawater_coriolis_frequency(latitude)
    
    # Уникнути ділення на нуль
    if abs(f) < 1e-10:
        return float('inf')
    
    # Швидкість звуку
    c = seawater_speed_of_sound(temperature, salinity)
    
    # Радіус Россбі = c / f
    rossby_radius = c / abs(f)
    return rossby_radius

def seawater_buoyancy_frequency(temperature_upper: float, temperature_lower: float,
                              salinity_upper: float, salinity_lower: float,
                              depth_upper: float, depth_lower: float) -> float:
    """
    Обчислити частоту буйянсу (частоту Брента-Вяйсяля).
    
    Параметри:
        temperature_upper: Температура верхнього шару (°C)
        temperature_lower: Температура нижнього шару (°C)
        salinity_upper: Солоність верхнього шару (‰)
        salinity_lower: Солоність нижнього шару (‰)
        depth_upper: Глибина верхнього шару (м)
        depth_lower: Глибина нижнього шару (м)
    
    Повертає:
        Частота буйянсу (с⁻¹)
    """
    if temperature_upper < -2 or temperature_upper > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if temperature_lower < -2 or temperature_lower > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity_upper < 0 or salinity_upper > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if salinity_lower < 0 or salinity_lower > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if depth_upper >= depth_lower:
        raise ValueError("Глибина верхнього шару повинна бути меншою за глибину нижнього шару")
    
    # Густина верхнього шару
    rho_upper = seawater_density(temperature_upper, salinity_upper)
    
    # Густина нижнього шару
    rho_lower = seawater_density(temperature_lower, salinity_lower)
    
    # Різниця глибин
    delta_z = depth_lower - depth_upper
    
    # Уникнути ділення на нуль
    if delta_z < 1e-10:
        return 0
    
    # Частота буйянсу = sqrt((g/ρ) * (dρ/dz))
    # Приблизно: sqrt((g/ρ) * (Δρ/Δz))
    rho_avg = (rho_upper + rho_lower) / 2
    delta_rho = rho_lower - rho_upper
    
    # Якщо нижній шар важчий, то стратифікація стабільна
    if delta_rho > 0:
        N_squared = (GRAVITY / rho_avg) * (delta_rho / delta_z)
        N = math.sqrt(max(0, N_squared))
    else:
        # Якщо нижній шар легший, то стратифікація нестабільна
        N = 0
    
    return N

def seawater_mixed_layer_depth(temperature_profile: List[float], 
                             depth_profile: List[float],
                             threshold: float = 0.2) -> float:
    """
    Обчислити глибину змішаного шару.
    
    Параметри:
        temperature_profile: Профіль температури (°C)
        depth_profile: Профіль глибин (м)
        threshold: Поріг температурної різниці (°C), за замовчуванням 0.2°C
    
    Повертає:
        Глибина змішаного шару (м)
    """
    if len(temperature_profile) != len(depth_profile):
        raise ValueError("Профілі температури та глибин повинні мати однакову довжину")
    if len(temperature_profile) < 2:
        raise ValueError("Профілі повинні містити принаймні 2 точки")
    if threshold <= 0:
        raise ValueError("Поріг повинен бути додатнім")
    
    # Температура поверхні
    surface_temp = temperature_profile[0]
    
    # Пошук глибини, де температура відрізняється більше ніж на поріг
    for i in range(1, len(temperature_profile)):
        temp_diff = abs(surface_temp - temperature_profile[i])
        if temp_diff > threshold:
            return depth_profile[i-1]  # Повертаємо глибину попередньої точки
    
    # Якщо поріг ніколи не перевищено, повертаємо максимальну глибину
    return depth_profile[-1]

def seawater_potential_temperature(temperature: float, salinity: float, 
                                 pressure: float, reference_pressure: float = 0) -> float:
    """
    Обчислити потенційну температуру морської води.
    
    Параметри:
        temperature: In-situ температура (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар)
        reference_pressure: Референсний тиск (дбар), за замовчуванням 0
    
    Повертає:
        Потенційна температура (°C)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    if reference_pressure < 0:
        raise ValueError("Референсний тиск повинен бути невід'ємним")
    
    # Спрощена формула для потенційної температури
    # У реальних океанографічних розрахунках використовують складніші алгоритми
    # Це приблизна формула
    
    # Коефіцієнт термічного розширення
    alpha = seawater_thermal_expansion(temperature, salinity, pressure)
    
    # Густина
    rho = seawater_density(temperature, salinity, pressure)
    
    # Питома теплоємність
    Cp = seawater_heat_capacity(temperature, salinity)
    
    # Різниця тисків
    delta_p = (reference_pressure - pressure) * 10000  # Перетворення дбар в Па
    
    # Потенційна температура
    # dT = (α * T * Δp) / (ρ * Cp)
    delta_T = (alpha * (temperature + 273.15) * delta_p) / (rho * Cp)
    potential_temp = temperature + delta_T
    
    return potential_temp

def seawater_conservative_temperature(temperature: float, salinity: float, 
                                   pressure: float) -> float:
    """
    Обчислити консервативну температуру морської води (приблизна формула).
    
    Параметри:
        temperature: In-situ температура (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар)
    
    Повертає:
        Консервативна температура (°C)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Консервативна температура приблизно дорівнює потенційній температурі
    # при референсному тиску 0 дбар
    conservative_temp = seawater_potential_temperature(temperature, salinity, pressure, 0)
    return conservative_temp

def seawater_absolute_salinity(salinity: float, pressure: float, 
                             latitude: float, longitude: float) -> float:
    """
    Обчислити абсолютну солоність морської води (приблизна формула).
    
    Параметри:
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар)
        latitude: Географічна широта (градуси)
        longitude: Географічна довгота (градуси)
    
    Повертає:
        Абсолютна солоність (г/кг)
    """
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    if longitude < -180 or longitude > 180:
        raise ValueError("Довгота повинна бути в діапазоні -180° до 180°")
    
    # Абсолютна солоність приблизно дорівнює практичній солоності
    # з невеликими корекціями на тиск і географічне положення
    # Це спрощена формула
    
    # Корекція на тиск (приблизна)
    pressure_correction = 0.0001 * pressure
    
    # Корекція на географічне положення (дуже приблизна)
    # Враховує вплив випаровування/опадів
    geographic_correction = 0.001 * math.sin(math.radians(latitude))
    
    absolute_salinity = salinity + pressure_correction + geographic_correction
    return absolute_salinity

def seawater_oxygen_solubility(temperature: float, salinity: float, 
                             pressure: float = 0) -> float:
    """
    Обчислити розчинність кисню в морській воді за формулою Гарві-Вільямса.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        pressure: Тиск (дбар), за замовчуванням 0
    
    Повертає:
        Розчинність кисню (мл/л)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if pressure < 0:
        raise ValueError("Тиск повинен бути невід'ємним")
    
    # Розчинність кисню в прісній воді при атмосферному тиску
    # Формула для прісної води
    Ts = math.log((273.15 + temperature) / 100)
    O2_fresh = math.exp(
        -177.7888 + 255.5907 * Ts + 146.4813 * Ts**2 - 22.2040 * Ts**3
    )
    
    # Корекція на солоність
    O2_salt = O2_fresh * (1 - 0.017 * salinity)
    
    # Корекція на тиск
    if pressure > 0:
        # Приблизна корекція на тиск
        pressure_correction = 1 + 0.0001 * pressure
        O2_solubility = O2_salt * pressure_correction
    else:
        O2_solubility = O2_salt
    
    return O2_solubility

def seawater_schmidt_number(temperature: float, salinity: float) -> float:
    """
    Обчислити число Шмідта для морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
    
    Повертає:
        Число Шмідта
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    
    # Кінематична в'язкість
    nu = seawater_kinematic_viscosity(temperature, salinity)
    
    # Коефіцієнт дифузії кисню в морській воді (приблизна оцінка)
    # D_O2 ≈ 1.5e-9 м²/с при 20°C
    # Температурна залежність: D ~ T^1.5
    D_O2_ref = 1.5e-9  # м²/с при 20°C
    T_ref = 20 + 273.15  # К
    T = temperature + 273.15  # К
    D_O2 = D_O2_ref * (T / T_ref)**1.5
    
    # Число Шмідта = ν / D
    Sc = nu / D_O2
    return Sc

def seawater_tidal_velocity(tidal_range: float, water_depth: float, 
                          period: float = 44714) -> float:
    """
    Обчислити швидкість приливного потоку.
    
    Параметри:
        tidal_range: Приливний діапазон (м)
        water_depth: Глибина води (м)
        period: Період приливів (с), за замовчуванням 12.42 години (44714 с)
    
    Повертає:
        Швидкість приливного потоку (м/с)
    """
    if tidal_range < 0:
        raise ValueError("Приливний діапазон повинен бути невід'ємним")
    if water_depth <= 0:
        raise ValueError("Глибина води повинна бути додатньою")
    if period <= 0:
        raise ValueError("Період повинен бути додатнім")
    
    # Швидкість приливного потоку = (tidal_range / 2) * (2π / period) * sqrt(g * depth)
    tidal_velocity = (tidal_range / 2) * (2 * math.pi / period) * math.sqrt(GRAVITY * water_depth)
    return tidal_velocity

def seawater_internal_wave_speed(density_upper: float, density_lower: float, 
                               water_depth: float) -> float:
    """
    Обчислити швидкість внутрішніх хвиль.
    
    Параметри:
        density_upper: Густина верхнього шару (кг/м³)
        density_lower: Густина нижнього шару (кг/м³)
        water_depth: Глибина води (м)
    
    Повертає:
        Швидкість внутрішніх хвиль (м/с)
    """
    if density_upper <= 0:
        raise ValueError("Густина верхнього шару повинна бути додатньою")
    if density_lower <= 0:
        raise ValueError("Густина нижнього шару повинна бути додатньою")
    if water_depth <= 0:
        raise ValueError("Глибина води повинна бути додатньою")
    if density_upper >= density_lower:
        raise ValueError("Густина верхнього шару повинна бути меншою за густину нижнього шару")
    
    # Швидкість внутрішніх хвиль = sqrt(g' * h)
    # де g' = g * (Δρ / ρ_avg) - зведена сила тяжіння
    delta_rho = density_lower - density_upper
    rho_avg = (density_upper + density_lower) / 2
    g_prime = GRAVITY * (delta_rho / rho_avg)
    
    # Для першої моди внутрішніх хвиль
    c = math.sqrt(g_prime * water_depth)
    return c

def seawater_molecular_diffusion_coefficient(temperature: float, 
                                           salinity: float) -> float:
    """
    Обчислити молекулярний коефіцієнт дифузії для морської води.
    
    Параметри:
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
    
    Повертає:
        Молекулярний коефіцієнт дифузії (м²/с)
    """
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    
    # Коефіцієнт дифузії солі в морській воді (приблизна формула)
    # При 20°C: D_salt ≈ 1.5e-9 м²/с
    D_salt_ref = 1.5e-9  # м²/с при 20°C
    T_ref = 20 + 273.15  # К
    T = temperature + 273.15  # К
    
    # Температурна залежність: D ~ T^1.5 / μ
    # де μ - динамічна в'язкість
    mu = seawater_viscosity(temperature, salinity)
    mu_ref = seawater_viscosity(20, 35)
    
    D_salt = D_salt_ref * (T / T_ref)**1.5 * (mu_ref / mu)
    return D_salt

def seawater_turbulent_diffusion_coefficient(velocity_scale: float, 
                                           length_scale: float) -> float:
    """
    Обчислити турбулентний коефіцієнт дифузії.
    
    Параметри:
        velocity_scale: Швидкісний масштаб (м/с)
        length_scale: Масштаб довжини (м)
    
    Повертає:
        Турбулентний коефіцієнт дифузії (м²/с)
    """
    if velocity_scale < 0:
        raise ValueError("Швидкісний масштаб повинен бути невід'ємним")
    if length_scale <= 0:
        raise ValueError("Масштаб довжини повинен бути додатнім")
    
    # Турбулентний коефіцієнт дифузії = швидкісний масштаб * масштаб довжини
    K_turb = velocity_scale * length_scale
    return K_turb

def seawater_mixed_layer_entrainment(temperature_surface: float, 
                                   temperature_below: float,
                                   wind_stress: float) -> float:
    """
    Обчислити швидкість втрати маси змішаного шару.
    
    Параметри:
        temperature_surface: Температура поверхні (°C)
        temperature_below: Температура нижче змішаного шару (°C)
        wind_stress: Напруження вітру (Па)
    
    Повертає:
        Швидкість втрати маси змішаного шару (м/с)
    """
    if temperature_surface < -2 or temperature_surface > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if temperature_below < -2 or temperature_below > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if wind_stress < 0:
        raise ValueError("Напруження вітру повинно бути невід'ємним")
    
    # Різниця температур
    delta_T = abs(temperature_surface - temperature_below)
    
    # Уникнути ділення на нуль
    if delta_T < 1e-10:
        return 0
    
    # Приблизна формула для швидкості втрати маси
    # E = (τ * κ) / (ρ * Cp * ΔT)
    # де τ - напруження вітру, κ - теплопровідність, ρ - густина, Cp - теплоємність
    rho = SEAWATER_DENSITY
    Cp = SEAWATER_SPECIFIC_HEAT
    kappa = 0.6  # Приблизна теплопровідність морської води (Вт/(м·К))
    
    entrainment = (wind_stress * kappa) / (rho * Cp * delta_T)
    return entrainment

def seawater_heat_flux(temperature_surface: float, temperature_air: float,
                     wind_speed: float, humidity_surface: float, 
                     humidity_air: float) -> Dict[str, float]:
    """
    Обчислити теплові потоки на поверхні океану.
    
    Параметри:
        temperature_surface: Температура поверхні океану (°C)
        temperature_air: Температура повітря (°C)
        wind_speed: Швидкість вітру (м/с)
        humidity_surface: Вологість поверхні океану (відносна, 0-1)
        humidity_air: Вологість повітря (відносна, 0-1)
    
    Повертає:
        Словник з тепловими потоками (Вт/м²):
        - sensible: Чутливий тепловий потік
        - latent: Латентний тепловий потік
        - total: Загальний тепловий потік
    """
    if temperature_surface < -2 or temperature_surface > 40:
        raise ValueError("Температура поверхні повинна бути в діапазоні -2°C до 40°C")
    if temperature_air < -50 or temperature_air > 50:
        raise ValueError("Температура повітря повинна бути в діапазоні -50°C до 50°C")
    if wind_speed < 0:
        raise ValueError("Швидкість вітру повинна бути невід'ємною")
    if humidity_surface < 0 or humidity_surface > 1:
        raise ValueError("Вологість поверхні повинна бути в діапазоні 0-1")
    if humidity_air < 0 or humidity_air > 1:
        raise ValueError("Вологість повітря повинна бути в діапазоні 0-1")
    
    # Густина повітря (приблизна)
    rho_air = 1.225  # кг/м³
    
    # Питома теплоємність повітря при постійному тиску
    Cp_air = 1005  # Дж/(кг·К)
    
    # Коефіцієнт тепловіддачі (приблизна формула)
    # h = 11.5 + 0.8 * wind_speed для швидкостей 1-20 м/с
    h_sensible = 11.5 + 0.8 * wind_speed
    
    # Чутливий тепловий потік
    sensible_flux = h_sensible * (temperature_surface - temperature_air)
    
    # Латентний тепловий потік
    # Приблизна формула
    latent_heat_vaporization = 2.5e6  # Дж/кг
    h_latent = 0.03 * wind_speed  # Приблизна формула для коефіцієнта масовіддачі
    latent_flux = h_latent * latent_heat_vaporization * (humidity_surface - humidity_air)
    
    # Загальний тепловий потік
    total_flux = sensible_flux + latent_flux
    
    return {
        'sensible': sensible_flux,
        'latent': latent_flux,
        'total': total_flux
    }

def seawater_carbonate_system(pH: float, temperature: float, salinity: float,
                            total_alkalinity: float, dissolved_inorganic_carbon: float) -> Dict[str, float]:
    """
    Обчислити параметри карбонатної системи морської води (спрощена версія).
    
    Параметри:
        pH: Водневий показник
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
        total_alkalinity: Загальна лужність (μmol/kg)
        dissolved_inorganic_carbon: Розчинений неорганічний вуглець (μmol/kg)
    
    Повертає:
        Словник з параметрами карбонатної системи:
        - CO2: Концентрація CO₂ (μmol/kg)
        - HCO3: Концентрація HCO₃⁻ (μmol/kg)
        - CO3: Концентрація CO₃²⁻ (μmol/kg)
        - Omega_aragonite: Насичення арагоніту
        - Omega_calcite: Насичення кальциту
    """
    if pH < 6 or pH > 9:
        raise ValueError("pH повинен бути в діапазоні 6-9")
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    if total_alkalinity < 0:
        raise ValueError("Загальна лужність повинна бути невід'ємною")
    if dissolved_inorganic_carbon < 0:
        raise ValueError("Розчинений неорганічний вуглець повинен бути невід'ємним")
    
    # Спрощені формули для карбонатної системи
    # У реальних океанографічних розрахунках використовують складніші алгоритми
    
    # Константи дисоціації (приблизні)
    K1 = 10**(-6.35)  # Перша константа дисоціації H₂CO₃
    K2 = 10**(-10.33)  # Друга константа дисоціації HCO₃⁻
    
    # Концентрація H⁺
    H = 10**(-pH)
    
    # Спрощені рівняння для розподілу DIC
    # [CO₂] = DIC / (1 + K1/H + K1*K2/H²)
    # [HCO₃⁻] = DIC / (H/K1 + 1 + K2/H)
    # [CO₃²⁻] = DIC / (H²/(K1*K2) + H/K2 + 1)
    
    DIC = dissolved_inorganic_carbon
    
    CO2 = DIC / (1 + K1/H + K1*K2/H**2)
    HCO3 = DIC / (H/K1 + 1 + K2/H)
    CO3 = DIC / (H**2/(K1*K2) + H/K2 + 1)
    
    # Насичення мінералів (дуже спрощено)
    Omega_aragonite = CO3 / 100  # Приблизна концентрація насичення арагоніту
    Omega_calcite = CO3 / 150    # Приблизна концентрація насичення кальциту
    
    return {
        'CO2': CO2,
        'HCO3': HCO3,
        'CO3': CO3,
        'Omega_aragonite': Omega_aragonite,
        'Omega_calcite': Omega_calcite
    }

def seawater_acidity(pH: float, temperature: float, salinity: float) -> Dict[str, float]:
    """
    Обчислити кислотність морської води.
    
    Параметри:
        pH: Водневий показник
        temperature: Температура води (°C)
        salinity: Практична солоність (‰)
    
    Повертає:
        Словник з параметрами кислотності:
        - pH: Водневий показник
        - pCO2: Парціальний тиск CO₂ (μatm)
        - H: Концентрація H⁺ (mol/kg)
        - OH: Концентрація OH⁻ (mol/kg)
    """
    if pH < 6 or pH > 9:
        raise ValueError("pH повинен бути в діапазоні 6-9")
    if temperature < -2 or temperature > 40:
        raise ValueError("Температура повинна бути в діапазоні -2°C до 40°C")
    if salinity < 0 or salinity > 42:
        raise ValueError("Солоність повинна бути в діапазоні 0‰ до 42‰")
    
    # Концентрація H⁺
    H = 10**(-pH)
    
    # Концентрація OH⁻ (Kw = 10^(-14) при 25°C)
    # Температурна корекція Kw
    T_K = temperature + 273.15
    Kw = 10**(-14 * (298.15 / T_K))
    OH = Kw / H
    
    # Приблизна оцінка pCO₂ (для pH ~ 8.1)
    pCO2 = 10**(pH - 8.1 + 3.5)  # μatm
    
    return {
        'pH': pH,
        'pCO2': pCO2,
        'H': H,
        'OH': OH
    }