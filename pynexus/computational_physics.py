"""
Модуль для обчислювальної фізики в PyNexus.
Включає функції для моделювання фізичних систем, розв'язання рівнянь фізики,
численних методів механіки, термодинаміки, електромагнетизму та квантової механіки.
"""

import math
import numpy as np
from typing import List, Tuple, Callable, Union, Optional
from scipy import constants, integrate, optimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Класична механіка
def newton_laws_motion(position_func: Callable[[float], List[float]], 
                      time_points: List[float]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Обчислення кінематичних величин на основі законів Ньютона.
    
    Параметри:
        position_func: Функція положення r(t) = [x(t), y(t), z(t)]
        time_points: Точки часу
    
    Повертає:
        Кортеж (positions, velocities, accelerations) з кінематичними величинами
    """
    positions = []
    velocities = []
    accelerations = []
    
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 1e-6
    
    for i, t in enumerate(time_points):
        # Положення
        r = position_func(t)
        positions.append(r)
        
        # Швидкість (численна похідна)
        if i == 0:
            v = [(position_func(t + dt)[j] - r[j]) / dt for j in range(len(r))]
        elif i == len(time_points) - 1:
            v = [(r[j] - position_func(t - dt)[j]) / dt for j in range(len(r))]
        else:
            v = [(position_func(t + dt)[j] - position_func(t - dt)[j]) / (2 * dt) for j in range(len(r))]
        velocities.append(v)
        
        # Прискорення (численна похідна)
        if i == 0:
            a = [(velocities[i+1][j] - v[j]) / dt for j in range(len(v))]
        elif i == len(time_points) - 1:
            a = [(v[j] - velocities[i-1][j]) / dt for j in range(len(v))]
        else:
            a = [(velocities[i+1][j] - velocities[i-1][j]) / (2 * dt) for j in range(len(v))]
        accelerations.append(a)
    
    return (positions, velocities, accelerations)

def kepler_orbits(semi_major_axis: float, 
                 eccentricity: float, 
                 true_anomaly: float) -> Tuple[float, float, float]:
    """
    Обчислення параметрів орбіти Кеплера.
    
    Параметри:
        semi_major_axis: Велика піввісь
        eccentricity: Ексцентриситет
        true_anomaly: Істинна аномалія (радіани)
    
    Повертає:
        Кортеж (radius, velocity, period) з радіусом, швидкістю та періодом
    """
    # Радіус орбіти
    radius = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * math.cos(true_anomaly))
    
    # Орбітальна швидкість (з закону збереження енергії)
    mu = constants.G * constants.M_sun  # Гравітаційний параметр для Сонця
    velocity = math.sqrt(mu * (2/radius - 1/semi_major_axis))
    
    # Орбітальний період
    period = 2 * math.pi * math.sqrt(semi_major_axis**3 / mu)
    
    return (radius, velocity, period)

def rigid_body_dynamics(inertia_tensor: List[List[float]], 
                       initial_angular_velocity: List[float], 
                       external_torque: List[float], 
                       time_span: Tuple[float, float], 
                       n_points: int = 1000) -> Tuple[List[float], List[List[float]]]:
    """
    Динаміка твердого тіла (рівняння Ейлера).
    
    Параметри:
        inertia_tensor: Тензор інерції 3x3
        initial_angular_velocity: Початкова кутова швидкість [wx, wy, wz]
        external_torque: Зовнішній момент [tx, ty, tz]
        time_span: Інтервал часу (t0, t_end)
        n_points: Кількість точок
    
    Повертає:
        Кортеж (time_points, angular_velocities) з розв'язком
    """
    def euler_equations(omega, t):
        """Рівняння Ейлера для твердого тіла"""
        I = np.array(inertia_tensor)
        omega_vec = np.array(omega)
        torque = np.array(external_torque)
        
        # I * dω/dt + ω × (I * ω) = τ
        # dω/dt = I^(-1) * (τ - ω × (I * ω))
        I_omega = np.dot(I, omega_vec)
        cross_product = np.cross(omega_vec, I_omega)
        domega_dt = np.linalg.solve(I, torque - cross_product)
        
        return domega_dt
    
    t = np.linspace(time_span[0], time_span[1], n_points)
    solution = odeint(euler_equations, initial_angular_velocity, t)
    
    return (t.tolist(), solution.tolist())

# Термодинаміка та статистична фізика
def maxwell_boltzmann_distribution(temperature: float, 
                                 particle_mass: float, 
                                 velocities: List[float]) -> List[float]:
    """
    Розподіл Максвелла-Больцмана для швидкостей частинок.
    
    Параметри:
        temperature: Температура (К)
        particle_mass: Маса частинки (кг)
        velocities: Список швидкостей (м/с)
    
    Повертає:
        Список значень функції розподілу
    """
    k_B = constants.Boltzmann  # Постійна Больцмана
    
    # Нормувальний множник
    normalization = (particle_mass / (2 * math.pi * k_B * temperature))**1.5
    
    # Функція розподілу
    distribution = []
    for v in velocities:
        f_v = normalization * 4 * math.pi * v**2 * math.exp(-particle_mass * v**2 / (2 * k_B * temperature))
        distribution.append(f_v)
    
    return distribution

def partition_function(energy_levels: List[float], 
                      temperature: float, 
                      degeneracies: Optional[List[int]] = None) -> float:
    """
    Функція розподілу статистичної фізики.
    
    Параметри:
        energy_levels: Рівні енергії
        temperature: Температура (К)
        degeneracies: Виродження рівнів (за замовчуванням 1)
    
    Повертає:
        Значення функції розподілу
    """
    k_B = constants.Boltzmann
    
    if degeneracies is None:
        degeneracies = [1] * len(energy_levels)
    
    Z = 0.0
    for i, energy in enumerate(energy_levels):
        Z += degeneracies[i] * math.exp(-energy / (k_B * temperature))
    
    return Z

def thermodynamic_properties(partition_function_func: Callable[[float], float], 
                           temperature: float, 
                           delta_t: float = 1e-6) -> dict:
    """
    Обчислення термодинамічних властивостей з функції розподілу.
    
    Параметри:
        partition_function_func: Функція розподілу Z(T)
        temperature: Температура (К)
        delta_t: Крок для чисельного диференціювання
    
    Повертає:
        Словник з термодинамічними властивостями
    """
    k_B = constants.Boltzmann
    
    # Функція розподілу
    Z = partition_function_func(temperature)
    
    # Внутрішня енергія: U = kT^2 * d(lnZ)/dT
    lnZ_plus = math.log(partition_function_func(temperature + delta_t))
    lnZ_minus = math.log(partition_function_func(temperature - delta_t))
    dlnZ_dT = (lnZ_plus - lnZ_minus) / (2 * delta_t)
    internal_energy = k_B * temperature**2 * dlnZ_dT
    
    # Ентропія: S = k * (lnZ + T * d(lnZ)/dT)
    entropy = k_B * (math.log(Z) + temperature * dlnZ_dT)
    
    # Вільна енергія Гельмгольца: F = -kT * lnZ
    helmholtz_free_energy = -k_B * temperature * math.log(Z)
    
    # Теплоємність: Cv = k * T^2 * d^2(lnZ)/dT^2
    lnZ_plus2 = math.log(partition_function_func(temperature + 2*delta_t))
    lnZ_minus2 = math.log(partition_function_func(temperature - 2*delta_t))
    d2lnZ_dT2 = (lnZ_plus2 - 2*lnZ_plus + 2*lnZ_minus - lnZ_minus2) / (4 * delta_t**2)
    heat_capacity = k_B * temperature**2 * d2lnZ_dT2
    
    return {
        'internal_energy': internal_energy,
        'entropy': entropy,
        'helmholtz_free_energy': helmholtz_free_energy,
        'heat_capacity': heat_capacity,
        'partition_function': Z
    }

# Електромагнетизм
def maxwell_equations(electric_field: Callable[[List[float], float], List[float]], 
                     magnetic_field: Callable[[List[float], float], List[float]], 
                     charge_density: Callable[[List[float], float], float], 
                     current_density: Callable[[List[float], float], List[float]], 
                     position: List[float], 
                     time: float) -> dict:
    """
    Перевірка рівнянь Максвелла в диференціальній формі.
    
    Параметри:
        electric_field: Функція E(x, y, z, t) = [Ex, Ey, Ez]
        magnetic_field: Функція B(x, y, z, t) = [Bx, By, Bz]
        charge_density: Функція ρ(x, y, z, t)
        current_density: Функція J(x, y, z, t) = [Jx, Jy, Jz]
        position: Положення [x, y, z]
        time: Час t
    
    Повертає:
        Словник з результатами перевірки рівнянь Максвелла
    """
    epsilon_0 = constants.epsilon_0  # Електрична стала
    mu_0 = constants.mu_0  # Магнітна стала
    c = constants.c  # Швидкість світла
    
    # Малі зміщення для чисельного диференціювання
    dx, dy, dz = 1e-6, 1e-6, 1e-6
    dt = 1e-12
    
    x, y, z = position
    
    # Обчислення часткових похідних методом скінченних різниць
    # Дивергенція електричного поля: ∇·E = ρ/ε₀
    dEx_dx = (electric_field([x+dx, y, z], time)[0] - electric_field([x-dx, y, z], time)[0]) / (2*dx)
    dEy_dy = (electric_field([x, y+dy, z], time)[1] - electric_field([x, y-dy, z], time)[1]) / (2*dy)
    dEz_dz = (electric_field([x, y, z+dz], time)[2] - electric_field([x, y, z-dz], time)[2]) / (2*dz)
    div_E = dEx_dx + dEy_dy + dEz_dz
    
    # Дивергенція магнітного поля: ∇·B = 0
    dBx_dx = (magnetic_field([x+dx, y, z], time)[0] - magnetic_field([x-dx, y, z], time)[0]) / (2*dx)
    dBy_dy = (magnetic_field([x, y+dy, z], time)[1] - magnetic_field([x, y-dy, z], time)[1]) / (2*dy)
    dBz_dz = (magnetic_field([x, y, z+dz], time)[2] - magnetic_field([x, y, z-dz], time)[2]) / (2*dz)
    div_B = dBx_dx + dBy_dy + dBz_dz
    
    # Ротор електричного поля: ∇×E = -∂B/∂t
    # x-компонента: ∂Ez/∂y - ∂Ey/∂z
    dEz_dy = (electric_field([x, y+dy, z], time)[2] - electric_field([x, y-dy, z], time)[2]) / (2*dy)
    dEy_dz = (electric_field([x, y, z+dz], time)[1] - electric_field([x, y, z-dz], time)[1]) / (2*dz)
    curl_E_x = dEz_dy - dEy_dz
    
    # y-компонента: ∂Ex/∂z - ∂Ez/∂x
    dEx_dz = (electric_field([x, y, z+dz], time)[0] - electric_field([x, y, z-dz], time)[0]) / (2*dz)
    dEz_dx = (electric_field([x+dx, y, z], time)[2] - electric_field([x-dx, y, z], time)[2]) / (2*dx)
    curl_E_y = dEx_dz - dEz_dx
    
    # z-компонента: ∂Ey/∂x - ∂Ex/∂y
    dEy_dx = (electric_field([x+dx, y, z], time)[1] - electric_field([x-dx, y, z], time)[1]) / (2*dx)
    dEx_dy = (electric_field([x, y+dy, z], time)[0] - electric_field([x, y-dy, z], time)[0]) / (2*dy)
    curl_E_z = dEy_dx - dEx_dy
    
    curl_E = [curl_E_x, curl_E_y, curl_E_z]
    
    # ∂B/∂t
    dBx_dt = (magnetic_field(position, time+dt)[0] - magnetic_field(position, time-dt)[0]) / (2*dt)
    dBy_dt = (magnetic_field(position, time+dt)[1] - magnetic_field(position, time-dt)[1]) / (2*dt)
    dBz_dt = (magnetic_field(position, time+dt)[2] - magnetic_field(position, time-dt)[2]) / (2*dt)
    dB_dt = [dBx_dt, dBy_dt, dBz_dt]
    
    # Ротор магнітного поля: ∇×B = μ₀J + μ₀ε₀∂E/∂t
    # x-компонента: ∂Bz/∂y - ∂By/∂z
    dBz_dy = (magnetic_field([x, y+dy, z], time)[2] - magnetic_field([x, y-dy, z], time)[2]) / (2*dy)
    dBy_dz = (magnetic_field([x, y, z+dz], time)[1] - magnetic_field([x, y, z-dz], time)[1]) / (2*dz)
    curl_B_x = dBz_dy - dBy_dz
    
    # y-компонента: ∂Bx/∂z - ∂Bz/∂x
    dBx_dz = (magnetic_field([x, y, z+dz], time)[0] - magnetic_field([x, y, z-dz], time)[0]) / (2*dz)
    dBz_dx = (magnetic_field([x+dx, y, z], time)[2] - magnetic_field([x-dx, y, z], time)[2]) / (2*dx)
    curl_B_y = dBx_dz - dBz_dx
    
    # z-компонента: ∂By/∂x - ∂Bx/∂y
    dBy_dx = (magnetic_field([x+dx, y, z], time)[1] - magnetic_field([x-dx, y, z], time)[1]) / (2*dx)
    dBx_dy = (magnetic_field([x, y+dy, z], time)[0] - magnetic_field([x, y-dy, z], time)[0]) / (2*dy)
    curl_B_z = dBy_dx - dBx_dy
    
    curl_B = [curl_B_x, curl_B_y, curl_B_z]
    
    # ∂E/∂t
    dEx_dt = (electric_field(position, time+dt)[0] - electric_field(position, time-dt)[0]) / (2*dt)
    dEy_dt = (electric_field(position, time+dt)[1] - electric_field(position, time-dt)[1]) / (2*dt)
    dEz_dt = (electric_field(position, time+dt)[2] - electric_field(position, time-dt)[2]) / (2*dt)
    dE_dt = [dEx_dt, dEy_dt, dEz_dt]
    
    # Перевірка рівнянь Максвелла
    gauss_law_check = abs(div_E - charge_density(position, time) / epsilon_0) < 1e-10
    gauss_magnetism_check = abs(div_B) < 1e-10
    faraday_law_check = all(abs(curl_E[i] + dB_dt[i]) < 1e-10 for i in range(3))
    ampere_maxwell_law_check = all(abs(curl_B[i] - mu_0 * current_density(position, time)[i] - 
                                      mu_0 * epsilon_0 * dE_dt[i]) < 1e-10 for i in range(3))
    
    return {
        'gauss_law': gauss_law_check,
        'gauss_magnetism': gauss_magnetism_check,
        'faraday_law': faraday_law_check,
        'ampere_maxwell_law': ampere_maxwell_law_check,
        'div_E': div_E,
        'div_B': div_B,
        'curl_E': curl_E,
        'dB_dt': dB_dt,
        'curl_B': curl_B,
        'dE_dt': dE_dt,
        'charge_density': charge_density(position, time),
        'current_density': current_density(position, time)
    }

def electromagnetic_wave_propagation(wave_vector: List[float], 
                                   angular_frequency: float, 
                                   polarization: List[float]) -> dict:
    """
    Властивості електромагнітної хвилі.
    
    Параметри:
        wave_vector: Хвильовий вектор [kx, ky, kz] (м^-1)
        angular_frequency: Кутова частота (рад/с)
        polarization: Вектор поляризації [Ex, Ey, Ez]
    
    Повертає:
        Словник з властивостями хвилі
    """
    c = constants.c  # Швидкість світла
    
    # Модуль хвильового вектора
    k_magnitude = math.sqrt(sum(ki**2 for ki in wave_vector))
    
    # Напрямок поширення
    direction = [ki / k_magnitude for ki in wave_vector] if k_magnitude > 0 else [0, 0, 1]
    
    # Перевірка дисперсійного співвідношення: ω = ck
    dispersion_relation = abs(angular_frequency - c * k_magnitude) < 1e-6
    
    # Ортогональність поляризації та напрямку поширення
    orthogonality = abs(sum(polarization[i] * direction[i] for i in range(3))) < 1e-10
    
    # Магнітне поле (ортогональне до електричного і напрямку поширення)
    magnetic_field = np.cross(direction, polarization).tolist()
    
    # Інтенсивність (пропорційна |E|^2)
    intensity = sum(ei**2 for ei in polarization)
    
    return {
        'wave_vector_magnitude': k_magnitude,
        'propagation_direction': direction,
        'dispersion_relation_satisfied': dispersion_relation,
        'polarization_orthogonal': orthogonality,
        'magnetic_field': magnetic_field,
        'intensity': intensity,
        'wavelength': 2 * math.pi / k_magnitude if k_magnitude > 0 else float('inf'),
        'frequency': angular_frequency / (2 * math.pi)
    }

# Квантова механіка
def schrodinger_equation_1d(potential_func: Callable[[float], float], 
                           mass: float, 
                           x_range: Tuple[float, float], 
                           n_points: int = 1000, 
                           n_states: int = 5) -> Tuple[List[float], List[List[float]]]:
    """
    Розв'язання одновимірного рівняння Шредінгера методом скінченних різниць.
    
    Параметри:
        potential_func: Функція потенціалу V(x)
        mass: Маса частинки
        x_range: Діапазон координат (x_min, x_max)
        n_points: Кількість точок
        n_states: Кількість станів для обчислення
    
    Повертає:
        Кортеж (energies, wave_functions) з власними значеннями та функціями
    """
    hbar = constants.hbar  # Зведена постійна Планка
    x_min, x_max = x_range
    dx = (x_max - x_min) / (n_points - 1)
    
    # Створення матриці гамільтоніана
    H = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        x = x_min + i * dx
        
        # Діагональний елемент
        H[i, i] = -hbar**2 / (2 * mass * dx**2) * (-2) + potential_func(x)
        
        # Недіагональні елементи
        if i > 0:
            H[i, i-1] = -hbar**2 / (2 * mass * dx**2)
        if i < n_points - 1:
            H[i, i+1] = -hbar**2 / (2 * mass * dx**2)
    
    # Знаходження власних значень та векторів
    energies, wave_functions = np.linalg.eigh(H)
    
    # Сортування за зростанням енергії
    sorted_indices = np.argsort(energies)
    energies = energies[sorted_indices][:n_states]
    wave_functions = wave_functions[:, sorted_indices][:, :n_states]
    
    # Нормалізація хвильових функцій
    x_points = np.linspace(x_min, x_max, n_points)
    normalized_wave_functions = []
    
    for i in range(n_states):
        # Нормалізація
        norm = np.trapz(wave_functions[:, i]**2, x_points)
        if norm > 0:
            normalized_psi = wave_functions[:, i] / np.sqrt(norm)
        else:
            normalized_psi = wave_functions[:, i]
        normalized_wave_functions.append(normalized_psi.tolist())
    
    return (energies.tolist(), normalized_wave_functions)

def quantum_harmonic_oscillator(n: int, 
                               mass: float, 
                               omega: float, 
                               x_points: List[float]) -> List[float]:
    """
    Хвильова функція квантового гармонічного осцилятора.
    
    Параметри:
        n: Квантове число (n = 0, 1, 2, ...)
        mass: Маса частинки
        omega: Кутова частота
        x_points: Точки координат
    
    Повертає:
        Значення хвильової функції в точках x
    """
    hbar = constants.hbar
    
    # Параметри осцилятора
    alpha = mass * omega / hbar
    
    # Енергія рівня
    energy = hbar * omega * (n + 0.5)
    
    # Нормувальний множник
    normalization = (alpha / math.pi)**0.25 / math.sqrt(2**n * math.factorial(n))
    
    # Хвильова функція
    wave_function = []
    for x in x_points:
        # Поліном Ерміта
        hermite = 0.0
        for k in range(n//2 + 1):
            sign = (-1)**k
            coeff = math.factorial(n) / (math.factorial(k) * math.factorial(n - 2*k))
            hermite += sign * coeff * (2**(n - 2*k)) * (math.sqrt(alpha) * x)**(n - 2*k)
        
        # Хвильова функція
        psi = normalization * hermite * math.exp(-alpha * x**2 / 2)
        wave_function.append(psi)
    
    return wave_function

def quantum_tunneling_barrier(energy: float, 
                            barrier_height: float, 
                            barrier_width: float, 
                            mass: float) -> float:
    """
    Коефіцієнт прозорості потенціального бар'єра.
    
    Параметри:
        energy: Енергія частинки
        barrier_height: Висота бар'єра
        barrier_width: Ширина бар'єра
        mass: Маса частинки
    
    Повертає:
        Коефіцієнт прозорості (ймовірність тунелювання)
    """
    hbar = constants.hbar
    
    if energy >= barrier_height:
        # Класичний випадок - частинка проходить
        return 1.0
    else:
        # Квантове тунелювання
        # k = sqrt(2m(V-E))/hbar
        k = math.sqrt(2 * mass * (barrier_height - energy)) / hbar
        # T = 1 / (1 + V₀² * sinh²(k*a) / (4*E*(V₀-E)))
        sinh_ka = math.sinh(k * barrier_width)
        transmission = 1.0 / (1.0 + (barrier_height**2 * sinh_ka**2) / (4 * energy * (barrier_height - energy)))
        return transmission

# Статистична фізика
def boltzmann_distribution(energy_levels: List[float], 
                         temperature: float, 
                         degeneracies: Optional[List[int]] = None) -> List[float]:
    """
    Розподіл Больцмана для рівнів енергії.
    
    Параметри:
        energy_levels: Рівні енергії
        temperature: Температура (К)
        degeneracies: Виродження рівнів (за замовчуванням 1)
    
    Повертає:
        Список ймовірностей населення рівнів
    """
    k_B = constants.Boltzmann
    
    if degeneracies is None:
        degeneracies = [1] * len(energy_levels)
    
    # Обчислення функції розподілу
    boltzmann_factors = [degeneracies[i] * math.exp(-energy_levels[i] / (k_B * temperature)) 
                        for i in range(len(energy_levels))]
    
    # Нормалізація
    partition_function = sum(boltzmann_factors)
    
    probabilities = [bf / partition_function for bf in boltzmann_factors]
    
    return probabilities

def fermi_dirac_distribution(energy: float, 
                           fermi_energy: float, 
                           temperature: float) -> float:
    """
    Розподіл Фермі-Дірака для ферміонів.
    
    Параметри:
        energy: Енергія
        fermi_energy: Енергія Фермі
        temperature: Температура (К)
    
    Повертає:
        Ймовірність заповнення стану
    """
    k_B = constants.Boltzmann
    
    if temperature == 0:
        return 1.0 if energy < fermi_energy else 0.0
    
    exponent = (energy - fermi_energy) / (k_B * temperature)
    # Уникаємо переповнення для великих значень експоненти
    if exponent > 700:  # Приблизно exp(700) = 10^304
        return 0.0
    
    return 1.0 / (math.exp(exponent) + 1.0)

def bose_einstein_distribution(energy: float, 
                             chemical_potential: float, 
                             temperature: float) -> float:
    """
    Розподіл Бозе-Ейнштейна для бозонів.
    
    Параметри:
        energy: Енергія
        chemical_potential: Хімічний потенціал
        temperature: Температура (К)
    
    Повертає:
        Середнє число частинок у стані
    """
    k_B = constants.Boltzmann
    
    if temperature == 0:
        return float('inf') if energy <= chemical_potential else 0.0
    
    exponent = (energy - chemical_potential) / (k_B * temperature)
    # Уникаємо переповнення для великих значень експоненти
    if exponent > 700:  # Приблизно exp(700) = 10^304
        return 0.0
    
    # Уникаємо сингулярності при exp(exponent) = 1
    if abs(exponent) < 1e-10:
        return float('inf')
    
    return 1.0 / (math.exp(exponent) - 1.0)

# Релятивістська фізика
def lorentz_transformation(time: float, 
                         position: List[float], 
                         velocity: List[float]) -> Tuple[float, List[float]]:
    """
    Перетворення Лоренца для часу та простору.
    
    Параметри:
        time: Час у вихідній системі (с)
        position: Положення [x, y, z] у вихідній системі (м)
        velocity: Відносна швидкість системи [vx, vy, vz] (м/с)
    
    Повертає:
        Кортеж (t', [x', y', z']) у новій системі
    """
    c = constants.c  # Швидкість світла
    
    # Модуль швидкості
    v = math.sqrt(sum(vi**2 for vi in velocity))
    
    if v >= c:
        raise ValueError("Швидкість не може бути більшою або рівною швидкості світла")
    
    # Фактор Лоренца
    gamma = 1.0 / math.sqrt(1 - (v/c)**2)
    
    # Напрямок швидкості
    if v > 0:
        direction = [vi / v for vi in velocity]
    else:
        direction = [0, 0, 1]
    
    # Проекція положення на напрямок швидкості
    r_parallel = sum(position[i] * direction[i] for i in range(3))
    
    # Перетворення Лоренца
    t_prime = gamma * (time - v * r_parallel / c**2)
    
    position_prime = []
    for i in range(3):
        # Паралельна компонента
        r_parallel_component = r_parallel * direction[i]
        # Перпендикулярна компонента
        r_perpendicular_component = position[i] - r_parallel_component
        
        # Перетворення
        r_prime = r_perpendicular_component + gamma * (r_parallel_component - v * time * direction[i])
        position_prime.append(r_prime)
    
    return (t_prime, position_prime)

def relativistic_kinematics(rest_mass: float, 
                           velocity: float) -> dict:
    """
    Релятивістська кінематика.
    
    Параметри:
        rest_mass: Маса спокою (кг)
        velocity: Швидкість (м/с)
    
    Повертає:
        Словник з релятивістськими величинами
    """
    c = constants.c
    
    if velocity >= c:
        raise ValueError("Швидкість не може бути більшою або рівною швидкості світла")
    
    # Фактор Лоренца
    gamma = 1.0 / math.sqrt(1 - (velocity/c)**2)
    
    # Релятивістський імпульс
    momentum = gamma * rest_mass * velocity
    
    # Повна енергія
    total_energy = gamma * rest_mass * c**2
    
    # Кінетична енергія
    kinetic_energy = total_energy - rest_mass * c**2
    
    return {
        'lorentz_factor': gamma,
        'momentum': momentum,
        'total_energy': total_energy,
        'kinetic_energy': kinetic_energy,
        'rest_energy': rest_mass * c**2
    }

# Атомна та ядерна фізика
def hydrogen_atom_wave_function(n: int, 
                               l: int, 
                               m: int, 
                               r: float, 
                               theta: float, 
                               phi: float) -> complex:
    """
    Хвильова функція атома водню.
    
    Параметри:
        n: Головне квантове число
        l: Орбітальне квантове число
        m: Магнітне квантове число
        r: Радіальна координата
        theta: Полярний кут
        phi: Азимутальний кут
    
    Повертає:
        Значення хвильової функції ψ(r, θ, φ)
    """
    a0 = constants.physical_constants['Bohr radius'][0]  # Радіус Бора
    alpha = 2.0 / (n * a0)
    
    # Радіальна частина
    # Спрощена форма для демонстрації
    radial_part = math.exp(-alpha * r / 2) * (2 * r / (n * a0))**l
    
    # Кутова частина (сферичні гармоніки)
    # Спрощена форма для m = 0
    if m == 0:
        angular_part = math.sqrt((2*l + 1) / (4 * math.pi)) * math.cos(l * theta)
    else:
        angular_part = (math.cos(m * phi) + 1j * math.sin(m * phi)) * math.sin(l * theta)
    
    # Нормувальний множник
    normalization = math.sqrt((2/(n*a0))**3 * math.factorial(n-l-1) / (2*n*math.factorial(n+l)))
    
    return normalization * radial_part * angular_part

def nuclear_decay_statistics(initial_amount: float, 
                           decay_constant: float, 
                           time_points: List[float]) -> List[float]:
    """
    Статистика радіоактивного розпаду.
    
    Параметри:
        initial_amount: Початкова кількість ядер
        decay_constant: Константа розпаду
        time_points: Точки часу
    
    Повертає:
        Кількість ядер у кожен момент часу
    """
    # Експоненційний розпад: N(t) = N₀ * e^(-λt)
    amounts = [initial_amount * math.exp(-decay_constant * t) for t in time_points]
    
    return amounts

def nuclear_binding_energy(atomic_number: int, 
                         mass_number: int) -> float:
    """
    Енергія зв'язку атомного ядра (формула Вайцзеккера).
    
    Параметри:
        atomic_number: Атомний номер (Z)
        mass_number: Масове число (A)
    
    Повертає:
        Енергія зв'язку (МеВ)
    """
    neutron_number = mass_number - atomic_number
    
    # Константи (МеВ)
    a_v = 15.8  # Об'ємний член
    a_s = 18.3  # Поверхневий член
    a_c = 0.714  # Кулонівський член
    a_a = 23.2  # Асиметрія
    a_p = 12.0  # Парний член
    
    # Об'ємний член
    volume_term = a_v * mass_number
    
    # Поверхневий член
    surface_term = a_s * mass_number**(2/3)
    
    # Кулонівський член
    coulomb_term = a_c * atomic_number * (atomic_number - 1) / (mass_number**(1/3))
    
    # Асиметрія
    asymmetry_term = a_a * (neutron_number - atomic_number)**2 / mass_number
    
    # Парний член
    if atomic_number % 2 == 0 and neutron_number % 2 == 0:
        pairing_term = a_p / mass_number**(3/4)
    elif atomic_number % 2 == 1 and neutron_number % 2 == 1:
        pairing_term = -a_p / mass_number**(3/4)
    else:
        pairing_term = 0
    
    binding_energy = volume_term - surface_term - coulomb_term - asymmetry_term + pairing_term
    
    return binding_energy

# Фізика твердого тіла
def band_structure_1d(lattice_constant: float, 
                     reciprocal_lattice_vector: float, 
                     n_points: int = 100) -> Tuple[List[float], List[float]]:
    """
    Проста модель зонної структури в 1D.
    
    Параметри:
        lattice_constant: Постійна ґратки
        reciprocal_lattice_vector: Вектор оберненої ґратки
        n_points: Кількість точок
    
    Повертає:
        Кортеж (k_points, energies) з хвильовими векторами та енергіями
    """
    # Хвильові вектори в першій зоні Бріллюена
    k_min = -reciprocal_lattice_vector / 2
    k_max = reciprocal_lattice_vector / 2
    k_points = np.linspace(k_min, k_max, n_points)
    
    # Проста модель: E(k) = -2t * cos(ka)
    t = 1.0  # Енергія зв'язку
    energies = [-2 * t * math.cos(k * lattice_constant) for k in k_points]
    
    return (k_points.tolist(), energies)

def debye_model_heat_capacity(temperature: float, 
                            debye_temperature: float) -> float:
    """
    Модель Дебая для теплоємності твердих тіл.
    
    Параметри:
        temperature: Температура (К)
        debye_temperature: Температура Дебая (К)
    
    Повертає:
        Теплоємність (Дж/К)
    """
    k_B = constants.Boltzmann
    
    if temperature == 0:
        return 0.0
    
    # Відношення температур
    theta_ratio = debye_temperature / temperature
    
    # Інтеграл Дебая
    def debye_integral(x):
        if x == 0:
            return 0
        return x**4 * math.exp(x) / (math.exp(x) - 1)**2
    
    # Чисельне інтегрування
    integral_value = integrate.quad(debye_integral, 1e-10, theta_ratio)[0]
    
    # Теплоємність: C_V = 9 * N * k_B * (T/Θ_D)^3 * ∫₀^(Θ_D/T) (x^4 * e^x)/(e^x - 1)^2 dx
    n_atoms = 1  # Для одного атома
    heat_capacity = 9 * n_atoms * k_B * (temperature / debye_temperature)**3 * integral_value
    
    return heat_capacity

# Астрофізика
def stellar_structure_equations(mass: float, 
                               radius: float, 
                               luminosity: float, 
                               surface_temperature: float) -> dict:
    """
    Основні рівняння структури зір.
    
    Параметри:
        mass: Маса зірки (кг)
        radius: Радіус зірки (м)
        luminosity: Світність зірки (Вт)
        surface_temperature: Температура поверхні (К)
    
    Повертає:
        Словник з астрофізичними параметрами
    """
    sigma = constants.Stefan_Boltzmann  # Константа Стефана-Больцмана
    G = constants.G  # Гравітаційна константа
    
    # Ефективна температура
    effective_temperature = (luminosity / (4 * math.pi * radius**2 * sigma))**0.25
    
    # Гравітаційний радіус Шварцшильда
    schwarzschild_radius = 2 * G * mass / constants.c**2
    
    # Середня густина
    volume = 4 * math.pi * radius**3 / 3
    average_density = mass / volume
    
    # Поверхневе прискорення вільного падіння
    surface_gravity = G * mass / radius**2
    
    return {
        'effective_temperature': effective_temperature,
        'schwarzschild_radius': schwarzschild_radius,
        'average_density': average_density,
        'surface_gravity': surface_gravity,
        'mass_luminosity_ratio': mass / luminosity,
        'surface_brightness': sigma * surface_temperature**4
    }

def hubble_law(redshift: float, 
              hubble_constant: float = 70.0) -> float:
    """
    Закон Хаббла для розширення Всесвіту.
    
    Параметри:
        redshift: Червоне зміщення
        hubble_constant: Константа Хаббла (км/с/Мпк)
    
    Повертає:
        Відстань до галактики (Мпк)
    """
    # Для малих червоних зміщень: v = H₀ * d, z = v/c
    speed_of_light_km_s = constants.c / 1000  # км/с
    velocity = redshift * speed_of_light_km_s
    distance = velocity / hubble_constant
    
    return distance

# Фізика плазми
def plasma_parameters(electron_density: float, 
                     electron_temperature: float, 
                     ion_temperature: float = None) -> dict:
    """
    Параметри плазми.
    
    Параметри:
        electron_density: Густина електронів (м^-3)
        electron_temperature: Температура електронів (К)
        ion_temperature: Температура іонів (К, за замовчуванням = electron_temperature)
    
    Повертає:
        Словник з параметрами плазми
    """
    if ion_temperature is None:
        ion_temperature = electron_temperature
    
    k_B = constants.Boltzmann
    e = constants.e  # Заряд електрона
    epsilon_0 = constants.epsilon_0
    m_e = constants.m_e  # Маса електрона
    
    # Дебаївська довжина
    debye_length = math.sqrt(epsilon_0 * k_B * electron_temperature / (e**2 * electron_density))
    
    # Плазмова частота
    plasma_frequency = math.sqrt(e**2 * electron_density / (epsilon_0 * m_e))
    
    # Тепловий показник
    thermal_velocity = math.sqrt(k_B * electron_temperature / m_e)
    
    # Параметр зв'язку (приблизний)
    average_distance = (3 / (4 * math.pi * electron_density))**(1/3)
    coupling_parameter = e**2 / (4 * math.pi * epsilon_0 * average_distance * k_B * electron_temperature)
    
    return {
        'debye_length': debye_length,
        'plasma_frequency': plasma_frequency,
        'thermal_velocity': thermal_velocity,
        'coupling_parameter': coupling_parameter,
        'electron_temperature_ev': electron_temperature * k_B / constants.e,
        'ion_temperature_ev': ion_temperature * k_B / constants.e
    }

def magnetohydrodynamics_equations(velocity_field: List[List[float]], 
                                 magnetic_field: List[List[float]], 
                                 density: List[List[float]], 
                                 pressure: List[List[float]], 
                                 viscosity: float, 
                                 magnetic_diffusivity: float) -> dict:
    """
    Рівняння магнітної гідродинаміки (МГД).
    
    Параметри:
        velocity_field: Поле швидкості
        magnetic_field: Магнітне поле
        density: Густина
        pressure: Тиск
        viscosity: Кінематична в'язкість
        magnetic_diffusivity: Магнітна дифузія
    
    Повертає:
        Словник з параметрами МГД
    """
    # Це спрощена модель для демонстрації
    # Повні рівняння МГД включають:
    # 1. Рівняння неперервності: ∂ρ/∂t + ∇·(ρv) = 0
    # 2. Рівняння руху: ρ(∂v/∂t + v·∇v) = -∇p + J×B + ρg + μ∇²v
    # 3. Рівняння індукції: ∂B/∂t = ∇×(v×B) + η∇²B
    # 4. Рівняння стану: p = p(ρ, T)
    
    # Обчислення кінетичної енергії
    kinetic_energy = 0.5 * sum(rho * sum(vi**2 for vi in v) 
                              for rho, v in zip(density, velocity_field))
    
    # Обчислення магнітної енергії
    magnetic_energy = sum(sum(bi**2 for bi in b) / (2 * constants.mu_0) 
                         for b in magnetic_field)
    
    # Число Рейнольдса (приблизне)
    characteristic_length = 1.0  # Потрібно задати
    characteristic_velocity = 1.0  # Потрібно задати
    reynolds_number = characteristic_velocity * characteristic_length / viscosity
    
    # Число магнітного Рейнольдса
    magnetic_reynolds_number = characteristic_velocity * characteristic_length / magnetic_diffusivity
    
    return {
        'kinetic_energy': kinetic_energy,
        'magnetic_energy': magnetic_energy,
        'reynolds_number': reynolds_number,
        'magnetic_reynolds_number': magnetic_reynolds_number,
        'plasma_beta': pressure[0][0] / (magnetic_field[0][0]**2 / (2 * constants.mu_0)) 
                      if magnetic_field[0][0] != 0 else float('inf')
    }

# Нелінійна динаміка та хаос
def logistic_map(r: float, 
                x0: float, 
                n_iterations: int) -> List[float]:
    """
    Логістичне відображення (дискретна динамічна система).
    
    Параметри:
        r: Параметр управління
        x0: Початкове значення
        n_iterations: Кількість ітерацій
    
    Повертає:
        Список значень x_n
    """
    x_values = [x0]
    x = x0
    
    for _ in range(n_iterations):
        x = r * x * (1 - x)
        x_values.append(x)
    
    return x_values

def lorenz_system(sigma: float, 
                 rho: float, 
                 beta: float, 
                 initial_conditions: List[float], 
                 time_span: Tuple[float, float], 
                 n_points: int = 10000) -> Tuple[List[float], List[List[float]]]:
    """
    Система Лоренца (класична система з хаотичною динамікою).
    
    Параметри:
        sigma: Параметр Прандтля
        rho: Параметр Релея
        beta: Геометричний параметр
        initial_conditions: Початкові умови [x0, y0, z0]
        time_span: Інтервал часу (t0, t_end)
        n_points: Кількість точок
    
    Повертає:
        Кортеж (time_points, solutions) з розв'язком
    """
    def lorenz_equations(state, t):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    t = np.linspace(time_span[0], time_span[1], n_points)
    solution = odeint(lorenz_equations, initial_conditions, t)
    
    return (t.tolist(), solution.tolist())

def fractal_dimension(time_series: List[float], 
                     min_box_size: float = 1e-3, 
                     max_box_size: float = 1.0, 
                     n_box_sizes: int = 20) -> float:
    """
    Обчислення фрактальної розмірності методом рахування ящиків.
    
    Параметри:
        time_series: Часовий ряд
        min_box_size: Мінімальний розмір ящика
        max_box_size: Максимальний розмір ящика
        n_box_sizes: Кількість розмірів ящиків
    
    Повертає:
        Фрактальна розмірність
    """
    # Створення множини точок у фазовому просторі
    # Використовуємо затримкове вкладення
    delay = 10
    embedding_dimension = 3
    
    points = []
    for i in range(len(time_series) - (embedding_dimension - 1) * delay):
        point = [time_series[i + j * delay] for j in range(embedding_dimension)]
        points.append(point)
    
    if len(points) == 0:
        return 0.0
    
    # Створення масиву розмірів ящиків
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), n_box_sizes)
    
    # Підрахунок кількості зайнятих ящиків для кожного розміру
    box_counts = []
    
    for box_size in box_sizes:
        # Створення сітки ящиків
        min_coords = [min(p[i] for p in points) for i in range(embedding_dimension)]
        max_coords = [max(p[i] for p in points) for i in range(embedding_dimension)]
        
        # Кількість ящиків у кожному вимірі
        n_boxes_per_dim = [int((max_coords[i] - min_coords[i]) / box_size) + 1 
                          for i in range(embedding_dimension)]
        
        # Відстеження зайнятих ящиків
        occupied_boxes = set()
        
        for point in points:
            # Визначення індексів ящика для точки
            box_indices = tuple(int((point[i] - min_coords[i]) / box_size) 
                               for i in range(embedding_dimension))
            occupied_boxes.add(box_indices)
        
        box_counts.append(len(occupied_boxes))
    
    # Обчислення фрактальної розмірності методом найменших квадратів
    if len(box_counts) < 2:
        return 0.0
    
    log_box_sizes = [math.log(1/bs) for bs in box_sizes]
    log_box_counts = [math.log(bc) for bc in box_counts]
    
    # Лінійна регресія
    n = len(log_box_sizes)
    sum_x = sum(log_box_sizes)
    sum_y = sum(log_box_counts)
    sum_xy = sum(log_box_sizes[i] * log_box_counts[i] for i in range(n))
    sum_x2 = sum(x**2 for x in log_box_sizes)
    
    if n * sum_x2 - sum_x**2 == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    
    return slope

# Квантові обчислення
def quantum_fourier_transform(qubit_states: List[complex]) -> List[complex]:
    """
    Квантове перетворення Фур'є.
    
    Параметри:
        qubit_states: Стан кубітів
    
    Повертає:
        Результат перетворення Фур'є
    """
    n = len(qubit_states)
    result = [0j] * n
    
    for k in range(n):
        for j in range(n):
            # QFT: |k⟩ → (1/√N) Σ_j exp(2πijk/N) |j⟩
            phase = 2 * math.pi * j * k / n
            result[k] += qubit_states[j] * (math.cos(phase) + 1j * math.sin(phase))
        result[k] /= math.sqrt(n)
    
    return result

def quantum_entanglement_measure(state_vector: List[complex], 
                               subsystem_a_indices: List[int], 
                               subsystem_b_indices: List[int]) -> float:
    """
    Вимірювання квантової заплутаності (ентропія заплутаності).
    
    Параметри:
        state_vector: Вектор стану системи
        subsystem_a_indices: Індекси підсистеми A
        subsystem_b_indices: Індекси підсистеми B
    
    Повертає:
        Ентропія заплутаності
    """
    # Це спрощена реалізація для демонстрації
    # Повне обчислення вимагає побудови матриці густини та її редукції
    
    total_qubits = len(subsystem_a_indices) + len(subsystem_b_indices)
    
    if len(state_vector) != 2**total_qubits:
        raise ValueError("Невідповідний розмір вектора стану")
    
    # Для максимально заплутаного стану (наприклад, стан Белла)
    # Ентропія заплутаності = 1 біт
    
    # Для розділюваного стану ентропія = 0
    
    # Проста оцінка на основі норми
    norm = math.sqrt(sum(abs(amplitude)**2 for amplitude in state_vector))
    
    # Нормалізація
    normalized_state = [amp / norm for amp in state_vector]
    
    # Ентропія заплутаності (спрощена оцінка)
    entanglement_entropy = 0.0
    for amplitude in normalized_state:
        prob = abs(amplitude)**2
        if prob > 1e-12:  # Уникаємо log(0)
            entanglement_entropy -= prob * math.log2(prob)
    
    return min(entanglement_entropy, 1.0)  # Обмежуємо максимальним значенням

# Фізика конденсованих середовищ
def superconductivity_gap(temperature: float, 
                         critical_temperature: float, 
                         zero_temperature_gap: float) -> float:
    """
    Енергетичний зазор у надпровідниках (модель БКШ).
    
    Параметри:
        temperature: Температура (К)
        critical_temperature: Критична температура (К)
        zero_temperature_gap: Зазор при T=0 (меВ)
    
    Повертає:
        Енергетичний зазор при заданій температурі (меВ)
    """
    if temperature >= critical_temperature:
        return 0.0
    
    # Модель БКШ: Δ(T) = Δ(0) * sqrt(1 - (T/Tc)^2)
    reduced_temperature = temperature / critical_temperature
    gap = zero_temperature_gap * math.sqrt(1 - reduced_temperature**2)
    
    return gap

def quantum_hall_effect(fermi_energy: float, 
                       magnetic_field: float, 
                       electron_density: float) -> dict:
    """
    Ефект квантового холу.
    
    Параметри:
        fermi_energy: Енергія Фермі (меВ)
        magnetic_field: Магнітне поле (Тл)
        electron_density: Густина електронів (м^-2)
    
    Повертає:
        Словник з параметрами ефекту квантового холу
    """
    h = constants.h  # Постійна Планка
    e = constants.e  # Заряд електрона
    
    # Магнітний поток квантування
    flux_quantum = h / e
    
    # Фактор заповнення
    filling_factor = electron_density * flux_quantum / magnetic_field
    
    # Опір Холу
    hall_resistance = magnetic_field / (electron_density * e)
    
    # Квантований опір Холу (якщо фактор заповнення цілий)
    if abs(filling_factor - round(filling_factor)) < 0.1:
        quantized_hall_resistance = h / (round(filling_factor) * e**2)
    else:
        quantized_hall_resistance = float('nan')
    
    return {
        'filling_factor': filling_factor,
        'hall_resistance': hall_resistance,
        'quantized_hall_resistance': quantized_hall_resistance,
        'landau_level_energy': h * magnetic_field / (2 * math.pi * constants.m_e)
    }

# Біофізика
def hodgkin_huxley_model(membrane_potential: float, 
                        time_span: Tuple[float, float], 
                        n_points: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Модель Ходжкіна-Хакслі для нервового імпульсу.
    
    Параметри:
        membrane_potential: Початковий мембранний потенціал (мВ)
        time_span: Інтервал часу (мс)
        n_points: Кількість точок
    
    Повертає:
        Кортеж (time_points, potentials) з розв'язком
    """
    def hh_equations(state, t):
        V, m, h, n = state
        
        # Константи
        C_m = 1.0  # Ємність мембрани (мкФ/см²)
        g_K = 36.0  # Провідність K+ (мСм/см²)
        g_Na = 120.0  # Провідність Na+ (мСм/см²)
        g_L = 0.3  # Провідність витоку (мСм/см²)
        E_K = -12.0  # Рівноважний потенціал K+ (мВ)
        E_Na = 115.0  # Рівноважний потенціал Na+ (мВ)
        E_L = 10.6  # Рівноважний потенціал витоку (мВ)
        
        # Струми
        I_K = g_K * n**4 * (V - E_K)
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_L = g_L * (V - E_L)
        
        # Зовнішній струм (імпульс)
        if 1.0 <= t <= 2.0:
            I_ext = 10.0  # мкА/см²
        else:
            I_ext = 0.0
        
        # Рівняння мембранного потенціалу
        dV_dt = (I_ext - I_K - I_Na - I_L) / C_m
        
        # Кінетичні рівняння для воріт
        alpha_m = 0.1 * (25 - V) / (math.exp((25 - V) / 10) - 1)
        beta_m = 4 * math.exp(-V / 18)
        alpha_h = 0.07 * math.exp(-V / 20)
        beta_h = 1 / (math.exp((30 - V) / 10) + 1)
        alpha_n = 0.01 * (10 - V) / (math.exp((10 - V) / 10) - 1)
        beta_n = 0.125 * math.exp(-V / 80)
        
        dm_dt = alpha_m * (1 - m) - beta_m * m
        dh_dt = alpha_h * (1 - h) - beta_h * h
        dn_dt = alpha_n * (1 - n) - beta_n * n
        
        return [dV_dt, dm_dt, dh_dt, dn_dt]
    
    # Початкові умови
    initial_conditions = [membrane_potential, 0.05, 0.6, 0.32]  # V, m, h, n
    
    t = np.linspace(time_span[0], time_span[1], n_points)
    solution = odeint(hh_equations, initial_conditions, t)
    
    # Повертаємо тільки мембранний потенціал
    potentials = [sol[0] for sol in solution]
    
    return (t.tolist(), potentials)

# Фізика елементарних частинок
def particle_decay_width(initial_mass: float, 
                        final_masses: List[float], 
                        coupling_constant: float) -> float:
    """
    Ширина розпаду елементарної частинки.
    
    Параметри:
        initial_mass: Маса початкової частинки (МеВ/c²)
        final_masses: Маси кінцевих частинок (МеВ/c²)
        coupling_constant: Константа зв'язку
    
    Повертає:
        Ширина розпаду (МеВ)
    """
    hbar = 6.582119569e-22  # Постійна Планка (МеВ·с)
    
    # Перевірка закону збереження енергії
    total_final_mass = sum(final_masses)
    if initial_mass < total_final_mass:
        return 0.0  # Розпад заборонений
    
    # Фазовий простір (спрощений)
    phase_space = (initial_mass - total_final_mass)**2 / initial_mass**2
    
    # Ширина розпаду: Γ = (ℏ/2) * |M|² * фазовий_простір
    decay_width = (hbar / 2) * coupling_constant**2 * phase_space
    
    return decay_width

def cross_section_scattering(energy: float, 
                            target_mass: float, 
                            coupling: float, 
                            scattering_angle: float) -> float:
    """
    Диференціальний переріз розсіяння.
    
    Параметри:
        energy: Енергія налетаючої частинки (МеВ)
        target_mass: Маса мішені (МеВ/c²)
        coupling: Константа зв'язку
        scattering_angle: Кут розсіяння (радіани)
    
    Повертає:
        Диференціальний переріз (барн/стерадіан)
    """
    # Редукована маса
    projectile_mass = 100.0  # Прикладна маса налетаючої частинки
    reduced_mass = (projectile_mass * target_mass) / (projectile_mass + target_mass)
    
    # Імпульс налетаючої частинки
    momentum = math.sqrt(2 * reduced_mass * energy)
    
    # Диференціальний переріз (формула Резерфорда для точкових частинок)
    # dσ/dΩ = (Z₁Z₂αℏc/(4E sin²(θ/2)))²
    # Де α - константа тонкої структури
    
    alpha = 1/137.0  # Константа тонкої структури
    hbar_c = 197.327  # МеВ·фм
    
    # Спрощений вираз
    differential_cross_section = (coupling**2 * hbar_c**2 / (4 * energy))**2 / math.sin(scattering_angle/2)**4
    
    # Перетворення в барни (1 барн = 10^-24 см² = 100 фм²)
    cross_section_barns = differential_cross_section * 100
    
    return cross_section_barns

if __name__ == "__main__":
    # Тестування деяких функцій
    print("Тестування модуля обчислювальної фізики PyNexus")
    
    # Тест кінематики
    def test_position(t):
        return [t, t**2, t**3]
    
    time_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    positions, velocities, accelerations = newton_laws_motion(test_position, time_points)
    print(f"Положення: {positions[2]}")
    print(f"Швидкість: {velocities[2]}")
    print(f"Прискорення: {accelerations[2]}")
    
    # Тест орбіт Кеплера
    radius, velocity, period = kepler_orbits(1.5e11, 0.0167, math.pi/4)
    print(f"Радіус орбіти Землі: {radius:.0f} м")
    print(f"Орбітальна швидкість: {velocity:.0f} м/с")
    print(f"Орбітальний період: {period/31536000:.2f} років")
    
    # Тест розподілу Максвелла-Больцмана
    velocities = np.linspace(0, 2000, 100)
    distribution = maxwell_boltzmann_distribution(300, 4.65e-26, velocities)
    print(f"Максимум розподілу при: {velocities[np.argmax(distribution)]:.0f} м/с")
    
    # Тест рівнянь Максвелла
    def test_e_field(pos, t):
        x, y, z = pos
        return [math.sin(x), math.cos(y), math.sin(z)]
    
    def test_b_field(pos, t):
        x, y, z = pos
        return [math.cos(x), math.sin(y), math.cos(z)]
    
    def test_charge_density(pos, t):
        return 1e-6
    
    def test_current_density(pos, t):
        return [1e-3, 0, 0]
    
    maxwell_results = maxwell_equations(test_e_field, test_b_field, test_charge_density, 
                                       test_current_density, [0, 0, 0], 0)
    print(f"Рівняння Максвелла задовольняються: {sum(maxwell_results[key] for key in 
          ['gauss_law', 'gauss_magnetism', 'faraday_law', 'ampere_maxwell_law']) == 4}")
    
    # Тест квантового осцилятора
    x_points = np.linspace(-5e-9, 5e-9, 100)
    wave_function = quantum_harmonic_oscillator(0, 9.1e-31, 1e14, x_points)
    print(f"Нормалізація хвильової функції: {np.trapz(np.array(wave_function)**2, x_points):.6f}")
    
    print("Тестування завершено!")