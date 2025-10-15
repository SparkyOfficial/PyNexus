"""
Модуль для обчислювальної науки про матеріали
Computational Materials Science Module
"""
from typing import Union, Tuple, List, Optional, Dict, Any
import math

# Фундаментальні константи
# Fundamental constants
BOLTZMANN_CONSTANT = 1.380649e-23  # Константа Больцмана (Дж/К)
PLANCK_CONSTANT = 6.62607015e-34  # Константа Планка (Дж·с)
SPEED_OF_LIGHT = 299792458  # Швидкість світла (м/с)
ELECTRON_CHARGE = 1.602176634e-19  # Заряд електрона (Кл)
ELECTRON_MASS = 9.1093837015e-31  # Маса електрона (кг)
AVOGADRO_CONSTANT = 6.02214076e23  # Число Авогадро (1/моль)
GAS_CONSTANT = 8.31446261815324  # Універсальна газова стала (Дж/(моль·К))
PERMITTIVITY_FREE_SPACE = 8.8541878128e-12  # Електрична стала (Ф/м)
PERMEABILITY_FREE_SPACE = 1.25663706212e-6  # Магнітна стала (Гн/м)

def debye_temperature(debye_frequency: float) -> float:
    """
    Обчислити температуру Дебая.
    
    Параметри:
        debye_frequency: Частота Дебая (Гц)
    
    Повертає:
        Температура Дебая (К)
    """
    if debye_frequency <= 0:
        raise ValueError("Частота Дебая повинна бути додатньою")
    
    return (PLANCK_CONSTANT * debye_frequency) / BOLTZMANN_CONSTANT

def debye_frequency(debye_temperature: float) -> float:
    """
    Обчислити частоту Дебая.
    
    Параметри:
        debye_temperature: Температура Дебая (К)
    
    Повертає:
        Частота Дебая (Гц)
    """
    if debye_temperature <= 0:
        raise ValueError("Температура Дебая повинна бути додатньою")
    
    return (BOLTZMANN_CONSTANT * debye_temperature) / PLANCK_CONSTANT

def specific_heat_debye(temperature: float, debye_temperature: float, 
                       atoms_per_unit: float = 1) -> float:
    """
    Обчислити питому теплоємність за моделлю Дебая.
    
    Параметри:
        temperature: Температура (К)
        debye_temperature: Температура Дебая (К)
        atoms_per_unit: Кількість атомів на одиницю, за замовчуванням 1
    
    Повертає:
        Питома теплоємність (Дж/(кг·К))
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if debye_temperature <= 0:
        raise ValueError("Температура Дебая повинна бути додатньою")
    if atoms_per_unit <= 0:
        raise ValueError("Кількість атомів на одиницю повинна бути додатньою")
    
    # Відношення температур
    theta_ratio = debye_temperature / temperature
    
    # Інтеграл Дебая (спрощене обчислення)
    if theta_ratio > 10:  # При низьких температурах
        # C_v ≈ 12π⁴/5 * R * (T/θ_D)³
        cv = (12 * math.pi**4 / 5) * GAS_CONSTANT * (temperature / debye_temperature)**3
    elif theta_ratio < 0.1:  # При високих температурах
        # C_v ≈ 3R (закон Дюлонга-Пті)
        cv = 3 * GAS_CONSTANT * atoms_per_unit
    else:  # При проміжних температурах
        # Приближене обчислення інтеграла Дебая
        def debye_integral(x):
            if x < 1e-10:
                return 0
            return (x**4 * math.exp(x)) / (math.exp(x) - 1)**2
        
        # Чисельне інтегрування (спрощене)
        integral_sum = 0
        dx = theta_ratio / 100
        for i in range(1, 101):
            x = i * dx
            integral_sum += debye_integral(x) * dx
        
        cv = 9 * GAS_CONSTANT * atoms_per_unit * (temperature / debye_temperature)**3 * integral_sum
    
    return cv

def thermal_conductivity(thermal_diffusivity: float, density: float, 
                        specific_heat: float) -> float:
    """
    Обчислити теплопровідність.
    
    Параметри:
        thermal_diffusivity: Температуропровідність (м²/с)
        density: Густина (кг/м³)
        specific_heat: Питома теплоємність (Дж/(кг·К))
    
    Повертає:
        Теплопровідність (Вт/(м·К))
    """
    if thermal_diffusivity < 0:
        raise ValueError("Температуропровідність повинна бути невід'ємною")
    if density <= 0:
        raise ValueError("Густина повинна бути додатньою")
    if specific_heat < 0:
        raise ValueError("Питома теплоємність повинна бути невід'ємною")
    
    return thermal_diffusivity * density * specific_heat

def electrical_conductivity(resistivity: float) -> float:
    """
    Обчислити електропровідність.
    
    Параметри:
        resistivity: Електричний опір (Ом·м)
    
    Повертає:
        Електропровідність (См/м)
    """
    if resistivity <= 0:
        raise ValueError("Електричний опір повинен бути додатнім")
    
    return 1 / resistivity

def resistivity(conductivity: float) -> float:
    """
    Обчислити електричний опір.
    
    Параметри:
        conductivity: Електропровідність (См/м)
    
    Повертає:
        Електричний опір (Ом·м)
    """
    if conductivity <= 0:
        raise ValueError("Електропровідність повинна бути додатньою")
    
    return 1 / conductivity

def drude_conductivity(relaxation_time: float, electron_density: float) -> float:
    """
    Обчислити електропровідність за моделлю Друде.
    
    Параметри:
        relaxation_time: Час релаксації (с)
        electron_density: Концентрація електронів (1/м³)
    
    Повертає:
        Електропровідність (См/м)
    """
    if relaxation_time <= 0:
        raise ValueError("Час релаксації повинен бути додатнім")
    if electron_density <= 0:
        raise ValueError("Концентрація електронів повинна бути додатньою")
    
    return (electron_density * ELECTRON_CHARGE**2 * relaxation_time) / ELECTRON_MASS

def fermi_energy(electron_density: float) -> float:
    """
    Обчислити енергію Фермі.
    
    Параметри:
        electron_density: Концентрація електронів (1/м³)
    
    Повертає:
        Енергія Фермі (Дж)
    """
    if electron_density <= 0:
        raise ValueError("Концентрація електронів повинна бути додатньою")
    
    # Енергія Фермі для вільного електронного газу
    h_bar = PLANCK_CONSTANT / (2 * math.pi)
    fermi_energy = (h_bar**2 / (2 * ELECTRON_MASS)) * (3 * math.pi**2 * electron_density)**(2/3)
    return fermi_energy

def fermi_temperature(fermi_energy: float) -> float:
    """
    Обчислити температуру Фермі.
    
    Параметри:
        fermi_energy: Енергія Фермі (Дж)
    
    Повертає:
        Температура Фермі (К)
    """
    if fermi_energy <= 0:
        raise ValueError("Енергія Фермі повинна бути додатньою")
    
    return fermi_energy / BOLTZMANN_CONSTANT

def debye_waller_factor(temperature: float, debye_temperature: float, 
                       atomic_mass: float, sound_velocity: float) -> float:
    """
    Обчислити фактор Дебая-Валлера.
    
    Параметри:
        temperature: Температура (К)
        debye_temperature: Температура Дебая (К)
        atomic_mass: Атомна маса (кг)
        sound_velocity: Швидкість звуку (м/с)
    
    Повертає:
        Фактор Дебая-Валлера
    """
    if temperature < 0:
        raise ValueError("Температура повинна бути невід'ємною")
    if debye_temperature <= 0:
        raise ValueError("Температура Дебая повинна бути додатньою")
    if atomic_mass <= 0:
        raise ValueError("Атомна маса повинна бути додатньою")
    if sound_velocity <= 0:
        raise ValueError("Швидкість звуку повинна бути додатньою")
    
    # Характерна частота
    omega_D = (BOLTZMANN_CONSTANT * debye_temperature) / PLANCK_CONSTANT
    
    # Характерна довжина хвилі
    lambda_D = sound_velocity / omega_D
    
    # Фактор Дебая-Валлера (спрощений)
    if temperature > 0:
        B = (PLANCK_CONSTANT / (2 * atomic_mass * sound_velocity)) * (temperature / debye_temperature)
        dw_factor = math.exp(-B)
    else:
        dw_factor = 1.0
    
    return dw_factor

def young_modulus(stress: float, strain: float) -> float:
    """
    Обчислити модуль Юнга.
    
    Параметри:
        stress: Напруження (Па)
        strain: Деформація (безрозмірна)
    
    Повертає:
        Модуль Юнга (Па)
    """
    if strain == 0:
        raise ValueError("Деформація не може дорівнювати нулю")
    
    return stress / strain

def shear_modulus(shear_stress: float, shear_strain: float) -> float:
    """
    Обчислити модуль зсуву.
    
    Параметри:
        shear_stress: Напруження зсуву (Па)
        shear_strain: Деформація зсуву (безрозмірна)
    
    Повертає:
        Модуль зсуву (Па)
    """
    if shear_strain == 0:
        raise ValueError("Деформація зсуву не може дорівнювати нулю")
    
    return shear_stress / shear_strain

def bulk_modulus(pressure: float, volume_change: float, initial_volume: float) -> float:
    """
    Обчислити об'ємний модуль.
    
    Параметри:
        pressure: Тиск (Па)
        volume_change: Зміна об'єму (м³)
        initial_volume: Початковий об'єм (м³)
    
    Повертає:
        Об'ємний модуль (Па)
    """
    if initial_volume <= 0:
        raise ValueError("Початковий об'єм повинен бути додатнім")
    if volume_change == 0:
        raise ValueError("Зміна об'єму не може дорівнювати нулю")
    
    relative_volume_change = volume_change / initial_volume
    return -pressure / relative_volume_change

def poisson_ratio(lateral_strain: float, axial_strain: float) -> float:
    """
    Обчислити коефіцієнт Пуассона.
    
    Параметри:
        lateral_strain: Поперечна деформація (безрозмірна)
        axial_strain: Осьова деформація (безрозмірна)
    
    Повертає:
        Коефіцієнт Пуассона (безрозмірна)
    """
    if axial_strain == 0:
        raise ValueError("Осьова деформація не може дорівнювати нулю")
    
    return -lateral_strain / axial_strain

def stress(force: float, area: float) -> float:
    """
    Обчислити напруження.
    
    Параметри:
        force: Сила (Н)
        area: Площа (м²)
    
    Повертає:
        Напруження (Па)
    """
    if area <= 0:
        raise ValueError("Площа повинна бути додатньою")
    
    return force / area

def strain(change_length: float, original_length: float) -> float:
    """
    Обчислити деформацію.
    
    Параметри:
        change_length: Зміна довжини (м)
        original_length: Початкова довжина (м)
    
    Повертає:
        Деформація (безрозмірна)
    """
    if original_length <= 0:
        raise ValueError("Початкова довжина повинна бути додатньою")
    
    return change_length / original_length

def dislocation_density(burgers_vector: float, dislocation_line_length: float, 
                       area: float) -> float:
    """
    Обчислити густину дислокацій.
    
    Параметри:
        burgers_vector: Вектор Бюргерса (м)
        dislocation_line_length: Довжина лінії дислокації (м)
        area: Площа (м²)
    
    Повертає:
        Густина дислокацій (1/м²)
    """
    if burgers_vector <= 0:
        raise ValueError("Вектор Бюргерса повинен бути додатнім")
    if dislocation_line_length <= 0:
        raise ValueError("Довжина лінії дислокації повинна бути додатньою")
    if area <= 0:
        raise ValueError("Площа повинна бути додатньою")
    
    return dislocation_line_length / area

def grain_boundary_energy(surface_energy: float, grain_boundary_area: float) -> float:
    """
    Обчислити енергію межі зерна.
    
    Параметри:
        surface_energy: Поверхнева енергія (Дж/м²)
        grain_boundary_area: Площа межі зерна (м²)
    
    Повертає:
        Енергія межі зерна (Дж)
    """
    if surface_energy < 0:
        raise ValueError("Поверхнева енергія повинна бути невід'ємною")
    if grain_boundary_area <= 0:
        raise ValueError("Площа межі зерна повинна бути додатньою")
    
    return surface_energy * grain_boundary_area

def diffusion_coefficient(pre_exponential: float, activation_energy: float, 
                         temperature: float) -> float:
    """
    Обчислити коефіцієнт дифузії за рівнянням Арреніуса.
    
    Параметри:
        pre_exponential: Предекспоненційний множник (м²/с)
        activation_energy: Енергія активації (Дж)
        temperature: Температура (К)
    
    Повертає:
        Коефіцієнт дифузії (м²/с)
    """
    if pre_exponential <= 0:
        raise ValueError("Предекспоненційний множник повинен бути додатнім")
    if activation_energy < 0:
        raise ValueError("Енергія активації повинна бути невід'ємною")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    return pre_exponential * math.exp(-activation_energy / (GAS_CONSTANT * temperature))

def vacancy_concentration(vacancy_formation_energy: float, temperature: float) -> float:
    """
    Обчислити концентрацію вакансій.
    
    Параметри:
        vacancy_formation_energy: Енергія утворення вакансії (Дж)
        temperature: Температура (К)
    
    Повертає:
        Концентрація вакансій (безрозмірна)
    """
    if vacancy_formation_energy < 0:
        raise ValueError("Енергія утворення вакансії повинна бути невід'ємною")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    return math.exp(-vacancy_formation_energy / (GAS_CONSTANT * temperature))

def grain_size(hall_petch_slope: float, yield_stress_0: float, 
              yield_stress: float) -> float:
    """
    Обчислити розмір зерна за рівнянням Холла-Петча.
    
    Параметри:
        hall_petch_slope: Коефіцієнт Холла-Петча (Па·м^(1/2))
        yield_stress_0: Межа міцності при нульовому розмірі зерна (Па)
        yield_stress: Межа міцності (Па)
    
    Повертає:
        Розмір зерна (м)
    """
    if hall_petch_slope <= 0:
        raise ValueError("Коефіцієнт Холла-Петча повинен бути додатнім")
    if yield_stress <= yield_stress_0:
        raise ValueError("Межа міцності повинна бути більшою за межу міцності при нульовому розмірі зерна")
    
    d_inv_sqrt = (yield_stress - yield_stress_0) / hall_petch_slope
    grain_size = 1 / (d_inv_sqrt**2)
    return grain_size

def fracture_toughness(stress: float, crack_length: float, 
                      geometry_factor: float = 1.0) -> float:
    """
    Обчислити критичний коефіцієнт інтенсивності напружень.
    
    Параметри:
        stress: Напруження (Па)
        crack_length: Довжина тріщини (м)
        geometry_factor: Геометричний фактор, за замовчуванням 1.0
    
    Повертає:
        Критичний коефіцієнт інтенсивності напружень (Па·м^(1/2))
    """
    if stress < 0:
        raise ValueError("Напруження повинно бути невід'ємним")
    if crack_length <= 0:
        raise ValueError("Довжина тріщини повинна бути додатньою")
    if geometry_factor <= 0:
        raise ValueError("Геометричний фактор повинен бути додатнім")
    
    return stress * geometry_factor * math.sqrt(crack_length)

def creep_rate(stress: float, temperature: float, pre_exponential: float, 
              stress_exponent: float, activation_energy: float) -> float:
    """
    Обчислити швидкість повзучості.
    
    Параметри:
        stress: Напруження (Па)
        temperature: Температура (К)
        pre_exponential: Предекспоненційний множник (1/с)
        stress_exponent: Експонента напруження
        activation_energy: Енергія активації (Дж)
    
    Повертає:
        Швидкість повзучості (1/с)
    """
    if stress < 0:
        raise ValueError("Напруження повинно бути невід'ємним")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pre_exponential <= 0:
        raise ValueError("Предекспоненційний множник повинен бути додатнім")
    if activation_energy < 0:
        raise ValueError("Енергія активації повинна бути невід'ємною")
    
    return pre_exponential * (stress**stress_exponent) * math.exp(-activation_energy / (GAS_CONSTANT * temperature))

def phase_transition_temperature(entropy_change: float, enthalpy_change: float) -> float:
    """
    Обчислити температуру фазового переходу.
    
    Параметри:
        entropy_change: Зміна ентропії (Дж/К)
        enthalpy_change: Зміна ентальпії (Дж)
    
    Повертає:
        Температура фазового переходу (К)
    """
    if entropy_change == 0:
        raise ValueError("Зміна ентропії не може дорівнювати нулю")
    
    return enthalpy_change / entropy_change

def gibbs_free_energy(enthalpy: float, temperature: float, entropy: float) -> float:
    """
    Обчислити вільну енергію Гіббса.
    
    Параметри:
        enthalpy: Ентальпія (Дж)
        temperature: Температура (К)
        entropy: Ентропія (Дж/К)
    
    Повертає:
        Вільна енергія Гіббса (Дж)
    """
    if temperature < 0:
        raise ValueError("Температура повинна бути невід'ємною")
    
    return enthalpy - temperature * entropy

def surface_energy(surface_tension: float, area: float) -> float:
    """
    Обчислити поверхневу енергію.
    
    Параметри:
        surface_tension: Поверхневий натяг (Н/м)
        area: Площа поверхні (м²)
    
    Повертає:
        Поверхнева енергія (Дж)
    """
    if surface_tension < 0:
        raise ValueError("Поверхневий натяг повинен бути невід'ємним")
    if area <= 0:
        raise ValueError("Площа поверхні повинна бути додатньою")
    
    return surface_tension * area

def work_function(fermi_energy: float, surface_potential: float) -> float:
    """
    Обчислити роботу виходу.
    
    Параметри:
        fermi_energy: Енергія Фермі (Дж)
        surface_potential: Поверхневий потенціал (Дж)
    
    Повертає:
        Робота виходу (Дж)
    """
    return fermi_energy + surface_potential

def band_gap(energy_conduction: float, energy_valence: float) -> float:
    """
    Обчислити заборонену зону.
    
    Параметри:
        energy_conduction: Енергія зони провідності (Дж)
        energy_valence: Енергія валентної зони (Дж)
    
    Повертає:
        Ширина забороненої зони (Дж)
    """
    if energy_conduction <= energy_valence:
        raise ValueError("Енергія зони провідності повинна бути більшою за енергію валентної зони")
    
    return energy_conduction - energy_valence

def carrier_concentration(effective_mass: float, temperature: float) -> float:
    """
    Обчислити концентрацію носіїв заряду.
    
    Параметри:
        effective_mass: Ефективна маса носія (кг)
        temperature: Температура (К)
    
    Повертає:
        Концентрація носіїв заряду (1/м³)
    """
    if effective_mass <= 0:
        raise ValueError("Ефективна маса повинна бути додатньою")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Концентрація носіїв у невиродженому випадку
    h_bar = PLANCK_CONSTANT / (2 * math.pi)
    return (2 * (2 * math.pi * effective_mass * BOLTZMANN_CONSTANT * temperature)**(3/2)) / (h_bar**3)

def mobility(drift_velocity: float, electric_field: float) -> float:
    """
    Обчислити рухливість носіїв заряду.
    
    Параметри:
        drift_velocity: Дрейфова швидкість (м/с)
        electric_field: Електричне поле (В/м)
    
    Повертає:
        Рухливість (м²/(В·с))
    """
    if electric_field == 0:
        raise ValueError("Електричне поле не може дорівнювати нулю")
    
    return drift_velocity / electric_field

def resistivity_temperature(resistivity_0: float, temperature_coefficient: float, 
                          temperature: float, reference_temperature: float = 293.15) -> float:
    """
    Обчислити температурну залежність електричного опору.
    
    Параметри:
        resistivity_0: Опір при референсній температурі (Ом·м)
        temperature_coefficient: Температурний коефіцієнт (1/К)
        temperature: Температура (К)
        reference_temperature: Референсна температура (К), за замовчуванням 293.15 К (20°C)
    
    Повертає:
        Електричний опір при заданій температурі (Ом·м)
    """
    if resistivity_0 <= 0:
        raise ValueError("Опір при референсній температурі повинен бути додатнім")
    
    delta_T = temperature - reference_temperature
    return resistivity_0 * (1 + temperature_coefficient * delta_T)

def thermal_expansion_coefficient(length_change: float, original_length: float, 
                                temperature_change: float) -> float:
    """
    Обчислити коефіцієнт теплового розширення.
    
    Параметри:
        length_change: Зміна довжини (м)
        original_length: Початкова довжина (м)
        temperature_change: Зміна температури (К)
    
    Повертає:
        Коефіцієнт теплового розширення (1/К)
    """
    if original_length <= 0:
        raise ValueError("Початкова довжина повинна бути додатньою")
    if temperature_change == 0:
        raise ValueError("Зміна температури не може дорівнювати нулю")
    
    strain = length_change / original_length
    return strain / temperature_change

def magnetic_moment(g_factor: float, bohr_magneton: float, 
                   angular_momentum: float) -> float:
    """
    Обчислити магнітний момент.
    
    Параметри:
        g_factor: g-фактор (безрозмірна)
        bohr_magneton: Магнетон Бора (Дж/Тл)
        angular_momentum: Кутовий момент (безрозмірна)
    
    Повертає:
        Магнітний момент (Дж/Тл)
    """
    return g_factor * bohr_magneton * angular_momentum

def bohr_magneton() -> float:
    """
    Обчислити магнетон Бора.
    
    Повертає:
        Магнетон Бора (Дж/Тл)
    """
    return (ELECTRON_CHARGE * PLANCK_CONSTANT) / (4 * math.pi * ELECTRON_MASS)

def curie_temperature(exchange_energy: float, boltzmann_constant: float = BOLTZMANN_CONSTANT) -> float:
    """
    Обчислити температуру Кюрі.
    
    Параметри:
        exchange_energy: Енергія обміну (Дж)
        boltzmann_constant: Константа Больцмана (Дж/К), за замовчуванням стандартне значення
    
    Повертає:
        Температура Кюрі (К)
    """
    if exchange_energy <= 0:
        raise ValueError("Енергія обміну повинна бути додатньою")
    if boltzmann_constant <= 0:
        raise ValueError("Константа Больцмана повинна бути додатньою")
    
    return exchange_energy / boltzmann_constant

def hall_coefficient(charge_carrier_density: float, charge: float) -> float:
    """
    Обчислити коефіцієнт Холла.
    
    Параметри:
        charge_carrier_density: Концентрація носіїв заряду (1/м³)
        charge: Заряд носія (Кл)
    
    Повертає:
        Коефіцієнт Холла (м³/Кл)
    """
    if charge_carrier_density <= 0:
        raise ValueError("Концентрація носіїв заряду повинна бути додатньою")
    if charge == 0:
        raise ValueError("Заряд носія не може дорівнювати нулю")
    
    return -1 / (charge_carrier_density * charge)

def superconducting_coherence_length(fermi_velocity: float, energy_gap: float) -> float:
    """
    Обчислити довжину когерентності для надпровідника.
    
    Параметри:
        fermi_velocity: Швидкість Фермі (м/с)
        energy_gap: Енергетичний зазор (Дж)
    
    Повертає:
        Довжина когерентності (м)
    """
    if fermi_velocity <= 0:
        raise ValueError("Швидкість Фермі повинна бути додатньою")
    if energy_gap <= 0:
        raise ValueError("Енергетичний зазор повинен бути додатнім")
    
    return (PLANCK_CONSTANT / (2 * math.pi)) * fermi_velocity / (2 * energy_gap)

def critical_magnetic_field(temperature: float, critical_temperature: float, 
                          critical_field_0: float) -> float:
    """
    Обчислити критичне магнітне поле для надпровідника.
    
    Параметри:
        temperature: Температура (К)
        critical_temperature: Критична температура (К)
        critical_field_0: Критичне поле при 0 К (Тл)
    
    Повертає:
        Критичне магнітне поле (Тл)
    """
    if temperature < 0:
        raise ValueError("Температура повинна бути невід'ємною")
    if critical_temperature <= 0:
        raise ValueError("Критична температура повинна бути додатньою")
    if critical_field_0 <= 0:
        raise ValueError("Критичне поле при 0 К повинно бути додатнім")
    if temperature > critical_temperature:
        return 0  # Немає надпровідності вище критичної температури
    
    reduced_temperature = temperature / critical_temperature
    return critical_field_0 * (1 - reduced_temperature**2)

def london_penetration_depth(superfluid_density: float, charge: float, 
                           mass: float) -> float:
    """
    Обчислити глибину проникнення Лондона.
    
    Параметри:
        superfluid_density: Густина надрідиної (1/м³)
        charge: Заряд носія (Кл)
        mass: Маса носія (кг)
    
    Повертає:
        Глибина проникнення Лондона (м)
    """
    if superfluid_density <= 0:
        raise ValueError("Густина надрідиної повинна бути додатньою")
    if charge == 0:
        raise ValueError("Заряд носія не може дорівнювати нулю")
    if mass <= 0:
        raise ValueError("Маса носія повинна бути додатньою")
    
    mu_0 = PERMEABILITY_FREE_SPACE
    return math.sqrt(mass / (mu_0 * superfluid_density * charge**2))

def einstein_frequency(einstein_temperature: float) -> float:
    """
    Обчислити частоту Ейнштейна.
    
    Параметри:
        einstein_temperature: Температура Ейнштейна (К)
    
    Повертає:
        Частота Ейнштейна (Гц)
    """
    if einstein_temperature <= 0:
        raise ValueError("Температура Ейнштейна повинна бути додатньою")
    
    return (BOLTZMANN_CONSTANT * einstein_temperature) / PLANCK_CONSTANT

def specific_heat_einstein(temperature: float, einstein_temperature: float, 
                         atoms_per_unit: float = 1) -> float:
    """
    Обчислити питому теплоємність за моделлю Ейнштейна.
    
    Параметри:
        temperature: Температура (К)
        einstein_temperature: Температура Ейнштейна (К)
        atoms_per_unit: Кількість атомів на одиницю, за замовчуванням 1
    
    Повертає:
        Питома теплоємність (Дж/(кг·К))
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if einstein_temperature <= 0:
        raise ValueError("Температура Ейнштейна повинна бути додатньою")
    if atoms_per_unit <= 0:
        raise ValueError("Кількість атомів на одиницю повинна бути додатньою")
    
    # Відношення температур
    theta_ratio = einstein_temperature / temperature
    
    # Функція Ейнштейна
    if theta_ratio > 100:  # При низьких температурах
        # C_v ≈ 3R * (θ_E/T)² * exp(-θ_E/T)
        exp_term = math.exp(-theta_ratio)
        cv = 3 * GAS_CONSTANT * atoms_per_unit * (theta_ratio**2) * exp_term
    else:
        # C_v = 3R * (θ_E/T)² * exp(θ_E/T) / (exp(θ_E/T) - 1)²
        exp_term = math.exp(theta_ratio)
        denominator = (exp_term - 1)**2
        if denominator < 1e-10:  # Уникнути ділення на нуль
            cv = 3 * GAS_CONSTANT * atoms_per_unit
        else:
            cv = 3 * GAS_CONSTANT * atoms_per_unit * (theta_ratio**2) * exp_term / denominator
    
    return cv

def thermal_stress(thermal_expansion: float, young_modulus: float, 
                  temperature_change: float, poisson_ratio: float) -> float:
    """
    Обчислити теплове напруження.
    
    Параметри:
        thermal_expansion: Коефіцієнт теплового розширення (1/К)
        young_modulus: Модуль Юнга (Па)
        temperature_change: Зміна температури (К)
        poisson_ratio: Коефіцієнт Пуассона (безрозмірна)
    
    Повертає:
        Теплове напруження (Па)
    """
    if young_modulus <= 0:
        raise ValueError("Модуль Юнга повинен бути додатнім")
    
    return (young_modulus * thermal_expansion * temperature_change) / (1 - poisson_ratio)