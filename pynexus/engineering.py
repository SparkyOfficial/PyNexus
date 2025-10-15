"""
Модуль обчислювальної інженерії для PyNexus.
Цей модуль містить функції для інженерних обчислень та моделювання інженерних систем.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional, Any, Dict, Callable
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def stress_analysis(force: float, 
                   area: float) -> float:
    """
    обчислити напруження в матеріалі.
    
    параметри:
        force: сила (Н)
        area: площа поперечного перерізу (м²)
    
    повертає:
        напруження (Па)
    """
    if area <= 0:
        raise ValueError("Площа поперечного перерізу повинна бути додатньою")
    
    return force / area

def strain_analysis(deformation: float, 
                   original_length: float) -> float:
    """
    обчислити деформацію матеріалу.
    
    параметри:
        deformation: абсолютна деформація (м)
        original_length: початкова довжина (м)
    
    повертає:
        відносна деформація
    """
    if original_length <= 0:
        raise ValueError("Початкова довжина повинна бути додатньою")
    
    return deformation / original_length

def young_modulus(stress: float, 
                 strain: float) -> float:
    """
    обчислити модуль Юнга.
    
    параметри:
        stress: напруження (Па)
        strain: деформація
    
    повертає:
        модуль Юнга (Па)
    """
    if strain == 0:
        raise ValueError("Деформація не може дорівнювати нулю")
    
    return stress / strain

def poisson_ratio(transverse_strain: float, 
                 axial_strain: float) -> float:
    """
    обчислити коефіцієнт Пуассона.
    
    параметри:
        transverse_strain: поперечна деформація
        axial_strain: осьова деформація
    
    повертає:
        коефіцієнт Пуассона
    """
    if axial_strain == 0:
        raise ValueError("Осьова деформація не може дорівнювати нулю")
    
    return -transverse_strain / axial_strain

def shear_modulus(young_modulus: float, 
                 poisson_ratio: float) -> float:
    """
    обчислити модуль зсуву.
    
    параметри:
        young_modulus: модуль Юнга (Па)
        poisson_ratio: коефіцієнт Пуассона
    
    повертає:
        модуль зсуву (Па)
    """
    if poisson_ratio == 0.5:
        raise ValueError("Коефіцієнт Пуассона не може дорівнювати 0.5")
    
    return young_modulus / (2 * (1 + poisson_ratio))

def bulk_modulus(young_modulus: float, 
                poisson_ratio: float) -> float:
    """
    обчислити об'ємний модуль.
    
    параметри:
        young_modulus: модуль Юнга (Па)
        poisson_ratio: коефіцієнт Пуассона
    
    повертає:
        об'ємний модуль (Па)
    """
    if poisson_ratio == 0.5:
        raise ValueError("Коефіцієнт Пуассона не може дорівнювати 0.5")
    
    return young_modulus / (3 * (1 - 2 * poisson_ratio))

def thermal_stress(thermal_expansion_coefficient: float, 
                  temperature_change: float, 
                  young_modulus: float, 
                  poisson_ratio: float = 0.0) -> float:
    """
    обчислити термічне напруження.
    
    параметри:
        thermal_expansion_coefficient: коефіцієнт теплового розширення (1/К)
        temperature_change: зміна температури (К)
        young_modulus: модуль Юнга (Па)
        poisson_ratio: коефіцієнт Пуассона
    
    повертає:
        термічне напруження (Па)
    """
    # Для одноосного напруження
    return young_modulus * thermal_expansion_coefficient * temperature_change

def beam_deflection_center_load(point_load: float, 
                               beam_length: float, 
                               elastic_modulus: float, 
                               moment_of_inertia: float) -> float:
    """
    обчислити прогин балки при навантаженні в центрі.
    
    параметри:
        point_load: зосереджена сила (Н)
        beam_length: довжина балки (м)
        elastic_modulus: модуль пружності (Па)
        moment_of_inertia: момент інерції перерізу (м⁴)
    
    повертає:
        прогин в центрі балки (м)
    """
    if beam_length <= 0:
        raise ValueError("Довжина балки повинна бути додатньою")
    if elastic_modulus <= 0:
        raise ValueError("Модуль пружності повинен бути додатнім")
    if moment_of_inertia <= 0:
        raise ValueError("Момент інерції повинен бути додатнім")
    
    return (point_load * beam_length**3) / (48 * elastic_modulus * moment_of_inertia)

def beam_deflection_uniform_load(uniform_load: float, 
                               beam_length: float, 
                               elastic_modulus: float, 
                               moment_of_inertia: float) -> float:
    """
    обчислити максимальний прогин балки при рівномірно розподіленому навантаженні.
    
    параметри:
        uniform_load: інтенсивність рівномірного навантаження (Н/м)
        beam_length: довжина балки (м)
        elastic_modulus: модуль пружності (Па)
        moment_of_inertia: момент інерції перерізу (м⁴)
    
    повертає:
        максимальний прогин балки (м)
    """
    if beam_length <= 0:
        raise ValueError("Довжина балки повинна бути додатньою")
    if elastic_modulus <= 0:
        raise ValueError("Модуль пружності повинен бути додатнім")
    if moment_of_inertia <= 0:
        raise ValueError("Момент інерції повинен бути додатнім")
    
    return (5 * uniform_load * beam_length**4) / (384 * elastic_modulus * moment_of_inertia)

def beam_bending_moment(point_load: float, 
                       beam_length: float, 
                       position: float) -> float:
    """
    обчислити згинальний момент у балці при навантаженні в центрі.
    
    параметри:
        point_load: зосереджена сила (Н)
        beam_length: довжина балки (м)
        position: положення перерізу (м)
    
    повертає:
        згинальний момент (Н·м)
    """
    if beam_length <= 0:
        raise ValueError("Довжина балки повинна бути додатньою")
    if position < 0 or position > beam_length:
        raise ValueError("Положення повинно бути в діапазоні [0, довжина балки]")
    
    # Для балки на двох опорах з навантаженням в центрі
    if position <= beam_length / 2:
        return (point_load * position) / 2
    else:
        return (point_load * (beam_length - position)) / 2

def torsion_angle(torque: float, 
                 shaft_length: float, 
                 shear_modulus: float, 
                 polar_moment_of_inertia: float) -> float:
    """
    обчислити кут закручування валу.
    
    параметри:
        torque: крутний момент (Н·м)
        shaft_length: довжина валу (м)
        shear_modulus: модуль зсуву (Па)
        polar_moment_of_inertia: полярний момент інерції (м⁴)
    
    повертає:
        кут закручування (рад)
    """
    if shaft_length <= 0:
        raise ValueError("Довжина валу повинна бути додатньою")
    if shear_modulus <= 0:
        raise ValueError("Модуль зсуву повинен бути додатнім")
    if polar_moment_of_inertia <= 0:
        raise ValueError("Полярний момент інерції повинен бути додатнім")
    
    return (torque * shaft_length) / (shear_modulus * polar_moment_of_inertia)

def column_critical_load(young_modulus: float, 
                        moment_of_inertia: float, 
                        column_length: float, 
                        effective_length_factor: float = 1.0) -> float:
    """
    обчислити критичне навантаження на стійкість колони (формула Ейлера).
    
    параметри:
        young_modulus: модуль Юнга (Па)
        moment_of_inertia: момент інерції перерізу (м⁴)
        column_length: довжина колони (м)
        effective_length_factor: коефіцієнт приведеної довжини
    
    повертає:
        критичне навантаження (Н)
    """
    if young_modulus <= 0:
        raise ValueError("Модуль Юнга повинен бути додатнім")
    if moment_of_inertia <= 0:
        raise ValueError("Момент інерції повинен бути додатнім")
    if column_length <= 0:
        raise ValueError("Довжина колони повинна бути додатньою")
    if effective_length_factor <= 0:
        raise ValueError("Коефіцієнт приведеної довжини повинен бути додатнім")
    
    effective_length = effective_length_factor * column_length
    return (np.pi**2 * young_modulus * moment_of_inertia) / (effective_length**2)

def stress_concentration_factor(nominal_stress: float, 
                               maximum_stress: float) -> float:
    """
    обчислити коефіцієнт концентрації напружень.
    
    параметри:
        nominal_stress: номінальне напруження (Па)
        maximum_stress: максимальне напруження (Па)
    
    повертає:
        коефіцієнт концентрації напружень
    """
    if nominal_stress == 0:
        raise ValueError("Номінальне напруження не може дорівнювати нулю")
    
    return maximum_stress / nominal_stress

def fatigue_life(stress_amplitude: float, 
                fatigue_strength_coefficient: float, 
                fatigue_strength_exponent: float) -> float:
    """
    обчислити ресурс втоми (формула Баскіна).
    
    параметри:
        stress_amplitude: амплітуда напружень (Па)
        fatigue_strength_coefficient: коефіцієнт міцності втоми
        fatigue_strength_exponent: показник степеня міцності втоми
    
    повертає:
        ресурс втоми (циклів)
    """
    if stress_amplitude <= 0:
        raise ValueError("Амплітуда напружень повинна бути додатньою")
    if fatigue_strength_coefficient <= 0:
        raise ValueError("Коефіцієнт міцності втоми повинен бути додатнім")
    if fatigue_strength_exponent >= 0:
        raise ValueError("Показник степеня міцності втоми повинен бути від'ємним")
    
    return (fatigue_strength_coefficient / stress_amplitude) ** (1 / fatigue_strength_exponent)

def von_mises_stress(stress_x: float, 
                    stress_y: float, 
                    stress_z: float, 
                    shear_xy: float, 
                    shear_yz: float, 
                    shear_xz: float) -> float:
    """
    обчислити напруження фон Мізеса.
    
    параметри:
        stress_x: нормальне напруження по осі x (Па)
        stress_y: нормальне напруження по осі y (Па)
        stress_z: нормальне напруження по осі z (Па)
        shear_xy: дотичне напруження в площині xy (Па)
        shear_yz: дотичне напруження в площині yz (Па)
        shear_xz: дотичне напруження в площині xz (Па)
    
    повертає:
        напруження фон Мізеса (Па)
    """
    # Формула фон Мізеса
    return np.sqrt(0.5 * (
        (stress_x - stress_y)**2 + 
        (stress_y - stress_z)**2 + 
        (stress_z - stress_x)**2 + 
        6 * (shear_xy**2 + shear_yz**2 + shear_xz**2)
    ))

def tresca_stress(stress_x: float, 
                 stress_y: float, 
                 stress_z: float) -> float:
    """
    обчислити максимальне дотичне напруження (критерій Треска).
    
    параметри:
        stress_x: нормальне напруження по осі x (Па)
        stress_y: нормальне напруження по осі y (Па)
        stress_z: нормальне напруження по осі z (Па)
    
    повертає:
        максимальне дотичне напруження (Па)
    """
    stresses = [stress_x, stress_y, stress_z]
    max_stress = max(stresses)
    min_stress = min(stresses)
    
    return (max_stress - min_stress) / 2

def safety_factor(ultimate_stress: float, 
                 working_stress: float) -> float:
    """
    обчислити коефіцієнт запасу міцності.
    
    параметри:
        ultimate_stress: граничне напруження (Па)
        working_stress: робоче напруження (Па)
    
    повертає:
        коефіцієнт запасу міцності
    """
    if working_stress <= 0:
        raise ValueError("Робоче напруження повинно бути додатнім")
    if ultimate_stress <= 0:
        raise ValueError("Граничне напруження повинно бути додатнім")
    
    return ultimate_stress / working_stress

def buckling_analysis(critical_load: float, 
                     applied_load: float) -> float:
    """
    обчислити коефіцієнт стійкості.
    
    параметри:
        critical_load: критичне навантаження (Н)
        applied_load: прикладене навантаження (Н)
    
    повертає:
        коефіцієнт стійкості
    """
    if applied_load <= 0:
        raise ValueError("Прикладене навантаження повинно бути додатнім")
    if critical_load <= 0:
        raise ValueError("Критичне навантаження повинно бути додатнім")
    
    return critical_load / applied_load

def thermal_expansion(original_length: float, 
                     thermal_expansion_coefficient: float, 
                     temperature_change: float) -> float:
    """
    обчислити теплове розширення.
    
    параметри:
        original_length: початкова довжина (м)
        thermal_expansion_coefficient: коефіцієнт теплового розширення (1/К)
        temperature_change: зміна температури (К)
    
    повертає:
        зміна довжини (м)
    """
    if original_length <= 0:
        raise ValueError("Початкова довжина повинна бути додатньою")
    
    return original_length * thermal_expansion_coefficient * temperature_change

def heat_conduction(thermal_conductivity: float, 
                   area: float, 
                   temperature_gradient: float) -> float:
    """
    обчислити тепловий потік за законом Фур'є.
    
    параметри:
        thermal_conductivity: коефіцієнт теплопровідності (Вт/(м·К))
        area: площа перерізу (м²)
        temperature_gradient: градієнт температури (К/м)
    
    повертає:
        тепловий потік (Вт)
    """
    if area <= 0:
        raise ValueError("Площа перерізу повинна бути додатньою")
    
    return -thermal_conductivity * area * temperature_gradient

def heat_convection(heat_transfer_coefficient: float, 
                   area: float, 
                   temperature_difference: float) -> float:
    """
    обчислити тепловий потік за законом Ньютона охолодження.
    
    параметри:
        heat_transfer_coefficient: коефіцієнт тепловіддачі (Вт/(м²·К))
        area: площа поверхні (м²)
        temperature_difference: різниця температур (К)
    
    повертає:
        тепловий потік (Вт)
    """
    if area <= 0:
        raise ValueError("Площа поверхні повинна бути додатньою")
    
    return heat_transfer_coefficient * area * temperature_difference

def heat_radiation(emissivity: float, 
                  area: float, 
                  temperature_object: float, 
                  temperature_environment: float) -> float:
    """
    обчислити тепловий потік випромінюванням (закон Стефана-Больцмана).
    
    параметри:
        emissivity: коефіцієнт випромінювання (0-1)
        area: площа поверхні (м²)
        temperature_object: температура об'єкта (К)
        temperature_environment: температура навколишнього середовища (К)
    
    повертає:
        тепловий потік (Вт)
    """
    if not (0 <= emissivity <= 1):
        raise ValueError("Коефіцієнт випромінювання повинен бути в діапазоні [0,1]")
    if area <= 0:
        raise ValueError("Площа поверхні повинна бути додатньою")
    if temperature_object <= 0 or temperature_environment <= 0:
        raise ValueError("Температури повинні бути додатніми")
    
    stefan_boltzmann_constant = 5.670374419e-8  # Вт/(м²·К⁴)
    
    return emissivity * stefan_boltzmann_constant * area * (
        temperature_object**4 - temperature_environment**4
    )

def fluid_flow_rate(velocity: float, 
                   area: float) -> float:
    """
    обчислити об'ємну витрату рідини.
    
    параметри:
        velocity: швидкість потоку (м/с)
        area: площа поперечного перерізу (м²)
    
    повертає:
        об'ємна витрата (м³/с)
    """
    if area <= 0:
        raise ValueError("Площа поперечного перерізу повинна бути додатньою")
    
    return velocity * area

def bernoulli_equation(pressure: float, 
                      density: float, 
                      velocity: float, 
                      height: float, 
                      gravity: float = 9.80665) -> float:
    """
    обчислити повну енергію потоку за рівнянням Бернуллі.
    
    параметри:
        pressure: тиск (Па)
        density: густина рідини (кг/м³)
        velocity: швидкість потоку (м/с)
        height: висота над рівнем відліку (м)
        gravity: прискорення вільного падіння (м/с²)
    
    повертає:
        повна енергія потоку (Дж/кг)
    """
    if density <= 0:
        raise ValueError("Густина рідини повинна бути додатньою")
    
    return pressure / density + 0.5 * velocity**2 + gravity * height

def reynolds_number(velocity: float, 
                   characteristic_length: float, 
                   density: float, 
                   viscosity: float) -> float:
    """
    обчислити число Рейнольдса.
    
    параметри:
        velocity: швидкість потоку (м/с)
        characteristic_length: характерна довжина (м)
        density: густина рідини (кг/м³)
        viscosity: динамічна в'язкість (Па·с)
    
    повертає:
        число Рейнольдса
    """
    if characteristic_length <= 0:
        raise ValueError("Характерна довжина повинна бути додатньою")
    if density <= 0:
        raise ValueError("Густина рідини повинна бути додатньою")
    if viscosity <= 0:
        raise ValueError("В'язкість рідини повинна бути додатньою")
    
    return (density * velocity * characteristic_length) / viscosity

def darcy_weisbach_friction_factor(reynolds_number: float, 
                                 relative_roughness: float, 
                                 pipe_diameter: float) -> float:
    """
    обчислити коефіцієнт тертя за формулою Дарсі-Вейсбаха.
    
    параметри:
        reynolds_number: число Рейнольдса
        relative_roughness: відносна шорсткість
        pipe_diameter: діаметр труби (м)
    
    повертає:
        коефіцієнт тертя
    """
    if reynolds_number <= 0:
        raise ValueError("Число Рейнольдса повинно бути додатнім")
    if pipe_diameter <= 0:
        raise ValueError("Діаметр труби повинен бути додатнім")
    
    # Для ламінарного потоку (Re < 2300)
    if reynolds_number < 2300:
        return 64 / reynolds_number
    
    # Для турбулентного потоку використовуємо наближену формулу Колбрука-Уайта
    # 1/√f = -2 * log10(ε/D/3.7 + 2.51/(Re*√f))
    # Розв'язуємо ітераційно
    
    # Початкове наближення
    f = 0.02
    
    for _ in range(100):  # Максимум 100 ітерацій
        f_new = (1 / (-2 * np.log10(relative_roughness / 3.7 + 2.51 / (reynolds_number * np.sqrt(f)))))**2
        
        if abs(f_new - f) < 1e-6:
            return f_new
        
        f = f_new
    
    return f

def pressure_drop_darcy_weisbach(friction_factor: float, 
                               pipe_length: float, 
                               pipe_diameter: float, 
                               fluid_density: float, 
                               fluid_velocity: float) -> float:
    """
    обчислити втрати тиску за формулою Дарсі-Вейсбаха.
    
    параметри:
        friction_factor: коефіцієнт тертя
        pipe_length: довжина труби (м)
        pipe_diameter: діаметр труби (м)
        fluid_density: густина рідини (кг/м³)
        fluid_velocity: швидкість рідини (м/с)
    
    повертає:
        втрати тиску (Па)
    """
    if pipe_length <= 0:
        raise ValueError("Довжина труби повинна бути додатньою")
    if pipe_diameter <= 0:
        raise ValueError("Діаметр труби повинен бути додатнім")
    if fluid_density <= 0:
        raise ValueError("Густина рідини повинна бути додатньою")
    
    return friction_factor * (pipe_length / pipe_diameter) * (fluid_density * fluid_velocity**2) / 2

def pump_power(flow_rate: float, 
              pressure_drop: float, 
              efficiency: float = 1.0) -> float:
    """
    обчислити потужність насоса.
    
    параметри:
        flow_rate: об'ємна витрата (м³/с)
        pressure_drop: перепад тиску (Па)
        efficiency: ККД насоса
    
    повертає:
        потужність насоса (Вт)
    """
    if flow_rate <= 0:
        raise ValueError("Об'ємна витрата повинна бути додатньою")
    if efficiency <= 0 or efficiency > 1:
        raise ValueError("ККД повинен бути в діапазоні (0, 1]")
    
    return (flow_rate * pressure_drop) / efficiency

def heat_exchanger_effectiveness(ntu: float, 
                               capacity_ratio: float, 
                               heat_exchanger_type: str = "counterflow") -> float:
    """
    обчислити ефективність теплообмінника.
    
    параметри:
        ntu: число передачі тепла
        capacity_ratio: відношення теплоємностей
        heat_exchanger_type: тип теплообмінника ("counterflow", "parallel", "crossflow")
    
    повертає:
        ефективність теплообмінника
    """
    if ntu <= 0:
        raise ValueError("Число передачі тепла повинно бути додатнім")
    if capacity_ratio < 0 or capacity_ratio > 1:
        raise ValueError("Відношення теплоємностей повинно бути в діапазоні [0, 1]")
    
    if heat_exchanger_type == "counterflow":
        if capacity_ratio == 1:
            return ntu / (1 + ntu)
        else:
            return (1 - np.exp(-ntu * (1 - capacity_ratio))) / (1 - capacity_ratio * np.exp(-ntu * (1 - capacity_ratio)))
    
    elif heat_exchanger_type == "parallel":
        return (1 - np.exp(-ntu * (1 + capacity_ratio))) / (1 + capacity_ratio)
    
    elif heat_exchanger_type == "crossflow":
        # Наближена формула для перехресного потоку
        return 1 - np.exp((1 / capacity_ratio) * ntu**0.22 * (np.exp(-capacity_ratio * ntu**0.78) - 1))
    
    else:
        raise ValueError("Невідомий тип теплообмінника")

def fin_efficiency(fin_length: float, 
                  fin_thickness: float, 
                  thermal_conductivity: float, 
                  heat_transfer_coefficient: float) -> float:
    """
    обчислити ефективність ребра.
    
    параметри:
        fin_length: довжина ребра (м)
        fin_thickness: товщина ребра (м)
        thermal_conductivity: коефіцієнт теплопровідності (Вт/(м·К))
        heat_transfer_coefficient: коефіцієнт тепловіддачі (Вт/(м²·К))
    
    повертає:
        ефективність ребра
    """
    if fin_length <= 0:
        raise ValueError("Довжина ребра повинна бути додатньою")
    if fin_thickness <= 0:
        raise ValueError("Товщина ребра повинна бути додатньою")
    if thermal_conductivity <= 0:
        raise ValueError("Коефіцієнт теплопровідності повинен бути додатнім")
    if heat_transfer_coefficient <= 0:
        raise ValueError("Коефіцієнт тепловіддачі повинен бути додатнім")
    
    # Параметр m
    m = np.sqrt(heat_transfer_coefficient * 2 / (thermal_conductivity * fin_thickness))
    
    # Ефективність прямокутного ребра
    if m * fin_length < 1e-10:
        return 1.0
    
    return np.tanh(m * fin_length) / (m * fin_length)

def thermal_resistance_series(resistances: List[float]) -> float:
    """
    обчислити загальний термічний опір при послідовному з'єднанні.
    
    параметри:
        resistances: список термічних опорів (К/Вт)
    
    повертає:
        загальний термічний опір (К/Вт)
    """
    if not resistances:
        raise ValueError("Список термічних опорів не може бути порожнім")
    
    return sum(resistances)

def thermal_resistance_parallel(resistances: List[float]) -> float:
    """
    обчислити загальний термічний опір при паралельному з'єднанні.
    
    параметри:
        resistances: список термічних опорів (К/Вт)
    
    повертає:
        загальний термічний опір (К/Вт)
    """
    if not resistances:
        raise ValueError("Список термічних опорів не може бути порожнім")
    
    conductances = [1/r for r in resistances if r > 0]
    if not conductances:
        return float('inf')
    
    total_conductance = sum(conductances)
    if total_conductance == 0:
        return float('inf')
    
    return 1 / total_conductance

def stress_intensity_factor(applied_stress: float, 
                           crack_length: float, 
                           geometry_factor: float = 1.12) -> float:
    """
    обчислити коефіцієнт інтенсивності напружень.
    
    параметри:
        applied_stress: прикладене напруження (Па)
        crack_length: довжина тріщини (м)
        geometry_factor: геометричний коефіцієнт
    
    повертає:
        коефіцієнт інтенсивності напружень (Па·√м)
    """
    if applied_stress <= 0:
        raise ValueError("Прикладене напруження повинно бути додатнім")
    if crack_length <= 0:
        raise ValueError("Довжина тріщини повинна бути додатньою")
    if geometry_factor <= 0:
        raise ValueError("Геометричний коефіцієнт повинен бути додатнім")
    
    return geometry_factor * applied_stress * np.sqrt(np.pi * crack_length)

def fracture_toughness(stress_intensity_factor: float, 
                      critical_stress: float) -> float:
    """
    обчислити критичний коефіцієнт інтенсивності напружень (вязкість руйнування).
    
    параметри:
        stress_intensity_factor: коефіцієнт інтенсивності напружень (Па·√м)
        critical_stress: критичне напруження (Па)
    
    повертає:
        в'язкість руйнування (Па·√м)
    """
    if critical_stress <= 0:
        raise ValueError("Критичне напруження повинно бути додатнім")
    
    return stress_intensity_factor / critical_stress

def creep_rate(stress: float, 
              temperature: float, 
              material_constant: float, 
              stress_exponent: float, 
              activation_energy: float) -> float:
    """
    обчислити швидкість повзучості (рівняння Нортона).
    
    параметри:
        stress: напруження (Па)
        temperature: температура (К)
        material_constant: матеріальна константа
        stress_exponent: показник степеня напруження
        activation_energy: енергія активації (Дж/моль)
    
    повертає:
        швидкість повзучості (1/с)
    """
    if stress <= 0:
        raise ValueError("Напруження повинно бути додатнім")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if activation_energy <= 0:
        raise ValueError("Енергія активації повинна бути додатньою")
    
    gas_constant = 8.31446261815324  # Дж/(моль·К)
    
    return material_constant * (stress ** stress_exponent) * np.exp(-activation_energy / (gas_constant * temperature))

def fatigue_crack_growth(stress_intensity_range: float, 
                        material_constant: float, 
                        paris_exponent: float) -> float:
    """
    обчислити швидкість зростання втомної тріщини (рівняння Паріса).
    
    параметри:
        stress_intensity_range: розмах коефіцієнта інтенсивності напружень (Па·√м)
        material_constant: матеріальна константа
        paris_exponent: показник степеня Паріса
    
    повертає:
        швидкість зростання тріщини (м/цикл)
    """
    if stress_intensity_range <= 0:
        raise ValueError("Розмах коефіцієнта інтенсивності напружень повинен бути додатнім")
    
    return material_constant * (stress_intensity_range ** paris_exponent)

def residual_stress(initial_stress: float, 
                   thermal_stress: float, 
                   mechanical_stress: float) -> float:
    """
    обчислити залишкові напруження.
    
    параметри:
        initial_stress: початкові напруження (Па)
        thermal_stress: термічні напруження (Па)
        mechanical_stress: механічні напруження (Па)
    
    повертає:
        залишкові напруження (Па)
    """
    return initial_stress + thermal_stress + mechanical_stress

def stress_concentration_hole(plate_width: float, 
                            hole_diameter: float, 
                            applied_stress: float) -> float:
    """
    обчислити коефіцієнт концентрації напружень для отвору в пластині.
    
    параметри:
        plate_width: ширина пластини (м)
        hole_diameter: діаметр отвору (м)
        applied_stress: прикладене напруження (Па)
    
    повертає:
        максимальне напруження біля отвору (Па)
    """
    if plate_width <= 0:
        raise ValueError("Ширина пластини повинна бути додатньою")
    if hole_diameter <= 0:
        raise ValueError("Діаметр отвору повинен бути додатнім")
    if hole_diameter >= plate_width:
        raise ValueError("Діаметр отвору повинен бути меншим за ширину пластини")
    if applied_stress <= 0:
        raise ValueError("Прикладене напруження повинно бути додатнім")
    
    # Для нескінченної пластини з отвором Kt = 3
    # Для скінченної пластини використовуємо наближення
    ratio = hole_diameter / plate_width
    if ratio < 0.1:
        kt = 3.0
    else:
        kt = 3.0 - 2.0 * ratio  # Спрощене наближення
    
    return kt * applied_stress

def composite_laminate_properties(fiber_properties: Dict[str, float], 
                                matrix_properties: Dict[str, float], 
                                fiber_volume_fraction: float) -> Dict[str, float]:
    """
    обчислити властивості композиту з однонаправленого шару.
    
    параметри:
        fiber_properties: властивості волокна {E1, E2, G12, ν12}
        matrix_properties: властивості матриці {E, G, ν}
        fiber_volume_fraction: об'ємна частка волокна
    
    повертає:
        словник властивостей композиту
    """
    if not (0 <= fiber_volume_fraction <= 1):
        raise ValueError("Об'ємна частка волокна повинна бути в діапазоні [0, 1]")
    
    matrix_volume_fraction = 1 - fiber_volume_fraction
    
    # Правило сум суміші для модуля пружності в напрямку волокна
    e1 = (fiber_properties['E1'] * fiber_volume_fraction + 
          matrix_properties['E'] * matrix_volume_fraction)
    
    # Модель по зворотному правилу сум суміші для поперечного модуля
    e2 = 1 / (fiber_volume_fraction / fiber_properties['E2'] + 
              matrix_volume_fraction / matrix_properties['E'])
    
    # Модуль зсуву по моделі Халпіна-Цаї
    xi = 2  # Емпіричний параметр
    g12 = matrix_properties['G'] * (
        (1 + xi * fiber_volume_fraction * (fiber_properties['G12'] / matrix_properties['G'] - 1)) /
        (1 - fiber_volume_fraction * (fiber_properties['G12'] / matrix_properties['G'] - 1))
    )
    
    # Коефіцієнт Пуассона
    nu12 = (fiber_properties['ν12'] * fiber_volume_fraction + 
            matrix_properties['ν'] * matrix_volume_fraction)
    
    return {
        'E1': e1,
        'E2': e2,
        'G12': g12,
        'ν12': nu12
    }

def failure_criteria_tsai_wu(sigma1: float, 
                           sigma2: float, 
                           tau12: float, 
                           strength_properties: Dict[str, float]) -> float:
    """
    обчислити критерій руйнування Цая-Ву.
    
    параметри:
        sigma1: напруження в напрямку 1 (Па)
        sigma2: напруження в напрямку 2 (Па)
        tau12: дотичне напруження (Па)
        strength_properties: міцнісні властивості {Xt, Xc, Yt, Yc, S}
    
    повертає:
        значення критерію Цая-Ву
    """
    # Міцнісні властивості
    xt = strength_properties['Xt']  # Міцність на розтяг в напрямку волокна
    xc = strength_properties['Xc']  # Міцність на стиск в напрямку волокна
    yt = strength_properties['Yt']  # Міцність на розтяг поперек волокна
    yc = strength_properties['Yc']  # Міцність на стиск поперек волокна
    s = strength_properties['S']    # Міцність на зсув
    
    # Коефіцієнти критерію Цая-Ву
    f1 = 1/xt - 1/xc
    f2 = 1/yt - 1/yc
    f11 = 1/(xt * xc)
    f22 = 1/(yt * yc)
    f66 = 1/(s * s)
    
    # Квадратична форма
    quadratic_term = (f11 * sigma1**2 + f22 * sigma2**2 + f66 * tau12**2 + 
                     2 * f1 * sigma1 + 2 * f2 * sigma2)
    
    # Лінійна форма
    linear_term = f1 * sigma1 + f2 * sigma2
    
    return quadratic_term + linear_term

def vibration_natural_frequency(effective_stiffness: float, 
                              effective_mass: float) -> float:
    """
    обчислити власну частоту коливань.
    
    параметри:
        effective_stiffness: ефективна жорсткість (Н/м)
        effective_mass: ефективна маса (кг)
    
    повертає:
        власна частота (Гц)
    """
    if effective_stiffness <= 0:
        raise ValueError("Ефективна жорсткість повинна бути додатньою")
    if effective_mass <= 0:
        raise ValueError("Ефективна маса повинна бути додатньою")
    
    return (1 / (2 * np.pi)) * np.sqrt(effective_stiffness / effective_mass)

def vibration_damping_ratio(logarithmic_decrement: float) -> float:
    """
    обчислити коефіцієнт демпфування.
    
    параметри:
        logarithmic_decrement: логарифмічний декремент
    
    повертає:
        коефіцієнт демпфування
    """
    if logarithmic_decrement <= 0:
        raise ValueError("Логарифмічний декремент повинен бути додатнім")
    
    return logarithmic_decrement / np.sqrt(4 * np.pi**2 + logarithmic_decrement**2)

def shock_response_spectrum(acceleration: float, 
                          frequency: float, 
                          damping_ratio: float) -> float:
    """
    обчислити спектр ударної відповіді.
    
    параметри:
        acceleration: прискорення (м/с²)
        frequency: частота (Гц)
        damping_ratio: коефіцієнт демпфування
    
    повертає:
        максимальне прискорення відповіді (м/с²)
    """
    if frequency <= 0:
        raise ValueError("Частота повинна бути додатньою")
    if damping_ratio < 0:
        raise ValueError("Коефіцієнт демпфування не може бути від'ємним")
    
    omega = 2 * np.pi * frequency
    if damping_ratio < 1:  # Недемпфовані коливання
        omega_d = omega * np.sqrt(1 - damping_ratio**2)
        return acceleration * np.exp(-damping_ratio * omega * 0) * np.cos(omega_d * 0)
    else:  # Критично або наддемпфовані коливання
        return acceleration

def fatigue_damage(miner_rule_cycles: List[Tuple[float, float]]) -> float:
    """
    обчислити накопичення втомного пошкодження за правилом Майнера.
    
    параметри:
        miner_rule_cycles: список пар (кількість_циклів, кількість_циклів_до_руйнування)
    
    повертає:
        сумарне накопичене пошкодження
    """
    if not miner_rule_cycles:
        raise ValueError("Список циклів не може бути порожнім")
    
    damage = 0.0
    for cycles_applied, cycles_to_failure in miner_rule_cycles:
        if cycles_to_failure <= 0:
            raise ValueError("Кількість циклів до руйнування повинна бути додатньою")
        damage += cycles_applied / cycles_to_failure
    
    return damage

def stress_life_curve(stress_amplitude: float, 
                     fatigue_strength_coefficient: float, 
                     fatigue_strength_exponent: float) -> float:
    """
    обчислити кількість циклів до руйнування з кривої напруження-ресурс.
    
    параметри:
        stress_amplitude: амплітуда напружень (Па)
        fatigue_strength_coefficient: коефіцієнт міцності втоми
        fatigue_strength_exponent: показник степеня міцності втоми
    
    повертає:
        кількість циклів до руйнування
    """
    if stress_amplitude <= 0:
        raise ValueError("Амплітуда напружень повинна бути додатньою")
    if fatigue_strength_coefficient <= 0:
        raise ValueError("Коефіцієнт міцності втоми повинен бути додатнім")
    if fatigue_strength_exponent >= 0:
        raise ValueError("Показник степеня міцності втоми повинен бути від'ємним")
    
    return (fatigue_strength_coefficient / stress_amplitude) ** (1 / fatigue_strength_exponent)

def strain_life_curve(strain_amplitude: float, 
                     fatigue_ductility_coefficient: float, 
                     fatigue_ductility_exponent: float,
                     elastic_modulus: float,
                     fatigue_strength_coefficient: float, 
                     fatigue_strength_exponent: float) -> float:
    """
    обчислити кількість циклів до руйнування з кривої деформації-ресурс.
    
    параметри:
        strain_amplitude: амплітуда деформацій
        fatigue_ductility_coefficient: коефіцієнт пластичної втоми
        fatigue_ductility_exponent: показник степеня пластичної втоми
        elastic_modulus: модуль пружності (Па)
        fatigue_strength_coefficient: коефіцієнт міцності втоми
        fatigue_strength_exponent: показник степеня міцності втоми
    
    повертає:
        кількість циклів до руйнування
    """
    if strain_amplitude <= 0:
        raise ValueError("Амплітуда деформацій повинна бути додатньою")
    if elastic_modulus <= 0:
        raise ValueError("Модуль пружності повинен бути додатнім")
    if fatigue_ductility_coefficient <= 0:
        raise ValueError("Коефіцієнт пластичної втоми повинен бути додатнім")
    if fatigue_ductility_exponent >= 0:
        raise ValueError("Показник степеня пластичної втоми повинен бути від'ємним")
    if fatigue_strength_coefficient <= 0:
        raise ValueError("Коефіцієнт міцності втоми повинен бути додатнім")
    if fatigue_strength_exponent >= 0:
        raise ValueError("Показник степеня міцності втоми повинен бути від'ємним")
    
    # Розділяємо на пружну та пластичну компоненти
    elastic_strain_amplitude = stress_amplitude / elastic_modulus
    plastic_strain_amplitude = strain_amplitude - elastic_strain_amplitude
    
    # Крива деформації-ресурс
    cycles_elastic = (elastic_modulus * strain_amplitude / fatigue_strength_coefficient) ** (1 / fatigue_strength_exponent)
    cycles_plastic = (plastic_strain_amplitude / fatigue_ductility_coefficient) ** (1 / fatigue_ductility_exponent)
    
    # Комбінована формула
    return 1 / (1/cycles_elastic + 1/cycles_plastic)

def rainflow_cycle_counting(stress_history: np.ndarray) -> List[Tuple[float, float]]:
    """
    виконати підрахунок циклів методом Rainflow.
    
    параметри:
        stress_history: історія напружень
    
    повертає:
        список циклів (амплітуда, середнє_напруження)
    """
    if len(stress_history) < 2:
        raise ValueError("Історія напружень повинна містити принаймні 2 точки")
    
    # Спрощена реалізація Rainflow алгоритму
    # У реальній інженерній практиці використовуються спеціалізовані бібліотеки
    
    # Знаходимо локальні екстремуми
    peaks = []
    valleys = []
    
    for i in range(1, len(stress_history) - 1):
        if (stress_history[i] > stress_history[i-1] and 
            stress_history[i] > stress_history[i+1]):
            peaks.append(stress_history[i])
        elif (stress_history[i] < stress_history[i-1] and 
              stress_history[i] < stress_history[i+1]):
            valleys.append(stress_history[i])
    
    # Формуємо цикли
    cycles = []
    all_extremes = sorted(peaks + valleys)
    
    for i in range(len(all_extremes) - 1):
        amplitude = abs(all_extremes[i+1] - all_extremes[i]) / 2
        mean = (all_extremes[i+1] + all_extremes[i]) / 2
        cycles.append((amplitude, mean))
    
    return cycles

def cumulative_damage_weibull(stress_amplitude: float, 
                            scale_parameter: float, 
                            shape_parameter: float) -> float:
    """
    обчислити накопичене пошкодження з розподілом Вейбулла.
    
    параметри:
        stress_amplitude: амплітуда напружень (Па)
        scale_parameter: параметр масштабу
        shape_parameter: параметр форми
    
    повертає:
        ймовірність руйнування
    """
    if stress_amplitude <= 0:
        raise ValueError("Амплітуда напружень повинна бути додатньою")
    if scale_parameter <= 0:
        raise ValueError("Параметр масштабу повинен бути додатнім")
    if shape_parameter <= 0:
        raise ValueError("Параметр форми повинен бути додатнім")
    
    return 1 - np.exp(-((stress_amplitude / scale_parameter) ** shape_parameter))

def fracture_mechanics_j_integral(strain_energy_density: float, 
                                crack_extension: float) -> float:
    """
    обчислити J-інтеграл механіки руйнування.
    
    параметри:
        strain_energy_density: щільність енергії деформації (Дж/м³)
        crack_extension: подовження тріщини (м)
    
    повертає:
        J-інтеграл (Дж/м²)
    """
    if crack_extension <= 0:
        raise ValueError("Подовження тріщини повинно бути додатнім")
    
    return strain_energy_density * crack_extension

def thermal_shock_analysis(thermal_shock_parameter: float, 
                          material_toughness: float) -> float:
    """
    обчислити параметр термічного удару.
    
    параметри:
        thermal_shock_parameter: параметр термічного удару
        material_toughness: в'язкість матеріалу (Па·√м)
    
    повертає:
        критичний температурний градієнт (К/м)
    """
    if material_toughness <= 0:
        raise ValueError("В'язкість матеріалу повинна бути додатньою")
    
    return thermal_shock_parameter / material_toughness

def creep_rupture_time(stress: float, 
                      temperature: float, 
                      larson_miller_parameter: float) -> float:
    """
    обчислити час до руйнування при повзучості (параметр Ларсона-Міллера).
    
    параметри:
        stress: напруження (Па)
        temperature: температура (К)
        larson_miller_parameter: параметр Ларсона-Міллера
    
    повертає:
        час до руйнування (години)
    """
    if stress <= 0:
        raise ValueError("Напруження повинно бути додатнім")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if larson_miller_parameter <= 0:
        raise ValueError("Параметр Ларсона-Міллера повинен бути додатнім")
    
    # P = T * (C + log(tr))
    # де P - параметр Ларсона-Міллера, T - температура в К, tr - час до руйнування
    c = 20  # Емпірична константа
    log_time = larson_miller_parameter / temperature - c
    
    return 10 ** log_time

def fatigue_limit_reduction(surf
