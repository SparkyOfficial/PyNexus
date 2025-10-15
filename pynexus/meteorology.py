"""
Модуль для обчислювальної метеорології
Computational Meteorology Module
"""
import numpy as np
from typing import Union, Tuple, List, Optional, Dict, Any
import math

# Константи для метеорологічних обчислень
# Meteorological constants
EARTH_RADIUS = 6371000  # Радіус Землі в метрах
GRAVITY = 9.80665  # Прискорення вільного падіння (м/с²)
GAS_CONSTANT_DRY_AIR = 287.058  # Газова стала для сухого повітря (Дж/(кг·К))
GAS_CONSTANT_WATER_VAPOR = 461.495  # Газова стала для водяної пари (Дж/(кг·К))
SPECIFIC_HEAT_DRY_AIR = 1005  # Питома теплоємність сухого повітря при постійному тиску (Дж/(кг·К))
LATENT_HEAT_VAPORIZATION = 2.501e6  # Питома теплота пароутворення (Дж/кг)
STEFAN_BOLTZMANN = 5.670374419e-8  # Константа Стефана-Больцмана (Вт/(м²·К⁴))

def potential_temperature(temperature: float, pressure: float, reference_pressure: float = 100000) -> float:
    """
    Обчислити потенційну температуру повітря.
    
    Потенційна температура - це температура, яку б мала повітряна маса, 
    якщо б її адіабатично підняти або опустити до рівня з тиском reference_pressure.
    
    Параметри:
        temperature: Абсолютна температура (К)
        pressure: Тиск (Па)
        reference_pressure: Базовий тиск (Па), за замовчуванням 100000 Па (1000 гПа)
    
    Повертає:
        Потенційна температура (К)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if reference_pressure <= 0:
        raise ValueError("Базовий тиск повинен бути додатнім")
    
    # Формула потенційної температури
    kappa = GAS_CONSTANT_DRY_AIR / SPECIFIC_HEAT_DRY_AIR
    theta = temperature * (reference_pressure / pressure) ** kappa
    return theta

def virtual_temperature(temperature: float, mixing_ratio: float) -> float:
    """
    Обчислити віртуальну температуру повітря.
    
    Віртуальна температура - це температура сухого повітря, 
    яка має таку ж густину, як і вологе повітря при тій же температурі.
    
    Параметри:
        temperature: Абсолютна температура (К)
        mixing_ratio: Відношення змішування водяної пари (кг/кг)
    
    Повертає:
        Віртуальна температура (К)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if mixing_ratio < 0:
        raise ValueError("Відношення змішування повинно бути невід'ємним")
    
    # Формула віртуальної температури
    epsilon = GAS_CONSTANT_DRY_AIR / GAS_CONSTANT_WATER_VAPOR
    Tv = temperature * (1 + mixing_ratio / epsilon) / (1 + mixing_ratio)
    return Tv

def equivalent_potential_temperature(temperature: float, pressure: float, 
                                  dew_point: float, reference_pressure: float = 100000) -> float:
    """
    Обчислити еквівалентну потенційну температуру.
    
    Еквівалентна потенційна температура - це потенційна температура повітряної маси, 
    якщо вся волога в ній сконденсувалася і випарувалася адіабатично.
    
    Параметри:
        temperature: Абсолютна температура (К)
        pressure: Тиск (Па)
        dew_point: Точка роси (К)
        reference_pressure: Базовий тиск (Па), за замовчуванням 100000 Па
    
    Повертає:
        Еквівалентна потенційна температура (К)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if dew_point <= 0:
        raise ValueError("Точка роси повинна бути додатньою")
    if reference_pressure <= 0:
        raise ValueError("Базовий тиск повинен бути додатнім")
    
    # Обчислити відношення змішування при точці роси
    vapor_pressure = saturation_vapor_pressure(dew_point)
    mixing_ratio = 0.622 * vapor_pressure / (pressure - vapor_pressure)
    
    # Обчислити потенційну температуру
    theta = potential_temperature(temperature, pressure, reference_pressure)
    
    # Обчислити еквівалентну потенційну температуру
    theta_e = theta * np.exp((LATENT_HEAT_VAPORIZATION * mixing_ratio) / 
                            (SPECIFIC_HEAT_DRY_AIR * temperature))
    return theta_e

def saturation_vapor_pressure(temperature: float) -> float:
    """
    Обчислити тиск насиченої водяної пари за формулою Magnus.
    
    Параметри:
        temperature: Температура (К)
    
    Повертає:
        Тиск насиченої водяної пари (Па)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Перетворити температуру в градуси Цельсія
    temp_c = temperature - 273.15
    
    # Формула Magnus для тиску насиченої пари
    # Для температур від -45°C до 60°C
    if temp_c >= -45 and temp_c <= 60:
        es = 6.1094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))
        # Перетворити з гПа в Па
        return es * 100
    else:
        raise ValueError("Температура поза діапазоном застосування формули Magnus")

def relative_humidity(temperature: float, dew_point: float) -> float:
    """
    Обчислити відносну вологість повітря.
    
    Параметри:
        temperature: Температура повітря (К)
        dew_point: Точка роси (К)
    
    Повертає:
        Відносна вологість (%)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if dew_point <= 0:
        raise ValueError("Точка роси повинна бути додатньою")
    
    # Обчислити тиск насиченої пари для обох температур
    e_actual = saturation_vapor_pressure(dew_point)
    e_saturation = saturation_vapor_pressure(temperature)
    
    # Обчислити відносну вологість
    rh = (e_actual / e_saturation) * 100
    return rh

def mixing_ratio_from_relative_humidity(temperature: float, pressure: float, 
                                      relative_humidity: float) -> float:
    """
    Обчислити відношення змішування з відносної вологості.
    
    Параметри:
        temperature: Температура повітря (К)
        pressure: Тиск (Па)
        relative_humidity: Відносна вологість (%)
    
    Повертає:
        Відношення змішування (кг/кг)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if relative_humidity < 0 or relative_humidity > 100:
        raise ValueError("Відносна вологість повинна бути в діапазоні 0-100%")
    
    # Обчислити тиск насиченої пари
    e_sat = saturation_vapor_pressure(temperature)
    
    # Обчислити фактичний тиск пари
    e_actual = (relative_humidity / 100) * e_sat
    
    # Обчислити відношення змішування
    mixing_ratio = 0.622 * e_actual / (pressure - e_actual)
    return mixing_ratio

def specific_humidity(mixing_ratio: float) -> float:
    """
    Обчислити питому вологість.
    
    Питома вологість - це відношення маси водяної пари до загальної маси повітряної суміші.
    
    Параметри:
        mixing_ratio: Відношення змішування (кг/кг)
    
    Повертає:
        Питома вологість (кг/кг)
    """
    if mixing_ratio < 0:
        raise ValueError("Відношення змішування повинно бути невід'ємним")
    
    # Формула питомої вологості
    q = mixing_ratio / (1 + mixing_ratio)
    return q

def absolute_humidity(temperature: float, pressure: float, mixing_ratio: float) -> float:
    """
    Обчислити абсолютну вологість.
    
    Абсолютна вологість - це маса водяної пари в одиниці об'єму повітря.
    
    Параметри:
        temperature: Температура повітря (К)
        pressure: Тиск (Па)
        mixing_ratio: Відношення змішування (кг/кг)
    
    Повертає:
        Абсолютна вологість (кг/м³)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if mixing_ratio < 0:
        raise ValueError("Відношення змішування повинно бути невід'ємним")
    
    # Обчислити густину повітря
    rho = pressure / (GAS_CONSTANT_DRY_AIR * temperature)
    
    # Обчислити абсолютну вологість
    rho_v = rho * mixing_ratio
    return rho_v

def heat_index(temperature: float, relative_humidity: float) -> float:
    """
    Обчислити індекс тепла (температуру по відчуттях).
    
    Параметри:
        temperature: Температура повітря (°C)
        relative_humidity: Відносна вологість (%)
    
    Повертає:
        Індекс тепла (°C)
    """
    if temperature < -50 or temperature > 60:
        raise ValueError("Температура поза діапазоном застосування (-50°C до 60°C)")
    if relative_humidity < 0 or relative_humidity > 100:
        raise ValueError("Відносна вологість повинна бути в діапазоні 0-100%")
    
    # Формула індексу тепла (Rothfusz регресія)
    T = temperature * 9/5 + 32  # Перетворити в °F
    RH = relative_humidity
    
    # Основна формула
    HI = (-42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH - 
          6.83783e-3*T*T - 5.481717e-2*RH*RH + 1.22874e-3*T*T*RH + 
          8.5282e-4*T*RH*RH - 1.99e-6*T*T*RH*RH)
    
    # Перетворити назад в °C
    HI_C = (HI - 32) * 5/9
    return HI_C

def wind_chill(temperature: float, wind_speed: float) -> float:
    """
    Обчислити індекс вітру (температурно-вітровий індекс).
    
    Параметри:
        temperature: Температура повітря (°C)
        wind_speed: Швидкість вітру (м/с)
    
    Повертає:
        Індекс вітру (°C)
    """
    if temperature > 10:
        raise ValueError("Індекс вітру застосовується лише при температурі ≤ 10°C")
    if wind_speed < 0:
        raise ValueError("Швидкість вітру повинна бути невід'ємною")
    
    # Перетворити швидкість вітру в км/год
    wind_kmh = wind_speed * 3.6
    
    # Формула індексу вітру (нова формула Environment Canada)
    if wind_kmh >= 5:
        WCI = 13.12 + 0.6215*temperature - 11.37*(wind_kmh**0.16) + 0.3965*temperature*(wind_kmh**0.16)
    else:
        WCI = temperature
    
    return WCI

def coriolis_parameter(latitude: float) -> float:
    """
    Обчислити параметр Коріоліса.
    
    Параметри:
        latitude: Географічна широта (градуси)
    
    Повертає:
        Параметр Коріоліса (с⁻¹)
    """
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    
    # Кутова швидкість обертання Землі
    omega = 7.292115e-5  # рад/с
    
    # Параметр Коріоліса
    f = 2 * omega * np.sin(np.radians(latitude))
    return f

def geostrophic_wind(height_gradient: float, latitude: float, 
                    density: float = 1.225) -> float:
    """
    Обчислити геострофний вітер.
    
    Геострофний вітер - це вітер, який встановлюється при балансі між 
    силою тиску і силою Коріоліса.
    
    Параметри:
        height_gradient: Градієнт висоти (м/км)
        latitude: Географічна широта (градуси)
        density: Густина повітря (кг/м³), за замовчуванням 1.225 кг/м³
    
    Повертає:
        Геострофний вітер (м/с)
    """
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    if density <= 0:
        raise ValueError("Густина повітря повинна бути додатньою")
    
    # Параметр Коріоліса
    f = coriolis_parameter(latitude)
    
    # Геострофний вітер
    if abs(f) < 1e-10:
        raise ValueError("Параметр Коріоліса занадто малий (екваторіальна область)")
    
    g_wind = (GRAVITY * height_gradient) / (f * 1000)  # 1000 для перетворення км в м
    return g_wind

def pressure_from_altitude(altitude: float, sea_level_pressure: float = 101325,
                          temperature: float = 288.15) -> float:
    """
    Обчислити тиск на заданій висоті за барометричною формулою.
    
    Параметри:
        altitude: Висота над рівнем моря (м)
        sea_level_pressure: Тиск на рівні моря (Па), за замовчуванням 101325 Па
        temperature: Температура (К), за замовчуванням 288.15 К (15°C)
    
    Повертає:
        Тиск на заданій висоті (Па)
    """
    if altitude < 0:
        raise ValueError("Висота повинна бути невід'ємною")
    if sea_level_pressure <= 0:
        raise ValueError("Тиск на рівні моря повинен бути додатнім")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Барометрична формула
    pressure = sea_level_pressure * np.exp(-GRAVITY * altitude / (GAS_CONSTANT_DRY_AIR * temperature))
    return pressure

def altitude_from_pressure(pressure: float, sea_level_pressure: float = 101325,
                          temperature: float = 288.15) -> float:
    """
    Обчислити висоту за тиском за барометричною формулою.
    
    Параметри:
        pressure: Тиск на заданій висоті (Па)
        sea_level_pressure: Тиск на рівні моря (Па), за замовчуванням 101325 Па
        temperature: Температура (К), за замовчуванням 288.15 К (15°C)
    
    Повертає:
        Висота над рівнем моря (м)
    """
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if sea_level_pressure <= 0:
        raise ValueError("Тиск на рівні моря повинен бути додатнім")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Обернена барометрична формула
    altitude = -(GAS_CONSTANT_DRY_AIR * temperature / GRAVITY) * np.log(pressure / sea_level_pressure)
    return altitude

def dew_point(temperature: float, relative_humidity: float) -> float:
    """
    Обчислити точку роси.
    
    Параметри:
        temperature: Температура повітря (К)
        relative_humidity: Відносна вологість (%)
    
    Повертає:
        Точка роси (К)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if relative_humidity < 0 or relative_humidity > 100:
        raise ValueError("Відносна вологість повинна бути в діапазоні 0-100%")
    
    # Перетворити температуру в °C
    temp_c = temperature - 273.15
    
    # Обчислити тиск насиченої пари при поточній температурі
    es = saturation_vapor_pressure(temperature)
    
    # Обчислити фактичний тиск пари
    e = (relative_humidity / 100) * es
    
    # Перетворити тиск пари в гПа
    e_hpa = e / 100
    
    # Формула Magnus для обчислення точки роси
    # Для точок роси вище 0°C
    if e_hpa >= 6.1094:
        b = 17.625
        c = 243.04
        dew_point_c = (c * np.log(e_hpa / 6.1094)) / (b - np.log(e_hpa / 6.1094))
    else:
        # Для точок роси нижче 0°C
        b = 22.46
        c = 272.62
        dew_point_c = (c * np.log(e_hpa / 6.1071)) / (b - np.log(e_hpa / 6.1071))
    
    # Перетворити в Кельвіни
    dew_point_k = dew_point_c + 273.15
    return dew_point_k

def lapse_rate(temperature1: float, temperature2: float, 
              altitude1: float, altitude2: float) -> float:
    """
    Обчислити температурний градієнт (лапс-рейт).
    
    Параметри:
        temperature1: Температура на висоті 1 (К)
        temperature2: Температура на висоті 2 (К)
        altitude1: Висота 1 (м)
        altitude2: Висота 2 (м)
    
    Повертає:
        Температурний градієнт (К/км)
    """
    if altitude1 == altitude2:
        raise ValueError("Висоти не повинні бути однаковими")
    
    # Обчислити температурний градієнт
    delta_temp = temperature2 - temperature1
    delta_alt = (altitude2 - altitude1) / 1000  # Перетворити в км
    
    lapse_rate = -delta_temp / delta_alt  # Негативний знак для стандартного визначення
    return lapse_rate

def atmospheric_stability(lapse_rate_value: float, 
                         environmental_lapse_rate: float = 9.8) -> str:
    """
    Визначити атмосферну стабільність за температурним градієнтом.
    
    Параметри:
        lapse_rate_value: Спостережуваний температурний градієнт (К/км)
        environmental_lapse_rate: Адіабатичний градієнт (К/км), за замовчуванням 9.8
    
    Повертає:
        Тип атмосферної стабільності
    """
    if lapse_rate_value < 0:
        return "ізотермічна або інверсійна"
    elif lapse_rate_value < environmental_lapse_rate:
        return "стабільна"
    elif abs(lapse_rate_value - environmental_lapse_rate) < 0.1:
        return "нейтральна"
    else:
        return "нестабільна"

def cloud_base_height(surface_temperature: float, dew_point: float,
                     lapse_rate_dry: float = 9.8, lapse_rate_moist: float = 6.5) -> float:
    """
    Обчислити висоту основи хмар.
    
    Параметри:
        surface_temperature: Температура на поверхні (К)
        dew_point: Точка роси (К)
        lapse_rate_dry: Сухий адіабатичний градієнт (К/км), за замовчуванням 9.8
        lapse_rate_moist: Вологий адіабатичний градієнт (К/км), за замовчуванням 6.5
    
    Повертає:
        Висота основи хмар (м)
    """
    if surface_temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if dew_point <= 0:
        raise ValueError("Точка роси повинна бути додатньою")
    if lapse_rate_dry <= 0 or lapse_rate_moist <= 0:
        raise ValueError("Градієнти повинні бути додатніми")
    
    # Перетворити температури в °C
    temp_c = surface_temperature - 273.15
    dew_c = dew_point - 273.15
    
    # Формула для висоти основи хмар (приблизна)
    # Висота = 125 * (температура - точка роси)
    cloud_base = 125 * (temp_c - dew_c)  # Висота в метрах
    
    return cloud_base

def precipitation_type(temperature: float, wet_bulb_temperature: float) -> str:
    """
    Визначити тип опадів.
    
    Параметри:
        temperature: Температура повітря (К)
        wet_bulb_temperature: Температура мокрого термометра (К)
    
    Повертає:
        Тип опадів
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if wet_bulb_temperature <= 0:
        raise ValueError("Температура мокрого термометра повинна бути додатньою")
    
    # Перетворити температури в °C
    temp_c = temperature - 273.15
    wet_bulb_c = wet_bulb_temperature - 273.15
    
    # Визначити тип опадів
    if temp_c >= 0 and wet_bulb_c >= 0:
        return "дощ"
    elif temp_c < 0 and wet_bulb_c >= 0:
        return "льодяний дощ"
    elif temp_c < 0 and wet_bulb_c < 0:
        return "сніг"
    else:
        return "морось"

def visibility_in_fog(visibility: float, relative_humidity: float) -> float:
    """
    Коригувати видимість з урахуванням туману.
    
    Параметри:
        visibility: Видимість в ясну погоду (м)
        relative_humidity: Відносна вологість (%)
    
    Повертає:
        Скоригована видимість в тумані (м)
    """
    if visibility < 0:
        raise ValueError("Видимість повинна бути невід'ємною")
    if relative_humidity < 0 or relative_humidity > 100:
        raise ValueError("Відносна вологість повинна бути в діапазоні 0-100%")
    
    # Проста емпірична формула для корекції видимості
    if relative_humidity < 90:
        return visibility
    else:
        # Експоненційне зменшення видимості
        reduction_factor = np.exp(-0.05 * (relative_humidity - 90))
        corrected_visibility = visibility * reduction_factor
        return max(corrected_visibility, 50)  # Мінімальна видимість 50 м

def evapotranspiration(potential_evaporation: float, 
                     vegetation_fraction: float = 1.0) -> float:
    """
    Обчислити евapotранспірацію.
    
    Параметри:
        potential_evaporation: Потенційне випаровування (мм/добу)
        vegetation_fraction: Доля рослинності (0-1), за замовчуванням 1.0
    
    Повертає:
        Евapotранспірація (мм/добу)
    """
    if potential_evaporation < 0:
        raise ValueError("Потенційне випаровування повинно бути невід'ємним")
    if vegetation_fraction < 0 or vegetation_fraction > 1:
        raise ValueError("Доля рослинності повинна бути в діапазоні 0-1")
    
    # Евapotранспірація = потенційне випаровування * доля рослинності
    et = potential_evaporation * vegetation_fraction
    return et

def atmospheric_pressure_at_height(height: float, 
                                 surface_pressure: float = 101325) -> float:
    """
    Обчислити атмосферний тиск на заданій висоті (модель ISO 2533).
    
    Параметри:
        height: Висота над рівнем моря (м)
        surface_pressure: Тиск на поверхні (Па), за замовчуванням 101325 Па
    
    Повертає:
        Атмосферний тиск (Па)
    """
    if height < 0:
        raise ValueError("Висота повинна бути невід'ємною")
    if surface_pressure <= 0:
        raise ValueError("Тиск на поверхні повинен бути додатнім")
    
    # Модель ISO 2533 для стандартної атмосфери
    if height <= 11000:
        # Тропосфера
        T0 = 288.15  # Температура на рівні моря (К)
        L = 0.0065  # Температурний градієнт (К/м)
        T = T0 - L * height
        P = surface_pressure * (T / T0) ** (GRAVITY / (GAS_CONSTANT_DRY_AIR * L))
    elif height <= 20000:
        # Нижня стратосфера
        T = 216.65  # Ізотермічний шар
        P11 = surface_pressure * (216.65 / 288.15) ** (GRAVITY / (GAS_CONSTANT_DRY_AIR * 0.0065))
        P = P11 * np.exp(-GRAVITY * (height - 11000) / (GAS_CONSTANT_DRY_AIR * T))
    else:
        # Верхня стратосфера
        # Спрощена формула
        P = surface_pressure * np.exp(-height / 8000)  # Приблизна формула
    
    return P

def wind_speed_from_components(u_component: float, v_component: float) -> float:
    """
    Обчислити швидкість вітру з компонентів.
    
    Параметри:
        u_component: Зональна компонента вітру (м/с)
        v_component: Меридіональна компонента вітру (м/с)
    
    Повертає:
        Швидкість вітру (м/с)
    """
    # Обчислити швидкість вітру
    wind_speed = np.sqrt(u_component**2 + v_component**2)
    return wind_speed

def wind_direction_from_components(u_component: float, v_component: float) -> float:
    """
    Обчислити напрямок вітру з компонентів.
    
    Параметри:
        u_component: Зональна компонента вітру (м/с)
        v_component: Меридіональна компонента вітру (м/с)
    
    Повертає:
        Напрямок вітру (градуси, від півночі за годинниковою стрілкою)
    """
    # Обчислити напрямок вітру
    wind_direction = (270 - np.degrees(np.arctan2(v_component, u_component))) % 360
    return wind_direction

def u_v_components_from_wind(speed: float, direction: float) -> Tuple[float, float]:
    """
    Обчислити компоненти вітру зі швидкості та напрямку.
    
    Параметри:
        speed: Швидкість вітру (м/с)
        direction: Напрямок вітру (градуси, від півночі за годинниковою стрілкою)
    
    Повертає:
        Кортеж (u_component, v_component)
    """
    if speed < 0:
        raise ValueError("Швидкість вітру повинна бути невід'ємною")
    if direction < 0 or direction > 360:
        raise ValueError("Напрямок вітру повинен бути в діапазоні 0-360°")
    
    # Перетворити напрямок в радіани
    direction_rad = np.radians(270 - direction)
    
    # Обчислити компоненти
    u_component = -speed * np.sin(direction_rad)
    v_component = -speed * np.cos(direction_rad)
    
    return u_component, v_component

def pressure_gradient_force(pressure_gradient: float, density: float = 1.225) -> float:
    """
    Обчислити силу тискового градієнта.
    
    Параметри:
        pressure_gradient: Градієнт тиску (Па/м)
        density: Густина повітря (кг/м³), за замовчуванням 1.225 кг/м³
    
    Повертає:
        Сила тискового градієнта (м/с²)
    """
    if density <= 0:
        raise ValueError("Густина повітря повинна бути додатньою")
    
    # Сила тискового градієнта на одиницю маси
    pgf = -pressure_gradient / density
    return pgf

def thermal_wind(u_wind_bottom: float, u_wind_top: float, 
                thickness: float, latitude: float) -> float:
    """
    Обчислити термічний вітер.
    
    Термічний вітер - це зміна геострофного вітру з висотою.
    
    Параметри:
        u_wind_bottom: Зональна компонента вітру на нижньому рівні (м/с)
        u_wind_top: Зональна компонента вітру на верхньому рівні (м/с)
        thickness: Товщина шару (м)
        latitude: Географічна широта (градуси)
    
    Повертає:
        Термічний вітер (м/с)
    """
    if thickness <= 0:
        raise ValueError("Товщина шару повинна бути додатньою")
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    
    # Параметр Коріоліса
    f = coriolis_parameter(latitude)
    
    # Термічний вітер
    if abs(f) < 1e-10:
        raise ValueError("Параметр Коріоліса занадто малий (екваторіальна область)")
    
    thermal_wind_speed = (u_wind_top - u_wind_bottom) * GRAVITY / (f * thickness)
    return thermal_wind_speed

def atmospheric_density(pressure: float, temperature: float, 
                      humidity: float = 0) -> float:
    """
    Обчислити густину атмосфери.
    
    Параметри:
        pressure: Атмосферний тиск (Па)
        temperature: Температура (К)
        humidity: Відносна вологість (0-1), за замовчуванням 0
    
    Повертає:
        Густина атмосфери (кг/м³)
    """
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if humidity < 0 or humidity > 1:
        raise ValueError("Відносна вологість повинна бути в діапазоні 0-1")
    
    # Обчислити густину сухого повітря
    rho_dry = pressure / (GAS_CONSTANT_DRY_AIR * temperature)
    
    # Якщо є вологість, скоригувати густину
    if humidity > 0:
        # Приблизно обчислити тиск водяної пари
        vapor_pressure = humidity * saturation_vapor_pressure(temperature)
        # Густина водяної пари
        rho_vapor = vapor_pressure / (GAS_CONSTANT_WATER_VAPOR * temperature)
        # Загальна густина
        rho = rho_dry - (vapor_pressure / (GAS_CONSTANT_DRY_AIR * temperature)) + rho_vapor
    else:
        rho = rho_dry
    
    return rho

def atmospheric_scale_height(temperature: float, molecular_weight: float = 0.0289644) -> float:
    """
    Обчислити масштабну висоту атмосфери.
    
    Параметри:
        temperature: Середня температура (К)
        molecular_weight: Молекулярна маса повітря (кг/моль), за замовчуванням 0.0289644
    
    Повертає:
        Масштабна висота (м)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if molecular_weight <= 0:
        raise ValueError("Молекулярна маса повинна бути додатньою")
    
    # Універсальна газова стала
    R_universal = 8.31446261815324  # Дж/(моль·К)
    
    # Масштабна висота
    H = R_universal * temperature / (molecular_weight * GRAVITY)
    return H

def atmospheric_optical_depth(aerosol_optical_depth: float, 
                            gas_optical_depth: float = 0) -> float:
    """
    Обчислити оптичну товщу атмосфери.
    
    Параметри:
        aerosol_optical_depth: Оптична товща аерозолів
        gas_optical_depth: Оптична товща газів, за замовчуванням 0
    
    Повертає:
        Загальна оптична товща атмосфери
    """
    if aerosol_optical_depth < 0:
        raise ValueError("Оптична товща аерозолів повинна бути невід'ємною")
    if gas_optical_depth < 0:
        raise ValueError("Оптична товща газів повинна бути невід'ємною")
    
    # Загальна оптична товща
    total_optical_depth = aerosol_optical_depth + gas_optical_depth
    return total_optical_depth

def atmospheric_transmittance(optical_depth: float) -> float:
    """
    Обчислити пропускання атмосфери.
    
    Параметри:
        optical_depth: Оптична товща атмосфери
    
    Повертає:
        Пропускання атмосфери (0-1)
    """
    if optical_depth < 0:
        raise ValueError("Оптична товща повинна бути невід'ємною")
    
    # Пропускання за законом Бера-Ламберта
    transmittance = np.exp(-optical_depth)
    return transmittance

def atmospheric_emissivity(temperature: float, humidity: float, 
                          cloud_cover: float = 0) -> float:
    """
    Обчислити емісивність атмосфери.
    
    Параметри:
        temperature: Температура атмосфери (К)
        humidity: Відносна вологість (0-1)
        cloud_cover: Хмарність (0-1), за замовчуванням 0
    
    Повертає:
        Емісивність атмосфери (0-1)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if humidity < 0 or humidity > 1:
        raise ValueError("Відносна вологість повинна бути в діапазоні 0-1")
    if cloud_cover < 0 or cloud_cover > 1:
        raise ValueError("Хмарність повинна бути в діапазоні 0-1")
    
    # Емісивність вологого повітря
    emissivity_humidity = 0.7 * humidity**0.5
    
    # Емісивність хмар
    emissivity_clouds = 0.95 * cloud_cover
    
    # Загальна емісивність
    emissivity = emissivity_humidity + emissivity_clouds * (1 - emissivity_humidity)
    
    # Обмежити значення діапазоном 0-1
    emissivity = max(0, min(1, emissivity))
    return emissivity

def atmospheric_radiation(temperature: float, emissivity: float) -> float:
    """
    Обчислити випромінювання атмосфери за законом Стефана-Больцмана.
    
    Параметри:
        temperature: Температура атмосфери (К)
        emissivity: Емісивність атмосфери (0-1)
    
    Повертає:
        Випромінювання атмосфери (Вт/м²)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if emissivity < 0 or emissivity > 1:
        raise ValueError("Емісивність повинна бути в діапазоні 0-1")
    
    # Закон Стефана-Больцмана
    radiation = emissivity * STEFAN_BOLTZMANN * temperature**4
    return radiation

def atmospheric_heat_capacity(pressure: float, temperature: float) -> float:
    """
    Обчислити теплоємність атмосфери.
    
    Параметри:
        pressure: Атмосферний тиск (Па)
        temperature: Температура (К)
    
    Повертає:
        Теплоємність атмосфери (Дж/К)
    """
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Обчислити об'єм повітря (приблизно для 1 м³)
    volume = 1.0  # м³
    
    # Обчислити масу повітря
    mass = pressure * volume / (GAS_CONSTANT_DRY_AIR * temperature)
    
    # Теплоємність при постійному тиску
    heat_capacity = mass * SPECIFIC_HEAT_DRY_AIR
    return heat_capacity

def atmospheric_energy_content(temperature: float, pressure: float, 
                             volume: float = 1.0) -> float:
    """
    Обчислити енергетичний вміст атмосфери.
    
    Параметри:
        temperature: Температура (К)
        pressure: Атмосферний тиск (Па)
        volume: Об'єм (м³), за замовчуванням 1.0 м³
    
    Повертає:
        Енергетичний вміст (Дж)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if volume <= 0:
        raise ValueError("Об'єм повинен бути додатнім")
    
    # Обчислити масу повітря
    mass = pressure * volume / (GAS_CONSTANT_DRY_AIR * temperature)
    
    # Енергетичний вміст
    energy = mass * SPECIFIC_HEAT_DRY_AIR * temperature
    return energy

def atmospheric_buoyancy(temperature_parcel: float, temperature_environment: float,
                        pressure: float) -> float:
    """
    Обчислити буйянс повітряної частинки.
    
    Параметри:
        temperature_parcel: Температура повітряної частинки (К)
        temperature_environment: Температура навколишнього середовища (К)
        pressure: Атмосферний тиск (Па)
    
    Повертає:
        Буйянс (м/с²)
    """
    if temperature_parcel <= 0:
        raise ValueError("Температура частинки повинна бути додатньою")
    if temperature_environment <= 0:
        raise ValueError("Температура середовища повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    
    # Буйянс
    buoyancy = GRAVITY * (temperature_parcel - temperature_environment) / temperature_environment
    return buoyancy

def atmospheric_stability_parameter(buoyancy_frequency: float) -> str:
    """
    Визначити параметр атмосферної стабільності за частотою Буйяна.
    
    Параметри:
        buoyancy_frequency: Частота Буйяна (с⁻¹)
    
    Повертає:
        Тип атмосферної стабільності
    """
    N_squared = buoyancy_frequency**2
    
    if N_squared > 0:
        return "стабільна"
    elif N_squared < 0:
        return "нестабільна"
    else:
        return "нейтральна"

def atmospheric_buoyancy_frequency(temperature_lapse_rate: float,
                                 temperature: float) -> float:
    """
    Обчислити частоту Буйяна (частоту Брента-Вяйсяля).
    
    Параметри:
        temperature_lapse_rate: Температурний градієнт (К/м)
        temperature: Температура (К)
    
    Повертає:
        Частота Буйяна (с⁻¹)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Частота Буйяна
    N_squared = (GRAVITY / temperature) * (GRAVITY / GAS_CONSTANT_DRY_AIR - temperature_lapse_rate)
    N = np.sqrt(max(0, N_squared))  # Уникнути уявних чисел
    return N

def atmospheric_rossby_number(characteristic_length: float, 
                             latitude: float, velocity: float = 10) -> float:
    """
    Обчислити число Россбі.
    
    Параметри:
        characteristic_length: Характерна довжина (м)
        latitude: Географічна широта (градуси)
        velocity: Характерна швидкість (м/с), за замовчуванням 10 м/с
    
    Повертає:
        Число Россбі
    """
    if characteristic_length <= 0:
        raise ValueError("Характерна довжина повинна бути додатньою")
    if latitude < -90 or latitude > 90:
        raise ValueError("Широта повинна бути в діапазоні -90° до 90°")
    if velocity <= 0:
        raise ValueError("Швидкість повинна бути додатньою")
    
    # Параметр Коріоліса
    f = coriolis_parameter(latitude)
    
    # Уникнути ділення на нуль
    if abs(f) < 1e-10:
        return float('inf')
    
    # Число Россбі
    rossby_number = velocity / (f * characteristic_length)
    return rossby_number

def atmospheric_reynolds_number(velocity: float, length: float, 
                              kinematic_viscosity: float = 1.5e-5) -> float:
    """
    Обчислити число Рейнольдса для атмосфери.
    
    Параметри:
        velocity: Швидкість (м/с)
        length: Характерна довжина (м)
        kinematic_viscosity: Кінематична в'язкість (м²/с), за замовчуванням 1.5e-5 м²/с
    
    Повертає:
        Число Рейнольдса
    """
    if velocity < 0:
        raise ValueError("Швидкість повинна бути невід'ємною")
    if length <= 0:
        raise ValueError("Довжина повинна бути додатньою")
    if kinematic_viscosity <= 0:
        raise ValueError("Кінематична в'язкість повинна бути додатньою")
    
    # Число Рейнольдса
    reynolds_number = velocity * length / kinematic_viscosity
    return reynolds_number

def atmospheric_prandtl_number() -> float:
    """
    Обчислити число Прандтля для повітря.
    
    Повертає:
        Число Прандтля для повітря
    """
    # Для повітря при кімнатній температурі
    prandtl_number = 0.71
    return prandtl_number

def atmospheric_schmidt_number() -> float:
    """
    Обчислити число Шмідта для повітря.
    
    Повертає:
        Число Шмідта для повітря
    """
    # Для повітря при кімнатній температурі
    schmidt_number = 0.6
    return schmidt_number

def atmospheric_froude_number(velocity: float, length: float) -> float:
    """
    Обчислити число Фруда для атмосфери.
    
    Параметри:
        velocity: Швидкість (м/с)
        length: Характерна довжина (м)
    
    Повертає:
        Число Фруда
    """
    if velocity < 0:
        raise ValueError("Швидкість повинна бути невід'ємною")
    if length <= 0:
        raise ValueError("Довжина повинна бути додатньою")
    
    # Число Фруда
    froude_number = velocity / np.sqrt(GRAVITY * length)
    return froude_number

def atmospheric_mach_number(velocity: float, temperature: float) -> float:
    """
    Обчислити число Маха для атмосфери.
    
    Параметри:
        velocity: Швидкість (м/с)
        temperature: Температура (К)
    
    Повертає:
        Число Маха
    """
    if velocity < 0:
        raise ValueError("Швидкість повинна бути невід'ємною")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Швидкість звуку в повітрі
    speed_of_sound = np.sqrt(GAS_CONSTANT_DRY_AIR * temperature * 1.4)  # 1.4 для двоатомного газу
    
    # Уникнути ділення на нуль
    if speed_of_sound < 1e-10:
        return float('inf')
    
    # Число Маха
    mach_number = velocity / speed_of_sound
    return mach_number

def atmospheric_karman_constant() -> float:
    """
    Повернути константу Кармана.
    
    Повертає:
        Константа Кармана
    """
    # Константа Кармана
    karman_constant = 0.4
    return karman_constant

def atmospheric_surface_layer_thickness(wind_speed: float, 
                                      friction_velocity: float) -> float:
    """
    Обчислити товщину приземного шару атмосфери.
    
    Параметри:
        wind_speed: Швидкість вітру (м/с)
        friction_velocity: Швидкість тертя (м/с)
    
    Повертає:
        Товщина приземного шару (м)
    """
    if wind_speed < 0:
        raise ValueError("Швидкість вітру повинна бути невід'ємною")
    if friction_velocity <= 0:
        raise ValueError("Швидкість тертя повинна бути додатньою")
    
    # Константа Кармана
    kappa = atmospheric_karman_constant()
    
    # Товщина приземного шару
    thickness = wind_speed * kappa / friction_velocity
    return thickness

def atmospheric_friction_velocity(stress: float, density: float = 1.225) -> float:
    """
    Обчислити швидкість тертя.
    
    Параметри:
        stress: Напруження тертя (Па)
        density: Густина повітря (кг/м³), за замовчуванням 1.225 кг/м³
    
    Повертає:
        Швидкість тертя (м/с)
    """
    if stress < 0:
        raise ValueError("Напруження тертя повинно бути невід'ємним")
    if density <= 0:
        raise ValueError("Густина повітря повинна бути додатньою")
    
    # Швидкість тертя
    friction_velocity = np.sqrt(stress / density)
    return friction_velocity

def atmospheric_obukhov_length(temperature: float, friction_velocity: float,
                             heat_flux: float) -> float:
    """
    Обчислити довжину Обухова.
    
    Параметри:
        temperature: Температура (К)
        friction_velocity: Швидкість тертя (м/с)
        heat_flux: Тепловий потік (Вт/м²)
    
    Повертає:
        Довжина Обухова (м)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if friction_velocity <= 0:
        raise ValueError("Швидкість тертя повинна бути додатньою")
    
    # Константа Кармана
    kappa = atmospheric_karman_constant()
    
    # Уникнути ділення на нуль
    if abs(heat_flux) < 1e-10:
        return float('inf') if heat_flux >= 0 else float('-inf')
    
    # Довжина Обухова
    obukhov_length = - (friction_velocity**3 * temperature) / (kappa * GRAVITY * heat_flux)
    return obukhov_length

def atmospheric_monin_obukhov_similarity(z_over_L: float) -> Tuple[float, float]:
    """
    Обчислити функції подібності Моніна-Обухова.
    
    Параметри:
        z_over_L: Відношення висоти до довжини Обухова
    
    Повертає:
        Кортеж (phi_m, phi_h) - функції подібності для імпульсу та тепла
    """
    # Функції подібності Моніна-Обухова
    if z_over_L >= 0:
        # Стабільна атмосфера
        phi_m = 1 + 5 * z_over_L
        phi_h = phi_m
    else:
        # Нестабільна атмосфера
        x = (1 - 16 * z_over_L)**0.25
        phi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x*x) / 2) - 2 * np.arctan(x) + np.pi/2
        phi_h = 2 * np.log((1 + x*x) / 2)
    
    return phi_m, phi_h

def atmospheric_boundary_layer_height(surface_heat_flux: float, 
                                    friction_velocity: float,
                                    coriolis_parameter: float) -> float:
    """
    Обчислити висоту приземного шару атмосфери.
    
    Параметри:
        surface_heat_flux: Поверхневий тепловий потік (Вт/м²)
        friction_velocity: Швидкість тертя (м/с)
        coriolis_parameter: Параметр Коріоліса (с⁻¹)
    
    Повертає:
        Висота приземного шару (м)
    """
    if friction_velocity <= 0:
        raise ValueError("Швидкість тертя повинна бути додатньою")
    if abs(coriolis_parameter) < 1e-10:
        raise ValueError("Параметр Коріоліса занадто малий")
    
    # Масштаб часу тертя
    tau = friction_velocity / (GRAVITY * surface_heat_flux) if abs(surface_heat_flux) > 1e-10 else 1
    
    # Висота приземного шару
    h = friction_velocity / (coriolis_parameter * np.sqrt(1 + (friction_velocity * tau)**2))
    return h

def atmospheric_convective_velocity(surface_heat_flux: float, 
                                  boundary_layer_height: float,
                                  density: float = 1.225) -> float:
    """
    Обчислити конвективну швидкість.
    
    Параметри:
        surface_heat_flux: Поверхневий тепловий потік (Вт/м²)
        boundary_layer_height: Висота приземного шару (м)
        density: Густина повітря (кг/м³), за замовчуванням 1.225 кг/м³
    
    Повертає:
        Конвективна швидкість (м/с)
    """
    if surface_heat_flux < 0:
        raise ValueError("Тепловий потік повинен бути невід'ємним для конвекції")
    if boundary_layer_height <= 0:
        raise ValueError("Висота приземного шару повинна бути додатньою")
    if density <= 0:
        raise ValueError("Густина повітря повинна бути додатньою")
    
    # Конвективна швидкість
    w_star = ((GRAVITY * surface_heat_flux * boundary_layer_height) / 
              (density * SPECIFIC_HEAT_DRY_AIR))**(1/3)
    return w_star

def atmospheric_convective_time_scale(boundary_layer_height: float, 
                                    convective_velocity: float) -> float:
    """
    Обчислити конвективну часову шкалу.
    
    Параметри:
        boundary_layer_height: Висота приземного шару (м)
        convective_velocity: Конвективна швидкість (м/с)
    
    Повертає:
        Конвективна часова шкала (с)
    """
    if boundary_layer_height <= 0:
        raise ValueError("Висота приземного шару повинна бути додатньою")
    if convective_velocity <= 0:
        raise ValueError("Конвективна швидкість повинна бути додатньою")
    
    # Конвективна часова шкала
    tau_conv = boundary_layer_height / convective_velocity
    return tau_conv

def atmospheric_turbulent_kinetic_energy(wind_shear: float, 
                                       buoyancy_flux: float) -> float:
    """
    Обчислити турбулентну кінетичну енергію.
    
    Параметри:
        wind_shear: Градієнт швидкості (с⁻¹)
        buoyancy_flux: Потік буйянсу (м²/с³)
    
    Повертає:
        Турбулентна кінетична енергія (м²/с²)
    """
    if wind_shear < 0:
        raise ValueError("Градієнт швидкості повинен бути невід'ємним")
    
    # Турбулентна кінетична енергія (спрощена формула)
    # TKE = (shear production + buoyancy production) * timescale
    # Тут ми використовуємо спрощену оцінку
    tke = 0.5 * (wind_shear**2 + max(0, buoyancy_flux))
    return tke

def atmospheric_mixing_ratio(pressure: float, vapor_pressure: float) -> float:
    """
    Обчислити відношення змішування.
    
    Параметри:
        pressure: Атмосферний тиск (Па)
        vapor_pressure: Тиск водяної пари (Па)
    
    Повертає:
        Відношення змішування (кг/кг)
    """
    if pressure <= 0:
        raise ValueError("Атмосферний тиск повинен бути додатнім")
    if vapor_pressure < 0:
        raise ValueError("Тиск водяної пари повинен бути невід'ємним")
    if vapor_pressure >= pressure:
        raise ValueError("Тиск водяної пари не може перевищувати атмосферний тиск")
    
    # Відношення змішування
    epsilon = GAS_CONSTANT_DRY_AIR / GAS_CONSTANT_WATER_VAPOR
    mixing_ratio = epsilon * vapor_pressure / (pressure - vapor_pressure)
    return mixing_ratio

def atmospheric_specific_volume(temperature: float, pressure: float) -> float:
    """
    Обчислити питомий об'єм повітря.
    
    Параметри:
        temperature: Температура (К)
        pressure: Атмосферний тиск (Па)
    
    Повертає:
        Питомий об'єм (м³/кг)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    
    # Питомий об'єм
    specific_volume = GAS_CONSTANT_DRY_AIR * temperature / pressure
    return specific_volume

def atmospheric_enthalpy(temperature: float, mixing_ratio: float) -> float:
    """
    Обчислити ентальпію повітря.
    
    Параметри:
        temperature: Температура (К)
        mixing_ratio: Відношення змішування (кг/кг)
    
    Повертає:
        Ентальпія (Дж/кг)
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if mixing_ratio < 0:
        raise ValueError("Відношення змішування повинно бути невід'ємним")
    
    # Ентальпія сухого повітря
    h_dry = SPECIFIC_HEAT_DRY_AIR * temperature
    
    # Ентальпія водяної пари
    h_vapor = (SPECIFIC_HEAT_DRY_AIR + 1.84e3) * temperature  # Приблизно
    
    # Загальна ентальпія
    enthalpy = h_dry + mixing_ratio * (h_vapor - LATENT_HEAT_VAPORIZATION)
    return enthalpy

def atmospheric_entropy(temperature: float, pressure: float, 
                      mixing_ratio: float) -> float:
    """
    Обчислити ентропію повітря.
    
    Параметри:
        temperature: Температура (К)
        pressure: Атмосферний тиск (Па)
        mixing_ratio: Відношення змішування (кг/кг)
    
    Повертає:
        Ентропія (Дж/(кг·К))
    """
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    if pressure <= 0:
        raise ValueError("Тиск повинен бути додатнім")
    if mixing_ratio < 0:
        raise ValueError("Відношення змішування повинно бути невід'ємним")
    
    # Ентропія сухого повітря
    s_dry = SPECIFIC_HEAT_DRY_AIR * np.log(temperature) - GAS_CONSTANT_DRY_AIR * np.log(pressure)
    
    # Ентропія водяної пари
    if mixing_ratio > 0:
        s_vapor = (SPECIFIC_HEAT_DRY_AIR + 1.84e3) * np.log(temperature) - GAS_CONSTANT_WATER_VAPOR * np.log(pressure)
        entropy = s_dry + mixing_ratio * s_vapor
    else:
        entropy = s_dry
    
    return entropy
