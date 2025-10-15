"""
Модуль для обчислювальної астрономії
Computational Astronomy Module
"""
from typing import Union, Tuple, List, Optional, Dict, Any
import math

# Астрономічні константи
# Astronomical constants
AU = 149597870700  # Астрономічна одиниця (м)
EARTH_RADIUS = 6371000  # Радіус Землі (м)
EARTH_MASS = 5.972e24  # Маса Землі (кг)
SUN_MASS = 1.989e30  # Маса Сонця (кг)
GRAVITATIONAL_CONSTANT = 6.67430e-11  # Гравітаційна константа (м³/(кг·с²))
SPEED_OF_LIGHT = 299792458  # Швидкість світла (м/с)
PLANCK_CONSTANT = 6.62607015e-34  # Константа Планка (Дж·с)
BOLTZMANN_CONSTANT = 1.380649e-23  # Константа Больцмана (Дж/К)
STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # Константа Стефана-Больцмана (Вт/(м²·К⁴))
HUBBLE_CONSTANT = 70  # Постійна Хаббла (км/(с·Мпк))
PARSEC = 3.08567758149137e16  # Парсек (м)
LIGHT_YEAR = 9.4607304725808e15  # Світловий рік (м)
SOLAR_RADIUS = 6.957e8  # Радіус Сонця (м)
SOLAR_LUMINOSITY = 3.828e26  # Світність Сонця (Вт)
JULIAN_YEAR = 365.25  # Юліанський рік (днів)
JULIAN_DAY = 86400  # Юліанський день (секунд)
SIDEREAL_YEAR = 365.256363004  # Зоряний рік (днів)
TROPICAL_YEAR = 365.242190419  # Тропічний рік (днів)

def julian_date(year: int, month: int, day: int, hour: float = 0, minute: float = 0, second: float = 0) -> float:
    """
    Обчислити юліанську дату.
    
    Параметри:
        year: Рік
        month: Місяць
        day: День
        hour: Година (0-23), за замовчуванням 0
        minute: Хвилина (0-59), за замовчуванням 0
        second: Секунда (0-59), за замовчуванням 0
    
    Повертає:
        Юліанська дата
    """
    if month <= 2:
        year -= 1
        month += 12
    
    A = year // 100
    B = 2 - A + A // 4
    
    jd = (
        int(365.25 * (year + 4716)) + 
        int(30.6001 * (month + 1)) + 
        day + B - 1524.5 +
        hour / 24 + minute / 1440 + second / 86400
    )
    
    return jd

def modified_julian_date(julian_date_value: float) -> float:
    """
    Обчислити модифіковану юліанську дату.
    
    Параметри:
        julian_date_value: Юліанська дата
    
    Повертає:
        Модифікована юліанська дата
    """
    return julian_date_value - 2400000.5

def julian_centuries(julian_date_value: float) -> float:
    """
    Обчислити юліанські століття від епохи J2000.0.
    
    Параметри:
        julian_date_value: Юліанська дата
    
    Повертає:
        Юліанські століття від J2000.0
    """
    J2000 = 2451545.0  # Юліанська дата для епохи J2000.0 (12:00:00 1.01.2000)
    return (julian_date_value - J2000) / 36525.0

def sidereal_time(julian_date_value: float, longitude: float = 0) -> float:
    """
    Обчислити зоряний час.
    
    Параметри:
        julian_date_value: Юліанська дата
        longitude: Географічна довгота (градуси), за замовчуванням 0
    
    Повертає:
        Зоряний час (години)
    """
    T = julian_centuries(julian_date_value)
    
    # Зоряний час в Гринвічі на 0h UT
    GMST = (
        280.46061837 + 
        360.98564736629 * (julian_date_value - 2451545.0) + 
        0.000387933 * T**2 - 
        T**3 / 38710000
    )
    
    # Нормалізація до 0-360 градусів
    GMST = GMST % 360
    if GMST < 0:
        GMST += 360
    
    # Додавання довготи
    LST = GMST + longitude
    
    # Нормалізація до 0-360 градусів
    LST = LST % 360
    if LST < 0:
        LST += 360
    
    # Перетворення в години
    LST_hours = LST / 15.0
    
    return LST_hours

def orbital_elements_to_position(semi_major_axis: float, eccentricity: float, 
                               inclination: float, longitude_ascending: float,
                               argument_periapsis: float, mean_anomaly: float,
                               distance: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Обчислити позицію тіла на орбіті з орбітальних елементів.
    
    Параметри:
        semi_major_axis: Велика піввісь (м)
        eccentricity: Ексцентриситет
        inclination: Нахил орбіти (градуси)
        longitude_ascending: Довгота висхідного вузла (градуси)
        argument_periapsis: Аргумент перицентру (градуси)
        mean_anomaly: Середня аномалія (градуси)
        distance: Відстань (м), якщо відома
    
    Повертає:
        Кортеж (x, y, z) - декартові координати (м)
    """
    if semi_major_axis <= 0:
        raise ValueError("Велика піввісь повинна бути додатньою")
    if eccentricity < 0 or eccentricity >= 1:
        raise ValueError("Ексцентриситет повинен бути в діапазоні [0, 1)")
    if inclination < 0 or inclination > 180:
        raise ValueError("Нахил орбіти повинен бути в діапазоні [0, 180]")
    if longitude_ascending < 0 or longitude_ascending > 360:
        raise ValueError("Довгота висхідного вузла повинна бути в діапазоні [0, 360]")
    if argument_periapsis < 0 or argument_periapsis > 360:
        raise ValueError("Аргумент перицентру повинен бути в діапазоні [0, 360]")
    if mean_anomaly < 0 or mean_anomaly > 360:
        raise ValueError("Середня аномалія повинна бути в діапазоні [0, 360]")
    
    # Перетворення в радіани
    inc_rad = math.radians(inclination)
    lan_rad = math.radians(longitude_ascending)
    arg_rad = math.radians(argument_periapsis)
    M_rad = math.radians(mean_anomaly)
    
    # Розв'язання рівняння Кеплера для ексцентричної аномалії
    E = M_rad
    for _ in range(10):  # Ітераційний метод
        E_new = M_rad + eccentricity * math.sin(E)
        if abs(E_new - E) < 1e-10:
            break
        E = E_new
    
    # Істинна аномалія
    nu = 2 * math.atan2(
        math.sqrt(1 + eccentricity) * math.sin(E / 2),
        math.sqrt(1 - eccentricity) * math.cos(E / 2)
    )
    
    # Відстань, якщо не задана
    if distance is None:
        r = semi_major_axis * (1 - eccentricity * math.cos(E))
    else:
        r = distance
    
    # Координати в площині орбіти
    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)
    z_orb = 0
    
    # Поворот до інерційної системи координат
    # Поворот навколо осі z на аргумент перицентру
    x1 = x_orb * math.cos(arg_rad) - y_orb * math.sin(arg_rad)
    y1 = x_orb * math.sin(arg_rad) + y_orb * math.cos(arg_rad)
    z1 = z_orb
    
    # Поворот навколо осі x на нахил орбіти
    x2 = x1
    y2 = y1 * math.cos(inc_rad) - z1 * math.sin(inc_rad)
    z2 = y1 * math.sin(inc_rad) + z1 * math.cos(inc_rad)
    
    # Поворот навколо осі z на довготу висхідного вузла
    x = x2 * math.cos(lan_rad) - y2 * math.sin(lan_rad)
    y = x2 * math.sin(lan_rad) + y2 * math.cos(lan_rad)
    z = z2
    
    return x, y, z

def position_to_orbital_elements(x: float, y: float, z: float, 
                               vx: float, vy: float, vz: float,
                               central_mass: float = SUN_MASS) -> Dict[str, float]:
    """
    Обчислити орбітальні елементи з декартових координат і швидкостей.
    
    Параметри:
        x, y, z: Декартові координати (м)
        vx, vy, vz: Компоненти швидкості (м/с)
        central_mass: Маса центрального тіла (кг), за замовчуванням маса Сонця
    
    Повертає:
        Словник з орбітальними елементами:
        - semi_major_axis: Велика піввісь (м)
        - eccentricity: Ексцентриситет
        - inclination: Нахил орбіти (градуси)
        - longitude_ascending: Довгота висхідного вузла (градуси)
        - argument_periapsis: Аргумент перицентру (градуси)
        - true_anomaly: Істинна аномалія (градуси)
    """
    # Позиція і швидкість як вектори
    r_vec = [x, y, z]
    v_vec = [vx, vy, vz]
    
    # Відстань і швидкість
    r = math.sqrt(x**2 + y**2 + z**2)
    v = math.sqrt(vx**2 + vy**2 + vz**2)
    
    # Кутовий момент
    h_vec = [
        y * vz - z * vy,
        z * vx - x * vz,
        x * vy - y * vx
    ]
    h = math.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2)
    
    # Енергія
    E = 0.5 * v**2 - GRAVITATIONAL_CONSTANT * central_mass / r
    
    # Велика піввісь
    a = -GRAVITATIONAL_CONSTANT * central_mass / (2 * E)
    
    # Ексцентриситет
    # Обчислення вектора ексцентриситету
    mu = GRAVITATIONAL_CONSTANT * central_mass
    e_vec = [
        (v**2 - mu/r) * x - (x*vx + y*vy + z*vz) * vx,
        (v**2 - mu/r) * y - (x*vx + y*vy + z*vz) * vy,
        (v**2 - mu/r) * z - (x*vx + y*vy + z*vz) * vz
    ]
    e_vec = [comp / mu for comp in e_vec]
    e = math.sqrt(e_vec[0]**2 + e_vec[1]**2 + e_vec[2]**2)
    
    # Нахил орбіти
    inc = math.acos(h_vec[2] / h) if h != 0 else 0
    inc_deg = math.degrees(inc)
    
    # Довгота висхідного вузла
    if h_vec[0] != 0 or h_vec[1] != 0:
        lan = math.atan2(h_vec[0], -h_vec[1])
        lan_deg = math.degrees(lan) % 360
    else:
        lan_deg = 0
    
    # Аргумент перицентру
    if e != 0:
        # Проекція вектора ексцентриситету на площину орбіти
        e_proj = math.sqrt(e_vec[0]**2 + e_vec[1]**2)
        if e_proj != 0:
            arg = math.atan2(e_vec[2], e_proj)
            arg_deg = math.degrees(arg) % 360
        else:
            arg_deg = 0
    else:
        arg_deg = 0
    
    # Істинна аномалія
    if e != 0:
        # Кут між вектором положення і вектором ексцентриситету
        r_dot_e = x * e_vec[0] + y * e_vec[1] + z * e_vec[2]
        cos_nu = r_dot_e / (r * e) if r * e != 0 else 1
        cos_nu = max(-1, min(1, cos_nu))  # Обмеження діапазону
        nu = math.acos(cos_nu)
        
        # Визначення знаку
        r_dot_v = x * vx + y * vy + z * vz
        if r_dot_v < 0:
            nu = 2 * math.pi - nu
        
        nu_deg = math.degrees(nu)
    else:
        # Для кругової орбіти використовуємо аргумент широти
        nu_deg = 0
    
    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'inclination': inc_deg,
        'longitude_ascending': lan_deg,
        'argument_periapsis': arg_deg,
        'true_anomaly': nu_deg
    }
