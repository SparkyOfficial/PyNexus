"""
Модуль для обчислювальної екології та науки про навколишнє середовище
Computational Environmental Science Module
"""
from typing import Union, Tuple, List, Optional, Dict, Any
import math

# Екологічні та природоохоронні константи
# Environmental and conservation constants
EARTH_RADIUS = 6371000  # Радіус Землі (м)
EARTH_SURFACE_AREA = 510072000000000  # Площа поверхні Землі (м²)
ATMOSPHERIC_MASS = 5.148e18  # Маса атмосфери (кг)
OCEAN_VOLUME = 1.332e18  # Об'єм океанів (м³)
CARBON_DIOXIDE_MOLAR_MASS = 0.04401  # Молярна маса CO₂ (кг/моль)
METHANE_MOLAR_MASS = 0.01604  # Молярна маса CH₄ (кг/моль)
NITROUS_OXIDE_MOLAR_MASS = 0.04401  # Молярна маса N₂O (кг/моль)
WATER_MOLAR_MASS = 0.018015  # Молярна маса H₂O (кг/моль)
AVOGADRO_CONSTANT = 6.02214076e23  # Число Авогадро (1/моль)
GAS_CONSTANT = 8.31446261815324  # Універсальна газова стала (Дж/(моль·К))
SPEED_OF_LIGHT = 299792458  # Швидкість світла (м/с)
STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # Константа Стефана-Больцмана (Вт/(м²·К⁴))
PLANCK_CONSTANT = 6.62607015e-34  # Константа Планка (Дж·с)
SOLAR_CONSTANT = 1361  # Сонячна стала (Вт/м²)
ALBEDO_EARTH = 0.3  # Альбедо Землі
GREENHOUSE_EFFECT_FACTOR = 0.39  # Фактор парникового ефекту

def carbon_footprint(energy_consumption: float, fuel_type: str = "coal") -> float:
    """
    Обчислити вуглецевий слід від споживання енергії.
    
    Параметри:
        energy_consumption: Споживання енергії (кВт·год)
        fuel_type: Тип палива ("coal", "natural_gas", "oil", "nuclear"), за замовчуванням "coal"
    
    Повертає:
        Вуглецевий слід (кг CO₂)
    """
    if energy_consumption < 0:
        raise ValueError("Споживання енергії повинне бути невід'ємним")
    
    # Коефіцієнти вуглецевого сліду для різних видів палива (кг CO₂/кВт·год)
    emission_factors = {
        "coal": 0.95,  # Вугілля
        "natural_gas": 0.45,  # Природний газ
        "oil": 0.75,  # Нафта
        "nuclear": 0.02,  # Ядерна енергія
        "hydro": 0.02,  # Гідроенергія
        "wind": 0.02,  # Вітрова енергія
        "solar": 0.04,  # Сонячна енергія
        "biomass": 0.23  # Біомаса
    }
    
    if fuel_type not in emission_factors:
        raise ValueError(f"Невідомий тип палива: {fuel_type}")
    
    return energy_consumption * emission_factors[fuel_type]

def greenhouse_gas_potential(co2_equivalent: float, gas_type: str = "co2") -> float:
    """
    Обчислити потенціал глобального потепління для різних парникових газів.
    
    Параметри:
        co2_equivalent: Еквівалент CO₂ (т CO₂)
        gas_type: Тип газу ("co2", "ch4", "n2o"), за замовчуванням "co2"
    
    Повертає:
        Потенціал глобального потепління (т CO₂-екв)
    """
    if co2_equivalent < 0:
        raise ValueError("Еквівалент CO₂ повинен бути невід'ємним")
    
    # Потенціал глобального потепління (GWP) на 100 років
    gwp_factors = {
        "co2": 1,  # Діоксид вуглецю
        "ch4": 28,  # Метан
        "n2o": 265  # Закис нітру
    }
    
    if gas_type not in gwp_factors:
        raise ValueError(f"Невідомий тип газу: {gas_type}")
    
    return co2_equivalent * gwp_factors[gas_type]

def biodiversity_index(species_counts: List[int]) -> float:
    """
    Обчислити індекс біорізноманіття Шеннона.
    
    Параметри:
        species_counts: Список кількості особин кожного виду
    
    Повертає:
        Індекс біорізноманіття Шеннона
    """
    if not species_counts:
        raise ValueError("Список кількості особин не може бути порожнім")
    
    if any(count < 0 for count in species_counts):
        raise ValueError("Кількість особин повинна бути невід'ємною")
    
    total_count = sum(species_counts)
    if total_count == 0:
        return 0
    
    # Обчислення індексу Шеннона
    shannon_index = 0
    for count in species_counts:
        if count > 0:
            p_i = count / total_count
            shannon_index -= p_i * math.log(p_i)
    
    return shannon_index

def ecosystem_services_value(biomass: float, area: float, service_type: str = "carbon_storage") -> float:
    """
    Обчислити економічну цінність екосистемних послуг.
    
    Параметри:
        biomass: Біомаса (т)
        area: Площа (га)
        service_type: Тип послуги ("carbon_storage", "water_purification", "pollination"), за замовчуванням "carbon_storage"
    
    Повертає:
        Економічна цінність (USD)
    """
    if biomass < 0:
        raise ValueError("Біомаса повинна бути невід'ємною")
    if area <= 0:
        raise ValueError("Площа повинна бути додатньою")
    
    # Цінність екосистемних послуг (USD/га/рік)
    service_values = {
        "carbon_storage": 150,  # Зберігання вуглецю
        "water_purification": 300,  # Очищення води
        "pollination": 500,  # Запилення
        "erosion_control": 200,  # Контроль ерозії
        "climate_regulation": 400  # Регулювання клімату
    }
    
    if service_type not in service_values:
        raise ValueError(f"Невідомий тип послуги: {service_type}")
    
    return area * service_values[service_type]

def population_growth(initial_population: float, growth_rate: float, time: float) -> float:
    """
    Обчислити зростання популяції за експоненційною моделлю.
    
    Параметри:
        initial_population: Початкова чисельність популяції
        growth_rate: Темп зростання (1/рік)
        time: Час (роки)
    
    Повертає:
        Чисельність популяції після заданого часу
    """
    if initial_population < 0:
        raise ValueError("Початкова чисельність популяції повинна бути невід'ємною")
    if time < 0:
        raise ValueError("Час повинен бути невід'ємним")
    
    return initial_population * math.exp(growth_rate * time)

def logistic_growth(initial_population: float, carrying_capacity: float, 
                   growth_rate: float, time: float) -> float:
    """
    Обчислити зростання популяції за логістичною моделлю.
    
    Параметри:
        initial_population: Початкова чисельність популяції
        carrying_capacity: Ємність середовища
        growth_rate: Темп зростання (1/рік)
        time: Час (роки)
    
    Повертає:
        Чисельність популяції після заданого часу
    """
    if initial_population < 0:
        raise ValueError("Початкова чисельність популяції повинна бути невід'ємною")
    if carrying_capacity <= 0:
        raise ValueError("Ємність середовища повинна бути додатньою")
    if initial_population > carrying_capacity:
        raise ValueError("Початкова чисельність не може перевищувати ємність середовища")
    if time < 0:
        raise ValueError("Час повинен бути невід'ємним")
    
    if initial_population == carrying_capacity:
        return carrying_capacity
    
    # Логістичне зростання
    numerator = carrying_capacity * initial_population
    denominator = initial_population + (carrying_capacity - initial_population) * math.exp(-growth_rate * time)
    
    if denominator == 0:
        return carrying_capacity
    
    return numerator / denominator

def species_area_relationship(area: float, constant: float = 10, exponent: float = 0.25) -> float:
    """
    Обчислити залежність кількості видів від площі (закон Саргента).
    
    Параметри:
        area: Площа (га)
        constant: Константа S = c * A^z, за замовчуванням 10
        exponent: Показник ступеня, за замовчуванням 0.25
    
    Повертає:
        Очікувана кількість видів
    """
    if area <= 0:
        raise ValueError("Площа повинна бути додатньою")
    if constant <= 0:
        raise ValueError("Константа повинна бути додатньою")
    if exponent <= 0:
        raise ValueError("Показник ступеня повинен бути додатнім")
    
    return constant * (area ** exponent)

def extinction_rate(species_count: int, time_period: float) -> float:
    """
    Обчислити темп вимирання видів.
    
    Параметри:
        species_count: Кількість вимерлих видів
        time_period: Період часу (роки)
    
    Повертає:
        Темп вимирання (види/рік)
    """
    if species_count < 0:
        raise ValueError("Кількість вимерлих видів повинна бути невід'ємною")
    if time_period <= 0:
        raise ValueError("Період часу повинен бути додатнім")
    
    return species_count / time_period

def ecological_footprint(population: float, consumption_per_capita: float, 
                       productivity_factor: float = 1.0) -> float:
    """
    Обчислити екологічний слід.
    
    Параметри:
        population: Чисельність населення
        consumption_per_capita: Споживання на душу населення (га)
        productivity_factor: Фактор продуктивності, за замовчуванням 1.0
    
    Повертає:
        Екологічний слід (га)
    """
    if population < 0:
        raise ValueError("Чисельність населення повинна бути невід'ємною")
    if consumption_per_capita < 0:
        raise ValueError("Споживання на душу населення повинне бути невід'ємним")
    if productivity_factor <= 0:
        raise ValueError("Фактор продуктивності повинен бути додатнім")
    
    return (population * consumption_per_capita) / productivity_factor

def water_footprint(production: float, water_intensity: float) -> float:
    """
    Обчислити водний слід виробництва.
    
    Параметри:
        production: Обсяг виробництва (одиниці)
        water_intensity: Водоінтенсивність (м³/одиницю)
    
    Повертає:
        Водний слід (м³)
    """
    if production < 0:
        raise ValueError("Обсяг виробництва повинен бути невід'ємним")
    if water_intensity < 0:
        raise ValueError("Водоінтенсивність повинна бути невід'ємною")
    
    return production * water_intensity

def air_quality_index(concentrations: Dict[str, float], 
                     standards: Dict[str, float]) -> float:
    """
    Обчислити індекс якості повітря.
    
    Параметри:
        concentrations: Словник концентрацій забруднювачів (мкг/м³)
        standards: Словник стандартів якості (мкг/м³)
    
    Повертає:
        Індекс якості повітря (0-500)
    """
    if not concentrations:
        raise ValueError("Словник концентрацій не може бути порожнім")
    if not standards:
        raise ValueError("Словник стандартів не може бути порожнім")
    
    # Перевірка наявності стандартів для всіх забруднювачів
    for pollutant in concentrations:
        if pollutant not in standards:
            raise ValueError(f"Відсутній стандарт для забруднювача: {pollutant}")
        if concentrations[pollutant] < 0:
            raise ValueError(f"Концентрація {pollutant} повинна бути невід'ємною")
        if standards[pollutant] <= 0:
            raise ValueError(f"Стандарт {pollutant} повинен бути додатнім")
    
    # Обчислення індексу для кожного забруднювача
    sub_indices = []
    for pollutant, concentration in concentrations.items():
        standard = standards[pollutant]
        sub_index = (concentration / standard) * 100
        sub_indices.append(sub_index)
    
    # Загальний індекс - максимальне значення
    return max(sub_indices) if sub_indices else 0

def renewable_energy_potential(area: float, irradiance: float, 
                              efficiency: float = 0.2) -> float:
    """
    Обчислити потенціал відновлюваної енергії.
    
    Параметри:
        area: Площа (м²)
        irradiance: Сонячна інсоляція (Вт/м²)
        efficiency: ККД перетворення, за замовчуванням 0.2
    
    Повертає:
        Потенціал енергії (Вт)
    """
    if area < 0:
        raise ValueError("Площа повинна бути невід'ємною")
    if irradiance < 0:
        raise ValueError("Сонячна інсоляція повинна бути невід'ємною")
    if efficiency < 0 or efficiency > 1:
        raise ValueError("ККД повинен бути в діапазоні [0, 1]")
    
    return area * irradiance * efficiency

def carbon_sequestration(biomass: float, sequestration_rate: float = 0.5) -> float:
    """
    Обчислити обсяг секвестрованого вуглецю.
    
    Параметри:
        biomass: Біомаса (т)
        sequestration_rate: Коефіцієнт секвестрації, за замовчуванням 0.5
    
    Повертає:
        Обсяг секвестрованого вуглецю (т)
    """
    if biomass < 0:
        raise ValueError("Біомаса повинна бути невід'ємною")
    if sequestration_rate < 0 or sequestration_rate > 1:
        raise ValueError("Коефіцієнт секвестрації повинен бути в діапазоні [0, 1]")
    
    return biomass * sequestration_rate

def pollution_dispersion(emission_rate: float, wind_speed: float, 
                        diffusion_coefficient: float, distance: float) -> float:
    """
    Обчислити дисперсію забруднень в атмосфері.
    
    Параметри:
        emission_rate: Швидкість емісії (г/с)
        wind_speed: Швидкість вітру (м/с)
        diffusion_coefficient: Коефіцієнт дифузії (м²/с)
        distance: Відстань від джерела (м)
    
    Повертає:
        Концентрація забруднень (г/м³)
    """
    if emission_rate < 0:
        raise ValueError("Швидкість емісії повинна бути невід'ємною")
    if wind_speed <= 0:
        raise ValueError("Швидкість вітру повинна бути додатньою")
    if diffusion_coefficient < 0:
        raise ValueError("Коефіцієнт дифузії повинен бути невід'ємним")
    if distance < 0:
        raise ValueError("Відстань повинна бути невід'ємною")
    
    if distance == 0:
        return float('inf') if emission_rate > 0 else 0
    
    # Модель дисперсії Гауса (спрощена)
    concentration = emission_rate / (wind_speed * math.sqrt(4 * math.pi * diffusion_coefficient * distance))
    return concentration

def ecosystem_stability(biodiversity_index: float, resilience_factor: float = 1.0) -> float:
    """
    Обчислити стабільність екосистеми.
    
    Параметри:
        biodiversity_index: Індекс біорізноманіття
        resilience_factor: Фактор стійкості, за замовчуванням 1.0
    
    Повертає:
        Стабільність екосистеми (0-1)
    """
    if biodiversity_index < 0:
        raise ValueError("Індекс біорізноманіття повинен бути невід'ємним")
    if resilience_factor <= 0:
        raise ValueError("Фактор стійкості повинен бути додатнім")
    
    # Стабільність пропорційна біорізноманіттю та стійкості
    stability = 1 - math.exp(-biodiversity_index * resilience_factor)
    return max(0, min(1, stability))  # Обмеження діапазону [0, 1]

def climate_sensitivity(temperature_change: float, forcing_change: float) -> float:
    """
    Обчислити кліматичну чутливість.
    
    Параметри:
        temperature_change: Зміна температури (°C)
        forcing_change: Зміна радіаційного форсингу (Вт/м²)
    
    Повертає:
        Кліматична чутливість (°C/(Вт/м²))
    """
    if forcing_change == 0:
        raise ValueError("Зміна радіаційного форсингу не може дорівнювати нулю")
    
    return temperature_change / forcing_change

def radiative_forcing(greenhouse_gas_concentration: float, 
                     reference_concentration: float, 
                     radiative_efficiency: float) -> float:
    """
    Обчислити радіаційний форсинг.
    
    Параметри:
        greenhouse_gas_concentration: Концентрація парникового газу
        reference_concentration: Референсна концентрація
        radiative_efficiency: Радіаційна ефективність (Вт/м²/(одиниця концентрації))
    
    Повертає:
        Радіаційний форсинг (Вт/м²)
    """
    if reference_concentration <= 0:
        raise ValueError("Референсна концентрація повинна бути додатньою")
    if radiative_efficiency < 0:
        raise ValueError("Радіаційна ефективність повинна бути невід'ємною")
    
    # Радіаційний форсинг
    concentration_ratio = greenhouse_gas_concentration / reference_concentration
    if concentration_ratio <= 0:
        return 0
    
    return radiative_efficiency * math.log(concentration_ratio)

def earth_energy_balance(solar_constant: float = SOLAR_CONSTANT, 
                        albedo: float = ALBEDO_EARTH) -> float:
    """
    Обчислити енергетичний баланс Землі.
    
    Параметри:
        solar_constant: Сонячна стала (Вт/м²), за замовчуванням SOLAR_CONSTANT
        albedo: Альбедо Землі, за замовчуванням ALBEDO_EARTH
    
    Повертає:
        Ефективна температура Землі (К)
    """
    if solar_constant < 0:
        raise ValueError("Сонячна стала повинна бути невід'ємною")
    if albedo < 0 or albedo > 1:
        raise ValueError("Альбедо повинно бути в діапазоні [0, 1]")
    
    # Ефективна температура Землі без парникового ефекту
    absorbed_radiation = solar_constant * (1 - albedo) / 4
    effective_temperature = (absorbed_radiation / STEFAN_BOLTZMANN_CONSTANT) ** 0.25
    return effective_temperature

def greenhouse_effect_temperature(effective_temperature: float, 
                                greenhouse_factor: float = GREENHOUSE_EFFECT_FACTOR) -> float:
    """
    Обчислити температуру з урахуванням парникового ефекту.
    
    Параметри:
        effective_temperature: Ефективна температура (К)
        greenhouse_factor: Фактор парникового ефекту, за замовчуванням GREENHOUSE_EFFECT_FACTOR
    
    Повертає:
        Температура з парниковим ефектом (К)
    """
    if effective_temperature < 0:
        raise ValueError("Ефективна температура повинна бути невід'ємною")
    if greenhouse_factor < 0:
        raise ValueError("Фактор парникового ефекту повинен бути невід'ємним")
    
    return effective_temperature * (1 + greenhouse_factor)

def carbon_cycle_atmosphere_ocean(co2_atmosphere: float, co2_ocean: float, 
                                 exchange_rate: float) -> Tuple[float, float]:
    """
    Обчислити обмін вуглекислого газу між атмосферою та океаном.
    
    Параметри:
        co2_atmosphere: Концентрація CO₂ в атмосфері (ppm)
        co2_ocean: Концентрація CO₂ в океані (ppm)
        exchange_rate: Швидкість обміну (1/рік)
    
    Повертає:
        Кортеж (нова концентрація в атмосфері, нова концентрація в океані)
    """
    if co2_atmosphere < 0:
        raise ValueError("Концентрація CO₂ в атмосфері повинна бути невід'ємною")
    if co2_ocean < 0:
        raise ValueError("Концентрація CO₂ в океані повинна бути невід'ємною")
    if exchange_rate < 0:
        raise ValueError("Швидкість обміну повинна бути невід'ємною")
    
    # Різниця концентрацій
    concentration_difference = co2_atmosphere - co2_ocean
    
    # Обмін між резервуарами
    exchange_amount = concentration_difference * exchange_rate
    
    # Нові концентрації
    new_atmosphere = co2_atmosphere - exchange_amount
    new_ocean = co2_ocean + exchange_amount
    
    return (new_atmosphere, new_ocean)

def deforestation_impact(forest_area_lost: float, carbon_density: float) -> float:
    """
    Обчислити вплив вирубки лісів на вуглецевий баланс.
    
    Параметри:
        forest_area_lost: Втрачена площа лісів (га)
        carbon_density: Щільність вуглецю в лісах (т/га)
    
    Повертає:
        Викиди вуглецю (т CO₂)
    """
    if forest_area_lost < 0:
        raise ValueError("Втрачена площа лісів повинна бути невід'ємною")
    if carbon_density < 0:
        raise ValueError("Щільність вуглецю повинна бути невід'ємною")
    
    # Приблизно 3.67 т CO₂ на 1 т вуглецю
    return forest_area_lost * carbon_density * 3.67

def renewable_energy_penetration(renewable_generation: float, 
                               total_generation: float) -> float:
    """
    Обчислити проникнення відновлюваної енергії.
    
    Параметри:
        renewable_generation: Виробництво відновлюваної енергії (МВт·год)
        total_generation: Загальне виробництво енергії (МВт·год)
    
    Повертає:
        Проникнення відновлюваної енергії (%)
    """
    if renewable_generation < 0:
        raise ValueError("Виробництво відновлюваної енергії повинне бути невід'ємним")
    if total_generation <= 0:
        raise ValueError("Загальне виробництво енергії повинне бути додатнім")
    if renewable_generation > total_generation:
        raise ValueError("Виробництво відновлюваної енергії не може перевищувати загальне виробництво")
    
    return (renewable_generation / total_generation) * 100

def water_quality_index(dissolved_oxygen: float, ph: float, 
                       turbidity: float, nutrients: float) -> float:
    """
    Обчислити індекс якості води.
    
    Параметри:
        dissolved_oxygen: Розчинений кисень (мг/л)
        ph: Рівень pH
        turbidity: Каламутьність (NTU)
        nutrients: Рівень поживних речовин (мг/л)
    
    Повертає:
        Індекс якості води (0-100)
    """
    if dissolved_oxygen < 0:
        raise ValueError("Розчинений кисень повинен бути невід'ємним")
    if ph < 0 or ph > 14:
        raise ValueError("Рівень pH повинен бути в діапазоні [0, 14]")
    if turbidity < 0:
        raise ValueError("Каламутьність повинна бути невід'ємною")
    if nutrients < 0:
        raise ValueError("Рівень поживних речовин повинен бути невід'ємним")
    
    # Нормалізація параметрів до шкали 0-100
    do_score = min(100, dissolved_oxygen * 10)  # Приблизно
    ph_score = 100 - abs(ph - 7) * 10  # Оптимальний pH = 7
    turbidity_score = max(0, 100 - turbidity)  # Менша каламутьність - краще
    nutrients_score = max(0, 100 - nutrients * 2)  # Менше поживних речовин - краще
    
    # Середнє значення
    return (do_score + ph_score + turbidity_score + nutrients_score) / 4

def ecosystem_services_valuation(carbon_storage: float, water_purification: float, 
                               biodiversity: float, recreation: float) -> float:
    """
    Обчислити загальну цінність екосистемних послуг.
    
    Параметри:
        carbon_storage: Цінність зберігання вуглецю (USD)
        water_purification: Цінність очищення води (USD)
        biodiversity: Цінність біорізноманіття (USD)
        recreation: Цінність рекреаційного використання (USD)
    
    Повертає:
        Загальна цінність екосистемних послуг (USD)
    """
    if carbon_storage < 0:
        raise ValueError("Цінність зберігання вуглецю повинна бути невід'ємною")
    if water_purification < 0:
        raise ValueError("Цінність очищення води повинна бути невід'ємною")
    if biodiversity < 0:
        raise ValueError("Цінність біорізноманіття повинна бути невід'ємною")
    if recreation < 0:
        raise ValueError("Цінність рекреаційного використання повинна бути невід'ємною")
    
    return carbon_storage + water_purification + biodiversity + recreation

def climate_resilience(biodiversity: float, ecosystem_stability: float, 
                      adaptive_capacity: float) -> float:
    """
    Обчислити кліматичну стійкість екосистеми.
    
    Параметри:
        biodiversity: Рівень біорізноманіття (0-1)
        ecosystem_stability: Стабільність екосистеми (0-1)
        adaptive_capacity: Адаптаційна здатність (0-1)
    
    Повертає:
        Кліматична стійкість (0-1)
    """
    if biodiversity < 0 or biodiversity > 1:
        raise ValueError("Рівень біорізноманіття повинен бути в діапазоні [0, 1]")
    if ecosystem_stability < 0 or ecosystem_stability > 1:
        raise ValueError("Стабільність екосистеми повинна бути в діапазоні [0, 1]")
    if adaptive_capacity < 0 or adaptive_capacity > 1:
        raise ValueError("Адаптаційна здатність повинна бути в діапазоні [0, 1]")
    
    # Комбінований індекс стійкості
    return (biodiversity * 0.4 + ecosystem_stability * 0.4 + adaptive_capacity * 0.2)

def pollution_health_impact(pollution_level: float, population_exposed: float, 
                          health_risk_coefficient: float = 0.001) -> float:
    """
    Обчислити вплив забруднення на здоров'я.
    
    Параметри:
        pollution_level: Рівень забруднення (індекс)
        population_exposed: Кількість населення, що зазнало впливу
        health_risk_coefficient: Коефіцієнт ризику здоров'я, за замовчуванням 0.001
    
    Повертає:
        Очікувана кількість випадків захворювань
    """
    if pollution_level < 0:
        raise ValueError("Рівень забруднення повинен бути невід'ємним")
    if population_exposed < 0:
        raise ValueError("Кількість населення повинна бути невід'ємною")
    if health_risk_coefficient < 0:
        raise ValueError("Коефіцієнт ризику здоров'я повинен бути невід'ємним")
    
    return pollution_level * population_exposed * health_risk_coefficient

def sustainable_yield(resource_stock: float, regeneration_rate: float, 
                     safety_factor: float = 0.1) -> float:
    """
    Обчислити сталу врожайність ресурсу.
    
    Параметри:
        resource_stock: Запас ресурсу
        regeneration_rate: Швидкість відновлення (1/рік)
        safety_factor: Фактор безпеки, за замовчуванням 0.1
    
    Повертає:
        Стала врожайність
    """
    if resource_stock < 0:
        raise ValueError("Запас ресурсу повинен бути невід'ємним")
    if regeneration_rate < 0:
        raise ValueError("Швидкість відновлення повинна бути невід'ємною")
    if safety_factor < 0 or safety_factor > 1:
        raise ValueError("Фактор безпеки повинен бути в діапазоні [0, 1]")
    
    maximum_sustainable_yield = resource_stock * regeneration_rate
    return maximum_sustainable_yield * (1 - safety_factor)

def ecological_connectivity(habitat_patches: int, connectivity_matrix: List[List[float]]) -> float:
    """
    Обчислити екологічну зв'язність ландшафту.
    
    Параметри:
        habitat_patches: Кількість ділянок середовища
        connectivity_matrix: Матриця зв'язності між ділянками
    
    Повертає:
        Індекс екологічної зв'язності (0-1)
    """
    if habitat_patches <= 0:
        raise ValueError("Кількість ділянок середовища повинна бути додатньою")
    
    if len(connectivity_matrix) != habitat_patches:
        raise ValueError("Розмір матриці зв'язності повинен відповідати кількості ділянок")
    
    for row in connectivity_matrix:
        if len(row) != habitat_patches:
            raise ValueError("Матриця зв'язності повинна бути квадратною")
        if any(value < 0 or value > 1 for value in row):
            raise ValueError("Значення в матриці зв'язності повинні бути в діапазоні [0, 1]")
    
    # Обчислення індексу зв'язності
    total_connections = sum(sum(row) for row in connectivity_matrix)
    maximum_connections = habitat_patches * (habitat_patches - 1)
    
    if maximum_connections == 0:
        return 1.0 if habitat_patches == 1 else 0.0
    
    return total_connections / maximum_connections

def carbon_neutral_balance(emissions: float, sequestration: float, 
                          offset_purchases: float = 0) -> float:
    """
    Обчислити вуглецевий баланс для досягнення нейтральності.
    
    Параметри:
        emissions: Викиди вуглекислого газу (т CO₂)
        sequestration: Секвестрація вуглекислого газу (т CO₂)
        offset_purchases: Покупка компенсацій (т CO₂), за замовчуванням 0
    
    Повертає:
        Вуглецевий баланс (т CO₂)
    """
    if emissions < 0:
        raise ValueError("Викиди повинні бути невід'ємними")
    if sequestration < 0:
        raise ValueError("Секвестрація повинна бути невід'ємною")
    if offset_purchases < 0:
        raise ValueError("Покупка компенсацій повинна бути невід'ємною")
    
    return emissions - sequestration - offset_purchases

def environmental_degradation_index(biodiversity_loss: float, soil_degradation: float, 
                                 water_pollution: float, air_pollution: float) -> float:
    """
    Обчислити індекс екологічного зневадження.
    
    Параметри:
        biodiversity_loss: Втрата біорізноманіття (0-1)
        soil_degradation: Деградація ґрунтів (0-1)
        water_pollution: Забруднення води (0-1)
        air_pollution: Забруднення повітря (0-1)
    
    Повертає:
        Індекс екологічного зневадження (0-1)
    """
    if not all(0 <= x <= 1 for x in [biodiversity_loss, soil_degradation, water_pollution, air_pollution]):
        raise ValueError("Всі параметри повинні бути в діапазоні [0, 1]")
    
    # Зважений індекс (можна налаштувати ваги)
    weights = [0.3, 0.25, 0.25, 0.2]  # Ваги для кожного фактора
    factors = [biodiversity_loss, soil_degradation, water_pollution, air_pollution]
    
    return sum(w * f for w, f in zip(weights, factors))

def renewable_energy_transition(current_renewable: float, target_renewable: float, 
                              transition_time: float) -> float:
    """
    Обчислити швидкість переходу на відновлювану енергію.
    
    Параметри:
        current_renewable: Поточна частка відновлюваної енергії (%)
        target_renewable: Цільова частка відновлюваної енергії (%)
        transition_time: Час переходу (роки)
    
    Повертає:
        Швидкість переходу (%/рік)
    """
    if current_renewable < 0 or current_renewable > 100:
        raise ValueError("Поточна частка повинна бути в діапазоні [0, 100]")
    if target_renewable < 0 or target_renewable > 100:
        raise ValueError("Цільова частка повинна бути в діапазоні [0, 100]")
    if transition_time <= 0:
        raise ValueError("Час переходу повинен бути додатнім")
    
    return (target_renewable - current_renewable) / transition_time

def ecosystem_recovery_time(disturbance_intensity: float, recovery_rate: float) -> float:
    """
    Обчислити час відновлення екосистеми.
    
    Параметри:
        disturbance_intensity: Інтенсивність порушення (0-1)
        recovery_rate: Швидкість відновлення (1/рік)
    
    Повертає:
        Час відновлення (роки)
    """
    if disturbance_intensity < 0 or disturbance_intensity > 1:
        raise ValueError("Інтенсивність порушення повинна бути в діапазоні [0, 1]")
    if recovery_rate <= 0:
        raise ValueError("Швидкість відновлення повинна бути додатньою")
    
    # Спрощена модель: час відновлення обернено пропорційний швидкості відновлення
    return disturbance_intensity / recovery_rate

def pollution_concentration_decay(initial_concentration: float, decay_rate: float, 
                                time: float) -> float:
    """
    Обчислити зменшення концентрації забруднювача з часом.
    
    Параметри:
        initial_concentration: Початкова концентрація
        decay_rate: Швидкість розпаду (1/рік)
        time: Час (роки)
    
    Повертає:
        Концентрація після заданого часу
    """
    if initial_concentration < 0:
        raise ValueError("Початкова концентрація повинна бути невід'ємною")
    if decay_rate < 0:
        raise ValueError("Швидкість розпаду повинна бути невід'ємною")
    if time < 0:
        raise ValueError("Час повинен бути невід'ємним")
    
    return initial_concentration * math.exp(-decay_rate * time)

def habitat_suitability(species_requirements: Dict[str, float], 
                       environmental_conditions: Dict[str, float]) -> float:
    """
    Обчислити придатність середовища для виду.
    
    Параметри:
        species_requirements: Вимоги виду до середовища
        environmental_conditions: Фактичні умови середовища
    
    Повертає:
        Індекс придатності середовища (0-1)
    """
    if not species_requirements:
        raise ValueError("Словник вимог виду не може бути порожнім")
    if not environmental_conditions:
        raise ValueError("Словник умов середовища не може бути порожнім")
    
    # Перевірка наявності всіх параметрів
    for param in species_requirements:
        if param not in environmental_conditions:
            raise ValueError(f"Відсутні умови для параметра: {param}")
    
    # Обчислення індексу придатності
    suitability_scores = []
    for param, required_value in species_requirements.items():
        actual_value = environmental_conditions[param]
        # Припущення: ідеальна придатність при відповідності вимогам
        # Можна вдосконалити з урахуванням допустимих відхилень
        if required_value > 0:
            score = min(1.0, actual_value / required_value) if actual_value <= required_value else max(0.0, 2 - actual_value / required_value)
        else:
            score = 1.0 if actual_value == required_value else 0.0
        suitability_scores.append(score)
    
    # Середнє значення
    return sum(suitability_scores) / len(suitability_scores) if suitability_scores else 0

def environmental_impact_assessment(emissions: Dict[str, float], 
                                  impact_factors: Dict[str, float]) -> float:
    """
    Обчислити загальний екологічний вплив.
    
    Параметри:
        emissions: Словник викидів різних забруднювачів
        impact_factors: Словник факторів впливу для кожного забруднювача
    
    Повертає:
        Загальний екологічний вплив
    """
    if not emissions:
        raise ValueError("Словник викидів не може бути порожнім")
    if not impact_factors:
        raise ValueError("Словник факторів впливу не може бути порожнім")
    
    # Перевірка наявності факторів впливу для всіх забруднювачів
    for pollutant in emissions:
        if pollutant not in impact_factors:
            raise ValueError(f"Відсутній фактор впливу для забруднювача: {pollutant}")
        if emissions[pollutant] < 0:
            raise ValueError(f"Викиди {pollutant} повинні бути невід'ємними")
        if impact_factors[pollutant] < 0:
            raise ValueError(f"Фактор впливу {pollutant} повинен бути невід'ємним")
    
    # Обчислення загального впливу
    total_impact = sum(emissions[pollutant] * impact_factors[pollutant] 
                      for pollutant in emissions)
    return total_impact

def climate_adaptation_index(vulnerability: float, exposure: float, 
                           adaptive_capacity: float) -> float:
    """
    Обчислити індекс кліматичної адаптації.
    
    Параметри:
        vulnerability: Вразливість (0-1)
        exposure: Експозиція (0-1)
        adaptive_capacity: Адаптаційна здатність (0-1)
    
    Повертає:
        Індекс кліматичної адаптації (0-1)
    """
    if not all(0 <= x <= 1 for x in [vulnerability, exposure, adaptive_capacity]):
        raise ValueError("Всі параметри повинні бути в діапазоні [0, 1]")
    
    # Індекс адаптації = адаптаційна здатність / (вразливість × експозиція)
    risk = vulnerability * exposure
    if risk == 0:
        return 1.0 if adaptive_capacity > 0 else 0.0
    
    adaptation_index = adaptive_capacity / risk
    return min(1.0, adaptation_index)  # Обмеження зверху

def ecosystem_services_supply(demand: float, supply: float) -> float:
    """
    Обчислити баланс екосистемних послуг.
    
    Параметри:
        demand: Попит на екосистемні послуги
        supply: Пропозиція екосистемних послуг
    
    Повертає:
        Баланс екосистемних послуг (додатній - надлишок, від'ємний - дефіцит)
    """
    if demand < 0:
        raise ValueError("Попит повинен бути невід'ємним")
    if supply < 0:
        raise ValueError("Пропозиція повинна бути невід'ємною")
    
    return supply - demand

def environmental_performance_index(air_quality: float, water_quality: float, 
                                  biodiversity: float, climate_action: float,
                                  weights: Optional[List[float]] = None) -> float:
    """
    Обчислити індекс екологічної ефективності.
    
    Параметри:
        air_quality: Якість повітря (0-100)
        water_quality: Якість води (0-100)
        biodiversity: Рівень біорізноманіття (0-100)
        climate_action: Дії з клімату (0-100)
        weights: Ваги для кожного показника, за замовчуванням [0.25, 0.25, 0.25, 0.25]
    
    Повертає:
        Індекс екологічної ефективності (0-100)
    """
    indicators = [air_quality, water_quality, biodiversity, climate_action]
    if not all(0 <= x <= 100 for x in indicators):
        raise ValueError("Всі показники повинні бути в діапазоні [0, 100]")
    
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    elif len(weights) != 4:
        raise ValueError("Кількість ваг повинна дорівнювати 4")
    elif abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError("Сума ваг повинна дорівнювати 1.0")
    elif any(w < 0 for w in weights):
        raise ValueError("Всі ваги повинні бути невід'ємними")
    
    return sum(indicator * weight for indicator, weight in zip(indicators, weights))

def pollution_transport_rate(wind_speed: float, atmospheric_stability: float, 
                           particle_size: float) -> float:
    """
    Обчислити швидкість транспортування забруднень.
    
    Параметри:
        wind_speed: Швидкість вітру (м/с)
        atmospheric_stability: Стабільність атмосфери (0-1)
        particle_size: Розмір частинок (мкм)
    
    Повертає:
        Швидкість транспортування (м/с)
    """
    if wind_speed < 0:
        raise ValueError("Швидкість вітру повинна бути невід'ємною")
    if atmospheric_stability < 0 or atmospheric_stability > 1:
        raise ValueError("Стабільність атмосфери повинна бути в діапазоні [0, 1]")
    if particle_size <= 0:
        raise ValueError("Розмір частинок повинен бути додатнім")
    
    # Спрощена модель: транспортна швидкість пропорційна швидкості вітру
    # та зворотно пропорційна стабільності та розміру частинок
    stability_factor = 1 - atmospheric_stability  # Менша стабільність - більше транспорт
    size_factor = 1 / particle_size  # Менші частинки транспортуються далі
    
    return wind_speed * stability_factor * size_factor

def ecosystem_connectivity_index(patch_areas: List[float], 
                               distances: List[List[float]]) -> float:
    """
    Обчислити індекс зв'язності екосистем.
    
    Параметри:
        patch_areas: Список площ ділянок середовища
        distances: Матриця відстаней між ділянками
    
    Повертає:
        Індекс зв'язності екосистем (0-1)
    """
    if not patch_areas:
        raise ValueError("Список площ ділянок не може бути порожнім")
    if any(area < 0 for area in patch_areas):
        raise ValueError("Всі площі ділянок повинні бути невід'ємними")
    
    num_patches = len(patch_areas)
    if len(distances) != num_patches:
        raise ValueError("Розмір матриці відстаней повинен відповідати кількості ділянок")
    
    for row in distances:
        if len(row) != num_patches:
            raise ValueError("Матриця відстаней повинна бути квадратною")
        if any(dist < 0 for dist in row):
            raise ValueError("Всі відстані повинні бути невід'ємними")
    
    if num_patches == 1:
        return 1.0
    
    # Обчислення індексу зв'язності
    total_connectivity = 0
    max_possible_connectivity = 0
    
    for i in range(num_patches):
        for j in range(i + 1, num_patches):
            # Зв'язність між ділянками i та j
            # Пропорційна площам та обернено пропорційна відстані
            if distances[i][j] > 0:
                connectivity = (patch_areas[i] * patch_areas[j]) / distances[i][j]
            else:
                connectivity = patch_areas[i] * patch_areas[j]  # Максимальна зв'язність при нульовій відстані
            
            total_connectivity += connectivity
            max_possible_connectivity += patch_areas[i] * patch_areas[j]
    
    if max_possible_connectivity == 0:
        return 0.0
    
    return total_connectivity / max_possible_connectivity

def carbon_credit_value(carbon_sequestered: float, market_price: float = 10.0) -> float:
    """
    Обчислити вартість вуглецевих кредитів.
    
    Параметри:
        carbon_sequestered: Обсяг секвестрованого вуглецю (т CO₂)
        market_price: Ринкова ціна за тонну CO₂ (USD), за замовчуванням 10.0
    
    Повертає:
        Вартість вуглецевих кредитів (USD)
    """
    if carbon_sequestered < 0:
        raise ValueError("Обсяг секвестрованого вуглецю повинен бути невід'ємним")
    if market_price < 0:
        raise ValueError("Ринкова ціна повинна бути невід'ємною")
    
    return carbon_sequestered * market_price

def environmental_risk_assessment(hazard_probability: float, 
                                consequence_severity: float) -> float:
    """
    Обчислити екологічний ризик.
    
    Параметри:
        hazard_probability: Ймовірність небезпеки (0-1)
        consequence_severity: Серйозність наслідків (0-1)
    
    Повертає:
        Екологічний ризик (0-1)
    """
    if hazard_probability < 0 or hazard_probability > 1:
        raise ValueError("Ймовірність небезпеки повинна бути в діапазоні [0, 1]")
    if consequence_severity < 0 or consequence_severity > 1:
        raise ValueError("Серйозність наслідків повинна бути в діапазоні [0, 1]")
    
    return hazard_probability * consequence_severity

def ecosystem_restoration_potential(degraded_area: float, 
                                  restoration_success_rate: float) -> float:
    """
    Обчислити потенціал відновлення екосистеми.
    
    Параметри:
        degraded_area: Деградована площа (га)
        restoration_success_rate: Швидкість успішного відновлення (0-1)
    
    Повертає:
        Потенціал відновлення (га)
    """
    if degraded_area < 0:
        raise ValueError("Деградована площа повинна бути невід'ємною")
    if restoration_success_rate < 0 or restoration_success_rate > 1:
        raise ValueError("Швидкість успішного відновлення повинна бути в діапазоні [0, 1]")
    
    return degraded_area * restoration_success_rate

def pollution_source_apportionment(sources: Dict[str, float], 
                                 total_pollution: float) -> Dict[str, float]:
    """
    Обчислити частку кожного джерела забруднення.
    
    Параметри:
        sources: Словник джерел забруднення та їх внеску
        total_pollution: Загальний рівень забруднення
    
    Повертає:
        Словник часток кожного джерела (0-1)
    """
    if not sources:
        raise ValueError("Словник джерел не може бути порожнім")
    if any(value < 0 for value in sources.values()):
        raise ValueError("Всі значення джерел повинні бути невід'ємними")
    if total_pollution <= 0:
        raise ValueError("Загальний рівень забруднення повинен бути додатнім")
    
    return {source: value / total_pollution for source, value in sources.items()}

def environmental_justice_index(population_exposure: List[float], 
                              population_vulnerability: List[float]) -> float:
    """
    Обчислити індекс екологічної справедливості.
    
    Параметри:
        population_exposure: Список рівнів експозиції для різних груп населення
        population_vulnerability: Список рівнів вразливості для різних груп населення
    
    Повертає:
        Індекс екологічної справедливості (0-1)
    """
    if not population_exposure:
        raise ValueError("Список експозицій не може бути порожнім")
    if not population_vulnerability:
        raise ValueError("Список вразливостей не може бути порожнім")
    if len(population_exposure) != len(population_vulnerability):
        raise ValueError("Списки експозицій та вразливостей повинні мати однакову довжину")
    if any(exp < 0 for exp in population_exposure):
        raise ValueError("Всі рівні експозиції повинні бути невід'ємними")
    if any(vuln < 0 or vuln > 1 for vuln in population_vulnerability):
        raise ValueError("Всі рівні вразливості повинні бути в діапазоні [0, 1]")
    
    if not population_exposure:
        return 0.0
    
    # Індекс справедливості - обернений до загального навантаження на вразливі групи
    weighted_exposure = sum(exp * vuln for exp, vuln in zip(population_exposure, population_vulnerability))
    total_exposure = sum(population_exposure)
    
    if total_exposure == 0:
        return 0.0
    
    # Нормалізація до діапазону [0, 1]
    # 0 - повна справедливість, 1 - максимальна несправедливість
    injustice_index = weighted_exposure / total_exposure
    return 1 - injustice_index

def climate_change_impact(agricultural_yield_change: float, 
                        water_availability_change: float,
                        biodiversity_loss: float) -> float:
    """
    Обчислити загальний вплив зміни клімату.
    
    Параметри:
        agricultural_yield_change: Зміна врожайності (%)
        water_availability_change: Зміна доступності води (%)
        biodiversity_loss: Втрата біорізноманіття (0-1)
    
    Повертає:
        Загальний вплив зміни клімату (0-1)
    """
    if agricultural_yield_change < -100 or agricultural_yield_change > 100:
        raise ValueError("Зміна врожайності повинна бути в діапазоні [-100, 100]")
    if water_availability_change < -100 or water_availability_change > 100:
        raise ValueError("Зміна доступності води повинна бути в діапазоні [-100, 100]")
    if biodiversity_loss < 0 or biodiversity_loss > 1:
        raise ValueError("Втрата біорізноманіття повинна бути в діапазоні [0, 1]")
    
    # Нормалізація показників до діапазону [0, 1]
    # 0 - позитивний вплив, 1 - негативний вплив
    normalized_agriculture = max(0, agricultural_yield_change / 100)
    normalized_water = max(0, water_availability_change / 100)
    
    # Зважений індекс впливу
    weights = [0.4, 0.4, 0.2]  # Ваги для сільського господарства, води та біорізноманіття
    impacts = [normalized_agriculture, normalized_water, biodiversity_loss]
    
    return sum(weight * impact for weight, impact in zip(weights, impacts))

def renewable_energy_intermittency(wind_speed_data: List[float], 
                                 solar_irradiance_data: List[float]) -> float:
    """
    Обчислити міру переривчастості відновлюваної енергії.
    
    Параметри:
        wind_speed_data: Дані про швидкість вітру
        solar_irradiance_data: Дані про сонячну інсоляцію
    
    Повертає:
        Міра переривчастості (0-1)
    """
    if not wind_speed_data or not solar_irradiance_data:
        raise ValueError("Списки даних не можуть бути порожніми")
    if len(wind_speed_data) != len(solar_irradiance_data):
        raise ValueError("Списки даних повинні мати однакову довжину")
    if any(ws < 0 for ws in wind_speed_data):
        raise ValueError("Швидкість вітру повинна бути невід'ємною")
    if any(si < 0 for si in solar_irradiance_data):
        raise ValueError("Сонячна інсоляція повинна бути невід'ємною")
    
    # Обчислення варіації для кожного джерела
    wind_variance = 0
    solar_variance = 0
    
    if len(wind_speed_data) > 1:
        wind_mean = sum(wind_speed_data) / len(wind_speed_data)
        wind_variance = sum((ws - wind_mean) ** 2 for ws in wind_speed_data) / (len(wind_speed_data) - 1)
    
    if len(solar_irradiance_data) > 1:
        solar_mean = sum(solar_irradiance_data) / len(solar_irradiance_data)
        solar_variance = sum((si - solar_mean) ** 2 for si in solar_irradiance_data) / (len(solar_irradiance_data) - 1)
    
    # Міра переривчастості - нормалізована варіація
    max_possible_variance = max(
        max(wind_speed_data) ** 2 if wind_speed_data else 0,
        max(solar_irradiance_data) ** 2 if solar_irradiance_data else 0
    )
    
    if max_possible_variance == 0:
        return 0.0
    
    combined_variance = (wind_variance + solar_variance) / 2
    return min(1.0, combined_variance / max_possible_variance)

def ecosystem_services_substitution(natural_service_value: float, 
                                  artificial_service_cost: float) -> float:
    """
    Обчислити коефіцієнт заміщення екосистемних послуг.
    
    Параметри:
        natural_service_value: Вартість природної екосистемної послуги (USD)
        artificial_service_cost: Вартість штучної заміни (USD)
    
    Повертає:
        Коефіцієнт заміщення (0-1)
    """
    if natural_service_value < 0:
        raise ValueError("Вартість природної послуги повинна бути невід'ємною")
    if artificial_service_cost < 0:
        raise ValueError("Вартість штучної заміни повинна бути невід'ємною")
    
    if artificial_service_cost == 0:
        return 1.0 if natural_service_value > 0 else 0.0
    
    # Коефіцієнт заміщення - наскільки штучна заміна ефективна порівняно з природною
    substitution_ratio = natural_service_value / artificial_service_cost
    return min(1.0, substitution_ratio)  # Обмеження зверху

def environmental_policy_effectiveness(policy_score: float, 
                                     implementation_rate: float,
                                     time_since_implementation: float) -> float:
    """
    Обчислити ефективність екологічної політики.
    
    Параметри:
        policy_score: Оцінка політики (0-100)
        implementation_rate: Швидкість реалізації (0-1)
        time_since_implementation: Час з моменту реалізації (роки)
    
    Повертає:
        Ефективність політики (0-100)
    """
    if policy_score < 0 or policy_score > 100:
        raise ValueError("Оцін