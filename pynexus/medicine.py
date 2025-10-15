"""
Модуль для обчислювальної медицини
Computational Medicine Module
"""
from typing import Union, Tuple, List, Optional, Dict, Any
import math

# Медичні константи
# Medical constants
BLOOD_DENSITY = 1060  # Густина крові (кг/м³)
BLOOD_VISCOSITY = 0.0035  # В'язкість крові (Па·с)
HEART_RATE_RESTING = 70  # Спокійна частота серця (ударів/хв)
STROKE_VOLUME_RESTING = 70e-6  # Спокійний ударний об'єм (м³)
AORTA_RADIUS = 0.01  # Радіус аорти (м)
CAPILLARY_RADIUS = 5e-6  # Радіус капіляра (м)
CAPILLARY_LENGTH = 1e-3  # Довжина капіляра (м)
OXYGEN_CONSUMPTION_RESTING = 250e-6  # Споживання кисню в спокої (м³/с)
ALVEOLAR_SURFACE_AREA = 70  # Площа альвеол (м²)
HEMOGLOBIN_CONCENTRATION = 150  # Концентрація гемоглобіну (г/л)
OXYGEN_CAPACITY_HEMOGLOBIN = 1.39  # Киснева ємність гемоглобіну (мл O₂/г Hb)
AVOGADRO_CONSTANT = 6.02214076e23  # Число Авогадро (1/моль)
GAS_CONSTANT = 8.31446261815324  # Універсальна газова стала (Дж/(моль·К))
OXYGEN_MOLAR_MASS = 0.032  # Молярна маса кисню (кг/моль)
CARBON_DIOXIDE_MOLAR_MASS = 0.044  # Молярна маса вуглекислого газу (кг/моль)
WATER_MOLAR_MASS = 0.018  # Молярна маса води (кг/моль)

def cardiac_output(heart_rate: float, stroke_volume: float) -> float:
    """
    Обчислити серцевий викид.
    
    Параметри:
        heart_rate: Частота серця (ударів/хв)
        stroke_volume: Ударний об'єм (м³)
    
    Повертає:
        Серцевий викид (м³/хв)
    """
    if heart_rate < 0:
        raise ValueError("Частота серця повинна бути невід'ємною")
    if stroke_volume < 0:
        raise ValueError("Ударний об'єм повинен бути невід'ємним")
    
    return heart_rate * stroke_volume

def mean_arterial_pressure(systolic_pressure: float, diastolic_pressure: float) -> float:
    """
    Обчислити середній артеріальний тиск.
    
    Параметри:
        systolic_pressure: Систолічний тиск (ммHg)
        diastolic_pressure: Діастолічний тиск (ммHg)
    
    Повертає:
        Середній артеріальний тиск (ммHg)
    """
    if systolic_pressure < 0:
        raise ValueError("Систолічний тиск повинен бути невід'ємним")
    if diastolic_pressure < 0:
        raise ValueError("Діастолічний тиск повинен бути невід'ємним")
    if diastolic_pressure > systolic_pressure:
        raise ValueError("Діастолічний тиск не може бути більшим за систолічний")
    
    return diastolic_pressure + (systolic_pressure - diastolic_pressure) / 3

def pulse_pressure(systolic_pressure: float, diastolic_pressure: float) -> float:
    """
    Обчислити пульсовий тиск.
    
    Параметри:
        systolic_pressure: Систолічний тиск (ммHg)
        diastolic_pressure: Діастолічний тиск (ммHg)
    
    Повертає:
        Пульсовий тиск (ммHg)
    """
    if systolic_pressure < 0:
        raise ValueError("Систолічний тиск повинен бути невід'ємним")
    if diastolic_pressure < 0:
        raise ValueError("Діастолічний тиск повинен бути невід'ємним")
    if diastolic_pressure > systolic_pressure:
        raise ValueError("Діастолічний тиск не може бути більшим за систолічний")
    
    return systolic_pressure - diastolic_pressure

def systemic_vascular_resistance(mean_arterial_pressure: float, central_venous_pressure: float, 
                               cardiac_output: float) -> float:
    """
    Обчислити загальний периферичний опір судин.
    
    Параметри:
        mean_arterial_pressure: Середній артеріальний тиск (ммHg)
        central_venous_pressure: Центральний венозний тиск (ммHg)
        cardiac_output: Серцевий викид (л/хв)
    
    Повертає:
        Загальний периферичний опір (дин·с/см⁵)
    """
    if mean_arterial_pressure < 0:
        raise ValueError("Середній артеріальний тиск повинен бути невід'ємним")
    if central_venous_pressure < 0:
        raise ValueError("Центральний венозний тиск повинен бути невід'ємним")
    if cardiac_output <= 0:
        raise ValueError("Серцевий викид повинен бути додатнім")
    
    pressure_difference = mean_arterial_pressure - central_venous_pressure
    # Конвертація з ммHg в дин/см²: 1 ммHg = 1333.22 дин/см²
    # Конвертація з л/хв в см³/с: 1 л/хв = 16.6667 см³/с
    return (pressure_difference * 1333.22) / (cardiac_output * 16.6667)

def oxygen_delivery(cardiac_output: float, cao2: float, cvo2: float) -> float:
    """
    Обчислити доставку кисню.
    
    Параметри:
        cardiac_output: Серцевий викид (л/хв)
        cao2: Артеріальна концентрація кисню (мл O₂/л крові)
        cvo2: Венозна концентрація кисню (мл O₂/л крові)
    
    Повертає:
        Доставка кисню (мл O₂/хв)
    """
    if cardiac_output < 0:
        raise ValueError("Серцевий викид повинен бути невід'ємним")
    if cao2 < 0:
        raise ValueError("Артеріальна концентрація кисню повинна бути невід'ємною")
    if cvo2 < 0:
        raise ValueError("Венозна концентрація кисню повинна бути невід'ємною")
    
    return cardiac_output * (cao2 - cvo2)

def oxygen_consumption(oxygen_delivery: float, oxygen_extraction_ratio: float) -> float:
    """
    Обчислити споживання кисню.
    
    Параметри:
        oxygen_delivery: Доставка кисню (мл O₂/хв)
        oxygen_extraction_ratio: Коефіцієнт екстракції кисню (від 0 до 1)
    
    Повертає:
        Споживання кисню (мл O₂/хв)
    """
    if oxygen_delivery < 0:
        raise ValueError("Доставка кисню повинна бути невід'ємною")
    if oxygen_extraction_ratio < 0 or oxygen_extraction_ratio > 1:
        raise ValueError("Коефіцієнт екстракції кисню повинен бути в діапазоні [0, 1]")
    
    return oxygen_delivery * oxygen_extraction_ratio

def oxygen_content(hemoglobin_concentration: float, oxygen_saturation: float, 
                  partial_pressure_o2: float) -> float:
    """
    Обчислити вміст кисню в крові.
    
    Параметри:
        hemoglobin_concentration: Концентрація гемоглобіну (г/л)
        oxygen_saturation: Насичення киснем (від 0 до 1)
        partial_pressure_o2: Парціальний тиск кисню (ммHg)
    
    Повертає:
        Вміст кисню (мл O₂/л крові)
    """
    if hemoglobin_concentration < 0:
        raise ValueError("Концентрація гемоглобіну повинна бути невід'ємною")
    if oxygen_saturation < 0 or oxygen_saturation > 1:
        raise ValueError("Насичення киснем повинне бути в діапазоні [0, 1]")
    if partial_pressure_o2 < 0:
        raise ValueError("Парціальний тиск кисню повинен бути невід'ємним")
    
    # Зв'язаний кисень (з гемоглобіном)
    bound_oxygen = hemoglobin_concentration * OXYGEN_CAPACITY_HEMOGLOBIN * oxygen_saturation
    
    # Розчинений кисень (закон Генрі)
    dissolved_oxygen = 0.0031 * partial_pressure_o2  # мл O₂/л крові на ммHg
    
    return bound_oxygen + dissolved_oxygen

def alveolar_gas_equation(pressure_atmospheric: float, water_vapor_pressure: float, 
                         respiratory_exchange_ratio: float, co2_pressure: float) -> float:
    """
    Обчислити парціальний тиск кисню в альвеолах.
    
    Параметри:
        pressure_atmospheric: Атмосферний тиск (ммHg)
        water_vapor_pressure: Тиск водяної пари (ммHg)
        respiratory_exchange_ratio: Респіраторне обмінне відношення (R)
        co2_pressure: Парціальний тиск CO₂ (ммHg)
    
    Повертає:
        Парціальний тиск кисню в альвеолах (ммHg)
    """
    if pressure_atmospheric < 0:
        raise ValueError("Атмосферний тиск повинен бути невід'ємним")
    if water_vapor_pressure < 0:
        raise ValueError("Тиск водяної пари повинен бути невід'ємним")
    if respiratory_exchange_ratio < 0:
        raise ValueError("Респіраторне обмінне відношення повинне бути невід'ємним")
    if co2_pressure < 0:
        raise ValueError("Парціальний тиск CO₂ повинен бути невід'ємним")
    
    alveolar_o2_pressure = (pressure_atmospheric - water_vapor_pressure) - (co2_pressure / respiratory_exchange_ratio)
    return max(0, alveolar_o2_pressure)

def minute_ventilation(tidal_volume: float, respiratory_rate: float) -> float:
    """
    Обчислити хвилинну вентиляцію легень.
    
    Параметри:
        tidal_volume: Дихальний об'єм (л)
        respiratory_rate: Частота дихання (вдихів/хв)
    
    Повертає:
        Хвилинна вентиляція (л/хв)
    """
    if tidal_volume < 0:
        raise ValueError("Дихальний об'єм повинен бути невід'ємним")
    if respiratory_rate < 0:
        raise ValueError("Частота дихання повинна бути невід'ємною")
    
    return tidal_volume * respiratory_rate

def alveolar_ventilation(tidal_volume: float, dead_space: float, 
                        respiratory_rate: float) -> float:
    """
    Обчислити альвеолярну вентиляцію.
    
    Параметри:
        tidal_volume: Дихальний об'єм (л)
        dead_space: Мертвий простір (л)
        respiratory_rate: Частота дихання (вдихів/хв)
    
    Повертає:
        Альвеолярна вентиляція (л/хв)
    """
    if tidal_volume < 0:
        raise ValueError("Дихальний об'єм повинен бути невід'ємним")
    if dead_space < 0:
        raise ValueError("Мертвий простір повинен бути невід'ємним")
    if respiratory_rate < 0:
        raise ValueError("Частота дихання повинна бути невід'ємною")
    if dead_space > tidal_volume:
        raise ValueError("Мертвий простір не може бути більшим за дихальний об'єм")
    
    return (tidal_volume - dead_space) * respiratory_rate

def compliance(volume_change: float, pressure_change: float) -> float:
    """
    Обчислити розтягуваність легень.
    
    Параметри:
        volume_change: Зміна об'єму (л)
        pressure_change: Зміна тиску (смH₂O)
    
    Повертає:
        Розтягуваність (л/смH₂O)
    """
    if pressure_change == 0:
        raise ValueError("Зміна тиску не може дорівнювати нулю")
    
    return volume_change / pressure_change

def airway_resistance(pressure_difference: float, flow_rate: float) -> float:
    """
    Обчислити опір дихальних шляхів.
    
    Параметри:
        pressure_difference: Різниця тисків (смH₂O)
        flow_rate: Швидкість потоку (л/с)
    
    Повертає:
        Опір дихальних шляхів (смH₂O/(л/с))
    """
    if flow_rate == 0:
        raise ValueError("Швидкість потоку не може дорівнювати нулю")
    
    return pressure_difference / flow_rate

def body_mass_index(weight: float, height: float) -> float:
    """
    Обчислити індекс маси тіла (ІМТ).
    
    Параметри:
        weight: Маса тіла (кг)
        height: Зріст (м)
    
    Повертає:
        Індекс маси тіла (кг/м²)
    """
    if weight <= 0:
        raise ValueError("Маса тіла повинна бути додатньою")
    if height <= 0:
        raise ValueError("Зріст повинен бути додатнім")
    
    return weight / (height ** 2)

def creatinine_clearance(creatinine_serum: float, creatinine_urine: float, 
                        urine_volume: float, time: float) -> float:
    """
    Обчислити кліренс креатиніну.
    
    Параметри:
        creatinine_serum: Концентрація креатиніну в сироватці (мг/дл)
        creatinine_urine: Концентрація креатиніну в сечі (мг/дл)
        urine_volume: Об'єм сечі (мл)
        time: Час збирання сечі (хв)
    
    Повертає:
        Кліренс креатиніну (мл/хв)
    """
    if creatinine_serum <= 0:
        raise ValueError("Концентрація креатиніну в сироватці повинна бути додатньою")
    if creatinine_urine < 0:
        raise ValueError("Концентрація креатиніну в сечі повинна бути невід'ємною")
    if urine_volume < 0:
        raise ValueError("Об'єм сечі повинен бути невід'ємним")
    if time <= 0:
        raise ValueError("Час збирання сечі повинен бути додатнім")
    
    return (creatinine_urine * urine_volume) / (creatinine_serum * time)

def glomerular_filtration_rate(creatinine_clearance: float, 
                             correction_factor: float = 1.0) -> float:
    """
    Обчислити швидкість клубочкової фільтрації.
    
    Параметри:
        creatinine_clearance: Кліренс креатиніну (мл/хв)
        correction_factor: Корекційний фактор, за замовчуванням 1.0
    
    Повертає:
        Швидкість клубочкової фільтрації (мл/хв)
    """
    if creatinine_clearance < 0:
        raise ValueError("Кліренс креатиніну повинен бути невід'ємним")
    if correction_factor <= 0:
        raise ValueError("Корекційний фактор повинен бути додатнім")
    
    return creatinine_clearance * correction_factor

def anion_gap(na: float, cl: float, hco3: float) -> float:
    """
    Обчислити аніонний проміжок.
    
    Параметри:
        na: Концентрація натрію (мЕкв/л)
        cl: Концентрація хлору (мЕкв/л)
        hco3: Концентрація бікарбонату (мЕкв/л)
    
    Повертає:
        Аніонний проміжок (мЕкв/л)
    """
    if na < 0:
        raise ValueError("Концентрація натрію повинна бути невід'ємною")
    if cl < 0:
        raise ValueError("Концентрація хлору повинна бути невід'ємною")
    if hco3 < 0:
        raise ValueError("Концентрація бікарбонату повинна бути невід'ємною")
    
    return na - (cl + hco3)

def fractional_excretion_sodium(urine_na: float, serum_na: float, 
                               urine_creatinine: float, serum_creatinine: float) -> float:
    """
    Обчислити фракційну екскрецію натрію.
    
    Параметри:
        urine_na: Концентрація натрію в сечі (мЕкв/л)
        serum_na: Концентрація натрію в сироватці (мЕкв/л)
        urine_creatinine: Концентрація креатиніну в сечі (мг/дл)
        serum_creatinine: Концентрація креатиніну в сироватці (мг/дл)
    
    Повертає:
        Фракційна екскреція натрію (%)
    """
    if urine_na < 0:
        raise ValueError("Концентрація натрію в сечі повинна бути невід'ємною")
    if serum_na <= 0:
        raise ValueError("Концентрація натрію в сироватці повинна бути додатньою")
    if urine_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сечі повинна бути додатньою")
    if serum_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сироватці повинна бути додатньою")
    
    fe_na = (urine_na * serum_creatinine) / (serum_na * urine_creatinine) * 100
    return fe_na

def corrected_calcium(total_calcium: float, albumin: float) -> float:
    """
    Обчислити коригований рівень кальцію.
    
    Параметри:
        total_calcium: Загальний кальцій (мг/дл)
        albumin: Рівень альбуміну (г/дл)
    
    Повертає:
        Коригований рівень кальцію (мг/дл)
    """
    if total_calcium < 0:
        raise ValueError("Загальний кальцій повинен бути невід'ємним")
    if albumin < 0:
        raise ValueError("Рівень альбуміну повинен бути невід'ємним")
    
    # Формула: коригований кальцій = загальний кальцій + 0.8 * (4 - альбумін)
    return total_calcium + 0.8 * (4 - albumin)

def anion_gap_with_potassium(na: float, k: float, cl: float, hco3: float) -> float:
    """
    Обчислити аніонний проміжок з урахуванням калію.
    
    Параметри:
        na: Концентрація натрію (мЕкв/л)
        k: Концентрація калію (мЕкв/л)
        cl: Концентрація хлору (мЕкв/л)
        hco3: Концентрація бікарбонату (мЕкв/л)
    
    Повертає:
        Аніонний проміжок з калієм (мЕкв/л)
    """
    if na < 0:
        raise ValueError("Концентрація натрію повинна бути невід'ємною")
    if k < 0:
        raise ValueError("Концентрація калію повинна бути невід'ємною")
    if cl < 0:
        raise ValueError("Концентрація хлору повинна бути невід'ємною")
    if hco3 < 0:
        raise ValueError("Концентрація бікарбонату повинна бути невід'ємною")
    
    return (na + k) - (cl + hco3)

def osmolar_gap(measured_osmolality: float, calculated_osmolarity: float) -> float:
    """
    Обчислити осмолярний проміжок.
    
    Параметри:
        measured_osmolality: Виміряна осмоляльність (мОсм/кг)
        calculated_osmolarity: Розрахована осмолярність (мОсм/л)
    
    Повертає:
        Осмолярний проміжок (мОсм/кг)
    """
    if measured_osmolality < 0:
        raise ValueError("Виміряна осмоляльність повинна бути невід'ємною")
    if calculated_osmolarity < 0:
        raise ValueError("Розрахована осмолярність повинна бути невід'ємною")
    
    return measured_osmolality - calculated_osmolarity

def calculated_osmolarity(na: float, glucose: float, bun: float) -> float:
    """
    Обчислити розраховану осмолярність.
    
    Параметри:
        na: Концентрація натрію (мЕкв/л)
        glucose: Рівень глюкози (мг/дл)
        bun: Азот сечовини крові (мг/дл)
    
    Повертає:
        Розрахована осмолярність (мОсм/л)
    """
    if na < 0:
        raise ValueError("Концентрація натрію повинна бути невід'ємною")
    if glucose < 0:
        raise ValueError("Рівень глюкози повинен бути невід'ємним")
    if bun < 0:
        raise ValueError("Азот сечовини крові повинен бути невід'ємним")
    
    # Формула: 2 * Na + глюкоза/18 + BUN/2.8
    return 2 * na + glucose / 18 + bun / 2.8

def arterial_oxygen_content(hemoglobin: float, oxygen_saturation: float, 
                           partial_pressure_o2: float) -> float:
    """
    Обчислити артеріальний вміст кисню.
    
    Параметри:
        hemoglobin: Концентрація гемоглобіну (г/дл)
        oxygen_saturation: Насичення киснем (від 0 до 1)
        partial_pressure_o2: Парціальний тиск кисню (ммHg)
    
    Повертає:
        Артеріальний вміст кисню (мл O₂/дл крові)
    """
    if hemoglobin < 0:
        raise ValueError("Концентрація гемоглобіну повинна бути невід'ємною")
    if oxygen_saturation < 0 or oxygen_saturation > 1:
        raise ValueError("Насичення киснем повинне бути в діапазоні [0, 1]")
    if partial_pressure_o2 < 0:
        raise ValueError("Парціальний тиск кисню повинен бути невід'ємним")
    
    # Зв'язаний кисень
    bound_oxygen = hemoglobin * 1.34 * oxygen_saturation
    
    # Розчинений кисень
    dissolved_oxygen = 0.003 * partial_pressure_o2
    
    return bound_oxygen + dissolved_oxygen

def a_a_gradient(pao2: float, paco2: float, fio2: float, 
                atmospheric_pressure: float = 760) -> float:
    """
    Обчислити альвеолярно-артеріальний градієнт.
    
    Параметри:
        pao2: Парціальний тиск кисню в артеріальній крові (ммHg)
        paco2: Парціальний тиск вуглекислого газу (ммHg)
        fio2: Фракція вдихуваного кисню (від 0 до 1)
        atmospheric_pressure: Атмосферний тиск (ммHg), за замовчуванням 760
    
    Повертає:
        Альвеолярно-артеріальний градієнт (ммHg)
    """
    if pao2 < 0:
        raise ValueError("Парціальний тиск кисню в артеріальній крові повинен бути невід'ємним")
    if paco2 < 0:
        raise ValueError("Парціальний тиск вуглекислого газу повинен бути невід'ємним")
    if fio2 < 0 or fio2 > 1:
        raise ValueError("Фракція вдихуваного кисню повинна бути в діапазоні [0, 1]")
    if atmospheric_pressure <= 0:
        raise ValueError("Атмосферний тиск повинен бути додатнім")
    
    # Альвеолярний тиск кисню
    pao2_alveolar = fio2 * (atmospheric_pressure - 47) - (paco2 / 0.8)
    
    return pao2_alveolar - pao2

def oxygenation_index(pao2: float, fio2: float, mean_airway_pressure: float) -> float:
    """
    Обчислити індекс оксигенації.
    
    Параметри:
        pao2: Парціальний тиск кисню в артеріальній крові (ммHg)
        fio2: Фракція вдихуваного кисню (від 0 до 1)
        mean_airway_pressure: Середній тиск в дихальних шляхах (смH₂O)
    
    Повертає:
        Індекс оксигенації
    """
    if pao2 < 0:
        raise ValueError("Парціальний тиск кисню в артеріальній крові повинен бути невід'ємним")
    if fio2 <= 0 or fio2 > 1:
        raise ValueError("Фракція вдихуваного кисню повинна бути в діапазоні (0, 1]")
    if mean_airway_pressure < 0:
        raise ValueError("Середній тиск в дихальних шляхах повинен бути невід'ємним")
    
    return (pao2 * mean_airway_pressure) / fio2

def cerebral_perfusion_pressure(mean_arterial_pressure: float, 
                               intracranial_pressure: float) -> float:
    """
    Обчислити церебральний перфузійний тиск.
    
    Параметри:
        mean_arterial_pressure: Середній артеріальний тиск (ммHg)
        intracranial_pressure: Внутрішньочерепний тиск (ммHg)
    
    Повертає:
        Церебральний перфузійний тиск (ммHg)
    """
    if mean_arterial_pressure < 0:
        raise ValueError("Середній артеріальний тиск повинен бути невід'ємним")
    if intracranial_pressure < 0:
        raise ValueError("Внутрішньочерепний тиск повинен бути невід'ємним")
    
    return mean_arterial_pressure - intracranial_pressure

def pulmonary_vascular_resistance(pulmonary_artery_pressure: float, 
                                pulmonary_capillary_pressure: float, 
                                cardiac_output: float) -> float:
    """
    Обчислити легеневий судинний опір.
    
    Параметри:
        pulmonary_artery_pressure: Тиск в легеневій артерії (ммHg)
        pulmonary_capillary_pressure: Тиск в легеневих капілярах (ммHg)
        cardiac_output: Серцевий викид (л/хв)
    
    Повертає:
        Легеневий судинний опір (дин·с/см⁵)
    """
    if pulmonary_artery_pressure < 0:
        raise ValueError("Тиск в легеневій артерії повинен бути невід'ємним")
    if pulmonary_capillary_pressure < 0:
        raise ValueError("Тиск в легеневих капілярах повинен бути невід'ємним")
    if cardiac_output <= 0:
        raise ValueError("Серцевий викид повинен бути додатнім")
    
    pressure_gradient = pulmonary_artery_pressure - pulmonary_capillary_pressure
    # Конвертація з ммHg в дин/см²: 1 ммHg = 1333.22 дин/см²
    # Конвертація з л/хв в см³/с: 1 л/хв = 16.6667 см³/с
    return (pressure_gradient * 1333.22) / (cardiac_output * 16.6667)

def systemic_vascular_resistance_index(systemic_vascular_resistance: float, 
                                     body_surface_area: float) -> float:
    """
    Обчислити індекс загального периферичного опору.
    
    Параметри:
        systemic_vascular_resistance: Загальний периферичний опір (дин·с/см⁵)
        body_surface_area: Площа тіла (м²)
    
    Повертає:
        Індекс загального периферичного опору (дин·с/см⁵/м²)
    """
    if systemic_vascular_resistance < 0:
        raise ValueError("Загальний периферичний опір повинен бути невід'ємним")
    if body_surface_area <= 0:
        raise ValueError("Площа тіла повинна бути додатньою")
    
    return systemic_vascular_resistance / body_surface_area

def oxygen_delivery_index(oxygen_delivery: float, body_surface_area: float) -> float:
    """
    Обчислити індекс доставки кисню.
    
    Параметри:
        oxygen_delivery: Доставка кисню (мл O₂/хв)
        body_surface_area: Площа тіла (м²)
    
    Повертає:
        Індекс доставки кисню (мл O₂/хв/м²)
    """
    if oxygen_delivery < 0:
        raise ValueError("Доставка кисню повинна бути невід'ємною")
    if body_surface_area <= 0:
        raise ValueError("Площа тіла повинна бути додатньою")
    
    return oxygen_delivery / body_surface_area

def oxygen_consumption_index(oxygen_consumption: float, body_surface_area: float) -> float:
    """
    Обчислити індекс споживання кисню.
    
    Параметри:
        oxygen_consumption: Споживання кисню (мл O₂/хв)
        body_surface_area: Площа тіла (м²)
    
    Повертає:
        Індекс споживання кисню (мл O₂/хв/м²)
    """
    if oxygen_consumption < 0:
        raise ValueError("Споживання кисню повинне бути невід'ємним")
    if body_surface_area <= 0:
        raise ValueError("Площа тіла повинна бути додатньою")
    
    return oxygen_consumption / body_surface_area

def body_surface_area(weight: float, height: float) -> float:
    """
    Обчислити площу тіла за формулою Дюбуа.
    
    Параметри:
        weight: Маса тіла (кг)
        height: Зріст (см)
    
    Повертає:
        Площа тіла (м²)
    """
    if weight <= 0:
        raise ValueError("Маса тіла повинна бути додатньою")
    if height <= 0:
        raise ValueError("Зріст повинен бути додатнім")
    
    # Формула Дюбуа: BSA = 0.007184 × вага^0.425 × зріст^0.725
    return 0.007184 * (weight ** 0.425) * (height ** 0.725)

def ideal_body_weight(height: float, gender: str = "male") -> float:
    """
    Обчислити ідеальну масу тіла за формулою Девіна.
    
    Параметри:
        height: Зріст (см)
        gender: Стать ("male" або "female"), за замовчуванням "male"
    
    Повертає:
        Ідеальна маса тіла (кг)
    """
    if height <= 0:
        raise ValueError("Зріст повинен бути додатнім")
    
    if gender.lower() == "male":
        # Для чоловіків: IBW = 50 + 2.3 × (зріст в дюймах - 60)
        height_inches = height / 2.54
        if height_inches <= 60:
            return 50
        else:
            return 50 + 2.3 * (height_inches - 60)
    elif gender.lower() == "female":
        # Для жінок: IBW = 45.5 + 2.3 × (зріст в дюймах - 60)
        height_inches = height / 2.54
        if height_inches <= 60:
            return 45.5
        else:
            return 45.5 + 2.3 * (height_inches - 60)
    else:
        raise ValueError("Стать повинна бути 'male' або 'female'")

def adjusted_body_weight(ideal_body_weight: float, actual_body_weight: float) -> float:
    """
    Обчислити скориговану масу тіла.
    
    Параметри:
        ideal_body_weight: Ідеальна маса тіла (кг)
        actual_body_weight: Фактична маса тіла (кг)
    
    Повертає:
        Скоригована маса тіла (кг)
    """
    if ideal_body_weight <= 0:
        raise ValueError("Ідеальна маса тіла повинна бути додатньою")
    if actual_body_weight <= 0:
        raise ValueError("Фактична маса тіла повинна бути додатньою")
    
    # Формула: ABW = IBW + 0.4 × (актуальна вага - IBW)
    return ideal_body_weight + 0.4 * (actual_body_weight - ideal_body_weight)

def creatinine_clearance_cockcroft_gault(age: int, weight: float, creatinine: float, 
                                        gender: str = "male") -> float:
    """
    Обчислити кліренс креатиніну за формулою Cockcroft-Gault.
    
    Параметри:
        age: Вік (роки)
        weight: Маса тіла (кг)
        creatinine: Рівень креатиніну в сироватці (мг/дл)
        gender: Стать ("male" або "female"), за замовчуванням "male"
    
    Повертає:
        Кліренс креатиніну (мл/хв)
    """
    if age <= 0:
        raise ValueError("Вік повинен бути додатнім")
    if weight <= 0:
        raise ValueError("Маса тіла повинна бути додатньою")
    if creatinine <= 0:
        raise ValueError("Рівень креатиніну повинен бути додатнім")
    
    # Формула Cockcroft-Gault
    if gender.lower() == "male":
        crcl = ((140 - age) * weight) / (72 * creatinine)
    else:  # female
        crcl = ((140 - age) * weight * 0.85) / (72 * creatinine)
    
    return crcl

def egfr_mdrd(creatinine: float, age: int, gender: str = "male", 
             race: str = "non-black") -> float:
    """
    Обчислити eGFR за формулою MDRD.
    
    Параметри:
        creatinine: Рівень креатиніну в сироватці (мг/дл)
        age: Вік (роки)
        gender: Стать ("male" або "female"), за замовчуванням "male"
        race: Раса ("black" або "non-black"), за замовчуванням "non-black"
    
    Повертає:
        eGFR (мл/хв/1.73 м²)
    """
    if creatinine <= 0:
        raise ValueError("Рівень креатиніну повинен бути додатнім")
    if age <= 0:
        raise ValueError("Вік повинен бути додатнім")
    
    # Формула MDRD
    if gender.lower() == "male":
        gender_coeff = 1
    else:
        gender_coeff = 0.742
    
    if race.lower() == "black":
        race_coeff = 1.212
    else:
        race_coeff = 1
    
    egfr = 175 * (creatinine ** (-1.154)) * (age ** (-0.203)) * gender_coeff * race_coeff
    return egfr

def egfr_ckd_epi(creatinine: float, age: int, gender: str = "male", 
                race: str = "non-black") -> float:
    """
    Обчислити eGFR за формулою CKD-EPI.
    
    Параметри:
        creatinine: Рівень креатиніну в сироватці (мг/дл)
        age: Вік (роки)
        gender: Стать ("male" або "female"), за замовчуванням "male"
        race: Раса ("black" або "non-black"), за замовчуванням "non-black"
    
    Повертає:
        eGFR (мл/хв/1.73 м²)
    """
    if creatinine <= 0:
        raise ValueError("Рівень креатиніну повинен бути додатнім")
    if age <= 0:
        raise ValueError("Вік повинен бути додатнім")
    
    # Формула CKD-EPI
    if gender.lower() == "male":
        if creatinine <= 0.9:
            egfr = 141 * (creatinine / 0.9) ** (-0.411) * (0.993 ** age)
        else:
            egfr = 141 * (creatinine / 0.9) ** (-1.209) * (0.993 ** age)
        gender_coeff = 1
    else:  # female
        if creatinine <= 0.7:
            egfr = 144 * (creatinine / 0.7) ** (-0.329) * (0.993 ** age)
        else:
            egfr = 144 * (creatinine / 0.7) ** (-1.209) * (0.993 ** age)
        gender_coeff = 1.018
    
    if race.lower() == "black":
        race_coeff = 1.159
    else:
        race_coeff = 1
    
    return egfr * gender_coeff * race_coeff

def fractional_excretion_electrolyte(urine_concentration: float, 
                                   serum_concentration: float,
                                   urine_creatinine: float, 
                                   serum_creatinine: float) -> float:
    """
    Обчислити фракційну екскрецію електроліту.
    
    Параметри:
        urine_concentration: Концентрація електроліту в сечі
        serum_concentration: Концентрація електроліту в сироватці
        urine_creatinine: Концентрація креатиніну в сечі (мг/дл)
        serum_creatinine: Концентрація креатиніну в сироватці (мг/дл)
    
    Повертає:
        Фракційна екскреція електроліту (%)
    """
    if urine_concentration < 0:
        raise ValueError("Концентрація електроліту в сечі повинна бути невід'ємною")
    if serum_concentration <= 0:
        raise ValueError("Концентрація електроліту в сироватці повинна бути додатньою")
    if urine_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сечі повинна бути додатньою")
    if serum_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сироватці повинна бути додатньою")
    
    fe = (urine_concentration * serum_creatinine) / (serum_concentration * urine_creatinine) * 100
    return fe

def urine_protein_excretion(urine_protein: float, urine_volume: float) -> float:
    """
    Обчислити екскрецію білка з сечею.
    
    Параметри:
        urine_protein: Концентрація білка в сечі (мг/дл)
        urine_volume: Об'єм сечі (л/добу)
    
    Повертає:
        Екскреція білка (мг/добу)
    """
    if urine_protein < 0:
        raise ValueError("Концентрація білка в сечі повинна бути невід'ємною")
    if urine_volume < 0:
        raise ValueError("Об'єм сечі повинен бути невід'ємним")
    
    return urine_protein * urine_volume * 10  # Перетворення з мг/дл в мг/л, потім в мг/добу

def transferrin_saturation(serum_iron: float, tibc: float) -> float:
    """
    Обчислити насичення трансферину.
    
    Параметри:
        serum_iron: Рівень заліза в сироватці (мкг/дл)
        tibc: Загальна зв'язуюча здатність до заліза (мкг/дл)
    
    Повертає:
        Насичення трансферину (%)
    """
    if serum_iron < 0:
        raise ValueError("Рівень заліза в сироватці повинен бути невід'ємним")
    if tibc <= 0:
        raise ValueError("Загальна зв'язуюча здатність до заліза повинна бути додатньою")
    
    return (serum_iron / tibc) * 100

def iron_binding_capacity(unbound_iron: float, serum_iron: float) -> float:
    """
    Обчислити зв'язуючу здатність до заліза.
    
    Параметри:
        unbound_iron: Незв'язане залізо (мкг/дл)
        serum_iron: Рівень заліза в сироватці (мкг/дл)
    
    Повертає:
        Зв'язуюча здатність до заліза (мкг/дл)
    """
    if unbound_iron < 0:
        raise ValueError("Незв'язане залізо повинне бути невід'ємним")
    if serum_iron < 0:
        raise ValueError("Рівень заліза в сироватці повинен бути невід'ємним")
    
    return unbound_iron + serum_iron

def anion_gap_delta_ratio(anion_gap: float, bicarbonate: float) -> float:
    """
    Обчислити дельта-відношення аніонного проміжку.
    
    Параметри:
        anion_gap: Аніонний проміжок (мЕкв/л)
        bicarbonate: Рівень бікарбонату (мЕкв/л)
    
    Повертає:
        Дельта-відношення
    """
    if anion_gap < 0:
        raise ValueError("Аніонний проміжок повинен бути невід'ємним")
    if bicarbonate < 0:
        raise ValueError("Рівень бікарбонату повинен бути невід'ємним")
    
    # Нормальні значення
    normal_ag = 12
    normal_hco3 = 24
    
    delta_ag = anion_gap - normal_ag
    delta_hco3 = normal_hco3 - bicarbonate
    
    if delta_hco3 == 0:
        return float('inf') if delta_ag > 0 else 0
    
    return delta_ag / delta_hco3

def corrected_sodium(measured_sodium: float, glucose: float) -> float:
    """
    Обчислити коригований рівень натрію.
    
    Параметри:
        measured_sodium: Виміряний рівень натрію (мЕкв/л)
        glucose: Рівень глюкози (мг/дл)
    
    Повертає:
        Коригований рівень натрію (мЕкв/л)
    """
    if measured_sodium < 0:
        raise ValueError("Виміряний рівень натрію повинен бути невід'ємним")
    if glucose < 0:
        raise ValueError("Рівень глюкози повинен бути невід'ємним")
    
    # Формула: коригований натрій = виміряний натрій + 0.016 * (глюкоза - 100)
    return measured_sodium + 0.016 * (glucose - 100)

def osmolal_gap(measured_osmolality: float, calculated_osmolality: float) -> float:
    """
    Обчислити осмоляльний проміжок.
    
    Параметри:
        measured_osmolality: Виміряна осмоляльність (мОсм/кг)
        calculated_osmolality: Розрахована осмоляльність (мОсм/кг)
    
    Повертає:
        Осмоляльний проміжок (мОсм/кг)
    """
    if measured_osmolality < 0:
        raise ValueError("Виміряна осмоляльність повинна бути невід'ємною")
    if calculated_osmolality < 0:
        raise ValueError("Розрахована осмоляльність повинна бути невід'ємною")
    
    return measured_osmolality - calculated_osmolality

def fractional_excretion_urea(urine_urea: float, serum_urea: float,
                            urine_creatinine: float, serum_creatinine: float) -> float:
    """
    Обчислити фракційну екскрецію сечовини.
    
    Параметри:
        urine_urea: Концентрація сечовини в сечі (мг/дл)
        serum_urea: Концентрація сечовини в сироватці (мг/дл)
        urine_creatinine: Концентрація креатиніну в сечі (мг/дл)
        serum_creatinine: Концентрація креатиніну в сироватці (мг/дл)
    
    Повертає:
        Фракційна екскреція сечовини (%)
    """
    if urine_urea < 0:
        raise ValueError("Концентрація сечовини в сечі повинна бути невід'ємною")
    if serum_urea <= 0:
        raise ValueError("Концентрація сечовини в сироватці повинна бути додатньою")
    if urine_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сечі повинна бути додатньою")
    if serum_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сироватці повинна бути додатньою")
    
    fe_urea = (urine_urea * serum_creatinine) / (serum_urea * urine_creatinine) * 100
    return fe_urea

def urine_protein_to_creatinine_ratio(urine_protein: float, urine_creatinine: float) -> float:
    """
    Обчислити відношення білка до креатиніну в сечі.
    
    Параметри:
        urine_protein: Концентрація білка в сечі (мг/дл)
        urine_creatinine: Концентрація креатиніну в сечі (мг/дл)
    
    Повертає:
        Відношення білка до креатиніну
    """
    if urine_protein < 0:
        raise ValueError("Концентрація білка в сечі повинна бути невід'ємною")
    if urine_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сечі повинна бути додатньою")
    
    return urine_protein / urine_creatinine

def arterial_alveolar_oxygen_ratio(pao2: float, pao2_alveolar: float) -> float:
    """
    Обчислити артеріально-альвеолярне відношення кисню.
    
    Параметри:
        pao2: Парціальний тиск кисню в артеріальній крові (ммHg)
        pao2_alveolar: Альвеолярний парціальний тиск кисню (ммHg)
    
    Повертає:
        Артеріально-альвеолярне відношення кисню
    """
    if pao2 < 0:
        raise ValueError("Парціальний тиск кисню в артеріальній крові повинен бути невід'ємним")
    if pao2_alveolar <= 0:
        raise ValueError("Альвеолярний парціальний тиск кисню повинен бути додатнім")
    
    return pao2 / pao2_alveolar

def oxygen_saturation_calculation(pao2: float, p50: float = 26.7) -> float:
    """
    Обчислити насичення киснем за формулою Сігма-Еді.
    
    Параметри:
        pao2: Парціальний тиск кисню (ммHg)
        p50: P50 (ммHg), за замовчуванням 26.7
    
    Повертає:
        Насичення киснем (від 0 до 1)
    """
    if pao2 < 0:
        raise ValueError("Парціальний тиск кисню повинен бути невід'ємним")
    if p50 <= 0:
        raise ValueError("P50 повинен бути додатнім")
    
    # Формула Сігма-Еді: SO₂ = (PaO₂^n) / (PaO₂^n + P50^n)
    # де n ≈ 2.7 для нормальної кривої дисоціації оксигемоглобіну
    n = 2.7
    so2 = (pao2 ** n) / ((pao2 ** n) + (p50 ** n))
    return max(0, min(1, so2))  # Обмеження діапазону [0, 1]

def alveolar_arterial_gradient(pao2_alveolar: float, pao2: float) -> float:
    """
    Обчислити альвеолярно-артеріальний градієнт кисню.
    
    Параметри:
        pao2_alveolar: Альвеолярний парціальний тиск кисню (ммHg)
        pao2: Парціальний тиск кисню в артеріальній крові (ммHg)
    
    Повертає:
        Альвеолярно-артеріальний градієнт (ммHg)
    """
    if pao2_alveolar < 0:
        raise ValueError("Альвеолярний парціальний тиск кисню повинен бути невід'ємним")
    if pao2 < 0:
        raise ValueError("Парціальний тиск кисню в артеріальній крові повинен бути невід'ємним")
    
    return pao2_alveolar - pao2

def oxygenation_ratio(pao2: float, fio2: float) -> float:
    """
    Обчислити відношення оксигенації (P/F ratio).
    
    Параметри:
        pao2: Парціальний тиск кисню в артеріальній крові (ммHg)
        fio2: Фракція вдихуваного кисню (від 0 до 1)
    
    Повертає:
        Відношення оксигенації
    """
    if pao2 < 0:
        raise ValueError("Парціальний тиск кисню в артеріальній крові повинен бути невід'ємним")
    if fio2 <= 0 or fio2 > 1:
        raise ValueError("Фракція вдихуваного кисню повинна бути в діапазоні (0, 1]")
    
    return pao2 / fio2

def dead_space_to_tidal_volume_ratio(dead_space: float, tidal_volume: float) -> float:
    """
    Обчислити відношення мертвого простору до дихального об'єму.
    
    Параметри:
        dead_space: Мертвий простір (л)
        tidal_volume: Дихальний об'єм (л)
    
    Повертає:
        Відношення мертвого простору до дихального об'єму
    """
    if dead_space < 0:
        raise ValueError("Мертвий простір повинен бути невід'ємним")
    if tidal_volume <= 0:
        raise ValueError("Дихальний об'єм повинен бути додатнім")
    if dead_space > tidal_volume:
        raise ValueError("Мертвий простір не може бути більшим за дихальний об'єм")
    
    return dead_space / tidal_volume

def cardiac_index(cardiac_output: float, body_surface_area: float) -> float:
    """
    Обчислити серцевий індекс.
    
    Параметри:
        cardiac_output: Серцевий викид (л/хв)
        body_surface_area: Площа тіла (м²)
    
    Повертає:
        Серцевий індекс (л/хв/м²)
    """
    if cardiac_output < 0:
        raise ValueError("Серцевий викид повинен бути невід'ємним")
    if body_surface_area <= 0:
        raise ValueError("Площа тіла повинна бути додатньою")
    
    return cardiac_output / body_surface_area

def stroke_volume_index(stroke_volume: float, body_surface_area: float) -> float:
    """
    Обчислити індекс ударного об'єму.
    
    Параметри:
        stroke_volume: Ударний об'єм (мл)
        body_surface_area: Площа тіла (м²)
    
    Повертає:
        Індекс ударного об'єму (мл/м²)
    """
    if stroke_volume < 0:
        raise ValueError("Ударний об'єм повинен бути невід'ємним")
    if body_surface_area <= 0:
        raise ValueError("Площа тіла повинна бути додатньою")
    
    return stroke_volume / body_surface_area

def systemic_vascular_resistance_units(systemic_vascular_resistance: float) -> float:
    """
    Обчислити загальний периферичний опір в одиницях Wood.
    
    Параметри:
        systemic_vascular_resistance: Загальний периферичний опір (дин·с/см⁵)
    
    Повертає:
        Загальний периферичний опір в одиницях Wood (Wood units)
    """
    if systemic_vascular_resistance < 0:
        raise ValueError("Загальний периферичний опір повинен бути невід'ємним")
    
    # 1 Wood unit = 80 dyn·s/cm⁵
    return systemic_vascular_resistance / 80

def pulmonary_vascular_resistance_units(pulmonary_vascular_resistance: float) -> float:
    """
    Обчислити легеневий судинний опір в одиницях Wood.
    
    Параметри:
        pulmonary_vascular_resistance: Легеневий судинний опір (дин·с/см⁵)
    
    Повертає:
        Легеневий судинний опір в одиницях Wood (Wood units)
    """
    if pulmonary_vascular_resistance < 0:
        raise ValueError("Легеневий судинний опір повинен бути невід'ємним")
    
    # 1 Wood unit = 80 dyn·s/cm⁵
    return pulmonary_vascular_resistance / 80

def oxygen_consumption_rate(oxygen_consumption: float, body_weight: float) -> float:
    """
    Обчислити швидкість споживання кисню на кг маси тіла.
    
    Параметри:
        oxygen_consumption: Споживання кисню (мл O₂/хв)
        body_weight: Маса тіла (кг)
    
    Повертає:
        Швидкість споживання кисню (мл O₂/хв/кг)
    """
    if oxygen_consumption < 0:
        raise ValueError("Споживання кисню повинне бути невід'ємним")
    if body_weight <= 0:
        raise ValueError("Маса тіла повинна бути додатньою")
    
    return oxygen_consumption / body_weight

def respiratory_quotient(oxygen_consumption: float, carbon_dioxide_production: float) -> float:
    """
    Обчислити респіраторне обмінне відношення.
    
    Параметри:
        oxygen_consumption: Споживання кисню (мл O₂/хв)
        carbon_dioxide_production: Виробництво вуглекислого газу (мл CO₂/хв)
    
    Повертає:
        Респіраторне обмінне відношення
    """
    if oxygen_consumption <= 0:
        raise ValueError("Споживання кисню повинне бути додатнім")
    if carbon_dioxide_production < 0:
        raise ValueError("Виробництво вуглекислого газу повинне бути невід'ємним")
    
    return carbon_dioxide_production / oxygen_consumption

def energy_expenditure_harris_benedict(weight: float, height: float, age: int, 
                                     gender: str = "male", activity_factor: float = 1.2) -> float:
    """
    Обчислити енергетичні витрати за формулою Harris-Benedict.
    
    Параметри:
        weight: Маса тіла (кг)
        height: Зріст (см)
        age: Вік (роки)
        gender: Стать ("male" або "female"), за замовчуванням "male"
        activity_factor: Фактор активності, за замовчуванням 1.2 (мінімальна активність)
    
    Повертає:
        Енергетичні витрати (ккал/добу)
    """
    if weight <= 0:
        raise ValueError("Маса тіла повинна бути додатньою")
    if height <= 0:
        raise ValueError("Зріст повинен бути додатнім")
    if age <= 0:
        raise ValueError("Вік повинен бути додатнім")
    if activity_factor <= 0:
        raise ValueError("Фактор активності повинен бути додатнім")
    
    # Формула Harris-Benedict
    if gender.lower() == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:  # female
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    
    return bmr * activity_factor

def fluid_balance(input_fluids: float, output_fluids: float) -> float:
    """
    Обчислити баланс рідини.
    
    Параметри:
        input_fluids: Надходження рідини (мл)
        output_fluids: Виведення рідини (мл)
    
    Повертає:
        Баланс рідини (мл)
    """
    if input_fluids < 0:
        raise ValueError("Надходження рідини повинне бути невід'ємним")
    if output_fluids < 0:
        raise ValueError("Виведення рідини повинне бути невід'ємним")
    
    return input_fluids - output_fluids

def urine_output_rate(urine_volume: float, time: float) -> float:
    """
    Обчислити швидкість виведення сечі.
    
    Параметри:
        urine_volume: Об'єм сечі (мл)
        time: Час (години)
    
    Повертає:
        Швидкість виведення сечі (мл/год)
    """
    if urine_volume < 0:
        raise ValueError("Об'єм сечі повинен бути невід'ємним")
    if time <= 0:
        raise ValueError("Час повинен бути додатнім")
    
    return urine_volume / time

def creatinine_clearance_simplified(weight: float, age: int, serum_creatinine: float, 
                                  gender: str = "male") -> float:
    """
    Обчислити спрощений кліренс креатиніну.
    
    Параметри:
        weight: Маса тіла (кг)
        age: Вік (роки)
        serum_creatinine: Рівень креатиніну в сироватці (мг/дл)
        gender: Стать ("male" або "female"), за замовчуванням "male"
    
    Повертає:
        Спрощений кліренс креатиніну (мл/хв)
    """
    if weight <= 0:
        raise ValueError("Маса тіла повинна бути додатньою")
    if age <= 0:
        raise ValueError("Вік повинен бути додатнім")
    if serum_creatinine <= 0:
        raise ValueError("Рівень креатиніну повинен бути додатнім")
    
    # Спрощена формула
    if gender.lower() == "male":
        crcl = ((140 - age) * weight) / (72 * serum_creatinine)
    else:  # female
        crcl = ((140 - age) * weight * 0.85) / (72 * serum_creatinine)
    
    return crcl

def fractional_sodium_excretion(urine_na: float, serum_na: float,
                              urine_creatinine: float, serum_creatinine: float) -> float:
    """
    Обчислити фракційну екскрецію натрію.
    
    Параметри:
        urine_na: Концентрація натрію в сечі (мЕкв/л)
        serum_na: Концентрація натрію в сироватці (мЕкв/л)
        urine_creatinine: Концентрація креатиніну в сечі (мг/дл)
        serum_creatinine: Концентрація креатиніну в сироватці (мг/дл)
    
    Повертає:
        Фракційна екскреція натрію (%)
    """
    if urine_na < 0:
        raise ValueError("Концентрація натрію в сечі повинна бути невід'ємною")
    if serum_na <= 0:
        raise ValueError("Концентрація натрію в сироватці повинна бути додатньою")
    if urine_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сечі повинна бути додатньою")
    if serum_creatinine <= 0:
        raise ValueError("Концентрація креатиніну в сироватці повинна бути додатньою")
    
    fe_na = (urine_na * serum_creatinine) / (serum_na * urine_creatinine) * 100
    return fe_na