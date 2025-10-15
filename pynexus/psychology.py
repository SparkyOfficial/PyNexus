"""
Модуль для обчислювальної психології в PyNexus.
Містить функції для психологічного тестування, аналізу поведінки, когнітивного моделювання та інших психологічних обчислень.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import numpy as np

# Константи для психології
AVERAGE_IQ = 100.0  # Середній IQ
IQ_STD_DEV = 15.0   # Стандартне відхилення IQ
STRESS_THRESHOLD = 7.0  # Поріг стресу (за шкалою від 1 до 10)
HAPPINESS_BASELINE = 5.0  # Базовий рівень щастя
ATTENTION_SPAN_AVERAGE = 20.0  # Середня тривалість уваги (хвилини)
MEMORY_DECAY_RATE = 0.05  # Швидкість забування
LEARNING_RATE_DEFAULT = 0.1  # Стандартна швидкість навчання

def big_five_personality_traits(answers: List[int]) -> Dict[str, float]:
    """
    Оцінити риси великої п'ятірки особистості.
    
    Параметри:
        answers: Відповіді на питання (1-5 шкала)
    
    Повертає:
        Словник з рисами особистості
    """
    if not answers:
        return {
            'openness': 0.0,
            'conscientiousness': 0.0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'neuroticism': 0.0
        }
    
    # Припускаємо, що питання впорядковані за рисами:
    # 0-9: Відкритість (Openness)
    # 10-19: Сумлінність (Conscientiousness)
    # 20-29: Екстраверсія (Extraversion)
    # 30-39: Лад з іншими (Agreeableness)
    # 40-49: Невротизм (Neuroticism)
    
    if len(answers) < 50:
        # Доповнюємо відповіді, якщо їх менше 50
        answers = answers + [3] * (50 - len(answers))
    elif len(answers) > 50:
        # Обрізаємо, якщо їх більше 50
        answers = answers[:50]
    
    openness = sum(answers[0:10]) / 10.0
    conscientiousness = sum(answers[10:20]) / 10.0
    extraversion = sum(answers[20:30]) / 10.0
    agreeableness = sum(answers[30:40]) / 10.0
    neuroticism = sum(answers[40:50]) / 10.0
    
    return {
        'openness': openness,
        'conscientiousness': conscientiousness,
        'extraversion': extraversion,
        'agreeableness': agreeableness,
        'neuroticism': neuroticism
    }

def iq_score_calculation(correct_answers: int, total_questions: int, 
                        time_taken: float, average_time: float) -> float:
    """
    Обчислити IQ-бал на основі результатів тесту.
    
    Параметри:
        correct_answers: Кількість правильних відповідей
        total_questions: Загальна кількість питань
        time_taken: Час, витрачений на тест (секунди)
        average_time: Середній час для тесту (секунди)
    
    Повертає:
        IQ-бал
    """
    if total_questions <= 0:
        raise ValueError("Загальна кількість питань повинна бути додатньою")
    
    if time_taken <= 0 or average_time <= 0:
        raise ValueError("Час повинен бути додатнім")
    
    # Обчислюємо базовий відсоток правильних відповідей
    accuracy = correct_answers / total_questions
    
    # Обчислюємо бонус за швидкість
    speed_bonus = 1.0
    if time_taken < average_time:
        # Чим швидше, тим більший бонус (до 20%)
        speed_bonus = min(1.2, 1.0 + (average_time - time_taken) / average_time * 0.2)
    
    # Обчислюємо IQ-бал
    raw_score = accuracy * speed_bonus
    # Нормалізуємо до шкали IQ (середнє 100, стандартне відхилення 15)
    iq_score = AVERAGE_IQ + (raw_score - 0.5) * 2 * IQ_STD_DEV
    
    return max(0.0, iq_score)

def stress_level_assessment(heart_rate: float, cortisol_level: float, 
                          sleep_quality: float, workload: float) -> Dict[str, float]:
    """
    Оцінити рівень стресу на основі фізіологічних та поведінкових показників.
    
    Параметри:
        heart_rate: Частота серцебиття (ударів/хв)
        cortisol_level: Рівень кортизолу (нг/мл)
        sleep_quality: Якість сну (1-10 шкала)
        workload: Навантаження (1-10 шкала)
    
    Повертає:
        Словник з оцінкою стресу
    """
    # Нормалізуємо показники
    # Нормальна частота серця: 60-100 ударів/хв
    normalized_heart_rate = max(0.0, min(1.0, abs(heart_rate - 80) / 40))
    
    # Нормальний рівень кортизолу: 50-200 нг/мл
    normalized_cortisol = max(0.0, min(1.0, abs(cortisol_level - 125) / 150))
    
    # Якість сну: 1-10 (10 - найкраща)
    normalized_sleep = 1.0 - (sleep_quality / 10.0)
    
    # Навантаження: 1-10 (10 - найбільше)
    normalized_workload = workload / 10.0
    
    # Комбінований індекс стресу
    stress_index = (normalized_heart_rate * 0.3 + 
                   normalized_cortisol * 0.3 + 
                   normalized_sleep * 0.2 + 
                   normalized_workload * 0.2)
    
    # Перетворюємо в шкалу 1-10
    stress_level = stress_index * 10.0
    
    # Класифікація стресу
    if stress_level < 3.0:
        stress_category = "Низький"
    elif stress_level < 6.0:
        stress_category = "Середній"
    elif stress_level < 8.0:
        stress_category = "Високий"
    else:
        stress_category = "Дуже високий"
    
    return {
        'stress_level': stress_level,
        'stress_category': stress_category,
        'heart_rate_component': normalized_heart_rate * 10.0,
        'cortisol_component': normalized_cortisol * 10.0,
        'sleep_component': normalized_sleep * 10.0,
        'workload_component': normalized_workload * 10.0
    }

def memory_retention_model(initial_memory: float, time_elapsed: float, 
                          decay_rate: float = MEMORY_DECAY_RATE) -> float:
    """
    Модель забування Еббінгауза.
    
    Параметри:
        initial_memory: Початковий рівень запам'ятовування (0-1)
        time_elapsed: Час, що минув (години)
        decay_rate: Швидкість забування
    
    Повертає:
        Поточний рівень запам'ятовування
    """
    if initial_memory < 0.0 or initial_memory > 1.0:
        raise ValueError("Початковий рівень пам'яті повинен бути між 0 та 1")
    
    if time_elapsed < 0:
        raise ValueError("Час не може бути від'ємним")
    
    if decay_rate < 0:
        raise ValueError("Швидкість забування не може бути від'ємною")
    
    # Формула Еббінгауза: R = e^(-λt)
    # де R - рівень запам'ятовування, λ - швидкість забування, t - час
    retention = initial_memory * math.exp(-decay_rate * time_elapsed)
    
    return max(0.0, min(1.0, retention))

def learning_curve_model(initial_skill: float, training_sessions: int, 
                        learning_rate: float = LEARNING_RATE_DEFAULT,
                        max_skill: float = 1.0) -> List[float]:
    """
    Модель кривої навчання.
    
    Параметри:
        initial_skill: Початковий рівень навичок (0-1)
        training_sessions: Кількість тренувальних сесій
        learning_rate: Швидкість навчання
        max_skill: Максимальний рівень навичок
    
    Повертає:
        Список рівнів навичок після кожної сесії
    """
    if initial_skill < 0.0 or initial_skill > max_skill:
        raise ValueError("Початковий рівень навичок повинен бути між 0 та максимальним значенням")
    
    if training_sessions < 0:
        raise ValueError("Кількість тренувальних сесій не може бути від'ємною")
    
    if learning_rate <= 0 or learning_rate > 1:
        raise ValueError("Швидкість навчання повинна бути між 0 та 1")
    
    if max_skill <= 0:
        raise ValueError("Максимальний рівень навичок повинен бути додатнім")
    
    skill_levels = [initial_skill]
    current_skill = initial_skill
    
    for _ in range(training_sessions):
        # Модель навчання: новий рівень = поточний + швидкість * (максимум - поточний)
        skill_gain = learning_rate * (max_skill - current_skill)
        current_skill = min(max_skill, current_skill + skill_gain)
        skill_levels.append(current_skill)
    
    return skill_levels

def attention_span_model(task_difficulty: float, fatigue_level: float, 
                        interest_level: float, time_elapsed: float) -> float:
    """
    Модель тривалості уваги.
    
    Параметри:
        task_difficulty: Складність завдання (1-10)
        fatigue_level: Рівень втоми (1-10)
        interest_level: Рівень інтересу (1-10)
        time_elapsed: Час, що минув (хвилини)
    
    Повертає:
        Очікувана тривалість уваги (хвилини)
    """
    if not all(1.0 <= x <= 10.0 for x in [task_difficulty, fatigue_level, interest_level]):
        raise ValueError("Усі параметри повинні бути в діапазоні 1-10")
    
    if time_elapsed < 0:
        raise ValueError("Час не може бути від'ємним")
    
    # Базова тривалість уваги
    base_attention = ATTENTION_SPAN_AVERAGE
    
    # Коригування за складністю (складніше завдання - менша увага)
    difficulty_factor = 1.0 - (task_difficulty - 5.0) / 10.0
    
    # Коригування за втомою (більше втоми - менша увага)
    fatigue_factor = 1.0 - (fatigue_level - 1.0) / 20.0
    
    # Коригування за інтересом (більше інтересу - більша увага)
    interest_factor = 1.0 + (interest_level - 5.0) / 10.0
    
    # Зменшення уваги з часом
    time_decay = math.exp(-time_elapsed / 30.0)
    
    attention_span = (base_attention * 
                     difficulty_factor * 
                     fatigue_factor * 
                     interest_factor * 
                     time_decay)
    
    return max(0.0, attention_span)

def cognitive_load_assessment(intrinsic_load: float, extraneous_load: float, 
                            germane_load: float) -> Dict[str, float]:
    """
    Оцінка когнітивного навантаження (теорія когнітивного навантаження).
    
    Параметри:
        intrinsic_load: Внутрішнє навантаження (1-10)
        extraneous_load: Зовнішнє навантаження (1-10)
        germane_load: Корисне навантаження (1-10)
    
    Повертає:
        Словник з оцінкою когнітивного навантаження
    """
    if not all(1.0 <= x <= 10.0 for x in [intrinsic_load, extraneous_load, germane_load]):
        raise ValueError("Усі параметри повинні бути в діапазоні 1-10")
    
    # Загальне когнітивне навантаження
    total_load = intrinsic_load + extraneous_load + germane_load
    
    # Ефективність навчання
    # Оптимально, коли germane_load високий, а extraneous_load низький
    learning_efficiency = germane_load / (extraneous_load + 1.0)  # +1 щоб уникнути ділення на 0
    
    # Класифікація навантаження
    if total_load < 15.0:
        load_category = "Низьке"
    elif total_load < 22.0:
        load_category = "Середнє"
    else:
        load_category = "Високе"
    
    return {
        'total_load': total_load,
        'load_category': load_category,
        'learning_efficiency': learning_efficiency,
        'intrinsic_load': intrinsic_load,
        'extraneous_load': extraneous_load,
        'germane_load': germane_load
    }

def emotional_valence_arousal(valence: float, arousal: float) -> Dict[str, Union[str, float]]:
    """
    Класифікація емоцій за моделлю валентності-збудження.
    
    Параметри:
        valence: Валентність (-1 до 1, де -1 - негативна, 1 - позитивна)
        arousal: Збудження (0 до 1, де 0 - низьке, 1 - високе)
    
    Повертає:
        Словник з класифікацією емоції
    """
    if not -1.0 <= valence <= 1.0:
        raise ValueError("Валентність повинна бути в діапазоні -1 до 1")
    
    if not 0.0 <= arousal <= 1.0:
        raise ValueError("Збудження повинно бути в діапазоні 0 до 1")
    
    # Класифікація валентності
    if valence < -0.5:
        valence_category = "Негативна"
    elif valence > 0.5:
        valence_category = "Позитивна"
    else:
        valence_category = "Нейтральна"
    
    # Класифікація збудження
    if arousal < 0.33:
        arousal_category = "Низьке"
    elif arousal < 0.66:
        arousal_category = "Середнє"
    else:
        arousal_category = "Високе"
    
    # Визначення конкретної емоції
    if valence_category == "Позитивна" and arousal_category == "Високе":
        emotion = "Радість"
    elif valence_category == "Позитивна" and arousal_category == "Низьке":
        emotion = "Спокій"
    elif valence_category == "Негативна" and arousal_category == "Високе":
        emotion = "Злість"
    elif valence_category == "Негативна" and arousal_category == "Низьке":
        emotion = "Сум"
    elif valence_category == "Нейтральна" and arousal_category == "Високе":
        emotion = "Здивування"
    else:
        emotion = "Сонливість"
    
    return {
        'emotion': emotion,
        'valence_category': valence_category,
        'arousal_category': arousal_category,
        'valence_score': valence,
        'arousal_score': arousal
    }

def decision_making_model(probability: float, outcome_value: float, 
                         risk_aversion: float = 0.5) -> float:
    """
    Модель прийняття рішень (очікувана корисність).
    
    Параметри:
        probability: Ймовірність успіху (0-1)
        outcome_value: Значення результату
        risk_aversion: Схильність до ризику (0-1, де 0 - любить ризик, 1 - уникає ризику)
    
    Повертає:
        Очікувана корисність
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Ймовірність повинна бути в діапазоні 0 до 1")
    
    if not 0.0 <= risk_aversion <= 1.0:
        raise ValueError("Схильність до ризику повинна бути в діапазоні 0 до 1")
    
    # Очікувана корисність
    expected_utility = probability * outcome_value
    
    # Коригування за схильністю до ризику
    # Якщо людина уникає ризику, очікувана корисність зменшується
    adjusted_utility = expected_utility * (1.0 - risk_aversion * 0.5)
    
    return adjusted_utility

def motivation_theory_model(expectancy: float, instrumentality: float, 
                          valence: float) -> float:
    """
    Модель мотивації Віктора Фрума (теорія очікування).
    
    Параметри:
        expectancy: Очікування (0-1, віра, що зусилля призведуть до результату)
        instrumentality: Інструментальність (0-1, віра, що результат призведе до винагороди)
        valence: Валентність (0-1, привабливість винагороди)
    
    Повертає:
        Мотиваційна сила
    """
    if not all(0.0 <= x <= 1.0 for x in [expectancy, instrumentality, valence]):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 1")
    
    # Формула Віктора Фрума: Мотивація = Очікування × Інструментальність × Валентність
    motivation = expectancy * instrumentality * valence
    
    return motivation

def maslow_hierarchy_needs(physiological: float, safety: float, 
                          love_belonging: float, esteem: float, 
                          self_actualization: float) -> Dict[str, Union[str, float]]:
    """
    Ієрархія потреб Маслоу.
    
    Параметри:
        physiological: Фізіологічні потреби (0-1)
        safety: Потреби в безпеці (0-1)
        love_belonging: Потреби в любові та належності (0-1)
        esteem: Потреби в повазі (0-1)
        self_actualization: Потреби в самореалізації (0-1)
    
    Повертає:
        Словник з домінуючою потребою
    """
    if not all(0.0 <= x <= 1.0 for x in [physiological, safety, love_belonging, esteem, self_actualization]):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 1")
    
    needs = {
        'physiological': physiological,
        'safety': safety,
        'love_belonging': love_belonging,
        'esteem': esteem,
        'self_actualization': self_actualization
    }
    
    # Знаходимо домінуючу потребу
    dominant_need = max(needs, key=needs.get)
    dominant_value = needs[dominant_need]
    
    # Класифікація рівня потреб
    if dominant_need == 'physiological':
        level = "Базовий рівень"
    elif dominant_need == 'safety':
        level = "Рівень безпеки"
    elif dominant_need == 'love_belonging':
        level = "Соціальний рівень"
    elif dominant_need == 'esteem':
        level = "Рівень поваги"
    else:
        level = "Рівень самореалізації"
    
    return {
        'dominant_need': dominant_need,
        'dominant_value': dominant_value,
        'level': level,
        'all_needs': needs
    }

def cognitive_bias_detection(decision_pattern: List[int]) -> Dict[str, float]:
    """
    Виявлення когнітивних упереджень у прийнятті рішень.
    
    Параметри:
        decision_pattern: Візерунок прийняття рішень (1 - вибір A, 0 - вибір B)
    
    Повертає:
        Словник з виявленими упередженнями
    """
    if not decision_pattern:
        return {
            'confirmation_bias': 0.0,
            'anchoring_bias': 0.0,
            'availability_bias': 0.0,
            'overconfidence_bias': 0.0
        }
    
    # Аналіз візерунка для виявлення упереджень
    
    # Упередження підтвердження (схильність вибирати те саме)
    if len(decision_pattern) > 1:
        same_choices = sum(1 for i in range(1, len(decision_pattern)) 
                          if decision_pattern[i] == decision_pattern[i-1])
        confirmation_bias = same_choices / (len(decision_pattern) - 1)
    else:
        confirmation_bias = 0.0
    
    # Ефект якорювання (якщо перший вибір впливає на наступні)
    if len(decision_pattern) > 2:
        first_choice = decision_pattern[0]
        influenced_choices = sum(1 for choice in decision_pattern[1:] if choice == first_choice)
        anchoring_bias = influenced_choices / (len(decision_pattern) - 1)
    else:
        anchoring_bias = 0.0
    
    # Ефект доступності (частота повторюваних виборів)
    choice_counts = Counter(decision_pattern)
    max_frequency = max(choice_counts.values()) if choice_counts else 0
    availability_bias = max_frequency / len(decision_pattern) if decision_pattern else 0.0
    
    # Упередження надмірної впевненості (відхилення від оптимального розподілу)
    expected_distribution = 0.5  # Очікуємо рівномірний розподіл
    actual_distribution = sum(decision_pattern) / len(decision_pattern) if decision_pattern else 0.0
    overconfidence_bias = abs(actual_distribution - expected_distribution)
    
    return {
        'confirmation_bias': confirmation_bias,
        'anchoring_bias': anchoring_bias,
        'availability_bias': availability_bias,
        'overconfidence_bias': overconfidence_bias
    }

def psychological_wellbeing_index(positive_affect: float, negative_affect: float, 
                                life_satisfaction: float, social_support: float,
                                autonomy: float, environmental_mastery: float,
                                personal_growth: float, purpose_in_life: float) -> Dict[str, float]:
    """
    Індекс психологічного добробуту (шкала Ріффа).
    
    Параметри:
        positive_affect: Позитивний афект (0-10)
        negative_affect: Негативний афект (0-10)
        life_satisfaction: Задоволеність життям (0-10)
        social_support: Соціальна підтримка (0-10)
        autonomy: Автономія (0-10)
        environmental_mastery: Опанування середовища (0-10)
        personal_growth: Особистісний ріст (0-10)
        purpose_in_life: Сенс життя (0-10)
    
    Повертає:
        Словник з індексом психологічного добробуту
    """
    parameters = [positive_affect, negative_affect, life_satisfaction, social_support,
                 autonomy, environmental_mastery, personal_growth, purpose_in_life]
    
    if not all(0.0 <= x <= 10.0 for x in parameters):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 10")
    
    # Нормалізуємо негативний афект (чим менше, тим краще)
    normalized_negative = 10.0 - negative_affect
    
    # Обчислюємо загальний індекс (середнє значення всіх компонентів)
    wellbeing_index = (positive_affect + normalized_negative + life_satisfaction + 
                      social_support + autonomy + environmental_mastery + 
                      personal_growth + purpose_in_life) / 8.0
    
    # Класифікація добробуту
    if wellbeing_index >= 8.0:
        wellbeing_category = "Високий"
    elif wellbeing_index >= 6.0:
        wellbeing_category = "Середній"
    elif wellbeing_index >= 4.0:
        wellbeing_category = "Нижче середнього"
    else:
        wellbeing_category = "Низький"
    
    return {
        'wellbeing_index': wellbeing_index,
        'wellbeing_category': wellbeing_category,
        'positive_affect': positive_affect,
        'negative_affect': negative_affect,
        'life_satisfaction': life_satisfaction,
        'social_support': social_support,
        'autonomy': autonomy,
        'environmental_mastery': environmental_mastery,
        'personal_growth': personal_growth,
        'purpose_in_life': purpose_in_life
    }

def behavioral_economics_model(choice_set: List[Tuple[float, float]], 
                              time_discount: float = 0.9) -> int:
    """
    Модель поведінкової економіки (вибір між варіантами з урахуванням дисконтування).
    
    Параметри:
        choice_set: Список варіантів (вигода, затримка)
        time_discount: Коефіцієнт дисконтування
    
    Повертає:
        Індекс найкращого варіанту
    """
    if not choice_set:
        raise ValueError("Набір варіантів не може бути порожнім")
    
    # Обчислюємо дисконтовану корисність для кожного варіанту
    discounted_utilities = []
    
    for benefit, delay in choice_set:
        # Формула: U = B / (1 + k)^t
        # де B - вигода, k - коефіцієнт дисконтування, t - затримка
        discounted_utility = benefit / ((1 + (1 - time_discount)) ** delay)
        discounted_utilities.append(discounted_utility)
    
    # Повертаємо індекс варіанту з найвищою дисконтованою корисністю
    best_choice_index = discounted_utilities.index(max(discounted_utilities))
    
    return best_choice_index

def social_influence_model(initial_opinion: float, social_pressure: float, 
                          conformity_tendency: float) -> float:
    """
    Модель соціального впливу.
    
    Параметри:
        initial_opinion: Початкова думка (-1 до 1)
        social_pressure: Соціальний тиск (0 до 1)
        conformity_tendency: Схильність до конформізму (0 до 1)
    
    Повертає:
        Кінцева думка після соціального впливу
    """
    if not -1.0 <= initial_opinion <= 1.0:
        raise ValueError("Початкова думка повинна бути в діапазоні -1 до 1")
    
    if not 0.0 <= social_pressure <= 1.0:
        raise ValueError("Соціальний тиск повинен бути в діапазоні 0 до 1")
    
    if not 0.0 <= conformity_tendency <= 1.0:
        raise ValueError("Схильність до конформізму повинна бути в діапазоні 0 до 1")
    
    # Модель: кінцева думка = початкова + (тиск * схильність * (0 - початкова))
    # Тобто, думка рухається до нейтральної точки (0) під впливом соціального тиску
    final_opinion = initial_opinion + (social_pressure * conformity_tendency * (0 - initial_opinion))
    
    return max(-1.0, min(1.0, final_opinion))

def cognitive_dissonance_model(expectation: float, reality: float, 
                              importance: float) -> Dict[str, float]:
    """
    Модель когнітивного дисонансу.
    
    Параметри:
        expectation: Очікування (0 до 1)
        reality: Реальність (0 до 1)
        importance: Важливість (0 до 1)
    
    Повертає:
        Словник з оцінкою дисонансу
    """
    if not all(0.0 <= x <= 1.0 for x in [expectation, reality, importance]):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 1")
    
    # Обчислюємо величину дисонансу
    dissonance_magnitude = abs(expectation - reality) * importance
    
    # Обчислюємо мотивацію до зменшення дисонансу
    # Чим більший дисонанс, тим більша мотивація його зменшити
    motivation_to_reduce = dissonance_magnitude
    
    # Стратегії зменшення дисонансу:
    # 1. Зміна очікувань
    change_expectations = dissonance_magnitude * 0.5
    # 2. Зміна сприйняття реальності
    change_reality_perception = dissonance_magnitude * 0.3
    # 3. Зменшення важливості
    reduce_importance = dissonance_magnitude * 0.2
    
    return {
        'dissonance_magnitude': dissonance_magnitude,
        'motivation_to_reduce': motivation_to_reduce,
        'change_expectations': change_expectations,
        'change_reality_perception': change_reality_perception,
        'reduce_importance': reduce_importance
    }

def psychological_resilience_score(stress_level: float, coping_strategies: float, 
                                 social_support: float, self_efficacy: float) -> float:
    """
    Обчислення психологічної стійкості.
    
    Параметри:
        stress_level: Рівень стресу (0 до 10)
        coping_strategies: Стратегії подолання (0 до 10)
        social_support: Соціальна підтримка (0 до 10)
        self_efficacy: Самоефективність (0 до 10)
    
    Повертає:
        Рівень психологічної стійкості (0 до 10)
    """
    if not all(0.0 <= x <= 10.0 for x in [stress_level, coping_strategies, social_support, self_efficacy]):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 10")
    
    # Нормалізуємо рівень стресу (чим менше стресу, тим краще)
    normalized_stress = 10.0 - stress_level
    
    # Обчислюємо загальний рівень стійкості
    resilience = (normalized_stress * 0.3 + 
                 coping_strategies * 0.3 + 
                 social_support * 0.2 + 
                 self_efficacy * 0.2)
    
    return max(0.0, min(10.0, resilience))

def group_dynamics_model(individual_opinions: List[float], 
                        leadership_influence: float,
                        communication_efficiency: float) -> Dict[str, Union[float, str]]:
    """
    Модель групової динаміки.
    
    Параметри:
        individual_opinions: Список індивідуальних думок (-1 до 1)
        leadership_influence: Вплив лідера (0 до 1)
        communication_efficiency: Ефективність комунікації (0 до 1)
    
    Повертає:
        Словник з результатами групової динаміки
    """
    if not individual_opinions:
        return {
            'group_opinion': 0.0,
            'consensus_level': 0.0,
            'group_cohesion': 0.0,
            'polarization': 'Немає'
        }
    
    if not all(-1.0 <= x <= 1.0 for x in individual_opinions):
        raise ValueError("Усі думки повинні бути в діапазоні -1 до 1")
    
    if not 0.0 <= leadership_influence <= 1.0:
        raise ValueError("Вплив лідера повинен бути в діапазоні 0 до 1")
    
    if not 0.0 <= communication_efficiency <= 1.0:
        raise ValueError("Ефективність комунікації повинна бути в діапазоні 0 до 1")
    
    # Початкова групова думка (середнє значення)
    initial_group_opinion = sum(individual_opinions) / len(individual_opinions)
    
    # Вплив лідера
    leader_opinion = 0.5  # Припускаємо, що лідер має позитивну думку
    influenced_opinions = [opinion + leadership_influence * (leader_opinion - opinion) 
                          for opinion in individual_opinions]
    
    # Фінальна групова думка після комунікації
    final_group_opinion = sum(influenced_opinions) / len(influenced_opinions)
    
    # Рівень консенсусу (обернений показник дисперсії)
    variance = sum((opinion - final_group_opinion) ** 2 for opinion in influenced_opinions) / len(influenced_opinions)
    consensus_level = max(0.0, 1.0 - variance)
    
    # Згуртованість групи
    group_cohesion = communication_efficiency * consensus_level
    
    # Поляризація
    if abs(final_group_opinion) > 0.5:
        if final_group_opinion > 0:
            polarization = "Позитивна"
        else:
            polarization = "Негативна"
    else:
        polarization = "Нейтральна"
    
    return {
        'group_opinion': final_group_opinion,
        'consensus_level': consensus_level,
        'group_cohesion': group_cohesion,
        'polarization': polarization
    }

def creativity_assessment(fluid_intelligence: float, divergent_thinking: float, 
                         domain_knowledge: float, intrinsic_motivation: float) -> float:
    """
    Оцінка креативності.
    
    Параметри:
        fluid_intelligence: Рідина інтелект (0 до 10)
        divergent_thinking: Дивергентне мислення (0 до 10)
        domain_knowledge: Предметні знання (0 до 10)
        intrinsic_motivation: Внутрішня мотивація (0 до 10)
    
    Повертає:
        Рівень креативності (0 до 10)
    """
    if not all(0.0 <= x <= 10.0 for x in [fluid_intelligence, divergent_thinking, domain_knowledge, intrinsic_motivation]):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 10")
    
    # Модель креативності Амабіле: креативність = інтелект * мислення * знання * мотивація
    # Але з урахуванням того, що занадто високий інтелект може обмежувати креативність,
    # використовуємо оптимальне значення для інтелекту (приблизно 7-8)
    optimal_intelligence = 7.5
    intelligence_factor = 1.0 - abs(fluid_intelligence - optimal_intelligence) / 10.0
    
    creativity = (intelligence_factor * divergent_thinking * 
                 domain_knowledge * intrinsic_motivation / 100.0 * 10.0)
    
    return max(0.0, min(10.0, creativity))

def memory_encoding_model(attention_level: float, emotional_intensity: float, 
                         repetition_count: int, semantic_processing: float) -> float:
    """
    Модель кодування пам'яті.
    
    Параметри:
        attention_level: Рівень уваги (0 до 1)
        emotional_intensity: Емоційна інтенсивність (0 до 1)
        repetition_count: Кількість повторень
        semantic_processing: Семантична обробка (0 до 1)
    
    Повертає:
        Ймовірність кодування в пам'ять (0 до 1)
    """
    if not 0.0 <= attention_level <= 1.0:
        raise ValueError("Рівень уваги повинен бути в діапазоні 0 до 1")
    
    if not 0.0 <= emotional_intensity <= 1.0:
        raise ValueError("Емоційна інтенсивність повинна бути в діапазоні 0 до 1")
    
    if repetition_count < 0:
        raise ValueError("Кількість повторень не може бути від'ємною")
    
    if not 0.0 <= semantic_processing <= 1.0:
        raise ValueError("Семантична обробка повинна бути в діапазоні 0 до 1")
    
    # Модель кодування пам'яті
    # Увага має найбільший вплив
    attention_factor = attention_level
    
    # Емоційна інтенсивність (ефект Зейгарник)
    emotional_factor = 1.0 - math.exp(-3.0 * emotional_intensity)
    
    # Ефект повторення (логарифмічний зростання)
    repetition_factor = min(1.0, math.log(repetition_count + 1))
    
    # Семантична обробка
    semantic_factor = semantic_processing
    
    # Комбінована ймовірність кодування
    encoding_probability = (attention_factor * 0.4 + 
                          emotional_factor * 0.3 + 
                          repetition_factor * 0.2 + 
                          semantic_factor * 0.1)
    
    return max(0.0, min(1.0, encoding_probability))

def psychological_flow_state(challenge_level: float, skill_level: float, 
                           clear_goals: float, feedback_clarity: float) -> Dict[str, Union[float, str]]:
    """
    Модель стану потоку (Міхай Чіксентміхайї).
    
    Параметри:
        challenge_level: Рівень виклику (0 до 10)
        skill_level: Рівень навичок (0 до 10)
        clear_goals: Чіткість цілей (0 до 10)
        feedback_clarity: Чіткість зворотного зв'язку (0 до 10)
    
    Повертає:
        Словник з оцінкою стану потоку
    """
    if not all(0.0 <= x <= 10.0 for x in [challenge_level, skill_level, clear_goals, feedback_clarity]):
        raise ValueError("Усі параметри повинні бути в діапазоні 0 до 10")
    
    # Баланс виклику та навичок
    balance = abs(challenge_level - skill_level)
    challenge_skill_balance = max(0.0, 1.0 - balance / 10.0)
    
    # Оптимальний стан потоку
    flow_probability = (challenge_skill_balance * 0.5 + 
                       clear_goals / 10.0 * 0.25 + 
                       feedback_clarity / 10.0 * 0.25)
    
    # Класифікація стану
    if flow_probability > 0.8:
        state = "Потік"
    elif flow_probability > 0.6:
        state = "Контроль"
    elif flow_probability > 0.4:
        state = "Розсіяна увага"
    elif flow_probability > 0.2:
        state = "Тривога"
    else:
        state = "Апатія"
    
    return {
        'flow_probability': flow_probability,
        'state': state,
        'challenge_skill_balance': challenge_skill_balance,
        'clear_goals': clear_goals / 10.0,
        'feedback_clarity': feedback_clarity / 10.0
    }

def attribution_theory_analysis(behavior: str, consensus: float, 
                              distinctiveness: float, consistency: float) -> Dict[str, Union[str, float]]:
    """
    Аналіз атрибуції (теорія відповідальності Келлі).
    
    Параметри:
        behavior: Поведінка (назва)
        consensus: Консенсус (0 до 1, чи інші також так поводяться)
        distinctiveness: Відмінність (0 до 1, чи особа поводиться так у всіх ситуаціях)
        consistency: Послідовність (0 до 1, чи особа завжди так поводиться)
    
    Повертає:
        Словник з типом атрибуції
    """
    if not 0.0 <= consensus <= 1.0:
        raise ValueError("Консенсус повинен бути в діапазоні 0 до 1")
    
    if not 0.0 <= distinctiveness <= 1.0:
        raise ValueError("Відмінність повинна бути в діапазоні 0 до 1")
    
    if not 0.0 <= consistency <= 1.0:
        raise ValueError("Послідовність повинна бути в діапазоні 0 до 1")
    
    # Внутрішня атрибуція (особистісна причина)
    internal_attribution = (1.0 - consensus) * (1.0 - distinctiveness) * consistency
    
    # Зовнішня атрибуція (ситуаційна причина)
    external_attribution = consensus * distinctiveness * (1.0 - consistency)
    
    # Визначення типу атрибуції
    if internal_attribution > external_attribution and internal_attribution > 0.5:
        attribution_type = "Внутрішня"
        explanation = f"Поведінка '{behavior}' приписується особистим рисам особи"
    elif external_attribution > internal_attribution and external_attribution > 0.5:
        attribution_type = "Зовнішня"
        explanation = f"Поведінка '{behavior}' приписується ситуаційним факторам"
    else:
        attribution_type = "Неоднозначна"
        explanation = f"Неможливо чітко визначити причину поведінки '{behavior}'"
    
    return {
        'attribution_type': attribution_type,
        'internal_attribution': internal_attribution,
        'external_attribution': external_attribution,
        'explanation': explanation
    }

def psychological_traits_correlation(trait1_scores: List[float], 
                                   trait2_scores: List[float]) -> float:
    """
    Обчислення кореляції між двома психологічними рисами.
    
    Параметри:
        trait1_scores: Оцінки першої риси
        trait2_scores: Оцінки другої риси
    
    Повертає:
        Коефіцієнт кореляції Пірсона
    """
    if len(trait1_scores) != len(trait2_scores):
        raise ValueError("Списки оцінок повинні мати однакову довжину")
    
    if len(trait1_scores) < 2:
        raise ValueError("Потрібно щонайменше дві пари оцінок для обчислення кореляції")
    
    n = len(trait1_scores)
    
    # Обчислюємо середні значення
    mean1 = sum(trait1_scores) / n
    mean2 = sum(trait2_scores) / n
    
    # Обчислюємо чисельник і знаменник для коефіцієнта кореляції
    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(trait1_scores, trait2_scores))
    
    sum_sq1 = sum((x - mean1) ** 2 for x in trait1_scores)
    sum_sq2 = sum((y - mean2) ** 2 for y in trait2_scores)
    
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    return correlation

def psychological_profile_matching(user_profile: Dict[str, float], 
                                 reference_profiles: List[Dict[str, float]]) -> int:
    """
    Визначення найбільш відповідного психологічного профілю.
    
    Параметри:
        user_profile: Профіль користувача
        reference_profiles: Список еталонних профілів
    
    Повертає:
        Індекс найбільш відповідного профілю
    """
    if not user_profile:
        raise ValueError("Профіль користувача не може бути порожнім")
    
    if not reference_profiles:
        raise ValueError("Список еталонних профілів не може бути порожнім")
    
    # Обчислюємо подібність між профілем користувача та кожним еталонним профілем
    similarities = []
    
    for ref_profile in reference_profiles:
        # Знаходимо спільні ключі
        common_keys = set(user_profile.keys()) & set(ref_profile.keys())
        
        if not common_keys:
            similarities.append(0.0)
            continue
        
        # Обчислюємо косинусну подібність
        dot_product = sum(user_profile[key] * ref_profile[key] for key in common_keys)
        
        user_magnitude = math.sqrt(sum(user_profile[key] ** 2 for key in common_keys))
        ref_magnitude = math.sqrt(sum(ref_profile[key] ** 2 for key in common_keys))
        
        if user_magnitude == 0 or ref_magnitude == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (user_magnitude * ref_magnitude)
        
        similarities.append(similarity)
    
    # Повертаємо індекс профілю з найвищою подібністю
    best_match_index = similarities.index(max(similarities))
    
    return best_match_index

def psychological_intervention_effect(pre_test_scores: List[float], 
                                   post_test_scores: List[float]) -> Dict[str, float]:
    """
    Оцінка ефективності психологічного втручання.
    
    Параметри:
        pre_test_scores: Результати до втручання
        post_test_scores: Результати після втручання
    
    Повертає:
        Словник з оцінкою ефективності
    """
    if len(pre_test_scores) != len(post_test_scores):
        raise ValueError("Списки результатів повинні мати однакову довжину")
    
    if not pre_test_scores:
        raise ValueError("Списки результатів не можуть бути порожніми")
    
    n = len(pre_test_scores)
    
    # Обчислюємо зміну для кожного учасника
    changes = [post - pre for post, pre in zip(post_test_scores, pre_test_scores)]
    
    # Середня зміна
    mean_change = sum(changes) / n
    
    # Стандартне відхилення змін
    if n > 1:
        variance = sum((change - mean_change) ** 2 for change in changes) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0
    
    # Коефіцієнт ефективності (розмір ефекту Коена)
    pre_mean = sum(pre_test_scores) / n
    post_mean = sum(post_test_scores) / n
    
    pooled_std = math.sqrt(
        (sum((x - pre_mean) ** 2 for x in pre_test_scores) + 
         sum((x - post_mean) ** 2 for x in post_test_scores)) / (2 * n - 2)
    ) if n > 1 else 1.0
    
    effect_size = (post_mean - pre_mean) / pooled_std if pooled_std != 0 else 0.0
    
    # Класифікація розміру ефекту
    if abs(effect_size) < 0.2:
        effect_interpretation = "Малий"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "Середній"
    else:
        effect_interpretation = "Великий"
    
    return {
        'mean_change': mean_change,
        'std_deviation': std_dev,
        'effect_size': effect_size,
        'effect_interpretation': effect_interpretation,
        'improvement_percentage': (sum(1 for change in changes if change > 0) / n) * 100
    }