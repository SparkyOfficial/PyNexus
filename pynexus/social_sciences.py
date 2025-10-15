"""
Модуль для обчислювальних соціальних наук в PyNexus.
Містить функції для аналізу соціальних мереж, демографії, економіки, політології та інших соціальних наук.
"""

import math
import random
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from collections import Counter, defaultdict
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# Константи для соціальних наук
WORLD_POPULATION_2023 = 8039836000  # Світова популяція у 2023 році
AVERAGE_LIFE_EXPECTANCY = 72.6  # Середня тривалість життя у світі (роки)
GDP_PER_CAPITA_WORLD = 12000  # Середній ВВП на душу населення у світі (USD)
GINI_COEFFICIENT_MAX = 1.0  # Максимальне значення коефіцієнта Джині
HUMAN_DEVELOPMENT_INDEX_MAX = 1.0  # Максимальне значення ІЛР
SOCIAL_MOBILITY_MIN = 0.0  # Мінімальна соціальна мобільність
SOCIAL_MOBILITY_MAX = 1.0  # Максимальна соціальна мобільність

def gini_coefficient(income_distribution: List[float]) -> float:
    """
    Обчислити коефіцієнт Джині для розподілу доходів.
    
    Параметри:
        income_distribution: Список доходів індивідів
    
    Повертає:
        Коефіцієнт Джині (від 0 до 1, де 0 - повна рівність, 1 - повна нерівність)
    """
    if not income_distribution:
        raise ValueError("Розподіл доходів не може бути порожнім")
    
    # Сортуємо доходи за зростанням
    sorted_incomes = sorted(income_distribution)
    n = len(sorted_incomes)
    
    # Обчислюємо кумулятивні частки населення
    cumulative_population = [i / n for i in range(1, n + 1)]
    
    # Обчислюємо кумулятивні частки доходів
    total_income = sum(sorted_incomes)
    if total_income == 0:
        return 0.0
    
    cumulative_income = []
    running_sum = 0
    for income in sorted_incomes:
        running_sum += income
        cumulative_income.append(running_sum / total_income)
    
    # Обчислюємо площу між лінією рівності та кривою Лоренца
    area_under_lorenz = 0
    for i in range(n):
        if i == 0:
            area_under_lorenz += cumulative_income[i] * cumulative_population[i] / 2
        else:
            # Площа трапеції
            area_under_lorenz += (cumulative_income[i] + cumulative_income[i-1]) * \
                               (cumulative_population[i] - cumulative_population[i-1]) / 2
    
    # Коефіцієнт Джині = 2 * (0.5 - площа під кривою Лоренца)
    gini = 1 - 2 * area_under_lorenz
    
    return max(0.0, min(1.0, gini))

def social_mobility_index(origin_distribution: List[float], 
                         destination_distribution: List[float]) -> float:
    """
    Обчислити індекс соціальної мобільності.
    
    Параметри:
        origin_distribution: Початковий розподіл соціального статусу
        destination_distribution: Кінцевий розподіл соціального статусу
    
    Повертає:
        Індекс соціальної мобільності (від 0 до 1)
    """
    if len(origin_distribution) != len(destination_distribution):
        raise ValueError("Розподіли повинні мати однакову довжину")
    
    if not origin_distribution:
        return 0.0
    
    n = len(origin_distribution)
    
    # Обчислюємо кореляцію між початковим та кінцевим статусом
    correlation = np.corrcoef(origin_distribution, destination_distribution)[0, 1]
    
    # Індекс мобільності = 1 - |кореляція|
    mobility_index = 1 - abs(correlation)
    
    return max(0.0, min(1.0, mobility_index))

def demographic_transition_model(population_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Проаналізувати демографічний перехід для країни.
    
    Параметри:
        population_data: Словник з демографічними даними
                        {'birth_rate': народжуваність, 'death_rate': смертність,
                         'migration_rate': міграція, 'population': населення}
    
    Повертає:
        Словник з аналізом демографічного перехіду
    """
    birth_rate = population_data.get('birth_rate', 0)
    death_rate = population_data.get('death_rate', 0)
    migration_rate = population_data.get('migration_rate', 0)
    population = population_data.get('population', 0)
    
    # Природний приріст
    natural_increase = birth_rate - death_rate
    
    # Загальний приріст
    total_increase = natural_increase + migration_rate
    
    # Фаза демографічного перехіду
    if birth_rate > 30 and death_rate > 15:
        phase = "Перша фаза (висока смертність та народжуваність)"
    elif birth_rate > 20 and death_rate < 15:
        phase = "Друга фаза (висока народжуваність, низька смертність)"
    elif birth_rate < 20 and death_rate < 10:
        phase = "Третя фаза (низька народжуваність та смертність)"
    else:
        phase = "Четверта фаза (дуже низька народжуваність)"
    
    return {
        'natural_increase': natural_increase,
        'total_increase': total_increase,
        'phase': phase,
        'growth_rate': total_increase / 10  # Відсоток приросту
    }

def social_network_analysis(adjacency_matrix: List[List[int]]) -> Dict[str, Union[float, int, List[int]]]:
    """
    Проаналізувати соціальну мережу.
    
    Параметри:
        adjacency_matrix: Матриця суміжності мережі
    
    Повертає:
        Словник з характеристиками мережі
    """
    n = len(adjacency_matrix)
    if n == 0:
        return {
            'nodes': 0,
            'edges': 0,
            'density': 0.0,
            'average_degree': 0.0,
            'clustering_coefficient': 0.0,
            'central_nodes': []
        }
    
    # Перевіряємо, чи матриця квадратна
    if any(len(row) != n for row in adjacency_matrix):
        raise ValueError("Матриця суміжності повинна бути квадратною")
    
    # Кількість ребер
    edges = sum(sum(row) for row in adjacency_matrix) // 2
    
    # Густина мережі
    max_edges = n * (n - 1) // 2 if n > 1 else 0
    density = edges / max_edges if max_edges > 0 else 0
    
    # Ступені вузлів
    degrees = [sum(row) for row in adjacency_matrix]
    average_degree = sum(degrees) / n if n > 0 else 0
    
    # Коефіцієнт кластеризації
    clustering_sum = 0
    for i in range(n):
        neighbors = [j for j in range(n) if adjacency_matrix[i][j] == 1]
        k = len(neighbors)
        if k < 2:
            continue
        
        # Підраховуємо кількість зв'язків між сусідами
        actual_edges = 0
        for u in neighbors:
            for v in neighbors:
                if u != v and adjacency_matrix[u][v] == 1:
                    actual_edges += 1
        actual_edges //= 2  # Кожен зв'язок враховано двічі
        
        # Максимальна кількість зв'язків між сусідами
        max_possible = k * (k - 1) // 2
        if max_possible > 0:
            clustering_sum += actual_edges / max_possible
    
    clustering_coefficient = clustering_sum / n if n > 0 else 0
    
    # Найцентральніші вузли (за ступенем)
    central_nodes = sorted(range(n), key=lambda i: degrees[i], reverse=True)[:min(5, n)]
    
    return {
        'nodes': n,
        'edges': edges,
        'density': density,
        'average_degree': average_degree,
        'clustering_coefficient': clustering_coefficient,
        'central_nodes': central_nodes
    }

def voting_system_analysis(votes: Dict[str, int], 
                          system: str = 'plurality') -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Проаналізувати результати голосування.
    
    Параметри:
        votes: Словник {кандидат: кількість голосів}
        system: Система голосування ('plurality', 'runoff', 'borda'), за замовчуванням 'plurality'
    
    Повертає:
        Словник з результатами аналізу
    """
    if not votes:
        return {'winner': 'Немає голосів', 'analysis': {}}
    
    total_votes = sum(votes.values())
    if total_votes == 0:
        return {'winner': 'Немає голосів', 'analysis': {}}
    
    # Аналіз залежить від системи
    if system == 'plurality':
        # Просте більшість
        winner = max(votes.items(), key=lambda x: x[1])[0]
        analysis = {candidate: votes[candidate] / total_votes 
                   for candidate in votes}
        
    elif system == 'runoff':
        # Двоетапне голосування
        sorted_candidates = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_candidates) < 2:
            winner = sorted_candidates[0][0] if sorted_candidates else 'Немає'
            analysis = {candidate: votes[candidate] / total_votes 
                       for candidate in votes}
        else:
            # Якщо ніхто не має більше 50%, проводимо другий тур
            first_place, first_votes = sorted_candidates[0]
            second_place, second_votes = sorted_candidates[1]
            
            if first_votes / total_votes > 0.5:
                winner = first_place
            else:
                winner = first_place if first_votes > second_votes else second_place
            
            analysis = {
                'first_round': {candidate: votes[candidate] / total_votes 
                              for candidate in votes},
                'runoff_candidates': [first_place, second_place]
            }
    
    elif system == 'borda':
        # Система Борда
        candidates = list(votes.keys())
        n_candidates = len(candidates)
        
        # Призначаємо бали (n-1 для першого місця, n-2 для другого і т.д.)
        borda_scores = {}
        for candidate in candidates:
            # У спрощеній версії використовуємо кількість голосів як ранг
            borda_scores[candidate] = votes[candidate]
        
        winner = max(borda_scores.items(), key=lambda x: x[1])[0]
        analysis = {candidate: borda_scores[candidate] / sum(borda_scores.values()) 
                   for candidate in candidates}
    
    else:
        winner = 'Невідома система'
        analysis = {}
    
    return {
        'winner': winner,
        'total_votes': total_votes,
        'analysis': analysis
    }

def public_opinion_poll(questions: List[str], 
                       sample_size: int, 
                       population_size: int = WORLD_POPULATION_2023) -> Dict[str, Dict[str, float]]:
    """
    Провести опитування громадської думки.
    
    Параметри:
        questions: Список питань
        sample_size: Розмір вибірки
        population_size: Розмір популяції, за замовчуванням світова популяція
    
    Повертає:
        Словник з результатами опитування
    """
    if sample_size <= 0:
        raise ValueError("Розмір вибірки повинен бути додатнім")
    
    if population_size <= 0:
        raise ValueError("Розмір популяції повинен бути додатнім")
    
    results = {}
    
    for question in questions:
        # Генеруємо випадкові результати опитування
        # Для спрощення припускаємо бінарні відповіді (так/ні)
        yes_votes = random.randint(0, sample_size)
        no_votes = sample_size - yes_votes
        
        # Обчислюємо відсотки
        yes_percentage = yes_votes / sample_size * 100
        no_percentage = no_votes / sample_size * 100
        
        # Обчислюємо похибку вибірки (при 95% довірчому інтервалі)
        # Формула: 1.96 * sqrt(p*(1-p)/n)
        p = yes_percentage / 100
        margin_of_error = 1.96 * math.sqrt(p * (1 - p) / sample_size) * 100
        
        # Коригуємо на скінченну популяцію
        finite_population_correction = math.sqrt((population_size - sample_size) / (population_size - 1))
        adjusted_margin_of_error = margin_of_error * finite_population_correction
        
        results[question] = {
            'yes': yes_percentage,
            'no': no_percentage,
            'margin_of_error': adjusted_margin_of_error,
            'confidence_interval': [
                max(0, yes_percentage - adjusted_margin_of_error),
                min(100, yes_percentage + adjusted_margin_of_error)
            ]
        }
    
    return results

def social_capital_index(trust_data: List[float], 
                        participation_data: List[float], 
                        networks_data: List[float]) -> float:
    """
    Обчислити індекс соціального капіталу.
    
    Параметри:
        trust_data: Дані про рівень довіри (від 0 до 1)
        participation_data: Дані про рівень участі (від 0 до 1)
        networks_data: Дані про соціальні мережі (від 0 до 1)
    
    Повертає:
        Індекс соціального капіталу (від 0 до 1)
    """
    if not trust_data or not participation_data or not networks_data:
        return 0.0
    
    # Обчислюємо середні значення для кожної компоненти
    avg_trust = sum(trust_data) / len(trust_data)
    avg_participation = sum(participation_data) / len(participation_data)
    avg_networks = sum(networks_data) / len(networks_data)
    
    # Індекс соціального капіталу як середнє арифметичне
    social_capital = (avg_trust + avg_participation + avg_networks) / 3
    
    return max(0.0, min(1.0, social_capital))

def inequality_decomposition(group_data: List[Dict[str, List[float]]]) -> Dict[str, float]:
    """
    Декомпозувати нерівність за групами.
    
    Параметри:
        group_data: Список словників {група: [доходи]}
    
    Повертає:
        Словник з декомпозицією нерівності
    """
    if not group_data:
        return {
            'overall_inequality': 0.0,
            'between_group_inequality': 0.0,
            'within_group_inequality': 0.0
        }
    
    # Збираємо всі доходи
    all_incomes = []
    group_incomes = {}
    
    for group_dict in group_data:
        for group, incomes in group_dict.items():
            all_incomes.extend(incomes)
            if group not in group_incomes:
                group_incomes[group] = []
            group_incomes[group].extend(incomes)
    
    if not all_incomes:
        return {
            'overall_inequality': 0.0,
            'between_group_inequality': 0.0,
            'within_group_inequality': 0.0
        }
    
    # Загальна нерівність
    overall_gini = gini_coefficient(all_incomes)
    
    # Міжгрупова нерівність
    group_means = {group: sum(incomes) / len(incomes) 
                   for group, incomes in group_incomes.items() if incomes}
    
    # Середній дохід по всій популяції
    total_mean = sum(all_incomes) / len(all_incomes)
    
    # Ваги груп (частка населення)
    total_population = len(all_incomes)
    group_weights = {group: len(incomes) / total_population 
                     for group, incomes in group_incomes.items() if incomes}
    
    # Міжгрупова нерівність (відхилення групових середніх від загального середнього)
    between_inequality = sum(weight * abs(mean - total_mean) 
                           for group, (weight, mean) in 
                           zip(group_weights.keys(), 
                               zip(group_weights.values(), group_means.values())))
    
    # Нормалізуємо міжгрупову нерівність
    max_possible_between = max(group_means.values()) - min(group_means.values()) if group_means else 0
    normalized_between = between_inequality / max_possible_between if max_possible_between > 0 else 0
    
    # Внутрішньогрупова нерівність
    within_inequality = sum(weight * gini_coefficient(group_incomes[group]) 
                          for group, weight in group_weights.items() 
                          if group in group_incomes and group_incomes[group])
    
    return {
        'overall_inequality': overall_gini,
        'between_group_inequality': normalized_between,
        'within_group_inequality': within_inequality
    }

def social_cohesion_index(interaction_frequency: List[float], 
                         shared_values: List[float], 
                         collective_action: List[float]) -> float:
    """
    Обчислити індекс соціальної згуртованості.
    
    Параметри:
        interaction_frequency: Частота соціальних взаємодій (від 0 до 1)
        shared_values: Ступінь спільних цінностей (від 0 до 1)
        collective_action: Рівень колективної дії (від 0 до 1)
    
    Повертає:
        Індекс соціальної згуртованості (від 0 до 1)
    """
    if not interaction_frequency or not shared_values or not collective_action:
        return 0.0
    
    # Обчислюємо середні значення
    avg_interaction = sum(interaction_frequency) / len(interaction_frequency)
    avg_shared_values = sum(shared_values) / len(shared_values)
    avg_collective_action = sum(collective_action) / len(collective_action)
    
    # Індекс згуртованості як середнє геометричне
    cohesion_index = (avg_interaction * avg_shared_values * avg_collective_action) ** (1/3)
    
    return max(0.0, min(1.0, cohesion_index))

def urbanization_analysis(population_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Проаналізувати процес урбанізації.
    
    Параметри:
        population_data: Словник {рік: {urban_population: міське населення, 
                                       total_population: загальне населення}}
    
    Повертає:
        Словник з аналізом урбанізації
    """
    if not population_data:
        return {
            'current_urbanization_rate': 0.0,
            'urbanization_trend': 0.0,
            'classification': 'Немає даних'
        }
    
    # Сортуємо роки
    years = sorted(population_data.keys())
    
    if not years:
        return {
            'current_urbanization_rate': 0.0,
            'urbanization_trend': 0.0,
            'classification': 'Немає даних'
        }
    
    # Поточний рівень урбанізації
    latest_year = years[-1]
    latest_data = population_data[latest_year]
    urban_pop = latest_data.get('urban_population', 0)
    total_pop = latest_data.get('total_population', 1)
    
    current_rate = urban_pop / total_pop if total_pop > 0 else 0
    
    # Тренд урбанізації
    if len(years) >= 2:
        first_year = years[0]
        first_data = population_data[first_year]
        first_rate = first_data.get('urban_population', 0) / first_data.get('total_population', 1) \
                    if first_data.get('total_population', 1) > 0 else 0
        
        trend = (current_rate - first_rate) / (latest_year - first_year) \
                if latest_year != first_year else 0
    else:
        trend = 0.0
    
    # Класифікація рівня урбанізації
    if current_rate < 0.3:
        classification = "Низький рівень урбанізації"
    elif current_rate < 0.6:
        classification = "Середній рівень урбанізації"
    elif current_rate < 0.8:
        classification = "Високий рівень урбанізації"
    else:
        classification = "Дуже високий рівень урбанізації"
    
    return {
        'current_urbanization_rate': current_rate * 100,  # У відсотках
        'urbanization_trend': trend * 100,  # Зміна на рік у відсотках
        'classification': classification
    }

def social_progress_index(basic_human_needs: float, 
                         foundations_of_wellbeing: float, 
                         opportunity: float) -> float:
    """
    Обчислити індекс соціального прогресу.
    
    Параметри:
        basic_human_needs: Основні людські потреби (від 0 до 100)
        foundations_of_wellbeing: Основи добробуту (від 0 до 100)
        opportunity: Можливості (від 0 до 100)
    
    Повертає:
        Індекс соціального прогресу (від 0 до 100)
    """
    # Нормалізуємо вхідні значення
    basic_human_needs = max(0, min(100, basic_human_needs))
    foundations_of_wellbeing = max(0, min(100, foundations_of_wellbeing))
    opportunity = max(0, min(100, opportunity))
    
    # Індекс соціального прогресу як середнє арифметичне
    spi = (basic_human_needs + foundations_of_wellbeing + opportunity) / 3
    
    return spi

def political_polarization_index(opinions: List[float]) -> float:
    """
    Обчислити індекс політичної поляризації.
    
    Параметри:
        opinions: Список політичних поглядів (від 0 до 1, де 0 - ліві, 1 - праві)
    
    Повертає:
        Індекс поляризації (від 0 до 1)
    """
    if not opinions:
        return 0.0
    
    # Обчислюємо середнє значення
    mean_opinion = sum(opinions) / len(opinions)
    
    # Обчислюємо стандартне відхилення
    variance = sum((opinion - mean_opinion) ** 2 for opinion in opinions) / len(opinions)
    std_dev = math.sqrt(variance)
    
    # Нормалізуємо стандартне відхилення до індексу поляризації
    # Використовуємо максимальне можливе стандартне відхилення (0.5 для рівномірного розподілу)
    max_std_dev = 0.5
    polarization_index = std_dev / max_std_dev if max_std_dev > 0 else 0
    
    return max(0.0, min(1.0, polarization_index))

def social_welfare_function(individual_utilities: List[float], 
                           weights: Optional[List[float]] = None) -> float:
    """
    Обчислити функцію соціального добробуту.
    
    Параметри:
        individual_utilities: Список корисностей індивідів
        weights: Ваги для кожного індивіда (за замовчуванням рівні)
    
    Повертає:
        Значення функції соціального добробуту
    """
    if not individual_utilities:
        return 0.0
    
    n = len(individual_utilities)
    
    if weights is None:
        weights = [1.0 / n] * n
    elif len(weights) != n:
        raise ValueError("Кількість ваг повинна відповідати кількості індивідів")
    
    # Нормалізуємо ваги
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0 / n] * n
    else:
        weights = [w / total_weight for w in weights]
    
    # Обчислюємо зважену суму корисностей
    social_welfare = sum(weight * utility for weight, utility in zip(weights, individual_utilities))
    
    return social_welfare

def collective_action_model(participants: int, 
                           benefits: float, 
                           costs: float, 
                           probability_success: float) -> Dict[str, float]:
    """
    Проаналізувати модель колективної дії.
    
    Параметри:
        participants: Кількість учасників
        benefits: Загальні вигоди від колективної дії
        costs: Індивідуальні витрати
        probability_success: Ймовірність успіху
    
    Повертає:
        Словник з аналізом моделі
    """
    if participants <= 0:
        return {
            'expected_benefit': 0.0,
            'expected_cost': 0.0,
            'net_benefit': 0.0,
            'participation_incentive': 0.0
        }
    
    # Очікувані вигоди для кожного учасника
    expected_benefit = (benefits / participants) * probability_success
    
    # Очікувані витрати
    expected_cost = costs
    
    # Чиста вигода
    net_benefit = expected_benefit - expected_cost
    
    # Стимул до участі
    participation_incentive = 1.0 if net_benefit > 0 else 0.0
    
    return {
        'expected_benefit': expected_benefit,
        'expected_cost': expected_cost,
        'net_benefit': net_benefit,
        'participation_incentive': participation_incentive
    }

def social_learning_model(initial_knowledge: List[float], 
                         learning_rate: float, 
                         social_influence: float, 
                         iterations: int) -> List[float]:
    """
    Симулювати процес соціального навчання.
    
    Параметри:
        initial_knowledge: Початкові рівні знань індивідів
        learning_rate: Швидкість індивідуального навчання
        social_influence: Сила соціального впливу
        iterations: Кількість ітерацій
    
    Повертає:
        Список рівнів знань після кожної ітерації
    """
    if not initial_knowledge or iterations <= 0:
        return initial_knowledge
    
    if learning_rate < 0 or learning_rate > 1:
        raise ValueError("Швидкість навчання повинна бути між 0 та 1")
    
    if social_influence < 0 or social_influence > 1:
        raise ValueError("Сила соціального впливу повинна бути між 0 та 1")
    
    knowledge_levels = initial_knowledge[:]
    history = [knowledge_levels[:]]
    
    n_individuals = len(knowledge_levels)
    
    for _ in range(iterations):
        new_knowledge = []
        
        for i in range(n_individuals):
            # Індивідуальне навчання
            individual_learning = knowledge_levels[i] + learning_rate * (1 - knowledge_levels[i])
            
            # Соціальне навчання (середнє знання інших)
            if n_individuals > 1:
                others_knowledge = sum(knowledge_levels[j] for j in range(n_individuals) if j != i) / (n_individuals - 1)
                social_learning = knowledge_levels[i] + social_influence * (others_knowledge - knowledge_levels[i])
            else:
                social_learning = knowledge_levels[i]
            
            # Комбінуємо обидва типи навчання
            new_level = (1 - social_influence) * individual_learning + social_influence * social_learning
            new_knowledge.append(max(0.0, min(1.0, new_level)))
        
        knowledge_levels = new_knowledge
        history.append(knowledge_levels[:])
    
    return history

def institutional_quality_index(rule_of_law: float, 
                               government_effectiveness: float, 
                               regulatory_quality: float, 
                               voice_and_accountability: float) -> float:
    """
    Обчислити індекс якості інститутів.
    
    Параметри:
        rule_of_law: Верховенство права (від 0 до 1)
        government_effectiveness: Ефективність уряду (від 0 до 1)
        regulatory_quality: Якість регулювання (від 0 до 1)
        voice_and_accountability: Голос та підзвітність (від 0 до 1)
    
    Повертає:
        Індекс якості інститутів (від 0 до 1)
    """
    # Нормалізуємо вхідні значення
    rule_of_law = max(0.0, min(1.0, rule_of_law))
    government_effectiveness = max(0.0, min(1.0, government_effectiveness))
    regulatory_quality = max(0.0, min(1.0, regulatory_quality))
    voice_and_accountability = max(0.0, min(1.0, voice_and_accountability))
    
    # Індекс якості інститутів як середнє арифметичне
    quality_index = (rule_of_law + government_effectiveness + regulatory_quality + voice_and_accountability) / 4
    
    return quality_index

def cultural_diversity_index(categories: List[int], 
                            population: int) -> float:
    """
    Обчислити індекс культурного різноманіття.
    
    Параметри:
        categories: Список кількостей людей у кожній культурній категорії
        population: Загальна кількість населення
    
    Повертає:
        Індекс культурного різноманіття (від 0 до 1)
    """
    if population <= 0:
        return 0.0
    
    if not categories:
        return 0.0
    
    # Обчислюємо частки кожної категорії
    proportions = [count / population for count in categories if count > 0]
    
    # Індекс різноманіття Шеннона
    diversity_index = -sum(p * math.log(p) for p in proportions if p > 0)
    
    # Нормалізуємо до діапазону [0, 1]
    max_diversity = math.log(len(proportions)) if proportions else 0
    normalized_diversity = diversity_index / max_diversity if max_diversity > 0 else 0
    
    return normalized_diversity

def social_network_growth_model(initial_nodes: int, 
                               growth_rate: float, 
                               preferential_attachment: bool = True,
                               time_steps: int = 100) -> Dict[str, Union[int, float, List[int]]]:
    """
    Симулювати зростання соціальної мережі.
    
    Параметри:
        initial_nodes: Початкова кількість вузлів
        growth_rate: Швидкість зростання (нові вузли на крок)
        preferential_attachment: Чи використовувати преференційне приєднання
        time_steps: Кількість часових кроків
    
    Повертає:
        Словник з характеристиками зростання мережі
    """
    if initial_nodes <= 0 or time_steps <= 0:
        return {
            'final_nodes': 0,
            'final_edges': 0,
            'average_degree': 0.0,
            'network_growth': []
        }
    
    # Початкова мережа - повністю зв'язний граф
    nodes = list(range(initial_nodes))
    edges = []
    
    # Створюємо зв'язки між усіма початковими вузлами
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            edges.append((i, j))
    
    # Історія зростання
    growth_history = [len(nodes)]
    
    # Ступені вузлів для преференційного приєднання
    node_degrees = defaultdict(int)
    for edge in edges:
        node_degrees[edge[0]] += 1
        node_degrees[edge[1]] += 1
    
    # Симуляція зростання
    for step in range(time_steps):
        # Додаємо нові вузли
        new_nodes = max(1, int(growth_rate))
        for _ in range(new_nodes):
            new_node_id = len(nodes)
            nodes.append(new_node_id)
            node_degrees[new_node_id] = 0
            
            # Додаємо зв'язки для нового вузла
            if preferential_attachment and len(nodes) > 1:
                # Преференційне приєднання - ймовірність приєднання пропорційна ступеню
                total_degree = sum(node_degrees.values())
                if total_degree > 0:
                    # Приєднуємо до одного існуючого вузла
                    probabilities = [node_degrees[node] / total_degree for node in nodes[:-1]]
                    chosen_node = random.choices(nodes[:-1], weights=probabilities)[0]
                    edges.append((new_node_id, chosen_node))
                    node_degrees[new_node_id] += 1
                    node_degrees[chosen_node] += 1
            else:
                # Випадкове приєднання
                if nodes[:-1]:  # Якщо є інші вузли
                    chosen_node = random.choice(nodes[:-1])
                    edges.append((new_node_id, chosen_node))
                    node_degrees[new_node_id] += 1
                    node_degrees[chosen_node] += 1
        
        growth_history.append(len(nodes))
    
    # Аналіз фінальної мережі
    final_nodes = len(nodes)
    final_edges = len(edges)
    average_degree = (2 * final_edges) / final_nodes if final_nodes > 0 else 0
    
    return {
        'final_nodes': final_nodes,
        'final_edges': final_edges,
        'average_degree': average_degree,
        'network_growth': growth_history
    }

def social_impact_model(adopters: int, 
                       total_population: int, 
                       innovation_coefficient: float,
                       communication_frequency: float) -> float:
    """
    Модель соціального впливу (модель поширення інновацій).
    
    Параметри:
        adopters: Кількість прихильників інновації
        total_population: Загальна кількість населення
        innovation_coefficient: Коефіцієнт інновації
        communication_frequency: Частота комунікації
    
    Повертає:
        Очікувана кількість нових прихильників
    """
    if total_population <= 0:
        return 0.0
    
    if adopters < 0 or adopters > total_population:
        raise ValueError("Кількість прихильників повинна бути між 0 та загальною популяцією")
    
    if innovation_coefficient < 0 or innovation_coefficient > 1:
        raise ValueError("Коефіцієнт інновації повинен бути між 0 та 1")
    
    if communication_frequency < 0:
        raise ValueError("Частота комунікації повинна бути невід'ємною")
    
    # Модель поширення інновацій
    # dA/dt = k * (M - A) * A / M
    # де A - кількість прихильників, M - загальна популяція, k - коефіцієнт
    
    non_adopters = total_population - adopters
    adoption_rate = innovation_coefficient * communication_frequency * \
                   (adopters * non_adopters) / total_population
    
    return adoption_rate

def social_entropy_measure(interactions: List[Tuple[int, int, float]]) -> float:
    """
    Обчислити ентропію соціальної системи.
    
    Параметри:
        interactions: Список взаємодій (вузол1, вузол2, інтенсивність)
    
    Повертає:
        Ентропія соціальної системи
    """
    if not interactions:
        return 0.0
    
    # Підраховуємо загальну інтенсивність
    total_intensity = sum(interaction[2] for interaction in interactions)
    
    if total_intensity == 0:
        return 0.0
    
    # Обчислюємо ймовірності кожної взаємодії
    probabilities = [interaction[2] / total_intensity for interaction in interactions]
    
    # Обчислюємо ентропію
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    
    return entropy

def social_resilience_index(vulnerability: float, 
                           adaptability: float, 
                           recovery_capacity: float) -> float:
    """
    Обчислити індекс соціальної стійкості.
    
    Параметри:
        vulnerability: Вразливість (від 0 до 1)
        adaptability: Адаптивність (від 0 до 1)
        recovery_capacity: Здатність до відновлення (від 0 до 1)
    
    Повертає:
        Індекс соціальної стійкості (від 0 до 1)
    """
    # Нормалізуємо вхідні значення
    vulnerability = max(0.0, min(1.0, vulnerability))
    adaptability = max(0.0, min(1.0, adaptability))
    recovery_capacity = max(0.0, min(1.0, recovery_capacity))
    
    # Індекс стійкості: висока адаптивність та здатність до відновлення 
    # компенсують високу вразливість
    resilience = (adaptability + recovery_capacity - vulnerability + 1) / 3
    
    # Обмежуємо діапазон [0, 1]
    return max(0.0, min(1.0, resilience))

def collective_intelligence_score(individual_performance: List[float], 
                                group_performance: float) -> float:
    """
    Обчислити рівень колективного інтелекту.
    
    Параметри:
        individual_performance: Список індивідуальних результатів
        group_performance: Результат групової роботи
    
    Повертає:
        Рівень колективного інтелекту
    """
    if not individual_performance:
        return 0.0
    
    # Середній індивідуальний результат
    avg_individual = sum(individual_performance) / len(individual_performance)
    
    if avg_individual <= 0:
        return 0.0
    
    # Рівень колективного інтелекту як відношення групового до індивідуального результату
    c_score = group_performance / avg_individual
    
    # Нормалізуємо до діапазону [0, 2] (де 1 - немає колективного ефекту)
    return max(0.0, min(2.0, c_score))

def social_dynamics_model(initial_state: Dict[str, float], 
                         transition_matrix: List[List[float]], 
                         time_steps: int) -> List[Dict[str, float]]:
    """
    Симулювати соціальну динаміку.
    
    Параметри:
        initial_state: Початковий стан системи {група: частка}
        transition_matrix: Матриця переходів між групами
        time_steps: Кількість часових кроків
    
    Повертає:
        Список станів системи на кожному кроці
    """
    if not initial_state or time_steps <= 0:
        return [initial_state]
    
    # Перевіряємо розміри матриці переходів
    n_groups = len(initial_state)
    if len(transition_matrix) != n_groups or any(len(row) != n_groups for row in transition_matrix):
        raise ValueError("Матриця переходів повинна бути квадратною та відповідати кількості груп")
    
    # Перевіряємо, що всі рядки матриці сумуються до 1
    for i, row in enumerate(transition_matrix):
        row_sum = sum(row)
        if abs(row_sum - 1.0) > 1e-10:
            raise ValueError(f"Рядок {i} матриці переходів не сумується до 1")
    
    # Сортуємо групи для узгодженості
    groups = sorted(initial_state.keys())
    
    # Початковий стан у вигляді вектора
    state_vector = [initial_state[group] for group in groups]
    
    # Історія станів
    history = [initial_state.copy()]
    
    # Симуляція динаміки
    current_state = state_vector[:]
    
    for _ in range(time_steps):
        # Обчислюємо новий стан
        new_state = [0.0] * n_groups
        for i in range(n_groups):
            for j in range(n_groups):
                new_state[i] += current_state[j] * transition_matrix[j][i]
        
        current_state = new_state
        state_dict = {group: current_state[i] for i, group in enumerate(groups)}
        history.append(state_dict)
    
    return history

def social_equilibrium_analysis(supply: List[float], 
                               demand: List[float]) -> Dict[str, Union[float, bool]]:
    """
    Проаналізувати соціальну рівновагу.
    
    Параметри:
        supply: Список значень пропозиції
        demand: Список значень попиту
    
    Повертає:
        Словник з аналізом рівноваги
    """
    if len(supply) != len(demand):
        raise ValueError("Списки пропозиції та попиту повинні мати однакову довжину")
    
    if not supply:
        return {
            'equilibrium_point': 0.0,
            'equilibrium_exists': False,
            'surplus_deficit': 0.0
        }
    
    # Знаходимо точку рівноваги (де supply = demand)
    differences = [abs(s - d) for s, d in zip(supply, demand)]
    min_diff_index = differences.index(min(differences))
    equilibrium_point = (supply[min_diff_index] + demand[min_diff_index]) / 2
    
    # Перевіряємо, чи існує рівновага
    equilibrium_exists = differences[min_diff_index] < 1e-6
    
    # Обчислюємо надлишок/дефіцит
    surplus_deficit = supply[min_diff_index] - demand[min_diff_index]
    
    return {
        'equilibrium_point': equilibrium_point,
        'equilibrium_exists': equilibrium_exists,
        'surplus_deficit': surplus_deficit
    }

def social_network_centrality(adjacency_matrix: List[List[int]], 
                             node_id: int) -> Dict[str, float]:
    """
    Обчислити центральність вузла у соціальній мережі.
    
    Параметри:
        adjacency_matrix: Матриця суміжності мережі
        node_id: ID вузла для аналізу
    
    Повертає:
        Словник з різними мірами центральності
    """
    n = len(adjacency_matrix)
    
    if n == 0:
        return {
            'degree_centrality': 0.0,
            'closeness_centrality': 0.0,
            'betweenness_centrality': 0.0
        }
    
    if node_id < 0 or node_id >= n:
        raise ValueError("Невірний ID вузла")
    
    # Перевіряємо, чи матриця квадратна
    if any(len(row) != n for row in adjacency_matrix):
        raise ValueError("Матриця суміжності повинна бути квадратною")
    
    # Ступенева центральність
    degree = sum(adjacency_matrix[node_id])
    degree_centrality = degree / (n - 1) if n > 1 else 0
    
    # Близькісна центральність
    # Обчислюємо найкоротші шляхи від вузла до всіх інших
    distances = [float('inf')] * n
    distances[node_id] = 0
    queue = [node_id]
    visited = set()
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in range(n):
            if adjacency_matrix[current][neighbor] == 1 and neighbor not in visited:
                if distances[neighbor] > distances[current] + 1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    
    # Сума відстаней до всіх інших вузлів
    total_distance = sum(d for d in distances if d != float('inf') and d != 0)
    closeness_centrality = (n - 1) / total_distance if total_distance > 0 else 0
    
    # Посередницька центральність (спрощена)
    # Для спрощення обчислюємо як частку зв'язків, що проходять через вузол
    betweenness_centrality = degree_centrality * (1 - degree_centrality)
    
    return {
        'degree_centrality': degree_centrality,
        'closeness_centrality': closeness_centrality,
        'betweenness_centrality': betweenness_centrality
    }

def social_choice_function(individual_preferences: List[List[int]], 
                          aggregation_method: str = 'borda') -> List[int]:
    """
    Агрегувати індивідуальні переваги в колективний вибір.
    
    Параметри:
        individual_preferences: Список переваг кожного індивіда (від 1 - найкраще)
        aggregation_method: Метод агрегації ('borda', 'majority', 'approval')
    
    Повертає:
        Агрегований список переваг
    """
    if not individual_preferences:
        return []
    
    n_individuals = len(individual_preferences)
    if n_individuals == 0:
        return []
    
    # Перевіряємо, що всі індивідуальні списки мають однакову довжину
    n_options = len(individual_preferences[0])
    if any(len(pref) != n_options for pref in individual_preferences):
        raise ValueError("Всі списки переваг повинні мати однакову довжину")
    
    if aggregation_method == 'borda':
        # Система Борда
        borda_scores = [0] * n_options
        
        for preferences in individual_preferences:
            for i, rank in enumerate(preferences):
                # Бали за системою Борда: найкраща опція отримує (n-1) балів
                borda_scores[i] += (n_options - rank)
        
        # Сортуємо опції за балами
        ranked_options = sorted(range(n_options), key=lambda i: borda_scores[i], reverse=True)
        
    elif aggregation_method == 'majority':
        # Правило простої більшості (для кожної пари опцій)
        # Спрощена реалізація: просто сортуємо за середнім рангом
        average_ranks = []
        for i in range(n_options):
            avg_rank = sum(pref[i] for pref in individual_preferences) / n_individuals
            average_ranks.append(avg_rank)
        
        ranked_options = sorted(range(n_options), key=lambda i: average_ranks[i])
        
    elif aggregation_method == 'approval':
        # Апробаційне голосування (припускаємо, що ранги 1-3 - одобрення)
        approval_scores = [0] * n_options
        
        for preferences in individual_preferences:
            for i, rank in enumerate(preferences):
                if rank <= 3:  # Одобрення
                    approval_scores[i] += 1
        
        ranked_options = sorted(range(n_options), key=lambda i: approval_scores[i], reverse=True)
        
    else:
        raise ValueError("Невідомий метод агрегації")
    
    return ranked_options

def social_norm_emergence(initial_norms: List[float], 
                         conformity_pressure: float, 
                         innovation_rate: float,
                         time_steps: int) -> List[List[float]]:
    """
    Симулювати емерджентність соціальних норм.
    
    Параметри:
        initial_norms: Початкові норми індивідів
        conformity_pressure: Тиск конформізму
        innovation_rate: Швидкість інновацій
        time_steps: Кількість часових кроків
    
    Повертає:
        Список норм на кожному кроці
    """
    if not initial_norms or time_steps <= 0:
        return [initial_norms]
    
    if conformity_pressure < 0 or conformity_pressure > 1:
        raise ValueError("Тиск конформізму повинен бути між 0 та 1")
    
    if innovation_rate < 0 or innovation_rate > 1:
        raise ValueError("Швидкість інновацій повинна бути між 0 та 1")
    
    norms_history = [initial_norms[:]]
    current_norms = initial_norms[:]
    n_individuals = len(current_norms)
    
    for _ in range(time_steps):
        new_norms = []
        
        for i in range(n_individuals):
            # Вплив групи (середнє норм інших)
            group_influence = sum(current_norms[j] for j in range(n_individuals) if j != i) / (n_individuals - 1) \
                            if n_individuals > 1 else current_norms[i]
            
            # Конформізм
            conformist_norm = conformity_pressure * group_influence + \
                            (1 - conformity_pressure) * current_norms[i]
            
            # Інновації
            innovation = random.uniform(-0.1, 0.1) * innovation_rate
            new_norm = conformist_norm + innovation
            
            # Обмежуємо норми діапазоном [0, 1]
            new_norm = max(0.0, min(1.0, new_norm))
            new_norms.append(new_norm)
        
        current_norms = new_norms
        norms_history.append(current_norms[:])
    
    return norms_history

def social_capital_accumulation(initial_capital: float, 
                               investment_rate: float, 
                               depreciation_rate: float,
                               time_periods: int) -> List[float]:
    """
    Модель акумуляції соціального капіталу.
    
    Параметри:
        initial_capital: Початковий соціальний капітал
        investment_rate: Швидкість інвестицій
        depreciation_rate: Швидкість знецінення
        time_periods: Кількість періодів
    
    Повертає:
        Список рівнів соціального капіталу
    """
    if initial_capital < 0:
        raise ValueError("Початковий капітал не може бути від'ємним")
    
    if investment_rate < 0 or depreciation_rate < 0:
        raise ValueError("Ставки не можуть бути від'ємними")
    
    if time_periods <= 0:
        return [initial_capital]
    
    capital_levels = [initial_capital]
    current_capital = initial_capital
    
    for _ in range(time_periods):
        # Інвестиції залежать від поточного рівня капіталу
        investment = investment_rate * current_capital
        
        # Знецінення
        depreciation = depreciation_rate * current_capital
        
        # Зміна капіталу
        delta_capital = investment - depreciation
        
        # Оновлюємо капітал
        current_capital = max(0, current_capital + delta_capital)
        capital_levels.append(current_capital)
    
    return capital_levels

def collective_memory_model(memories: List[float], 
                           forgetting_rate: float,
                           sharing_rate: float,
                           time_steps: int) -> List[float]:
    """
    Модель колективної пам'яті.
    
    Параметри:
        memories: Початкові рівні пам'яті індивідів
        forgetting_rate: Швидкість забування
        sharing_rate: Швидкість поширення пам'яті
        time_steps: Кількість часових кроків
    
    Повертає:
        Список рівнів колективної пам'яті
    """
    if not memories or time_steps <= 0:
        return [sum(memories) / len(memories)] if memories else [0.0]
    
    if forgetting_rate < 0 or forgetting_rate > 1:
        raise ValueError("Швидкість забування повинна бути між 0 та 1")
    
    if sharing_rate < 0 or sharing_rate > 1:
        raise ValueError("Швидкість поширення повинна бути між 0 та 1")
    
    collective_memory_history = []
    current_memories = memories[:]
    n_individuals = len(current_memories)
    
    for _ in range(time_steps):
        # Обчислюємо колективну пам'ять (середнє)
        collective_memory = sum(current_memories) / n_individuals if n_individuals > 0 else 0
        collective_memory_history.append(collective_memory)
        
        # Оновлюємо індивідуальні рівні пам'яті
        new_memories = []
        
        for i in range(n_individuals):
            # Забування
            forgotten_memory = current_memories[i] * (1 - forgetting_rate)
            
            # Поширення пам'яті (вплив колективної пам'яті)
            shared_memory = sharing_rate * collective_memory
            retained_memory = (1 - sharing_rate) * forgotten_memory
            
            new_memory = retained_memory + shared_memory
            new_memories.append(max(0.0, min(1.0, new_memory)))
        
        current_memories = new_memories
    
    # Додаємо фінальний рівень
    final_collective_memory = sum(current_memories) / n_individuals if n_individuals > 0 else 0
    collective_memory_history.append(final_collective_memory)
    
    return collective_memory_history

def social_identity_dynamics(group_memberships: List[List[bool]], 
                            identity_salience: List[float],
                            interaction_intensity: float) -> List[float]:
    """
    Модель динаміки соціальної ідентичності.
    
    Параметри:
        group_memberships: Матриця належності до груп [індивід][група]
        identity_salience: Важливість кожної ідентичності
        interaction_intensity: Інтенсивність міжгрупових взаємодій
    
    Повертає:
        Список активності ідентичностей
    """
    if not group_memberships or not identity_salience:
        return []
    
    n_individuals = len(group_memberships)
    n_groups = len(identity_salience)
    
    # Перевіряємо розміри
    if any(len(memberships) != n_groups for memberships in group_memberships):
        raise ValueError("Всі вектори належності повинні мати однакову довжину")
    
    # Нормалізуємо важливість ідентичностей
    total_salience = sum(identity_salience)
    if total_salience > 0:
        normalized_salience = [s / total_salience for s in identity_salience]
    else:
        normalized_salience = [1.0 / n_groups] * n_groups
    
    # Обчислюємо активність кожної ідентичності
    identity_activity = [0.0] * n_groups
    
    for group_id in range(n_groups):
        # Підраховуємо кількість членів групи
        group_members = sum(1 for memberships in group_memberships if memberships[group_id])
        
        if group_members == 0:
            identity_activity[group_id] = 0
            continue
        
        # Обчислюємо внутрішню когезію групи
        intra_group_similarity = 0
        group_member_indices = [i for i, memberships in enumerate(group_memberships) 
                              if memberships[group_id]]
        
        for i in group_member_indices:
            for j in group_member_indices:
                if i != j:
                    # Схожість між членами групи
                    similarity = sum(m1 and m2 for m1, m2 in 
                                   zip(group_memberships[i], group_memberships[j]))
                    intra_group_similarity += similarity / n_groups if n_groups > 0 else 0
        
        # Нормалізуємо внутрішню когезію
        max_possible_similarities = len(group_member_indices) * (len(group_member_indices) - 1)
        normalized_intra_similarity = (intra_group_similarity / max_possible_similarities 
                                     if max_possible_similarities > 0 else 0)
        
        # Обчислюємо міжгрупову взаємодію
        inter_group_interaction = 0
        for other_group_id in range(n_groups):
            if other_group_id != group_id:
                other_group_members = sum(1 for memberships in group_memberships 
                                        if memberships[other_group_id])
                # Інтенсивність взаємодії пропорційна розмірам груп
                interaction_strength = (group_members * other_group_members * 
                                      interaction_intensity / (n_individuals ** 2))
                inter_group_interaction += interaction_strength
        
        # Активність ідентичності
        identity_activity[group_id] = (normalized_intra_similarity * normalized_salience[group_id] - 
                                     inter_group_interaction * (1 - normalized_salience[group_id]))
        
        # Обмежуємо діапазон [0, 1]
        identity_activity[group_id] = max(0.0, min(1.0, identity_activity[group_id]))
    
    return identity_activity

def social_influence_network(influence_matrix: List[List[float]], 
                            initial_opinions: List[float],
                            time_steps: int) -> List[List[float]]:
    """
    Модель мережі соціального впливу.
    
    Параметри:
        influence_matrix: Матриця впливу [впливаючий][впливований]
        initial_opinions: Початкові думки індивідів
        time_steps: Кількість часових кроків
    
    Повертає:
        Список думок на кожному кроці
    """
    if not influence_matrix or not initial_opinions or time_steps <= 0:
        return [initial_opinions] if initial_opinions else []
    
    n_individuals = len(initial_opinions)
    
    # Перевіряємо розміри матриці
    if len(influence_matrix) != n_individuals or any(len(row) != n_individuals for row in influence_matrix):
        raise ValueError("Матриця впливу повинна бути квадратною та відповідати кількості індивідів")
    
    # Нормалізуємо рядки матриці (сума впливів на кожного індивіда = 1)
    normalized_influence = []
    for row in influence_matrix:
        row_sum = sum(row)
        if row_sum > 0:
            normalized_row = [influence / row_sum for influence in row]
        else:
            # Якщо немає впливів, індивід зберігає свою думку
            normalized_row = [0] * n_individuals
            normalized_row[row.index(max(row))] = 1 if row else 0
        normalized_influence.append(normalized_row)
    
    opinions_history = [initial_opinions[:]]
    current_opinions = initial_opinions[:]
    
    for _ in range(time_steps):
        new_opinions = []
        
        for i in range(n_individuals):
            # Нова думка як зважена сума думок інших індивідів
            new_opinion = sum(normalized_influence[j][i] * current_opinions[j] 
                            for j in range(n_individuals))
            # Обмежуємо діапазон [-1, 1]
            new_opinion = max(-1.0, min(1.0, new_opinion))
            new_opinions.append(new_opinion)
        
        current_opinions = new_opinions
        opinions_history.append(current_opinions[:])
    
    return opinions_history

def social_system_stability(equilibrium_state: List[float], 
                           perturbation: List[float],
                           system_matrix: List[List[float]]) -> Dict[str, Union[float, bool]]:
    """
    Аналіз стійкості соціальної системи.
    
    Параметри:
        equilibrium_state: Стан рівноваги
        perturbation: Збурення системи
        system_matrix: Матриця системи
    
    Повертає:
        Словник з аналізом стійкості
    """
    if not equilibrium_state or not perturbation or not system_matrix:
        return {
            'is_stable': False,
            'stability_measure': 0.0,
            'return_time': float('inf')
        }
    
    n = len(equilibrium_state)
    
    # Перевіряємо розміри
    if len(perturbation) != n or len(system_matrix) != n or any(len(row) != n for row in system_matrix):
        raise ValueError("Невідповідність розмірів вхідних даних")
    
    # Спрощений аналіз стійкості
    # Обчислюємо власні значення матриці системи
    try:
        # Для спрощення використовуємо слід матриці як наближення
        trace = sum(system_matrix[i][i] for i in range(n))
        
        # Визначник
        if n == 2:
            determinant = (system_matrix[0][0] * system_matrix[1][1] - 
                          system_matrix[0][1] * system_matrix[1][0])
        else:
            # Для більших матриць використовуємо наближення
            determinant = trace / n if n > 0 else 0
        
        # Система стійка, якщо всі власні значення мають від'ємні дійсні частини
        # Спрощений критерій: слід < 0 та визначник > 0
        is_stable = trace < 0 and determinant > 0
        
        # Міра стійкості
        stability_measure = abs(trace) / (1 + abs(determinant)) if determinant != 0 else 0
        
        # Час повернення до рівноваги (наближений)
        return_time = 1 / abs(trace) if trace != 0 else float('inf')
        
    except:
        # Якщо не вдалося обчислити, припускаємо нестійкість
        is_stable = False
        stability_measure = 0.0
        return_time = float('inf')
    
    return {
        'is_stable': is_stable,
        'stability