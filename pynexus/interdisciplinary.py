"""
Модуль для міждисциплінрних досліджень в PyNexus.
Містить функції для комплексного аналізу, що поєднує методи з різних наукових дисциплін.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import numpy as np

# Константи для міждисциплінрних досліджень
COMPLEXITY_THRESHOLD = 0.7  # Поріг складності для міждисциплінрного аналізу
INNOVATION_RATE = 0.05  # Швидкість інновацій
SYNERGY_FACTOR = 1.5  # Коефіцієнт синергії
UNCERTAINTY_TOLERANCE = 0.1  # Толерантність до невизначеності

def interdisciplinary_integration_score(disciplines: List[str], 
                                      collaboration_strength: float,
                                      knowledge_overlap: float) -> float:
    """
    Обчислити рівень інтеграції між дисциплінами.
    
    Параметри:
        disciplines: Список дисциплін
        collaboration_strength: Сила співпраці (0-1)
        knowledge_overlap: Перекриття знань (0-1)
    
    Повертає:
        Рівень інтеграції (0-1)
    """
    if not disciplines:
        raise ValueError("Список дисциплін не може бути порожнім")
    
    if not 0.0 <= collaboration_strength <= 1.0:
        raise ValueError("Сила співпраці повинна бути в діапазоні 0-1")
    
    if not 0.0 <= knowledge_overlap <= 1.0:
        raise ValueError("Перекриття знань повинно бути в діапазоні 0-1")
    
    # Кількість дисциплін
    num_disciplines = len(disciplines)
    
    # Різноманітність дисциплін (чим більше різних дисциплін, тим краще)
    diversity_score = min(1.0, num_disciplines / 10.0)
    
    # Індекс інтеграції
    integration_index = (collaboration_strength * 0.4 + 
                        knowledge_overlap * 0.3 + 
                        diversity_score * 0.3)
    
    return max(0.0, min(1.0, integration_index))

def complexity_science_model(system_components: int, 
                           interactions: int,
                           emergence_level: float) -> Dict[str, Union[float, str]]:
    """
    Модель складних систем.
    
    Параметри:
        system_components: Кількість компонентів системи
        interactions: Кількість взаємодій
        emergence_level: Рівень емерджентності (0-1)
    
    Повертає:
        Словник з характеристиками складної системи
    """
    if system_components <= 0:
        raise ValueError("Кількість компонентів повинна бути додатньою")
    
    if interactions < 0:
        raise ValueError("Кількість взаємодій не може бути від'ємною")
    
    if not 0.0 <= emergence_level <= 1.0:
        raise ValueError("Рівень емерджентності повинен бути в діапазоні 0-1")
    
    # Ступінь зв'язності
    connectivity = interactions / (system_components * (system_components - 1) / 2) if system_components > 1 else 0
    
    # Складність системи
    complexity = (system_components * connectivity * emergence_level) / 100.0
    
    # Класифікація складності
    if complexity < 0.3:
        complexity_category = "Низька"
    elif complexity < 0.6:
        complexity_category = "Середня"
    elif complexity < 0.8:
        complexity_category = "Висока"
    else:
        complexity_category = "Дуже висока"
    
    # Передбачуваність
    predictability = max(0.0, 1.0 - complexity * 1.5)
    
    return {
        'complexity': complexity,
        'complexity_category': complexity_category,
        'connectivity': connectivity,
        'predictability': predictability,
        'emergence_level': emergence_level
    }

def systems_biology_model(gene_interactions: List[Tuple[str, str]], 
                         environmental_factors: List[float],
                         time_points: List[float]) -> Dict[str, Any]:
    """
    Модель системної біології.
    
    Параметри:
        gene_interactions: Список взаємодій генів
        environmental_factors: Впливи середовища
        time_points: Точки часу для аналізу
    
    Повертає:
        Словник з результатами моделювання
    """
    if not gene_interactions:
        return {
            'network_complexity': 0.0,
            'robustness': 0.0,
            'adaptability': 0.0,
            'critical_nodes': []
        }
    
    if not environmental_factors or not time_points:
        raise ValueError("Фактори середовища та точки часу не можуть бути порожніми")
    
    # Кількість генів
    genes = set()
    for gene1, gene2 in gene_interactions:
        genes.add(gene1)
        genes.add(gene2)
    
    num_genes = len(genes)
    
    # Кількість взаємодій
    num_interactions = len(gene_interactions)
    
    # Складність мережі
    network_complexity = num_interactions / (num_genes * (num_genes - 1) / 2) if num_genes > 1 else 0
    
    # Робастність (стійкість до збурень)
    avg_environmental_factor = sum(environmental_factors) / len(environmental_factors)
    robustness = max(0.0, min(1.0, 1.0 - abs(avg_environmental_factor - 0.5)))
    
    # Адаптивність
    adaptability = min(1.0, len(time_points) / 100.0)
    
    # Критичні вузли (гени з найбільшою кількістю зв'язків)
    gene_connections = Counter()
    for gene1, gene2 in gene_interactions:
        gene_connections[gene1] += 1
        gene_connections[gene2] += 1
    
    # Топ-10% критичних вузлів
    threshold = int(len(gene_connections) * 0.1) or 1
    critical_nodes = [gene for gene, count in gene_connections.most_common(threshold)]
    
    return {
        'network_complexity': network_complexity,
        'robustness': robustness,
        'adaptability': adaptability,
        'critical_nodes': critical_nodes,
        'num_genes': num_genes,
        'num_interactions': num_interactions
    }

def computational_sustainability_model(resource_consumption: List[float], 
                                     renewable_ratio: float,
                                     efficiency_improvements: List[float],
                                     time_horizon: int) -> Dict[str, Union[float, List[float]]]:
    """
    Модель обчислювальної сталості.
    
    Параметри:
        resource_consumption: Споживання ресурсів по роках
        renewable_ratio: Частка відновлюваних ресурсів (0-1)
        efficiency_improvements: Покращення ефективності по роках
        time_horizon: Горизонт планування (роки)
    
    Повертає:
        Словник з прогнозами сталості
    """
    if not resource_consumption:
        raise ValueError("Споживання ресурсів не може бути порожнім")
    
    if not 0.0 <= renewable_ratio <= 1.0:
        raise ValueError("Частка відновлюваних ресурсів повинна бути в діапазоні 0-1")
    
    if time_horizon <= 0:
        raise ValueError("Горизонт планування повинен бути додатнім")
    
    # Середнє споживання
    avg_consumption = sum(resource_consumption) / len(resource_consumption)
    
    # Тренд споживання
    if len(resource_consumption) > 1:
        consumption_trend = (resource_consumption[-1] - resource_consumption[0]) / len(resource_consumption)
    else:
        consumption_trend = 0.0
    
    # Середнє покращення ефективності
    avg_efficiency_improvement = sum(efficiency_improvements) / len(efficiency_improvements) if efficiency_improvements else 0.0
    
    # Прогноз споживання
    projected_consumption = []
    current_consumption = resource_consumption[-1] if resource_consumption else avg_consumption
    
    for year in range(time_horizon):
        # Модель: споживання змінюється згідно з трендом та ефективністю
        adjusted_consumption = current_consumption * (1 + consumption_trend / 100) * (1 - avg_efficiency_improvement / 100)
        projected_consumption.append(adjusted_consumption)
        current_consumption = adjusted_consumption
    
    # Індекс сталості
    sustainability_index = renewable_ratio * 0.5 + (1.0 - avg_consumption / 1000.0) * 0.3 + (1.0 - consumption_trend / 100.0) * 0.2
    
    # Класифікація сталості
    if sustainability_index > 0.7:
        sustainability_status = "Висока"
    elif sustainability_index > 0.4:
        sustainability_status = "Середня"
    else:
        sustainability_status = "Низька"
    
    return {
        'sustainability_index': max(0.0, min(1.0, sustainability_index)),
        'sustainability_status': sustainability_status,
        'projected_consumption': projected_consumption,
        'avg_consumption': avg_consumption,
        'consumption_trend': consumption_trend,
        'renewable_ratio': renewable_ratio
    }

def cognitive_science_integration(cognitive_processes: List[str], 
                                 neural_data: List[float],
                                 behavioral_data: List[float]) -> Dict[str, Union[float, str]]:
    """
    Інтеграція когнітивних наук.
    
    Параметри:
        cognitive_processes: Список когнітивних процесів
        neural_data: Нейронні дані
        behavioral_data: Поведінкові дані
    
    Повертає:
        Словник з інтегрованими результатами
    """
    if not cognitive_processes:
        return {
            'integration_score': 0.0,
            'consistency_level': 0.0,
            'explanatory_power': 0.0,
            'domain': 'Немає'
        }
    
    if len(neural_data) != len(behavioral_data):
        raise ValueError("Нейронні та поведінкові дані повинні мати однакову довжину")
    
    # Кількість когнітивних процесів
    num_processes = len(cognitive_processes)
    
    # Узгодженість між нейронними та поведінковими даними
    if neural_data and behavioral_data:
        # Кореляція між нейронними та поведінковими даними
        if len(neural_data) > 1:
            # Обчислюємо кореляцію (спрощений підхід)
            neural_mean = sum(neural_data) / len(neural_data)
            behavioral_mean = sum(behavioral_data) / len(behavioral_data)
            
            numerator = sum((n - neural_mean) * (b - behavioral_mean) 
                           for n, b in zip(neural_data, behavioral_data))
            
            neural_variance = sum((n - neural_mean) ** 2 for n in neural_data)
            behavioral_variance = sum((b - behavioral_mean) ** 2 for b in behavioral_data)
            
            denominator = math.sqrt(neural_variance * behavioral_variance) if neural_variance * behavioral_variance > 0 else 1
            
            consistency = abs(numerator / denominator) if denominator != 0 else 0
        else:
            consistency = 1.0
    else:
        consistency = 0.0
    
    # Рівень інтеграції
    integration_score = (num_processes / 10.0) * 0.4 + consistency * 0.6
    
    # Пояснювальна сила
    explanatory_power = integration_score * (1.0 - abs(consistency - 0.5))
    
    # Домен дослідження
    if 'memory' in cognitive_processes and 'attention' in cognitive_processes:
        domain = "Когнітивна психологія"
    elif 'perception' in cognitive_processes and 'vision' in cognitive_processes:
        domain = "Нейронаука"
    elif 'decision' in cognitive_processes and 'reasoning' in cognitive_processes:
        domain = "Когнітивна економіка"
    else:
        domain = "Загальна когнітивна наука"
    
    return {
        'integration_score': max(0.0, min(1.0, integration_score)),
        'consistency_level': consistency,
        'explanatory_power': max(0.0, min(1.0, explanatory_power)),
        'domain': domain,
        'num_processes': num_processes
    }

def data_science_interdisciplinary_model(features: List[str], 
                                       methodologies: List[str],
                                       domain_knowledge: float) -> Dict[str, Union[float, str]]:
    """
    Міждисциплінрна модель дата-сайенсу.
    
    Параметри:
        features: Список ознак
        methodologies: Список методологій
        domain_knowledge: Глибина предметних знань (0-1)
    
    Повертає:
        Словник з оцінкою міждисциплінрного підходу
    """
    if not features or not methodologies:
        return {
            'interdisciplinary_score': 0.0,
            'methodology_diversity': 0.0,
            'feature_richness': 0.0,
            'approach_type': 'Немає'
        }
    
    if not 0.0 <= domain_knowledge <= 1.0:
        raise ValueError("Глибина предметних знань повинна бути в діапазоні 0-1")
    
    # Різноманітність методологій
    unique_methodologies = len(set(methodologies))
    methodology_diversity = min(1.0, unique_methodologies / 5.0)
    
    # Багатство ознак
    num_features = len(features)
    feature_richness = min(1.0, num_features / 100.0)
    
    # Міждисциплінрний рахунок
    interdisciplinary_score = (methodology_diversity * 0.4 + 
                              feature_richness * 0.3 + 
                              domain_knowledge * 0.3)
    
    # Тип підходу
    if interdisciplinary_score > 0.7 and methodology_diversity > 0.5:
        approach_type = "Високо міждисциплінрний"
    elif interdisciplinary_score > 0.4:
        approach_type = "Середньо міждисциплінрний"
    else:
        approach_type = "Однодисциплінрний"
    
    return {
        'interdisciplinary_score': interdisciplinary_score,
        'methodology_diversity': methodology_diversity,
        'feature_richness': feature_richness,
        'approach_type': approach_type,
        'domain_knowledge': domain_knowledge
    }

def network_science_interdisciplinary( nodes: List[str], 
                                      edges: List[Tuple[str, str]],
                                      node_attributes: Dict[str, Dict[str, Any]],
                                      edge_attributes: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Union[float, int, str]]:
    """
    Міждисциплінрний аналіз мереж.
    
    Параметри:
        nodes: Список вузлів
        edges: Список ребер
        node_attributes: Атрибути вузлів
        edge_attributes: Атрибути ребер
    
    Повертає:
        Словник з міждисциплінрними характеристиками мережі
    """
    if not nodes:
        return {
            'network_size': 0,
            'interdisciplinary_index': 0.0,
            'heterogeneity': 0.0,
            'modularity': 0.0,
            'network_type': 'Порожня'
        }
    
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # Густина мережі
    max_possible_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    # Аналіз атрибутів вузлів для визначення міждисциплінрності
    if node_attributes:
        # Підрахунок різних типів вузлів
        node_types = set()
        for attrs in node_attributes.values():
            if 'type' in attrs:
                node_types.add(attrs['type'])
        
        type_diversity = len(node_types) / len(node_attributes) if node_attributes else 0
    else:
        type_diversity = 0.0
    
    # Аналіз атрибутів ребер
    if edge_attributes:
        # Підрахунок різних типів зв'язків
        edge_types = set()
        for attrs in edge_attributes.values():
            if 'type' in attrs:
                edge_types.add(attrs['type'])
        
        edge_diversity = len(edge_types) / len(edge_attributes) if edge_attributes else 0
    else:
        edge_diversity = 0.0
    
    # Індекс міждисциплінрності
    interdisciplinary_index = (type_diversity * 0.5 + edge_diversity * 0.3 + density * 0.2)
    
    # Гетерогенність мережі
    # На основі різноманітності атрибутів
    heterogeneity = (type_diversity + edge_diversity) / 2.0
    
    # Модулярність (спрощений підхід)
    # Припускаємо, що висока модулярність вказує на чіткі спільноти
    modularity = 1.0 - density if density > 0 else 0
    
    # Тип мережі
    if interdisciplinary_index > 0.6:
        network_type = "Міждисциплінрна"
    elif interdisciplinary_index > 0.3:
        network_type = "Мультиспектральна"
    else:
        network_type = "Однорідна"
    
    return {
        'network_size': num_nodes,
        'edges_count': num_edges,
        'density': density,
        'interdisciplinary_index': interdisciplinary_index,
        'heterogeneity': heterogeneity,
        'modularity': modularity,
        'network_type': network_type
    }

def computational_social_science_model( social_interactions: List[Tuple[str, str, float]],
                                      demographic_data: Dict[str, Dict[str, Any]],
                                      behavioral_patterns: List[Dict[str, Any]]) -> Dict[str, Union[float, str]]:
    """
    Модель обчислювальних соціальних наук.
    
    Параметри:
        social_interactions: Список соціальних взаємодій (особа1, особа2, інтенсивність)
        demographic_data: Демографічні дані
        behavioral_patterns: Поведінкові патерни
    
    Повертає:
        Словник з результатами моделювання
    """
    if not social_interactions:
        return {
            'social_complexity': 0.0,
            'behavioral_diversity': 0.0,
            'demographic_heterogeneity': 0.0,
            'social_dynamics': 'Статична'
        }
    
    # Аналіз соціальних взаємодій
    participants = set()
    total_intensity = 0.0
    for person1, person2, intensity in social_interactions:
        participants.add(person1)
        participants.add(person2)
        total_intensity += intensity
    
    num_participants = len(participants)
    avg_intensity = total_intensity / len(social_interactions) if social_interactions else 0
    
    # Соціальна складність
    # Базується на кількості учасників, інтенсивності та кількості взаємодій
    social_complexity = min(1.0, (num_participants * len(social_interactions) * avg_intensity) / 10000.0)
    
    # Поведінкова різноманітність
    if behavioral_patterns:
        # Кількість унікальних патернів
        unique_patterns = len(behavioral_patterns)
        behavioral_diversity = min(1.0, unique_patterns / 50.0)
    else:
        behavioral_diversity = 0.0
    
    # Демографічна гетерогенність
    if demographic_data:
        # Аналіз різноманітності за ключовими демографічними характеристиками
        attributes = ['age', 'gender', 'education', 'income']
        diversity_scores = []
        
        for attr in attributes:
            values = [data[attr] for data in demographic_data.values() if attr in data]
            if values:
                # Ентропія як міра різноманітності
                value_counts = Counter(values)
                total = sum(value_counts.values())
                entropy = -sum((count/total) * math.log(count/total) for count in value_counts.values()) if total > 0 else 0
                max_entropy = math.log(len(value_counts)) if len(value_counts) > 1 else 1
                normalized_diversity = entropy / max_entropy if max_entropy > 0 else 0
                diversity_scores.append(normalized_diversity)
        
        demographic_heterogeneity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
    else:
        demographic_heterogeneity = 0.0
    
    # Соціальна динаміка
    if len(social_interactions) > 100:
        social_dynamics = "Динамічна"
    elif len(social_interactions) > 10:
        social_dynamics = "Помірно динамічна"
    else:
        social_dynamics = "Статична"
    
    return {
        'social_complexity': social_complexity,
        'behavioral_diversity': behavioral_diversity,
        'demographic_heterogeneity': demographic_heterogeneity,
        'social_dynamics': social_dynamics,
        'participants_count': num_participants,
        'interactions_count': len(social_interactions),
        'avg_interaction_intensity': avg_intensity
    }

def interdisciplinary_innovation_model(disciplinary_knowledge: Dict[str, float], 
                                     collaboration_network: List[Tuple[str, str]],
                                     resource_allocation: Dict[str, float]) -> Dict[str, Union[float, List[str]]]:
    """
    Модель міждисциплінрних інновацій.
    
    Параметри:
        disciplinary_knowledge: Рівень знань по дисциплінам
        collaboration_network: Мережа співпраці
        resource_allocation: Розподіл ресурсів
    
    Повертає:
        Словник з прогнозами інновацій
    """
    if not disciplinary_knowledge:
        return {
            'innovation_potential': 0.0,
            'interdisciplinary_synergy': 0.0,
            'breakthrough_likelihood': 0.0,
            'promising_areas': []
        }
    
    # Різноманітність знань
    knowledge_areas = list(disciplinary_knowledge.keys())
    knowledge_diversity = len(knowledge_areas) / 10.0  # Нормалізація
    
    # Середній рівень знань
    avg_knowledge = sum(disciplinary_knowledge.values()) / len(disciplinary_knowledge)
    
    # Аналіз мережі співпраці
    collaborators = set()
    for collab1, collab2 in collaboration_network:
        collaborators.add(collab1)
        collaborators.add(collab2)
    
    collaboration_diversity = len(collaborators) / 20.0  # Нормалізація
    
    # Різноманітність ресурсів
    resource_diversity = len(resource_allocation) / 5.0  # Нормалізація
    
    # Потенціал інновацій
    innovation_potential = (knowledge_diversity * 0.3 + 
                           avg_knowledge * 0.3 + 
                           collaboration_diversity * 0.2 + 
                           resource_diversity * 0.2)
    
    # Міждисциплінрна синергія
    # Базується на поєднанні різних дисциплін
    interdisciplinary_synergy = min(1.0, innovation_potential * SYNERGY_FACTOR)
    
    # Ймовірність прориву
    breakthrough_likelihood = innovation_potential * (1.0 - UNCERTAINTY_TOLERANCE)
    
    # Перспективні області
    # Сортуємо дисципліни за рівнем знань
    sorted_areas = sorted(disciplinary_knowledge.items(), key=lambda x: x[1], reverse=True)
    promising_areas = [area for area, level in sorted_areas[:3]]  # Топ-3 області
    
    return {
        'innovation_potential': max(0.0, min(1.0, innovation_potential)),
        'interdisciplinary_synergy': max(0.0, min(1.0, interdisciplinary_synergy)),
        'breakthrough_likelihood': max(0.0, min(1.0, breakthrough_likelihood)),
        'promising_areas': promising_areas,
        'knowledge_diversity': knowledge_diversity,
        'collaboration_diversity': collaboration_diversity
    }

def complexity_economics_model(agents: List[Dict[str, Any]], 
                              interaction_rules: List[str],
                              environmental_constraints: List[float]) -> Dict[str, Union[float, str]]:
    """
    Модель економіки складних систем.
    
    Параметри:
        agents: Список агентів з їх атрибутами
        interaction_rules: Правила взаємодії
        environmental_constraints: Обмеження середовища
    
    Повертає:
        Словник з економічними показниками складної системи
    """
    if not agents:
        return {
            'system_emergence': 0.0,
            'adaptive_efficiency': 0.0,
            'market_stability': 0.0,
            'economic_regime': 'Статична'
        }
    
    num_agents = len(agents)
    
    # Адаптивність агентів
    if all('adaptability' in agent for agent in agents):
        avg_adaptability = sum(agent['adaptability'] for agent in agents) / num_agents
    else:
        avg_adaptability = 0.5  # За замовчуванням
    
    # Різноманітність стратегій
    if all('strategy' in agent for agent in agents):
        strategies = [agent['strategy'] for agent in agents]
        strategy_diversity = len(set(strategies)) / len(strategies) if strategies else 0
    else:
        strategy_diversity = 0.5  # За замовчуванням
    
    # Складність взаємодій
    interaction_complexity = len(interaction_rules) / 10.0  # Нормалізація
    
    # Обмеження середовища
    avg_constraint = sum(environmental_constraints) / len(environmental_constraints) if environmental_constraints else 0.5
    
    # Емерджентність системи
    system_emergence = (avg_adaptability * 0.3 + 
                       strategy_diversity * 0.3 + 
                       interaction_complexity * 0.2 + 
                       (1.0 - avg_constraint) * 0.2)
    
    # Адаптивна ефективність
    adaptive_efficiency = avg_adaptability * (1.0 - avg_constraint)
    
    # Стабільність ринку
    market_stability = 1.0 - system_emergence * 0.7  # Зворотний зв'язок
    
    # Економічний режим
    if system_emergence > 0.7:
        economic_regime = "Катастрофічний"
    elif system_emergence > 0.5:
        economic_regime = "Нелінійний"
    elif system_emergence > 0.3:
        economic_regime = "Складно-адаптивний"
    else:
        economic_regime = "Статичний"
    
    return {
        'system_emergence': max(0.0, min(1.0, system_emergence)),
        'adaptive_efficiency': max(0.0, min(1.0, adaptive_efficiency)),
        'market_stability': max(0.0, min(1.0, market_stability)),
        'economic_regime': economic_regime,
        'num_agents': num_agents,
        'strategy_diversity': strategy_diversity
    }

def interdisciplinary_optimization_model(objectives: List[str], 
                                       constraints: List[str],
                                       methodologies: List[str]) -> Dict[str, Union[float, str]]:
    """
    Міждисциплінрна модель оптимізації.
    
    Параметри:
        objectives: Цілі оптимізації
        constraints: Обмеження
        methodologies: Методології
    
    Повертає:
        Словник з результатами оптимізації
    """
    if not objectives:
        return {
            'optimization_score': 0.0,
            'methodology_appropriateness': 0.0,
            'constraint_handling': 0.0,
            'solution_quality': 0.0
        }
    
    # Кількість цілей
    num_objectives = len(objectives)
    objective_complexity = min(1.0, num_objectives / 5.0)
    
    # Кількість обмежень
    num_constraints = len(constraints)
    constraint_complexity = min(1.0, num_constraints / 10.0)
    
    # Різноманітність методологій
    num_methodologies = len(methodologies)
    methodology_diversity = min(1.0, num_methodologies / 3.0)
    
    # Оцінка відповідності методологій цілям
    # Спрощений підхід: більше методологій для більшої кількості цілей
    if num_objectives > 0:
        methodology_appropriateness = min(1.0, num_methodologies / num_objectives)
    else:
        methodology_appropriateness = 1.0
    
    # Обробка обмежень
    constraint_handling = 1.0 - constraint_complexity * 0.5
    
    # Загальна оцінка оптимізації
    optimization_score = (objective_complexity * 0.3 + 
                         methodology_diversity * 0.3 + 
                         methodology_appropriateness * 0.2 + 
                         constraint_handling * 0.2)
    
    # Якість рішення
    solution_quality = optimization_score * (1.0 - constraint_complexity * 0.3)
    
    return {
        'optimization_score': max(0.0, min(1.0, optimization_score)),
        'methodology_appropriateness': methodology_appropriateness,
        'constraint_handling': constraint_handling,
        'solution_quality': max(0.0, min(1.0, solution_quality)),
        'num_objectives': num_objectives,
        'num_constraints': num_constraints
    }

def knowledge_integration_model(source_domains: List[str], 
                               integration_methods: List[str],
                               validation_criteria: List[str]) -> Dict[str, Union[float, str]]:
    """
    Модель інтеграції знань.
    
    Параметри:
        source_domains: Домени джерел знань
        integration_methods: Методи інтеграції
        validation_criteria: Критерії валідації
    
    Повертає:
        Словник з результатами інтеграції знань
    """
    if not source_domains:
        return {
            'integration_completeness': 0.0,
            'knowledge_coherence': 0.0,
            'validation_strength': 0.0,
            'integration_quality': 0.0
        }
    
    # Різноманітність доменів
    domain_diversity = len(set(source_domains)) / len(source_domains) if source_domains else 0
    
    # Кількість методів інтеграції
    num_methods = len(integration_methods)
    method_richness = min(1.0, num_methods / 5.0)
    
    # Кількість критеріїв валідації
    num_criteria = len(validation_criteria)
    validation_completeness = min(1.0, num_criteria / 10.0)
    
    # Повнота інтеграції
    integration_completeness = (domain_diversity * 0.4 + 
                               method_richness * 0.3 + 
                               validation_completeness * 0.3)
    
    # Когерентність знань
    # Припускаємо, що більше різноманітність доменів вимагає кращої інтеграції
    knowledge_coherence = 1.0 - (1.0 - integration_completeness) * domain_diversity
    
    # Сила валідації
    validation_strength = validation_completeness * (1.0 - domain_diversity * 0.2)
    
    # Якість інтеграції
    integration_quality = (integration_completeness * 0.4 + 
                          knowledge_coherence * 0.3 + 
                          validation_strength * 0.3)
    
    # Тип інтеграції
    if integration_quality > 0.8:
        integration_type = "Глибока"
    elif integration_quality > 0.5:
        integration_type = "Середня"
    else:
        integration_type = "Поверхнева"
    
    return {
        'integration_completeness': max(0.0, min(1.0, integration_completeness)),
        'knowledge_coherence': max(0.0, min(1.0, knowledge_coherence)),
        'validation_strength': max(0.0, min(1.0, validation_strength)),
        'integration_quality': max(0.0, min(1.0, integration_quality)),
        'integration_type': integration_type
    }

def interdisciplinary_research_impact(citations: List[int], 
                                    collaborations: List[Tuple[str, str]],
                                    publications: List[Dict[str, Any]]) -> Dict[str, Union[float, str]]:
    """
    Оцінка впливу міждисциплінрних досліджень.
    
    Параметри:
        citations: Кількість цитувань для кожної публікації
        collaborations: Список співпраць
        publications: Список публікацій з атрибутами
    
    Повертає:
        Словник з оцінкою впливу
    """
    if not citations:
        return {
            'impact_factor': 0.0,
            'interdisciplinary_reach': 0.0,
            'collaboration_strength': 0.0,
            'innovation_index': 0.0
        }
    
    # Середній фактор впливу
    avg_citations = sum(citations) / len(citations) if citations else 0
    impact_factor = min(1.0, avg_citations / 50.0)  # Нормалізація
    
    # Міждисциплінрний охоплення
    if publications:
        # Аналізуємо різноманітність галузей
        fields = []
        for pub in publications:
            if 'fields' in pub:
                fields.extend(pub['fields'])
        
        field_diversity = len(set(fields)) / len(fields) if fields else 0
        interdisciplinary_reach = field_diversity
    else:
        interdisciplinary_reach = 0.0
    
    # Сила співпраці
    unique_collaborators = set()
    for collab1, collab2 in collaborations:
        unique_collaborators.add(collab1)
        unique_collaborators.add(collab2)
    
    collaboration_strength = min(1.0, len(unique_collaborators) / 20.0)
    
    # Індекс інновацій
    innovation_index = (impact_factor * 0.4 + 
                       interdisciplinary_reach * 0.3 + 
                       collaboration_strength * 0.3)
    
    # Категорія впливу
    if innovation_index > 0.7:
        impact_category = "Високий"
    elif innovation_index > 0.4:
        impact_category = "Середній"
    else:
        impact_category = "Низький"
    
    return {
        'impact_factor': impact_factor,
        'interdisciplinary_reach': interdisciplinary_reach,
        'collaboration_strength': collaboration_strength,
        'innovation_index': innovation_index,
        'impact_category': impact_category,
        'total_citations': sum(citations),
        'num_publications': len(citations)
    }

def synthesis_evaluation_model(components: List[Dict[str, Any]], 
                              integration_principles: List[str],
                              evaluation_metrics: List[str]) -> Dict[str, Union[float, str]]:
    """
    Модель оцінки синтезу знань.
    
    Параметри:
        components: Компоненти для синтезу
        integration_principles: Принципи інтеграції
        evaluation_metrics: Метрики оцінки
    
    Повертає:
        Словник з результатами синтезу
    """
    if not components:
        return {
            'synthesis_quality': 0.0,
            'integration_coherence': 0.0,
            'evaluation_completeness': 0.0,
            'synthesis_maturity': 0.0
        }
    
    # Кількість компонентів
    num_components = len(components)
    component_richness = min(1.0, num_components / 20.0)
    
    # Якість компонентів (середня оцінка)
    if all('quality' in comp for comp in components):
        avg_quality = sum(comp['quality'] for comp in components) / num_components
    else:
        avg_quality = 0.5  # За замовчуванням
    
    # Різноманітність принципів інтеграції
    principle_diversity = len(set(integration_principles)) / len(integration_principles) if integration_principles else 0
    
    # Повнота метрик оцінки
    metric_completeness = min(1.0, len(evaluation_metrics) / 10.0)
    
    # Якість синтезу
    synthesis_quality = (component_richness * 0.3 + 
                       avg_quality * 0.3 + 
                       principle_diversity * 0.2 + 
                       metric_completeness * 0.2)
    
    # Когерентність інтеграції
    integration_coherence = principle_diversity * (1.0 - abs(avg_quality - 0.5))
    
    # Повнота оцінки
    evaluation_completeness = metric_completeness * (1.0 + principle_diversity * 0.2)
    
    # Зрілість синтезу
    synthesis_maturity = (synthesis_quality + integration_coherence + evaluation_completeness) / 3.0
    
    # Рівень синтезу
    if synthesis_maturity > 0.8:
        maturity_level = "Високий"
    elif synthesis_maturity > 0.5:
        maturity_level = "Середній"
    else:
        maturity_level = "Низький"
    
    return {
        'synthesis_quality': max(0.0, min(1.0, synthesis_quality)),
        'integration_coherence': max(0.0, min(1.0, integration_coherence)),
        'evaluation_completeness': max(0.0, min(1.0, evaluation_completeness)),
        'synthesis_maturity': max(0.0, min(1.0, synthesis_maturity)),
        'maturity_level': maturity_level
    }

def interdisciplinary_trend_analysis(historical_data: List[Dict[str, Any]], 
                                   current_indicators: Dict[str, float],
                                   future_projections: List[Dict[str, float]]) -> Dict[str, Union[float, str]]:
    """
    Аналіз міждисциплінрних трендів.
    
    Параметри:
        historical_data: Історичні дані
        current_indicators: Поточні індикатори
        future_projections: Майбутні проекції
    
    Повертає:
        Словник з аналізом трендів
    """
    if not historical_data:
        return {
            'trend_strength': 0.0,
            'interdisciplinary_growth': 0.0,
            'innovation_rate': 0.0,
            'future_outlook': 'Невизначений'
        }
    
    # Аналіз історичних даних
    time_span = len(historical_data)
    
    # Зростання міждисциплінрності
    if time_span > 1:
        initial_level = historical_data[0].get('interdisciplinary_index', 0)
        final_level = historical_data[-1].get('interdisciplinary_index', 0)
        growth_rate = (final_level - initial_level) / time_span if time_span > 0 else 0
        interdisciplinary_growth = max(0.0, min(1.0, growth_rate * 10))  # Нормалізація
    else:
        interdisciplinary_growth = 0.0
    
    # Сила тренду
    if current_indicators:
        avg_current = sum(current_indicators.values()) / len(current_indicators)
        trend_strength = min(1.0, avg_current)
    else:
        trend_strength = 0.0
    
    # Швидкість інновацій
    innovation_rate = INNOVATION_RATE * (1.0 + interdisciplinary_growth)
    
    # Майбутні перспективи
    if future_projections:
        # Аналіз проекцій
        avg_projections = []
        for proj in future_projections:
            if proj:
                avg_proj = sum(proj.values()) / len(proj)
                avg_projections.append(avg_proj)
        
        if avg_projections:
            avg_future = sum(avg_projections) / len(avg_projections)
            if avg_future > 0.7:
                future_outlook = "Сприятливий"
            elif avg_future > 0.4:
                future_outlook = "Помірний"
            else:
                future_outlook = "Несприятливий"
        else:
            future_outlook = "Невизначений"
    else:
        future_outlook = "Невизначений"
    
    return {
        'trend_strength': trend_strength,
        'interdisciplinary_growth': interdisciplinary_growth,
        'innovation_rate': innovation_rate,
        'future_outlook': future_outlook,
        'time_span': time_span,
        'data_points': len(historical_data)
    }

def cross_domain_mapping(source_domain: str, 
                        target_domain: str,
                        mapping_principles: List[str]) -> Dict[str, Union[float, List[str]]]:
    """
    Відображення між доменами знань.
    
    Параметри:
        source_domain: Вихідний домен
        target_domain: Цільовий домен
        mapping_principles: Принципи відображення
    
    Повертає:
        Словник з результатами відображення
    """
    if not source_domain or not target_domain:
        return {
            'mapping_quality': 0.0,
            'conceptual_distance': 0.0,
            'transfer_potential': 0.0,
            'common_principles': []
        }
    
    # Концептуальна відстань (спрощений підхід)
    # Базується на схожості назв доменів
    if source_domain == target_domain:
        conceptual_distance = 0.0
    else:
        # Проста міра відстані - кількість спільних символів
        common_chars = len(set(source_domain) & set(target_domain))
        total_chars = len(set(source_domain) | set(target_domain))
        conceptual_distance = 1.0 - (common_chars / total_chars) if total_chars > 0 else 1.0
    
    # Якість відображення
    num_principles = len(mapping_principles)
    principle_applicability = min(1.0, num_principles / 10.0)
    
    mapping_quality = (1.0 - conceptual_distance) * 0.6 + principle_applicability * 0.4
    
    # Потенціал перенесення
    transfer_potential = (1.0 - conceptual_distance) * (1.0 + principle_applicability * 0.5)
    
    # Спільні принципи (спрощений підхід)
    common_principles = [p for p in mapping_principles if 'common' in p.lower() or 'universal' in p.lower()]
    
    return {
        'mapping_quality': max(0.0, min(1.0, mapping_quality)),
        'conceptual_distance': conceptual_distance,
        'transfer_potential': min(1.0, transfer_potential),
        'common_principles': common_principles,
        'source_domain': source_domain,
        'target_domain': target_domain
    }

def interdisciplinary_collaboration_network(researchers: List[Dict[str, Any]], 
                                          collaboration_strengths: List[float],
                                          knowledge_domains: List[str]) -> Dict[str, Union[float, str]]:
    """
    Мережа міждисциплінрної співпраці.
    
    Параметри:
        researchers: Список дослідників з атрибутами
        collaboration_strengths: Сили співпраці
        knowledge_domains: Домени знань
    
    Повертає:
        Словник з характеристиками мережі співпраці
    """
    if not researchers:
        return {
            'network_density': 0.0,
            'interdisciplinary_connectivity': 0.0,
            'knowledge_flow': 0.0,
            'collaboration_efficiency': 0.0
        }
    
    num_researchers = len(researchers)
    
    # Густина мережі співпраці
    if collaboration_strengths:
        avg_collaboration = sum(collaboration_strengths) / len(collaboration_strengths)
        network_density = min(1.0, avg_collaboration)
    else:
        network_density = 0.0
    
    # Міждисциплінрна зв'язність
    # Аналізуємо різноманітність доменів знань серед дослідників
    if all('domains' in researcher for researcher in researchers):
        all_domains = set()
        for researcher in researchers:
            all_domains.update(researcher['domains'])
        
        domain_diversity = len(all_domains) / (len(knowledge_domains) or 1)
        interdisciplinary_connectivity = domain_diversity * network_density
    else:
        interdisciplinary_connectivity = network_density * 0.5
    
    # Потік знань
    knowledge_flow = interdisciplinary_connectivity * (1.0 + len(knowledge_domains) / 20.0)
    
    # Ефективність співпраці
    collaboration_efficiency = network_density * (1.0 - abs(interdisciplinary_connectivity - 0.5))
    
    # Тип мережі
    if interdisciplinary_connectivity > 0.7:
        network_type = "Високо міждисциплінрна"
    elif interdisciplinary_connectivity > 0.4:
        network_type = "Середньо міждисциплінрна"
    else:
        network_type = "Низько міждисциплінрна"
    
    return {
        'network_density': network_density,
        'interdisciplinary_connectivity': min(1.0, interdisciplinary_connectivity),
        'knowledge_flow': min(1.0, knowledge_flow),
        'collaboration_efficiency': collaboration_efficiency,
        'network_type': network_type,
        'num_researchers': num_researchers,
        'num_domains': len(knowledge_domains)
    }

def interdisciplinary_research_framework(domains: List[str], 
                                       methodologies: List[str],
                                       tools: List[str]) -> Dict[str, Union[float, str]]:
    """
    Фреймворк міждисциплінрних досліджень.
    
    Параметри:
        domains: Наукові домени
        methodologies: Методології
        tools: Інструменти
    
    Повертає:
        Словник з характеристиками фреймворку
    """
    if not domains:
        return {
            'framework_completeness': 0.0,
            'methodological_rigor': 0.0,
            'tool_integration': 0.0,
            'framework_maturity': 0.0
        }
    
    # Повнота фреймворку
    domain_coverage = min(1.0, len(domains) / 10.0)
    methodological_coverage = min(1.0, len(methodologies) / 15.0)
    tool_coverage = min(1.0, len(tools) / 20.0)
    
    framework_completeness = (domain_coverage * 0.4 + 
                             methodological_coverage * 0.3 + 
                             tool_coverage * 0.3)
    
    # Методологічна строгость
    # Базується на різноманітності методологій
    methodological_rigor = methodological_coverage * (1.0 + len(set(methodologies)) / len(methodologies) if methodologies else 0)
    
    # Інтеграція інструментів
    # Базується на різноманітності та сумісності інструментів
    tool_integration = tool_coverage * (1.0 - abs(tool_coverage - 0.5))
    
    # Зрілість фреймворку
    framework_maturity = (framework_completeness + methodological_rigor + tool_integration) / 3.0
    
    # Рівень фреймворку
    if framework_maturity > 0.8:
        framework_level = "Високий"
    elif framework_maturity > 0.5:
        framework_level = "Середній"
    else:
        framework_level = "Низький"
    
    return {
        'framework_completeness': max(0.0, min(1.0, framework_completeness)),
        'methodological_rigor': max(0.0, min(1.0, methodological_rigor)),
        'tool_integration': max(0.0, min(1.0, tool_integration)),
        'framework_maturity': max(0.0, min(1.0, framework_maturity)),
        'framework_level': framework_level
    }