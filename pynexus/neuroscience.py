"""
Модуль для обчислювальної нейронауки
Computational Neuroscience Module
"""
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
import math

# Нейронаукові константи
# Neuroscience constants
NEURON_RESTING_POTENTIAL = -70e-3  # Потенціал спокою нейрона (В)
NEURON_SPIKE_THRESHOLD = -55e-3  # Поріг спайку (В)
NEURON_ACTION_POTENTIAL = 100e-3  # Амплітуда потенціалу дії (В)
MEMBRANE_CAPACITANCE = 1e-9  # Ємність мембрани (Ф)
MEMBRANE_RESISTANCE = 1e8  # Опір мембрани (Ом)
TIME_CONSTANT = MEMBRANE_RESISTANCE * MEMBRANE_CAPACITANCE  # Часова константа (с)
AXON_LENGTH = 1e-3  # Довжина аксона (м)
AXON_DIAMETER = 1e-6  # Діаметр аксона (м)
CONDUCTION_VELOCITY = 100  # Швидкість проведення (м/с)
SYNAPTIC_DELAY = 1e-3  # Синаптична затримка (с)
SYNAPTIC_STRENGTH = 1e-9  # Синаптична сила (См)
AVOGADRO_CONSTANT = 6.02214076e23  # Число Авогадро (1/моль)
GAS_CONSTANT = 8.31446261815324  # Універсальна газова стала (Дж/(моль·К))
FARADAY_CONSTANT = 96485.33212  # Константа Фарадея (Кл/моль)
TEMPERATURE = 310.15  # Температура (К), 37°C

def leaky_integrate_and_fire(input_current: float, dt: float, 
                           membrane_potential: float = NEURON_RESTING_POTENTIAL,
                           resistance: float = MEMBRANE_RESISTANCE,
                           capacitance: float = MEMBRANE_CAPACITANCE,
                           spike_threshold: float = NEURON_SPIKE_THRESHOLD) -> Tuple[float, bool]:
    """
    Модель інтегрування з витоком та вогникового спайку.
    
    Параметри:
        input_current: Вхідний струм (А)
        dt: Крок часу (с)
        membrane_potential: Початковий мембранный потенціал (В)
        resistance: Опір мембрани (Ом)
        capacitance: Ємність мембрани (Ф)
        spike_threshold: Поріг спайку (В)
    
    Повертає:
        Кортеж (новий мембранный потенціал, чи відбувся спайк)
    """
    if dt <= 0:
        raise ValueError("Крок часу повинен бути додатнім")
    if capacitance <= 0:
        raise ValueError("Ємність мембрани повинна бути додатньою")
    if resistance <= 0:
        raise ValueError("Опір мембрани повинен бути додатнім")
    
    # Диференціальне рівняння для мембранного потенціалу
    # τ * dV/dt = -V + R * I
    time_constant = resistance * capacitance
    dv_dt = (-membrane_potential + resistance * input_current) / time_constant
    new_potential = membrane_potential + dv_dt * dt
    
    # Перевірка на спайк
    if new_potential >= spike_threshold:
        return NEURON_RESTING_POTENTIAL, True  # Спайк відбувся, скидання потенціалу
    else:
        return new_potential, False  # Спайк не відбувся

def hodgkin_huxley_model(input_current: float, dt: float, t: float,
                        v: float = NEURON_RESTING_POTENTIAL,
                        n: float = 0.3177, m: float = 0.0529, h: float = 0.5961) -> Tuple[float, float, float, float]:
    """
    Модель Ходжкіна-Хакслі для нейрона.
    
    Параметри:
        input_current: Вхідний струм (мкА/см²)
        dt: Крок часу (мс)
        t: Час (мс)
        v: Початковий мембранный потенціал (мВ)
        n: Змінна активації К⁺ каналів
        m: Змінна активації Na⁺ каналів
        h: Змінна інактивації Na⁺ каналів
    
    Повертає:
        Кортеж (мембранный потенціал, n, m, h)
    """
    if dt <= 0:
        raise ValueError("Крок часу повинен бути додатнім")
    
    # Константи моделі Ходжкіна-Хакслі
    C_m = 1.0  # Ємність мембрани (мкФ/см²)
    g_K = 36.0  # Провідність К⁺ (мСм/см²)
    g_Na = 120.0  # Провідність Na⁺ (мСм/см²)
    g_L = 0.3  # Провідність витоку (мСм/см²)
    E_K = -12.0  # Рівноважний потенціал К⁺ (мВ)
    E_Na = 115.0  # Рівноважний потенціал Na⁺ (мВ)
    E_L = 10.613  # Рівноважний потенціал витоку (мВ)
    
    # Струми через іонні канали
    I_K = g_K * (n**4) * (v - E_K)
    I_Na = g_Na * (m**3) * h * (v - E_Na)
    I_L = g_L * (v - E_L)
    
    # Диференціальне рівняння для мембранного потенціалу
    dv_dt = (input_current - I_K - I_Na - I_L) / C_m
    
    # Рівняння для змінних ворот
    alpha_n = (0.01 * (v + 10)) / (math.exp((v + 10) / 10) - 1) if v != -10 else 0.1
    beta_n = 0.125 * math.exp(v / 80)
    dn_dt = alpha_n * (1 - n) - beta_n * n
    
    alpha_m = (0.1 * (v + 25)) / (math.exp((v + 25) / 10) - 1) if v != -25 else 1.0
    beta_m = 4 * math.exp(v / 18)
    dm_dt = alpha_m * (1 - m) - beta_m * m
    
    alpha_h = 0.07 * math.exp(v / 20)
    beta_h = 1 / (math.exp((v + 30) / 10) + 1)
    dh_dt = alpha_h * (1 - h) - beta_h * h
    
    # Оновлення значень
    v_new = v + dv_dt * dt
    n_new = n + dn_dt * dt
    m_new = m + dm_dt * dt
    h_new = h + dh_dt * dt
    
    return v_new, n_new, m_new, h_new

def synaptic_transmission(pre_spike: bool, synaptic_strength: float = SYNAPTIC_STRENGTH,
                         delay: float = SYNAPTIC_DELAY) -> float:
    """
    Модель синаптичної передачі.
    
    Параметри:
        pre_spike: Чи відбувся спайк у пресинаптичному нейроні
        synaptic_strength: Сила синапсу (См)
        delay: Синаптична затримка (с)
    
    Повертає:
        Синаптичний струм (А)
    """
    if synaptic_strength < 0:
        raise ValueError("Сила синапсу повинна бути невід'ємною")
    if delay < 0:
        raise ValueError("Синаптична затримка повинна бути невід'ємною")
    
    if pre_spike:
        return synaptic_strength  # Спрощена модель - миттєвий струм
    else:
        return 0.0

def action_potential_propagation(distance: float, 
                               conduction_velocity: float = CONDUCTION_VELOCITY) -> float:
    """
    Обчислити час поширення потенціалу дії.
    
    Параметри:
        distance: Відстань (м)
        conduction_velocity: Швидкість проведення (м/с)
    
    Повертає:
        Час поширення (с)
    """
    if distance < 0:
        raise ValueError("Відстань повинна бути невід'ємною")
    if conduction_velocity <= 0:
        raise ValueError("Швидкість проведення повинна бути додатньою")
    
    return distance / conduction_velocity

def neural_network_activation(weights: List[float], inputs: List[float], 
                            bias: float, activation_function: str = "sigmoid") -> float:
    """
    Обчислити активацію нейрона в нейронній мережі.
    
    Параметри:
        weights: Ваги зв'язків
        inputs: Вхідні сигнали
        bias: Зміщення
        activation_function: Функція активації ("sigmoid", "relu", "tanh")
    
    Повертає:
        Активація нейрона
    """
    if len(weights) != len(inputs):
        raise ValueError("Кількість ваг повинна дорівнювати кількості входів")
    
    # Обчислення зваженої суми
    weighted_sum = sum(w * x for w, x in zip(weights, inputs)) + bias
    
    # Застосування функції активації
    if activation_function == "sigmoid":
        return 1 / (1 + math.exp(-weighted_sum))
    elif activation_function == "relu":
        return max(0, weighted_sum)
    elif activation_function == "tanh":
        return math.tanh(weighted_sum)
    else:
        raise ValueError(f"Невідома функція активації: {activation_function}")

def ion_channel_conductance(voltage: float, ion_type: str = "Na") -> float:
    """
    Обчислити провідність іонного каналу.
    
    Параметри:
        voltage: Мембранный потенціал (мВ)
        ion_type: Тип іону ("Na", "K", "Ca", "Cl")
    
    Повертає:
        Провідність каналу (См)
    """
    # Максимальні провідності для різних іонів (См/см²)
    max_conductances = {
        "Na": 120e-3,
        "K": 36e-3,
        "Ca": 10e-3,
        "Cl": 3e-3
    }
    
    if ion_type not in max_conductances:
        raise ValueError(f"Невідомий тип іону: {ion_type}")
    
    max_g = max_conductances[ion_type]
    
    # Спрощена модель залежності провідності від напруги
    if ion_type == "Na":
        # Натрієві канали - активація при деполяризації
        return max_g * (1 / (1 + math.exp(-(voltage + 30) / 10)))
    elif ion_type == "K":
        # Калієві канали - активація при деполяризації з затримкою
        return max_g * (1 / (1 + math.exp(-(voltage + 10) / 15)))
    else:
        # Для інших іонів - проста залежність
        return max_g * (1 / (1 + math.exp(-(voltage + 20) / 20)))

def nernst_potential(ion_concentration_out: float, ion_concentration_in: float, 
                    ion_charge: int, temperature: float = TEMPERATURE) -> float:
    """
    Обчислити рівноважний потенціал Нернста.
    
    Параметри:
        ion_concentration_out: Концентрація іону зовні (мМ)
        ion_concentration_in: Концентрація іону всередині (мМ)
        ion_charge: Заряд іону
        temperature: Температура (К)
    
    Повертає:
        Рівноважний потенціал (В)
    """
    if ion_concentration_out <= 0:
        raise ValueError("Концентрація іону зовні повинна бути додатньою")
    if ion_concentration_in <= 0:
        raise ValueError("Концентрація іону всередині повинна бути додатньою")
    if ion_charge == 0:
        raise ValueError("Заряд іону не може дорівнювати нулю")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Рівняння Нернста: E = (RT/zF) * ln([ion]_out / [ion]_in)
    R = GAS_CONSTANT
    F = FARADAY_CONSTANT
    z = ion_charge
    
    nernst_potential = (R * temperature / (z * F)) * math.log(ion_concentration_out / ion_concentration_in)
    return nernst_potential

def goldman_equation(ion_concentrations: Dict[str, Tuple[float, float]], 
                    ion_permeabilities: Dict[str, float], 
                    temperature: float = TEMPERATURE) -> float:
    """
    Обчислити мембранный потенціал за рівнянням Голдмана.
    
    Параметри:
        ion_concentrations: Словник концентрацій іонів (зовні, всередині) (мМ)
        ion_permeabilities: Словник проникності іонів (см/с)
        temperature: Температура (К)
    
    Повертає:
        Мембранный потенціал (В)
    """
    if not ion_concentrations:
        raise ValueError("Словник концентрацій іонів не може бути порожнім")
    if not ion_permeabilities:
        raise ValueError("Словник проникності іонів не може бути порожнім")
    
    # Перевірка наявності проникності для всіх іонів
    for ion in ion_concentrations:
        if ion not in ion_permeabilities:
            raise ValueError(f"Відсутня проникність для іону: {ion}")
        if any(c <= 0 for c in ion_concentrations[ion]):
            raise ValueError(f"Концентрації іону {ion} повинні бути додатніми")
        if ion_permeabilities[ion] < 0:
            raise ValueError(f"Проникність іону {ion} повинна бути невід'ємною")
    
    R = GAS_CONSTANT
    F = FARADAY_CONSTANT
    
    # Чисельник і знаменник для рівняння Голдмана
    numerator = 0
    denominator = 0
    
    for ion, (conc_out, conc_in) in ion_concentrations.items():
        permeability = ion_permeabilities[ion]
        if permeability > 0:
            numerator += permeability * conc_out
            denominator += permeability * conc_in
    
    if denominator == 0:
        return 0.0
    
    # Рівняння Голдмана: V_m = (RT/F) * ln(numerator/denominator)
    membrane_potential = (R * temperature / F) * math.log(numerator / denominator)
    return membrane_potential

def hodgkin_huxley_steady_state(voltage: float, gate_type: str) -> float:
    """
    Обчислити стаціонарне значення змінної ворот у моделі Ходжкіна-Хакслі.
    
    Параметри:
        voltage: Мембранный потенціал (мВ)
        gate_type: Тип ворот ("n", "m", "h")
    
    Повертає:
        Стаціонарне значення змінної ворот
    """
    if gate_type == "n":
        alpha = (0.01 * (voltage + 10)) / (math.exp((voltage + 10) / 10) - 1) if voltage != -10 else 0.1
        beta = 0.125 * math.exp(voltage / 80)
    elif gate_type == "m":
        alpha = (0.1 * (voltage + 25)) / (math.exp((voltage + 25) / 10) - 1) if voltage != -25 else 1.0
        beta = 4 * math.exp(voltage / 18)
    elif gate_type == "h":
        alpha = 0.07 * math.exp(voltage / 20)
        beta = 1 / (math.exp((voltage + 30) / 10) + 1)
    else:
        raise ValueError(f"Невідомий тип ворот: {gate_type}")
    
    if alpha + beta == 0:
        return 0.0
    
    return alpha / (alpha + beta)

def synaptic_plasticity(pre_spike_time: float, post_spike_time: float, 
                       weight: float, learning_rate: float = 0.01) -> float:
    """
    Модель синаптичної пластичності (правило Хебба).
    
    Параметри:
        pre_spike_time: Час спайку пресинаптичного нейрона (с)
        post_spike_time: Час спайку постсинаптичного нейрона (с)
        weight: Початкова вага синапсу
        learning_rate: Швидкість навчання
    
    Повертає:
        Нова вага синапсу
    """
    if learning_rate < 0:
        raise ValueError("Швидкість навчання повинна бути невід'ємною")
    
    # Вікно співпадіння для STDP (спайк-тайм залежної пластичності)
    time_difference = post_spike_time - pre_spike_time
    time_window = 20e-3  # 20 мс
    
    # Функція STDP
    if abs(time_difference) <= time_window:
        if time_difference > 0:
            # Потенціація: пресинаптичний спайк перед постсинаптичним
            delta_weight = learning_rate * math.exp(-time_difference / time_window)
        else:
            # Депресія: постсинаптичний спайк перед пресинаптичним
            delta_weight = -learning_rate * math.exp(time_difference / time_window)
    else:
        delta_weight = 0
    
    new_weight = weight + delta_weight
    return max(0, new_weight)  # Вага не може бути від'ємною

def neural_oscillation(frequency: float, time: float, phase: float = 0) -> float:
    """
    Модель нейрональних осциляцій.
    
    Параметри:
        frequency: Частота осциляцій (Гц)
        time: Час (с)
        phase: Фаза (рад)
    
    Повертає:
        Значення осциляцій
    """
    if frequency < 0:
        raise ValueError("Частота повинна бути невід'ємною")
    
    return math.sin(2 * math.pi * frequency * time + phase)

def brain_network_connectivity(adjacency_matrix: List[List[float]]) -> Dict[str, float]:
    """
    Обчислити параметри зв'язності мозкової мережі.
    
    Параметри:
        adjacency_matrix: Матриця суміжності мозкової мережі
    
    Повертає:
        Словник параметрів зв'язності
    """
    if not adjacency_matrix:
        raise ValueError("Матриця суміжності не може бути порожньою")
    
    n = len(adjacency_matrix)
    if any(len(row) != n for row in adjacency_matrix):
        raise ValueError("Матриця суміжності повинна бути квадратною")
    
    # Перевірка на симетричність (для ненаправлених графів)
    is_symmetric = all(adjacency_matrix[i][j] == adjacency_matrix[j][i] 
                      for i in range(n) for j in range(n))
    
    # Ступені вузлів
    degrees = [sum(row) for row in adjacency_matrix]
    
    # Густина мережі
    total_possible_connections = n * (n - 1)
    if not is_symmetric:
        total_possible_connections = n * (n - 1)
    else:
        total_possible_connections = n * (n - 1) / 2
    
    actual_connections = sum(sum(row) for row in adjacency_matrix) / (2 if is_symmetric else 1)
    density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
    
    # Середній ступінь
    avg_degree = sum(degrees) / n if n > 0 else 0
    
    # Коефіцієнт кластеризації (спрощений)
    clustering_coefficient = 0
    if n > 2:
        triangles = 0
        triplets = 0
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i][j] > 0]
            k = len(neighbors)
            if k >= 2:
                triplets += k * (k - 1) / 2
                for j1 in range(len(neighbors)):
                    for j2 in range(j1 + 1, len(neighbors)):
                        if adjacency_matrix[neighbors[j1]][neighbors[j2]] > 0:
                            triangles += 1
        if triplets > 0:
            clustering_coefficient = triangles / triplets
    
    return {
        "density": density,
        "average_degree": avg_degree,
        "clustering_coefficient": clustering_coefficient,
        "is_connected": is_symmetric
    }

def information_entropy(probabilities: List[float]) -> float:
    """
    Обчислити інформаційну ентропію.
    
    Параметри:
        probabilities: Список ймовірностей
    
    Повертає:
        Інформаційна ентропія (біти)
    """
    if not probabilities:
        raise ValueError("Список ймовірностей не може бути порожнім")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Всі ймовірності повинні бути в діапазоні [0, 1]")
    if abs(sum(probabilities) - 1) > 1e-10:
        raise ValueError("Сума ймовірностей повинна дорівнювати 1")
    
    entropy = 0
    for p in probabilities:
        if p > 0:  # Уникаємо log(0)
            entropy -= p * math.log2(p)
    
    return entropy

def mutual_information(joint_probabilities: List[List[float]], 
                      marginal_x: List[float], 
                      marginal_y: List[float]) -> float:
    """
    Обчислити взаємну інформацію між двома змінними.
    
    Параметри:
        joint_probabilities: Спільні ймовірності P(X,Y)
        marginal_x: Граничні ймовірності P(X)
        marginal_y: Граничні ймовірності P(Y)
    
    Повертає:
        Взаємна інформація (біти)
    """
    if not joint_probabilities:
        raise ValueError("Матриця спільних ймовірностей не може бути порожньою")
    
    rows = len(joint_probabilities)
    cols = len(joint_probabilities[0]) if rows > 0 else 0
    
    if len(marginal_x) != rows:
        raise ValueError("Довжина граничних ймовірностей X повинна дорівнювати кількості рядків")
    if len(marginal_y) != cols:
        raise ValueError("Довжина граничних ймовірностей Y повинна дорівнювати кількості стовпців")
    
    # Перевірка нормалізації
    if abs(sum(sum(row) for row in joint_probabilities) - 1) > 1e-10:
        raise ValueError("Сума спільних ймовірностей повинна дорівнювати 1")
    if abs(sum(marginal_x) - 1) > 1e-10:
        raise ValueError("Сума граничних ймовірностей X повинна дорівнювати 1")
    if abs(sum(marginal_y) - 1) > 1e-10:
        raise ValueError("Сума граничних ймовірностей Y повинна дорівнювати 1")
    
    mutual_info = 0
    for i in range(rows):
        for j in range(cols):
            p_xy = joint_probabilities[i][j]
            p_x = marginal_x[i]
            p_y = marginal_y[j]
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mutual_info += p_xy * math.log2(p_xy / (p_x * p_y))
    
    return mutual_info

def firing_rate(spike_times: List[float], time_window: float) -> float:
    """
    Обчислити частоту спайків нейрона.
    
    Параметри:
        spike_times: Список часів спайків (с)
        time_window: Часове вікно (с)
    
    Повертає:
        Частота спайків (Гц)
    """
    if not spike_times:
        return 0.0
    if time_window <= 0:
        raise ValueError("Часове вікно повинно бути додатнім")
    if any(t < 0 for t in spike_times):
        raise ValueError("Всі часи спайків повинні бути невід'ємними")
    
    return len(spike_times) / time_window

def coefficient_of_variation(spike_intervals: List[float]) -> float:
    """
    Обчислити коефіцієнт варіації інтервалів між спайками.
    
    Параметри:
        spike_intervals: Список інтервалів між спайками (с)
    
    Повертає:
        Коефіцієнт варіації
    """
    if not spike_intervals:
        raise ValueError("Список інтервалів не може бути порожнім")
    if any(interval <= 0 for interval in spike_intervals):
        raise ValueError("Всі інтервали повинні бути додатніми")
    
    mean_interval = sum(spike_intervals) / len(spike_intervals)
    variance = sum((interval - mean_interval) ** 2 for interval in spike_intervals) / len(spike_intervals)
    std_deviation = math.sqrt(variance)
    
    if mean_interval == 0:
        return float('inf')
    
    return std_deviation / mean_interval

def neural_correlation(signal1: List[float], signal2: List[float]) -> float:
    """
    Обчислити кореляцію між двома нейрональними сигналами.
    
    Параметри:
        signal1: Перший сигнал
        signal2: Другий сигнал
    
    Повертає:
        Коефіцієнт кореляції
    """
    if not signal1 or not signal2:
        raise ValueError("Сигнали не можуть бути порожніми")
    if len(signal1) != len(signal2):
        raise ValueError("Сигнали повинні мати однакову довжину")
    
    n = len(signal1)
    if n < 2:
        return 0.0
    
    mean1 = sum(signal1) / n
    mean2 = sum(signal2) / n
    
    numerator = sum((signal1[i] - mean1) * (signal2[i] - mean2) for i in range(n))
    sum_sq1 = sum((signal1[i] - mean1) ** 2 for i in range(n))
    sum_sq2 = sum((signal2[i] - mean2) ** 2 for i in range(n))
    
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def spike_triggered_average(stimulus: List[List[float]], 
                          spike_times: List[int]) -> List[float]:
    """
    Обчислити усереднений стимул, що викликає спайк.
    
    Параметри:
        stimulus: Стимул (часовий рядок)
        spike_times: Часи спайків (індекси)
    
    Повертає:
        Усереднений стимул
    """
    if not stimulus or not spike_times:
        return []
    
    stimulus_length = len(stimulus[0]) if stimulus else 0
    if any(len(s) != stimulus_length for s in stimulus):
        raise ValueError("Всі стимули повинні мати однакову довжину")
    if any(t < 0 or t >= len(stimulus) for t in spike_times):
        raise ValueError("Часи спайків повинні бути в межах стимулу")
    
    # Сума стимулів перед спайками
    sta_sum = [0.0] * stimulus_length
    count = 0
    
    # Вікно усереднення (наприклад, 100 мс перед спайком)
    window_size = min(100, stimulus_length)
    
    for spike_time in spike_times:
        if spike_time >= window_size:
            start_index = spike_time - window_size
            for i in range(window_size):
                sta_sum[i] += stimulus[start_index + i]
            count += 1
    
    if count == 0:
        return [0.0] * window_size
    
    return [s / count for s in sta_sum]

def receptive_field_size(visual_angle: float, distance: float) -> float:
    """
    Обчислити розмір рецептивного поля.
    
    Параметри:
        visual_angle: Візуальний кут (градуси)
        distance: Відстань до об'єкта (м)
    
    Повертає:
        Розмір рецептивного поля (м)
    """
    if visual_angle < 0 or visual_angle > 180:
        raise ValueError("Візуальний кут повинен бути в діапазоні [0, 180]")
    if distance < 0:
        raise ValueError("Відстань повинна бути невід'ємною")
    
    # Розмір = 2 * відстань * tan(візуальний_кут/2)
    return 2 * distance * math.tan(math.radians(visual_angle / 2))

def spatial_frequency_tuning(spatial_frequency: float, preferred_frequency: float, 
                           bandwidth: float = 1.0) -> float:
    """
    Обчислити налаштування на просторову частоту.
    
    Параметри:
        spatial_frequency: Просторова частота (циклів/градус)
        preferred_frequency: Бажана частота (циклів/градус)
        bandwidth: Смуга пропускання
    
    Повертає:
        Ступінь налаштування (0-1)
    """
    if spatial_frequency < 0:
        raise ValueError("Просторова частота повинна бути невід'ємною")
    if preferred_frequency < 0:
        raise ValueError("Бажана частота повинна бути невід'ємною")
    if bandwidth <= 0:
        raise ValueError("Смуга пропускання повинна бути додатньою")
    
    # Гаусівська функція налаштування
    diff = spatial_frequency - preferred_frequency
    tuning = math.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    return tuning

def contrast_sensitivity(contrast: float, threshold: float = 0.01) -> float:
    """
    Обчислити чутливість до контрасту.
    
    Параметри:
        contrast: Контраст (0-1)
        threshold: Поріг чутливості
    
    Повертає:
        Чутливість до контрасту
    """
    if contrast < 0 or contrast > 1:
        raise ValueError("Контраст повинен бути в діапазоні [0, 1]")
    if threshold < 0 or threshold > 1:
        raise ValueError("Поріг чутливості повинен бути в діапазоні [0, 1]")
    
    if contrast < threshold:
        return 0.0
    else:
        # Логарифмічна залежність
        return math.log10(contrast / threshold + 1)

def visual_field_mapping(eccentricity: float, polar_angle: float) -> Tuple[float, float]:
    """
    Відобразити ексцентриситет та полярний кут у координати кори.
    
    Параметри:
        eccentricity: Ексцентриситет (градуси)
        polar_angle: Полярний кут (градуси)
    
    Повертає:
        Кортеж (x, y) координат кори
    """
    if eccentricity < 0:
        raise ValueError("Ексцентриситет повинен бути невід'ємним")
    
    # Спрощене логарифмічне відображення
    if eccentricity == 0:
        return (0.0, 0.0)
    
    # Логарифмічне відображення ексцентриситету
    log_ecc = math.log(eccentricity + 1)
    
    # Полярне відображення
    rad_angle = math.radians(polar_angle)
    x = log_ecc * math.cos(rad_angle)
    y = log_ecc * math.sin(rad_angle)
    
    return (x, y)

def orientation_selectivity(angle: float, preferred_angle: float, 
                          bandwidth: float = 30.0) -> float:
    """
    Обчислити селективність до орієнтації.
    
    Параметри:
        angle: Кут орієнтації (градуси)
        preferred_angle: Бажана орієнтація (градуси)
        bandwidth: Смуга пропускання (градуси)
    
    Повертає:
        Ступінь селективності (0-1)
    """
    if bandwidth <= 0:
        raise ValueError("Смуга пропускання повинна бути додатньою")
    
    # Різниця кутів (з урахуванням періодичності)
    diff = abs(angle - preferred_angle)
    diff = min(diff, 180 - diff)  # Максимальна різниця 180°
    
    # Гаусівська функція селективності
    selectivity = math.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    return selectivity

def temporal_frequency_tuning(temporal_frequency: float, preferred_frequency: float, 
                            bandwidth: float = 2.0) -> float:
    """
    Обчислити налаштування на тимчасову частоту.
    
    Параметри:
        temporal_frequency: Тимчасова частота (Гц)
        preferred_frequency: Бажана частота (Гц)
        bandwidth: Смуга пропускання (Гц)
    
    Повертає:
        Ступінь налаштування (0-1)
    """
    if temporal_frequency < 0:
        raise ValueError("Тимчасова частота повинна бути невід'ємною")
    if preferred_frequency < 0:
        raise ValueError("Бажана частота повинна бути невід'ємною")
    if bandwidth <= 0:
        raise ValueError("Смуга пропускання повинна бути додатньою")
    
    # Гаусівська функція налаштування
    diff = temporal_frequency - preferred_frequency
    tuning = math.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    return tuning

def neural_decoding(neural_responses: List[float], 
                   tuning_curves: List[Callable[[float], float]]) -> float:
    """
    Декодувати стимул з нейрональних відповідей.
    
    Параметри:
        neural_responses: Відповіді нейронів
        tuning_curves: Криві налаштування нейронів
    
    Повертає:
        Декодований стимул
    """
    if not neural_responses or not tuning_curves:
        raise ValueError("Списки відповідей та кривих налаштування не можуть бути порожніми")
    if len(neural_responses) != len(tuning_curves):
        raise ValueError("Кількість відповідей повинна дорівнювати кількості кривих налаштування")
    
    # Спрощене декодування методом максимальної правдоподібності
    # Знаходимо стимул, який максимізує ймовірність відповідей
    
    best_stimulus = 0
    max_likelihood = -float('inf')
    
    # Пошук по дискретній сітці стимулів
    for stimulus in [i * 0.1 for i in range(100)]:
        likelihood = 0
        for response, curve in zip(neural_responses, tuning_curves):
            predicted_response = curve(stimulus)
            # Гаусівська правдоподібність
            if predicted_response > 0:
                likelihood -= (response - predicted_response) ** 2 / (2 * predicted_response ** 2)
        
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_stimulus = stimulus
    
    return best_stimulus

def population_coding(neural_responses: List[float], 
                     preferred_stimuli: List[float]) -> float:
    """
    Обчислити кодування популяцією нейронів.
    
    Параметри:
        neural_responses: Відповіді нейронів
        preferred_stimuli: Бажані стимули для кожного нейрона
    
    Повертає:
        Декодований стимул
    """
    if not neural_responses or not preferred_stimuli:
        raise ValueError("Списки відповідей та бажаних стимулів не можуть бути порожніми")
    if len(neural_responses) != len(preferred_stimuli):
        raise ValueError("Списки повинні мати однакову довжину")
    if any(r < 0 for r in neural_responses):
        raise ValueError("Відповіді нейронів повинні бути невід'ємними")
    
    total_response = sum(neural_responses)
    if total_response == 0:
        return 0.0
    
    # Зважене середнє бажаних стимулів
    decoded_stimulus = sum(r * s for r, s in zip(neural_responses, preferred_stimuli)) / total_response
    return decoded_stimulus

def neural_noise(noise_level: float, signal: float) -> float:
    """
    Додати шум до нейронального сигналу.
    
    Параметри:
        noise_level: Рівень шуму
        signal: Оригінальний сигнал
    
    Повертає:
        Сигнал з шумом
    """
    if noise_level < 0:
        raise ValueError("Рівень шуму повинен бути невід'ємним")
    
    # Гаусівський шум
    import random
    noise = random.gauss(0, noise_level)
    return signal + noise

def synaptic_integration(dendritic_inputs: List[Tuple[float, float]], 
                       time_constant: float = TIME_CONSTANT) -> float:
    """
    Інтегрувати синаптичні вхідні сигнали.
    
    Параметри:
        dendritic_inputs: Список кортежів (сила синапсу, час)
        time_constant: Часова константа (с)
    
    Повертає:
        Інтегрований сигнал
    """
    if time_constant <= 0:
        raise ValueError("Часова константа повинна бути додатньою")
    
    # Поточний час (відносний)
    current_time = 0
    integrated_signal = 0
    
    for synapse_strength, synapse_time in dendritic_inputs:
        if synapse_strength < 0:
            raise ValueError("Сила синапсу повинна бути невід'ємною")
        if synapse_time < 0:
            raise ValueError("Час синапсу повинен бути невід'ємним")
        
        # Експоненційне згасання сигналу
        time_diff = current_time - synapse_time
        decay_factor = math.exp(-time_diff / time_constant) if time_diff >= 0 else 0
        integrated_signal += synapse_strength * decay_factor
    
    return integrated_signal

def neural_field_dynamics(activity: List[float], connectivity: List[List[float]], 
                         external_input: List[float], dt: float) -> List[float]:
    """
    Обчислити динаміку нейронального поля.
    
    Параметри:
        activity: Поточна активність поля
        connectivity: Матриця зв'язності
        external_input: Зовнішні вхідні сигнали
        dt: Крок часу
    
    Повертає:
        Нова активність поля
    """
    if not activity or not connectivity or not external_input:
        raise ValueError("Всі списки повинні бути непорожніми")
    
    n = len(activity)
    if len(connectivity) != n or any(len(row) != n for row in connectivity):
        raise ValueError("Матриця зв'язності повинна бути квадратною та відповідати розміру активності")
    if len(external_input) != n:
        raise ValueError("Зовнішній вхід повинен відповідати розміру активності")
    if dt <= 0:
        raise ValueError("Крок часу повинен бути додатнім")
    
    # Динаміка: dA/dt = -A + W * A + I
    new_activity = []
    for i in range(n):
        # Сума зв'язаних активностей
        weighted_sum = sum(connectivity[i][j] * activity[j] for j in range(n))
        # Динаміка
        dA_dt = -activity[i] + weighted_sum + external_input[i]
        new_activity.append(activity[i] + dA_dt * dt)
    
    return new_activity

def spike_train_distance(train1: List[float], train2: List[float], 
                       cost_spike: float = 1.0, cost_time: float = 1.0) -> float:
    """
    Обчислити відстань між двома спайковими поїздами (відстань Вікрама).
    
    Параметри:
        train1: Перший спайковий поїзд (часи спайків)
        train2: Другий спайковий поїзд (часи спайків)
        cost_spike: Вартість вставки/видалення спайку
        cost_time: Вартість зсуву у часі
    
    Повертає:
        Відстань між спайковими поїздами
    """
    if any(t < 0 for t in train1 + train2):
        raise ValueError("Всі часи спайків повинні бути невід'ємними")
    if cost_spike < 0 or cost_time < 0:
        raise ValueError("Вартості повинні бути невід'ємними")
    
    # Спрощена реалізація відстані Вікрама
    # Для повної реалізації потрібна динамічна програмування
    
    # Відстань як сума абсолютних різниць
    # (це спрощення для демонстрації)
    if not train1 and not train2:
        return 0.0
    if not train1:
        return len(train2) * cost_spike
    if not train2:
        return len(train1) * cost_spike
    
    # Спрощене порівняння
    max_len = max(len(train1), len(train2))
    min_len = min(len(train1), len(train2))
    
    # Вартість вставки/видалення
    spike_cost = abs(len(train1) - len(train2)) * cost_spike
    
    # Вартість зсуву у часі для відповідних спайків
    time_cost = 0
    for i in range(min_len):
        time_cost += abs(train1[i] - train2[i]) * cost_time
    
    return spike_cost + time_cost

def neural_manifold_dimensionality(activity_patterns: List[List[float]]) -> int:
    """
    Оцінити розмірність нейронального многовиду.
    
    Параметри:
        activity_patterns: Список патернів активності
    
    Повертає:
        Оцінка розмірності многовиду
    """
    if not activity_patterns:
        raise ValueError("Список патернів активності не може бути порожнім")
    
    # Перевірка однакової довжини патернів
    pattern_length = len(activity_patterns[0])
    if any(len(pattern) != pattern_length for pattern in activity_patterns):
        raise ValueError("Всі патерни повинні мати однакову довжину")
    
    # Спрощена оцінка розмірності через SVD
    # Для реалізації потрібна матрична алгебра
    
    # Кількість патернів
    n_patterns = len(activity_patterns)
    
    # Максимальна можлива розмірність
    max_dimension = min(n_patterns, pattern_length)
    
    # Спрощена оцінка - логарифмічна залежність
    if n_patterns <= 1:
        return 1
    
    estimated_dimension = int(math.log(n_patterns) * math.log(pattern_length))
    return max(1, min(max_dimension, estimated_dimension))

def consciousness_integration(Phi: float, information: float) -> float:
    """
    Обчислити інтеграцію інформації як міру свідомості (на основі IIT).
    
    Параметри:
        Phi: Інтегрована інформація
        information: Загальна інформація
    
    Повертає:
        Міра свідомості
    """
    if Phi < 0 or information < 0:
        raise ValueError("Параметри повинні бути невід'ємними")
    
    # Спрощена модель IIT (Інтегрована інформаційна теорія)
    # Свідомість пропорційна інтегрованій інформації
    return Phi * information

def neural_entropy_production(activity: List[float], 
                            transition_matrix: List[List[float]]) -> float:
    """
    Обчислити ентропію, що виробляється нейрональною системою.
    
    Параметри:
        activity: Стан активності
        transition_matrix: Матриця переходів
    
    Повертає:
        Ентропія, що виробляється (бит/с)
    """
    if not activity or not transition_matrix:
        raise ValueError("Списки не можуть бути порожніми")
    
    n = len(activity)
    if len(transition_matrix) != n or any(len(row) != n for row in transition_matrix):
        raise ValueError("Матриця переходів повинна бути квадратною")
    
    # Перевірка нормалізації ймовірностей
    for row in transition_matrix:
        if abs(sum(row) - 1) > 1e-10:
            raise ValueError("Рядки матриці переходів повинні сумуватися до 1")
    
    # Ентропія виробництва: Σ P(i) Σ P(j|i) log P(j|i)
    entropy_production = 0
    for i in range(n):
        for j in range(n):
            if transition_matrix[i][j] > 0 and activity[i] > 0:
                entropy_production += activity[i] * transition_matrix[i][j] * math.log2(transition_matrix[i][j])
    
    return -entropy_production

def brain_energy_consumption(neuron_count: int, firing_rate: float, 
                           synapse_count: int) -> float:
    """
    Оцінити енергоспоживання мозком.
    
    Параметри:
        neuron_count: Кількість нейронів
        firing_rate: Частота спайків (Гц)
        synapse_count: Кількість синапсів
    
    Повертає:
        Енергоспоживання (Вт)
    """
    if neuron_count < 0 or firing_rate < 0 or synapse_count < 0:
        raise ValueError("Всі параметри повинні бути невід'ємними")
    
    # Енергія на один спайк (приблизно 5e-11 Дж)
    energy_per_spike = 5e-11
    
    # Енергія на синаптичну передачу (приблизно 1e-12 Дж)
    energy_per_synapse = 1e-12
    
    # Спайкова енергія
    spike_energy = neuron_count * firing_rate * energy_per_spike
    
    # Синаптична енергія
    synaptic_energy = synapse_count * firing_rate * energy_per_synapse
    
    # Загальна енергія
    total_energy = spike_energy + synaptic_energy
    
    return total_energy

def neural_coding_efficiency(information_rate: float, energy_consumption: float) -> float:
    """
    Обчислити ефективність нейронального кодування.
    
    Параметри:
        information_rate: Швидкість передачі інформації (біт/с)
        energy_consumption: Споживання енергії (Вт)
    
    Повертає:
        Ефективність кодування (біт/Дж)
    """
    if information_rate < 0:
        raise ValueError("Швидкість передачі інформації повинна бути невід'ємною")
    if energy_consumption <= 0:
        raise ValueError("Споживання енергії повинне бути додатнім")
    
    return information_rate / energy_consumption

def cognitive_load(processing_demand: float, available_resources: float) -> float:
    """
    Обчислити когнітивне навантаження.
    
    Параметри:
        processing_demand: Потреба в обробці
        available_resources: Доступні ресурси
    
    Повертає:
        Когнітивне навантаження (0-1)
    """
    if processing_demand < 0:
        raise ValueError("Потреба в обробці повинна бути невід'ємною")
    if available_resources <= 0:
        raise ValueError("Доступні ресурси повинні бути додатніми")
    
    load = processing_demand / available_resources
    return min(1.0, load)  # Обмеження зверху

def attention_allocation(stimulus_salience: List[float], 
                        attention_weights: List[float]) -> List[float]:
    """
    Розподілити увагу між стимулами.
    
    Параметри:
        stimulus_salience: Виразність стимулів
        attention_weights: Ваги уваги
    
    Повертає:
        Розподіл уваги
    """
    if not stimulus_salience or not attention_weights:
        raise ValueError("Списки не можуть бути порожніми")
    if len(stimulus_salience) != len(attention_weights):
        raise ValueError("Списки повинні мати однакову довжину")
    if any(s < 0 for s in stimulus_salience):
        raise ValueError("Виразність стимулів повинна бути невід'ємною")
    if any(w < 0 for w in attention_weights):
        raise ValueError("Ваги уваги повинні бути невід'ємними")
    
    # Комбінована салієнтність
    combined_salience = [s * w for s, w in zip(stimulus_salience, attention_weights)]
    
    # Нормалізація
    total_salience = sum(combined_salience)
    if total_salience == 0:
        return [1/len(combined_salience)] * len(combined_salience)
    
    return [s / total_salience for s in combined_salience]

def memory_decay(time_since_encoding: float, decay_rate: float = 0.1) -> float:
    """
    Обчислити зменшення пам'яті з часом.
    
    Параметри:
        time_since_encoding: Час з моменту кодування (с)
        decay_rate: Швидкість зменшення
    
    Повертає:
        Рівень пам'яті (0-1)
    """
    if time_since_encoding < 0:
        raise ValueError("Час з моменту кодування повинен бути невід'ємним")
    if decay_rate < 0:
        raise ValueError("Швидкість зменшення повинна бути невід'ємною")
    
    return math.exp(-decay_rate * time_since_encoding)

def learning_rate_adaptation(performance_error: float, 
                           previous_error: float, 
                           current_learning_rate: float,
                           adaptation_rate: float = 0.01) -> float:
    """
    Адаптувати швидкість навчання на основі помилки.
    
    Параметри:
        performance_error: Поточна помилка
        previous_error: Попередня помилка
        current_learning_rate: Поточна швидкість навчання
        adaptation_rate: Швидкість адаптації
    
    Повертає:
        Нова швидкість навчання
    """
    if current_learning_rate <= 0:
        raise ValueError("Поточна швидкість навчання повинна бути додатньою")
    if adaptation_rate < 0:
        raise ValueError("Швидкість адаптації повинна бути невід'ємною")
    
    # Якщо помилка зменшується, збільшуємо швидкість навчання
    # Якщо помилка збільшується, зменшуємо швидкість навчання
    error_change = performance_error - previous_error
    
    if error_change < 0:  # Помилка зменшується
        new_rate = current_learning_rate * (1 + adaptation_rate)
    else:  # Помилка збільшується або не змінюється
        new_rate = current_learning_rate * (1 - adaptation_rate)
    
    # Обмеження діапазону
    return max(1e-6, min(1.0, new_rate))

def neural_network_pruning(connection_strengths: List[float], 
                          pruning_threshold: float) -> List[bool]:
    """
    Виконати обрізку слабких зв'язків у нейронній мережі.
    
    Параметри:
        connection_strengths: Сили зв'язків
        pruning_threshold: Поріг обрізки
    
    Повертає:
        Список булевих значень (True - зберегти, False - обрізати)
    """
    if not connection_strengths:
        raise ValueError("Список сил зв'язків не може бути порожнім")
    if pruning_threshold < 0:
        raise ValueError("Поріг обрізки повинен бути невід'ємним")
    
    return [strength >= pruning_threshold for strength in connection_strengths]

def synaptic_scaling(synaptic_weights: List[float], 
                    target_activity: float, 
                    current_activity: float,
                    scaling_factor: float = 0.01) -> List[float]:
    """
    Масштабувати синаптичні ваги для підтримки цільової активності.
    
    Параметри:
        synaptic_weights: Синаптичні ваги
        target_activity: Цільова активність
        current_activity: Поточна активність
        scaling_factor: Фактор масштабування
    
    Повертає:
        Нові синаптичні ваги
    """
    if not synaptic_weights:
        raise ValueError("Список синаптичних ваг не може бути порожнім")
    if target_activity < 0 or current_activity < 0:
        raise ValueError("Активності повинні бути невід'ємними")
    if scaling_factor < 0:
        raise ValueError("Фактор масштабування повинен бути невід'ємним")
    
    if current_activity == 0:
        return synaptic_weights[:]  # Немає чого масштабувати
    
    # Відношення цільової до поточної активності
    activity_ratio = target_activity / current_activity
    
    # Корекція ваг
    adjustment = scaling_factor * (activity_ratio - 1)
    
    # Масштабування ваг
    new_weights = [max(0, w * (1 + adjustment)) for w in synaptic_weights]
    
    return new_weights

def neural_differentiation(activity_patterns: List[List[float]]) -> float:
    """
    Обчислити різноманітність нейрональних патернів.
    
    Параметри:
        activity_patterns: Список патернів активності
    
    Повертає:
        Міра диференціації (0-1)
    """
    if not activity_patterns:
        raise ValueError("Список патернів активності не може бути порожнім")
    
    # Перевірка однакової довжини патернів
    pattern_length = len(activity_patterns[0])
    if any(len(pattern) != pattern_length for pattern in activity_patterns):
        raise ValueError("Всі патерни повинні мати однакову довжину")
    
    if len(activity_patterns) < 2:
        return 0.0
    
    # Обчислення попарних відстаней між патернами
    distances = []
    for i in range(len(activity_patterns)):
        for j in range(i + 1, len(activity_patterns)):
            # Евклідова відстань
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(activity_patterns[i], activity_patterns[j])))
            distances.append(dist)
    
    if not distances:
        return 0.0
    
    # Середня відстань
    avg_distance = sum(distances) / len(distances)
    
    # Максимально можлива відстань (для нормалізації)
    max_possible_distance = math.sqrt(pattern_length)  # При максимальній різниці
    
    if max_possible_distance == 0:
        return 0.0
    
    return min(1.0, avg_distance / max_possible_distance)

def neural_integration(connectivity_matrix: List[List[float]]) -> float:
    """
    Обчислити інтеграцію нейрональної мережі.
    
    Параметри:
        connectivity_matrix: Матриця зв'язності
    
    Повертає:
        Міра інтеграції (0-1)
    """
    if not connectivity_matrix:
        raise ValueError("Матриця зв'язності не може бути порожньою")
    
    n = len(connectivity_matrix)
    if any(len(row) != n for row in connectivity_matrix):
        raise ValueError("Матриця зв'язності повинна бути квадратною")
    
    if n < 2:
        return 0.0
    
    # Спрощена міра інтеграції - густина зв'язності
    total_possible_connections = n * (n - 1)
    actual_connections = sum(sum(1 for w in row if w > 0) for row in connectivity_matrix) - n  # Віднімаємо діагональ
    
    if total_possible_connections == 0:
        return 0.0
    
    return actual_connections / total_possible_connections

def consciousness_measure(integration: float, information: float, 
                        differentiation: float) -> float:
    """
    Обчислити міру свідомості (на основі IIT).
    
    Параметри:
        integration: Інтеграція
        information: Інформація
        differentiation: Диференціація
    
    Повертає:
        Міра свідомості (0-1)
    """
    if not all(0 <= x <= 1 for x in [integration, information, differentiation]):
        raise ValueError("Всі параметри повинні бути в діапазоні [0, 1]")
    
    # Φ = інтеграція × інформація × диференціація
    return integration * information * differentiation

def neural_plasticity_rule(pre_activity: float, post_activity: float, 
                          weight: float, learning_rate: float = 0.01) -> float:
    """
    Правило пластичності нейронів (спрощене STDP).
    
    Параметри:
        pre_activity: Активність пресинаптичного нейрона
        post_activity: Активність постсинаптичного нейрона
        weight: Поточна вага синапсу
        learning_rate: Швидкість навчання
    
    Повертає:
        Нова вага синапсу
    """
    if not all(0 <= x <= 1 for x in [pre_activity, post_activity]):
        raise ValueError("Активності повинні бути в діапазоні [0, 1]")
    if weight < 0:
        raise ValueError("Вага синапсу повинна бути невід'ємною")
    if learning_rate < 0:
        raise ValueError("Швидкість навчання повинна бути невід'ємною")
    
    # Спрощене правило STDP
    # Якщо пресинаптичний нейрон активний перед постсинаптичним - потенціація
    # Якщо постсинаптичний нейрон активний перед пресинаптичним - депресія
    if pre_activity > post_activity:
        delta_weight = learning_rate * (1 - weight)  # Потенціація
    else:
        delta_weight = -learning_rate * weight  # Депресія
    
    new_weight = weight + delta_weight
    return max(0, min(1, new_weight))  # Обмеження діапазону [0, 1]

def cognitive_flexibility(task_switch_cost: float, switch_frequency: float) -> float:
    """
    Обчислити когнітивну гнучкість.
    
    Параметри:
        task_switch_cost: Вартість перемикання між завданнями
        switch_frequency: Частота перемикання
    
    Повертає:
        Міра когнітивної гнучкості (0-1)
    """
    if task_switch_cost < 0:
        raise ValueError("Вартість перемикання повинна бути невід'ємною")
    if switch_frequency < 0:
        raise ValueError("Частота перемикання повинна бути невід'ємною")
    
    # Когнітивна гнучкість обернено пропорційна вартості перемикання
    if task_switch_cost == 0:
        return 1.0
    
    flexibility = 1 / (1 + task_switch_cost * switch_frequency)
    return flexibility

def neural_robustness(activity_patterns: List[List[float]], 
                     noise_level: float) -> float:
    """
    Обчислити стійкість нейрональної системи до шуму.
    
    Параметри:
        activity_patterns: Список патернів активності
        noise_level: Рівень шуму
    
    Повертає:
        Міра стійкості (0-1)
    """
    if not activity_patterns:
        raise ValueError("Список патернів активності не може бути порожнім")
    if noise_level < 0:
        raise ValueError("Рівень шуму повинен бути невід'ємним")
    
    # Перевірка однакової довжини патернів
    pattern_length = len(activity_patterns[0])
    if any(len(pattern) != pattern_length for pattern in activity_patterns):
        raise ValueError("Всі патерни повинні мати однакову довжину")
    
    if len(activity_patterns) < 2:
        return 1.0  # Немає чого порівнювати
    
    # Спрощена міра стійкості - відношення сигнал/шум
    # Тут ми припускаємо, що шум додається до патернів
    
    # Середній патерн
    avg_pattern = [sum(pattern[i] for pattern in activity_patterns) / len(activity_patterns) 
                   for i in range(pattern_length)]
    
    # Варіація навколо середнього (сигнал)
    signal_variance = sum(sum((pattern[i] - avg_pattern[i]) ** 2 for i in range(pattern_length)) 
                         for pattern in activity_patterns) / len(activity_patterns)
    
    # Вплив шуму (шум)
    noise_variance = noise_level ** 2
    
    if noise_variance == 0:
        return 1.0 if signal_variance > 0 else 0.0
    
    # Співвідношення сигнал/шум
    snr = signal_variance / noise_variance
    
    # Нормалізація до [0, 1]
    return min(1.0, snr / (1 + snr))

def neural_efficiency(processing_speed: float, energy_consumption: float) -> float:
    """
    Обчислити ефективність нейрональної обробки.
    
    Параметри:
        processing_speed: Швидкість обробки (операцій/с)
        energy_consumption: Споживання енергії (Вт)
    
    Повертає:
        Ефективність (операцій/Дж)
    """
    if processing_speed < 0:
        raise ValueError("Швидкість обробки повинна бути невід'ємною")
    if energy_consumption <= 0:
        raise ValueError("Споживання енергії повинне бути додатнім")
    
    return processing_speed / energy_consumption

def cognitive_reserve(lifetime_learning: float, brain_volume: float, 
                     education_level: float) -> float:
    """
    Обчислити когнітивний резерв.
    
    Параметри:
        lifetime_learning: Навчання протягом життя
        brain_volume: Об'єм мозку
        education_level: Рівень освіти
    
    Повертає:
        Когнітивний резерв
    """
    if lifetime_learning < 0:
        raise ValueError("Навчання протягом життя повинне бути невід'ємним")
    if brain_volume < 0:
        raise ValueError("Об'єм мозку повинен бути невід'ємним")
    if education_level < 0:
        raise ValueError("Рівень освіти повинен бути невід'ємним")
    
    # Спрощена модель: когнітивний резерв пропорційний навчанню, об'єму мозку та освіті
    return lifetime_learning * brain_volume * education_level

def neural_synchronization(oscillation_phases: List[float], 
                          coupling_strength: float) -> float:
    """
    Обчислити синхронізацію нейрональних осциляцій.
    
    Параметри:
        oscillation_phases: Фази осциляцій
        coupling_strength: Сила зв'язку
    
    Повертає:
        Міра синхронізації (0-1)
    """
    if not oscillation_phases:
        raise ValueError("Список фаз не може бути порожнім")
    if coupling_strength < 0:
        raise ValueError("Сила зв'язку повинна бути невід'ємною")
    
    if len(oscillation_phases) < 2:
        return 1.0
    
    # Спрощена міра синхронізації - порядковий параметр
    # R = |Σ exp(iφ)| / N
    
    real_sum = sum(math.cos(phase) for phase in oscillation_phases)
    imag_sum = sum(math.sin(phase) for phase in oscillation_phases)
    
    order_parameter = math.sqrt(real_sum**2 + imag_sum**2) / len(oscillation_phases)
    
    # Нормалізація з урахуванням сили зв'язку
    return min(1.0, order_parameter * (1 + coupling_strength))

def brain_network_small_world(sigma: float, gamma: float) -> float:
    """
    Обчислити міру "малого світу" мозкової мережі.
    
    Параметри:
        sigma: Відношення кластеризації до випадкової мережі
        gamma: Відношення шляху до випадкової мережі
    
    Повертає:
        Міра "малого світу" (0-1)
    """
    if sigma < 0 or gamma < 0:
        raise ValueError("Параметри повинні бути невід'ємними")
    
    # Міра "малого світу" - добуток sigma та 1/gamma
    if gamma == 0:
        return 1.0 if sigma > 0 else 0.0
    
    small_world_measure = sigma / gamma
    return min(1.0, small_world_measure)

def neural_information_capacity(channel_bandwidth: float, 
                               signal_to_noise_ratio: float) -> float:
    """
    Обчислити інформаційну ємність нейронального каналу (теорема Шеннона).
    
    Параметри:
        channel_bandwidth: Смуга пропускання каналу (Гц)
        signal_to_noise_ratio: Відношення сигнал/шум
    
    Повертає:
        Інформаційна ємність (біт/с)
    """
    if channel_bandwidth < 0:
        raise ValueError("Смуга пропускання повинна бути невід'ємною")
    if signal_to_noise_ratio < 0:
        raise ValueError("Відношення сигнал/шум повинне бути невід'ємним")
    
    # Теорема Шеннона: C = B * log2(1 + SNR)
    if signal_to_noise_ratio == 0:
        return 0.0
    
    return channel_bandwidth * math.log2(1 + signal_to_noise_ratio)

def neural_field_pattern_formation(activity: List[float], 
                                  kernel: List[List[float]], 
                                  threshold: float) -> List[bool]:
    """
    Визначити формування патернів у нейрональному полі.
    
    Параметри:
        activity: Активність поля
        kernel: Ядро зв'язності
        threshold: Поріг активації
    
    Повертає:
        Список булевих значень (True - активний, False - неактивний)
    """
    if not activity or not kernel:
        raise ValueError("Списки не можуть бути порожніми")
    
    field_size = len(activity)
    if len(kernel) != field_size or any(len(row) != field_size for row in kernel):
        raise ValueError("Ядро повинно відповідати розміру поля")
    if threshold < 0:
        raise ValueError("Поріг активації повинен бути невід'ємним")
    
    # Конволюція активності з ядром
    pattern_activity = []
    for i in range(field_size):
        activation = sum(kernel[i][j] * activity[j] for j in range(field_size))
        pattern_activity.append(activation)
    
    # Порівняння з порогом
    return [act >= threshold for act in pattern_activity]

def consciousness_integration_complexity(Phi: float, system_size: int) -> float:
    """
    Обчислити складність інтеграції свідомості.
    
    Параметри:
        Phi: Інтегрована інформація
        system_size: Розмір системи
    
    Повертає:
        Складність інтеграції (0-1)
    """
    if Phi < 0:
        raise ValueError("Інтегрована інформація повинна бути невід'ємною")
    if system_size <= 0:
        raise ValueError("Розмір системи повинен бути додатнім")
    
    # Складність пропорційна Phi та обернено пропорційна розміру системи
    complexity = Phi / math.log2(system_size + 1)
    return min(1.0, complexity)

def neural_adaptation(sensory_input: float, adaptation_rate: float, 
                     previous_state: float) -> float:
    """
    Модель нейрональної адаптації до сенсорного вхідного сигналу.
    
    Параметри:
        sensory_input: Сенсорний вхід
        adaptation_rate: Швидкість адаптації
        previous_state: Попередній стан адаптації
    
    Повертає:
        Новий стан адаптації
    """
    if adaptation_rate < 0 or adaptation_rate > 1:
        raise ValueError("Швидкість адаптації повинна бути в діапазоні [0, 1]")
    
    # Експоненційна адаптація
    return previous_state + adaptation_rate * (sensory_input - previous_state)

def cognitive_control(conflict_level: float, control_efficiency: float) -> float:
    """
    Обчислити когнітивний контроль.
    
    Параметри:
        conflict_level: Рівень конфлікту
        control_efficiency: Ефективність контролю
    
    Повертає:
        Рівень когнітивного контролю
    """
    if conflict_level < 0:
        raise ValueError("Рівень конфлікту повинен бути невід'ємним")
    if control_efficiency < 0 or control_efficiency > 1:
        raise ValueError("Ефективність контролю повинна бути в діапазоні [0, 1]")
    
    # Когнітивний контроль зменшує вплив конфлікту
    return max(0, 1 - conflict_level * (1 - control_efficiency))

def neural_resilience(damage_level: float, recovery_capacity: float) -> float:
    """
    Обчислити стійкість нейронної системи до пошкоджень.
    
    Параметри:
        damage_level: Рівень пошкоджень
        recovery_capacity: Здатність до відновлення
    
    Повертає:
        Міра стійкості (0-1)
    """
    if damage_level < 0 or damage_level > 1:
        raise ValueError("Рівень пошкоджень повинен бути в діапазоні [0, 1]")
    if recovery_capacity < 0:
        raise ValueError("Здатність до відновлення повинна бути невід'ємною")
    
    # Стійкість = (1 - рівень_пошкоджень) * здатність_до_відновлення
    resilience = (1 - damage_level) * recovery_capacity
    return min(1.0, resilience)

def neural_diversity(activity_variance: float, optimal_variance: float) -> float:
    """
    Обчислити нейрональну різноманітність.
    
    Параметри:
        activity_variance: Дисперсія активності
        optimal_variance: Оптимальна дисперсія
    
    Повертає:
        Міра різноманітності (0-1)
    """
    if activity_variance < 0:
        raise ValueError("Дисперсія активності повинна бути невід'ємною")
    if optimal_variance <= 0:
        raise ValueError("Оптимальна дисперсія повинна бути додатньою")
    
    # Різноманітність максимальна при оптимальній дисперсії
    if activity_variance == 0:
        return 0.0
    
    # Гаусівська функція різноманітності
    diversity = math.exp(-((activity_variance - optimal_variance) ** 2) / (2 * optimal_variance ** 2))
    return diversity

def cognitive_fusion(sensory_modalities: List[float], 
                    integration_weights: List[float]) -> float:
    """
    Обчислити когнітивне злиття сенсорних модальностей.
    
    Параметри:
        sensory_modalities: Сенсорні модальності
        integration_weights: Ваги інтеграції
    
    Повертає:
        Ступінь когнітивного злиття
    """
    if not sensory_modalities or not integration_weights:
        raise ValueError("Списки не можуть бути порожніми")
    if len(sensory_modalities) != len(integration_weights):
        raise ValueError("Списки повинні мати однакову довжину")
    if any(m < 0 for m in sensory_modalities):
        raise ValueError("Сенсорні модальності повинні бути невід'ємними")
    if any(w < 0 for w in integration_weights):
        raise ValueError("Ваги інтеграції повинні бути невід'ємними")
    
    # Зважене інтеграційне злиття
    total_weight = sum(integration_weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(m * w for m, w in zip(sensory_modalities, integration_weights))
    return weighted_sum / total_weight

def neural_plasticity_potential(age_factor: float, learning_history: float) -> float:
    """
    Обчислити потенціал нейрональної пластичності.
    
    Параметри:
        age_factor: Фактор віку (0-1, де 0 - молодий, 1 - старий)
        learning_history: Історія навчання
    
    Повертає:
        Потенціал пластичності (0-1)
    """
    if age_factor < 0 or age_factor > 1:
        raise ValueError("Фактор віку повинен бути в діапазоні [0, 1]")
    if learning_history < 0:
        raise ValueError("Історія навчання повинна бути невід'ємною")
    
    # Пластичність зменшується з віком, але збільшується з навчанням
    age_effect = 1 - age_factor
    learning_effect = 1 - math.exp(-learning_history / 10)  # Насичення
    
    plasticity = age_effect * (1 + learning_effect)
    return min(1.0, plasticity)

def consciousness_complexity(integrated_information: float, 
                           system_diversity: float) -> float:
    """
    Обчислити складність свідомості.
    
    Параметри:
        integrated_information: Інтегрована інформація (Φ)
        system_diversity: Різноманітність системи
    
    Повертає:
        Складність свідомості
    """
    if integrated_information < 0:
        raise ValueError("Інтегрована інформація повинна бути невід'ємною")
    if system_diversity < 0:
        raise ValueError("Різноманітність системи повинна бути невід'ємною")
    
    # Складність свідомості - добуток інтегрованої інформації та різноманітності
    return integrated_information * system_diversity

def neural_network_criticality(connection_density: float, 
                              critical_density: float = 0.5) -> float:
    """
    Обчислити критичність нейронної мережі.
    
    Параметри:
        connection_density: Густина зв'язків
        critical_density: Критична густина, за замовчуванням 0.5
    
    Повертає:
        Міра критичності (0-1)
    """
    if connection_density < 0:
        raise ValueError("Густина зв'язків повинна бути невід'ємною")
    if critical_density <= 0 or critical_density > 1:
        raise ValueError("Критична густина повинна бути в діапазоні (0, 1]")
    
    # Критичність максимальна при критичній густині
    criticality = math.exp(-((connection_density - critical_density) ** 2) / (2 * critical_density ** 2))
    return criticality

def cognitive_emergence(complexity: float, integration: float) -> float:
    """
    Обчислити когнітивну емергентність.
    
    Параметри:
        complexity: Складність системи
        integration: Інтеграція компонентів
    
    Повертає:
        Міра емергентності
    """
    if complexity < 0:
        raise ValueError("Складність повинна бути невід'ємною")
    if integration < 0:
        raise ValueError("Інтеграція повинна бути невід'ємною")
    
    # Емергентність виникає при високій складності та інтеграції
    emergence = complexity * integration
    return min(1.0, emergence)

def neural_entropy_balance(entropy_production: float, entropy_consumption: float) -> float:
    """
    Обчислити баланс ентропії в нейронній системі.
    
    Параметри:
        entropy_production: Виробництво ентропії
        entropy_consumption: Споживання ентропії
    
    Повертає:
        Баланс ентропії
    """
    if entropy_production < 0:
        raise ValueError("Виробництво ентропії повинне бути невід'ємним")
    if entropy_consumption < 0:
        raise ValueError("Споживання ентропії повинне бути невід'ємним")
    
    # Баланс = споживання - виробництво
    balance = entropy_consumption - entropy_production
    return balance

def consciousness_resonance(frequency: float, resonance_threshold: float) -> float:
    """
    Обчислити резонанс свідомості.
    
    Параметри:
        frequency: Частота осциляцій
        resonance_threshold: Поріг резонансу
    
    Повертає:
        Міра резонансу (0-1)
    """
    if frequency < 0:
        raise ValueError("Частота повинна бути невід'ємною")
    if resonance_threshold <= 0:
        raise ValueError("Поріг резонансу повинен бути додатнім")
    
    # Резонанс максимальний при відповідності частот
    resonance = math.exp(-((frequency - resonance_threshold) ** 2) / (2 * resonance_threshold ** 2))
    return resonance

def neural_field_coherence(field_activities: List[List[float]]) -> float:
    """
    Обчислити когерентність нейронального поля.
    
    Параметри:
        field_activities: Активності в різних точках поля
    
    Повертає:
        Міра когерентності (0-1)
    """
    if not field_activities or not field_activities[0]:
        raise ValueError("Списки активностей не можуть бути порожніми")
    
    # Перевірка однакової довжини
    length = len(field_activities[0])
    if any(len(activity) != length for activity in field_activities):
        raise ValueError("Всі списки активностей повинні мати однакову довжину")
    
    if len(field_activities) < 2:
        return 1.0
    
    # Обчислення когерентності як середньої кореляції між активностями
    correlations = []
    for i in range(len(field_activities)):
        for j in range(i + 1, len(field_activities)):
            # Кореляція між двома активностями
            mean1 = sum(field_activities[i]) / length
            mean2 = sum(field_activities[j]) / length
            
            numerator = sum((field_activities[i][k] - mean1) * (field_activities[j][k] - mean2) 
                           for k in range(length))
            sum_sq1 = sum((field_activities[i][k] - mean1) ** 2 for k in range(length))
            sum_sq2 = sum((field_activities[j][k] - mean2) ** 2 for k in range(length))
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator > 0:
                correlation = abs(numerator / denominator)
                correlations.append(correlation)
    
    if not correlations:
        return 0.0
    
    return sum(correlations) / len(correlations)

def cognitive_emergence_threshold(system_complexity: float, 
                                 integration_level: float) -> bool:
    """
    Визначити, чи досягнуто поріг когнітивної емергентності.
    
    Параметри:
        system_complexity: Складність системи
        integration_level: Рівень інтеграції
    
    Повертає:
        Чи досягнуто поріг емергентності (True/False)
    """
    if system_complexity < 0:
        raise ValueError("Складність системи повинна бути невід'ємною")
    if integration_level < 0:
        raise ValueError("Рівень інтеграції повинен бути невід'ємним")
    
    # Поріг емергентності - добуток складності та інтеграції перевищує критичне значення
    emergence_measure = system_complexity * integration_level
    critical_threshold = 0.5  # Приклад критичного порогу
    
    return emergence_measure >= critical_threshold

def neural_information_integration(activity_patterns: List[List[float]], 
                                  time_windows: List[float]) -> float:
    """
    Обчислити інтеграцію інформації в нейронній системі.
    
    Параметри:
        activity_patterns: Патерни активності в різних часових вікнах
        time_windows: Часові вікна
    
    Повертає:
        Міра інтеграції інформації
    """
    if not activity_patterns or not time_windows:
        raise ValueError("Списки не можуть бути порожніми")
    if len(activity_patterns) != len(time_windows):
        raise ValueError("Кількість патернів повинна відповідати кількості часових вікон")
    
    # Перевірка однакової довжини патернів
    if activity_patterns:
        pattern_length = len(activity_patterns[0])
        if any(len(pattern) != pattern_length for pattern in activity_patterns):
            raise ValueError("Всі патерни повинні мати однакову довжину")
    
    if len(activity_patterns) < 2:
        return 0.0
    
    # Спрощена міра інтеграції - сума взаємної інформації між послідовними патернами
    total_integration = 0
    for i in range(len(activity_patterns) - 1):
        # Обчислення кореляції між послідовними патернами
        pattern1 = activity_patterns[i]
        pattern2 = activity_patterns[i + 1]
        
        if not pattern1 or not pattern2:
            continue
            
        mean1 = sum(pattern1) / len(pattern1)
        mean2 = sum(pattern2) / len(pattern2)
        
        numerator = sum((a - mean1) * (b - mean2) for a, b in zip(pattern1, pattern2))
        sum_sq1 = sum((a - mean1) ** 2 for a in pattern1)
        sum_sq2 = sum((b - mean2) ** 2 for b in pattern2)
        
        denominator = math.sqrt(sum_sq1 * sum_sq2) if sum_sq1 * sum_sq2 > 0 else 1
        
        correlation = abs(numerator / denominator) if denominator > 0 else 0
        total_integration += correlation
    
    return total_integration / (len(activity_patterns) - 1) if len(activity_patterns) > 1 else 0.0"""
Модуль для обчислювальної нейронауки
Computational Neuroscience Module
"""
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
import math

# Нейронаукові константи
# Neuroscience constants
NEURON_RESTING_POTENTIAL = -70e-3  # Потенціал спокою нейрона (В)
NEURON_SPIKE_THRESHOLD = -55e-3  # Поріг спайку (В)
NEURON_ACTION_POTENTIAL = 100e-3  # Амплітуда потенціалу дії (В)
MEMBRANE_CAPACITANCE = 1e-9  # Ємність мембрани (Ф)
MEMBRANE_RESISTANCE = 1e8  # Опір мембрани (Ом)
TIME_CONSTANT = MEMBRANE_RESISTANCE * MEMBRANE_CAPACITANCE  # Часова константа (с)
AXON_LENGTH = 1e-3  # Довжина аксона (м)
AXON_DIAMETER = 1e-6  # Діаметр аксона (м)
CONDUCTION_VELOCITY = 100  # Швидкість проведення (м/с)
SYNAPTIC_DELAY = 1e-3  # Синаптична затримка (с)
SYNAPTIC_STRENGTH = 1e-9  # Синаптична сила (См)
AVOGADRO_CONSTANT = 6.02214076e23  # Число Авогадро (1/моль)
GAS_CONSTANT = 8.31446261815324  # Універсальна газова стала (Дж/(моль·К))
FARADAY_CONSTANT = 96485.33212  # Константа Фарадея (Кл/моль)
TEMPERATURE = 310.15  # Температура (К), 37°C

def leaky_integrate_and_fire(input_current: float, dt: float, 
                           membrane_potential: float = NEURON_RESTING_POTENTIAL,
                           resistance: float = MEMBRANE_RESISTANCE,
                           capacitance: float = MEMBRANE_CAPACITANCE,
                           spike_threshold: float = NEURON_SPIKE_THRESHOLD) -> Tuple[float, bool]:
    """
    Модель інтегрування з витоком та вогникового спайку.
    
    Параметри:
        input_current: Вхідний струм (А)
        dt: Крок часу (с)
        membrane_potential: Початковий мембранный потенціал (В)
        resistance: Опір мембрани (Ом)
        capacitance: Ємність мембрани (Ф)
        spike_threshold: Поріг спайку (В)
    
    Повертає:
        Кортеж (новий мембранный потенціал, чи відбувся спайк)
    """
    if dt <= 0:
        raise ValueError("Крок часу повинен бути додатнім")
    if capacitance <= 0:
        raise ValueError("Ємність мембрани повинна бути додатньою")
    if resistance <= 0:
        raise ValueError("Опір мембрани повинен бути додатнім")
    
    # Диференціальне рівняння для мембранного потенціалу
    # τ * dV/dt = -V + R * I
    time_constant = resistance * capacitance
    dv_dt = (-membrane_potential + resistance * input_current) / time_constant
    new_potential = membrane_potential + dv_dt * dt
    
    # Перевірка на спайк
    if new_potential >= spike_threshold:
        return NEURON_RESTING_POTENTIAL, True  # Спайк відбувся, скидання потенціалу
    else:
        return new_potential, False  # Спайк не відбувся

def hodgkin_huxley_model(input_current: float, dt: float, t: float,
                        v: float = NEURON_RESTING_POTENTIAL,
                        n: float = 0.3177, m: float = 0.0529, h: float = 0.5961) -> Tuple[float, float, float, float]:
    """
    Модель Ходжкіна-Хакслі для нейрона.
    
    Параметри:
        input_current: Вхідний струм (мкА/см²)
        dt: Крок часу (мс)
        t: Час (мс)
        v: Початковий мембранный потенціал (мВ)
        n: Змінна активації К⁺ каналів
        m: Змінна активації Na⁺ каналів
        h: Змінна інактивації Na⁺ каналів
    
    Повертає:
        Кортеж (мембранный потенціал, n, m, h)
    """
    if dt <= 0:
        raise ValueError("Крок часу повинен бути додатнім")
    
    # Константи моделі Ходжкіна-Хакслі
    C_m = 1.0  # Ємність мембрани (мкФ/см²)
    g_K = 36.0  # Провідність К⁺ (мСм/см²)
    g_Na = 120.0  # Провідність Na⁺ (мСм/см²)
    g_L = 0.3  # Провідність витоку (мСм/см²)
    E_K = -12.0  # Рівноважний потенціал К⁺ (мВ)
    E_Na = 115.0  # Рівноважний потенціал Na⁺ (мВ)
    E_L = 10.613  # Рівноважний потенціал витоку (мВ)
    
    # Струми через іонні канали
    I_K = g_K * (n**4) * (v - E_K)
    I_Na = g_Na * (m**3) * h * (v - E_Na)
    I_L = g_L * (v - E_L)
    
    # Диференціальне рівняння для мембранного потенціалу
    dv_dt = (input_current - I_K - I_Na - I_L) / C_m
    
    # Рівняння для змінних ворот
    alpha_n = (0.01 * (v + 10)) / (math.exp((v + 10) / 10) - 1) if v != -10 else 0.1
    beta_n = 0.125 * math.exp(v / 80)
    dn_dt = alpha_n * (1 - n) - beta_n * n
    
    alpha_m = (0.1 * (v + 25)) / (math.exp((v + 25) / 10) - 1) if v != -25 else 1.0
    beta_m = 4 * math.exp(v / 18)
    dm_dt = alpha_m * (1 - m) - beta_m * m
    
    alpha_h = 0.07 * math.exp(v / 20)
    beta_h = 1 / (math.exp((v + 30) / 10) + 1)
    dh_dt = alpha_h * (1 - h) - beta_h * h
    
    # Оновлення значень
    v_new = v + dv_dt * dt
    n_new = n + dn_dt * dt
    m_new = m + dm_dt * dt
    h_new = h + dh_dt * dt
    
    return v_new, n_new, m_new, h_new

def synaptic_transmission(pre_spike: bool, synaptic_strength: float = SYNAPTIC_STRENGTH,
                         delay: float = SYNAPTIC_DELAY) -> float:
    """
    Модель синаптичної передачі.
    
    Параметри:
        pre_spike: Чи відбувся спайк у пресинаптичному нейроні
        synaptic_strength: Сила синапсу (См)
        delay: Синаптична затримка (с)
    
    Повертає:
        Синаптичний струм (А)
    """
    if synaptic_strength < 0:
        raise ValueError("Сила синапсу повинна бути невід'ємною")
    if delay < 0:
        raise ValueError("Синаптична затримка повинна бути невід'ємною")
    
    if pre_spike:
        return synaptic_strength  # Спрощена модель - миттєвий струм
    else:
        return 0.0

def action_potential_propagation(distance: float, 
                               conduction_velocity: float = CONDUCTION_VELOCITY) -> float:
    """
    Обчислити час поширення потенціалу дії.
    
    Параметри:
        distance: Відстань (м)
        conduction_velocity: Швидкість проведення (м/с)
    
    Повертає:
        Час поширення (с)
    """
    if distance < 0:
        raise ValueError("Відстань повинна бути невід'ємною")
    if conduction_velocity <= 0:
        raise ValueError("Швидкість проведення повинна бути додатньою")
    
    return distance / conduction_velocity

def neural_network_activation(weights: List[float], inputs: List[float], 
                            bias: float, activation_function: str = "sigmoid") -> float:
    """
    Обчислити активацію нейрона в нейронній мережі.
    
    Параметри:
        weights: Ваги зв'язків
        inputs: Вхідні сигнали
        bias: Зміщення
        activation_function: Функція активації ("sigmoid", "relu", "tanh")
    
    Повертає:
        Активація нейрона
    """
    if len(weights) != len(inputs):
        raise ValueError("Кількість ваг повинна дорівнювати кількості входів")
    
    # Обчислення зваженої суми
    weighted_sum = sum(w * x for w, x in zip(weights, inputs)) + bias
    
    # Застосування функції активації
    if activation_function == "sigmoid":
        return 1 / (1 + math.exp(-weighted_sum))
    elif activation_function == "relu":
        return max(0, weighted_sum)
    elif activation_function == "tanh":
        return math.tanh(weighted_sum)
    else:
        raise ValueError(f"Невідома функція активації: {activation_function}")

def ion_channel_conductance(voltage: float, ion_type: str = "Na") -> float:
    """
    Обчислити провідність іонного каналу.
    
    Параметри:
        voltage: Мембранный потенціал (мВ)
        ion_type: Тип іону ("Na", "K", "Ca", "Cl")
    
    Повертає:
        Провідність каналу (См)
    """
    # Максимальні провідності для різних іонів (См/см²)
    max_conductances = {
        "Na": 120e-3,
        "K": 36e-3,
        "Ca": 10e-3,
        "Cl": 3e-3
    }
    
    if ion_type not in max_conductances:
        raise ValueError(f"Невідомий тип іону: {ion_type}")
    
    max_g = max_conductances[ion_type]
    
    # Спрощена модель залежності провідності від напруги
    if ion_type == "Na":
        # Натрієві канали - активація при деполяризації
        return max_g * (1 / (1 + math.exp(-(voltage + 30) / 10)))
    elif ion_type == "K":
        # Калієві канали - активація при деполяризації з затримкою
        return max_g * (1 / (1 + math.exp(-(voltage + 10) / 15)))
    else:
        # Для інших іонів - проста залежність
        return max_g * (1 / (1 + math.exp(-(voltage + 20) / 20)))

def nernst_potential(ion_concentration_out: float, ion_concentration_in: float, 
                    ion_charge: int, temperature: float = TEMPERATURE) -> float:
    """
    Обчислити рівноважний потенціал Нернста.
    
    Параметри:
        ion_concentration_out: Концентрація іону зовні (мМ)
        ion_concentration_in: Концентрація іону всередині (мМ)
        ion_charge: Заряд іону
        temperature: Температура (К)
    
    Повертає:
        Рівноважний потенціал (В)
    """
    if ion_concentration_out <= 0:
        raise ValueError("Концентрація іону зовні повинна бути додатньою")
    if ion_concentration_in <= 0:
        raise ValueError("Концентрація іону всередині повинна бути додатньою")
    if ion_charge == 0:
        raise ValueError("Заряд іону не може дорівнювати нулю")
    if temperature <= 0:
        raise ValueError("Температура повинна бути додатньою")
    
    # Рівняння Нернста: E = (RT/zF) * ln([ion]_out / [ion]_in)
    R = GAS_CONSTANT
    F = FARADAY_CONSTANT
    z = ion_charge
    
    nernst_potential = (R * temperature / (z * F)) * math.log(ion_concentration_out / ion_concentration_in)
    return nernst_potential

def goldman_equation(ion_concentrations: Dict[str, Tuple[float, float]], 
                    ion_permeabilities: Dict[str, float], 
                    temperature: float = TEMPERATURE) -> float:
    """
    Обчислити мембранный потенціал за рівнянням Голдмана.
    
    Параметри:
        ion_concentrations: Словник концентрацій іонів (зовні, всередині) (мМ)
        ion_permeabilities: Словник проникності іонів (см/с)
        temperature: Температура (К)
    
    Повертає:
        Мембранный потенціал (В)
    """
    if not ion_concentrations:
        raise ValueError("Словник концентрацій іонів не може бути порожнім")
    if not ion_permeabilities:
        raise ValueError("Словник проникності іонів не може бути порожнім")
    
    # Перевірка наявності проникності для всіх іонів
    for ion in ion_concentrations:
        if ion not in ion_permeabilities:
            raise ValueError(f"Відсутня проникність для іону: {ion}")
        if any(c <= 0 for c in ion_concentrations[ion]):
            raise ValueError(f"Концентрації іону {ion} повинні бути додатніми")
        if ion_permeabilities[ion] < 0:
            raise ValueError(f"Проникність іону {ion} повинна бути невід'ємною")
    
    R = GAS_CONSTANT
    F = FARADAY_CONSTANT
    
    # Чисельник і знаменник для рівняння Голдмана
    numerator = 0
    denominator = 0
    
    for ion, (conc_out, conc_in) in ion_concentrations.items():
        permeability = ion_permeabilities[ion]
        if permeability > 0:
            numerator += permeability * conc_out
            denominator += permeability * conc_in
    
    if denominator == 0:
        return 0.0
    
    # Рівняння Голдмана: V_m = (RT/F) * ln(numerator/denominator)
    membrane_potential = (R * temperature / F) * math.log(numerator / denominator)
    return membrane_potential

def hodgkin_huxley_steady_state(voltage: float, gate_type: str) -> float:
    """
    Обчислити стаціонарне значення змінної ворот у моделі Ходжкіна-Хакслі.
    
    Параметри:
        voltage: Мембранный потенціал (мВ)
        gate_type: Тип ворот ("n", "m", "h")
    
    Повертає:
        Стаціонарне значення змінної ворот
    """
    if gate_type == "n":
        alpha = (0.01 * (voltage + 10)) / (math.exp((voltage + 10) / 10) - 1) if voltage != -10 else 0.1
        beta = 0.125 * math.exp(voltage / 80)
    elif gate_type == "m":
        alpha = (0.1 * (voltage + 25)) / (math.exp((voltage + 25) / 10) - 1) if voltage != -25 else 1.0
        beta = 4 * math.exp(voltage / 18)
    elif gate_type == "h":
        alpha = 0.07 * math.exp(voltage / 20)
        beta = 1 / (math.exp((voltage + 30) / 10) + 1)
    else:
        raise ValueError(f"Невідомий тип ворот: {gate_type}")
    
    if alpha + beta == 0:
        return 0.0
    
    return alpha / (alpha + beta)

def synaptic_plasticity(pre_spike_time: float, post_spike_time: float, 
                       weight: float, learning_rate: float = 0.01) -> float:
    """
    Модель синаптичної пластичності (правило Хебба).
    
    Параметри:
        pre_spike_time: Час спайку пресинаптичного нейрона (с)
        post_spike_time: Час спайку постсинаптичного нейрона (с)
        weight: Початкова вага синапсу
        learning_rate: Швидкість навчання
    
    Повертає:
        Нова вага синапсу
    """
    if learning_rate < 0:
        raise ValueError("Швидкість навчання повинна бути невід'ємною")
    
    # Вікно співпадіння для STDP (спайк-тайм залежної пластичності)
    time_difference = post_spike_time - pre_spike_time
    time_window = 20e-3  # 20 мс
    
    # Функція STDP
    if abs(time_difference) <= time_window:
        if time_difference > 0:
            # Потенціація: пресинаптичний спайк перед постсинаптичним
            delta_weight = learning_rate * math.exp(-time_difference / time_window)
        else:
            # Депресія: постсинаптичний спайк перед пресинаптичним
            delta_weight = -learning_rate * math.exp(time_difference / time_window)
    else:
        delta_weight = 0
    
    new_weight = weight + delta_weight
    return max(0, new_weight)  # Вага не може бути від'ємною

def neural_oscillation(frequency: float, time: float, phase: float = 0) -> float:
    """
    Модель нейрональних осциляцій.
    
    Параметри:
        frequency: Частота осциляцій (Гц)
        time: Час (с)
        phase: Фаза (рад)
    
    Повертає:
        Значення осциляцій
    """
    if frequency < 0:
        raise ValueError("Частота повинна бути невід'ємною")
    
    return math.sin(2 * math.pi * frequency * time + phase)

def brain_network_connectivity(adjacency_matrix: List[List[float]]) -> Dict[str, float]:
    """
    Обчислити параметри зв'язності мозкової мережі.
    
    Параметри:
        adjacency_matrix: Матриця суміжності мозкової мережі
    
    Повертає:
        Словник параметрів зв'язності
    """
    if not adjacency_matrix:
        raise ValueError("Матриця суміжності не може бути порожньою")
    
    n = len(adjacency_matrix)
    if any(len(row) != n for row in adjacency_matrix):
        raise ValueError("Матриця суміжності повинна бути квадратною")
    
    # Перевірка на симетричність (для ненаправлених графів)
    is_symmetric = all(adjacency_matrix[i][j] == adjacency_matrix[j][i] 
                      for i in range(n) for j in range(n))
    
    # Ступені вузлів
    degrees = [sum(row) for row in adjacency_matrix]
    
    # Густина мережі
    total_possible_connections = n * (n - 1)
    if not is_symmetric:
        total_possible_connections = n * (n - 1)
    else:
        total_possible_connections = n * (n - 1) / 2
    
    actual_connections = sum(sum(row) for row in adjacency_matrix) / (2 if is_symmetric else 1)
    density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
    
    # Середній ступінь
    avg_degree = sum(degrees) / n if n > 0 else 0
    
    # Коефіцієнт кластеризації (спрощений)
    clustering_coefficient = 0
    if n > 2:
        triangles = 0
        triplets = 0
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency_matrix[i][j] > 0]
            k = len(neighbors)
            if k >= 2:
                triplets += k * (k - 1) / 2
                for j1 in range(len(neighbors)):
                    for j2 in range(j1 + 1, len(neighbors)):
                        if adjacency_matrix[neighbors[j1]][neighbors[j2]] > 0:
                            triangles += 1
        if triplets > 0:
            clustering_coefficient = triangles / triplets
    
    return {
        "density": density,
        "average_degree": avg_degree,
        "clustering_coefficient": clustering_coefficient,
        "is_connected": is_symmetric
    }

def information_entropy(probabilities: List[float]) -> float:
    """
    Обчислити інформаційну ентропію.
    
    Параметри:
        probabilities: Список ймовірностей
    
    Повертає:
        Інформаційна ентропія (біти)
    """
    if not probabilities:
        raise ValueError("Список ймовірностей не може бути порожнім")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Всі ймовірності повинні бути в діапазоні [0, 1]")
    if abs(sum(probabilities) - 1) > 1e-10:
        raise ValueError("Сума ймовірностей повинна дорівнювати 1")
    
    entropy = 0
    for p in probabilities:
        if p > 0:  # Уникаємо log(0)
            entropy -= p * math.log2(p)
    
    return entropy

def mutual_information(joint_probabilities: List[List[float]], 
                      marginal_x: List[float], 
                      marginal_y: List[float]) -> float:
    """
    Обчислити взаємну інформацію між двома змінними.
    
    Параметри:
        joint_probabilities: Спільні ймовірності P(X,Y)
        marginal_x: Граничні ймовірності P(X)
        marginal_y: Граничні ймовірності P(Y)
    
    Повертає:
        Взаємна інформація (біти)
    """
    if not joint_probabilities:
        raise ValueError("Матриця спільних ймовірностей не може бути порожньою")
    
    rows = len(joint_probabilities)
    cols = len(joint_probabilities[0]) if rows > 0 else 0
    
    if len(marginal_x) != rows:
        raise ValueError("Довжина граничних ймовірностей X повинна дорівнювати кількості рядків")
    if len(marginal_y) != cols:
        raise ValueError("Довжина граничних ймовірностей Y повинна дорівнювати кількості стовпців")
    
    # Перевірка нормалізації
    if abs(sum(sum(row) for row in joint_probabilities) - 1) > 1e-10:
        raise ValueError("Сума спільних ймовірностей повинна дорівнювати 1")
    if abs(sum(marginal_x) - 1) > 1e-10:
        raise ValueError("Сума граничних ймовірностей X повинна дорівнювати 1")
    if abs(sum(marginal_y) - 1) > 1e-10:
        raise ValueError("Сума граничних ймовірностей Y повинна дорівнювати 1")
    
    mutual_info = 0
    for i in range(rows):
        for j in range(cols):
            p_xy = joint_probabilities[i][j]
            p_x = marginal_x[i]
            p_y = marginal_y[j]
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mutual_info += p_xy * math.log2(p_xy / (p_x * p_y))
    
    return mutual_info

def firing_rate(spike_times: List[float], time_window: float) -> float:
    """
    Обчислити частоту спайків нейрона.
    
    Параметри:
        spike_times: Список часів спайків (с)
        time_window: Часове вікно (с)
    
    Повертає:
        Частота спайків (Гц)
    """
    if not spike_times:
        return 0.0
    if time_window <= 0:
        raise ValueError("Часове вікно повинно бути додатнім")
    if any(t < 0 for t in spike_times):
        raise ValueError("Всі часи спайків повинні бути невід'ємними")
    
    return len(spike_times) / time_window

def coefficient_of_variation(spike_intervals: List[float]) -> float:
    """
    Обчислити коефіцієнт варіації інтервалів між спайками.
    
    Параметри:
        spike_intervals: Список інтервалів між спайками (с)
    
    Повертає:
        Коефіцієнт варіації
    """
    if not spike_intervals:
        raise ValueError("Список інтервалів не може бути порожнім")
    if any(interval <= 0 for interval in spike_intervals):
        raise ValueError("Всі інтервали повинні бути додатніми")
    
    mean_interval = sum(spike_intervals) / len(spike_intervals)
    variance = sum((interval - mean_interval) ** 2 for interval in spike_intervals) / len(spike_intervals)
    std_deviation = math.sqrt(variance)
    
    if mean_interval == 0:
        return float('inf')
    
    return std_deviation / mean_interval

def neural_correlation(signal1: List[float], signal2: List[float]) -> float:
    """
    Обчислити кореляцію між двома нейрональними сигналами.
    
    Параметри:
        signal1: Перший сигнал
        signal2: Другий сигнал
    
    Повертає:
        Коефіцієнт кореляції
    """
    if not signal1 or not signal2:
        raise ValueError("Сигнали не можуть бути порожніми")
    if len(signal1) != len(signal2):
        raise ValueError("Сигнали повинні мати однакову довжину")
    
    n = len(signal1)
    if n < 2:
        return 0.0
    
    mean1 = sum(signal1) / n
    mean2 = sum(signal2) / n
    
    numerator = sum((signal1[i] - mean1) * (signal2[i] - mean2) for i in range(n))
    sum_sq1 = sum((signal1[i] - mean1) ** 2 for i in range(n))
    sum_sq2 = sum((signal2[i] - mean2) ** 2 for i in range(n))
    
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def spike_triggered_average(stimulus: List[List[float]], 
                          spike_times: List[int]) -> List[float]:
    """
    Обчислити усереднений стимул, що викликає спайк.
    
    Параметри:
        stimulus: Стимул (часовий рядок)
        spike_times: Часи спайків (індекси)
    
    Повертає:
        Усереднений стимул
    """
    if not stimulus or not spike_times:
        return []
    
    stimulus_length = len(stimulus[0]) if stimulus else 0
    if any(len(s) != stimulus_length for s in stimulus):
        raise ValueError("Всі стимули повинні мати однакову довжину")
    if any(t < 0 or t >= len(stimulus) for t in spike_times):
        raise ValueError("Часи спайків повинні бути в межах стимулу")
    
    # Сума стимулів перед спайками
    sta_sum = [0.0] * stimulus_length
    count = 0
    
    # Вікно усереднення (наприклад, 100 мс перед спайком)
    window_size = min(100, stimulus_length)
    
    for spike_time in spike_times:
        if spike_time >= window_size:
            start_index = spike_time - window_size
            for i in range(window_size):
                sta_sum[i] += stimulus[start_index + i]
            count += 1
    
    if count == 0:
        return [0.0] * window_size
    
    return [s / count for s in sta_sum]

def receptive_field_size(visual_angle: float, distance: float) -> float:
    """
    Обчислити розмір рецептивного поля.
    
    Параметри:
        visual_angle: Візуальний кут (градуси)
        distance: Відстань до об'єкта (м)
    
    Повертає:
        Розмір рецептивного поля (м)
    """
    if visual_angle < 0 or visual_angle > 180:
        raise ValueError("Візуальний кут повинен бути в діапазоні [0, 180]")
    if distance < 0:
        raise ValueError("Відстань повинна бути невід'ємною")
    
    # Розмір = 2 * відстань * tan(візуальний_кут/2)
    return 2 * distance * math.tan(math.radians(visual_angle / 2))

def spatial_frequency_tuning(spatial_frequency: float, preferred_frequency: float, 
                           bandwidth: float = 1.0) -> float:
    """
    Обчислити налаштування на просторову частоту.
    
    Параметри:
        spatial_frequency: Просторова частота (циклів/градус)
        preferred_frequency: Бажана частота (циклів/градус)
        bandwidth: Смуга пропускання
    
    Повертає:
        Ступінь налаштування (0-1)
    """
    if spatial_frequency < 0:
        raise ValueError("Просторова частота повинна бути невід'ємною")
    if preferred_frequency < 0:
        raise ValueError("Бажана частота повинна бути невід'ємною")
    if bandwidth <= 0:
        raise ValueError("Смуга пропускання повинна бути додатньою")
    
    # Гаусівська функція налаштування
    diff = spatial_frequency - preferred_frequency
    tuning = math.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    return tuning

def contrast_sensitivity(contrast: float, threshold: float = 0.01) -> float:
    """
    Обчислити чутливість до контрасту.
    
    Параметри:
        contrast: Контраст (0-1)
        threshold: Поріг чутливості
    
    Повертає:
        Чутливість до контрасту
    """
    if contrast < 0 or contrast > 1:
        raise ValueError("Контраст повинен бути в діапазоні [0, 1]")
    if threshold < 0 or threshold > 1:
        raise ValueError("Поріг чутливості повинен бути в діапазоні [0, 1]")
    
    if contrast < threshold:
        return 0.0
    else:
        # Логарифмічна залежність
        return math.log10(contrast / threshold + 1)

def visual_field_mapping(eccentricity: float, polar_angle: float) -> Tuple[float, float]:
    """
    Відобразити ексцентриситет та полярний кут у координати кори.
    
    Параметри:
        eccentricity: Ексцентриситет (градуси)
        polar_angle: Полярний кут (градуси)
    
    Повертає:
        Кортеж (x, y) координат кори
    """
    if eccentricity < 0:
        raise ValueError("Ексцентриситет повинен бути невід'ємним")
    
    # Спрощене логарифмічне відображення
    if eccentricity == 0:
        return (0.0, 0.0)
    
    # Логарифмічне відображення ексцентриситету
    log_ecc = math.log(eccentricity + 1)
    
    # Полярне відображення
    rad_angle = math.radians(polar_angle)
    x = log_ecc * math.cos(rad_angle)
    y = log_ecc * math.sin(rad_angle)
    
    return (x, y)

def orientation_selectivity(angle: float, preferred_angle: float, 
                          bandwidth: float = 30.0) -> float:
    """
    Обчислити селективність до орієнтації.
    
    Параметри:
        angle: Кут орієнтації (градуси)
        preferred_angle: Бажана орієнтація (градуси)
        bandwidth: Смуга пропускання (градуси)
    
    Повертає:
        Ступінь селективності (0-1)
    """
    if bandwidth <= 0:
        raise ValueError("Смуга пропускання повинна бути додатньою")
    
    # Різниця кутів (з урахуванням періодичності)
    diff = abs(angle - preferred_angle)
    diff = min(diff, 180 - diff)  # Максимальна різниця 180°
    
    # Гаусівська функція селективності
    selectivity = math.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    return selectivity

def temporal_frequency_tuning(temporal_frequency: float, preferred_frequency: float, 
                            bandwidth: float = 2.0) -> float:
    """
    Обчислити налаштування на тимчасову частоту.
    
    Параметри:
        temporal_frequency: Тимчасова частота (Гц)
        preferred_frequency: Бажана частота (Гц)
        bandwidth: Смуга пропускання (Гц)
    
    Повертає:
        Ступінь налаштування (0-1)
    """
    if temporal_frequency < 0:
        raise ValueError("Тимчасова частота повинна бути невід'ємною")
    if preferred_frequency < 0:
        raise ValueError("Бажана частота повинна бути невід'ємною")
    if bandwidth <= 0:
        raise ValueError("Смуга пропускання повинна бути додатньою")
    
    # Гаусівська функція налаштування
    diff = temporal_frequency - preferred_frequency
    tuning = math.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    return tuning

def neural_decoding(neural_responses: List[float], 
                   tuning_curves: List[Callable[[float], float]]) -> float:
    """
    Декодувати стимул з нейрональних відповідей.
    
    Параметри:
        neural_responses: Відповіді нейронів
        tuning_curves: Криві налаштування нейронів
    
    Повертає:
        Декодований стимул
    """
    if not neural_responses or not tuning_curves:
        raise ValueError("Списки відповідей та кривих налаштування не можуть бути порожніми")
    if len(neural_responses) != len(tuning_curves):
        raise ValueError("Кількість відповідей повинна дорівнювати кількості кривих налаштування")
    
    # Спрощене декодування методом максимальної правдоподібності
    # Знаходимо стимул, який максимізує ймовірність відповідей
    
    best_stimulus = 0
    max_likelihood = -float('inf')
    
    # Пошук по дискретній сітці стимулів
    for stimulus in [i * 0.1 for i in range(100)]:
        likelihood = 0
        for response, curve in zip(neural_responses, tuning_curves):
            predicted_response = curve(stimulus)
            # Гаусівська правдоподібність
            if predicted_response > 0:
                likelihood -= (response - predicted_response) ** 2 / (2 * predicted_response ** 2)
        
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_stimulus = stimulus
    
    return best_stimulus

def population_coding(neural_responses: List[float], 
                     preferred_stimuli: List[float]) -> float:
    """
    Обчислити кодування популяцією нейронів.
    
    Параметри:
        neural_responses: Відповіді нейронів
        preferred_stimuli: Бажані стимули для кожного нейрона
    
    Повертає:
        Декодований стимул
    """
    if not neural_responses or not preferred_stimuli:
        raise ValueError("Списки відповідей та бажаних стимулів не можуть бути порожніми")
    if len(neural_responses) != len(preferred_stimuli):
        raise ValueError("Списки повинні мати однакову довжину")
    if any(r < 0 for r in neural_responses):
        raise ValueError("Відповіді нейронів повинні бути невід'ємними")
    
    total_response = sum(neural_responses)
    if total_response == 0:
        return 0.0
    
    # Зважене середнє бажаних стимулів
    decoded_stimulus = sum(r * s for r, s in zip(neural_responses, preferred_stimuli)) / total_response
    return decoded_stimulus

def neural_noise(noise_level: float, signal: float) -> float:
    """
    Додати шум до нейронального сигналу.
    
    Параметри:
        noise_level: Рівень шуму
        signal: Оригінальний сигнал
    
    Повертає:
        Сигнал з шумом
    """
    if noise_level < 0:
        raise ValueError("Рівень шуму повинен бути невід'ємним")
    
    # Гаусівський шум
    import random
    noise = random.gauss(0, noise_level)
    return signal + noise

def synaptic_integration(dendritic_inputs: List[Tuple[float, float]], 
                       time_constant: float = TIME_CONSTANT) -> float:
    """
    Інтегрувати синаптичні вхідні сигнали.
    
    Параметри:
        dendritic_inputs: Список кортежів (сила синапсу, час)
        time_constant: Часова константа (с)
    
    Повертає:
        Інтегрований сигнал
    """
    if time_constant <= 0:
        raise ValueError("Часова константа повинна бути додатньою")
    
    # Поточний час (відносний)
    current_time = 0
    integrated_signal = 0
    
    for synapse_strength, synapse_time in dendritic_inputs:
        if synapse_strength < 0:
            raise ValueError("Сила синапсу повинна бути невід'ємною")
        if synapse_time < 0:
            raise ValueError("Час синапсу повинен бути невід'ємним")
        
        # Експоненційне згасання сигналу
        time_diff = current_time - synapse_time
        decay_factor = math.exp(-time_diff / time_constant) if time_diff >= 0 else 0
        integrated_signal += synapse_strength * decay_factor
    
    return integrated_signal

def neural_field_dynamics(activity: List[float], connectivity: List[List[float]], 
                         external_input: List[float], dt: float) -> List[float]:
    """
    Обчислити динаміку нейронального поля.
    
    Параметри:
        activity: Поточна активність поля
        connectivity: Матриця зв'язності
        external_input: Зовнішні вхідні сигнали
        dt: Крок часу
    
    Повертає:
        Нова активність поля
    """
    if not activity or not connectivity or not external_input:
        raise ValueError("Всі списки повинні бути непорожніми")
    
    n = len(activity)
    if len(connectivity) != n or any(len(row) != n for row in connectivity):
        raise ValueError("Матриця зв'язності повинна бути квадратною та відповідати розміру активності")
    if len(external_input) != n:
        raise ValueError("Зовнішній вхід повинен відповідати розміру активності")
    if dt <= 0:
        raise ValueError("Крок часу повинен бути додатнім")
    
    # Динаміка: dA/dt = -A + W * A + I
    new_activity = []
    for i in range(n):
        # Сума зв'язаних активностей
        weighted_sum = sum(connectivity[i][j] * activity[j] for j in range(n))
        # Динаміка
        dA_dt = -activity[i] + weighted_sum + external_input[i]
        new_activity.append(activity[i] + dA_dt * dt)
    
    return new_activity

def spike_train_distance(train1: List[float], train2: List[float], 
                       cost_spike: float = 1.0, cost_time: float = 1.0) -> float:
    """
    Обчислити відстань між двома спайковими поїздами (відстань Вікрама).
    
    Параметри:
        train1: Перший спайковий поїзд (часи спайків)
        train2: Другий спайковий поїзд (часи спайків)
        cost_spike: Вартість вставки/видалення спайку
        cost_time: Вартість зсуву у часі
    
    Повертає:
        Відстань між спайковими поїздами
    """
    if any(t < 0 for t in train1 + train2):
        raise ValueError("Всі часи спайків повинні бути невід'ємними")
    if cost_spike < 0 or cost_time < 0:
        raise ValueError("Вартості повинні бути невід'ємними")
    
    # Спрощена реалізація відстані Вікрама
    # Для повної реалізації потрібна динамічна програмування
    
    # Відстань як сума абсолютних різниць
    # (це спрощення для демонстрації)
    if not train1 and not train2:
        return 0.0
    if not train1:
        return len(train2) * cost_spike
    if not train2:
        return len(train1) * cost_spike
    
    # Спрощене порівняння
    max_len = max(len(train1), len(train2))
    min_len = min(len(train1), len(train2))
    
    # Вартість вставки/видалення
    spike_cost = abs(len(train1) - len(train2)) * cost_spike
    
    # Вартість зсуву у часі для відповідних спайків
    time_cost = 0
    for i in range(min_len):
        time_cost += abs(train1[i] - train2[i]) * cost_time
    
    return spike_cost + time_cost

def neural_manifold_dimensionality(activity_patterns: List[List[float]]) -> int:
    """
    Оцінити розмірність нейронального многовиду.
    
    Параметри:
        activity_patterns: Список патернів активності
    
    Повертає:
        Оцінка розмірності многовиду
    """
    if not activity_patterns:
        raise ValueError("Список патернів активності не може бути порожнім")
    
    # Перевірка однакової довжини патернів
    pattern_length = len(activity_patterns[0])
    if any(len(pattern) != pattern_length for pattern in activity_patterns):
        raise ValueError("Всі патерни повинні мати однакову довжину")
    
    # Спрощена оцінка розмірності через SVD
    # Для реалізації потрібна матрична алгебра
    
    # Кількість патернів
    n_patterns = len(activity_patterns)
    
    # Максимальна можлива розмірність
    max_dimension = min(n_patterns, pattern_length)
    
    # Спрощена оцінка - логарифмічна залежність
    if n_patterns <= 1:
        return 1
    
    estimated_dimension = int(math.log(n_patterns) * math.log(pattern_length))
    return max(1, min(max_dimension, estimated_dimension))

def consciousness_integration(Phi: float, information: float) -> float:
    """
    Обчислити інтеграцію інформації як міру свідомості (на основі IIT).
    
    Параметри:
        Phi: Інтегрована інформація
        information: Загальна інформація
    
    Повертає:
        Міра свідомості
    """
    if Phi < 0 or information < 0:
        raise ValueError("Параметри повинні бути невід'ємними")
    
    # Спрощена модель IIT (Інтегрована інформаційна теорія)
    # Свідомість пропорційна інтегрованій інформації
    return Phi * information

def neural_entropy_production(activity: List[float], 
                            transition_matrix: List[List[float]]) -> float:
    """
    Обчислити ентропію, що виробляється нейрональною системою.
    
    Параметри:
        activity: Стан активності
        transition_matrix: Матриця переходів
    
    Повертає:
        Ентропія, що виробляється (бит/с)
    """
    if not activity or not transition_matrix:
        raise ValueError("Списки не можуть бути порожніми")
    
    n = len(activity)
    if len(transition_matrix) != n or any(len(row) != n for row in transition_matrix):
        raise ValueError("Матриця переходів повинна бути квадратною")
    
    # Перевірка нормалізації ймовірностей
    for row in transition_matrix:
        if abs(sum(row) - 1) > 1e-10:
            raise ValueError("Рядки матриці переходів повинні сумуватися до 1")
    
    # Ентропія виробництва: Σ P(i) Σ P(j|i) log P(j|i)
    entropy_production = 0
    for i in range(n):
        for j in range(n):
            if transition_matrix[i][j] > 0 and activity[i] > 0:
                entropy_production += activity[i] * transition_matrix[i][j] * math.log2(transition_matrix[i][j])
    
    return -entropy_production

def brain_energy_consumption(neuron_count: int, firing_rate: float, 
                           synapse_count: int) -> float:
    """
    Оцінити енергоспоживання мозком.
    
    Параметри:
        neuron_count: Кількість нейронів
        firing_rate: Частота спайків (Гц)
        synapse_count: Кількість синапсів
    
    Повертає:
        Енергоспоживання (Вт)
    """
    if neuron_count < 0 or firing_rate < 0 or synapse_count < 0:
        raise ValueError("Всі параметри повинні бути невід'ємними")
    
    # Енергія на один спайк (приблизно 5e-11 Дж)
    energy_per_spike = 5e-11
    
    # Енергія на синаптичну передачу (приблизно 1e-12 Дж)
    energy_per_synapse = 1e-12
    
    # Спайкова енергія
    spike_energy = neuron_count * firing_rate * energy_per_spike
    
    # Синаптична енергія
    synaptic_energy = synapse_count * firing_rate * energy_per_synapse
    
    # Загальна енергія
    total_energy = spike_energy + synaptic_energy
    
    return total_energy

def neural_coding_efficiency(information_rate: float, energy_consumption: float) -> float:
    """
    Обчислити ефективність нейронального кодування.
    
    Параметри:
        information_rate: Швидкість передачі інформації (біт/с)
        energy_consumption: Споживання енергії (Вт)
    
    Повертає:
        Ефективність кодування (біт/Дж)
    """
    if information_rate < 0:
        raise ValueError("Швидкість передачі інформації повинна бути невід'ємною")
    if energy_consumption <= 0:
        raise ValueError("Споживання енергії повинне бути додатнім")
    
    return information_rate / energy_consumption

def cognitive_load(processing_demand: float, available_resources: float) -> float:
    """
    Обчислити когнітивне навантаження.
    
    Параметри:
        processing_demand: Потреба в обробці
        available_resources: Доступні ресурси
    
    Повертає:
        Когнітивне навантаження (0-1)
    """
    if processing_demand < 0:
        raise ValueError("Потреба в обробці повинна бути невід'ємною")
    if available_resources <= 0:
        raise ValueError("Доступні ресурси повинні бути додатніми")
    
    load = processing_demand / available_resources
    return min(1.0, load)  # Обмеження зверху

def attention_allocation(stimulus_salience: List[float], 
                        attention_weights: List[float]) -> List[float]:
    """
    Розподілити увагу між стимулами.
    
    Параметри:
        stimulus_salience: Виразність стимулів
        attention_weights: Ваги уваги
    
    Повертає:
        Розподіл уваги
    """
    if not stimulus_salience or not attention_weights:
        raise ValueError("Списки не можуть бути порожніми")
    if len(stimulus_salience) != len(attention_weights):
        raise ValueError("Списки повинні мати однакову довжину")
    if any(s < 0 for s in stimulus_salience):
        raise ValueError("Виразність стимулів повинна бути невід'ємною")
    if any(w < 0 for w in attention_weights):
        raise ValueError("Ваги уваги повинні бути невід'ємними")
    
    # Комбінована салієнтність
    combined_salience = [s * w for s, w in zip(stimulus_salience, attention_weights)]
    
    # Нормалізація
    total_salience = sum(combined_salience)
    if total_salience == 0:
        return [1/len(combined_salience)] * len(combined_salience)
    
    return [s / total_salience for s in combined_salience]

def memory_decay(time_since_encoding: float, decay_rate: float = 0.1) -> float:
    """
    Обчислити зменшення пам'яті з часом.
    
    Параметри:
        time_since_encoding: Час з моменту кодування (с)
        decay_rate: Швидкість зменшення
    
    Повертає:
        Рівень пам'яті (0-1)
    """
    if time_since_encoding < 0:
        raise ValueError("Час з моменту кодування повинен бути невід'ємним")
    if decay_rate < 0:
        raise ValueError("Швидкість зменшення повинна бути невід'ємною")
    
    return math.exp(-decay_rate * time_since_encoding)

def learning_rate_adaptation(performance_error: float, 
                           previous_error: float, 
                           current_learning_rate: float,
                           adaptation_rate: float = 0.01) -> float:
    """
    Адаптувати швидкість навчання на основі помилки.
    
    Параметри:
        performance_error: Поточна помилка
        previous_error: Попередня помилка
        current_learning_rate: Поточна швидкість навчання
        adaptation_rate: Швидкість адаптації
    
    Повертає:
        Нова швидкість навчання
    """
    if current_learning_rate <= 0:
        raise ValueError("Поточна швидкість навчання повинна бути додатньою")
    if adaptation_rate < 0:
        raise ValueError("Швидкість адаптації повинна бути невід'ємною")
    
    # Якщо помилка зменшується, збільшуємо швидкість навчання
    # Якщо помилка збільшується, зменшуємо швидкість навчання
    error_change = performance_error - previous_error
    
    if error_change < 0:  # Помилка зменшується
        new_rate = current_learning_rate * (1 + adaptation_rate)
    else:  # Помилка збільшується або не змінюється
        new_rate = current_learning_rate * (1 - adaptation_rate)
    
    # Обмеження діапазону
    return max(1e-6, min(1.0, new_rate))

def neural_network_pruning(connection_strengths: List[float], 
                          pruning_threshold: float) -> List[bool]:
    """
    Виконати обрізку слабких зв'язків у нейронній мережі.
    
    Параметри:
        connection_strengths: Сили зв'язків
        pruning_threshold: Поріг обрізки
    
    Повертає:
        Список булевих значень (True - зберегти, False - обрізати)
    """
    if not connection_strengths:
        raise ValueError("Список сил зв'язків не може бути порожнім")
    if pruning_threshold < 0:
        raise ValueError("Поріг обрізки повинен бути невід'ємним")
    
    return [strength >= pruning_threshold for strength in connection_strengths]

def synaptic_scaling(synaptic_weights: List[float], 
                    target_activity: float, 
                    current_activity: float,
                    scaling_factor: float = 0.01) -> List[float]:
    """
    Масштабувати синаптичні ваги для підтримки цільової активності.
    
    Параметри:
        synaptic_weights: Синаптичні ваги
        target_activity: Цільова активність
        current_activity: Поточна активність
        scaling_factor: Фактор масштабування
    
    Повертає:
        Нові синаптичні ваги
    """
    if not synaptic_weights:
        raise ValueError("Список синаптичних ваг не може бути порожнім")
    if target_activity < 0 or current_activity < 0:
        raise ValueError("Активності повинні бути невід'ємними")
    if scaling_factor < 0:
        raise ValueError("Фактор масштабування повинен бути невід'ємним")
    
    if current_activity == 0:
        return synaptic_weights[:]  # Немає чого масштабувати
    
    # Відношення цільової до поточної активності
    activity_ratio = target_activity / current_activity
    
    # Корекція ваг
    adjustment = scaling_factor * (activity_ratio - 1)
    
    # Масштабування ваг
    new_weights = [max(0, w * (1 + adjustment)) for w in synaptic_weights]
    
    return new_weights

def neural_differentiation(activity_patterns: List[List[float]]) -> float:
    """
    Обчислити різноманітність нейрональних патернів.
    
    Параметри:
        activity_patterns: Список патернів активності
    
    Повертає:
        Міра диференціації (0-1)
    """
    if not activity_patterns:
        raise ValueError("Список патернів активності не може бути порожнім")
    
    # Перевірка однакової довжини патернів
    pattern_length = len(activity_patterns[0])
    if any(len(pattern) != pattern_length for pattern in activity_patterns):
        raise ValueError("Всі патерни повинні мати однакову довжину")
    
    if len(activity_patterns) < 2:
        return 0.0
    
    # Обчислення попарних відстаней між патернами
    distances = []
    for i in range(len(activity_patterns)):
        for j in range(i + 1, len(activity_patterns)):
            # Евклідова відстань
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(activity_patterns[i], activity_patterns[j])))
            distances.append(dist)
    
    if not distances:
        return 0.0
    
    # Середня відстань
    avg_distance = sum(distances) / len(distances)
    
    # Максимально можлива відстань (для нормалізації)
    max_possible_distance = math.sqrt(pattern_length)  # При максимальній різниці
    
    if max_possible_distance == 0:
        return 0.0
    
    return min(1.0, avg_distance / max_possible_distance)

def neural_integration(connectivity_matrix: List[List[float]]) -> float:
    """
    Обчислити інтеграцію нейрональної мережі.
    
    Параметри:
        connectivity_matrix: Матриця зв'язності
    
    Повертає:
        Міра інтеграції (0-1)
    """
    if not connectivity_matrix:
        raise ValueError("Матриця зв'язності не може бути порожньою")
    
    n = len(connectivity_matrix)
    if any(len(row) != n for row in connectivity_matrix):
        raise ValueError("Матриця зв'язності повинна бути квадратною")
    
    if n < 2:
        return 0.0
    
    # Спрощена міра інтеграції - густина зв'язності
    total_possible_connections = n * (n - 1)
    actual_connections = sum(sum(1 for w in row if w > 0) for row in connectivity_matrix) - n  # Віднімаємо діагональ
    
    if total_possible_connections == 0:
        return 0.0
    
    return actual_connections / total_possible_connections

def consciousness_measure(integration: float, information: float, 
                        differentiation: float) -> float:
    """
    Обчислити міру свідомості (на основі IIT).
    
    Параметри:
        integration: Інтеграція
        information: Інформація
        differentiation: Диференціація
    
    Повертає:
        Міра свідомості (0-1)
    """
    if not all(0 <= x <= 1 for x in [integration, information, differentiation]):
        raise ValueError("Всі параметри повинні бути в діапазоні [0, 1]")
    
    # Φ = інтеграція × інформація × диференціація
    return integration * information * differentiation

def neural_plasticity_rule(pre_activity: float, post_activity: float, 
                          weight: float, learning_rate: float = 0.01) -> float:
    """
    Правило пластичності нейронів (спрощене STDP).
    
    Параметри:
        pre_activity: Активність пресинаптичного нейрона
        post_activity: Активність постсинаптичного нейрона
        weight: Поточна вага синапсу
        learning_rate: Швидкість навчання
    
    Повертає:
        Нова вага синапсу
    """
    if not all(0 <= x <= 1 for x in [pre_activity, post_activity]):
        raise ValueError("Активності повинні бути в діапазоні [0, 1]")
    if weight < 0:
        raise ValueError("Вага синапсу повинна бути невід'ємною")
    if learning_rate < 0:
        raise ValueError("Швидкість навчання повинна бути невід'ємною")
    
    # Спрощене правило STDP
    # Якщо пресинаптичний нейрон активний перед постсинаптичним - потенціація
    # Якщо постсинаптичний нейрон активний перед пресинаптичним - депресія
    if pre_activity > post_activity:
        delta_weight = learning_rate * (1 - weight)  # Потенціація
    else:
        delta_weight = -learning_rate * weight  # Депресія
    
    new_weight = weight + delta_weight
    return max(0, min(1, new_weight))  # Обмеження діапазону [0, 1]

def cognitive_flexibility(task_switch_cost: float, switch_frequency: float) -> float:
    """
    Обчислити когнітивну гнучкість.
    
    Параметри:
        task_switch_cost: Вартість перемикання між завданнями
        switch_frequency: Частота перемикання
    
    Повертає:
        Міра когнітивної гнучкості (0-1)
    """
    if task_switch_cost < 0:
        raise ValueError("Вартість перемикання повинна бути невід'ємною")
    if switch_frequency < 0:
        raise ValueError("Частота перемикання повинна бути невід'ємною")
    
    # Когнітивна гнучкість обернено пропорційна вартості перемикання
    if task_switch_cost == 0:
        return 1.0
    
    flexibility = 1 / (1 + task_switch_cost * switch_frequency)
    return flexibility

def neural_robustness(activity_patterns: List[List[float]], 
                     noise_level: float) -> float:
    """
    Обчислити стійкість нейрональної системи до шуму.
    
    Параметри:
        activity_patterns: Список патернів активності
        noise_level: Рівень шуму
    
    Повертає:
        Міра стійкості (0-1)
    """
    if not activity_patterns:
        raise ValueError("Список патернів активності не може бути порожнім")
    if noise_level < 0:
        raise ValueError("Рівень шуму повинен бути невід'ємним")
    
    # Перевірка однакової довжини патернів
    pattern_length = len(activity_patterns[0])
    if any(len(pattern) != pattern_length for pattern in activity_patterns):
        raise ValueError("Всі патерни повинні мати однакову довжину")
    
    if len(activity_patterns) < 2:
        return 1.0  # Немає чого порівнювати
    
    # Спрощена міра стійкості - відношення сигнал/шум
    # Тут ми припускаємо, що шум додається до патернів
    
    # Середній патерн
    avg_pattern = [sum(pattern[i] for pattern in activity_patterns) / len(activity_patterns) 
                   for i in range(pattern_length)]
    
    # Варіація навколо середнього (сигнал)
    signal_variance = sum(sum((pattern[i] - avg_pattern[i]) ** 2 for i in range(pattern_length)) 
                         for pattern in activity_patterns) / len(activity_patterns)
    
    # Вплив шуму (шум)
    noise_variance = noise_level ** 2
    
    if noise_variance == 0:
        return 1.0 if signal_variance > 0 else 0.0
    
    # Співвідношення сигнал/шум
    snr = signal_variance / noise_variance
    
    # Нормалізація до [0, 1]
    return min(1.0, snr / (1 + snr))

def neural_efficiency(processing_speed: float, energy_consumption: float) -> float:
    """
    Обчислити ефективність нейрональної обробки.
    
    Параметри:
        processing_speed: Швидкість обробки (операцій/с)
        energy_consumption: Споживання енергії (Вт)
    
    Повертає:
        Ефективність (операцій/Дж)
    """
    if processing_speed < 0:
        raise ValueError("Швидкість обробки повинна бути невід'ємною")
    if energy_consumption <= 0:
        raise ValueError("Споживання енергії повинне бути додатнім")
    
    return processing_speed / energy_consumption

def cognitive_reserve(lifetime_learning: float, brain_volume: float, 
                     education_level: float) -> float:
    """
    Обчислити когнітивний резерв.
    
    Параметри:
        lifetime_learning: Навчання протягом життя
        brain_volume: Об'єм мозку
        education_level: Рівень освіти
    
    Повертає:
        Когнітивний резерв
    """
    if lifetime_learning < 0:
        raise ValueError("Навчання протягом життя повинне бути невід'ємним")
    if brain_volume < 0:
        raise ValueError("Об'єм мозку повинен бути невід'ємним")
    if education_level < 0:
        raise ValueError("Рівень освіти повинен бути невід'ємним")
    
    # Спрощена модель: когнітивний резерв пропорційний навчанню, об'єму мозку та освіті
    return lifetime_learning * brain_volume * education_level

def neural_synchronization(oscillation_phases: List[float], 
                          coupling_strength: float) -> float:
    """
    Обчислити синхронізацію нейрональних осциляцій.
    
    Параметри:
        oscillation_phases: Фази осциляцій
        coupling_strength: Сила зв'язку
    
    Повертає:
        Міра синхронізації (0-1)
    """
    if not oscillation_phases:
        raise ValueError("Список фаз не може бути порожнім")
    if coupling_strength < 0:
        raise ValueError("Сила зв'язку повинна бути невід'ємною")
    
    if len(oscillation_phases) < 2:
        return 1.0
    
    # Спрощена міра синхронізації - порядковий параметр
    # R = |Σ exp(iφ)| / N
    
    real_sum = sum(math.cos(phase) for phase in oscillation_phases)
    imag_sum = sum(math.sin(phase) for phase in oscillation_phases)
    
    order_parameter = math.sqrt(real_sum**2 + imag_sum**2) / len(oscillation_phases)
    
    # Нормалізація з урахуванням сили зв'язку
    return min(1.0, order_parameter * (1 + coupling_strength))

def brain_network_small_world(sigma: float, gamma: float) -> float:
    """
    Обчислити міру "малого світу" мозкової мережі.
    
    Параметри:
        sigma: Відношення кластеризації до випадкової мережі
        gamma: Відношення шляху до випадкової мережі
    
    Повертає:
        Міра "малого світу" (0-1)
    """
    if sigma < 0 or gamma < 0:
        raise ValueError("Параметри повинні бути невід'ємними")
    
    # Міра "малого світу" - добуток sigma та 1/gamma
    if gamma == 0:
        return 1.0 if sigma > 0 else 0.0
    
    small_world_measure = sigma / gamma
    return min(1.0, small_world_measure)

def neural_information_capacity(channel_bandwidth: float, 
                               signal_to_noise_ratio: float) -> float:
    """
    Обчислити інформаційну ємність нейронального каналу (теорема Шеннона).
    
    Параметри:
        channel_bandwidth: Смуга пропускання каналу (Гц)
        signal_to_noise_ratio: Відношення сигнал/шум
    
    Повертає:
        Інформаційна ємність (біт/с)
    """
    if channel_bandwidth < 0:
        raise ValueError("Смуга пропускання повинна бути невід'ємною")
    if signal_to_noise_ratio < 0:
        raise ValueError("Відношення сигнал/шум повинне бути невід'ємним")
    
    # Теорема Шеннона: C = B * log2(1 + SNR)
    if signal_to_noise_ratio == 0:
        return 0.0
    
    return channel_bandwidth * math.log2(1 + signal_to_noise_ratio)

def neural_field_pattern_formation(activity: List[float], 
                                  kernel: List[List[float]], 
                                  threshold: float) -> List[bool]:
    """
    Визначити формування патернів у нейрональному полі.
    
    Параметри:
        activity: Активність поля
        kernel: Ядро зв'язності
        threshold: Поріг активації
    
    Повертає:
        Список булевих значень (True - активний, False - неактивний)
    """
    if not activity or not kernel:
        raise ValueError("Списки не можуть бути порожніми")
    
    field_size = len(activity)
    if len(kernel) != field_size or any(len(row) != field_size for row in kernel):
        raise ValueError("Ядро повинно відповідати розміру поля")
    if threshold < 0:
        raise ValueError("Поріг активації повинен бути невід'ємним")
    
    # Конволюція активності з ядром
    pattern_activity = []
    for i in range(field_size):
        activation = sum(kernel[i][j] * activity[j] for j in range(field_size))
        pattern_activity.append(activation)
    
    # Порівняння з порогом
    return [act >= threshold for act in pattern_activity]

def consciousness_integration_complexity(Phi: float, system_size: int) -> float:
    """
    Обчислити складність інтеграції свідомості.
    
    Параметри:
        Phi: Інтегрована інформація
        system_size: Розмір системи
    
    Повертає:
        Складність інтеграції (0-1)
    """
    if Phi < 0:
        raise ValueError("Інтегрована інформація повинна бути невід'ємною")
    if system_size <= 0:
        raise ValueError("Розмір системи повинен бути додатнім")
    
    # Складність пропорційна Phi та обернено пропорційна розміру системи
    complexity = Phi / math.log2(system_size + 1)
    return min(1.0, complexity)

def neural_adaptation(sensory_input: float, adaptation_rate: float, 
                     previous_state: float) -> float:
    """
    Модель нейрональної адаптації до сенсорного вхідного сигналу.
    
    Параметри:
        sensory_input: Сенсорний вхід
        adaptation_rate: Швидкість адаптації
        previous_state: Попередній стан адаптації
    
    Повертає:
        Новий стан адаптації
    """
    if adaptation_rate < 0 or adaptation_rate > 1:
        raise ValueError("Швидкість адаптації повинна бути в діапазоні [0, 1]")
    
    # Експоненційна адаптація
    return previous_state + adaptation_rate * (sensory_input - previous_state)

def cognitive_control(conflict_level: float, control_efficiency: float) -> float:
    """
    Обчислити когнітивний контроль.
    
    Параметри:
        conflict_level: Рівень конфлікту
        control_efficiency: Ефективність контролю
    
    Повертає:
        Рівень когнітивного контролю
    """
    if conflict_level < 0:
        raise ValueError("Рівень конфлікту повинен бути невід'ємним")
    if control_efficiency < 0 or control_efficiency > 1:
        raise ValueError("Ефективність контролю повинна бути в діапазоні [0, 1]")
    
    # Когнітивний контроль зменшує вплив конфлікту
    return max(0, 1 - conflict_level * (1 - control_efficiency))

def neural_resilience(damage_level: float, recovery_capacity: float) -> float:
    """
    Обчислити стійкість нейронної системи до пошкоджень.
    
    Параметри:
        damage_level: Рівень пошкоджень
        recovery_capacity: Здатність до відновлення
    
    Повертає:
        Міра стійкості (0-1)
    """
    if damage_level < 0 or damage_level > 1:
        raise ValueError("Рівень пошкоджень повинен бути в діапазоні [0, 1]")
    if recovery_capacity < 0:
        raise ValueError("Здатність до відновлення повинна бути невід'ємною")
    
    # Стійкість = (1 - рівень_пошкоджень) * здатність_до_відновлення
    resilience = (1 - damage_level) * recovery_capacity
    return min(1.0, resilience)

def neural_diversity(activity_variance: float, optimal_variance: float) -> float:
    """
    Обчислити нейрональну різноманітність.
    
    Параметри:
        activity_variance: Дисперсія активності
        optimal_variance: Оптимальна дисперсія
    
    Повертає:
        Міра різноманітності (0-1)
    """
    if activity_variance < 0:
        raise ValueError("Дисперсія активності повинна бути невід'ємною")
    if optimal_variance <= 0:
        raise ValueError("Оптимальна дисперсія повинна бути додатньою")
    
    # Різноманітність максимальна при оптимальній дисперсії
    if activity_variance == 0:
        return 0.0
    
    # Гаусівська функція різноманітності
    diversity = math.exp(-((activity_variance - optimal_variance) ** 2) / (2 * optimal_variance ** 2))
    return diversity

def cognitive_fusion(sensory_modalities: List[float], 
                    integration_weights: List[float]) -> float:
    """
    Обчислити когнітивне злиття сенсорних модальностей.
    
    Параметри:
        sensory_modalities: Сенсорні модальності
        integration_weights: Ваги інтеграції
    
    Повертає:
        Ступінь когнітивного злиття
    """
    if not sensory_modalities or not integration_weights:
        raise ValueError("Списки не можуть бути порожніми")
    if len(sensory_modalities) != len(integration_weights):
        raise ValueError("Списки повинні мати однакову довжину")
    if any(m < 0 for m in sensory_modalities):
        raise ValueError("Сенсорні модальності повинні бути невід'ємними")
    if any(w < 0 for w in integration_weights):
        raise ValueError("Ваги інтеграції повинні бути невід'ємними")
    
    # Зважене інтеграційне злиття
    total_weight = sum(integration_weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(m * w for m, w in zip(sensory_modalities, integration_weights))
    return weighted_sum / total_weight

def neural_plasticity_potential(age_factor: float, learning_history: float) -> float:
    """
    Обчислити потенціал нейрональної пластичності.
    
    Параметри:
        age_factor: Фактор віку (0-1, де 0 - молодий, 1 - старий)
        learning_history: Історія навчання
    
    Повертає:
        Потенціал пластичності (0-1)
    """
    if age_factor < 0 or age_factor > 1:
        raise ValueError("Фактор віку повинен бути в діапазоні [0, 1]")
    if learning_history < 0:
        raise ValueError("Історія навчання повинна бути невід'ємною")
    
    # Пластичність зменшується з віком, але збільшується з навчанням
    age_effect = 1 - age_factor
    learning_effect = 1 - math.exp(-learning_history / 10)  # Насичення
    
    plasticity = age_effect * (1 + learning_effect)
    return min(1.0, plasticity)

def consciousness_complexity(integrated_information: float, 
                           system_diversity: float) -> float:
    """
    Обчислити складність свідомості.
    
    Параметри:
        integrated_information: Інтегрована інформація (Φ)
        system_diversity: Різноманітність системи
    
    Повертає:
        Складність свідомості
    """
    if integrated_information < 0:
        raise ValueError("Інтегрована інформація повинна бути невід'ємною")
    if system_diversity < 0:
        raise ValueError("Різноманітність системи повинна бути невід'ємною")
    
    # Складність свідомості - добуток інтегрованої інформації та різноманітності
    return integrated_information * system_diversity

def neural_network_criticality(connection_density: float, 
                              critical_density: float = 0.5) -> float:
    """
    Обчислити критичність нейронної мережі.
    
    Параметри:
        connection_density: Густина зв'язків
        critical_density: Критична густина, за замовчуванням 0.5
    
    Повертає:
        Міра критичності (0-1)
    """
    if connection_density < 0:
        raise ValueError("Густина зв'язків повинна бути невід'ємною")
    if critical_density <= 0 or critical_density > 1:
        raise ValueError("Критична густина повинна бути в діапазоні (0, 1]")
    
    # Критичність максимальна при критичній густині
    criticality = math.exp(-((connection_density - critical_density) ** 2) / (2 * critical_density ** 2))
    return criticality

def cognitive_emergence(complexity: float, integration: float) -> float:
    """
    Обчислити когнітивну емергентність.
    
    Параметри:
        complexity: Складність системи
        integration: Інтеграція компонентів
    
    Повертає:
        Міра емергентності
    """
    if complexity < 0:
        raise ValueError("Складність повинна бути невід'ємною")
    if integration < 0:
        raise ValueError("Інтеграція повинна бути невід'ємною")
    
    # Емергентність виникає при високій складності та інтеграції
    emergence = complexity * integration
    return min(1.0, emergence)

def neural_entropy_balance(entropy_production: float, entropy_consumption: float) -> float:
    """
    Обчислити баланс ентропії в нейронній системі.
    
    Параметри:
        entropy_production: Виробництво ентропії
        entropy_consumption: Споживання ентропії
    
    Повертає:
        Баланс ентропії
    """
    if entropy_production < 0:
        raise ValueError("Виробництво ентропії повинне бути невід'ємним")
    if entropy_consumption < 0:
        raise ValueError("Споживання ентропії повинне бути невід'ємним")
    
    # Баланс = споживання - виробництво
    balance = entropy_consumption - entropy_production
    return balance

def consciousness_resonance(frequency: float, resonance_threshold: float) -> float:
    """
    Обчислити резонанс свідомості.
    
    Параметри:
        frequency: Частота осциляцій
        resonance_threshold: Поріг резонансу
    
    Повертає:
        Міра резонансу (0-1)
    """
    if frequency < 0:
        raise ValueError("Частота повинна бути невід'ємною")
    if resonance_threshold <= 0:
        raise ValueError("Поріг резонансу повинен бути додатнім")
    
    # Резонанс максимальний при відповідності частот
    resonance = math.exp(-((frequency - resonance_threshold) ** 2) / (2 * resonance_threshold ** 2))
    return resonance

def neural_field_coherence(field_activities: List[List[float]]) -> float:
    """
    Обчислити когерентність нейронального поля.
    
    Параметри:
        field_activities: Активності в різних точках поля
    
    Повертає:
        Міра когерентності (0-1)
    """
    if not field_activities or not field_activities[0]:
        raise ValueError("Списки активностей не можуть бути порожніми")
    
    # Перевірка однакової довжини
    length = len(field_activities[0])
    if any(len(activity) != length for activity in field_activities):
        raise ValueError("Всі списки активностей повинні мати однакову довжину")
    
    if len(field_activities) < 2:
        return 1.0
    
    # Обчислення когерентності як середньої кореляції між активностями
    correlations = []
    for i in range(len(field_activities)):
        for j in range(i + 1, len(field_activities)):
            # Кореляція між двома активностями
            mean1 = sum(field_activities[i]) / length
            mean2 = sum(field_activities[j]) / length
            
            numerator = sum((field_activities[i][k] - mean1) * (field_activities[j][k] - mean2) 
                           for k in range(length))
            sum_sq1 = sum((field_activities[i][k] - mean1) ** 2 for k in range(length))
            sum_sq2 = sum((field_activities[j][k] - mean2) ** 2 for k in range(length))
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator > 0:
                correlation = abs(numerator / denominator)
                correlations.append(correlation)
    
    if not correlations:
        return 0.0
    
    return sum(correlations) / len(correlations)

def cognitive_emergence_threshold(system_complexity: float, 
                                 integration_level: float) -> bool:
    """
    Визначити, чи досягнуто поріг когнітивної емергентності.
    
    Параметри:
        system_complexity: Складність системи
        integration_level: Рівень інтеграції
    
    Повертає:
        Чи досягнуто поріг емергентності (True/False)
    """
    if system_complexity < 0:
        raise ValueError("Складність системи повинна бути невід'ємною")
    if integration_level < 0:
        raise ValueError("Рівень інтеграції повинен бути невід'ємним")
    
    # Поріг емергентності - добуток складності та інтеграції перевищує критичне значення
    emergence_measure = system_complexity * integration_level
    critical_threshold = 0.5  # Приклад критичного порогу
    
    return emergence_measure >= critical_threshold

def neural_information_integration(activity_patterns: List[List[float]], 
                                  time_windows: List[float]) -> float:
    """
    Обчислити інтеграцію інформації в нейронній системі.
    
    Параметри:
        activity_patterns: Патерни активності в різних часових вікнах
        time_windows: Часові вікна
    
    Повертає:
        Міра інтеграції інформації
    """
    if not activity_patterns or not time_windows:
        raise ValueError("Списки не можуть бути порожніми")
    if len(activity_patterns) != len(time_windows):
        raise ValueError("Кількість патернів повинна відповідати кількості часових вікон")
    
    # Перевірка однакової довжини патернів
    if activity_patterns:
        pattern_length = len(activity_patterns[0])
        if any(len(pattern) != pattern_length for pattern in activity_patterns):
            raise ValueError("Всі патерни повинні мати однакову довжину")
    
    if len(activity_patterns) < 2:
        return 0.0
    
    # Спрощена міра інтеграції - сума взаємної інформації між послідовними патернами
    total_integration = 0
    for i in range(len(activity_patterns) - 1):
        # Обчислення кореляції між послідовними патернами
        pattern1 = activity_patterns[i]
        pattern2 = activity_patterns[i + 1]
        
        if not pattern1 or not pattern2:
            continue
            
        mean1 = sum(pattern1) / len(pattern1)
        mean2 = sum(pattern2) / len(pattern2)
        
        numerator = sum((a - mean1) * (b - mean2) for a, b in zip(pattern1, pattern2))
        sum_sq1 = sum((a - mean1) ** 2 for a in pattern1)
        sum_sq2 = sum((b - mean2) ** 2 for b in pattern2)
        
        denominator = math.sqrt(sum_sq1 * sum_sq2) if sum_sq1 * sum_sq2 > 0 else 1
        
        correlation = abs(numerator / denominator) if denominator > 0 else 0
        total_integration += correlation
    
    return total_integration / (len(activity_patterns) - 1) if len(activity_patterns) > 1 else 0.0