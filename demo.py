#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстраційний скрипт для бібліотеки PyNexus
Показує основні можливості бібліотеки
"""

import math
import random
import pynexus as px

def demonstrate_mathematics():
    """Демонстрація математичних можливостей"""
    print("=== Математичні обчислення ===")
    
    # Чисельне інтегрування
    result = px.numerical_integral(lambda x: x**2, 0, 1)
    print(f"∫₀¹ x² dx = {result:.6f}")
    
    # Чисельне диференціювання
    derivative = px.numerical_derivative(math.sin, math.pi/4)
    print(f"d/dx sin(x) при x=π/4 = {derivative:.6f}")
    
    # Розв'язання рівняння
    try:
        root = px.newton_raphson(
            lambda x: x**3 - 2*x - 5,
            lambda x: 3*x**2 - 2,
            2.0
        )
        print(f"Корінь рівняння x³ - 2x - 5 = 0: {root[0]:.6f}")
    except:
        print("Не вдалося знайти корінь")
    
    print()

def demonstrate_statistics():
    """Демонстрація статистичних можливостей"""
    print("=== Статистичний аналіз ===")
    
    # Генеруємо тестові дані
    data = [random.gauss(100, 15) for _ in range(1000)]
    
    # Основні статистики
    mean = sum(data) / len(data)
    sorted_data = sorted(data)
    median = sorted_data[len(sorted_data) // 2]
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = math.sqrt(variance)
    
    print(f"Середнє значення: {mean:.2f}")
    print(f"Медіана: {median:.2f}")
    print(f"Стандартне відхилення: {std:.2f}")
    
    # Коефіцієнт кореляції
    x = [float(i) for i in range(100)]
    y = [float(2*i + random.gauss(0, 5)) for i in range(100)]
    # Використовуємо функцію з statistics.py
    correlation = px.pearson_correlation(x, y)
    print(f"Коефіцієнт кореляції: {correlation:.4f}")
    
    print()

def demonstrate_physics():
    """Демонстрація фізичних обчислень"""
    print("=== Фізичні обчислення ===")
    
    # Обчислення енергії
    mass = 1.0  # кг
    velocity = 10.0  # м/с
    kinetic_energy = 0.5 * mass * velocity ** 2
    print(f"Кінетична енергія: {kinetic_energy:.2f} Дж")
    
    # Закон Ома
    voltage = 12.0  # В
    resistance = 4.0  # Ом
    current = voltage / resistance
    print(f"Сила струму: {current:.2f} А")
    
    print()

def demonstrate_chemistry():
    """Демонстрація хімічних обчислень"""
    print("=== Хімічні обчислення ===")
    
    # Ідеальний газовий закон
    pressure = 101325  # Па
    volume = 0.0224  # м³
    temperature = 273.15  # К
    R = 8.314  # Дж/(моль·К)
    moles = (pressure * volume) / (R * temperature)
    print(f"Кількість речовини: {moles:.4f} моль")
    
    # Молярна маса води
    # H = 1.008, O = 15.999
    water_molar_mass = 2 * 1.008 + 15.999
    print(f"Молярна маса води: {water_molar_mass:.2f} г/моль")
    
    print()

def demonstrate_biology():
    """Демонстрація біологічних обчислень"""
    print("=== Біологічні обчислення ===")
    
    # Індекс маси тіла
    weight = 70.0  # кг
    height = 1.75  # м
    bmi = weight / (height ** 2)
    print(f"Індекс маси тіла: {bmi:.2f}")
    
    # Частота серця
    age = 30  # років
    max_heart_rate = 220 - age
    print(f"Максимальна частота серця: {max_heart_rate:.0f} ударів/хв")
    
    print()

def demonstrate_economics():
    """Демонстрація економічних обчислень"""
    print("=== Економічні обчислення ===")
    
    # Темп зростання ВВП
    gdp_current = 1000000
    gdp_previous = 950000
    growth_rate = ((gdp_current - gdp_previous) / gdp_previous) * 100
    print(f"Темп зростання ВВП: {growth_rate:.2f}%")
    
    # Рівень інфляції
    cpi_current = 110
    cpi_previous = 105
    inflation = ((cpi_current - cpi_previous) / cpi_previous) * 100
    print(f"Рівень інфляції: {inflation:.2f}%")
    
    print()

def demonstrate_machine_learning():
    """Демонстрація машинного навчання"""
    print("=== Машинне навчання ===")
    
    # Генеруємо тестові дані для лінійної регресії
    X = [random.uniform(-1, 1) for _ in range(100)]
    y = [2 * x + 1 + random.gauss(0, 0.1) for x in X]
    
    # Лінійна регресія
    try:
        # Використовуємо функцію з ml.py
        model = px.linear_regression(X, y)
        print(f"Коефіцієнт регресії: {model['slope']:.4f}")
        print(f"Вільний член: {model['intercept']:.4f}")
    except:
        print("Не вдалося виконати лінійну регресію")
    
    print()

def demonstrate_signal_processing():
    """Демонстрація обробки сигналів"""
    print("=== Обробка сигналів ===")
    
    # Генеруємо тестовий сигнал
    t = [i/100 for i in range(1000)]
    signal = [math.sin(2*math.pi*5*t[i]) + 0.5*math.sin(2*math.pi*10*t[i]) for i in range(len(t))]
    
    # Обчислення FFT
    try:
        # Перетворюємо в комплексні числа
        complex_signal = [complex(x) for x in signal[:64]]  # Обмежуємо для швидкості
        spectrum = px.fft(complex_signal)
        print(f"Розмір спектра: {len(spectrum)}")
        print(f"Максимальна амплітуда: {max(abs(x) for x in spectrum):.4f}")
    except:
        print("Не вдалося обчислити FFT")
    
    print()

def demonstrate_data_analysis():
    """Демонстрація аналізу даних"""
    print("=== Аналіз даних ===")
    
    # Статистика набору даних
    data = [random.gauss(50, 10) for _ in range(1000)]
    mean = sum(data) / len(data)
    sorted_data = sorted(data)
    median = sorted_data[len(sorted_data) // 2]
    min_val = min(data)
    max_val = max(data)
    
    print(f"Мінімум: {min_val:.2f}")
    print(f"Максимум: {max_val:.2f}")
    print(f"Середнє: {mean:.2f}")
    print(f"Медіана: {median:.2f}")
    
    print()

def main():
    """Головна функція демонстрації"""
    print("PyNexus - Універсальна науково-аналітична бібліотека")
    print("=" * 50)
    print()
    
    # Демонстрація різних модулів
    demonstrate_mathematics()
    demonstrate_statistics()
    demonstrate_physics()
    demonstrate_chemistry()
    demonstrate_biology()
    demonstrate_economics()
    demonstrate_machine_learning()
    demonstrate_signal_processing()
    demonstrate_data_analysis()
    
    print("Демонстрація завершена!")
    print()
    print("Для більш детального вивчення зверніться до документації.")

if __name__ == "__main__":
    main()