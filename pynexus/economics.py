"""
Модуль для обчислювальної економіки в PyNexus.
Містить функції для економетрії, фінансового моделювання, макроекономічного аналізу та інших економічних обчислень.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Union

# Примітка: Ці бібліотеки не встановлені в середовищі, тому вони закоментовані
# import numpy as np
# from scipy import stats, optimize
# import matplotlib.pyplot as plt

# Константи для економіки
GDP_GROWTH_TARGET = 0.03  # Цільовий темп зростання ВВП (3%)
INFLATION_TARGET = 0.02   # Цільовий рівень інфляції (2%)
UNEMPLOYMENT_NATURAL = 0.05  # Природний рівень безробіття (5%)
INTEREST_RATE_NEUTRAL = 0.03  # Нейтральна процентна ставка (3%)
MONEY_VELOCITY = 1.5  # Швидкість обігу грошей
TAX_RATE_DEFAULT = 0.2  # Стандартна ставка податку (20%)
GOVERNMENT_SPENDING_MULTIPLIER = 1.5  # Мультиплікатор державних витрат

def supply_and_demand_equilibrium(supply_func, demand_func, 
                               price_range: Tuple[float, float], 
                               tolerance: float = 1e-6) -> Tuple[float, float]:
    """
    Знайти рівноважну ціну та кількість для функцій попиту та пропозиції.
    
    Параметри:
        supply_func: Функція пропозиції (кількість від ціни)
        demand_func: Функція попиту (кількість від ціни)
        price_range: Діапазон цін для пошуку
        tolerance: Точність обчислень
    
    Повертає:
        Кортеж (рівноважна ціна, рівноважна кількість)
    """
    min_price, max_price = price_range
    
    if min_price >= max_price:
        raise ValueError("Мінімальна ціна повинна бути меншою за максимальну")
    
    # Пошук рівноваги методом дихотомії
    for _ in range(1000):  # Максимум 1000 ітерацій
        mid_price = (min_price + max_price) / 2
        supply_qty = supply_func(mid_price)
        demand_qty = demand_func(mid_price)
        
        if abs(supply_qty - demand_qty) < tolerance:
            return mid_price, (supply_qty + demand_qty) / 2
        
        if supply_qty > demand_qty:
            max_price = mid_price
        else:
            min_price = mid_price
    
    # Якщо не знайдено точну рівновагу, повертаємо найближче наближення
    final_price = (min_price + max_price) / 2
    final_quantity = (supply_func(final_price) + demand_func(final_price)) / 2
    return final_price, final_quantity

def consumer_surplus(demand_func, equilibrium_price: float, 
                    max_price: float, num_points: int = 1000) -> float:
    """
    Обчислити споживчий надлишок.
    
    Параметри:
        demand_func: Функція попиту
        equilibrium_price: Рівноважна ціна
        max_price: Максимальна ціна (де попит = 0)
        num_points: Кількість точок для чисельного інтегрування
    
    Повертає:
        Споживчий надлишок
    """
    if equilibrium_price < 0 or max_price <= equilibrium_price:
        raise ValueError("Невірні параметри ціни")
    
    if num_points <= 0:
        raise ValueError("Кількість точок повинна бути додатньою")
    
    # Чисельне інтегрування для обчислення споживчого надлишку
    step = (max_price - equilibrium_price) / num_points
    surplus = 0.0
    
    for i in range(num_points):
        price = equilibrium_price + i * step
        quantity = demand_func(price)
        surplus += quantity * step
    
    return surplus

def producer_surplus(supply_func, equilibrium_price: float, 
                    min_price: float = 0, num_points: int = 1000) -> float:
    """
    Обчислити надлишок виробника.
    
    Параметри:
        supply_func: Функція пропозиції
        equilibrium_price: Рівноважна ціна
        min_price: Мінімальна ціна (де пропозиція = 0)
        num_points: Кількість точок для чисельного інтегрування
    
    Повертає:
        Надлишок виробника
    """
    if equilibrium_price < min_price:
        raise ValueError("Рівноважна ціна повинна бути не меншою за мінімальну")
    
    if num_points <= 0:
        raise ValueError("Кількість точок повинна бути додатньою")
    
    # Чисельне інтегрування для обчислення надлишку виробника
    step = (equilibrium_price - min_price) / num_points
    surplus = 0.0
    
    for i in range(num_points):
        price = min_price + i * step
        quantity = supply_func(price)
        surplus += (equilibrium_price - price) * quantity * step
    
    return surplus

def elasticity_demand(price: float, quantity: float, 
                     delta_price: float, delta_quantity: float) -> float:
    """
    Обчислити коефіцієнт еластичності попиту.
    
    Параметри:
        price: Початкова ціна
        quantity: Початкова кількість
        delta_price: Зміна ціни
        delta_quantity: Зміна кількості
    
    Повертає:
        Коефіцієнт еластичності попиту
    """
    if price == 0 or quantity == 0:
        raise ValueError("Ціна та кількість повинні бути ненульовими")
    
    if delta_price == 0:
        return float('inf')  # Нескінченна еластичність
    
    # Формула еластичності: (% зміна кількості) / (% зміна ціни)
    percent_change_quantity = (delta_quantity / quantity) * 100
    percent_change_price = (delta_price / price) * 100
    
    if percent_change_price == 0:
        return float('inf')
    
    elasticity = percent_change_quantity / percent_change_price
    return elasticity

def gdp_growth_rate(gdp_current: float, gdp_previous: float) -> float:
    """
    Обчислити темп зростання ВВП.
    
    Параметри:
        gdp_current: Поточний ВВП
        gdp_previous: Попередній ВВП
    
    Повертає:
        Темп зростання ВВП у відсотках
    """
    if gdp_previous <= 0:
        raise ValueError("Попередній ВВП повинен бути додатнім")
    
    growth_rate = ((gdp_current - gdp_previous) / gdp_previous) * 100
    return growth_rate

def inflation_rate(cpi_current: float, cpi_previous: float) -> float:
    """
    Обчислити рівень інфляції.
    
    Параметри:
        cpi_current: Поточний індекс споживчих цін
        cpi_previous: Попередній індекс споживчих цін
    
    Повертає:
        Рівень інфляції у відсотках
    """
    if cpi_previous <= 0:
        raise ValueError("Попередній ІСЦ повинен бути додатнім")
    
    inflation_rate = ((cpi_current - cpi_previous) / cpi_previous) * 100
    return inflation_rate

def unemployment_rate(labor_force: float, unemployed: float) -> float:
    """
    Обчислити рівень безробіття.
    
    Параметри:
        labor_force: Робоча сила
        unemployed: Кількість безробітних
    
    Повертає:
        Рівень безробіття у відсотках
    """
    if labor_force <= 0:
        raise ValueError("Робоча сила повинна бути додатньою")
    
    unemployment_rate = (unemployed / labor_force) * 100
    return unemployment_rate

def phillips_curve(inflation_expectations: float, 
                  unemployment_rate: float, 
                  natural_unemployment: float = UNEMPLOYMENT_NATURAL,
                  Phillips_coefficient: float = 0.5) -> float:
    """
    Модель кривої Філліпса - зв'язок між інфляцією та безробіттям.
    
    Параметри:
        inflation_expectations: Очікувана інфляція
        unemployment_rate: Поточний рівень безробіття
        natural_unemployment: Природний рівень безробіття
        Phillips_coefficient: Коефіцієнт кривої Філліпса
    
    Повертає:
        Очікувана інфляція
    """
    if natural_unemployment < 0 or natural_unemployment > 1:
        raise ValueError("Природний рівень безробіття повинен бути між 0 та 1")
    
    if unemployment_rate < 0:
        raise ValueError("Рівень безробіття не може бути від'ємним")
    
    # Формула кривої Філліпса: π = π_e - α(u - u_n)
    inflation = inflation_expectations - Phillips_coefficient * (unemployment_rate - natural_unemployment)
    return max(0, inflation)  # Інфляція не може бути від'ємною

def is_lm_model(interest_rate: float, income: float,
               liquidity_preference: float = 0.5,
               investment_sensitivity: float = 0.3,
               money_supply: float = 1000,
               price_level: float = 1.0) -> Dict[str, float]:
    """
    Модель IS-LM - рівновага на товарному та грошовому ринках.
    
    Параметри:
        interest_rate: Процентна ставка
        income: Дохід (ВВП)
        liquidity_preference: Схильність до ліквідності
        investment_sensitivity: Чутливість інвестицій до процентної ставки
        money_supply: Грошова пропозиція
        price_level: Рівень цін
    
    Повертає:
        Словник з результатами моделі
    """
    if liquidity_preference <= 0 or investment_sensitivity <= 0:
        raise ValueError("Параметри повинні бути додатніми")
    
    if money_supply <= 0 or price_level <= 0:
        raise ValueError("Грошова пропозиція та рівень цін повинні бути додатніми")
    
    # Крива IS: Y = C(Y-T) + I(r) + G + NX
    # Спрощена форма: Y = A - B*r, де A - автономні витрати, B - чутливість до ставки
    autonomous_spending = 1000  # Автономні витрати
    is_curve = autonomous_spending - investment_sensitivity * interest_rate
    
    # Крива LM: M/P = L(r, Y)
    # Спрощена форма: Y = C + D*r, де C, D - параметри
    real_money_supply = money_supply / price_level
    lm_constant = real_money_supply / liquidity_preference
    lm_curve = lm_constant + liquidity_preference * interest_rate
    
    return {
        'is_curve': is_curve,
        'lm_curve': lm_curve,
        'equilibrium_income': (autonomous_spending + lm_constant) / (1 + investment_sensitivity * liquidity_preference),
        'equilibrium_interest_rate': (autonomous_spending - lm_constant) / (investment_sensitivity + liquidity_preference)
    }

def solow_growth_model(capital: float, labor: float, 
                      productivity: float, 
                      depreciation_rate: float = 0.05,
                      savings_rate: float = 0.2) -> Dict[str, float]:
    """
    Модель економічного зростання Солоу.
    
    Параметри:
        capital: Капітал
        labor: Праця
        productivity: Продуктивність
        depreciation_rate: Норма амортизації
        savings_rate: Норма заощадження
    
    Повертає:
        Словник з результатами моделі
    """
    if capital <= 0 or labor <= 0 or productivity <= 0:
        raise ValueError("Капітал, праця та продуктивність повинні бути додатніми")
    
    if depreciation_rate < 0 or depreciation_rate > 1:
        raise ValueError("Норма амортизації повинна бути між 0 та 1")
    
    if savings_rate < 0 or savings_rate > 1:
        raise ValueError("Норма заощадження повинна бути між 0 та 1")
    
    # Виробнича функція Кобба-Дугласа: Y = A * K^α * L^(1-α)
    # Припустимо α = 0.3 (капіталова доля)
    capital_share = 0.3
    output = productivity * (capital ** capital_share) * (labor ** (1 - capital_share))
    
    # Інвестиції
    investment = savings_rate * output
    
    # Амортизація
    depreciation = depreciation_rate * capital
    
    # Зміна капіталу
    delta_capital = investment - depreciation
    
    # Вихід на сталий стан
    # У сталому стані: s * A * k^α = δ * k, де k = K/L
    labor_productivity = output / labor
    capital_labor_ratio = capital / labor
    
    # Золоте правило сталого стану
    golden_rule_savings = capital_share
    
    return {
        'output': output,
        'output_per_worker': labor_productivity,
        'capital_labor_ratio': capital_labor_ratio,
        'investment': investment,
        'depreciation': depreciation,
        'delta_capital': delta_capital,
        'golden_rule_savings_rate': golden_rule_savings,
        'is_steady_state': abs(delta_capital) < 1e-6
    }

def keynesian_multiplier(government_spending: float, 
                        tax_rate: float = TAX_RATE_DEFAULT,
                        marginal_propensity_to_consume: float = 0.8) -> float:
    """
    Кейнсіанський мультиплікатор фіскальної політики.
    
    Параметри:
        government_spending: Державні витрати
        tax_rate: Ставка податку
        marginal_propensity_to_consume: Гранична схильність до споживання
    
    Повертає:
        Мультиплікатор та зміну ВВП
    """
    if tax_rate < 0 or tax_rate >= 1:
        raise ValueError("Ставка податку повинна бути між 0 та 1")
    
    if marginal_propensity_to_consume < 0 or marginal_propensity_to_consume > 1:
        raise ValueError("Гранична схильність до споживання повинна бути між 0 та 1")
    
    # Мультиплікатор: 1 / (1 - MPC * (1 - t))
    multiplier = 1 / (1 - marginal_propensity_to_consume * (1 - tax_rate))
    
    # Зміна ВВП
    gdp_change = government_spending * multiplier
    
    return gdp_change

def fisher_equation(nominal_rate: float, inflation_rate: float) -> float:
    """
    Рівняння Фішера - зв'язок між номінальною та реальною процентною ставкою.
    
    Параметри:
        nominal_rate: Номінальна процентна ставка
        inflation_rate: Очікувана інфляція
    
    Повертає:
        Реальна процентна ставка
    """
    # Точна формула: 1 + r = (1 + i) / (1 + π)
    # Приблизна формула: r ≈ i - π
    real_rate = (1 + nominal_rate) / (1 + inflation_rate) - 1
    return real_rate

def okun_law(gdp_gap: float, natural_unemployment: float = UNEMPLOYMENT_NATURAL) -> float:
    """
    Закон Окена - зв'язок між розривом ВВП та безробіттям.
    
    Параметри:
        gdp_gap: Розрив ВВП (фактичний - потенційний)
        natural_unemployment: Природний рівень безробіття
    
    Повертає:
        Рівень безробіття
    """
    # Закон Окена: u - u_n = -β * (Y - Y_p) / Y_p
    # При β = 0.5: u = u_n - 0.5 * (Y - Y_p) / Y_p
    okun_coefficient = 0.5
    unemployment = natural_unemployment - okun_coefficient * gdp_gap
    return max(0, unemployment)

def quantity_theory_of_money(money_supply: float, velocity: float = MONEY_VELOCITY, 
                           price_level: float = 1.0) -> float:
    """
    Кількісна теорія грошей (MV = PY).
    
    Параметри:
        money_supply: Грошова пропозиція
        velocity: Швидкість обігу грошей
        price_level: Рівень цін
    
    Повертає:
        Реальний ВВП
    """
    if money_supply <= 0 or velocity <= 0:
        raise ValueError("Грошова пропозиція та швидкість обігу повинні бути додатніми")
    
    if price_level <= 0:
        raise ValueError("Рівень цін повинен бути додатнім")
    
    # MV = PY => Y = MV/P
    real_gdp = (money_supply * velocity) / price_level
    return real_gdp

def lorenz_curve(income_distribution: List[float]) -> Tuple[List[float], List[float]]:
    """
    Побудувати криву Лоренца для розподілу доходів.
    
    Параметри:
        income_distribution: Розподіл доходів
    
    Повертає:
        Кортеж (кумулятивна частка населення, кумулятивна частка доходів)
    """
    if not income_distribution:
        return [0.0], [0.0]
    
    # Сортуємо доходи за зростанням
    sorted_incomes = sorted(income_distribution)
    n = len(sorted_incomes)
    
    # Обчислюємо кумулятивні частки
    cumulative_population: List[float] = [float(i) / n for i in range(n + 1)]
    
    # Обчислюємо кумулятивні частки доходів
    total_income = sum(sorted_incomes)
    if total_income == 0:
        cumulative_income: List[float] = [0.0] * (n + 1)
    else:
        cumulative_income = [0.0]
        running_sum = 0.0
        for income in sorted_incomes:
            running_sum += income
            cumulative_income.append(float(running_sum) / total_income)
    
    return cumulative_population, cumulative_income

def tax_revenue(tax_base: float, tax_rate: float, 
               elasticity: float = 0) -> float:
    """
    Обчислити податкові надходження з урахуванням еластичності.
    
    Параметри:
        tax_base: База оподаткування
        tax_rate: Ставка податку
        elasticity: Еластичність бази оподаткування до ставки податку
    
    Повертає:
        Податкові надходження
    """
    if tax_rate < 0 or tax_rate > 1:
        raise ValueError("Ставка податку повинна бути між 0 та 1")
    
    # З урахуванням еластичності: нова база = база * (1 - elasticity * ставка)
    adjusted_base = tax_base * (1 - elasticity * tax_rate)
    revenue = adjusted_base * tax_rate
    return revenue

def portfolio_optimization(expected_returns: List[float], 
                          covariance_matrix: List[List[float]], 
                          risk_free_rate: float = 0.02) -> Dict[str, Union[List[float], float]]:
    """
    Оптимізація портфеля інвестицій (модель Марковіца).
    
    Параметри:
        expected_returns: Очікувані доходності активів
        covariance_matrix: Матриця коваріацій
        risk_free_rate: Безризикова ставка
    
    Повертає:
        Словник з оптимальними вагами та характеристиками портфеля
    """
    n_assets = len(expected_returns)
    
    if n_assets == 0:
        return {'weights': [], 'expected_return': 0, 'risk': 0, 'sharpe_ratio': 0}
    
    if len(covariance_matrix) != n_assets or any(len(row) != n_assets for row in covariance_matrix):
        raise ValueError("Матриця коваріацій повинна бути квадратною та відповідати кількості активів")
    
    # Спрощена оптимізація - рівноважний портфель
    weights = [1.0 / n_assets] * n_assets  # Рівноважні ваги
    
    # Очікувана доходність портфеля
    expected_return = sum(w * r for w, r in zip(weights, expected_returns))
    
    # Ризик портфеля (стандартне відхилення)
    portfolio_variance = 0
    for i in range(n_assets):
        for j in range(n_assets):
            portfolio_variance += weights[i] * weights[j] * covariance_matrix[i][j]
    
    risk = math.sqrt(portfolio_variance)
    
    # Коефіцієнт Шарпа
    sharpe_ratio = (expected_return - risk_free_rate) / risk if risk > 0 else 0
    
    return {
        'weights': weights,
        'expected_return': expected_return,
        'risk': risk,
        'sharpe_ratio': sharpe_ratio
    }

def exchange_rate_parity(interest_rate_domestic: float, 
                        interest_rate_foreign: float,
                        spot_rate: float,
                        time_period: float = 1.0) -> float:
    """
    Паритет процентних ставок для валютних курсів.
    
    Параметри:
        interest_rate_domestic: Внутрішня процентна ставка
        interest_rate_foreign: Зовнішня процентна ставка
        spot_rate: Спот-курс
        time_period: Період часу
    
    Повертає:
        Очікуваний форвардний курс
    """
    # Формула: F = S * (1 + r_d)^t / (1 + r_f)^t
    forward_rate = spot_rate * ((1 + interest_rate_domestic) ** time_period) / ((1 + interest_rate_foreign) ** time_period)
    return forward_rate

def purchasing_power_parity(price_domestic: float, 
                           price_foreign: float,
                           exchange_rate: float) -> float:
    """
    Паритет купівельної спроможності.
    
    Параметри:
        price_domestic: Ціна товару в країні
        price_foreign: Ціна товару за кордоном
        exchange_rate: Валютний курс
    
    Повертає:
        Відношення реального валютного курсу до номінального
    """
    if price_foreign <= 0 or exchange_rate <= 0:
        raise ValueError("Ціна за кордоном та валютний курс повинні бути додатніми")
    
    # Реальний валютний курс: q = e * P_foreign / P_domestic
    real_exchange_rate = exchange_rate * price_foreign / price_domestic if price_domestic > 0 else float('inf')
    return real_exchange_rate

def economic_value_added(capital: float, 
                        cost_of_capital: float,
                        nopat: float) -> float:
    """
    Додана економічна вартість (EVA).
    
    Параметри:
        capital: Інвестований капітал
        cost_of_capital: Вартість капіталу
        nopat: Чиста операційна прибутковість після податків
    
    Повертає:
        Додана економічна вартість
    """
    # EVA = NOPAT - (капітал * вартість капіталу)
    capital_charge = capital * cost_of_capital
    eva = nopat - capital_charge
    return eva

def compound_interest(principal: float, 
                     rate: float, 
                     time: float, 
                     compound_frequency: int = 1) -> float:
    """
    Обчислити складні відсотки.
    
    Параметри:
        principal: Початкова сума
        rate: Річна процентна ставка
        time: Період часу (роки)
        compound_frequency: Частота нарахування відсотків за рік
    
    Повертає:
        Нарощена сума
    """
    if principal < 0:
        raise ValueError("Початкова сума не може бути від'ємною")
    
    if time < 0:
        raise ValueError("Час не може бути від'ємним")
    
    if compound_frequency <= 0:
        raise ValueError("Частота нарахування повинна бути додатньою")
    
    # Формула: A = P(1 + r/n)^(nt)
    amount = principal * (1 + rate / compound_frequency) ** (compound_frequency * time)
    return amount

def present_value(future_value: float, 
                 rate: float, 
                 time: float) -> float:
    """
    Обчислити теперішню вартість.
    
    Параметри:
        future_value: Майбутня вартість
        rate: Ставка дисконтування
        time: Період часу
    
    Повертає:
        Теперішня вартість
    """
    if time < 0:
        raise ValueError("Час не може бути від'ємним")
    
    # Формула: PV = FV / (1 + r)^t
    present_value = future_value / ((1 + rate) ** time)
    return present_value

def net_present_value(cash_flows: List[float], 
                     discount_rate: float) -> float:
    """
    Обчислити чисту теперішню вартість (NPV).
    
    Параметри:
        cash_flows: Грошові потоки (індекс 0 - початкові інвестиції)
        discount_rate: Ставка дисконтування
    
    Повертає:
        Чиста теперішня вартість
    """
    if not cash_flows:
        return 0.0
    
    npv = 0.0
    for t, cash_flow in enumerate(cash_flows):
        npv += cash_flow / ((1 + discount_rate) ** t)
    return npv

def internal_rate_of_return(cash_flows: List[float], 
                           max_iterations: int = 1000,
                           tolerance: float = 1e-6) -> float:
    """
    Обчислити внутрішню норму доходності (IRR).
    
    Параметри:
        cash_flows: Грошові потоки
        max_iterations: Максимальна кількість ітерацій
        tolerance: Точність
    
    Повертає:
        Внутрішня норма доходності
    """
    if not cash_flows:
        raise ValueError("Грошові потоки не можуть бути порожніми")
    
    if len(cash_flows) < 2:
        raise ValueError("Потрібно щонайменше два грошових потоки")
    
    # Метод Ньютона-Рафсона для знаходження IRR
    # Початкове наближення
    irr = 0.1
    
    for _ in range(max_iterations):
        npv = 0.0
        npv_derivative = 0.0
        
        for t, cash_flow in enumerate(cash_flows):
            if t == 0:
                npv += cash_flow
            else:
                npv += cash_flow / ((1 + irr) ** t)
                npv_derivative -= t * cash_flow / ((1 + irr) ** (t + 1))
        
        if abs(npv_derivative) < tolerance:
            break
            
        new_irr = irr - npv / npv_derivative
        
        if abs(new_irr - irr) < tolerance:
            return new_irr
            
        irr = new_irr
    
    return irr

def payback_period(cash_flows: List[float]) -> float:
    """
    Обчислити строк окупності інвестицій.
    
    Параметри:
        cash_flows: Грошові потоки (індекс 0 - початкові інвестиції)
    
    Повертає:
        Строк окупності в роках
    """
    if not cash_flows:
        raise ValueError("Грошові потоки не можуть бути порожніми")
    
    if cash_flows[0] >= 0:
        raise ValueError("Початкові інвестиції повинні бути від'ємними")
    
    cumulative_cash_flow = 0.0
    for t, cash_flow in enumerate(cash_flows):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow >= 0:
            # Якщо точно на межі, повертаємо цілий рік
            if cumulative_cash_flow == 0:
                return float(t)
            # Інакше інтерполюємо
            if t > 0:
                previous_cumulative = cumulative_cash_flow - cash_flow
                fraction = abs(previous_cumulative) / cash_flow
                return t - 1 + fraction
            else:
                return 0.0
    
    # Якщо інвестиції не окупаються
    return float('inf')

def monte_carlo_simulation(initial_value: float, 
                          expected_return: float,
                          volatility: float,
                          time_horizon: float,
                          num_simulations: int = 10000) -> List[float]:
    """
    Монте-Карло симуляція для оцінки ризику інвестицій.
    
    Параметри:
        initial_value: Початкова вартість
        expected_return: Очікувана доходність
        volatility: Волатильність
        time_horizon: Горизонт планування
        num_simulations: Кількість симуляцій
    
    Повертає:
        Список кінцевих значень
    """
    if initial_value <= 0:
        raise ValueError("Початкова вартість повинна бути додатньою")
    
    if num_simulations <= 0:
        raise ValueError("Кількість симуляцій повинна бути додатньою")
    
    if time_horizon <= 0:
        raise ValueError("Горизонт планування повинен бути додатнім")
    
    final_values = []
    
    for _ in range(num_simulations):
        # Генеруємо випадкову доходність
        random_return = random.normalvariate(expected_return, volatility)
        # Обчислюємо кінцеву вартість
        final_value = initial_value * math.exp(random_return * time_horizon)
        final_values.append(final_value)
    
    return final_values

def value_at_risk(returns: List[float], 
                 confidence_level: float = 0.05) -> float:
    """
    Обчислити ризик у вигляді значення VaR.
    
    Параметри:
        returns: Історичні доходності
        confidence_level: Рівень довіри (наприклад, 0.05 для 95%)
    
    Повертає:
        Значення VaR
    """
    if not returns:
        raise ValueError("Доходності не можуть бути порожніми")
    
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Рівень довіри повинен бути між 0 та 1")
    
    # Сортуємо доходності
    sorted_returns = sorted(returns)
    
    # Знаходимо квантиль
    index = int(confidence_level * len(sorted_returns))
    var = -sorted_returns[index]  # VaR зазвичай виражається як додатне число
    return var

def economic_cycle_analysis(gdp_growth_rates: List[float]) -> Dict[str, Union[float, str]]:
    """
    Аналіз економічного циклу.
    
    Параметри:
        gdp_growth_rates: Історія темпів зростання ВВП
    
    Повертає:
        Словник з характеристиками економічного циклу
    """
    if not gdp_growth_rates:
        return {
            'trend_growth': 0.0,
            'volatility': 0.0,
            'cycle_phase': 'Немає даних',
            'amplitude': 0.0
        }
    
    # Трендовий темп зростання
    trend_growth = sum(gdp_growth_rates) / len(gdp_growth_rates)
    
    # Волатильність (стандартне відхилення)
    if len(gdp_growth_rates) > 1:
        mean = trend_growth
        variance = sum((rate - mean) ** 2 for rate in gdp_growth_rates) / (len(gdp_growth_rates) - 1)
        volatility = math.sqrt(variance)
    else:
        volatility = 0.0
    
    # Фаза циклу (спрощений аналіз)
    if len(gdp_growth_rates) >= 2:
        recent_growth = gdp_growth_rates[-1]
        previous_growth = gdp_growth_rates[-2]
        
        if recent_growth > trend_growth and previous_growth < recent_growth:
            cycle_phase = 'Експансія'
        elif recent_growth < trend_growth and previous_growth > recent_growth:
            cycle_phase = 'Рецесія'
        elif recent_growth > trend_growth:
            cycle_phase = 'Відновлення'
        else:
            cycle_phase = 'Пік'
    else:
        cycle_phase = 'Недостатньо даних'
    
    # Амплітуда циклу
    amplitude = max(gdp_growth_rates) - min(gdp_growth_rates) if gdp_growth_rates else 0.0
    
    return {
        'trend_growth': trend_growth,
        'volatility': volatility,
        'cycle_phase': cycle_phase,
        'amplitude': amplitude
    }

def market_concentration( market_shares: List[float]) -> Dict[str, float]:
    """
    Обчислити показники концентрації ринку.
    
    Параметри:
        market_shares: Ринкові частки компаній
    
    Повертає:
        Словник з індексами концентрації
    """
    if not market_shares:
        return {
            'cr4': 0.0,
            'hhi': 0.0,
            'entropy': 0.0
        }
    
    # Нормалізуємо частки
    total_share = sum(market_shares)
    if total_share == 0:
        normalized_shares = [0.0] * len(market_shares)
    else:
        normalized_shares = [share / total_share for share in market_shares]
    
    # CR4 (сума часток 4 найбільших компаній)
    sorted_shares = sorted(normalized_shares, reverse=True)
    cr4 = sum(sorted_shares[:4])
    
    # Індекс Херфіндаля-Хіршмана (HHI)
    hhi = sum(share ** 2 for share in normalized_shares) * 10000  # Зазвичай множиться на 10000
    
    # Ентропія ринку
    entropy = -sum(share * math.log(share) for share in normalized_shares if share > 0)
    
    return {
        'cr4': cr4,
        'hhi': hhi,
        'entropy': entropy
    }

def game_theory_nash_equilibrium(payoff_matrix_a: List[List[float]], 
                                payoff_matrix_b: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Знайти рівновагу Неша в іграх двох осіб.
    
    Параметри:
        payoff_matrix_a: Матриця виплат для гравця A
        payoff_matrix_b: Матриця виплат для гравця B
    
    Повертає:
        Список рівноваг Неша
    """
    if not payoff_matrix_a or not payoff_matrix_b:
        return []
    
    rows = len(payoff_matrix_a)
    cols = len(payoff_matrix_a[0]) if payoff_matrix_a else 0
    
    if rows == 0 or cols == 0:
        return []
    
    # Перевіряємо розміри
    if len(payoff_matrix_b) != rows or any(len(row) != cols for row in payoff_matrix_b):
        raise ValueError("Матриці виплат повинні мати однакові розміри")
    
    nash_equilibria = []
    
    # Для кожної стратегії гравця A
    for i in range(rows):
        # Знаходимо найкращу відповідь гравця B
        max_b_payoff = max(payoff_matrix_b[i][j] for j in range(cols))
        best_responses_b = [j for j in range(cols) if payoff_matrix_b[i][j] == max_b_payoff]
        
        # Для кожної стратегії гравця B
        for j in range(cols):
            # Знаходимо найкращу відповідь гравця A
            max_a_payoff = max(payoff_matrix_a[k][j] for k in range(rows))
            best_responses_a = [k for k in range(rows) if payoff_matrix_a[k][j] == max_a_payoff]
            
            # Якщо (i,j) є взаємно найкращими відповідями, то це рівновага Неша
            if i in best_responses_a and j in best_responses_b:
                nash_equilibria.append((i, j))
    
    return nash_equilibria

def regression_analysis(x_values: List[float], 
                       y_values: List[float]) -> Dict[str, float]:
    """
    Провести регресійний аналіз.
    
    Параметри:
        x_values: Незалежні змінні
        y_values: Залежні змінні
    
    Повертає:
        Словник з результатами регресії
    """
    if len(x_values) != len(y_values):
        raise ValueError("Списки x та y повинні мати однакову довжину")
    
    if len(x_values) < 2:
        raise ValueError("Потрібно щонайменше дві точки для регресії")
    
    n = len(x_values)
    
    # Обчислюємо середні значення
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    
    # Обчислюємо коефіцієнти регресії
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    
    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator
    
    intercept = mean_y - slope * mean_x
    
    # Обчислюємо коефіцієнт детермінації (R²)
    y_predicted = [slope * x + intercept for x in x_values]
    ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(y_values, y_predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in y_values)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Стандартна помилка оцінки
    standard_error = math.sqrt(ss_res / (n - 2)) if n > 2 else 0.0
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'standard_error': standard_error,
        'n': n
    }

def time_series_forecast(data: List[float], 
                        forecast_periods: int = 1) -> List[float]:
    """
    Прогнозування часових рядів методом експоненційного згладжування.
    
    Параметри:
        data: Історичні дані
        forecast_periods: Кількість періодів для прогнозу
    
    Повертає:
        Список прогнозованих значень
    """
    if not data:
        return [0.0] * forecast_periods
    
    if forecast_periods <= 0:
        return []
    
    # Параметр згладжування (альфа)
    alpha = 0.3
    
    # Ініціалізуємо згладжені значення
    smoothed = [data[0]]
    
    # Застосовуємо експоненційне згладжування
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        smoothed.append(smoothed_value)
    
    # Прогнозуємо майбутні значення
    forecasts = []
    last_smoothed = smoothed[-1]
    
    for _ in range(forecast_periods):
        forecasts.append(last_smoothed)
    
    return forecasts

def economic_indicator_synthesis(gdp: float, 
                                inflation: float, 
                                unemployment: float,
                                interest_rate: float) -> Dict[str, Union[float, str]]:
    """
    Синтез економічних індикаторів у комплексний показник.
    
    Параметри:
        gdp: Темп зростання ВВП
        inflation: Рівень інфляції
        unemployment: Рівень безробіття
        interest_rate: Процентна ставка
    
    Повертає:
        Словник з комплексними економічними показниками
    """
    # Нормалізуємо показники до шкали 0-100
    # ВВП: 0-5% = 0-100 (більше краще)
    normalized_gdp = min(100.0, max(0.0, gdp * 20.0))
    
    # Інфляція: 0-10% = 100-0 (оптимально 2%)
    if inflation <= 2.0:
        normalized_inflation = 100.0 - abs(inflation - 2.0) * 10.0
    else:
        normalized_inflation = max(0.0, 100.0 - (inflation - 2.0) * 10.0)
    
    # Безробіття: 0-20% = 100-0 (оптимально 5%)
    if unemployment <= 5.0:
        normalized_unemployment = 100.0 - abs(unemployment - 5.0) * 5.0
    else:
        normalized_unemployment = max(0.0, 100.0 - (unemployment - 5.0) * 5.0)
    
    # Процентна ставка: 0-20% = 100-0 (оптимально 3%)
    if interest_rate <= 3.0:
        normalized_interest = 100.0 - abs(interest_rate - 3.0) * 10.0
    else:
        normalized_interest = max(0.0, 100.0 - (interest_rate - 3.0) * 10.0)
    
    # Композитний індекс
    composite_index = (normalized_gdp + normalized_inflation + 
                      normalized_unemployment + normalized_interest) / 4.0
    
    # Економічне здоров'я
    if composite_index >= 80.0:
        health = "Дуже добре"
    elif composite_index >= 60.0:
        health = "Добре"
    elif composite_index >= 40.0:
        health = "Задовільно"
    elif composite_index >= 20.0:
        health = "Погано"
    else:
        health = "Дуже погано"
    
    return {
        'composite_index': composite_index,
        'gdp_component': normalized_gdp,
        'inflation_component': normalized_inflation,
        'unemployment_component': normalized_unemployment,
        'interest_rate_component': normalized_interest,
        'economic_health': health
    }
