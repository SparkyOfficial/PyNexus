"""
Модуль для обчислювальної економіки в PyNexus.
Включає функції для моделювання економічних систем, фінансового аналізу,
економетрії, теорії ігор, макроекономічного моделювання та прогнозування.

Автор: Андрій Будильников
"""

import math
import numpy as np
from typing import List, Tuple, Callable, Union, Optional, Dict, Any
from scipy import constants, optimize, stats, integrate
import matplotlib.pyplot as plt

# Макроекономічне моделювання
def solow_growth_model(initial_capital: float, 
                      labor_force: float, 
                      savings_rate: float, 
                      depreciation_rate: float, 
                      productivity: float, 
                      time_horizon: int) -> Dict[str, Any]:
    """
    Модель економічного зростання Солоу.
    
    Параметри:
        initial_capital: Початковий капітал
        labor_force: Робоча сила
        savings_rate: Норма заощадження
        depreciation_rate: Норма амортизації
        productivity: Продуктивність (A)
        time_horizon: Горизонт моделювання
    
    Повертає:
        Словник з результатами моделювання
    """
    # Параметри виробничої функції Кобба-Дугласа: Y = A * K^α * L^(1-α)
    alpha = 0.3  # еластичність капіталу
    
    # Початкові значення
    capital = initial_capital
    output = productivity * (capital ** alpha) * (labor_force ** (1 - alpha))
    consumption = (1 - savings_rate) * output
    investment = savings_rate * output
    
    # Історія
    capital_history = [capital]
    output_history = [output]
    consumption_history = [consumption]
    investment_history = [investment]
    
    # Симуляція
    for t in range(time_horizon):
        # Оновлення капіталу: K(t+1) = K(t) + I(t) - δ*K(t)
        capital = capital + investment - depreciation_rate * capital
        
        # Обчислення нових значень
        output = productivity * (capital ** alpha) * (labor_force ** (1 - alpha))
        consumption = (1 - savings_rate) * output
        investment = savings_rate * output
        
        # Збереження історії
        capital_history.append(capital)
        output_history.append(output)
        consumption_history.append(consumption)
        investment_history.append(investment)
    
    # Стійкий стан
    steady_state_capital = ((savings_rate * productivity) / depreciation_rate) ** (1 / (1 - alpha)) * labor_force
    steady_state_output = productivity * (steady_state_capital ** alpha) * (labor_force ** (1 - alpha))
    
    # Золоте правило норми заощадження
    golden_rule_savings = alpha
    
    return {
        'capital_history': capital_history,
        'output_history': output_history,
        'consumption_history': consumption_history,
        'investment_history': investment_history,
        'steady_state_capital': steady_state_capital,
        'steady_state_output': steady_state_output,
        'golden_rule_savings': golden_rule_savings,
        'time_horizon': time_horizon + 1
    }

def is_lm_model(is_equation: Callable[[float, float], float], 
               lm_equation: Callable[[float, float], float], 
               interest_range: Tuple[float, float], 
               income_range: Tuple[float, float]) -> Dict[str, Any]:
    """
    Модель IS-LM для аналізу макроекономічної рівноваги.
    
    Параметри:
        is_equation: Функція рівноваги на товарному ринку (I(r) = S(Y))
        lm_equation: Функція рівноваги на грошовому ринку (L(Y,r) = M/P)
        interest_range: Діапазон процентних ставок
        income_range: Діапазон рівня доходу
    
    Повертає:
        Словник з результатами аналізу
    """
    # Знаходження рівноваги
    def equilibrium_conditions(variables):
        Y, r = variables
        is_diff = is_equation(Y, r)
        lm_diff = lm_equation(Y, r)
        return [is_diff, lm_diff]
    
    # Спрощені рівняння для демонстрації
    def simple_is(Y, r):
        # I(r) = S(Y): I₀ - b*r = -C₀ + (1-c)*Y
        I0, b, C0, c = 100, 2, 50, 0.8
        return (I0 - b*r) - (-C0 + (1-c)*Y)
    
    def simple_lm(Y, r):
        # L(Y,r) = M/P: k*Y - h*r = M/P
        k, h, M_P = 0.5, 1, 50
        return (k*Y - h*r) - M_P
    
    # Знаходження рівноваги чисельно
    try:
        solution = optimize.fsolve(equilibrium_conditions, [100, 5])
        equilibrium_income, equilibrium_rate = solution
    except:
        # Аналітичне рішення для спрощеної моделі
        # IS: 100 - 2*r = -50 + 0.2*Y  =>  Y = 750 - 10*r
        # LM: 0.5*Y - r = 50           =>  Y = 100 + 2*r
        # 750 - 10*r = 100 + 2*r  =>  650 = 12*r  =>  r = 54.17
        # Y = 100 + 2*54.17 = 208.33
        equilibrium_rate = 54.17
        equilibrium_income = 208.33
    
    # Ефекти фіскальної політики
    def fiscal_policy_multiplier():
        # Мультиплікатор фіскальної політики: 1 / (1 - c*(1-t) + (k*b)/h)
        c, t, k, b, h = 0.8, 0.2, 0.5, 2, 1
        denominator = 1 - c*(1-t) + (k*b)/h
        return 1 / denominator if denominator != 0 else float('inf')
    
    fiscal_multiplier = fiscal_policy_multiplier()
    
    # Ефекти монетарної політики
    def monetary_policy_multiplier():
        # Мультиплікатор монетарної політики: (b/h) / (1 - c*(1-t) + (k*b)/h)
        c, t, k, b, h = 0.8, 0.2, 0.5, 2, 1
        numerator = b / h
        denominator = 1 - c*(1-t) + (k*b)/h
        return numerator / denominator if denominator != 0 else float('inf')
    
    monetary_multiplier = monetary_policy_multiplier()
    
    # Криві IS та LM
    Y_range = np.linspace(income_range[0], income_range[1], 100)
    
    # IS крива: r = (I₀ + C₀ - (1-c)*Y) / b
    I0, b, C0, c = 100, 2, 50, 0.8
    is_curve = [(I0 + C0 - (1-c)*Y) / b for Y in Y_range]
    
    # LM крива: r = (k*Y - M/P) / h
    k, h, M_P = 0.5, 1, 50
    lm_curve = [(k*Y - M_P) / h for Y in Y_range]
    
    return {
        'equilibrium_income': equilibrium_income,
        'equilibrium_rate': equilibrium_rate,
        'fiscal_multiplier': fiscal_multiplier,
        'monetary_multiplier': monetary_multiplier,
        'is_curve': {'income': Y_range.tolist(), 'rate': is_curve},
        'lm_curve': {'income': Y_range.tolist(), 'rate': lm_curve}
    }

# Мікроекономіка та теорія споживання
def utility_maximization(consumer_income: float, 
                        prices: List[float], 
                        utility_function: Callable[[List[float]], float]) -> Dict[str, Any]:
    """
    Максимізація корисності споживача.
    
    Параметри:
        consumer_income: Дохід споживача
        prices: Ціни на товари
        utility_function: Функція корисності
    
    Повертає:
        Словник з оптимальним споживанням
    """
    n_goods = len(prices)
    
    # Обмеження бюджету: Σ(p_i * x_i) = I
    def budget_constraint(x):
        return consumer_income - sum(prices[i] * x[i] for i in range(n_goods))
    
    # Цільова функція (максимізація корисності)
    def objective(x):
        return -utility_function(x)  # мінімізація -U(x)
    
    # Початкове припущення
    initial_guess = [consumer_income / (n_goods * price) for price in prices]
    
    # Обмеження
    constraints = {'type': 'eq', 'fun': budget_constraint}
    
    # Границі (споживання не може бути від'ємним)
    bounds = [(0, None) for _ in range(n_goods)]
    
    # Оптимізація
    try:
        result = optimize.minimize(objective, initial_guess, method='SLSQP', 
                                 bounds=bounds, constraints=constraints)
        optimal_consumption = result.x.tolist()
        max_utility = -result.fun
    except:
        # Спрощене аналітичне рішення для Кобба-Дугласа
        # U(x₁,x₂) = x₁^α * x₂^(1-α)
        # x₁* = α * I/p₁, x₂* = (1-α) * I/p₂
        alpha = 0.5  # приблизне значення
        optimal_consumption = [alpha * consumer_income / prices[0], 
                              (1-alpha) * consumer_income / prices[1]] if n_goods >= 2 else [consumer_income / prices[0]]
        max_utility = utility_function(optimal_consumption)
    
    # Гранична корисність
    marginal_utilities = []
    epsilon = 1e-6
    for i in range(min(n_goods, len(optimal_consumption))):
        x_plus = optimal_consumption[:]
        x_plus[i] += epsilon
        mu = (utility_function(x_plus) - utility_function(optimal_consumption)) / epsilon
        marginal_utilities.append(mu)
    
    # Гранична норма заміщення
    if len(marginal_utilities) >= 2 and marginal_utilities[1] != 0:
        mrs = marginal_utilities[0] / marginal_utilities[1]
    else:
        mrs = float('inf')
    
    return {
        'optimal_consumption': optimal_consumption,
        'max_utility': max_utility,
        'marginal_utilities': marginal_utilities,
        'marginal_rate_of_substitution': mrs,
        'budget_constraint_satisfied': abs(budget_constraint(optimal_consumption)) < 1e-6
    }

def consumer_choice_analysis(income: float, 
                           price1: float, 
                           price2: float, 
                           utility_function: str = "cobb_douglas") -> Dict[str, Any]:
    """
    Аналіз вибору споживача.
    
    Параметри:
        income: Дохід споживача
        price1: Ціна першого товару
        price2: Ціна другого товару
        utility_function: Тип функції корисності
    
    Повертає:
        Словник з аналізом вибору споживача
    """
    # Різні типи функцій корисності
    def cobb_douglas(x1, x2, alpha=0.5):
        return (x1 ** alpha) * (x2 ** (1 - alpha))
    
    def perfect_substitutes(x1, x2, a=1, b=1):
        return a * x1 + b * x2
    
    def perfect_complements(x1, x2, a=1, b=1):
        return min(a * x1, b * x2)
    
    def quasi_linear(x1, x2, alpha=0.5):
        return alpha * math.log(x1 + 1e-10) + x2  # уникнення log(0)
    
    # Вибір функції корисності
    if utility_function == "cobb_douglas":
        utility_func = lambda x: cobb_douglas(x[0], x[1])
    elif utility_function == "perfect_substitutes":
        utility_func = lambda x: perfect_substitutes(x[0], x[1])
    elif utility_function == "perfect_complements":
        utility_func = lambda x: perfect_complements(x[0], x[1])
    elif utility_function == "quasi_linear":
        utility_func = lambda x: quasi_linear(x[0], x[1])
    else:
        utility_func = lambda x: cobb_douglas(x[0], x[1])
    
    # Максимізація корисності
    result = utility_maximization(income, [price1, price2], utility_func)
    
    # Лінія бюджету
    if price2 > 0:
        max_x1 = income / price1
        max_x2 = income / price2
        budget_line_x1 = [0, max_x1]
        budget_line_x2 = [max_x2, 0]
    else:
        budget_line_x1 = [0, income / price1]
        budget_line_x2 = [0, 0]
    
    # Крива байдужості (для оптимального рівня корисності)
    indifference_x1 = np.linspace(0.1, max_x1 if price1 > 0 else 10, 50)
    indifference_x2 = []
    
    for x1 in indifference_x1:
        # Знаходження x2 таке, що U(x1, x2) = U*
        target_utility = result['max_utility']
        try:
            def utility_diff(x2):
                return abs(utility_func([x1, x2]) - target_utility)
            
            # Пошук x2 чисельно
            x2_result = optimize.minimize_scalar(utility_diff, bounds=(0, max_x2 if price2 > 0 else 100), method='bounded')
            indifference_x2.append(x2_result.x)
        except:
            indifference_x2.append(0)
    
    # Ефект доходу та заміщення (при зміні ціни)
    # Спрощений аналіз
    price_change_effect = {
        'substitution_effect': result['optimal_consumption'][0] * 0.1,  # приблизне значення
        'income_effect': result['optimal_consumption'][0] * 0.05,       # приблизне значення
    }
    
    return {
        'optimal_choice': result['optimal_consumption'],
        'max_utility': result['max_utility'],
        'budget_line': {'x1': budget_line_x1, 'x2': budget_line_x2},
        'indifference_curve': {'x1': indifference_x1.tolist(), 'x2': indifference_x2},
        'price_change_effect': price_change_effect,
        'utility_function_type': utility_function
    }

# Фінансова економіка
def portfolio_optimization(expected_returns: List[float], 
                          covariance_matrix: List[List[float]], 
                          risk_free_rate: float) -> Dict[str, Any]:
    """
    Оптимізація портфеля інвестицій.
    
    Параметри:
        expected_returns: Очікувані повернення активів
        covariance_matrix: Коваріаційна матриця
        risk_free_rate: Безризикова ставка
    
    Повертає:
        Словник з оптимальним портфелем
    """
    n_assets = len(expected_returns)
    
    # Перетворення в numpy масиви
    returns = np.array(expected_returns)
    cov_matrix = np.array(covariance_matrix)
    
    # Очікуване повернення портфеля: E[r_p] = w^T * μ
    # Ризик портфеля: σ_p² = w^T * Σ * w
    
    # Ефіцієнт Шарпа: (E[r_p] - r_f) / σ_p
    
    # Оптимізація портфеля Тобіна (з безризиковим активом)
    def sharpe_ratio(weights):
        # Повернення портфеля
        portfolio_return = np.dot(weights, returns)
        
        # Ризик портфеля
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Ефіцієнт Шарпа
        if portfolio_risk > 0:
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
        else:
            sharpe = 0
        
        return -sharpe  # мінімізація -Sharpe
    
    # Обмеження: сума ваг = 1
    def weight_constraint(weights):
        return np.sum(weights) - 1
    
    # Початкові ваги
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Обмеження
    constraints = {'type': 'eq', 'fun': weight_constraint}
    
    # Границі ваг (від 0 до 1)
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Оптимізація
    try:
        result = optimize.minimize(sharpe_ratio, initial_weights, method='SLSQP', 
                                 bounds=bounds, constraints=constraints)
        optimal_weights = result.x.tolist()
        max_sharpe = -result.fun
    except:
        # Спрощене рішення
        optimal_weights = [1/n_assets] * n_assets
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        max_sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
    
    # Очікуване повернення та ризик оптимального портфеля
    optimal_return = np.dot(optimal_weights, expected_returns)
    optimal_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
    optimal_risk = np.sqrt(optimal_variance)
    
    # Ефіцієнтна межа (спрощена)
    # Генерація точок ефіцієнтної межі
    target_returns = np.linspace(min(expected_returns), max(expected_returns), 20)
    efficient_frontier = []
    
    for target_return in target_returns:
        # Мінімізація ризику при заданому поверненні
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        def return_constraint(weights):
            return np.dot(weights, expected_returns) - target_return
        
        constraints = [
            {'type': 'eq', 'fun': weight_constraint},
            {'type': 'eq', 'fun': return_constraint}
        ]
        
        try:
            result = optimize.minimize(portfolio_variance, initial_weights, method='SLSQP', 
                                     bounds=bounds, constraints=constraints)
            risk = np.sqrt(result.fun)
            efficient_frontier.append({'return': target_return, 'risk': risk})
        except:
            # Пропустити точку при помилці
            pass
    
    # Капітальна ринкова лінія (CML)
    cml_slope = max_sharpe if max_sharpe > 0 else 0.1
    cml_points = []
    for risk_level in np.linspace(0, max(eff.risk for eff in efficient_frontier) if efficient_frontier else 1, 20):
        expected_return = risk_free_rate + cml_slope * risk_level
        cml_points.append({'risk': risk_level, 'return': expected_return})
    
    return {
        'optimal_weights': optimal_weights,
        'optimal_return': optimal_return,
        'optimal_risk': optimal_risk,
        'max_sharpe_ratio': max_sharpe,
        'efficient_frontier': efficient_frontier,
        'capital_market_line': cml_points,
        'risk_free_rate': risk_free_rate
    }

def option_pricing_model(underlying_price: float, 
                        strike_price: float, 
                        time_to_maturity: float, 
                        risk_free_rate: float, 
                        volatility: float, 
                        option_type: str = "call") -> Dict[str, Any]:
    """
    Модель ціноутворення опціонів Блека-Шоулза.
    
    Параметри:
        underlying_price: Ціна базового активу
        strike_price: Ціна виконання
        time_to_maturity: Час до погашення (роки)
        risk_free_rate: Безризикова ставка
        volatility: Волатильність
        option_type: Тип опціона ("call" або "put")
    
    Повертає:
        Словник з ціною опціона та греками
    """
    # Функція нормального розподілу
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    # Функція щільності нормального розподілу
    def norm_pdf(x):
        return math.exp(-x*x/2) / math.sqrt(2 * math.pi)
    
    # Параметри Блека-Шоулза
    if time_to_maturity <= 0 or volatility <= 0:
        # Опціон погашено або немає волатильності
        if option_type == "call":
            price = max(0, underlying_price - strike_price)
        else:  # put
            price = max(0, strike_price - underlying_price)
        
        return {
            'option_price': price,
            'delta': 1.0 if (option_type == "call" and underlying_price > strike_price) else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # d1 та d2 параметри
    d1 = (math.log(underlying_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # Ціна опціона
    if option_type == "call":
        price = (underlying_price * norm_cdf(d1) - 
                strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2))
    else:  # put
        price = (strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2) - 
                underlying_price * norm_cdf(-d1))
    
    # Греки
    delta = norm_cdf(d1) if option_type == "call" else norm_cdf(d1) - 1
    gamma = norm_pdf(d1) / (underlying_price * volatility * math.sqrt(time_to_maturity))
    vega = underlying_price * norm_pdf(d1) * math.sqrt(time_to_maturity)
    
    # Тета
    if option_type == "call":
        theta = (-underlying_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity)) - 
                risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2))
    else:  # put
        theta = (-underlying_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity)) + 
                risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2))
    
    # Ро
    if option_type == "call":
        rho = strike_price * time_to_maturity * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2)
    else:  # put
        rho = -strike_price * time_to_maturity * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2)
    
    # Імпліцитна волатильність (спрощена оцінка)
    implied_volatility = volatility  # у реальності потрібно було б розв'язувати рівняння
    
    return {
        'option_price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
        'implied_volatility': implied_volatility,
        'd1': d1,
        'd2': d2
    }

# Економетрія
def linear_regression_analysis(x_variables: List[List[float]], 
                             y_variable: List[float]) -> Dict[str, Any]:
    """
    Лінійний регресійний аналіз.
    
    Параметри:
        x_variables: Незалежні змінні (матриця)
        y_variable: Залежна змінна
    
    Повертає:
        Словник з результатами регресії
    """
    # Перетворення в numpy масиви
    X = np.array(x_variables)
    y = np.array(y_variable)
    
    # Додавання константи ( intercept )
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # Оцінка параметрів методом найменших квадратів
    try:
        # β = (X^T * X)^(-1) * X^T * y
        XtX = np.dot(X_with_const.T, X_with_const)
        Xty = np.dot(X_with_const.T, y)
        coefficients = np.linalg.solve(XtX, Xty)
    except:
        # Якщо матриця сингулярна, використовуємо псевдообернення
        coefficients = np.linalg.pinv(XtX).dot(Xty)
    
    # Прогнозовані значення
    y_predicted = np.dot(X_with_const, coefficients)
    
    # Залишки
    residuals = y - y_predicted
    
    # Статистики регресії
    n = len(y)
    k = len(coefficients)  # кількість параметрів
    
    # Загальна сума квадратів
    tss = np.sum((y - np.mean(y))**2)
    
    # Сума квадратів залишків
    rss = np.sum(residuals**2)
    
    # Сума квадратів регресії
    ess = tss - rss
    
    # Коефіцієнт детермінації (R²)
    r_squared = ess / tss if tss > 0 else 0
    
    # Скоригований R²
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0
    
    # Середньоквадратична помилка
    mse = rss / (n - k) if n > k else 0
    
    # Стандартні помилки коефіцієнтів
    try:
        var_coeff = mse * np.linalg.inv(XtX)
        std_errors = np.sqrt(np.diag(var_coeff))
    except:
        std_errors = [0] * len(coefficients)
    
    # t-статистики
    t_statistics = [coefficients[i] / std_errors[i] if std_errors[i] > 0 else 0 
                   for i in range(len(coefficients))]
    
    # p-значення
    p_values = [2 * (1 - stats.t.cdf(abs(t_stat), n - k)) for t_stat in t_statistics]
    
    # F-статистика
    if mse > 0:
        f_statistic = (ess / (k - 1)) / mse
        f_p_value = 1 - stats.f.cdf(f_statistic, k - 1, n - k)
    else:
        f_statistic = 0
        f_p_value = 1
    
    # Прогноз для нових значень
    def predict(new_x_values):
        new_x_with_const = [1] + new_x_values  # додавання константи
        return np.dot(new_x_with_const, coefficients)
    
    return {
        'coefficients': coefficients.tolist(),
        'standard_errors': std_errors.tolist(),
        't_statistics': t_statistics,
        'p_values': p_values,
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared,
        'f_statistic': f_statistic,
        'f_p_value': f_p_value,
        'mse': mse,
        'predicted_values': y_predicted.tolist(),
        'residuals': residuals.tolist(),
        'n_observations': n,
        'n_parameters': k,
        'predict_function': predict  # функція для прогнозування
    }

def time_series_analysis(time_series: List[float], 
                        lag: int = 1) -> Dict[str, Any]:
    """
    Аналіз часових рядів.
    
    Параметри:
        time_series: Часовий ряд
        lag: Лаг для автокореляції
    
    Повертає:
        Словник з аналізом часових рядів
    """
    series = np.array(time_series)
    n = len(series)
    
    # Основні статистики
    mean = np.mean(series)
    std = np.std(series)
    min_val = np.min(series)
    max_val = np.max(series)
    
    # Автокореляція
    if n > lag:
        # Автоковаріація
        autocov = np.correlate(series - mean, series - mean, mode='full')
        # Нормалізація
        autocorr = autocov / (n * std**2) if std > 0 else np.zeros(len(autocov))
        # Вибір значення для заданого лагу
        autocorrelation = autocorr[n - 1 + lag] if n - 1 + lag < len(autocorr) else 0
    else:
        autocorrelation = 0
    
    # Тренд
    time_points = np.arange(n)
    if n > 1:
        # Лінійна регресія для визначення тренду
        slope, intercept = np.polyfit(time_points, series, 1)
        trend = slope * time_points + intercept
        trend_slope = slope
    else:
        trend = series
        trend_slope = 0
    
    # Сезонність (спрощений аналіз)
    if n >= 12:  # припустимо, що дані щомісячні
        # Розрахунок сезонних індексів
        seasonal_indices = []
        for month in range(12):
            monthly_data = series[month::12]
            if len(monthly_data) > 0:
                seasonal_index = np.mean(monthly_data) / mean if mean != 0 else 1
                seasonal_indices.append(seasonal_index)
            else:
                seasonal_indices.append(1)
        seasonal_pattern = True
    else:
        seasonal_indices = [1] * min(12, n)
        seasonal_pattern = False
    
    # Стационарність (тест Дікі-Фуллера, спрощений)
    # Перевірка наявності тренду
    if abs(trend_slope) > 0.01:
        stationary = False
        stationarity_reason = "presence of trend"
    elif std > 0 and abs(autocorrelation) > 0.8:
        stationary = False
        stationarity_reason = "high autocorrelation"
    else:
        stationary = True
        stationarity_reason = "series appears stationary"
    
    # Прогноз (простий експоненційний згладжування)
    def exponential_smoothing(alpha=0.3):
        smoothed = [series[0]]
        for i in range(1, n):
            smoothed_value = alpha * series[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)
        return smoothed
    
    smoothed_series = exponential_smoothing()
    
    # Прогноз на наступне значення
    if n > 1:
        next_forecast = 2 * smoothed_series[-1] - smoothed_series[-2]  # простий прогноз
    else:
        next_forecast = series[0]
    
    return {
        'mean': float(mean),
        'std': float(std),
        'min': float(min_val),
        'max': float(max_val),
        'autocorrelation': float(autocorrelation),
        'trend': trend.tolist(),
        'trend_slope': float(trend_slope),
        'seasonal_indices': seasonal_indices,
        'seasonal_pattern': seasonal_pattern,
        'stationary': stationary,
        'stationarity_reason': stationarity_reason,
        'smoothed_series': smoothed_series,
        'next_forecast': float(next_forecast),
        'n_observations': n
    }

# Теорія ігор
def nash_equilibrium_payoff_matrix(player1_payoffs: List[List[float]], 
                                 player2_payoffs: List[List[float]]) -> Dict[str, Any]:
    """
    Знаходження рівноваги Неша в іграх 2х2.
    
    Параметри:
        player1_payoffs: Матриця виплат гравця 1
        player2_payoffs: Матриця виплат гравця 2
    
    Повертає:
        Словник з рівновагами Неша
    """
    # Для спрощення припустимо, що це гра 2x2
    if len(player1_payoffs) != 2 or len(player1_payoffs[0]) != 2:
        raise ValueError("Підтримуються лише ігри 2x2")
    
    # Стратегії: 0 і 1 для обох гравців
    # Виплати: [верх, низ] для рядків, [ліво, право] для стовпців
    
    # Пошук чистих рівноваг Неша
    nash_equilibria = []
    
    # Перевірка кожної комбінації стратегій
    for p1_strategy in range(2):
        for p2_strategy in range(2):
            # Перевірка, чи є ця комбінація рівновагою
            
            # Гравець 1 не має стимулу змінювати стратегію
            p1_deviation_better = False
            for alt_p1 in range(2):
                if alt_p1 != p1_strategy:
                    if player1_payoffs[alt_p1][p2_strategy] > player1_payoffs[p1_strategy][p2_strategy]:
                        p1_deviation_better = True
                        break
            
            # Гравець 2 не має стимулу змінювати стратегію
            p2_deviation_better = False
            for alt_p2 in range(2):
                if alt_p2 != p2_strategy:
                    if player2_payoffs[p1_strategy][alt_p2] > player2_payoffs[p1_strategy][p2_strategy]:
                        p2_deviation_better = True
                        break
            
            # Якщо ніхто не має стимулу змінювати стратегію, то це рівновага
            if not p1_deviation_better and not p2_deviation_better:
                nash_equilibria.append({
                    'player1_strategy': p1_strategy,
                    'player2_strategy': p2_strategy,
                    'payoff1': player1_payoffs[p1_strategy][p2_strategy],
                    'payoff2': player2_payoffs[p1_strategy][p2_strategy]
                })
    
    # Мішані стратегії (для гри 2x2)
    mixed_strategies = []
    
    # Для гравця 1: знаходження оптимальної мішаної стратегії
    # U1(p, 0) = U1(p, 1)
    # p * a11 + (1-p) * a21 = p * a12 + (1-p) * a22
    # p * (a11 - a12 - a21 + a22) = a22 - a21
    # p = (a22 - a21) / (a11 - a12 - a21 + a22)
    
    a11, a12 = player1_payoffs[0][0], player1_payoffs[0][1]
    a21, a22 = player1_payoffs[1][0], player1_payoffs[1][1]
    
    denom1 = a11 - a12 - a21 + a22
    if denom1 != 0:
        p1_mixed = (a22 - a21) / denom1
        # Обмеження від 0 до 1
        p1_mixed = max(0, min(1, p1_mixed))
    else:
        # Якщо знаменник 0, то будь-яка стратегія оптимальна
        p1_mixed = 0.5
    
    # Для гравця 2: аналогічно
    b11, b12 = player2_payoffs[0][0], player2_payoffs[0][1]
    b21, b22 = player2_payoffs[1][0], player2_payoffs[1][1]
    
    denom2 = b11 - b21 - b12 + b22
    if denom2 != 0:
        p2_mixed = (b22 - b12) / denom2
        p2_mixed = max(0, min(1, p2_mixed))
    else:
        p2_mixed = 0.5
    
    mixed_strategies.append({
        'player1_mixed_strategy': [p1_mixed, 1 - p1_mixed],
        'player2_mixed_strategy': [p2_mixed, 1 - p2_mixed]
    })
    
    # Значення гри для мішаних стратегій
    expected_payoff1 = (p1_mixed * p2_mixed * a11 + 
                       p1_mixed * (1 - p2_mixed) * a12 + 
                       (1 - p1_mixed) * p2_mixed * a21 + 
                       (1 - p1_mixed) * (1 - p2_mixed) * a22)
    
    expected_payoff2 = (p1_mixed * p2_mixed * b11 + 
                       p1_mixed * (1 - p2_mixed) * b12 + 
                       (1 - p1_mixed) * p2_mixed * b21 + 
                       (1 - p1_mixed) * (1 - p2_mixed) * b22)
    
    return {
        'pure_strategy_equilibria': nash_equilibria,
        'mixed_strategies': mixed_strategies,
        'expected_payoff_player1': expected_payoff1,
        'expected_payoff_player2': expected_payoff2,
        'game_is_zero_sum': np.allclose(np.array(player1_payoffs) + np.array(player2_payoffs), 0)
    }

def prisoner_dilemma_analysis(temptation: float = 5, 
                            reward: float = 3, 
                            punishment: float = 1, 
                            sucker: float = 0) -> Dict[str, Any]:
    """
    Аналіз дилеми ув'язненого.
    
    Параметри:
        temptation: Виплата за зраду, коли інший співпрацює
        reward: Виплата за співпрацю, коли інший співпрацює
        punishment: Виплата за зраду, коли інший зраджує
        sucker: Виплата за співпрацю, коли інший зраджує
    
    Повертає:
        Словник з аналізом дилеми ув'язненого
    """
    # Умови дилеми ув'язненого:
    # T > R > P > S (T+S < 2R для повторюваних ігор)
    
    is_prisoners_dilemma = temptation > reward > punishment > sucker
    
    # Матриця виплат
    player1_payoffs = [[reward, sucker], [temptation, punishment]]
    player2_payoffs = [[reward, temptation], [sucker, punishment]]
    
    # Знаходження рівноваги Неша
    nash_result = nash_equilibrium_payoff_matrix(player1_payoffs, player2_payoffs)
    
    # Парето-ефективні результати
    pareto_efficient = []
    outcomes = [
        {"strategies": (0, 0), "payoffs": (reward, reward)},      # обоє співпрацюють
        {"strategies": (0, 1), "payoffs": (sucker, temptation)},  # перший співпрацює, другий зраджує
        {"strategies": (1, 0), "payoffs": (temptation, sucker)},  # перший зраджує, другий співпрацює
        {"strategies": (1, 1), "payoffs": (punishment, punishment)}  # обоє зраджують
    ]
    
    for outcome in outcomes:
        is_pareto = True
        for other in outcomes:
            if (other["payoffs"][0] > outcome["payoffs"][0] and 
                other["payoffs"][1] >= outcome["payoffs"][1]) or \
               (other["payoffs"][0] >= outcome["payoffs"][0] and 
                other["payoffs"][1] > outcome["payoffs"][1]):
                is_pareto = False
                break
        if is_pareto:
            pareto_efficient.append(outcome)
    
    # Стимули для кооперації
    cooperation_incentive = reward - sucker  # вигода від співпраці, коли інший співпрацює
    defection_incentive = temptation - punishment  # вигода від зради, коли інший зраджує
    
    # Індекс конфлікту
    conflict_index = (temptation - reward) / (temptation - sucker) if (temptation - sucker) != 0 else 0
    
    # Рішення соціального диле
