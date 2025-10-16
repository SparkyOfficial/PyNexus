"""
Модуль для обчислювальної фінансової математики в PyNexus.
Включає функції для фінансового аналізу, оцінки активів, управління ризиками,
кількісної торгівлі та фінансового моделювання.

Автор: Андрій Будильников
"""

# Спроба імпорту бібліотек з безпечними заглушками
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy import stats, optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    optimize = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from typing import List, Tuple, Callable, Union, Optional, Dict, Any
import math
import random

# Допоміжні функції для роботи без зовнішніх бібліотек
def safe_import_call(module, func_name, *args, **kwargs):
    """Безпечний виклик функцій з імпортованих модулів"""
    try:
        if module is not None:
            func = getattr(module, func_name, None)
            if func is not None:
                return func(*args, **kwargs)
    except:
        pass
    return None

def create_array(size, value=0):
    """Створення масиву без numpy"""
    if NUMPY_AVAILABLE and np is not None:
        try:
            return np.full(size, value)
        except:
            pass
    return [value] * size

def create_zeros_array(size):
    """Створення масиву з нулями без numpy"""
    if NUMPY_AVAILABLE and np is not None:
        try:
            return np.zeros(size)
        except:
            pass
    return [0.0] * size

def my_linspace(start, stop, num):
    """Створення рівномірно розподілених значень без numpy"""
    if NUMPY_AVAILABLE and np is not None:
        try:
            return np.linspace(start, stop, num)
        except:
            pass
    
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1) if num > 1 else 0
    return [start + i * step for i in range(num)]

def safe_get_item(arr, index, default=0):
    """Безпечне отримання елемента масиву"""
    try:
        if isinstance(arr, list) and 0 <= index < len(arr):
            return arr[index]
        elif hasattr(arr, '__getitem__') and hasattr(arr, '__len__'):
            # Для numpy масивів та інших послідовностей
            if 0 <= index < len(arr):
                return arr[index]
        return default
    except:
        return default

# Оцінка активів
def black_scholes_option_pricing(spot_price: float, 
                                strike_price: float, 
                                time_to_maturity: float, 
                                risk_free_rate: float, 
                                volatility: float, 
                                option_type: str = "call") -> Dict[str, Any]:
    """
    Модель ціноутворення опціонів Блека-Шоулза.
    
    Параметри:
        spot_price: Поточна ціна активу
        strike_price: Ціна виконання
        time_to_maturity: Час до погашення (в роках)
        risk_free_rate: Безризикова ставка
        volatility: Волатильність активу
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
    
    # Перевірка вхідних параметрів
    if time_to_maturity <= 0 or volatility <= 0 or spot_price <= 0 or strike_price <= 0:
        # Опціон погашено або немає волатильності
        if option_type == "call":
            price = max(0, spot_price - strike_price)
        else:  # put
            price = max(0, strike_price - spot_price)
        
        return {
            'option_price': price,
            'delta': 1.0 if (option_type == "call" and spot_price > strike_price) else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # d1 та d2 параметри
    d1 = (math.log(spot_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # Ціна опціона
    if option_type == "call":
        price = (spot_price * norm_cdf(d1) - 
                strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2))
    else:  # put
        price = (strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2) - 
                spot_price * norm_cdf(-d1))
    
    # Греки
    delta = norm_cdf(d1) if option_type == "call" else norm_cdf(d1) - 1
    gamma = norm_pdf(d1) / (spot_price * volatility * math.sqrt(time_to_maturity))
    vega = spot_price * norm_pdf(d1) * math.sqrt(time_to_maturity)
    
    # Тета
    if option_type == "call":
        theta = (-spot_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity)) - 
                risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2))
    else:  # put
        theta = (-spot_price * norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity)) + 
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

def monte_carlo_option_pricing(spot_price: float, 
                              strike_price: float, 
                              time_to_maturity: float, 
                              risk_free_rate: float, 
                              volatility: float, 
                              n_simulations: int = 10000, 
                              option_type: str = "call") -> Dict[str, Any]:
    """
    Оцінка опціонів методом Монте-Карло.
    
    Параметри:
        spot_price: Поточна ціна активу
        strike_price: Ціна виконання
        time_to_maturity: Час до погашення (в роках)
        risk_free_rate: Безризикова ставка
        volatility: Волатильність активу
        n_simulations: Кількість симуляцій
        option_type: Тип опціона ("call" або "put")
    
    Повертає:
        Словник з результатами симуляції
    """
    # Генерація випадкових шляхів ціни
    payoffs = []
    
    for _ in range(n_simulations):
        # Генерація випадкового шляху (геометричне броунівське рух)
        # S_T = S_0 * exp((r - 0.5*σ²)*T + σ*√T*Z)
        # де Z ~ N(0,1)
        z = random.gauss(0, 1)
        terminal_price = spot_price * math.exp(
            (risk_free_rate - 0.5 * volatility**2) * time_to_maturity + 
            volatility * math.sqrt(time_to_maturity) * z
        )
        
        # Обчислення виплати
        if option_type == "call":
            payoff = max(0, terminal_price - strike_price)
        else:  # put
            payoff = max(0, strike_price - terminal_price)
        
        payoffs.append(payoff)
    
    # Середня виплата з дисконтуванням
    average_payoff = sum(payoffs) / len(payoffs)
    option_price = math.exp(-risk_free_rate * time_to_maturity) * average_payoff
    
    # Статистика
    squared_deviations = [(p - average_payoff)**2 for p in payoffs]
    variance = sum(squared_deviations) / (len(payoffs) - 1) if len(payoffs) > 1 else 0
    standard_error = math.sqrt(variance / len(payoffs)) if len(payoffs) > 0 else 0
    
    # Довірчий інтервал (95%)
    confidence_interval = [
        option_price - 1.96 * standard_error,
        option_price + 1.96 * standard_error
    ]
    
    return {
        'option_price': option_price,
        'standard_error': standard_error,
        'confidence_interval': confidence_interval,
        'n_simulations': n_simulations,
        'payoffs': payoffs[:100]  # перші 100 виплат для аналізу
    }

# Ризик-менеджмент
def value_at_risk(returns: List[float], 
                 confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Обчислення Value at Risk (VaR).
    
    Параметри:
        returns: Історичні прибутки/збитки
        confidence_level: Рівень довіри
    
    Повертає:
        Словник з VaR показниками
    """
    if not returns:
        return {
            'var': 0,
            'expected_shortfall': 0,
            'confidence_level': confidence_level
        }
    
    # Сортування прибутків
    sorted_returns = sorted(returns)
    
    # Індекс для VaR
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_index = max(0, min(var_index, len(sorted_returns) - 1))
    
    # VaR
    var = -sorted_returns[var_index]
    
    # Expected Shortfall (усереднене значення втрат, що перевищують VaR)
    if var_index > 0:
        exceedances = [-r for r in sorted_returns[:var_index]]
        expected_shortfall = sum(exceedances) / len(exceedances) if exceedances else 0
    else:
        expected_shortfall = 0
    
    # Статистика
    mean_return = sum(returns) / len(returns) if returns else 0
    std_deviation = math.sqrt(sum((r - mean_return)**2 for r in returns) / (len(returns) - 1)) if len(returns) > 1 else 0
    
    return {
        'var': var,
        'expected_shortfall': expected_shortfall,
        'mean_return': mean_return,
        'std_deviation': std_deviation,
        'confidence_level': confidence_level,
        'worst_loss': -min(returns) if returns else 0,
        'best_gain': max(returns) if returns else 0
    }

def portfolio_risk_metrics(assets_returns: List[List[float]], 
                          weights: List[float]) -> Dict[str, Any]:
    """
    Обчислення ризикових метрик портфеля.
    
    Параметри:
        assets_returns: Прибутки активів (по стовпцях)
        weights: Ваги активів у портфелі
    
    Повертає:
        Словник з ризиковими метриками
    """
    n_assets = len(weights)
    n_periods = len(assets_returns[0]) if assets_returns and assets_returns[0] else 0
    
    if n_assets == 0 or n_periods == 0:
        return {
            'portfolio_volatility': 0,
            'sharpe_ratio': 0,
            'beta': 0,
            'max_drawdown': 0
        }
    
    # Прибутки портфеля
    portfolio_returns = []
    for t in range(n_periods):
        port_return = sum(weights[i] * safe_get_item(assets_returns[i], t, 0) for i in range(n_assets))
        portfolio_returns.append(port_return)
    
    # Середній прибуток портфеля
    mean_portfolio_return = sum(portfolio_returns) / len(portfolio_returns) if portfolio_returns else 0
    
    # Волатильність портфеля
    if len(portfolio_returns) > 1:
        variance = sum((r - mean_portfolio_return)**2 for r in portfolio_returns) / (len(portfolio_returns) - 1)
        portfolio_volatility = math.sqrt(variance)
    else:
        portfolio_volatility = 0
    
    # Коефіцієнт Шарпа (припускаємо безризикову ставку 0 для спрощення)
    sharpe_ratio = mean_portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Максимальна просадка
    cumulative_returns = [1.0]
    for r in portfolio_returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))
    
    max_drawdown = 0
    peak = cumulative_returns[0]
    for value in cumulative_returns:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Бета (відносно ринку, припускаємо останній актив як ринковий індекс)
    if n_assets > 1 and len(portfolio_returns) > 1:
        market_returns = assets_returns[-1] if len(assets_returns) > 0 else []
        if len(market_returns) >= len(portfolio_returns):
            # Коваріація портфеля та ринку
            market_mean = sum(market_returns[:len(portfolio_returns)]) / len(portfolio_returns)
            covariance = sum((portfolio_returns[i] - mean_portfolio_return) * 
                           (market_returns[i] - market_mean) for i in range(len(portfolio_returns))) / (len(portfolio_returns) - 1)
            
            # Дисперсія ринку
            market_variance = sum((r - market_mean)**2 for r in market_returns[:len(portfolio_returns)]) / (len(portfolio_returns) - 1)
            
            # Бета
            beta = covariance / market_variance if market_variance > 0 else 0
        else:
            beta = 0
    else:
        beta = 0
    
    return {
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'beta': beta,
        'max_drawdown': max_drawdown,
        'mean_return': mean_portfolio_return,
        'cumulative_returns': cumulative_returns,
        'portfolio_returns': portfolio_returns
    }

# Технічний аналіз
def moving_averages(prices: List[float], 
                   short_window: int = 20, 
                   long_window: int = 50) -> Dict[str, Any]:
    """
    Обчислення рухомих середніх.
    
    Параметри:
        prices: Ціни активу
        short_window: Короткий період
        long_window: Довгий період
    
    Повертає:
        Словник з рухомими середніми
    """
    if not prices or len(prices) < max(short_window, long_window):
        return {
            'short_ma': [],
            'long_ma': [],
            'signals': []
        }
    
    # Коротке рухоме середнє
    short_ma = []
    for i in range(len(prices)):
        if i >= short_window - 1:
            window = prices[i - short_window + 1:i + 1]
            ma = sum(window) / len(window)
            short_ma.append(ma)
        else:
            short_ma.append(None)
    
    # Довге рухоме середнє
    long_ma = []
    for i in range(len(prices)):
        if i >= long_window - 1:
            window = prices[i - long_window + 1:i + 1]
            ma = sum(window) / len(window)
            long_ma.append(ma)
        else:
            long_ma.append(None)
    
    # Сигнали (перетини)
    signals = []
    for i in range(1, len(prices)):
        # Перевірка наявності значень
        if (i < len(short_ma) and i < len(long_ma) and 
            short_ma[i] is not None and long_ma[i] is not None and
            short_ma[i-1] is not None and long_ma[i-1] is not None):
            
            # Попередній та поточний розрив
            prev_diff = short_ma[i-1] - long_ma[i-1]
            curr_diff = short_ma[i] - long_ma[i]
            
            # Сигнал на покупку (коротке перетинає довге знизу вгору)
            if prev_diff < 0 and curr_diff > 0:
                signals.append({'index': i, 'type': 'buy', 'price': prices[i]})
            # Сигнал на продаж (коротке перетинає довге згори вниз)
            elif prev_diff > 0 and curr_diff < 0:
                signals.append({'index': i, 'type': 'sell', 'price': prices[i]})
            else:
                signals.append({'index': i, 'type': 'hold', 'price': prices[i]})
        else:
            signals.append({'index': i, 'type': 'hold', 'price': prices[i] if i < len(prices) else 0})
    
    return {
        'short_ma': short_ma,
        'long_ma': long_ma,
        'signals': signals,
        'current_short': short_ma[-1] if short_ma else None,
        'current_long': long_ma[-1] if long_ma else None
    }

def relative_strength_index(prices: List[float], 
                          period: int = 14) -> Dict[str, Any]:
    """
    Обчислення індексу відносної сили (RSI).
    
    Параметри:
        prices: Ціни активу
        period: Період розрахунку
    
    Повертає:
        Словник з RSI значеннями
    """
    if not prices or len(prices) < period + 1:
        return {
            'rsi_values': [],
            'overbought': False,
            'oversold': False
        }
    
    # Обчислення змін цін
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # RSI значення
    rsi_values = []
    
    for i in range(period, len(changes) + 1):
        # Відбір змін за період
        period_changes = changes[i - period:i]
        
        # Середні зростання та спади
        gains = [max(0, change) for change in period_changes]
        losses = [max(0, -change) for change in period_changes]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # RSI
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    # Додавання None для початкових значень
    rsi_values = [None] * period + rsi_values
    
    # Перевірка на перекупленість/перепроданість
    current_rsi = rsi_values[-1] if rsi_values else None
    overbought = current_rsi is not None and current_rsi > 70
    oversold = current_rsi is not None and current_rsi < 30
    
    return {
        'rsi_values': rsi_values,
        'current_rsi': current_rsi,
        'overbought': overbought,
        'oversold': oversold,
        'n_periods': len(rsi_values)
    }

# Фінансове моделювання
def financial_projections(initial_revenue: float, 
                        growth_rate: float, 
                        operating_margin: float, 
                        years: int = 5) -> Dict[str, Any]:
    """
    Фінансові проекції компанії.
    
    Параметри:
        initial_revenue: Початковий дохід
        growth_rate: Річний темп зростання
        operating_margin: Операційна маржа
        years: Горизонт прогнозування
    
    Повертає:
        Словник з фінансовими проекціями
    """
    # Проекції доходів
    revenues = []
    ebit = []  # Прибуток до сплати відсотків та податків
    
    current_revenue = initial_revenue
    
    for year in range(years):
        revenues.append(current_revenue)
        ebit.append(current_revenue * operating_margin)
        current_revenue *= (1 + growth_rate)
    
    # Внутрішня норма прибутковості (спрощено)
    # Припускаємо початкові інвестиції як 2x початковий дохід
    initial_investment = initial_revenue * 2
    
    # Грошові потоки (спрощено)
    cash_flows = [-initial_investment] + ebit
    
    # IRR (спрощене обчислення)
    def npv(rate):
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))
    
    # Пошук IRR
    irr = 0
    try:
        if SCIPY_AVAILABLE and optimize is not None:
            result = optimize.root_scalar(npv, bracket=[-0.99, 10], method='brentq')
            irr = result.root if result.converged else 0
        else:
            # Простий пошук методом бісекції
            low, high = -0.99, 10
            for _ in range(100):
                mid = (low + high) / 2
                if abs(npv(mid)) < 1e-6:
                    irr = mid
                    break
                elif npv(low) * npv(mid) < 0:
                    high = mid
                else:
                    low = mid
            else:
                irr = (low + high) / 2
    except:
        irr = 0
    
    # Чиста приведена вартість (NPV) при ставці 10%
    npv_10 = npv(0.10)
    
    return {
        'revenues': revenues,
        'ebit': ebit,
        'operating_margin': operating_margin,
        'growth_rate': growth_rate,
        'irr': irr,
        'npv_10_percent': npv_10,
        'total_revenue': sum(revenues),
        'total_ebit': sum(ebit),
        'years': years
    }

def portfolio_optimization(expected_returns: List[float], 
                          covariance_matrix: List[List[float]], 
                          risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Оптимізація портфеля інвестицій.
    
    Параметри:
        expected_returns: Очікувані прибутки активів
        covariance_matrix: Коваріаційна матриця
        risk_free_rate: Безризикова ставка
    
    Повертає:
        Словник з оптимальним портфелем
    """
    n_assets = len(expected_returns)
    
    if n_assets == 0 or len(covariance_matrix) != n_assets:
        return {
            'optimal_weights': [],
            'expected_return': 0,
            'risk': 0,
            'sharpe_ratio': 0
        }
    
    # Перевірка коваріаційної матриці
    for row in covariance_matrix:
        if len(row) != n_assets:
            return {
                'optimal_weights': [],
                'expected_return': 0,
                'risk': 0,
                'sharpe_ratio': 0
            }
    
    # Спрощена оптимізація (рівноважений портфель)
    weights = [1/n_assets] * n_assets
    
    # Очікуваний прибуток портфеля
    expected_return = sum(weights[i] * expected_returns[i] for i in range(n_assets))
    
    # Ризик портфеля (стандартне відхилення)
    portfolio_variance = 0
    for i in range(n_assets):
        for j in range(n_assets):
            portfolio_variance += weights[i] * weights[j] * covariance_matrix[i][j]
    
    risk = math.sqrt(portfolio_variance) if portfolio_variance >= 0 else 0
    
    # Коефіцієнт Шарпа
    sharpe_ratio = (expected_return - risk_free_rate) / risk if risk > 0 else 0
    
    return {
        'optimal_weights': weights,
        'expected_return': expected_return,
        'risk': risk,
        'sharpe_ratio': sharpe_ratio,
        'portfolio_variance': portfolio_variance
    }

# Статистика та економетрія
def time_series_analysis(returns: List[float]) -> Dict[str, Any]:
    """
    Аналіз часових рядів фінансових прибутків.
    
    Параметри:
        returns: Часовий ряд прибутків
    
    Повертає:
        Словник з результатами аналізу
    """
    if not returns or len(returns) < 2:
        return {
            'mean': 0,
            'std_dev': 0,
            'skewness': 0,
            'kurtosis': 0,
            'sharpe_ratio': 0
        }
    
    n = len(returns)
    
    # Середнє значення
    mean = sum(returns) / n
    
    # Стандартне відхилення
    variance = sum((r - mean)**2 for r in returns) / (n - 1) if n > 1 else 0
    std_dev = math.sqrt(variance)
    
    # Асиметрія
    if std_dev > 0:
        skewness = sum(((r - mean) / std_dev)**3 for r in returns) / n
    else:
        skewness = 0
    
    # Ексцес
    if std_dev > 0:
        kurtosis = sum(((r - mean) / std_dev)**4 for r in returns) / n - 3
    else:
        kurtosis = 0
    
    # Коефіцієнт Шарпа (припускаємо безризикову ставку 0)
    sharpe_ratio = mean / std_dev if std_dev > 0 else 0
    
    # Автокореляція (лаг 1)
    if n > 1:
        # Коваріація
        covariance = sum((returns[i] - mean) * (returns[i-1] - mean) for i in range(1, n)) / (n - 1)
        # Автокореляція
        autocorrelation = covariance / variance if variance > 0 else 0
    else:
        autocorrelation = 0
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'sharpe_ratio': sharpe_ratio,
        'autocorrelation': autocorrelation,
        'n_observations': n
    }

def garch_volatility_model(returns: List[float], 
                          alpha: float = 0.1, 
                          beta: float = 0.8) -> Dict[str, Any]:
    """
    Спрощена модель GARCH(1,1) для прогнозування волатильності.
    
    Параметри:
        returns: Часовий ряд прибутків
        alpha: Параметр ARCH
        beta: Параметр GARCH
    
    Повертає:
        Словник з прогнозами волатильності
    """
    if not returns or len(returns) < 2:
        return {
            'conditional_volatility': [],
            'long_term_volatility': 0,
            'persistence': 0
        }
    
    n = len(returns)
    
    # Середній прибуток
    mean_return = sum(returns) / n
    
    # Оцінка довгострокової волатильності
    long_term_variance = sum((r - mean_return)**2 for r in returns) / n
    long_term_volatility = math.sqrt(long_term_variance)
    
    # Постійність
    persistence = alpha + beta
    
    # Умовна волатильність (модель GARCH(1,1))
    conditional_variance = [long_term_variance]  # початкове значення
    conditional_volatility = [long_term_volatility]
    
    omega = long_term_variance * (1 - persistence)  # довгостроковий параметр
    
    for i in range(1, n):
        # Модель: σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
        # де ε_{t-1} = r_{t-1} - μ
        error = returns[i-1] - mean_return
        new_variance = omega + alpha * error**2 + beta * conditional_variance[-1]
        conditional_variance.append(new_variance)
        conditional_volatility.append(math.sqrt(new_variance))
    
    # Прогноз на майбутнє
    forecast_variance = omega + persistence * conditional_variance[-1]
    forecast_volatility = math.sqrt(forecast_variance)
    
    return {
        'conditional_volatility': conditional_volatility,
        'long_term_volatility': long_term_volatility,
        'persistence': persistence,
        'forecast_volatility': forecast_volatility,
        'omega': omega,
        'alpha': alpha,
        'beta': beta
    }

if __name__ == "__main__":
    # Тестування функцій модуля
    print("Тестування модуля обчислювальної фінансової математики PyNexus")
    
    # Тест ціноутворення опціонів Блека-Шоулза
    bs_result = black_scholes_option_pricing(
        spot_price=100,
        strike_price=105,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option_type="call"
    )
    print(f"Ціна опціона Блека-Шоулза: {bs_result['option_price']:.2f}")
    print(f"Дельта опціона: {bs_result['delta']:.3f}")
    
    # Тест VaR
    sample_returns = [random.gauss(0.001, 0.02) for _ in range(1000)]  # 1000 днів прибутків
    var_result = value_at_risk(sample_returns, confidence_level=0.95)
    print(f"Value at Risk (95%): {var_result['var']:.4f}")
    print(f"Expected Shortfall: {var_result['expected_shortfall']:.4f}")
    
    # Тест RSI
    sample_prices = [100 + i*0.5 + random.gauss(0, 1) for i in range(100)]
    rsi_result = relative_strength_index(sample_prices, period=14)
    print(f"Поточний RSI: {rsi_result['current_rsi']:.2f}")
    print(f"Перекупленість: {rsi_result['overbought']}")
    print(f"Перепроданість: {rsi_result['oversold']}")
    
    # Тест фінансових проекцій
    projection_result = financial_projections(
        initial_revenue=1000000,  # 1 мільйон
        growth_rate=0.10,  # 10% річний ріст
        operating_margin=0.15,  # 15% маржа
        years=5
    )
    print(f"Загальний дохід за 5 років: {projection_result['total_revenue']:,.0f}")
    print(f"IRR: {projection_result['irr']:.2%}")
    print(f"NPV при 10%: {projection_result['npv_10_percent']:,.0f}")
    
    print("Тестування завершено!")