"""
analysis module for PyNexus.

цей модуль надає функції для аналізу даних.
автор: Андрій Будильников
"""

import pandas as pd
import numpy as np
from scipy import stats


def describe(data, extended=False):
    """
    generate descriptive statistics for a dataset.
    
    args:
        data: pandas.dataframe or similar data structure
        extended: bool, if true, provides extended statistics
        
    returns:
        pandas.dataframe: summary statistics
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> desc = describe(df)
        >>> print(desc)
    """
    if extended:
        # provide extended statistics
        desc = data.describe()
        # add additional statistics
        extended_stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            extended_stats[col] = {
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis(),
                'median': data[col].median()
            }
        return desc, extended_stats
    else:
        return data.describe()


def filter_data(data, expr):
    """
    filter data based on an expression.
    
    args:
        data: pandas.dataframe to filter
        expr: string expression to filter by
        
    returns:
        pandas.dataframe: filtered data
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> filtered = filter_data(df, 'a > 1')
        >>> print(filtered)
           a  b
        1  2  5
        2  3  6
    """
    return data.query(expr)


def groupby_stats(data, group_col, agg_col, agg_func='mean'):
    """
    calculate statistics grouped by a column.
    
    args:
        data: pandas.dataframe
        group_col: column to group by
        agg_col: column to aggregate
        agg_func: aggregation function (mean, sum, count, etc.)
        
    returns:
        pandas.series: grouped statistics
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'a', 'b'], 'value': [1, 2, 3, 4]})
        >>> stats = groupby_stats(df, 'category', 'value', 'mean')
        >>> print(stats)
    """
    return data.groupby(group_col)[agg_col].agg(agg_func)


def correlation_matrix(data, method='pearson'):
    """
    compute correlation matrix for numeric columns.
    
    args:
        data: pandas.dataframe
        method: correlation method ('pearson', 'spearman', 'kendall')
        
    returns:
        pandas.dataframe: correlation matrix
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> corr = correlation_matrix(df)
        >>> print(corr)
    """
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.corr(method=method)


def covariance_matrix(data, min_periods=None):
    """
    обчислити коваріаційну матрицю для числових стовпців.
    
    args:
        data: pandas.dataframe
        min_periods: мінімальна кількість спостережень (за замовчуванням none)
        
    returns:
        pandas.dataframe: коваріаційна матриця
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> cov = covariance_matrix(df)
        >>> print(cov)
    """
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.cov(min_periods=min_periods)


def rolling_mean(data, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
    """
    обчислити ковзне середнє значення.
    
    args:
        data: pandas.dataframe або series
        window: розмір вікна
        min_periods: мінімальна кількість спостережень у вікні
        center: чи центрувати вікно
        win_type: тип вікна
        on: мітка часу для віконного індексування
        axis: вісь для обчислення
        closed: які кінці вікна включаються
        
    returns:
        pandas.dataframe або series: ковзне середнє
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> rolling = rolling_mean(df['a'], window=3)
        >>> print(rolling)
    """
    return data.rolling(window=window, min_periods=min_periods, center=center, 
                       win_type=win_type, on=on, axis=axis, closed=closed).mean()


def rolling_std(data, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
    """
    обчислити ковзне стандартне відхилення.
    
    args:
        data: pandas.dataframe або series
        window: розмір вікна
        min_periods: мінімальна кількість спостережень у вікні
        center: чи центрувати вікно
        win_type: тип вікна
        on: мітка часу для віконного індексування
        axis: вісь для обчислення
        closed: які кінці вікна включаються
        
    returns:
        pandas.dataframe або series: ковзне стандартне відхилення
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> rolling = rolling_std(df['a'], window=3)
        >>> print(rolling)
    """
    return data.rolling(window=window, min_periods=min_periods, center=center, 
                       win_type=win_type, on=on, axis=axis, closed=closed).std()


def expanding_mean(data, min_periods=1, center=None, axis=0):
    """
    обчислити розширювальне середнє значення.
    
    args:
        data: pandas.dataframe або series
        min_periods: мінімальна кількість спостережень
        center: чи центрувати вікно
        axis: вісь для обчислення
        
    returns:
        pandas.dataframe або series: розширювальне середнє
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> expanding = expanding_mean(df['a'])
        >>> print(expanding)
    """
    return data.expanding(min_periods=min_periods, center=center, axis=axis).mean()


def expanding_std(data, min_periods=1, center=None, axis=0):
    """
    обчислити розширювальне стандартне відхилення.
    
    args:
        data: pandas.dataframe або series
        min_periods: мінімальна кількість спостережень
        center: чи центрувати вікно
        axis: вісь для обчислення
        
    returns:
        pandas.dataframe або series: розширювальне стандартне відхилення
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> expanding = expanding_std(df['a'])
        >>> print(expanding)
    """
    return data.expanding(min_periods=min_periods, center=center, axis=axis).std()


def ewm_mean(data, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0):
    """
    обчислити експоненційно зважене середнє.
    
    args:
        data: pandas.dataframe або series
        com: центр маса (com >= 0)
        span: період (span >= 1)
        halflife: період напіврозпаду
        alpha: параметр згладжування (0 < alpha <= 1)
        min_periods: мінімальна кількість спостережень
        adjust: метод обчислення
        ignore_na: чи ігнорувати na значення
        axis: вісь для обчислення
        
    returns:
        pandas.dataframe або series: експоненційно зважене середнє
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> ewm = ewm_mean(df['a'], span=3)
        >>> print(ewm)
    """
    return data.ewm(com=com, span=span, halflife=halflife, alpha=alpha, 
                   min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis).mean()


def quantile(data, q, axis=0, numeric_only=True, interpolation='linear'):
    """
    обчислити квантилі даних.
    
    args:
        data: pandas.dataframe або series
        q: квантилі (0 <= q <= 1)
        axis: вісь для обчислення
        numeric_only: чи враховувати тільки числові стовпці
        interpolation: метод інтерполяції
        
    returns:
        pandas.series або dataframe: квантилі
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> quantiles = quantile(df, [0.25, 0.5, 0.75])
        >>> print(quantiles)
    """
    return data.quantile(q=q, axis=axis, numeric_only=numeric_only, interpolation=interpolation)


def rank_data(data, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False):
    """
    проранжувати дані.
    
    args:
        data: pandas.dataframe або series
        axis: вісь для ранжування
        method: метод обробки зв'язків ('average', 'min', 'max', 'first', 'dense')
        numeric_only: чи враховувати тільки числові стовпці
        na_option: як обробляти na значення ('keep', 'top', 'bottom')
        ascending: чи сортувати за зростанням
        pct: чи повертати відсоткові ранги
        
    returns:
        pandas.series або dataframe: ранги
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 2, 3]})
        >>> ranks = rank_data(df['a'])
        >>> print(ranks)
    """
    return data.rank(axis=axis, method=method, numeric_only=numeric_only, 
                    na_option=na_option, ascending=ascending, pct=pct)


def value_counts(data, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
    """
    підрахувати унікальні значення.
    
    args:
        data: pandas.series
        normalize: чи нормалізувати до відносних частот
        sort: чи сортувати результати
        ascending: чи сортувати за зростанням
        bins: кількість інтервалів для числових даних
        dropna: чи видаляти na значення
        
    returns:
        pandas.series: підрахунок унікальних значень
        
    example:
        >>> s = pd.series([1, 2, 2, 3, 3, 3])
        >>> counts = value_counts(s)
        >>> print(counts)
    """
    return data.value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)


def crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='all', dropna=True, normalize=False):
    """
    обчислити таблицю спряженості.
    
    args:
        index: значення для рядків
        columns: значення для стовпців
        values: значення для агрегації
        rownames: імена рядків
        colnames: імена стовпців
        aggfunc: функція агрегації
        margins: чи додавати підсумкові рядки/стовпці
        margins_name: ім'я для підсумкових рядків/стовпців
        dropna: чи видаляти рядки/стовпці з na
        normalize: чи нормалізувати результати
        
    returns:
        pandas.dataframe: таблиця спряженості
        
    example:
        >>> index = pd.series(['a', 'b', 'a', 'b'])
        >>> columns = pd.series(['x', 'y', 'x', 'y'])
        >>> crosstab_result = crosstab(index, columns)
        >>> print(crosstab_result)
    """
    return pd.crosstab(index, columns, values=values, rownames=rownames, colnames=colnames, 
                      aggfunc=aggfunc, margins=margins, margins_name=margins_name, 
                      dropna=dropna, normalize=normalize)


def pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='all', observed=False, sort=True):
    """
    створити зведenu таблицю.
    
    args:
        data: pandas.dataframe
        values: стовпці для агрегації
        index: стовпці для рядків
        columns: стовпці для стовпців
        aggfunc: функція агрегації
        fill_value: значення для заповнення відсутніх комірок
        margins: чи додавати підсумкові рядки/стовпці
        dropna: чи видаляти стовпці з na
        margins_name: ім'я для підсумкових рядків/стовпців
        observed: чи враховувати тільки спостережувані значення
        sort: чи сортувати результати
        
    returns:
        pandas.dataframe: зведена таблиця
        
    example:
        >>> df = pd.dataframe({'a': ['x', 'y', 'x', 'y'], 'b': [1, 2, 3, 4], 'c': [10, 20, 30, 40]})
        >>> pivot = pivot_table(df, values='b', index='a', columns='c', aggfunc='sum')
        >>> print(pivot)
    """
    return pd.pivot_table(data, values=values, index=index, columns=columns, 
                         aggfunc=aggfunc, fill_value=fill_value, margins=margins, 
                         dropna=dropna, margins_name=margins_name, observed=observed, sort=sort)


def melt(data, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True):
    """
    перетворити таблицю з широкої форми у довгу.
    
    args:
        data: pandas.dataframe
        id_vars: стовпці для збереження
        value_vars: стовпці для перетворення (за замовчуванням всі інші)
        var_name: ім'я стовпця для назв змінних
        value_name: ім'я стовпця для значень
        col_level: рівень колонок для використання
        ignore_index: чи ігнорувати оригінальний індекс
        
    returns:
        pandas.dataframe: перетворена таблиця
        
    example:
        >>> df = pd.dataframe({'id': [1, 2], 'a': [3, 4], 'b': [5, 6]})
        >>> melted = melt(df, id_vars=['id'], value_vars=['a', 'b'])
        >>> print(melted)
    """
    return pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name=var_name, 
                  value_name=value_name, col_level=col_level, ignore_index=ignore_index)


def merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
    """
    об'єднати два dataframe.
    
    args:
        left: лівий dataframe
        right: правий dataframe
        how: тип об'єднання ('left', 'right', 'outer', 'inner', 'cross')
        on: стовпці для об'єднання
        left_on: стовпці з лівого dataframe
        right_on: стовпці з правого dataframe
        left_index: чи використовувати індекс лівого dataframe
        right_index: чи використовувати індекс правого dataframe
        sort: чи сортувати результат
        suffixes: суфікси для однакових назв стовпців
        copy: чи копіювати дані
        indicator: чи додавати індикатор джерела
        validate: перевірка типу об'єднання
        
    returns:
        pandas.dataframe: об'єднаний dataframe
        
    example:
        >>> df1 = pd.dataframe({'key': ['a', 'b'], 'value1': [1, 2]})
        >>> df2 = pd.dataframe({'key': ['a', 'b'], 'value2': [3, 4]})
        >>> merged = merge(df1, df2, on='key')
        >>> print(merged)
    """
    return pd.merge(left, right, how=how, on=on, left_on=left_on, right_on=right_on, 
                   left_index=left_index, right_index=right_index, sort=sort, 
                   suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)


def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True):
    """
    об'єднати об'єкти вздовж вказаної осі.
    
    args:
        objs: послідовність об'єктів для об'єднання
        axis: вісь для об'єднання (0 - рядки, 1 - стовпці)
        join: тип об'єднання ('inner', 'outer')
        ignore_index: чи ігнорувати оригінальні індекси
        keys: ключі для створення ієрархічного індексу
        levels: рівні для ієрархічного індексу
        names: імена рівнів ієрархічного індексу
        verify_integrity: чи перевіряти унікальність індексу
        sort: чи сортувати стовпці
        copy: чи копіювати дані
        
    returns:
        pandas.dataframe або series: об'єднаний об'єкт
        
    example:
        >>> df1 = pd.dataframe({'a': [1, 2]})
        >>> df2 = pd.dataframe({'a': [3, 4]})
        >>> concatenated = concat([df1, df2])
        >>> print(concatenated)
    """
    return pd.concat(objs, axis=axis, join=join, ignore_index=ignore_index, keys=keys, 
                    levels=levels, names=names, verify_integrity=verify_integrity, 
                    sort=sort, copy=copy)


def drop_duplicates(data, subset=None, keep='first', inplace=False, ignore_index=False):
    """
    видалити дублікати з даних.
    
    args:
        data: pandas.dataframe або series
        subset: стовпці для перевірки дублікатів
        keep: які дублікати зберігати ('first', 'last', false)
        inplace: чи змінювати оригінал
        ignore_index: чи ігнорувати оригінальний індекс
        
    returns:
        pandas.dataframe або series: дані без дублікатів
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 2, 3], 'b': [4, 5, 5, 6]})
        >>> deduplicated = drop_duplicates(df)
        >>> print(deduplicated)
    """
    return data.drop_duplicates(subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index)


def fillna(data, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    """
    заповнити відсутні значення.
    
    args:
        data: pandas.dataframe або series
        value: значення для заповнення
        method: метод заповнення ('backfill', 'bfill', 'pad', 'ffill', none)
        axis: вісь для заповнення
        inplace: чи змінювати оригінал
        limit: максимальна кількість заповнень
        downcast: тип даних для зменшення розміру
        
    returns:
        pandas.dataframe або series: дані з заповненими значеннями
        
    example:
        >>> df = pd.dataframe({'a': [1, np.nan, 3]})
        >>> filled = fillna(df, value=0)
        >>> print(filled)
    """
    return data.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)


def dropna(data, axis=0, how='any', thresh=None, subset=None, inplace=False):
    """
    видалити рядки/стовпці з відсутніми значеннями.
    
    args:
        data: pandas.dataframe або series
        axis: вісь для видалення (0 - рядки, 1 - стовпці)
        how: як видаляти ('any', 'all')
        thresh: мінімальна кількість не-na значень
        subset: стовпці для перевірки
        inplace: чи змінювати оригінал
        
    returns:
        pandas.dataframe або series: дані без na значень
        
    example:
        >>> df = pd.dataframe({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        >>> cleaned = dropna(df)
        >>> print(cleaned)
    """
    return data.dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace)


def replace(data, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
    """
    замінити значення в даних.
    
    args:
        data: pandas.dataframe або series
        to_replace: значення для заміни
        value: нове значення
        inplace: чи змінювати оригінал
        limit: максимальна кількість замін
        regex: чи використовувати регулярні вирази
        method: метод заміни
        
    returns:
        pandas.dataframe або series: дані з заміненими значеннями
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3]})
        >>> replaced = replace(df, to_replace=2, value=99)
        >>> print(replaced)
    """
    return data.replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit, regex=regex, method=method)


def apply(data, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
    """
    застосувати функцію до даних.
    
    args:
        data: pandas.dataframe або series
        func: функція для застосування
        axis: вісь для застосування (0 - індекс, 1 - стовпці)
        raw: чи передавати raw масиви
        result_type: тип результату
        args: додаткові аргументи для функції
        **kwargs: додаткові ключові аргументи
        
    returns:
        pandas.series, dataframe або scalar: результат застосування функції
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3]})
        >>> result = apply(df, lambda x: x * 2)
        >>> print(result)
    """
    return data.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs)


def applymap(data, func, na_action=None, **kwargs):
    """
    застосувати функцію до кожного елемента.
    
    args:
        data: pandas.dataframe
        func: функція для застосування
        na_action: як обробляти na значення (none або 'ignore')
        **kwargs: додаткові ключові аргументи
        
    returns:
        pandas.dataframe: результат застосування функції
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3]})
        >>> result = applymap(df, lambda x: x * 2)
        >>> print(result)
    """
    return data.applymap(func, na_action=na_action, **kwargs)


def map_series(data, arg, na_action=None):
    """
    відобразити значення серії за допомогою словника або функції.
    
    args:
        data: pandas.series
        arg: словник, функція або series для відображення
        na_action: як обробляти na значення (none або 'ignore')
        
    returns:
        pandas.series: відображена серія
        
    example:
        >>> s = pd.series(['cat', 'dog', 'cat'])
        >>> mapped = map_series(s, {'cat': 'feline', 'dog': 'canine'})
        >>> print(mapped)
    """
    return data.map(arg, na_action=na_action)


def transform(data, func, axis=0, *args, **kwargs):
    """
    трансформувати дані за допомогою функції.
    
    args:
        data: pandas.dataframe або groupby об'єкт
        func: функція для трансформації
        axis: вісь для трансформації
        *args: додаткові аргументи
        **kwargs: додаткові ключові аргументи
        
    returns:
        pandas.dataframe або series: трансформовані дані
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3]})
        >>> transformed = transform(df, lambda x: x - x.mean())
        >>> print(transformed)
    """
    return data.transform(func, axis=axis, *args, **kwargs)


def aggregate(data, func, axis=0, *args, **kwargs):
    """
    агрегувати дані за допомогою функції.
    
    args:
        data: pandas.dataframe або groupby об'єкт
        func: функція для агрегації
        axis: вісь для агрегації
        *args: додаткові аргументи
        **kwargs: додаткові ключові аргументи
        
    returns:
        pandas.dataframe, series або scalar: агреговані дані
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3]})
        >>> aggregated = aggregate(df, 'mean')
        >>> print(aggregated)
    """
    return data.agg(func, axis=axis, *args, **kwargs)


def resample(data, rule, axis=0, closed=None, label=None, convention='start', kind=None, loffset=None, base=None, on=None, level=None, origin='start_day', offset=None):
    """
    змінити частоту часових рядів.
    
    args:
        data: pandas.dataframe або series з часовим індексом
        rule: правило зміни частоти
        axis: вісь для зміни частоти
        closed: який бік інтервалу закритий
        label: який бік інтервалу використовувати для міток
        convention: конвенція для періодів
        kind: тип перетворення
        loffset: зсув міток
        base: зсув початку інтервалу
        on: стовпець для використання як індекс
        level: рівень ієрархічного індексу
        origin: початок інтервалу
        offset: зсув інтервалу
        
    returns:
        pandas.resampler: об'єкт для подальшої обробки
        
    example:
        >>> dates = pd.date_range('2023-01-01', periods=100, freq='d')
        >>> df = pd.dataframe({'value': range(100)}, index=dates)
        >>> resampled = resample(df, 'm').mean()
        >>> print(resampled)
    """
    return data.resample(rule, axis=axis, closed=closed, label=label, convention=convention, 
                        kind=kind, loffset=loffset, base=base, on=on, level=level, 
                        origin=origin, offset=offset)


def shift(data, periods=1, freq=None, axis=0, fill_value=None):
    """
    зсунути індекс на передану частоту або кількість періодів.
    
    args:
        data: pandas.dataframe або series
        periods: кількість періодів для зсуву
        freq: частота для зсуву
        axis: вісь для зсуву
        fill_value: значення для заповнення нових позицій
        
    returns:
        pandas.dataframe або series: зсунуті дані
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4]})
        >>> shifted = shift(df, periods=1)
        >>> print(shifted)
    """
    return data.shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)


def diff(data, periods=1, axis=0):
    """
    обчислити різницю елементів.
    
    args:
        data: pandas.dataframe або series
        periods: кількість періодів для різниці
        axis: вісь для обчислення різниці
        
    returns:
        pandas.dataframe або series: різниця елементів
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 4, 7]})
        >>> difference = diff(df['a'])
        >>> print(difference)
    """
    return data.diff(periods=periods, axis=axis)


def pct_change(data, periods=1, fill_method='pad', limit=None, freq=None, **kwargs):
    """
    обчислити відсоткову зміну.
    
    args:
        data: pandas.dataframe або series
        periods: кількість періодів для обчислення зміни
        fill_method: метод заповнення na значень
        limit: максимальна кількість заповнень
        freq: частота для зміни індексу
        **kwargs: додаткові аргументи
        
    returns:
        pandas.dataframe або series: відсоткова зміна
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 4, 8]})
        >>> pct = pct_change(df['a'])
        >>> print(pct)
    """
    return data.pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq, **kwargs)


def rolling_window(data, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
    """
    створити ковзне вікно для обчислень.
    
    args:
        data: pandas.dataframe або series
        window: розмір вікна або змінна для визначення вікна
        min_periods: мінімальна кількість спостережень у вікні
        center: чи центрувати вікно
        win_type: тип вікна
        on: мітка часу для віконного індексування
        axis: вісь для обчислення
        closed: які кінці вікна включаються
        
    returns:
        pandas.rolling: об'єкт ковзного вікна
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> rolling_obj = rolling_window(df['a'], window=3)
        >>> result = rolling_obj.mean()
        >>> print(result)
    """
    return data.rolling(window=window, min_periods=min_periods, center=center, 
                       win_type=win_type, on=on, axis=axis, closed=closed)


def expanding_window(data, min_periods=1, center=None, axis=0):
    """
    створити розширювальне вікно для обчислень.
    
    args:
        data: pandas.dataframe або series
        min_periods: мінімальна кількість спостережень
        center: чи центрувати вікно
        axis: вісь для обчислення
        
    returns:
        pandas.expanding: об'єкт розширювального вікна
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> expanding_obj = expanding_window(df['a'])
        >>> result = expanding_obj.mean()
        >>> print(result)
    """
    return data.expanding(min_periods=min_periods, center=center, axis=axis)


def ewm_window(data, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0):
    """
    створити експоненційно зважене вікно для обчислень.
    
    args:
        data: pandas.dataframe або series
        com: центр маса (com >= 0)
        span: період (span >= 1)
        halflife: період напіврозпаду
        alpha: параметр згладжування (0 < alpha <= 1)
        min_periods: мінімальна кількість спостережень
        adjust: метод обчислення
        ignore_na: чи ігнорувати na значення
        axis: вісь для обчислення
        
    returns:
        pandas.ewm: об'єкт експоненційно зваженого вікна
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3, 4, 5]})
        >>> ewm_obj = ewm_window(df['a'], span=3)
        >>> result = ewm_obj.mean()
        >>> print(result)
    """
    return data.ewm(com=com, span=span, halflife=halflife, alpha=alpha, 
                   min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis)


def ttest_1samp(data, popmean, axis=0, nan_policy='propagate', alternative='two-sided'):
    """
    одновибірковий t-тест.
    
    args:
        data: масив даних
        popmean: гіпотетичне середнє значення генеральної сукупності
        axis: вісь для обчислення
        nan_policy: як обробляти na значення ('propagate', 'raise', 'omit')
        alternative: альтернативна гіпотеза ('two-sided', 'less', 'greater')
        
    returns:
        scipy.stats._stats_py.Ttest_1sampResult: результати t-тесту
        
    example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> result = ttest_1samp(data, 3)
        >>> print(result)
    """
    return stats.ttest_1samp(data, popmean, axis=axis, nan_policy=nan_policy, alternative=alternative)


def ttest_ind(data1, data2, axis=0, equal_var=True, nan_policy='propagate', alternative='two-sided'):
    """
    двовибірковий t-тест для незалежних вибірок.
    
    args:
        data1: перша вибірка
        data2: друга вибірка
        axis: вісь для обчислення
        equal_var: чи приймати рівність дисперсій
        nan_policy: як обробляти na значення ('propagate', 'raise', 'omit')
        alternative: альтернативна гіпотеза ('two-sided', 'less', 'greater')
        
    returns:
        scipy.stats._stats_py.Ttest_indResult: результати t-тесту
        
    example:
        >>> data1 = np.array([1, 2, 3, 4, 5])
        >>> data2 = np.array([2, 3, 4, 5, 6])
        >>> result = ttest_ind(data1, data2)
        >>> print(result)
    """
    return stats.ttest_ind(data1, data2, axis=axis, equal_var=equal_var, nan_policy=nan_policy, alternative=alternative)


def ttest_rel(data1, data2, axis=0, nan_policy='propagate', alternative='two-sided'):
    """
    парний t-тест.
    
    args:
        data1: перша вибірка
        data2: друга вибірка
        axis: вісь для обчислення
        nan_policy: як обробляти na значення ('propagate', 'raise', 'omit')
        alternative: альтернативна гіпотеза ('two-sided', 'less', 'greater')
        
    returns:
        scipy.stats._stats_py.Ttest_relResult: результати t-тесту
        
    example:
        >>> data1 = np.array([1, 2, 3, 4, 5])
        >>> data2 = np.array([2, 3, 4, 5, 6])
        >>> result = ttest_rel(data1, data2)
        >>> print(result)
    """
    return stats.ttest_rel(data1, data2, axis=axis, nan_policy=nan_policy, alternative=alternative)


def chi2_contingency(observed, correction=True, lambda_=None):
    """
    хі-квадрат тест для таблиць спряженості.
    
    args:
        observed: таблиця спостережень
        correction: чи застосовувати корекцію юейта
        lambda_: параметр для статистики
        
    returns:
        tuple: (chi2, p, dof, expected)
        
    example:
        >>> observed = np.array([[10, 10, 20], [20, 20, 20]])
        >>> chi2, p, dof, expected = chi2_contingency(observed)
        >>> print(chi2, p)
    """
    return stats.chi2_contingency(observed, correction=correction, lambda_=lambda_)


def anova(data, *args, **kwargs):
    """
    аналіз дисперсії (anova).
    
    args:
        data: дані для аналізу
        *args: додаткові аргументи
        **kwargs: додаткові ключові аргументи
        
    returns:
        scipy.stats._stats_py.F_onewayResult: результати anova
        
    example:
        >>> data1 = np.array([1, 2, 3])
        >>> data2 = np.array([4, 5, 6])
        >>> data3 = np.array([7, 8, 9])
        >>> result = anova(data1, data2, data3)
        >>> print(result)
    """
    return stats.f_oneway(*args, **kwargs)


def kruskal(*args, **kwargs):
    """
    критерій крускала-уолліса (непараметричний anova).
    
    args:
        *args: вибірки для порівняння
        **kwargs: додаткові ключові аргументи
        
    returns:
        scipy.stats._stats_py.KruskalResult: результати критерію
        
    example:
        >>> data1 = np.array([1, 2, 3])
        >>> data2 = np.array([4, 5, 6])
        >>> data3 = np.array([7, 8, 9])
        >>> result = kruskal(data1, data2, data3)
        >>> print(result)
    """
    return stats.kruskal(*args, **kwargs)


def mannwhitneyu(x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto'):
    """
    u-критерій манна-уітні.
    
    args:
        x: перша вибірка
        y: друга вибірка
        use_continuity: чи застосовувати корекцію неперервності
        alternative: альтернативна гіпотеза ('two-sided', 'less', 'greater')
        axis: вісь для обчислення
        method: метод обчислення ('auto', 'exact', 'asymptotic')
        
    returns:
        scipy.stats._mannwhitneyu.MannwhitneyuResult: результати критерію
        
    example:
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> result = mannwhitneyu(x, y)
        >>> print(result)
    """
    return stats.mannwhitneyu(x, y, use_continuity=use_continuity, alternative=alternative, axis=axis, method=method)


def wilcoxon(x, y=None, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto'):
    """
    знаковий ранговий критерій уілкоксона.
    
    args:
        x: вибірка або різниці пар
        y: друга вибірка (необов'язково)
        zero_method: метод обробки нульових різниць
        correction: чи застосовувати корекцію неперервності
        alternative: альтернативна гіпотеза ('two-sided', 'less', 'greater')
        mode: метод обчислення ('auto', 'exact', 'approx')
        
    returns:
        scipy.stats._morestats.WilcoxonResult: результати критерію
        
    example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 3, 4, 5, 6])
        >>> result = wilcoxon(x, y)
        >>> print(result)
    """
    return stats.wilcoxon(x, y=y, zero_method=zero_method, correction=correction, alternative=alternative, mode=mode)


def pearsonr(x, y):
    """
    коефіцієнт кореляції пірсона.
    
    args:
        x: перша вибірка
        y: друга вибірка
        
    returns:
        tuple: (correlation, p-value)
        
    example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>> corr, p_value = pearsonr(x, y)
        >>> print(corr, p_value)
    """
    return stats.pearsonr(x, y)


def spearmanr(a, b=None, axis=0, nan_policy='propagate', alternative='two-sided'):
    """
    рангова кореляція спірмена.
    
    args:
        a: масив або двовимірна матриця
        b: другий масив (необов'язково)
        axis: вісь для обчислення
        nan_policy: як обробляти na значення
        alternative: альтернативна гіпотеза
        
    returns:
        scipy.stats._stats_py.SpearmanrResult: результати кореляції
        
    example:
        >>> a = np.array([1, 2, 3, 4, 5])
        >>> b = np.array([2, 4, 6, 8, 10])
        >>> result = spearmanr(a, b)
        >>> print(result)
    """
    return stats.spearmanr(a, b=b, axis=axis, nan_policy=nan_policy, alternative=alternative)


def kendalltau(x, y, initial_lexsort=None, nan_policy='propagate', method='auto'):
    """
    тау-кореляція кендала.
    
    args:
        x: перша вибірка
        y: друга вибірка
        initial_lexsort: чи використовувати лексикографічне сортування
        nan_policy: як обробляти na значення
        method: метод обчислення
        
    returns:
        scipy.stats._stats_py.KendalltauResult: результати кореляції
        
    example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>> result = kendalltau(x, y)
        >>> print(result)
    """
    return stats.kendalltau(x, y, initial_lexsort=initial_lexsort, nan_policy=nan_policy, method=method)


def linregress(x, y=None):
    """
    лінійна регресія методом найменших квадратів.
    
    args:
        x: незалежна змінна
        y: залежна змінна (необов'язково)
        
    returns:
        scipy.stats._stats_py.LinregressResult: результати регресії
        
    example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>> result = linregress(x, y)
        >>> print(result)
    """
    return stats.linregress(x, y=y)


def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    """
    обчислити z-оцінки (стандартизовані оцінки).
    
    args:
        a: масив даних
        axis: вісь для обчислення
        ddof: степені свободи
        nan_policy: як обробляти na значення
        
    returns:
        numpy.ndarray: z-оцінки
        
    example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> z_scores = zscore(data)
        >>> print(z_scores)
    """
    return stats.zscore(a, axis=axis, ddof=ddof, nan_policy=nan_policy)


def iqr(x, axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False):
    """
    міжквартильний розмах.
    
    args:
        x: масив даних
        axis: вісь для обчислення
        rng: квантилі для обчислення (за замовчуванням 25-75)
        scale: масштабний множник
        nan_policy: як обробляти na значення
        interpolation: метод інтерполяції
        keepdims: чи зберігати розмірності
        
    returns:
        float або numpy.ndarray: міжквартильний розмах
        
    example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> iqr_value = iqr(data)
        >>> print(iqr_value)
    """
    return stats.iqr(x, axis=axis, rng=rng, scale=scale, nan_policy=nan_policy, 
                    interpolation=interpolation, keepdims=keepdims)


def entropy(pk, qk=None, base=None, axis=0):
    """
    обчислити ентропію шеннона.
    
    args:
        pk: розподіл ймовірностей
        qk: другий розподіл для відносної ентропії
        base: основа логарифма
        axis: вісь для обчислення
        
    returns:
        float або numpy.ndarray: ентропія
        
    example:
        >>> pk = np.array([0.5, 0.5])
        >>> entropy_value = entropy(pk)
        >>> print(entropy_value)
    """
    return stats.entropy(pk, qk=qk, base=base, axis=axis)


def normaltest(a, axis=0, nan_policy='propagate'):
    """
    тест на нормальність розподілу (дарлінг-пірсон).
    
    args:
        a: масив даних
        axis: вісь для обчислення
        nan_policy: як обробляти na значення
        
    returns:
        scipy.stats._morestats.NormaltestResult: результати тесту
        
    example:
        >>> data = np.random.normal(0, 1, 100)
        >>> result = normaltest(data)
        >>> print(result)
    """
    return stats.normaltest(a, axis=axis, nan_policy=nan_policy)


def shapiro(x, axis=None, nan_policy='propagate', keepdims=False):
    """
    тест шапіро-уілка на нормальність.
    
    args:
        x: масив даних
        axis: вісь для обчислення
        nan_policy: як обробляти na значення
        keepdims: чи зберігати розмірності
        
    returns:
        scipy.stats._morestats.ShapiroResult: результати тесту
        
    example:
        >>> data = np.random.normal(0, 1, 100)
        >>> result = shapiro(data)
        >>> print(result)
    """
    return stats.shapiro(x, axis=axis, nan_policy=nan_policy, keepdims=keepdims)


def anderson(x, dist='norm'):
    """
    тест андерсона-дарлінга.
    
    args:
        x: масив даних
        dist: тип розподілу ('norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1')
        
    returns:
        scipy.stats._morestats.AndersonResult: результати тесту
        
    example:
        >>> data = np.random.normal(0, 1, 100)
        >>> result = anderson(data)
        >>> print(result)
    """
    return stats.anderson(x, dist=dist)


def kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='auto'):
    """
    критерій колмогорова-смірнова.
    
    args:
        rvs: вибірка або функція для генерації вибірки
        cdf: функція кумулятивного розподілу
        args: аргументи для cdf
        n: кількість точок для обчислення
        alternative: альтернативна гіпотеза
        mode: метод обчислення
        
    returns:
        scipy.stats._ksstats.KstestResult: результати тесту
        
    example:
        >>> data = np.random.normal(0, 1, 100)
        >>> result = kstest(data, 'norm')
        >>> print(result)
    """
    return stats.kstest(rvs, cdf, args=args, N=N, alternative=alternative, mode=mode)


def skew(data, axis=0, bias=True, nan_policy='propagate'):
    """
    обчислити коефіцієнт асиметрії.
    
    args:
        data: масив даних
        axis: вісь для обчислення
        bias: чи коригувати на упередженість
        nan_policy: як обробляти na значення
        
    returns:
        float або numpy.ndarray: коефіцієнт асиметрії
        
    example:
        >>> data = np.random.normal(0, 1, 100)
        >>> skewness = skew(data)
        >>> print(skewness)
    """
    return stats.skew(data, axis=axis, bias=bias, nan_policy=nan_policy)


def kurtosis(data, axis=0, fisher=True, bias=True, nan_policy='propagate'):
    """
    обчислити коефіцієнт ексцесу.
    
    args:
        data: масив даних
        axis: вісь для обчислення
        fisher: чи використовувати визначення фішера
        bias: чи коригувати на упередженість
        nan_policy: як обробляти na значення
        
    returns:
        float або numpy.ndarray: коефіцієнт ексцесу
        
    example:
        >>> data = np.random.normal(0, 1, 100)
        >>> kurt = kurtosis(data)
        >>> print(kurt)
    """
    return stats.kurtosis(data, axis=axis, fisher=fisher, bias=bias, nan_policy=nan_policy)