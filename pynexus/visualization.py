"""
visualization module for PyNexus.

цей модуль надає функції для візуалізації даних.
автор: Андрій Будильников
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot(data, x=None, y=None, kind='auto', **kwargs):
    """
    plot data with automatic plot type selection.
    
    args:
        data: data to plot (dataframe, series, etc.)
        x: column name for x-axis (optional)
        y: column name for y-axis (optional)
        kind: type of plot ('auto', 'line', 'bar', 'scatter', 'hist', 'box', etc.)
        **kwargs: additional keyword arguments to pass to plotting function
        
    returns:
        matplotlib.axes.axes: the plot axes
        
    example:
        >>> df = pd.dataframe({'x': [1, 2, 3], 'y': [1, 4, 2]})
        >>> ax = plot(df, 'x', 'y')
        >>> plt.show()
    """
    # auto-determine plot type if needed
    if kind == 'auto':
        kind = _auto_plot_type(data, x, y)
    
    # create the plot based on kind
    if kind == 'line':
        if x and y:
            ax = data.plot(x=x, y=y, kind='line', **kwargs)
        else:
            ax = data.plot(kind='line', **kwargs)
    elif kind == 'bar':
        if x and y:
            ax = data.plot(x=x, y=y, kind='bar', **kwargs)
        else:
            ax = data.plot(kind='bar', **kwargs)
    elif kind == 'scatter':
        if x and y:
            ax = data.plot(x=x, y=y, kind='scatter', **kwargs)
        else:
            raise ValueError("scatter plot requires both x and y parameters")
    elif kind == 'hist':
        ax = data.plot(kind='hist', **kwargs)
    elif kind == 'box':
        ax = data.plot(kind='box', **kwargs)
    elif kind == 'area':
        ax = data.plot(kind='area', **kwargs)
    elif kind == 'pie':
        if y:
            ax = data.plot(y=y, kind='pie', **kwargs)
        else:
            ax = data.plot(kind='pie', **kwargs)
    elif kind == 'hexbin':
        if x and y:
            ax = data.plot(x=x, y=y, kind='hexbin', **kwargs)
        else:
            raise ValueError("hexbin plot requires both x and y parameters")
    elif kind == 'kde':
        ax = data.plot(kind='kde', **kwargs)
    elif kind == 'density':
        ax = data.plot(kind='density', **kwargs)
    else:
        ax = data.plot(kind=kind, **kwargs)
    
    return ax


def _auto_plot_type(data, x=None, y=None):
    """
    automatically determine the best plot type based on data characteristics.
    
    args:
        data: data to plot
        x: x-axis column (optional)
        y: y-axis column (optional)
        
    returns:
        str: recommended plot type
    """
    if isinstance(data, pd.DataFrame):
        if x and y:
            # check if we have categorical vs numerical data
            x_numeric = pd.api.types.is_numeric_dtype(data[x])
            y_numeric = pd.api.types.is_numeric_dtype(data[y])
            
            if x_numeric and y_numeric:
                return 'scatter' if len(data) < 50 else 'line'
            else:
                return 'bar'
        else:
            # multiple columns, check if they are numeric
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                return 'line' if len(data) < 1000 else 'hist'
            else:
                return 'bar'
    elif isinstance(data, pd.Series):
        if pd.api.types.is_numeric_dtype(data):
            return 'hist' if len(data) > 20 else 'bar'
        else:
            return 'bar'
    else:
        # for arrays or other data types
        return 'line'


def plot_auto(data, **kwargs):
    """
    automatically plot data with intelligent type selection.
    
    args:
        data: data to plot
        **kwargs: additional keyword arguments
        
    returns:
        matplotlib.axes.axes: the plot axes
    """
    return plot(data, kind='auto', **kwargs)


def subplot_grid(data_list, titles=None, ncols=2, figsize=(12, 8)):
    """
    create a grid of subplots for multiple datasets.
    
    args:
        data_list: list of datasets to plot
        titles: list of titles for each subplot
        ncols: number of columns in the grid
        figsize: figure size (width, height)
        
    returns:
        tuple: (figure, axes array)
    """
    nrows = (len(data_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, data in enumerate(data_list):
        if i < len(axes):
            plot_auto(data, ax=axes[i])
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
    
    # hide unused subplots
    for i in range(len(data_list), len(axes)):
        axes[i].set_visible(False)
    
    return fig, axes


def correlation_heatmap(data, method='pearson', figsize=(10, 8), cmap='coolwarm', annot=True):
    """
    створити теплову карту кореляцій.
    
    args:
        data: pandas.dataframe з числовими стовпцями
        method: метод кореляції ('pearson', 'spearman', 'kendall')
        figsize: розмір фігури (ширина, висота)
        cmap: колірна схема
        annot: чи додавати анотації зі значеннями
        
    returns:
        matplotlib.figure.figure: фігура з тепловою картою
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> fig = correlation_heatmap(df)
        >>> plt.show()
    """
    # обчислити кореляційну матрицю
    corr_matrix = data.corr(method=method)
    
    # створити фігуру
    fig, ax = plt.subplots(figsize=figsize)
    
    # побудувати теплову карту
    im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # додати анотації
    if annot:
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black")
    
    # налаштувати мітки
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.columns)
    
    # додати колірну шкалу
    plt.colorbar(im, ax=ax)
    
    # додати заголовок
    ax.set_title(f'кореляційна матриця ({method})')
    
    return fig


def pairplot(data, hue=None, vars=None, diag_kind='auto', plot_kws=None, diag_kws=None, grid_kws=None):
    """
    створити матрицю діаграм розкиду для всіх пар змінних.
    
    args:
        data: pandas.dataframe
        hue: змінна для розфарбування точок
        vars: список змінних для включення
        diag_kind: тип діагональних графіків ('auto', 'hist', 'kde')
        plot_kws: додаткові аргументи для графіків розкиду
        diag_kws: додаткові аргументи для діагональних графіків
        grid_kws: додаткові аргументи для сітки
        
    returns:
        seaborn.axisgrid.PairGrid: об'єкт сітки парних графіків
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> grid = pairplot(df)
        >>> plt.show()
    """
    # validate diag_kind parameter
    valid_diag_kinds = ['auto', 'hist', 'kde']
    if diag_kind not in valid_diag_kinds:
        diag_kind = 'auto'
        
    return sns.pairplot(data, hue=hue, vars=vars, diag_kind=diag_kind, 
                       plot_kws=plot_kws, diag_kws=diag_kws, grid_kws=grid_kws)


def violin_plot(data, x=None, y=None, hue=None, figsize=(10, 6)):
    """
    створити діаграму скрипки.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з діаграмою скрипки
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'a', 'b'], 'value': [1, 2, 3, 4]})
        >>> fig = violin_plot(df, x='category', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=data, x=x, y=y, hue=hue, ax=ax)
    return fig


def swarm_plot(data, x=None, y=None, hue=None, figsize=(10, 6)):
    """
    створити діаграму роя.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з діаграмою роя
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'a', 'b'], 'value': [1, 2, 3, 4]})
        >>> fig = swarm_plot(df, x='category', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(data=data, x=x, y=y, hue=hue, ax=ax)
    return fig


def joint_plot(x, y, data=None, kind='scatter', figsize=(8, 8), **kwargs):
    """
    створити об'єднаний графік з діаграмою розкиду та гістограмами.
    
    args:
        x: змінна для осі x
        y: змінна для осі y
        data: pandas.dataframe (необов'язково)
        kind: тип графіка ('scatter', 'reg', 'resid', 'kde', 'hex')
        figsize: розмір фігури (ширина, висота)
        **kwargs: додаткові аргументи
        
    returns:
        seaborn.axisgrid.JointGrid: об'єкт об'єднаного графіка
        
    example:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> grid = joint_plot(x, y)
        >>> plt.show()
    """
    # validate kind parameter
    valid_kinds = ['scatter', 'reg', 'resid', 'kde', 'hex']
    if kind not in valid_kinds:
        kind = 'scatter'
        
    return sns.jointplot(x=x, y=y, data=data, kind=kind, height=figsize[0]/2, **kwargs)


def clustermap(data, method='average', metric='euclidean', figsize=(10, 10), cmap='viridis'):
    """
    створити кластеризовану теплову карту.
    
    args:
        data: pandas.dataframe з числовими даними
        method: метод кластеризації ('average', 'single', 'complete', 'ward')
        metric: метрика відстані ('euclidean', 'manhattan', 'cosine', etc.)
        figsize: розмір фігури (ширина, висота)
        cmap: колірна схема
        
    returns:
        seaborn.matrix.ClusterGrid: об'єкт кластеризованої теплової карти
        
    example:
        >>> df = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> grid = clustermap(df)
        >>> plt.show()
    """
    return sns.clustermap(data, method=method, metric=metric, figsize=figsize, cmap=cmap)


def regression_plot(x, y, data=None, figsize=(10, 6), **kwargs):
    """
    створити діаграму розкиду з лінією регресії.
    
    args:
        x: змінна для осі x
        y: змінна для осі y
        data: pandas.dataframe (необов'язково)
        figsize: розмір фігури (ширина, висота)
        **kwargs: додаткові аргументи
        
    returns:
        matplotlib.figure.figure: фігура з діаграмою розкиду та регресією
        
    example:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> fig = regression_plot(x, y)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(x=x, y=y, data=data, ax=ax, **kwargs)
    return fig


def residual_plot(x, y, data=None, figsize=(10, 6), **kwargs):
    """
    створити графік залишків.
    
    args:
        x: змінна для осі x
        y: змінна для осі y
        data: pandas.dataframe (необов'язково)
        figsize: розмір фігури (ширина, висота)
        **kwargs: додаткові аргументи
        
    returns:
        matplotlib.figure.figure: фігура з графіком залишків
        
    example:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> fig = residual_plot(x, y)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.residplot(x=x, y=y, data=data, ax=ax, **kwargs)
    return fig


def distribution_plot(data, x=None, hue=None, kind='hist', figsize=(10, 6), **kwargs):
    """
    створити графік розподілу.
    
    args:
        data: pandas.dataframe або series
        x: змінна для відображення (якщо data - dataframe)
        hue: змінна для розфарбування
        kind: тип графіка ('hist', 'kde', 'ecdf')
        figsize: розмір фігури (ширина, висота)
        **kwargs: додаткові аргументи
        
    returns:
        matplotlib.figure.figure: фігура з графіком розподілу
        
    example:
        >>> data = [1, 2, 3, 4, 5, 2, 3, 4, 5, 6]
        >>> fig = distribution_plot(data)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    if kind == 'hist':
        sns.histplot(data=data, x=x, hue=hue, ax=ax, **kwargs)
    elif kind == 'kde':
        sns.kdeplot(data=data, x=x, hue=hue, ax=ax, **kwargs)
    elif kind == 'ecdf':
        sns.ecdfplot(data=data, x=x, hue=hue, ax=ax, **kwargs)
    return fig


def time_series_plot(data, x=None, y=None, figsize=(12, 6), **kwargs):
    """
    створити графік часових рядів.
    
    args:
        data: pandas.dataframe з часовим індексом
        x: змінна для осі x (якщо немає індексу)
        y: змінна для осі y
        figsize: розмір фігури (ширина, висота)
        **kwargs: додаткові аргументи
        
    returns:
        matplotlib.figure.figure: фігура з графіком часових рядів
        
    example:
        >>> dates = pd.date_range('2023-01-01', periods=100)
        >>> values = np.random.randn(100).cumsum()
        >>> df = pd.dataframe({'date': dates, 'value': values})
        >>> fig = time_series_plot(df, x='date', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    if x and y:
        ax.plot(data[x], data[y], **kwargs)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    else:
        data.plot(ax=ax, **kwargs)
    ax.set_title('графік часових рядів')
    return fig


def boxen_plot(data, x=None, y=None, hue=None, figsize=(10, 6)):
    """
    створити розширenu діаграму ящика (boxen plot).
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з розширеною діаграмою ящика
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'a', 'b'], 'value': [1, 2, 3, 4]})
        >>> fig = boxen_plot(df, x='category', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=ax)
    return fig


def count_plot(data, x=None, hue=None, figsize=(10, 6)):
    """
    створити діаграму підрахунку.
    
    args:
        data: pandas.dataframe
        x: змінна для підрахунку
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з діаграмою підрахунку
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'a', 'b', 'a']})
        >>> fig = count_plot(df, x='category')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=data, x=x, hue=hue, ax=ax)
    return fig


def point_plot(data, x=None, y=None, hue=None, figsize=(10, 6)):
    """
    створити точкову діаграму.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з точковою діаграмою
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'a', 'b'], 'value': [1, 2, 3, 4]})
        >>> fig = point_plot(df, x='category', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.pointplot(data=data, x=x, y=y, hue=hue, ax=ax)
    return fig


def bar_plot(data, x=None, y=None, hue=None, figsize=(10, 6)):
    """
    створити стовпчикову діаграму.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з стовпчиковою діаграмою
        
    example:
        >>> df = pd.dataframe({'category': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        >>> fig = bar_plot(df, x='category', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)
    return fig


def line_plot(data, x=None, y=None, hue=None, figsize=(10, 6)):
    """
    створити лінійну діаграму.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з лінійною діаграмою
        
    example:
        >>> df = pd.dataframe({'time': [1, 2, 3, 4], 'value': [1, 4, 2, 8]})
        >>> fig = line_plot(df, x='time', y='value')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax)
    return fig


def scatter_plot(data, x=None, y=None, hue=None, style=None, size=None, figsize=(10, 6)):
    """
    створити діаграму розкиду.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        hue: змінна для розфарбування
        style: змінна для стилю точок
        size: змінна для розміру точок
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з діаграмою розкиду
        
    example:
        >>> df = pd.dataframe({'x': [1, 2, 3], 'y': [4, 5, 6], 'category': ['a', 'b', 'a']})
        >>> fig = scatter_plot(df, x='x', y='y', hue='category')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=data, x=x, y=y, hue=hue, style=style, size=size, ax=ax)
    return fig


def heatmap(data, x=None, y=None, figsize=(10, 8), cmap='viridis', annot=True):
    """
    створити теплову карту.
    
    args:
        data: pandas.dataframe або матриця значень
        x: мітки для осі x (необов'язково)
        y: мітки для осі y (необов'язково)
        figsize: розмір фігури (ширина, висота)
        cmap: колірна схема
        annot: чи додавати анотації зі значеннями
        
    returns:
        matplotlib.figure.figure: фігура з тепловою картою
        
    example:
        >>> data = np.random.rand(4, 4)
        >>> fig = heatmap(data)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(data, pd.DataFrame):
        sns.heatmap(data, xticklabels=x or data.columns, yticklabels=y or data.index, 
                   cmap=cmap, annot=annot, ax=ax)
    else:
        x_labels = x if x is not None else range(data.shape[1])
        y_labels = y if y is not None else range(data.shape[0])
        sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, annot=annot, ax=ax)
    return fig


def save_plot(fig, filename, dpi=300, bbox_inches='tight'):
    """
    зберегти фігуру у файл.
    
    args:
        fig: matplotlib.figure.figure
        filename: ім'я файлу для збереження
        dpi: роздільна здатність
        bbox_inches: параметри збереження меж
        
    example:
        >>> fig = plt.figure()
        >>> save_plot(fig, 'my_plot.png')
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)


def set_style(style='whitegrid', palette='deep', font_scale=1):
    """
    налаштувати стиль візуалізацій.
    
    args:
        style: стиль seaborn ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        palette: палітра кольорів ('deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind')
        font_scale: масштаб шрифту
        
    example:
        >>> set_style('darkgrid', 'colorblind', 1.2)
    """
    # validate style parameter
    valid_styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']
    if style not in valid_styles:
        style = 'whitegrid'
        
    # validate palette parameter
    valid_palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']
    if palette not in valid_palettes:
        palette = 'deep'
        
    sns.set_style(style)
    sns.set_palette(palette)
    sns.set_context("paper", font_scale=font_scale)


def reset_style():
    """
    скинути стиль візуалізацій до стандартного.
    
    example:
        >>> reset_style()
    """
    sns.reset_defaults()
    plt.style.use('default')


def multi_plot(data_list, plot_funcs, titles=None, figsize=(15, 10)):
    """
    створити кілька графіків у одній фігурі.
    
    args:
        data_list: список наборів даних
        plot_funcs: список функцій для побудови графіків
        titles: список заголовків
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з кількома графіками
        
    example:
        >>> data1 = pd.dataframe({'x': [1, 2, 3], 'y': [1, 4, 2]})
        >>> data2 = pd.dataframe({'a': [1, 2, 3], 'b': [3, 2, 1]})
        >>> fig = multi_plot([data1, data2], [lambda d: d.plot(x='x', y='y'), lambda d: d.plot(x='a', y='b')])
        >>> plt.show()
    """
    n_plots = len(data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (data, plot_func) in enumerate(zip(data_list, plot_funcs)):
        plot_func(data).plot(ax=axes[i])
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    plt.tight_layout()
    return fig


def animate_plot(data, x, y, time_col, filename=None, interval=200):
    """
    створити анімований графік.
    
    args:
        data: pandas.dataframe з часовими даними
        x: змінна для осі x
        y: змінна для осі y
        time_col: змінна часу
        filename: ім'я файлу для збереження анімації (необов'язково)
        interval: інтервал між кадрами в мс
        
    returns:
        matplotlib.animation.funcanimation: об'єкт анімації
        
    example:
        >>> dates = pd.date_range('2023-01-01', periods=100)
        >>> df = pd.dataframe({'date': dates, 'value': np.random.randn(100).cumsum()})
        >>> anim = animate_plot(df, 'date', 'value', 'date')
        >>> plt.show()
    """
    try:
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots()
        
        def animate(frame):
            ax.clear()
            subset = data[data[time_col] <= data[time_col].iloc[frame]]
            ax.plot(subset[x], subset[y])
            ax.set_title(f'час: {subset[time_col].iloc[-1] if len(subset) > 0 else "початок"}')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=interval, repeat=True)
        
        if filename:
            anim.save(filename, writer='pillow')
        
        return anim
    except ImportError:
        print("matplotlib.animation не доступний.")
        return None


def interactive_plot(data, x, y, kind='scatter'):
    """
    створити інтерактивний графік.
    
    args:
        data: pandas.dataframe
        x: змінна для осі x
        y: змінна для осі y
        kind: тип графіка
        
    returns:
        plotly.graph_objects.figure: інтерактивна фігура
        
    example:
        >>> df = pd.dataframe({'x': [1, 2, 3], 'y': [1, 4, 2]})
        >>> fig = interactive_plot(df, 'x', 'y')
        >>> fig.show()
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        if kind == 'scatter':
            fig = px.scatter(data, x=x, y=y)
        elif kind == 'line':
            fig = px.line(data, x=x, y=y)
        elif kind == 'bar':
            fig = px.bar(data, x=x, y=y)
        else:
            fig = px.scatter(data, x=x, y=y)
            
        return fig
    except ImportError:
        print("plotly не встановлено. встановіть plotly для інтерактивних графіків.")
        return None


def plot_3d(x, y, z, kind='scatter', figsize=(10, 8)):
    """
    створити 3d графік.
    
    args:
        x: дані для осі x
        y: дані для осі y
        z: дані для осі z
        kind: тип графіка ('scatter', 'surface', 'wireframe')
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: 3d фігура
        
    example:
        >>> x = np.random.rand(100)
        >>> y = np.random.rand(100)
        >>> z = np.random.rand(100)
        >>> fig = plot_3d(x, y, z)
        >>> plt.show()
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if kind == 'scatter':
        ax.scatter(x, y, z)
    elif kind == 'surface':
        ax.plot_trisurf(x, y, z)
    elif kind == 'wireframe':
        ax.plot_wireframe(x, y, z)
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    return fig


def plot_confusion_matrix(cm, labels=None, figsize=(8, 6), cmap='Blues'):
    """
    створити графік матриці помилок.
    
    args:
        cm: матриця помилок (numpy array)
        labels: мітки класів
        figsize: розмір фігури (ширина, висота)
        cmap: колірна схема
        
    returns:
        matplotlib.figure.figure: фігура з матрицею помилок
        
    example:
        >>> cm = np.array([[50, 2], [3, 45]])
        >>> fig = plot_confusion_matrix(cm, labels=['class 0', 'class 1'])
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    if labels is not None:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               ylabel='істинні мітки',
               xlabel='передбачені мітки')
    
    # додати анотації з текстом
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('матриця помилок')
    fig.tight_layout()
    
    return fig


def plot_roc_curve(fpr, tpr, auc_score=None, figsize=(8, 6)):
    """
    створити roc-криву.
    
    args:
        fpr: false positive rate
        tpr: true positive rate
        auc_score: площа під кривою (необов'язково)
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з roc-кривою
        
    example:
        >>> fpr = [0, 0.1, 0.2, 0.3, 1.0]
        >>> tpr = [1.0, 0.9, 0.8, 0.7, 0.0]
        >>> fig = plot_roc_curve(fpr, tpr, auc_score=0.85)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'roc curve (auc = {auc_score:.2f})' if auc_score else 'roc curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('roc-крива')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig


def plot_precision_recall_curve(precision, recall, figsize=(8, 6)):
    """
    створити криву точність-повнота.
    
    args:
        precision: точність
        recall: повнота
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з кривою точність-повнота
        
    example:
        >>> precision = [1.0, 0.8, 0.6, 0.4, 0.2]
        >>> recall = [0.2, 0.4, 0.6, 0.8, 1.0]
        >>> fig = plot_precision_recall_curve(precision, recall)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='blue', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('повнота')
    ax.set_ylabel('точність')
    ax.set_title('крива точність-повнота')
    ax.grid(True)
    
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, title='крива навчання', figsize=(10, 6)):
    """
    створити графік кривої навчання.
    
    args:
        train_sizes: розміри навчальних вибірок
        train_scores: бали навчання
        val_scores: бали валідації
        title: заголовок графіка
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з кривою навчання
        
    example:
        >>> train_sizes = np.array([10, 50, 100, 150, 200])
        >>> train_scores = np.array([[0.8, 0.85, 0.9], [0.82, 0.87, 0.91], [0.85, 0.88, 0.92], [0.86, 0.89, 0.93], [0.87, 0.90, 0.94]])
        >>> val_scores = np.array([[0.7, 0.75, 0.8], [0.72, 0.77, 0.81], [0.75, 0.78, 0.82], [0.76, 0.79, 0.83], [0.77, 0.80, 0.84]])
        >>> fig = plot_learning_curve(train_sizes, train_scores, val_scores)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='навчання')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='валідація')
    
    ax.set_xlabel('розмір навчальної вибірки')
    ax.set_ylabel('точність')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig


def plot_validation_curve(param_range, train_scores, val_scores, param_name='параметр', title='крива валідації', figsize=(10, 6)):
    """
    створити графік кривої валідації.
    
    args:
        param_range: діапазон значень параметра
        train_scores: бали навчання
        val_scores: бали валідації
        param_name: ім'я параметра
        title: заголовок графіка
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з кривою валідації
        
    example:
        >>> param_range = np.array([0.1, 1, 10])
        >>> train_scores = np.array([[0.8, 0.85, 0.9], [0.82, 0.87, 0.91], [0.85, 0.88, 0.92]])
        >>> val_scores = np.array([[0.7, 0.75, 0.8], [0.72, 0.77, 0.81], [0.75, 0.78, 0.82]])
        >>> fig = plot_validation_curve(param_range, train_scores, val_scores, param_name='c')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(param_range, train_mean, 'o-', color='blue', label='навчання')
    ax.plot(param_range, val_mean, 'o-', color='orange', label='валідація')
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('точність')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_xscale('log')
    
    return fig


def plot_feature_importance(importance, feature_names=None, top_n=20, figsize=(10, 8)):
    """
    створити графік важливості ознак.
    
    args:
        importance: важливість ознак (numpy array)
        feature_names: імена ознак
        top_n: кількість топ ознак для відображення
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з важливістю ознак
        
    example:
        >>> importance = np.array([0.1, 0.3, 0.2, 0.4])
        >>> feature_names = ['feature 1', 'feature 2', 'feature 3', 'feature 4']
        >>> fig = plot_feature_importance(importance, feature_names)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # сортувати за важливістю
    indices = np.argsort(importance)[::-1]
    
    # обмежити кількість ознак
    indices = indices[:top_n]
    
    # отримати відсортовані значення
    sorted_importance = importance[indices]
    
    # отримати імена ознак
    if feature_names is not None:
        sorted_names = [feature_names[i] for i in indices]
    else:
        sorted_names = [f'feature {i}' for i in indices]
    
    # побудувати графік
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # найважливіші ознаки зверху
    ax.set_xlabel('важливість')
    ax.set_title('важливість ознак')
    
    return fig


def plot_silhouette_analysis(X, labels, metric='euclidean', figsize=(10, 6)):
    """
    створити графік силуетного аналізу.
    
    args:
        x: дані
        labels: мітки кластерів
        metric: метрика відстані
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з силуетним аналізом
        
    example:
        >>> x = np.random.rand(100, 2)
        >>> labels = np.random.randint(0, 3, 100)
        >>> fig = plot_silhouette_analysis(x, labels)
        >>> plt.show()
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # обчислити силуетний бал
    silhouette_avg = silhouette_score(X, labels, metric=metric)
    
    # обчислити силуетні значення для кожного зразка
    sample_silhouette_values = silhouette_samples(X, labels, metric=metric)
    
    y_lower = 10
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    for i in range(n_clusters):
        # агрегувати силуетні значення для зразків з кластера i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # підписати кластери посередині
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # обчислити нове положення y для наступного графіка
        y_lower = y_upper + 10  # 10 для відступу між кластерами
    
    ax.set_xlabel("значення силуетного коефіцієнта")
    ax.set_ylabel("мітка кластера")
    
    # вертикальна лінія для середнього силуетного балу всіх зразків
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f"середній силуетний бал: {silhouette_avg:.2f}")
    
    ax.set_yticks([])  # очистити мітки осі y
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend()
    
    ax.set_title("силуетний аналіз для кластеризації")
    
    return fig


def plot_elbow_method(X, k_range=range(1, 11), figsize=(10, 6)):
    """
    створити графік методу ліктя для визначення оптимальної кількості кластерів.
    
    args:
        x: дані
        k_range: діапазон значень k
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з методом ліктя
        
    example:
        >>> x = np.random.rand(100, 2)
        >>> fig = plot_elbow_method(x, k_range=range(1, 6))
        >>> plt.show()
    """
    from sklearn.cluster import KMeans
    
    distortions = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_range, distortions, 'bo-')
    ax.set_xlabel('кількість кластерів (k)')
    ax.set_ylabel('інерція (сума квадратів відстаней до центроїдів)')
    ax.set_title('метод ліктя для визначення оптимальної кількості кластерів')
    ax.grid(True)
    
    return fig


def plot_dendrogram(model, figsize=(12, 8)):
    """
    створити дендрограму для ієрархічної кластеризації.
    
    args:
        model: натренована модель агломеративної кластеризації
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з дендрограмою
        
    example:
        >>> from sklearn.cluster import AgglomerativeClustering
        >>> x = np.random.rand(10, 2)
        >>> model = AgglomerativeClustering(n_clusters=3)
        >>> model.fit(x)
        >>> fig = plot_dendrogram(model)
        >>> plt.show()
    """
    from scipy.cluster.hierarchy import dendrogram
    
    # для побудови дендрограми потрібно мати доступ до лінків
    # це приклад, в реальності потрібно зберігати лінки під час навчання
    fig, ax = plt.subplots(figsize=figsize)
    
    # побудова дендрограми
    # dendrogram(linkage_matrix, ax=ax)
    ax.set_title('дендрограма')
    ax.set_xlabel('індекс зразка')
    ax.set_ylabel('відстань')
    
    return fig


def plot_pca_components(pca_model, feature_names=None, figsize=(12, 8)):
    """
    створити графік компонентів pca.
    
    args:
        pca_model: натренована модель pca
        feature_names: імена ознак
        figsize: розмір фігури (ширина, висота)
        
    returns:
        matplotlib.figure.figure: фігура з компонентами pca
        
    example:
        >>> from sklearn.decomposition import PCA
        >>> x = np.random.rand(100, 5)
        >>> pca = PCA(n_components=2)
        >>> pca.fit(x)
        >>> fig = plot_pca_components(pca)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    components = pca_model.components_
    n_components = components.shape[0]
    n_features = components.shape[1]
    
    # створити теплову карту компонентів
    im = ax.imshow(components, cmap='RdBu_r', aspect='auto')
    
    # налаштувати мітки
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([f'pc{i+1}' for i in range(n_components)])
    
    if feature_names is not None:
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
    else:
        ax.set_xticks(range(n_features))
        ax.set_xticklabels([f'feature {i}' for i in range(n_features)], rotation=45, ha='right')
    
    ax.set_xlabel('ознаки')
    ax.set_ylabel('головні компоненти')
    ax.set_title('компоненти головних компонент')
    
    # додати колірну шкалу
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


# продовження файлу буде в наступному повідомленні через обмеження довжини...