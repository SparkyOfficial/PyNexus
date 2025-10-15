"""
machine learning module for PyNexus.

цей модуль надає функції для машинного навчання.
автор: Андрій Будильников
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# data preprocessing functions
def scale_data(data, method='standard'):
    """
    масштабувати дані.
    
    args:
        data: вхідні дані (numpy array або pandas dataframe)
        method: метод масштабування ('standard', 'minmax')
        
    returns:
        numpy array: масштабовані дані
        
    example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> scaled = scale_data(data, method='standard')
        >>> print(scaled)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    return scaler.fit_transform(data)


def encode_labels(data):
    """
    закодувати категоріальні мітки.
    
    args:
        data: вхідні дані з категоріальними мітками
        
    returns:
        numpy array: закодовані мітки
        
    example:
        >>> data = ['cat', 'dog', 'cat', 'bird']
        >>> encoded = encode_labels(data)
        >>> print(encoded)
    """
    encoder = LabelEncoder()
    return encoder.fit_transform(data)


def split_data(X, y, test_size=0.2, random_state=42):
    """
    розділити дані на навчальну та тестову вибірки.
    
    args:
        x: ознаки
        y: цільова змінна
        test_size: розмір тестової вибірки
        random_state: початкове значення для відтворюваності
        
    returns:
        tuple: (x_train, x_test, y_train, y_test)
        
    example:
        >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 1, 0, 1])
        >>> x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.5)
        >>> print(x_train.shape, x_test.shape)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# regression models
def linear_regression(X_train, y_train, X_test=None):
    """
    лінійна регресія.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.linear_model.LinearRegression: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = linear_regression(x_train, y_train)
        >>> print(model.coef_)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def polynomial_features(X, degree=2):
    """
    створити поліноміальні ознаки.
    
    args:
        x: вхідні ознаки
        degree: ступінь полінома
        
    returns:
        numpy array: поліноміальні ознаки
        
    example:
        >>> x = np.array([[1], [2], [3]])
        >>> poly_x = polynomial_features(x, degree=2)
        >>> print(poly_x)
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)


def ridge_regression(X_train, y_train, alpha=1.0, X_test=None):
    """
    ридж-регресія.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        alpha: параметр регуляризації
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.linear_model.Ridge: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = ridge_regression(x_train, y_train, alpha=0.5)
        >>> print(model.coef_)
    """
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def lasso_regression(X_train, y_train, alpha=1.0, X_test=None):
    """
    лассо-регресія.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        alpha: параметр регуляризації
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.linear_model.Lasso: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = lasso_regression(x_train, y_train, alpha=0.5)
        >>> print(model.coef_)
    """
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def elastic_net_regression(X_train, y_train, alpha=1.0, l1_ratio=0.5, X_test=None):
    """
    еластична мережева регресія.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        alpha: параметр регуляризації
        l1_ratio: співвідношення l1/l2 регуляризації
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.linear_model.ElasticNet: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = elastic_net_regression(x_train, y_train, alpha=0.5, l1_ratio=0.3)
        >>> print(model.coef_)
    """
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


# classification models
def logistic_regression(X_train, y_train, X_test=None):
    """
    логістична регресія.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.linear_model.LogisticRegression: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = logistic_regression(x_train, y_train)
        >>> print(model.coef_)
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=None, X_test=None):
    """
    класифікатор випадкового лісу.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        n_estimators: кількість дерев
        max_depth: максимальна глибина дерев
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.ensemble.RandomForestClassifier: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = random_forest_classifier(x_train, y_train, n_estimators=50)
        >>> print(model.feature_importances_)
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def support_vector_classifier(X_train, y_train, kernel='rbf', C=1.0, X_test=None):
    """
    класифікатор опорних векторів.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        kernel: тип ядра ('linear', 'poly', 'rbf', 'sigmoid')
        c: параметр регуляризації
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.svm.SVC: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = support_vector_classifier(x_train, y_train, kernel='linear')
        >>> print(model.support_vectors_)
    """
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def k_neighbors_classifier(X_train, y_train, n_neighbors=5, weights='uniform', X_test=None):
    """
    класифікатор k-найближчих сусідів.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        n_neighbors: кількість сусідів
        weights: тип зважування ('uniform', 'distance')
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.neighbors.KNeighborsClassifier: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = k_neighbors_classifier(x_train, y_train, n_neighbors=3)
        >>> print(model.n_neighbors)
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


# regression models for continuous targets
def random_forest_regressor(X_train, y_train, n_estimators=100, max_depth=None, X_test=None):
    """
    регресор випадкового лісу.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        n_estimators: кількість дерев
        max_depth: максимальна глибина дерев
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.ensemble.RandomForestRegressor: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = random_forest_regressor(x_train, y_train, n_estimators=50)
        >>> print(model.feature_importances_)
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def support_vector_regressor(X_train, y_train, kernel='rbf', C=1.0, X_test=None):
    """
    регресор опорних векторів.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        kernel: тип ядра ('linear', 'poly', 'rbf', 'sigmoid')
        c: параметр регуляризації
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.svm.SVR: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = support_vector_regressor(x_train, y_train, kernel='linear')
        >>> print(model.support_vectors_)
    """
    model = SVR(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def k_neighbors_regressor(X_train, y_train, n_neighbors=5, weights='uniform', X_test=None):
    """
    регресор k-найближчих сусідів.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        n_neighbors: кількість сусідів
        weights: тип зважування ('uniform', 'distance')
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.neighbors.KNeighborsRegressor: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([2, 4, 6])
        >>> model = k_neighbors_regressor(x_train, y_train, n_neighbors=3)
        >>> print(model.n_neighbors)
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


# clustering models
def k_means_clustering(X, n_clusters=3, init='k-means++', n_init=10, max_iter=300):
    """
    кластеризація методом k-середніх.
    
    args:
        x: вхідні дані
        n_clusters: кількість кластерів
        init: метод ініціалізації ('k-means++', 'random')
        n_init: кількість запусків алгоритму
        max_iter: максимальна кількість ітерацій
        
    returns:
        sklearn.cluster.KMeans: натренована модель
        numpy array: мітки кластерів
        
    example:
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> model, labels = k_means_clustering(x, n_clusters=2)
        >>> print(labels)
    """
    model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(X)
    return model, labels


def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    кластеризація методом dbscan.
    
    args:
        x: вхідні дані
        eps: максимальна відстань між точками в одному кластері
        min_samples: мінімальна кількість точок для формування кластера
        
    returns:
        sklearn.cluster.DBSCAN: натренована модель
        numpy array: мітки кластерів
        
    example:
        >>> x = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
        >>> model, labels = dbscan_clustering(x, eps=3, min_samples=2)
        >>> print(labels)
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels


# dimensionality reduction
def principal_component_analysis(X, n_components=None):
    """
    аналіз головних компонент.
    
    args:
        x: вхідні дані
        n_components: кількість компонент (за замовчуванням min(n_samples, n_features))
        
    returns:
        sklearn.decomposition.PCA: натренована модель
        numpy array: перетворені дані
        
    example:
        >>> x = np.array([[1, 2], [3, 4], [5, 6]])
        >>> model, transformed = principal_component_analysis(x, n_components=1)
        >>> print(transformed.shape)
    """
    model = PCA(n_components=n_components)
    transformed = model.fit_transform(X)
    return model, transformed


# model evaluation functions
def evaluate_classifier(y_true, y_pred, average='weighted'):
    """
    оцінити класифікатор.
    
    args:
        y_true: справжні мітки
        y_pred: передбачені мітки
        average: тип усереднення ('micro', 'macro', 'weighted', 'samples')
        
    returns:
        dict: метрики оцінки
        
    example:
        >>> y_true = [0, 1, 2, 2, 2]
        >>> y_pred = [0, 0, 2, 2, 1]
        >>> metrics = evaluate_classifier(y_true, y_pred)
        >>> print(metrics)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_regressor(y_true, y_pred):
    """
    оцінити регресор.
    
    args:
        y_true: справжні значення
        y_pred: передбачені значення
        
    returns:
        dict: метрики оцінки
        
    example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> metrics = evaluate_regressor(y_true, y_pred)
        >>> print(metrics)
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2_score': r2
    }


def confusion_matrix_report(y_true, y_pred, labels=None, sample_weight=None):
    """
    згенерувати матрицю помилок та звіт про класифікацію.
    
    args:
        y_true: справжні мітки
        y_pred: передбачені мітки
        labels: список міток для включення у матрицю
        sample_weight: ваги зразків
        
    returns:
        tuple: (матриця помилок, звіт про класифікацію)
        
    example:
        >>> y_true = [0, 1, 2, 2, 2]
        >>> y_pred = [0, 0, 2, 2, 1]
        >>> cm, report = confusion_matrix_report(y_true, y_pred)
        >>> print(cm)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return cm, report


def cross_validation_score(model, X, y, cv=5, scoring=None):
    """
    оцінити модель методом перехресної валідації.
    
    args:
        model: модель для оцінки
        x: ознаки
        y: цільова змінна
        cv: кількість блоків перехресної валідації
        scoring: метрика оцінки
        
    returns:
        numpy array: результати перехресної валідації
        
    example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> x = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> scores = cross_validation_score(model, x, y, cv=3)
        >>> print(scores)
    """
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)


def grid_search_cv(model, param_grid, X, y, cv=5, scoring=None):
    """
    підібрати гіперпараметри методом сіткового пошуку.
    
    args:
        model: модель для налаштування
        param_grid: сітка параметрів
        x: ознаки
        y: цільова змінна
        cv: кількість блоків перехресної валідації
        scoring: метрика оцінки
        
    returns:
        sklearn.model_selection.GridSearchCV: натренований пошуковий об'єкт
        
    example:
        >>> from sklearn.svm import SVC
        >>> model = SVC()
        >>> param_grid = {'c': [1, 10], 'kernel': ['linear', 'rbf']}
        >>> x = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> search = grid_search_cv(model, param_grid, x, y, cv=3)
        >>> print(search.best_params_)
    """
    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    search.fit(X, y)
    return search


def silhouette_analysis(X, labels):
    """
    обчислити силуетний аналіз для кластеризації.
    
    args:
        x: вхідні дані
        labels: мітки кластерів
        
    returns:
        float: середній силуетний бал
        
    example:
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> labels = [0, 0, 0, 1, 1, 1]
        >>> score = silhouette_analysis(x, labels)
        >>> print(score)
    """
    return silhouette_score(X, labels)


# feature selection functions
def feature_importance(model, feature_names=None):
    """
    отримати важливість ознак для моделі.
    
    args:
        model: натренована модель з атрибутом feature_importances_
        feature_names: імена ознак
        
    returns:
        pandas.series: важливість ознак
        
    example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier().fit([[1], [2], [3]], [0, 1, 1])
        >>> importance = feature_importance(model)
        >>> print(importance)
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        if feature_names is not None:
            return pd.Series(importance, index=feature_names)
        else:
            return pd.Series(importance)
    else:
        raise ValueError("model does not have feature_importances_ attribute")


def correlation_filter(X, y, threshold=0.1):
    """
    відфільтрувати ознаки за кореляцією з цільовою змінною.
    
    args:
        x: ознаки (pandas dataframe)
        y: цільова змінна
        threshold: поріг кореляції
        
    returns:
        list: імена відфільтрованих ознак
        
    example:
        >>> x = pd.dataframe({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> y = [1, 2, 3]
        >>> filtered_features = correlation_filter(x, y, threshold=0.5)
        >>> print(filtered_features)
    """
    correlations = X.corrwith(pd.Series(y))
    return correlations[abs(correlations) >= threshold].index.tolist()


# ensemble methods
def voting_classifier(estimators, X_train, y_train, voting='hard', X_test=None):
    """
    класифікатор голосування.
    
    args:
        estimators: список (ім'я, модель) кортежів
        x_train: навчальні ознаки
        y_train: навчальні цілі
        voting: тип голосування ('hard', 'soft')
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.ensemble.VotingClassifier: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.svm import SVC
        >>> estimators = [('lr', LogisticRegression()), ('svc', SVC(probability=True))]
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = voting_classifier(estimators, x_train, y_train)
        >>> print(model.named_estimators_)
    """
    from sklearn.ensemble import VotingClassifier
    model = VotingClassifier(estimators=estimators, voting=voting)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def bagging_classifier(base_estimator, X_train, y_train, n_estimators=10, X_test=None):
    """
    класифікатор бегінгу.
    
    args:
        base_estimator: базова модель
        x_train: навчальні ознаки
        y_train: навчальні цілі
        n_estimators: кількість базових моделей
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.ensemble.BaggingClassifier: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> base = DecisionTreeClassifier()
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = bagging_classifier(base, x_train, y_train, n_estimators=5)
        >>> print(model.n_estimators)
    """
    from sklearn.ensemble import BaggingClassifier
    model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


def boosting_classifier(X_train, y_train, n_estimators=50, learning_rate=1.0, X_test=None):
    """
    класифікатор бустингу.
    
    args:
        x_train: навчальні ознаки
        y_train: навчальні цілі
        n_estimators: кількість базових моделей
        learning_rate: швидкість навчання
        x_test: тестові ознаки (необов'язково)
        
    returns:
        sklearn.ensemble.AdaBoostClassifier: натренована модель
        numpy array: передбачення (якщо x_test надано)
        
    example:
        >>> x_train = np.array([[1], [2], [3]])
        >>> y_train = np.array([0, 1, 1])
        >>> model = boosting_classifier(x_train, y_train, n_estimators=20)
        >>> print(model.n_estimators)
    """
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    
    if X_test is not None:
        predictions = model.predict(X_test)
        return model, predictions
    
    return model


# anomaly detection
def isolation_forest(X, contamination=0.1, random_state=42):
    """
    виявлення аномалій методом ізоляційного лісу.
    
    args:
        x: вхідні дані
        contamination: очікувана частка аномалій
        random_state: початкове значення для відтворюваності
        
    returns:
        sklearn.ensemble.IsolationForest: натренована модель
        numpy array: мітки аномалій (-1 для аномалій, 1 для нормальних)
        
    example:
        >>> x = np.array([[1], [2], [3], [100]])
        >>> model, labels = isolation_forest(x, contamination=0.1)
        >>> print(labels)
    """
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def one_class_svm(X, kernel='rbf', nu=0.1):
    """
    виявлення аномалій методом однокласового svm.
    
    args:
        x: вхідні дані
        kernel: тип ядра
        nu: верхня межа частки помилок
        
    returns:
        sklearn.svm.OneClassSVM: натренована модель
        numpy array: мітки аномалій (-1 для аномалій, 1 для нормальних)
        
    example:
        >>> x = np.array([[1], [2], [3], [100]])
        >>> model, labels = one_class_svm(x, nu=0.1)
        >>> print(labels)
    """
    from sklearn.svm import OneClassSVM
    model = OneClassSVM(kernel=kernel, nu=nu)
    labels = model.fit_predict(X)
    return model, labels


# time series forecasting
def time_series_split(X, y, n_splits=5, test_size=None, gap=0):
    """
    розділити часові ряди для перехресної валідації.
    
    args:
        x: ознаки
        y: цільова змінна
        n_splits: кількість блоків
        test_size: розмір тестового блоку
        gap: розрив між навчальним і тестовим блоками
        
    returns:
        sklearn.model_selection.TimeSeriesSplit: об'єкт розділення
        
    example:
        >>> x = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> tscv = time_series_split(x, y, n_splits=3)
        >>> for train_index, test_index in tscv.split(x):
        ...     print("train:", train_index, "test:", test_index)
    """
    from sklearn.model_selection import TimeSeriesSplit
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)


def exponential_smoothing(data, alpha=0.3, beta=None, gamma=None, seasonal_periods=None):
    """
    експоненційне згладжування для прогнозування часових рядів.
    
    args:
        data: часовий ряд
        alpha: параметр згладжування рівня
        beta: параметр згладжування тренду
        gamma: параметр згладжування сезонності
        seasonal_periods: кількість періодів сезонності
        
    returns:
        numpy array: згладжені значення
        
    example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> smoothed = exponential_smoothing(data, alpha=0.5)
        >>> print(smoothed)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(data, trend='add' if beta else None, 
                                seasonal='add' if gamma else None, 
                                seasonal_periods=seasonal_periods)
    fitted = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
    return fitted.fittedvalues


# model persistence
def save_model(model, filepath):
    """
    зберегти модель у файл.
    
    args:
        model: модель для збереження
        filepath: шлях до файлу
        
    example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression().fit([[1], [2]], [2, 4])
        >>> save_model(model, 'model.pkl')
    """
    import joblib
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    завантажити модель з файлу.
    
    args:
        filepath: шлях до файлу
        
    returns:
        модель: завантажена модель
        
    example:
        >>> model = load_model('model.pkl')
        >>> print(model.coef_)
    """
    import joblib
    return joblib.load(filepath)


# utility functions
def train_test_validation_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    розділити дані на навчальну, валідаційну та тестову вибірки.
    
    args:
        x: ознаки
        y: цільова змінна
        train_size: розмір навчальної вибірки
        val_size: розмір валідаційної вибірки
        test_size: розмір тестової вибірки
        random_state: початкове значення для відтворюваності
        
    returns:
        tuple: (x_train, x_val, x_test, y_train, y_val, y_test)
        
    example:
        >>> x = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> x_train, x_val, x_test, y_train, y_val, y_test = train_test_validation_split(x, y)
        >>> print(x_train.shape, x_val.shape, x_test.shape)
    """
    # normalize sizes
    total = train_size + val_size + test_size
    train_size /= total
    val_size /= total
    
    # first split: train and (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state
    )
    
    # second split: val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(val_size + test_size), random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def bootstrap_sample(X, y, n_samples=None, random_state=42):
    """
    створити бутстреп-вибірку.
    
    args:
        x: ознаки
        y: цільова змінна
        n_samples: кількість зразків у вибірці
        random_state: початкове значення для відтворюваності
        
    returns:
        tuple: (x_bootstrap, y_bootstrap)
        
    example:
        >>> x = np.array([[1], [2], [3]])
        >>> y = np.array([1, 2, 3])
        >>> x_boot, y_boot = bootstrap_sample(x, y, n_samples=5)
        >>> print(x_boot.shape)
    """
    if n_samples is None:
        n_samples = len(X)
    
    np.random.seed(random_state)
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    
    if isinstance(X, pd.DataFrame):
        X_bootstrap = X.iloc[indices]
    else:
        X_bootstrap = X[indices]
    
    if isinstance(y, pd.Series):
        y_bootstrap = y.iloc[indices]
    else:
        y_bootstrap = y[indices]
    
    return X_bootstrap, y_bootstrap


def learning_curve(model, X, y, train_sizes=None, cv=5, scoring=None):
    """
    обчислити криву навчання.
    
    args:
        model: модель для оцінки
        x: ознаки
        y: цільова змінна
        train_sizes: розміри навчальних вибірок
        cv: кількість блоків перехресної валідації
        scoring: метрика оцінки
        
    returns:
        tuple: (train_sizes, train_scores, val_scores)
        
    example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> x = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> sizes, train_scores, val_scores = learning_curve(model, x, y)
        >>> print(sizes)
    """
    from sklearn.model_selection import learning_curve
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring
    )
    
    return train_sizes, train_scores, val_scores


def validation_curve(model, X, y, param_name, param_range, cv=5, scoring=None):
    """
    обчислити криву валідації.
    
    args:
        model: модель для оцінки
        x: ознаки
        y: цільова змінна
        param_name: ім'я параметра для налаштування
        param_range: діапазон значень параметра
        cv: кількість блоків перехресної валідації
        scoring: метрика оцінки
        
    returns:
        tuple: (train_scores, val_scores)
        
    example:
        >>> from sklearn.svm import SVC
        >>> model = SVC()
        >>> x = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> param_range = [0.1, 1, 10]
        >>> train_scores, val_scores = validation_curve(model, x, y, 'c', param_range)
        >>> print(train_scores.shape)
    """
    from sklearn.model_selection import validation_curve
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring
    )
    
    return train_scores, val_scores


def plot_learning_curve(train_sizes, train_scores, val_scores, title='learning curve'):
    """
    побудувати графік кривої навчання.
    
    args:
        train_sizes: розміри навчальних вибірок
        train_scores: бали навчання
        val_scores: бали валідації
        title: заголовок графіка
        
    example:
        >>> train_sizes = np.array([1, 2, 3])
        >>> train_scores = np.array([[0.8, 0.9], [0.85, 0.95], [0.9, 0.98]])
        >>> val_scores = np.array([[0.7, 0.8], [0.75, 0.85], [0.8, 0.9]])
        >>> plot_learning_curve(train_sizes, train_scores, val_scores)
    """
    import matplotlib.pyplot as plt
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('training examples')
    plt.ylabel('score')
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='training score')
    plt.plot(train_sizes, val_mean, 'o-', color='orange', label='validation score')
    
    plt.legend(loc='best')
    plt.show()


def plot_validation_curve(param_range, train_scores, val_scores, param_name='parameter', title='validation curve'):
    """
    побудувати графік кривої валідації.
    
    args:
        param_range: діапазон значень параметра
        train_scores: бали навчання
        val_scores: бали валідації
        param_name: ім'я параметра
        title: заголовок графіка
        
    example:
        >>> param_range = np.array([0.1, 1, 10])
        >>> train_scores = np.array([[0.8, 0.9], [0.85, 0.95], [0.9, 0.98]])
        >>> val_scores = np.array([[0.7, 0.8], [0.75, 0.85], [0.8, 0.9]])
        >>> plot_validation_curve(param_range, train_scores, val_scores, 'c')
    """
    import matplotlib.pyplot as plt
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('score')
    
    plt.grid()
    
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    
    plt.plot(param_range, train_mean, 'o-', color='blue', label='training score')
    plt.plot(param_range, val_mean, 'o-', color='orange', label='validation score')
    
    plt.legend(loc='best')
    plt.show()