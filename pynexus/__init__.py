"""
PyNexus - Універсальна науково-аналітична бібліотека

Автор: Андрій Будильников
"""

__version__ = "1.0.0"
__author__ = "Андрій Будильников"

# Імпортування основних модулів
from .core import *
from .analysis import *
from .solver import *
from .visualization import *

# Імпортування спеціалізованих модулів
from .mathematics import *
from .physics import *
from .chemistry import *
from .biology import *
from .engineering import *
from .economics import *
from .finance import *
from .statistics import *
from .ml import *
from .numerics import *
from .data_processing import *
from .signal_processing import *
from .optimization import *
from .advanced_math import *
from .advanced_plots import *
from .geoscience import *
from .meteorology import *
from .oceanography import *
from .environmental import *
from .medicine import *
from .neuroscience import *
from .psychology import *
from .social_sciences import *
from .linguistics import *
from .interdisciplinary import *
from .materials import *
from .astronomy import *
from .computational_science import *
from .computational_mathematics import *
from .computational_physics import *
from .computational_chemistry import *
from .computational_biology import *
from .computational_engineering import *
from .computational_economics import *
from .computational_finance import *

# Імпортування утиліт
from .utils.cache import *
from .utils.lazy import *
from .utils.profile import *

# Створення зручних псевдонімів
px = globals()

__all__ = [
    # Основні модулі
    'core', 'analysis', 'solver', 'visualization',
    
    # Спеціалізовані модулі
    'mathematics', 'physics', 'chemistry', 'biology',
    'engineering', 'economics', 'finance', 'statistics',
    'ml', 'numerics', 'data_processing', 'signal_processing',
    'optimization', 'advanced_math', 'advanced_plots',
    'geoscience', 'meteorology', 'oceanography', 'environmental',
    'medicine', 'neuroscience', 'psychology', 'social_sciences',
    'linguistics', 'interdisciplinary', 'materials', 'astronomy',
    'computational_science', 'computational_mathematics',
    'computational_physics', 'computational_chemistry',
    'computational_biology', 'computational_engineering',
    'computational_economics', 'computational_finance',
    
    # Утиліти
    'utils'
]