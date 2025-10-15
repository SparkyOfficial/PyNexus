# PyNexus - Універсальна науково-аналітична бібліотека
# Інтелектуальна система для комплексного наукового аналізу

"""
PyNexus - це універсальна науково-аналітична бібліотека, розроблена для об'єднання 
можливостей різних наукових бібліотек Python в єдиний інтуїтивний інтерфейс. 
Бібліотека забезпечує доступ до широкого спектру функцій для обчислень, аналізу даних, 
візуалізації, машинного навчання та багатьох інших наукових задач.
"""

__version__ = "0.2.0"
__author__ = "PyNexus Development Team"
__email__ = "pynexus.dev@example.com"
__license__ = "MIT"
__description__ = "Універсальна науково-аналітична бібліотека"

# Імпортуємо основні модулі
from .core import *
from .analysis import *
from .solver import *
from .visualization import *
from .ml import *
from .numerics import *
from .signal_processing import *
from .data_processing import *
from .statistics import *
from .advanced_plots import *
from .advanced_math import *
from .optimization import *
from .physics import *
from .chemistry import *
from .biology import *
from .finance import *
from .engineering import *
from .geoscience import *
from .meteorology import *
from .oceanography import *
from .astronomy import *
from .materials import *
from .medicine import *
from .environmental import *
from .neuroscience import *
from .mathematics import *
from .linguistics import *
from .social_sciences import *
from .economics import *
from .psychology import *
from .interdisciplinary import *
from .computational_science import *

# Імпортуємо CLI інтерфейс
from .cli import main as cli_main

# Імпортуємо утиліти
from .utils import *

# Визначаємо публічний API
__all__ = [
    # Основні модулі
    'core',
    'analysis', 
    'solver',
    'visualization',
    'ml',
    'numerics',
    'signal_processing',
    'data_processing',
    'statistics',
    'advanced_plots',
    'advanced_math',
    'optimization',
    'physics',
    'chemistry',
    'biology',
    'finance',
    'engineering',
    'geoscience',
    'meteorology',
    'oceanography',
    'astronomy',
    'materials',
    'medicine',
    'environmental',
    'neuroscience',
    'mathematics',
    'linguistics',
    'social_sciences',
    'economics',
    'psychology',
    'interdisciplinary',
    'computational_science',
    
    # CLI
    'cli_main',
    
    # Утиліти
    'utils'
]

# Налаштування для імпорту
import os
import sys

# Встановлюємо змінні середовища для роботи з бібліотекою
os.environ['PYNEXUS_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['PYNEXUS_VERSION'] = __version__

# Ініціалізація бібліотеки
def init():
    """
    Ініціалізація PyNexus бібліотеки.
    """
    print(f"PyNexus v{__version__} ініціалізовано успішно!")
    print("Готовий до наукових обчислень та аналізу даних.")

# Автоматична ініціалізація при імпорті
init()