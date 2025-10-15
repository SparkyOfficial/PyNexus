"""
lazy loading utilities for PyNexus.

цей модуль надає функції для лінивого завантаження модулів.
автор: Андрій Будильников
"""

import importlib
import sys
from typing import Any, Optional


class LazyLoader:
    """
    ліниве завантаження модулів.
    """
    
    def __init__(self, module_name: str):
        """
        ініціалізація лінивого завантажувача.
        
        args:
            module_name: ім'я модуля для лінивого завантаження
        """
        self.module_name = module_name
        self.module = None
    
    def _load_module(self) -> Any:
        """
        завантажити модуль.
        
        returns:
            any: завантажений модуль
        """
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return self.module
    
    def __getattr__(self, name: str) -> Any:
        """
        отримати атрибут з модуля.
        
        args:
            name: ім'я атрибута
            
        returns:
            any: атрибут модуля
        """
        module = self._load_module()
        return getattr(module, name)


def lazy_import(module_name: str) -> LazyLoader:
    """
    створити лінивий імпортер для модуля.
    
    args:
        module_name: ім'я модуля
        
    returns:
        lazyloader: об'єкт для лінивого завантаження
    """
    return LazyLoader(module_name)


# lazy loaders for heavy modules
numpy_loader = LazyLoader('numpy')
pandas_loader = LazyLoader('pandas')
matplotlib_loader = LazyLoader('matplotlib')
scipy_loader = LazyLoader('scipy')
sympy_loader = LazyLoader('sympy')


def get_numpy():
    """отримати numpy з лінивим завантаженням."""
    return numpy_loader._load_module()


def get_pandas():
    """отримати pandas з лінивим завантаженням."""
    return pandas_loader._load_module()


def get_matplotlib():
    """отримати matplotlib з лінивим завантаженням."""
    return matplotlib_loader._load_module()


def get_scipy():
    """отримати scipy з лінивим завантаженням."""
    return scipy_loader._load_module()


def get_sympy():
    """отримати sympy з лінивим завантаженням."""
    return sympy_loader._load_module()