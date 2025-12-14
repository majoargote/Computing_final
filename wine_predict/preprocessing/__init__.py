
# Preprocessing module for wine quality prediction

from .base import BasePreprocessor
from .scalers import StandardScalerPreprocessor

__all__ = [
    'BasePreprocessor',
    'StandardScalerPreprocessor'
]
