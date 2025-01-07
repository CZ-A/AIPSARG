# aipsarg/data/feature_engineering/base_indicator.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseIndicator(ABC):
    """Abstract base class for all technical indicators."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the indicator.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.

        Returns:
            pd.DataFrame: DataFrame with added indicator column(s).
        """
        pass
