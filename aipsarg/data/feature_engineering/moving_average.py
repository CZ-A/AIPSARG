# aipsarg/data/feature_engineering/moving_average.py
import pandas as pd
from aipsarg.data.feature_engineering.base_indicator import BaseIndicator
from typing import List
class MovingAverage(BaseIndicator):
    """Concrete implementation of Moving Average indicator."""

    def __init__(self, window:int, column:str ='close') -> None:
         self.window = window
         self.column = column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
         df[f'ma_{self.window}'] = df[self.column].rolling(window = self.window).mean()
         return df
