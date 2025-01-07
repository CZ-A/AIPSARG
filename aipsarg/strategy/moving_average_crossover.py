# aipsarg/strategy/moving_average_crossover.py
from aipsarg.strategy.base_strategy import BaseStrategy, TradingStyle
import pandas as pd
from typing import Optional
from aipsarg.data.feature_engineering.moving_average import MovingAverage


class MovingAverageCrossover(BaseStrategy):
     """Concrete implementation of a moving average crossover strategy."""

     def __init__(self, short_window: int, long_window: int, trading_style : TradingStyle):
          super().__init__(trading_style)
          self.short_window = short_window
          self.long_window = long_window
          self.ma_short = MovingAverage(window = short_window)
          self.ma_long = MovingAverage(window = long_window)

     def generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Generates a trading signal based on moving average crossover.
        
        Args:
            df (pd.DataFrame): Input DataFrame with market data.
        
        Returns:
            Optional[str]: "buy" or "sell" signal, or None if no signal.
        """
        df = self.ma_short.calculate(df)
        df = self.ma_long.calculate(df)

        if df[f'ma_{self.short_window}'].iloc[-1] > df[f'ma_{self.long_window}'].iloc[-1] and df[f'ma_{self.short_window}'].iloc[-2] <= df[f'ma_{self.long_window}'].iloc[-2]:
            return "buy"
        elif df[f'ma_{self.short_window}'].iloc[-1] < df[f'ma_{self.long_window}'].iloc[-1] and df[f'ma_{self.short_window}'].iloc[-2] >= df[f'ma_{self.long_window}'].iloc[-2]:
            return "sell"
        return None
