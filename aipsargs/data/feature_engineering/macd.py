# aipsarg/data/feature_engineering/macd.py
import pandas as pd
from data.feature_engineering.base_indicator import BaseIndicator

class MACD(BaseIndicator):
    """Concrete implementation of Moving Average Convergence Divergence (MACD)."""

    def __init__(self, short_window: int, long_window: int, signal_window: int = 9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Moving Average Convergence Divergence (MACD).

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            short_window (int): The short window size for the EMA.
            long_window (int): The long window size for the EMA.
            signal_window (int): The window size for the signal line EMA.

        Returns:
            pd.DataFrame: The DataFrame with added 'macd', 'signal_line', and 'macd_histogram' columns.
        """
        ema_short = df['close'].ewm(span=self.short_window, adjust=False).mean()
        ema_long = df['close'].ewm(span=self.long_window, adjust=False).mean()
        df['macd'] = ema_short - ema_long
        df['signal_line'] = df['macd'].ewm(span=self.signal_window, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['signal_line']
        return df