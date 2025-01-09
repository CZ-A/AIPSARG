# aipsarg/data/feature_engineering/rsi.py
import pandas as pd
import numpy as np
from data.feature_engineering.base_indicator import BaseIndicator

class RSI(BaseIndicator):
    """Concrete implementation of Relative Strength Index (RSI)."""
    def __init__(self, window: int):
         self.window = window

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI).

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for the RSI.

        Returns:
            pd.DataFrame: The DataFrame with an added 'rsi' column.
         """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

class StochasticOscillator(BaseIndicator):
     """Concrete implementation of Stochastic Oscillator."""
     def __init__(self, window: int) -> None:
          self.window = window
     
     def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
          """
          Calculates the Stochastic Oscillator.

          Args:
             df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            window (int): The window size for the Stochastic Oscillator.

          Returns:
            pd.DataFrame: The DataFrame with added '%k' and '%d' columns.
          """
          lowest_low = df['low'].rolling(window=self.window).min()
          highest_high = df['high'].rolling(window=self.window).max()
          df['%k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
          df['%d'] = df['%k'].rolling(window=3).mean()  # 3-period moving average
          return df

class BollingerBands(BaseIndicator):
    """Concrete implementation of Bollinger Bands."""
    def __init__(self, window: int, num_std_dev: int = 2) -> None:
        self.window = window
        self.num_std_dev = num_std_dev

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Bollinger Bands.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for the moving average.
            num_std_dev (int): The number of standard deviations for the bands.

        Returns:
             pd.DataFrame: The DataFrame with added 'bollinger_upper' and 'bollinger_lower' columns.
        """
        ma = df['close'].rolling(window=self.window).mean()
        std_dev = df['close'].rolling(window=self.window).std()
        df['bollinger_upper'] = ma + (std_dev * self.num_std_dev)
        df['bollinger_lower'] = ma - (std_dev * self.num_std_dev)
        return df

class ATR(BaseIndicator):
    """Concrete implementation of Average True Range (ATR)."""
    def __init__(self, window: int) -> None:
       self.window = window

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Average True Range (ATR).

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            window (int): The window size for the ATR.

        Returns:
            pd.DataFrame: The DataFrame with added 'atr' column.
        """
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.window).mean()
        return df
    
class ADX(BaseIndicator):
    """Concrete implementation of Average Directional Index (ADX)."""
    def __init__(self, window: int) -> None:
       self.window = window

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Average Directional Index (ADX).

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            window (int): The window size for the ADX.

        Returns:
           pd.DataFrame: The DataFrame with added 'adx' column.
        """
        df['plus_dm'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
        df['minus_dm'] = (df['low'].shift(1) - df['low']).clip(lower=0)
        df['true_range'] = ATR(self.window).calculate(df)['true_range']
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=self.window).mean() / df['true_range'].rolling(window=self.window).mean())
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=self.window).mean() / df['true_range'].rolling(window=self.window).mean())
        df['dx'] = 100 * np.abs((df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        df['adx'] = df['dx'].rolling(window=self.window).mean()
        return df

class PriceSentiment(BaseIndicator):
    """Concrete implementation of Price Sentiment."""
    def __init__(self, window: int) -> None:
        self.window = window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates simple price-based sentiment.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for calculating the sentiment.
        
        Returns:
            pd.DataFrame: The DataFrame with added 'price_sentiment' column.
        """
        df['price_change'] = df['close'].diff()
        df['price_sentiment'] = df['price_change'].rolling(window=self.window).mean()
        return df