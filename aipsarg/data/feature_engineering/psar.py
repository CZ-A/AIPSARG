# aipsarg/data/feature_engineering/psar.py
import pandas as pd
from data.feature_engineering.base_indicator import BaseIndicator


class PSAR(BaseIndicator):
    """Concrete implementation of Parabolic SAR."""
    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, max_af: float = 0.2):
      self.af_start = af_start
      self.af_increment = af_increment
      self.max_af = max_af
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Parabolic SAR for the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low' columns.
            af_start (float): Acceleration factor starting value.
            af_increment (float): Acceleration factor increment.
            max_af (float): Maximum acceleration factor.

        Returns:
            pd.DataFrame: The DataFrame with an added 'psar' column.
        """
        if df is None or df.empty:
                raise ValueError("Empty dataframe passed to PSAR calculation.")
        psar = pd.Series(index=df.index, dtype='float64')
        af = self.af_start
        trend = 1  # 1 for uptrend, -1 for downtrend
        high = df['high'].iloc[0]
        low = df['low'].iloc[0]
        extreme = high
        psar.iloc[0] = low
        for i in range(1, len(df)):
            if trend == 1:  # Uptrend
                psar.iloc[i] = psar.iloc[i-1] + af * (extreme - psar.iloc[i-1])
                if df['high'].iloc[i] > high:
                    high = df['high'].iloc[i]
                    extreme = df['high'].iloc[i]
                    af = min(af + self.af_increment, self.max_af)
                if df['low'].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = high
                    extreme = df['low'].iloc[i]
                    af = self.af_start
            else:  # Downtrend
                psar.iloc[i] = psar.iloc[i-1] + af * (extreme - psar.iloc[i-1])
                if df['low'].iloc[i] < low:
                    low = df['low'].iloc[i]
                    extreme = df['low'].iloc[i]
                    af = min(af + self.af_increment, self.max_af)
                if df['high'].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = low
                    extreme = df['high'].iloc[i]
                    af = self.af_start
        df['psar'] = psar
        return df
