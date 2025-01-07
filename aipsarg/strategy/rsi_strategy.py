# aipsarg/strategy/rsi_strategy.py
from strategy.base_strategy import BaseStrategy, TradingStyle
import pandas as pd
from typing import Optional
from data.feature_engineering.rsi import RSI

class RSIStrategy(BaseStrategy):
    """Concrete implementation of RSI Strategy"""
    def __init__(self, rsi_window: int, overbought: int = 70, oversold: int = 30, trading_style: TradingStyle = TradingStyle.DAY_TRADING):
        super().__init__(trading_style)
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold
        self.rsi = RSI(window = rsi_window)

    def generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Generates a trading signal based on RSI.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.

        Returns:
            Optional[str]: "buy" or "sell" signal, or None if no signal.
        """
        df = self.rsi.calculate(df)
        if df['rsi'].iloc[-1] > self.overbought:
            return "sell"
        elif df['rsi'].iloc[-1] < self.oversold:
           return "buy"
        else:
           return None
