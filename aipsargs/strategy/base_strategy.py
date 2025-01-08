# aipsarg/strategy/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from enum import Enum

class TradingStyle(Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    LONG_TERM = "long_term"

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, trading_style: TradingStyle):
        """
        Initializes the base strategy with a trading style.
        
        Args:
            trading_style (TradingStyle): The trading style for the strategy.
        """
        self.trading_style = trading_style
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Generates a trading signal based on the data.

        Args:
            df (pd.DataFrame): Input DataFrame with market data and indicators.

        Returns:
            Optional[str]: "buy" or "sell" signal, or None if no signal.
        """
        pass
    
    def get_trading_style(self) -> TradingStyle:
        """
        Returns the trading style of the strategy.
        
        Returns:
           TradingStyle: The trading style for the strategy.
        """
        return self.trading_style
