# AIPSARG/data/base_data.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseData(ABC):
    """Abstract base class for all data handling."""

    @abstractmethod
    def fetch_okx_candlesticks(self, pair: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetches candlestick data from OKX and returns a pandas DataFrame.
         
         Args:
            pair (str): The trading pair symbol (e.g., "CORE/USDT").
            timeframe (str): The timeframe for the candlesticks (e.g., "1h").
            limit (int): The number of candlesticks to fetch.

         Returns:
            pd.DataFrame: A DataFrame containing candlestick data.
        """
        pass

    @abstractmethod
    def add_all_indicators(self, df: pd.DataFrame, trading_style: str) -> pd.DataFrame:
        """Adds all technical indicators based on trading style.

        Args:
            df (pd.DataFrame): The input DataFrame.
            trading_style (str): type of trading style ('scalping', 'day_trading', 'swing_trading', 'long_term')
        
        Returns:
            pd.DataFrame: The DataFrame with all the required technical indicator
        """
        pass
    
    @abstractmethod
    def prepare_training_data(self, df: pd.DataFrame, scaler: object = None) -> tuple:
        """
        Prepares data for training the LSTM model.

        Args:
            df (pd.DataFrame): The input DataFrame.
            scaler (object): A MinMaxScaler object to scale the data.

        Returns:
            tuple: A tuple containing scaled features (X), target labels (y), and the scaler.
        """
        pass
    
    @abstractmethod
    def prepare_data_for_model(self, df: pd.DataFrame, sequence_length: int, scaler :object = None) -> pd.DataFrame :
        """
        Prepares data for model prediction.

        Args:
            df (pd.DataFrame): The input DataFrame.
            sequence_length (int): The sequence length for the model.
            scaler (object): A MinMaxScaler object to scale the data.

        Returns:
            np.ndarray: The prepared data for the model.
        """
        pass
