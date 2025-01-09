# trading_bot/data/market_data.py
import pandas as pd
import logging
from typing import Dict
from api.api_utils import ExchangeAPI
from configs.config import TRADING_CONFIG

class MarketData:
    """
    A class to handle fetching market data with different timeframes.
    """
    def __init__(self):
        """Initializes MarketData with an ExchangeAPI instance."""
        self.api_handler = ExchangeAPI()
    
    def fetch_market_data(self, pair: str, timeframes: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """
        Fetches market data for a given pair with multiple timeframes.
        
        Args:
            pair (str): The trading pair symbol (e.g., "CORE/USDT").
            timeframes (Dict[str, int]): A dictionary of timeframes and limit (e.g., {"1m": 100, "5m": 200}).
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing pandas DataFrames for each timeframe.
        """
        market_data = {}
        for timeframe, limit in timeframes.items():
            try:
                candles = self.api_handler.fetch_candlesticks(pair, timeframe, limit)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logging.info(f"Fetched {len(df)} candlesticks for {pair} with timeframe {timeframe}.")
                market_data[timeframe] = df
            except Exception as e:
                logging.error(f"Error fetching candlesticks for {pair} with timeframe {timeframe}: {e}")
                market_data[timeframe] = pd.DataFrame() # Return empty df on error

        return market_data