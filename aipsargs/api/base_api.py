# aipsarg/api/base_api.py
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseAPI(ABC):
    """Abstract base class for all API interactions."""

    @abstractmethod
    def make_request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Makes a request to the exchange API.

        Args:
            method (str): HTTP method (GET, POST, etc.).
            path (str): API endpoint path.
            params (Optional[Dict]): Query parameters.
            data (Optional[Dict]): Request body data.

        Returns:
            Dict: JSON response from the API.
        """
        pass

    @abstractmethod
    def fetch_candlesticks(self, pair: str, timeframe: str, limit: int) -> list:
        """Fetches candlestick data from the exchange.
           
        Args:
            pair (str) : trading pair (ex: BTC/USDT)
            timeframe (str) : timeframe
            limit (int) : limit candlestick data

        Returns:
           list: list candlestick data
        """
        pass

    @abstractmethod
    def fetch_balance(self, pair:str) -> Dict:
        """
        Fetches the account balance from OKX and structures it.
        
        Returns:
            Dict: A dictionary containing the available balance, formatted as 'currency' and 'asset'.
        """
        pass
