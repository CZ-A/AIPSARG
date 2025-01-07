# AIPSARG/api/mock_api.py
from aisarg.api.base_api import BaseAPI
from typing import Dict, Optional

class MockAPI(BaseAPI):
    """Mock implementation of the API for testing."""

    def make_request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
       return {"mocked": True, "method": method, "path": path}

    def fetch_candlesticks(self, pair: str, timeframe: str, limit: int) -> list:
         return []

    def fetch_balance(self, pair:str) -> Dict:
         return {}
