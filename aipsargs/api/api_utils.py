# aipsarg/api/api_utils.py
import time
import hmac
import base64
import hashlib
import json
import requests
import ccxt
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Dict, Optional

from configs.config import API_KEY, API_SECRET, PASSWORD

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_signature(timestamp: str, method: str, request_path: str, body: str) -> str:
    """
    Generates a signature for API requests.
    """
    message = str(timestamp) + method + request_path + body
    mac = hmac.new(bytes(API_SECRET, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    return base64.b64encode(mac.digest()).decode('utf-8')

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def make_request(method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
    """
    Makes a request to the OKX API.
    """
    timestamp = str(int(time.time()))
    if params is None:
        params = {}
    if data is None:
        data = ''
    elif isinstance(data, dict):
        data = json.dumps(data)
    
    signature = generate_signature(timestamp, method.upper(), path, str(data))
    headers = {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSWORD,
    }
    
    try:
        url = f"https://www.okx.com{path}"
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(url, headers=headers, data=data, params=params)
        else:
            raise ValueError(f"Invalid HTTP Method: {method}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {}
    except ValueError as e:
         logging.error(f"Value Error: {e}")
         return {}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_instrument_info(pair: str) -> Optional[Dict]:
    """
    Fetches instrument information from OKX for a specific trading pair.

    Args:
        pair (str): The trading pair symbol (e.g., 'BTC/USDT').

    Returns:
        Optional[Dict]: A dictionary containing instrument info, including minimum order size, or None if the request fails.
    """
    try:
        path = "/api/v5/public/instruments"
        params = {"instType": "SPOT", "instId": pair}
        response = make_request("GET", path, params=params)

        if response and response.get("data"):
            instrument_data = response["data"][0]
            min_size = instrument_data.get("minSz")
            return {"min_size": min_size}
        else:
            logging.error(f"Failed to fetch instrument info for {pair}.")
            return None
    except Exception as e:
        logging.error(f"Error fetching instrument info: {e}")
        return None

class ExchangeAPI:
    """A class to encapsulate interactions with the exchange API."""

    def __init__(self):
        """Initializes the ExchangeAPI with ccxt object and load market data."""
        self.exchange = ccxt.okx({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'password': PASSWORD,
            'enableRateLimit': True
        })
        self.exchange.load_markets()  # Load market data during initialization
    
    def get_exchange(self) -> ccxt.Exchange:
        """Returns the ccxt exchange object."""
        return self.exchange
    
    def place_order(self, pair: str, side: str, amount: float) -> Optional[Dict]:
        """
        Places an order on the exchange.
        """
        try:
            order = self.exchange.create_order(pair, 'market', side, amount)
            logging.info(f"Order placed: {side} {amount} of {pair} at price {order['price']}")
            return order
        except ccxt.ExchangeError as e:
            logging.error(f"Error placing order: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error placing order: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def fetch_candlesticks(self, pair: str, timeframe: str, limit: int) -> list:
        """
        Fetches candlestick data from the exchange.
        """
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            return candles
        except Exception as e:
            logging.error(f"Error fetching candlesticks: {e}")
            raise
    
    def fetch_balance(self, pair:str) -> Dict:
        """
        Fetches the account balance from OKX and structures it.
        
        Returns:
            Dict: A dictionary containing the available balance, formatted as 'currency' and 'asset'.
        """
        try:
            balance = self.exchange.fetch_balance()
            if balance and 'free' in balance:
                currency = "USDT"  # Assuming USDT is always the quote currency
                asset = pair.split('/')[0]
                free_currency = balance['free'].get(currency, 0.0)
                free_asset = balance['free'].get(asset, 0.0)

                return {
                    "currency": {
                        "name": currency,
                        "available": free_currency
                    },
                     "asset": {
                        "name": asset,
                        "available": free_asset
                    }
                }
            else:
                logging.error("Could not retrieve balance or balance is empty")
                return {}
        except Exception as e:
            logging.error(f"Error fetching balance: {e}")
            return {}
