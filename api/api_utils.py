import time
import hmac
import base64
import hashlib
import json
import requests
import ccxt
import logging
from config.config import API_KEY, API_SECRET, PASSWORD
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Dict, Optional


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_signature(timestamp: str, method: str, request_path: str, body: str) -> str:
    """
    Generates a signature for API requests.

    Args:
        timestamp (str): The timestamp of the request.
        method (str): The HTTP method of the request (e.g., 'GET', 'POST').
        request_path (str): The API endpoint path.
        body (str): The request body.

    Returns:
        str: The generated signature.
    
    Raises:
         Exception: If an error occurs during signature generation.
    """
    message = str(timestamp) + method + request_path + body
    mac = hmac.new(bytes(API_SECRET, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    return base64.b64encode(mac.digest()).decode('utf-8')

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def make_request(method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
    """
    Makes a request to the OKX API.

    Args:
        method (str): The HTTP method of the request (e.g., 'GET', 'POST').
        path (str): The API endpoint path.
        params (Optional[Dict]): The query parameters for the request.
        data (Optional[Dict]): The request body data.

    Returns:
        Dict: The JSON response from the API, or an empty dictionary if the request fails.
    
    Raises:
        ValueError: If an invalid HTTP method is provided.
        requests.exceptions.RequestException: If an error occurs during the request.
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
        """Returns the ccxt exchange object.

        Returns:
            ccxt.Exchange: The ccxt exchange object.
        """
        return self.exchange
    
    def place_order(self, pair: str, side: str, amount: float) -> Optional[Dict]:
        """
        Places an order on the exchange.

        Args:
            pair (str): The trading pair symbol (e.g., 'BTC/USDT').
            side (str): The side of the order ('buy' or 'sell').
            amount (float): The amount of the asset to order.

        Returns:
            Optional[Dict]: The order details from the API if successful, otherwise None.
        
        Raises:
            ccxt.ExchangeError: If an error occurs during order placing.
            Exception: If an unexpected error occurs during order placing.
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

        Args:
            pair (str): The trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): The timeframe for the candlesticks (e.g., '1h').
            limit (int): The number of candlesticks to fetch.

        Returns:
            list: A list of candlestick data from the API.

        Raises:
            Exception: If an error occurs during data fetching.
        """
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            return candles
        except Exception as e:
            logging.error(f"Error fetching candlesticks: {e}")
            raise
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    # Example usage of ExchangeAPI
    api = ExchangeAPI()
    print(api.exchange.markets)
    print(api.exchange.symbols)
    # Sample data for make_request
    sample_path = "/api/v5/market/tickers"
    sample_params = {"instType": "SPOT"}
    sample_response = make_request("GET",sample_path, sample_params)
    if sample_response:
        logging.info(f"Successfully fetched ticker data. Response: {sample_response}")
    else:
        logging.error("Failed to fetch ticker data")
