# aipsarg/api/okx_api.py
import time,hmac,base64,hashlib,json,requests,ccxt,logging
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_fixed
from configs.config import API_KEY, API_SECRET, PASSWORD
from api.base_api import BaseAPI

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_signature(timestamp: str, method: str, request_path: str, body: str) -> str:
    message = str(timestamp) + method + request_path + body
    mac = hmac.new(bytes(API_SECRET, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    return base64.b64encode(mac.digest()).decode('utf-8')

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def make_request(method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
    timestamp = str(int(time.time()))
    if params is None: params = {}
    if data is None: data = ''
    elif isinstance(data, dict): data = json.dumps(data)
    signature = generate_signature(timestamp, method.upper(), path, str(data))
    headers = {"Content-Type": "application/json", "OK-ACCESS-KEY": API_KEY, "OK-ACCESS-SIGN": signature, "OK-ACCESS-TIMESTAMP": timestamp, "OK-ACCESS-PASSPHRASE": PASSWORD}
    try:
        url = f"https://www.okx.com{path}"
        if method == "GET": response = requests.get(url, headers=headers, params=params)
        elif method == "POST": response = requests.post(url, headers=headers, data=data, params=params)
        else: raise ValueError(f"Invalid HTTP Method: {method}")
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e: logging.error(f"Request failed: {e}"); return {}
    except ValueError as e: logging.error(f"Value Error: {e}"); return {}

class OKXAPI(BaseAPI):
    """Concrete implementation of the OKX API."""

    def __init__(self):
         self.exchange = ccxt.okx({'apiKey': API_KEY,'secret': API_SECRET,'password': PASSWORD,'enableRateLimit': True}); self.exchange.load_markets()

    def make_request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
         try:
            return make_request(method, path, params, data)
         except Exception as e:
            logging.error(f"Error make request: {e}")
            return {}
    
    def fetch_candlesticks(self, pair: str, timeframe: str, limit: int) -> list:
         try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit); return candles
         except Exception as e: logging.error(f"Error fetching candlesticks: {e}"); raise
    
    def fetch_balance(self, pair:str) -> Dict:
        try:
            balance = self.exchange.fetch_balance()
            if balance and 'free' in balance:
                currency = "USDT"; asset = pair.split('/')[0]; free_currency = balance['free'].get(currency, 0.0); free_asset = balance['free'].get(asset, 0.0)
                return {"currency": {"name": currency,"available": free_currency},"asset": {"name": asset,"available": free_asset}}
            else: logging.error("Could not retrieve balance or balance is empty"); return {}
        except Exception as e: logging.error(f"Error fetching balance: {e}"); return {}