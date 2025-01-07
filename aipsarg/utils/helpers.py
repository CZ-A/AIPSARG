# aipsarg/utils/helpers.py
from datetime import datetime

def is_market_open(market_open_hour=0, market_close_hour=23):
    """
    Check if the market is open based on the current hour.
    """
    now = datetime.now()
    current_hour = now.hour
    return market_open_hour <= current_hour < market_close_hour
