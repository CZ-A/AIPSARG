# AIPSARG/config/validation.py
import logging
from trading_bot.strategy.base_strategy import TradingStyle

def validate_config(config: dict) -> bool:
    """Validates if essential configuration variables are set."""
    if not all([config.get("API_KEY"), config.get("API_SECRET"), config.get("PASSWORD"), config.get("PAIRS")]):
        logging.error("Essential API or trading configuration variables are missing.")
        return False
    if not all(config.get("TRADING_CONFIG").values()):
        logging.error("Essential trading configuration variables are missing.")
        return False

    # Validate trading style
    valid_trading_styles = [style.value for style in TradingStyle]
    if config.get("TRADING_CONFIG").get("TRADING_STYLE") not in valid_trading_styles:
        logging.error(f"Invalid TRADING_STYLE: {config.get('TRADING_CONFIG').get('TRADING_STYLE')}. Must be one of {valid_trading_styles}")
        return False
    return True
