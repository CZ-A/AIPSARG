# aipsarg/main.py
import logging
from ai.trading_ai import TradingAI
from configs.app_config import AppConfig
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


if __name__ == "__main__":
    try:
        config = AppConfig()
        if config.validate_config():
            ai = TradingAI()
            ai.run()
        else:
            logger.error("Invalid configuration, cannot start bot")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
