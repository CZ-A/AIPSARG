# aipsarg/main.py
from configs.config import PAIRS, TRADING_CONFIG, INDICATOR_CONFIG
from api.api_utils import ExchangeAPI
from data.data_utils import DataHandler
from model.model_utils import ModelManager
from ai.trading_ai import TradingAI
import logging
import time
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Initialize API, DataHandler, ModelManager, and TradingAI
exchange_api = ExchangeAPI()
data_handler = DataHandler()
model_manager = ModelManager()
trading_ai = TradingAI()

def main():
    try:
        logging.info("Starting trading bot...")
        while True:
            # Fetch market data
            df = data_handler.fetch_market_data(PAIRS)

            # Load model and scaler
            model, scaler = model_manager.load_model_and_scaler()

            # Execute trading strategy
            trading_ai.trading_strategy(df, PAIRS, model, scaler, trading_style='short_term')

            # Sleep for a defined interval before the next iteration
            time.sleep(TRADING_CONFIG['MONITOR_SLEEP_INTERVAL'])
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()