# aipsarg/main.py
import logging
import os
from dotenv import load_dotenv
from configs import Config
from trading_ai import TradingAI

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
    )

def load_configuration():
    load_dotenv()
    required_vars = ['API_KEY', 'SECRET_KEY', 'PASSPHRASE', 'INST_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required configuration variables: {', '.join(missing_vars)}")
    return Config(
        api_key=os.getenv('API_KEY'),
        secret_key=os.getenv('SECRET_KEY'),
        passphrase=os.getenv('PASSPHRASE'),
        inst_id=os.getenv('INST_ID')
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Jika tidak menggunakan GPU, tambahkan ini:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        config = load_configuration()
        trading_ai = TradingAI(
            api_key=config.api_key,
            secret_key=config.secret_key,
            passphrase=config.passphrase,
            inst_id=config.inst_id
        )
        logger.info("Bot trading berhasil dimulai.")
    except Exception as e:
        logger.error(f"Bot trading gagal dimulai: {e}")

if __name__ == "__main__":
    main()