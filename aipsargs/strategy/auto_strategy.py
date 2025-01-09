# aipsarg/strategy/auto_strategy.py
import logging
import pandas as pd
from typing import Optional
from data.data_utils import DataHandler
from model.model_utils import ModelManager
from strategy.base_strategy import BaseStrategy, TradingStyle
from utils.logger import setup_logger

logger = setup_logger(__name__)

class AutoStrategy:
    """
    A class to handle automatic trading strategy flow, from data to trading signal.
    """

    def __init__(self, trading_style: TradingStyle, strategy : BaseStrategy, data_handler : DataHandler, model_manager : ModelManager):
        """
        Initializes the AutoStrategy.
        
        Args:
            trading_style (TradingStyle): The trading style for the strategy.
            strategy (BaseStrategy) : Concrete strategy
            data_handler (DataHandler) : Data handler for fetching data
            model_manager (ModelManager) : Model manager for price prediction
        """
        self.trading_style = trading_style
        self.data_handler = data_handler
        self.model_manager = model_manager
        self.strategy = strategy

    def generate_trading_signal(self, df: pd.DataFrame, pair: str) -> Optional[str]:
        """Generates a trading signal based on data, indicators, and prediction."""
        try:
            # 1. Data Pasar dan Indikator
            df = self.data_handler.add_all_indicators(df, trading_style=self.trading_style.value)
            if df.empty:
                logger.warning("Cannot proceed trading, data is empty")
                return None

            columns_to_scale = ['close', 'volume','psar','ma_20','ema_20','macd', 'signal_line', 'macd_histogram','rsi', '%k', '%d',
                     'bollinger_upper', 'bollinger_lower', 'atr', 'adx', 'obv', 'pivot', 'r1', 's1', 'r2', 's2', 'price_sentiment']
            df = self.data_handler.standardize_data(df, columns_to_scale=columns_to_scale)

            data_for_model = self.data_handler.prepare_data_for_model(df)
            if data_for_model is None:
               logger.warning("Cannot proceed trading, data for model is not enough")
               return None

            # 2. Prediksi Model
            model, scaler = self.model_manager.load_trained_model()
            prediction = self.model_manager.predict_price(df, model, scaler)
            if prediction == -1:
                logger.warning("Cannot proceed trading, error during model prediction")
                return None

            # 3. Sinyal Trading (menggabungkan prediksi dan indikator)
            signal = self._generate_trading_signal_with_prediction(df, prediction)
            
            if signal is None:
                return None
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None

    def _generate_trading_signal_with_prediction(self, df, prediction):
         """
         Generates trading signal based on strategy, indicators and model prediction

         Args:
            df (pd.DataFrame): Input DataFrame with market data and indicators.
            prediction (float): Predicted price movement (0-1).

         Returns:
            Optional[str]: Trading signal ("buy" or "sell"), or None if no signal.
         """
         try:
            if prediction == -1:  # Jika prediksi error, gunakan strategi PSAR
                signal = 'buy' if df['psar'].iloc[-1] < df['close'].iloc[-1] else 'sell'
            else :
                signal = self.strategy.generate_signal(df=df)
            return signal
         except Exception as e:
            logger.error(f"Error generate trading signal with prediction: {e}")
            return None