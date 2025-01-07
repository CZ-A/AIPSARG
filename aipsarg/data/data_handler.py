# aipsarg/data/data_handler.py
import pandas as pd
import logging
import numpy as np
from api.api_utils import ExchangeAPI
from config.config import TRADING_CONFIG, INDICATOR_CONFIG, MODEL_CONFIG
from data.base_data import BaseData
from feature_engineering import moving_average, rsi, macd, psar
from transformers import DataTransformer

class DataHandler(BaseData):
    """A class to handle data fetching and processing."""
    def __init__(self):
        """Initializes the DataHandler with an ExchangeAPI instance."""
        self.api_handler = ExchangeAPI()
        self.data_transformer = DataTransformer()
    
    def fetch_okx_candlesticks(self, pair: str, timeframe: str = TRADING_CONFIG["TIMEFRAME"], limit: int = TRADING_CONFIG["LIMIT"]) -> pd.DataFrame:
        """
        Fetches candlestick data from OKX and returns a pandas DataFrame.

        Args:
            pair (str): The trading pair symbol (e.g., "CORE/USDT").
            timeframe (str): The timeframe for the candlesticks (e.g., "1h").
            limit (int): The number of candlesticks to fetch.

        Returns:
            pd.DataFrame: A DataFrame containing candlestick data.
        """
        try:
            candles = self.api_handler.fetch_candlesticks(pair, timeframe, limit)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info(f"Fetched {len(df)} candlesticks for {pair}.")
            return df
        except Exception as e:
            logging.error(f"Error fetching candlesticks: {e}")
            return pd.DataFrame()
    
    def add_all_indicators(self, df: pd.DataFrame, trading_style: str) -> pd.DataFrame:
        """Adds all technical indicators based on trading style."""
        
        if trading_style == 'scalping':
            ma_window = INDICATOR_CONFIG["SCALPING_MA_WINDOW"]
            ema_window = INDICATOR_CONFIG["SCALPING_EMA_WINDOW"]
            macd_short_window = INDICATOR_CONFIG["SCALPING_MACD_SHORT_WINDOW"]
            macd_long_window = INDICATOR_CONFIG["SCALPING_MACD_LONG_WINDOW"]
            rsi_window = INDICATOR_CONFIG["SCALPING_RSI_WINDOW"]
            stochastic_window = INDICATOR_CONFIG["SCALPING_STOCHASTIC_WINDOW"]
            bollinger_window = INDICATOR_CONFIG["SCALPING_BOLLINGER_WINDOW"]
            atr_window = INDICATOR_CONFIG["SCALPING_ATR_WINDOW"]
            adx_window = INDICATOR_CONFIG["SCALPING_ADX_WINDOW"]
            sentiment_window = INDICATOR_CONFIG["SCALPING_SENTIMENT_WINDOW"]
            
        elif trading_style == 'day_trading':
            ma_window = INDICATOR_CONFIG["DAY_TRADING_MA_WINDOW"]
            ema_window = INDICATOR_CONFIG["DAY_TRADING_EMA_WINDOW"]
            macd_short_window = INDICATOR_CONFIG["DAY_TRADING_MACD_SHORT_WINDOW"]
            macd_long_window = INDICATOR_CONFIG["DAY_TRADING_MACD_LONG_WINDOW"]
            rsi_window = INDICATOR_CONFIG["DAY_TRADING_RSI_WINDOW"]
            stochastic_window = INDICATOR_CONFIG["DAY_TRADING_STOCHASTIC_WINDOW"]
            bollinger_window = INDICATOR_CONFIG["DAY_TRADING_BOLLINGER_WINDOW"]
            atr_window = INDICATOR_CONFIG["DAY_TRADING_ATR_WINDOW"]
            adx_window = INDICATOR_CONFIG["DAY_TRADING_ADX_WINDOW"]
            sentiment_window = INDICATOR_CONFIG["DAY_TRADING_SENTIMENT_WINDOW"]
            
        elif trading_style == 'swing_trading':
            ma_window = INDICATOR_CONFIG["SWING_TRADING_MA_WINDOW"]
            ema_window = INDICATOR_CONFIG["SWING_TRADING_EMA_WINDOW"]
            macd_short_window = INDICATOR_CONFIG["SWING_TRADING_MACD_SHORT_WINDOW"]
            macd_long_window = INDICATOR_CONFIG["SWING_TRADING_MACD_LONG_WINDOW"]
            rsi_window = INDICATOR_CONFIG["SWING_TRADING_RSI_WINDOW"]
            stochastic_window = INDICATOR_CONFIG["SWING_TRADING_STOCHASTIC_WINDOW"]
            bollinger_window = INDICATOR_CONFIG["SWING_TRADING_BOLLINGER_WINDOW"]
            atr_window = INDICATOR_CONFIG["SWING_TRADING_ATR_WINDOW"]
            adx_window = INDICATOR_CONFIG["SWING_TRADING_ADX_WINDOW"]
            sentiment_window = INDICATOR_CONFIG["SWING_TRADING_SENTIMENT_WINDOW"]
            
        elif trading_style == 'long_term':
            ma_window = INDICATOR_CONFIG["LONG_TERM_MA_WINDOW"]
            ema_window = INDICATOR_CONFIG["LONG_TERM_EMA_WINDOW"]
            macd_short_window = INDICATOR_CONFIG["LONG_TERM_MACD_SHORT_WINDOW"]
            macd_long_window = INDICATOR_CONFIG["LONG_TERM_MACD_LONG_WINDOW"]
            rsi_window = INDICATOR_CONFIG["LONG_TERM_RSI_WINDOW"]
            stochastic_window = INDICATOR_CONFIG["LONG_TERM_STOCHASTIC_WINDOW"]
            bollinger_window = INDICATOR_CONFIG["LONG_TERM_BOLLINGER_WINDOW"]
            atr_window = INDICATOR_CONFIG["LONG_TERM_ATR_WINDOW"]
            adx_window = INDICATOR_CONFIG["LONG_TERM_ADX_WINDOW"]
            sentiment_window = INDICATOR_CONFIG["LONG_TERM_SENTIMENT_WINDOW"]
        else:
            logging.error("Invalid trading style.")
            return pd.DataFrame()
        try:
           df = psar.PSAR().calculate(df)
           df = moving_average.MovingAverage(window=ma_window).calculate(df)
           df = moving_average.MovingAverage(window=ema_window, column='close').calculate(df)
           df = macd.MACD(short_window=macd_short_window, long_window=macd_long_window).calculate(df)
           df = rsi.RSI(window=rsi_window).calculate(df)
           df = rsi.StochasticOscillator(window=stochastic_window).calculate(df)
           df = rsi.BollingerBands(window=bollinger_window).calculate(df)
           df = rsi.ATR(window=atr_window).calculate(df)
           df = rsi.ADX(window=adx_window).calculate(df)
           df = rsi.PriceSentiment(window=sentiment_window).calculate(df)
           return df
        except Exception as e:
            logging.error(f"Error calculating indicators {e}")
            return pd.DataFrame()

    def prepare_training_data(self, df: pd.DataFrame, scaler: object = None) -> tuple:
        """
        Prepares data for training the LSTM model.

        Args:
            df (pd.DataFrame): The input DataFrame.
            scaler (object): A MinMaxScaler object to scale the data.

        Returns:
            tuple: A tuple containing scaled features (X), target labels (y), and the scaler.
        """
        try:
            df = df[['open', 'high', 'low', 'close', 'volume', 'psar']].copy()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            if scaler is None:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df.drop('target', axis=1).values)
            else:
                scaled_data = scaler.transform(df.drop('target', axis=1).values)
            X, y = [], []
            sequence_length = MODEL_CONFIG["SEQUENCE_LENGTH"]
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:i+sequence_length])
                y.append(df['target'].iloc[i+sequence_length])
            X = np.array(X)
            y = np.array(y)
            return X, y, scaler
        except Exception as e:
            logging.error(f"Error prepare training data: {e}")
            return [],[],None

    def prepare_data_for_model(self, df: pd.DataFrame, sequence_length: int = MODEL_CONFIG["SEQUENCE_LENGTH"], scaler :object = None) -> pd.DataFrame :
         """
        Prepares data for model prediction.

        Args:
            df (pd.DataFrame): The input DataFrame.
            sequence_length (int): The sequence length for the model.
            scaler (object): A MinMaxScaler object to scale the data.

        Returns:
            np.ndarray: The prepared data for the model.
        """
         try:
            df_copy = df[['open', 'high', 'low', 'close', 'volume', 'psar']].copy()
            if len(df_copy) < sequence_length:
                logging.warning(f"Not enough data to prepare for model. Returning the last available data.")
                return None
            return df_copy.tail(sequence_length)
         except Exception as e:
             logging.error(f"Error preparing data for model: {e}")
             return None
