import pandas as pd
import logging
import numpy as np
from api.api_utils import ExchangeAPI
from configs.config import TRADING_CONFIG, INDICATOR_CONFIG, MODEL_CONFIG
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    """A class to handle data fetching and processing."""
    def __init__(self):
        """Initializes the DataHandler with an ExchangeAPI instance."""
        self.api_handler = ExchangeAPI()
    
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
            raise
    
    def calculate_psar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, max_af: float = 0.2) -> pd.DataFrame:
        """
        Calculates Parabolic SAR for the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low' columns.
            af_start (float): Acceleration factor starting value.
            af_increment (float): Acceleration factor increment.
            max_af (float): Maximum acceleration factor.

        Returns:
            pd.DataFrame: The DataFrame with an added 'psar' column.
        """
        try:
            logging.info("Starting PSAR calculation.")
            if df is None or df.empty:
                raise ValueError("Empty dataframe passed to PSAR calculation.")
            psar = pd.Series(index=df.index, dtype='float64')
            af = af_start
            trend = 1  # 1 for uptrend, -1 for downtrend
            high = df['high'].iloc[0]
            low = df['low'].iloc[0]
            extreme = high
            psar.iloc[0] = low
            for i in range(1, len(df)):
                if trend == 1:  # Uptrend
                    psar.iloc[i] = psar.iloc[i-1] + af * (extreme - psar.iloc[i-1])
                    if df['high'].iloc[i] > high:
                        high = df['high'].iloc[i]
                        extreme = df['high'].iloc[i]
                        af = min(af + af_increment, max_af)
                    if df['low'].iloc[i] < psar.iloc[i]:
                        trend = -1
                        psar.iloc[i] = high
                        extreme = df['low'].iloc[i]
                        af = af_start
                else:  # Downtrend
                    psar.iloc[i] = psar.iloc[i-1] + af * (extreme - psar.iloc[i-1])
                    if df['low'].iloc[i] < low:
                        low = df['low'].iloc[i]
                        extreme = df['low'].iloc[i]
                        af = min(af + af_increment, max_af)
                    if df['high'].iloc[i] > psar.iloc[i]:
                        trend = 1
                        psar.iloc[i] = low
                        extreme = df['high'].iloc[i]
                        af = af_start
            df['psar'] = psar
            logging.info("PSAR calculated successfully.")
            return df
        except Exception as e:
            logging.error(f"Error in PSAR calculation: {e}")
            return df
        
    def calculate_moving_average(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculates the Moving Average for the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for the moving average.

        Returns:
            pd.DataFrame: The DataFrame with an added 'ma_{window}' column.
        """
        try:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            return df
        except Exception as e:
            logging.error(f"Error calculating moving average: {e}")
            return df
    
    def calculate_exponential_moving_average(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculates the Exponential Moving Average for the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for the EMA.

        Returns:
            pd.DataFrame: The DataFrame with an added 'ema_{window}' column.
        """
        try:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            return df
        except Exception as e:
            logging.error(f"Error calculating exponential moving average: {e}")
            return df
    
    def calculate_macd(self, df: pd.DataFrame, short_window: int, long_window: int, signal_window: int = 9) -> pd.DataFrame:
        """
        Calculates the Moving Average Convergence Divergence (MACD).

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            short_window (int): The short window size for the EMA.
            long_window (int): The long window size for the EMA.
            signal_window (int): The window size for the signal line EMA.

        Returns:
            pd.DataFrame: The DataFrame with added 'macd', 'signal_line', and 'macd_histogram' columns.
        """
        try:
            ema_short = df['close'].ewm(span=short_window, adjust=False).mean()
            ema_long = df['close'].ewm(span=long_window, adjust=False).mean()
            df['macd'] = ema_short - ema_long
            df['signal_line'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['signal_line']
            return df
        except Exception as e:
             logging.error(f"Error calculating MACD: {e}")
             return df
    
    def calculate_rsi(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
         """
         Calculates the Relative Strength Index (RSI).

         Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for the RSI.

         Returns:
            pd.DataFrame: The DataFrame with an added 'rsi' column.
         """
         try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            return df
         except Exception as e:
              logging.error(f"Error calculating RSI: {e}")
              return df

    def calculate_stochastic_oscillator(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculates the Stochastic Oscillator.

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            window (int): The window size for the Stochastic Oscillator.

        Returns:
            pd.DataFrame: The DataFrame with added '%k' and '%d' columns.
        """
        try:
            lowest_low = df['low'].rolling(window=window).min()
            highest_high = df['high'].rolling(window=window).max()
            df['%k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            df['%d'] = df['%k'].rolling(window=3).mean()  # 3-period moving average
            return df
        except Exception as e:
             logging.error(f"Error calculating stochastic oscillator: {e}")
             return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int, num_std_dev: int = 2) -> pd.DataFrame:
        """
        Calculates the Bollinger Bands.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'close' column.
            window (int): The window size for the moving average.
            num_std_dev (int): The number of standard deviations for the bands.

        Returns:
             pd.DataFrame: The DataFrame with added 'bollinger_upper' and 'bollinger_lower' columns.
        """
        try:
            ma = df['close'].rolling(window=window).mean()
            std_dev = df['close'].rolling(window=window).std()
            df['bollinger_upper'] = ma + (std_dev * num_std_dev)
            df['bollinger_lower'] = ma - (std_dev * num_std_dev)
            return df
        except Exception as e:
            logging.error(f"Error calculating bollinger bands: {e}")
            return df

    def calculate_atr(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculates the Average True Range (ATR).

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            window (int): The window size for the ATR.

        Returns:
            pd.DataFrame: The DataFrame with added 'atr' column.
        """
        try:
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift())
            df['low_close'] = np.abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=window).mean()
            return df
        except Exception as e:
            logging.error(f"Error calculating Average True Range: {e}")
            return df
    
    def calculate_adx(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Calculates the Average Directional Index (ADX).

        Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            window (int): The window size for the ADX.

        Returns:
           pd.DataFrame: The DataFrame with added 'adx' column.
        """
        try:
            df['plus_dm'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
            df['minus_dm'] = (df['low'].shift(1) - df['low']).clip(lower=0)
            df['true_range'] = self.calculate_atr(df, window)['true_range']
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=window).mean() / df['true_range'].rolling(window=window).mean())
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=window).mean() / df['true_range'].rolling(window=window).mean())
            df['dx'] = 100 * np.abs((df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
            df['adx'] = df['dx'].rolling(window=window).mean()
            return df
        except Exception as e:
              logging.error(f"Error calculating ADX: {e}")
              return df

    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the On Balance Volume (OBV).

        Args:
            df (pd.DataFrame): The input DataFrame with 'close' and 'volume' columns.

        Returns:
            pd.DataFrame: The DataFrame with an added 'obv' column.
        """
        try:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            return df
        except Exception as e:
              logging.error(f"Error calculating OBV: {e}")
              return df
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
         """
         Calculates Pivot Points.

         Args:
            df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.

         Returns:
             pd.DataFrame: The DataFrame with added 'pivot', 'r1', 's1', 'r2', 's2' columns.
         """
         try:
              df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
              df['r1'] = (2 * df['pivot']) - df['low'].shift(1)
              df['s1'] = (2 * df['pivot']) - df['high'].shift(1)
              df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
              df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
              return df
         except Exception as e:
             logging.error(f"Error calculating pivot points: {e}")
             return df
    
    def calculate_price_sentiment(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
          """
          Calculates simple price-based sentiment.

          Args:
              df (pd.DataFrame): The input DataFrame with a 'close' column.
              window (int): The window size for calculating the sentiment.
          
          Returns:
            pd.DataFrame: The DataFrame with added 'price_sentiment' column.
          """
          try:
                df['price_change'] = df['close'].diff()
                df['price_sentiment'] = df['price_change'].rolling(window=window).mean()
                return df
          except Exception as e:
              logging.error(f"Error calculating price sentiment: {e}")
              return df
    
    def standardize_data(self, df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
        """
        Standardize specific columns of the DataFrame using MinMaxScaler.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_to_scale (list): A list of column names to scale.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns scaled.
        """
        try:
            scaler = MinMaxScaler()
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            return df
        except Exception as e:
            logging.error(f"Error standardizing data: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame, scaler: object = None) -> tuple:
        """
        Prepares data for training the LSTM model.

        Args:
            df (pd.DataFrame): The input DataFrame.
            scaler (object): A MinMaxScaler object to scale the data.

        Returns:
            tuple: A tuple containing scaled features (X), target labels (y), and the scaler.
        """
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

    def prepare_data_for_model(self, df: pd.DataFrame, sequence_length: int = MODEL_CONFIG["SEQUENCE_LENGTH"], scaler :object = None) -> np.ndarray :
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
            return df

        df = self.calculate_moving_average(df, window=ma_window)
        df = self.calculate_exponential_moving_average(df, window=ema_window)
        df = self.calculate_macd(df, short_window=macd_short_window, long_window=macd_long_window)
        df = self.calculate_rsi(df, window=rsi_window)
        df = self.calculate_stochastic_oscillator(df, window=stochastic_window)
        df = self.calculate_bollinger_bands(df, window=bollinger_window)
        df = self.calculate_atr(df, window=atr_window)
        df = self.calculate_adx(df, window=adx_window)
        df = self.calculate_obv(df)
        df = self.calculate_pivot_points(df)
        df = self.calculate_price_sentiment(df, window = sentiment_window)
        return df