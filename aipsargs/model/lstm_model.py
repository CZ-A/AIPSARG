# aipsarg/model/lstm_model.py
import logging
import numpy as np
import json
from typing import Tuple, Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
from configs.config import MODEL_CONFIG
from model.base_model import BaseModel

class LSTMModel(BaseModel):
    """Concrete implementation of LSTM Model."""
    
    def create_model(self, input_shape: Tuple[int, int], lstm_units: int = MODEL_CONFIG["LSTM_UNITS"]) -> Sequential:
        """
        Creates and returns a compiled LSTM model.

        Args:
            input_shape (Tuple[int, int]): The shape of the input data (sequence_length, num_features).
            lstm_units (int): The number of LSTM units.

        Returns:
            Sequential: A compiled LSTM model.
        
        Raises:
            Exception: If an error occurs during model creation.
        """
        try:
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(LSTM(units=lstm_units, return_sequences=True))
            model.add(LSTM(units=lstm_units))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])
            return model
        except Exception as e:
            logging.error(f"Error creating LSTM model: {e}")
            raise

    def train_model(self, model_file: str, scaler_file: str, training_limit: int, pair: str) -> Tuple[bool, Optional[Sequential], Optional[MinMaxScaler]]:
        """
        Trains the LSTM model.

        Args:
            model_file (str): Path to save the trained model.
            scaler_file (str): Path to save the scaler.
            training_limit (int): Limit of data to use for training.
            pair (str): Trading pair symbol.

        Returns:
            Tuple[bool, Optional[Sequential], Optional[MinMaxScaler]]: 
                A tuple indicating success status, trained model, and the scaler.
        
        Raises:
            Exception: If an error occurs during the training process.
        """
        from trading_bot.data.data_utils import DataHandler
        try:
            logging.info("Starting model training.")
            data_handler = DataHandler()
            train_df = data_handler.fetch_okx_candlesticks(pair = pair,limit=training_limit)
            if train_df.empty:
              logging.error("No data fetched for training model")
              return False, None, None
            train_df = data_handler.add_all_indicators(train_df, trading_style='swing_trading')
            X, y, scaler = data_handler.prepare_training_data(train_df)
            if not X or not y or scaler is None:
               logging.error("Failed to prepare data for training")
               return False, None, None
            input_shape = (X.shape[1], X.shape[2])
            model = self.create_model(input_shape)
            history = model.fit(X, y, epochs=MODEL_CONFIG["EPOCHS"], batch_size=MODEL_CONFIG["BATCH_SIZE"], verbose=1, validation_split=0.1)
            self._save_model(model, model_file)
            self._save_scaler(scaler, scaler_file)
            logging.info("Model trained and saved successfully.")
            return True, model, scaler
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            return False, None, None
    
    def _save_model(self, model: Sequential, model_file: str) -> None:
         """
         Saves the trained model to a file.
         
         Args:
            model (Sequential): Trained model.
            model_file (str): Path where model will be saved.
         
         Raises:
            Exception: If an error occurs during model saving.
        """
         try:
            model.save(model_file)
            logging.info(f"Model saved successfully to {model_file}")
         except Exception as e:
             logging.error(f"Error saving model to {model_file}: {e}")
             raise
    
    def _save_scaler(self, scaler: MinMaxScaler, scaler_file: str) -> None:
        """
        Saves the scaler to a JSON file.
        
        Args:
           scaler (MinMaxScaler): Trained scaler.
           scaler_file (str): Path where scaler data will be saved.

         Raises:
            Exception: If an error occurs during scaler saving.
        """
        try:
            scaler_data = {
                "min": scaler.min_.tolist(),
                "scale": scaler.scale_.tolist(),
                "feature_range": scaler.feature_range
            }
            with open(scaler_file, 'w') as f:
                json.dump(scaler_data, f)
            logging.info(f"Scaler saved successfully to {scaler_file}")
        except Exception as e:
            logging.error(f"Error saving scaler to {scaler_file}: {e}")
            raise
    
    def load_trained_model(self, model_file: str, scaler_file: str) -> Tuple[Optional[Sequential], Optional[MinMaxScaler]]:
        """
        Loads a trained model and scaler from files.

        Args:
            model_file (str): Path to the saved model file.
            scaler_file (str): Path to the saved scaler file.

        Returns:
            Tuple[Optional[Sequential], Optional[MinMaxScaler]]: 
                A tuple containing the loaded model and scaler, or None if loading fails.
        
        Raises:
            Exception: If an error occurs during the loading process.
        """
        try:
            logging.info("Loading trained model and scaler.")
            model = self._load_model(model_file)
            if model:
                 model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])
            scaler = self._load_scaler(scaler_file)
            logging.info("Model and scaler loaded successfully.")
            return model, scaler
        except Exception as e:
            logging.error(f"Error loading model and scaler: {e}")
            return None, None

    def _load_model(self, model_file: str) -> Optional[Sequential]:
         """
         Loads the trained model from file.
         
         Args:
             model_file (str): Path to the saved model file.
         
         Returns:
            Optional[Sequential]: The loaded model or None in case error.
        
         Raises:
            Exception: If an error occurs during model loading.
         """
         try:
            model = load_model(model_file)
            logging.info(f"Model loaded successfully from {model_file}")
            return model
         except Exception as e:
              logging.error(f"Error loading model from {model_file}: {e}")
              return None
    
    def _load_scaler(self, scaler_file: str) -> Optional[MinMaxScaler]:
        """
        Loads the scaler from the JSON file.

        Args:
            scaler_file (str): Path to the saved scaler file.

        Returns:
            Optional[MinMaxScaler]: The loaded scaler or None in case error.
        
        Raises:
            Exception: If an error occurs during scaler loading.
        """
        try:
            with open(scaler_file, 'r') as f:
                scaler_data = json.load(f)
            scaler = MinMaxScaler()
            scaler.min_ = np.array(scaler_data["min"])
            scaler.scale_ = np.array(scaler_data["scale"])
            scaler.feature_range = tuple(scaler_data["feature_range"])
            logging.info(f"Scaler loaded successfully from {scaler_file}")
            return scaler
        except Exception as e:
            logging.error(f"Error loading scaler from {scaler_file}: {e}")
            return None

    def predict_price(self, df: pd.DataFrame, model: Optional[Sequential], scaler: Optional[MinMaxScaler]) -> float:
        """
        Predicts the next price movement using the trained model.

        Args:
           df (object): Pandas DataFrame containing market data.
           model (Optional[Sequential]): Trained LSTM model.
           scaler (Optional[MinMaxScaler]): Trained scaler.
           
        Returns:
            float: Predicted price movement (0-1), or -1 on error.
        
        Raises:
            Exception: If an error occurs during the prediction.
        """
        from trading_bot.data.data_utils import DataHandler
        try:
            if model is None or scaler is None:
                logging.error("Model or scaler not loaded. Prediction cannot be made.")
                return -1
            data_handler = DataHandler()
            df_prepared = data_handler.prepare_data_for_model(df)
            if df_prepared is None:
                 return -1
            scaled_data = scaler.transform(df_prepared)
            X_pred = np.array([scaled_data])
            prediction = model.predict(X_pred)
            return prediction[0][0]
        except Exception as e:
            logging.error(f"Error during price prediction: {e}")
            return -1