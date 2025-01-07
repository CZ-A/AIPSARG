# aipsarg/model/model_manager.py
import logging
import numpy as np
import json
from typing import Tuple, Optional
from model.base_model import BaseModel
from config.config import MODEL_FILE, SCALER_FILE, PAIRS
from data.data_utils import DataHandler
from model.lstm_model import LSTMModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential


class ModelManager:
    """A class to handle model creation, training, loading, and prediction."""

    def __init__(self, model_type : BaseModel = LSTMModel()) -> None:
        """Initializes ModelManager with a DataHandler instance."""
        self.model_type = model_type
    
    def create_model(self, input_shape: Tuple[int, int], lstm_units: int) -> object:
        return self.model_type.create_model(input_shape=input_shape, lstm_units=lstm_units)
    
    def train_model(self, model_file: str = MODEL_FILE, scaler_file: str = SCALER_FILE, training_limit: int = 500, pair: str = PAIRS) -> Tuple[bool, Optional[object], Optional[MinMaxScaler]]:
        """
        Trains the model.

        Args:
            model_file (str): Path to save the trained model.
            scaler_file (str): Path to save the scaler.
            training_limit (int): Limit of data to use for training.
            pair (str): Trading pair symbol.

        Returns:
            Tuple[bool, Optional[object], Optional[MinMaxScaler]]: 
                A tuple indicating success status, trained model, and the scaler.
        
        Raises:
            Exception: If an error occurs during the training process.
        """
        return self.model_type.train_model(model_file=model_file,scaler_file=scaler_file,training_limit=training_limit,pair=pair)
    
    def load_trained_model(self, model_file: str = MODEL_FILE, scaler_file: str = SCALER_FILE) -> Tuple[Optional[object], Optional[MinMaxScaler]]:
        """
        Loads a trained model and scaler from files.

        Args:
            model_file (str): Path to the saved model file.
            scaler_file (str): Path to the saved scaler file.

        Returns:
            Tuple[Optional[object], Optional[MinMaxScaler]]: 
                A tuple containing the loaded model and scaler, or None if loading fails.
        
        Raises:
            Exception: If an error occurs during the loading process.
        """
        return self.model_type.load_trained_model(model_file=model_file,scaler_file=scaler_file)

    def predict_price(self, df: object, model: Optional[object], scaler: Optional[MinMaxScaler]) -> float:
        """
        Predicts the next price movement using the trained model.

        Args:
           df (object): Pandas DataFrame containing market data.
           model (Optional[object]): Trained model.
           scaler (Optional[MinMaxScaler]): Trained scaler.
           
        Returns:
            float: Predicted price movement (0-1), or -1 on error.
        
        Raises:
            Exception: If an error occurs during the prediction.
        """
        return self.model_type.predict_price(df=df,model=model,scaler=scaler)
