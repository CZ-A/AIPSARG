# aipsarg/model/base_model.py
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import pandas as pd
class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def create_model(self, input_shape: Tuple[int, int], lstm_units: int) -> object:
        """Creates and returns a compiled model.

        Args:
            input_shape (Tuple[int, int]): The shape of the input data (sequence_length, num_features).
            lstm_units (int): The number of LSTM units.

        Returns:
            object: A compiled model
        """
        pass

    @abstractmethod
    def train_model(self,  model_file: str, scaler_file: str, training_limit: int, pair: str) -> Tuple[bool, object, object]:
        """
        Trains the  model.

        Args:
            model_file (str): Path to save the trained model.
            scaler_file (str): Path to save the scaler.
            training_limit (int): Limit of data to use for training.
            pair (str): Trading pair symbol.

        Returns:
            Tuple[bool, Optional[object], Optional[object]]: 
                A tuple indicating success status, trained model, and the scaler.
        """
        pass
    
    @abstractmethod
    def load_trained_model(self, model_file: str, scaler_file: str) -> Tuple[Optional[object], Optional[object]]:
        """
        Loads a trained model and scaler from files.

        Args:
            model_file (str): Path to the saved model file.
            scaler_file (str): Path to the saved scaler file.

        Returns:
            Tuple[Optional[object], Optional[object]]: 
                A tuple containing the loaded model and scaler, or None if loading fails.
        """
        pass

    @abstractmethod
    def predict_price(self, df: pd.DataFrame, model: Optional[object], scaler: Optional[object]) -> float:
        """
        Predicts the next price movement using the trained model.

        Args:
           df (object): Pandas DataFrame containing market data.
           model (Optional[object]): Trained model.
           scaler (Optional[object]): Trained scaler.
           
        Returns:
            float: Predicted price movement (0-1), or -1 on error.
        """
        pass
