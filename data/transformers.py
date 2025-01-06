# aipsarg/data/transformers.py
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

class DataTransformer:
    """A class to handle data transformations."""

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
