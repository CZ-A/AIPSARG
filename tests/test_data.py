# aipsarg/tests/test_data.py
import unittest
import pandas as pd
from aipsarg.data.data_utils import DataHandler

class TestData(unittest.TestCase):
    def test_fetch_okx_candlesticks(self):
        data_handler = DataHandler()
        df = data_handler.fetch_okx_candlesticks(pair="BTC/USDT", timeframe="1m", limit=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df),10)
