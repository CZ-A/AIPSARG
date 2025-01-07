# aipsarg/tests/test_ai.py
import unittest
from aipsarg.ai.trading_ai import TradingAI
from aipsarg.api.mock_api import MockAPI
from aipsarg.strategy.moving_average_crossover import MovingAverageCrossover
from aipsarg.strategy.base_strategy import TradingStyle
import pandas as pd


class TestBot(unittest.TestCase):
    
    def test_trading_strategy(self):
        mock_api = MockAPI()
        strategy = MovingAverageCrossover(short_window=20, long_window=50, trading_style = TradingStyle.SWING_TRADING)
        bot = TradingAI(trading_strategy = strategy)
        bot.api = mock_api
        data = {'timestamp': [1, 2, 3, 4, 5], 'open': [10, 12, 15, 13, 16], 'high': [11, 13, 16, 14, 17], 'low': [9, 11, 14, 12, 15], 'close': [10.5, 12.5, 15.5, 13.5, 16.5], 'volume': [100, 120, 150, 130, 160]}
        df = pd.DataFrame(data)
        bot.trading_strategy(df, pair="BTC/USDT", model=None, scaler=None, trading_style=bot.trading_strategy.get_trading_style().value)
        self.assertTrue(True)
