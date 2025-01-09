# aipsarg/ai/trading_ai.py
import time
import logging
import json
from config.config import PAIRS, TRADING_CONFIG
from api.api_utils import ExchangeAPI, fetch_instrument_info
from data.data_utils import DataHandler
from model.model_utils import ModelManager
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
from strategy.base_strategy import BaseStrategy, TradingStyle
from strategy.moving_average_crossover import MovingAverageCrossover
from strategy.auto_strategy import AutoStrategy
from utils.logger import setup_logger
from utils.helpers import is_market_open

# Setup logger
logger = setup_logger(__name__)

class TradingAI:
    """A class to handle the trading AI logic."""

    def __init__(self, trading_strategy : BaseStrategy =  MovingAverageCrossover(short_window=20, long_window=50, trading_style=TradingStyle.SWING_TRADING)):
        """Initializes the TradingAI."""
        self.api = ExchangeAPI()
        self.data_handler = DataHandler()
        self.model_manager = ModelManager()
        self.ai_state = self._load_ai_state()
        self._stop_monitoring = threading.Event()
        self.min_size = None  # Initialize min_size here
        self._initialize_min_size() # Get min size
        self.trading_strategy = trading_strategy
        self.auto_strategy = AutoStrategy(trading_style = self.trading_strategy.get_trading_style(), strategy=self.trading_strategy, data_handler=self.data_handler, model_manager=self.model_manager)
        self.previous_pair = None
        self.previous_balance = None

    def _initialize_min_size(self):
        """Initializes the minimum order size."""
        try:
          instrument_info = fetch_instrument_info(PAIRS)
          if instrument_info and 'min_size' in instrument_info:
               self.min_size = float(instrument_info['min_size'])
               logger.info(f"Minimum order size for {PAIRS} set to {self.min_size}")
          else:
               logger.error(f"Failed to fetch or set minimum order size for {PAIRS}")
        except Exception as e:
             logger.error(f"Error initialize min size: {e}")

    def _load_ai_state(self):
        """Loads the AI state from file."""
        try:
            with open(TRADING_CONFIG["BOT_STATE_FILE"], "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                    "purchase_price": 0.0,
                    "action": None,
                    "amount": 0.0,
                    "previous_pair": None,
                    "previous_balance" : None
                    }
        except Exception as e:
            logger.error(f"Error load ai state: {e}")
            return {
                    "purchase_price": 0.0,
                    "action": None,
                    "amount": 0.0,
                    "previous_pair": None,
                    "previous_balance": None
                    }
    
    def _save_ai_state(self):
        """Saves the AI state to file."""
        try:
          with open(TRADING_CONFIG["BOT_STATE_FILE"], "w") as f:
             json.dump(self.ai_state, f)
        except Exception as e:
            logger.error(f"Error saving ai state: {e}")

    def check_take_profit_stop_loss(self, df, pair, action, purchase_price, amount):
        """Checks for take profit and stop loss conditions."""
        try:
           current_price = df['close'].iloc[-1]
           take_profit_price = purchase_price * (1 + TRADING_CONFIG["TAKE_PROFIT_PERCENTAGE"])
           stop_loss_price = purchase_price * (1 - TRADING_CONFIG["STOP_LOSS_PERCENTAGE"])
           
           if action == 'buy':
              if current_price >= take_profit_price or current_price <= stop_loss_price:
                  order = self.api.place_order(pair, 'sell', amount)
                  if order:
                      logger.info(f"Action triggered for BUY. Selling {amount} of {pair} at {current_price}.")
                      self.ai_state["action"] = None
                      self._save_ai_state()
           elif action == 'sell':
               if current_price <= stop_loss_price or current_price >= take_profit_price:
                   order = self.api.place_order(pair, 'buy', amount)
                   if order:
                       logger.info(f"Action triggered for SELL. Buying back {amount} of {pair} at {current_price}.")
                       self.ai_state["action"] = None
                       self._save_ai_state()
        except Exception as e:
            logger.error(f"Error check take profit and stop loss: {e}")

    def trading_strategy(self, df, pair, model, scaler, trading_style):
        """Executes the trading strategy."""
        try:
            # 1. Generate trading signal using AutoStrategy
            signal = self.auto_strategy.generate_trading_signal(df = df, pair = pair)

            if signal is None:
               return
            # 2. Perhitungan Profit/Stop Loss & Eksekusi
            balance = self.api.fetch_balance(pair)
            
            if not balance or 'currency' not in balance or 'asset' not in balance:
                logger.error("Could not retrieve balance information.")
                return
            
            free_usdt = balance['currency']['available']
            free_stock = balance['asset']['available']
           
            if signal == 'buy' and free_usdt > 0:
                amount_to_buy = free_usdt * TRADING_CONFIG["BUY_PERCENTAGE"]
                price = df['close'].iloc[-1]
                amount_to_buy_stock = amount_to_buy / price
                if self.min_size is not None and amount_to_buy_stock >= self.min_size:  # Check against minimum order size
                   order = self.api.place_order(pair, 'buy', amount_to_buy_stock)
                   if order:
                       self._log_trade_decision(df, 0, 'buy', pair) #prediction diganti 0, karena di handle pada auto strategy
                       self.check_take_profit_stop_loss(df, pair, 'buy', price, amount_to_buy_stock)
                       self.ai_state["action"] = 'buy'
                       self.ai_state["purchase_price"] = price
                       self.ai_state["amount"] = amount_to_buy_stock
                       self._save_ai_state()
                else:
                     logger.warning(f"Amount to buy {amount_to_buy_stock} too small, minimum amount is {self.min_size}")

            elif signal == 'sell' and free_stock > 0:
                amount_to_sell = free_stock * TRADING_CONFIG["SELL_PERCENTAGE"]
                if self.min_size is not None and amount_to_sell >= self.min_size:  # Check against minimum order size
                    order = self.api.place_order(pair, 'sell', amount_to_sell)
                    if order:
                        self._log_trade_decision(df, 0, 'sell', pair) #prediction diganti 0, karena di handle pada auto strategy
                        self.check_take_profit_stop_loss(df, pair, 'sell', df['close'].iloc[-1], amount_to_sell)
                        self.ai_state["action"] = 'sell'
                        self.ai_state["purchase_price"] = df['close'].iloc[-1]
                        self.ai_state["amount"] = amount_to_sell
                        self._save_ai_state()
                else:
                    logger.warning(f"Amount to sell {amount_to_sell} too small, minimum amount is {self.min_size}")

            self.display_summary(df, pair)
            self._handle_balance_display(df, pair)
        
        except Exception as e:
            logger.error(f"Error in strategy: {e}")

    def _log_trade_decision(self, df, prediction, signal, pair):
        """Logs the trade decision with relevant information."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        macd_value = df['macd'].iloc[-1]
        signal_line_value = df['signal_line'].iloc[-1]
        logger.info(f"Trade Decision: {signal.upper()} at {timestamp}, "
                     f"Pair: {pair}, "
                     f"Predicted Price Movement: {prediction:.4f}, "
                     f"MACD: {macd_value:.4f}, "
                     f"Signal Line: {signal_line_value:.4f}")
    
    def _generate_trading_signal(self, df, prediction):
         """Generates trading signal based on indicators and prediction."""
         try:
            return self.trading_strategy.generate_signal(df=df)
         except Exception as e:
             logger.error(f"Error generate trading signal: {e}")
             return None
    
    def monitor_market(self, pair, model, scaler):
        """Monitors the market and executes trading strategy."""
        while not self._stop_monitoring.is_set():
           try:
                logger.info("Monitoring market...")
                df = self.data_handler.fetch_okx_candlesticks(pair)
                if df is not None:
                    self.trading_strategy(df, pair, model, scaler, trading_style = self.trading_strategy.get_trading_style().value )
                else:
                     logger.error("Failed to fetch candlestick data.")
           except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
           time.sleep(TRADING_CONFIG["MONITOR_SLEEP_INTERVAL"])

    def _handle_balance_display(self, df, pair):
         """Handles display of balance and PnL calculation."""
         try:
              if pair != self.ai_state["previous_pair"] and self.ai_state["previous_pair"] is not None:
                self.display_summary(df, pair)
                self.previous_pair = self.ai_state["previous_pair"]
                self.previous_balance = self.ai_state["previous_balance"]
                self.ai_state["previous_pair"] = pair
                self._save_ai_state()

              self.display_balance(df, pair)
              self.ai_state["previous_balance"] = self.api.fetch_balance(pair)
              self.ai_state["previous_pair"] = pair
              self._save_ai_state()
              
         except Exception as e:
              logger.error(f"Error handling balance display {e}")

    def display_summary(self, df, pair):
        """Displays summary of the latest data and account status."""
        try:
            summary = df[['timestamp', 'open', 'high', 'low', 'close', 'psar']].tail(5)
            print("\n=== Ringkasan Data Terbaru ===")
            print(summary.to_string(index=False))

            balance = self.api.fetch_balance(pair)
            if not balance or 'currency' not in balance or 'asset' not in balance:
                 logger.error("Could not retrieve balance for summary.")
                 return
            
            free_usdt = balance['currency']['available']
            free_stock = balance['asset']['available']
            current_price = df['close'].iloc[-1]
            
            # Calculate PnL (Profit and Loss)
            if self.ai_state["action"] == 'buy' and self.ai_state["purchase_price"] > 0:
                 purchase_price = self.ai_state["purchase_price"]
                 pnl = (current_price - purchase_price) * self.ai_state["amount"]
                 pnl_percentage = (pnl / (purchase_price * self.ai_state["amount"])) * 100
            elif self.ai_state["action"] == 'sell' and self.ai_state["purchase_price"] > 0:
                 purchase_price = self.ai_state["purchase_price"]
                 pnl = (purchase_price - current_price) * self.ai_state["amount"]
                 pnl_percentage = (pnl / (purchase_price * self.ai_state["amount"])) * 100
            else:
                pnl = 0.0
                pnl_percentage = 0.0

            print("\n=== Status Akun ===")
            print(f"Saldo USDT: {free_usdt:.2f} USDT")
            print(f"Saldo {pair.split('/')[0]}: {free_stock:.2f} {pair.split('/')[0]}")
            print(f"Potensi Untung/Rugi: {pnl:.2f} USDT ({pnl_percentage:.2f}%)")

        except Exception as e:
            logger.error(f"Error displaying summary: {e}") 

    def display_balance(self, df, pair):
        """Displays the balance of the current trading pair and calculates PnL"""
        try:
            balance = self.api.fetch_balance(pair)
            if not balance or 'currency' not in balance or 'asset' not in balance:
                logger.error("Could not retrieve balance for display balance.")
                return
            
            free_usdt = balance['currency']['available']
            free_stock = balance['asset']['available']
            current_price = df['close'].iloc[-1]
            
            # Calculate PnL (Profit and Loss)
            pnl = 0.0
            pnl_percentage = 0.0
            if self.previous_balance and self.previous_pair == pair :
               previous_free_usdt = self.previous_balance['currency']['available']
               previous_free_stock = self.previous_balance['asset']['available']
               asset_name = pair.split('/')[0]
               
               if self.ai_state["action"] == 'buy':
                  pnl = (current_price * free_stock) - (previous_free_usdt + (previous_free_stock * current_price))
                  pnl_percentage = (pnl / (previous_free_usdt + (previous_free_stock * current_price))) * 100
               elif self.ai_state["action"] == 'sell':
                   pnl = (previous_free_usdt + (previous_free_stock * current_price)) - (current_price * free_stock)
                   pnl_percentage = (pnl / (current_price * free_stock)) * 100
               else:
                    pnl = 0.0
                    pnl_percentage = 0.0
                   
            print("\n=== Balance ===")
            print(f"Saldo USDT: {free_usdt:.2f} USDT")
            print(f"Saldo {pair.split('/')[0]}: {free_stock:.2f} {pair.split('/')[0]}")
            print(f"Potensi Untung/Rugi: {pnl:.2f} USDT ({pnl_percentage:.2f}%)")


        except Exception as e:
            logger.error(f"Error display balance: {e}")

    def run(self):
        """Runs the trading AI."""
        try:
           if not is_market_open():
              logger.info("Market is close, waiting for market to open")
              return
           print("=== A1PSARG Auto-Trading Dimulai ===")

           # Cek apakah model sudah ada, jika tidak, latih model
           model, scaler = self.model_manager.load_trained_model()
           if model is None or scaler is None:
              logger.info("Model file not found. Training model...")
              model_trained, model, scaler = self.model_manager.train_model(pair = PAIRS)
              if not model_trained:
                  print("Model failed to trained")
                  return

           with ThreadPoolExecutor() as executor:
                self._stop_monitoring.clear()
                executor.submit(self.monitor_market, PAIRS, model, scaler)
            
        except KeyboardInterrupt:
             self._stop_monitoring.set()
             logger.info("Program dihentikan oleh pengguna.")
             print("\n=== Program Berhenti ===")
        except Exception as e:
              logger.error(f"Error in main loop: {e}")