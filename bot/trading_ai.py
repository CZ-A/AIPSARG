import time
import logging
import json
from config.config import PAIRS, TRADING_CONFIG, INDICATOR_CONFIG
from api.api_utils import ExchangeAPI
from data.data_utils import DataHandler
from model.model_utils import ModelManager
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading

class TradingAI:
    """A class to handle the trading AI logic."""

    def __init__(self):
        """Initializes the TradingAI."""
        self.api = ExchangeAPI()
        self.data_handler = DataHandler()
        self.model_manager = ModelManager()
        self.ai_state = self._load_ai_state()
        self._stop_monitoring = threading.Event()
    
    def _load_ai_state(self):
        """Loads the AI state from file."""
        try:
            with open(TRADING_CONFIG["BOT_STATE_FILE"], "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                    "purchase_price": 0.0,
                    "action": None,
                    "amount": 0.0
                    }
    
    def _save_ai_state(self):
        """Saves the AI state to file."""
        with open(TRADING_CONFIG["BOT_STATE_FILE"], "w") as f:
            json.dump(self.ai_state, f)

    def check_take_profit_stop_loss(self, df, pair, action, purchase_price, amount):
        """Checks for take profit and stop loss conditions."""
        current_price = df['close'].iloc[-1]
        take_profit_price = purchase_price * (1 + TRADING_CONFIG["TAKE_PROFIT_PERCENTAGE"])
        stop_loss_price = purchase_price * (1 - TRADING_CONFIG["STOP_LOSS_PERCENTAGE"])
        
        if action == 'buy':
            if current_price >= take_profit_price or current_price <= stop_loss_price:
                order = self.api.place_order(pair, 'sell', amount)
                if order:
                    logging.info(f"Action triggered for BUY. Selling {amount} of {pair} at {current_price}.")
                    self.ai_state["action"] = None
                    self._save_ai_state()
        elif action == 'sell':
            if current_price <= stop_loss_price or current_price >= take_profit_price:
                order = self.api.place_order(pair, 'buy', amount)
                if order:
                    logging.info(f"Action triggered for SELL. Buying back {amount} of {pair} at {current_price}.")
                    self.ai_state["action"] = None
                    self._save_ai_state()
    
    def trading_strategy(self, df, pair, model, scaler, trading_style):
        """Executes the trading strategy."""
        try:
            # 1. Data Pasar dan Indikator
            df = self.data_handler.add_all_indicators(df, trading_style=trading_style)
            columns_to_scale = ['close', 'volume','psar','ma_20','ema_20','macd', 'signal_line', 'macd_histogram','rsi', '%k', '%d',
                     'bollinger_upper', 'bollinger_lower', 'atr', 'adx', 'obv', 'pivot', 'r1', 's1', 'r2', 's2', 'price_sentiment']
            df = self.data_handler.standardize_data(df, columns_to_scale=columns_to_scale)
            data_for_model = self.data_handler.prepare_data_for_model(df, scaler=scaler)
            if data_for_model is None:
                logging.warning("Cannot proceed trading, not enough data")
                self.display_summary(df, pair)
                return  # Stop trading jika tidak ada data
            
            # 2. Prediksi Model
            prediction = self.model_manager.predict_price(df, model, scaler)

            # 3. Sinyal Trading (menggabungkan prediksi dan indikator)
            signal = self._generate_trading_signal(df, prediction, trading_style)
            
            if signal is None:
                return

            # 4. Perhitungan Profit/Stop Loss & Eksekusi
            balance = self.api.get_exchange().fetch_balance()
            free_usdt = balance['free']['USDT']
            free_stock = balance['free'].get(pair.split('/')[0], 0.0)

            if signal == 'buy' and free_usdt > 0:
                amount_to_buy = free_usdt * TRADING_CONFIG["BUY_PERCENTAGE"]
                price = df['close'].iloc[-1]
                amount_to_buy_stock = amount_to_buy / price
                if amount_to_buy_stock > TRADING_CONFIG["MINIMUM_ORDER_AMOUNT"]:  # Ganti <minimum_amount> dengan nilai minimum order
                   order = self.api.place_order(pair, 'buy', amount_to_buy_stock)
                   if order:
                       self._log_trade_decision(df, prediction, 'buy', pair)
                       self.check_take_profit_stop_loss(df, pair, 'buy', price, amount_to_buy_stock)
                else:
                    logging.warning(f"Amount to buy {amount_to_buy_stock} too small, minimum amount is {TRADING_CONFIG['MINIMUM_ORDER_AMOUNT']}")

            elif signal == 'sell' and free_stock > 0:
                amount_to_sell = free_stock * TRADING_CONFIG["SELL_PERCENTAGE"]
                if amount_to_sell > TRADING_CONFIG["MINIMUM_ORDER_AMOUNT"]:  # Ganti <minimum_amount> dengan nilai minimum order
                    order = self.api.place_order(pair, 'sell', amount_to_sell)
                    if order:
                        self._log_trade_decision(df, prediction, 'sell', pair)
                        self.check_take_profit_stop_loss(df, pair, 'sell', df['close'].iloc[-1], amount_to_sell)
                else:
                    logging.warning(f"Amount to sell {amount_to_sell} too small, minimum amount is {TRADING_CONFIG['MINIMUM_ORDER_AMOUNT']}")

            self.display_summary(df, pair)
        
        except Exception as e:
            logging.error(f"Error in strategy: {e}")

    def _log_trade_decision(self, df, prediction, signal, pair):
        """Logs the trade decision with relevant information."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        macd_value = df['macd'].iloc[-1]
        signal_line_value = df['signal_line'].iloc[-1]
        logging.info(f"Trade Decision: {signal.upper()} at {timestamp}, "
                     f"Pair: {pair}, "
                     f"Predicted Price Movement: {prediction:.4f}, "
                     f"MACD: {macd_value:.4f}, "
                     f"Signal Line: {signal_line_value:.4f}")
    
    def _generate_trading_signal(self, df, prediction, trading_style):
         """Generates trading signal based on indicators and prediction."""
         try:
            if prediction == -1:  # Jika prediksi error, gunakan strategi PSAR
                signal = 'buy' if df['psar'].iloc[-1] < df['close'].iloc[-1] else 'sell'
            elif trading_style == 'scalping':
                if prediction >= 0.6 and df['rsi'].iloc[-1] < 70 and df['%k'].iloc[-1] < 70:
                    signal = 'buy'
                elif prediction < 0.4 and df['rsi'].iloc[-1] > 30 and df['%k'].iloc[-1] > 30:
                     signal = 'sell'
                else:
                     signal = None
            elif trading_style == 'day_trading':
                if prediction >= 0.6 and df['macd'].iloc[-1] > df['signal_line'].iloc[-1] and df['rsi'].iloc[-1] < 70:
                     signal = 'buy'
                elif prediction < 0.4 and df['macd'].iloc[-1] < df['signal_line'].iloc[-1] and df['rsi'].iloc[-1] > 30:
                     signal = 'sell'
                else:
                     signal = None
            elif trading_style == 'swing_trading':
                if prediction >= 0.6 and df['adx'].iloc[-1] > 25 and df['%k'].iloc[-1] < 80:
                     signal = 'buy'
                elif prediction < 0.4 and df['adx'].iloc[-1] > 25 and df['%k'].iloc[-1] > 20:
                     signal = 'sell'
                else:
                     signal = None
            elif trading_style == 'long_term':
                if prediction >= 0.6 and df['ma_100'].iloc[-1] < df['close'].iloc[-1] and df['obv'].iloc[-1] > df['obv'].iloc[-2] :
                    signal = 'buy'
                elif prediction < 0.4 and df['ma_100'].iloc[-1] > df['close'].iloc[-1] and df['obv'].iloc[-1] < df['obv'].iloc[-2] :
                     signal = 'sell'
                else:
                    signal = None
            else:
                signal = None
            
            return signal
         except Exception as e:
             logging.error(f"Error generate trading signal: {e}")
             return None
    
    def monitor_market(self, pair, model, scaler, trading_style):
        """Monitors the market and executes trading strategy."""
        while not self._stop_monitoring.is_set():
           try:
                logging.info("Monitoring market...")
                df = self.data_handler.fetch_okx_candlesticks(pair)
                if df is not None:
                    df = self.data_handler.calculate_psar(df)
                    self.trading_strategy(df, pair, model, scaler, trading_style)
                else:
                     logging.error("Failed to fetch candlestick data.")
           except Exception as e:
                logging.error(f"Error in market monitoring: {e}")
           time.sleep(TRADING_CONFIG["MONITOR_SLEEP_INTERVAL"])
    
    def display_summary(self, df, pair):
        """Displays summary of the latest data and account status."""
        try:
            summary = df[['timestamp', 'open', 'high', 'low', 'close', 'psar']].tail(5)
            print("\n=== Ringkasan Data Terbaru ===")
            print(summary.to_string(index=False))
            balance = self.api.get_exchange().fetch_balance()
            free_usdt = balance['free']['USDT']
            free_stock = balance['free'].get(pair.split('/')[0], 0.0)
            print("\n=== Status Akun ===")
            print(f"Saldo USDT: {free_usdt:.2f} USDT")
            print(f"Saldo {pair.split('/')[0]}: {free_stock:.2f} {pair.split('/')[0]}")
        except Exception as e:
            logging.error(f"Error displaying summary: {e}") 

    def run(self):
        """Runs the trading AI."""
        try:
            print("=== A1PSARG Auto-Trading Dimulai ===")

            # Cek apakah model sudah ada, jika tidak, latih model
            model, scaler = self.model_manager.load_trained_model()
            if model is None or scaler is None:
                logging.info("Model file not found. Training model...")
                model_trained, model, scaler = self.model_manager.train_model(pair = PAIRS)
                if not model_trained:
                     print("Model failed to trained")
                     return

            with ThreadPoolExecutor() as executor:
                 self._stop_monitoring.clear()
                 executor.submit(self.monitor_market, PAIRS, model, scaler, TRADING_CONFIG["TRADING_STYLE"])
            
        except KeyboardInterrupt:
             self._stop_monitoring.set()
             logging.info("Program dihentikan oleh pengguna.")
             print("\n=== Program Berhenti ===")
        except Exception as e:
              logging.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    ai = TradingAI()
    ai.run()
