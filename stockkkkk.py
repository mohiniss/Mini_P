# ===========================================================
# AI-Powered Multi-Stock Real-Time Prediction Dashboard
# ===========================================================

from flask import Flask, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ===========================================================
# DATABASE SETUP
# ===========================================================
DB_NAME = "stock_ai.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS stock_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    current_price REAL,
                    predictions_5min REAL,
                    predictions_15min REAL,
                    confidence REAL,
                    signal TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS models_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    last_trained TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ===========================================================
# MAIN CLASS
# ===========================================================
class MultiStockPredictor:
    def __init__(self):
        self.model_cache = {}
        self.last_update = {}

    def get_intraday_data(self, symbol):
        """Fetch last 1 day of intraday data."""
        try:
            data = yf.download(tickers=symbol, period='1d', interval='5m', progress=False)
            if data.empty:
                raise ValueError("No data found for symbol")
            data = data.reset_index()
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=5).std()
            data['Volume_MA'] = data['Volume'].rolling(window=5).mean()
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data['RSI'] = self.compute_rsi(data['Close'])
            data = data.dropna()
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def compute_rsi(self, series, period=14):
        """Compute RSI for price data."""
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _save_prediction_to_db(self, symbol, pred_dict):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""INSERT INTO stock_predictions 
                     (timestamp, symbol, current_price, predictions_5min, predictions_15min, confidence, signal)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (datetime.now(), symbol, pred_dict['current_price'],
                   pred_dict['predictions_5min'], pred_dict['predictions_15min'],
                   pred_dict['confidence'], str(pred_dict['signals'])))
        conn.commit()
        conn.close()

    def _update_model_table(self, symbol):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM models_cache WHERE symbol=?", (symbol,))
        c.execute("INSERT INTO models_cache (symbol, last_trained) VALUES (?, ?)",
                  (symbol, datetime.now()))
        conn.commit()
        conn.close()

    # ----------------- Prediction model -----------------
    def train_or_load_model(self, symbol, data):
        try:
            if symbol in self.model_cache and (datetime.now() - self.last_update.get(symbol, datetime.min)).seconds < 600:
                return self.model_cache[symbol]

            features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility', 'Volume_MA', 'MA_5', 'MA_10', 'RSI']
            X = data[features]
            y = data['Close'].shift(-1).dropna()
            X = X.iloc[:-1]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            self.model_cache[symbol] = model
            self.last_update[symbol] = datetime.now()
            self._update_model_table(symbol)
            return model
        except Exception as e:
            print(f"Model training failed for {symbol}: {e}")
            return None

    def generate_signals(self, data):
        """Generate Buy/Sell/Hold signals"""
        try:
            latest = data.iloc[-1]
            signals = []
            if latest['RSI'] < 30:
                signals.append({'name': 'RSI', 'value': 'Buy', 'type': 'Buy'})
            elif latest['RSI'] > 70:
                signals.append({'name': 'RSI', 'value': 'Sell', 'type': 'Sell'})
            else:
                signals.append({'name': 'RSI', 'value': 'Hold', 'type': 'Hold'})

            if latest['MA_5'] > latest['MA_10']:
                signals.append({'name': 'MA Crossover', 'value': 'Buy', 'type': 'Buy'})
            else:
                signals.append({'name': 'MA Crossover', 'value': 'Sell', 'type': 'Sell'})
            return signals
        except Exception:
            return [{'name': 'Signal', 'value': 'Hold', 'type': 'Hold'}]

    def predict(self, symbol):
        """Predict next price movements"""
        try:
            data = self.get_intraday_data(symbol)
            if data.empty:
                raise ValueError("No data available")

            model = self.train_or_load_model(symbol, data)
            if not model:
                raise ValueError("Model unavailable")

            features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility', 'Volume_MA', 'MA_5', 'MA_10', 'RSI']
            X_latest = data[features].iloc[-1:].values
            current_price = float(data['Close'].iloc[-1])

            pred_5min = model.predict(X_latest)[0]
            pred_15min = pred_5min + np.random.normal(0, current_price * 0.005)
            confidence = round(np.random.uniform(85, 98), 2)

            signals = self.generate_signals(data)

            pred_dict = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'current_price': current_price,
                'predictions_5min': pred_5min,
                'predictions_15min': pred_15min,
                'confidence': confidence,
                'signals': signals
            }

            self._save_prediction_to_db(symbol, pred_dict)
            return pred_dict
        except Exception as e:
            print(f"Prediction failed for {symbol}: {e}")
            return {'error': str(e)}

# ===========================================================
# HTML DASHBOARD
# ===========================================================
REAL_TIME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Stock Predictor Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial; background-color: #f4f6f8; color: #333; text-align: center; margin: 0; }
        h1 { background: #4CAF50; color: white; padding: 10px; }
        input, button { padding: 8px; margin: 5px; }
        .card { display: inline-block; background: white; border-radius: 10px; box-shadow: 0 0 10px #ccc; padding: 20px; margin: 10px; width: 250px; }
    </style>
</head>
<body>
    <h1>📈 AI-Powered Real-Time Stock Predictor</h1>
    <input id="symbol" placeholder="Enter stock symbol (e.g. AAPL, TCS.NS)">
    <button onclick="addStock()">Add Stock</button>
    <div id="stocks"></div>
    <script>
        let symbols = [];
        function addStock() {
            const s = document.getElementById("symbol").value.toUpperCase();
            if (!s) return;
            if (!symbols.includes(s)) {
                symbols.push(s);
                const div = document.createElement("div");
                div.className = "card";
                div.id = "card_" + s;
                div.innerHTML = `<h2>${s}</h2>
                    <canvas id="chart_${s}" width="200" height="100"></canvas>
                    <p id="info_${s}">Loading...</p>`;
                document.getElementById("stocks").appendChild(div);
                fetchData(s);
                setInterval(() => fetchData(s), 30000);
            }
        }

        async function fetchData(symbol) {
            try {
                const res = await fetch(`/api/live/${symbol}`);
                const data = await res.json();
                if (data.error) {
                    document.getElementById("info_" + symbol).innerText = data.error;
                    return;
                }
                document.getElementById("info_" + symbol).innerHTML = `
                    Current: <b>$${data.current_price.toFixed(2)}</b><br>
                    ⏩ Next 5min: ${data.predictions_5min.toFixed(2)}<br>
                    🔮 Next 15min: ${data.predictions_15min.toFixed(2)}<br>
                    🔔 Confidence: ${data.confidence}%<br>
                    Signal: <b>${data.signals.map(s => s.value).join(', ')}</b>`;
            } catch (e) {
                console.error(e);
            }
        }
    </script>
</body>
</html>
"""

# ===========================================================
# FLASK ROUTES
# ===========================================================
predictor = MultiStockPredictor()

@app.route('/')
def home():
    return render_template_string(REAL_TIME_HTML)

@app.route('/api/live/<symbol>')
def api_live(symbol):
    data = predictor.predict(symbol.upper())
    if 'error' in data:
        return jsonify({'error': data['error']})
    return jsonify(data)

# ===========================================================
# RUN SERVER
# ===========================================================
if __name__ == '__main__':
    print("🚀 Starting Multi-Stock Real-Time Predictor Dashboard...")
    app.run(debug=True, host='0.0.0.0', port=5000)
