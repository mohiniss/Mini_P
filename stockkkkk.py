from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import threading
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

DB_FILE = 'stocks.db'

app = Flask(__name__)

# Real-time HTML Template with Multiple Stock Graphs
REAL_TIME_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Stock Real-Time Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #0f0f23; color: #ffffff; font-family: 'Arial', sans-serif; }
        .terminal { background: #1a1a2e; border: 1px solid #00ff00; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
        .stock-card { background: #162447; border: 1px solid #00ff00; border-radius: 8px; padding: 10px; margin: 5px; }
        .price-up { color: #00ff00; }
        .price-down { color: #ff0000; }
        .chart-container { height: 300px; margin-bottom: 15px; }
        .live-badge { background: #ff0000; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .stock-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; }
        .signal-buy { background: #28a745 !important; }
        .signal-sell { background: #dc3545 !important; }
        .signal-hold { background: #ffc107 !important; color: #000 !important; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <div class="row mt-2">
            <div class="col-12">
                <div class="terminal text-center">
                    <h2 class="mb-1"><span class="live-badge badge me-2">LIVE</span>MULTI-STOCK REAL-TIME PREDICTION SYSTEM</h2>
                    <p class="mb-0">Tracking multiple stocks simultaneously with AI predictions</p>
                </div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="row">
            <div class="col-md-3">
                <div class="terminal">
                    <h5>CONTROL PANEL</h5>
                    <form id="controlForm">
                        <div class="mb-2">
                            <label class="form-label small">Add Stock Symbol</label>
                            <div class="input-group input-group-sm">
                                <input type="text" class="form-control bg-dark text-light" id="newSymbol" placeholder="e.g., AAPL">
                                <button type="button" class="btn btn-success" onclick="addStock()">Add</button>
                            </div>
                        </div>
                        
                        <div class="mb-2">
                            <label class="form-label small">Update Interval</label>
                            <select class="form-select form-select-sm bg-dark text-light" id="updateInterval">
                                <option value="3000">3 seconds</option>
                                <option value="5000" selected>5 seconds</option>
                                <option value="10000">10 seconds</option>
                            </select>
                        </div>
                        
                        <div class="mb-2">
                            <label class="form-label small">Tracked Stocks</label>
                            <div id="stockList" class="d-flex flex-wrap gap-1">
                                <!-- Stock badges will appear here -->
                            </div>
                        </div>
                        
                        <button type="button" class="btn btn-warning btn-sm w-100 mt-2" onclick="clearAllStocks()">Clear All</button>
                    </form>
                    
                    <div class="mt-3">
                        <h6 class="small">Quick Add:</h6>
                        <div class="d-flex flex-wrap gap-1">
                            {% for stock in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC'] %}
                            <span class="badge bg-secondary stock-btn" style="cursor: pointer;" onclick="addQuickStock('{{ stock }}')">{{ stock }}</span>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                
                <!-- Summary Stats -->
                <div class="terminal mt-3">
                    <h6>MARKET SUMMARY</h6>
                    <div id="marketSummary">
                        <div class="text-center text-muted">Waiting for data...</div>
                    </div>
                </div>
            </div>

            <!-- Main Content - Stock Grid -->
            <div class="col-md-9">
                <div class="terminal">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="mb-0">LIVE STOCK CHARTS & PREDICTIONS</h5>
                        <span id="lastUpdate" class="text-muted small">Last update: --:--:--</span>
                    </div>
                    
                    <div id="stockGrid" class="stock-grid">
                        <!-- Stock cards will be dynamically added here -->
                        <div class="text-center text-muted" id="noStocksMessage">
                            <p>No stocks being tracked. Add stocks using the control panel.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Stock Card Template -->
    <template id="stockCardTemplate">
        <div class="stock-card" data-symbol="{symbol}">
            <div class="d-flex justify-content-between align-items-start mb-2">
                <h6 class="mb-0">{symbol}</h6>
                <button type="button" class="btn-close btn-close-white" onclick="removeStock('{symbol}')" aria-label="Close"></button>
            </div>
            
            <!-- Price Display -->
            <div class="row text-center mb-2">
                <div class="col-4">
                    <small>Current</small>
                    <div class="h6 mb-0 price-up" id="current_{symbol}">--</div>
                </div>
                <div class="col-4">
                    <small>Pred 5min</small>
                    <div class="h6 mb-0 price-up" id="pred5_{symbol}">--</div>
                </div>
                <div class="col-4">
                    <small>Confidence</small>
                    <div class="h6 mb-0 text-warning" id="conf_{symbol}">--%</div>
                </div>
            </div>
            
            <!-- Chart -->
            <div class="chart-container">
                <canvas id="chart_{symbol}"></canvas>
            </div>
            
            <!-- Signals -->
            <div class="mt-2">
                <small>Signals:</small>
                <div id="signals_{symbol}" class="d-flex flex-wrap gap-1">
                    <span class="badge signal-hold">Loading...</span>
                </div>
            </div>
        </div>
    </template>

    <script>
        let trackedStocks = new Set(['AAPL', 'MSFT', 'TSLA']); // Default stocks
        let updateInterval = 5000;
        let charts = {};
        let stockData = {};
        let updateIntervalId = null;

        // Initialize the application
        function initializeApp() {
            updateStockList();
            initializeDefaultStocks();
            startLiveUpdates();
        }

        // Add a stock to tracking
        function addStock() {
            const symbol = document.getElementById('newSymbol').value.trim().toUpperCase();
            if (symbol && !trackedStocks.has(symbol)) {
                trackedStocks.add(symbol);
                updateStockList();
                createStockCard(symbol);
                document.getElementById('newSymbol').value = '';
            }
        }

        // Add quick stock
        function addQuickStock(symbol) {
            if (!trackedStocks.has(symbol)) {
                trackedStocks.add(symbol);
                updateStockList();
                createStockCard(symbol);
            }
        }

        // Remove a stock
        function removeStock(symbol) {
            trackedStocks.delete(symbol);
            updateStockList();
            const card = document.querySelector(`[data-symbol="${symbol}"]`);
            if (card) {
                card.remove();
            }
            if (charts[symbol]) {
                charts[symbol].destroy();
                delete charts[symbol];
            }
            updateNoStocksMessage();
        }

        // Clear all stocks
        function clearAllStocks() {
            trackedStocks.clear();
            updateStockList();
            document.getElementById('stockGrid').innerHTML = '';
            Object.values(charts).forEach(chart => chart.destroy());
            charts = {};
            updateNoStocksMessage();
        }

        // Update stock list display
        function updateStockList() {
            const stockList = document.getElementById('stockList');
            stockList.innerHTML = '';
            trackedStocks.forEach(symbol => {
                const badge = document.createElement('span');
                badge.className = 'badge bg-primary';
                badge.textContent = symbol;
                badge.style.cursor = 'pointer';
                badge.onclick = () => removeStock(symbol);
                stockList.appendChild(badge);
            });
        }

        // Update no stocks message
        function updateNoStocksMessage() {
            const message = document.getElementById('noStocksMessage');
            message.style.display = trackedStocks.size === 0 ? 'block' : 'none';
        }

        // Create stock card
        function createStockCard(symbol) {
            updateNoStocksMessage();
            
            const template = document.getElementById('stockCardTemplate');
            const cardHTML = template.innerHTML
                .replace(/{symbol}/g, symbol);
            
            const card = document.createElement('div');
            card.innerHTML = cardHTML;
            document.getElementById('stockGrid').appendChild(card.firstElementChild);
            
            initializeChart(symbol);
        }

        // Initialize chart for a stock
        function initializeChart(symbol) {
            const ctx = document.getElementById(`chart_${symbol}`).getContext('2d');
            
            if (charts[symbol]) {
                charts[symbol].destroy();
            }
            
            charts[symbol] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: [],
                            borderColor: '#00ff00',
                            backgroundColor: 'rgba(0, 255, 0, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Predicted (5min)',
                            data: [],
                            borderColor: '#ff9900',
                            backgroundColor: 'rgba(255, 153, 0, 0.1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            tension: 0.4,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#ffffff',
                                font: { size: 10 }
                            }
                        },
                        title: {
                            display: true,
                            text: symbol,
                            color: '#ffffff',
                            font: { size: 12 }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#ffffff',
                                font: { size: 8 },
                                maxTicksLimit: 6
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: {
                                color: '#ffffff',
                                font: { size: 8 },
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
            
            // Initialize empty data
            stockData[symbol] = {
                timestamps: [],
                actualPrices: [],
                predictedPrices: []
            };
        }

        // Initialize default stocks
        function initializeDefaultStocks() {
            trackedStocks.forEach(symbol => {
                createStockCard(symbol);
            });
        }

        // Start live updates
        function startLiveUpdates() {
            // Clear existing interval
            if (updateIntervalId) {
                clearInterval(updateIntervalId);
            }
            
            // Get update interval from select
            updateInterval = parseInt(document.getElementById('updateInterval').value);
            
            // Fetch data immediately
            fetchAllStockData();
            
            // Set up interval
            updateIntervalId = setInterval(fetchAllStockData, updateInterval);
        }

        // Fetch data for all tracked stocks
        async function fetchAllStockData() {
            if (trackedStocks.size === 0) return;
            
            const promises = Array.from(trackedStocks).map(symbol => 
                fetch(`/api/live/${symbol}`).then(res => res.json())
            );
            
            try {
                const results = await Promise.allSettled(promises);
                
                results.forEach((result, index) => {
                    if (result.status === 'fulfilled') {
                        const symbol = Array.from(trackedStocks)[index];
                        const data = result.value;
                        updateStockDisplay(symbol, data);
                    }
                });
                
                updateMarketSummary();
                document.getElementById('lastUpdate').textContent = 
                    'Last update: ' + new Date().toLocaleTimeString();
                    
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        }

        // Update stock display
        function updateStockDisplay(symbol, data) {
            if (!data || data.error) return;
            
            // Update prices
            document.getElementById(`current_${symbol}`).textContent = '$' + data.current_price.toFixed(2);
            document.getElementById(`pred5_${symbol}`).textContent = '$' + data.predictions_5min.toFixed(2);
            document.getElementById(`conf_${symbol}`).textContent = data.confidence + '%';
            
            // Update price colors
            const change = ((data.predictions_5min - data.current_price) / data.current_price * 100);
            const currentEl = document.getElementById(`current_${symbol}`);
            const predEl = document.getElementById(`pred5_${symbol}`);
            
            currentEl.className = change >= 0 ? 'h6 mb-0 price-up' : 'h6 mb-0 price-down';
            predEl.className = change >= 0 ? 'h6 mb-0 price-up' : 'h6 mb-0 price-down';
            
            // Update chart
            updateStockChart(symbol, data);
            
            // Update signals
            updateStockSignals(symbol, data.signals);
        }

        // Update stock chart
        function updateStockChart(symbol, data) {
            if (!charts[symbol]) return;
            
            const now = new Date().toLocaleTimeString();
            const chart = charts[symbol];
            
            // Add new data point
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(data.current_price);
            chart.data.datasets[1].data.push(data.predictions_5min);
            
            // Keep only last 10 points
            if (chart.data.labels.length > 10) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }
            
            chart.update('none');
        }

        // Update stock signals
        function updateStockSignals(symbol, signals) {
            const signalsContainer = document.getElementById(`signals_${symbol}`);
            if (!signalsContainer || !signals) return;
            
            signalsContainer.innerHTML = signals.map(signal => 
                `<span class="badge signal-${signal.type.toLowerCase()}">${signal.name}: ${signal.value}</span>`
            ).join('');
        }

        // Update market summary
        function updateMarketSummary() {
            const summary = document.getElementById('marketSummary');
            const stocks = Array.from(trackedStocks);
            
            if (stocks.length === 0) {
                summary.innerHTML = '<div class="text-center text-muted">No stocks tracked</div>';
                return;
            }
            
            let bullish = 0;
            let bearish = 0;
            
            stocks.forEach(symbol => {
                const currentEl = document.getElementById(`current_${symbol}`);
                const predEl = document.getElementById(`pred5_${symbol}`);
                
                if (currentEl && predEl) {
                    const currentText = currentEl.textContent.replace('$', '');
                    const predText = predEl.textContent.replace('$', '');
                    
                    if (currentText !== '--' && predText !== '--') {
                        const current = parseFloat(currentText);
                        const pred = parseFloat(predText);
                        
                        if (pred > current) bullish++;
                        else bearish++;
                    }
                }
            });
            
            summary.innerHTML = `
                <div class="row text-center small">
                    <div class="col-6">
                        <div class="price-up">Bullish: ${bullish}</div>
                    </div>
                    <div class="col-6">
                        <div class="price-down">Bearish: ${bearish}</div>
                    </div>
                    <div class="col-12 mt-1">
                        <div class="text-info">Total: ${stocks.length} stocks</div>
                    </div>
                </div>
            `;
        }

        // Event listeners
        document.getElementById('updateInterval').addEventListener('change', startLiveUpdates);
        document.getElementById('newSymbol').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addStock();
            }
        });

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeApp);
    </script>
</body>
</html>
'''

class MultiStockPredictor:
    def __init__(self, db_file=DB_FILE):
        self.model_cache = {}
        self.stock_data_cache = {}
        self.last_update = {}
        self.db_file = db_file
        self._init_db()

    # ----------------- Database utilities -----------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                current_price REAL,
                pred_5min REAL,
                pred_15min REAL,
                confidence REAL,
                signals TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS models (
                symbol TEXT PRIMARY KEY,
                last_trained TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _save_prediction_to_db(self, symbol, pred_dict):
        try:
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions (symbol, timestamp, current_price, pred_5min, pred_15min, confidence, signals) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, pred_dict['timestamp'].isoformat(), pred_dict['current_price'], pred_dict['predictions_5min'], pred_dict['predictions_15min'], pred_dict['confidence'], json.dumps(pred_dict['signals']))
            )
            conn.commit()
        except Exception as e:
            print('DB save error:', e)
        finally:
            conn.close()

    def _update_model_table(self, symbol):
        try:
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute("REPLACE INTO models (symbol, last_trained) VALUES (?, ?)", (symbol, datetime.now().isoformat()))
            conn.commit()
        except Exception as e:
            print('DB model update error:', e)
        finally:
            conn.close()

    # ----------------- Data fetching & features -----------------
    def get_live_price(self, symbol):
        """Get real-time stock price"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = info.get('regularMarketPrice', info.get('currentPrice', None))

            if current_price is None:
                hist = stock.history(period='1d', interval='1m')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    # Generate realistic dummy data
                    base_prices = {
                        'AAPL': 180, 'GOOGL': 140, 'MSFT': 330, 'TSLA': 200,
                        'NVDA': 450, 'AMZN': 150, 'META': 320, 'NFLX': 500,
                        'AMD': 120, 'INTC': 40
                    }
                    base_price = base_prices.get(symbol, 100)
                    current_price = base_price + np.random.normal(0, base_price * 0.02)

            return float(current_price)
        except Exception as e:
            print(f"Error getting live price for {symbol}: {e}")
            base_prices = {
                'AAPL': 180, 'GOOGL': 140, 'MSFT': 330, 'TSLA': 200,
                'NVDA': 450, 'AMZN': 150, 'META': 320, 'NFLX': 500,
                'AMD': 120, 'INTC': 40
            }
            base_price = base_prices.get(symbol, 100)
            return float(base_price + np.random.normal(0, base_price * 0.02))

    def get_intraday_data(self, symbol, period='2d', interval='5m'):
        """Get intraday data for real-time prediction"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                return self.generate_sample_data(symbol)

            # Calculate technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Close'].rolling(10).std()
            data['Volume_MA'] = data['Volume'].rolling(10).mean()
            data['MA_5'] = data['Close'].rolling(5).mean()
            data['MA_10'] = data['Close'].rolling(10).mean()
            data['MA_20'] = data['Close'].rolling(20).mean()

            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            data = data.dropna()
            return data

        except Exception as e:
            print(f"Error getting intraday data for {symbol}: {e}")
            return self.generate_sample_data(symbol)

    def generate_sample_data(self, symbol):
        """Generate sample data when real data fails"""
        base_prices = {
            'AAPL': 180, 'GOOGL': 140, 'MSFT': 330, 'TSLA': 200,
            'NVDA': 450, 'AMZN': 150, 'META': 320, 'NFLX': 500,
            'AMD': 120, 'INTC': 40
        }
        base_price = base_prices.get(symbol, 100)

        dates = pd.date_range(end=datetime.now(), periods=50, freq='5min')
        prices = [base_price + i * 0.1 + np.random.normal(0, base_price * 0.01) for i in range(50)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + abs(np.random.normal(0, base_price * 0.005)) for p in prices],
            'Low': [p - abs(np.random.normal(0, base_price * 0.005)) for p in prices],
            'Close': prices,
            'Volume': [abs(np.random.normal(1000000, 100000)) for _ in prices]
        }, index=dates)

        # Calculate basic features
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Close'].rolling(10).std()
        data['Volume_MA'] = data['Volume'].rolling(10).mean()
        data['MA_5'] = data['Close'].rolling(5).mean()
        data['MA_10'] = data['Close'].rolling(10).mean()
        data['RSI'] = 50 + np.random.normal(0, 10, len(data))

        return data.dropna()

    def create_features(self, data):
        """Create features for prediction"""
        if data is None or len(data) < 20:
            return None, None

        features = []
        targets_5min = []

        # start after some lookback
        for i in range(10, len(data) - 1):
            current = data.iloc[i]
            feature_vector = [
                current['Close'],
                current.get('MA_5', current['Close']),
                current.get('MA_10', current['Close']),
                current.get('RSI', 50),
                current.get('Volatility', 0),
                current.get('Volume_MA', 0),
                data['Close'].iloc[i - 1] if i > 0 else current['Close'],
                data['Close'].iloc[i - 2] if i > 1 else current['Close'],
            ]

            features.append(feature_vector)
            targets_5min.append(data['Close'].iloc[i + 1])

        if len(features) == 0:
            return None, None

        return np.array(features), np.array(targets_5min)

    # ----------------- Training & Prediction -----------------
    def train_model(self, symbol):
        """Train model for a stock"""
        try:
            data = self.get_intraday_data(symbol)
            if data is None or len(data) < 30:
                return False

            features, targets_5min = self.create_features(data)

            if features is None or len(features) < 15:
                return False

            model = RandomForestRegressor(
                n_estimators=30,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(features, targets_5min)

            self.model_cache[symbol] = {
                'model': model,
                'last_trained': datetime.now(),
                'data': data
            }

            # store model metadata in DB
            self._update_model_table(symbol)

            return True

        except Exception as e:
            print(f"Training error for {symbol}: {e}")
            return False

    def calculate_confidence(self, data):
        """Calculate prediction confidence"""
        try:
            if data is None or len(data) < 10:
                return 75.0
            volatility = data['Returns'].std()
            confidence = 80.0 - (abs(volatility) * 1000)
            return float(max(60.0, min(90.0, confidence)))
        except Exception:
            return 75.0

    def generate_signals(self, current_price, pred_5min, data):
        """Generate trading signals"""
        signals = []

        try:
            # Price momentum
            change = ((pred_5min - current_price) / current_price * 100)
            if change > 0.5:
                signals.append({'name': 'Momentum', 'value': f'+{change:.2f}%', 'type': 'BUY'})
            elif change < -0.5:
                signals.append({'name': 'Momentum', 'value': f'{change:.2f}%', 'type': 'SELL'})
            else:
                signals.append({'name': 'Momentum', 'value': f'{change:.2f}%', 'type': 'HOLD'})

            # RSI signal
            if data is not None and 'RSI' in data.columns and len(data) > 0:
                rsi = data['RSI'].iloc[-1]
                if rsi < 30:
                    signals.append({'name': 'RSI', 'value': f'{rsi:.1f}', 'type': 'BUY'})
                elif rsi > 70:
                    signals.append({'name': 'RSI', 'value': f'{rsi:.1f}', 'type': 'SELL'})
                else:
                    signals.append({'name': 'RSI', 'value': f'{rsi:.1f}', 'type': 'HOLD'})

        except Exception as e:
            print(f"Error generating signals: {e}")
            signals.append({'name': 'System', 'value': 'Active', 'type': 'HOLD'})

        return signals

    def generate_demo_prediction(self, symbol):
        """Generate demo prediction when real prediction fails"""
        current_price = self.get_live_price(symbol)
        change = np.random.normal(0, 0.02)  # Random change between -2% to +2%
        pred_5min = current_price * (1 + change)

        return {
            'current_price': float(current_price),
            'predictions_5min': float(pred_5min),
            'predictions_15min': float(pred_5min * (1 + change * 1.5)),
            'confidence': int(max(60, min(90, 75 + np.random.normal(0, 10)))),
            'signals': [
                {'name': 'Trend', 'value': f'{change*100:+.2f}%', 'type': 'BUY' if change > 0 else 'SELL'},
                {'name': 'Volatility', 'value': 'Medium', 'type': 'HOLD'}
            ],
            'timestamp': datetime.now()
        }

    def predict_stock(self, symbol):
        """Make prediction for a single stock"""
        try:
            data = self.get_intraday_data(symbol)
            if data is None or len(data) < 20:
                demo = self.generate_demo_prediction(symbol)
                # Save demo to DB as well
                self._save_prediction_to_db(symbol, demo)
                return demo

            # Train model if needed
            if (symbol not in self.model_cache or
                datetime.now() - self.model_cache[symbol]['last_trained'] > timedelta(minutes=30)):
                self.train_model(symbol)

            features, _ = self.create_features(data)
            if features is None or len(features) == 0:
                demo = self.generate_demo_prediction(symbol)
                self._save_prediction_to_db(symbol, demo)
                return demo

            last_features = features[-1].reshape(1, -1)
            model = self.model_cache.get(symbol, {}).get('model', None)
            if model is None:
                demo = self.generate_demo_prediction(symbol)
                self._save_prediction_to_db(symbol, demo)
                return demo

            pred_5min = float(model.predict(last_features)[0])

            current_price = float(data['Close'].iloc[-1])
            confidence = float(self.calculate_confidence(data))
            signals = self.generate_signals(current_price, pred_5min, data)

            result = {
                'current_price': current_price,
                'predictions_5min': pred_5min,
                'predictions_15min': float(pred_5min * 1.002),
                'confidence': int(confidence),
                'signals': signals,
                'timestamp': datetime.now()
            }

            # Cache and save
            self.stock_data_cache[symbol] = result
            self._save_prediction_to_db(symbol, result)
            return result

        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            demo = self.generate_demo_prediction(symbol)
            self._save_prediction_to_db(symbol, demo)
            return demo

    def get_all_predictions(self, symbols):
        """Get predictions for multiple stocks"""
        predictions = {}
        for symbol in symbols:
            predictions[symbol] = self.predict_stock(symbol)
        return predictions

    # ----------------- DB read helpers -----------------
    def get_recent_predictions(self, symbol, limit=50):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, current_price, pred_5min, pred_15min, confidence, signals FROM predictions WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit))
        rows = cur.fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append({
                'timestamp': r[0],
                'current_price': r[1],
                'pred_5min': r[2],
                'pred_15min': r[3],
                'confidence': r[4],
                'signals': json.loads(r[5]) if r[5] else []
            })
        return results

# Initialize predictor
predictor = MultiStockPredictor()

# Flask Routes
@app.route('/')
def index():
    return render_template_string(REAL_TIME_HTML)

@app.route('/api/live/<symbol>')
def get_live_data(symbol):
    """API endpoint for live stock data"""
    try:
        prediction_data = predictor.predict_stock(symbol.upper())
        # convert timestamp to iso
        if isinstance(prediction_data.get('timestamp'), datetime):
            prediction_data['timestamp'] = prediction_data['timestamp'].isoformat()
        return jsonify(prediction_data)
    except Exception as e:
        print(f"API error for {symbol}: {e}")
        demo = predictor.generate_demo_prediction(symbol.upper())
        if isinstance(demo.get('timestamp'), datetime):
            demo['timestamp'] = demo['timestamp'].isoformat()
        return jsonify(demo)

@app.route('/api/batch')
def get_batch_data():
    """API endpoint for batch stock data"""
    symbols = request.args.get('symbols', '').upper().split(',')
    symbols = [s.strip() for s in symbols if s.strip()]
    if not symbols:
        symbols = ['AAPL', 'MSFT', 'TSLA']

    predictions = predictor.get_all_predictions(symbols)
    # convert timestamps
    for sym, p in predictions.items():
        if isinstance(p.get('timestamp'), datetime):
            p['timestamp'] = p['timestamp'].isoformat()
    return jsonify(predictions)

@app.route('/api/stats')
def get_stats():
    """API endpoint for system statistics"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total_preds = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM models")
        total_models = cur.fetchone()[0]
        conn.close()

        return jsonify({
            'total_models': total_models,
            'total_predictions': total_preds,
            'cache_size': len(predictor.stock_data_cache),
            'status': 'running'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/history/<symbol>')
def get_history(symbol):
    """Return recent predictions from DB"""
    try:
        rows = predictor.get_recent_predictions(symbol.upper())
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)})
    
# =========================================================
# 🧠 DATABASE STRUCTURE (Auto-created if not exists)
# =========================================================
def init_database():
    conn = sqlite3.connect("stocks.db")
    cursor = conn.cursor()

    # Table to store predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        current_price REAL,
        predicted_5min REAL,
        predicted_15min REAL,
        confidence REAL,
        signal TEXT
    )
    """)

    # Table to store model info
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        last_trained TEXT
    )
    """)

    conn.commit()
    conn.close()

# Initialize database at app startup
init_database()


# =========================================================
# 📊 HISTORY TABLE ROUTE (View Past Predictions)
# =========================================================
@app.route("/history")
def history():
    conn = sqlite3.connect("stocks.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()

    # If table is empty
    if df.empty:
        return "<h3 style='text-align:center;'>📭 No prediction history found yet!</h3>"

    # Convert to pretty HTML table
    html_table = df.to_html(classes='table table-striped table-bordered', index=False)
    page = f"""
    <html>
    <head>
        <title>Stock Prediction History</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <style>
            body {{
                background-color: #f8f9fa;
                padding: 30px;
            }}
            h2 {{
                text-align: center;
                margin-bottom: 30px;
            }}
            table {{
                background-color: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <h2>📊 Prediction History</h2>
        {html_table}
        <div style='text-align:center; margin-top:20px;'>
            <a href='/download_history' class='btn btn-success'>⬇️ Download as CSV</a>
            <a href='/' class='btn btn-secondary'>🏠 Back to Dashboard</a>
        </div>
    </body>
    </html>
    """
    return page


# =========================================================
# 📥 DOWNLOAD HISTORY AS CSV
# =========================================================
@app.route("/download_history")
def download_history():
    conn = sqlite3.connect("stocks.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        return "No data to download."

    filename = "prediction_history.csv"
    df.to_csv(filename, index=False)
    return send_file(filename, as_attachment=True)
    

if __name__ == '__main__':
    print("STARTING MULTI-STOCK REAL-TIME PREDICTION SYSTEM")
    print("=" * 50)
    print("Features:")
    print("- Real-time graphs for all tracked stocks")
    print("- Simultaneous multi-stock monitoring")
    print("- Live price updates every 3-10 seconds")
    print("- Individual charts with predictions")
    print("- Trading signals for each stock")
    print("- Market summary statistics")
    print("")
    print("Server: http://localhost:5000")
    print("")
    print("Pre-training models for popular stocks...")

    # Pre-train for popular stocks (non-blocking thread-safe)
    popular = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA']
    for symbol in popular:
        print(f"  Training {symbol}...", end=" ")
        if predictor.train_model(symbol):
            print("✓")
        else:
            print("✗ (using demo mode)")
        time.sleep(0.3)

    print("")
    print("System ready! Open http://localhost:5000 in your browser")
    print("=" * 50)



    
app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
