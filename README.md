# 📈 Enhanced Stock Market Dashboard

A modern, full-featured stock market dashboard with real-time price tracking, advanced AI predictions, and interactive charts. Built with FastAPI backend and vanilla JavaScript frontend.



## ✨ Features

### 📊 **Real-Time Data**
- **WebSocket-based live price updates** (5-second intervals)
- **Real-time market status** indicator (Open/Closed)
- **Live watchlist** with instant price changes
- **Toast notifications** for connection status

### 🤖 **Advanced AI Predictions**
- **5 Machine Learning Models:**
  - Linear Regression (baseline)
  - Random Forest (ensemble method)
  - XGBoost (gradient boosting)
  - LSTM Neural Network (deep learning)
  - Ensemble Model (weighted combination)
- **Technical indicators** integration for better predictions
- **Compare all models** side-by-side

### 📈 **Interactive Charts**
- **Multiple chart types:** Candlestick, Line, Area
- **11 time periods:** 1D, 5D, 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, YTD, MAX
- **Technical indicators:** MA20, MA50, RSI, MACD, Bollinger Bands
- **Volume analysis** with dedicated charts
- **Responsive design** for all devices

### 💼 **Trading Tools**
- **60+ Popular stocks** with company information
- **Key metrics dashboard:** High/Low, Volume, Volatility, Trend
- **Support/Resistance levels** calculation
- **Market trend analysis** (Bullish/Bearish/Sideways)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/chand051104/stockdashboard/>
   cd stockdashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**
   ```bash
   # Option 1: Direct Python
   cd backend
   python main.py

   # Option 2: Using uvicorn
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the dashboard**
   ```
   Open your browser to: http://localhost:8000
   ```

## 📁 Project Structure

```
enhanced-stock-dashboard/
├── backend/                    # Backend API
│   ├── __init__.py
│   ├── main.py                # FastAPI app with WebSocket
│   ├── ai.py                  # ML models for predictions
│   ├── crud.py                # Data fetching and processing
│   ├── database.py            # SQLAlchemy database config
│   ├── models.py              # Database models
│   └── schemas.py             # Pydantic response schemas
├── static/                     # Frontend assets
│   ├── index.html             # Main HTML template
│   ├── script.js              # JavaScript functionality
│   └── style.css              # Modern UI styling
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── stocks.db                   # SQLite database (auto-created)
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
# Database
DATABASE_URL=sqlite:///./stocks.db

# API Settings  
HOST=0.0.0.0
PORT=8000

# Market Data
UPDATE_INTERVAL=5  # seconds for real-time updates
```

### Customizing Stock Lists
Edit `POPULAR_TICKERS` in `backend/crud.py`:
```python
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    # Add your preferred stocks here
]
```

## 🎯 Usage Guide

### Basic Usage
1. **Search stocks** by ticker symbol (e.g., AAPL, TSLA)
2. **Select time period** using the sidebar buttons
3. **Enable real-time updates** with the green "Start Real-time" button
4. **Get AI predictions** by choosing a model and clicking "Predict Price"
5. **Toggle chart types** and technical indicators as needed

### Advanced Features
- **WebSocket subscriptions:** Multiple stocks in watchlist update simultaneously
- **Chart interactions:** Zoom, pan, and hover for detailed information
- **Prediction comparison:** Use "All Models" to compare different AI approaches
- **Technical analysis:** Enable/disable indicators based on your strategy

## 📊 API Endpoints

### REST API
```bash
GET  /                          # Main dashboard
GET  /companies                 # List of popular stocks
GET  /historical/{ticker}       # Historical data with metrics
GET  /predict/{ticker}          # AI price predictions
GET  /realtime/{ticker}         # Current price snapshot
GET  /time-periods              # Available time periods
GET  /market-status             # Market open/close status
```

### WebSocket
```bash
WS   /ws                        # Real-time price updates
```

**WebSocket Message Format:**
```json
// Subscribe to ticker
{"type": "subscribe", "ticker": "AAPL"}

// Unsubscribe from ticker  
{"type": "unsubscribe", "ticker": "AAPL"}

// Price update (server to client)
{
  "type": "price_update",
  "ticker": "AAPL", 
  "data": {
    "price": 150.25,
    "change": 2.15,
    "change_percent": 1.45
  },
  "timestamp": "2025-08-15T10:30:00"
}
```

## 🤖 AI Models Details

### Linear Regression
- **Use case:** Baseline predictions
- **Features:** SMA20, SMA50, RSI, Volume, Price change
- **Speed:** Fastest (~10ms)

### Random Forest  
- **Use case:** Robust predictions with feature importance
- **Features:** OHLCV + technical indicators
- **Speed:** Fast (~50ms)

### XGBoost
- **Use case:** High-performance gradient boosting
- **Features:** Same as Random Forest
- **Speed:** Medium (~100ms)

### LSTM Neural Network
- **Use case:** Time series pattern recognition
- **Features:** 60-day price sequences
- **Speed:** Slower (~2-3 seconds)

### Ensemble Model
- **Use case:** Best overall accuracy
- **Method:** Weighted average of multiple models
- **Weights:** Linear (20%), RF (40%), XGBoost (40%)

## 🎨 UI Features

### Modern Design
- **Glass-morphism** styling with backdrop blur
- **Gradient backgrounds** and smooth transitions
- **Font Awesome icons** for better UX
- **Responsive grid layout** for all screen sizes

### Interactive Elements
- **Period selector** buttons with active states
- **Chart type toggles** (Candlestick/Line/Area)
- **Real-time indicators** with color coding
- **Toast notifications** for user feedback

### Dark/Light Theme Support
The dashboard automatically adapts to system preferences and provides excellent contrast in both modes.

## 🔍 Technical Indicators

| Indicator | Purpose | Calculation |
|-----------|---------|-------------|
| **SMA20** | Short-term trend | 20-day Simple Moving Average |
| **SMA50** | Medium-term trend | 50-day Simple Moving Average |
| **RSI** | Momentum oscillator | 14-day Relative Strength Index |
| **MACD** | Trend and momentum | 12-day EMA - 26-day EMA |
| **Bollinger Bands** | Volatility bands | SMA20 ± (2 × Standard Deviation) |

## ⚡ Performance

### Optimizations
- **Database indexing** on ticker and date columns
- **Efficient data queries** with SQLAlchemy ORM
- **WebSocket connection pooling** for multiple clients
- **Frontend caching** of chart data
- **Asynchronous operations** for non-blocking requests

### Benchmarks
- **API Response Time:** < 200ms for historical data
- **WebSocket Latency:** < 50ms for price updates
- **ML Prediction Time:** 10ms - 3s depending on model
- **Frontend Rendering:** < 100ms for chart updates

## 🛠️ Development

### Running in Development Mode
```bash
# Backend with hot reload
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Frontend development (serve static files)
python -m http.server 8080 --directory static
```

### Adding New Features

**1. New AI Model:**
```python
# Add to backend/ai.py
def your_model_predict(self, db: Session, ticker: str):
    # Your model implementation
    return prediction

# Register in get_model_predictions()
models = {
    "Your Model": predictor.your_model_predict,
    # ... existing models
}
```

**2. New Technical Indicator:**
```python
# Add to backend/crud.py in calculate_technical_indicators()
df["your_indicator"] = calculate_your_indicator(df["close"])
```

**3. New Chart Type:**
```javascript
// Add to static/script.js in plotStockData()
if (currentChartType === "your_type") {
    tracePrice = {
        // Your chart configuration
    };
}
```

## 🧪 Testing

### Manual Testing
1. **API Tests:** Use browser or Postman to test endpoints
2. **WebSocket Tests:** Use WebSocket testing tools
3. **UI Tests:** Test across different browsers and devices
4. **Performance Tests:** Monitor response times under load

### Unit Tests (Future Enhancement)
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

**1. WebSocket Connection Failed**
```bash
# Check firewall settings
# Verify port 8000 is available
netstat -an | grep 8000

# Try different port
uvicorn backend.main:app --port 8001
```

**2. Stock Data Not Loading**
```bash
# Check internet connection
# Verify yfinance package version
pip install --upgrade yfinance

# Check API rate limits (Yahoo Finance)
```

**3. AI Predictions Failing**
```bash
# Check TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"

# Install missing ML dependencies  
pip install --upgrade scikit-learn xgboost tensorflow
```

**4. Charts Not Displaying**
```bash
# Check Plotly.js loading
# Open browser console for JavaScript errors
# Verify internet connection for CDN resources
```

### Debug Mode
```python
# Enable debug logging in main.py
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Monitoring

### Built-in Metrics
- WebSocket connection count
- API response times (logged)
- Database query performance
- ML model execution times

### Monitoring Dashboard (Future)
Consider adding tools like:
- **Prometheus + Grafana** for metrics
- **Sentry** for error tracking  
- **Uptime monitoring** for availability

## 🔐 Security Considerations

### Current Security
- **CORS enabled** for development (configure for production)
- **Input validation** via Pydantic schemas
- **SQL injection prevention** via SQLAlchemy ORM
- **No authentication required** (add for production use)

### Production Security (Recommended)
```python
# Add authentication
from fastapi.security import HTTPBearer

# Add rate limiting
from slowapi import Limiter

# Configure CORS properly
origins = ["https://yourdomain.com"]
```

## 🚀 Deployment

### Local Deployment
Already covered in Quick Start section.



## 🙏 Acknowledgments

### Data Sources
- **Yahoo Finance** (via yfinance) for stock market data
- **Alpha Vantage** alternative for premium features

### Technologies Used
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - SQL toolkit and ORM
- **Plotly.js** - Interactive charting library
- **TensorFlow** - Machine learning framework
- **XGBoost** - Gradient boosting library
- **scikit-learn** - Machine learning library

### Inspiration
- Modern trading platforms like Robinhood, E*TRADE
- Financial news websites like Yahoo Finance, Bloomberg
- Open source projects in fintech space



### Documentation
- **API Docs:** Available at `http://localhost:8000/docs` (Swagger UI)
- **ReDoc:** Available at `http://localhost:8000/redoc`


*Last updated: August 15, 2025*
