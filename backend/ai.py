import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from sqlalchemy.orm import Session
from backend.models import StockPrice
import warnings
warnings.filterwarnings('ignore')

Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def prepare_data(self, db: Session, ticker: str, days_back=252):
        """Prepare data with technical indicators for better predictions"""
        data = db.query(StockPrice).filter(
            StockPrice.ticker == ticker
        ).order_by(StockPrice.date.desc()).limit(days_back).all()

        if len(data) < 60:
            return None, None, None

        # Reverse to get chronological order
        data = data[::-1]

        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': d.date,
            'open': d.open,
            'high': d.high,
            'low': d.low,
            'close': d.close,
            'volume': d.volume
        } for d in data])

        # Add technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # RSI calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        bb_period = 20
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['sma_20'] + (bb_std * 2)
        df['bb_lower'] = df['sma_20'] - (bb_std * 2)

        # Price change percentage
        df['pct_change'] = df['close'].pct_change()

        # Remove NaN values
        df = df.dropna()

        if len(df) < 30:
            return None, None, None

        return df, len(data), ticker

    def linear_regression_predict(self, db: Session, ticker: str):
        """Enhanced linear regression with more features"""
        df, data_len, _ = self.prepare_data(db, ticker)
        if df is None:
            return None

        # Select features for linear regression
        features = ['sma_20', 'sma_50', 'rsi', 'volume', 'pct_change']
        available_features = [f for f in features if f in df.columns]

        if len(available_features) < 3:
            # Fallback to simple linear regression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['close'].values
        else:
            X = df[available_features].values
            y = df['close'].values

        # Remove any remaining NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]

        if len(X) < 10:
            return None

        model = LinearRegression().fit(X, y)

        if len(available_features) < 3:
            next_X = np.array([[len(df)]])
        else:
            # Use last known values for prediction
            next_X = X[-1:] 

        prediction = model.predict(next_X)[0]
        return round(float(prediction), 2)

    def random_forest_predict(self, db: Session, ticker: str):
        """Random Forest prediction with technical indicators"""
        df, data_len, _ = self.prepare_data(db, ticker)
        if df is None:
            return None

        # Select features
        features = ['open', 'high', 'low', 'volume', 'sma_20', 'sma_50', 'rsi', 'macd']
        available_features = [f for f in features if f in df.columns]

        if len(available_features) < 4:
            return None

        X = df[available_features].values
        y = df['close'].values

        # Remove NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]

        if len(X) < 20:
            return None

        # Split data for training (use 80% for training)
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]

        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        ).fit(X_train, y_train)

        # Predict next day using last available data
        prediction = model.predict(X[-1:].reshape(1, -1))[0]
        return round(float(prediction), 2)

    def xgboost_predict(self, db: Session, ticker: str):
        """XGBoost prediction"""
        df, data_len, _ = self.prepare_data(db, ticker)
        if df is None:
            return None

        features = ['open', 'high', 'low', 'volume', 'sma_20', 'sma_50', 'rsi', 'macd']
        available_features = [f for f in features if f in df.columns]

        if len(available_features) < 4:
            return None

        X = df[available_features].values
        y = df['close'].values

        # Remove NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]

        if len(X) < 20:
            return None

        # Split for training
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]

        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ).fit(X_train, y_train)

        prediction = model.predict(X[-1:].reshape(1, -1))[0]
        return round(float(prediction), 2)

    def lstm_predict(self, db: Session, ticker: str):
        """LSTM neural network prediction"""
        df, data_len, _ = self.prepare_data(db, ticker, days_back=500)
        if df is None or len(df) < 60:
            return None

        # Use closing prices for LSTM
        prices = df['close'].values.reshape(-1, 1)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Create sequences for LSTM (60 days to predict 1 day)
        sequence_length = 60
        X, y = [], []

        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])

        if len(X) < 20:
            return None

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model (reduced epochs for faster execution)
        model.fit(X, y, batch_size=32, epochs=10, verbose=0)

        # Make prediction
        last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
        prediction = model.predict(last_sequence, verbose=0)

        # Inverse transform to get actual price
        prediction = scaler.inverse_transform(prediction)[0][0]

        return round(float(prediction), 2)

# Main prediction functions
def predict_next_day(db: Session, ticker: str, model_type: str = "linear"):
    """Predict next day price using specified model"""
    predictor = StockPredictor()

    try:
        if model_type == "linear":
            return predictor.linear_regression_predict(db, ticker)
        elif model_type == "random_forest":
            return predictor.random_forest_predict(db, ticker)
        elif model_type == "xgboost":
            return predictor.xgboost_predict(db, ticker)
        elif model_type == "lstm":
            return predictor.lstm_predict(db, ticker)
        else:
            return predictor.linear_regression_predict(db, ticker)
    except Exception as e:
        print(f"Error in {model_type} prediction: {str(e)}")
        return None

def predict_ensemble(db: Session, ticker: str):
    """Ensemble prediction using multiple models"""
    predictor = StockPredictor()
    predictions = []

    # Get predictions from different models
    try:
        linear_pred = predictor.linear_regression_predict(db, ticker)
        if linear_pred: predictions.append(linear_pred)
    except: pass

    try:
        rf_pred = predictor.random_forest_predict(db, ticker)
        if rf_pred: predictions.append(rf_pred)
    except: pass

    try:
        xgb_pred = predictor.xgboost_predict(db, ticker)
        if xgb_pred: predictions.append(xgb_pred)
    except: pass

    if len(predictions) >= 2:
        # Return weighted average (give more weight to tree-based methods)
        weights = [0.2, 0.4, 0.4] if len(predictions) == 3 else [0.3, 0.7]
        ensemble_pred = sum(p * w for p, w in zip(predictions, weights[:len(predictions)]))
        return round(float(ensemble_pred), 2)
    elif len(predictions) == 1:
        return predictions[0]
    else:
        return None

def get_model_predictions(db: Session, ticker: str):
    """Get predictions from all available models"""
    predictor = StockPredictor()
    results = {}

    models = {
        "Linear Regression": predictor.linear_regression_predict,
        "Random Forest": predictor.random_forest_predict,
        "XGBoost": predictor.xgboost_predict,
        "LSTM": predictor.lstm_predict
    }

    for name, model_func in models.items():
        try:
            pred = model_func(db, ticker)
            results[name] = pred
        except Exception as e:
            results[name] = None
            print(f"Error in {name}: {str(e)}")

    # Add ensemble prediction
    ensemble_pred = predict_ensemble(db, ticker)
    results["Ensemble"] = ensemble_pred

    return results
