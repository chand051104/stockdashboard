
import yfinance as yf
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import backend.models as models
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Extended list of popular tickers
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", 
    "INTC", "AMD", "CRM", "PYPL", "ORCL", "CSCO", "QCOM", "AVGO", "TXN", "IBM", 
    "SHOP", "UBER", "LYFT", "ZOOM", "DOCU", "SQ", "ROKU", "TWTR", "SNAP", "PINS",
    "V", "MA", "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
    "JNJ", "PFE", "ABBV", "MRK", "UNH", "CVS", "AMGN", "GILD", "BMY", "LLY",
    "KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "CMCSA", "VZ", "T", "TMUS"
]

# Time period mapping for yfinance
PERIOD_MAPPING = {
    "1d": "1d",
    "5d": "5d", 
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "10y": "10y",
    "ytd": "ytd",
    "max": "max"
}

def get_companies():
    """Return list of popular stock tickers with company names"""
    companies = []
    for ticker in POPULAR_TICKERS[:20]:  # Limit to first 20 for performance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            companies.append({
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown")
            })
        except:
            companies.append({
                "ticker": ticker,
                "name": ticker,
                "sector": "Unknown"
            })
    return companies

def fetch_and_store_historical(db: Session, ticker: str, period: str = "1y"):
    """
    Enhanced function to fetch and store historical data with support for different time periods
    """
    try:
        # Map period to yfinance format
        yf_period = PERIOD_MAPPING.get(period, "1y")

        # Check if we have recent data
        latest_record = db.query(models.StockPrice).filter(
            models.StockPrice.ticker == ticker
        ).order_by(models.StockPrice.date.desc()).first()

        # Determine start date for fetching new data
        if latest_record and period in ["1d", "5d"]:
            # For short periods, always fetch fresh data
            start_date = datetime.now() - timedelta(days=7)
        elif latest_record:
            # For longer periods, only fetch data after the latest record
            start_date = latest_record.date + timedelta(days=1)
        else:
            # No existing data, fetch based on period
            if period == "1d":
                start_date = datetime.now() - timedelta(days=2)
            elif period == "5d":
                start_date = datetime.now() - timedelta(days=7)
            elif period == "1mo":
                start_date = datetime.now() - timedelta(days=35)
            elif period == "3mo":
                start_date = datetime.now() - timedelta(days=95)
            elif period == "6mo":
                start_date = datetime.now() - timedelta(days=185)
            elif period == "1y":
                start_date = datetime.now() - timedelta(days=370)
            else:
                start_date = datetime.now() - timedelta(days=365)

        # For very short periods, use interval parameter
        interval = "1m" if period == "1d" else "1d"

        # Fetch data from yfinance
        if period in ["1d"]:
            # Use download with interval for intraday data
            data = yf.download(ticker, start=start_date, interval=interval)
        else:
            # Use period parameter for longer timeframes
            data = yf.download(ticker, period=yf_period)

        if data.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return

        # Clean data
        data = data.dropna()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        if data.empty:
            logger.warning(f"No valid data after cleaning for {ticker}")
            return

        # Store new data
        new_records = 0
        for idx, row in data.iterrows():
            # Convert timezone-aware datetime to date
            if hasattr(idx, 'date'):
                date_only = idx.date()
            else:
                date_only = idx.date() if hasattr(idx, 'date') else idx

            # Check if record already exists
            existing = db.query(models.StockPrice).filter(
                models.StockPrice.ticker == ticker,
                models.StockPrice.date == date_only
            ).first()

            if existing:
                # Update existing record
                existing.open = float(row["Open"])
                existing.high = float(row["High"])
                existing.low = float(row["Low"])
                existing.close = float(row["Close"])
                existing.volume = int(row["Volume"])
            else:
                # Create new record
                db.add(models.StockPrice(
                    ticker=ticker,
                    date=date_only,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"])
                ))
                new_records += 1

        db.commit()
        logger.info(f"Stored {new_records} new records for {ticker}")

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        db.rollback()
        raise

def get_historical_with_metrics(db: Session, ticker: str, period: str = "1y"):
    """
    Enhanced function to get historical data with metrics, filtered by time period
    """
    try:
        # Calculate date range based on period
        end_date = datetime.now().date()

        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "5d":
            start_date = end_date - timedelta(days=5)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        elif period == "10y":
            start_date = end_date - timedelta(days=3650)
        elif period == "ytd":
            start_date = datetime(end_date.year, 1, 1).date()
        else:  # max
            start_date = None

        # Query records within date range
        query = db.query(models.StockPrice).filter(models.StockPrice.ticker == ticker)

        if start_date:
            query = query.filter(models.StockPrice.date >= start_date)

        records = query.order_by(models.StockPrice.date.asc()).all()

        if not records:
            logger.warning(f"No records found for {ticker} in period {period}")
            return None

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([{
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume
        } for r in records])

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Calculate period-specific metrics
        metrics = calculate_metrics(df, period)

        # Prepare response data
        response_data = []
        for _, row in df.iterrows():
            record = {
                "date": row["date"].isoformat(),
                "open": round(row["open"], 2),
                "high": round(row["high"], 2),
                "low": round(row["low"], 2),
                "close": round(row["close"], 2),
                "volume": int(row["volume"])
            }

            # Add technical indicators if available
            for indicator in ["ma20", "ma50", "rsi", "macd", "bb_upper", "bb_lower"]:
                if indicator in row and pd.notna(row[indicator]):
                    record[indicator] = round(row[indicator], 2)

            response_data.append(record)

        return {
            "metrics": metrics,
            "data": response_data,
            "period": period,
            "ticker": ticker,
            "total_records": len(response_data)
        }

    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the dataframe"""
    try:
        # Moving averages
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma50"] = df["close"].rolling(window=50).mean()
        df["ma200"] = df["close"].rolling(window=200).mean()

        # Exponential moving averages
        df["ema12"] = df["close"].ewm(span=12).mean()
        df["ema26"] = df["close"].ewm(span=26).mean()

        # RSI calculation
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        bb_period = 20
        bb_std = df["close"].rolling(bb_period).std()
        df["bb_upper"] = df["ma20"] + (bb_std * 2)
        df["bb_lower"] = df["ma20"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Price change indicators
        df["pct_change"] = df["close"].pct_change()
        df["price_range"] = df["high"] - df["low"]
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )

        return df

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return df

def calculate_metrics(df: pd.DataFrame, period: str) -> dict:
    """Calculate various metrics for the given period"""
    try:
        if df.empty:
            return {}

        # Basic price metrics
        current_price = df["close"].iloc[-1]
        period_start_price = df["close"].iloc[0]
        price_change = current_price - period_start_price
        price_change_pct = (price_change / period_start_price) * 100 if period_start_price != 0 else 0

        # High/Low metrics
        period_high = df["high"].max()
        period_low = df["low"].min()

        # Volume metrics
        avg_volume = df["volume"].mean()
        current_volume = df["volume"].iloc[-1] if len(df) > 0 else 0

        # Volatility
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility

        # Support and resistance levels
        support_level = df["low"].rolling(window=20).min().iloc[-1] if len(df) >= 20 else period_low
        resistance_level = df["high"].rolling(window=20).max().iloc[-1] if len(df) >= 20 else period_high

        metrics = {
            "current_price": round(current_price, 2),
            "period_change": round(price_change, 2),
            "period_change_percent": round(price_change_pct, 2),
            "period_high": round(period_high, 2),
            "period_low": round(period_low, 2),
            "avg_volume": round(avg_volume, 0),
            "current_volume": int(current_volume),
            "volatility": round(volatility, 2),
            "support_level": round(support_level, 2),
            "resistance_level": round(resistance_level, 2)
        }

        # Add period-specific metrics
        if len(df) >= 20:
            metrics["sma_20"] = round(df["ma20"].iloc[-1], 2) if pd.notna(df["ma20"].iloc[-1]) else None
            metrics["rsi"] = round(df["rsi"].iloc[-1], 2) if pd.notna(df["rsi"].iloc[-1]) else None

        if len(df) >= 50:
            metrics["sma_50"] = round(df["ma50"].iloc[-1], 2) if pd.notna(df["ma50"].iloc[-1]) else None

        # Trend analysis
        if len(df) >= 5:
            recent_closes = df["close"].tail(5)
            if all(recent_closes.iloc[i] <= recent_closes.iloc[i+1] for i in range(len(recent_closes)-1)):
                trend = "Bullish"
            elif all(recent_closes.iloc[i] >= recent_closes.iloc[i+1] for i in range(len(recent_closes)-1)):
                trend = "Bearish"
            else:
                trend = "Sideways"
            metrics["trend"] = trend

        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {}

def get_real_time_quote(ticker: str) -> Optional[dict]:
    """Get real-time quote for a ticker using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "previous_close": info.get("previousClose", 0),
            "open": info.get("regularMarketOpen", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "volume": info.get("regularMarketVolume", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0)
        }
    except Exception as e:
        logger.error(f"Error fetching real-time quote for {ticker}: {str(e)}")
        return None
