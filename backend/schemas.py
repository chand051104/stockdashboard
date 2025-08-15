
from pydantic import BaseModel
from datetime import date
from typing import List, Optional, Dict, Any

class StockData(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None

    class Config:
        orm_mode = True

class StockMetrics(BaseModel):
    current_price: Optional[float] = None
    period_change: Optional[float] = None
    period_change_percent: Optional[float] = None
    period_high: Optional[float] = None
    period_low: Optional[float] = None
    avg_volume: Optional[float] = None
    current_volume: Optional[int] = None
    volatility: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi: Optional[float] = None
    trend: Optional[str] = None

class HistoricalResponse(BaseModel):
    metrics: Dict[str, Any]  # Changed to Dict for flexibility
    data: List[StockData]
    period: Optional[str] = None
    ticker: Optional[str] = None
    total_records: Optional[int] = None

class Prediction(BaseModel):
    ticker: str
    predicted_price: Optional[float] = None
    model: Optional[str] = None

class PredictionAll(BaseModel):
    ticker: str
    predictions: Dict[str, Optional[float]]

class Company(BaseModel):
    ticker: str
    name: str
    sector: Optional[str] = "Unknown"

class TimePeriod(BaseModel):
    value: str
    label: str
    yfinance: str

class TimePeriodsResponse(BaseModel):
    periods: List[TimePeriod]

class MarketStatus(BaseModel):
    is_open: Optional[bool] = None
    current_time: str
    next_open: Optional[str] = None
    next_close: Optional[str] = None
    error: Optional[str] = None

class RealTimePrice(BaseModel):
    ticker: str
    price: float
    previous_close: float
    change: float
    change_percent: float
    timestamp: str

class WSMessage(BaseModel):
    type: str
    ticker: Optional[str] = None
    message: Optional[str] = None

class WSPriceUpdate(BaseModel):
    type: str
    ticker: str
    data: Dict[str, Any]
    timestamp: str
