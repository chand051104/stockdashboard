from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import backend.database as database
import backend.models as models
import backend.crud as crud
import backend.ai as ai
import backend.schemas as schemas
import asyncio
import json
from typing import Dict, Set
from datetime import datetime
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Enhanced Stock Market Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - FIXED: using "static" directory as per user's structure
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.client_subscriptions[websocket] = set()
        logger.info("New WebSocket connection established")

    def disconnect(self, websocket: WebSocket):
        # Remove from all ticker subscriptions
        if websocket in self.client_subscriptions:
            for ticker in self.client_subscriptions[websocket]:
                if ticker in self.active_connections:
                    self.active_connections[ticker].discard(websocket)
                    if not self.active_connections[ticker]:
                        del self.active_connections[ticker]
            del self.client_subscriptions[websocket]
        logger.info("WebSocket connection closed")

    def subscribe_to_ticker(self, websocket: WebSocket, ticker: str):
        if ticker not in self.active_connections:
            self.active_connections[ticker] = set()
        self.active_connections[ticker].add(websocket)
        self.client_subscriptions[websocket].add(ticker)

    def unsubscribe_from_ticker(self, websocket: WebSocket, ticker: str):
        if ticker in self.active_connections:
            self.active_connections[ticker].discard(websocket)
            if not self.active_connections[ticker]:
                del self.active_connections[ticker]
        if websocket in self.client_subscriptions:
            self.client_subscriptions[websocket].discard(ticker)

    async def send_price_update(self, ticker: str, data: dict):
        if ticker in self.active_connections:
            disconnected = []
            for connection in self.active_connections[ticker]:
                try:
                    await connection.send_json({
                        "type": "price_update",
                        "ticker": ticker,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    disconnected.append(connection)

            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)

manager = ConnectionManager()

# Background task for real-time price updates
async def fetch_and_broadcast_prices():
    """Background task to fetch real-time prices and broadcast to WebSocket clients"""
    while True:
        try:
            # Get all subscribed tickers
            subscribed_tickers = set(manager.active_connections.keys())

            if subscribed_tickers:
                # Fetch current prices using yfinance
                for ticker in subscribed_tickers:
                    try:
                        stock = yf.Ticker(ticker)
                        # Get fast info (real-time data)
                        fast_info = stock.fast_info

                        price_data = {
                            "price": round(fast_info.last_price, 2),
                            "previous_close": round(fast_info.previous_close, 2),
                            "change": round(fast_info.last_price - fast_info.previous_close, 2),
                            "change_percent": round(((fast_info.last_price - fast_info.previous_close) / fast_info.previous_close) * 100, 2) if fast_info.previous_close != 0 else 0,
                            "volume": getattr(fast_info, 'last_volume', 0),
                            "market_cap": getattr(fast_info, 'market_cap', 0)
                        }

                        await manager.send_price_update(ticker, price_data)

                    except Exception as e:
                        logger.error(f"Error fetching price for {ticker}: {str(e)}")

            await asyncio.sleep(5)  # Update every 5 seconds

        except Exception as e:
            logger.error(f"Error in price broadcast loop: {str(e)}")
            await asyncio.sleep(10)

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_and_broadcast_prices())

# Serve the main page - FIXED: using correct path
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/companies")
def get_companies():
    return crud.get_companies()

@app.get("/historical/{ticker}")
def get_historical(ticker: str, period: str = "1y", db=Depends(database.get_db)):
    """Get historical data with support for different time periods"""
    try:
        crud.fetch_and_store_historical(db, ticker, period)
        result = crud.get_historical_with_metrics(db, ticker, period)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{ticker}")
def predict_price(ticker: str, model: str = "ensemble", db=Depends(database.get_db)):
    """Get price prediction using specified model"""
    try:
        if model == "all":
            predictions = ai.get_model_predictions(db, ticker)
            return {"ticker": ticker, "predictions": predictions}
        else:
            price = ai.predict_next_day(db, ticker, model)
            return {"ticker": ticker, "model": model, "predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/realtime/{ticker}")
async def get_realtime_price(ticker: str):
    """Get current real-time price for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        fast_info = stock.fast_info

        return {
            "ticker": ticker,
            "price": round(fast_info.last_price, 2),
            "previous_close": round(fast_info.previous_close, 2),
            "change": round(fast_info.last_price - fast_info.previous_close, 2),
            "change_percent": round(((fast_info.last_price - fast_info.previous_close) / fast_info.previous_close) * 100, 2) if fast_info.previous_close != 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "subscribe":
                ticker = message["ticker"].upper()
                manager.subscribe_to_ticker(websocket, ticker)
                await websocket.send_json({
                    "type": "subscribed",
                    "ticker": ticker,
                    "message": f"Subscribed to {ticker}"
                })

            elif message["type"] == "unsubscribe":
                ticker = message["ticker"].upper()
                manager.unsubscribe_from_ticker(websocket, ticker)
                await websocket.send_json({
                    "type": "unsubscribed", 
                    "ticker": ticker,
                    "message": f"Unsubscribed from {ticker}"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.get("/time-periods")
def get_time_periods():
    """Get available time periods for historical data"""
    return {
        "periods": [
            {"value": "1d", "label": "1 Day", "yfinance": "1d"},
            {"value": "5d", "label": "5 Days", "yfinance": "5d"},
            {"value": "1mo", "label": "1 Month", "yfinance": "1mo"},
            {"value": "3mo", "label": "3 Months", "yfinance": "3mo"},
            {"value": "6mo", "label": "6 Months", "yfinance": "6mo"},
            {"value": "1y", "label": "1 Year", "yfinance": "1y"},
            {"value": "2y", "label": "2 Years", "yfinance": "2y"},
            {"value": "5y", "label": "5 Years", "yfinance": "5y"},
            {"value": "10y", "label": "10 Years", "yfinance": "10y"},
            {"value": "ytd", "label": "Year to Date", "yfinance": "ytd"},
            {"value": "max", "label": "Max", "yfinance": "max"}
        ]
    }

@app.get("/market-status")
async def get_market_status():
    """Get current market status"""
    try:
        import pytz
        from datetime import datetime

        # US market hours (Eastern Time)
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)

        # Market is open Monday-Friday 9:30 AM - 4:00 PM ET
        is_weekday = now.weekday() < 5
        market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

        is_market_hours = market_open_time <= now <= market_close_time
        market_open = is_weekday and is_market_hours

        return {
            "is_open": market_open,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "next_open": "Next trading day 9:30 AM ET" if not market_open else None,
            "next_close": market_close_time.strftime("%H:%M:%S") if market_open else None
        }
    except Exception as e:
        return {
            "is_open": None,
            "error": str(e),
            "current_time": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
