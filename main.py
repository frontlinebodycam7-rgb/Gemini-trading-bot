import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Gemini Trading Analytics Engine")

class CandleData(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class TradeDataInput(BaseModel):
    candles: List[CandleData]

@app.get("/")
def home():
    return {"status": "Active", "message": "Gemini Quant Model is Running Live!"}

@app.post("/api/v1/analyze")
async def analyze_market_data(data: TradeDataInput):
    try:
        raw_candles = [candle.model_dump() for candle in data.candles]
        df = pd.DataFrame(raw_candles)
        
        if df.empty or len(df) < 26:
            raise HTTPException(status_code=400, detail="Data kam hai! Kam se kam 26 candles chahiye.")

        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- PURE PANDAS MATHEMATICAL FORMULAS ---
        # 1. EMA
        df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # 2. MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # 3. RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # 4. Bollinger Bands
        df["BB_Middle"] = df["close"].rolling(window=20).mean()
        std_dev = df["close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (std_dev * 2)
        df["BB_Lower"] = df["BB_Middle"] - (std_dev * 2)

        # 5. Log Returns
        df["Log_Returns"] = np.log(df["close"] / df["close"].shift(1))
        
        df.dropna(inplace=True)
        
        # Simple Forecast Signal Engine
        latest = df.iloc[-1]
        signal = "NEUTRAL"
        if latest["RSI_14"] < 35:
            signal = "BULLISH_BUY_ZONE"
        elif latest["RSI_14"] > 65:
            signal = "BEARISH_SELL_ZONE"

        return {
            "status": "Success",
            "latest_price": float(latest["close"]),
            "market_signal": signal,
            "extracted_features": {
                "RSI": float(latest["RSI_14"]) if not np.isnan(latest["RSI_14"]) else 50.0,
                "EMA_12": float(latest["EMA_12"]),
                "EMA_26": float(latest["EMA_26"]),
                "BB_Upper": float(latest["BB_Upper"]) if not np.isnan(latest["BB_Upper"]) else 0.0,
                "BB_Lower": float(latest["BB_Lower"]) if not np.isnan(latest["BB_Lower"]) else 0.0,
                "Log_Return": float(latest["Log_Returns"]) if not np.isnan(latest["Log_Returns"]) else 0.0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
