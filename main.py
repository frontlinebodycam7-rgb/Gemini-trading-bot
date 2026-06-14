import numpy as np
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Gemini Trading Analytics Engine")

# Data format structure check karne ke liye Schema
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
        # 1. Incoming JSON data ko Pandas DataFrame mein convert karein
        raw_candles = [candle.model_dump() for candle in data.candles]
        df = pd.DataFrame(raw_candles)
        
        if df.empty or len(df) < 26:
            raise HTTPException(status_code=400, detail="Data kam hai! Kam se kam 26 candles chahiye.")

        # Data sorting
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 2. FEATURE EXTRACTION PIPELINE
        df["EMA_12"] = ta.ema(df["close"], length=12)
        df["EMA_26"] = ta.ema(df["close"], length=26)

        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is not None:
            df["MACD"] = macd_df["MACD_12_26_9"]
            df["MACD_Signal"] = macd_df["MACDS_12_26_9"]

        df["RSI_14"] = ta.rsi(df["close"], length=14)

        bbands_df = ta.bbands(df["close"], length=20, std=2)
        if bbands_df is not None:
            df["BB_Upper"] = bbands_df["BBU_20_2.0"]
            df["BB_Lower"] = bbands_df["BBL_20_2.0"]

        df["Log_Returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Cleanup
        df.dropna(inplace=True)
        
        # 3. Forecast / Signal Engine (Rule-based Baseline)
        # Agar RSI < 35 aur Price Lower Bollinger Band ke paas hai -> BUY
        # Agar RSI > 65 aur Price Upper Bollinger Band ke paas hai -> SELL
        latest = df.iloc[-1]
        signal = "NEUTRAL"
        if latest["RSI_14"] < 35:
            signal = "BULLISH_BUY_ZONE"
        elif latest["RSI_14"] > 65:
            signal = "BEARISH_SELL_ZONE"

        # Output payload response
        return {
            "status": "Success",
            "latest_price": float(latest["close"]),
            "market_signal": signal,
            "extracted_features": {
                "RSI": float(latest["RSI_14"]),
                "EMA_12": float(latest["EMA_12"]),
                "EMA_26": float(latest["EMA_26"]),
                "BB_Upper": float(latest["BB_Upper"]),
                "BB_Lower": float(latest["BB_Lower"]),
                "Log_Return": float(latest["Log_Returns"])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
