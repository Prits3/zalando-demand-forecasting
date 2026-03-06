from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Demand Forecast API")

FORECAST_PATH = Path("reports/demand_forecast.csv")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Demand Forecast API is running",
        "docs": "/docs",
        "predict_example": "/predict?product_id=85123A",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/predict")
def predict(product_id: str):
    if not FORECAST_PATH.exists():
        raise HTTPException(status_code=404, detail="Forecast file not found")

    df = pd.read_csv(FORECAST_PATH)
    result = df[df["StockCode"].astype(str) == str(product_id)]
    if result.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    return {
        "product_id": str(product_id),
        "forecasts": result[["ForecastWeek", "ForecastQty"]].to_dict(orient="records"),
    }
