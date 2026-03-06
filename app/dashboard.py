from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("Demand Forecast and Inventory Dashboard")

forecast_path = Path("reports/demand_forecast.csv")
inventory_path = Path("reports/inventory_recommendations.csv")
backtest_path = Path("reports/forecast_actual_vs_predicted.csv")

if not forecast_path.exists() or not inventory_path.exists():
    st.error(
        "Missing forecast outputs. Run `make train` first to generate report files."
    )
    st.stop()

forecast_df = pd.read_csv(forecast_path)
inventory_df = pd.read_csv(inventory_path)

forecast_df["ForecastWeek"] = pd.to_datetime(forecast_df["ForecastWeek"])
forecast_df["series_id"] = (
    forecast_df["StockCode"].astype(str) + " | " + forecast_df["Country"].astype(str)
)
inventory_df["series_id"] = (
    inventory_df["StockCode"].astype(str) + " | " + inventory_df["Country"].astype(str)
)

series_options = sorted(forecast_df["series_id"].unique().tolist())
selected_series = st.selectbox("Select Product-Country Series", series_options)

sel_forecast = forecast_df[forecast_df["series_id"] == selected_series].copy()
sel_inventory = inventory_df[inventory_df["series_id"] == selected_series].copy()

left, right = st.columns([2, 1])
with left:
    st.subheader("Forecast Curve")
    chart_df = sel_forecast[["ForecastWeek", "ForecastQty"]].set_index("ForecastWeek")
    st.line_chart(chart_df)
    st.dataframe(
        sel_forecast[
            ["StockCode", "Country", "Description", "ForecastWeek", "ForecastQty"]
        ].sort_values("ForecastWeek"),
        use_container_width=True,
    )

with right:
    st.subheader("Inventory Recommendation")
    if sel_inventory.empty:
        st.warning("No inventory recommendation found for this series.")
    else:
        row = sel_inventory.iloc[0]
        st.metric("Recommended Inventory Qty", int(row["RecommendedInventoryQty"]))
        st.metric("Reorder Point Qty", float(row["ReorderPointQty"]))
        st.metric("Safety Stock Qty", float(row["SafetyStockQty"]))
        st.metric("Avg Weekly Forecast", float(row["AvgWeeklyForecastQty"]))
        st.dataframe(sel_inventory, use_container_width=True)

st.subheader("Global Backtest: Actual vs Predicted")
if backtest_path.exists():
    backtest_df = pd.read_csv(backtest_path)
    backtest_df["WeekStart"] = pd.to_datetime(backtest_df["WeekStart"])
    st.line_chart(backtest_df.set_index("WeekStart")[["actual", "prediction"]])
else:
    st.info("Run `make train` to generate reports/forecast_actual_vs_predicted.csv.")
