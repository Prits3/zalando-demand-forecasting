PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
VENV_DIR ?= .venv
UVICORN ?= uvicorn
STREAMLIT ?= streamlit

.PHONY: help venv install train train-fast viz api dashboard clean

help:
	@echo "Available targets:"
	@echo "  make venv        Create virtual environment (.venv)"
	@echo "  make install     Install dependencies from requirements.txt"
	@echo "  make train       Train model and generate forecast/inventory outputs"
	@echo "  make train-fast  Faster training preset for quick demos"
	@echo "  make viz         Generate forecast/inventory figures"
	@echo "  make api         Run FastAPI prediction service"
	@echo "  make dashboard   Run Streamlit dashboard"
	@echo "  make clean       Remove generated artifacts"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install:
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) src/demand_inventory_system.py \
		--input data/raw/processed/online_retail_data.csv \
		--output-dir reports \
		--model-dir models \
		--top-series 300 \
		--n-estimators 20

train-fast:
	$(PYTHON) src/demand_inventory_system.py \
		--input data/raw/processed/online_retail_data.csv \
		--output-dir reports \
		--model-dir models \
		--top-series 100 \
		--n-estimators 10 \
		--test-weeks 4

viz:
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; import matplotlib.pyplot as plt; fig_dir=Path('reports/figures'); fig_dir.mkdir(parents=True, exist_ok=True); backtest=pd.read_csv('reports/backtest_predictions.csv'); backtest['WeekStart']=pd.to_datetime(backtest['WeekStart']); trend=backtest.groupby('WeekStart', as_index=False).agg(actual=('demand_qty','sum'), prediction=('prediction','sum')).sort_values('WeekStart'); plt.figure(figsize=(12,6)); plt.plot(trend['WeekStart'], trend['actual'], label='Actual Demand'); plt.plot(trend['WeekStart'], trend['prediction'], label='Predicted Demand'); plt.legend(); plt.title('Demand Forecast vs Actual'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(fig_dir/'forecast_plot.png', dpi=160); plt.close(); inv=pd.read_csv('reports/inventory_recommendations.csv'); top=inv.sort_values('ReorderPointQty', ascending=False).head(10); plt.figure(figsize=(11,5)); plt.bar(top['StockCode'].astype(str), top['ReorderPointQty']); plt.title('Top Products by Reorder Point'); plt.xlabel('Product ID (StockCode)'); plt.ylabel('Reorder Point Qty'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(fig_dir/'inventory_reorder_top10.png', dpi=160); plt.close(); print('Saved reports/figures/forecast_plot.png and reports/figures/inventory_reorder_top10.png')"

api:
	$(UVICORN) api.app:app --reload

dashboard:
	$(STREAMLIT) run app/dashboard.py

clean:
	rm -f models/demand_model.joblib models/model_metadata.json
	rm -f reports/metrics.json reports/backtest_predictions.csv reports/demand_forecast.csv reports/inventory_recommendations.csv
	rm -f reports/figures/forecast_plot.png reports/figures/inventory_reorder_top10.png
