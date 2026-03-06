import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@dataclass
class DemandConfig:
    forecast_horizon_weeks: int = 8
    test_weeks: int = 8
    min_history_weeks: int = 12
    lead_time_weeks: int = 2
    review_period_weeks: int = 1
    service_level: float = 0.95
    top_series: int = 1000
    n_estimators: int = 40
    max_depth: int = 10
    min_samples_leaf: int = 5
    random_state: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a demand forecasting model and produce inventory recommendations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/processed/online_retail_data.csv"),
        help="Path to transaction-level input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument("--forecast-horizon-weeks", type=int, default=8)
    parser.add_argument("--test-weeks", type=int, default=8)
    parser.add_argument("--min-history-weeks", type=int, default=12)
    parser.add_argument("--lead-time-weeks", type=int, default=2)
    parser.add_argument("--review-period-weeks", type=int, default=1)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument(
        "--top-series",
        type=int,
        default=1000,
        help=(
            "Train/forecast only the top N product-country series by total demand. "
            "Set to 0 to use all series."
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    return parser.parse_args()


def load_and_clean_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Invoice" not in df.columns:
        raise ValueError("Expected 'Invoice' column in input data.")

    # Filter cancellation/returns and invalid pricing rows.
    df = df[~df["Invoice"].astype(str).str.startswith("C", na=False)].copy()
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    df = df.dropna(subset=["InvoiceDate", "StockCode"]).copy()

    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Country"] = df["Country"].fillna("Unknown").astype(str)
    df["Description"] = df["Description"].fillna("Unknown").astype(str)

    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    # Use Monday week-start bins consistently for aggregation and reindexing.
    df["WeekStart"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time

    weekly = (
        df.groupby(["StockCode", "Country", "WeekStart"], as_index=False)
        .agg(
            demand_qty=("Quantity", "sum"),
            avg_price=("Price", "mean"),
            description=("Description", "first"),
        )
        .sort_values(["StockCode", "Country", "WeekStart"])
    )

    filled = []
    for (stock, country), group in weekly.groupby(["StockCode", "Country"], sort=False):
        full_weeks = pd.date_range(
            group["WeekStart"].min(), group["WeekStart"].max(), freq="W-MON"
        )
        reindexed = (
            group.set_index("WeekStart")
            .reindex(full_weeks)
            .rename_axis("WeekStart")
            .reset_index()
        )
        reindexed["StockCode"] = stock
        reindexed["Country"] = country
        reindexed["description"] = (
            reindexed["description"]
            .astype("string")
            .ffill()
            .bfill()
            .fillna("Unknown")
            .astype(str)
        )
        reindexed["demand_qty"] = reindexed["demand_qty"].fillna(0.0)
        reindexed["avg_price"] = reindexed["avg_price"].ffill().bfill()
        reindexed["avg_price"] = reindexed["avg_price"].fillna(0.0)
        filled.append(reindexed)

    weekly_filled = pd.concat(filled, ignore_index=True)
    return weekly_filled.sort_values(["StockCode", "Country", "WeekStart"])


def build_features(weekly: pd.DataFrame, min_history_weeks: int) -> pd.DataFrame:
    df = weekly.copy()

    group_keys = ["StockCode", "Country"]
    df["lag_1"] = df.groupby(group_keys)["demand_qty"].shift(1)
    df["lag_2"] = df.groupby(group_keys)["demand_qty"].shift(2)
    df["lag_4"] = df.groupby(group_keys)["demand_qty"].shift(4)
    df["rolling_mean_4"] = df.groupby(group_keys)["demand_qty"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    )
    df["rolling_std_4"] = (
        df.groupby(group_keys)["demand_qty"]
        .transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).std())
        .fillna(0.0)
    )
    df["price_lag_1"] = df.groupby(group_keys)["avg_price"].shift(1)

    df["week_of_year"] = df["WeekStart"].dt.isocalendar().week.astype(int)
    df["month"] = df["WeekStart"].dt.month
    df["quarter"] = df["WeekStart"].dt.quarter

    history_counts = df.groupby(group_keys)["demand_qty"].transform("count")
    df = df[history_counts >= min_history_weeks].copy()

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "price_lag_1",
        "week_of_year",
        "month",
        "quarter",
    ]

    df = df.dropna(subset=feature_cols).copy()
    return df


def select_top_series(weekly_df: pd.DataFrame, top_series: int) -> pd.DataFrame:
    if top_series <= 0:
        return weekly_df

    totals = (
        weekly_df.groupby(["StockCode", "Country"], as_index=False)["demand_qty"]
        .sum()
        .sort_values("demand_qty", ascending=False)
        .head(top_series)
    )
    keys = totals[["StockCode", "Country"]]
    return weekly_df.merge(keys, on=["StockCode", "Country"], how="inner")


def make_time_split(df: pd.DataFrame, test_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_week = df["WeekStart"].max()
    cutoff = last_week - pd.Timedelta(weeks=test_weeks)
    train = df[df["WeekStart"] <= cutoff].copy()
    test = df[df["WeekStart"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError(
            "Time split produced empty train/test set. Reduce --test-weeks or --min-history-weeks."
        )
    return train, test


def encode_categories(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = train.copy()
    test = test.copy()

    stock_categories = pd.Categorical(train["StockCode"])
    country_categories = pd.Categorical(train["Country"])

    stock_map = {cat: i for i, cat in enumerate(stock_categories.categories)}
    country_map = {cat: i for i, cat in enumerate(country_categories.categories)}

    train["stock_code_cat"] = train["StockCode"].map(stock_map).astype(int)
    train["country_cat"] = train["Country"].map(country_map).astype(int)

    test["stock_code_cat"] = test["StockCode"].map(stock_map).fillna(-1).astype(int)
    test["country_cat"] = test["Country"].map(country_map).fillna(-1).astype(int)

    category_maps = {
        "stock_code": stock_map,
        "country": country_map,
    }
    return train, test, category_maps


def train_model(
    train_df: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> RandomForestRegressor:
    feature_cols = [
        "stock_code_cat",
        "country_cat",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "price_lag_1",
        "week_of_year",
        "month",
        "quarter",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["demand_qty"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestRegressor, test_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    feature_cols = [
        "stock_code_cat",
        "country_cat",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "price_lag_1",
        "week_of_year",
        "month",
        "quarter",
    ]
    preds = np.clip(model.predict(test_df[feature_cols]), 0, None)

    mae = mean_absolute_error(test_df["demand_qty"], preds)
    rmse = root_mean_squared_error(test_df["demand_qty"], preds)

    total_actual = test_df["demand_qty"].sum()
    wmape = (
        np.abs(test_df["demand_qty"] - preds).sum() / total_actual if total_actual > 0 else np.nan
    )

    scored = test_df[["StockCode", "Country", "WeekStart", "demand_qty"]].copy()
    scored["prediction"] = preds
    scored["residual"] = scored["demand_qty"] - scored["prediction"]

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wmape": float(wmape),
        "test_rows": int(len(test_df)),
    }, scored


def build_series_state(weekly_df: pd.DataFrame) -> dict:
    state = {}
    for (stock, country), group in weekly_df.groupby(["StockCode", "Country"], sort=False):
        grp = group.sort_values("WeekStart")
        state[(stock, country)] = {
            "history_demand": list(grp["demand_qty"].astype(float).values),
            "history_price": list(grp["avg_price"].astype(float).values),
            "last_week": grp["WeekStart"].max(),
            "description": grp["description"].dropna().iloc[-1],
        }
    return state


def recursive_forecast(
    model: RandomForestRegressor,
    weekly_df: pd.DataFrame,
    category_maps: dict,
    horizon_weeks: int,
) -> pd.DataFrame:
    stock_map = category_maps["stock_code"]
    country_map = category_maps["country"]
    state = build_series_state(weekly_df)

    records = []
    for (stock, country), info in state.items():
        demand_hist = info["history_demand"][:]
        price_hist = info["history_price"][:]
        current_week = info["last_week"]

        stock_cat = stock_map.get(stock, -1)
        country_cat = country_map.get(country, -1)

        for _ in range(horizon_weeks):
            next_week = current_week + pd.Timedelta(weeks=1)
            lag_1 = demand_hist[-1] if len(demand_hist) >= 1 else 0.0
            lag_2 = demand_hist[-2] if len(demand_hist) >= 2 else lag_1
            lag_4 = demand_hist[-4] if len(demand_hist) >= 4 else lag_2
            recent = demand_hist[-4:] if len(demand_hist) >= 4 else demand_hist
            rolling_mean_4 = float(np.mean(recent)) if recent else 0.0
            rolling_std_4 = float(np.std(recent)) if recent else 0.0
            price_lag_1 = price_hist[-1] if price_hist else 0.0

            row = pd.DataFrame(
                [
                    {
                        "stock_code_cat": stock_cat,
                        "country_cat": country_cat,
                        "lag_1": lag_1,
                        "lag_2": lag_2,
                        "lag_4": lag_4,
                        "rolling_mean_4": rolling_mean_4,
                        "rolling_std_4": rolling_std_4,
                        "price_lag_1": price_lag_1,
                        "week_of_year": int(next_week.isocalendar().week),
                        "month": int(next_week.month),
                        "quarter": int(next_week.quarter),
                    }
                ]
            )
            pred = float(np.clip(model.predict(row)[0], 0, None))

            records.append(
                {
                    "StockCode": stock,
                    "Country": country,
                    "Description": info["description"],
                    "ForecastWeek": next_week,
                    "ForecastQty": pred,
                }
            )

            demand_hist.append(pred)
            price_hist.append(price_lag_1)
            current_week = next_week

    return pd.DataFrame(records)


def inventory_recommendations(
    forecast_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    config: DemandConfig,
) -> pd.DataFrame:
    z = NormalDist().inv_cdf(config.service_level)

    residual_std = (
        scored_df.groupby(["StockCode", "Country"]) ["residual"].std().fillna(0.0)
    )
    global_std = float(scored_df["residual"].std()) if len(scored_df) > 1 else 1.0

    weekly_summary = (
        forecast_df.groupby(["StockCode", "Country", "Description"], as_index=False)
        .agg(avg_weekly_forecast=("ForecastQty", "mean"))
    )

    recs = []
    for row in weekly_summary.itertuples(index=False):
        sigma = float(
            residual_std.get((row.StockCode, row.Country), np.nan)
        )
        if np.isnan(sigma) or sigma == 0:
            sigma = global_std if global_std > 0 else max(1.0, row.avg_weekly_forecast * 0.3)

        lead_time_demand = row.avg_weekly_forecast * config.lead_time_weeks
        safety_stock = z * sigma * np.sqrt(config.lead_time_weeks)
        reorder_point = lead_time_demand + safety_stock

        target_cover = config.lead_time_weeks + config.review_period_weeks
        target_stock_level = (
            row.avg_weekly_forecast * target_cover
            + z * sigma * np.sqrt(target_cover)
        )

        recs.append(
            {
                "StockCode": row.StockCode,
                "Country": row.Country,
                "Description": row.Description,
                "AvgWeeklyForecastQty": round(float(row.avg_weekly_forecast), 2),
                "LeadTimeDemandQty": round(float(lead_time_demand), 2),
                "SafetyStockQty": round(float(safety_stock), 2),
                "ReorderPointQty": round(float(reorder_point), 2),
                "TargetStockLevelQty": round(float(target_stock_level), 2),
                "RecommendedInventoryQty": int(np.ceil(target_stock_level)),
                "ServiceLevel": config.service_level,
                "LeadTimeWeeks": config.lead_time_weeks,
                "ReviewPeriodWeeks": config.review_period_weeks,
            }
        )

    rec_df = pd.DataFrame(recs)
    return rec_df.sort_values("RecommendedInventoryQty", ascending=False)


def main() -> None:
    args = parse_args()
    config = DemandConfig(
        forecast_horizon_weeks=args.forecast_horizon_weeks,
        test_weeks=args.test_weeks,
        min_history_weeks=args.min_history_weeks,
        lead_time_weeks=args.lead_time_weeks,
        review_period_weeks=args.review_period_weeks,
        service_level=args.service_level,
        top_series=args.top_series,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_clean_transactions(args.input)
    weekly_df = aggregate_weekly(raw_df)
    weekly_df = select_top_series(weekly_df, top_series=config.top_series)
    feature_df = build_features(weekly_df, min_history_weeks=config.min_history_weeks)

    train_df, test_df = make_time_split(feature_df, test_weeks=config.test_weeks)
    train_df, test_df, category_maps = encode_categories(train_df, test_df)

    model = train_model(
        train_df,
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
    )
    metrics, scored_df = evaluate_model(model, test_df)

    forecast_df = recursive_forecast(
        model,
        weekly_df=weekly_df,
        category_maps=category_maps,
        horizon_weeks=config.forecast_horizon_weeks,
    )

    rec_df = inventory_recommendations(forecast_df, scored_df, config)

    model_path = args.model_dir / "demand_model.joblib"
    meta_path = args.model_dir / "model_metadata.json"
    metrics_path = args.output_dir / "metrics.json"
    forecast_path = args.output_dir / "demand_forecast.csv"
    inventory_path = args.output_dir / "inventory_recommendations.csv"
    backtest_path = args.output_dir / "backtest_predictions.csv"
    plot_ready_path = args.output_dir / "forecast_actual_vs_predicted.csv"

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "category_maps": category_maps,
                "config": config.__dict__,
                "train_end_week": str(train_df["WeekStart"].max().date()),
                "test_start_week": str(test_df["WeekStart"].min().date()),
                "test_end_week": str(test_df["WeekStart"].max().date()),
            },
            f,
            indent=2,
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    forecast_df.to_csv(forecast_path, index=False)
    rec_df.to_csv(inventory_path, index=False)
    scored_df.to_csv(backtest_path, index=False)
    plot_ready = (
        scored_df.groupby("WeekStart", as_index=False)
        .agg(actual=("demand_qty", "sum"), prediction=("prediction", "sum"))
        .sort_values("WeekStart")
    )
    plot_ready.to_csv(plot_ready_path, index=False)

    print("Training and forecasting complete")
    print(f"Model saved: {model_path}")
    print(f"Forecasts saved: {forecast_path}")
    print(f"Inventory recommendations saved: {inventory_path}")
    print(f"Plot-ready actual vs predicted saved: {plot_ready_path}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
