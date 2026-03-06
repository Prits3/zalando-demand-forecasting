import argparse
import json
from pathlib import Path

import joblib

from demand_inventory_system import (
    aggregate_weekly,
    load_and_clean_transactions,
    recursive_forecast,
    select_top_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demand forecasts from saved model.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/processed/online_retail_data.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/demand_model.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("models/model_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/demand_forecast.csv"))
    parser.add_argument("--forecast-horizon-weeks", type=int, default=8)
    parser.add_argument("--top-series", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    raw_df = load_and_clean_transactions(args.input)
    weekly_df = aggregate_weekly(raw_df)
    weekly_df = select_top_series(weekly_df, args.top_series)

    forecast_df = recursive_forecast(
        model=model,
        weekly_df=weekly_df,
        category_maps=metadata["category_maps"],
        horizon_weeks=args.forecast_horizon_weeks,
    )
    forecast_df.to_csv(args.output, index=False)

    print(f"Forecast saved to {args.output}")


if __name__ == "__main__":
    main()
