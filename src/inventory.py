import argparse
from pathlib import Path

import pandas as pd

from demand_inventory_system import DemandConfig, inventory_recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inventory recommendations from forecasts.")
    parser.add_argument("--forecast", type=Path, default=Path("reports/demand_forecast.csv"))
    parser.add_argument("--backtest", type=Path, default=Path("reports/backtest_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/inventory_recommendations.csv"))
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--lead-time-weeks", type=int, default=2)
    parser.add_argument("--review-period-weeks", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.read_csv(args.forecast)
    scored_df = pd.read_csv(args.backtest)

    config = DemandConfig(
        service_level=args.service_level,
        lead_time_weeks=args.lead_time_weeks,
        review_period_weeks=args.review_period_weeks,
    )

    rec_df = inventory_recommendations(forecast_df, scored_df, config)
    rec_df.to_csv(args.output, index=False)

    print(f"Inventory recommendations saved to {args.output}")


if __name__ == "__main__":
    main()
