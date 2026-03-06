import argparse
from pathlib import Path

from demand_inventory_system import (
    aggregate_weekly,
    load_and_clean_transactions,
    select_top_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare weekly demand data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/processed/online_retail_data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/processed/weekly_demand.csv"),
    )
    parser.add_argument("--top-series", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_clean_transactions(args.input)
    weekly_df = aggregate_weekly(raw_df)
    weekly_df = select_top_series(weekly_df, args.top_series)
    weekly_df.to_csv(args.output, index=False)

    print(f"Weekly dataset saved to {args.output}")


if __name__ == "__main__":
    main()
