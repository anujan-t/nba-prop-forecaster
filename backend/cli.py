import argparse
import json
from pathlib import Path

from .data import demo_frame, metadata, read_csv
from .engine import backtest, forecast


def main():
    parser = argparse.ArgumentParser(description="Courtside player-prop research CLI")
    parser.add_argument("command", choices=["demo", "forecast", "backtest"])
    parser.add_argument("--csv", type=Path, help="Use a validated game-log CSV instead of synthetic data")
    parser.add_argument("--output", type=Path, help="Save demo CSV or result JSON")
    parser.add_argument("--player", default="203999")
    parser.add_argument("--stat", default="points", choices=["points", "rebounds", "assists", "steals", "blocks"])
    parser.add_argument("--model", default="auto", choices=["auto", "empirical", "normal", "poisson", "negative_binomial"])
    parser.add_argument("--line", type=float, default=26.5)
    parser.add_argument("--lookback", type=int, default=30)
    args = parser.parse_args()
    frame = read_csv(args.csv.read_text(encoding="utf-8-sig")) if args.csv else demo_frame()
    if args.command == "demo":
        output = frame.to_csv(index=False)
    else:
        from .app import BacktestRequest, ForecastRequest
        request = {
            "player_id": args.player, "stat": args.stat, "model": args.model,
            "line": args.line, "lookback": args.lookback, "use_context": True,
        }
        if args.command == "backtest":
            result = backtest(frame, BacktestRequest(**request).model_dump())
        else:
            info = metadata(frame, "cli", args.csv.name if args.csv else "Synthetic season", "csv" if args.csv else "synthetic")
            result = forecast(frame, info, ForecastRequest(**request).model_dump())
        output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Saved {args.output.resolve()}")
    else:
        print(output)


if __name__ == "__main__":
    main()
