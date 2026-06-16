import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_START_DATE = "2018-02-01"
FEATURE_COLUMNS = ["Open", "High", "Low", "Volume", "Volatility", "SMA_7"]
HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
]
HALVING_REWARDS = [25.0, 12.5, 6.25, 3.125]


@dataclass
class ForecastResult:
    latest_date: pd.Timestamp
    latest_close: float
    predictions: pd.DataFrame


def download_btc_history(start_date: str) -> pd.DataFrame:
    btc = yf.download("BTC-USD", start=start_date, auto_adjust=False, progress=False)
    btc = btc.reset_index()
    btc = btc[["Date", "Close", "High", "Low", "Open", "Volume"]]

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [column[0] if isinstance(column, tuple) else column for column in btc.columns]

    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    for column in numeric_cols:
        btc[column] = pd.to_numeric(btc[column], errors="coerce")

    btc["Date"] = pd.to_datetime(btc["Date"], errors="coerce").astype("datetime64[ns]")
    btc = btc.dropna(subset=["Date", *numeric_cols]).sort_values("Date").reset_index(drop=True)
    return btc


def download_fear_greed_history() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?date_format=%2701%2F01%2F2018%27&format=csv&limit=50000"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    content = response.content.decode("utf-8")
    start = content.find("fng_value")
    csv_data = content[start:]
    fear_greed = pd.read_csv(io.StringIO(csv_data))
    fear_greed = fear_greed.rename(
        columns={
            "fng_value": "date",
            "fng_classification": "fng_value",
            "date": "fng_classification",
        }
    )

    fear_greed = fear_greed.iloc[:-5].copy()
    fear_greed = fear_greed.rename(columns={"date": "Date"})
    fear_greed["Date"] = pd.to_datetime(
        fear_greed["Date"].astype(str).str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )
    fear_greed["fng_value"] = pd.to_numeric(fear_greed["fng_value"], errors="coerce")
    fear_greed = fear_greed.dropna(subset=["Date", "fng_value"])
    fear_greed["Date"] = fear_greed["Date"].dt.normalize().astype("datetime64[ns]")
    fear_greed = fear_greed.sort_values("Date").reset_index(drop=True)
    return fear_greed


def build_full_dataset(start_date: str) -> pd.DataFrame:
    btc = download_btc_history(start_date)
    btc["Daily_Change"] = btc["Close"] - btc["Open"]
    btc["Volatility"] = btc["High"] - btc["Low"]
    btc["Pct_Change"] = btc["Close"].pct_change()
    btc["Volume_Change_pct"] = btc["Volume"].pct_change()
    btc["SMA_7"] = btc["Close"].rolling(7).mean()
    btc["SMA_30"] = btc["Close"].rolling(30).mean()
    btc["Rolling_volatility_30"] = btc["Pct_Change"].rolling(30).std()

    for lag in [1, 2, 3, 7]:
        btc[f"BTC_Close_t-{lag}"] = btc["Close"].shift(lag)

    fear_greed = download_fear_greed_history()
    fear_greed["fng_diff_day"] = fear_greed["fng_value"].diff()
    fear_greed["fng_SMA_7"] = fear_greed["fng_value"].rolling(7).mean()
    fear_greed["fng_SMA_30"] = fear_greed["fng_value"].rolling(30).mean()
    fear_greed["fng_trend"] = fear_greed["fng_SMA_7"] - fear_greed["fng_SMA_30"]

    full_data = pd.merge_asof(
        btc.sort_values("Date"),
        fear_greed.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    full_data["Is_Halving_Date"] = 0
    full_data["Block_reward"] = np.nan
    full_data.loc[full_data["Date"].isin(HALVINGS), "Is_Halving_Date"] = 1

    for index, halving_date in enumerate(HALVINGS):
        end_date = HALVINGS[index + 1] if index + 1 < len(HALVINGS) else full_data["Date"].max()
        mask = (full_data["Date"] >= halving_date) & (full_data["Date"] < end_date)
        full_data.loc[mask, "Block_reward"] = HALVING_REWARDS[index]

    full_data = full_data.sort_values("Date").reset_index(drop=True)
    return full_data


def train_and_forecast(full_data: pd.DataFrame, horizon: int) -> ForecastResult:
    target_columns = [f"Close_t+{day}" for day in range(1, horizon + 1)]
    supervised_data = full_data.copy()

    for day in range(1, horizon + 1):
        supervised_data[f"Close_t+{day}"] = supervised_data["Close"].shift(-day)

    inference_data = full_data.dropna(subset=FEATURE_COLUMNS).copy()
    latest_row = inference_data.iloc[[-1]].copy()
    latest_date = pd.Timestamp(latest_row.iloc[0]["Date"])
    latest_close = float(latest_row.iloc[0]["Close"])

    train_data = supervised_data.dropna(subset=FEATURE_COLUMNS + target_columns).copy()
    if train_data.empty:
        raise ValueError("No hay suficientes datos para entrenar el modelo.")

    mapper = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("mapper", mapper),
        ("model", LinearRegression()),
    ])

    pipeline.fit(train_data[FEATURE_COLUMNS], train_data[target_columns])
    predictions = pipeline.predict(latest_row[FEATURE_COLUMNS])[0]

    forecast_rows = []
    for day, predicted_close in enumerate(predictions, start=1):
        forecast_rows.append(
            {
                "day": day,
                "forecast_date": latest_date + pd.Timedelta(days=day),
                "predicted_close": float(predicted_close),
            }
        )

    prediction_frame = pd.DataFrame(forecast_rows)
    return ForecastResult(
        latest_date=latest_date,
        latest_close=latest_close,
        predictions=prediction_frame,
    )


def format_currency(value: float) -> str:
    return f"USD {value:,.2f}"


def save_outputs(full_data: pd.DataFrame, forecast: ForecastResult, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_data.to_csv(output_path / "full_data_generated.csv", index=False)
    forecast.predictions.to_csv(output_path / "btc_forecast_7d.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera datos de BTC y estima el precio de cierre para los próximos 7 días.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Fecha inicial para descargar BTC (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=int, default=7, help="Cantidad de días a predecir.")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directorio opcional para guardar el dataset generado y la predicción en CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("El horizonte debe ser mayor o igual a 1.")

    full_data = build_full_dataset(args.start_date)
    forecast = train_and_forecast(full_data, args.horizon)

    print("Pronóstico de Bitcoin a 7 días")
    print(f"Última fecha disponible: {forecast.latest_date.date()}")
    print(f"Último cierre observado: {format_currency(forecast.latest_close)}")
    print()

    for row in forecast.predictions.itertuples(index=False):
        print(
            f"t+{row.day} | {row.forecast_date.date()} | {format_currency(row.predicted_close)}"
        )

    if args.save_dir:
        save_outputs(full_data, forecast, args.save_dir)
        print()
        print(f"Archivos guardados en: {args.save_dir}")


if __name__ == "__main__":
    main()