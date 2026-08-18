import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf


DEFAULT_DATA_PATH = Path("Data/full_data.csv")
MODEL_DIR = Path(__file__).resolve().parent / "Experimentos" / "models_final"

# features usadas para entrenar los MLP guardados en models_final (experimento 3)
FEATURE_COLUMNS = [
    "Volume",
    "Pct_Change",
    "Volume_Change_pct",
    "Volatility",
    "SMA_7",
    "SMA_30",
    "fng_value",
    "fng_SMA_7",
    "fng_SMA_30",
    "BTC_Close_t-1",
    "BTC_Close_t-2",
    "BTC_Close_t-3",
    "BTC_Close_t-7",
    "Rolling_volatility_30",
    "Block_reward",
]


@dataclass
class ForecastResult:
    latest_date: pd.Timestamp
    latest_close: float
    model_name: str
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carga la red neuronal del experimento 3 y realiza la prediccion "
            "del precio de cierre de BTC para los proximos horizontes."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al CSV con el dataset final (default: Data/full_data.csv).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Cantidad de dias a pronosticar (default: 7).",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path, horizon: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("El dataset debe incluir las columnas 'Date' y 'Close'.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Faltan columnas requeridas para la red neuronal: "
            + ", ".join(missing_features)
        )

    for day in range(1, horizon + 1):
        df[f"Close_t+{day}"] = df["Close"].shift(-day)

    return df


def load_model_for_horizon(horizon: int):
    target = f"Close_t+{horizon}"
    model_path = MODEL_DIR / f"mlp_{target}.keras"
    scaler_path = MODEL_DIR / f"scaler_{target}.pkl"
    imputer_path = MODEL_DIR / f"imputer_{target}.pkl"

    if not model_path.exists() or not scaler_path.exists() or not imputer_path.exists():
        raise FileNotFoundError(
            f"No se encontraron artefactos para el horizonte {horizon}. "
            f"Buscados en: {MODEL_DIR}"
        )

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    return model, scaler, imputer


def prepare_last_row(df: pd.DataFrame):
    row = df.dropna(subset=FEATURE_COLUMNS).iloc[[-1]].copy()
    if row.empty:
        raise ValueError("No hay filas con features validas para predecir.")
    return row


def forecast_with_mlp(df: pd.DataFrame, horizon: int) -> ForecastResult:
    last_row = prepare_last_row(df)
    last_date = pd.Timestamp(last_row.iloc[0]["Date"])
    last_close = float(last_row.iloc[0]["Close"])

    forecast_rows = []
    for day in range(1, horizon + 1):
        model, scaler, imputer = load_model_for_horizon(day)

        x_row = last_row[FEATURE_COLUMNS].to_numpy(dtype=float)
        x_imp = imputer.transform(x_row.reshape(1, -1))
        x_scaled = scaler.transform(x_imp)

        prediction = model.predict(x_scaled, verbose=0)
        value = float(prediction[0][0])

        forecast_rows.append(
            {
                "horizon": f"t+{day}",
                "forecast_date": last_date + pd.Timedelta(days=day),
                "predicted_close": value,
            }
        )

    return ForecastResult(
        latest_date=last_date,
        latest_close=last_close,
        model_name="MLP_Exp3",
        predictions=pd.DataFrame(forecast_rows),
    )


def print_result(result: ForecastResult) -> None:
    print("\n=== Prediccion BTC con la red neuronal del Experimento 3 ===")
    print(f"Ultima fecha disponible: {result.latest_date.date()}")
    print(f"Ultimo cierre observado: USD {result.latest_close:,.2f}")
    print(f"Modelo usado: {result.model_name}")
    print("\nPronostico del cierre:")
    for row in result.predictions.itertuples(index=False):
        print(
            f"{row.horizon} | {row.forecast_date.date()} | "
            f"USD {row.predicted_close:,.2f}"
        )


def main() -> None:
    args = parse_args()

    if args.horizon < 1 or args.horizon > 7:
        raise ValueError("El horizonte debe estar entre 1 y 7 dias.")

    df = load_dataset(args.data_path, args.horizon)
    result = forecast_with_mlp(df, args.horizon)
    print_result(result)


if __name__ == "__main__":
    main()
