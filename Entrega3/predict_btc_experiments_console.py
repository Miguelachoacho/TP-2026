import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path("Data/full_data.csv")
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Volatility",
    "SMA_7",
    "SMA_30",
    "fng_value",
    "fng_SMA_7",
    "fng_SMA_30",
]


@dataclass
class ModelScore:
    name: str
    mae_mean: float
    rmse_mean: float


@dataclass
class ForecastResult:
    latest_date: pd.Timestamp
    latest_close: float
    model_name: str
    scores: list[ModelScore]
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalua modelos de los experimentos con validacion temporal y "
            "muestra la prediccion de BTC para t+1 a t+7."
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
        help="Dias a predecir desde la ultima fecha disponible (default: 7).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Cantidad de folds para validacion temporal (default: 5).",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path, horizon: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("El dataset debe incluir la columna 'Date'.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Faltan columnas requeridas para entrenar: " + ", ".join(missing_features)
        )

    for day in range(1, horizon + 1):
        df[f"Close_t+{day}"] = df["Close"].shift(-day)

    return df


def build_models() -> dict[str, object]:
    np.random.seed(42)
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=21),
    }


def make_pipeline(model: object) -> Pipeline:
    estimator = model
    if not hasattr(model, "predict"):
        raise ValueError("El modelo no implementa metodo predict().")

    # KNN y RandomForest soportan salida multiple; esta envoltura mantiene robustez.
    if not hasattr(model, "n_outputs_") and model.__class__.__name__ == "SVR":
        estimator = MultiOutputRegressor(model)

    mapper = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )
    return Pipeline([
        ("mapper", mapper),
        ("model", estimator),
    ])


def evaluate_models(df: pd.DataFrame, horizon: int, splits: int) -> tuple[str, list[ModelScore]]:
    targets = [f"Close_t+{day}" for day in range(1, horizon + 1)]
    train_df = df.dropna(subset=FEATURE_COLUMNS + targets).copy()

    X = train_df[FEATURE_COLUMNS]
    Y = train_df[targets]

    tvt = TimeSeriesSplit(n_splits=splits)
    models = build_models()
    scores: list[ModelScore] = []

    for model_name, model in models.items():
        fold_mae = []
        fold_rmse = []

        for train_idx, val_idx in tvt.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

            pipeline = make_pipeline(model)
            pipeline.fit(X_train, Y_train)
            Y_pred = pipeline.predict(X_val)

            mae = mean_absolute_error(Y_val, Y_pred)
            rmse = mean_squared_error(Y_val, Y_pred) ** 0.5
            fold_mae.append(mae)
            fold_rmse.append(rmse)

        scores.append(
            ModelScore(
                name=model_name,
                mae_mean=float(np.mean(fold_mae)),
                rmse_mean=float(np.mean(fold_rmse)),
            )
        )

    best_model = min(scores, key=lambda score: score.mae_mean)
    return best_model.name, sorted(scores, key=lambda score: score.mae_mean)


def train_best_and_forecast(
    df: pd.DataFrame,
    best_model_name: str,
    scores: list[ModelScore],
    horizon: int,
) -> ForecastResult:
    targets = [f"Close_t+{day}" for day in range(1, horizon + 1)]
    train_df = df.dropna(subset=FEATURE_COLUMNS + targets).copy()

    latest_features_df = df.dropna(subset=FEATURE_COLUMNS).copy()
    latest_row = latest_features_df.iloc[[-1]]

    X_train = train_df[FEATURE_COLUMNS]
    Y_train = train_df[targets]

    model = build_models()[best_model_name]
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, Y_train)

    pred = pipeline.predict(latest_row[FEATURE_COLUMNS])[0]

    latest_date = pd.Timestamp(latest_row.iloc[0]["Date"])
    latest_close = float(latest_row.iloc[0]["Close"])

    forecast_rows = []
    for day, close_pred in enumerate(pred, start=1):
        forecast_rows.append(
            {
                "horizon": f"t+{day}",
                "forecast_date": latest_date + pd.Timedelta(days=day),
                "predicted_close": float(close_pred),
            }
        )

    return ForecastResult(
        latest_date=latest_date,
        latest_close=latest_close,
        model_name=best_model_name,
        scores=scores,
        predictions=pd.DataFrame(forecast_rows),
    )


def print_result(result: ForecastResult) -> None:
    print("\n=== Prediccion BTC (basada en experimentos) ===")
    print(f"Ultima fecha disponible: {result.latest_date.date()}")
    print(f"Ultimo cierre observado: USD {result.latest_close:,.2f}")

    print("\nModelos evaluados (CV temporal):")
    for score in result.scores:
        print(
            f"- {score.name}: MAE={score.mae_mean:.4f} | RMSE={score.rmse_mean:.4f}"
        )

    print(f"\nModelo seleccionado: {result.model_name}")
    print("\nPronostico:")
    for row in result.predictions.itertuples(index=False):
        print(
            f"{row.horizon} | {row.forecast_date.date()} | "
            f"USD {row.predicted_close:,.2f}"
        )


def main() -> None:
    args = parse_args()

    if args.horizon < 1:
        raise ValueError("El horizonte debe ser mayor o igual a 1.")
    if args.splits < 2:
        raise ValueError("La validacion temporal requiere al menos 2 folds.")

    df = load_dataset(args.data_path, args.horizon)
    best_model_name, scores = evaluate_models(df, args.horizon, args.splits)
    result = train_best_and_forecast(df, best_model_name, scores, args.horizon)
    print_result(result)


if __name__ == "__main__":
    main()
