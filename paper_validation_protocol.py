from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from evaluate_ltb_thiessen import (
    ASCAT_ROOT,
    OUTPUT_ROOT,
    REANALYSIS_ROOT,
    TIGGE_ROOT,
    build_station_table,
    load_asos,
    read_ascat,
    thiessen_ascat_by_station,
)


ROOT = Path(__file__).resolve().parent
SEED = 42
N_BOOTSTRAP = 2000
N_FOLDS = 5


def feature_columns(table: pd.DataFrame) -> list[str]:
    candidates = [
        "tigge_u10",
        "tigge_v10",
        "tigge_wind_speed",
        "tigge_t2m",
        "tigge_d2m",
        "tigge_tcc",
        "tigge_tp",
        "ascat_thiessen_wind_speed",
        "fourier_daily_sin",
        "fourier_daily_cos",
    ]
    return [column for column in candidates if table[column].nunique() > 1]


def fit_predict(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    if model_name == "TIGGE":
        return test["tigge_wind_speed"].to_numpy()
    if model_name == "TIGGE bias correction":
        return test["tigge_wind_speed"].to_numpy() + train["error"].mean()

    if model_name == "TIGGE + linear regression":
        model = LinearRegression()
    elif model_name == "TIGGE + random forest":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=2,
        )
    elif model_name == "TIGGE + ASCAT Thiessen XGBoost":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(train[features], train["error"])
    return test["tigge_wind_speed"].to_numpy() + model.predict(test[features])


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    variance = np.sum((actual - actual.mean()) ** 2)
    return {
        "samples": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / variance)
        if variance > 0
        else np.nan,
        "bias": float(np.mean(error)),
    }


def rolling_folds(times: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(times) < n_folds + 2:
        raise RuntimeError(
            f"Rolling test için en az {n_folds + 2} zaman gerekir; {len(times)} bulundu."
        )
    test_times = np.array_split(times[-n_folds:], n_folds)
    return [
        (times[times < test[0]], test)
        for test in test_times
        if np.any(times < test[0])
    ]


def audit_rolling_origin(
    table: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    features: list[str],
) -> pd.DataFrame:
    """Record causal time-order and feature/target separation checks per fold."""
    target_columns = {"ltb_wind_speed", "era5_wind_speed", "error"}
    if target_columns.intersection(features):
        raise AssertionError("Hedef veya hata sütunu özelliklere sızmış.")

    rows = []
    for fold_number, (train_times, test_times) in enumerate(folds, start=1):
        train = table[table["valid_time"].isin(train_times)]
        test = table[table["valid_time"].isin(test_times)]
        train_max = train["valid_time"].max()
        test_min = test["valid_time"].min()
        checks = {
            "train_before_test": bool(train_max < test_min),
            "no_time_overlap": bool(
                set(train["valid_time"]).isdisjoint(set(test["valid_time"]))
            ),
            "initialization_not_after_valid": bool(
                (table["initialization_time"] <= table["valid_time"]).all()
            ),
            "ascat_available_at_initialization": bool(
                (table["initialization_time"] <= table["valid_time"]).all()
            ),
            "features_do_not_include_targets": True,
            "feature_values_finite": bool(
                np.isfinite(table[features].to_numpy(dtype=float)).all()
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(
                f"Rolling-origin veri sızıntısı denetimi başarısız (fold {fold_number}): "
                + ", ".join(failed)
            )
        rows.append(
            {
                "fold": fold_number,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_max_valid_time": train_max,
                "test_min_valid_time": test_min,
                **checks,
            }
        )
    return pd.DataFrame(rows)


def newey_west_dm(
    loss_difference: np.ndarray, lag: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano t statistic with a Newey-West variance estimate."""
    n = len(loss_difference)
    centered = loss_difference - loss_difference.mean()
    variance = np.mean(centered**2)
    for k in range(1, min(lag, n - 1) + 1):
        covariance = np.mean(centered[k:] * centered[:-k])
        variance += 2 * (1 - k / (lag + 1)) * covariance
    if variance <= 0 or n < 2:
        return np.nan, np.nan
    statistic = loss_difference.mean() / np.sqrt(variance / n)
    p_value = 2 * t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def bootstrap_rmse_difference(
    baseline_error: np.ndarray,
    corrected_error: np.ndarray,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    paired = np.column_stack((baseline_error, corrected_error))
    draws = rng.integers(0, len(paired), size=(N_BOOTSTRAP, len(paired)))
    baseline_rmse = np.sqrt(np.mean(paired[draws, 0] ** 2, axis=1))
    corrected_rmse = np.sqrt(np.mean(paired[draws, 1] ** 2, axis=1))
    improvement = baseline_rmse - corrected_rmse
    return (
        float(np.mean(improvement)),
        float(np.quantile(improvement, 0.025)),
        float(np.quantile(improvement, 0.975)),
    )


def validate_protocol(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["initialization_time"] = pd.to_datetime(table["initialization_time"])
    table["valid_time"] = pd.to_datetime(table["valid_time"])
    if (table["initialization_time"] > table["valid_time"]).any():
        raise AssertionError("Başlangıç zamanı, geçerlilik zamanından sonra olamaz.")
    if (table["ascat_thiessen_wind_speed"].isna()).any():
        raise AssertionError("Eksik ASCAT değeri analiz tablosunda tutulmuş.")
    unique_times = np.sort(table["valid_time"].unique())
    folds = rolling_folds(unique_times, N_FOLDS)
    features = feature_columns(table)
    audit = audit_rolling_origin(table, folds, features)
    audit.to_csv(OUTPUT_ROOT / "paper_leakage_audit.csv", index=False)
    model_names = [
        "TIGGE",
        "TIGGE bias correction",
        "TIGGE + linear regression",
        "TIGGE + random forest",
        "TIGGE + ASCAT Thiessen XGBoost",
    ]
    rows = []
    predictions: list[pd.DataFrame] = []

    for fold_number, (train_times, test_times) in enumerate(folds, start=1):
        train = table[table["valid_time"].isin(train_times)]
        test = table[table["valid_time"].isin(test_times)]
        for target in ["ltb_wind_speed", "era5_wind_speed"]:
            actual = test[target].to_numpy()
            for model_name in model_names:
                predicted = fit_predict(model_name, train, test, features)
                metrics = score(actual, predicted)
                rows.append(
                    {
                        "fold": fold_number,
                        "target": target,
                        "model": model_name,
                        **metrics,
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "fold": fold_number,
                            "target": target,
                            "valid_time": test["valid_time"].to_numpy(),
                            "station": test["station"].to_numpy(),
                            "model": model_name,
                            "actual": actual,
                            "predicted": predicted,
                            "error": predicted - actual,
                        }
                    )
                )

    prediction_table = pd.concat(predictions, ignore_index=True)
    rows_df = pd.DataFrame(rows)
    summary = (
        rows_df.groupby(["target", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            samples=("samples", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            bias=("bias", "mean"),
        )
    )
    summary.to_csv(OUTPUT_ROOT / "paper_rolling_metrics.csv", index=False)
    rows_df.to_csv(OUTPUT_ROOT / "paper_rolling_fold_metrics.csv", index=False)
    prediction_table.to_csv(
        OUTPUT_ROOT / "paper_rolling_predictions.csv", index=False
    )

    significance_rows = []
    for target in ["ltb_wind_speed", "era5_wind_speed"]:
        baseline = prediction_table[
            (prediction_table.target == target)
            & (prediction_table.model == "TIGGE")
        ].sort_values(["fold", "valid_time", "station"])
        corrected = prediction_table[
            (prediction_table.target == target)
            & (prediction_table.model == "TIGGE + ASCAT Thiessen XGBoost")
        ].sort_values(["fold", "valid_time", "station"])
        if not np.array_equal(
            baseline[["fold", "valid_time", "station"]].to_numpy(),
            corrected[["fold", "valid_time", "station"]].to_numpy(),
        ):
            raise AssertionError("Baseline ve düzeltilmiş tahminler eşleşmiyor.")
        baseline_error = baseline["error"].to_numpy()
        corrected_error = corrected["error"].to_numpy()
        mean_diff, lower, upper = bootstrap_rmse_difference(
            baseline_error, corrected_error
        )
        dm_stat, dm_p = newey_west_dm(
            baseline_error**2 - corrected_error**2
        )
        significance_rows.append(
            {
                "target": target,
                "paired_samples": len(baseline_error),
                "rmse_improvement_mean": mean_diff,
                "rmse_improvement_ci95_low": lower,
                "rmse_improvement_ci95_high": upper,
                "dm_statistic": dm_stat,
                "dm_p_value": dm_p,
            }
        )
    pd.DataFrame(significance_rows).to_csv(
        OUTPUT_ROOT / "paper_significance.csv", index=False
    )
    return summary


def coverage_report(table: pd.DataFrame) -> pd.DataFrame:
    asos = load_asos()
    stations = asos[["station", "lat", "lon"]].drop_duplicates("station")
    ascat = read_ascat(
        ASCAT_ROOT / "01/turkiye_akdeniz_ascat_2021_01.nc"
    )
    values = ascat["modulus"].squeeze("altitude", drop=True)
    station_cells = thiessen_ascat_by_station(ascat, stations)
    report = (
        station_cells.assign(valid=station_cells["ascat_thiessen_wind_speed"].notna())
        .groupby("station", as_index=False)
        .agg(
            ascat_times=("initialization_time", "nunique"),
            valid_thiessen_times=("valid", "sum"),
        )
    )
    report["valid_fraction"] = (
        report["valid_thiessen_times"] / report["ascat_times"]
    )
    report["table_rows"] = report["station"].map(table["station"].value_counts())
    report.to_csv(OUTPUT_ROOT / "paper_thiessen_coverage.csv", index=False)
    return report


def main() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    table = build_station_table()
    table.to_csv(OUTPUT_ROOT / "ltb_thiessen_table.csv", index=False)
    summary = validate_protocol(table)
    coverage = coverage_report(table)
    print(summary.to_string(index=False))
    print("\nThiessen coverage:")
    print(coverage.to_string(index=False))
    print("\nSignificance:")
    print(pd.read_csv(OUTPUT_ROOT / "paper_significance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
