from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from evaluate_ltb_thiessen import (
    ASCAT_COMPOSITE_HALF_WIDTH,
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
    target: str,
) -> np.ndarray:
    error_column = f"{target.removesuffix('_wind_speed')}_error"
    if model_name == "TIGGE":
        return test["tigge_wind_speed"].to_numpy()
    if model_name == "Persistence":
        # Target-specific value observed at forecast start (t_valid - 12 h).
        prefix = target.removesuffix("_wind_speed")
        return test[f"{prefix}_persistence_wind_speed"].to_numpy()
    if model_name == "Persistence (24 h)":
        prefix = target.removesuffix("_wind_speed")
        return test[f"{prefix}_persistence24_wind_speed"].to_numpy()
    if model_name == "TIGGE bias correction":
        return test["tigge_wind_speed"].to_numpy() + train[error_column].mean()

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
        unique_train_times = np.sort(train["valid_time"].unique())
        candidates = [
            {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 3},
            {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 1},
            {"max_depth": 4, "learning_rate": 0.06, "min_child_weight": 5},
        ]
        if len(unique_train_times) >= 10:
            split = max(1, int(len(unique_train_times) * 0.8))
            inner_train = train[train["valid_time"] < unique_train_times[split]]
            inner_valid = train[train["valid_time"] >= unique_train_times[split]]
            scores = []
            for params in candidates:
                candidate = XGBRegressor(
                    n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                    objective="reg:squarederror", random_state=SEED, n_jobs=2,
                    **params,
                )
                candidate.fit(inner_train[features], inner_train[error_column])
                residual = candidate.predict(inner_valid[features]) - inner_valid[
                    error_column
                ].to_numpy()
                scores.append((float(np.mean(residual**2)), params))
            selected = min(scores, key=lambda item: item[0])[1]
        else:
            selected = candidates[1]
        model = XGBRegressor(
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=SEED, n_jobs=2,
            **selected,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(train[features], train[error_column])
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


def rolling_folds(times: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(times) < 2:
        raise RuntimeError(
            f"Rolling test için en az 2 zaman gerekir; {len(times)} bulundu."
        )
    return [
        (times[:index], times[index : index + 1])
        for index in range(1, len(times))
    ]


def audit_rolling_origin(
    table: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    features: list[str],
) -> pd.DataFrame:
    """Record causal time-order and feature/target separation checks per fold."""
    target_columns = {
        "ltb_wind_speed", "era5_wind_speed", "ltb_error", "era5_error",
        "ltb_observed_error", "error"
    }
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
            # The whole 1-day ASCAT composite must end before forecast start.
            "ascat_composite_before_initialization": bool(
                (
                    table["ascat_time"] + ASCAT_COMPOSITE_HALF_WIDTH
                    <= table["initialization_time"]
                ).all()
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
    """Diebold-Mariano asymptotic z statistic with Newey-West variance."""
    n = len(loss_difference)
    centered = loss_difference - loss_difference.mean()
    variance = np.mean(centered**2)
    for k in range(1, min(lag, n - 1) + 1):
        covariance = np.mean(centered[k:] * centered[:-k])
        variance += 2 * (1 - k / (lag + 1)) * covariance
    if variance <= 0 or n < 2:
        return np.nan, np.nan
    statistic = loss_difference.mean() / np.sqrt(variance / n)
    p_value = 2 * norm.sf(abs(statistic))
    return float(statistic), float(p_value)


def block_bootstrap_rmse_difference(
    baseline_mse: np.ndarray,
    corrected_mse: np.ndarray,
    block_length: int = 7,
) -> tuple[float, float, float]:
    """Bootstrap origin-level losses while preserving short-term dependence."""
    rng = np.random.default_rng(SEED)
    n = len(baseline_mse)
    if n != len(corrected_mse) or n < 2:
        return np.nan, np.nan, np.nan
    block_length = min(block_length, n)
    starts = rng.integers(0, n, size=(N_BOOTSTRAP, int(np.ceil(n / block_length))))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % n
    indices = indices.reshape(N_BOOTSTRAP, -1)[:, :n]
    baseline_rmse = np.sqrt(np.mean(baseline_mse[indices], axis=1))
    corrected_rmse = np.sqrt(np.mean(corrected_mse[indices], axis=1))
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
    table["ascat_time"] = pd.to_datetime(table["ascat_time"])
    table = table.sort_values(["station", "valid_time"]).copy()
    # 24 h persistence: the target's own value at valid_time - 24 h, which is
    # known before the 12:00 UTC forecast start and keeps the diurnal phase.
    for target in ["ltb_wind_speed", "era5_wind_speed"]:
        prefix = target.removesuffix("_wind_speed")
        previous = table[["station", "valid_time", target]].assign(
            valid_time=table["valid_time"] + pd.Timedelta(days=1)
        ).rename(columns={target: f"{prefix}_persistence24_wind_speed"})
        table = table.merge(previous, on=["station", "valid_time"], how="left")
        table[f"{prefix}_persistence24_missing"] = table[
            f"{prefix}_persistence24_wind_speed"
        ].isna()
        table[f"{prefix}_persistence24_wind_speed"] = table[
            f"{prefix}_persistence24_wind_speed"
        ].fillna(table["tigge_wind_speed"])
    if (table["initialization_time"] > table["valid_time"]).any():
        raise AssertionError("Başlangıç zamanı, geçerlilik zamanından sonra olamaz.")
    if (table["ascat_thiessen_wind_speed"].isna()).any():
        raise AssertionError("Eksik ASCAT değeri analiz tablosunda tutulmuş.")
    if table[["ltb_error", "era5_error"]].isna().any().any():
        raise AssertionError("Hedefe özgü hata sütunlarında eksik değer var.")
    unique_times = np.sort(table["valid_time"].unique())
    folds = rolling_folds(unique_times)
    features = feature_columns(table)
    audit = audit_rolling_origin(table, folds, features)
    audit.to_csv(OUTPUT_ROOT / "paper_leakage_audit.csv", index=False)
    model_names = [
        "TIGGE",
        "Persistence",
        "Persistence (24 h)",
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
                predicted = fit_predict(model_name, train, test, features, target)
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
                            "observation_source": test[
                                "ltb_observation_source"
                            ].to_numpy(),
                            "model": model_name,
                            "actual": actual,
                            "predicted": predicted,
                            "error": predicted - actual,
                        }
                    )
                )

    prediction_table = pd.concat(predictions, ignore_index=True)
    rows_df = pd.DataFrame(rows)
    summary_rows = []
    for (target, model), data in prediction_table.groupby(["target", "model"]):
        error = data["error"].to_numpy()
        actual = data["actual"].to_numpy()
        variance = np.sum((actual - actual.mean()) ** 2)
        summary_rows.append(
            {
                "target": target,
                "model": model,
                "folds": data["fold"].nunique(),
                "samples": len(data),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "r2": float(1 - np.sum(error**2) / variance)
                if variance > 0
                else np.nan,
                "bias": float(np.mean(error)),
                "era5_fallback_rows": int(
                    (data["observation_source"] == "ERA5 fallback").sum()
                )
                if target == "ltb_wind_speed"
                else 0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_ROOT / "paper_rolling_metrics.csv", index=False)
    rows_df.to_csv(OUTPUT_ROOT / "paper_rolling_fold_metrics.csv", index=False)
    prediction_table.to_csv(
        OUTPUT_ROOT / "paper_rolling_predictions.csv", index=False
    )
    balanced_rows = []
    for (target, model), data in prediction_table.groupby(["target", "model"]):
        station_scores = data.groupby("station").apply(
            lambda group: pd.Series(
                {
                    "mae": np.mean(np.abs(group["error"])),
                    "rmse": np.sqrt(np.mean(group["error"] ** 2)),
                    "bias": np.mean(group["error"]),
                    "samples": len(group),
                }
            ),
            include_groups=False,
        )
        balanced_rows.append(
            {
                "target": target,
                "model": model,
                "stations": len(station_scores),
                "equal_station_mae": station_scores["mae"].mean(),
                "equal_station_rmse": station_scores["rmse"].mean(),
                "equal_station_bias": station_scores["bias"].mean(),
                "min_station_samples": int(station_scores["samples"].min()),
                "max_station_samples": int(station_scores["samples"].max()),
            }
        )
    pd.DataFrame(balanced_rows).to_csv(
        OUTPUT_ROOT / "paper_station_balanced_metrics.csv", index=False
    )

    significance_rows = []
    # The first pair is the main result; the others support the discussion.
    comparisons = [
        ("TIGGE", "TIGGE + ASCAT Thiessen XGBoost"),
        ("Persistence", "TIGGE + ASCAT Thiessen XGBoost"),
        ("Persistence (24 h)", "TIGGE + ASCAT Thiessen XGBoost"),
        ("TIGGE", "TIGGE + random forest"),
        ("TIGGE + random forest", "TIGGE + ASCAT Thiessen XGBoost"),
    ]
    for target, (baseline_model, corrected_model) in [
        (target, pair)
        for target in ["ltb_wind_speed", "era5_wind_speed"]
        for pair in comparisons
    ]:
        baseline = prediction_table[
            (prediction_table.target == target)
            & (prediction_table.model == baseline_model)
        ].sort_values(["fold", "valid_time", "station"])
        corrected = prediction_table[
            (prediction_table.target == target)
            & (prediction_table.model == corrected_model)
        ].sort_values(["fold", "valid_time", "station"])
        if not np.array_equal(
            baseline[["fold", "valid_time", "station"]].to_numpy(),
            corrected[["fold", "valid_time", "station"]].to_numpy(),
        ):
            raise AssertionError("Baseline ve düzeltilmiş tahminler eşleşmiyor.")
        paired = baseline[["valid_time", "station", "error"]].rename(
            columns={"error": "baseline_error"}
        ).merge(
            corrected[["valid_time", "station", "error"]].rename(
                columns={"error": "corrected_error"}
            ),
            on=["valid_time", "station"],
            how="inner",
        )
        # Stations sharing a forecast origin are one temporal cluster, not
        # independent replications.
        origin_losses = paired.assign(
            baseline_loss=paired["baseline_error"] ** 2,
            corrected_loss=paired["corrected_error"] ** 2,
        ).groupby("valid_time", as_index=False)[
            ["baseline_loss", "corrected_loss"]
        ].mean()
        baseline_loss = origin_losses["baseline_loss"].to_numpy()
        corrected_loss = origin_losses["corrected_loss"].to_numpy()
        mean_diff, lower, upper = block_bootstrap_rmse_difference(
            baseline_loss, corrected_loss
        )
        # Point estimates: pooled RMSE difference (as in the metrics table)
        # and the same origin-weighted statistic the bootstrap resamples.
        pooled_baseline_rmse = float(np.sqrt(np.mean(paired["baseline_error"] ** 2)))
        pooled_corrected_rmse = float(
            np.sqrt(np.mean(paired["corrected_error"] ** 2))
        )
        origin_point = float(
            np.sqrt(np.mean(baseline_loss)) - np.sqrt(np.mean(corrected_loss))
        )
        # DM uses the same loss as the bootstrap: mean squared error over the
        # stations of one origin. Averaging signed errors first would let
        # opposite station errors cancel and test a different quantity.
        loss_difference = baseline_loss - corrected_loss
        dm_stat, dm_p = newey_west_dm(loss_difference, lag=1)
        sensitivity = {}
        for lag in (3, 7):
            stat, p_value = newey_west_dm(loss_difference, lag=lag)
            sensitivity[f"dm_p_value_lag{lag}"] = p_value
        significance_rows.append(
            {
                "target": target,
                "baseline_model": baseline_model,
                "corrected_model": corrected_model,
                "paired_samples": len(baseline_loss),
                "paired_records": len(paired),
                "pooled_baseline_rmse": pooled_baseline_rmse,
                "pooled_corrected_rmse": pooled_corrected_rmse,
                "rmse_improvement_point": pooled_baseline_rmse
                - pooled_corrected_rmse,
                "rmse_improvement_point_pct": 100
                * (pooled_baseline_rmse - pooled_corrected_rmse)
                / pooled_baseline_rmse,
                "rmse_improvement_origin_point": origin_point,
                "rmse_improvement_bootstrap_mean": mean_diff,
                "rmse_improvement_ci95_low": lower,
                "rmse_improvement_ci95_high": upper,
                "bootstrap_block_length": 7,
                "dm_statistic": dm_stat,
                "dm_p_value": dm_p,
                **sensitivity,
            }
        )
    pd.DataFrame(significance_rows).to_csv(
        OUTPUT_ROOT / "paper_significance.csv", index=False
    )
    return summary


def coverage_report(table: pd.DataFrame) -> pd.DataFrame:
    asos = load_asos()
    stations = asos[["station", "lat", "lon"]].drop_duplicates("station")
    station_cells = []
    for month in range(1, 5):
        path = ASCAT_ROOT / f"{month:02d}/turkiye_akdeniz_ascat_2021_{month:02d}.nc"
        if not path.exists():
            continue
        ascat = read_ascat(path)
        station_cells.append(thiessen_ascat_by_station(ascat, stations))
        ascat.close()
    if not station_cells:
        raise FileNotFoundError("Ocak-Nisan ASCAT dosyaları bulunamadı.")
    station_cells = pd.concat(station_cells, ignore_index=True)
    report = (
        station_cells.assign(valid=station_cells["ascat_thiessen_wind_speed"].notna())
        .groupby("station", as_index=False)
        .agg(
            ascat_times=("ascat_time", "nunique"),
            valid_thiessen_times=("valid", "sum"),
        )
    )
    report["valid_fraction"] = (
        report["valid_thiessen_times"] / report["ascat_times"]
    )
    report["table_rows"] = report["station"].map(table["station"].value_counts())
    report.to_csv(OUTPUT_ROOT / "paper_thiessen_coverage.csv", index=False)
    source_report = (
        table["ltb_observation_source"]
        .value_counts(dropna=False)
        .rename_axis("source")
        .reset_index(name="rows")
    )
    source_report["fraction"] = source_report["rows"] / len(table)
    source_report.to_csv(
        OUTPUT_ROOT / "paper_ltb_observation_coverage.csv", index=False
    )
    sample_composition(table)
    return report


def sample_composition(table: pd.DataFrame) -> pd.DataFrame:
    """Per-station sample sizes and data-quality flags for the thesis text."""
    first_origin = table["valid_time"].min()
    composition = (
        table.assign(
            out_of_sample=table["valid_time"] > first_origin,
        )
        .groupby("station", as_index=False)
        .agg(
            table_rows=("valid_time", "size"),
            out_of_sample_rows=("out_of_sample", "sum"),
            first_valid_time=("valid_time", "min"),
            last_valid_time=("valid_time", "max"),
            ltb_gap_filled_rows=("ltb_gap_filled", "sum"),
            ltb_persistence_missing_rows=("ltb_persistence_missing", "sum"),
            era5_persistence_missing_rows=("era5_persistence_missing", "sum"),
        )
    )
    composition["share_of_rows"] = composition["table_rows"] / len(table)
    composition.to_csv(OUTPUT_ROOT / "paper_sample_composition.csv", index=False)
    stations_per_origin = (
        table.groupby("valid_time")["station"]
        .nunique()
        .value_counts()
        .sort_index()
        .rename_axis("stations_in_origin")
        .reset_index(name="origins")
    )
    stations_per_origin.to_csv(
        OUTPUT_ROOT / "paper_stations_per_origin.csv", index=False
    )
    return composition


def main() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    table = build_station_table()
    table.to_csv(OUTPUT_ROOT / "ltb_thiessen_table.csv", index=False)
    summary = validate_protocol(table)
    coverage = coverage_report(
        table.assign(valid_time=pd.to_datetime(table["valid_time"]))
    )
    pd.DataFrame(
        [
            {
                "product": "ASCAT",
                "operational_latency_verified": False,
                "status": "not_verified",
                "reason": (
                    "Arşiv NetCDF zaman damgaları ürünün operasyonel yayın "
                    "gecikmesini göstermemektedir."
                ),
                "required_evidence": (
                    "Ürün dokümantasyonu veya gerçek zamanlı erişim loglarında "
                    "yayın ve erişim zamanlarının karşılaştırılması."
                ),
            }
        ]
    ).to_csv(OUTPUT_ROOT / "paper_operational_availability.csv", index=False)
    print(summary.to_string(index=False))
    print("\nThiessen coverage:")
    print(coverage.to_string(index=False))
    print("\nSignificance:")
    print(pd.read_csv(OUTPUT_ROOT / "paper_significance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
