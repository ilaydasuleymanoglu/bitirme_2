from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
TIGGE_ROOT = ROOT / "ERA5_data"
ASCAT_ROOT = ROOT / "ascat_data"
REANALYSIS_ROOT = ROOT / "ERA_5_REANALYSİS_Data"
ASOS_PATH = ROOT / "asos.csv"
OUTPUT_ROOT = ROOT / "wind_correction_output"
KNOT_TO_MS = 0.514444
# erdQMwind1day is a 1-day composite whose timestamp is the day centre
# (12:00 UTC); it contains passes after the 12:00 forecast start. Using the
# previous day's composite keeps every ASCAT pass before initialization.
ASCAT_LAG_DAYS = 1
ASCAT_COMPOSITE_HALF_WIDTH = pd.Timedelta("12h")


def kalman_fill(values: pd.Series, process_var: float = 0.05,
                observation_var: float = 1.0) -> pd.Series:
    """Fill gaps with a causal local-level Kalman filter.

    Observed values are returned unchanged; only missing values receive the
    filtered state. Leading gaps remain missing.
    """
    measurements = values.to_numpy(dtype=float)
    filled = np.empty(len(measurements), dtype=float)
    finite = np.isfinite(measurements)
    if not finite.any():
        return values.copy()
    state = np.nan
    variance = observation_var
    for index, measurement in enumerate(measurements):
        if not np.isfinite(state):
            if not np.isfinite(measurement):
                filled[index] = np.nan
                continue
            state = measurement
        variance += process_var
        if np.isfinite(measurement):
            gain = variance / (variance + observation_var)
            state += gain * (measurement - state)
            variance *= 1.0 - gain
            filled[index] = measurement
        else:
            filled[index] = state
    return pd.Series(filled, index=values.index, name=values.name)


def open_grib_datasets(path: Path) -> list[xr.Dataset]:
    return cfgrib.open_datasets(str(path), indexpath="")


def read_tigge_file(path: Path) -> pd.DataFrame:
    merged = xr.merge(open_grib_datasets(path), compat="override", join="outer")
    frames = []
    for index in range(merged.sizes.get("time", 1)):
        current = merged.isel(time=index) if "time" in merged.dims else merged
        frames.append(
            pd.DataFrame(
                {
                    "latitude": current.latitude.values,
                    "longitude": current.longitude.values,
                    "tigge_u10": current.u10.values,
                    "tigge_v10": current.v10.values,
                    "tigge_t2m": current.t2m.values,
                    "tigge_d2m": current.d2m.values,
                    "tigge_tcc": current.tcc.values,
                    "tigge_tp": current.tp.values,
                    "initialization_time": pd.Timestamp(current.time.values),
                    "valid_time": pd.Timestamp(current.valid_time.values),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def read_wind_reanalysis(path: Path) -> xr.Dataset:
    wind = next(
        dataset
        for dataset in open_grib_datasets(path)
        if {"u10", "v10"} <= set(dataset.data_vars)
    )
    return wind[["u10", "v10"]].sortby("time")


def read_ascat(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)[["modulus"]].sortby("time")


def load_asos() -> pd.DataFrame:
    asos = pd.read_csv(ASOS_PATH, low_memory=False)
    asos["valid"] = pd.to_datetime(asos["valid"], utc=True).dt.tz_localize(None)
    asos["sknt"] = pd.to_numeric(asos["sknt"], errors="coerce")
    asos = asos.dropna(subset=["station", "lat", "lon", "valid"])
    asos = asos.sort_values(["station", "valid"]).copy()
    asos["sknt_kalman"] = asos.groupby("station", group_keys=False)["sknt"].apply(
        kalman_fill
    )
    asos["sknt_gap_filled"] = asos["sknt"].isna() & asos["sknt_kalman"].notna()
    stations = (
        asos[["station", "lat", "lon"]]
        .drop_duplicates("station")
        .sort_values("station")
        .reset_index(drop=True)
    )
    return asos.merge(stations, on=["station", "lat", "lon"], how="left")


def thiessen_ascat_by_station(
    ascat: xr.Dataset, stations: pd.DataFrame
) -> pd.DataFrame:
    """Return area-weighted ASCAT wind for each station's Thiessen cell."""
    station_xy = stations[["lat", "lon"]].to_numpy()
    latitudes = ascat.latitude.to_numpy()
    longitudes = ascat.longitude.to_numpy()
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    scale = np.cos(np.deg2rad(lat_grid))
    grid_xy = np.column_stack((lat_grid.ravel(), lon_grid.ravel() * scale.ravel()))
    station_scaled = np.column_stack(
        (station_xy[:, 0], station_xy[:, 1] * np.cos(np.deg2rad(station_xy[:, 0])))
    )
    cell_station = cKDTree(station_scaled).query(grid_xy)[1]
    cell_weights = scale.ravel()

    values = ascat["modulus"].squeeze("altitude", drop=True)
    rows = []
    for timestamp in pd.to_datetime(values.time.values):
        flat_values = np.asarray(values.sel(time=timestamp).values).ravel()
        for station_index, (_, station) in enumerate(stations.iterrows()):
            mask = cell_station == station_index
            valid = mask & np.isfinite(flat_values)
            if not valid.any():
                value = np.nan
            else:
                value = np.average(flat_values[valid], weights=cell_weights[valid])
            rows.append(
                {
                    "ascat_time": timestamp,
                    "station": station["station"],
                    "ascat_thiessen_wind_speed": value,
                }
            )
    return pd.DataFrame(rows)


def station_model_features(
    tigge: pd.DataFrame,
    reanalysis: xr.Dataset,
    ascat_station: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    station_rows = []
    initialization_time = tigge["initialization_time"].iloc[0]
    valid_time = tigge["valid_time"].iloc[0]
    ascat_time = initialization_time - pd.Timedelta(days=ASCAT_LAG_DAYS)
    reanalysis_times = set(pd.to_datetime(reanalysis["time"].values))
    for _, station in stations.iterrows():
        lat = xr.DataArray([station["lat"]], dims="point")
        lon = xr.DataArray([station["lon"]], dims="point")
        point = tigge.iloc[
            np.argmin(
                (tigge["latitude"] - station["lat"]) ** 2
                + (tigge["longitude"] - station["lon"]) ** 2
            )
        ].copy()
        era = reanalysis.sel(
            time=valid_time,
            latitude=lat,
            longitude=lon,
            method="nearest",
        )
        point["era5_u10"] = float(era["u10"].squeeze().values)
        point["era5_v10"] = float(era["v10"].squeeze().values)
        # Persistence for the ERA5 target: ERA5 wind at forecast start.
        if initialization_time in reanalysis_times:
            era_init = reanalysis.sel(
                time=initialization_time,
                latitude=lat,
                longitude=lon,
                method="nearest",
            )
            point["era5_persistence_wind_speed"] = float(
                np.hypot(era_init["u10"].squeeze(), era_init["v10"].squeeze())
            )
        else:
            point["era5_persistence_wind_speed"] = np.nan
        point["station"] = station["station"]
        valid_minutes = valid_time.hour * 60 + valid_time.minute
        point["fourier_daily_sin"] = np.sin(2 * np.pi * valid_minutes / 1440)
        point["fourier_daily_cos"] = np.cos(2 * np.pi * valid_minutes / 1440)
        point["ascat_time"] = ascat_time
        ascat_value = ascat_station.loc[
            ascat_station["station"].eq(station["station"])
            & ascat_station["ascat_time"].eq(ascat_time),
            "ascat_thiessen_wind_speed",
        ]
        point["ascat_thiessen_wind_speed"] = (
            ascat_value.iloc[0] if len(ascat_value) else np.nan
        )
        station_rows.append(point)
    return pd.DataFrame(station_rows)


def match_ltb_observations(
    asos: pd.DataFrame, valid_time: pd.Timestamp
) -> pd.DataFrame:
    candidates = asos.assign(
        time_difference=(asos["valid"] - valid_time).abs()
    )
    candidates = candidates[candidates["time_difference"] <= pd.Timedelta("40min")]
    columns = ["station", "ltb_wind_speed", "ltb_gap_filled"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    # Prefer a real observation over a Kalman-filled gap inside the window.
    matched = (
        candidates.sort_values(["sknt_gap_filled", "time_difference"])
        .drop_duplicates("station")
        .copy()
    )
    matched["ltb_wind_speed"] = matched["sknt_kalman"] * KNOT_TO_MS
    matched["ltb_gap_filled"] = matched["sknt_gap_filled"]
    return matched[columns]


def build_station_table() -> pd.DataFrame:
    asos = load_asos()
    stations = asos[["station", "lat", "lon"]].drop_duplicates("station")
    rows = []

    monthly_paths = sorted(
        TIGGE_ROOT.glob("tigge_2021_ascat_monthly/2021_*.grib")
    )
    daily_paths = sorted(
        TIGGE_ROOT.glob("*/turkiye_akdeniz_era5_2021_*.grib")
    )
    input_paths = monthly_paths or daily_paths
    # All months are read first so that the lagged ASCAT composite of the
    # first day of a month can come from the previous month's file.
    ascat_station = []
    for ascat_path in sorted(ASCAT_ROOT.glob("*/turkiye_akdeniz_ascat_2021_*.nc")):
        ascat = read_ascat(ascat_path)
        ascat_station.append(thiessen_ascat_by_station(ascat, stations))
        ascat.close()
    if not ascat_station:
        raise FileNotFoundError("ASCAT dosyaları bulunamadı.")
    ascat_station = pd.concat(ascat_station, ignore_index=True)
    ascat_times = set(ascat_station["ascat_time"])

    for path in input_paths:
        month = path.stem[-2:] if monthly_paths else path.parent.name
        reanalysis_path = next(
            REANALYSIS_ROOT.glob(
                f"**/*era5_reanalysis_2021_{month}.grib"
            ),
            None,
        )
        if reanalysis_path is None:
            continue
        reanalysis = read_wind_reanalysis(reanalysis_path)
        tigge_all = read_tigge_file(path)
        for _, tigge in tigge_all.groupby("initialization_time"):
            initialization_time = tigge["initialization_time"].iloc[0]
            ascat_time = initialization_time - pd.Timedelta(days=ASCAT_LAG_DAYS)
            if ascat_time not in ascat_times:
                continue
            valid_time = pd.Timestamp(tigge["valid_time"].iloc[0])
            if valid_time.to_datetime64() not in reanalysis["time"].values:
                continue
            station_features = station_model_features(
                tigge, reanalysis, ascat_station, stations
            )
            valid_time = station_features["valid_time"].iloc[0]
            observations = match_ltb_observations(asos, valid_time)
            station_features = station_features.merge(
                observations,
                on="station",
                how="left",
            )
            # Persistence for the LTB target: LTB observation at forecast start.
            start_observations = match_ltb_observations(
                asos, initialization_time
            ).rename(
                columns={
                    "ltb_wind_speed": "ltb_persistence_wind_speed",
                    "ltb_gap_filled": "ltb_persistence_gap_filled",
                }
            )
            station_features = station_features.merge(
                start_observations, on="station", how="left"
            )
            station_features["ltb_observation_source"] = np.where(
                station_features["ltb_wind_speed"].notna(),
                "LTB/ASOS",
                "ERA5 fallback",
            )
            station_features["ltb_observed_wind_speed"] = station_features[
                "ltb_wind_speed"
            ]
            station_features["ltb_wind_speed"] = station_features[
                "ltb_wind_speed"
            ].fillna(
                np.hypot(
                    station_features["era5_u10"],
                    station_features["era5_v10"],
                )
            )
            rows.append(station_features)

    if not rows:
        raise RuntimeError("ASCAT başlangıç zamanı ile eşleşen TIGGE çevrimi bulunamadı.")
    table = pd.concat(rows, ignore_index=True)
    table["ltb_gap_filled"] = table["ltb_gap_filled"].fillna(False).astype(bool)
    table["tigge_wind_speed"] = np.hypot(table["tigge_u10"], table["tigge_v10"])
    table["era5_wind_speed"] = np.hypot(table["era5_u10"], table["era5_v10"])
    table["ltb_error"] = table["ltb_wind_speed"] - table["tigge_wind_speed"]
    table["ltb_observed_error"] = (
        table["ltb_observed_wind_speed"] - table["tigge_wind_speed"]
    )
    table["era5_error"] = table["era5_wind_speed"] - table["tigge_wind_speed"]
    table["error"] = table["era5_error"]
    table = table.replace([np.inf, -np.inf], np.nan)
    # Persistence columns are baselines, not model inputs; a missing start
    # observation must not remove the row from the evaluation.
    persistence_columns = [
        "ltb_persistence_wind_speed",
        "ltb_persistence_gap_filled",
        "era5_persistence_wind_speed",
    ]
    table = table.dropna(
        subset=[column for column in table.columns if column not in persistence_columns]
    ).copy()
    for target in ["ltb", "era5"]:
        column = f"{target}_persistence_wind_speed"
        table[f"{target}_persistence_missing"] = table[column].isna()
        table[column] = table[column].fillna(table["tigge_wind_speed"])
    return table.reset_index(drop=True)


def metrics(name: str, actual: pd.Series, predicted: np.ndarray) -> dict:
    error = predicted - actual.to_numpy()
    variance = np.sum((actual.to_numpy() - actual.mean()) ** 2)
    return {
        "target": name.split(" | ")[0],
        "model": name.split(" | ")[1],
        "samples": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / variance) if variance else np.nan,
        "bias": float(np.mean(error)),
    }


def main() -> None:
    table = build_station_table().sort_values("valid_time")
    times = table["valid_time"].drop_duplicates().sort_values().to_numpy()
    split = times[max(1, int(len(times) * 0.8))]
    train = table[table["valid_time"] < split]
    test = table[table["valid_time"] >= split]
    if train.empty or test.empty:
        raise RuntimeError("İstasyon bazlı zaman ayrımı için yeterli kayıt yok.")

    features = [
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
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[features], train["error"])
    corrected = test["tigge_wind_speed"].to_numpy() + model.predict(test[features])
    baseline = test["tigge_wind_speed"].to_numpy()
    output_metrics = [
        metrics("LTB | TIGGE", test["ltb_wind_speed"], baseline),
        metrics("LTB | TIGGE + ASCAT Thiessen XGBoost", test["ltb_wind_speed"], corrected),
        metrics("ERA5 | TIGGE", test["era5_wind_speed"], baseline),
        metrics("ERA5 | TIGGE + ASCAT Thiessen XGBoost", test["era5_wind_speed"], corrected),
    ]
    zscore_rows = []
    for name in [
        "ltb_wind_speed",
        "era5_wind_speed",
        "tigge_wind_speed",
        "ascat_thiessen_wind_speed",
        "error",
    ]:
        values = test[name].to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        zscore_rows.append(
            {
                "variable": name,
                "mean": mean,
                "std": std,
                "min_z": float(np.min((values - mean) / std)) if std else np.nan,
                "max_z": float(np.max((values - mean) / std)) if std else np.nan,
            }
        )
    OUTPUT_ROOT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_ROOT / "ltb_thiessen_table.csv", index=False)
    pd.DataFrame(output_metrics).to_csv(
        OUTPUT_ROOT / "ltb_thiessen_metrics.csv", index=False
    )
    pd.DataFrame(zscore_rows).to_csv(
        OUTPUT_ROOT / "ltb_thiessen_zscores.csv", index=False
    )
    test.assign(
        corrected_wind_speed=corrected,
        baseline_wind_speed=baseline,
    ).to_csv(OUTPUT_ROOT / "ltb_thiessen_predictions.csv", index=False)
    print(pd.DataFrame(output_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
