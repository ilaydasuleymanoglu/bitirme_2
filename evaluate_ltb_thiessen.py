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


def kalman_fill(values: pd.Series, process_var: float = 0.05,
                observation_var: float = 1.0) -> pd.Series:
    """Causal local-level Kalman filter; leading gaps remain missing."""
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
                    "initialization_time": timestamp,
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
        point["station"] = station["station"]
        valid_minutes = valid_time.hour * 60 + valid_time.minute
        point["fourier_daily_sin"] = np.sin(2 * np.pi * valid_minutes / 1440)
        point["fourier_daily_cos"] = np.cos(2 * np.pi * valid_minutes / 1440)
        point["ascat_thiessen_wind_speed"] = ascat_station.loc[
            ascat_station["station"].eq(station["station"])
            & ascat_station["initialization_time"].eq(initialization_time),
            "ascat_thiessen_wind_speed",
        ].iloc[0]
        station_rows.append(point)
    return pd.DataFrame(station_rows)


def match_ltb_observations(
    asos: pd.DataFrame, valid_time: pd.Timestamp
) -> pd.DataFrame:
    candidates = asos.assign(
        time_difference=(asos["valid"] - valid_time).abs()
    )
    candidates = candidates[candidates["time_difference"] <= pd.Timedelta("40min")]
    if candidates.empty:
        return pd.DataFrame(columns=["station", "ltb_wind_speed"])
    matched = (
        candidates.sort_values("time_difference")
        .drop_duplicates("station")
        .copy()
    )
    matched["ltb_wind_speed"] = matched["sknt_kalman"] * KNOT_TO_MS
    return matched[["station", "ltb_wind_speed"]]


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
    for path in input_paths:
        month = path.stem[-2:] if monthly_paths else path.parent.name
        ascat_path = ASCAT_ROOT / month / f"turkiye_akdeniz_ascat_2021_{month}.nc"
        reanalysis_path = next(
            REANALYSIS_ROOT.glob(
                f"**/*era5_reanalysis_2021_{month}.grib"
            ),
            None,
        )
        if not ascat_path.exists() or reanalysis_path is None:
            continue
        ascat = read_ascat(ascat_path)
        reanalysis = read_wind_reanalysis(reanalysis_path)
        ascat_station = thiessen_ascat_by_station(ascat, stations)
        tigge_all = read_tigge_file(path)
        for _, tigge in tigge_all.groupby("initialization_time"):
            initialization_time = tigge["initialization_time"].iloc[0]
            if initialization_time not in set(pd.to_datetime(ascat["time"].values)):
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
                observations[["station", "ltb_wind_speed"]],
                on="station",
                how="left",
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
        ascat.close()

    if not rows:
        raise RuntimeError("ASCAT başlangıç zamanı ile eşleşen TIGGE çevrimi bulunamadı.")
    table = pd.concat(rows, ignore_index=True)
    table["tigge_wind_speed"] = np.hypot(table["tigge_u10"], table["tigge_v10"])
    table["era5_wind_speed"] = np.hypot(table["era5_u10"], table["era5_v10"])
    table["ltb_error"] = table["ltb_wind_speed"] - table["tigge_wind_speed"]
    table["ltb_observed_error"] = (
        table["ltb_observed_wind_speed"] - table["tigge_wind_speed"]
    )
    table["era5_error"] = table["era5_wind_speed"] - table["tigge_wind_speed"]
    table["error"] = table["era5_error"]
    return table.replace([np.inf, -np.inf], np.nan).dropna()


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
