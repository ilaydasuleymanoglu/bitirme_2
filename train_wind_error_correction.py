from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
TIGGE_ROOT = ROOT / "ERA5_data"
ASCAT_ROOT = ROOT / "ascat_data"
REANALYSIS_ROOT = ROOT / "ERA_5_REANALYSİS_Data"
OUTPUT_ROOT = ROOT / "wind_correction_output"


def open_grib_datasets(path: Path) -> list[xr.Dataset]:
    return cfgrib.open_datasets(str(path), indexpath="")


def read_tigge_file(path: Path) -> pd.DataFrame:
    datasets = open_grib_datasets(path)
    merged = xr.merge(datasets, compat="override", join="outer")
    initialization_time = pd.Timestamp(merged.time.values)
    valid_time = pd.Timestamp(merged.valid_time.values)
    frame = pd.DataFrame(
        {
            "latitude": merged.latitude.values,
            "longitude": merged.longitude.values,
            "tigge_u10": merged.u10.values,
            "tigge_v10": merged.v10.values,
            "tigge_t2m": merged.t2m.values,
            "tigge_d2m": merged.d2m.values,
            "tigge_tcc": merged.tcc.values,
            "tigge_tp": merged.tp.values,
        }
    )
    frame["initialization_time"] = initialization_time
    frame["valid_time"] = valid_time
    return frame


def read_reanalysis(path: Path) -> xr.Dataset:
    datasets = open_grib_datasets(path)
    wind = next(ds for ds in datasets if {"u10", "v10"} <= set(ds.data_vars))
    return wind[["u10", "v10"]].sortby("time")


def read_ascat(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)[["modulus"]].sortby("time")


def build_training_table() -> pd.DataFrame:
    reanalysis_paths = sorted(
        REANALYSIS_ROOT.glob("**/*era5_reanalysis_2021_01*.grib")
    )
    if not reanalysis_paths:
        raise FileNotFoundError("Ocak 2021 ERA5 reanalysis GRIB dosyası bulunamadı.")
    reanalysis = read_reanalysis(reanalysis_paths[0])

    ascat_path = ASCAT_ROOT / "01/turkiye_akdeniz_ascat_2021_01.nc"
    if not ascat_path.exists():
        raise FileNotFoundError(f"ASCAT dosyası bulunamadı: {ascat_path}")
    ascat = read_ascat(ascat_path)

    rows = []
    for path in sorted(TIGGE_ROOT.glob("01/turkiye_akdeniz_era5_2021_01_*.grib")):
        frame = read_tigge_file(path)
        initialization_time = (
            frame["initialization_time"].iloc[0].to_datetime64()
        )
        valid_time = frame["valid_time"].iloc[0].to_datetime64()
        lat = xr.DataArray(frame["latitude"].to_numpy(), dims="point")
        lon = xr.DataArray(frame["longitude"].to_numpy(), dims="point")

        era = reanalysis.sel(
            time=valid_time, latitude=lat, longitude=lon, method="nearest"
        )
        sat = ascat.sel(
            time=initialization_time, latitude=lat, longitude=lon, method="nearest"
        )
        frame["ascat_wind_speed"] = sat["modulus"].squeeze().values
        frame["era5_u10"] = era["u10"].squeeze().values
        frame["era5_v10"] = era["v10"].squeeze().values
        rows.append(frame)

    table = pd.concat(rows, ignore_index=True)
    table["tigge_wind_speed"] = np.hypot(
        table["tigge_u10"], table["tigge_v10"]
    )
    table["era5_wind_speed"] = np.hypot(table["era5_u10"], table["era5_v10"])
    table["error"] = table["era5_wind_speed"] - table["tigge_wind_speed"]
    return table.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_metrics(name: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - actual
    squared_error = np.mean(errors**2)
    total_variance = np.sum((actual - np.mean(actual)) ** 2)
    return {
        "model": name,
        "samples": len(actual),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(squared_error)),
        "r2": float(1 - np.sum(errors**2) / total_variance),
        "bias": float(np.mean(errors)),
    }


def main() -> None:
    table = build_training_table()
    table = table.sort_values("valid_time")
    split_time = table["valid_time"].sort_values().unique()[-1]
    train = table[table["valid_time"] < split_time]
    test = table[table["valid_time"] == split_time]
    if train.empty or test.empty:
        raise RuntimeError("Zaman bazlı eğitim/test ayrımı için yeterli kayıt yok.")

    features = [
        "tigge_u10",
        "tigge_v10",
        "tigge_wind_speed",
        "tigge_t2m",
        "tigge_d2m",
        "tigge_tcc",
        "tigge_tp",
        "ascat_wind_speed",
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
    actual = test["era5_wind_speed"].to_numpy()
    baseline = test["tigge_wind_speed"].to_numpy()

    OUTPUT_ROOT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_ROOT / "training_table.csv", index=False)
    pd.DataFrame(
        [
            calculate_metrics("TIGGE baseline", actual, baseline),
            calculate_metrics("TIGGE + ASCAT XGBoost correction", actual, corrected),
        ]
    ).to_csv(OUTPUT_ROOT / "metrics.csv", index=False)
    model.save_model(OUTPUT_ROOT / "xgboost_wind_error_correction.json")
    test.assign(
        corrected_wind_speed=corrected,
        baseline_wind_speed=baseline,
    ).to_csv(OUTPUT_ROOT / "test_predictions.csv", index=False)
    print(pd.DataFrame(
        [
            calculate_metrics("TIGGE baseline", actual, baseline),
            calculate_metrics("TIGGE + ASCAT XGBoost correction", actual, corrected),
        ]
    ).to_string(index=False))


if __name__ == "__main__":
    main()
