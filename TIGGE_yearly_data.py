from datetime import datetime
from os import environ
from pathlib import Path
import time

import cdsapi
import pandas as pd
import requests
import xarray as xr
from dotenv import load_dotenv


YEAR = 2021
ASCAT_ROOT = Path(__file__).resolve().parent / "ascat_data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "ERA5_data"
DATASET = "tigge-forecasts"
OUTPUT_FILE = OUTPUT_ROOT / f"turkiye_akdeniz_tigge_{YEAR}_ascat_days.grib"
PARTS_ROOT = OUTPUT_ROOT / f"tigge_{YEAR}_ascat_monthly"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 60

load_dotenv(Path(__file__).resolve().parent / ".env")
API_URL = environ.get("TIGGE_API_URL", "https://ecds.ecmwf.int/api")
API_KEY = environ.get("TIGGE_API_KEY")


def ascat_days(year: int) -> list[str]:
    """Return days having an ASCAT observation at the requested forecast time."""
    paths = sorted(ASCAT_ROOT.glob(f"*/turkiye_akdeniz_ascat_{year}_*.nc"))
    if not paths:
        raise FileNotFoundError(
            f"{year} yılı ASCAT dosyası bulunamadı: {ASCAT_ROOT}"
        )

    timestamps: list[pd.Timestamp] = []
    for path in paths:
        with xr.open_dataset(path) as dataset:
            timestamps.extend(pd.to_datetime(dataset["time"].values))

    selected = sorted(
        {
            timestamp.strftime("%Y-%m-%d")
            for timestamp in timestamps
            if timestamp.year == year and timestamp.hour == 12 and timestamp.minute == 0
        }
    )
    if not selected:
        raise RuntimeError(
            f"{year} ASCAT verisinde 12:00 zaman damgası bulunamadı."
        )
    return selected


def build_request(month: int, days: list[str], year: int) -> dict[str, object]:
    grouped_days = [
        datetime.strptime(day, "%Y-%m-%d").strftime("%d")
        for day in days
    ]
    return {
        "origin": "ecmf",
        "year": str(year),
        "month": f"{month:02d}",
        "day": grouped_days,
        "time": "12:00",
        "level_type": "single_level",
        "variable": [
            "10_m_u_component_of_wind",
            "10_m_v_component_of_wind",
            "2_m_dewpoint_temperature",
            "2_m_temperature",
            "total_cloud_cover",
            "total_precipitation",
        ],
        "forecast_type": "control_forecast",
        "leadtime_hour": "12",
        "data_format": "grib",
        "area": [41, 23, 34, 37],
    }


def download_month(
    client: cdsapi.Client,
    month: int,
    days: list[str],
    output_file: Path,
) -> None:
    request = build_request(month, days, YEAR)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"TIGGE indiriliyor: {YEAR}-{month:02d}, "
                f"{len(days)} ASCAT günü, {request['time']} "
                f"(deneme {attempt}/{MAX_RETRIES})"
            )
            temporary_file = output_file.with_suffix(".grib.part")
            client.retrieve(DATASET, request).download(str(temporary_file))
            temporary_file.replace(output_file)
            return
        except requests.HTTPError as error:
            message = str(error)
            if "Maximum number queued requests" not in message:
                raise
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"{YEAR}-{month:02d} isteği ECMWF kuyruğu nedeniyle "
                    f"{MAX_RETRIES} denemeden sonra başarısız oldu."
                ) from error
            wait_seconds = RETRY_DELAY_SECONDS * attempt
            print(
                f"ECMWF kuyruğu dolu; {wait_seconds} saniye sonra yeniden "
                f"denenecek: {message}"
            )
            time.sleep(wait_seconds)


def main() -> None:
    if not API_KEY:
        raise RuntimeError("TIGGE_API_KEY .env içinde tanımlı değil.")

    days = ascat_days(YEAR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PARTS_ROOT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url=API_URL, key=API_KEY)
    month_files = []
    for month in range(1, 13):
        month_days = [
            day for day in days
            if datetime.strptime(day, "%Y-%m-%d").month == month
        ]
        if not month_days:
            continue
        month_file = PARTS_ROOT / f"{YEAR}_{month:02d}.grib"
        if month_file.exists() and month_file.stat().st_size > 0:
            print(f"Atlanıyor, ay dosyası zaten var: {month_file}")
        else:
            download_month(client, month, month_days, month_file)
        month_files.append(month_file)

    partial_output = OUTPUT_FILE.with_suffix(".grib.part")
    with partial_output.open("wb") as output:
        for month_file in month_files:
            with month_file.open("rb") as source:
                output.write(source.read())
    partial_output.replace(OUTPUT_FILE)
    print(f"Tamamlandı: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
