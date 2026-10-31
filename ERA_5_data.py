from calendar import monthrange
from datetime import date
from os import environ
from pathlib import Path

import cdsapi
from dotenv import load_dotenv


YEAR = 2021
OUTPUT_ROOT = Path(__file__).resolve().parent / "ERA5_data"
DATASET = "tigge-forecasts"
load_dotenv(Path(__file__).resolve().parent / ".env")
API_URL = environ.get("TIGGE_API_URL", "https://ecds.ecmwf.int/api")
API_KEY = environ.get("TIGGE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TIGGE_API_KEY ortam değişkeni tanımlı değil. "
        "API anahtarınızı bu değişkene atayın."
    )


def download_daily_data(client: cdsapi.Client, day: date) -> None:
    month_name = f"{day.month:02d}"
    day_name = f"{day.day:02d}"
    output_dir = OUTPUT_ROOT / month_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"turkiye_akdeniz_era5_{day.year}_{month_name}_{day_name}.grib"
    )

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Atlanıyor, dosya zaten var: {output_file}")
        return

    request = {
        "origin": "ecmwf",
        "year": str(day.year),
        "month": month_name,
        "day": day_name,
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
        "forecast_type": "high_resolution_forecast",
        "leadtime_hour": ["12"],
        "data_format": "grib",
        "area": [41, 23, 34, 37],
    }

    print(f"İndiriliyor: {day.isoformat()} -> {output_file}")
    client.retrieve(DATASET, request).download(str(output_file))


def main() -> None:
    client = cdsapi.Client(url=API_URL, key=API_KEY)
    for month in range(1, 13):
        for day_number in range(1, monthrange(YEAR, month)[1] + 1):
            download_daily_data(client, date(YEAR, month, day_number))


if __name__ == "__main__":
    main()
