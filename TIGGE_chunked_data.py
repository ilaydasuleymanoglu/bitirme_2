from calendar import monthrange
from datetime import date, timedelta
from os import environ
from pathlib import Path

import cdsapi
from dotenv import load_dotenv


YEAR = 2021
CHUNK_DAYS = 3
OUTPUT_ROOT = Path(__file__).resolve().parent / "ERA5_data"
DATASET = "tigge-forecasts"

load_dotenv(Path(__file__).resolve().parent / ".env")
API_URL = environ.get("TIGGE_API_URL", "https://ecds.ecmwf.int/api")
API_KEY = environ.get("TIGGE_API_KEY")

if not API_KEY:
    raise RuntimeError("TIGGE_API_KEY .env içinde tanımlı değil.")


def download_chunk(client: cdsapi.Client, start: date, end: date) -> None:
    month_name = f"{start.month:02d}"
    output_dir = OUTPUT_ROOT / month_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"turkiye_akdeniz_tigge_{start:%Y_%m_%d}"
        f"_to_{end:%Y_%m_%d}.grib"
    )

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Atlanıyor, dosya zaten var: {output_file}")
        return

    days = [
        f"{day:02d}"
        for day in range(start.day, end.day + 1)
    ]
    request = {
        "origin": "ecmf",
        "year": str(YEAR),
        "month": month_name,
        "day": days,
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

    print(f"TIGGE indiriliyor: {start} - {end} -> {output_file}")
    client.retrieve(DATASET, request).download(str(output_file))
    print(f"Tamamlandı: {output_file}")


def main() -> None:
    client = cdsapi.Client(url=API_URL, key=API_KEY)
    current = date(YEAR, 1, 1)
    final = date(YEAR, 12, 31)
    while current <= final:
        end = min(current + timedelta(days=CHUNK_DAYS - 1), final)
        download_chunk(client, current, end)
        current = end + timedelta(days=1)


if __name__ == "__main__":
    main()
