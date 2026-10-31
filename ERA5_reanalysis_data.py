from calendar import monthrange
from os import environ
from pathlib import Path

import cdsapi
from dotenv import load_dotenv


YEAR = 2021
OUTPUT_ROOT = Path(__file__).resolve().parent / "ERA_5_REANALYSİS_Data"
DATASET = "reanalysis-era5-single-levels"

load_dotenv(Path(__file__).resolve().parent / ".env")
API_URL = environ.get("CDS_API_URL", "https://ecds.ecmwf.int/api")
API_KEY = environ.get("CDS_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "CDS_API_KEY ortam değişkeni tanımlı değil. "
        "API anahtarınızı .env dosyasına ekleyin."
    )


def download_month(client: cdsapi.Client, month: int) -> None:
    month_name = f"{month:02d}"
    days = [
        f"{day:02d}"
        for day in range(1, monthrange(YEAR, month)[1] + 1)
    ]
    times = [f"{hour:02d}:00" for hour in range(24)]
    output_dir = OUTPUT_ROOT / month_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"turkiye_akdeniz_era5_reanalysis_{YEAR}_{month_name}.grib"
    )

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Atlanıyor, dosya zaten var: {output_file}")
        return

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "10m_u_component_of_neutral_wind",
            "10m_v_component_of_neutral_wind",
            "instantaneous_10m_wind_gust",
        ],
        "year": [str(YEAR)],
        "month": [month_name],
        "day": days,
        "time": times,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [42.75, 21.25, 35, 29],
    }

    print(f"ERA5 reanalysis indiriliyor: {output_file}")
    client.retrieve(DATASET, request).download(str(output_file))
    print(f"İndirme tamamlandı: {output_file}")


def main() -> None:
    client = cdsapi.Client(url=API_URL, key=API_KEY)
    for month in range(1, 13):
        download_month(client, month)


if __name__ == "__main__":
    main()
