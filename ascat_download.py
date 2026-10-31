import xarray as xr
from calendar import monthrange
from pathlib import Path

YEAR = 2021

# 1. Doğru veri seti ID'sine sahip ERDDAP OPeNDAP bağlantısı
url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdQMwind1day"

print("Sunucuya bağlanılıyor, lütfen bekleyin...")
ds = xr.open_dataset(url)

# 2. İstediğiniz koordinatlara göre filtreleme (Slice)
# Orijinal veri setindeki isimler: 'latitude', 'longitude' ve 'time'
filtered_ds = ds.sel(
    latitude=slice(30.0, 42.5),
    longitude=slice(25.0, 45.0)
)
if ds.time.values[-1].astype("datetime64[Y]").astype(int) + 1970 < YEAR:
    raise RuntimeError(
        f"ERDDAP kaynağında {YEAR} verisi yok; son tarih: {ds.time.values[-1]}"
    )

# 3. Her ayı ayrı bir klasöre, sadece rüzgar hızı (modulus) verisiyle kaydet.
output_root = Path(__file__).resolve().parent / "ascat_data"
output_root.mkdir(parents=True, exist_ok=True)

for month in range(1, 13):
    month_name = f"{month:02d}"
    start = f"{YEAR}-{month_name}-01T12:00:00"
    last_day = monthrange(YEAR, month)[1]
    end = f"{YEAR}-{month_name}-{last_day:02d}T12:00:00"
    month_dir = output_root / month_name
    month_dir.mkdir(parents=True, exist_ok=True)
    output_name = month_dir / f"turkiye_akdeniz_ascat_{YEAR}_{month_name}.nc"

    print(f"{month_name}. ay verileri filtreleniyor...")
    month_slice = filtered_ds.sel(time=slice(start, end))
    x_wind = month_slice["x_wind"].load()
    y_wind = month_slice["y_wind"].load()
    month_data = xr.Dataset({"x_wind": x_wind, "y_wind": y_wind})
    month_data = month_data.assign(
        modulus=(month_data["x_wind"] ** 2 + month_data["y_wind"] ** 2) ** 0.5
    )[["modulus"]]
    print(f"'{output_name}' kaydediliyor...")
    month_data.to_netcdf(output_name)

ds.close()
print(f"İşlem başarıyla tamamlandı! Dosyalar '{output_root}' klasörüne kaydedildi.")
