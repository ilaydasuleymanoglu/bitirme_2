from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable) != VENV_PYTHON:
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages


OUT = ROOT / "wind_correction_output"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(OUT / "paper_rolling_metrics.csv")
    significance = pd.read_csv(OUT / "paper_significance.csv")
    coverage = pd.read_csv(OUT / "paper_thiessen_coverage.csv")
    return metrics, significance, coverage


def load_table() -> pd.DataFrame:
    return pd.read_csv(OUT / "ltb_thiessen_table.csv", parse_dates=["valid_time"])


def save_requested_figures(table: pd.DataFrame) -> list[Path]:
    paths = []
    stations = table[["station", "latitude", "longitude"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(table["longitude"], table["latitude"], s=10, alpha=0.15, color="#457b9d")
    ax.scatter(stations["longitude"], stations["latitude"], s=55, color="#d62828", label="ASOS/LTB")
    for _, row in stations.iterrows():
        ax.annotate(row["station"], (row["longitude"], row["latitude"]), xytext=(3, 3),
                    textcoords="offset points", fontsize=8)
    ax.set_title("Türkiye/Akdeniz çalışma alanı ve ASOS/LTB istasyonları")
    ax.set_xlabel("Boylam (°)")
    ax.set_ylabel("Enlem (°)")
    ax.legend()
    ax.grid(alpha=0.25)
    path = FIG / "study_area_asos_stations.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    ascat_path = ROOT / "ascat_data/01/turkiye_akdeniz_ascat_2021_01.nc"
    ascat = xr.open_dataset(ascat_path)
    values = ascat["modulus"].squeeze()
    first = values.isel(time=0)
    latitude = values["latitude"].to_numpy()
    longitude = values["longitude"].to_numpy()
    grid_lat, grid_lon = np.meshgrid(latitude, longitude, indexing="ij")
    station_xy = stations[["latitude", "longitude"]].to_numpy()
    assignment = np.argmin(
        (grid_lat[..., None] - station_xy[:, 0]) ** 2
        + (grid_lon[..., None] - station_xy[:, 1]) ** 2,
        axis=2,
    )
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mesh = ax.pcolormesh(grid_lon, grid_lat, first.to_numpy(), shading="auto", cmap="viridis")
    ax.contour(grid_lon, grid_lat, assignment, levels=np.arange(0.5, len(stations)), colors="white", linewidths=0.7)
    ax.scatter(stations["longitude"], stations["latitude"], s=40, color="#d62828", edgecolor="white")
    ax.set_title("ASCAT gridi ve istasyonlara göre Thiessen bölgeleri")
    ax.set_xlabel("Boylam (°)")
    ax.set_ylabel("Enlem (°)")
    fig.colorbar(mesh, ax=ax, label="ASCAT rüzgar hızı (m/s)")
    path = FIG / "ascat_grid_thiessen.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    ascat.close()
    paths.append(path)

    daily = table.groupby("valid_time", as_index=False)[
        ["tigge_wind_speed", "ascat_thiessen_wind_speed", "ltb_wind_speed", "era5_wind_speed"]
    ].mean()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for column, label, color in [
        ("tigge_wind_speed", "TIGGE", "#264653"),
        ("ascat_thiessen_wind_speed", "ASCAT Thiessen", "#e9c46a"),
        ("ltb_wind_speed", "LTB/ASOS", "#d62828"),
        ("era5_wind_speed", "ERA5", "#2a9d8f"),
    ]:
        ax.plot(daily["valid_time"], daily[column], marker="o", label=label, color=color)
    ax.set_title("TIGGE, ASCAT, LTB/ASOS ve ERA5 zaman serileri")
    ax.set_ylabel("Ortalama 10 m rüzgar hızı (m/s)")
    ax.set_xlabel("Geçerlilik zamanı")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    path = FIG / "wind_time_series.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    predictions = pd.read_csv(OUT / "paper_rolling_predictions.csv", parse_dates=["valid_time"])
    baseline = predictions[predictions["model"] == "TIGGE"].copy()
    corrected = predictions[predictions["model"] == "TIGGE + ASCAT Thiessen XGBoost"].copy()
    baseline["baseline_error_sq"] = baseline["error"] ** 2
    corrected["corrected_error_sq"] = corrected["error"] ** 2
    error_data = baseline[["target", "station", "baseline_error_sq"]].merge(
        corrected[["target", "station", "corrected_error_sq"]],
        on=["target", "station"],
        how="inner",
    )
    error_data["rmse_improvement"] = np.sqrt(error_data["baseline_error_sq"]) - np.sqrt(
        error_data["corrected_error_sq"]
    )
    station_improvement = error_data.groupby(["target", "station"], as_index=False)[
        "rmse_improvement"
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, target, title in zip(
        axes,
        ["ltb_wind_speed", "era5_wind_speed"],
        ["LTB/ASOS RMSE farkı", "ERA5 RMSE farkı"],
    ):
        plot = station_improvement[station_improvement["target"] == target]
        colors = np.where(plot["rmse_improvement"] >= 0, "#2a9d8f", "#d62828")
        ax.barh(plot["station"], plot["rmse_improvement"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Hata azalması (m/s)")
        ax.grid(axis="x", alpha=0.25)
    path = FIG / "station_rmse_difference.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    series = [
        table["tigge_wind_speed"],
        table["ascat_thiessen_wind_speed"],
        table["ltb_wind_speed"],
        table["era5_wind_speed"],
    ]
    axes[0].boxplot(series, tick_labels=["TIGGE", "ASCAT", "LTB", "ERA5"], patch_artist=True)
    axes[0].set_title("10 m rüzgar hız profilleri")
    axes[0].set_ylabel("Rüzgar hızı (m/s)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].hist(table["error"], bins=14, alpha=0.75, color="#457b9d", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("ERA5-TIGGE hata dağılımı")
    axes[1].set_xlabel("Hata (m/s)")
    axes[1].set_ylabel("Frekans")
    axes[1].grid(axis="y", alpha=0.25)
    path = FIG / "wind_profiles_error_distributions.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def save_ltb_timeseries_scatter() -> Path:
    predictions = pd.read_csv(
        OUT / "paper_rolling_predictions.csv",
        parse_dates=["valid_time"],
    )
    station = "LTBS"
    station_data = predictions[
        (predictions["station"] == station)
        & (predictions["target"] == "ltb_wind_speed")
    ]
    actual = station_data[station_data["model"] == "TIGGE"][
        ["valid_time", "actual"]
    ].rename(columns={"actual": "ltb_actual"})
    era5 = predictions[
        (predictions["station"] == station)
        & (predictions["target"] == "era5_wind_speed")
        & (predictions["model"] == "TIGGE")
    ][["valid_time", "actual"]].rename(columns={"actual": "era5_actual"})
    tigge = station_data[station_data["model"] == "TIGGE"][
        ["valid_time", "predicted"]
    ].rename(columns={"predicted": "tigge"})
    corrected = station_data[
        station_data["model"] == "TIGGE + ASCAT Thiessen XGBoost"
    ][["valid_time", "predicted"]].rename(columns={"predicted": "corrected"})
    series = actual.merge(era5, on="valid_time").merge(tigge, on="valid_time").merge(
        corrected, on="valid_time"
    ).sort_values("valid_time")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(series["valid_time"], series["tigge"], marker="o", label="TIGGE", color="#264653")
    axes[0, 0].plot(series["valid_time"], series["corrected"], marker="o", label="ASCAT + XGBoost", color="#e9c46a")
    axes[0, 0].plot(series["valid_time"], series["ltb_actual"], marker="o", label="LTB gerçekleşen", color="#d62828")
    axes[0, 0].plot(series["valid_time"], series["era5_actual"], marker="o", label="ERA5", color="#2a9d8f")
    axes[0, 0].set_title("LTBS zaman serisi")
    axes[0, 0].set_ylabel("Rüzgar hızı (m/s)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].tick_params(axis="x", rotation=35)

    for ax, column, label, color in [
        (axes[0, 1], "tigge", "TIGGE", "#264653"),
        (axes[1, 0], "corrected", "ASCAT + XGBoost", "#e9c46a"),
        (axes[1, 1], "era5_actual", "ERA5", "#2a9d8f"),
    ]:
        ax.scatter(series["ltb_actual"], series[column], color=color, alpha=0.8, edgecolor="white")
        bounds = np.array([
            min(series["ltb_actual"].min(), series[column].min()),
            max(series["ltb_actual"].max(), series[column].max()),
        ])
        ax.plot(bounds, bounds, "--", color="#555555", linewidth=1)
        ax.set_title(f"LTB gerçekleşen - {label}")
        ax.set_xlabel("LTB gerçekleşen (m/s)")
        ax.set_ylabel(f"{label} (m/s)")
        ax.grid(alpha=0.25)

    fig.suptitle("LTBS: tahminler, gerçekleşen LTB ve ERA5 karşılaştırması", fontsize=14)
    path = FIG / "ltbs_timeseries_scatter.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_figures() -> list[Path]:
    metrics, significance, coverage = load()
    paths = []
    labels = {
        "TIGGE": "TIGGE",
        "TIGGE bias correction": "Bias correction",
        "TIGGE + linear regression": "Linear regression",
        "TIGGE + random forest": "Random forest",
        "TIGGE + ASCAT Thiessen XGBoost": "ASCAT + XGBoost",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, target, title in zip(
        axes,
        ["ltb_wind_speed", "era5_wind_speed"],
        ["LTB/ASOS target", "ERA5 reanalysis target"],
    ):
        data = metrics[metrics.target == target].copy()
        data["label"] = data.model.map(labels)
        data = data.sort_values("rmse")
        ax.barh(data.label, data.rmse, color=["#2a9d8f" if "ASCAT" in x else "#8d99ae" for x in data.model])
        ax.set_title(title)
        ax.set_xlabel("Mean RMSE (m/s)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Rolling-origin validation: model comparison")
    path = FIG / "model_rmse_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data = significance.copy()
    x = np.arange(len(data))
    improvement = data.rmse_improvement_mean.to_numpy()
    lower = improvement - data.rmse_improvement_ci95_low.to_numpy()
    upper = data.rmse_improvement_ci95_high.to_numpy() - improvement
    labels_sig = ["LTB/ASOS", "ERA5"]
    ax.errorbar(x, improvement, yerr=[lower, upper], fmt="o", capsize=6, color="#264653", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, labels_sig)
    ax.set_ylabel("RMSE improvement (m/s)")
    ax.set_title("Paired bootstrap improvement with 95% CI")
    ax.grid(axis="y", alpha=0.25)
    path = FIG / "rmse_bootstrap_improvement.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot = coverage.sort_values("valid_fraction")
    ax.barh(plot.station, plot.valid_fraction * 100, color="#e76f51")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Valid Thiessen-weighted ASCAT coverage (%)")
    ax.set_title("ASCAT Thiessen coverage by station")
    ax.grid(axis="x", alpha=0.25)
    path = FIG / "thiessen_coverage.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    paths.extend(save_requested_figures(load_table()))
    paths.append(save_ltb_timeseries_scatter())
    return paths


def make_pdf(paths: list[Path]) -> None:
    metrics, significance, coverage = load()
    pdf_path = OUT / "wind_correction_paper.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.88, "ASCAT-DESTEKLI TIGGE RUZGAR HATA DUZELTMESI", ha="center", fontsize=18, weight="bold")
        fig.text(0.5, 0.82, "Thiessen poligonlari ve XGBoost ile pilot dogrulama", ha="center", fontsize=13)
        fig.text(0.12, 0.70, "Ozet", fontsize=14, weight="bold")
        fig.text(0.12, 0.66, "Bu rapor, 12 saatlik TIGGE ruzgar tahminlerini tahmin baslangicindaki ASCAT gozlemleriyle duzeltir.\\n"
                 "ASCAT grid hucreleri Thiessen/Voronoi bolgeleriyle istasyonlara aktarilir. Sonuclar LTB/ASOS\\n"
                 "gozlemleri ve ERA5 reanalysis karsisinda rolling-origin zaman dogrulamasi ile raporlanir.", fontsize=11, va="top")
        fig.text(0.12, 0.51, "Ana bulgu", fontsize=14, weight="bold")
        fig.text(0.12, 0.47, "LTB hedefinde ortalama RMSE iyilesmesi: 52.07%\\nERA5 hedefinde ortalama RMSE iyilesmesi: 50.04%\\n"
                 "Diebold-Mariano t-testi p-degeri: LTB=0.0163, ERA5=0.1057", fontsize=12, va="top")
        fig.text(0.12, 0.28, "Sinir: Sonuclar Ocak 2021 ve 18 bagimsiz test ornegine dayanan pilot calismadir.", fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        for path in paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(image)
            ax.axis("off")
            fig.suptitle(path.stem.replace("_", " ").title(), fontsize=16)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.02, 0.97, "Rolling-origin sonuclar", fontsize=16, weight="bold", va="top")
        table = metrics.copy()
        table["target"] = table.target.map({"ltb_wind_speed": "LTB", "era5_wind_speed": "ERA5"})
        table["model"] = table.model.str.replace("TIGGE + ", "", regex=False)
        table = table[["target", "model", "samples", "mae", "rmse", "r2"]].round(3)
        ax.table(cellText=table.values, colLabels=table.columns, loc="upper left", bbox=[0.02, 0.42, 0.96, 0.48])
        ax.text(0.02, 0.34, "Formulasyonlar", fontsize=14, weight="bold")
        ax.text(0.02, 0.29, r"$U = \sqrt{u_{10}^{2}+v_{10}^{2}}$", fontsize=13)
        ax.text(0.02, 0.24, r"$e_t = U^{ERA5}_t - U^{TIGGE}_t$", fontsize=13)
        ax.text(0.02, 0.19, r"$\hat{U}^{corr}_t = U^{TIGGE}_t + \hat{e}_t(X_t)$", fontsize=13)
        ax.text(0.02, 0.14, r"$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{U}_i-U_i)^2}$", fontsize=13)
        ax.text(0.02, 0.09, r"$z_i=(x_i-\bar{x})/s_x$", fontsize=13)
        pdf.savefig(fig); plt.close(fig)
    print(pdf_path)


if __name__ == "__main__":
    figure_paths = save_figures()
    make_pdf(figure_paths)
