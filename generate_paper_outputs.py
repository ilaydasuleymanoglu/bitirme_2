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
from matplotlib.backends.backend_pdf import PdfPages


OUT = ROOT / "wind_correction_output"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(OUT / "paper_rolling_metrics.csv")
    significance = pd.read_csv(OUT / "paper_significance.csv")
    coverage = pd.read_csv(OUT / "paper_thiessen_coverage.csv")
    return metrics, significance, coverage


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
                 "Diebold-Mariano p-degeri: LTB=0.0077, ERA5=0.0875", fontsize=12, va="top")
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
