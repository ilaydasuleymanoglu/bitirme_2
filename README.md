# ASCAT Destekli TIGGE Rüzgar Hata Düzeltmesi

Bu klasör, **12 saatlik TIGGE rüzgar tahminlerinin**, tahmin başlangıç zamanındaki ASCAT gözlemleri ve ASOS/LTB istasyonlarıyla değerlendirilmesini içeren pilot makale çıktısını barındırır.

## İndirilebilir makale

- [PDF makale raporu](./wind_correction_output/wind_correction_paper.pdf)
- [LaTeX kaynak dosyası](./wind_correction_paper.tex)

LaTeX PDF’si üretmek için:

```bash
pdflatex wind_correction_paper.tex
```

Bu ortamda `pdflatex` bulunmadığı için PDF, aynı metrik ve grafiklerle doğrudan oluşturulmuştur. LaTeX kaynağı gerçek makale düzeni ve formülasyonları içerir.

## Özet

TIGGE tahmin çevrimi 12:00’da başlatılmış, aynı başlangıç zamanındaki ASCAT rüzgar gözlemi feature olarak kullanılmış ve +12 saat sonraki ERA5/ASOS değerleriyle doğrulama yapılmıştır. ASCAT hücreleri istasyonlara Thiessen/Voronoi bölgeleriyle atanmış, enlem alan düzeltmesi `cos(latitude)` ile yapılmıştır. XGBoost, TIGGE tahmin hatasını öğrenmiştir.

Rolling-origin doğrulamada:

| Hedef | TIGGE RMSE | TIGGE + ASCAT + XGBoost RMSE | İyileşme |
|---|---:|---:|---:|
| LTB/ASOS | 2.988 m/s | 1.432 m/s | %52.07 |
| ERA5 | 2.386 m/s | 1.192 m/s | %50.04 |

Diebold–Mariano sonuçları:

| Hedef | RMSE iyileşmesi | %95 bootstrap aralığı | DM p-değeri |
|---|---:|---:|---:|
| LTB/ASOS | 1.498 m/s | 0.793–2.158 m/s | 0.0077 |
| ERA5 | 1.174 m/s | 0.325–1.966 m/s | 0.0875 |

LTB hedefinde iyileşme istatistiksel olarak anlamlıdır. ERA5 hedefinde iyileşme yönü güçlü olsa da mevcut örneklemle `p < 0.05` seviyesine ulaşmamıştır.

> **Bilimsel kapsam:** Sonuçlar Ocak 2021 ve 18 bağımsız test örneğine dayanan pilot bulgulardır. Mevsimsel ve yıllararası genelleme için daha uzun TIGGE dönemi gereklidir.

## Formülasyonlar

10 m rüzgar hızı:

$$
U = \sqrt{u_{10}^{2}+v_{10}^{2}}
$$

TIGGE hata hedefi:

$$
e_t = U_t^{ERA5} - U_t^{TIGGE}
$$

XGBoost düzeltmesi:

$$
\widehat{U}_t^{corr}
= U_t^{TIGGE}+\widehat{e}_t(X_t)
$$

Thiessen alan ağırlıklı ASCAT:

$$
\overline{U}_{s,t}^{ASCAT}
=
\frac{\sum_{j\in V_s}\cos(\phi_j)U_{j,t}^{ASCAT}}
{\sum_{j\in V_s}\cos(\phi_j)}
$$

Değerlendirme:

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\widehat{U}_i-U_i)^2}
$$

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|\widehat{U}_i-U_i|
$$

Z-skoru:

$$
z_i=\frac{x_i-\bar{x}}{s_x}
$$

## Grafikler

![Model RMSE karşılaştırması](./wind_correction_output/figures/model_rmse_comparison.png)

![Bootstrap RMSE iyileşmesi](./wind_correction_output/figures/rmse_bootstrap_improvement.png)

![Thiessen ASCAT kapsamı](./wind_correction_output/figures/thiessen_coverage.png)

## Ham sonuçlar

- [Rolling tüm model metrikleri](./wind_correction_output/paper_rolling_metrics.csv)
- [Rolling katman metrikleri](./wind_correction_output/paper_rolling_fold_metrics.csv)
- [Tüm tahminler](./wind_correction_output/paper_rolling_predictions.csv)
- [Bootstrap ve DM sonuçları](./wind_correction_output/paper_significance.csv)
- [Thiessen kapsamı](./wind_correction_output/paper_thiessen_coverage.csv)
- [Z-skorları](./wind_correction_output/ltb_thiessen_zscores.csv)
- [İstasyon-temelli eğitim tablosu](./wind_correction_output/ltb_thiessen_table.csv)

## Kullanıcı tarafından eklenecek 5 görsel

LaTeX dosyasında beş ayrı placeholder bırakılmıştır. Aşağıdaki önerilen görseller sonradan eklenebilir:

1. Türkiye/Akdeniz çalışma alanı ve ASOS istasyonları.
2. ASCAT grid ve Thiessen poligonları.
3. TIGGE, ASCAT, LTB ve ERA5 zaman serileri.
4. İstasyon veya grid bazlı RMSE fark haritası.
5. Rüzgar profilleri ve hata dağılımları.

## Yeniden üretim

```bash
python evaluate_ltb_thiessen.py
python paper_validation_protocol.py
python generate_paper_outputs.py
```

`generate_paper_outputs.py`, sistem Python'ındaki NumPy/Matplotlib ABI
uyumsuzluğunu önlemek için mevcutsa otomatik olarak `.venv/bin/python`
ortamında çalışır. İsterseniz ortamı doğrudan etkinleştirerek de çalıştırabilirsiniz:

```bash
source .venv/bin/activate
python generate_paper_outputs.py
```

İstatistiksel sonuçların yeniden üretilebilmesi için rastgele tohum `42`, bootstrap tekrar sayısı `2000` ve rolling katman sayısı `5` olarak sabitlenmiştir.
