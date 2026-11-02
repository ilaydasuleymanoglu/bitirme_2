# ASCAT Destekli TIGGE Rüzgar Tahmin Düzeltmesi

Bu klasör, lisans bitirme tezi kapsamında 12 saatlik TIGGE 10 m rüzgar
tahminlerinin ASCAT gözlemleriyle düzeltilmesini ve LTB/ASOS ile ERA5
referanslarına karşı değerlendirilmesini içerir. Çalışma alanı Türkiye'nin
Akdeniz bölgesidir.

## Tez için kısa özet

TIGGE tahminleri her gün 12:00 başlangıç zamanında alınmış, aynı zamandaki
ASCAT rüzgar bilgisi istasyonlara yaklaşık Thiessen/en yakın hücre atamasıyla
aktarılmış ve +12 saat sonraki değerler değerlendirilmiştir. XGBoost, TIGGE
hata hedefini öğrenmiştir. LTB ve ERA5 hedefleri ayrı modellerle ele alınmış,
LTB hatası ERA5 hatasından türetilmemiştir.

Analiz Ocak--Nisan 2021 dönemindeki dört aylık aylık GRIB dosyalarını kullanır:

- 114 geçerli forecast başlangıcı,
- ilk başlangıçta geçmiş eğitim verisi olmadığı için 113 dış örnek origin,
- 432 eşleşmiş istasyon-zaman kaydı,
- gerekli hedef ve özellikleri sonlu olan 428 nihai dış örnek test kaydı.

Buradaki 432 kayıt, TIGGE--ASCAT--LTB/ASOS eşleştirmesinden sonra oluşan ham
analiz tablosudur. Dışarıda kalan 4 kayıt ilk geçerlilik zamanına
(2021-01-04) aittir: öncesinde eğitim verisi olmadığı için yalnızca eğitimde
kullanılır, dış örnek test edilmez.

> **Not (düzeltme sonrası):** Bu README'deki sayılar Kalman, ASCAT (D-1
> kompozit), persistence ve DM kaybı düzeltmelerinden önceki çalıştırmaya
> aittir. `evaluate_ltb_thiessen.py` ve `paper_validation_protocol.py`
> yeniden çalıştırıldıktan sonra güncellenmelidir. Eski grid pilotunun
> çıktıları (`metrics.csv` vb.) `wind_correction_output/legacy_grid_pilot/`
> klasörüne taşınmıştır ve tez sonuçlarında kullanılmaz.
Bu nedenle sonuç tablolarındaki örnek sayısı 428'dir; 432 ve 428 farklı işlem
aşamalarını ifade eder ve aynı örneklem büyüklüğü olarak yorumlanmamalıdır.

## Güncel ana sonuçlar

Metrikler tüm dış örnek tahmin hatalarının havuzlanmasıyla hesaplanmıştır;
günlük fold metriklerinin ortalaması değildir.

| Hedef | TIGGE RMSE | ASCAT-XGBoost RMSE | Göreli iyileşme |
|---|---:|---:|---:|
| LTB/ASOS | 1.863 m/s | 1.483 m/s | %20.37 |
| ERA5 reanalysis referansı | 1.439 m/s | 1.029 m/s | %28.54 |

Karşılaştırmaya istasyon bazlı persistence baseline'ı da eklenmiştir. Her
istasyon için önceki geçerli rüzgar hızı kullanılmış, ilk kullanılabilir
zamanda TIGGE değeriyle başlatılmıştır. Persistence RMSE değerleri LTB için
2.575 m/s, ERA5 için 2.663 m/s'dir.

### Anlamlılık ve belirsizlik

İstasyonlar aynı forecast origin içinde bağımsız gözlemler kabul edilmemiştir.
RMSE farkları 113 origin kümesi üzerinden, 7 günlük blok bootstrap ile
hesaplanmıştır. DM testi lag-1 Newey--West varyansı ve asimptotik z yaklaşımı
kullanır.

| Hedef | RMSE iyileşmesi | %95 blok bootstrap aralığı | DM z p-değeri |
|---|---:|---:|---:|
| LTB/ASOS | 0.396 m/s | 0.258--0.548 m/s | 0.1060 |
| ERA5 | 0.468 m/s | 0.338--0.599 m/s | 0.0025 |

Bu nedenle ERA5 sonucu %10 düzeyinde anlamlıdır; LTB/ASOS sonucu %10
düzeyinde anlamlı değildir. Newey--West lag duyarlılığı da raporlanmıştır:

| Hedef | Lag 1 | Lag 3 | Lag 7 |
|---|---:|---:|---:|
| LTB/ASOS | 0.1060 | 0.1190 | 0.1657 |
| ERA5 | 0.0025 | 0.0013 | 0.0017 |

### İstasyon-eşitlikli sonuçlar

Toplu metrikler kayıt sayısı yüksek istasyonlardan daha fazla etkilenebileceği
için istasyonların eşit ağırlıklı sonuçları ayrıca hesaplanmıştır:

| Hedef | TIGGE eşit-istasyon RMSE | ASCAT-XGBoost eşit-istasyon RMSE |
|---|---:|---:|
| LTB/ASOS | 2.560 m/s | 1.964 m/s |
| ERA5 | 1.301 m/s | 1.244 m/s |

Bu tablo iyileşmenin yalnızca toplam kayıt sayısı yüksek istasyonlardan
kaynaklanıp kaynaklanmadığını incelemek için kullanılır; istasyonların veri
kapsamları yine de eşit değildir.

## Yöntem ve veri sızıntısı kontrolü

Her forecast origin için:

1. Eğitim verisi yalnızca geçmiş valid time kayıtlarından oluşturulur.
2. Test origin'i eğitim kümesine alınmaz.
3. Tahmin başlangıç zamanı valid time'dan sonra olamaz.
4. ASCAT zamanı tahmin başlangıcından sonra kullanılamaz.
5. Hedef ve hata sütunları model özelliklerine alınmaz.
6. Özelliklerin sonlu değerlerden oluştuğu denetlenir.

XGBoost hiperparametreleri her origin'de yalnızca geçmiş eğitim kümesinin son
%20'si üzerinde iç zaman doğrulamasıyla seçilir. Seçilen ayarlarla model,
gelecekteki test origin'i görülmeden tüm geçmiş eğitim kümesinde yeniden
eğitilir. Bu nested temporal validation'dır; bağımsız bir yıl arası test
dönemi değildir.

## Eksik LTB verileri ve ERA5 fallback

Veri hattı, LTB/ASOS kaydı bulunmadığında aynı valid time ve istasyon konumu
için ERA5 rüzgarını `ERA5 fallback` olarak işaretleyebilir. Bu değer bağımsız
LTB gözlemi olarak yorumlanmaz.

Ocak--Nisan 2021 analiz tablosunda:

- LTB/ASOS kaydı: 432,
- ERA5 fallback kaydı: 0.

Dolayısıyla mevcut sayısal sonuçlarda ERA5 fallback kullanılmamıştır. LTBH
istasyonu, LTB verisi nedeniyle değil, geçerli ASCAT Thiessen değeri
bulunamadığı için ana model tablosuna alınmamıştır.

## Veri kapsamı ve sınırlılıklar

Ocak--Nisan ASCAT Thiessen kapsamı istasyonlar arasında farklıdır:

- LTBS/LTBU: %100,
- LTBG: %98.31,
- LTFE: %87.29,
- LTFK: %84.75,
- LTFD: %79.66,
- LTBH: %0.

Çalışmanın temel sınırlılıkları şunlardır:

- Veri yalnızca 2021'in ilk dört ayını kapsar; yıllararası genelleme yapılamaz.
- ERA5 bağımsız bir yer gözlemi değil, reanalysis referansıdır.
- ASCAT'ın operasyonel yayın gecikmesi arşiv dosyalarından doğrulanmamıştır.
- LTB ve ASCAT/TIGGE gridleri noktasal ve mekansal olarak tam eşdeğer değildir.
- İstasyonların gözlem sayıları eşit değildir.
- Thiessen yaklaşımı gerçek atmosferik akış yönünü ve kıyı/deniz etkisini
  tamamen temsil etmez.
- Z-testi ve blok uzunluğu duyarlılık analizleri yapılmış olsa da tek yıllık
  veri dönemi istatistiksel genellemeyi sınırlar.

ASCAT operasyonel uygunluk durumu kasıtlı olarak `not_verified` tutulmuştur.
Bunu doğrulamak için ürün dokümantasyonu veya gerçek zamanlı yayın ve erişim
zamanlarını içeren operasyonel loglar gerekir.

## Tez raporu ve görseller

- [LaTeX tez/makale kaynağı](./wind_correction_paper.tex)
- [Üretilmiş PDF raporu](./wind_correction_output/wind_correction_paper.pdf)
- [Model RMSE grafiği](./wind_correction_output/figures/model_rmse_comparison.png)
- [Blok bootstrap iyileşme grafiği](./wind_correction_output/figures/rmse_bootstrap_improvement.png)
- [Thiessen ASCAT kapsam grafiği](./wind_correction_output/figures/thiessen_coverage.png)
- [LTB zaman serileri](./wind_correction_output/figures/ltb_forecast_timeseries.png)
- [LTB saçılım grafikleri](./wind_correction_output/figures/ltb_forecast_scatter.png)
- [Çalışma alanı ve istasyonlar](./wind_correction_output/figures/study_area_asos_stations.png)
- [ASCAT grid ve Thiessen ataması](./wind_correction_output/figures/ascat_grid_thiessen.png)
- [TIGGE, ASCAT, LTB ve ERA5 zaman serileri](./wind_correction_output/figures/wind_time_series.png)
- [İstasyon RMSE farkları](./wind_correction_output/figures/station_rmse_difference.png)
- [10 m hız ve hata dağılımları](./wind_correction_output/figures/wind_profiles_error_distributions.png)

Beşinci görsel gerçek düşey rüzgar profili değildir; veri setinde farklı
düşey seviyeler bulunmadığı için 10 m hız ve hata dağılımlarını gösterir.

## Temel çıktı dosyaları

- [Tüm model metrikleri](./wind_correction_output/paper_rolling_metrics.csv)
- [Fold metrikleri](./wind_correction_output/paper_rolling_fold_metrics.csv)
- [Dış örnek tahminleri](./wind_correction_output/paper_rolling_predictions.csv)
- [Bootstrap ve DM sonuçları](./wind_correction_output/paper_significance.csv)
- [İstasyon-eşitlikli metrikler](./wind_correction_output/paper_station_balanced_metrics.csv)
- [Rolling-origin leakage denetimi](./wind_correction_output/paper_leakage_audit.csv)
- [Ocak--Nisan ASCAT kapsamı](./wind_correction_output/paper_thiessen_coverage.csv)
- [LTB ve ERA5 fallback kapsamı](./wind_correction_output/paper_ltb_observation_coverage.csv)
- [Operasyonel ASCAT uygunluk durumu](./wind_correction_output/paper_operational_availability.csv)
- [İstasyon eğitim tablosu](./wind_correction_output/ltb_thiessen_table.csv)

## Yeniden üretim

### 1. TIGGE verilerini indirme

ASCAT'ta bulunan günler ve 12:00 başlangıç zamanı için aylık TIGGE dosyaları:

```bash
python TIGGE_yearly_data.py
```

Dosyalar şu klasöre yazılır:

```text
ERA5_data/tigge_2021_ascat_monthly/2021_01.grib
ERA5_data/tigge_2021_ascat_monthly/2021_02.grib
ERA5_data/tigge_2021_ascat_monthly/2021_03.grib
ERA5_data/tigge_2021_ascat_monthly/2021_04.grib
```

ECMWF kuyruk hatası durumunda tamamlanan aylar korunur; script yeniden
çalıştırıldığında tamamlanmış ayları tekrar indirmez. Mayıs--Aralık dosyaları
bulunmadığından çalışma tam yıllık değerlendirme değildir.

### 2. Veri tablosunu ve rolling-origin sonuçlarını üretme

```bash
python evaluate_ltb_thiessen.py
python paper_validation_protocol.py
```

### 3. Grafik ve PDF üretme

```bash
python generate_paper_outputs.py
```

Mevcut `.venv` ortamı kullanılıyorsa:

```bash
source .venv/bin/activate
python generate_paper_outputs.py
```

## Tezde kullanılabilecek sonuç cümlesi

> Ocak--Nisan 2021 döneminde 113 dış örnek forecast-origin ve 428
> nihai istasyon-zaman dış örnek kaydı üzerinde, nested temporal validation ile seçilen
> ASCAT destekli XGBoost düzeltmesi ham TIGGE tahminine göre LTB/ASOS
> hedefinde pooled RMSE'yi %20.37, ERA5 reanalysis referansında %28.54
> azaltmıştır. ERA5 iyileşmesi blok bootstrap ve Newey--West düzeltilmiş DM
> z-testiyle anlamlı bulunurken (p=0.0025), LTB/ASOS iyileşmesi anlamlılık
> düzeyine ulaşmamıştır (p=0.1060). Sonuçlar tek yılın dört ayıyla sınırlıdır
> ve ASCAT'ın operasyonel yayın gecikmesi bu çalışmada doğrulanmamıştır.
