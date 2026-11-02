# Eski grid-tabanlı pilot (tez sonuçlarında kullanılmaz)

Bu klasördeki dosyalar `train_wind_error_correction.py` tarafından üretilmiştir.
O betik istasyon/Thiessen yapısından önceki ilk denemedir: grid noktalarında,
yalnızca ERA5 hedefiyle, son tek günü test olarak ayıran (554 örnek) bir
eğitimdir. Rolling-origin doğrulama, LTB hedefi ve Thiessen ağırlıklandırma
içermez; bu yüzden `metrics.csv` içindeki XGBoost sonucu tezdeki sonuçlarla
karşılaştırılamaz.

Tezdeki tüm sayılar `paper_validation_protocol.py` çıktılarından
(`../paper_*.csv`) gelir.
