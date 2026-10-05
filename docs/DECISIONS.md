# Metodolojik Kararlar

Bu dosya projede alınan metodolojik ve teknik kararların kalıcı kaydıdır. Her fazda alınan yeni kararlar
sona yeni bir girdi olarak eklenir. Mevcut girdiler silinmez; bir karar değişirse yeni girdi eskisine
atıf yaparak onu geçersiz kılar ve eski girdinin durumu "Geçersiz (K-xxx ile)" olarak güncellenir.

Girdi biçimi: **Numara — Başlık**, faz, tarih, durum, bağlam, karar, gerekçe, etkisi.

---

## K-001 — Python sürümü: 3.13

- **Faz / tarih:** P0 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** Proje planı (F0 istemi) ve öneri formu (Bölüm 15) Python 3.11 belirtiyor. Geliştirme makinesinde kurulu tek yorumlayıcı Python 3.13.15.
- **Karar:** Proje Python 3.13 ile yürütülür.
- **Gerekçe:** Tüm bağımlılıklar (torch 2.14.1+cu130, TensorFlow 2.21, ONNX 1.23 / ONNX Runtime 1.30, scikit-learn 1.9, XGBoost 3.4, pandas 3.0, pyarrow 25) 3.13 için resmî wheel sağlıyor ve sorunsuz kuruldu; `pip check` temiz, smoke testleri geçti. Ayrı bir 3.11 kurulumu ek bakım yükü getirir, araştırma sorusuna katkısı yoktur.
- **Etkisi:** Sürümler `requirements.txt`'te sabit. Her koşu kütüphane sürümlerini sonuç meta verisine gömer (CLAUDE.md kural 7), dolayısıyla tekrarlanabilirlik korunur.

## K-002 — Pencereleme: örtüşme yalnızca eğitimde, testte stride = pencere boyu

- **Faz / tarih:** P0 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** Öneri formu (Bölüm 9) N = 100 mesajlık, %50 örtüşmeli kayan pencere belirtiyor. Proje planı (Bölüm 2.2) ve CLAUDE.md (kural 2) test kümesinde örtüşmesiz pencereleme şart koşuyor.
- **Karar:** Pencere boyu N = 100. Örtüşme (stride = 50) yalnızca eğitim kümesinde kullanılır; doğrulama ve test kümelerinde stride = N = 100 (örtüşmesiz). Eğitim/doğrulama/test ayrımı dosya düzeyinde yapılır.
- **Gerekçe:** Örtüşmeli test pencerelerinde komşu pencereler büyük ölçüde aynı mesajları içerir; aynı saldırı olayı birden çok kez sayılır, örnekler bağımsız olmaz ve başarım yapay olarak yükselir. Örtüşmesiz test pencereleri her mesajın tek bir kez değerlendirilmesini sağlar.
- **Etkisi:** Öneri formundan bilinçli bir sapmadır; danışmana **ilk haftalık raporda** bildirilecek. Makalenin Yöntem bölümünde açıkça gerekçelendirilecek. F1'de `tests/test_no_leakage.py` test stride'ının pencere boyuna eşit olduğunu doğrular.

## K-003 — Budama (Q3): temel ızgaradan sonra, zaman kalırsa

- **Faz / tarih:** P0 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** Öneri formu (Bölüm 9, 11) budanmış sürümü kesin bir çıktı olarak listeliyor; proje planı (Bölüm 2.4) Q3'ü opsiyonel tutuyor. Öneri formunun risk tablosu (Bölüm 16) da zaman darlığında budamayı ikincil önceliğe alıyor.
- **Karar:** Temel ızgara (Q0 float32, Q1 float16, Q2 INT8 PTQ) tamamlanana kadar budama uygulanmaz. Temel ızgara ve analiz tamamlandıktan sonra zaman kalırsa Q3 (yapısal budama + float16) eklenir.
- **Gerekçe:** Araştırma soruları (AS-1…AS-3) Q0–Q2 ile yanıtlanabilir; budama ek bir boyuttur ve deney maliyetini artırır.
- **Etkisi:** `src/quantize.py` Q3 için genişletilebilir tasarlanır ama F3'te Q3 zorunlu değildir. Q3 uygulanmazsa bu durum raporda ve makalenin Sınırlılıklar bölümünde belirtilir.

## K-004 — `.gitignore`: veri kuralı köke sabitlendi (`/data/`)

- **Faz / tarih:** P0 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** `.gitignore`'daki `data/` kuralı her düzeydeki `data` adlı klasörü eşleştirdiği için `src/data/` kaynak paketini de dışlıyordu.
- **Karar:** Kural `/data/` olarak (aynı şekilde `/logs/`) depo köküne sabitlendi.
- **Gerekçe:** Veri setleri lisanslı ve büyüktür, yeniden dağıtılamaz; yalnızca kökteki `data/` dışlanmalı, kaynak kod paketi sürüm kontrolünde kalmalıdır.
- **Etkisi:** `src/data/` commit'lenir; `data/raw/` ve `data/processed/` hiçbir zaman commit'lenmez (`git check-ignore` ile doğrulandı).
