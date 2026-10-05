**YZM5015 — YAPAY ZEKÂ İLE NESNELERİN İNTERNETİ · DÖNEM PROJESİ**

**Kaynak Kısıtlı Uç Cihazlarda Araç İçi CAN Ağı Saldırı Tespiti:
Model Nicemlemenin Saldırı Türüne Göre Etkisinin Analizi**

*Proje ve Makale Üretim Planı
Claude Code (VS Code) ile uçtan uca yürütme rehberi*

Yusuf Olcay Dirik · [gizli]

Danışman: Dr. Öğr. Üyesi Muhammed Furkan Küçük · Bursa Teknik Üniversitesi

Final sunumu: 23 Aralık 2026 · Sürüm 1.0

# 0. Bu Plan Nasıl Kullanılır

Bölüm 1 ve 2 projenin tanımıdır; bunları sen okursun. Bölüm 3’ten itibaren her şey Claude Code’a verilecek talimattır. Fazları sırayla ilerlet, her fazın kabul kriterini sağlamadan sonrakine geçme.

| **Bu projenin en büyük riski teknik değil, metodolojiktir**<br>CAN saldırı tespiti literatüründe %99+ doğruluk raporlamak kolaydır ve bu nedenle alan güvenilirlik sorunu yaşamaktadır. Senin katkın "yüksek skor" değil, **dürüst ölçüm**. Veri sızıntısı, pencere örtüşmesinden kaynaklanan optimistik sonuçlar veya test verisi üzerinde eşik seçimi gibi hatalar sonucu değersiz kılar. Plan boyunca bu noktalar özellikle işaretlenmiştir. |
|---|

# 1. Projenin Tanımı

## 1.1 Problem

CAN protokolü (ISO 11898) düşük maliyet ve gerçek zamanlılık için tasarlandığından kimlik doğrulama, şifreleme ve bütünlük koruması içermez. Veri yoluna erişen bir saldırgan isteğe bağlı çerçeve enjekte edebilir. UNECE R155 ve ISO/SAE 21434, saldırı tespitini zorunlu bir savunma katmanı haline getirmiştir.

Derin öğrenme tabanlı tespit modelleri araç içine ek bir ECU olarak yerleştirilmek üzere **nicemlenir** (float32 → float16 → INT8). Literatür bu işlemin kazancını (bellek, gecikme) raporlar; bedelini ise yalnızca **toplam doğruluk** üzerinden verir.

## 1.2 Araştırma boşluğu ve temel sezgi

CAN saldırıları tespit zorluğu bakımından homojen değildir:

| **Sınıf** | **Saldırılar** | **Trafiğe etkisi** | **Beklenen nicemleme duyarlılığı** |
|---|---|---|---|
| **Kaba** | DoS / flooding, fuzzing | Mesaj kimliği dağılımını ve varış süresi düzenini belirgin biçimde bozar | Düşük — kaba istatistiklerle bile ayırt edilir |
| **Gizli** | Spoofing, malfunction, masquerade | Geçerli kimlik kullanır; yalnızca yük baytlarında ince sapma üretir | Yüksek — sayısal çözünürlük kaybı sapmayı bastırabilir |

**Sezgi:** INT8 nicemleme toplam F1’i az düşürse bile, bu düşüşün büyük kısmı gizli saldırılardan geliyor olabilir. Veri setlerinde kaba saldırı örnekleri çoğunlukta olduğundan, toplam skor bu kaybı maskeler. İncelenen çalışmalarda bu ayrıştırma yapılmamıştır.

## 1.3 Araştırma soruları

| **Kod** | **Soru** |
|---|---|
| **AS-1** | Nicemleme (float32 → float16 → INT8), tespit başarımı kaybını saldırı türüne göre farklı oranlarda mı üretir? Gizli saldırılardaki F1 kaybı, kaba saldırılardakinden istatistiksel olarak anlamlı biçimde büyük müdür? |
| **AS-2** | Belirlenen çıkarım gecikmesi bütçesi altında başarım–gecikme–model boyutu ödünleşiminde en uygun model/nicemleme kombinasyonu hangisidir? |
| **AS-3** | Denetimli ve denetimsiz yaklaşımlar nicemlemeye karşı farklı dayanıklılık sergiler mi? |

## 1.4 Hipotezler

- **H1:** Toplam F1 kaybı float32→INT8 geçişinde sınırlı kalır (birkaç puan), ancak saldırı türleri arasında eşit dağılmaz.

- **H2:** Gizli saldırılardaki F1 kaybı, kaba saldırılardakinden anlamlı biçimde büyüktür. DoS ve fuzzing INT8 altında dahi yüksek başarımla tespit edilmeye devam eder.

- **H3:** float16, başarım kaybı ile kazanç arasında en dengeli noktadır; INT8 en yüksek kazancı sağlarken gizli saldırılarda belirgin kayba yol açar.

- **H4:** Yeniden oluşturma tabanlı denetimsiz modeller nicemlemeye denetimli sınıflandırıcılardan daha duyarlıdır, çünkü anomali skoru küçük hata farklarına dayanır.

# 2. Deney Tasarımı

## 2.1 Veri setleri

| **Veri seti** | **Kaynak** | **Saldırı türleri** | **Rolü** |
|---|---|---|---|
| **Car-Hacking** | HCRL, Korea University<br>ocslab.hksecurity.net | DoS, Fuzzy, Gear spoofing, RPM spoofing | Literatürle kıyaslanabilirlik; kaba saldırı sınıfının temsili |
| **ROAD** | Oak Ridge National Laboratory<br>Verma vd. (2024), PLOS ONE | Fuzzing, hedefli kimlik enjeksiyonu, malfunction, **masquerade** | Gizli saldırı sınıfının temsili. Fiziksel olarak doğrulanmış maskeleme saldırıları içeren tek kamuya açık set. |

İki veri setinin birlikte kullanılması ayrıca **veri setleri arası genelleme** deneyine imkân verir: farklı araç, farklı toplama prosedürü.

## 2.2 Temsil ve ön işleme

Mesaj başına 10 boyutlu öznitelik vektörü:

- Normalize arbitrasyon kimliği (11-bit ID / 0x7FF)

- 8 yük baytı, her biri /255 ile ölçeklenmiş (DLC < 8 ise sıfır doldurma)

- Mesajlar arası varış süresi (IAT), üst sınırla kırpılıp normalize edilmiş

Pencereleme: N = 100 ardışık mesaj. Bir pencere, içinde en az bir enjekte edilmiş mesaj varsa saldırı olarak etiketlenir.

| **Kritik metodolojik karar: pencere örtüşmesi**<br>Literatürde %50 örtüşmeli kayan pencere yaygındır. Ancak eğitim ve test kümeleri aynı kayıt dosyasından örtüşmeli pencerelerle üretilirse, komşu pencereler neredeyse aynı mesajları içerdiği için **veri sızıntısı** oluşur ve sonuçlar yapay olarak yükselir.<br>Bu projede: eğitim/doğrulama/test ayrımı **dosya düzeyinde** yapılır. Örtüşme yalnızca eğitim kümesi içinde kullanılır; test kümesinde örtüşmesiz (stride = N) pencereleme uygulanır. Bu karar makalede açıkça belirtilmelidir ve kendi başına bir metodolojik katkıdır. |
|---|

## 2.3 Modeller

| **Kod** | **Model** | **Tür** | **Not** |
|---|---|---|---|
| **M0** | Random Forest / XGBoost | Denetimli, klasik | Zamanlama ve kimlik öznitelikleri üzerinde. Referans alt sınır. |
| **M1** | 1D-CNN | Denetimli, derin | Pencere üzerinde evrişim. Birincil denetimli model. |
| **M2** | LSTM Otokodlayıcı | Denetimsiz | Yalnızca normal trafikle eğitilir. Eşik, eğitim verisinin skor dağılımından belirlenir. |
| **M2-alt** | 1D Evrişimli Otokodlayıcı | Denetimsiz | M2 kararsız çıkarsa yedek. Eğitimi daha kararlıdır. |

## 2.4 Nicemleme seviyeleri

| **Kod** | **Seviye** | **Yöntem** |
|---|---|---|
| **Q0** | float32 | Referans. Hiçbir dönüşüm uygulanmaz. |
| **Q1** | float16 | Eğitim sonrası yarı hassasiyet dönüşümü |
| **Q2** | INT8 (PTQ) | Eğitim sonrası tam tamsayı nicemleme; kalibrasyon kümesi **yalnızca eğitim verisinden** seçilir |
| **Q3** | Budama (opsiyonel) | Yapısal budama + float16. Zaman kalırsa. |

Araç: ONNX Runtime ve/veya TensorFlow Lite. Hangi katmanların nicemlenemediği raporlanır; seçmeli (hybrid) nicemleme uygulandıysa açıkça belirtilir.

## 2.5 Deney ızgarası

| **Faktör** | **Seviyeler** | **Sayı** |
|---|---|---|
| Model | M0, M1, M2 | 3 |
| Nicemleme | Q0, Q1, Q2 | 3 |
| Veri seti | Car-Hacking, ROAD | 2 |
| Tohum | 0, 1, 2 (eğitim tekrarı) | 3 |
| **Eğitim koşusu** | 3 model × 2 veri seti × 3 tohum | **18** |
| **Değerlendirme** | Her eğitilmiş model × 3 nicemleme × saldırı türü bazında | ~54 model değerlendirmesi |

Not: Nicemleme eğitim sonrası uygulandığından yeniden eğitim gerekmez; bu, deney maliyetini düşük tutar. Her eğitilmiş modelden üç nicemlenmiş sürüm türetilir.

## 2.6 Değerlendirme metrikleri

| **Metrik** | **Birim** | **Neden** |
|---|---|---|
| **F1 (saldırı türü bazında)** | — | Birincil metrik. Sınıf dengesizliği nedeniyle accuracy yanıltıcıdır. Projenin sorusu gereği toplam değil, tür bazında hesaplanır. |
| Precision / Recall | — | Kaybın hangi bileşenden geldiğini ayırt etmek için |
| Yanlış alarm oranı (FPR) | — | Araç içi IDS’in pratik kullanılabilirliğini belirler |
| ROC-AUC | — | Denetimsiz modellerde eşikten bağımsız karşılaştırma |
| Çıkarım gecikmesi | ms / pencere | Gerçek zamanlılığın doğrudan ölçütü. Ortalama ve 95. persentil birlikte raporlanır. |
| Model boyutu | MB | Nicemlemenin doğrudan çıktısı |
| Tepe bellek kullanımı | MB | Kaynak kısıtlı cihazda çalıştırılabilirlik |
| **ΔF1 (nicemleme kaybı)** | puan | **Türetilmiş ana değişken:** F1(Q0) − F1(Qk), saldırı türü bazında. Projenin tüm analizi bu değişken üzerinedir. |

## 2.7 İstatistiksel protokol

- Her model 3 tohumla eğitilir; raporlanan değer ortalama ± standart sapmadır.

- Ana test: gizli saldırı grubu ile kaba saldırı grubu arasında ΔF1 farkının karşılaştırılması (Mann-Whitney U).

- Etki büyüklüğü: Cliff’s delta. Çoklu karşılaştırmada Holm-Bonferroni düzeltmesi.

- Gecikme ölçümleri en az 1000 tekrarla yapılır; ısınma (warm-up) koşuları ölçüme dahil edilmez.

# 3. Claude Code Çalışma Çerçevesi

## 3.1 Depo yapısı

```text
can-ids-quantization/
  CLAUDE.md
  README.md
  requirements.txt
  configs/
    data.yaml           — veri seti yolları, dosya düzeyinde ayrım tanımı
    models.yaml         — model hiperparametreleri
    grid.yaml           — deney ızgarası
  src/
    data/
      loaders.py        — Car-Hacking ve ROAD okuyucuları (ortak şemaya dönüştürür)
      features.py       — 10 boyutlu öznitelik vektörü
      windows.py        — pencereleme; eğitim/test için farklı stride
      splits.py         — DOSYA DÜZEYİNDE ayrım; sızıntı kontrolü
    models/
      cnn1d.py
      lstm_autoencoder.py
      conv_autoencoder.py
      classical.py
    train.py
    quantize.py         — Q0/Q1/Q2/Q3 dönüşümleri
    evaluate.py         — saldırı türü bazında metrikler
    benchmark.py        — gecikme, model boyutu, bellek ölçümü
    run_grid.py
  analysis/
    aggregate.py
    stats.py
    figures.py
  tests/
    test_no_leakage.py  — EN ÖNEMLİ TEST
    test_features.py
    test_windows.py
    test_quantize.py
  results/ { raw, tables, figures }
  paper/ { main.tex, refs.bib, sections/ }
```

## 3.2 CLAUDE.md içeriği

| **CLAUDE.md**<br># Proje: CAN IDS’te Nicemlemenin Saldırı Türüne Göre Etkisi<br>## Amaç<br>Araç içi CAN saldırı tespit modellerinde nicemlemenin (float32 → float16 → INT8) başarım kaybını saldırı türü bazında ölçmek. Çıktı: bir konferans bildirisi.<br>## Değişmez kurallar<br>1. **Veri sızıntısı yasaktır.** Eğitim/doğrulama/test ayrımı DOSYA düzeyinde yapılır. Aynı kayıt dosyasından hem eğitim hem test penceresi üretilmez.<br>2. Test kümesinde pencereler ÖRTÜŞMESİZ (stride = pencere boyu). Örtüşme yalnızca eğitimde.<br>3. Anomali eşiği ve tüm hiperparametreler YALNIZCA eğitim/doğrulama verisinden belirlenir. Test verisine bakarak eşik seçmek yasaktır.<br>4. INT8 kalibrasyon kümesi yalnızca eğitim verisinden örneklenir.<br>5. Her sonuç en az 3 tohumla üretilir; ortalama ± standart sapma raporlanır.<br>6. Metrikler saldırı türü bazında hesaplanır. Yalnızca toplam skor raporlamak yasaktır.<br>7. Her koşu kendi yapılandırmasını, kütüphane sürümlerini ve git commit hash’ini sonuç dosyasına gömer.<br>8. Gecikme ölçümü en az 1000 tekrarlıdır ve ısınma koşuları hariç tutulur.<br>## Yasaklar<br>- Sonuç uydurma, örnek değer yazma, çalıştırılmamış deneyin skorunu tahmin etme.<br>- Test setinde model seçimi.<br>- Sınıf dengesizliğini görmezden gelip accuracy raporlama.<br>- Veri setinin tamamını belleğe yükleyip rastgele bölme (sızıntı üretir).<br>## Faz disiplini<br>Aynı anda bir faz. Kabul kriteri sağlanmadan ilerleme yok. Her faz sonunda commit. |
|---|

# 4. Faz Faz Uygulama Planı

## F0 — Kurulum ve veri keşfi

**Süre:** 3–4 gün · **Hafta 1**

| **Claude Code istemi**<br>Bu klasörde Python 3.11 projesi kur. Bağımlılıklar: numpy, pandas, scikit-learn, xgboost, torch, onnx, onnxruntime, tensorflow (lite dönüşümü için), scipy, pyyaml, matplotlib, seaborn, pytest, psutil.<br>Bölüm 3.1’deki klasör yapısını kur ve CLAUDE.md’yi oluştur (içeriğini vereceğim).<br>src/data/loaders.py yaz: Car-Hacking ve ROAD veri setlerini okuyup ortak bir şemaya dönüştüren iki okuyucu. Ortak şema: timestamp (float, saniye), can_id (int), dlc (int), data (8 baytlık dizi), label (0/1), attack_type (string).<br>Keşifsel analiz raporu üret (results/tables/data_profile.csv ve kısa bir markdown özeti):<br>- Her dosyada toplam mesaj sayısı, normal/saldırı oranı<br>- Benzersiz CAN kimliği sayısı ve dağılımı<br>- Kimlik başına varış süresi istatistikleri (ortalama, std) — CAN trafiğinin periyodikliğini doğrula<br>- Saldırı türü başına örnek sayısı<br>- Saldırı sırasında kimlik dağılımının ve varış süresinin nasıl değiştiğini gösteren 3-4 grafik |
|---|

### Kabul kriteri

- Her iki veri seti de ortak şemaya hatasız yükleniyor.

- data_profile.csv üretildi ve saldırı türü başına örnek sayıları biliniyor.

- Grafiklerde kaba saldırıların trafiği belirgin bozduğu, gizli saldırıların bozmadığı gözle görülüyor. **Bu, projenin temel sezgisinin görsel doğrulamasıdır ve makalede Şekil 2 olarak kullanılacaktır.**

## F1 — Öznitelik, pencereleme ve sızıntı testi

**Süre:** 4–5 gün · **Hafta 2** · Projenin geçerliliği bu fazda belirlenir.

| **Claude Code istemi**<br>src/data/features.py yaz: ortak şemadaki mesajları 10 boyutlu vektöre dönüştürür (normalize ID + 8 yük baytı + normalize IAT). Normalizasyon parametreleri yalnızca eğitim verisinden hesaplanır ve kaydedilir.<br>src/data/windows.py yaz: pencereleme. Parametreler window_size=100, stride (eğitimde 50, testte 100). Pencere etiketi: içinde en az bir saldırı mesajı varsa 1.<br>src/data/splits.py yaz: DOSYA DÜZEYİNDE eğitim/doğrulama/test ayrımı. Ayrım configs/data.yaml içinde açıkça listelenir; rastgele bölme yapılmaz.<br>tests/test_no_leakage.py yaz. Şunları test et:<br>1. Eğitim ve test pencereleri hiçbir ortak mesaj (timestamp) içermiyor.<br>2. Normalizasyon istatistikleri test verisinden hesaplanmamış.<br>3. Test penceresi stride’ı pencere boyuna eşit (örtüşme yok).<br>4. Her saldırı türü test kümesinde yeterli sayıda pencereyle temsil ediliyor (minimum eşiği belirt ve kontrol et).<br>tests/test_features.py: bilinen bir mesaj için öznitelik vektörünün elle hesaplanan değere eşit olduğunu doğrula. |
|---|

### Kabul kriteri

- Tüm sızıntı testleri geçiyor.

- Eğitim/doğrulama/test dosya listesi configs/data.yaml içinde açık ve sabit.

- Saldırı türü başına test penceresi sayısı tabloda yazılı.

## F2 — Baseline modeller ve fizibilite

**Süre:** 1 hafta · **Hafta 3–4**

| **Claude Code istemi**<br>src/models/ altına üç modeli yaz: cnn1d.py (1D-CNN sınıflandırıcı), lstm_autoencoder.py (denetimsiz), classical.py (Random Forest ve XGBoost, zamanlama/kimlik öznitelikleri üzerinde).<br>src/train.py yaz: model, veri seti ve tohum alır; eğitir; modeli ve eğitim meta verisini results/raw/<run_id>/ altına kaydeder. Denetimsiz model için anomali eşiğini YALNIZCA eğitim verisinin skor dağılımından hesaplar (örneğin 99. persentil) ve kaydeder.<br>Her modeli her veri setinde 1 tohumla eğit (6 koşu). Saldırı türü bazında F1 raporla.<br>Karar noktası: LSTM otokodlayıcı anlamlı anomali skoru üretiyor mu (ROC-AUC rastgeleden belirgin yüksek mi)? Üretmiyorsa bir kez hiperparametre ayarı dene; yine olmazsa 1D evrişimli otokodlayıcıya geç. |
|---|

### Kabul kriteri

- Altı baseline koşusu tamamlandı, saldırı türü bazında F1 tablosu üretildi.

- Denetimsiz modelin çalışıp çalışmadığına karar verildi ve gerekçe README’ye yazıldı.

- Sonuçlar makul aralıkta: kaba saldırılarda yüksek F1, gizli saldırılarda daha düşük. **%100’e yakın her sonuç sızıntı şüphesiyle incelenir.**

## F3 — Nicemleme hattı

**Süre:** 1 hafta · **Hafta 5**

| **Claude Code istemi**<br>src/quantize.py yaz. Eğitilmiş bir modeli alır ve üç sürüm üretir: Q0 (float32 referans, ONNX’e dönüştürülmüş), Q1 (float16), Q2 (INT8 post-training quantization).<br>INT8 kalibrasyonu için temsili veri kümesi YALNIZCA eğitim verisinden örneklenir (örneğin 500 pencere). Bu kısıt kodda açıkça uygulanmalı ve yorum satırıyla belirtilmeli.<br>Nicemlenemeyen katman varsa (ör. LSTM), bunu raporla ve seçmeli nicemleme uygula; hangi katmanların float kaldığını sonuç meta verisine yaz.<br>tests/test_quantize.py: (1) nicemlenmiş model aynı girdi için çıktı üretiyor, (2) model dosya boyutu beklenen oranda küçülmüş, (3) Q0 ONNX çıktısı orijinal PyTorch çıktısına sayısal toleransla eşit.<br>src/benchmark.py yaz: bir modelin pencere başına çıkarım gecikmesini (ortalama ve 95. persentil, en az 1000 tekrar, ilk 100 ısınma hariç), model dosya boyutunu ve tepe bellek kullanımını ölçer. Ölçüm ortamının künyesini (CPU, çekirdek sayısı, bellek sınırı) kaydeder. |
|---|

### Kabul kriteri

- Üç nicemleme seviyesi de üretilebiliyor ve testler geçiyor.

- Model boyutları beklenen oranda küçülüyor (float16 ≈ yarı, INT8 ≈ dörtte bir).

- Benchmark ölçümü tekrarlanabilir; aynı model iki kez ölçüldüğünde sonuçlar tutarlı.

## F4 — Tam ızgara ve uç cihaz ölçümü

**Süre:** 2 hafta · **Hafta 6–7**

| **Claude Code istemi**<br>src/run_grid.py yaz: configs/grid.yaml’daki ızgarayı yürütür (3 model × 2 veri seti × 3 tohum eğitim; her eğitilmiş modelden 3 nicemleme sürümü; her sürüm için saldırı türü bazında değerlendirme ve benchmark). Devam ettirilebilir ve hata dayanıklı olmalı.<br>src/evaluate.py yaz: bir model sürümünü test kümesinde değerlendirir ve metrikleri SALDIRI TÜRÜ BAZINDA kaydeder. Toplam metrikler de ayrıca hesaplanır ama ikincil olarak.<br>Izgarayı çalıştır.<br>Uç cihaz ölçümü: Raspberry Pi erişimi varsa benchmark’ı orada tekrarla. Yoksa CPU çekirdek sayısı (2) ve bellek (2 GB) sınırlaması uygulanmış bir Docker konteynerinde tekrarla ve bu durumu sonuç meta verisine açıkça yaz. Her iki durumda da Q0’a göre GÖRECELİ değerleri (oran) birincil olarak raporla.<br>Ek deney (zaman varsa): veri setleri arası genelleme. Car-Hacking üzerinde eğitilen modeli ROAD üzerinde test et ve tersi. |
|---|

## F5 — Analiz ve istatistik

**Süre:** 1 hafta · **Hafta 8–9**

| **Claude Code istemi**<br>analysis/aggregate.py: results/raw altındaki her şeyi tek bir tidy CSV’ye topla. Sütunlar: run_id, model, dataset, seed, quantization, attack_type, metric, value. results/tables/all_results.csv.<br>Türetilmiş değişkeni hesapla: her (model, veri seti, tohum, saldırı türü) için ΔF1 = F1(Q0) − F1(Qk). Bu, analizin ana değişkenidir.<br>analysis/stats.py:<br>- AS-1: Saldırı türlerini "kaba" (DoS, fuzzing) ve "gizli" (spoofing, malfunction, masquerade) olarak grupla. İki grubun ΔF1 dağılımlarını Mann-Whitney U ile karşılaştır. Cliff’s delta hesapla. Holm-Bonferroni uygula.<br>- AS-2: Gecikme bütçesi altında Pareto sınırını hesapla (F1 vs gecikme vs model boyutu).<br>- AS-3: Denetimli ve denetimsiz modellerin ΔF1 dağılımlarını karşılaştır.<br>Çıktı: results/tables/stats.csv ve her araştırma sorusu için birer paragraflık okunabilir bulgu özeti. |
|---|

### Kabul kriteri

- all_results.csv eksiksiz; her hücre dolu.

- ΔF1 hesaplanmış ve saldırı türü bazında dağılımı çıkarılmış.

- AS-1, AS-2, AS-3 için bulgu özetleri yazılmış; hipotezlerin doğrulanıp doğrulanmadığı net.

## F6 — Şekiller ve tablolar

**Süre:** 4–5 gün · **Hafta 9–10**

| **No** | **Şekil / Tablo** | **İçerik** |
|---|---|---|
| Şekil 1 | Sistem şeması | CAN veri yolu → öznitelik çıkarımı → pencereleme → model → nicemleme → uç cihaz |
| Şekil 2 | Saldırı imzaları | Normal ve saldırı altında kimlik dağılımı ve varış süresi (F0 çıktısı). Kaba/gizli ayrımının görsel kanıtı. |
| Şekil 3 | **Ana şekil:** ΔF1 dağılımı | x: nicemleme seviyesi, y: ΔF1. Kaba ve gizli saldırılar ayrı renkte, kutu grafiği veya noktalı dağılım. Bu şekil makalenin merkezidir. |
| Şekil 4 | Saldırı türü × nicemleme ısı haritası | Her hücrede F1; renk ölçeği kaybı gösterir |
| Şekil 5 | Pareto grafiği | x: çıkarım gecikmesi (ms), y: gizli saldırılarda F1, nokta boyutu: model boyutu |
| Şekil 6 | Model boyutu ve gecikme kazancı | Nicemleme seviyesine göre bar grafiği, Q0’a göre oran |
| Şekil 7 | Denetimli vs denetimsiz | İki model ailesinin ΔF1 karşılaştırması |
| Tablo I | Veri seti künyesi | Dosya, mesaj sayısı, saldırı türü, pencere sayısı, eğitim/test ayrımı |
| Tablo II | Ana sonuç tablosu | Model × nicemleme × saldırı türü için F1 (ort ± std) |
| Tablo III | Kaynak ölçümleri | Model boyutu, gecikme (ort ve p95), tepe bellek |
| Tablo IV | İstatistiksel test | Kaba vs gizli ΔF1 karşılaştırması, düzeltilmiş p, etki büyüklüğü |

| **Claude Code istemi**<br>analysis/figures.py yaz ve yukarıdaki yedi şekli ve dört tabloyu üret.<br>Şekil kuralları: vektörel PDF + 300 dpi PNG, IEEE sütun genişliğine uygun (tek sütun 3.5", çift sütun 7.16"), 8–9 pt yazı, renk körü dostu palet, renk dışında işaretçi/desen ile de ayrışma, her şekil kendi başına anlaşılır.<br>Tablolar CSV ve LaTeX (booktabs) olarak üretilsin. Her şekil için makaleye girebilecek bir caption öner. |
|---|

# 5. Makale Planı

Makale F5’ten itibaren paralel yazılır. Hedef: IEEE konferans şablonu, 6 sayfa.

## 5.1 Bölüm yapısı

| **Bölüm** | **İçerik** | **Kaynak** |
|---|---|---|
| **Öz** | 150–250 kelime. En son yazılır, Bölüm V’teki gerçek sayıları içerir. | F5 |
| **I. Giriş** | CAN güvenliği, UNECE R155 bağlamı, uç cihaz kısıtı, nicemleme gerekliliği, toplam doğruluk üzerinden raporlamanın sorunu, katkı listesi | Bölüm 1 |
| **II. İlgili Çalışmalar** | (a) CAN IDS yöntemleri, (b) denetimsiz/sıfırıncı gün tespiti, (c) uç cihazda konuşlandırma ve nicemleme. Her birinde boşluğu işaretle. | Bölüm 6 |
| **III. Tehdit Modeli ve Saldırı Taksonomisi** | Kaba ve gizli saldırı ayrımının tanımı ve gerekçesi. Şekil 2 burada. **Bu bölüm makalenin özgün çerçevesidir, kısa geçme.** | F0 |
| **IV. Yöntem** | Öznitelik temsili, pencereleme, dosya düzeyinde ayrım gerekçesi, üç model, üç nicemleme seviyesi. Şekil 1 burada. | F1, F2, F3 |
| **V. Deneysel Kurulum** | Veri setleri (Tablo I), eğitim protokolü, eşik belirleme, ölçüm ortamı künyesi, metrik tanımları ve ΔF1 tanımı | F1–F4 |
| **VI. Sonuçlar** | AS-1 → Şekil 3, 4, Tablo II, IV. AS-2 → Şekil 5, 6, Tablo III. AS-3 → Şekil 7. | F5, F6 |
| **VII. Tartışma** | Pratik çıkarım: uç cihaz optimizasyonu yapan bir mühendis ne yapmalı? UNECE R155 doğrulama süreçleri açısından anlamı. | F5 |
| **VIII. Sınırlılıklar** | İki veri seti, dinamometre verisi, sentetik olmayan ama sınırlı saldırı çeşitliliği, gerçek ECU donanımı yerine benzetilmiş kısıt, PTQ dışında QAT denenmemiş olması | — |
| **IX. Sonuç** | Ana bulgu bir cümle; gelecek iş: QAT, gerçek ECU, daha fazla veri seti | — |

## 5.2 Makale için Claude Code istemleri

### Adım 1 — İskelet

| paper/ altında IEEEtran conference şablonunda LaTeX projesi kur. main.tex, sections/ altında bölüm dosyaları, refs.bib.<br>analysis/figures.py çıktılarını paper/figures/ altına kopyalayan bir betik ekle.<br>Şimdilik sadece iskelet ve her bölümün altında ne yazılacağını belirten yorum satırları. |
|---|

### Adım 2 — Tehdit modeli, yöntem, kurulum

| Bölüm III, IV ve V’i yaz.<br>Bölüm III’te kaba/gizli saldırı ayrımını tanımla ve F0 keşifsel analizinden gelen somut gözlemlerle (kimlik dağılımı ve varış süresi değişimi) destekle. Bu ayrım makalenin kavramsal katkısıdır.<br>Bölüm IV’te dosya düzeyinde ayrım ve test kümesinde örtüşmesiz pencereleme kararını açıkça gerekçelendir; literatürde yaygın uygulamanın neden sızıntı riski taşıdığını anlat.<br>Tüm sayılar configs/ ve results/tables/ dosyalarından okunsun; elle yazılmasın. Kodda olmayan hiçbir şeyi yazma. |
|---|

### Adım 3 — Sonuçlar

| Bölüm VI’yı yaz. Veri kaynağı: results/tables/all_results.csv ve stats.csv.<br>Kurallar:<br>- Her sayı CSV’den gelsin.<br>- Her iddia bir şekle/tabloya referans versin.<br>- Anlamlılık belirtilirken düzeltilmiş p ve etki büyüklüğü birlikte verilsin.<br>- Doğrulanmayan hipotezler açıkça belirtilsin.<br>- Toplam metrikler ile tür bazında metrikler arasındaki fark özellikle vurgulansın; bu makalenin ana mesajıdır. |
|---|

### Adım 4 — Giriş, ilgili çalışmalar, tartışma, öz

| Bölüm I, II, VII, VIII, IX ve Öz’ü yaz.<br>refs.bib’deki kaynakların dışına çıkma, yeni atıf uydurma. Eksik gördüğün kaynakları liste olarak bana bildir.<br>Tartışma bölümünde pratik çıkarımı net ver: bir mühendis bu makaleyi okuyup ne yapacak? |
|---|

### Adım 5 — Son kontrol

| Denetim raporu üret:<br>1. Metindeki her sayı CSV’de var mı?<br>2. Her şekil ve tabloya atıf yapılmış mı?<br>3. refs.bib ile metindeki atıflar örtüşüyor mu?<br>4. Sayfa sınırı?<br>5. Öz ile sonuçlar tutarlı mı?<br>6. Sızıntı önlemleri (dosya düzeyi ayrım, örtüşmesiz test, eğitimden kalibrasyon) makalede açıkça yazılmış mı? **Hakemin ilk soracağı budur.** |
|---|

## 5.3 Hedef konferanslar

ASYU, SIU, UBMK, IDAP, ISMSIT. Ayrıca araç güvenliği odaklı uluslararası seçenekler: escar, VehicleSec (NDSS yan sempozyumu). Son gönderim tarihlerini dönem başında takvime işle.

# 6. Kaynakça

Proje öneri formunda kullanılan beş çekirdek kaynak. Literatür taramasında bunların üzerine en az 10 kaynak daha eklenmelidir.

- Hanselmann, M., Strauss, T., Dormann, K., & Ulmer, H. (2020). CANet: An unsupervised intrusion detection system for high dimensional CAN bus data. IEEE Access, 8, 58194–58205.

- Oualil, S., Ait Yahia, I., El Kamili, M., Fardousse, K., & Berrada, I. (2026). CAN-TEMPO: Unsupervised CAN bus intrusion detection via temporal multi-period oscillation encoding. Technologies, 14(6), 375.

- Rajapaksha, S., Kalutarage, H., Al-Kadri, M. O., Petrovski, A., & Madzudzo, G. (2023). Improving in-vehicle networks intrusion detection using on-device transfer learning. VehicleSec 2023.

- Song, H. M., Woo, J., & Kim, H. K. (2020). In-vehicle network intrusion detection using deep convolutional neural network. Vehicular Communications, 21, 100198.

- Verma, M. E., Bridges, R. A., Iannacone, M. D., Hollifield, S. C., Moriano, P., Hespeler, S. C., Kay, B., & Combs, F. L. (2024). A comprehensive guide to CAN IDS data and introduction of the ROAD dataset. PLOS ONE, 19(1), e0296879.

**Eklenecek alanlar:** nicemleme ve model sıkıştırma temel kaynakları (post-training quantization, knowledge distillation), CAN IDS derlemeleri, UNECE R155 ve ISO/SAE 21434 ile ilgili teknik kaynaklar.

**Arama terimleri:** CAN bus intrusion detection quantization, edge IDS automotive INT8, masquerade attack CAN detection, in-vehicle network anomaly detection embedded, TinyML automotive security, post-training quantization impact accuracy.

## Haftalık kontrol listesi

- ☐ H1: F0 tamam · veri profili çıkarıldı · saldırı imzası grafikleri hazır

- ☐ H2: F1 tamam · **sızıntı testleri geçiyor** · ayrım sabitlendi

- ☐ H3–H4: F2 tamam · baseline F1 tablosu · denetimsiz model kararı

- ☐ H5: F3 tamam · üç nicemleme seviyesi üretilebiliyor · benchmark tekrarlanabilir

- ☐ H6–H7: F4 yürüyor · ızgara ve uç cihaz ölçümü

- ☐ H8–H9: F5 tamam · ΔF1 analizi ve istatistik hazır

- ☐ H9–H10: F6 tamam · yedi şekil, dört tablo

- ☐ H10–H12: makale yazımı

- ☐ 23 Aralık: sunum ve teslim

> CAN IDS Proje Planı ·
