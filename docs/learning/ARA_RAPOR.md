# Ara Rapor — CAN IDS'te Nicemlemenin Saldırı Türüne Göre Etkisi

**Durum tarihi:** 2026-10-07 (P0 … P5 tamamlandı; sıradaki faz P6 = plan F5, analiz ve istatistik)
**Proje:** YZM5015 — Yapay Zekâ ile Nesnelerin İnterneti dönem projesi
**Öğrenci:** Yusuf Olcay Dirik · **Danışman:** Dr. Öğr. Üyesi Muhammed Furkan Küçük (Bursa Teknik Üniversitesi)
**Final sunumu:** 23 Aralık 2026

> Bu rapor çalışma ve sunum hazırlığı içindir. Her sayı depodaki bir dosyadan gelir; kaynak dosya her
> tablonun yanında yazılıdır. Bir sayıyı sunumda kullanmadan önce kaynağına bakın. Karar numaraları
> (K-xxx) `docs/DECISIONS.md`'ye, faz adları (P0 … P5) `docs/phase-logs/`'a atıftır. Planın faz adları
> (F0 … F5) ile projenin faz adları bir kaydırmalıdır: P1 = F0, P2 = F1, P3 = F2, P4 = F3, P5 = F4.

## İçindekiler

1. Problem ve motivasyon
2. Temel kavramlar ve tanımlar
3. Veri ve ortam
4. Projenin genel mimarisi
5. Faz faz ilerleme (P0 … P5)
6. Kararlar ve hikâyeleri
7. Şu ana kadarki sonuçlar ve nasıl okunmaları gerektiği
8. Kalan işler ve sıradaki adımlar
9. Sunumda gelebilecek sorular ve cevapları
10. Terimler sözlüğü

---

# 1. Problem ve motivasyon

## 1.1 Neyi çalışıyoruz?

Bir otomobilde onlarca elektronik kontrol ünitesi (ECU) **CAN veri yolu** (Controller Area Network, ISO 11898)
üzerinden haberleşir. CAN düşük maliyet ve gerçek zamanlılık için tasarlanmıştır; **kimlik doğrulama,
şifreleme ve bütünlük koruması yoktur**. Veri yoluna erişen bir saldırgan (tanı portu, bilgi-eğlence birimi,
telematik ağ geçidi) istediği çerçeveyi enjekte edebilir. UNECE R155 ve ISO/SAE 21434 gibi düzenlemeler bu
yüzden saldırı tespitini (IDS) zorunlu bir savunma katmanı yapar (`docs/PLAN.md` §1.1, `docs/PROPOSAL.md` §3).

Derin öğrenme tabanlı bir IDS araca **ek bir ECU** olarak yerleştirilecekse küçük ve hızlı olmalıdır. Bunun
yaygın yolu **nicemlemedir** (quantization): modelin sayılarını float32'den float16'ya ya da INT8'e indirmek.
Literatür nicemlemenin **kazancını** (bellek, gecikme) raporlar; **bedelini** ise yalnızca toplam doğruluk
üzerinden verir.

**Projenin sorusu:** Nicemlemenin tespit başarımına bedeli bütün saldırı türlerinde aynı mıdır, yoksa bazı
türlerde (özellikle "gizli" saldırılarda) daha mı büyüktür? Toplam skor bu farkı gizliyor olabilir mi?

## 1.2 Temel sezgi (plan §1.2)

| Sınıf | Örnek | Trafiğe etkisi | Beklenen nicemleme duyarlılığı |
|---|---|---|---|
| Kaba (coarse) | DoS, fuzzing | Kimlik dağılımını ve varış zamanlarını belirgin bozar | Düşük |
| Gizli (stealthy) | masquerade (plana göre spoofing ve malfunction da) | Geçerli kimlik kullanır; imza yalnızca yük baytlarında | Yüksek: sayısal çözünürlük kaybı ince sapmayı bastırabilir |

Sezgi: INT8 toplam F1'i az düşürse bile bu düşüşün büyük kısmı gizli saldırılardan gelebilir; veri setinde
kaba saldırı örnekleri çok olduğu için toplam skor bunu maskeler.

**Önemli düzeltme (P1, K-009 … K-012):** Bu sezginin "hangi saldırı gizli" kısmı veriyle sınandı ve
kısmen reddedildi. Spoofing (Car-Hacking) ve fabrication (ROAD) saldırıları hedef kimliğin gönderim
hızını 2–7 kat artırdığı için trafiği bozuyor; yalnızca **masquerade** saldırıları zamanlamayı ve kimlik
dağılımını koruyor. Projede "gizli" artık **mekanizmaya göre** tanımlı: zamanlamayı ve kimlik dağılımını
koruyan, imzası yalnızca yükte olan saldırı (K-012). Ayrıntı Bölüm 5.2'de.

## 1.3 Araştırma soruları (plan §1.3; güncel hâlleri)

| Kod | Soru | Güncel durum |
|---|---|---|
| **AS-1** | Nicemleme (float32 → float16 → INT8) başarım kaybını saldırı türüne göre farklı oranlarda mı üretir? Gizli saldırılardaki kayıp kaba saldırılardakinden anlamlı biçimde büyük mü? | Birincil ölçü F1 değil **kalibre eşikte recall** ve **ΔRecall** (K-037). Birincil analiz **eşleştirilmiş fabrication–masquerade** karşılaştırması (K-013), taban kuralıyla (K-041). Test tasarımı P6 planında konuşulacak (K-056). |
| **AS-2** | Belirlenen gecikme bütçesi altında başarım–gecikme–boyut ödünleşiminde en uygun model/nicemleme kombinasyonu hangisi? | Gecikme ana makinede ölçüldü; tekrarlanabilirlik sınırlı olduğu için yalnızca **oturum içi oranların yönü** yorumlanır (K-049, K-050). Q1 (fp16) CPU'da desteklenmiyor (K-043). Uç cihaz/konteyner ölçümü yok (K-051). |
| **AS-3** | Denetimli ve denetimsiz yaklaşımlar nicemlemeye farklı dayanıklılık gösterir mi? | Denetimsiz model LSTM-AE değil **conv-AE** (K-039). İki veri setinde de değerlendirilebilir. |

## 1.4 Hipotezler (plan §1.4)

- **H1:** Toplam F1 kaybı float32 → INT8'de sınırlı kalır ama türler arasında eşit dağılmaz.
- **H2:** Gizli saldırılardaki kayıp kaba saldırılardakinden anlamlı biçimde büyüktür.
- **H3:** float16 en dengeli noktadır; INT8 en çok kazancı sağlar ama gizli saldırılarda belirgin kayba yol açar.
- **H4:** Yeniden oluşturma tabanlı denetimsiz modeller nicemlemeye denetimli sınıflandırıcılardan daha duyarlıdır.

Öneri formu (§13) şunu da söylüyor: hipotezlerin aksi çıkması da anlamlı bir bulgudur (agresif nicemlemenin
güvenli olduğunu gösterir). **Hipotezler henüz test edilmedi**; istatistiksel testler P6'da. Bölüm 7'deki
gözlemler tanımlayıcıdır.

## 1.5 Literatürdeki yeri (öneri formu §6–7)

| Kaynak | Ne yapıyor | Bu projeyle ilişkisi |
|---|---|---|
| Rajapaksha vd. (2023), VehicleSec — CAN-ODTL | Raspberry Pi üzerinde cihaz üstü transfer öğrenme, float16 nicemleme; ~125 ms gecikme | Nicemlemenin **kazancını** raporluyor; tespit başarımına maliyetini saldırı türüne göre ayrıştırmıyor. Projenin hedeflediği boşluk. |
| Oualil vd. (2026), Technologies — CAN-TEMPO | Denetimsiz, çok periyotlu kodlayıcı–kod çözücü, yeniden oluşturma skoru; tür bazında sonuç | Uç cihaz, gecikme ve nicemleme yok. |
| Verma vd. (2024), PLOS ONE — ROAD | Veri seti rehberi ve fiziksel olarak doğrulanmış saldırılar içeren ROAD | Kullandığımız ikinci veri seti; masquerade saldırılarının kaynağı. |
| Hanselmann vd. (2020), IEEE Access — CANet | Yalnızca normal trafikle eğitilen denetimsiz IDS | Denetimsiz yaklaşımın örneği; gömülü maliyet incelenmemiş. |
| Song, Woo & Kim (2020), Veh. Commun. — DCNN | Car-Hacking veri setini yayımlayan denetimli CNN çalışması | Kullandığımız birinci veri seti; yüksek skorların kaynağı. |

**Boşluk:** İncelenen çalışmaların hiçbirinde nicemlemenin tespit başarımına etkisi saldırı türüne göre
ayrıştırılmamış. Projenin katkısı yeni bir mimari değil, **dürüst ve saldırı türüne duyarlı bir değerlendirme
protokolü**: sızıntısız ayrım, yanlış alarm bütçesine kalibre eşik, eşleştirilmiş tasarım, çoklu tohum ve
nicemlemenin gerçekte nasıl yürütüldüğünün (hangi katman INT8 çekirdekle, hangisi benzetimle) belgelenmesi.

> Not: Plan (§0) projenin en büyük riskini "teknik değil metodolojik" diye tanımlar: CAN IDS literatüründe
> %99+ doğruluk raporlamak kolaydır, çoğu zaman sızıntı yüzünden. Bu rapordaki kararların çoğu bu riske
> karşı alınmıştır.

---

# 2. Temel kavramlar ve tanımlar

Her kavram için önce sezgi, sonra teknik tanım ve projede nerede kullanıldığı verilir.

### CAN çerçevesi (frame)
- **Sezgi:** Veri yoluna atılan kısa bir mesaj: "ben 0x316'yım, devir sayısı şu".
- **Teknik:** Arbitrasyon kimliği (11 bit standart), DLC (yük uzunluğu 0–8), en fazla 8 bayt yük. Kimlik hem
  mesajın anlamını hem önceliğini belirler; gönderen bilgisi yoktur.
- **Projede:** Ortak şema `timestamp, can_id, dlc, data[8], label, attack_type, source_file, source_capture`
  (K-005; `src/data/loaders.py`).

### Varış süresi (IAT, inter-arrival time)
- **Sezgi:** Aynı kimlikli iki mesaj arasında geçen süre. Periyodik ECU'lar sabit aralıkla gönderir
  (ROAD'da baskın kimlikler tam 10 ms veya 20 ms; K-010), saldırı bu düzeni bozabilir.
- **Teknik:** `iat_i = t_i − t_{önceki aynı ID}`, segment içinde hesaplanır; bir kimliğin segmentteki ilk çerçevesi
  tavan değerini alır (K-021, K-025).
- **Projede:** Özniteliğin 10. boyutu, log ölçekli (aşağıda).

### Saldırı türleri
- **DoS:** Yüksek öncelikli kimliğin (Car-Hacking'de 0x000) sel gibi gönderilmesi.
- **Fuzzing / Fuzzy:** Rastgele kimlik ve/veya yük enjeksiyonu (ROAD fuzzing'de yük FF×8).
- **Spoofing (Car-Hacking gear/RPM):** Geçerli bir kimlikle sahte değer gönderme. Veride hedef kimliğin hızı
  ×6.5–7 artıyor (K-011).
- **Fabrication (ROAD hedefli saldırılar):** Gerçek ECU susmadan, aynı kimlikle sahte çerçeve eklemek. Hedef
  kimliğin hızı tam ×2 (K-011).
- **Masquerade (ROAD `_masquerade`):** Gerçek ECU'nun çerçeveleri yerine sahte çerçeve. Hız ve kimlik dağılımı
  korunur, fark yalnızca yükte. ROAD'da bunlar fabrication kayıtlarından meşru çerçevelerin **sonradan
  silinmesiyle** üretilmiştir (K-007, K-014).

### Kaba / gizli (coarse / stealthy) grubu
- **Teknik tanım (K-009):** Aşağıdakilerden biri sağlanırsa **coarse**:
  - C1: görülmemiş kimlik payı > 0.01;
  - C2: kimlik dağılımının Jensen–Shannon ıraksaması, saldırısız trafikten çekilen boş dağılımın 99. persentilinden büyük;
  - C3: hedef kimlik hız oranı [1/1.5, 1.5] dışında;
  - C4: hedef kimlik medyan IAT oranı [1/1.5, 1.5] dışında.
  Hiçbiri sağlanmazsa **stealthy**. Kural metrikler hesaplanmadan önce `configs/eda.yaml`'a yazıldı.
- **Jensen–Shannon ıraksaması:** `JSD(P‖Q) = ½ KL(P‖M) + ½ KL(Q‖M)`, `M = ½(P+Q)`; iki olasılık dağılımı
  arasındaki simetrik ve sınırlı uzaklık.
- **Projede:** `configs/attack_groups.yaml` (K-011): 10 coarse, 5 stealthy tür.

### Pencere (window)
- **Sezgi:** Modelin tek seferde baktığı 100 ardışık mesajlık dilim.
- **Teknik:** `N = 100`; stride train'de 50 (örtüşmeli), val/test'te 100 (örtüşmesiz) (K-002). Pencere etiketi:
  içinde ≥ 1 enjekte çerçeve varsa 1. Baskın `attack_type`: en sık enjekte tür (`src/data/windows.py`).
- **Neden test örtüşmesiz:** Örtüşmeli test pencerelerinde aynı saldırı olayı birden çok kez sayılır, örnekler
  bağımsız olmaz ve başarım yapay yükselir.

### Veri sızıntısı (data leakage)
- **Sezgi:** Modelin test sırasında, eğitimde zaten gördüğü bilgiyle değerlendirilmesi; sonuç gerçekte
  olduğundan iyi görünür.
- **Projede önlemler:** Dosya/kaynak kayıt düzeyinde ayrım (K-017, K-018); masquerade kaydı kaynağıyla aynı
  kümede (K-007); ortak kuyruğun tek kullanımı (K-016); normalizasyon yalnızca train'den (K-021, K-024, K-025);
  eşik yalnızca val'den (K-035); INT8 kalibrasyonu yalnızca train'den (K-042); 32 çerçevelik dizi tabanlı
  sızıntı testi (K-023).

### Denetimli ve denetimsiz öğrenme
- **Denetimli (1D-CNN, RF, XGBoost):** Saldırı ve normal etiketli pencerelerle öğrenir; bilinen saldırılarda güçlü.
- **Denetimsiz (otokodlayıcı):** Yalnızca normal trafikle öğrenir; normalden sapmayı anomali sayar. Görülmemiş
  saldırıyı yakalama potansiyeli vardır.

### 1D-CNN (M1)
- **Sezgi:** Zaman ekseninde kayan küçük filtrelerle yerel desenleri (ör. bir baytın ani değişimi) arar.
- **Teknik (`src/models/cnn1d.py`):** 3 blok Conv1d (kanal 32, 64, 64; çekirdek 5, 5, 3) + BatchNorm + ReLU,
  ilk ikisinden sonra MaxPool(2); zaman üzerinde ortalama; Linear(64→32) + ReLU + Dropout(0.2) + Linear(32→1).
  Çıktı tek **logit**.
- **Kayıp:** `BCEWithLogitsLoss(pos_weight = N_neg/N_pos)` (K-027).

### Otokodlayıcı (autoencoder) ve yeniden oluşturma hatası
- **Sezgi:** Girdiyi küçük bir temsile sıkıştırıp geri açan ağ. Yalnızca normal trafikle eğitildiği için
  normali iyi, anormali kötü geri üretir. Kötü geri üretim = anomali.
- **Skor:** `s(x) = (1 / (T·F)) Σ_t Σ_f (x̂_{t,f} − x_{t,f})²` (T = 100 adım, F = 10 öznitelik;
  `reconstruction_score`, `src/models/lstm_autoencoder.py`).
- **LSTM-AE (M2):** Kodlayıcı LSTM'in son gizli durumu → doğrusal gizli temsil → zamana kopyalanır → çözücü LSTM →
  sigmoid. Fizibilite protokolünü ROAD'da geçemedi (K-039).
- **conv-AE (M2-alt, projede kullanılan):** Conv1d(10→32, stride 2) → Conv1d(32→64, stride 2) →
  Conv1d(64→16) | ConvTranspose1d(16→64) → ConvTranspose1d(64→32) → Conv1d(32→10) → sigmoid
  (`src/models/conv_autoencoder.py`). Pencere 100 → 50 → 25 → 50 → 100.

### Klasik referanslar (M0): Random Forest, XGBoost
- Pencere başına 24 özet öznitelik üzerinde çalışırlar (`src/models/classical.py`):
  - kimlik entropisi, benzersiz kimlik sayısı, görülmemiş kimlik payı;
  - IAT ortalama, std, min, q10;
  - 8 baytın ortalaması ve varyansı;
  - yük değişim oranı.
- Nicemlenmezler (K-042). P5 ızgarasında yalnızca XGBoost var (K-053).
- XGBoost tohumdan bağımsız çıktı (K-055, K-056).

### Karar eşiği ve kalibrasyon (K-035, K-038)
- **Sezgi:** Model bir skor üretir; "saldırı" demek için bir eşik gerekir. Eşik, temiz normal trafikte yanlış
  alarm oranı %1 olacak biçimde seçilir. Böylece bütün modeller **aynı yanlış alarm bütçesinde** karşılaştırılır.
- **Teknik (`src/calibration.py`):** `eşik = quantile(skor_normal_val, 0.99, method="higher")`; karar
  `skor > eşik`. Kalibrasyon kümesi: val'de label 0, masquerade kaydı olmayan ve hiçbir enjekte çerçeveye ±5 s'den
  yakın olmayan pencereler. Saldırı etiketleri kullanılmaz; val dışı skor verilirse fonksiyon hata verir.

### Sınıflandırma metrikleri
- `TP, FP, FN, TN`: doğru pozitif, yanlış pozitif, kaçırılan, doğru negatif.
- **Precision** = TP / (TP + FP); **Recall (TPR)** = TP / (TP + FN); **F1** = 2TP / (2TP + FP + FN);
  **FPR** = FP / (FP + TN).
- **ROC-AUC:** rastgele bir saldırı penceresinin rastgele bir normal pencereden yüksek skor alma olasılığı;
  eşikten bağımsız.
- **AP (average precision):** precision–recall eğrisinin özeti; dengesiz sınıflarda AUC'dan daha bilgilendirici.
- **ROC-TPR@%1 (K-040):** Değerlendirilen kümenin kendi ROC eğrisinde FPR ≤ %1 olan en yüksek TPR. Yalnızca
  açıklayıcıdır, çünkü test etiketlerini kullanır; hiçbir seçimde kullanılmaz.
- **Accuracy raporlanmaz:** ROAD'da saldırı oranı %0.07–2.9; her şeye "normal" diyen model %97+ accuracy alır.

### Saldırı türü bazında değerlendirme kümesi (K-029)
- t türü için küme = t'nin saldırı pencereleri + bu saldırıların bulunduğu kayıtların normal pencereleri.
- Saldırısız kayıtlar (`ambient`) ayrıca FPR için raporlanır.
- **Birincil havuz (pooled):** coolant çifti hariç bütün pencereler. Toplam metrikler ikincildir.

### Recall ve ΔRecall (K-037)
- **AS-1'in birincil ölçüsü:** kalibre eşikte tür bazında recall.
- **Nicemleme kaybı:** `ΔRecall = Recall(Q0) − Recall(Qk)`; pozitif değer **kayıp** demektir (K-045).
- **Neden F1 değil:** Recall yalnızca saldırı pencerelerine bağlıdır, türler arası saldırı oranı farkından
  etkilenmez. Eşik sabit FPR'a kalibre olduğu için modeller arası adildir.

### Nicemleme (quantization)
- **Sezgi:** Sayıları daha az bitle temsil etmek: daha küçük dosya, (destekleyen donanımda) daha hızlı hesap,
  ama yuvarlama hatası.
- **Afin INT8 nicemleme:** `q = clip(round(x / s) + z, q_min, q_max)`, `x̂ = s · (q − z)`. `s` ölçek, `z` sıfır noktası.
  - **Asimetrik (projede aktivasyon, QUInt8):** [0, 1] girdisinde `s = 1/255`, `z = 0` → 256 seviye.
  - **Simetrik (projede ağırlık, QInt8):** `z = 0`; [0, 1] aralığında yalnızca 128 seviye kullanılabilir (P2 Ek).
- **Kanal başına (per-channel):** Her çıkış kanalının kendi ölçeği; ağırlık hatasını azaltır.
- **PTQ (post-training quantization):** Eğitimden sonra, yeniden eğitim olmadan. Statik PTQ'da aktivasyon
  aralıkları bir **kalibrasyon kümesiyle** (MinMax) belirlenir; bu küme yalnızca train'den gelir (kural 4).
- **QDQ biçimi:** Grafikte QuantizeLinear → DequantizeLinear çiftleri. Çalışma zamanı bunları INT8 çekirdeğe
  (ör. QLinearConv) birleştirebilir; birleştiremezse DQ → float işlem → Q **benzetimi** yapar (K-044).
- **Seçmeli (hybrid) nicemleme:** Bazı düğümler float kalır (projede Transpose, ReduceMean, skor kuyruğu).

### Nicemleme seviyeleri (plan §2.4)
| Kod | Seviye | Projede nasıl |
|---|---|---|
| Q0 | float32 | PyTorch modelinin ONNX (opset 17) hâli; ONNX Runtime CPU'da değerlendirilir (K-045) |
| Q1 | float16 | Dosya: fp16 ONNX. **Doğruluk:** PyTorch `.half()` ile CUDA'da gerçek fp16 aritmetiği; CPU'da fp16 çekirdeği olmadığı için CPU gecikmesi ölçülmez (K-043) |
| Q2 | INT8 PTQ | ONNX Runtime `quantize_static`, QDQ, MinMax, train'den 1000 pencerelik kalibrasyon (K-042) |
| Q3 | budama | Kapsam dışı (K-003; zaman kalırsa) |

### İki eşik kipi (K-036, K-045)
- **`recalibrated` (birincil):** Her nicemlenmiş sürümün eşiği kendi val skorlarından yeniden hesaplanır →
  ΔRecall yalnızca **ayırt edicilik** kaybını ölçer.
- **`q0_threshold` (ikincil):** Q0 eşiği aynen uygulanır → **kalibrasyon kayması** (dağıtımda eşik yeniden
  ayarlanmazsa yanlış alarm ne olur?).

### Tohum (seed) ve tekrarlanabilirlik
- Her yapılandırma 3 tohumla (0, 1, 2) eğitilir; ortalama ± std raporlanır (kural 5).
- `run_id = <model>_<veri seti>_s<tohum>_<sha8>`; sha8 hiperparametreler, veri özeti, tohum ve pencere boyundan
  türetilir (K-031).
- Her koşu `run_config.json`'a sürümleri ve git commit'ini yazar (kural 7).

### Gecikme ölçümü ve tekrarlanabilirlik (K-046, K-049)
- Batch = 1, pencere başına çıkarım süresi; ısınma (≥ 100 çıkarım ve ≥ 1 s) atılır; 10.000 ölçüm; medyan, p95, p99.
- İki oturum tutarlı sayılır ⇔ medyan göreli farkı ≤ %5 ve p95 göreli farkı ≤ %10 (`|a − b| / ortalama`).
- **Oturum içi oran:** Aynı oturumda ölçülen Q2 medyanı / Q0 medyanı (K-050).

### Eşleştirilmiş tasarım (K-013) ve taban kuralı (K-041)
- **Eşleştirme:** Aynı kaynak kayıttan gelen fabrication (coarse) ve masquerade (stealthy) kayıtları aynı enjekte
  yük baytlarını taşır. Tek fark zamanlama izi (hız ×2) var mı yok mu. Çiftin ΔRecall farkı doğrudan
  "zamanlama ipucu yokken nicemleme daha mı çok zarar veriyor" sorusuna bağlanır; araç, sürüş ve yük sabittir.
- **Taban kuralı:** Bir (model, çift) biriminde Q0 recall'u (3 tohum ortalaması) iki taraftan birinde < 0.10 ise
  birim birincil testten çıkar, çünkü zaten tespit edilemeyen bir türde kayıp ölçülemez.

### İstatistik (plan §2.7; P6'da uygulanacak)
- **Mann–Whitney U:** İki bağımsız grubun dağılımlarını sıra tabanlı karşılaştıran parametrik olmayan test.
- **Wilcoxon işaretli sıra:** Eşleştirilmiş farklar için parametrik olmayan test (K-013'te öngörülen).
- **Cliff's delta:** `δ = P(X > Y) − P(X < Y)`; [−1, 1] aralığında etki büyüklüğü.
- **Holm–Bonferroni:** Çoklu karşılaştırmada p değerlerini sıralayıp kademeli düzelten aile bazlı hata kontrolü.

---

# 3. Veri ve ortam

## 3.1 Veri setleri (`docs/DATA.md`)

| | Car-Hacking (HCRL, Korea Univ.) | ROAD (ORNL) |
|---|---|---|
| Kaynak | HCRL sayfasındaki Dropbox; yayıncı sağlama özeti yok, SHA-256 kaydedildi | Zenodo 10.5281/zenodo.10462796; MD5 doğrulandı |
| Lisans/koşul | Akademik kullanım, atıf istenir; yeniden dağıtım izni yok | CC-BY-4.0 |
| Format | Başlıksız CSV `Timestamp, ID, DLC, DATA…, Flag` (DLC < 8'de sütun sayısı azalıyor; bayrak `3 + DLC` konumunda); normal dosya farklı metin biçiminde, son satırı kesik | 46 baytlık sabit genişlikte candump satırları; anonimleştirilmiş zaman damgaları; etiket yok, enjeksiyon bilgisi `capture_metadata.json`'da |
| Saldırılar | dos, fuzzy, gear_spoofing, rpm_spoofing | fuzzing; 5 hedefli fabrication (correlated_signal, max_speedometer, max_engine_coolant_temp, reverse_light_on/off) ve her birinin masquerade sürümü |
| Kayıt | 5 dosya (1 normal + 4 saldırı) | 12 ambient, 3 fuzzing, 13 fabrication, 13 masquerade, 4 accelerator (dışlandı, K-008) |

**Boyut (P1 veri profili, `data_profile.csv`):** Car-Hacking 17.56 M çerçeve (2.33 M saldırı), ROAD 28.24 M
çerçeve (62.5 k saldırı). Saldırı oranı Car-Hacking'de %13–16 (ortak kuyruk çıkınca %16.7–21.2; K-016),
ROAD'da %0.07–2.9.

## 3.2 Dönüşümler ve nedenleri

1. **Ortak şema** (K-005): Her kayıt ayrı bir zstd parquet (`data/processed/<set>/<kayıt>.parquet`) +
   `index.csv`. Neden: kayıt başına dosya, kayıt düzeyinde ayrımı doğal kılar.
2. **ROAD etiketlemesi** (K-006): Çerçeve enjektedir ⇔ kimlik = `injection_id` ∧ yük `injection_data_str`
   desenine uyar (X = joker yarım bayt) ∧ zaman `injection_interval` içinde. Fuzzing'de yük FF×8 ∧ zaman ≥
   başlangıç. Veriyle doğrulandı (hedef kimlik hızı aralıkta tam iki katına çıkıyor, çerçevelerin tam yarısı
   desene uyuyor).
3. **Soy ilişkisi** (K-007): 13 masquerade kaydının her biri kaynağına karşı doğrulandı (çerçevelerin %100'ü
   kaynakta, silinenlerin tamamı aralık içi meşru hedef-ID çerçeveleri; `capture_lineage.csv`).
   `source_capture` alanı eklendi.
4. **Car-Hacking ortak kuyruğu** (K-016): Dört saldırı dosyası aynı 891,068 çerçevelik saldırısız kayıtla
   bitiyor. Bir kez, train'de, `car_hacking/shared_tail` görünümü olarak kullanılır.
5. **Öznitelikler** (K-024, K-025): Mesaj başına 10 boyut, hepsi [0, 1]:
   - `x[0] = kod/K`: kimlik, train normal çerçevelerinin sıralı sözlüğündeki sıra numarası (1..K); sözlük dışı
     kimlik 0 (CH K = 38, ROAD K = 106).
   - `x[1:9] = bayt/255`.
   - `x[9]`: log ölçekli IAT, `(log10(clip(iat, taban, tavan)) − log10(taban)) / (log10(tavan) − log10(taban))`.
     Taban ve tavan, train normal IAT'lerinin 0.001 ve 0.999 kantilleri (CH: 0.2382 ms ve 0.999 s;
     ROAD: 1 µs ve 1.007 s).
   - **Neden:** Girdi INT8'e nicemlendiğinde kayıpsız ya da kayba çok yakın olsun; ölçülen kayıp girdi
     ölçeklemesinden değil modelin iç nicemlemesinden gelsin. Doğrusal IAT'de 1 ms ile 10 ms arası 3 INT8
     seviyesiydi, log ölçekte CH'de 70, ROAD'da 42 seviye (`feature_int8_levels.csv`).
6. **Ayrım** (K-017, K-018, K-019):
   - **Car-Hacking (zamansal blok):** Her saldırı türü tek dosyada olduğu için dosya düzeyinde ayrım türü ya
     eğitimden ya testten tamamen düşürürdü. Saldırı dosyası 300 patlamadır: train 1–180, val 183–240,
     test 243–300; her kesimde 2 patlama atılır. Normal dosya %60/20/20 zamanla bölünür; kesimlerde 15 s
     atılır. Boşluklar ≥ 10 s.
   - **ROAD (source_capture düzeyi):** `_1` train, `_2` val, `_3` test. Ambient kayıtlar ~%60/20/20.
     - **Fuzzing istisnası:** `_2` + `_3` testte, val yok. `_3` tek başına 17 pencere bırakıyordu.
     - **Coolant:** Çiftin tamamı testte; "eğitimde görülmemiş tür" olarak ayrı raporlanır.
7. **Pencere dizini** (`data/processed/windows/windows.parquet`, depoya girmez) ve `split_summary.csv`.

## 3.3 Bölünmüş veri (pencere sayıları, `split_summary.csv`)

| Veri seti | Train | Val | Test |
|---|---:|---:|---:|
| Car-Hacking (saldırı + normal) | 185,379 | 26,069 | 27,849 |
| ROAD (saldırı + normal) | 311,496 | 58,768 | 59,621 |
| **Toplam** | 496,875 | 84,837 | 87,470 |

Test saldırı pencereleri, ROAD'da tür başına: correlated_signal 315 / masq 302; max_speedometer 1,522 / 1,461;
reverse_light_off 607 / 583; reverse_light_on 587 / 564; fuzzing 68; coolant 43 / 43. Her tür ≥ 30 pencere
(`min_test_attack_windows`, sızıntı testinde doğrulanır).

## 3.4 Hesaplama ortamı (`docs/ENVIRONMENT.md`)

- **Makine:**
  - Windows 11 Pro;
  - Intel Core i7-12650H: 6 performans + 4 verimlilik çekirdeği, 16 iş parçacığı;
  - 15.7 GB RAM;
  - NVIDIA RTX 4050 Laptop, 6 GB.
- **Yazılım:** Python 3.13.15 (plan 3.11 diyordu; K-001). Başlıca sürümler:
  - torch 2.14.1+cu130, ONNX 1.23.1, ONNX Runtime 1.30.0 (CPU);
  - scikit-learn 1.9.1, XGBoost 3.4.1, pandas 3.0.6, numpy 2.5.3.
  - Hepsi `requirements.txt`'te sabit.
- **Kaynak kullanımı:**
  - Eğitim GPU'da yapılır, gecikme ölçümü CPU'da.
  - Makinede aynı anda bir DRL deneyi çalıştığı için CPU iş parçacığı bütçesi 4 (K-031).
- **Uç cihaz:** Raspberry Pi yok. Docker konteyneri tanımlandı ama ölçüm kapsam dışı bırakıldı (K-047, K-051).

---

# 4. Projenin genel mimarisi

## 4.1 Klasör yapısı (gerçek depo)

```text
CLAUDE.md                 proje kuralları (değişmez kurallar, faz disiplini, onay gerektiren işlemler)
README.md                 İngilizce kullanım ve yeniden üretim
requirements.txt          sabit sürümler
configs/
  data.yaml               yollar, dışlanan kayıtlar, ortak kuyruk, öznitelik, pencere, ayrım, analiz rolleri
  eda.yaml                kaba/gizli gruplama kuralı (C1–C4 eşikleri, K-009)
  attack_groups.yaml      15 saldırı türünün grubu (K-011)
  models.yaml             hiperparametreler, karar kuralı (FPR %1), iş parçacığı
  quantization.yaml       Q0/Q1/Q2 ayarları, benchmark protokolü (K-049)
  grid.yaml               aşamalar: baseline, ae_feasibility, full (P5)
src/
  data/
    download.py           indirme ve bütünlük özetleri
    loaders.py            Car-Hacking ve ROAD okuyucuları → ortak şema, soy doğrulaması
    sources.py            görünüm katmanı (ortak kuyruk tek kaynak)
    splits.py             zamansal blok / source_capture ayrımı, pencere dizini
    features.py           10 boyutlu öznitelik, yalnızca train'den uydurulan normalizasyon
    windows.py            pencere başlangıçları, etiket, baskın tür, ±5 s pay
    dataset.py            öznitelik önbelleği, FeatureStore, küme seçicileri (split denetimli)
  models/                 cnn1d, lstm_autoencoder, conv_autoencoder, classical (RF/XGB)
  calibration.py          val normallerinde FPR %1 eşiği
  train.py                eğitim, tohumlama, checkpoint, run_id, run_config.json
  evaluate.py             tür bazında metrikler, çiftler, baseline tablosu
  quantize.py             Q0/Q1/Q2, kapsam/çekirdek raporu, K-036 değerlendirmesi, pilot tablosu
  benchmark.py            gecikme, boyut, bellek; çekirdek sabitleme, güç modu, tekrarlanabilirlik
  run_grid.py             ızgara yürütücüsü (P3 aşamaları + P5 full), K-033 protokolü
  sanity.py               yüksek skorlar için sızıntı şüphesi kontrolleri
analysis/
  eda.py                  keşifsel analiz, imza şekilleri
  road_cnn_gap.py         ROAD CNN genelleme açığı incelemesi
  aggregate.py, stats.py, figures.py   yer tutucu (P6/P7)
tests/                    207 test (sızıntı, öznitelik, pencere, okuyucu, model, eğitim, nicemleme, ızgara …)
results/
  raw/<run_id>/           her koşu: run_config.json, meta.json, metrics_*.csv, quant/… (ağırlıklar git dışı)
  tables/                 bütün özet tablolar
  figures/                imza şekilleri
docs/                     PLAN, PROPOSAL, DATA, ENVIRONMENT, DECISIONS, phase-logs/, learning/
docker/                   konteyner tanımı (kullanılmadı, K-051)
```

## 4.2 Verinin baştan sona akışı

```text
 Ham dosyalar (data/raw)
   │  src/data/download.py   (SHA-256 / MD5, manifest)
   ▼
 Ortak şema parquet (data/processed/<set>/<kayıt>.parquet) + index.csv + capture_lineage.csv
   │  src/data/loaders.py    (ROAD etiket kuralı K-006, soy doğrulaması K-007)
   ▼
 Segmentler (configs/data.yaml)
   │  src/data/sources.py    (ortak kuyruk görünümü K-016)
   │  src/data/splits.py     (CH zamansal blok K-017; ROAD source_capture K-018/K-019)
   ▼
 Normalizasyon (yalnızca train normal çerçeveleri) → results/tables/normalization_params.json
   │  src/data/features.py   (kimlik sözlüğü K-024, log-IAT K-025)
   ▼
 Pencere dizini (start satırları; stride 50 train / 100 val-test; etiket; baskın tür; ±5 s pay)
   │  src/data/windows.py, splits.py  → split_summary.csv
   ▼
 Küme seçicileri (src/data/dataset.py)
   ├─ supervised_train  (train, masquerade normalleri hariç K-028)   → CNN, RF, XGB
   ├─ ae_train          (train normal alt kümesi, ±5 s K-022)        → otokodlayıcılar
   ├─ calibration_windows = ae_val (temiz val normalleri K-038)      → bütün modellerin eşiği
   └─ eval_windows      (val / test, bütün pencereler)               → değerlendirme
   ▼
 Eğitim (src/train.py, GPU)  → model.pt / model.ubj, run_config.json, meta.json (eşik)
   ▼
 Değerlendirme (src/evaluate.py) → metrics_{val,test}.csv, pairs_test.csv
   ▼
 Nicemleme (src/quantize.py)
   ├─ Q0: ONNX float32 (PyTorch'la denklik kontrolü)
   ├─ Q1: fp16 ONNX (dosya) + PyTorch fp16 CUDA (doğruluk)
   └─ Q2: INT8 QDQ (train'den 1000 pencerelik kalibrasyon)
   → quant/quant_meta.json, quant/metrics.csv (iki eşik kipi, Δ = Q0 − Qk)
   ▼
 Benchmark (src/benchmark.py, CPU, yalnızca tohum 0) → benchmark_*.csv
   ▼
 Izgara (src/run_grid.py --stage full) → grid_quant.csv, grid_classical.csv, grid_k041_units.csv
   ▼
 [P6] analysis/aggregate.py, stats.py → istatistiksel testler   [P7] şekiller, makale
```

## 4.3 Koruma mekanizmaları (kodda zorlanan kurallar)

| Kural | Kodda nerede |
|---|---|
| Normalizasyon yalnızca train | `features.fit_normalization` train dışı segmentte hata verir |
| Eşik yalnızca val | `calibration.calibrate_threshold` `split != "val"` ise hata verir |
| Seçici doğru kümeden | `dataset._require_split` başka kümeden pencere gelirse hata verir |
| INT8 kalibrasyonu yalnızca train, ≥ 500 pencere | `quantize.check_calibration_selection` |
| Kirli ağaçta koşu yok (kural 7) | `train.git_info` + `train()` / `run_job` reddeder |
| Q1 kapsamı düşükse kullanıcı kararı olmadan durma | `quantize.q1_status` → `Q1CoverageError` |
| Girdi nicemleyicisi asimetrik ve kayıpsız | `quantize.verify_input_quantization` → `InputQuantizationError` |
| Benchmark döngüsünde uyarı varsa ölçüm geçersiz | `benchmark` → `MeasurementInvalid` (K-048) |
| Güç modu "En iyi performans" değilse ölçüm yok | `benchmark` (K-049) |
| Aynı anda iki ızgara yok | `run_grid.acquire_lock` |
