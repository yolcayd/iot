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

---

# 5. Faz faz ilerleme (P0 … P5)

Her faz aynı disiplinle yürüdü:
- **Aynı anda tek faz.** Faz sonunda `docs/phase-logs/<faz>.md` yazılır; kabul kriterleri ✔/✘ olarak işaretlenir.
- **Kararlar** `docs/DECISIONS.md`'ye girer.
- Faz bitince **durulur ve onay beklenir.**
- **5 dakikadan uzun işleri** kullanıcı ayrı terminalde başlatır.

| Faz | Plan karşılığı | Tarih | Ana çıktı | Test sayısı (faz sonunda) |
|---|---|---|---|---:|
| P0 | kurulum | 2026-10-05 | depo, ortam, belgeler | 34 |
| P1 | F0 | 2026-10-05 | veri, okuyucular, keşifsel analiz, kaba/gizli gruplama | 57 |
| P2 | F1 | 2026-10-05 | öznitelik, pencere, ayrım, sızıntı testleri, INT8 dostu girdi | 101 |
| P3 | F2 | 2026-10-05 … 06 | modeller, eşik kuralı, denetimsiz model seçimi | 156 |
| P4 | F3 | 2026-10-06 | nicemleme hattı, benchmark | 199 |
| P5 | F4 | 2026-10-07 | 3 tohumlu ızgara | 207 |

## 5.1 P0 — Kurulum (`docs/phase-logs/P0.md`)

**Ne yapıldı:**
- Ortam tespit edildi (`docs/ENVIRONMENT.md`). Plan ve öneri formu .docx'ten markdown'a çevrildi.
- `CLAUDE.md` (değişmez kurallar) yazıldı.
- Depo iskeleti kuruldu: yalnızca docstring'li yer tutucular.
- `.venv` ve sabit sürümlü `requirements.txt`; git ve remote; smoke testleri.

**Neden böyle:** Planın ilk kuralı fazların sırayla ve kabul kriteriyle ilerlemesi. Kurallar baştan yazıldı ki
sonraki her adım onlara göre denetlensin:
- sızıntı yasağı; test pencereleri örtüşmesiz;
- eşik yalnızca train/val'den; INT8 kalibrasyonu yalnızca train'den;
- 3 tohum; tür bazında metrik;
- her koşunun commit'ini gömmesi; ≥ 1000 tekrarlı gecikme.

**Sorunlar ve çözümleri:**
- **Python sürümü:** Plan 3.11 diyordu, makinede 3.13 vardı. Bütün bağımlılıklar 3.13'te kuruldu; karar
  K-001 olarak kaydedildi.
- **`.gitignore`:** `data/` kuralı `src/data/` kaynak paketini de dışlıyordu. `/data/` olarak köke sabitlendi (K-004).
- **Docker motoru kapalıydı:** Açık sorun olarak bırakıldı. Sonunda konteyner ölçümü kapsam dışı kaldı (K-051).
- **Pencere örtüşmesi:** Öneri formu %50 örtüşme diyordu. Testte örtüşme sızıntı ürettiği için örtüşme yalnızca
  eğitimde kullanılıyor (K-002).

## 5.2 P1 — Veri edinme, okuyucular, keşifsel analiz (`P1.md`)

**Ne yapıldı:**
- İki veri seti indirildi; 106 dosyanın SHA-256'sı kaydedildi.
- Okuyucudan önce gerçek formatlar incelendi. Car-Hacking'de DLC < 8 satırlar daha az sütunlu; ROAD
  metadata alan adları readme'dekinden farklı.
- Ortak şemaya dönüştürüldü. Keşifsel analiz ve veri temelli gruplama yapıldı.

**Kritik kod — ROAD çerçeve etiketi** (`src/data/loaders.py: road_labels`, K-006):

```python
start, end = (int(round(x * 1e6)) for x in cap.meta["injection_interval"])
if kind == "fuzzing":
    return ((data == FUZZ_PAYLOAD).all(axis=1) & (rel_us >= start)).astype(np.int8)
mask, value = parse_injection_pattern(cap.meta["injection_data_str"])
target = int(cap.meta["injection_id"], 16)
match = ((data & mask) == value).all(axis=1)
inside = (rel_us >= start) & (rel_us <= end)
return ((can_id == target) & match & inside).astype(np.int8)
```

- **Satır 1:** Enjeksiyon aralığı saniye cinsinden; **mikrosaniye tamsayıya** çevriliyor. Kayan nokta
  karşılaştırmasında uçtaki çerçeveler yanlış tarafa düşmesin diye.
- **Satır 2–3:** Fuzzing'de enjekte çerçeve = 8 baytın hepsi 0xFF **ve** zaman ≥ başlangıç. Üst sınır yok, çünkü
  fuzzing_attack_2/3'te metadata bitişinden < 6 ms sonra birer FF×8 çerçevesi daha var.
- **Satır 4:** `XXXX0CXXXXXXXXXX` gibi bir desen (X = joker yarım bayt) iki 8 baytlık diziye çevriliyor:
  `mask` hangi bitlerin önemli olduğunu, `value` beklenen değeri tutar.
- **Satır 6:** `(data & mask) == value` maskelenmiş bitler beklenen değere eşitse doğru olur; `all(axis=1)` 8
  baytın hepsi için.
- **Satır 8:** Üç koşulun birlikte sağlanması gerekir: doğru kimlik ∧ desen ∧ zaman aralığı.
- **Doğrulama:** Aralık içinde hedef kimliğin hızı tam iki katına çıkıyor ve hedef kimlik çerçevelerinin tam
  yarısı desene uyuyor. `reverse_light_off`'ta desene uyan meşru çerçeveler yalnızca aralık dışında; doğru
  biçimde normal kalıyor.

**Bulgular:**
- **Masquerade kayıtları türetilmiş** (K-007): 13 masquerade kaydı, fabrication kaynağından aralık içi meşru
  hedef-ID çerçeveleri silinerek üretilmiş (`capture_lineage.csv`; ör. max_speedometer_attack_3: 213,551 →
  207,444 çerçeve, 6,107 silinen, 6,108 enjekte). Aynı enjekte çerçeveleri taşıyan iki kayıt farklı kümelere
  düşerse sızıntı olur. Bu yüzden `source_capture` alanı eklendi.
- **Accelerator kayıtları** (4 adet) çerçeve etiketsiz; dışlandı (K-008).
- **Periyodiklik** (K-010):
  - Car-Hacking: 27/27 kimlik IAT CV < 0.2.
  - ROAD: std tabanlı CV ECU duraklamalarıyla şişiyor (18/106). Medyan tabanlı yayılımla 81/106 periyodik;
    baskın kimlikler tam 10 ms veya 20 ms.
- **Gruplama ve sezginin düzeltilmesi** (K-009, K-011): Kural metriklerden **önce** yazıldı. Sonuç
  (`attack_signatures.csv`, P2 güncel değerleri):

| Tür | C1 | C2 JSD | C3 hız oranı | C4 IAT oranı | Grup | Planın tahmini |
|---|---:|---:|---:|---:|---|---|
| CH dos | 0.545 | 0.349 | ∞ | ∞ | coarse | coarse |
| CH fuzzy | 0.379 | 0.226 | — | — | coarse | coarse |
| CH gear / rpm spoofing | 0 | 0.111 / 0.112 | 6.48 / 7.06 | 0.11 / 0.11 | **coarse** | stealthy ✘ |
| ROAD fuzzing | 0.066 | 0.040 | — | — | coarse | coarse |
| ROAD 5 fabrication | 0 | 0.006–0.011 | 2.00–2.03 | 0.36–0.50 | **coarse** | stealthy ✘ |
| ROAD 5 masquerade | 0 | 0.005–0.006 | 1.00–1.03 | 1.00–1.14 | stealthy | stealthy |

  C2 eşiği: CH 0.00059, ROAD 0.0769. Masquerade'in imzası yalnızca yükte: en büyük bayt medyan kayması
  correlated_signal 252, speedometer 254, coolant 118, reverse_light_on 68, reverse_light_off 8 (P1).

**Kullanıcı kararları:**
- **K-012:** "Gizli" mekanizmaya göre tanımlanır.
- **K-013:** AS-1'in birincil analizi eşleştirilmiş fabrication–masquerade tasarımı.
- **K-014:** Gizli grubun tamamı sentetik ROAD masquerade kayıtları. Bu bir sınırlılık olarak raporlanacak.

**Neden önemli:** Planın "spoofing gizlidir" varsayımıyla devam edilseydi AS-1, trafiği belirgin bozan
saldırıları "gizli" sayacak ve test ettiği mekanizmayı (ince yük sapmasının nicemlemede kaybolması)
bulanıklaştıracaktı.

## 5.3 P2 — Öznitelik, pencere, ayrım, sızıntı testleri (`P2.md`)

**Ne yapıldı:**
- 10 boyutlu öznitelik, pencereleme, sabit ayrım ve otokodlayıcı normal alt kümesi.
- 22 sızıntı testi.
- Ortak kuyruk bulgusundan sonra P1 analizi yeniden üretildi.
- "Ek" bölümünde girdi temsili INT8 için yeniden tasarlandı.

**Ortak kuyruk bulgusu (K-016):**
- Dört Car-Hacking saldırı dosyası aynı 891,068 çerçeveyle bitiyor (mutlak zaman damgaları, kimlikler, yükler
  birebir aynı).
- Bu kuyruk dosyalarla birlikte farklı kümelere gitseydi aynı mesajlar hem eğitimde hem testte olacaktı. Bir
  kez, train'de kullanılıyor.
- P1 tabloları yeniden üretildi. Gruplar değişmedi. Örneğin gear_spoofing'in yük kayması 196'dan 77'ye düştü;
  eski değer kuyruktan geliyordu.

**Kritik kod — öznitelikler** (`src/data/features.py`):

```python
def encode_ids(can_id, vocab):
    k = len(vocab)
    pos = np.searchsorted(vocab, can_id)
    known = (pos < k) & (vocab[np.minimum(pos, k - 1)] == can_id)
    return np.where(known, pos + 1, 0) / k
```

- `vocab` train'deki normal çerçevelerin sıralı kimlik listesi. `searchsorted` her kimliğin listede nereye
  gireceğini bulur.
- `known`: o konumdaki değer gerçekten aynı kimlik mi? Sözlük dışı bir kimlik de bir konuma "girer", bu yüzden
  ayrıca kontrol edilir.
- Bilinen kimlik 1..K sırasını, bilinmeyen 0'ı alır; K'ya bölünerek [0, 1]'e taşınır. K + 1 değer (CH 39,
  ROAD 107) INT8'in 256 seviyesine çakışmasız sığar.

```python
def scale_iat(iat, floor, cap):
    lo, hi = np.log10(floor), np.log10(cap)
    v = (np.log10(np.clip(iat, floor, cap)) - lo) / (hi - lo)
    return np.where(np.isnan(iat), 1.0, v)
```

- IAT önce [taban, tavan] aralığına kırpılır, sonra log ölçekte [0, 1]'e taşınır. Her büyüklük mertebesi eşit
  çözünürlük alır.
- `NaN` = kimliğin segmentteki ilk çerçevesi (önceki çerçeve yok); 1.0 alır. Böylece öznitelik segment
  sınırının ötesine bakmaz.

```python
def fit_normalization(segments, floor_quantile, cap_quantile):
    bad = [s.id for s in segments if s.split != "train"]
    if bad:
        raise ValueError(...)
```

- Normalizasyon parametreleri yalnızca train segmentlerinden uydurulabilir; başka bir şey verilirse kod durur.
  Sızıntı kuralının kodda zorlanması budur.

**Kritik kod — pencere etiketi** (`src/data/windows.py`):

```python
def injected_counts(label, starts, size):
    cs = np.r_[0, np.cumsum(label.astype(np.int64))]
    return cs[starts + size] - cs[starts]

def window_labels(label, starts, size):
    return (injected_counts(label, starts, size) > 0).astype(np.int8)
```

- Kümülatif toplam hilesi: `cs[i]`, ilk i çerçevedeki enjekte sayısıdır. `[start, start+size)` aralığındaki
  enjekte sayısı `cs[start+size] − cs[start]` ile O(1)'de bulunur.
- Pencerede ≥ 1 enjekte çerçeve varsa etiket 1.

**Ayrım kararları:**
- K-017: Car-Hacking zamansal blok.
- K-018: ROAD `source_capture`; fuzzing istisnası.
- K-019: Coolant yalnızca testte.
- Bu kararlar Bölüm 3.2'de; sayılar Bölüm 3.3'te.

**Sızıntı testi sorunu ve çözümü (K-023):**
- **Sorun:** Planın testi "eğitim ve test pencereleri ortak zaman damgası içermiyor" diyordu. İlk sürüm
  (zaman damgası, kimlik, yük) anahtarıyla ROAD'da 54 eşleşme verdi.
- **İnceleme:** Bunlar, anonimleştirilmiş ve aynı yuvarlak değerden başlayan ilişkisiz kayıtlar arasındaki
  tesadüfi, tekil eşleşmelerdi. Ayrıca yeniden zaman damgalanmış kopyalar (masquerade) mutlak zamanla hiç
  yakalanamıyordu.
- **Çözüm:** Mesaj kimliği "başlattığı 32 çerçevelik dizi" oldu: kimlikler, yükler ve çerçeveler arası µs
  farkları; zamandan bağımsız. Pozitif kontrol eklendi: test bilinen kopyaları (soy çiftleri, ortak kuyruk)
  yakalıyor mu?

**Ek — INT8 dostu girdi (K-024, K-025; model eğitiminden ve hiçbir sonuç görülmeden önce):**
- **Kimlik:** Eski ölçeklemede (ID / 0x7FF ya da / 0xFFF) bir INT8 seviyesine ≈ 8 (CH) ya da ≈ 16 (ROAD)
  ardışık kimlik düşüyordu. Yoğun sözlük indeksiyle her kimlik kendi seviyesinde.
- **IAT:** Doğrusal ölçekte 1 ms ile 10 ms arası 3 seviyeydi; log ölçekte CH 70, ROAD 42 seviye.
- **Neden:** Ölçülen nicemleme kaybı girdi ölçeklemesinden değil modelin iç nicemlemesinden gelmeli; araştırma
  sorusu ikincisi.

**Otokodlayıcı normal alt kümesi (K-022):**
- Kaynaklar: train, label 0, enjekte çerçevelere ±5 s'den uzak; masquerade kayıtları hariç.
- CH 29,685 pencere (normal dosya 11,865 + kuyruk 17,820); ROAD 295,689.
- CH saldırı bloklarında patlama arası 3 s olduğu için ±5 s payı sağlayan pencere yok.

## 5.4 P3 — Baseline modeller ve fizibilite (`P3.md`)

**Ne yapıldı:**
- Dört model ailesinin kodu, eğitim/değerlendirme hattı, ızgara (`baseline`, `ae_feasibility` aşamaları),
  akıl sağlığı kontrolleri.
- Baseline **iki kez** koşuldu (v1, v2), çünkü v1'den sonra kurallar değişti.

**Duman testinde bulunan hatalar:**
- Çifti olmayan veri setinde `pairs_*.csv` sütunsuz yazılıyordu.
- Yeniden yüklenen XGBoost CPU'da tahmin yapıyordu (1.2e-7 fark).
- Checkpoint `map_location=cuda` ile yüklenip devamı bozacaktı.

Üçü de düzeltildi. Süreci öldürüp devam ettirmek, kesintisiz koşuyla **bit düzeyinde aynı** ağırlıkları verdi.

**v1'in sorunları ve hikâyesi:**
1. **ROAD CNN dejenere modeli seçti:**
   - Ağırlıklı val kaybıyla (pos_weight ≈ 41) erken durdurma 1. epoch'u seçti. Bu model her pencereye saldırı
     diyor (FPR ≈ 1.0).
   - İnceleme (`analysis/road_cnn_gap.py`): 2. epoch'tan sonra val kaybının %100'ü pozitiflerden geliyor; val
     AUC 0.79 → 0.94 ve AP 0.33 → 0.81 artarken kayıp 9.8'den 22–37'ye çıkıyor.
   - Yani kayıp sıralamayı değil olasılık ölçeğini izliyor. **K-034:** Erken durdurma val AP ile yapılır.
2. **Sabit 0.5 eşiği kayıtlar arasında aktarılmadı:**
   - CNN'de FPR ≈ 1.0; RF correlated_signal'da test AUC 0.955, F1 0.091.
   - **K-035:** Bütün modellerde eşik, val normal pencerelerinde FPR %1 olacak biçimde seçilir.
   - **K-038:** Kalibrasyon kümesi "temiz" val normalleri. Masquerade kayıtları ve saldırıya ±5 s'den yakın
     pencereler hariç; masquerade normalleri fabrication'ın kopyası, saldırıya yakın normaller geçiş etkisi
     taşıyabilir.
3. **LSTM-AE ROAD'da başarısız:** Val ROC-AUC 0.394 < 0.6. Önceden kayıtlı protokol (K-033) işletildi.

**Kritik kod — eşik** (`src/calibration.py`):

```python
def calibrate_threshold(scores, labels, split, target_fpr):
    if split != "val":
        raise ValueError(...)
    normal = scores[labels == 0]                 # attack windows are ignored
    thr = float(np.quantile(normal, 1.0 - target_fpr, method="higher"))
    return {"rule": "score > threshold", "threshold": thr,
            "realized_val_fpr": float((normal > thr).mean()), ...}
```

- **Satır 2–3:** Eşik yalnızca val'den. Test ya da train skorları verilirse durur (kural 3).
- **Satır 4:** Saldırı pencereleri kullanılmaz; eşik etiketsiz normal trafiğe göre seçilir.
- **Satır 5:** 0.99 kantili; `method="higher"` gerçek bir skor değerini seçer. Karar `skor > eşik` olduğu için
  gerçekleşen FPR ≤ %1 olur; bağ varsa daha düşük.
- **Satır 6–7:** Gerçekleşen val FPR kaydedilir.
- **Skorlar neden logit:** `score_deep` olasılık değil logit döndürür. Sigmoid float32'de tam 1.0'a doyar ve
  eşikte bağlar oluşturur; sıralama ise aynıdır.

**Kritik kod — CNN'de val AP ile erken durdurma** (`src/train.py: train_deep`):

```python
monitor = -row["val_ap"] if use_ap else val
state["epoch"] += 1
if monitor < state["best_val"]:
    state.update(best_val=monitor, best_epoch=state["epoch"], bad=0, best_state=...)
else:
    state["bad"] += 1
...
tmp = ckpt.with_suffix(".tmp"); torch.save({...}, tmp); os.replace(tmp, ckpt)
```

- **İzlenen değer:** "Küçük daha iyi" mantığı korunur; CNN'de −AP, otokodlayıcıda val kaybı. Otokodlayıcının
  val kümesinde saldırı yok, AP tanımsız.
- **Sabır:** İyileşme olursa en iyi ağırlıklar saklanır, yoksa sabır sayacı artar (patience 5, en fazla 30 epoch).
- **Checkpoint:** Önce geçici dosyaya yazılır, sonra `os.replace` ile atomik olarak değiştirilir. Yazma
  sırasında süreç ölürse önceki checkpoint sağlam kalır. Checkpoint ayrıca karıştırma üretecinin ve
  CPU/CUDA rastgelelik durumlarını taşır; devam ettirilen koşu bit düzeyinde aynıdır.

**Denetimsiz model seçimi (K-033 → K-039):**
- **Protokol:** Sonuçlar görülmeden yazıldı. Aday sırası (1) LSTM-AE, (2) ayar denemesi LSTM-AE (gizli 128,
  lr 3e-4, ≤ 60 epoch), (3) conv-AE.
- **Seçim:** İki veri setinde de val ROC-AUC ≥ 0.6 olan **ilk** aday seçilir; en yüksek AUC aranmaz. Sonraki
  aday yalnızca öncekinin başarısızlığında incelenir.

| Aday | CH val ROC-AUC | ROAD val ROC-AUC |
|---|---:|---:|
| (1) LSTM-AE | 0.807 ✔ | 0.395 ✘ |
| (2) LSTM-AE ayar denemesi | 0.809 ✔ | 0.372 ✘ |
| (3) conv-AE | 0.999 ✔ | 0.722 ✔ |

- **Sonuç:** İki veri setinde de conv-AE seçildi. Test yalnızca seçimden sonra değerlendirildi.
- **Danışman uyumu:** Danışman yönlendirmesindeki "LSTM-AE başarısız olursa conv-AE yedek" kuralına uyuyor.

**v2 baseline sonuçları** (test, tek tohum; `baseline_f1.csv`):
- **Car-Hacking:** Denetimli modellerde bütün türlerde recall 1.000; conv-AE 0.954–0.996.
- **ROAD, toplam recall:** CNN 0.611, RF 0.264, XGB 0.276, conv-AE 0.102.
- **ROAD, ROC-AUC:** 0.888–0.923.
- **Eşik aktarımı (K-040):** conv-AE testte FPR ≈ %0.1'de çalışıyor (CH %0.18, ROAD %0.11), yani hedeften on
  kat sıkı. Bu yüzden açıklayıcı ölçü ROC-TPR@%1 eklendi.
- **Sızıntı incelemesi:**
  - CH'deki ≈ 1.0 skorlar tek öznitelikli bir karar kütüğüyle de elde ediliyor: DoS/Fuzzy için
    `unseen_id_share`, gear/RPM için `iat_min`; F1 ≥ 0.9997.
  - Yani yüksek skorlar sızıntıdan değil kaba imzalardan geliyor.
  - Sızıntı testleri yeniden çalıştırıldı: 22/22 ✔.

**K-041 taban kuralı:** Sonuçlar görülmeden yazıldı. Q0 recall'u < 0.10 olan bir çift birimi AS-1'in birincil
testinden çıkar.

## 5.5 P4 — Nicemleme hattı ve benchmark (`P4.md`)

**Ne yapıldı:**
- CNN ve conv-AE için Q0/Q1/Q2.
- K-036'ya göre iki eşik kipiyle değerlendirme.
- Benchmark kodu.
- Pilot: dört tohum-0 koşusu.

**Q1 sorunu (K-043):**
- **Ölçüm:** ONNX Runtime profilinden okunan gerçek çekirdek türleri, x86 CPU sağlayıcısında bu modellerin
  hiçbir ağır katmanı için fp16 çekirdeği olmadığını gösterdi (CNN 0/5, conv-AE 0/6).
- **Sonuç:** ORT her ağır katmanın önüne fp32'ye dönüştürme ekliyor. CPU'daki "Q1" fp16'ya yuvarlanmış
  ağırlıklarla fp32 hesap yapıyor.
- **Önceden konmuş kural tetiklendi:** Ağır katmanların yarısından fazlası float32'de kalıyorsa iş durur ve
  kullanıcıya sorulur (kodda `Q1CoverageError`).
- **Kullanıcı kararı:** Doğruluk PyTorch fp16 ile CUDA'da ölçülür; CPU gecikmesi ölçülmez
  (`cpu_unsupported`); boyut fp16 ONNX dosyasından alınır.

**Kritik kod — fp16 benzetimi** (`src/quantize.py`):

```python
def fp16_scorer(module, device):
    m16 = copy.deepcopy(module).to(device).half().eval()
    def run(x):
        with torch.no_grad():
            return m16(torch.from_numpy(x).to(device).half()).float().cpu().numpy()
    return run
```

- Modelin bir kopyası `.half()` ile ağırlıklarıyla birlikte fp16'ya çevrilir. Girdi de fp16'ya çevrilir; ara
  aktivasyonlar fp16'da kalır.
- Çıktı karşılaştırma için float32'ye geri çevrilir.
- **Sınırlılık:** cuDNN/cuBLAS ara toplamaları genellikle fp32 tutar. Bu bir ECU'nun fp16 donanımıyla birebir
  aynı değildir (K-043).

**Kritik kod — INT8 kalibrasyon kümesi** (`src/quantize.py: calibration_selection`, kural 4):

```python
pool = D.ae_train(win, dataset) if model in T.AUTOENCODERS else D.supervised_train(win, dataset, segs)
pool = check_calibration_selection(pool, n_min)
idx = np.sort(np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False))
return check_calibration_selection(pool.iloc[idx].reset_index(drop=True), n_min)
```

- **Satır 1:** Havuz, modelin **kendi** eğitim seçimidir. Otokodlayıcı yalnızca normal pencerelerle
  kalibre edilir; aktivasyon aralıkları saldırı görmez.
- **Satır 2 ve 4:** `check_calibration_selection` havuzda ve örnekte train dışı pencere ya da 500'den az
  pencere varsa durur.
- **Satır 3:** Koşu tohumuyla 1000 pencere seçilir. Seçimin özeti (`digest`) `quant_meta.json`'a yazılır.

**Kritik kod — iki eşik kipi** (`src/quantize.py: evaluate_run`):

```python
decisions[q] = {**C.calibrate_threshold(scores[q]["val"][cal_pos], cal["label"].to_numpy(), "val", target_fpr), ...}
...
for mode in THRESHOLD_MODES:
    dec = decisions[q] if mode == "recalibrated" else decisions["q0"]   # (b): Q0 threshold unchanged
```

- Her sürüm (Q0, Q1, Q2) temiz val normallerinde kendi eşiğini alır; bu `recalibrated` kipidir.
- `q0_threshold` kipinde Q0'ın eşiği Q1/Q2'ye aynen uygulanır.
- Referans Q0, P3'ün PyTorch/GPU sonucu değil Q0-ONNX'in CPU sonucudur. Bütün Δ'lar aynı hatta hesaplanır
  (K-045). P3 ile tutarlılık ayrıca kaydedilir: eşik göreli farkı ≤ 1.8e-4; test kararlarında 0–2 değişen
  pencere.

**Q2'nin yürütülmesi (K-044):**
- **CNN:** 5/5 ağır katman INT8 çekirdekle çalışıyor.
- **conv-AE:** 4/6. İki ConvTranspose, ORT CPU'da INT8 çekirdeği olmadığı için DQ → float → Q benzetimiyle
  çalışıyor. Sayısal olarak int8 ama aritmetik ve gecikme float.
- **Float kalan düğümler:** CNN'de Transpose, ReduceMean, Squeeze; conv-AE'de Transpose ×2 ve skor kuyruğu
  (Sigmoid, Sub, Pow, ReduceMean).
- **Girdi nicemleyicisi doğrulandı:** Ölçek 1/255, sıfır noktası 0, 256 seviye; kimlik ve bayt sütunları kayıpsız.

**Boyut:**
- Q1/Q0 = 0.52.
- Q2/Q0 = 0.376–0.379. Plan ≈ ¼ bekliyordu. Model küçük (≈ 26–28 bin parametre) olduğu için int32 bias'lar,
  ölçekler ve ≈ 11.5 KB'lik QDQ grafik ek yükü oranı yükseltiyor; tensör verisi oranı 0.28.

**Benchmark hikâyesi (K-046 → K-048 → K-049 → K-050 → K-051):**
1. **K-046 protokolü:** 100 ısınma, 1000 ölçüm, batch 1, alt süreç başına bir hedef; iki oturumda tutarlılık
   toleransı medyan ≤ %5, p95 ≤ %10.
2. **Session 1'de RF ROAD ölçümü bozuldu (K-048):**
   - sklearn uyarısı ölçülen döngünün içinde 164,442 kez basıldı; medyan 23 ms'den 578 ms'ye fırladı.
   - **Kök neden:** RF'nin çok iş parçacıklı tahmininde `warnings.catch_warnings()` iş parçacığı güvenli
     değil. Süreç çapındaki uyarı filtre listesini boşaltan bir yarış oluşuyor. Ayrı süreçte yeniden üretildi.
   - **Çözüm:** RF tek iş parçacığıyla tahmin yapar. Bütün hedeflere "döngüde uyarı varsa ölçüm geçersiz"
     koruması eklendi.
3. **Eski protokolle iki oturum: 12 hedeften 8'i tutarsız.** Oturum içi örnekleme hatası, oturumlar arası
   farkın çok altındaydı; fark sistematik.
4. **K-049 protokolü** (hiçbir ölçüm yapılmadan yazıldı):
   - 10.000 ölçüm;
   - ısınma ≥ 100 çıkarım ve ≥ 1 s;
   - 4 performans çekirdeğine sabitleme (mantıksal işlemciler 2, 4, 6, 8);
   - "En iyi performans" güç modu zorunlu;
   - **önceden kayıtlı kural:** iki oturum yine tutarsızsa yeniden ölçüm yok, iki oturum birlikte raporlanır,
     karşılaştırmalar oturum içi oranlara dayanır.
5. **Sonuç: 12 hedeften 3'ü tutarlı.** Medyan farkı en çok %15.5, p95 farkı en çok %54. Kural uygulandı; K-050
   yorum kısıtını ekledi:
   - yalnızca iki oturumda aynı yönlü medyan oranları için yönlü ifade kullanılır;
   - p95 oranlarından sonuç çıkarılmaz;
   - mutlak değerler yalnızca büyüklük mertebesi olarak verilir.
6. **K-051:** Ana makinede bile tekrarlanabilirlik sağlanamadığı için konteyner ölçümü kapsam dışı.
7. **P4 kabul kriteri 12** ("benchmark iki kez tutarlı") **✘** olarak kaldı. Faz kullanıcı kararıyla kapatıldı.

## 5.6 P5 — Tam ızgara (`P5.md`)

**Ne yapıldı:**
- `src/run_grid.py --stage full`: 3 model × 2 veri seti × 3 tohum.
- 6 tohum-0 koşusu aynı run_id ile yeniden kullanıldı; 12 yeni koşu yapıldı.
- Kullanıcı ızgarayı başlattı: 15 dk 16 s, 0 başarısız.
- Tablolar: `grid_quant.csv`, `grid_classical.csv`, `grid_k041_units.csv`.

**Kritik kod — devam ettirilebilirlik** (`src/run_grid.py`):

```python
def pending_steps(r, raw=T.RAW_DIR):
    d = raw / r["run_id"]
    out = []
    if not (d / "DONE").exists():
        out.append("train")
    if any(not (d / f"metrics_{s}.csv").exists() for s in r["evaluate_splits"]):
        out.append("evaluate")
    if r.get("quantize"):
        meta = _read_json(d / "quant" / "quant_meta.json")
        if meta is None:
            out.append("quantize")
        if meta is None or "evaluated_utc" not in meta or not (d / "quant" / "metrics.csv").exists():
            out.append("quant_eval")
    return out
```

- Her adımın tamamlanmış sayılması bir dosyanın varlığına bağlıdır: `DONE` eğitim bitince en son yazılır;
  `metrics_<split>.csv`; `quant_meta.json`.
- `evaluated_utc`, nicemleme değerlendirmesinden **sonra** meta dosyasına yazılır. Değerlendirme yarıda kalırsa
  `metrics.csv` var ama `evaluated_utc` yoktur ve adım yeniden yapılır.
- **Tohum-0 yeniden kullanımı:** Tohum-0 koşularının bütün dosyaları zaten var. Bu yüzden "done" görünürler ve
  hiç alt süreç açılmaz.

**Diğer özellikler:**
- İş başına ayrı alt süreç; `--workers`; çöken iş loglanır, ızgara devam eder.
- Tek örnek kilidi; PROGRESS satırları ve bitiş tahmini.
- `--dry-run`, `--status`, `--only`, `--table`.

**Doğrulamalar:**
- Yeni 12 koşunun hepsi temiz ağaçta, commit `354e4b0`.
- Q0 ONNX çıktıları PyTorch'la `allclose` toleransı içinde.
- INT8 kalibrasyonu train'den.
- Tohum-0 satırları pilot tablosuyla birebir aynı (420/420).

**Bulgular:**
- **K-055:** XGBoost'un üç tohumu bit düzeyinde aynı modeli veriyor (`subsample = colsample = 1.0`).
  Kullanıcı kararı (K-056): XGBoost'a dokunulmaz; nicemlenmeyen referans model olduğu için deterministik
  olması raporlanır.
- **K-054:** conv-AE her tohumda 30. epoch sınırına dayandı. `max_epochs` değiştirilmedi; kullanıcı onayladı.
