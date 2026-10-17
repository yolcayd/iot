# Danışman geri bildirimi 1 — genişletme sonuçları (P10)

**Tarih:** 2026-10-17 · **Kararlar:** K-071 … K-077 (`docs/DECISIONS.md`) · **Önceki raporlar:** `advisor_ext_status.md` (mevcut durum), `literature_preprocessing.md` (literatür)

Bu rapordaki sonuçlar H1–H4 hükümlerinden ve makaleden ayrıdır (K-071). Genişletme, danışman isteğiyle sonradan eklendi. Bütün kurallar sonuçlardan önce yazıldı ve commit'lendi:
- tasarım: `c23cb02`;
- yalnızca val ile yapılan seçim: `b3cac3f`. Test sonuçları bu commit'ten sonra üretildi.

**Ortak koşullar:**
- ROAD ve Car-Hacking (CH) test kümeleri;
- 3 tohum ortalaması (± std);
- eşik temiz val normallerinde FPR %1 (K-035/K-038);
- TPR@1% = test ROC eğrisinde FPR %1'deki TPR. Yalnızca açıklayıcıdır, hiçbir seçimde kullanılmadı (K-040);
- köşeli parantez = %95 hiyerarşik bootstrap aralığı (K-058 yöntemi).

**Kaynak tablolar** (`results/tables/`): `ext_ablation*.csv`, `ext_diag_*.csv`, `ext_quant.csv`. **Şekil:** `results/figures/ext_ablation_road.pdf`.

## Özet

1. **Öznitelikler ROAD'da 1D-CNN'i belirgin iyileştiriyor, ama kazanç yük değişimi özniteliklerinden değil, ID frekansından (S3) geliyor:**
   - Toplam recall 0.545 → 0.780; kaba türler +0.28 [0.21, 0.33].
   - Gizli (masquerade) türlerde kazanç daha küçük ve belirsiz: recall +0.086 [−0.004, 0.168], AUC +0.038 [0.006, 0.075].
   - Bit ve bayt değişimi (S1, S2) gizli türlerde recall'u artırmadı.
2. **conv-AE'nin düşük recall'u üç etkenin birleşimi:**
   - **Eşik aktarımı:** val'de kalibre edilen eşik testte 11 kat sıkı çalışıyor (FPR 0.0009). Recall'u ≈ 0.19 düşürüyor.
   - **Model/paradigma:** aynı FPR'da CNN'in 0.22 gerisinde.
   - **Skor bileşimi:** yeniden oluşturma hatasının %87'si zayıf ayırt edici yük kanallarından geliyor.

   Yeni öznitelikler conv-AE'ye yalnızca fabrication türlerinde yardım etti; val seçim kuralı S0'ı korudu.
3. **Nicemleme sonucu değişmedi:** CNN S3'te INT8 kaybı küçük. Toplam ΔRecall 0.007 (S0: 0.006). Eski eşikle de FPR kayması yok. conv-AE'de S0 korunduğu için bilinen kalibrasyon kayması aynen geçerli.

## 1. Ablasyon (K-074): ROAD, test

Setler:
- S0 = 10 öznitelik;
- S1 = + bit değişimi (Hamming / 64);
- S2 = + bayt değişimi (8 sütun);
- S3 = + ID frekansı (son 1 s / train referansı);
- S4 = + son 100 çerçevede ID payı ve entropi.

### 1.1 1D-CNN

| Ölçü (ROAD test) | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Toplam recall | 0.545 | 0.515 | 0.643 | **0.780** | 0.802 |
| Toplam ROC-AUC | 0.892 | 0.890 | 0.925 | **0.971** | 0.972 |
| Toplam TPR@1% | 0.463 | 0.361 | 0.576 | **0.717** | 0.725 |
| Gizli türler, recall | 0.552 | 0.479 | 0.540 | **0.638** | 0.636 |
| Gizli türler, F1 | 0.646 | 0.596 | 0.645 | **0.732** | 0.724 |
| Gizli türler, ROC-AUC | 0.909 | 0.908 | 0.916 | **0.947** | 0.963 |
| Kaba türler, recall | 0.596 | 0.586 | 0.676 | **0.874** | 0.902 |
| Saldırısız test FPR | 0.021 | 0.032 | 0.016 | 0.017 | 0.024 |

Tür bazında, seçilen S3 − S0 (bootstrap aralığı; `ext_ablation_contrast.csv`):

| Tür | Recall S0 → S3 | ΔRecall | ΔF1 | ΔAUC |
|---|---|---|---|---|
| reverse_light_off | 0.356 → 0.891 | +0.535 [0.399, 0.677] | +0.426 [0.302, 0.559] | +0.090 [0.033, 0.178] |
| reverse_light_on | 0.088 → 0.488 | +0.400 [0.140, 0.576] | +0.465 [0.212, 0.647] | +0.282 [0.164, 0.424] |
| max_speedometer | 0.642 → 0.995 | +0.354 [0.258, 0.456] | +0.214 [0.146, 0.291] | +0.138 [0.092, 0.190] |
| correlated_signal | 0.901 → 0.996 | +0.095 [0.030, 0.193] | +0.052 [0.015, 0.109] | +0.001 [0.000, 0.003] |
| fuzzing | 0.995 → 1.000 | +0.005 [0.000, 0.015] | 0.000 [−0.031, 0.028] | 0.000 |
| reverse_light_off_masquerade | 0.540 → 0.774 | +0.234 [0.035, 0.452] | +0.184 [0.027, 0.363] | +0.035 [0.002, 0.082] |
| reverse_light_on_masquerade | 0.096 → 0.207 | +0.111 [−0.027, 0.257] | +0.157 [−0.041, 0.355] | +0.112 [−0.008, 0.256] |
| max_speedometer_masquerade | 0.718 → 0.794 | +0.076 [−0.066, 0.216] | +0.049 [−0.042, 0.139] | +0.017 [−0.032, 0.063] |
| correlated_signal_masquerade | 0.854 → 0.776 | **−0.078 [−0.138, −0.003]** | −0.047 [−0.083, −0.001] | −0.013 [−0.027, 0.000] |

Eğitimde görülmemiş coolant çifti ayrı raporlanır (K-019). İki türde de recall S3'te düştü (0.047 → 0.023, 0.085 → 0.031); aralıklar 0'ı içeriyor.

### 1.2 conv-AE

| Ölçü (ROAD test) | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Toplam recall | 0.055 | 0.042 | 0.168 | 0.197 | 0.136 |
| Toplam ROC-AUC | 0.882 | 0.872 | 0.871 | 0.886 | 0.867 |
| Toplam TPR@1% | 0.246 | 0.208 | 0.377 | 0.404 | 0.354 |
| Gizli türler, recall | 0.034 | 0.020 | 0.011 | 0.013 | 0.007 |
| Gizli türler, ROC-AUC | 0.850 | 0.841 | 0.834 | 0.840 | 0.809 |
| Gizli türler, TPR@1% | 0.186 | 0.160 | 0.247 | 0.244 | 0.131 |
| Kaba türler, recall | 0.135 | 0.169 | 0.355 | 0.419 | 0.346 |

- Bayt değişimi (S2), conv-AE'nin fabrication türlerini ayırt etmesini çok artırıyor. Örneğin correlated_signal'da AUC 0.869 → 1.000, recall 0.036 → 0.981.
- Gizli türlerde ise AUC her eklemeyle düşüyor (0.850 → 0.809) ve recall 0'a yakın kalıyor.
- reverse_light_off/on'da (tek bitlik yük değişimi) conv-AE'nin AUC'si S2'den sonra geriliyor.

### 1.3 Car-Hacking

- **1D-CNN:** her sette tavanda. Recall ve AUC 1.000; F1 0.974 → 0.986 (S3).
- **conv-AE:**
  - S1'de gear_spoofing recall'u 0.874 → 0.982;
  - S2'de 0.485'e düşüyor. Val'de de görülen bu düşüş, S2'yi seçimde CH şartından eledi;
  - S3'te DoS TPR@1% 0.996 → 0.872.
- CH'nin ayırt edicilik bakımından bilgilendirici olmadığı yine görülüyor (`advisor_ext_status.md` §2.4).

### 1.4 Seçim (K-075, yalnızca val; test görülmeden commit'lendi)

| Model | Val kazanç G (4 masquerade türü, AUC) S1 / S2 / S3 / S4 | CH şartı | Seçilen |
|---|---|---|---|
| 1D-CNN | −0.009 / −0.005 / **+0.022** / +0.019 | hepsi geçti | **S3** |
| conv-AE | −0.000 / −0.028 / −0.012 / −0.035 | S2 kaldı | **S0** (G\* < 0.01) |

CNN S3'ün testteki gizli tür AUC kazancı (+0.038), val'deki kazançla (+0.022) aynı yönde.

### 1.5 Yorum ve uyarılar

- **Danışmanın hipotezi bu tasarımda desteklenmedi:** yük değişimi ve bit değişim oranı (S1, S2) gizli saldırı tespitini iyileştirmedi. CNN'de S1, gizli recall'u 0.552 → 0.479'a düşürdü.
- **Kazancın kaynağı zamanlama:** CNN'deki gizli tür kazancı, zamanlama alanındaki ID frekansıyla (S3) geldi.
  - ROAD masquerade kayıtları, meşru çerçevelerin sonradan silinmesiyle üretilmiş (K-007, K-014). Hedef ID hızında küçük bir iz bırakıyorlar (hız oranı 1.00–1.14; K-011).
  - Literatürdeki CAN-ODTL de masquerade'i yük görmeden, bu türden bir zaman izi üzerinden yakalıyor (`literature_preprocessing.md`, madde 4).
  - Bu yüzden S3'ün masquerade kazancı, gerçek bir masquerade saldırısına genellenecek bir "gizli saldırı tespiti" olarak yorumlanmamalı. **Hipotez:** kazancın bir kısmı veri setinin bu yapıntısından geliyor; ayrı bir deneyle sınanmadı.
- **Örneklem:** 3 tohum ve tür başına tek test kaydı. Masquerade satırlarının çoğunda aralıklar geniş ve 0'ı içeriyor. Kümülatif sıra grupların tek tek katkısını ayırmıyor (K-074).

## 2. conv-AE'nin düşük recall'u: neden? (K-076)

**Dürüstlük notu:** S0 değerleri P5/P6'da görülmüştü. Ayrıştırma betimleyicidir, nedensellik iddiası yoktur. Bütün değerler ROAD, toplam havuz.

### 2.1 Eşik

| | Val kalibrasyon FPR | Test saldırısız FPR | Kalibre recall | TPR@1% | Eşik açığı (TPR@1% − recall) |
|---|---|---|---|---|---|
| conv-AE S0 | 0.010 | **0.0009** | 0.055 | 0.246 | **+0.191** |
| conv-AE S1–S4 | 0.010 | 0.0007–0.0011 | 0.042–0.197 | 0.208–0.404 | +0.166 … +0.218 |
| 1D-CNN S0 | 0.010 | 0.021 | 0.545 | 0.463 | −0.081 |
| 1D-CNN S1–S4 | 0.010 | 0.016–0.032 | 0.515–0.802 | 0.361–0.725 | −0.063 … −0.154 |

- conv-AE'nin val eşiği, saldırısız test pencerelerinin %99.9 kantiline düşüyor. Test normalleri val normallerinden daha "kolay" yeniden oluşturuluyor: S0'da skor kantilleri testte her düzeyde %12–15 düşük (medyan 0.0128 → 0.0109, %99 0.0188 → 0.0164; S2–S4'te %12–23). Bu yüzden eşik testte hedeften on kat sıkı çalışıyor.
- Bu, setten bağımsız ve tutarlı bir etken: aynı test FPR'ında recall ≈ 0.2 daha yüksek olurdu.
- CNN'de etki ters yönde: test FPR'ı hedefin iki katı, recall bu yüzden TPR@1%'den yüksek.
- **Kalibrasyon kümesi:** temiz val normalleri (K-038) seçimde yalnızca normal veriyi kullanıyor (doğru). Ama ROAD'un val ambient kayıtları, test ambient kayıtlarından farklı sürüş koşullarında. Eşiğin teste aktarımı bu farka bağlı.

### 2.2 Öznitelik

- conv-AE'nin TPR@1%'si S0 → S3: 0.246 → 0.404, AUC 0.882 → 0.886.
- Artış neredeyse tamamen fabrication türlerinden: correlated_signal TPR@1% 0.22 → 1.00, max_speedometer 0.26 → 0.99.
- Gizli türlerde ölçü 0.186'dan en fazla 0.247'ye çıkıyor, S4'te 0.131'e düşüyor.
- **Sonuç:** öznitelik yetersizliği conv-AE'nin fabrication türlerindeki zayıflığını açıklıyor; masquerade'deki zayıflığını açıklamıyor.

### 2.3 Model / paradigma (aynı öznitelikler)

| Set | TPR@1% CNN / conv-AE | ROC-AUC CNN / conv-AE |
|---|---|---|
| S0 | 0.463 / 0.246 | 0.892 / 0.882 |
| S3 | 0.717 / 0.404 | 0.971 / 0.886 |

- AUC'ler S0'da yakın, ama düşük FPR bölgesinde (TPR@1%) CNN açıkça önde. Öznitelik ekledikçe fark büyüyor: CNN yeni bilgiyi kullanıyor, conv-AE kullanamıyor.
- Bu fark, denetimli ile denetimsiz öğrenme farkını da içeriyor; saf mimari etkisi değil.

### 2.4 Skor bileşimi (yalnızca val; `ext_diag_channels.csv`)

Skordaki pay = kanal grubunun yeniden oluşturma hatasının toplam MSE'deki payı. Oran = saldırı penceresindeki hata / temiz normal penceredeki hata.

| Set | Kanal grubu | Normalde pay | Saldırıda pay | Saldırı/normal oranı |
|---|---|---|---|---|
| S0 | yük baytları (8) | **0.870** | 0.852 | **1.08** |
| S0 | ID kodu | 0.099 | 0.112 | 1.25 |
| S0 | IAT | 0.031 | 0.036 | 1.26 |
| S3 | yük baytları | 0.611 | 0.580 | 1.12 |
| S3 | bayt değişimi (8) | 0.297 | 0.313 | 1.24 |
| S3 | ID frekansı | **0.003** | 0.007 | **3.67** |

- Skorun büyük kısmını, normal trafikte de yüksek hata veren ve saldırıyı en az ayırt eden yük kanalları oluşturuyor.
- En ayırt edici kanal (ID frekansı, oran 3.7) skora binde 3 katkı yapıyor. Eşit ağırlıklı MSE ortalaması, ayırt edici sinyali gürültüde boğuyor.
- Bu, CNN ile conv-AE arasındaki farkın olası mekanizmalarından biri: CNN kanal ağırlıklarını etiketlerden öğreniyor, conv-AE öğrenemiyor.
- **Sınanmamış yollar:** kanal başına normalleştirilmiş hata skoru, kanal seçimi, ağırlıklı skor. Bunlar yeni bir model seçimi olur ve bu fazın kapsamı dışında. Yapılacaksa önceden kayıtlı bir kuralla ve val'de yapılmalı.

### 2.5 Kısa yanıt

> conv-AE'nin %5.5 recall'u özellikten mi, mimariden mi, eşikten mi?

Üçünden de ve skor tasarımından. S0'daki toplam fark (CNN 0.545 − conv-AE 0.055 = 0.49) betimleyici olarak şöyle ayrışıyor:

| Bileşen | Hesap | Büyüklük |
|---|---|---|
| Eşiğin teste sıkı aktarımı | conv-AE TPR@1% − kalibre recall | ≈ 0.19 |
| Aynı test FPR'ında ayırt edicilik farkı (model/paradigma ve skor bileşimi) | CNN TPR@1% − conv-AE TPR@1% | ≈ 0.22 |
| CNN'in kendi gevşek eşiği | CNN kalibre recall − CNN TPR@1% | ≈ 0.08 |

Öznitelik eklemek conv-AE'nin fabrication türlerindeki sıralamasını iyileştiriyor, ama masquerade'de etkisi yok ya da olumsuz.

## 3. Nicemleme ve eşik (K-077): test

### 3.1 1D-CNN, S3 ile (S0 yanında)

| ROAD | Kip | Toplam recall Q0 → Q2 | ΔRecall (Q0 − Q2) | Saldırısız FPR Q0 → Q2 |
|---|---|---|---|---|
| S0 | yeniden kalibre | 0.545 → 0.538 | 0.006 | 0.0209 → 0.0203 |
| S0 | eski (float32) eşik | 0.545 → 0.539 | 0.005 | 0.0209 → 0.0201 |
| **S3** | yeniden kalibre | 0.780 → 0.773 | 0.007 | 0.0173 → 0.0175 |
| **S3** | eski (float32) eşik | 0.780 → 0.767 | 0.013 | 0.0173 → 0.0167 |

- Tür bazında Q2 ΔRecall S3'te (yeniden kalibre) en fazla 0.013 (reverse_light_on); S0'da 0.008. Eski eşikle en fazla 0.026 (reverse_light_on). Tür içi FPR değişimi ≤ 0.008.
- Q1 (fp16), S3: tür bazında |ΔRecall| ≤ 0.0006 (S0'da ≤ 0.0023).
- Q2 dosyası Q0'ın 0.372'si.
- **Girdi nicemlemesi:** ızgara sütunları (ID kodu, baytlar, bit/bayt değişimi) kayıpsız. Sürekli sütunlar (IAT, ID frekansı) tasarım gereği kayıplı (K-073).
- **CH (S3):** recall 0.9999 bütün düzeylerde; saldırısız FPR 0.0139 → 0.0123 (Q2, yeniden kalibre).
- **Sonuç:** daha zengin öznitelikle CNN'in INT8 kaybı biraz büyüdü (tür bazında en fazla 0.008 → 0.013), ama hâlâ ≈ 1 recall puanının altında. Eski eşik kullanılsa da yanlış alarm oranı kaymıyor.

### 3.2 conv-AE (S0 korundu; P5 sonuçları)

| ROAD | Kip | Saldırısız FPR Q0 → Q2 | Toplam recall Q0 → Q2 |
|---|---|---|---|
| S0 | yeniden kalibre | 0.0009 → 0.0010 | 0.055 → 0.056 |
| S0 | eski eşik | 0.0009 → **0.062** (tohuma göre 0.002–0.174) | 0.055 → 0.364 |

- Eski eşikte INT8 conv-AE'nin yanlış alarm oranı tohuma bağlı olarak 1.7 ile 372 kat artıyor (tohum 2, 0, 1: 0.0012 → 0.0020, 0.0011 → 0.0089, 0.0005 → 0.174).
- Recall'daki artış tespit iyileşmesi değil; yanlış alarmdan geliyor.
- Eşik nicemlenmiş modelin kendi val skorlarıyla yeniden kalibre edilince kayma kayboluyor (P6 bulgusu, değişmedi).

## 4. Sınırlılıklar

- **Kümülatif ablasyon:** grupların tek başına etkisi ölçülmedi.
- **Kapsam:** XGBoost/RF yeni özniteliklerle eğitilmedi. Genişletilmiş setlerde gecikme ölçülmedi.
- **Gizli grup sentetik:** yalnızca ROAD'un sonradan üretilmiş masquerade kayıtları (K-014). Zamanlama özniteliklerinin bu kayıtlardaki kazancı yapıntı olabilir (§1.5).
- **Literatür:** 3 makalenin tam metni okunamadı.

## 5. Görüşünüze ihtiyaç duyulan noktalar

1. **Makale:** genişletme sonuçları makaleye girsin mi? Önerim: S3 ablasyonu ve conv-AE teşhisi kısa bir alt bölüm ya da ek olarak. Ana hükümler (H1–H4) değişmiyor.
2. **conv-AE skoru:** kanal başına normalleştirilmiş skor gibi bir düzeltme, önceden kayıtlı bir kuralla ve val'de denensin mi? Bu, yeni bir model seçimi olur.
3. **Masquerade:** S3 kazancının sentetik masquerade yapıntısından gelip gelmediği ayrı bir deneyle sınansın mı? Örneğin yalnızca hedef ID'nin zamanlama izini maskeleyen bir kontrol.
