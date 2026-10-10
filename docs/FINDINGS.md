# Bulgular — P6 (Plan F5: analiz ve istatistik)

**Tarih:** 2026-10-10.
**Kaynaklar:**
- `results/tables/stats.csv`, `as1_pairs.csv`, `per_type_delta.csv`, `pareto.csv`, `calibration_shift.csv`, `decision_churn.csv`, `delta_f1.csv`, `all_results.csv`;
- üretim: `python -m analysis.aggregate`, `python -m analysis.stats` (B = 10.000; aggregate commit `1233f00`, stats commit `88eed36`; ikisi de kirli değil; `analysis_manifest.json`). Bütün sayısal çıktılar `1233f00`'dan beri aynı; `88eed36` yalnızca H4 satırının hüküm metnini değiştirdi (K-064).

**Kararlar:** K-057 … K-063 (`docs/DECISIONS.md`).

**Ortak koşullar:**
- Test kümesi; eşik kipi `recalibrated` (K-036 a): her sürüm kendi val eşiğiyle, %1 FPR.
- Δ = Q0 − Qk; pozitif değer kayıp demek.
- Değerler 3 tohumun ortalaması.
- Köşeli parantezler %95 hiyerarşik bootstrap aralığıdır (tohum × pencere, hareketli blok; K-058).
- Sayılar CSV'lerden yuvarlanarak alındı.

**Genel sınırlılık:**
- Gizli grubun tamamı ROAD'un sentetik türetilmiş masquerade kayıtlarıdır (K-014).
- Gecikme yalnızca ana makinede, tohum 0'da ölçüldü (K-050, K-051, K-052).
- Aralıklar test penceresi ve eğitim tohumu belirsizliğini gösterir; veri setinin kendisinden gelen belirsizliği göstermez.

## Özet: hipotez hükümleri (K-062)

| Hipotez | Hüküm | Dayanak (kısa) |
|---|---|---|
| **H1** — toplam kayıp sınırlı, türler arasında eşit dağılmıyor | **Doğrulandı** (kurala göre). Ancak eşitsizlik ihmal edilebilir büyüklükte. | (a) Toplam ΔF1(Q2) en fazla 0.0029 (ROAD CNN), sınır 0.05. (b) 26 birincil tür × model satırının 3'ünde ΔRecall(Q2) aralığı 0'ın üstünde (CNN ROAD, ≈ 0.007); 22'sinde aralık 0'ı içeriyor. |
| **H2** — gizli saldırılarda kayıp daha büyük; kaba saldırılar INT8'de de yüksek başarımla tespit ediliyor | **Doğrulanmadı** | (a) Birincil birimlerde D (Q2) = 0.0006 [−0.0035, 0.0057]. (b) conv-AE ROAD fuzzing recall'u Q2'de 0.539 < 0.90; bu değer zaten Q0'da 0.564, yani nicemlemeden kaynaklanmıyor. |
| **H3** — float16 en dengeli nokta; INT8 en yüksek kazancı veriyor ama gizli saldırılarda belirgin kayıp yaratıyor | **Kısmen doğrulandı** | Q1'de \|ΔRecall\| Q2'dekinden küçük; Q1 dosyası Q0'ın 0.52'si. Q1'in gecikme kazancı test edilemedi (K-043). INT8'de gizli saldırılara özgü kayıp yok (H2 a). |
| **H4** — denetimsiz model nicemlemeye daha duyarlı | **Kısmen doğrulandı** (K-064, kullanıcı kararı) | Önceden kayıtlı kural (K-062): **doğrulandı**; mutlak değer farkı ΔAUC(Q2) 0.0062 [0.0048, 0.0272]. Bu istatistik gürültülü model lehine yanlı (K-063). K-063 işaretli denetimi: **doğrulanmadı**; ΔAUC_AE − ΔAUC_CNN = 0.0025 [−0.0041, 0.0088]. Çıktı değişimi conv-AE'de daha büyük (ROAD Spearman 0.973 ve 0.9997). Üç aşama birlikte aşağıda. |

## AS-1 — Nicemleme kaybı saldırı türüne göre farklı mı?

### Birimler ve XGBoost'un durumu (K-057)

- AS-1 birimi (model, fabrication–masquerade çifti). Yalnızca nicemlenen modeller birim olabilir (`as1_pairs.csv`).
- **XGBoost AS-1 birimi değildir.** Nicemlenmediği için (K-042) Qk'si ve ΔRecall'u tanımsızdır.
  - `grid_k041_units.csv`'de XGB'nin iki birimi `in_primary_test = True` görünüyor: max_speedometer (Q0 recall fabrication / masquerade 0.486 / 0.110) ve reverse_light_off (0.624 / 0.158).
  - Bunun nedeni K-041 taban kuralının yalnızca Q0'a bakması ve P5'te ızgaradaki her modele uygulanmasıdır.
  - Tablo değiştirilmedi. `as1_pairs.csv` bu birimleri `as1_eligible = False` ("not quantized: no Qk, delta undefined") olarak işaretliyor. RF ızgarada yok (K-053).
- **Birincil birimler:** 3 CNN çifti. Q0 recall'ları (fabrication / masquerade): correlated_signal 0.901 / 0.854, max_speedometer 0.642 / 0.718, reverse_light_off 0.356 / 0.540.
- **Raporlanıp birincil analize girmeyenler:**
  - taban kuralıyla (K-041): CNN reverse_light_on (0.088 / 0.096) ve conv-AE'nin 4 çifti (Q0 recall'ları 0.000–0.113);
  - birincil havuz dışında (K-019): coolant çifti.
- Analizdeki birim kümesi, P5'teki K-041 tablosuyla kodda karşılaştırıldı ve aynı çıktı.

### Birincil eşleştirilmiş analiz (keşifsel; hipotez testi yok)

D = ΔR_masq − ΔR_fab. D > 0 ise gizli (masquerade) tarafta kayıp daha büyüktür.

| Birim (CNN, ROAD) | k | ΔR_fab | ΔR_masq | D | Tohum bazında D |
|---|---|---|---|---|---|
| correlated_signal | Q2 | 0.0021 [−0.0042, 0.0095] | 0.0022 [−0.0044, 0.0132] | 0.0001 [−0.0095, 0.0112] | −0.0063; −0.0001; 0.0068 |
| max_speedometer | Q2 | 0.0068 [0.0026, 0.0112] | 0.0071 [0.0027, 0.0119] | 0.0003 [−0.0048, 0.0057] | −0.0004; 0.0002; 0.0010 |
| reverse_light_off | Q2 | 0.0066 [0.0011, 0.0137] | 0.0080 [−0.0023, 0.0223] | 0.0014 [−0.0067, 0.0139] | 0.0074; −0.0033; 0.0001 |
| **3 birimin ortalaması** | **Q2** | | | **0.0006 [−0.0035, 0.0057]** | 0.0002; −0.0011; 0.0026 |
| **3 birimin ortalaması** | **Q1** | | | **0.0010 [0.0000, 0.0029]** | 0.0002; 0; 0.0028 |

- **Bulgu:** INT8, CNN'de iki fabrication türünde ve bir masquerade türünde küçük ama 0'dan ayrılan bir recall kaybı üretiyor (≈ 0.007, yani 0.7 recall puanı). Kayıp çiftin iki tarafında **aynı büyüklükte**.
- Gizli tarafa özgü ek kayıp (D) Q2'de 0.0006. Aralığın üst sınırı 0.0057: bu tasarımda, INT8'in masquerade'e özgü ek kaybı 3 birimin ortalamasında ≈ 0.6 recall puanından büyük görünmüyor.
- **Q1:** kayıp neredeyse yok. Aralık [0, 0.003] tohum 2'deki tek pencerelik değişimlerden geliyor.
- **ΔF1 (ikincil):** D_F1(Q2) = −0.0002 [−0.0033, 0.0035]. Aynı sonuç.
- **Ulaşılabilir en küçük p:** 3 birimde Wilcoxon için 0.25. Test yapılmadı (K-057).
- **Duyarlılık (iid pencere, L = 1):** aralıklar blok sürümüyle neredeyse aynı. Örnek: Q2 ortalama D [−0.0037, 0.0051]. Pencereler arası ilişki aralığı bu veride belirgin biçimde değiştirmiyor (ayrıca "Beklenmedik sonuçlar").
- **Taban altı birimler** (yalnızca bilgi):
  - conv-AE'nin Q2 D değerleri −0.0006 ile 0.0041 arasında; aralıkların hepsi 0'ı içeriyor.
  - conv-AE recall'u tabanda olduğu için bu değerler kayıp ölçüsü olarak anlamlı değil (K-041).

### İkincil grup karşılaştırması (K-059; ROAD, 4 gizli ve 5 kaba tür)

| Model | k | Cliff's δ (ΔRecall) | p (kesin MWU) | p (Holm) |
|---|---|---|---|---|
| CNN | Q1 | 0.65 [0.00, 0.75] | 0.135 | 0.444 |
| CNN | Q2 | 0.70 [−0.30, 0.90] | 0.111 | 0.444 |
| conv-AE | Q1 | 0.20 [−0.20, 0.50] | 0.683 | 1.000 |
| conv-AE | Q2 | −0.10 [−0.45, 0.50] | 0.905 | 1.000 |

- **Ulaşılabilir en küçük p** 2/126 = 0.016. Holm ile (4 test) 0.063. **Bu tasarımda düzeltilmiş anlamlılık yapısal olarak ulaşılamaz;** sonuç ne olursa olsun.
- **CNN Q2:** gizli türlerin ortalama ΔRecall'u 0.0061, kaba türlerinki 0.0031. Etki büyüklüğü δ = 0.70 büyük görünüyor, ama mutlak fark 0.3 recall puanı ve aralık 0'ı içeriyor.
- **ΔF1 ailesi:** CNN Q2 δ = 0.10. Bütün Holm p'leri ≥ 0.44.
- **Duyarlılık (CH'nin kaba türleri eklenince, 9'a 4):**
  - CNN Q2: δ = 0.83 [−0.06, 0.94], p = 0.015, Holm p = 0.062.
  - CH'de CNN'in ΔRecall'u tavan nedeniyle tam 0. Kaba gruba eklenen türler bu yüzden yalnızca veri seti farkını taşıyor (karıştırıcı).

### AS-1 yanıtı

Bu veriyle, INT8'in gizli (masquerade) saldırılarda kaba saldırılardan daha büyük kayıp ürettiğine dair kanıt yok. Eşleştirilmiş çiftlerde iki tarafın kaybı aynı büyüklükte (≤ 1 recall puanı). Tür bazında kayıplar küçük ve ROAD CNN'de yoğunlaşıyor; Car-Hacking'de CNN tavanda (ΔRecall = 0).

## AS-2 — Gecikme, boyut ve recall ödünleşimi (K-060)

| Veri seti | Aday | Boyut (MB) | Medyan gecikme, oturum 1 / 2 (ms) | CNN Q0'a oran (iki oturumun aralığı) | Gizli tür recall'u (ROAD) | Toplam recall |
|---|---|---|---|---|---|---|
| ROAD | CNN Q0 | 0.103 | 0.040 / 0.047 | 1 | 0.552 [0.516, 0.592] | 0.545 |
| ROAD | CNN Q2 | 0.039 | 0.030 / 0.030 | 0.64–0.75 | 0.546 [0.511, 0.584] | 0.538 |
| ROAD | conv-AE Q0 | 0.111 | 0.102 / 0.107 | 2.30–2.57 | 0.034 | 0.055 |
| ROAD | conv-AE Q2 | 0.042 | 0.107 / 0.108 | 2.32–2.69 | 0.036 | 0.056 |
| ROAD | XGB | 0.274 | 0.500 / 0.484 | 10.4–12.6 | 0.089 | 0.276 |
| CH | CNN Q0 | 0.103 | 0.042 / 0.040 | 1 | — | 0.9997 |
| CH | CNN Q2 | 0.039 | 0.030 / 0.030 | 0.70–0.74 | — | 0.9997 |
| CH | conv-AE Q0 / Q2 | 0.111 / 0.042 | 0.102–0.116 | 2.42–2.88 | — | 0.955 / 0.963 |
| CH | XGB | 0.205 | 0.518 / 0.489 | 12.1–12.3 | — | 1.000 |

- **Sağlam Pareto cephesi** (bir aday ancak iki oturumda da baskılanırsa elenir):
  - **ROAD** (gizli tür recall'u): CNN Q0 ve CNN Q2.
    - Q2: medyan gecikme %25–36 daha düşük, dosya 0.38 kat. Karşılığında gizli tür recall'u 0.006 daha düşük; aralıklar büyük ölçüde örtüşüyor.
    - conv-AE ve XGB baskılanıyor.
  - **CH** (toplam recall): CNN Q2 ve XGB. XGB'nin recall'u 1.000, CNN Q2'ninki 0.9997; buna karşılık XGB ≈ 12 kat yavaş ve ≈ 5 kat büyük.
- **Boyut–recall cephesi** (Q1 dahil, gecikme yok): ROAD'da CNN Q0, Q1 ve Q2'nin üçü de cephede. Q1, boyut (0.054 MB) ve recall (0.551) bakımından ikisinin arasında.
- **Gecikme bütçeleri** (ileti hızı: saldırısız kayıtların medyanı; CH 1901, ROAD 2384 ileti/s):
  - **Pencere bütçesi** (100 ileti): CH 52.6 ms, ROAD 41.9 ms. Bütün adaylar bütçenin çok altında.
  - **Sıkı bütçe** (1 ileti): CH 0.526 ms, ROAD 0.419 ms.
    - Derin modellerin hepsi bu bütçeye uyuyor (≤ 0.12 ms).
    - XGB ROAD'da uymuyor (0.50 ms).
    - XGB CH'de sınırda uyuyor (0.518 < 0.526 ms). Mutlak değerler K-050'ye göre yalnızca büyüklük mertebesi olarak yorumlanmalı; bu marj o mertebenin altında.
- **Gecikmenin yönü:**
  - CNN'de INT8, iki veri setinde de ve iki oturumda da medyan gecikmeyi düşürüyor (oran 0.64–0.75).
  - conv-AE'de düşürmüyor; Q2/Q0 oranı 1.01–1.08 (K-050). ConvTranspose katmanları INT8 çekirdeksiz, QDQ benzetimiyle çalışıyor (K-044); bu bir açıklama önerisi, doğrulanmadı.
- **AS-2 yanıtı:** Ana makine ölçümüyle iki bütçe de bağlayıcı değil. ROAD'da en uygun kombinasyon **CNN INT8**: en küçük ve en hızlı aday; gizli tür recall'unda CNN float32'den farkı aralıkların içinde kalıyor. Bir ECU üzerinde ölçüm yapılmadı (K-051); bütçe uyumu ECU için kanıt değildir.

## AS-3 — Denetimli ve denetimsiz modelin nicemleme duyarlılığı (K-061, K-063)

### K-061 birincil analizi (önceden kayıtlı)

d_t = |Δ_AE,t| − |Δ_CNN,t|, türler üzerinden ortalama.

| Ölçü | k | Ortalama d | Aralık | Wilcoxon p | Holm p | Birim |
|---|---|---|---|---|---|---|
| ΔROC-AUC | Q1 | 0.000008 | [0.000002, 0.000125] | 0.893 | 1.000 | 13 tür |
| ΔROC-AUC | Q2 | 0.0062 | [0.0048, 0.0272] | 0.0007 | 0.0029 | 13 tür |
| ΔRecall (taban ≥ 0.10) | Q1 | 0.0009 | [−0.0003, 0.0048] | 0.500 | 1.000 | 6 tür |
| ΔRecall (taban ≥ 0.10) | Q2 | 0.0062 | [0.0033, 0.0265] | 0.250 | 0.750 | 6 tür |

K-062 kuralına göre H4 doğrulanıyor: Q2'de iki ölçünün aralığı da 0'ın üstünde.

### Neden bu sonuç tek başına hüküm değil (K-063)

- Mutlak değerin beklenen değeri, gerçek etki sıfır olsa bile gürültüyle büyür.
- conv-AE'nin Δ'sı tohumlar arasında çok daha değişken. Örnekler:
  - ROAD correlated_signal ΔAUC(Q2): 0.024 ± 0.071 (CNN: 0.00005 ± 0.0002);
  - ROAD fuzzing ΔRecall(Q2): 0.025 ± 0.095.
- Bu yüzden d_t, AE lehine yukarı yanlıdır. Belirti: nokta tahmini aralığın alt ucuna yakın, aralık sağa doğru uzanıyor (0.0062 [0.0048, 0.0272]).
- Wilcoxon testi de aynı d_t değerleri üzerinde yapıldığı için bu yanlılığı taşır.
- Yanlılığın varlığı birim testle gösterildi (`tests/test_stats.py::test_absolute_difference_is_biased_upward_for_the_noisier_model`).

### Sağlamlık denetimleri (K-063; sonuç görüldükten sonra eklendi)

**(a) İşaretli fark** (Δ_AE − Δ_CNN; pozitif: AE daha çok kaybediyor):

| Ölçü | k | Ortalama | Aralık |
|---|---|---|---|
| ΔROC-AUC | Q2 | 0.0025 | [−0.0041, 0.0088] |
| ΔRecall (taban) | Q2 | −0.0002 | [−0.0176, 0.0172] |
| ΔROC-AUC | Q1 | −0.00008 | [−0.00016, −0.0000002] |

- Q2'de conv-AE'nin ortalama ΔAUC'si 0.0031 [−0.0034, 0.0093], CNN'inki 0.0006 [−0.0001, 0.0015].
- Hiçbir model için ortalama kayıp 0'dan ayrılmıyor; iki modelin farkı da ayrılmıyor.
- Q1'deki fark 0'dan ayrılıyor, ama büyüklüğü 10⁻⁴'ün altında ve yönü AE lehine (AE daha az kaybediyor).

**(b) Karar değişimi ve sıra korunumu** (`decision_churn.csv`; birincil havuz; Q2; ortalama, tohumlar arasında min–max):

| Veri seti | Model | Saldırı pencerelerinde karar değişimi | Normal pencerelerde karar değişimi | Spearman (Q0 ile Q2 skoru) |
|---|---|---|---|---|
| ROAD | CNN | 0.0075 (0.0055–0.0093) | 0.0009 | 0.9997 (0.9997–0.9998) |
| ROAD | conv-AE | 0.0197 (0.0107–0.0276) | 0.0004 | **0.973 (0.946–0.990)** |
| CH | CNN | 0 | 0.0060 | 0.9988 |
| CH | conv-AE | 0.0084 (0.0040–0.0140) | 0.00002 | 0.9994 |

- ROAD'da INT8 conv-AE'nin skor sıralamasını CNN'inkinden belirgin biçimde daha çok bozuyor. Saldırı pencerelerinde karar değişimi ≈ 2.6 kat.
- Ancak değişen kararlar iki yöne de gidiyor; net recall ve AUC kaybı 0'dan ayrılmıyor.
- CH'de iki model benzer.

### AS-3 yanıtı ve H4

- **"Denetimsiz model nicemlemeyle daha çok başarım kaybeder"** (yönlü okuma): veri desteklemiyor (denetim a).
- **"Denetimsiz modelin çıktısı nicemlemeyle daha çok değişir"** (büyüklük okuması): ROAD'da destekleniyor (denetim b, kalibrasyon kayması). Ancak bu değişim yeniden kalibre edilmiş eşikte net bir kayba dönüşmüyor.
### H4'ün hükmü: üç aşama birlikte (K-062 → K-063 → K-064)

| Aşama | Ne zaman yazıldı | Ölçü (Q2) | Sonuç | Hüküm |
|---|---|---|---|---|
| **1. Önceden kayıtlı kural (K-062)** | P6 aralıklarından önce (P5 betimleyicilerinden sonra) | Mutlak değer farkı \|Δ_AE\| − \|Δ_CNN\|: ΔAUC 0.0062 [0.0048, 0.0272]; ΔRecall 0.0062 [0.0033, 0.0265] | İki aralık da 0'ın üstünde | **Doğrulandı** |
| **2. Düzeltme / sağlamlık denetimi (K-063)** | B = 200 duman koşusu görüldükten sonra, tam koşudan önce | İşaretli fark Δ_AE − Δ_CNN: ΔAUC 0.0025 [−0.0041, 0.0088]; ΔRecall −0.0002 [−0.0176, 0.0172]. Çıktı ölçüleri (ROAD): Spearman 0.973 ve 0.9997; saldırı pencerelerinde karar değişimi 0.0197 ve 0.0075 | Başarım kaybı farkı yok; çıktı değişimi farkı var | **Doğrulanmadı** (yönlü okuma) |
| **3. Nihai hüküm (K-064)** | Bütün sonuçlar görüldükten sonra, kullanıcı kararı | — | — | **Kısmen doğrulandı** |

**Neden kayıtlı kuralın hükmü tek başına kabul edilmedi?**
- Kuralın istatistiği mutlak değer farkıdır. Mutlak değerin beklenen değeri, gerçek etki sıfır olsa bile gürültüyle büyür.
- conv-AE'nin Δ'sı tohumlar arasında CNN'inkinden çok daha değişken. Örneğin ROAD ΔAUC(Q2) tohum std'si correlated_signal'da 0.071 ve 0.0002, max_speedometer'da 0.017 ve 0.003 (`per_type_delta.csv`). Bu yüzden istatistik, gerçek bir fark olmasa da conv-AE'yi "daha duyarlı" gösterme eğilimindedir.
- Belirtisi: nokta tahmini aralığın alt ucunda, aralık sağa doğru uzanıyor.
- Bu bir yöntem özelliğidir, veriye bakılarak seçilmiş bir yorum değildir (birim test: `tests/test_stats.py::test_absolute_difference_is_biased_upward_for_the_noisier_model`).
- Yine de kural sonuçlardan önce kaydedildiği için hükmü silinmedi; burada ve `stats.csv`'nin H4 satırında nihai hükümle birlikte duruyor.

**"Kısmen"in içeriği:**
- **Desteklenen:** INT8, denetimsiz modelin çıktısını denetimli modelinkinden daha çok değiştiriyor. ROAD'da skor sırası daha çok bozuluyor, kararlar daha çok değişiyor ve float32 eşiği kullanılırsa yanlış alarm oranı büyük ölçüde kayıyor (kalibrasyon kayması).
- **Desteklenmeyen:** eşik nicemlenmiş modelin kendi val skorlarıyla yeniden kalibre edildiğinde, denetimsiz model denetimliden daha çok başarım (recall, ROC-AUC) kaybetmiyor.

**Zamanlama:** K-063 sonuç görüldükten sonra eklendi; nihai hüküm (K-064) bütün sonuçlar görüldükten sonra kullanıcı tarafından verildi. Makalede bu sıra açıkça yazılacak.

## Kalibrasyon kayması (K-036 b: Q0 eşiği Qk'ye aynen uygulandığında)

`calibration_shift.csv`. Saldırısız (ambient) kayıtlarda FPR; tohum ortalaması (min–max).

| Koşu | Val FPR Q0 → Q2 | Test FPR Q0 → Q2 | Toplam recall Q0 → Q2 |
|---|---|---|---|
| ROAD conv-AE | 0.0102 → **0.185 (0.020–0.486)** | 0.0009 → **0.062 (0.002–0.174)** | 0.055 → 0.363 (0.070–0.733) |
| CH conv-AE | 0.0095 → 0.0125 | 0.0020 → 0.0028 | 0.955 → 0.971 |
| ROAD CNN | 0.0095 → 0.0094 | 0.0209 → 0.0201 | 0.545 → 0.539 |
| CH CNN | 0.0095 → 0.0091 | 0.0146 → 0.0129 | 0.9997 → 0.9997 |

- **Q1'de kayma yok:** iki modelde de FPR değişimi ≤ 0.0006.
- **Bulgu:** INT8, ROAD'daki conv-AE'nin yeniden oluşturma hatası dağılımını kaydırıyor. Float32 eşiğiyle kullanılırsa yanlış alarm oranı tohuma bağlı olarak 2 ile 48 kat artıyor (val 0.020–0.486).
  - Toplam recall'daki artış (0.055 → 0.363) tespit iyileşmesi değil; yanlış alarmdan geliyor. Bu kipte tür bazında "ΔRecall" −0.90'a kadar iniyor (tohum 1).
  - CNN'de kayma pratikte yok.
  - **Dağıtım için sonuç:** nicemlenmiş otokodlayıcının eşiği, nicemlenmiş modelin kendi val skorlarıyla yeniden kalibre edilmelidir. Birincil analiz (K-036 a) bunu zaten yapıyor; bu yüzden AS-1 ve AS-3'te kayma görünmüyor.
- Bu kayma, AS-3'teki "çıktı duyarlılığı" bulgusuyla tutarlıdır.

## Beklenmedik sonuçlar

1. **Masquerade, CNN için fabrication'dan zor değil.**
   - CNN'in Q0 recall'u üç birincil çiftin ikisinde masquerade'de daha yüksek: max_speedometer 0.718 / 0.642, reverse_light_off 0.540 / 0.356. correlated_signal'da daha düşük: 0.854 / 0.901.
   - XGB'de ise fabrication açıkça daha kolay: max_speedometer 0.486 / 0.110, reverse_light_off 0.624 / 0.158.
   - Pencere özet öznitelikleriyle (zamanlama, kimlik) çalışan XGB zamanlama izine dayanıyor; yük baytlarını gören CNN bu ize ihtiyaç duymuyor. Bu bir yorum önerisidir; ayrı bir deneyle sınanmadı.
   - "Gizli saldırı daha zor" sezgisi model ailesine bağlı görünüyor.
2. **INT8, conv-AE'nin CH gear_spoofing recall'unu artırıyor:** ΔRecall(Q2) = −0.024 [−0.044, −0.011], üç tohumda da aynı yönde. ΔAUC ise ihmal edilebilir (−0.00003). Ayırt edicilik değişmiyor; değişen, kalibrasyon kuyruğunda eşiğin yeri. CH kalibrasyon kümesi 1,684 pencere ve %1 eşiği ≈ 17 pencerelik kuyruğa dayanıyor (K-038). Açıklama doğrulanmadı.
3. **Blok bootstrap aralıkları iid ile neredeyse aynı.** Ardışık pencerelerin ilişkili olması aralığı genişletmedi. Olası neden: nicemlemeyle değişen kararlar seyrek (saldırı pencerelerinin en fazla %2.8'i, `decision_churn.csv`) ve dağınık.
4. **conv-AE'de INT8 gecikme kazancı yok** (oran 1.01–1.08 / CNN Q0'a göre 2.3–2.9 kat). P4'te biliniyordu (K-050); AS-2'de conv-AE'yi cepheden çıkaran etkenlerden biri.
5. **Q1'de AS-3 işaretli farkı 0'dan ayrılıyor, ama AE lehine ve 10⁻⁴'ün altında.** Pratik anlamı yok; belirtilmesi gereken bir istatistiksel ayrıntı.
6. **H2 (b)'nin düşmesinin nedeni nicemleme değil:** conv-AE'nin ROAD fuzzing recall'u Q0'da da 0.564 ± 0.221.

## Kapsam ve dürüstlük notları

- **Önceden kayıtlı olmayanlar:** H1–H4 hüküm kuralları (K-062) P5 betimleyici sonuçlarından sonra, P6 aralıklarından önce yazıldı. K-063 ise P6 duman koşusundan sonra yazıldı. H4'ün nihai hükmü (K-064) bütün sonuçlar görüldükten sonra kullanıcı tarafından verildi.
- **Bu fazda bulunan hesaplama hataları** (sonuçlar raporlanmadan önce düzeltildi; P6 günlüğünde ayrıntılı):
  - Q1 eşiklerinin `metrics.csv`'deki 16 haneli yuvarlaması;
  - aralık sınırlarındaki kayan nokta artığı.
- **Car-Hacking'de CNN tavanda:** ΔRecall'un aşağı yönde payı yok. Car-Hacking'in AS-1 ve AS-3'e katkısı sınırlı.
- **Pencere skorları git dışında:** analiz, ham `scores_*.parquet` dosyalarını gerektiriyor. Girdi özetleri (`sha256`) `analysis_manifest.json`'da.
