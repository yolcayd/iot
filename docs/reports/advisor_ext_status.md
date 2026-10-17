# Danışman geri bildirimi 1 — mevcut durum raporu

**Tarih:** 2026-10-17 · **Faz:** P10 (genişletme; K-071) · **Geri bildirim metni:** `docs/reports/advisor_feedback_01.md`

Bu rapor yeni deneylerden **önce** yazıldı. İçeriği:
- danışmanın her isteğinin bugünkü karşılanma durumu;
- bu fazda yapılan ek doğrulamaların sonuçları (sonuç olarak yalnızca veri denetimleri var; model sonucu yok);
- henüz yapılmamış olanlar ve nasıl yapılacakları.

Durum sütunu: **✔** zaten karşılanıyor · **◐** kısmen · **✚** yeni (bu fazda yapılacak).

## 1. İsteklerin durumu

| # | Danışman isteği | Durum | Kanıt | Bu fazda |
|---|---|---|---|---|
| 1 | Literatürdeki ön işlemelerin incelenmesi | ✚ | — | Ayrı rapor: `docs/reports/literature_preprocessing.md` (7 makale, kaynak atıflı tablo) |
| 2a | Normalizasyon yalnızca eğitimden | ✔ | `src/data/features.py:111` `fit_normalization` train dışı segmenti reddediyor; testler `test_normalisation_fitted_on_training_segments_only`, `test_id_vocabulary_from_benign_training_frames_only`; K-024, K-025 | Köken tablosu eklendi (§2.3) |
| 2b | Zaman sırasının korunması | ◐ → ✔ | Pencereler ardışık satırlardır (`src/data/windows.py`). Zaman sırası testi yoktu | Yeni test `test_timestamps_monotonic_within_segments`: **geçti** (§2.1) |
| 2c | DLC < 8 sıfır dolgu | ◐ → ✔ | Yükleyici `byte_cells_to_uint8` (`src/data/loaders.py:136`); CH'de bayrak `3 + DLC` sütununda (`docs/DATA.md`); test `test_short_dlc_payload_is_zero_padded_in_features` | Dolgu belirsizliği ölçüldü (§2.2) |
| 2d | Eğitim–test sızıntısı yok | ✔ | `tests/test_no_leakage.py` (20 test): 32 çerçevelik dizi kimliğiyle kopya taraması (K-023), masquerade çiftleri aynı kümede (K-007, K-018), ortak kuyruk tek kullanım (K-016), örtüşmesiz val/test (K-002), segmentlerin çakışmaması | — |
| 2e | CH zaman bloğu ayrımının bağımsızlık sınırlılığı | ◐ | K-017: sınırlılık kayıtlı; makalenin Sınırlılıklar bölümüne aday | Değerlendirme §2.4; literatürle karşılaştırma ayrı raporda |
| 3a | Aynı ID'nin ardışık mesajları arası süre | ✔ | Bizdeki IAT **zaten ID bazında**: `per_id_iat` (`src/data/features.py:53`), segment içinde, log ölçekli (K-021, K-025) | — |
| 3b | "Mesajlar arası zaman farkı" (ID'den bağımsız, global) | ✘ (not) | Temsilde yok | Ablasyona eklenmedi (P10 listesinde yok); not olarak raporlanır (K-072 madde 4) |
| 3c | Mesaj frekansı | ✚ | — | G3 `id_rate`: son 1 s'deki aynı ID sayısı / train normal referansı (K-073) |
| 3d | ID dağılımı | ✚ | Yalnızca XGBoost'un pencere özetinde vardı (`id_entropy`, `n_unique_ids`; `src/models/classical.py`) | G4 `id_share_w`, `id_entropy_w`: son 100 çerçeve (K-073) |
| 3e | Payload değişimi | ✚ | Yalnızca XGBoost'ta `payload_change_rate` | G2 `dbyte_0..7` (K-073) |
| 3f | Bit değişim oranı | ✚ | — | G1 `bit_flips` = Hamming / 64 (K-073) |
| 4 | Aşamalı ablasyon (10 öznitelik referans; 1D-CNN ve conv-AE; aynı ayrımlar; özellikle ROAD, tür bazında Recall, F1, AUC) | ✚ | — | S0 → S4, 48 yeni koşu, 3 tohum; önce yalnızca val, seçimden sonra test (K-074, K-075) |
| 5 | conv-AE'nin ROAD'daki ≈ %5.5 recall'unun nedeni | ◐ | K-040 (P3): conv-AE'nin testteki gerçekleşen FPR'ı hedefin onda biri (ROAD %0.11). Ayrıştırma yapılmamıştı | Eşik / öznitelik / mimari / skor bileşimi ayrıştırması (K-076) |
| 6 | FP32/FP16/INT8 ve INT8'de eski ile yeniden kalibre eşik (FPR, recall; bağımsız test) | ◐ | S0 için yapıldı: K-036 (a, b), `results/tables/calibration_shift.csv`; ROAD conv-AE INT8 eski eşikle val FPR 0.010 → 0.185 | Seçilen setle tekrarı (K-077) |

## 2. Bu fazda yapılan ek doğrulamalar

### 2.1 Zaman sırası

- **Yöntem:** her split segmentinde ardışık çerçevelerin zaman damgası farkı ≥ 0 olmalı (`tests/test_no_leakage.py::test_timestamps_monotonic_within_segments`).
- **Sonuç:** iki veri setinin bütün segmentlerinde geçti; azalan zaman damgası yok.
- **Ne anlama geliyor:** pencereler, ID bazındaki IAT ve yeni öznitelikler (hepsi "önceki çerçeve" tanımına dayanır) çerçeveleri gönderildikleri sırayla görüyor. Ayrıca bir sıralama işlemi uygulanmadığı için ham kayıt sırası korunuyor.

### 2.2 DLC < 8 ve sıfır dolgu

Kaynak: `results/tables/ext_dlc_profile.csv` ve `ext_dlc_by_id.csv` (`python -m analysis.ext_preproc_checks`).

| Veri seti | Split | Çerçeve | DLC < 8 | Pay | Bunların enjekte olanı |
|---|---|---:|---:|---:|---:|
| Car-Hacking | train | 9,269,396 | 163,945 | %1.77 | 0 |
| Car-Hacking | val | 2,607,204 | 30,966 | %1.19 | 0 |
| Car-Hacking | test | 2,785,127 | 32,660 | %1.17 | 0 |
| ROAD | hepsi | 27,415,983 | 0 | %0 | 0 |

- **ROAD'da DLC < 8 çerçeve yok.** Bütün yükler 8 bayt; dolgu konusu ROAD'u etkilemiyor.
- **Car-Hacking:**
  - DLC < 8 çerçevelerin hepsi normal. Enjekte çerçevelerin (DoS, Fuzzy, gear, RPM) hepsi DLC 8.
  - Train normal çerçevelerinde DLC < 8 alan 4 ID var. DLC'si değişen tek ID 0x105: 4 çerçevede DLC 6 ve 8. Bu, Fuzzy dosyasındaki normal etiketli tanı yanıtı kimliklerinden biri (K-026).
  - Diğer ID'lerde DLC sabit. Kimlik yük uzunluğunu belirlediği için dolgu baytı gerçek bir 0x00 ile karışmıyor.
- **Sonuç:** dolgu doğru yönetiliyor. Bilgi kaybı adayı yalnızca 0x105'in 4 çerçevesi; pratikte ihmal edilebilir. DLC'yi öznitelik olarak eklemek bu veride yeni bilgi taşımaz. Bu yüzden eklenmedi (K-072).

### 2.3 Normalizasyon kökeni

Kaynak: `results/tables/ext_normalization_provenance.csv`.

| Veri seti | Parametre | Uydurulan segment | Beklenen (config train) | Aynı mı | Split |
|---|---|---:|---:|---|---|
| Car-Hacking | ID sözlüğü + IAT taban/tavan (K-024, K-025) | 6 | 6 | ✔ | train |
| Car-Hacking | `id_rate` referansı (K-073) | 6 | 6 | ✔ | train |
| ROAD | ID sözlüğü + IAT taban/tavan | 14 | 14 | ✔ | train |
| ROAD | `id_rate` referansı | 10 | 10 (masquerade hariç) | ✔ | train |

Ayrıca:
- ID sözlüğü yalnızca **normal** train çerçevelerinden çıkarılıyor;
- IAT taban ve tavanı normal train çerçevelerinin 0.001/0.999 kantilleri;
- `id_rate` referansı, enjeksiyona ±5 s'den uzak normal train çerçevelerinden.

Val ve test hiçbir parametreye girmiyor. Eğitim, parametreler config ile uyuşmuyorsa reddediliyor (`check_params_current`, `check_ext_params_current`).

### 2.4 Car-Hacking zaman bloğu ayrımı: bağımsızlık değerlendirmesi

**Neden blok ayrımı:**
- Car-Hacking'de her saldırı türü tek bir dosyada; normal trafik de tek dosyada.
- Dosya düzeyinde ayrım her türü ya eğitimden ya testten tamamen düşürürdü (K-015b, K-017).

**Uygulanan ayrım** (K-017):
- **Saldırı dosyaları:** zaman sırasıyla train = 1–180., val = 183–240., test = 243–300. patlamalar. Her kesimde 2 tam patlama atılıyor; kesim patlamalar arasındaki saldırısız aralığın ortasından.
- **Normal dosya:** %60/20/20, kesimlerde 15 s atılıyor.
- **Gerçekleşen boşluklar:** DoS 15.1–15.3 s, Fuzzy 16.3–16.5 s, gear/RPM 12.9–13.0 s, normal 15 s. Hepsi ≥ 10 × IAT tavanı (≈ 1 s).

**Güvence altında olanlar:**
1. **Kopya yok:** 32 çerçevelik hiçbir dizi (ID'ler, yükler, µs farkları) iki kümede birden yok (K-023). Sınırın iki yanındaki pencereler aynı mesajları paylaşmıyor.
2. **Öznitelik sınırı aşmıyor:** IAT ve bütün yeni öznitelikler segment içinde hesaplanıyor. Bir test çerçevesinin özniteliği eğitim çerçevesine bakmıyor.
3. **Örtüşme yok:** val/test pencereleri örtüşmesiz (K-002). En uzun test penceresi (p99 0.45 s) boşluğun çok altında.

**Güvence altında olmayanlar (sınırlılık):**
1. **Aynı sürüş oturumu:** train, val ve test aynı aracın, aynı oturumunun ardışık dilimleri. Sürüş durumu, ECU sayaçları, sinyal değerlerinin dağılımı ve saldırı aracının davranışı oturum boyunca benzer. Bu yüzden test, "yeni bir kayıt" değil "aynı kaydın ilerisi" niteliğinde. Dosya düzeyindeki bağımsızlıktan zayıf.
2. **Saldırı parametreleri aynı:** her türde aynı saldırı (DoS: 0x000; spoofing: aynı hedef ID ve yük) her bölümde tekrar ediyor. Test, saldırının yeni bir örneğine genellemeyi değil, aynı saldırının sonraki patlamalarını tanımayı ölçüyor.
3. **Tavan etkisi:** bunun bir sonucu olarak CNN, CH'de val ve testte tavanda (S0, 3 tohum ortalaması: recall 0.999–1.000, AUC ≥ 0.9998; `results/tables/grid_quant.csv`). CH sonuçları ayrım gücü bakımından bilgilendirici değil; bu P6'da da kaydedildi (FINDINGS "Kapsam ve dürüstlük notları").
4. **Ölçülemeyen:** oturum içi bağımlılığın skoru ne kadar şişirdiği bu tasarımla ölçülemiyor. Bunun için başka bir oturumdan kayıt gerekir; CH'de böyle bir kayıt yok.

**Literatürle ilişki:**
- Yaygın uygulama, CH'yi rastgele çerçeve ya da pencere bölmesiyle ayırmaktır (karşılaştırma `literature_preprocessing.md`'de).
- Rastgele bölmede komşu çerçeveler (ve örtüşmeli pencerelerde aynı çerçeveler) eğitimde ve testte birlikte bulunur.
- Bizim blok ayrımımız ondan belirgin biçimde sıkı, ama ROAD'un kayıt düzeyindeki ayrımından gevşek.
- Bu nedenle: AS-1'in ve genişletmenin asıl kanıtı ROAD'dan gelmeli; CH sonuçları ikincil kalmalı.

## 3. Henüz yapılmayanlar ve yapılış biçimi

- **Öznitelik genişletmesi ve ablasyon** (K-073, K-074):
  - S1 = +bit değişimi;
  - S2 = +bayt değişimi;
  - S3 = +ID frekansı;
  - S4 = +pencere ID dağılımı.

  İki model × iki veri seti × 3 tohum. Önce yalnızca val değerlendirilir.
- **Set seçimi** (K-075): model başına, yalnızca val'den, önceden yazılmış kuralla. Seçim commit'lenmeden test değerlendirmesi yapılamıyor (kodda kapı var).
- **conv-AE teşhisi** (K-076): eşik, öznitelik, mimari/paradigma ve skor bileşimi ayrı ölçülecek. Not: S0 değerleri P5/P6'da görüldü; S0 için ayrıştırma önceden kayıtlı sayılmaz.
- **Nicemleme** (K-077): seçilen setle FP32/FP16/INT8, iki eşik kipinde, test kümesinde.
- Bütün yeni sonuçlar `docs/reports/advisor_ext_01.md`'de ayrı raporlanacak. H1–H4 hükümleri ve makale değişmeyecek (K-071).
