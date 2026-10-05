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

## K-005 — Veri kaynakları ve ortak şema

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** Plan F0 iki veri setinin ortak şemaya dönüştürülmesini istiyor. Kaynaklar ve gerçek dosya formatları P1'de incelendi (`docs/DATA.md`).
- **Karar:** Car-Hacking, HCRL sayfasındaki herkese açık Dropbox klasöründen; ROAD, Zenodo kaydından (10.5281/zenodo.10462796, MD5 doğrulamalı) indirilir (`src/data/download.py`). Ortak şema: `timestamp` float64, `can_id` int32, `dlc` int8, `data` uint8[8] (DLC < 8 ise sıfır dolgu), `label` int8, `attack_type` (enjekte çerçevede saldırı adı, aksi hâlde `normal`), `source_file`, `source_capture`. Kayıt başına bir parquet (`data/processed/<set>/<kayıt>.parquet`) ve `data/processed/index.csv`. Saldırı adları: Car-Hacking `dos, fuzzy, gear_spoofing, rpm_spoofing`; ROAD kayıt adından (`fuzzing, correlated_signal, max_speedometer, max_engine_coolant_temp, reverse_light_on, reverse_light_off` ve her birinin `_masquerade` sürümü).
- **Gerekçe:** Kayıt başına ayrı dosya, dosya/kaynak düzeyinde ayrımı (CLAUDE.md kural 1) doğal kılar. Ham zaman damgaları izlenebilirlik için korunur.
- **Etkisi:** Car-Hacking normal dosyasının son satırı kesik olduğundan atılır (988,871 çerçeve). Etiket sayıları `tests/test_loaders.py` içinde ham dosyadan bağımsız sayımla doğrulanır.

## K-006 — ROAD çerçeve düzeyi etiketleme kuralı

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** ROAD çerçeve etiketi içermez; enjeksiyon bilgisi `capture_metadata.json`'dadır. Gerçek alan adları readme'dekinden farklıdır (`injection_id`, `injection_data_str`, `injection_interval`, `modified`).
- **Karar:** Hedefli saldırılarda bir çerçeve enjektedir ⇔ `can_id == injection_id` ∧ yük `injection_data_str` desenine uyar (`X` = joker yarım bayt) ∧ kaydın ilk çerçevesine göre zamanı `injection_interval` içinde (uçlar dahil, mikro-saniye tamsayı karşılaştırması). Fuzzing'de enjektedir ⇔ yük FF×8 ∧ zaman ≥ enjeksiyon başlangıcı.
- **Gerekçe:** Veriyle doğrulandı: `injection_interval` kaydın başına görelidir (ilk–son çerçeve farkı her kayıtta `elapsed_sec` ile birebir eşit); aralık içinde hedef ID hızı tam iki katına çıkar ve hedef ID çerçevelerinin tam yarısı desene uyar; masquerade kayıtlarında aralık içinde kalan hedef ID çerçevelerinin tamamı desene uyar. FF×8 yükü 12 ambient kaydın hiçbirinde yoktur. fuzzing_attack_2 ve _3'te metadata bitişinden < 6 ms sonra birer FF×8 çerçevesi daha vardır; bu yüzden fuzzing'de üst sınır uygulanmaz. reverse_light_off kayıtlarında desene uyan meşru çerçeveler yalnızca aralık dışında bulunur (araç gerçekten o durumdayken) ve doğru biçimde normal kalır.
- **Etkisi:** Etiket sayıları ham log'dan bağımsız bir yeniden sayımla test edilir.

## K-007 — Kayıt soy ilişkisi (`source_capture`) ve ayrım birimi

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** ROAD readme'si `_masquerade` kayıtlarının, çakışan enjekte edilmemiş çerçeveleri çıkarılmış özdeş log'lar olduğunu söylüyor. Kullanıcı, türetilmiş kayıtların kaynağının belirlenmesini ve ortak şemaya eklenmesini istedi.
- **Karar:** Ortak şemaya `source_capture` alanı eklendi: türetilmiş kayıtlar kaynak kaydın kimliğini (ör. `road/max_speedometer_attack_1`), diğer tüm kayıtlar kendi kimliğini taşır. Dönüştürme sırasında 13 masquerade kaydının hepsi kaynağına karşı doğrulanır (`verify_masquerade_lineage`); sonuç `results/tables/capture_lineage.csv`'ye yazılır.
- **Gerekçe:** Veriyle doğrulandı: göreli zaman + ID + yük üzerinden masquerade çerçevelerinin %100'ü kaynak kayıtta bulunur; silinen çerçevelerin tamamı hedef ID'li, aralık içinde ve enjekte olmayan (meşru) çerçevelerdir; enjekte çerçeve sayısı iki sürümde aynıdır. Aynı enjekte çerçeveleri taşıyan iki kaydın farklı kümelere düşmesi sızıntıdır.
- **Etkisi:** F1'de ayrım birimi `source_capture`'dır (bkz. K-015c). Masquerade kayıtları fiziksel olarak kaydedilmemiş, sonradan işlenmiştir (bkz. K-014).

## K-008 — ROAD accelerator kayıtlarının etiketli kullanımdan dışlanması

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** 4 accelerator kaydında `injection_id`, `injection_data_str` ve `injection_interval` null. Readme'ye göre enjeksiyon kayıttan önce yapılmış; kayıt aracın ele geçirilmiş durumunu içerir ama enjekte çerçeve yoktur.
- **Karar:** Kayıtlar ortak şemaya dönüştürülür (tüm çerçeveler `label = 0`), `index.csv`'de `excluded = True` olarak işaretlenir ve `configs/data.yaml`'da dışlanmış olarak listelenir. Eğitim/test ve imza analizinde kullanılmaz.
- **Gerekçe:** Çerçeve düzeyinde doğru etiket yoktur. Normal veri olarak kullanmak eğitim verisini kirletir; saldırı olarak kullanmak etiket uydurmak olur.
- **Etkisi:** ROAD'da kullanılabilir saldırı kaydı 29'dur (3 fuzzing, 13 fabrication, 13 masquerade).

## K-009 — Veri temelli kaba/gizli gruplama kuralı (a priori)

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** Plan (Bölüm 1.2) saldırıları ada göre gruplar (DoS/fuzzing kaba; spoofing/malfunction/masquerade gizli). Kullanıcı, grubun saldırının mekanizmasına göre ve yalnızca trafik istatistikleriyle belirlenmesine karar verdi.
- **Karar:** Tanımlar: **coarse** = trafiğin zamanlamasını veya kimlik dağılımını belirgin değiştiren (fabrication dahil); **stealthy** = zamanlamayı ve kimlik dağılımını koruyan, imzası yalnızca yükte olan. Kural, metrikler hesaplanmadan önce `configs/eda.yaml`'a yazıldı. Herhangi biri sağlanırsa coarse:
  - C1: görülmemiş ID payı > 0.01.
  - C2: 1000 çerçevelik saldırı pencerelerinin, aynı kaydın normal dönemine göre medyan JSD'si > saldırısız trafikten 200 çekilişlik boş dağılımın 99. persentili.
  - C3: hedef ID hız oranı [1/1.5, 1.5] dışında.
  - C4: hedef ID medyan IAT oranı [1/1.5, 1.5] dışında.

  Dönemler: Car-Hacking'de < 1 s aralıklı T patlamalarının kapsadığı süre saldırı dönemi, 1 s kenar payı dışındaki geri kalan normal dönemdir; ROAD'da `injection_interval` saldırı, geri kalan normal dönemdir. Plandaki öncül grup `plan_group` alanında tutulur.
- **Gerekçe:** Model sonuçlarına bakarak grup seçmek analizi döngüsel yapar. Eşit çerçeve sayılı pencereler, JSD'nin örneklem büyüklüğüne bağımlılığını boş dağılımla aynı tutar.
- **Etkisi:** Sonuçlar K-011'de. ROAD'da ID uzayı 0xFFF'e kadar uzandığından (bkz. K-015a) analiz 4096 değerlik ID uzayı kullanır.

## K-010 — Kayıt boşlukları ve periyodiklik ölçütü (ilk EDA koşusundan sonra)

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** İlk koşuda Car-Hacking gear dosyasında 5,588 s'lik, DoS'ta 22 s'lik veri yolu sessizliği (kayıt boşluğu) bulundu. Dönem süreleri uç uca hesaplandığında çerçeve/s oranları anlamsızlaştı (gear dosyasının ortalaması ~550 çerçeve/s göründü; boşluk hariç gerçek değer ~1,830). ROAD'da std tabanlı IAT değişkenlik katsayısı (CV), ECU'ya özgü duraklamalar yüzünden şişti (106 ID'nin 0'ı CV < 0.2).
- **Karar:** `configs/eda.yaml`'a `recording_gap_s: 1.0` eklendi: veri yolunda hiç çerçeve olmayan > 1 s aralıklar dönem sürelerinden (hız hesapları) ve ID IAT istatistiklerinden çıkarılır. Periyodiklik için std tabanlı CV (eşik 0.2) korunur; yanına medyan tabanlı yayılım `(p95 − p05)/medyan` ve çerçeve ağırlıklı paylar raporlanır.
- **Gerekçe:** Boşluklar trafiğin değil kayıt sürecinin özelliğidir. Gruplama eşiklerinin hiçbiri değişmedi ve düzeltme öncesi ile sonrasında grupların hepsi aynı kaldı (C3 dışındaki metrikler medyan veya pencere tabanlı olduğundan boşluklardan etkilenmez).
- **Etkisi:** Car-Hacking: 27/27 ID CV < 0.2 (medyan CV 0.043). ROAD: CV ölçütüyle 18/106, medyan tabanlı yayılımla 81/106 ID periyodik; baskın ID'lerin medyan IAT'si tam 10 ms veya 20 ms; 0xFFF periyodik değil.

## K-011 — Gruplama sonucu ve plan ile çelişkiler

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli (kullanıcı onayı: 2026-10-05)
- **Bağlam:** K-009 kuralı 15 saldırı türüne uygulandı (`results/tables/attack_signatures.csv`).
- **Karar:** `configs/attack_groups.yaml` kesinleşti.
  - **coarse (10 tür):** CH dos, fuzzy, gear_spoofing, rpm_spoofing; ROAD fuzzing, correlated_signal, max_engine_coolant_temp, max_speedometer, reverse_light_off, reverse_light_on.
  - **stealthy (5 tür):** ROAD'un 5 masquerade türü.
- **Gerekçe:** 8 türde plan_group = stealthy iken veri coarse gösterdi:
  - Car-Hacking spoofing hedef ID hızını ×6.5–7.0 artırır, medyan IAT'yi ×0.11'e düşürür ve kimlik dağılımını boş dağılımın çok üstünde değiştirir.
  - ROAD fabrication hedef ID hızını tam ×2.0 artırır, medyan IAT'yi ×0.36–0.50'ye düşürür; kimlik dağılımı değişimi ise gürültü tabanının altında kalır (JSD 0.006–0.011 < 0.077).

  Masquerade türlerinde hız ve IAT oranları 1.00–1.14, JSD ≈ 0.006 ve görülmemiş ID yok; imza yalnızca yükte (bayt medyan kayması 8–254).
- **Etkisi:** Planın "gizli saldırılar trafiği bozmaz" sezgisi yalnızca masquerade için geçerli; spoofing ve fabrication için veride görülmedi. Grup dengesi: coarse 10 tür / 22 kayıt, stealthy 5 tür / 13 kayıt (yalnızca ROAD).

## K-012 — AS-1'in ifadesi: "gizli saldırı" mekanizmaya göre tanımlanır

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı)
- **Bağlam:** Öneri formu ve plan, AS-1'deki gizli grubu "spoofing, malfunction, masquerade" listesiyle tanımlıyor. K-011'e göre bu liste veriyle desteklenmedi.
- **Karar:** AS-1'de "gizli saldırılar" saldırının adına göre değil mekanizmasına göre tanımlanır: zamanlamayı ve kimlik dağılımını koruyan, imzası yalnızca yükte olan saldırılar. Grup ataması `configs/attack_groups.yaml`'dan okunur.
- **Gerekçe:** Ada dayalı liste, trafik yapısını belirgin bozan spoofing/fabrication saldırılarını gizli sayıyordu. Bu, AS-1'in test ettiği mekanizmayı (nicemlemenin ince yük sapmalarını bastırması) bulanıklaştırırdı.
- **Etkisi:** Öneri formundan sapmadır; danışmana **ilk haftalık raporda** bildirilecek. Makalenin Tehdit Modeli bölümü bu tanımı kullanacak.

## K-013 — AS-1 birincil analizi: eşleştirilmiş fabrication–masquerade tasarımı

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı)
- **Bağlam:** Gizli grubun her kaydı, aynı kaynak kayıttan türetilmiş bir fabrication (coarse) kaydıyla eşleşiyor (K-007): 5 tür, 13 çift.
- **Karar:** AS-1'in birincil analizi eşleştirilmiş tasarımdır: aynı `source_capture`'dan gelen fabrication (coarse) ve masquerade (stealthy) çiftlerinin ΔF1 değerleri karşılaştırılır. Grup düzeyindeki coarse–stealthy karşılaştırması (Mann-Whitney U, Cliff's delta; plan Bölüm 2.7) ikincil analiz olarak kalır.
- **Gerekçe:** İki sürüm aynı enjekte yük baytlarını taşır; aradaki tek fark zamanlama izinin (hedef ID hızının iki katına çıkması) varlığıdır. Böylece ΔF1 farkı doğrudan zamanlama ipucunun yokluğuna bağlanabilir; araç, sürüş durumu ve yük değeri gibi karıştırıcılar sabit kalır.
- **Etkisi:** F5'te eşleştirilmiş test (ör. Wilcoxon işaretli sıra) ve eşleştirilmiş etki büyüklüğü kullanılacak; ayrıntılar F5'te sabitlenecek. Çiftlerin test kümesinde birlikte bulunması gerekir (K-015c).

## K-014 — Gizli grubun tamamı sentetik türetilmiş ROAD masquerade kayıtları (sınırlılık)

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Açık sorun (makale Sınırlılıklar bölümü)
- **Bağlam:** K-011'e göre stealthy grubun 5 türünün hepsi ROAD'dan gelir ve fabrication kayıtlarından meşru çerçevelerin sonradan silinmesiyle türetilmiştir (K-007). Car-Hacking'de gizli tür yoktur.
- **Karar:** Bu durum makalenin Sınırlılıklar bölümünde açıkça yazılacak: gizli saldırı bulguları tek araca, tek veri setine ve fiziksel olarak kaydedilmemiş (sonradan işlenmiş) kayıtlara dayanır.
- **Gerekçe:** Gerçek bir masquerade saldırısında meşru ECU susturulur; sentetik silme, susturma sırasında oluşabilecek zamanlama yan etkilerini içermez. Genellenebilirlik ve AS-1'in istatistiksel gücü sınırlıdır.
- **Etkisi:** Gizli grup için veri setleri arası genelleme deneyi (F4 ek deneyi) yapılamaz.

## K-015 — F1'e devredilen açık sorunlar

- **Faz / tarih:** P1 · 2026-10-05
- **Durum:** Açık (F1'de karara bağlanacak)
- **Sorunlar:**
  - **(a)** ROAD'da 0xFFF kimliği var (çerçevelerin %6.3'ü; ID'ler 3 hex hane, muhtemelen anonimleştirilmiş). Plan Bölüm 2.2'deki ID/0x7FF normalizasyonu ROAD için 1'i aşar; normalizasyon yöntemi F1'de belirlenecek.
  - **(b)** Car-Hacking'de her saldırı türü tek dosyada; ROAD'da max_engine_coolant_temp (ve masquerade sürümü) tek kaynak kayıtta. Dosya/kaynak düzeyinde ayrım bu türleri test kümesinden düşürebilir; F1'de çözülecek.
  - **(c)** Ayrım birimi `source_capture`'dır; aynı kaynaktan türeyen fabrication ve masquerade kayıtları her zaman aynı kümede (eğitim, doğrulama veya test) kalır.
- **Etkisi:** F1'de `tests/test_no_leakage.py` (c)'yi doğrulayacak.
