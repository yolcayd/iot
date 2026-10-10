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
- **Durum:** Kapandı (P2): (a) K-020, (b) K-017 ve K-019, (c) K-018 ile karara bağlandı
- **Sorunlar:**
  - **(a)** ROAD'da 0xFFF kimliği var (çerçevelerin %6.3'ü; ID'ler 3 hex hane, muhtemelen anonimleştirilmiş). Plan Bölüm 2.2'deki ID/0x7FF normalizasyonu ROAD için 1'i aşar; normalizasyon yöntemi F1'de belirlenecek.
  - **(b)** Car-Hacking'de her saldırı türü tek dosyada; ROAD'da max_engine_coolant_temp (ve masquerade sürümü) tek kaynak kayıtta. Dosya/kaynak düzeyinde ayrım bu türleri test kümesinden düşürebilir; F1'de çözülecek.
  - **(c)** Ayrım birimi `source_capture`'dır; aynı kaynaktan türeyen fabrication ve masquerade kayıtları her zaman aynı kümede (eğitim, doğrulama veya test) kalır.
- **Etkisi:** F1'de `tests/test_no_leakage.py` (c)'yi doğrulayacak.

## K-016 — Car-Hacking saldırı dosyalarının ortak kuyruğu tek kaynak olarak sayılır

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçerli (kullanıcı onayı)
- **Bağlam:** Zamansal blok ayrımı hazırlanırken dört Car-Hacking saldırı dosyasının kuyruğu incelendi. Her dosya uzun kayıt boşluğundan sonra aynı 891,068 çerçevelik saldırısız kayıtla bitiyor: mutlak zaman damgaları (1478200727.419470 … 1478201209.058944), ID'ler ve yükler birebir aynı. Kuyruk zamanda DoS kaydının devamı (son DoS çerçevesinden 22 s sonra); normal dosyada yok.
- **Karar:**
  - Kuyruk `configs/data.yaml` → `shared_segments` altında tanımlanır, dört dosyanın her birinden kesilir ve tek bir görünüm (`car_hacking/shared_tail`, `src/data/sources.py`) olarak yalnızca bir kez kullanılır. Bu görünüm train kümesindedir.
  - P1 keşifsel analiz tabloları ve şekilleri (`data_profile.csv`, `attack_signatures.csv`, imza şekilleri, özet şekli) aynı görünüm katmanıyla yeniden üretildi. Car-Hacking saldırı dosyalarının normal dönemi artık kuyruğu içermiyor.
- **Gerekçe:** Kuyruk dosyalarla birlikte farklı kümelere giderse aynı mesajlar hem eğitimde hem testte bulunur (sızıntı). Keşifsel analizde aynı kayıt dört kez sayılıyor ve dört saldırı dosyasının normal dönemine karışıyordu; tehdit modeli ve Şekil 2 sayıları doğru olmalı.
- **Etkisi:**
  - Gruplama değişmedi: 15 türün hepsinde tetiklenen ölçütler aynı; `tests/test_attack_groups.py` geçiyor.
  - Car-Hacking imza sayıları küçük oranda değişti (P2.md'de eski/yeni tablo). Car-Hacking null q99: 0.00053 → 0.00059.
  - Saldırı oranları %12.8–16.0 → %16.7–21.2; dosya süreleri artık kayıt boşluğunu içermiyor.
  - ROAD tabloları değişmedi.
  - `tests/test_no_leakage.py` kuyruğun dört dosyada aynı olduğunu ve tek bir segmentte kullanıldığını doğrular.

## K-017 — Car-Hacking: zamansal blok ayrımı

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçerli (kullanıcı onayı)
- **Bağlam:** Car-Hacking'de her saldırı türü tek dosyada, normal trafik de tek dosyada (K-015b). Dosya düzeyinde ayrım, türleri ya eğitimden ya testten tamamen düşürür. Her saldırı dosyasında 300 patlama var (DoS ~4.6 s, Fuzzy ~5.2 s, gear/RPM ~3.4 s); patlamalar arası aralık 3.0 s.
- **Karar:**
  - Saldırı dosyası (kuyruk çıkarılmış) zaman sırasıyla bölünür: train = 1–180. patlamalar, val = 183–240, test = 243–300.
  - Her kesimde 2 tam patlama atılır. Kesimler patlama arası saldırısız aralığın zaman orta noktasına düşer.
  - Normal dosya zamana göre %60/20/20 bölünür, kesimlerde 15 s atılır.
  - Satır aralıkları `configs/data.yaml`'a açıkça yazıldı (`splits.car_hacking.blocks`). `python -m src.data.splits --propose-blocks` kuraldan aynı aralıkları yeniden üretir; testler ikisinin eşit olduğunu doğrular.
- **Gerekçe (boşluk süresi):**
  - 100 çerçevelik bir pencere medyanda 51–55 ms, p99'da en fazla 0.45 s sürer.
  - IAT özniteliği her segmentin içinde sıfırdan hesaplanır (K-021); öznitelik sınırın ötesine bakmaz.
  - Boşluk, zamansal bağımlılığa (sürüş durumu, otokorelasyon) karşı güvenlik payıdır. Kural: boşluk ≥ `min_block_gap_s` = 10 s ≥ 10 × IAT tavanı (0.999 s).
  - Gerçekleşen boşluklar: DoS 15.1–15.3 s, Fuzzy 16.3–16.5 s, gear/RPM 12.9–13.0 s, normal dosya 15 s.
  - Kesim çerçevelerinin en yakın enjekte çerçeveye uzaklığı > 1 s.
- **Etkisi:**
  - Test pencereleri dosyanın son bölümünden gelir. Train, val ve test aynı sürüş oturumunun farklı zaman dilimleridir; bu, dosya düzeyinde ayrımdan zayıf bir bağımsızlıktır ve makalede Sınırlılıklar bölümünde belirtilecek.
  - Plandan (dosya düzeyinde ayrım) bilinçli bir sapmadır; danışmana bildirilecek.

## K-018 — ROAD: source_capture düzeyinde ayrım ve fuzzing istisnası

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı)
- **Bağlam:** ROAD'da hedefli fabrication türlerinin (correlated_signal, max_speedometer, reverse_light_off/on) ve fuzzing'in her birinde 3 kaynak kayıt var. Ambient kayıt sayısı 12.
- **Karar:**
  - Ayrım birimi `source_capture`'dır (K-015c); masquerade kaydı her zaman fabrication kaynağıyla aynı kümeye düşer.
  - Önceden belirlenmiş kural: `_1` train, `_2` val, `_3` test.
  - Ambient kayıtlar çerçeve sayısına göre ~%60/20/20 dağıtıldı (train 14.54 M, val 4.91 M, test 4.70 M çerçeve). `highway_street_driving_long` ve `exercise_all_bits` train'de; `benign_anomaly` gerçekçi yanlış alarm ölçümü için test'te.
  - **Fuzzing istisnası:** `fuzzing_attack_1` train'de; `fuzzing_attack_2` ve `fuzzing_attack_3` birlikte test'te (68 test saldırı penceresi). Fuzzing için val kümesi yoktur; erken durdurma diğer türlerle yürür.
- **Gerekçe:**
  - Önceden belirlenmiş kuralla `fuzzing_attack_3` (116 enjekte çerçeve) testte yalnızca 17 saldırı penceresi bırakıyordu; bu sayıyla F1 çok gürültülü olur.
  - İstisna model sonuçlarına değil, yalnızca pozitif örnek sayısına dayanır; model çıktısı veya test skoru kullanılmadı.
  - Fuzzing, birincil AS-1 analizinin (eşleştirilmiş fabrication–masquerade, K-013) parçası değildir.
- **Etkisi:** `min_test_attack_windows` = 30 (her tür bu eşiğin üstünde; en küçükleri coolant 43, fuzzing 68). `configs/data.yaml` → `analysis_roles.no_validation_windows` fuzzing'i listeler.

## K-019 — ROAD max_engine_coolant_temp çifti yalnızca testte

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı)
- **Bağlam:** Coolant fabrication ve masquerade kayıtları tek kaynak kayıttan geliyor. Enjeksiyon kaydın 19.98–24.17 s aralığında (4.2 s, 43 enjekte çerçeve).
- **Karar:**
  - Çiftin tamamı test kümesine konur; tür eğitimde ve doğrulamada görülmez.
  - Sonuçlar AS-1 birincil analizine katılmaz; ayrı bir "eğitimde görülmemiş saldırı türü" vakası olarak raporlanır (`analysis_roles.unseen_in_training`, `excluded_from_primary_as1`).
- **Gerekçe:** Enjeksiyon aralığı içinde zamansal blok, testte yalnızca 8 saldırı penceresi bırakıyordu. Tüm çift testte olunca 43 saldırı penceresi var. Aynı mekanizma (hedef ID hızının iki katına çıkması) diğer 4 fabrication türünde eğitimde bulunuyor; LSTM otokodlayıcı zaten denetimsiz.
- **Etkisi:** 13 fabrication–masquerade çiftinin 12'si birincil AS-1 analizinde kalır; testte 4 çift (her tür için `_3`) bulunur.

## K-020 — ID normalizasyonu: veri setine özgü sabit ID uzayı

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçersiz (K-024 ile)
- **Bağlam:** Plan Bölüm 2.2 ID/0x7FF diyor. ROAD'da 0x7FF'i aşan tek ID 0xFFF (çerçevelerin %6.3'ü); Car-Hacking'de en büyük ID 0x7FF (K-015a).
- **Karar:** Car-Hacking ID/0x7FF, ROAD ID/0xFFF. Sabitler `configs/data.yaml` → `features.id_space` altındadır, veriden öğrenilmez. ID uzayını aşan bir ID hata verir.
- **Gerekçe:**
  - Vektör 10 boyutta kalır.
  - Her iki veri seti [0, 1] aralığını tam kullanır; bu, INT8 girdi nicemlemesinde ID çözünürlüğü için önemlidir. Ortak /0xFFF, Car-Hacking ID'lerini [0, 0.5]'e sıkıştırırdı.
  - Ayrı bir ikili 0xFFF bayrağı 11 boyuta çıkarırdı ve plandan sapardı.
  - ROAD'un ID alanı 3 hex hanedir (12 bit, anonimleştirilmiş).
- **Etkisi:** İki veri setindeki ID değerleri aynı ölçekte değildir; veri setleri arası genelleme deneyinde (F4) ID özniteliği zaten anonimleştirme nedeniyle karşılaştırılamaz.

## K-021 — IAT özniteliği: tanım ve tavan

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçersiz (K-025 ile)
- **Karar:**
  - IAT = aynı ID'li bir önceki çerçeveden bu yana geçen süre. Hesap segment içinde yapılır; bir ID'nin segmentteki ilk çerçevesi tavan değerini (1.0) alır.
  - Öznitelik min(IAT, tavan) / tavan.
  - Tavan, veri setinin **train segmentlerindeki normal (label 0) çerçevelerin** ID başına IAT'sinin 0.999 kantilidir. Car-Hacking 0.999111 s (7.84 M örnek), ROAD 1.007227 s (15.56 M örnek).
  - Parametreler, uydurulduğu segment listesi ve yapılandırma özetiyle `results/tables/normalization_params.json`'a yazılır.
  - `fit_iat_cap` train dışı segment verilirse hata verir.
- **Gerekçe:**
  - Segment içi hesap, bir segmentin özniteliğinin başka bir kümenin çerçevelerine bakmasını önler.
  - Saldırı çerçeveleri tavan hesabından çıkarıldı: Fuzzy'nin rastgele ID'leri p99.9'u ~60 s'ye çıkarıyordu.
- **Etkisi:** Açık sorun (P2.md): tavan ≈ 1 s olduğundan tipik IAT değerleri (10–100 ms) özniteliğin 0.01–0.1 aralığına düşer. [0, 1] aralığında 1/255 adımlı INT8 girdi nicemlemesi 10 ms'yi ≈ 2.5 adıma, 1 ms'yi < 1 adıma indirir. Bu, F3'ten önce karara bağlanmalı.

## K-022 — LSTM otokodlayıcının normal eğitim alt kümesi

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçerli (kullanıcı onayı)
- **Karar:**
  - Alt küme yalnızca train kümesinden, label 0 olan ve hiçbir çerçevesi kaydın herhangi bir enjekte çerçevesine 5 s'den yakın olmayan pencerelerden oluşur.
  - Kaynaklar: ROAD ambient (train), ROAD train fabrication kayıtlarının saldırısız dönemleri, Car-Hacking normal dosyasının train bloğu, Car-Hacking ortak kuyruğu.
  - Masquerade kayıtlarının saldırısız dönemleri, fabrication kaynağındaki çerçevelerle aynı olduğu için dahil edilmez.
  - Car-Hacking train saldırı bloklarının 3 s'lik patlama arası aralıkları ±5 s payın içinde kaldığından alt kümeye girmez.
- **Gerekçe:** Saldırıya yakın normal pencereler, saldırının hemen öncesi veya sonrasındaki geçiş etkilerini içerebilir. Kenar payı bu etkileri eğitimden uzak tutar.
- **Etkisi:**
  - Car-Hacking alt kümesi 29,685 pencere (normal dosya 11,865 + kuyruk 17,820).
  - ROAD alt kümesi 295,689 pencere (ambient 290,835 + saldırısız dönemler 4,854).
  - Anomali eşiği F2'de yalnızca bu alt kümenin skor dağılımından belirlenecek.

## K-023 — Sızıntı testinde mesaj kimliği: zaman kaymasından bağımsız 32 çerçevelik dizi

- **Faz / tarih:** P2 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:**
  - Plan F1 testi 1 "eğitim ve test pencereleri ortak mesaj (timestamp) içermiyor" diyor. ROAD zaman damgaları kayıt başına anonimleştirilmiş ve birçok kayıt aynı yuvarlak değerden başlıyor (ör. üç kayıt 1110000000.0'dan).
  - (zaman damgası, ID, yük) anahtarı ilişkisiz kayıtlar arasında tek tek, ardışık olmayan 54 tesadüfi eşleşme verdi.
  - Yeniden zaman damgalanmış kopyalar (masquerade kayıtları) mutlak zaman anahtarıyla hiç yakalanamaz.
- **Karar:**
  - Mesaj kimliği, başlattığı 32 çerçevelik dizidir: ID'ler, yükler ve çerçeveler arası µs farkları.
  - Test, hiçbir dizinin iki kümede bulunmadığını doğrular ve eşleşmeyi ham çerçevelerle teyit eder.
  - Pozitif kontrol: dedektör masquerade–fabrication çiftini ve ortak kuyruk kopyasını yakalar.
  - Ayrıca aynı dosyanın segmentleri çakışmaz.
- **Gerekçe:** Kopya bir kayıt binlerce ortak dizi üretir. Tüm kayıtlarda 8 çerçevelik dizilerle yapılan tarama, yalnızca bilinen 13 soy çiftinde (%75–99) ve DoS ile kuyruk arasında tek bir dizide eşleşme buldu; başka bir kopya kayıt yok.
- **Etkisi:** Test 1, planın ifadesinden daha sıkı bir biçimde uygulanır.

## K-024 — ID kodlaması: train sözlüğüne göre yoğun indeks

- **Faz / tarih:** P2 (ek) · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı). K-020'yi geçersiz kılar.
- **Bağlam:**
  - K-020'deki ID / ID uzayı ölçeklemesinde, tek ölçekli (per-tensor) INT8 girdi nicemlemesi bir seviyeye ≈ 8 (0x7FF/255) veya ≈ 16 (0xFFF/255) ardışık kimlik düşürür; komşu kimlikler birleşebilir.
  - Bu, yükün hangi kimlikte değiştiğini ayırt etmeyi zorlaştırır ve masquerade tespitini yapay olarak bozabilir. Ölçülen ΔF1 o zaman modelin iç nicemlemesinden değil girdi ölçeklemesinden gelir; araştırma sorusu ise birincisidir.
  - Karar model eğitiminden önce, hiçbir model sonucu görülmeden alındı.
- **Karar:**
  - Kimlik, train kümesindeki **normal (label 0) çerçevelerde** görülen kimliklerin sıralı sözlüğüne göre yoğun indekse çevrilir: sözlükteki kimlik sıra numarası 1..K'yi, sözlük dışı her kimlik ayrılmış 0 kodunu alır. Öznitelik = kod / K.
  - Sözlük yalnızca train segmentlerinden çıkarılır ve `results/tables/normalization_params.json`'a kaydedilir.
  - Sözlük: Car-Hacking K = 38, ROAD K = 106.
- **Gerekçe:**
  - K + 1 değer (CH 39, ROAD 107), asimetrik INT8'in 256 seviyesine de simetrik INT8'in [0, 1]'deki 128 seviyesine de çakışmasız sığar; testle doğrulandı.
  - 0xFFF sorunu kendiliğinden ortadan kalkar.
  - Sözlük normal çerçevelerden çıkarılır, çünkü train'in tüm çerçevelerinde Car-Hacking'de 2,048 kimlik var (Fuzzy rastgele kimlik enjekte ediyor); bu 256 seviyeye sığmaz ve saldırı çerçevelerinin temsil kuralını belirlemesine izin verir.
  - Enjeksiyonla gelen sözlük dışı kimlikler (DoS 0x000, fuzzing kimliklerinin çoğu) 0 koduna düşer; "görülmemiş kimlik" bilgisi korunur.
- **Etkisi:**
  - Car-Hacking sözlüğü normal dosyadaki 27 kimliğin yanında, Fuzzy dosyasında normal etiketli 3–149 çerçeveyle görülen 11 kimliği (0x105, 0x608, 0x6A1, 0x6AA, 0x71B, 0x7CF, 0x7D9, 0x7DA, 0x7DE, 0x7E8, 0x7E9; toplam 385 çerçeve) içerir. Bunlar veri setindeki etiketlerdir, değiştirilmedi.
  - Val/test normal çerçevelerinde train sözlüğü dışında kimlik yoktur.
  - Kimlik değerleri iki veri seti arasında karşılaştırılamaz (sözlükler farklı).

## K-025 — IAT özniteliği: log ölçeği

- **Faz / tarih:** P2 (ek) · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı). K-021'i geçersiz kılar (segment içi hesap ve yalnızca train'den uydurma kuralları korunur).
- **Bağlam:**
  - K-021'deki doğrusal min(IAT, tavan)/tavan ölçeklemesinde tavan ≈ 1 s'dir. Asimetrik INT8'de 1 ms seviye 0'a, 10 ms seviye 3'e düşer. Tipik periyotlar (10–100 ms) 3–26 seviyeye sıkışır.
  - Bu, ölçülen nicemleme kaybını girdi ölçeklemesine bağlar (K-024'teki gerekçe). Karar model eğitiminden önce alındı.
- **Karar:**
  - Öznitelik (log10(clip(IAT, taban, tavan)) − log10(taban)) / (log10(tavan) − log10(taban)), [0, 1] aralığında.
  - Taban ve tavan, train segmentlerindeki normal çerçevelerin ID başına IAT dağılımının (tüm ID'ler birlikte) 0.001 ve 0.999 kantilleridir:
    - Car-Hacking: taban 0.2382 ms, tavan 0.999111 s.
    - ROAD: taban 0.0010 ms (1 µs), tavan 1.007227 s.
  - IAT segment içinde hesaplanır; bir ID'nin segmentteki ilk çerçevesi 1.0 alır.
- **Gerekçe:** Log ölçeği, IAT'nin birkaç büyüklük mertebesine yayılan dağılımında her mertebeye eşit çözünürlük verir (`results/tables/feature_int8_levels.csv`):

  | Veri seti | Şema | Seviye/mertebe | 1 ms seviyesi | 10 ms seviyesi | 1 ms ile 10 ms arası |
  |---|---|---:|---:|---:|---:|
  | Car-Hacking | asimetrik | 70.4 | 44 | 114 | 70 seviye |
  | Car-Hacking | simetrik | 35.1 | 22 | 57 | 35 seviye |
  | ROAD | asimetrik | 42.3 | 128 | 170 | 42 seviye |
  | ROAD | simetrik | 21.1 | 64 | 85 | 21 seviye |

  Doğrusal ölçekte bu fark asimetrikte 3 seviyeydi.
- **Etkisi:**
  - ROAD tabanı 1 µs'dir: train normal IAT'lerinin 791'i tam olarak 0, 0.01 ve 0.1 kantilleri 1 µs, 1 kantili 1.02 ms. Ölçek bu yüzden 6 mertebeye yayılır. Asimetrik INT8'in 128 seviyesi (0–128) IAT'lerin ≈ %1'ini oluşturan 1 ms altı bölüme ayrılır; yine de 1 ms–10 ms arası 42 seviyedir.
  - Car-Hacking blok boşluğu kuralı (≥ 10 × tavan, K-017) değişmez; tavan aynıdır.

## K-026 — Gözlem: Car-Hacking Fuzzy dosyasında normal etiketli tanı yanıtı kimlikleri

- **Faz / tarih:** P2 (ek) · 2026-10-05
- **Durum:** Gözlem (karar değil; etiketler değiştirilmedi)
- **Bağlam:**
  - K-024'teki Car-Hacking sözlüğü, normal dosyada bulunmayan 11 kimlik içeriyor. Train kümesinde bu kimlikler yalnızca Fuzzy dosyasının train bloğundaki normal (R) etiketli çerçevelerde görülüyor (toplam 385 çerçeve).
  - Bunların ikisi OBD-II tanı yanıtı aralığında: 0x7E8 (13 çerçeve) ve 0x7E9 (16 çerçeve).
  - K-024 ve buradaki üç sapma (geçersiz kılınan kararlar K-020/K-021, sözlüğün normal çerçevelerden çıkarılması, K = 38) kullanıcı tarafından onaylandı.
- **Gözlem:**
  - 0x7E8/0x7E9 kimlikleri muhtemelen ECU'ların, fuzzing saldırısının rastgele kimliklerle gönderdiği tanı isteklerine verdiği yanıtlardır. Yani saldırının dolaylı bir sonucu olabilirler: çerçeveyi saldırgan enjekte etmemiş, ama çerçeve saldırı olmasaydı da oluşmayacaktı.
  - Bu bir varsayımdır; istek–yanıt zamanlaması incelenerek doğrulanmadı.
- **Karar:** Etiketlere dokunulmaz; veri setinin kendi etiketleri (R = normal) korunur. Bu çerçeveler normal kalır ve sözlükte yer alır.
- **Etkisi:**
  - Bu kimlikleri taşıyan pencereler "normal" etiketli olsa da saldırıyla ilişkili trafik içerebilir. Car-Hacking fuzzy sonuçları yorumlanırken (ör. yanlış alarm incelemesinde) bu göz önünde tutulmalı.
  - LSTM otokodlayıcı alt kümesini etkilemez: Fuzzy saldırı bloğunun hiçbir penceresi ±5 s payı sağlamadığı için alt kümede yer almaz (K-022).
  - Makalede veri seti sınırlılığı olarak kısaca belirtilebilir.

## K-027 — Denetimli modellerde sınıf dengesizliği: ters frekans ağırlığı, sabit 0.5 eşiği

- **Faz / tarih:** P3 · 2026-10-05
- **Durum:** Kısmen geçersiz: sınıf ağırlıkları geçerli; erken durdurma ölçütü K-034 ile, sabit 0.5 karar eşiği K-035 ile geçersiz
- **Bağlam:** Train pencerelerinde saldırı/normal oranı veri setleri arasında çok farklı: Car-Hacking ≈ 78 k saldırı / 107 k normal (≈ 1.4 normal : 1 saldırı), ROAD ≈ 7.2 k saldırı / 304 k normal (≈ 42 : 1; `split_summary.csv`). Kesin sayılar ve kullanılan ağırlık her koşunun `run_config.json`'unda.
- **Karar:**
  - 1D-CNN: `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)`, N'ler o koşunun train pencerelerinden.
  - Random Forest: `class_weight="balanced"`. XGBoost: `scale_pos_weight = N_neg / N_pos`.
  - Yeniden örnekleme (over/undersampling) yok. Karar eşiği olasılık ≥ 0.5, sabit; val'de ayarlanmaz.
  - Accuracy raporlanmaz; tür bazında precision/recall/F1/FPR raporlanır (K-029).
- **Gerekçe:**
  - Ağırlıklandırma örnek kümesini değiştirmez, ek rastgelelik getirmez ve tekrarlanan örnek üretmez; ROAD'da 42:1 oranında ağırlıksız kayıp saldırı sınıfını göz ardı etmeye yatkındır.
  - Sabit eşik ek bir seçim serbestliği yaratmaz; F3'te nicemlenmiş sürümler aynı eşikle karşılaştırılır, ΔF1 eşik ayarından değil nicemlemeden gelir.
- **Etkisi:** Erken durdurma val üzerinde aynı ağırlıklı kayıpla yapılır (`configs/models.yaml`).

## K-028 — Denetimli eğitimde masquerade kayıtlarının normal pencereleri dışlanır

- **Faz / tarih:** P3 · 2026-10-05
- **Durum:** Geçerli
- **Bağlam:** P2 açık sorunu: ROAD masquerade kayıtlarının enjeksiyon aralığı dışındaki çerçeveleri fabrication eşleriyle aynı (K-007); ikisi de train'de olduğundan denetimli model bu normal pencereleri iki kez görür.
- **Karar:** Denetimli eğitim kümesi (`src/data/dataset.py: supervised_train`) masquerade kayıtlarının label-0 pencerelerini içermez; saldırı pencereleri korunur. Val/test değerlendirmesinde bütün pencereler kullanılır.
- **Gerekçe:** Tekrar, sızıntı değil ama örnek ağırlığını bozar. Otokodlayıcı alt kümesindeki `exclude_masquerade` kuralıyla (K-022) tutarlı.
- **Etkisi:** `tests/test_train.py` gerçek pencere dizininde bu kuralı doğrular.

## K-029 — Değerlendirme tanımları: tür kümesi, recall, saldırısız FPR, çiftler

- **Faz / tarih:** P3 · 2026-10-05
- **Durum:** Geçerli (kullanıcı kararı); AS-1 birincil ölçüsü K-037 ile belirlendi
- **Karar:**
  - **Tür bazında (birincil):** t türünün değerlendirme kümesi = t'nin saldırı pencereleri + t saldırı pencerelerini içeren kayıtların normal pencereleri. Metrikler: precision, recall (TPR), F1, FPR, ROC-AUC (bütün modeller; denetimsiz model için zorunlu), pencere sayıları.
  - **Recall (TPR) ve ΔRecall:** Recall yalnızca saldırı pencerelerine bağlıdır, türler arası saldırı oranı farklarından etkilenmez. AS-1'in türler arası karşılaştırmasında F1'in yanında recall ve nicemleme kaybı ΔRecall da kullanılır (F3/F5).
  - **Saldırısız kayıtlar:** tüm saldırısız test kayıtlarında birlikte (`ambient`) ve kayıt başına FPR (ROAD `benign_anomaly` ayrı satır).
  - **Kayıt ve çift düzeyi:** kayıt başına metrikler; aynı `source_capture`'ın fabrication–masquerade çifti ayrı tabloda (K-013).
  - **Coolant** (`analysis_roles.unseen_in_training`): `role = unseen`, ayrı raporlanır; birincil havuza ve toplam metriklere katılmaz (K-019).
  - **Toplam (ikincil):** birincil havuzun bütün pencereleri.
  - ROAD fuzzing'in val kümesi yoktur (K-018): fuzzing için val satırı üretilmez.
- **Gerekçe:** Aynı kaydın normal pencereleri, saldırı ile aynı araç/sürüş koşulundaki yanlış alarmı ölçer (Car-Hacking literatüründeki dosya başına değerlendirmeyle uyumlu). Tüm test normallerini her türe eklemek, küçük türlerde precision'ı ortak FP sayısına bağlardı.
- **Etkisi:** `src/evaluate.py`, `tests/test_evaluate.py`.

## K-030 — Denetimsiz model: skor, eşik, val kullanımı ve fizibilite kararı

- **Faz / tarih:** P3 · 2026-10-05
- **Durum:** Kısmen geçersiz: skor, val erken durdurma ve val go/no-go geçerli; train p99 eşiği K-035 ile geçersiz; protokolün yürütülmesi K-033'te
- **Karar:**
  - Skor = pencere yeniden oluşturma MSE'si (100 adım × 10 öznitelik ortalaması).
  - Eşik = `normal_subset` train pencerelerinin (K-022) skorlarının 99. persentili; eğitim bittikten sonra en iyi ağırlıklarla hesaplanır, `meta.json` ve `run_config.json`'a yazılır. `fit_threshold` train dışı skorları reddeder.
  - Val kümesinin normal pencereleri (label 0, masquerade kaydı değil, ±5 s payı sağlayan) yalnızca erken durdurma kaybı içindir.
  - **Fizibilite (go/no-go):** veri seti başına **val** ROC-AUC (birincil havuz). < 0.6 ise bir kez ayar (gizli 128, lr 3e-4, en fazla 60 epoch; `configs/models.yaml → ae_feasibility`), yine val'de; yine < 0.6 ise 1D evrişimli otokodlayıcı eğitilir ve val'de karşılaştırılır. Test ROC-AUC yalnızca raporlanır.
- **Gerekçe:** CLAUDE.md "test setinde model seçimi" yasağı; karar test üzerinde verilirse model ailesi seçimi teste dayanır. Val saldırı etiketleri yalnızca bu tek go/no-go kararında kullanılır, eşikte asla.
- **Etkisi:** Görev isteminin "doğrulama yalnızca erken durdurma içindir" ifadesi bu tek karar için genişletildi.

## K-031 — Hesaplama ortamı, iş parçacığı sınırı ve determinizm

- **Faz / tarih:** P3 · 2026-10-05
- **Durum:** Geçerli (iş parçacığı kuralı kullanıcı kararı)
- **Bağlam:** Aynı makinede bir DRL projesi CPU kullanıyor.
- **Karar:**
  - Derin modeller ve XGBoost GPU'da (CUDA). Bir veri setinin öznitelik matrisi GPU'ya bir kez yüklenir, pencereler batch başına GPU'da toplanır (CPU yükü düşük).
  - CPU iş parçacığı sayısı `configs/models.yaml → runtime.threads` (varsayılan 4) tek parametresidir; `--threads` ile değiştirilir (DRL çalışmazken 12). torch, sklearn, XGBoost ve BLAS ortam değişkenleri bu değeri kullanır.
  - `run_id = <model>_<veri seti>_s<tohum>_<sha8>`; sha8 hiperparametreler + veri seti + tohum + öznitelik/ayrım config özeti + pencere boyundan türetilir. İş parçacığı, cihaz ve git commit run_id'ye girmez; `run_config.json`'a yazılır.
  - Tohumlama: `random`, `numpy`, `torch` (CPU + CUDA), karıştırma için ayrı tohumlu `torch.Generator`, `cudnn.deterministic = True`, `benchmark = False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG`; RF/XGBoost `random_state`.
  - Kirli çalışma ağacında (`results/raw` hariç) eğitim `--allow-dirty` olmadan reddedilir (CLAUDE.md kural 7).
- **Gerekçe:** İş parçacığı sayısı sonucu değil yalnızca süreyi etkiler (RF `random_state` ile n_jobs'tan bağımsızdır); bu yüzden run kimliğine girmez ama kayıt altındadır.
- **Etkisi:** Benchmark (gecikme) iş parçacığı sayısı bundan bağımsızdır ve F3'te sabitlenecek.

## K-032 — Açık sorun (F3): nicemlenmiş otokodlayıcının anomali eşiği

- **Faz / tarih:** P3 · 2026-10-05
- **Durum:** Kapandı (K-036 ile; nicemlenmiş sonuçlar görülmeden)
- **Soru:** Otokodlayıcının eşiği nicemlenmiş sürümlerde (Q1 float16, Q2 INT8) nasıl belirlenecek?
  - (a) float32 modelin eşiği aynen kullanılır;
  - (b) her nicemlenmiş sürüm için eşik, o sürümün **train normal alt kümesi** skorlarından aynı persentille yeniden hesaplanır.
- **Neden önemli:** Nicemleme yeniden oluşturma hatası dağılımını kaydırabilir. (a)'da kayma doğrudan FPR/recall değişimi olarak ΔF1'e yansır; (b)'de eşik kaymayı telafi eder ve ΔF1 yalnızca sıralama (ayırt edicilik) kaybını ölçer. Seçim ΔF1'i doğrudan etkiler ve AS-3 (denetimli vs denetimsiz dayanıklılık) karşılaştırmasının anlamını değiştirir.
- **Kısıt:** Her iki seçenekte de eşik yalnızca train verisinden gelir (CLAUDE.md kural 3). P3 kodu bunu kısıtlamaz: eşik `meta.json`'da saklanır, `fit_threshold` yeniden kullanılabilir.

## K-033 — Denetimsiz model fizibilite protokolünün yürütülmesi ve önceden kayıtlı çıkış kuralı

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı). Ayar denemesi ve conv-AE sonuçları görülmeden yazıldı.
- **Bağlam:** Baseline LSTM-AE (gizli 64, lr 1e-3) val ROC-AUC'si (birincil havuz): Car-Hacking 0.807 (geçti), ROAD 0.394 (eşik 0.6; geçemedi). K-030 protokolü bir kez ayar denemesi, sonra 1D evrişimli otokodlayıcı öngörüyor.
- **Karar:**
  - Aday sırası: (1) baseline LSTM-AE, (2) ayar denemesi LSTM-AE (gizli 128, lr 3e-4, en fazla 60 epoch; `configs/models.yaml → ae_feasibility.retry`), (3) conv-AE (`models.conv_ae`, hiperparametreler a priori). Her aday her iki veri setinde, tohum 0 ile eğitilir.
  - (2) ve (3) tek arka plan işinde peş peşe eğitilir (`python -m src.run_grid --stage ae_feasibility`). Bu aşamada yalnızca **val** değerlendirmesi yapılır; test ancak model seçildikten sonra değerlendirilir.
  - Karar ölçüsü: veri seti başına val ROC-AUC (birincil havuz) ≥ 0.6. Eşikten bağımsızdır, K-035'teki eşik kuralından etkilenmez.
  - **İnceleme sırası:** (3)'ün val sonucuna yalnızca (2) en az bir veri setinde başarısız olursa bakılır (`--ae-check` bu sırayı uygular ve gerekmediği sürece conv-AE satırını göstermez).
  - **Seçim kuralı** (işlem tanımı: AUC'si en yüksek olan değil, sıradaki ilk geçen aday): seçilen otokodlayıcı, sırada iki veri setinde de geçen ilk adaydır. Böyle bir aday yoksa (üç aday da incelenmiştir) her veri seti için sırada o veri setinde geçen ilk aday kullanılır. Böylece val'de yalnızca geç/kal kararı verilir, en iyi model aranmaz. Bir aday iki veri setinde de geçerse sonraki adaylar incelenmez.
  - **Çıkış kuralı:** (2) ve (3) ROAD'da val ROC-AUC < 0.6 kalırsa otokodlayıcı ROAD analizlerinden çıkarılır. AS-3 (denetimli vs denetimsiz) yalnızca Car-Hacking'de değerlendirilir. Bu durum bir bulgu olarak raporlanır: bu girdi temsiliyle yeniden oluşturma tabanlı modeller ROAD'un hedefli saldırılarını ayıramadı. Car-Hacking'de sırada Car-Hacking'de geçen ilk aday kullanılır.
- **Gerekçe:** Çıkış kuralı ve seçim kuralı sonuçlar görülmeden sabitlenir. Bu, denetimsiz ailede val üzerinde arama yapılmasını ve sonuca göre kural uydurulmasını önler.
- **Etkisi:** `configs/grid.yaml → ae_feasibility`, `src/run_grid.py --ae-check`. Sonuç P3 faz günlüğüne ve README'ye yazılacak.

## K-034 — CNN erken durdurma ölçütü: val average precision

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı). K-027'nin erken durdurma kısmını geçersiz kılar.
- **Bağlam:** ROAD baseline CNN'de ağırlıklı val kaybı (pos_weight ≈ 41) 1. epoch'u seçti; bu model her pencereyi saldırı diyor (val/test FPR ≈ 1.0). Analiz (`analysis/road_cnn_gap.py`, `results/tables/road_cnn_gap_*.csv`):
  - 2. epoch'tan itibaren val kaybının %100'ü pozitiflerden geliyor.
  - En kötü 10 pozitifin payı %0.4–0.8 (gürültü değil, yaygın ve yüksek güvenli hata).
  - Val AUC 0.79 → 0.94 ve AP 0.33 → 0.81 artarken kayıp 9.8'den 22–37'ye çıkıyor.
  - Kayıp sıralamayı değil olasılık ölçeğini izliyor.
- **Karar:** 1D-CNN'de erken durdurma ve en iyi epoch seçimi, val pencerelerinin tamamı üzerinde **average precision**'ı (logit skoru, eşikten bağımsız) en büyütmeye göre yapılır. Patience 5 ve en fazla 30 epoch aynı. Kayıp fonksiyonu, pos_weight ve diğer hiperparametreler değişmez. Ağırlıklı val kaybı yine kaydedilir.
- **Gerekçe:** AP, sınıf dengesizliğinde pozitif sınıfın sıralama kalitesini ölçer ve kalibrasyondan bağımsızdır. Karar eşiği zaten ayrı bir kuralla belirlenir (K-035).
- **Etkisi:** CNN her iki veri setinde yeniden eğitilir (tutarlılık). `cnn1d.early_stopping: val_average_precision` hiperparametre olarak run_id'ye girer.

## K-035 — Karar eşiği: val normal pencerelerinde FPR = %1 (tüm modeller)

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Kısmen geçersiz: kalibrasyon kümesi K-038 ile değişti (kural, FPR hedefi ve skorlar geçerli). K-027'nin sabit 0.5 eşiğini ve K-030'daki train p99 eşiğini geçersiz kılar.
- **Karar:**
  - Her model için (CNN, RF, XGB, otokodlayıcılar) karar eşiği, **val kümesinin bütün label-0 pencerelerinin** skor dağılımından belirlenir.
  - Eşik = bu skorların 0.99 kantili (`numpy.quantile(..., method="higher")`); karar kuralı skor > eşik. Böylece val normal pencerelerinde gerçekleşen FPR ≤ %1 olur; bağlarda daha düşük olabilir.
  - Eşik, gerçekleşen val FPR'ı ve kullanılan normal pencere sayısı `meta.json` ve `run_config.json`'a yazılır.
  - Saldırı etiketleri kullanılmaz. Test hiçbir zaman kullanılmaz (`calibrate_threshold` val dışı skorları reddeder).
  - Skorlar: CNN logit, XGBoost ham marj (`output_margin`), RF olasılık, otokodlayıcı yeniden oluşturma MSE'si. Hepsi tekdüze skorlar; olasılık doygunluğundan (float32'de 1.0) doğan bağları önler.
  - "Val normal pencereleri" kelimesi kelimesine uygulanır: val'deki bütün label-0 pencereleri. Saldırı kayıtlarının normal dönemleri ve masquerade kayıtlarının normal pencereleri de dahildir. Otokodlayıcının erken durdurma alt kümesi (±5 s payı, K-030) bu amaçla kullanılmaz.
- **Gerekçe:** Sabit 0.5 eşiği ROAD'da kayıtlar arasında aktarılmadı: CNN'de FPR ≈ 1.0; RF correlated_signal'da test AUC 0.955 ama F1 0.091. Yanlış alarm oranına kalibre edilmiş eşik, modelleri aynı işletim noktasında karşılaştırır ve araç içi IDS'in pratik kısıtına (yanlış alarm bütçesi) karşılık gelir.
- **Etkisi:** Bütün baseline koşuları yeni kuralla yeniden üretilir (eşik artık hiperparametre değil; run_id'ler değişir). P3 v1 koşuları (`results/raw/*_{634b5aea,328acff9,d4eb6502,8588b4c9,e89b45da,d0365e45,f1c8183c,d06e60c4}`) tarihçe olarak tutulur, tablolarda kullanılmaz.

## K-036 — K-032'nin kapanışı: nicemlenmiş sürümlerin eşiği

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı; nicemlenmiş sonuçlar görülmeden). K-032'yi kapatır.
- **Karar** (denetimli ve denetimsiz bütün modeller):
  - **(a) Birincil:** her nicemlenmiş sürüm (Q1, Q2) için eşik aynı kuralla (K-035: val normal pencereleri, FPR %1) o sürümün kendi val skorlarından yeniden hesaplanır.
  - **(b) İkincil:** float32 (Q0) eşiği nicemlenmiş sürüme aynen uygulanarak kalibrasyon kayması ölçülür (val/test FPR ve recall değişimi).
  - **(c)** Eşikten bağımsız ΔROC-AUC ve ΔAP raporlanır.
- **Gerekçe:** (a) her sürümü aynı yanlış alarm bütçesinde karşılaştırır; ΔRecall ayırt edicilik kaybını ölçer. (b) dağıtımda eşik yeniden kalibre edilmezse ne olacağını gösterir. (c) eşik seçiminden bağımsız bir kontroldür.
- **Etkisi:** F3'te `src/quantize.py` ve değerlendirme, her sürüm için val skorlarını da üretir.

## K-037 — AS-1 birincil ölçüsü: kalibre eşikte recall (%1 FPR'da TPR) ve ΔRecall

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı). K-029'daki "F1 birincil" ifadesini AS-1 için günceller. Birincil eşleştirilmiş testin birim kümesi K-041'deki taban kuralıyla sınırlanır; açıklayıcı ROC ölçüsü K-040'ta.
- **Karar:** AS-1'in birincil ölçüsü, K-035'e göre kalibre edilmiş eşikteki saldırı türü bazında recall (val normal FPR = %1'de TPR) ve onun nicemleme kaybı ΔRecall = Recall(Q0) − Recall(Qk). F1, ROC-AUC ve AP ikincildir ve raporlanır.
- **Gerekçe:**
  - Recall yalnızca saldırı pencerelerine bağlıdır; türler arasındaki saldırı oranı farklarından etkilenmez (ROAD tür kümelerinde 103–973 normal pencere).
  - Eşikler sabit bir yanlış alarm oranına kalibre edildiği için recall modeller arasında adil karşılaştırılır.
- **Etkisi:** Plan Bölüm 2.6/2.7'den (ΔF1 ana değişken) sapmadır; danışmana **haftalık raporda** bildirilecek. F5 istatistiksel testleri ΔRecall üzerinde birincil, ΔF1 üzerinde ikincil olarak yürütülür.

## K-038 — K-035'e ek: kalibrasyon kümesi temiz val normal alt kümesidir

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı; v2 koşularından ve hiçbir eşik sonucundan önce). K-035'in "val'deki bütün label-0 pencereleri" maddesini geçersiz kılar; K-035'in geri kalanı geçerli.
- **Karar:** Bütün modellerin (CNN, RF, XGB, otokodlayıcılar) karar eşiği, otokodlayıcının erken durdurmada kullandığı temiz val normal alt kümesinden belirlenir (`src/data/dataset.py: calibration_windows` = `ae_val`):
  - label 0;
  - masquerade kayıtlarının pencereleri hariç;
  - kaydın herhangi bir enjekte çerçevesine ±5 s'den yakın çerçeve içeren pencereler hariç (K-022 ile aynı pay).

  Eşik bu pencerelerin skorlarının 0.99 kantilidir, FPR hedefi %1, kural skor > eşik.
- **Gerekçe:**
  - Masquerade kayıtlarının normal pencereleri fabrication kayıtlarının kopyasıdır; dahil edilirse aynı trafik iki kez sayılır.
  - Saldırıya yakın normal etiketli pencereler saldırının etkisini (geçiş, gecikme, yanıt trafiği) taşıyabilir; eşik "temiz" normal trafiğe kalibre edilmelidir.
- **Yapısal sonuç** (sonuç değil, ayrım tanımından): Car-Hacking'de patlamalar arası aralık 3 s < ±5 s olduğundan saldırı dosyalarının val normal pencerelerinin hiçbiri bu kümeye girmez. Car-Hacking kalibrasyon kümesi yalnızca normal dosyanın val bloğudur (1,684 pencere; ROAD 50,337). %1 FPR eşiği CH'de ≈ 17 pencerelik kuyruğa dayanır. Saldırı dosyalarının normal pencerelerindeki FPR testte ayrıca raporlanır.
- **Etkisi:** Val ve testte raporlanan FPR, kalibrasyon kümesinden farklı pencerelerde ölçülür; %1'den sapabilir ve sapma raporlanır.

## K-039 — K-033 protokolünün sonucu: iki veri setinde 1D evrişimli otokodlayıcı

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli. K-033'ün önceden kayıtlı kuralının uygulanmasıdır; yeni bir seçim serbestliği içermez.
- **Girdi** (val ROC-AUC, birincil havuz; `results/tables/ae_feasibility.csv`; eşik 0.6):

  | Aday | Car-Hacking | ROAD |
  |---|---:|---:|
  | (1) baseline LSTM-AE (gizli 64, lr 1e-3) | 0.807 ✔ | 0.395 ✘ |
  | (2) ayar denemesi LSTM-AE (gizli 128, lr 3e-4, ≤ 60 epoch) | 0.809 ✔ | 0.372 ✘ |
  | (3) conv-AE | 0.999 ✔ | 0.722 ✔ |

- **Karar:** (1) ROAD'da geçemedi, (2) incelendi; (2) ROAD'da geçemedi, (3) incelendi. (3) iki veri setinde de geçen ilk aday; **iki veri setinde de denetimsiz model conv-AE'dir** (M2-alt). Çıkış kuralı tetiklenmedi; AS-3 iki veri setinde değerlendirilir.
- **Gözlem:** Baseline LSTM-AE v2 koşusu v1 val AUC'lerini birebir yeniden üretti (0.807448 / 0.394506). LSTM-AE'nin iki sürümünde de eğitim kaybı ≈ 0.073–0.08'de düzleşti; ayar denemesi bunu değiştirmedi.
- **Etkisi:** Test değerlendirmesi yalnızca seçimden sonra, seçilen conv-AE koşuları için yapıldı. `baseline_f1.csv`'de LSTM-AE satırları `rejected_k033`, conv-AE satırları `selected_k033` olarak işaretli. F3–F5'te M2 yerine conv-AE kullanılır. Plandan (M2 = LSTM-AE) bu sapma README'de ve faz günlüğünde belirtilir; danışmana haftalık raporda bildirilecek.

## K-040 — Açıklayıcı ölçü: test ROC eğrisinde FPR = %1'deki TPR

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı)
- **Bağlam:** v2 baseline'da kalibre eşik (K-035, K-038) teste farklı aktarıldı. Temiz saldırısız test kayıtlarında gerçekleşen FPR denetimli modellerde ≈ %1 (CNN 1.2%, RF 1.0%, XGB 0.9%; ROAD), conv-AE'de ise ≈ %0.1 (ROAD 0.11%, CH 0.18%). conv-AE testte hedeften on kat sıkı bir işletim noktasında çalışıyor; "kalibre recall" modeller arasında aynı noktayı temsil etmiyor.
- **Karar:**
  - Her değerlendirme kümesi için `roc_tpr_at_fpr_0.01` raporlanır: o kümenin kendi ROC eğrisinde FPR ≤ %1 olan en yüksek TPR.
  - **Yalnızca açıklayıcıdır**, ROC-AUC ile aynı statüdedir: eşik, model, epoch veya aday seçiminde kullanılmaz.
  - Kalibrasyon kuralı değişmez. Birincil ölçü K-037'deki kalibre recall olarak kalır; aktarım farkı bulgu olarak raporlanır.
- **Gerekçe:** Eşik kalibrasyonunun aktarımından bağımsız, sabit test FPR'ında bir ayırt edicilik ölçüsü verir. Testin kendi etiketlerini kullandığı için karar ölçüsü olamaz.
- **Etkisi:** `src/evaluate.py: roc_tpr_at_fpr`. Normal pencere sayısı küçük kümelerde (ROAD tür kümeleri 103–973) gürültülüdür; FPR ≤ %1, 103 normalde en fazla 1 yanlış alarm demektir.

## K-041 — AS-1 birincil eşleştirilmiş testinde taban kuralı (yalnızca Q0'a dayalı)

- **Faz / tarih:** P3 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı; nicemlenmiş hiçbir sonuç görülmeden)
- **Karar:**
  - Birim: (model, fabrication–masquerade çifti) (K-013; birincil havuz, coolant hariç K-019).
  - Bir birimde Q0'daki (float32) **fabrication veya masquerade recall'undan biri < 0.10** ise (taban etkisi), o birim AS-1'in birincil eşleştirilmiş testinden çıkarılır. Tablolarda ve şekillerde raporlanmaya devam eder, çıkarılma nedeniyle işaretlenir.
  - Kural yalnızca Q0 değerlerine dayanır. Q1/Q2 sonuçları ve ΔRecall kullanılmaz. Aynı birim kümesi her nicemleme düzeyinde (Q1, Q2) kullanılır.
  - **İşlem tanımı** (kullanıcı onayına açık): "Q0 recall" = test kümesinde kalibre eşikteki (K-035/K-038) recall'un **3 tohum üzerinden ortalaması**.
  - Car-Hacking'de çift yoktur; denetimli modellerin Car-Hacking'de tavanda olması (recall ≈ 1, ΔRecall'un aşağı yönde payı yok) bilinen sınırlılık olarak raporlanır. Grup düzeyindeki ikincil analiz (K-013) bu kuraldan etkilenmez.
- **Gerekçe:** Recall'u Q0'da sıfıra yakın bir türde nicemleme kaybı ölçülemez; ΔRecall tabanla sınırlıdır ve "kayıp yok" sonucu yanıltıcı olur. Eşik ve uygulama Q0'a bağlı olduğundan kural ΔRecall'a bakılarak ayarlanamaz.
- **Etkisi (yalnızca bilgi; P3 tek tohumu, kesin değil):** P3 seed-0 Q0 recall'larıyla 16 birimden 7'si eşiği geçiyor (CNN 3, RF 1, XGB 2, conv-AE 1). Birincil testin istatistiksel gücü sınırlı olabilir; kesin birim listesi F4'teki 3 tohumla belirlenir.

## K-042 — Nicemleme araç zinciri, kapsamı ve INT8 kalibrasyon kümesi

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli (plan kullanıcı tarafından onaylandı; nicemlenmiş hiçbir sonuç görülmeden)
- **Karar:**
  - Nicemlenen modeller: CNN ve conv-AE (K-039). RF ve XGBoost nicemlenmez. Yalnızca boyut ve gecikme ölçümüne kendi biçimleriyle girerler: `model.joblib`, `model.ubj`.
  - Araç: ONNX Runtime 1.30. Q0, PyTorch modelinin TorchScript dışa aktarıcısıyla üretilmiş float32 ONNX hâlidir (opset 17).
  - Q2 için `quantize_static` QDQ biçiminde kullanılır:
    - aktivasyon QUInt8, asimetrik;
    - ağırlık QInt8, simetrik, kanal başına;
    - kalibrasyon yöntemi MinMax.
  - Nicemlenecek işlem türleri izin listesiyle verilir: Conv, ConvTranspose, Gemm, MatMul, MaxPool, Relu, Transpose.
  - **INT8 kalibrasyon kümesi:** modelin kendi eğitim seçiminden (CNN: `supervised_train`; conv-AE: `ae_train`) koşu tohumuyla rastgele seçilen 1000 penceredir.
    - `check_calibration_selection`, train dışı pencereyi ya da 500'den az pencereyi reddeder (CLAUDE.md kural 4).
    - Seçimin özeti (`digest`) `quant_meta.json`'a yazılır.
  - Pilot kapsamı: CNN ve conv-AE, iki veri seti, tohum 0. İstemin "bir baseline model" asgarisinin üstündedir; kullanıcı onayladı.
  - Ayarlar `configs/quantization.yaml` dosyasındadır.
- **Gerekçe:**
  - QDQ biçimi, hangi katmanların nicemlendiğini grafikte açıkça gösterir.
  - Asimetrik uint8 aktivasyon, [0, 1] girdisine 256 seviye verir (P2 açık sorunu, K-024/K-025).
  - Kalibrasyon kümesinin modelin kendi eğitim dağılımından gelmesi, otokodlayıcıyı saldırı penceresi olmadan kalibre eder.
- **Etkisi:** Q2 dosya boyutu Q0'ın 0.376–0.379'u. Tensör verisinin oranı 0.28: int8 ağırlıklar, int32 bias'lar, ölçekler ve sıfır noktaları. Kalan fark QDQ düğümlerinin grafik ek yüküdür (≈ 11.5 KB). Model küçük (≈ 26–28 bin parametre) olduğu için bu ek yük oranı belirgin biçimde etkiliyor.

## K-043 — Q1 (float16): CPU sağlayıcısında desteklenmiyor; doğruluk fp16 benzetimiyle

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı; Q1 doğruluk sonuçları görülmeden)
- **Bağlam:**
  - Q1, Q0 grafiğinin `onnxruntime.transformers.float16` ile dönüştürülmüş hâlidir (`keep_io_types=True`).
  - ONNX Runtime profilinden okunan gerçek çekirdek girdi türleri şunu gösteriyor: x86 CPU sağlayıcısının bu modellerde hiçbir ağır katman için fp16 çekirdeği yok.
    - CNN: 0/5 ağır katman fp16 çalışıyor (Conv ×3, Gemm ×2). Bütün çekirdeklerde fp16 payı 2/10 (Transpose, Squeeze).
    - conv-AE: 0/6 (Conv ×4, ConvTranspose ×2). Bütün çekirdeklerde 1/13.
    - Grafik optimizasyonu kapalıyken de sonuç aynı.
  - ORT her ağır katmanın önüne fp32'ye dönüştürme (Cast) ekliyor. CPU'da çalışan "Q1" dosyası bu yüzden fp16'ya yuvarlanmış ağırlıklarla fp32 hesap yapıyor. Gecikmesi de fp16 hesabı değil, Cast ek yükünü ölçer.
  - Kullanıcının önceden koyduğu kural: ağır katmanların yarısından fazlası float32'de kalıyorsa Q1 olduğu gibi raporlanmaz; iş durur ve kullanıcıya sorulur. Kural kodda `Q1CoverageError` ile uygulanır.
- **Karar** (kullanıcının seçtiği seçenek (a)):
  - Q1 "CPU'da desteklenmiyor" (`cpu_status = cpu_unsupported`) olarak işaretlenir.
  - **Doğruluk:** gerçek float16 aritmetiğiyle ölçülür. PyTorch modelinin `.half()` kopyası kullanılır: ağırlıklar, girdi ve aktivasyonlar float16. Hesap CUDA'da (RTX 4050) yapılır; cuDNN/cuBLAS ara toplamaları genellikle fp32 tutar.
  - **Boyut:** fp16 ONNX dosyasının boyutudur (fp16 depolama).
  - **CPU gecikmesi ve belleği:** ölçülmez; tablolarda `cpu_unsupported` olarak boş kalır.
  - Kural `configs/quantization.yaml → q1.low_cpu_coverage_action: simulate_fp16` ile kayıtlıdır. Bu ayar olmadan kod durur.
  - Kapsam raporu (`fp16_coverage`: işlem türü bazında, ağır katman payı) her koşunun `quant_meta.json` dosyasında.
- **Gerekçe:**
  - H3 (float16 en dengeli nokta) float16'nın sayısal etkisini ölçmeyi gerektirir. Ancak CPU'daki dosya bu etkiyi temsil etmez; gecikme ölçümü de yanıltıcı olur.
  - Benzetim sayısal etkiyi verir. Gecikme–boyut ödünleşimi (AS-2) Q1 için CPU'da tanımsızdır ve bu bir bulgu olarak raporlanır.
- **Etkisi:** Q1'in gecikme kazancı bu projede ölçülmez; fp16 çekirdekli bir donanım gerekir (ör. ARM, GPU). Bu, sınırlılık olarak yazılacak.

## K-044 — Q2 grafiği: skor grafikte, seçmeli nicemleme, INT8 çekirdek ile benzetimin ayrımı, girdi nicemlemesi

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli
- **Karar:**
  - **Skor grafikte:** ONNX grafiği eşiklenen skoru üretir: CNN'de logit, conv-AE'de pencere yeniden oluşturma MSE'si. Böylece benchmark skor dahil tüm çıkarımı ölçer.
  - **Constant/Identity düzeltmesi:** TorchScript dışa aktarıcısının `Constant` ve paylaşılan `Identity` bias'ları initializer'a çevrilir. Aksi hâlde nicemleyici o katmanın bias'ını tanımaz ve Conv katmanı float kalır; sentetik testte bu durum görüldü.
  - **Seçmeli (hybrid) nicemleme:**
    - İzin listesinde olmayan ya da nicemlenemeyen düğümler float kalır ve `quant_meta.json → levels.q2.execution.float_nodes` alanına türleriyle yazılır.
    - Pilotta float kalan düğümler: CNN'de Transpose, ReduceMean, Squeeze; conv-AE'de Transpose ×2 ve skor kuyruğu (Sigmoid, Sub, Pow, ReduceMean).
    - LSTM katmanları nicemlenmez ve listelenir (sentetik testle doğrulandı; LSTM-AE projede nicemlenmiyor).
  - **Çalışma biçiminin ayrımı:** Her ağır katman için CPU sağlayıcısının gerçekte INT8 çekirdeği mi (QLinearConv/QGemm) kullandığı, yoksa DQ → float işlem → Q biçiminde benzetim mi yaptığı profilden okunup kaydedilir.
    - CNN: 5/5 INT8 çekirdek.
    - conv-AE: 4/6 INT8 çekirdek. İki ConvTranspose, ORT CPU'da INT8 çekirdeği olmadığı için benzetimle çalışıyor. Sayısal olarak int8 ağırlık ve aktivasyonlar korunuyor; yalnızca aritmetik (ve gecikme) float. P3 açık sorun 7c'nin yanıtı budur.
  - **Girdi nicemlemesi (P2 açık sorunu):** Q2 girdisinin uint8 nicemleyicisi okunup doğrulanır: ölçek 1/255, sıfır noktası 0, temsil aralığı [0, 1], yani asimetrik ve 256 seviye.
    - Doğrulanan: bayt ızgarası (k/255) 256 değer → 256 seviye; ID ızgarası (k/K) K+1 değer → K+1 seviye (çakışmasız); kalibrasyon girdisi aralığı [0, 1].
    - Simetrik şema ya da ID/bayt çakışması `InputQuantizationError` ile durdurur.
    - IAT sürekli bir öznitelik; kalibrasyon penceresinde 55–256 seviyeye iner (P2'deki seviye analiziyle uyumlu, kayıp beklenen).
- **Gerekçe:** Plan Bölüm 2.4 nicemlenemeyen katmanların ve seçmeli nicemlemenin raporlanmasını istiyor. Ayrıca benzetimle çalışan katman, doğruluk etkisini korurken gecikme kazancı sağlamaz; bu ayrım AS-2 için gerekli.

## K-045 — K-036'nın uygulanması: referans Q0-ONNX (CPU), iki eşik kipi, Δ = Q0 − Qk

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli
- **Karar:**
  - Her sürüm şu kümeleri puanlar: temiz val normalleri (K-038), val ve test. Q0 ve Q2 ONNX Runtime CPU'da (4 iş parçacığı), Q1 K-043'e göre PyTorch fp16'da puanlanır.
  - **(a) Birincil (`recalibrated`):** her sürümün eşiği kendi temiz val normal skorlarından K-035/K-038 kuralıyla (FPR %1) yeniden hesaplanır.
  - **(b) Kalibrasyon kayması (`q0_threshold`):** Q0-ONNX eşiği Q1/Q2'ye aynen uygulanır. Val/test FPR ve recall değişimi raporlanır.
  - **(c) Eşikten bağımsız ölçüler:** ΔROC-AUC, ΔAP ve test FPR %1'deki TPR değişimi (K-040).
  - **Referans:** Q0, P3'ün PyTorch/GPU değerlendirmesi değil, **Q0-ONNX'in CPU'daki değerlendirmesidir**. Böylece bütün Δ değerleri aynı hatta hesaplanır.
  - P3 ile tutarlılık kaydedilir (`q0_consistency`):
    - eşik göreli farkı ≤ 1.8e-4;
    - test kararlarında 0–2 değişen pencere (59,621 pencerede);
    - tür bazında recall farkı ≤ 0.0147. Bu üst değer, ROAD conv-AE'de az pencereli bir türde eşiğe çok yakın tek bir pencerenin GPU/CPU sayısal farkıyla yön değiştirmesinden geliyor.
  - **Δ tanımı:** Q0 − Qk; pozitif değer kayıp demektir (K-037).
  - Çıktılar: `results/raw/<run>/quant/metrics.csv` ve `results/tables/quant_pilot.csv`. Pilot tablosu `attack_type`, `attack_free` ve `pooled` satırlarını içerir. Kayıt düzeyindeki satırlar koşu klasöründe.

## K-046 — Benchmark protokolü ve tekrarlanabilirlik ölçütü

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Kısmen geçersiz: RF iş parçacığı maddesi K-048 ile (RF artık n_jobs = 1); ölçüm sayısı (1000) ve ısınma (100) maddeleri K-049 ile (10.000 ölçüm; en az 100 çıkarım ve 1 s ısınma; çekirdek sabitlemesi; güç modu). Toleranslar (%5 / %10) ve geri kalanı geçerli.
- **Karar:**
  - **Gecikme:** batch = 1, pencere başına. 100 ısınma koşusu atılır, ardından 1000 ölçüm yapılır (`perf_counter_ns`). Raporlananlar: ortalama, medyan, p95, p99, std. 1000'den az tekrar reddedilir.
  - **Girdi:** sabit tohumla seçilmiş 1100 val penceresi (`data/processed/bench/<veri seti>.npy`); etiket kullanılmaz, test kümesine dokunulmaz.
  - **İş parçacığı:**
    - Ana makine profili 4 (kullanıcı kararı).
    - ORT: intra-op = T, inter-op = 1, sıralı yürütme, `CPUExecutionProvider`.
    - OMP/MKL/OpenBLAS = T.
    - XGBoost: `n_jobs = T`, CPU.
    - RF: `n_jobs = T`. Batch = 1'de bu, joblib iş dağıtımının ek yükünü içerir (deneme koşusunda ≈ 36 ms). Bu, kütüphanenin bu ayarla davranışıdır, düzeltilmedi.
  - RF/XGB gecikmesine pencere özet özniteliklerinin hesabı dahildir.
  - **Bellek:** Her hedef ayrı bir alt süreçte ölçülür. Raporlananlar:
    - tepe RSS: Windows'ta peak working set, Linux'ta `ru_maxrss`;
    - tepe RSS'in, kütüphaneler yüklendikten sonra ama model yüklenmeden önce ölçülen değere göre farkı.
  - **Künye:** CPU adı ve özellik bayrakları, fiziksel/mantıksal çekirdek, affinity, toplam bellek, cgroup sınırları, ORT sürümü ve kullanılan sağlayıcı, iş parçacıkları, pil/güç planı, git commit, kütüphane sürümleri. Her sonuç JSON'una yazılır.
  - **Tekrarlanabilirlik:** iki ayrı oturum (`--session 1/2`) yapılır. Her hedefte medyan göreli farkı ≤ %5 ve p95 göreli farkı ≤ %10 ise tutarlı sayılır (göreli fark = |a − b| / ortalama). Çıktı `results/tables/benchmark_repeatability_<profil>.csv`.
  - **Q1:** K-043 gereği ölçülmez.
- **Gerekçe:** CLAUDE.md kural 8; plan F3 kabul kriteri ("aynı model iki kez ölçüldüğünde tutarlı"). Toleranslar ölçümden önce sabitlendi.
- **Etkisi:** Ölçümler 2026-10-06'da çalıştırılmadı (makinede DRL deneyi vardı); kullanıcı makine boşken çalıştıracak. O zamana kadar kabul kriterinin bu maddesi beklemede.

## K-047 — Kısıtlı ölçüm ortamı: Docker konteyneri (2 CPU, 2 GB) ve ana makine yaklaşımı

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Kısmen geçersiz: konteynerde ölçüm K-051 ile kapsam dışı. Konteyner tanımı (depoda kalıyor) ve ana makine yaklaşımı maddeleri geçerli.
- **Bağlam:** Raspberry Pi yok. Docker Desktop kurulu, ancak motor 2026-10-06'da kapalı.
- **Karar:**
  - **Konteyner profili:** `docker/benchmark.Dockerfile` (python:3.13-slim; yalnızca numpy, onnxruntime, scikit-learn, xgboost, psutil, PyYAML; `requirements.txt` sürümleri) ve `docker/run_benchmark.ps1`.
    - Çalıştırma: `docker run --cpus=2 --memory=2g --memory-swap=2g`, iş parçacığı 2.
    - Depo salt okunur, `results/` yazılabilir bağlanır. Commit ve kirli ağaç bilgisi ortam değişkeniyle geçirilir.
    - Benchmark modülü PyTorch'a ve veri hattına ihtiyaç duymaz (testle doğrulandı).
  - **Ana makine profili:** 4 iş parçacığı; CPU kotası ya da bellek sınırı yok. Bu bir **yaklaşımdır** ve her sonuçta `profile_note` alanıyla belirtilir.
  - Plan F4'e uygun olarak, iki profilde de birincil raporlama Q0'a göre oranlardır.
- **Gerekçe:** Plan F3/F4'ün kısıtlı ortam yönergesi. Kullanıcı, iş parçacığı sayısının 4 ile sınırlanmasını istedi (ana makine profili).

## K-048 — Benchmark: RF tek iş parçacığıyla tahmin; sklearn uyarı yarışı ve zamanlanan döngü koruması

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli. K-046'nın RF iş parçacığı maddesini (`n_jobs = T`) geçersiz kılar; K-046'nın geri kalanı geçerli.
- **Bağlam** (ana makine profili, session 1, kullanıcı tarafından çalıştırıldı):
  - RF ROAD ölçümü sırasında scikit-learn şu uyarıyı 164,442 kez bastı: "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` …" (`sklearn/utils/parallel.py:144`).
  - Uyarı, zamanlanan döngünün içinde, `predict_proba` çağrısında basıldı.
  - Gecikme serisi: ilk 506 çıkarım ≈ 23 ms; 507. çıkarımdan sonra 590–1250 ms, kalan ≈ 594 çıkarımın her birinde ≈ 277 uyarı. Medyan 577.8 ms, p95 1268.7 ms. Sonuç kirli.
  - Diğer 14 hedef ayrı alt süreçlerde koştu. Loglarında hiç uyarı satırı yok ve gecikme serileri temiz (RF CH dahil, medyan 23.1 ms).
- **Kök neden:**
  - sklearn'ün `_FuncWrapper.__call__` fonksiyonu, `Parallel.__call__` sırasında yakalanan uyarı filtre listesi boşsa uyarı basar ve her görevde `warnings.catch_warnings()` + `resetwarnings()` çağırır.
  - RF tahmini `n_jobs > 1` iken threading backend'iyle birden fazla işçi iş parçacığında koşar. `catch_warnings` iş parçacığı güvenli değildir: eşzamanlı giriş ve çıkışlar süreç çapındaki `warnings.filters` listesini başka (boşaltılmış) bir listeyle kalıcı olarak değiştirebilir.
  - Bundan sonra her ağaç görevi uyarı basar. Filtre listesi boş olduğu için uyarı her seferinde yeniden gösterilir.
  - **Yeniden üretme** (ayrı süreç, RF ROAD, `n_jobs = 4`, 1100 çıkarım): bir koşuda filtre listesi 11 girdiden 3'e değişti, diğerinde değişmedi. Yarış olasılıksal ve ağaç işinin uzunluğuyla artıyor. Session 1'deki boşalma bunun bir örneği.
  - Yalnızca mesaja özgü bir filtre yeterli değil, çünkü yarış o filtreyi de siliyor (testle gösterildi).
- **Karar:**
  - Benchmark'ta RF tahmini **`n_jobs = 1`** ile yapılır. İşçi iş parçacığı olmadığı için yarış oluşamaz. Bu, iki profilde de (ana makine ve konteyner) geçerlidir.
  - Diğer modellerin iş parçacığı ayarları değişmez: ORT intra-op = T, XGBoost `n_jobs = T`.
  - Yalnızca bu mesaja özgü ve yalnızca RF ölçümünde bir `ignore` filtresi eklenir: `UserWarning`, mesaj düzenli ifadesi `SKLEARN_DELAYED_WARNING`. Genel bir uyarı bastırması yapılmaz.
  - **Zamanlanan döngü koruması (bütün hedefler):** ısınma sonrası ölçülen çıkarımlarda herhangi bir uyarı gösterilirse ya da süreç çapındaki uyarı filtreleri döngü boyunca değişirse ölçüm `MeasurementInvalid` ile reddedilir ve sonuç yazılmaz. Uyarılar yine de basılır ve sayılır; sayılar sonuçta `warnings_guard` alanında kayıtlıdır.
  - **Session 1'in RF sonuçları geçersiz sayılır** (`invalid` alanıyla işaretlendi; dosyalar silinmedi):
    - RF ROAD: kirli.
    - RF CH: kirli değil, ancak `n_jobs = 4` ile ölçüldüğü için yeni protokolle karşılaştırılamaz.
  - Geçersiz sonuçlar bütün okuyucularca (`--compare`, `--table`, pilot tablosu) yok sayılır. Yeniden ölçümden önce `.invalid-<zaman>.json` adıyla arşivlenir.
  - Session 1'in diğer 10 sonucu geçerli kalır.
- **Gerekçe:**
  - Kök nedeni (iş parçacıklarının eşzamanlı `catch_warnings` çağrısı) ortadan kaldıran tek ayar, RF'nin tek iş parçacığıyla çalışmasıdır. sklearn sürümü sabit olduğu (K-001) ve RF modelleri bu sürümle kaydedildiği için sürüm yükseltme seçenek değil.
  - Batch = 1'de RF'nin ağaçlar arası paralelliği zaten joblib dağıtım ek yükü taşıyordu (K-046, deneme koşusunda ≈ 36 ms). Tek iş parçacığı, 4 iş parçacığı bütçesinin içinde kalır.
- **Etkisi:**
  - RF gecikmesi artık tek iş parçacıklı tahmini ölçer. Karşılaştırmalarda RF'nin 4 değil 1 iş parçacığı kullandığı belirtilir.
  - Session 1'de iki RF hedefi yeniden ölçülmelidir (`--only rf_ --force`).

## K-049 — Benchmark protokolü v2: 10.000 ölçüm, süreye bağlı ısınma, performans çekirdeğine sabitleme, güç modu; önceden kayıtlı raporlama kuralı

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı; bu protokolle hiçbir ölçüm yapılmadan yazıldı). K-046'nın ölçüm sayısı ve ısınma maddelerini geçersiz kılar. K-046'nın toleransları (%5 / %10) ve K-048 (RF `n_jobs = 1`, uyarı koruması) geçerli.
- **Bağlam** (ana makine, eski protokol: 1000 ölçüm, 100 ısınma; sabitleme yok; güç modu "Dengeli", plan "Office"):
  - Session 1 ile 2'nin karşılaştırmasında 12 hedeften 8'i tutarsız çıktı (`results/tables/benchmark_repeatability_host_k048.csv`).
  - **XGB ROAD session 1** (medyan 841 µs; session 2'de 468 µs): uyarı selinin bittiği saniyede ölçüldü (18:33:09). Bütün seri boyunca yüksek; dış yük altında olduğu düşünülüyor (konsol çizimi), kanıtlanmadı.
  - **RF CH session 1 yeniden ölçümü** (18:50): 100'lük blokların üçünde ~21 ms; 314 ölçüm medyanın iki katını aşıyor. Session 2'de yok. Kaynak belirsiz.
  - **Derin modeller:**
    - Tutarsız 4 Car-Hacking çifti uyarı selinden **önce** ölçüldü; sel bu tutarsızlığı açıklamıyor.
    - Oturum içi örnekleme hatası (bootstrap) medyanda %0.1–1.1, p95'te %0.5–6. Oturumlar arası fark medyanda %6.6–9.9, p95'te %10–49. Fark örnekleme hatası değil, sistematik.
    - Bir derin model ölçümünün tamamı 50–130 ms, 100 ısınma koşusu 4–25 ms sürüyor.
    - 100'lük blok medyanlarında kademeli seviye sıçramaları var (ör. CNN ROAD Q0 session 1: 66 → 48 → 40 → 50 µs). Bu desen, hibrit CPU'da (i7-12650H: 6 performans + 4 verimlilik çekirdeği) iş parçacıklarının çekirdekler arasında taşınmasıyla ya da frekans değişimiyle uyumlu; doğrulanmadı.
- **Karar** (`configs/quantization.yaml → benchmark`, `protocol: K-049`):
  1. **Ölçüm sayısı:** her hedefte 10.000; iki oturumda da. Girdiler döngüsel kullanılır (1100 val penceresi).
  2. **Isınma:** en az 100 çıkarım *ve* en az 1 saniye; istatistiklerin dışında (kural 8). Gerçekleşen ısınma sayısı ve süresi sonuca yazılır.
  3. **Çekirdek sabitlemesi** (yalnızca ana makine profili):
     - Süreç, model yüklenmeden önce, 4 performans çekirdeğinin her birinden en küçük indeksli mantıksal işlemciye sabitlenir.
     - Performans çekirdeği = Windows `GetSystemCpuSetInformation` ile en yüksek verimlilik sınıfı.
     - İşletim sistemi kesmelerinin çoğunu alan çekirdek 0 atlanır. Bu makinede çekirdekler 2, 4, 6, 8; mantıksal işlemciler [2, 4, 6, 8].
     - Sabitleme uygulanamazsa ölçüm yapılmaz.
     - ORT iş parçacığı sayısı 4 olarak kalır. RF tek iş parçacığıyla çalışır (K-048).
  4. **Güç modu:** Windows güç modu "En iyi performans" (`PowerGetEffectiveOverlayScheme` = `ded574b5-…`) değilse ana makine ölçümü başlamaz. Güç modu ve güç planı her sonuçta kayıtlı.
  5. **Tolerans değişmez:** medyan göreli farkı ≤ %5, p95 göreli farkı ≤ %10.
  6. **Eski sonuçlar:** K-046/K-048 protokolüyle alınmış bütün ana makine sonuçları (session 1 ve 2, 24 dosya) `invalid` olarak işaretlendi; silinmedi.
     - Her sonuç `protocol` alanı taşır. Başka protokolün sonuçları okuyucularca yok sayılır ve yeniden ölçümden önce arşivlenir.
     - Eski tablolar `benchmark_host_k048.csv` ve `benchmark_repeatability_host_k048.csv` adıyla saklandı.
  7. **Önceden kayıtlı raporlama kuralı:** Bu protokolle yapılan iki oturumdan sonra bazı hedefler yine tolerans dışında kalırsa **yeniden ölçüm yapılmaz**. Bu durumda:
     - iki oturumun değerleri birlikte raporlanır;
     - tutarsız hedefler açıkça işaretlenir;
     - makalede gecikme karşılaştırmaları ağırlıklı olarak **aynı oturum içindeki oranlara** (ör. Q2/Q0) dayandırılır.

     `--compare` çıktısı bu kuralı hatırlatır.
- **Gerekçe:**
  - Gözlenen değişkenlik süreçten sürece ve zaman penceresinden pencereye sistematik.
  - Ölçüm sayısını artırmak pencereyi derin modellerde 0.5–1.3 saniyeye uzatır; tek başına ise kaynağı kontrol etmez.
  - Süreye bağlı ısınma, frekans ve önbelleklerin oturmasını sağlar. Çekirdek sabitlemesi, performans ve verimlilik çekirdekleri arasındaki taşınmayı kaldırır. Güç modu, frekans politikasını iki oturumda sabitler.
  - Kural 7, ölçümden önce kaydedildi. Böylece tutarlılık kontrolü geçene kadar yeniden ölçme yapılamaz.
- **Etkisi:**
  - Çekirdek sabitlemesi bir ECU'nun koşullarını temsil etmekten çok ölçümün tekrarlanabilirliği içindir; makalede ve sınırlılıklarda belirtilecek.
  - Konteyner profili değişmez: sabitleme yok; 10.000 ölçüm ve 1 saniyelik ısınma ona da uygulanır.
  - Oturum başına süre ≈ 5 dk. Bunun ≈ 3 dk'sı iki RF hedefi (10.000 × ~9 ms).

## K-050 — K-049 kural 7'nin uygulanması: ana makine gecikme sonuçlarının raporlanma biçimi

- **Faz / tarih:** P4 · 2026-10-06
- **Durum:** Geçerli (kullanıcı onayı, 2026-10-06).
  - Kural 7'nin uygulanması önceden kayıtlıdır (K-049).
  - Madde 4'teki **yorum kısıtı**, ana makine sonuçları görüldükten **sonra** yazıldı. Bu kısıt iddiaları güçlendirmiyor, kısıtlıyor:
    - belirsizlik olarak tek oturum ya da oturum içi güven aralığı yerine daha geniş olan iki oturum aralığı kullanılıyor;
    - yalnızca iki oturumda aynı yönde olan medyan oranları için yönlü ifade kullanılıyor;
    - p95 oranlarından hiçbir sonuç çıkarılmıyor;
    - mutlak gecikmeler yalnızca büyüklük mertebesi olarak veriliyor.

    Sonuca bakılarak gevşetilen ya da lehte seçilen bir ölçüt yok. Kullanıcı bu gerekçeyle onayladı.
- **Bağlam:** K-049 protokolüyle iki ana makine oturumu yapıldı (21:19–21:27; 24 sonucun hepsi K-049 koşullarında). Sonuçlar:
  - **Tutarlılık:** 12 hedeften 3'ü tutarlı, 9'u tolerans dışı (`results/tables/benchmark_repeatability_host.csv`).
  - **Medyan göreli farkı:** en büyük %15.5 (CNN ROAD Q0).
  - **p95 göreli farkı:** en büyük %54 (conv-AE CH Q0).
  - **Oturum içi örnekleme hatası:** medyanda ≤ %0.2 (bootstrap). Farklar sistematik; ölçüm sayısından bağımsız.
  - **Oturum içi Q2/Q0 oranları** (medyan; oturum 1 / 2):
    - CNN CH 0.70 / 0.74; CNN ROAD 0.75 / 0.64;
    - conv-AE CH 1.08 / 1.05; conv-AE ROAD 1.05 / 1.01.
    - Oranların oturumlar arası göreli farkı %2.5–16; oturum içi bootstrap aralıkları (genişlik ≤ 0.004) bu farkı kapsamıyor.
  - **p95 oranları:** conv-AE CH'de yön değiştiriyor (1.48 / 0.84).
- **Karar:**
  1. **Yeniden ölçüm yapılmaz** (K-049 kural 7).
  2. **İki oturum birlikte raporlanır** ve tutarsız hedefler işaretlenir:
     - `results/tables/benchmark_ratios_host.csv`: oturum içi Qk/Q0 oranları, iki oturum, yön kararlılığı, tutarlılık bayrakları.
     - `quant_pilot.csv`: `latency_median_ms_s1/s2`, `latency_p95_ms_s1/s2`, `latency_consistent`.
  3. **Gecikme karşılaştırmaları oturum içi oranlara dayanır.** Raporlanan değer, iki oturumun oran aralığıdır (min–max). Tek oturumun değeri ya da oturum içi güven aralığı belirsizlik ölçüsü olarak kullanılmaz.
  4. **Yorum kısıtı** (kullanıcı onayına açık):
     - Yalnızca **medyan oranının yönü iki oturumda da aynıysa** yönlü bir sonuç ifade edilir. Bu, dört çiftin dördünde sağlanıyor.
     - **p95 oranları yorumlanmaz**, yalnızca tabloda verilir; bir çiftte yön değiştiriyor ve göreli farkları %14–56.
     - Mutlak gecikmeler yalnızca **büyüklük mertebesi** olarak verilir: derin modeller ~30–115 µs; XGB ~0.5 ms; RF ~9–10 ms.
- **Gerekçe:**
  - Kural 7 oranları "ağırlıklı olarak" kullanmayı öngörüyor. Ancak veriler, oranların da oturumdan oturuma %16'ya kadar değiştiğini gösteriyor.
  - Bu nedenle tek bir oran değeri ya da oturum içi güven aralığı, gerçek belirsizliği eksik gösterir. İki oturumun aralığı ve yalnızca yönlü ifade, verinin desteklediği en güçlü iddiadır.
- **Etkisi:**
  - AS-2'nin gecikme boyutu ana makine profilinde nitel kalır: "CNN'de INT8 medyan gecikmeyi %25–36 azaltıyor; conv-AE'de azaltmıyor (Q2/Q0 1.01–1.08)."
  - conv-AE sonucu K-044 ile uyumlu: iki ConvTranspose katmanı CPU'da QDQ benzetimiyle (float) çalışıyor. Bu bir açıklama önerisidir; ayrı bir deneyle doğrulanmadı.
  - Bellek ve boyut ölçümleri iki oturumda tutarlı (tepe bellek farkı ≤ %4); bunlar mutlak değer olarak raporlanabilir.
  - P4 kabul kriteri "benchmark iki kez çalıştırıldığında tutarlı" **sağlanmadı**. P4'ün bu durumda kapatılıp kapatılmayacağı kullanıcı kararıdır.

## K-051 — Konteyner profilinde gecikme ölçümü kapsam dışı

- **Faz / tarih:** P4 (kapanış) · 2026-10-06
- **Durum:** Geçerli (kullanıcı kararı). K-047'nin konteynerde ölçüm yapılması öngörüsünü ve plan F3/F4'ün "Docker konteynerinde tekrarla" adımını geçersiz kılar. K-047'nin konteyner tanımı ve ana makine yaklaşımı maddeleri geçerli.
- **Bağlam:**
  - Ana makinede, iş parçacığı sabitlemesi, performans çekirdeklerine sabitleme, "En iyi performans" güç modu, ≥ 1 s ısınma ve 10.000 ölçümle bile iki oturum 12 hedeften yalnızca 3'ünde tutarlı çıktı (K-049, K-050).
  - Derin modellerin gecikmesi 30–115 µs düzeyinde. Oturumlar arası medyan farkı %16'ya, p95 farkı %54'e kadar çıkıyor.
- **Karar:**
  - Konteyner profilinde (2 CPU, 2 GB) gecikme ölçümü yapılmaz.
  - `docker/benchmark.Dockerfile`, `docker/requirements-benchmark.txt`, `docker/run_benchmark.ps1` ve yapılandırmadaki `container` profili depoda kalır. İmaj hiç derlenmedi, betik hiç çalıştırılmadı; bu da belirtilir.
  - Makalede sınırlılık olarak yazılır: kaynak kısıtlı uç cihaz ne gerçek donanımda (Raspberry Pi / ECU) ne de konteyner kısıtıyla ölçüldü. Gecikme sonuçları tek bir dizüstü bilgisayarın ana makine profilinden gelir ve oturumlar arası değişkenlikleri K-050'ye göre raporlanır.
- **Gerekçe:** Ana makinede bile mikrosaniye düzeyinde tekrarlanabilirlik sağlanamadı. Konteyner (CPU kotası ve sanallaştırma katmanı) ek değişkenlik getirir, ek bilgi getirmez.
- **Etkisi:**
  - AS-2'nin gecikme boyutu yalnızca ana makine profiline dayanır ve nitel/yönlü kalır (K-050).
  - Bellek ve boyut ölçümleri bu karardan etkilenmez.

## K-052 — Gecikme tohum başına ölçülmez; P4 tohum-0 ölçümleri kullanılır

- **Faz / tarih:** P5 · 2026-10-07
- **Durum:** Geçerli (kullanıcı kararı; ızgaranın hiçbir sonucu görülmeden)
- **Karar:**
  - Tam ızgarada (P5) benchmark adımı yoktur. Gecikme ve tepe bellek için P4'teki tohum-0 ana makine ölçümleri (K-049 protokolü, K-050 raporlama kuralı) kullanılır: `quant_pilot.csv`, `benchmark_host.csv`, `benchmark_ratios_host.csv`.
  - Model dosya boyutu her tohum için `quant_meta.json`'dan raporlanır (ücretsiz, ölçüm değil).
  - Konteyner profili K-051 gereği kapsam dışı kalır.
- **Gerekçe:**
  - Gecikme mimariye ve nicemleme düzeyine bağlıdır, tohuma değil. Tohumlar aynı grafiği, aynı tensör şekillerini ve aynı çekirdekleri kullanır; yalnızca ağırlık değerleri farklıdır. Q2'de float kalan düğümler ve INT8 çekirdek / QDQ benzetim ayrımı (K-044) de grafik yapısına bağlıdır.
  - Ana makinede oturumlar arası değişkenlik (K-050: medyanda %16'ya kadar) olası bir tohum etkisinden büyük. Tohum başına ölçüm yeni bilgi getirmez, yalnızca oturum gürültüsü ekler.
- **Etkisi:**
  - `configs/grid.yaml → full.latency: reuse_seed0`. `grid_quant.csv`'nin `latency` sütunu bu kaynağa işaret eder.
  - Makalede gecikme tohum-0 modellerine dayanır; bu durum yöntem bölümünde yazılır.
  - CLAUDE.md kural 8 (≥ 1000 tekrar, ısınma hariç) P4 ölçümlerinde sağlanmıştır (10.000 ölçüm, ≥ 1 s ısınma).

## K-053 — P5 tam ızgarasının kapsamı ve yürütülmesi

- **Faz / tarih:** P5 · 2026-10-07
- **Durum:** Geçerli (kapsam kullanıcı kararı; yürütme ayrıntıları bu fazda)
- **Karar:**
  - **Izgara:** {cnn1d, conv_ae (K-039), xgb} × {car_hacking, road} × tohum {0, 1, 2} = 18 eğitim koşusu (`configs/grid.yaml → full`).
    - Derin modeller (12 koşu): Q0/Q1/Q2 ve K-036/K-045 değerlendirmesi. Q1 doğruluğu GPU'da fp16 (K-043).
    - XGBoost nicemlenmez (K-042), yalnızca val/test değerlendirmesi.
  - **RF ızgarada yok:** M0 referansı olarak XGBoost seçildi. RF yalnızca P3/P4'teki tohum-0 sonuçlarıyla referans olarak kalır; 3 tohumlu analizlere girmez.
  - **Tohum 0 yeniden kullanılır:** run_id hiperparametre, veri özeti, tohum ve pencere boyundan türetildiği için (K-031) P3/P4'ün tohum-0 koşuları aynı run_id'yi taşır ve atlanır; yeniden eğitilmez ve yeniden nicemlenmez. Bu koşular kendi commit'lerini kayıtlı taşır (`30d9762`, `8fe34e4`; nicemleme `a341053`). Kural 7 koşu başına sağlanır.
  - **Yürütme** (`src/run_grid.py`):
    - İş birimi bir eğitim koşusudur. Adımlar: eğitim → val/test değerlendirmesi → nicemleme → K-036 değerlendirmesi. Çıktısı olan adım atlanır; yarım derin eğitim checkpoint'ten devam eder.
    - Her iş ayrı alt süreçte koşar, en fazla `--workers` eşzamanlı. Çöken iş loglanır; ızgara durmaz.
    - CPU iş parçacığı toplamı 4 (DRL deneyi aynı makinede): varsayılan 1 işçi × 4 iş parçacığı.
    - Paralel çalışma sonucu değiştirmez: her süreç kendi tohumlamasını ve deterministik ayarlarını yapar (K-031).
  - **K-041 birim listesi** ızgaradan sonra Q0 test recall'unun 3 tohum ortalamasından hesaplanır (`grid_k041_units.csv`). Derin modellerde Q0 = Q0-ONNX (K-045), XGBoost'ta yerel model.
  - **Veri setleri arası genelleme deneyi kapsam dışı** (kullanıcı kararı). Izgaradan sonra zaman kalırsa ayrıca karar verilecek.
- **Gerekçe:** Plan F4 ve CLAUDE.md kural 5 (en az 3 tohum). Tohum-0'ın yeniden eğitilmesi yalnızca aynı sonucu yeniden üretirdi (P3'te bit düzeyinde tekrar üretilebilirlik doğrulandı).

## K-054 — conv-AE `max_epochs` değiştirilmez (P3 açık sorunu 2)

- **Faz / tarih:** P5 · 2026-10-07
- **Durum:** Geçerli (kullanıcı kararı, 2026-10-07; 1 ve 2. tohumların hiçbir sonucu görülmeden)
- **Bağlam:** P3'te conv-AE iki veri setinde de en iyi epoch'u `max_epochs` = 30'da buldu; val kaybı hâlâ düşüyordu. P3 bunu "F4 öncesi karar; değiştirilecekse sonuçlara bakmadan" olarak bıraktı.
- **Karar:** `max_epochs` 30'da kalır; conv-AE hiperparametreleri değişmez.
- **Gerekçe:**
  - Hiperparametreler a priori sabittir (`configs/models.yaml`). Tohum-0 test sonuçları P3/P4'te görüldü; şimdi değiştirmek sonuç görüldükten sonra yapılan bir ayar olurdu.
  - Değişiklik run_id'yi değiştirir; tohum-0'ın yeniden eğitilmesi gerekir (K-053'e aykırı).
  - Seçim protokolü (K-033) conv-AE'yi bu ayarla seçti; seçilen model değişmemeli.
- **Etkisi:** Makalede sınırlılık: conv-AE eğitim bütçesinin sınırında; daha uzun eğitim yeniden oluşturma hatasını düşürebilirdi. Bu, nicemleme kaybının (ΔRecall) ölçümünü geçersiz kılmaz, çünkü Q0 ile Qk aynı ağırlıklardan türetilir.

## K-055 — XGBoost tohumdan bağımsız: üç tohum aynı modeli veriyor (bulgu ve raporlama)

- **Faz / tarih:** P5 · 2026-10-07
- **Durum:** Geçerli. Açık bıraktığı soru (hiperparametre değişikliği) K-056 ile kapandı: XGBoost'a dokunulmaz.
- **Bulgu:** XGBoost'un tohum 0, 1 ve 2 koşuları bit düzeyinde aynı modeli verdi.
  - Test skorlarının en büyük farkı 0.0; en iyi iterasyon aynı (CH 107, ROAD 42).
  - Neden: `subsample = 1.0` ve `colsample_bytree = 1.0` (`configs/models.yaml`, P3'ten beri a priori). Satır ve sütun örneklemesi olmadığı için `random_state` hiçbir rastgele seçimi etkilemiyor; GPU `hist` algoritması bu veride deterministik.
- **Karar:**
  - Hiperparametreler değiştirilmez. Örnekleme oranlarını şimdi 1'in altına çekmek, sonuçlar görüldükten sonra yapılmış bir ayar olurdu.
  - XGBoost için 3 tohum raporlanır (kural 5 biçimsel olarak sağlanıyor); standart sapma 0'dır ve tablolarda "deterministik, tohumdan bağımsız" diye işaretlenir. Bu, eğitim değişkenliğinin olmadığı anlamına gelir, ölçüm belirsizliğinin olmadığı anlamına gelmez.
  - Tohumlar arası değişkenlik yorumları (ör. AS-3, AS-1'in tohum bazlı analizleri) yalnızca derin modeller için yapılır; XGBoost satırları bu karşılaştırmada etkin olarak tek gözlemdir.
- **Etkisi:**
  - K-041 birim listesinde XGBoost'un "3 tohum ortalaması" tek koşunun değeridir.
  - Makalede yöntem bölümünde belirtilir.
  - F5'te XGBoost için tohum bazlı istatistik (ör. tohumları bağımsız gözlem sayan testler) kullanılmaz.

## K-056 — P5 kapanışında kullanıcı kararları: XGBoost deterministik kalır; AS-1 tasarımı P6'da; genelleme kapsam dışı

- **Faz / tarih:** P5 (kapanış) · 2026-10-07
- **Durum:** Geçerli (kullanıcı kararı). K-055'in açık sorusunu kapatır; K-053'ün genelleme maddesini teyit eder.
- **Karar:**
  1. **XGBoost'a dokunulmaz** (K-055). Hiperparametreleri (`subsample = colsample_bytree = 1.0`) değişmez, yeniden eğitilmez.
     - XGBoost nicemlenmeyen referans modeldir (K-042). Projenin ana değişkeni olan nicemleme kaybını (ΔRecall) taşımaz. Bu yüzden tohumlar arası eğitim değişkenliğinin olmaması ana analizi etkilemez.
     - Raporlamada XGBoost'un deterministik olduğu (üç tohumun aynı modeli verdiği, std = 0) açıkça yazılır.
  2. **AS-1'in test tasarımı P6'nın (plan F5) planında konuşulacak.** Birincil testin gücü sınırlı: K-041 listesinde 12 birimden 5'i kalıyor (`grid_k041_units.csv`).
     - P6'da şu da açıklanacak: XGBoost nicemlenmediği için ΔRecall'u yoktur. `grid_k041_units.csv`, K-041'in taban kuralını ızgaradaki her modele uyguladığı için XGBoost birimlerini de listeliyor (`in_primary_test = True` iki birim).
     - Bu birimlerin ΔRecall'a dayalı birincil teste nasıl girebileceği (ya da giremeyeceği) P6 planında netleştirilecek. Tabloda bir değişiklik yapılmadı; sonuç elle düzeltilmez.
  3. **Veri setleri arası genelleme deneyi kapsam dışı** (K-053 geçerli).
- **Etkisi:** P5 açık sorunları 1 ve 2'nin durumu `docs/phase-logs/P5.md`'de güncellendi. P6 planı bu iki maddeyi ele alacak.

## K-057 — AS-1 tasarımı (P6): keşifsel eşleştirilmiş analiz, birimler ve XGBoost'un durumu

- **Faz / tarih:** P6 · 2026-10-10
- **Durum:** Geçerli (kullanıcı onayı, P6 planı; P6 analiz kodu çalıştırılmadan yazıldı). K-013'ün "eşleştirilmiş test (ör. Wilcoxon)" niyetini değiştirir; K-013'ün birim tanımı, K-037 (ΔRecall birincil) ve K-041 (taban kuralı) geçerli.
- **Bağlam:** K-041'in kesin birim listesinde (P5, `grid_k041_units.csv`) 12 birimden 5'i kalıyor: CNN 3, XGB 2. XGBoost nicemlenmez (K-042), bu yüzden Qk'si ve ΔRecall'u yoktur. Tablo, taban kuralını (yalnızca Q0'a bakar) ızgaradaki her modele uyguladığı için XGB birimlerini `in_primary_test = True` gösteriyor (K-056 madde 2).
- **Karar:**
  - **AS-1 birimi:** (model, fabrication–masquerade çifti). Yalnızca nicemlenen modeller (CNN, conv-AE) birim olabilir. XGBoost ve RF (K-053) AS-1 birimi değildir. Gerekçe: ΔRecall tanımsız. `grid_k041_units.csv` değiştirilmez; AS-1 tablosunda (`as1_pairs.csv`) `as1_eligible` ve `as1_excluded_reason` sütunlarıyla işaretlenir.
  - **Birincil birimler:** taban kuralını geçen nicemlenmiş birimler, yani 3 CNN çifti (correlated_signal, max_speedometer, reverse_light_off). Taban kuralıyla dışarıda kalanlar (CNN reverse_light_on, conv-AE'nin 4 çifti) ve `unseen` coolant çifti işaretli olarak raporlanır.
  - **Ölçüler** (her birim × k ∈ {Q1, Q2}; test, `recalibrated` kipi):
    - ΔR_fab = R_fab(Q0) − R_fab(Qk);
    - ΔR_masq = R_masq(Q0) − R_masq(Qk);
    - **D = ΔR_masq − ΔR_fab.** D > 0, gizli tarafta daha çok kayıp demektir (H2 yönü).
    - ΔF1 için aynı üçlü ikincil olarak raporlanır.
    - Ayrıca birincil birimlerin D ortalaması verilir.
  - **Raporlanan değerler:** tohum bazında değerler, 3 tohumun ortalaması ± std ve %95 hiyerarşik bootstrap aralığı (K-058).
  - **Hipotez testi yapılmaz.** n = 3 birimde Wilcoxon işaretli sıra testinin ulaşılabilir en küçük iki yönlü p'si 2/2³ = 0.25'tir. Test hangi veriyle yapılırsa yapılsın anlamlılığa ulaşamaz. Bu değer bilgi olarak yazılır. Analiz **keşifseldir**: etki büyüklüğü (D) ve aralıklar önceliklidir.
- **Gerekçe:** Kullanıcı yönlendirmesi (P6): küçük örneklemde anlamsız kalacak testler zorlanmaz. Taban kuralı K-041'de sonuçlardan önce kayıtlıydı ve değişmez. Yalnızca nicemlenmeyen modellerin AS-1'e giremeyeceği açıkça yazılır.
- **Etkisi:** AS-1'in birincil kanıtı tek modele (CNN) ve tek veri setine (ROAD) dayanır; makalede sınırlılık olarak yazılır (K-014 ile birlikte).

## K-058 — Bootstrap yöntemi: tohum × pencere çapraz hiyerarşik bootstrap, hareketli blok

- **Faz / tarih:** P6 · 2026-10-10
- **Durum:** Geçerli (kullanıcı kararı: blok birincil, iid duyarlılık; P6 analiz kodu çalıştırılmadan)
- **Karar:**
  - **Kaynak:** pencere düzeyindeki test skorları (`results/raw/<run>/quant/scores_q{0,1,2}_test.parquet`; XGB `scores_test.parquet`) ve her sürümün val'den kalibre edilmiş eşiği (`quant/metrics.csv`, `metrics_test.csv`).
    - Eşikler bootstrap'ta **yeniden hesaplanmaz** (kural 3). Yeniden hesaplanan nokta tahminleri `metrics.csv` ile aynı olmalıdır; değilse analiz durur.
  - **Çapraz tasarım:**
    - Her replikte 3 tohum yerine konarak çekilir.
    - Pencereler tek bir ortak çekilişle yeniden örneklenir. Test kümesi bütün tohumlarda, nicemleme düzeylerinde ve modellerde aynıdır; aynı replikte hepsi aynı pencere örneğini görür.
    - Bu sayede Q0–Qk, fabrication–masquerade ve CNN–conv-AE farkları eşleştirilmiş kalır.
    - Replik değeri = çekilen tohumların, çekilen pencerelerdeki istatistiğinin ortalaması.
  - **Pencere örneklemesi:**
    - Hareketli blok bootstrap'ı. Her (segment, label) katmanında pencereler zaman sırasına (`start`) dizilir; L = ⌈n^{1/3}⌉ uzunluğunda bloklar, katmanın boyutu korunacak şekilde çekilir.
    - Katmanlama, her tür kümesinde saldırı ve normal pencere sayısını sabit tutar.
    - **Duyarlılık:** AS-1 için iid sürümü (L = 1) ayrıca raporlanır.
  - **Ayarlar:** B = 10.000 replik, %95 yüzdelik aralığı, sabit RNG tohumu (`configs/analysis.yaml`).
- **Gerekçe:**
  - Ardışık test pencereleri aynı enjeksiyon aralığından gelir; tespit kararları zaman içinde ilişkilidir. iid bootstrap bu durumda aralığı daraltır. Blok uzunluğu kuralı standart bir seçimdir ve sonuçlardan önce sabitlenir.
  - 3 tohumla tohum düzeyindeki yeniden örnekleme kabadır (10 farklı çoklu küme). Yine de eğitim değişkenliğini aralığa katar. Tohum bazında değerler ayrıca verilir.
- **Etkisi:** Aralıklar test penceresi örneklemi ve 3 eğitim tohumu üzerinden belirsizliği gösterir. Veri setinin kendisinden (tek araç, sentetik masquerade; K-014) gelen belirsizliği göstermez.

## K-059 — İkincil grup düzeyi karşılaştırma: ROAD, Mann-Whitney U, Cliff's δ, Holm ve ulaşılabilir en küçük p

- **Faz / tarih:** P6 · 2026-10-10
- **Durum:** Geçerli (kullanıcı onayı, P6 planı; P6 analiz kodu çalıştırılmadan)
- **Karar:**
  - **Kapsam ve gruplar** (plan 2.7'nin birincil analizi; K-013'e göre ikincil):
    - Kapsam: ROAD birincil havuzu (coolant hariç, K-019).
    - Gruplar: coarse = fuzzing + 4 fabrication türü (5 tür); stealthy = 4 masquerade türü.
    - Birim: tür; değer, 3 tohumun ortalaması olan ΔRecall (ikincil aile: ΔF1).
  - **Her model (CNN, conv-AE) × k (Q1, Q2) için:**
    - Mann-Whitney U (iki yönlü, kesin dağılım);
    - Cliff's δ = P(Δ_stealthy > Δ_coarse) − P(Δ_stealthy < Δ_coarse); δ > 0 H2 yönüdür;
    - δ için K-058'e göre hiyerarşik bootstrap aralığı.
  - **Holm-Bonferroni aileleri:** ΔRecall ailesi 4 test; ΔF1 ailesi 4 test.
  - **Ulaşılabilir en küçük p:** her satırda raporlanır. 5'e 4 için kesin MWU'nun en küçük iki yönlü p'si 2/C(9,4) = 2/126 = 0.0159. Holm'da en küçük p 4 ile çarpılır: 0.063 > 0.05. **Bu tasarımda düzeltilmiş anlamlılık yapısal olarak ulaşılamaz.** Bu durum testin sonucundan bağımsızdır ve açıkça yazılır.
  - **Taban kuralı uygulanmaz** (K-041 son madde). conv-AE'nin ROAD recall'u çoğu türde tabanda olduğu için ΔRecall'u yapısal olarak 0'a yakındır; bu, yorumda not edilir.
  - **Duyarlılık:** Car-Hacking'in 4 coarse türü coarse gruba eklenir (9'a 4). Veri seti ve gruplar karışır (gizli türlerin hepsi ROAD'da), bu yüzden yalnızca duyarlılık analizi olarak raporlanır.
- **Gerekçe:** Plan 2.7 ve F5 bu testi istiyor. Küçük örneklem nedeniyle etki büyüklüğü ve aralığı öncelikli tutuluyor; p değerleri ulaşılabilir sınırlarıyla birlikte veriliyor.

## K-060 — AS-2: Pareto cephesi, iki oturum kuralı ve gecikme bütçeleri

- **Faz / tarih:** P6 · 2026-10-10
- **Durum:** Geçerli (bütçe tanımı kullanıcı kararı; P6 analiz kodu çalıştırılmadan)
- **Karar:**
  - **Adaylar** (veri seti başına):
    - CNN Q0, CNN Q2, conv-AE Q0, conv-AE Q2, XGB (nicemlenmez).
    - Q1'in CPU gecikmesi yoktur (K-043). Q1 yalnızca boyut–recall cephesine girer.
    - RF 3 tohumlu analizlere girmez (K-053).
  - **Eksenler:**
    - gecikme: tohum-0, ana makine, medyan (en küçük daha iyi; K-052);
    - model dosya boyutu: tohum ortalaması (en küçük daha iyi);
    - recall: en büyük daha iyi.
      - ROAD birincil cephe: 4 masquerade türünün recall ortalaması (tohum ortalaması).
      - İkincil cephe: toplam (pooled) recall.
      - Car-Hacking'de gizli tür olmadığı için yalnızca toplam recall cephesi kurulur.
  - **Gecikme (K-050):**
    - Pareto her oturumda ayrı hesaplanır. Bir aday ancak **iki oturumda da** baskılanıyorsa baskılanmış sayılır; sağlam cephe, en az bir oturumda baskılanmayan adaylardır.
    - Gecikme, aynı oturumda aynı veri setinin CNN Q0'ına oranla da verilir.
    - Mutlak değerler yalnızca büyüklük mertebesi olarak yorumlanır; p95 yorumlanmaz.
  - **Gecikme bütçeleri** (`configs/analysis.yaml`; ileti hızı `results/tables/data_profile.csv`'den, veri setinin dışlanmamış saldırısız (`ambient`) kayıtlarının `msg_rate_per_s` medyanı):
    - (i) **pencere bütçesi** = 100 ileti / medyan ileti hızı (stride = N ile pencere başına tespit);
    - (ii) **sıkı bütçe** = 1 ileti aralığı = 1 / medyan ileti hızı (her yeni iletide kayan pencereyle tespit).
    - Bir aday, iki oturumun medyan gecikmesinin büyüğü bütçenin altındaysa bütçeye uyar.
  - **Kapsam notu:** ölçümler ana makine CPU'sundandır. Gerçek ECU ya da konteyner ölçümü yoktur (K-051). Bütçe uyumu bir ECU için kanıt değildir.
- **Gerekçe:** Plan ve öneri formu "belirlenen gecikme bütçesi"nden söz ediyor, ancak bir değer tanımlanmamıştı. Veriden türetilen iki bütçe, iki tespit kipinin gerçek zamanlılık sınırıdır ve sonuçlardan bağımsızdır.

## K-061 — AS-3: denetimli ve denetimsiz modelin nicemleme duyarlılığı; kalibrasyon kayması ayrı

- **Faz / tarih:** P6 · 2026-10-10
- **Durum:** Geçerli (kullanıcı onayı, P6 planı; P6 analiz kodu çalıştırılmadan)
- **Karar:**
  - **Karşılaştırma:** CNN (denetimli) ile conv-AE (denetimsiz; K-039), aynı (veri seti, tür) üzerinde eşleştirilmiş. Birincil havuzun 13 türü: CH 4, ROAD 9. Test, `recalibrated` kipi.
  - **Birincil ölçü ΔROC-AUC:** eşikten bağımsızdır ve recall tabanından etkilenmez.
  - **ΔRecall:** yalnızca iki modelin de Q0 recall'u (3 tohum ortalaması) ≥ 0.10 olan türlerde. Bu, K-041'in taban mantığıdır; recall'u tabanda olan modelin ΔRecall'u yapısal olarak sıfıra yakın kalır ve yanlış biçimde "dayanıklı" görünür. Bütün türler ayrıca raporlanır.
  - **Duyarlılık büyüklüğü:** tür başına d_t = |Δ_conv-AE,t| − |Δ_CNN,t|. d_t > 0 denetimsiz modelin daha duyarlı olduğu anlamına gelir (H4 yönü).
    - Birincil istatistik: türler üzerinden d_t ortalaması ve K-058'e göre hiyerarşik bootstrap aralığı. İki modelin tohumları birbirinden bağımsız çekilir, pencere çekilişi ortaktır.
    - İkincil: Wilcoxon işaretli sıra (`zero_method = "pratt"`), Holm ailesi 4 test (ΔAUC, ΔRecall × Q1, Q2); Cliff's δ (|Δ| dağılımları).
    - İşaretli Δ (kayıp yönü) betimleyici olarak verilir.
  - **Kalibrasyon kayması** (K-036 b, `q0_threshold` kipi) ayrı başlıkta raporlanır:
    - saldırısız kayıtlarda val ve test FPR'ı (Q0 eşiği Qk'ye aynen uygulandığında);
    - tür bazında recall değişimi;
    - tohum bazında değerler, ortalama, min–max.
- **Gerekçe:** H4'ün öne sürdüğü mekanizma ("anomali skoru küçük hata farklarına dayanır"), eşik yeniden kalibre edildiğinde ayırt edicilik kaybı (ΔAUC) olarak, eşik sabit kaldığında ise kalibrasyon kayması olarak görünür. İki kip ayrı raporlanır.

## K-062 — Hipotez hüküm kuralları (H1–H4)

- **Faz / tarih:** P6 · 2026-10-10
- **Durum:** Geçerli (kullanıcı onayı, P6 planı)
- **Zamanlama notu:** Bu kurallar P5'in betimleyici sonuçları (tohum ortalamaları, `grid_quant.csv`) görüldükten **sonra**, ancak P6'nın bootstrap aralıkları ve testleri hesaplanmadan **önce** yazıldı. Önceden kayıtlı değildirler. Makalede böyle belirtilir.
- **Hüküm etiketleri:** doğrulandı / doğrulanmadı / kısmen doğrulandı / test edilemedi.
- **Kurallar** (Q2 = INT8; test; `recalibrated` kipi; birincil havuz):
  - **H1:**
    - **(a) Toplam kayıp sınırlı:** her derin model × veri setinde toplam (pooled) ΔF1'in tohum ortalaması ≤ 0.05.
    - **(b) Eşit dağılmıyor:** türler arasında, ΔRecall aralığı 0'ı kayıp yönünde dışlayan en az bir tür ve aralığı 0'ı içeren en az bir tür var.
    - Hüküm: (a) ve (b) birlikte doğrulandı; yalnızca biri sağlanırsa kısmen.
  - **H2:**
    - **(a) Gizli tarafta daha büyük kayıp:** birincil birimlerin D ortalaması > 0 ve aralığı 0'ı dışlıyor (K-057). Grup düzeyindeki ikincil analiz (K-059) destekleyici olarak raporlanır, hükmü belirlemez.
    - **(b) Kaba saldırılar yüksek başarımla tespit ediliyor:** Q2'de CH dos, CH fuzzy ve ROAD fuzzing recall'unun tohum ortalaması ≥ 0.90; her derin model için ayrı.
    - Hüküm: (a) ve (b) birlikte doğrulandı; yalnızca biri sağlanırsa kısmen.
  - **H3:**
    - **(a) Doğruluk:** her derin model × veri setinde, tür bazında |ΔRecall| ortalaması Q1'de Q2'dekinden küçük ya da ona eşit.
    - **(b) Kazanç:** Q1 dosya boyutu < Q0. Gecikme kazancı **test edilemedi** (CPU'da fp16 çekirdeği yok; K-043).
    - **(c) INT8'de gizli kayıp:** H2 (a) ölçütü.
    - Hüküm: gecikme bileşeni test edilemediği için en fazla "kısmen doğrulandı".
  - **H4:** AS-3 birincil ölçüsünün (ΔROC-AUC; d_t ortalaması) Q2'deki aralığı tamamen 0'ın üstünde.
    - ΔRecall bileşeni (taban kısıtlı türler) aynı yöndeyse doğrulandı;
    - yalnızca biri sağlanıyorsa kısmen;
    - ikisinin de aralığı 0'ı içeriyor ya da 0'ın altındaysa doğrulanmadı.
    - Kalibrasyon kayması hükme girmez, ayrıca yorumlanır.
- **Gerekçe:** Hükümler sayısal ve yeniden üretilebilir olmalı. Kurallar aralıkların hesaplanmasından önce sabitlenir.
