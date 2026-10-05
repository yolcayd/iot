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
