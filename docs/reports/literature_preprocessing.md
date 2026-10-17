# Literatürde Car-Hacking ve ROAD ön işlemeleri — karşılaştırma

**Tarih:** 2026-10-17 · **Faz:** P10 (K-071) · **Danışman isteği:** önerideki makalelerde Car-Hacking (CH) ve ROAD'un modele girmeden önce hangi ön işlemlerden geçtiğinin anlaşılması.

## Yöntem ve kapsam

- **Kaynaklar:** öneri formundaki 5 makale + LCSMC-Net + CANival.
- **Okuma:** tam metinler 2026-10-17'de web üzerinden okundu. Her hücre makalenin bölüm ya da tablo numarasına dayanır. Metinde bulunmayan bilgi "belirtilmemiş" diye yazıldı; tahminle hücre doldurulmadı.
- **Tam metni okunamayanlar:** 3 makale (yayıncılar otomatik erişimi engelledi, HTTP 403). Bunlar ayrı tabloda; yalnızca özet düzeyindeki bilgi verildi ve satırlar boş bırakıldı.
- **Okunan sürümler:**
  - Rajapaksha 2023: NDSS yayıncı PDF'i;
  - Hanselmann 2020 (CANet): arXiv 1906.02492 (IEEE Access sürümünün ön baskısı);
  - Verma 2024: PLOS ONE HTML;
  - Hou 2026 (LCSMC-Net): PMC12944881 (PMC sayfasının ilk ≈ 100 bin karakteri; Bölüm 5.4 sonrası okunmadı).

## 1. Tam metni okunan çalışmalar

| | **Rajapaksha vd. 2023 (CAN-ODTL)** | **Hou vd. 2026 (LCSMC-Net)** | **Hanselmann vd. 2020 (CANet)** | **Verma vd. 2024 (ROAD)** | **Bu proje** |
|---|---|---|---|---|---|
| Veri seti | **ROAD**: fuzzing, max speedometer, reverse light on/off ve bunların masquerade sürümleri (7 tür; Tablo I, Bölüm V-A). Accelerator ve correlated_signal kullanılmamış | **CH**: DoS, Fuzzy, RPM ve Gear spoofing; ayrıca HCRL CAN-FD (Bölüm 5.1). ROAD yok | Kendi aracı (13 ID, 20 sinyal) ve **SynCAN** (Bölüm 4). CH ve ROAD yok | ROAD'u tanıtıyor, CH'yi inceliyor (Bölüm 3.3, 4) | CH (4 tür) + ROAD (fuzzing, 5 fabrication, 5 masquerade; accelerator dışlandı, K-008) |
| Temsil / öznitelik | Yalnızca **ID dizisi**: 11 ID'lik bağlamdan ortadaki ID'nin tahmini; yük ve zaman öznitelik değil. Zaman damgası yalnızca 100 ms gözlem penceresini kurmak için (Bölüm IV, V-B) | 6 öznitelik (Bölüm 4): ID'nin alt 8 biti, üst 3 biti, yük ortalaması, yük entropisi, **ardışık iki mesaj arası süre** (ID'den bağımsız), pencere içi ID sapma skoru. DLC öznitelik değil, yalnızca uzunluk için | Ham bayt değil, **sinyal uzayı**: yük önceden fiziksel sinyallere çözülmüş (Tablo 1, Bölüm 2). ID başına bir LSTM | Veri hazırlığı önerisi yok. ROAD'da ID'ler yeniden eşlenmiş, yük baytları ID başına karıştırılmış (Bölüm 4.2) | Çerçeve başına 10 öznitelik: train sözlüğüyle ID kodu, 8 bayt, **ID bazında** log IAT (K-024, K-025). P10'da +12 sütun (K-073) |
| Normalizasyon | Belirtilmemiş (gömme katmanı; ID'ler sınıf) | Zaman aralıkları [0, 1]'e ölçeklenmiş; hangi veriyle uydurulduğu belirtilmemiş (Bölüm 4.5, 5.1) | "Sinyal bazında 0–1"; hangi veriyle uydurulduğu belirtilmemiş (Bölüm 4.2) | — | Yalnızca train normal çerçevelerinden; test reddediliyor (K-024, K-025; testler) |
| Pencere | Model: 11 ID'lik kayan bağlam. Karar: **100 ms** zaman pencereleri (Bölüm V-B); örtüşme belirtilmemiş | **8 mesaj, stride 4** (%50 örtüşme); train ve test için ayrı bir kural belirtilmemiş (Bölüm 5.1) | Mesaj bazında akış; eğitimde rastgele başlangıçlı 5000 mesajlık diziler (Bölüm 4.2) | — | 100 çerçeve; train stride 50, **val/test stride 100 (örtüşmesiz)** (K-002) |
| Etiket birimi | 100 ms pencere: zayıf anomali oranı > ψ = 0.04 ise saldırı (Algoritma 2). Pencerenin gerçek etiket kuralı belirtilmemiş | Belirtilmemiş | Nokta (mesaj) bazında; ayrıca aralık bazında ölçüt (Bölüm 4.3) | Etiketler aralık düzeyinde, mesaj düzeyinde değil (Bölüm 4.3) | Pencere: içinde ≥ 1 enjekte çerçeve varsa saldırı. Çerçeve etiketi metadata'dan türetiliyor (K-006) |
| Ayrım | Eğitim ve eşik ayrı **normal** kayıtlarda; saldırı kayıtları yalnızca testte. Hangi saldırı kaydının (`_1/_2/_3`) kullanıldığı belirtilmemiş (Bölüm V-B) | **8:2 tabakalı ayrım**, dengeli alt örneklemeden sonra (Bölüm 5.1); rastgele mi zaman/dosya bazında mı belirtilmemiş. Ayrıca 5 katlı çapraz doğrulama (Bölüm 5.2.6) | Gerçek veri: 12.5 saat train, 0.5 saat test (zamana göre). SynCAN: 16.5 / 7.5 saat. Saldırılar test kesitlerine sonradan eklenmiş (Bölüm 4, 4.1) | — | Dosya/kaynak kayıt düzeyinde (ROAD, K-018); CH'de zaman blokları, ≥ 12.9 s boşluk (K-017) |
| Test penceresi örtüşmesi | Belirtilmemiş | Belirtilmemiş; aynı stride 4 kullanılıyorsa %50 örtüşme | Mesaj bazında; yok | — | Yok |
| Eşik | Yalnızca ayrı bir **normal** kümeden: ID başına softmax 0.001 kantili; pencere eşiği ψ, normal pencerelerde en düşük FPR'a göre (Bölüm IV-C, V-B) | Sınıflandırıcı (softmax); eşik belirtilmemiş | Eğitimden ayrılmış küçük bir normal kümede, sinyal başına hata dağılımının %99.99 persentili (Bölüm 3, 4.2) | — | Temiz val normallerinde FPR %1 (K-035, K-038) |
| Metrik / skor | Tür başına makro F1, FP, FN (Tablo II): **CAN-ODTL 7 türün hepsinde F1 %99.8–100**, masquerade dahil | Accuracy %99.89, F1 0.9985, recall %99.91 (Tablo 7–8); fuzzing recall %99.68 (Bölüm 5.2.2) | Nokta bazında TPR/TNR: TNR > 0.99; plateau/playback TPR 0.85–0.955; continuous change > 0.70 (Bölüm 4.3, Tablo 3) | ROAD'da IDS denenmemiş (Bölüm 5) | Tür bazında kalibre recall (K-037), F1, AUC, TPR@1% |
| Tür bazında sonuç | Evet (Tablo II) | Kısmen (yalnızca fuzzing sayısal) | Evet (saldırı türü başına) | — | Evet (birincil) |

### Verma vd.'nin (ROAD makalesi) CH hakkındaki tespitleri (Bölüm 3.3)

- Her CH saldırı dosyası, saldırıdan sonra uzun bir veri yolu sessizliği içeriyor; yazarlar dosyaların bu boşluktan hemen önce kesilmesini öneriyor.
  - **Bizde:** boşluktan sonraki bölüm, dört dosyada da aynı olan bir kayıt. Bu bölüm dosyalardan kesilip tek bir segment olarak bir kez kullanılıyor (K-016); boşluklar K-010'da ele alındı.
- Yazarlara göre araç normal kayıtta sürülüyor, saldırı kayıtlarında sürülmüyor. Bu nedenle CH test verisi, enjeksiyonlar dışında da eğitim verisinden temelden farklı; uygun bir test kümesi değil.
  - **Bizde:**
    - Her türün değerlendirmesi, saldırı pencereleriyle **aynı kaydın** normal pencerelerinden oluşuyor (K-029). Model "sürüş durumu" ile saldırıyı karıştırırsa bu, tür içi FPR'da görünür.
    - Yine de eğitim kümesinde normal dosya (sürüş) ile saldırı dosyaları (durağan) bir arada. Bir sınıflandırıcının saldırı dosyasına ait olmayı öğrenmesi mümkün.
    - CH'deki tavan sonuçları (CNN recall ≈ 1.0) bu yüzden de iyimser olabilir. Bu, K-017 sınırlılığına ek bir nokta; makalede yazılmalı.

## 2. Tam metni okunamayan çalışmalar

| Çalışma | Erişim girişimi | Özetten bilinen (yalnızca bu) | Ön işleme hücreleri |
|---|---|---|---|
| **Song, Woo & Kim 2020** (DCNN; Vehicular Communications 21) | DOI → Elsevier (açık erişim değil; Semantic Scholar'da açık PDF yok); Korea Univ. deposunda yalnızca özet | Gerçek araçtan toplanan veri setleri (CH'nin kaynağı); Inception-ResNet'ten sadeleştirilmiş DCNN; geleneksel ML'den düşük yanlış negatif ve hata oranı | Okunamadı: hepsi boş |
| **Oualil vd. 2026** (CAN-TEMPO; Technologies 14(6)) | MDPI HTML ve PDF: 403 (açık erişim, CC BY; bot engeli) | Denetimsiz; FFT ile periyot çıkarımı, çok ölçekli 2B temsil; "birden çok açık CAN IDS veri seti"; AUC-ROC ve F1'de üstünlük, düşük FPR | Okunamadı: hepsi boş |
| **Kang vd. 2024** (CANival; Vehicular Communications 50) | ScienceDirect ve Elsevier API: 403 / ulaşılamadı (açık erişim, CC BY) | Zaman aralığı (TIL) ve sinyal (CANet) analizörleri; **X-CANIDS ve SynCAN** (CH ve ROAD değil); ortalama TPR 0.960 / 0.912, TNR 0.997 / 0.996 | Okunamadı: hepsi boş |

**Tamamlamak için:** bu üç makalenin PDF'i depo dışında bir klasöre (ör. `data/literature/`; `/data/` git'e girmez) konursa tablo aynı kuralla tamamlanabilir. Kullanıcı Chrome uzantısını bağlarsa iki açık erişimli makale (CAN-TEMPO, CANival) tarayıcıdan da okunabilir.

## 3. Bizim tasarımla farklar ve skor farkının olası nedenleri

Aşağıdakiler **hipotezdir**. Hiçbiri bu projede deneyle sınanmadı. Literatür protokolünün sızıntılı bir kopyasını (rastgele bölme, örtüşmeli test) çalıştırmak CLAUDE.md'de yasak, bu yüzden yapılmadı.

| # | Fark | Kimde | Skora olası etkisi |
|---|---|---|---|
| 1 | **Rastgele / tabakalı pencere bölmesi ve örtüşmeli pencereler.** Train ve testte %50 örtüşen 8 mesajlık pencereler; ayrımın zaman/dosya bazında olduğu belirtilmemiş | LCSMC-Net (CH) | Ayrım rastgeleyse komşu ve örtüşen pencereler iki kümede birlikte bulunur; test, eğitimde görülmüş mesajları içerir. Accuracy %99.89 bu koşullarda iyimserdir. Bizde CH'de 32 çerçevelik hiçbir dizi iki kümede yok (K-023) |
| 2 | **Dengeli alt örnekleme ve accuracy** | LCSMC-Net | Dengeli kümede accuracy, gerçek saldırı/normal oranında FPR'ı göstermez. Biz accuracy raporlamıyoruz; FPR'ı saldırısız kayıtlarda ölçüyoruz (K-027, K-029) |
| 3 | **Eşik ve işletim noktası.** CAN-ODTL'de eşik yalnızca normal veriden (doğru uygulama). Ama karar 100 ms pencerede, zayıf anomali sayımıyla veriliyor (ψ = 0.04). Biz %1 FPR'da, 100 çerçevelik (≈ 40 ms) pencerede kalibre recall raporluyoruz | CAN-ODTL | Farklı işletim noktası ve pencere tanımı; skorlar doğrudan karşılaştırılamaz. Bizde conv-AE'nin val'de kalibre edilen eşiği teste sıkı aktarılıyor (test FPR %0.1; K-040); bu, recall'u düşüren ayrı bir etken (P10 teşhisi, K-076) |
| 4 | **Masquerade'in yakalanma yolu.** CAN-ODTL yük görmüyor, yalnızca ID dizisini görüyor; yine de masquerade'de F1 ≈ %100. Yazarlar bunu, sentetik masquerade kayıtlarındaki zaman uyumsuzluğunun ID dizisini değiştirmesine bağlıyor (Bölüm V-C) | CAN-ODTL | ROAD masquerade kayıtları sonradan işlenmiş (K-007, K-014). Meşru çerçevelerin silinmesi ID bağlamında bir iz bırakıyor olabilir. Bu bir **veri yapıntısıdır**, gerçek bir masquerade'de olmayabilir. Bizim 10 özniteliğimiz ID'yi pencere içinde sıralı tutuyor; ama "ID bağlamı" özniteliği (ör. önceki ID'ler) yok. P10'daki G4 (pencere ID dağılımı) bunu kısmen yakalar. Bu yapıntıdan gelen bir kazanç, gerçek gizli saldırı tespiti olarak yorumlanmamalı |
| 5 | **Hangi kayıtların test edildiği.** CAN-ODTL'de saldırı kayıtlarının hangisinin (`_1/_2/_3`) test edildiği ve tür başına normal pencerelerin nereden geldiği belirtilmemiş | CAN-ODTL | Bizde test yalnızca `_3` kayıtları (fuzzing `_2`, `_3`; coolant ayrı) ve görülmemiş saldırısız kayıtlar (K-018). Eğitim saldırısı ile test saldırısı farklı kayıtlar |
| 6 | **Eğitimde saldırı verisi yok** (denetimsiz; eşik normal veriden). Bu bizim conv-AE'mizle aynı ilke. Fark: CAN-ODTL ID dizisi modeli, bizimki yeniden oluşturma (yük + IAT + ID kodu) | CAN-ODTL, CANet | Yeniden oluşturma hatasının bütün kanalların ortalaması olması, yükteki küçük sapmayı ID/IAT gürültüsü içinde boğabilir (P10 teşhis (d), K-076) |
| 7 | **Sinyal uzayı ve sentetik saldırı.** CANet'te yük önceden fiziksel sinyallere çözülmüş; saldırılar test kesitlerine yazılımla eklenmiş | CANet | Sinyal çözümü, yük baytlarının ham temsilinde kaybolan yapıyı (bit alanları, ölçek) korur. Bizim ham bayt temsilimizde bir sinyal birden çok bayta yayılabilir ya da bir baytın birkaç bitinde olabilir. Reverse light (tek bit) gibi türlerde bu bilgi kaybı olabilir; G1 (bit değişimi) bunu kısmen hedefliyor. CANet CH ve ROAD kullanmadığı için skorlar karşılaştırılamaz |
| 8 | **Global (ID'den bağımsız) varış süresi** | LCSMC-Net (öznitelik F5) | Bizde yok (yalnızca ID bazında IAT). DoS/flooding'de ayırt edici; bizde CH DoS/Fuzzy zaten tavanda, ROAD gizli türlerinde zaman izi yok. Etkisinin küçük olması beklenir, sınanmadı (K-072) |
| 9 | **CH test kümesinin niteliği** (Verma vd., Bölüm 3.3) | Bütün CH kullanan çalışmalar | CH saldırı kayıtlarında araç durağan, normal kayıtta sürülüyor. Hem literatürün yüksek CH skorları hem bizim CH tavanımız bu farktan beslenebilir |

## 4. Danışmanın sorusuna kısa yanıt

> "Bazı özellikler kaybolmuş olabilir; bu yüzden sonuçlar düşük çıkmış olabilir."

- **CH:** sonuçlarımız düşük değil. CNN tavanda (recall ≈ 1.0), conv-AE 0.87–0.99. Literatürle fark, CH'de bir sorun göstermiyor.
- **ROAD:** fark gerçek. Ana adaylar:
  1. **İşletim noktası:** %1 FPR'da kalibre recall ile 100 ms pencerede, eşik sayımlı F1 aynı ölçü değil.
  2. **Ayrım:** biz eğitim, doğrulama ve test için farklı kayıtlar kullanıyoruz.
  3. **Temsil:** ID bağlamı ve sinyal yapısı ham bayt temsilinde zayıf. P10'da G1–G4 ile sınanıyor.
  4. **conv-AE'ye özgü:** eşiğin teste aktarımı ve MSE ortalamasının bileşimi (K-076).
- **Kaybolan bilgi:**
  - **DLC kaybolmuyor:** ROAD'da DLC < 8 yok; CH'de DLC ID başına sabit, tek istisna 4 çerçeve (`advisor_ext_status.md` §2.2).
  - **Kaybolanlar:** global varış süresi ve ID bağlam dizisi temsilde yok.
- **Uyarı:** literatürdeki masquerade skorlarının bir kısmı ROAD'un sentetik masquerade yapıntısından gelebilir (madde 4). Bu iz bizim modellerimizde de bir kazanç olarak ortaya çıkarsa, raporda gerçek "gizli saldırı tespiti" diye sunulmamalı.
