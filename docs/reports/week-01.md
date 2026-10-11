# Haftalık danışman raporu — 1. hafta (5–11 Ekim 2026)

**Proje:** CAN IDS'te nicemlemenin saldırı türüne göre etkisi · **Kapsam:** P0–P7 (plan F0–F6) · Ayrıntılar: `docs/phase-logs/`, `docs/DECISIONS.md` (K-001 … K-065), `docs/FINDINGS.md`

## 1. Bu hafta yapılanlar

- **P0–P2:** ortam ve veri profili (Car-Hacking, ROAD); trafik ölçütleriyle kaba/gizli gruplama; sızıntı testli bölümleme; çerçeve öznitelikleri ve pencereleme.
- **P3:** 1D-CNN, otokodlayıcı ve XGBoost/RF temel modelleri; önceden kayıtlı fizibilite protokolüyle otokodlayıcı seçimi.
- **P4:** float16/INT8 eğitim sonrası nicemleme (ONNX Runtime) ve 10.000 ölçümlü gecikme protokolü.
- **P5–P6:** 18 koşuluk ızgara (3 tohum); hiyerarşik bootstrap, hipotez hükümleri.
- **P7:** 8 şekil ve 4 tablo, makale taslağına hazır.
- **Henüz yok:** makale metni.

## 2. Sayısal sonuçlar (test; Δ = float32 − nicemli, pozitif = kayıp; köşeli parantez %95 bootstrap aralığı)

| Hipotez | Hüküm | Dayanak |
|---|---|---|
| H1 toplam kayıp sınırlı, türlere eşit dağılmıyor | Doğrulandı (eşitsizlik çok küçük) | Toplam ΔF1(INT8) ≤ 0.003. Aralığı 0'ın üstünde kalan 3 türde (CNN, ROAD) kayıp ≈ 0.007 recall. En büyük ortalama 0.025 (conv-AE fuzzing), ama aralığı 0'ı içeriyor |
| H2 gizli saldırıda kayıp daha büyük | **Doğrulanmadı** | Eşleştirilmiş fark D(INT8) = 0.0006 [−0.0035, 0.0057]; grup testi Holm p = 0.44 |
| H3 fp16 dengeli, INT8 en kazançlı | Kısmen | Dosya 0.52× / 0.38×; CNN INT8 gecikmesi 0.64–0.75×; conv-AE'de kazanç yok (1.01–1.08×); fp16 gecikmesi CPU'da ölçülemedi |
| H4 denetimsiz model daha duyarlı | Kısmen | Başarım kaybı farkı yok: ΔAUC farkı 0.0025 [−0.0041, 0.0088]. Çıktı daha çok değişiyor: Spearman 0.973 (CNN 0.9997) |

**Kalibrasyon kayması** (Şekil 8): float32 eşiği INT8 conv-AE'ye aynen uygulanınca ROAD'da val FPR 0.010 → 0.185 (tohuma göre 0.020–0.486). Eşik nicemli modelin kendi val skorlarıyla yeniden belirlenince kayma kayboluyor. Ana sonuç şekli: Şekil 3 (`results/figures/fig3_*`).

## 3. Sorunlar ve kararlar (öneri formundan/plandan sapmalar)

1. **Gruplama mekanizmaya göre (K-011, K-012):** spoofing ve fabrication trafik hızını 2–7 kat değiştiriyor; ada göre "gizli" sayılamazlar. Gizli grup = zamanlamayı ve kimlik dağılımını koruyan saldırılar. Veride bunlar yalnızca ROAD'un 5 masquerade türü ve hepsi sentetik türetilmiş (K-014, sınırlılık).
2. **Eşleştirilmiş fabrication–masquerade tasarımı (K-013, K-057):** aynı kaynak kayıttan iki sürüm; tek fark zamanlama izi. Taban kuralından sonra 3 CNN birimi kaldı. Bu sayıda test anlamlılığa ulaşamaz (en küçük p 0.25), bu yüzden analiz keşifsel.
3. **Pencere örtüşmesi (K-002):** öneri formundaki %50 örtüşme yalnızca eğitimde kaldı. Val ve testte pencereler örtüşmesiz; aynı olay iki kez sayılmıyor.
4. **Birincil ölçü kalibre recall (K-037):** %1 FPR eşiğinde recall ve ΔRecall birincil; F1 ikincil. Gerekçe: türler arasında normal pencere oranı çok farklı, F1 bundan etkileniyor.
5. **LSTM-AE yerine conv-AE (K-039):** önceden kayıtlı protokolde LSTM-AE ROAD'da val AUC 0.39 ile eşiği (0.6) geçemedi, ayar denemesinden sonra da geçemedi. Yedek model conv-AE iki veri setinde de geçti. Ancak ROAD'daki toplam recall'u yalnızca 0.055.
6. **Car-Hacking'de dosya içi zaman bloğu ayrımı (K-017):** her saldırı tek dosyada. Bloklar arasında ≥ 12 s boşluk var; bağımsızlık dosya düzeyinden zayıf.
7. **Gecikme tekrarlanabilirliği (K-050):** iki ölçüm oturumu %5 toleransı sağlamadı. Gecikme oturum aralığıyla ve yalnızca yön olarak raporlanıyor; ECU ölçümü yok.

## 4. Gelecek hafta planı

- Makale iskeleti ve Bölüm III–V (tehdit modeli, yöntem, kurulum), ardından Bölüm VI (sonuçlar).
- Şekil seçimi. Önerim: ana metinde Şekil 1, 3, 5, 8; Şekil 2, 4, 7 ekte.

## 5. Görüşünüze ihtiyaç duyulan noktalar

1. LSTM-AE yerine conv-AE (madde 5) onaylı yönteme uygun mu?
2. H2 doğrulanmadı. Makaleyi "INT8 gizli saldırılara özgü kayıp yaratmıyor; asıl risk kalibrasyon kayması" çerçevesiyle yazmak uygun mu?
3. H4'ün "kısmen" hükmü, kayıtlı kuralın yanlılığı fark edildikten sonra ve bütün sonuçlar görüldükten sonra verildi (K-063, K-064). Makalede bu sırayla açıkça yazılacak; bu yaklaşım yeterli mi?
4. Car-Hacking zaman bloğu ayrımı (madde 6) hakem açısından kabul edilebilir mi?
