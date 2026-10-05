# Proje: CAN IDS’te Nicemlemenin Saldırı Türüne Göre Etkisi

## Amaç

Araç içi CAN saldırı tespit modellerinde nicemlemenin (float32 → float16 → INT8) başarım kaybını saldırı türü bazında ölçmek. Çıktı: bir konferans bildirisi.

## Değişmez kurallar

1. **Veri sızıntısı yasaktır.** Eğitim/doğrulama/test ayrımı DOSYA düzeyinde yapılır. Aynı kayıt dosyasından hem eğitim hem test penceresi üretilmez.
2. Test kümesinde pencereler ÖRTÜŞMESİZ (stride = pencere boyu). Örtüşme yalnızca eğitimde.
3. Anomali eşiği ve tüm hiperparametreler YALNIZCA eğitim/doğrulama verisinden belirlenir. Test verisine bakarak eşik seçmek yasaktır.
4. INT8 kalibrasyon kümesi yalnızca eğitim verisinden örneklenir.
5. Her sonuç en az 3 tohumla üretilir; ortalama ± standart sapma raporlanır.
6. Metrikler saldırı türü bazında hesaplanır. Yalnızca toplam skor raporlamak yasaktır.
7. Her koşu kendi yapılandırmasını, kütüphane sürümlerini ve git commit hash’ini sonuç dosyasına gömer.
8. Gecikme ölçümü en az 1000 tekrarlıdır ve ısınma koşuları hariç tutulur.

## Yasaklar

- Sonuç uydurma, örnek değer yazma, çalıştırılmamış deneyin skorunu tahmin etme.
- Test setinde model seçimi.
- Sınıf dengesizliğini görmezden gelip accuracy raporlama.
- Veri setinin tamamını belleğe yükleyip rastgele bölme (sızıntı üretir).

## Faz disiplini

Aynı anda bir faz. Kabul kriteri sağlanmadan ilerleme yok. Her faz sonunda commit.

## Danışman yönlendirmesi

Danışman (Dr. Muhammed Furkan Küçük) iki yöntemi onayladı: (i) denetimli 1D-CNN pencere sınıflandırıcı, (ii) yalnızca normal trafikle eğitilen, anomali skoru yeniden oluşturma hatası olan ve eşiği yalnızca eğitim verisinin skor dağılımından belirlenen LSTM otokodlayıcı. Bu iki model projenin çekirdeğidir. Klasik model (Random Forest/XGBoost) referans içindir; 1D evrişimli otokodlayıcı yalnızca LSTM otokodlayıcı başarısız olursa yedektir. Sonuçlar danışmana haftalık rapor olarak sunulur.

## Faz kapanış kuralı

Her fazın sonunda docs/phase-logs/<faz>.md yazılır: yapılanlar, oluşturulan/değiştirilen dosyalar, elde edilen sayısal sonuçlar, plandan sapmalar ve gerekçeleri, açık sorunlar, kabul kriterlerinin tek tek durumu (✔/✘). Faz bitince DUR ve onay bekle. Sonraki faza kendiliğinden geçme.

## Uzun işler kuralı

5 dakikadan uzun sürecek eğitim/deney komutlarını kendin çalıştırma. Arka planda çalışacak, kesintiye dayanıklı bir komut hazırla; işletim sistemine uygun çalıştırma talimatını (ayrı terminal, log dosyası, ilerleme izleme, kesinti sonrası devam) bana ver.

## Dil

Kod, yorum, docstring ve README İngilizce. docs/phase-logs ve docs/reports Türkçe.
