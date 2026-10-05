# Proje: CAN IDS’te Nicemlemenin Saldırı Türüne Göre Etkisi

## Amaç

Araç içi CAN saldırı tespit modellerinde nicemlemenin (float32 → float16 → INT8) başarım kaybını saldırı türü bazında ölçmek. Çıktı: bir konferans bildirisi.

## Ortam

Python 3.13 (proje `.venv`'i, sabit sürümler `requirements.txt`'te). Gerekçe: docs/DECISIONS.md K-001.

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

## Karar kaydı kuralı

Her fazda alınan metodolojik kararlar (plandan veya öneri formundan sapmalar dahil) gerekçeleriyle birlikte docs/DECISIONS.md'ye yeni bir girdi olarak eklenir. Mevcut girdiler silinmez; bir karar değişirse yeni girdi eski girdiye atıf yaparak onu geçersiz kılar. Faz günlüğü ilgili karar numaralarına atıf yapar.

## Uzun işler kuralı

5 dakikadan uzun sürecek eğitim/deney komutlarını kendin çalıştırma. Arka planda çalışacak, kesintiye dayanıklı bir komut hazırla; işletim sistemine uygun çalıştırma talimatını (ayrı terminal, log dosyası, ilerleme izleme, kesinti sonrası devam) bana ver.

## Onay gerektiren işlemler

Aşağıdaki işlemlerden önce ne yapılacağını ve nedenini açıklayıp kullanıcıdan açık onay al; onay gelmeden çalıştırma:

- Dosya veya klasör silme (yer tutucular, üretilmiş çıktılar ve `results/` altındakiler dahil).
- Git geçmişini değiştiren her işlem: force-push, rebase, `reset --hard`, commit amend, geçmişten dosya temizleme.
- Sistem çapında kurulum veya yapılandırma değişikliği (proje `.venv`'i dışına paket kurulumu, global git ayarı, sürücü/CUDA/Docker kurulumu vb.).
- 5 dakikadan uzun sürecek komutlar (bkz. "Uzun işler kuralı": bu komutlar kendin çalıştırılmaz, çalıştırma talimatı verilir).

Serbest olanlar: faz sonunda commit ve normal (force olmayan) push; proje `.venv`'i içine paket kurulumu.

## Dil

Kod, yorum, docstring ve README İngilizce. docs/phase-logs ve docs/reports Türkçe.