# Danışman geri bildirimi — 1

**Tarih:** 2026-10-17
**Kaynak:** Dr. Öğr. Üyesi Muhammed Furkan Küçük; 1. hafta raporuna (`docs/reports/week-01.md`) yanıt.
Metin aşağıda aynen verilmiştir. Bu geri bildirim P10 genişletme fazı olarak planlandı (`docs/DECISIONS.md` K-071 …).

---

Merhaba Olcay,

Aşağıda yazdığım denemelere bakıp uygulayabilirsin. Özellikle "proje önerisinde" taradığın makalelerde Car-Hacking ile ROAD veri setlerinin modele sokmadan önce nasıl bir ön işlemlerden geçirildiğini iyi anlamak gerekiyor. Bazı özellikler kaybolmuş olabilir bu sebeple sonuçlar diğer makalelerde elde edilenlere göre düşük çıkmış olabilir.

Ön işlemeyi doğrula: Normalizasyon parametrelerinin yalnızca eğitim verisinden hesaplandığını, CAN mesajlarının zaman sırasının korunduğunu, DLC değeri 8'den küçük mesajlarda yapılan sıfır doldurmanın doğru yönetildiğini ve eğitim–test arasında veri sızıntısı olmadığını kontrol et. Özellikle Car-Hacking veri setindeki zaman bloğu ayrımının bağımsızlık açısından sınırlılıklarını değerlendir.

Özellik çıkarımını genişlet: Mevcut CAN ID, payload ve mesajlar arası zaman farkına ek olarak aynı CAN ID'nin ardışık mesajları arasındaki zaman farkını, mesaj frekansını, ID dağılımını, payload değişimini ve bit değişim oranını incele. Bu özellikler özellikle gizli saldırıların tespitinde faydalı olabilir.

Özellik karşılaştırma deneyi yap: Mevcut 10 özellikli modeli referans al. Yeni özellikleri aşamalı olarak ekleyerek aynı veri bölünmeleri üzerinde 1D-CNN ve Conv-AE modellerini yeniden eğit. Özellikle ROAD veri setinde saldırı türü bazında Recall, F1 ve AUC değerlerinin iyileşip iyileşmediğini göster.

Düşük başarımın nedenini araştır: ROAD üzerinde Conv-AE'nin yaklaşık %5,5 Recall üretmesinin özellik yetersizliğinden mi, model mimarisinden mi, yoksa karar eşiğinden mi kaynaklandığını belirlemeye çalış.

Nicemleme ve kalibrasyonu karşılaştır: Özellik çıkarımı deneylerinden sonra FP32, FP16 ve INT8 sonuçlarını karşılaştır. Özellikle INT8 için eski eşik ile yeniden kalibre edilmiş eşik kullanıldığında FPR ve Recall değerlerinin nasıl değiştiğini bağımsız test verisinde göster.
