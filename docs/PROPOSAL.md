> YZM5015 | Yapay Zekâ ile Nesnelerin İnterneti

**YZM5015**

**YAPAY ZEKÂ İLE NESNELERİN İNTERNETİ**

**PROJE ÖNERİSİ FORMU**

| **SON TESLİM TARİHİ: 7 EKİM 2026 ÇARŞAMBA** |
|---|

| **Ad Soyad** | Yusuf Olcay Dirik |
|---|---|
| **Öğrenci Numarası** | [gizli] |
| **Program / Bölüm** | Yapay Zekâ ve Makine Öğrenmesi Yüksek Lisans Programı |

**NOT:** Hazırladığınız bu dökümanı muhammed.kucuk@btu.edu.tr adresinden email olarak gönderiniz.

| Konu seçimi: Öğrenci, ders kapsamında ele alınan genel IoT konu alanlarından birini seçmeli; bu alanı belirli, uygulanabilir ve ölçülebilir bir araştırma problemine dönüştürmelidir. Yalnızca geniş alan adı yazılması yeterli değildir. |
|---|

# 1. Proje Konusunun Seçimi

Aşağıdaki genel IoT alanlarından konular vardır. 1. Maddeki genel konunun yanısıra biraz daha daraltılan 3 farklı alt konu başlığı da verilmiştir (survey makalelerde den bu şekilde daraltarak gidilebilir) .Bu alanlardan hareketle özgün ve sınırlandırılmış bir problem tanımlamalıdır.

- IoT ve Görüntü İşleme

- Edge cihazlarda gerçek zamanlı nesne/olay tespiti ve hafif görüntü işleme,

- Visual IoT sistemlerinde enerji-verimli görüntü analizi ve model sıkıştırma,

- IoT kameralarında mahremiyeti koruyan görüntü analizi ve yerel çıkarım.

- 6G ve IoT kavramı

- Dijital Ajanlar

- IoT de güvenlik

- IoT sistemleri ve mimarileri

- IoT haberleşme teknolojileri

- Sensörler ve veri toplama sistemleri

- IoT platformları

- IoT verilerinin işlenmesi ve analizi

- IoT verilerinde özellik çıkarımı

- Makine öğrenmesi tabanlı IoT uygulamaları

- Derin öğrenme tabanlı IoT uygulamaları

- IoT sistemlerinde anomali tespiti

- Edge AI / Edge Computing

- TinyML ve kaynak kısıtlı cihazlarda yapay zekâ

- Dağıtık öğrenme ve Federated Learning

- Endüstriyel IoT (IIoT)

- Akıllı şehir uygulamaları

- Akıllı ev ve bina sistemleri

- Sağlıkta IoT

- Akıllı tarım

- Akıllı ulaşım ve araç IoT sistemleri

- Enerji ve çevre izleme sistemleri

- IoT sistemlerinde güvenlik ve gizlilik

- IoT tabanlı kestirimci bakım

- İnsan aktivitesi tanıma

- IoT tabanlı olay / durum algılama

- Dijital İkiz ve IoT Kavramı

- …….

| Listede bulunmayan bir konu seçilebilir; ancak proje önerisinde konunun IoT ve Yapay Zeka Uygulaması ile doğrudan ilişkisi açıkça gösterilmelidir. |
|---|

# 2. Proje Başlığı

Projenin konusunu açık ve mümkün olduğunca spesifik biçimde ifade eden geçici bir başlık yazınız.

| **Kaynak Kısıtlı Uç Cihazlarda Araç İçi CAN Ağı Saldırı Tespiti: Model Nicemlemenin Saldırı Türüne Göre Tespit Başarımı ve Çıkarım Gecikmesi Üzerindeki Etkisinin Analizi**<br>İngilizce başlık: Attack-Type-Aware Analysis of Model Quantization Effects on In-Vehicle CAN Intrusion Detection for Resource-Constrained Edge Devices |
|---|

# 3. Problem Tanımı ve Motivasyon

Projenin çözmeye çalışacağı problemi açıklayınız. Yazacağınız paragraf şu sorulara cevap verebilmelidir:

- Ele alınacak problem nedir?

- Problem hangi IoT uygulama alanında ortaya çıkmaktadır?

- Bu problemin çözülmesi neden önemlidir?

- Literatürdeki mevcut yaklaşımların hangi sınırlılıkları bulunmaktadır?

- Önerilen çalışmaya neden ihtiyaç vardır?

- Problem tanımını yaparken literatürdeki diğer çalışmalara atıf yaparak yazmalısınız.

Önerilen uzunluk: 150–300 kelime.

| Modern araçlar, onlarca elektronik kontrol ünitesinin (ECU) Controller Area Network (CAN) veri yolu üzerinden haberleştiği, gerçek zamanlı çalışan gömülü IoT sistemleridir. ISO 11898 ile standartlaştırılan CAN protokolü düşük maliyet ve gerçek zamanlılık gözetilerek tasarlandığından kimlik doğrulama, şifreleme ve bütünlük koruması içermez. Tanı portu, bilgi-eğlence birimi veya telematik ağ geçidi üzerinden veri yoluna erişen bir saldırgan, isteğe bağlı çerçeveleri doğrudan enjekte edebilir. Bu durum, araç içi IoT sistemlerinde güvenlik boyutunu doğrudan can güvenliği meselesi haline getirmekte, UNECE R155 ve ISO/SAE 21434 gibi düzenleyici çerçeveler de saldırı tespitini zorunlu bir savunma katmanı olarak konumlandırmaktadır.<br>Literatürde CAN veri yolu için çok sayıda yapay zekâ tabanlı saldırı tespit sistemi (IDS) önerilmiştir. Denetimli yaklaşımlar bilinen saldırılarda yüksek doğruluk sağlamakta, ancak etiketli veriye bağımlılık ve görülmemiş tehditlere uyum sağlayamama sorunları taşımaktadır (Oualil vd., 2026). Denetimsiz yaklaşımlar bu sınırlılığı aşmayı hedeflese de (Hanselmann vd., 2020), derin öğrenme modellerinin hesaplama maliyeti, araç içine ek bir ECU olarak yerleştirilecek kaynak kısıtlı donanımlarda ciddi bir engel oluşturmaktadır.<br>Bu engeli aşmak için literatürde model nicemleme (quantization) yaygın biçimde kullanılmaktadır. Örneğin Rajapaksha vd. (2023), Raspberry Pi üzerinde float16 nicemleme ile bellek kullanımında ve tespit gecikmesinde önemli oranda azalma raporlamıştır. Ancak bu çalışmalar nicemlemenin maliyetini yalnızca toplam doğruluk üzerinden değerlendirmekte, hangi saldırı türünün bu işlemden ne ölçüde etkilendiğini ayrıştırmamaktadır. Oysa DoS ve fuzzing gibi kaba saldırılar trafiğin zamansal yapısını belirgin biçimde bozarken; maskeleme (masquerade), spoofing ve malfunction gibi gizli saldırılar geçerli kimlikler kullanarak yalnızca yük içeriğinde ince sapmalar yaratır ve tespiti çok daha güçtür (Verma vd., 2024). Toplam doğruluk üzerinden yapılan bir değerlendirme, gizli saldırılardaki kaybı kaba saldırılardaki yüksek başarının ardında gizleyebilir.<br>Bu çalışma, uç cihazda çalışacak bir CAN IDS tasarlanırken nicemlemenin bedelinin saldırı türü bazında ölçülmesi gerektiği savından hareketle, doğruluk–gecikme–model boyutu ödünleşimini saldırı türüne duyarlı biçimde analiz etmeyi amaçlamaktadır. |
|---|

# 4. Projenin/Problemin Amacı

Projenin temel amacını açık ve ölçülebilir biçimde belirtiniz. Proje sonunda hangi problemin çözülmesinin veya hangi sonucun elde edilmesinin hedeflendiğini açıklayınız.

| Bu projenin amacı, araç içi CAN ağı için geliştirilen yapay zekâ tabanlı saldırı tespit modellerinin kaynak kısıtlı uç cihazlarda çalıştırılabilmesi için uygulanan nicemleme (float32 → float16 → INT8) ve model küçültme tekniklerinin, tespit başarımı üzerindeki etkisini saldırı türü bazında ölçülebilir biçimde ortaya koymaktır.<br>Proje sonunda elde edilmesi hedeflenen somut çıktılar şunlardır:<br>1) Her nicemleme seviyesi için saldırı türü bazında F1-skoru, yanlış alarm oranı, çıkarım gecikmesi (ms), model boyutu (MB) ve bellek kullanımını içeren karşılaştırmalı bir başarım tablosu.<br>2) Kaba saldırılar (DoS, fuzzing) ile gizli saldırılar (spoofing, malfunction, masquerade) arasındaki nicemleme kaynaklı başarım kaybı farkının niceliksel olarak belirlenmesi.<br>3) Belirlenen gecikme bütçesi altında en düşük başarım kaybını veren model–nicemleme kombinasyonunun tespiti ve uç cihazda konuşlandırma için pratik bir öneri setinin oluşturulması. |
|---|

# 5. Araştırma Sorusu

En az bir temel araştırma sorusu oluşturunuz. Soru, deneysel olarak değerlendirilebilir ve mümkün olduğunca ölçülebilir olmalıdır.

| Örnek: “X yöntemi, Y IoT veri setinde Z problemi için baseline yönteme göre performansı artırabilir mi?” veya “X özelliğinin kullanılması Y metriğini nasıl etkilemektedir?” |
|---|
| **AS-1 (temel soru): Model nicemleme (float32 → float16 → INT8), araç içi CAN ağı saldırı tespitinde başarım kaybını saldırı türüne göre farklı oranlarda mı üretmektedir? Özellikle gizli saldırılarda (spoofing, malfunction, masquerade) ölçülen F1-skoru kaybı, kaba saldırılardakinden (DoS, fuzzing) istatistiksel olarak anlamlı biçimde büyük müdür?**<br>**AS-2: Uç cihaz için belirlenen çıkarım gecikmesi bütçesi altında, tespit başarımı ile model boyutu/gecikme arasındaki ödünleşimde en uygun model–nicemleme kombinasyonu hangisidir?**<br>**AS-3 (destekleyici): Denetimsiz (otokodlayıcı tabanlı) ve denetimli (sınıflandırma tabanlı) yaklaşımlar, nicemlemeye karşı farklı dayanıklılık sergilemekte midir?** |

# 6. Kısa Literatür Araştırması

Proje konusuyla(belirlenen ana doğrudan ilişkili son 3-5 yıl arasında yayınlanan en az 5 bilimsel çalışma(makale, tez, uluslararası bildiriler vs.) belirleyiniz. Tercihen SCI/SCIE kapsamındaki dergi makaleleri ve IEEE Xplore, ACM Digital Library, ScienceDirect, SpringerLink, Wiley Online Library, DBLP, Scopus, Web of Science, Google Scholar, Semantic Scholar ve arXiv , Elsevier gibi güvenilir bilimsel kaynaklar kullanılmalıdır.

| Konu belirlerken arama yaptığınızi veri tabanlarında önce geniş alan + “survey/review” araması yapılması; ardından ilgi alanınıza göre problem, yöntem ve veri seti anahtar kelimeleri ile aramanın daraltılması önerilir. |
|---|

## Kaynak 1

| **Makale / Kaynak** | Rajapaksha, S., Kalutarage, H., Al-Kadri, M. O., Petrovski, A. & Madzudzo, G. (2023). Improving In-vehicle Networks Intrusion Detection Using On-Device Transfer Learning. VehicleSec 2023. doi:10.14722/vehiclesec.2023.23088 |
|---|---|
| **Problem** | Derin öğrenme tabanlı CAN IDS modellerinin kaynak kısıtlı araç içi donanımda yeniden eğitilmesi ve gerçek zamanlıya yakın çalıştırılması. |
| **Yöntem** | CAN-ODTL: Raspberry Pi üzerinde akan CAN verisiyle cihaz üstü transfer öğrenme; optimize edilmiş ön işleme ve float16 nicemleme. |
| **Temel Sonuç** | Maskeleme saldırıları dâhil yedi saldırı türünde yüksek tespit oranı; float16 nicemleme ile bellek ve tespit gecikmesinde önemli oranda azalma; Raspberry Pi üzerinde yaklaşık 125 ms gecikme. |
| **Sınırlılık / Future Work** | Nicemlemenin kazancı yalnızca bellek ve gecikme üzerinden raporlanmış; nicemlemenin tespit başarımına maliyeti saldırı türü bazında ayrıştırılmamıştır. Bu proje tam olarak bu boşluğu hedeflemektedir. |

## Kaynak 2

| **Makale / Kaynak** | Oualil, S., Ait Yahia, I., El Kamili, M., Fardousse, K. & Berrada, I. (2026). CAN-TEMPO: Unsupervised CAN Bus Intrusion Detection via Temporal Multi-Period Oscillation Encoding. Technologies, 14(6), 375. doi:10.3390/technologies14060375 |
|---|---|
| **Problem** | CAN trafiğinin çok periyotlu yapısının mevcut denetimsiz IDS yaklaşımlarında açıkça modellenmemesi. |
| **Yöntem** | FFT tabanlı otomatik periyot çıkarımı ve çok ölçekli iki boyutlu temsil üreten TEMPO blokları; yalnızca normal trafikle eğitilen kodlayıcı–kod çözücü mimarisi ve yeniden oluşturma hatası tabanlı anomali skoru. |
| **Temel Sonuç** | Farklı araçlar ve saldırı türlerinde AUC-ROC ve F1-skoru açısından karşılaştırılan yöntemlerin üzerinde başarım; daha düşük yanlış alarm oranı. Saldırı türü bazında ayrıştırılmış sonuç raporlanmıştır. |
| **Sınırlılık / Future Work** | Değerlendirme masaüstü ortamda yapılmış; gömülü/uç cihazda çıkarım gecikmesi, model boyutu ve nicemleme etkisi incelenmemiştir. |

## Kaynak 3

| **Makale / Kaynak** | Verma, M. E., Bridges, R. A., Iannacone, M. D., Hollifield, S. C., Moriano, P., Hespeler, S. C., Kay, B. & Combs, F. L. (2024). A comprehensive guide to CAN IDS data and introduction of the ROAD dataset. PLOS ONE, 19(1), e0296879. |
|---|---|
| **Problem** | CAN IDS araştırmalarında gerçekçi, etiketli ve fiziksel olarak doğrulanmış saldırı verisinin bulunmaması; sonuçların karşılaştırılabilirlik ve tekrarlanabilirlik sorunu. |
| **Yöntem** | Açık CAN IDS veri setlerinin kapsamlı taranması ve gerçek bir araçtan dinamometre ortamında toplanan ROAD veri setinin üretilmesi. |
| **Temel Sonuç** | Saldırıların araç üzerindeki etkisi fiziksel olarak doğrulanmış, maskeleme saldırılarını da içeren yüksek doğrulukta bir veri seti yayımlanmıştır. |
| **Sınırlılık / Future Work** | Verinin dinamometre ortamında toplanmış olması gerçek yol koşullarından ince farklar içerebilmektedir; çalışma veri seti üretimine odaklı olup uç cihazda konuşlandırma incelenmemiştir. |

## Kaynak 4

| **Makale / Kaynak** | Hanselmann, M., Strauss, T., Dormann, K. & Ulmer, H. (2020). CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data. IEEE Access, 8, 58194–58205. |
|---|---|
| **Problem** | Bilinen saldırıların yanında bilinmeyen (sıfırıncı gün) saldırıların da tespit edilebilmesi ihtiyacı. |
| **Yöntem** | Yalnızca normal trafikle eğitilen, çok boyutlu CAN sinyal verisi üzerinde çalışan denetimsiz sinir ağı mimarisi. |
| **Temel Sonuç** | Denetimsiz yaklaşımın bilinmeyen saldırı senaryolarında da anlamlı tespit başarımı sağladığı gösterilmiştir. |
| **Sınırlılık / Future Work** | Hesaplama maliyeti ve gömülü donanımda gerçek zamanlı çalışabilirlik değerlendirilmemiştir. |

## Kaynak 5

| **Makale / Kaynak** | Song, H. M., Woo, J. & Kim, H. K. (2020). In-vehicle network intrusion detection using deep convolutional neural network. Vehicular Communications, 21, 100198. |
|---|---|
| **Problem** | Araç içi ağda enjeksiyon saldırılarının yüksek doğrulukla sınıflandırılması. |
| **Yöntem** | CAN mesaj kimliği dizilerinden üretilen iki boyutlu temsiller üzerinde derin evrişimli sinir ağı (DCNN) tabanlı denetimli sınıflandırma. |
| **Temel Sonuç** | DoS, fuzzy, gear ve RPM spoofing saldırılarında yüksek tespit başarımı; Car-Hacking veri setinin literatüre kazandırılması. |
| **Sınırlılık / Future Work** | Denetimli yaklaşım etiketli saldırı verisine bağımlıdır; model karmaşıklığının uç cihaz kısıtları altındaki davranışı ele alınmamıştır. |

Makale sayısına göre tablolar çoğaltılabilir.

# 7. Araştırma Boşluğu (Research Gap)

İncelenen literatüre göre mevcut çalışmaların hangi yönünün geliştirilebileceğini açıklayınız. Örneğin düşük doğruluk, yüksek hesaplama maliyeti, sınırlı veri seti, gerçek zamanlı çalışamama, Edge cihazlara uygun olmama, yüksek enerji tüketimi, güvenlik/gizlilik sorunu veya belirli yöntemlerin karşılaştırılmamış olması araştırma boşluğu oluşturabilir.

**Not:** Derse göstermiştik, survey ve diğer araştırma makalelerin genelde son kısımlarında veya ayrı bir başlık halinde “research gap” bahsedilir.

| İncelenen literatür iki ayrı kümede yoğunlaşmaktadır.<br>**(a) Başarım odaklı çalışmalar (Oualil vd., 2026; Hanselmann vd., 2020; Song vd., 2020), tespit doğruluğunu artırmayı hedeflemekte ve değerlendirmelerini masaüstü/sunucu donanımında yapmaktadır. Bu çalışmalarda model boyutu, bellek kullanımı ve çıkarım gecikmesi genellikle raporlanmamaktadır; dolayısıyla önerilen modellerin araç içine ek bir ECU olarak yerleştirilebilirliği belirsiz kalmaktadır.**<br>**(b) Uç cihaz odaklı çalışmalar (Rajapaksha vd., 2023), nicemleme ve model küçültme ile bellek ve gecikme kazanımlarını göstermektedir. Ancak nicemlemenin bedeli yalnızca toplam doğruluk üzerinden raporlanmaktadır.**<br>**Tespit edilen boşluk: İncelenen çalışmaların hiçbirinde nicemlemenin tespit başarımına etkisi saldırı türü bazında ayrıştırılmamıştır. Bu önemlidir; çünkü CAN saldırıları tespit zorluğu bakımından homojen değildir. DoS ve fuzzing saldırıları trafiğin zamansal düzenini ve mesaj kimliği dağılımını belirgin biçimde bozduğundan düşük hassasiyetli modellerle dahi kolayca ayırt edilebilir. Buna karşılık spoofing, malfunction ve maskeleme saldırıları geçerli kimlikler kullanarak yalnızca yük baytlarında ince sapmalar üretir ve bu sapmaların ayırt edilmesi modelin sayısal çözünürlüğüne çok daha duyarlıdır.**<br>Bu durumda toplam doğruluk üzerinden yapılan bir değerlendirme yanıltıcı olabilir: veri setindeki kaba saldırı örneklerinin ağırlığı nedeniyle toplam F1-skoru yüksek görünürken, güvenlik açısından en kritik olan gizli saldırılardaki kayıp gizlenebilir. İncelenen çalışmalarda bu riskin niceliksel olarak ölçüldüğü bir değerlendirmeye rastlanmamıştır. |
|---|

# 8. Veri Seti / Veri Kaynağı

Projede kullanılacak veri veya veri seti aşağıdaki bilgilerle açıklanmalıdır:

- Veri setinin adı ve kaynağı

- Veri türü ve veri miktarı

- Sınıf sayısı veya hedef değişken

- Veri setine erişim durumu (Github, web siteleri, ilgili link etc.)

- Veri setinin neden seçildiği

| **Veri Seti 1 — Car-Hacking Dataset (HCRL)**<br>Kaynak: Hacking and Countermeasure Research Lab, Korea University. Erişim: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset<br>Veri türü: Gerçek bir araçtan OBD-II portu üzerinden kaydedilmiş ham CAN çerçeveleri (zaman damgası, arbitrasyon kimliği, DLC, 8 bayta kadar yük).<br>Veri miktarı: Saldırı türü başına milyonlarca mesaj içeren ayrı kayıt dosyaları.<br>Sınıflar / hedef değişken: Normal ve dört saldırı türü (DoS, Fuzzy, Gear spoofing, RPM spoofing) — ikili (normal/saldırı) ve çok sınıflı etiketleme mümkündür.<br>Seçilme nedeni: Alandaki en yaygın karşılaştırma ölçütü olduğundan sonuçların literatürle kıyaslanabilirliğini sağlar; ayrıca kaba saldırı sınıflarını (DoS, fuzzing) temsil eder.<br>**Veri Seti 2 — ROAD: Real ORNL Automotive Dynamometer CAN Intrusion Dataset**<br>Kaynak: Oak Ridge National Laboratory. Verma vd. (2024), PLOS ONE 19(1), e0296879. Erişim: Zenodo / ORNL kamuya açık deposu.<br>Veri türü: Gerçek bir binek araçtan dinamometre ve yol sürüşü koşullarında toplanmış ham CAN verisi; yaklaşık 3,5 saatlik kayıt.<br>Sınıflar / hedef değişken: Normal trafik ile fuzzing, hedefli kimlik enjeksiyonu, malfunction ve maskeleme (masquerade) saldırıları.<br>Seçilme nedeni: Bu projenin araştırma sorusu gizli saldırılar üzerine kurulu olduğundan, fiziksel olarak doğrulanmış maskeleme saldırıları içeren tek kamuya açık veri seti olması kritik önemdedir. Ayrıca iki veri setinin birlikte kullanılması, farklı araç ve farklı toplama prosedürü altında genelleme davranışının gözlenmesine imkân verir.<br>Her iki veri seti de açık erişimlidir ve akademik kullanım için ek izin gerektirmemektedir; projede ayrıca veri toplama donanımına ihtiyaç duyulmamaktadır. |
|---|

# 9. Önerilen Yöntem

Projede uygulanması planlanan yöntem veya algoritmaları açıklayınız. Mümkün olduğunca aşağıdaki akışı takip ediniz:

**Veri → Ön İşleme → Özellik Çıkarımı / Temsil → Model → Eğitim → Test → Değerlendirme**

Örnek yöntemler: Makine öğrenmesi, CNN, LSTM, GRU, Autoencoder, Transformer, Transfer Learning, Federated Learning, anomali tespit yöntemleri, Edge AI veya TinyML modelleri.

| **Veri: Car-Hacking ve ROAD veri setlerinden ham CAN çerçeveleri.**<br>**Ön İşleme: Mesajların zaman damgasına göre sıralanması; mesajlar arası varış süresinin (inter-arrival time) hesaplanması ve kırpılarak normalize edilmesi; arbitrasyon kimliğinin ve 8 yük baytının [0,1] aralığına ölçeklenmesi; DLC değeri 8’den küçük çerçevelerde sıfır doldurma.**<br>**Özellik Çıkarımı / Temsil: Mesaj başına 10 boyutlu öznitelik vektörü (normalize kimlik + 8 yük baytı + varış süresi). Bu vektörlerin N=100 mesajlık, %50 örtüşmeli kayan pencerelere gruplanması. Pencere, içinde en az bir enjekte edilmiş mesaj varsa saldırı olarak etiketlenir.**<br>**Model: İki model ailesi karşılaştırılacaktır.**<br>(i) Denetimli: 1B evrişimli sinir ağı (1D-CNN) tabanlı pencere sınıflandırıcı.<br>(ii) Denetimsiz: LSTM tabanlı otokodlayıcı; yalnızca normal trafikle eğitilir, yeniden oluşturma hatası anomali skoru olarak kullanılır ve eşik yalnızca eğitim verisinin skor dağılımından belirlenir.<br>**Eğitim: Sızıntıyı önlemek için eğitim/doğrulama/test ayrımı dosya düzeyinde yapılacaktır. Her yapılandırma en az 3 farklı rastgele tohumla eğitilecek ve sonuçlar ortalama ± standart sapma olarak raporlanacaktır.**<br>**Nicemleme ve Model Küçültme: Eğitilen her model için (a) float32 referans, (b) float16 nicemleme, (c) eğitim sonrası INT8 nicemleme (post-training quantization) ve (d) yapısal budama uygulanmış sürüm üretilecektir. Dönüşümler ONNX Runtime ve/veya TensorFlow Lite ile gerçekleştirilecektir.**<br>**Test ve Değerlendirme: Her model–nicemleme kombinasyonu, saldırı türüne göre ayrı test kümelerinde değerlendirilecek; başarım metrikleriyle birlikte çıkarım gecikmesi, model boyutu ve bellek kullanımı ölçülecektir. Gruplar arası farklar parametrik olmayan bir istatistiksel testle (Mann-Whitney U) karşılaştırılacaktır.** |
|---|

# 10. Baseline ve Karşılaştırma Yöntemleri

Projenin başarısını gösterebilmek için önerilen yöntem mümkün olduğunda en az bir baseline veya alternatif yöntemle karşılaştırılmalıdır.

**Baseline yöntem:**

| Nicemleme uygulanmamış float32 referans modeller: (i) 1D-CNN tabanlı denetimli sınıflandırıcı, (ii) LSTM otokodlayıcı tabanlı denetimsiz model. Ayrıca klasik makine öğrenmesi referansı olarak zamanlama ve mesaj kimliği özniteliklerine dayalı Random Forest / XGBoost sınıflandırıcı kullanılacaktır. |
|---|

**Önerilen yöntem:**

| Aynı modellerin float16, INT8 ve budanmış sürümleri; saldırı türüne duyarlı değerlendirme protokolü ile birlikte uç cihaz kısıtları altında ölçülmesi. |
|---|

**Karşılaştırılacak diğer yöntem(ler):**

| (1) Denetimli ve denetimsiz yaklaşımların nicemlemeye karşı dayanıklılığının karşılaştırılması.<br>(2) Veri setleri arası değerlendirme: Car-Hacking üzerinde eğitilen modelin ROAD üzerinde test edilmesi (farklı araç ve farklı toplama prosedürü altında genelleme).<br>(3) Literatürde raporlanan uç cihaz gecikme değerleriyle (ör. Rajapaksha vd., 2023) niteliksel karşılaştırma. |
|---|

# 11. Deney Planı

Projenin nasıl gerçekleştirileceğini adım adım açıklayınız. Aşağıdaki yapı örnek olarak kullanılabilir:

1. Veri setinin elde edilmesi

1. Verilerin incelenmesi ve ön işleme

1. Eğitim / doğrulama / test ayrımı

1. Baseline modelin uygulanması

1. Önerilen yöntemin uygulanması

1. Model eğitimi ve hiperparametrelerin belirlenmesi

1. Model testi

1. Sonuçların karşılaştırılması

1. Bulguların yorumlanması

1. …

1. …

1. …

| 1. Car-Hacking ve ROAD veri setlerinin indirilmesi, formatlarının incelenmesi ve saldırı etiketlerinin doğrulanması.<br>2. CAN protokolü ve veri yapısının incelenmesi; mesaj kimliği dağılımı, varış süresi istatistikleri ve saldırı türlerinin trafiğe etkisinin keşifsel analizi.<br>3. Ön işleme hattının kurulması: normalizasyon, varış süresi hesaplama, kayan pencere oluşturma ve pencere düzeyinde etiketleme.<br>4. Eğitim / doğrulama / test ayrımının dosya düzeyinde yapılması; test kümelerinin saldırı türüne göre ayrıştırılması.<br>5. Baseline modellerin (float32 1D-CNN, LSTM otokodlayıcı, ağaç tabanlı sınıflandırıcı) eğitilmesi ve başarımlarının saldırı türü bazında raporlanması.<br>6. Hiperparametrelerin yalnızca eğitim/doğrulama verisi üzerinden belirlenmesi; anomali eşiğinin test verisine bakılmadan hesaplanması.<br>7. Her modelin float16, INT8 ve budanmış sürümlerinin üretilmesi.<br>8. Tüm model–nicemleme kombinasyonlarının saldırı türü bazında test edilmesi; F1, yanlış alarm oranı ve ROC-AUC ölçümü.<br>9. Çıkarım gecikmesi, model boyutu ve bellek kullanımının ölçülmesi; ölçümlerin kaynak kısıtlı bir hedefte (Raspberry Pi sınıfı cihaz veya çekirdek/bellek sınırlaması uygulanmış konteyner) tekrarlanması.<br>10. Veri setleri arası genelleme deneyinin yürütülmesi.<br>11. Her yapılandırmanın en az 3 rastgele tohumla tekrarlanması; sonuçların ortalama ± standart sapma ile raporlanması ve istatistiksel testin uygulanması.<br>12. Bulguların yorumlanması: kaba ve gizli saldırılar arasındaki nicemleme kaynaklı kayıp farkının değerlendirilmesi.<br>13. Raporun ve sunumun hazırlanması. |
|---|

# 12. Değerlendirme Metrikleri

Probleme uygun performans ölçütlerini seçiniz ve neden uygun olduklarını kısaca açıklayınız.

| • Accuracy | • Precision |
|---|---|
| • Recall | • F1-score |
| • ROC-AUC | • MAE / MSE / RMSE |
| • R² | • Inference time |
| • Model size | • Memory consumption |
| • Energy consumption | • Communication overhead |
| • Latency<br>………. | ……<br>…… |
| **ROC-AUC: Denetimsiz modellerde eşikten bağımsız karşılaştırma sağlar.**<br>**Inference time / Latency (ms, pencere başına): Uç cihazda gerçek zamanlı çalışabilirliğin doğrudan ölçütüdür ve projenin ikinci ekseni olan gecikme ödünleşimini tanımlar.**<br>**Model size (MB): Nicemleme ve budamanın doğrudan çıktısıdır; gömülü donanımda depolama kısıtını temsil eder.**<br>**Memory consumption (MB): Kaynak kısıtlı cihazda çalıştırılabilirliğin belirleyicisidir.**<br>Not: Enerji tüketimi ölçümü uygun donanım erişimi sağlanırsa ek metrik olarak raporlanacaktır. |  |

# 13. Beklenen Sonuçlar

Projenin sonunda elde edilmesi beklenen sonuçları açıklayınız. Kesin sonuç ileri sürmek yerine test edilmesi beklenen hipotez veya öngörüyü belirtiniz.

| Örnek: “Önerilen yöntemin baseline modele göre F1-score açısından daha yüksek performans göstermesi beklenmektedir.” |
|---|
| **H1: Nicemleme seviyesi arttıkça (float32 → float16 → INT8) toplam F1-skorundaki düşüş sınırlı kalacak; ancak bu düşüş saldırı türleri arasında eşit dağılmayacaktır.**<br>**H2: Gizli saldırılarda (spoofing, malfunction, masquerade) ölçülen F1-skoru kaybının, kaba saldırılardakinden (DoS, fuzzing) anlamlı biçimde büyük olması beklenmektedir. DoS ve fuzzing saldırılarının INT8 nicemleme altında dahi yüksek başarımla tespit edilmeye devam etmesi öngörülmektedir.**<br>**H3: Float16 nicemlemenin, başarım kaybı ile gecikme/bellek kazancı arasında en dengeli noktayı sunması; INT8 nicemlemenin ise en yüksek kazancı sağlarken gizli saldırılarda belirgin kayba yol açması beklenmektedir.**<br>**H4: Denetimsiz (yeniden oluşturma tabanlı) modellerin nicemlemeye karşı denetimli sınıflandırıcılardan daha duyarlı olması beklenmektedir; çünkü anomali skoru küçük yeniden oluşturma hatası farklarına dayanmakta ve sayısal çözünürlük kaybı bu farkları bastırabilmektedir.**<br>Hipotezlerin aksi yönde sonuçlanması da anlamlı bir bulgu oluşturacaktır: bu durum, uç cihaz için agresif nicemlemenin güvenlik açısından güvenli bir tercih olduğunu gösterecektir. |

# 14. Projenin Beklenen Katkısı

Projenin sağlayacağı bilimsel veya teknik katkıyı belirtiniz. Katkı yeni bir yöntem, mevcut yöntemlerin karşılaştırılması, farklı bir IoT problemine uygulama, daha hafif model, daha düşük maliyet, Edge uyumluluğu, performans iyileştirmesi veya daha kapsamlı deneysel analiz biçiminde olabilir.

| Projenin katkısı yeni bir mimari önermek değil, mevcut yöntemlerin uç cihaz koşullarında daha kapsamlı ve daha dürüst biçimde değerlendirilmesidir. Beklenen katkılar:<br>1) Saldırı türüne duyarlı değerlendirme protokolü: Nicemlemenin bedelinin toplam doğruluk yerine saldırı türü bazında raporlanması gerektiğini gösteren sistematik bir analiz. İncelenen çalışmalarda bu ayrıştırma yapılmamıştır.<br>2) Pratik tasarım rehberi: Araç içi IDS geliştiricileri için, belirlenen gecikme ve bellek bütçesi altında hangi model–nicemleme kombinasyonunun tercih edilmesi gerektiğine dair deneysel olarak desteklenmiş bir öneri seti.<br>3) Güvenlik açısından bir uyarı niteliği: Eğer hipotez doğrulanırsa, uç cihaz optimizasyonunun en kritik saldırı sınıflarında sessiz bir başarım kaybına yol açtığı gösterilmiş olacaktır. Bu bulgu, UNECE R155 kapsamında konuşlandırılacak sistemlerin doğrulama süreçleri açısından doğrudan pratik değer taşır.<br>4) Tekrarlanabilirlik: Deney kodunun ve ön işleme hattının açık biçimde paylaşılması; iki farklı veri seti üzerinde çoklu tohumla doğrulanmış sonuçlar. |
|---|

# 15. Kullanılacak Araçlar

Kullanılması planlanan yazılım, kütüphane ve platformları belirtiniz (örn. Python, Google Colab, Jupyter Notebook, TensorFlow, Keras, PyTorch, Scikit-learn, Pandas, NumPy, MATLAB).

| **Programlama dili: Python 3.11**<br>**Derin öğrenme: PyTorch, TensorFlow / Keras**<br>**Model dönüşümü ve nicemleme: ONNX, ONNX Runtime, TensorFlow Lite (post-training quantization), PyTorch quantization API**<br>**Klasik makine öğrenmesi: Scikit-learn, XGBoost**<br>**Veri işleme ve analiz: Pandas, NumPy, SciPy (istatistiksel testler)**<br>**Görselleştirme: Matplotlib, Seaborn**<br>**Geliştirme ortamı: Jupyter Notebook, Google Colab, yerel GPU iş istasyonu**<br>**Uç cihaz ölçümü: Raspberry Pi 4 sınıfı cihaz (erişim sağlanabilirse); aksi hâlde CPU çekirdeği ve bellek sınırlaması uygulanmış Docker konteyneri**<br>**Sürüm kontrolü: Git / GitHub** |
|---|

# 16. Riskler ve Alternatif Plan

Çalışma sırasında karşılaşılabilecek olası problemleri ve alternatif çözümleri açıklayınız.

| **Olası Risk** | **Alternatif Plan** |
|---|---|
| Uç cihaz donanımına (Raspberry Pi vb.) erişim sağlanamaması; gecikme ve bellek ölçümlerinin gerçek donanımda yapılamaması. | Ölçümler, CPU çekirdek sayısı ve bellek sınırlaması uygulanmış bir Docker konteynerinde tekrarlanabilir biçimde yapılır. Mutlak değerler yerine göreli karşılaştırma (float32 referansına göre oran) raporlanır; bu, araştırma sorusunu cevaplamak için yeterlidir. |
| ROAD veri setine erişimde gecikme veya kullanım izni sorunu yaşanması. | Çalışma Car-Hacking veri seti ve açık erişimli Survival Analysis / SynCAN veri setleriyle sürdürülür. Gizli saldırı temsili için spoofing ve malfunction sınıfları kullanılır; maskeleme saldırısı kapsam dışına alınarak araştırma sorusu daraltılır. |
| INT8 nicemleme sonrası bazı model katmanlarının desteklenmemesi veya dönüşümün başarısız olması. | Katman bazlı seçmeli nicemleme (hybrid quantization) uygulanır; desteklenmeyen katmanlar float olarak bırakılır ve bu durum sonuçlarda açıkça belirtilir. Alternatif olarak nicemleme farkındalıklı eğitim (QAT) denenir. |
| LSTM otokodlayıcı eğitiminin kararsız olması veya anlamlı anomali skoru üretememesi. | Daha kararlı olan 1D evrişimli otokodlayıcıya geçilir. Denetimsiz eksen tamamen sorun çıkarırsa çalışma denetimli modeller üzerinde daraltılır; bu durumda AS-3 kapsam dışına alınır. |
| Deney sayısının dönem içinde tamamlanamaması. | Nicemleme seviyeleri önce üç (float32, float16, INT8) ile sınırlandırılır; budama ve veri setleri arası genelleme deneyi ikincil öncelik olarak planlanır ve zaman kalırsa yürütülür. |

# 17. Ön Çalışma Takvimi

| **Aşama** | **Planlanan Tarih / Hafta** |
|---|---|
| Literatür araştırması | 1 – 19 Ekim 2026 (Hafta 1–3) |
| Veri setinin hazırlanması | 13 – 26 Ekim 2026 (Hafta 3–4) |
| Baseline uygulaması | 27 Ekim – 9 Kasım 2026 (Hafta 5–6) |
| Önerilen yöntemin uygulanması | 10 – 23 Kasım 2026 (Hafta 7–8) |
| Deneylerin gerçekleştirilmesi | 24 Kasım – 7 Aralık 2026 (Hafta 9–10) |
| Sonuçların analizi | 8 – 14 Aralık 2026 (Hafta 11) |
| Raporun hazırlanması | 15 – 22 Aralık 2026 (Hafta 12) |
| Final sunumu | **23 Aralık** **Çarşamba** |

# 18. Kaynakça

En az 5 bilimsel kaynak(proje konusu çerçevesinde daraltıp elediğiniz nihai makaleler) uygun akademik kaynak gösterme biçiminde verilmelidir. Halisünasyon olmadığına dikkat ediniz. Google scholarda bulduğunuz bir makaleyi “ Alıntı yap” kısmında “APA” formatını kopyala/yapıştır olarak yapabilirsiniz.

| Hanselmann, M., Strauss, T., Dormann, K., & Ulmer, H. (2020). CANet: An unsupervised intrusion detection system for high dimensional CAN bus data. IEEE Access, 8, 58194–58205.<br>Oualil, S., Ait Yahia, I., El Kamili, M., Fardousse, K., & Berrada, I. (2026). CAN-TEMPO: Unsupervised CAN bus intrusion detection via temporal multi-period oscillation encoding. Technologies, 14(6), 375. https://doi.org/10.3390/technologies14060375<br>Rajapaksha, S., Kalutarage, H., Al-Kadri, M. O., Petrovski, A., & Madzudzo, G. (2023). Improving in-vehicle networks intrusion detection using on-device transfer learning. Inaugural International Symposium on Vehicle Security and Privacy (VehicleSec 2023). https://doi.org/10.14722/vehiclesec.2023.23088<br>Song, H. M., Woo, J., & Kim, H. K. (2020). In-vehicle network intrusion detection using deep convolutional neural network. Vehicular Communications, 21, 100198.<br>Verma, M. E., Bridges, R. A., Iannacone, M. D., Hollifield, S. C., Moriano, P., Hespeler, S. C., Kay, B., & Combs, F. L. (2024). A comprehensive guide to CAN IDS data and introduction of the ROAD dataset. PLOS ONE, 19(1), e0296879. https://doi.org/10.1371/journal.pone.0296879 |
|---|

# Proje Önerisinin Kabul Edilebilmesi İçin Asgari Şartlar

- IoT ile açık ve doğrudan bir ilişki bulunması

- Belirli ve sınırlandırılmış bir problem tanımlanması

- Araştırma sorusunun oluşturulması

- En az 3 bilimsel çalışmanın incelenmesi

- Kullanılacak veri veya veri setinin belirlenmesi

- Uygulanacak yöntemin açıklanması

- En az bir baseline / karşılaştırma yaklaşımının düşünülmesi

- Değerlendirme metriklerinin belirtilmesi

- Uygulanabilir bir deney planının hazırlanması

| “IoT’de Yapay Zekâ”, “Akıllı Ev”, “IoT Güvenliği” veya “Derin Öğrenme ile IoT” gibi geniş başlıklar tek başına proje konusu olarak yeterli değildir. Genel alan, spesifik ve ölçülebilir bir araştırma problemine dönüştürülmelidir. |
|---|
