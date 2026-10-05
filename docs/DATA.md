# Veri Setleri

Bu dosya projede kullanılan ham veri setlerinin kaynağını, kullanım koşulunu, atıf bilgisini,
dosya formatını ve bütünlük özetlerini kaydeder. Ham veri depoya eklenmez (`/data/` yoksayılır);
`python -m src.data.download all` ile yeniden indirilir, `python -m src.data.loaders` ile ortak
şemaya dönüştürülür. Bütünlük özetleri `data/raw/manifest.csv`'den üretilmiştir (indirme tarihi 2026-10-05).

## 1. Car-Hacking Dataset (HCRL, Korea University)

- **Sayfa:** https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
- **İndirme:** sayfadaki herkese açık Dropbox klasörü (kayıt/form yok); klasör `dl=1` ile tek zip
  olarak indirildi (sunucu adı: `9) Car-Hacking Dataset.zip`, yerel adı `car_hacking_dataset.zip`).
  Yayıncı bir sağlama özeti yayımlamıyor; SHA-256 aşağıda.
- **Kullanım koşulu:** sayfaya göre veri seti akademik amaçla kullanıma açıktır ve kullanımda aşağıdaki
  makalelere atıf istenir. Yeniden dağıtım izni belirtilmemiştir; veri depoya konmaz.
- **Atıf:**
  - Song, H. M., Woo, J., & Kim, H. K. (2020). In-vehicle network intrusion detection using deep
    convolutional neural network. *Vehicular Communications*, 21, 100198.
  - Seo, E., Song, H. M., & Kim, H. K. (2018). GIDS: GAN based intrusion detection system for
    in-vehicle network. *16th Annual Conference on Privacy, Security and Trust (PST)*, IEEE.
    (Sayfa, PST sürümündeki Tablo 2 hatalarının arXiv sürümünde düzeltildiğini belirtiyor.)

```bibtex
@article{song2020invehicle,
  title   = {In-vehicle network intrusion detection using deep convolutional neural network},
  author  = {Song, Hyun Min and Woo, Jiyoung and Kim, Huy Kang},
  journal = {Vehicular Communications},
  volume  = {21},
  pages   = {100198},
  year    = {2020}
}
@inproceedings{seo2018gids,
  title     = {{GIDS}: {GAN} based Intrusion Detection System for In-Vehicle Network},
  author    = {Seo, Eunbi and Song, Hyun Min and Kim, Huy Kang},
  booktitle = {2018 16th Annual Conference on Privacy, Security and Trust (PST)},
  year      = {2018},
  publisher = {IEEE}
}
```

### Format (dosyalar incelenerek doğrulandı)

- Saldırı dosyaları (`DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv`):
  başlıksız CSV, `Timestamp, CAN ID (hex), DLC, DATA[0..DLC-1] (hex), Flag`. **DLC < 8 olan satırlarda
  sütun sayısı azalır** (DLC 2 → 6 alan, DLC 5 → 9 alan, DLC 6 → 10 alan); bayrak her zaman son alandadır
  (`3 + DLC` konumu). Bayrak `T` = enjekte, `R` = normal. Zaman damgaları monoton artan.
- Normal dosya (`normal_run_data/normal_run_data.txt`, `normal_run_data.7z` içinden de gelir):
  `Timestamp: <ts>  ID: <hex>  000  DLC: <n>  <b0> ... <bn-1>` biçiminde boşlukla ayrılmış metin.
  **Son satır kesik** (yalnızca `Timestamp: 1479121941.287013`); okuyucu bu tek satırı atar.
- Ham dosyadan sayılan değerler (okuyucu çıktısıyla `tests/test_loaders.py` içinde karşılaştırılır):

| Dosya | Satır | Enjekte (T) | Enjekte ID |
|---|---:|---:|---|
| DoS_dataset.csv | 3,665,771 | 587,521 | 0x000 |
| Fuzzy_dataset.csv | 3,838,860 | 491,847 | rastgele (çok sayıda) |
| gear_dataset.csv | 4,443,142 | 597,252 | 0x43F |
| RPM_dataset.csv | 4,621,702 | 654,897 | 0x316 |
| normal_run_data.txt | 988,871 (+1 kesik) | 0 | — |

HCRL sayfası mesaj sayısı tablosunu metin olarak yayımlamıyor; tutarlılık testi ham dosyadan bağımsız
yeniden sayıma dayanır.

## 2. ROAD — Real ORNL Automotive Dynamometer CAN Intrusion Dataset

- **Sayfa:** https://0xsam.com/road/ · Zenodo: https://zenodo.org/records/10462796 (DOI 10.5281/zenodo.10462796)
- **İndirme:** `road.zip` (556,718,276 bayt), form/kayıt yok. Zenodo'nun yayımladığı MD5
  `cab184cfc2fe12c0834bc46188c0f330` indirme sonrası doğrulandı.
- **Kullanım koşulu:** Creative Commons Attribution 4.0 (CC-BY-4.0).
- **Atıf:** Verma, M. E., Bridges, R. A., Iannacone, M. D., Hollifield, S. C., Moriano, P., Hespeler,
  S. C., Kay, B., & Combs, F. L. (2024). A comprehensive guide to CAN IDS data and introduction of the
  ROAD dataset. *PLOS ONE*, 19(1), e0296879. (Arşivdeki readme arXiv:2012.14600 sürümüne atıf istiyor.)

```bibtex
@article{verma2024road,
  title   = {A comprehensive guide to {CAN IDS} data and introduction of the {ROAD} dataset},
  author  = {Verma, Miki E. and Bridges, Robert A. and Iannacone, Michael D. and Hollifield, Samuel C.
             and Moriano, Pablo and Hespeler, Steven C. and Kay, Bill and Combs, Frank L.},
  journal = {PLOS ONE},
  volume  = {19},
  number  = {1},
  pages   = {e0296879},
  year    = {2024},
  doi     = {10.1371/journal.pone.0296879}
}
@misc{verma2020road,
  title  = {{ROAD}: The Real {ORNL} Automotive Dynamometer Controller Area Network Intrusion Detection Dataset},
  author = {Verma, Miki E. and Iannacone, Michael D. and Bridges, Robert A. and Hollifield, Samuel C.
            and Kay, Bill and Combs, Frank L.},
  year   = {2020},
  eprint = {2012.14600},
  archivePrefix = {arXiv}
}
```

### Format ve etiketleme (readme, metadata ve veri incelenerek doğrulandı)

- Yapı: `road/ambient/*.log` (12 kayıt), `road/attacks/*.log` (33 kayıt), her klasörde
  `capture_metadata.json`; ayrıca `signal_extractions/` (DBC ile çözülmüş sinyal CSV'leri — bu projede
  kullanılmıyor) ve macOS arşiv artıkları (`__MACOSX/`, `.DS_Store`; manifest dışı).
- Log satırı (SocketCAN candump): `(SSSSSSSSSS.UUUUUU) can0 III#PPPPPPPPPPPPPPPP` — tüm satırlar 46 bayt,
  ID 3 hex hane, yük her zaman 8 bayt. Zaman damgaları anonimleştirilmiş: her kayıt yuvarlak bir değerden
  başlar (ör. `1030000000.000000`), masquerade kaydının başlangıcı kaynak kaydından **farklıdır**.
- Metadata alan adları readme'den farklıdır: gerçek dosyada `injection_id`, `injection_data_str`,
  `injection_interval`, `modified`, `elapsed_sec`, `on_dyno`, `description`.
  - `injection_interval`: kaydın ilk çerçevesinden itibaren saniye (ilk-son çerçeve farkı her kayıtta
    `elapsed_sec` ile birebir aynı).
  - `injection_data_str`: 16 hex hane, `X` = değiştirilmeyen yarım bayt.
  - Fuzzing: `injection_id = "XXX"`, `injection_data_str = "FF…FF"`; enjekte çerçeveler yükü FF×8 olan,
    her biri farklı ID taşıyan çerçevelerdir. FF×8 yükü 12 ambient kaydın hiçbirinde geçmez.
  - Accelerator: `injection_id`, `injection_data_str`, `injection_interval` = null. Readme'ye göre
    enjeksiyon kayıttan önce yapılmış; kayıtta enjekte çerçeve yoktur → çerçeve düzeyinde etiket yok.
- Okuyucunun etiket kuralı (K-006): hedefli saldırılarda `ID == injection_id ∧ yük desene uyar ∧
  zaman aralık içinde (uçlar dahil, mikro-saniye tamsayı ile)`; fuzzing'de `yük == FF×8 ∧ zaman ≥
  enjeksiyon başlangıcı` (fuzzing_attack_2 ve _3'te metadata bitişinden < 6 ms sonra birer FF×8 çerçevesi
  daha var). Hedefli fabrication kayıtlarında aralık içinde hedef ID hızı tam iki katına çıkar
  (100 → 200 çerçeve/s; coolant 10 → 20) ve hedef ID çerçevelerinin tam yarısı desene uyar.
- **Masquerade kayıtları türetilmiştir** (K-007): her `*_masquerade.log`, aynı adlı fabrication
  kaydından aralık içindeki çakışan meşru hedef-ID çerçeveleri silinerek elde edilmiştir. Veriyle
  doğrulandı: göreli zaman + ID + yük üzerinden masquerade kaydının çerçevelerinin %100'ü kaynakta
  bulunur; silinen çerçevelerin tamamı hedef ID'li, aralık içinde ve desene uymayan (meşru) çerçevelerdir.
  Ayrıntı: `results/tables/capture_lineage.csv`.

## 3. Dosya listesi ve SHA-256 özetleri

#### car_hacking

| Tür | Dosya (data/raw/ altında) | Boyut (bayt) | SHA-256 |
|---|---|---:|---|
| arşiv | `car_hacking_dataset.zip` | 953,365,680 | `c47c139142e83965fc74e416d76eee76dc8de48b3d6c6b639e44b663484c8de9` |
| açılmış | `car_hacking/DoS_dataset.csv` | 190,058,708 | `09bf3136647b1d846d13c374f26208f06ffeba18369093fc57c8076fc56b795e` |
| açılmış | `car_hacking/Fuzzy_dataset.csv` | 198,520,767 | `a0e78dff6365a1ad3d314973f1ac5879b57f9b80c40eb07b1e3bbe4aeddaecfe` |
| açılmış | `car_hacking/RPM_dataset.csv` | 239,581,936 | `6718c6cb0a5fe6bdf0e966a680f1d6d7d79c063d603254d6118c16aaeb74974d` |
| açılmış | `car_hacking/gear_dataset.csv` | 230,320,414 | `b4eba43605fff337117ff30e13c33f63b56eec3b00927c8bccb31192df3bee23` |
| açılmış | `car_hacking/normal_run_data.7z` | 7,510,943 | `ea326338163c613e77739d8fea372fb252e875c68e35f70c744455e3711ffa15` |
| açılmış | `car_hacking/normal_run_data/normal_run_data.txt` | 87,371,776 | `efbbafbf8e965c0d2f75eb97f1d118d0ae1c4b9f3077a6c4c000a26eb5be97ea` |

#### road

| Tür | Dosya (data/raw/ altında) | Boyut (bayt) | SHA-256 |
|---|---|---:|---|
| arşiv | `road.zip` | 556,718,276 | `0e4fe6ed7f99b5cdabf6b772c91a2025c614a071ba8579c532904d0ef7aea5f6` |
| açılmış | `road/road/ambient/ambient_dyno_drive_basic_long.log` | 137,598,512 | `38b6cf1b806dbff515ce31f287d445ae6148db513d58c79509b2311d62bcffe7` |
| açılmış | `road/road/ambient/ambient_dyno_drive_basic_short.log` | 48,926,704 | `c9f65e0050dfbbce10105a48d7da6aae9facb1d3d499b8968262c29cff9612bb` |
| açılmış | `road/road/ambient/ambient_dyno_drive_benign_anomaly.log` | 35,489,874 | `2578f97340613b93b62a222769332019e78e4014278840c5e569f6b395b4d10d` |
| açılmış | `road/road/ambient/ambient_dyno_drive_extended_long.log` | 65,513,568 | `f06d22c9f67ce5bf42945441b9b0f48b677a4fc3004cb67147cffb1a58c72788` |
| açılmış | `road/road/ambient/ambient_dyno_drive_extended_short.log` | 36,360,332 | `ddc11a7ed5dcb85b7de063c609638b123eaf59b52939da55ea8a0cbfe6fe24b0` |
| açılmış | `road/road/ambient/ambient_dyno_drive_radio_infotainment.log` | 42,911,192 | `8e06679eaa54d627a9346ab400a6fc51bba9313ed3f50883aeb4ca9ad0365ff4` |
| açılmış | `road/road/ambient/ambient_dyno_drive_winter.log` | 5,250,808 | `29868df8bdab5380ed7b643ebc4bcfdd6c11cca3486830b7b326ff4745433894` |
| açılmış | `road/road/ambient/ambient_dyno_exercise_all_bits.log` | 203,293,412 | `710d0deff388686b05765573b36bccdbbc175fa693caafd6f18fadec0f226cd7` |
| açılmış | `road/road/ambient/ambient_dyno_idle_radio_infotainment.log` | 72,326,628 | `a410eb564cab15a3de84f97f4b8d84753c7fcaaed24da7eb05074ac7ef4d800c` |
| açılmış | `road/road/ambient/ambient_dyno_reverse.log` | 5,660,162 | `c21004cd476155bde8459d7a96b87a8eec996d7410d294f51ee66c3217a13001` |
| açılmış | `road/road/ambient/ambient_highway_street_driving_diagnostics.log` | 51,738,546 | `95ccdc033aadf9156dc055f09ede2ca849b7c69a889c61c74107016c4956fa54` |
| açılmış | `road/road/ambient/ambient_highway_street_driving_long.log` | 405,808,136 | `5d4d2715dfac8089fdea4e3bfd39609f1e863336657379a705c9ce790efcfc76` |
| açılmış | `road/road/ambient/capture_metadata.json` | 2,995 | `a2fb03f861c241e2dfb564fff92006917e748d11a874af261aad5918c161f149` |
| açılmış | `road/road/attacks/accelerator_attack_drive_1.log` | 9,418,960 | `f25657c7b5b792299d5ff0718fd74192b43ce970daa1e6b12dba0286c27a63c4` |
| açılmış | `road/road/attacks/accelerator_attack_drive_2.log` | 7,909,056 | `b17381d3915b7fe2518866f7b7d293abe280d503ba2fa46e8517e8a365385985` |
| açılmış | `road/road/attacks/accelerator_attack_reverse_1.log` | 9,312,562 | `ad7fdd209587bfb5530d43ac12b9b7fcaa67322244755811aa9bb2be9387afe7` |
| açılmış | `road/road/attacks/accelerator_attack_reverse_2.log` | 11,472,998 | `584a5c102e3cb23a306978f76aef91ea3a7bd5fd5c8ba3493d211c43914b287b` |
| açılmış | `road/road/attacks/capture_metadata.json` | 15,577 | `5340b274dc54894950140d14b0d42ad475800454405e87198117246fe6220fe8` |
| açılmış | `road/road/attacks/correlated_signal_attack_1.log` | 3,738,052 | `46f1c4ecd3392515f8e9b4d6ef8d7ebc5890f6695258dc45ece896ad99cd06f1` |
| açılmış | `road/road/attacks/correlated_signal_attack_1_masquerade.log` | 3,642,096 | `6b6a32c590f766fe338d29840679b5e6aac44c9b193430cdceda06d80e2d4c7e` |
| açılmış | `road/road/attacks/correlated_signal_attack_2.log` | 3,204,222 | `d0f659b172554110e9b67f9951ac478d67ce37481efd6a2dd9ef53b8e305e421` |
| açılmış | `road/road/attacks/correlated_signal_attack_2_masquerade.log` | 3,105,782 | `53cb0ba23ee6a36ae63b731d924a10402a5b28974f746af4aaa801322f3f70eb` |
| açılmış | `road/road/attacks/correlated_signal_attack_3.log` | 1,924,134 | `ef8c97a6a4bd6eaaf14dfc6ff5908c38c02676d3a384e9bcf5b857fe71866110` |
| açılmış | `road/road/attacks/correlated_signal_attack_3_masquerade.log` | 1,865,990 | `e7053700495a18a67882c7666a0344bb0d43e0743744d42e3f502c7a0962e490` |
| açılmış | `road/road/attacks/fuzzing_attack_1.log` | 2,269,732 | `04173a8403e8fc9996d3228889685b93f8327240fcbb84965e64ed853b3d39bc` |
| açılmış | `road/road/attacks/fuzzing_attack_2.log` | 1,488,146 | `99f9ca4c9ebb0fbdebb1bedd3c0ff51869c9efeb8aba6dd8e8006083b7bd9e4f` |
| açılmış | `road/road/attacks/fuzzing_attack_3.log` | 608,948 | `53a67622b04e0ff0c63ca10b567a8ee31f17effbb96af8d369b8bbaadc0040aa` |
| açılmış | `road/road/attacks/max_engine_coolant_temp_attack.log` | 2,848,458 | `0f907546586b3f9401772df4d7642e995f30e764baa86f0aec46b72598e2b541` |
| açılmış | `road/road/attacks/max_engine_coolant_temp_attack_masquerade.log` | 2,846,526 | `6a6f3e419963b2118a9cc116021e62ad22c4138497aec0840b30f47320595c73` |
| açılmış | `road/road/attacks/max_speedometer_attack_1.log` | 9,816,262 | `6d36e0985dc7e962f600f328d2056d013f759df8a4daeb2db1505274fcecaca1` |
| açılmış | `road/road/attacks/max_speedometer_attack_1_masquerade.log` | 9,703,838 | `9905b3b4305c86f1562659e48327aa9ba471941078dcbf93aea4f001620e20a6` |
| açılmış | `road/road/attacks/max_speedometer_attack_2.log` | 6,711,124 | `20e8877307407de41b66c84d29205e99329590cbf47c1894ffe4713bd29d68dd` |
| açılmış | `road/road/attacks/max_speedometer_attack_2_masquerade.log` | 6,566,684 | `e0fdb8078d2aa9ce26cae34d2dbe216aab654cb77eddfb4917b41f0f574298e3` |
| açılmış | `road/road/attacks/max_speedometer_attack_3.log` | 9,823,346 | `dff0445e7941722227668865e10947e6c7a7bb15d1bf39b9425de4347288bd10` |
| açılmış | `road/road/attacks/max_speedometer_attack_3_masquerade.log` | 9,542,424 | `f8bea0945d47fcbcdea26dcab38845274a20ba17698eb417e68bb3ba57863388` |
| açılmış | `road/road/attacks/reverse_light_off_attack_1.log` | 3,123,492 | `0e253c574f6c8ba2a03c39f7fbbdb2c450bf2c4c072ac817f22c961d32996560` |
| açılmış | `road/road/attacks/reverse_light_off_attack_1_masquerade.log` | 3,092,580 | `4cd8616e2bfd0623b99a72e63876dc966d0c3a29252a2ac4cba736d0bfb0398b` |
| açılmış | `road/road/attacks/reverse_light_off_attack_2.log` | 4,582,888 | `e2247262c4ff7b2d3f9b509916c8c18487a4e5dc8ee0c282ba5c7da0d321ba48` |
| açılmış | `road/road/attacks/reverse_light_off_attack_2_masquerade.log` | 4,473,822 | `e576b6a7a608afebcd6438d92ea0e9f8314936d676a4e46e7700e4cbefd17771` |
| açılmış | `road/road/attacks/reverse_light_off_attack_3.log` | 6,479,330 | `b359e592f92775a9f10540b83f29a2e3de89e77b6828dee698fec3f9a8ee29ad` |
| açılmış | `road/road/attacks/reverse_light_off_attack_3_masquerade.log` | 6,367,366 | `19d02cc4c85c22aaab8e80ee700e0b139cc888c3232b2bd7c371c0877d6a9582` |
| açılmış | `road/road/attacks/reverse_light_on_attack_1.log` | 6,128,902 | `c5c34833e27241ced63b0b6e074d80651897a505f8f4fd6d005f859432384c82` |
| açılmış | `road/road/attacks/reverse_light_on_attack_1_masquerade.log` | 6,037,316 | `9a1d6907104e4a94495552818b8d8aaf0852c8ffc58c12b2f602ee5ee1bffd1e` |
| açılmış | `road/road/attacks/reverse_light_on_attack_2.log` | 8,093,470 | `1f8bb86b754127e61149cecee33ce924a66f1d7e3f7b06890715c240c6843fd3` |
| açılmış | `road/road/attacks/reverse_light_on_attack_2_masquerade.log` | 7,923,776 | `c15e11e83895deeaecf904ca0757efb0acd84053f58637d6d7029c6f58344da1` |
| açılmış | `road/road/attacks/reverse_light_on_attack_3.log` | 7,178,392 | `8ad277b7b19e66a4744f8309c02e4c7a9c149ce4c3b141f557780baefe9dcb8e` |
| açılmış | `road/road/attacks/reverse_light_on_attack_3_masquerade.log` | 7,070,246 | `4fbbce65cbe81c5fd762dd0440486a5dc0bb53cff3f29b408ab798236ba9c6e1` |
| açılmış | `road/road/data_table.csv` | 723 | `2d29e180cb290e84679d514015664f00ec6524f2763bc8ac001ed739ddb7d6d6` |
| açılmış | `road/road/readme.md` | 5,498 | `00764a90086aec19f75381377f0e62e92073fabe50824ff6877543b67a892f9c` |
| açılmış | `road/road/signal_extractions/DBC/anonymized.dbc` | 30,785 | `84b5e72b604d79f360a0fcb107ef84014e6fa77f36ea2ba03895e9f549eb95b9` |
| açılmış | `road/road/signal_extractions/ambient/accelerator_attack_drive_1.csv` | 12,877,037 | `20744e81bbdb58292295243f81a832e150a197fbe443d1f1e7281c1653409661` |
| açılmış | `road/road/signal_extractions/ambient/accelerator_attack_drive_2.csv` | 10,817,267 | `9908389d069c410f7734619ba48031527bc48e231a5c1176fa7e7e41ef38f0e9` |
| açılmış | `road/road/signal_extractions/ambient/accelerator_attack_reverse_1.csv` | 12,705,269 | `be247bf7e2f58d30386f62eba1e2e00eab802f18ff097deee825958817335cfc` |
| açılmış | `road/road/signal_extractions/ambient/accelerator_attack_reverse_2.csv` | 15,701,591 | `95ce09b7de3cff8545a07889969dbbfd33179f88b1ea323e6b28c8fdcd1adb74` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_basic_long.csv` | 188,101,947 | `fc4af90420e5ea02328a2d958db5a72e0e8f283859e60f70f2f5702e0723a2ce` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_basic_short.csv` | 67,058,935 | `764d3e3963ca3a3e61447944166f724d8a82ec64c3d25e89b16db88c871be5ab` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_benign_anomaly.csv` | 47,781,360 | `91778c539da53189c91923e9df91317a0d104c6b98e97681172023d9e13f40ee` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_extended_long.csv` | 89,366,969 | `3b69fb021476aab7948a6c88afc495441b533da1c0a6a03df721c7d27e7faf62` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_extended_short.csv` | 49,650,718 | `63c7bdb1d403559850da97e3d7e01190030e22c618b4b39819bee020d828af92` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_radio_infotainment.csv` | 58,362,767 | `00be3c7562863dd6fc1a66fbfeead12a6d3bec1c316a4c48c121b00d899d493d` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_drive_winter.csv` | 7,179,274 | `d422e3f0aeb105492e3cfef0469ce1b932aaa1cb51b75b384d0abe25a003c27d` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_exercise_all_bits.csv` | 275,877,891 | `8870e18beec66ef189cdf21ef3a223063f444d56c39f709897e1ab8c84f99e63` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_idle_radio_infotainment.csv` | 97,350,804 | `8d7762a24fbb1607a08af45bfe221b46981fb2daabef570801f798c02553d61c` |
| açılmış | `road/road/signal_extractions/ambient/ambient_dyno_reverse.csv` | 7,699,053 | `a80e2d6beaccfecdca6b0fc46accd26fd8e8660e1a074ffbf8f3e2c08379c404` |
| açılmış | `road/road/signal_extractions/ambient/ambient_highway_street_driving_diagnostics.csv` | 70,991,688 | `a591305781f57e09f323b996a4aae09133568b36c983e6798d0d1e01807fb172` |
| açılmış | `road/road/signal_extractions/ambient/ambient_highway_street_driving_long.csv` | 556,424,225 | `5aa9053ac44d127e5cea0e20045d19b031b75c02758ddebc81c705d15f303ef0` |
| açılmış | `road/road/signal_extractions/ambient/correlated_signal_attack_1_masquerade.csv` | 4,974,953 | `0e03e5b74ccc2e26a7dd05984749c09bb7e247894ba0dea795603514045c350f` |
| açılmış | `road/road/signal_extractions/ambient/correlated_signal_attack_2_masquerade.csv` | 4,255,996 | `9ab5791efd48ebbc4ceea53f2f0fac911e4e4788d570bb0be580716cf2314820` |
| açılmış | `road/road/signal_extractions/ambient/correlated_signal_attack_3_masquerade.csv` | 2,550,755 | `56ee6ff7a4ce39c1322261df39fa27f63f6bee579e0f40de0b3c7701b4fc8e18` |
| açılmış | `road/road/signal_extractions/ambient/max_engine_coolant_temp_attack_masquerade.csv` | 3,873,588 | `1234d16773e394b01507473e8f0524275b282d2c3eb75f424626aa13cdb78279` |
| açılmış | `road/road/signal_extractions/ambient/max_speedometer_attack_1_masquerade.csv` | 13,171,552 | `385c51df86417cc62817a1f4a31770ebba1d732cfaf8d471a1cc79ae260a3cae` |
| açılmış | `road/road/signal_extractions/ambient/max_speedometer_attack_2_masquerade.csv` | 8,971,951 | `2331683a010be5b5034f07c95e9ec8a365baf93ac6fdca31865d0bac4bb8f72b` |
| açılmış | `road/road/signal_extractions/ambient/max_speedometer_attack_3_masquerade.csv` | 12,986,141 | `129baa9346a4f61718bb9c12984b9f10c9758b4ff3d8dbe937c816edef6c3826` |
| açılmış | `road/road/signal_extractions/ambient/metadata.json` | 2,995 | `a2fb03f861c241e2dfb564fff92006917e748d11a874af261aad5918c161f149` |
| açılmış | `road/road/signal_extractions/ambient/reverse_light_off_attack_1_masquerade.csv` | 4,211,797 | `7063d4621fd3e78ba5704c666f325809df77e429c570043618212c7e08f3aea0` |
| açılmış | `road/road/signal_extractions/ambient/reverse_light_off_attack_2_masquerade.csv` | 6,092,391 | `0f05ad82d1371f1cb20e6a6564bba4a6cdbbac8af5014f577c0172fcf72216a1` |
| açılmış | `road/road/signal_extractions/ambient/reverse_light_off_attack_3_masquerade.csv` | 8,658,429 | `ff45324f9e2cb0a20dc5e20794a02095c61fe4913d16146cd0dbe667b95c01e2` |
| açılmış | `road/road/signal_extractions/ambient/reverse_light_on_attack_1_masquerade.csv` | 8,202,799 | `da1f73e5a8c0bbfb5d95185f06ab15ebbc3753a84191b00be84d70cf42e415e8` |
| açılmış | `road/road/signal_extractions/ambient/reverse_light_on_attack_2_masquerade.csv` | 10,788,136 | `fc47e8cd595c8adab79aff7053ae13b10d81f0014cfc33d4807332182f4aaddc` |
| açılmış | `road/road/signal_extractions/ambient/reverse_light_on_attack_3_masquerade.csv` | 9,640,775 | `3d1b7264bf77d3cd7a3088bf9145737e9db3ded78539e97688fddfecd9ae70a2` |
| açılmış | `road/road/signal_extractions/attacks/accelerator_attack_drive_1.csv` | 12,877,037 | `20744e81bbdb58292295243f81a832e150a197fbe443d1f1e7281c1653409661` |
| açılmış | `road/road/signal_extractions/attacks/accelerator_attack_drive_2.csv` | 10,817,267 | `9908389d069c410f7734619ba48031527bc48e231a5c1176fa7e7e41ef38f0e9` |
| açılmış | `road/road/signal_extractions/attacks/accelerator_attack_reverse_1.csv` | 12,705,269 | `be247bf7e2f58d30386f62eba1e2e00eab802f18ff097deee825958817335cfc` |
| açılmış | `road/road/signal_extractions/attacks/accelerator_attack_reverse_2.csv` | 15,701,591 | `95ce09b7de3cff8545a07889969dbbfd33179f88b1ea323e6b28c8fdcd1adb74` |
| açılmış | `road/road/signal_extractions/attacks/correlated_signal_attack_1_masquerade.csv` | 4,974,953 | `0e03e5b74ccc2e26a7dd05984749c09bb7e247894ba0dea795603514045c350f` |
| açılmış | `road/road/signal_extractions/attacks/correlated_signal_attack_2_masquerade.csv` | 4,255,996 | `9ab5791efd48ebbc4ceea53f2f0fac911e4e4788d570bb0be580716cf2314820` |
| açılmış | `road/road/signal_extractions/attacks/correlated_signal_attack_3_masquerade.csv` | 2,550,755 | `56ee6ff7a4ce39c1322261df39fa27f63f6bee579e0f40de0b3c7701b4fc8e18` |
| açılmış | `road/road/signal_extractions/attacks/max_engine_coolant_temp_attack_masquerade.csv` | 3,873,588 | `1234d16773e394b01507473e8f0524275b282d2c3eb75f424626aa13cdb78279` |
| açılmış | `road/road/signal_extractions/attacks/max_speedometer_attack_1_masquerade.csv` | 13,171,552 | `385c51df86417cc62817a1f4a31770ebba1d732cfaf8d471a1cc79ae260a3cae` |
| açılmış | `road/road/signal_extractions/attacks/max_speedometer_attack_2_masquerade.csv` | 8,971,951 | `2331683a010be5b5034f07c95e9ec8a365baf93ac6fdca31865d0bac4bb8f72b` |
| açılmış | `road/road/signal_extractions/attacks/max_speedometer_attack_3_masquerade.csv` | 12,986,141 | `129baa9346a4f61718bb9c12984b9f10c9758b4ff3d8dbe937c816edef6c3826` |
| açılmış | `road/road/signal_extractions/attacks/metadata.json` | 8,149 | `46c08c6ae3225a115100a7688a7d99b10aabff2bb542898784b0d07a87e680c3` |
| açılmış | `road/road/signal_extractions/attacks/reverse_light_off_attack_1_masquerade.csv` | 4,211,797 | `7063d4621fd3e78ba5704c666f325809df77e429c570043618212c7e08f3aea0` |
| açılmış | `road/road/signal_extractions/attacks/reverse_light_off_attack_2_masquerade.csv` | 6,092,391 | `0f05ad82d1371f1cb20e6a6564bba4a6cdbbac8af5014f577c0172fcf72216a1` |
| açılmış | `road/road/signal_extractions/attacks/reverse_light_off_attack_3_masquerade.csv` | 8,658,429 | `ff45324f9e2cb0a20dc5e20794a02095c61fe4913d16146cd0dbe667b95c01e2` |
| açılmış | `road/road/signal_extractions/attacks/reverse_light_on_attack_1_masquerade.csv` | 8,202,799 | `da1f73e5a8c0bbfb5d95185f06ab15ebbc3753a84191b00be84d70cf42e415e8` |
| açılmış | `road/road/signal_extractions/attacks/reverse_light_on_attack_2_masquerade.csv` | 10,788,136 | `fc47e8cd595c8adab79aff7053ae13b10d81f0014cfc33d4807332182f4aaddc` |
| açılmış | `road/road/signal_extractions/attacks/reverse_light_on_attack_3_masquerade.csv` | 9,640,775 | `3d1b7264bf77d3cd7a3088bf9145737e9db3ded78539e97688fddfecd9ae70a2` |
