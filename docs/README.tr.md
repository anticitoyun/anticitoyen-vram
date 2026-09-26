<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-soutenir-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Belleği bir hiyerarşi olarak ele alan, her GPU'ya kendi donanımının en iyi okuduğu sayısal biçimi veren ve her token'i saniye kadar joule cinsinden de optimize eden, OpenAI API ile uyumlu bir çıkarım ağ geçidi.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · **🇹🇷 Türkçe** · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="vLLM ve llama.cpp'ye karşı verim ve enerji karşılaştırması" width="720"></p>

---

## İçindekiler

- [İki fikir](#idees)
- [Hızlı başlangıç](#demarrage)
- [Kurulum](#installer)
- [`acvram plan` ne söylüyor](#plan)
- [Hızlanmak](#optimisations)
- [HTTP uç noktaları](#http)
- [Sayılar nereden geliyor](#chiffres)
- [Belgeler](#documentation)
- [Ölçülen sonuçlar](#resultats)
- [Durum](#etat)
- [Katkılar](#credits)
- [Lisans](#licence)
- [Projeyi destekleyin](#soutien)

---

<a id="idees"></a>

## İki fikir

Belirli bir makine için tasarlandı:

| | |
|---|---|
| İşlemci | Intel Core i9-14900K (8 P çekirdek + 16 E çekirdek) |
| Anakart | ASUS ROG Maximus Z790 Dark Hero |
| Bellek | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistem | Ubuntu 26.04 LTS (CUDA 13); iki kart PCIe x8/x8'de, 400 W / 275 W ile sınırlı |

**GPU başına bir biçim.** RTX 5090'ın FP4 tensor çekirdekleri var; RTX 3080 Ti'nin ne bunlar ne de FP8 var. İkisini ortak bir biçime hizalamak 5090'ı israf ederdi. Bu yüzden dönüştürücü *aynı modeli iki kez* yazar, her hedefin gerçekten kullanabileceği biçimde:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| ağırlıklar | **NVFP4** — her 16'da bir E2M1 + FP8 E4M3 ölçeği | **INT4** — her 128'de bir uint4 + fp16 ölçek ve sıfır |
| ağırlık başına bit | 4,50 | 4,16 |
| BF16'ya karşı | ×3,56 daha küçük | ×3,85 daha küçük |
| hesaplama modu | FP4 tensor çekirdekleri | çekirdek içinde FP16'ya dekuantize, FP16 tensor çekirdekleri |
| KV önbelleği | INT8 | INT8 |

Ağırlık başına 4,5 bit'te 32 Go VRAM, BF16'daki 16 milyara karşılık yaklaşık **56 milyar parametre** barındırır. İki kart birlikte, ana belleğe hiç dokunmadan yaklaşık **78 milyar yerleşik parametre** verir.

**Bellek bir duvar değil, bir hiyerarşidir.** Üç katman, ve zamanlayıcı modelin sığacağını ummak yerine her birinin maliyetini ölçer:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Hızlı başlangıç

```bash
./install.sh                       # sanal ortam + torch cu128 + acvram
acvram doctor                      # bu makine hazır mı, ve ne için
acvram detect                      # burada gerçekte ne var

acvram plan  ~/modeles/Qwen3-32B                    # her katman nereye gidecek
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Herhangi bir OpenAI istemcisi ardından bağlanır:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Merhaba"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="inutilise")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Merhaba"}])
```

---

<a id="installer"></a>

## Kurulum

Kaynaktan (tüm platformlar):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Veya paket olarak, her [GitHub sürümüne](https://github.com/anticitoyun/anticitoyen-vram/releases/latest) eklenmiş bir dosya:

| Kanal | Sürüme eklenen dosya | Komut |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (`rpmbuild` tarafından üretilen adlar, sabit değil) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (veya `.src.rpm`'den `rpmbuild --rebuild *.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Bir `.flatpakref` her zaman deponun en son yayımlanan sürümünü kurar.

Kurulumdan önce, indirilen dosyayı sürüme eklenen sağlama toplamlarına göre doğrulayın (`SHA256SUMS`, diğer tüm dosyalar mevcut olduktan sonra yayımlanır):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip paket olarak yayımlanmaz (derlenmiş wheel yok): `pip install -e '.[dev]'`, `./install.sh` gibi, kaynağın bir klonundan kurar.

---

<a id="plan"></a>

## `acvram plan` ne söylüyor

Zamanlayıcının her indirmeden önce çalıştırılması gerekir. Bir modelin bu makinede kullanılabilir olup olmadığına karar veren soruları yanıtlar:

```
$ acvram plan ~/modeles/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  etage   format      capacite      poids         KV  tranche
  cuda:0  nvfp4        30,3 Gio   25,5 Gio   4,5 Gio  couches 0-58
  cuda:1  int4_awq     10,9 Gio    8,7 Gio   1,6 Gio  couches 59-79
  cpu     nvfp4        74,8 Gio    3,4 Gio       0 o  -

  poids au total     37,6 Gio
  lu par jeton       35,1 Gio
  KV par jeton       162,5 Kio  -> 39 843 jetons en cache
  MLP en RAM hote    55-58

  decodage estime    17,8 jetons/s  (lot de 1)
  prefill estime     847 jetons/s
```

İlk uyanı tutmak yerine yapılandırma uzayını araştırır, ve kararlarından ikisi açıklanmayı hak edecek kadar sezgiye aykırıdır:

* **Bir model yalnızca 5090'a sığdığında 3080 Ti'yi kullanılmadan bırakır.** Bir hattın aşamaları sıralı çalışır: 1790 Go/s'lik bir hatta 912 Go/s'lik bir aşama eklemek tekli-akış çözümlemesini yavaşlatır. `--gpus all` ile zorlanır.
* **Ağırlıkları VRAM'de tutmak için KV önbelleğini küçültür.** Önbelleğe verilen her gibibyte, PCIe veri yoluna itilmiş bir gibibyte ağırlıktır, ve bir ağırlığı PCIe üzerinden okumak VRAM'den okumanın yaklaşık otuz katı maliyetlidir. Yukarıdaki 70B'de, tek başına bu tercih hızı 2,3'ten 17,8 token/s'ye çıkarır.

---

<a id="optimisations"></a>

## Hızlanmak

Her biri yalnızca bir kronometreyle değil, bir eşdeğerlik kanıtıyla doğrulanmış dört optimizasyon: yanıtı değiştiren bir optimizasyon bir hatadır.

Yoğun modellerin NVFP4 doğrusalları varsayılan olarak Marlin düzeninden geçer (b = 8'de +%57 ila +%90 verim, TTFT +2 ila +4 ms, revue/poste6-piece147-verdict-24-09.md'ye göre; geri dönüş `ACVRAM_PROJ_MARLIN=0`, bkz. [CHANGELOG.md](../CHANGELOG.md)).

### Spekülatif kod çözme (`--speculative`)

1 boyutlu bir toplu işle tek bir token'i kod çözmek bellek sınırlıdır: makine tek bir token üretmek için tüm etkin ağırlıkları okur. Önerilen K token'i doğrulamak aynı ağırlıkları **yalnızca bir kez** okur. İki öneri sağlayıcı:

* `ngram` (varsayılan) — geçerli soneki bağlamda daha önce arar ve onu takip edeni önerir. Hiçbir maliyeti yoktur, hiçbir model gerektirmez. Çıktı girdiyi kopyaladığında karlıdır: kod düzenleme, RAG, özetleme.
* `draft` — ikinci bir cihazda küçük bir model. Bu düzende, bu cihaz, zamanlayıcının 5090'a sığan her model için bilerek boşta bıraktığı RTX 3080 Ti'dir.

`mtp` (modelin `nextn` başlığı) ve `auto` de mevcuttur; şu haliyle karlı değildir ve varsayılan olarak etkin değildir — bkz. `docs/ARCHITECTURE.md`.

Kabul yaklaşık değil, kesindir: bir öneri `min(1, p/q)` olasılığıyla kabul edilir ve bir ret, `p - q`'nun normalize edilmiş pozitif kısmından yeniden örnekleme yapar. Kasıtlı olarak kötü kalibre edilmiş bir taslağa karşı 40.000 çekimde ölçüldüğünde, yayılan dağılım hedefe göre 0,002 toplam varyasyonda kalır — spekülasyon hız satın alır, asla farklı bir yanıt değil.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Önek önbelleği (varsayılan olarak etkin)

Bloklar, token dilimlerinin *zincirlenmiş* özeti ile adreslenir: bir sistem talimatını paylaşan iki istek onun bloklarını paylaşır, ve ikincisinin artık onları önceden hesaplaması gerekmez. Zincirleme vazgeçilmezdir: farklı bir bağlamdaki aynı on altı token aynı anahtarları ve değerleri içermez, ve yalnızca dilimi özetlemek bir dizinin önbelleğini bir başkasına hizmet ettirirdi.

İçeriği tanınabilir kalan serbest bırakılmış bir blok, boş bloklar listesi yerine bir LRU kuyruğuna katılır: önbellek böylece isteklerin arasında hayatta kalır, hizmet edebileceği bir tahsisi asla reddetmeden.

### Ana makine yürütmesi (`--host-exec`)

Ağırlıkları RAM'de bulunan bir katman, GPU'ya kopyalanabilir veya yerinde hesaplanabilir. Her iki yol da bellek sınırlıdır ve aynı byte'ları okur: en hızlısı en geniş veri yoluna sahip olandır — PCIe 5.0 x16 yaklaşık 54 Go/s verir, çift kanallı DDR5 yaklaşık 70 Go/s — ve yerinde hesaplama ayrıca GPU'yu bir kopyayı beklemek yerine serbest bırakır.

Bu yalnızca işlemci 4 bit'e paketlenmiş ağırlıkları doğrudan okuduğunda değerlidir. Bu yüzden AVX2 yolu olan küçük bir C++ çekirdeği (`acvram_cpu.cpp`, ctypes ile yüklenir, Python başlıkları veya ninja olmadan). **Skaler** geri dönüş dalında bile, `dequantize() @ x`'i INT4'te 1,44 kat ve NVFP4'te 3,21 kat geçer, çünkü ikincisi önce tüm matrisin 32 bit'lik bir kopyasını yazar.

Mistral-Large-123B'de, zamanlayıcının tahmini 1,35'ten 2,42 token/s'ye çıkar.

### Karışık hassasiyet (`--snr-floor`, varsayılan olarak kapalı)

Dönüştürücü, her tensör için her katmanın çıkışındaki sinyal/gürültü oranını ölçer ve `--snr-floor`'un altına düşenleri, tensörlerin %15'i ve bir üst fiyat (`--promotion-cout-max`, eklenen mebibayt cinsinden) sınırında, daha geniş bir biçime yükseltebilir.

Eşik **varsayılan olarak sıfırdır**: hiçbir şey yükseltilmez. Kod çözme bellek bant genişliği ile sınırlıdır, ve `Huihui-Qwen3.8-27B` üzerindeki ölçüm karar verir — 25 dB'lik bir eşik, %2,0'lık şaşkınlık (perplexity) (16.383 token'lık bir külliyatta 43,447'ye karşı 42,591) karşılığında %13,4 bellek ve %10,6 verim maliyetine yol açar (16,02 ve 46,2'ye karşı 18,50 Gio ve 41,8 t/s). `--snr-floor 25`, kalite hızdan önce geldiğinde eski davranışı geri getirir.

### Ve `acvram eval`

Sinyal/gürültü oranı ve logit kosinüsü yaklaşımlardır. `acvram eval REP [REP ...]`, bir biçim seçiminin kanıtlarla karara bağlanması için kayan pencereyle şaşkınlığı ölçer:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP uç noktaları

| uç nokta | notlar |
|---|---|
| `POST /v1/chat/completions` | SSE akışı veya tekil yanıt; modelin sohbet şablonunu kullanır |
| `POST /v1/completions` | metin veya token kimlikleri olarak istem |
| `POST /v1/embeddings` | ortalanmış son gizli durumlar, L2 normalize edilmiş, `dimensions` uygulanır |
| `GET /v1/models` | artı bir `acvram` bloğu: biçimler, cihazlar, KV önbellek kapasitesi |
| `GET /health`, `GET /metrics` | kod çözme hızı, KV blok doluluğu |

Bu yanıtların alan adları İngilizce kalır: bu OpenAI protokolüdür, ve çevirmek mevcut tüm istemcileri bozardı.

---

<a id="chiffres"></a>

## Sayılar nereden geliyor

Yukarıda belirtilen her değer bu depodaki kodla üretilir ve `pytest` ile doğrulanır. Referans çekirdeklerle işlemci üzerinde yapılan ölçümler:

| biçim | bit/ağırlık | ağırlık SNR'si | BF16'ya karşı logit kosinüsü |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Bu ölçümlerden çıkan iki tespit, varsayılan değerleri değiştirdi:

* **Bir Hadamard dönüşü INT4'e yardımcı olur, NVFP4'e değil.** INT4'ün 128'lik grupları tek başına bir kanal aykırı değerini ememez, bu yüzden uç değerleri yaymak, aktivasyon başına n log n'lik bir dönüşüme değer. NVFP4'ün 16'lık blokları zaten kendi ölçeklerini taşır. Bu yüzden yalnızca INT4'e uygulanan `--hadamard auto`.
* **INT8, KV önbelleği için FP8 E4M3'ü geçer**, aynı boyutta 32 dB'ye karşı 44 dB, çünkü (token, başlık) başına bir ölçek zaten FP8'in üstel bitler harcadığı dinamik aralığı sağlar. Bu yüzden her iki kart da, 5090 FP8 yapabilecek olsa bile, INT8'de bir KV önbelleği kullanır. Bir `k8v4` biçimi (INT4'te değerler, %−22 önbellek byte'ı) opsiyonel olarak mevcuttur, **onaylanmamıştır** — bkz. `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Belgeler

| Belge | İçerik |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **projeyi başka bir makinede sürdürmek** (Fransızca) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | parçaların nasıl bir araya geldiği |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | saf NVFP4 ya da kanal başına int8 ile attention+GDN, bir Gated DeltaNet hibrit üzerinde |
| [`docs/MATERIEL.md`](MATERIEL.md) | bu belirli makineyi ayarlamak |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **henüz yapılmamış olan**, önce bunu okuyun |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | kod üzerinde çalışma kuralları (dil, stil, itmeden önceki kontroller) |

---

<a id="resultats"></a>

## Ölçülen sonuçlar (22/09/2026, 400 W'ta RTX 5090, enerji sayacında ≥ 20 s rejim)

Tüm motorlar için aynı protokol (`outils/`, tek kart, `energie.py`) ile NVFP4 (uzmanlar) + INT8 (dikkat, başlık) içinde Qwen3-Coder-30B-A3B:

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| 12 dizi kod çözme | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| 1 dizi kod çözme | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, kaldırıldı) |

¹ 22/09 düzeltmesi: `serve` varsayılan olarak spekülasyon yapar (`--speculative ngram`, cli.py), rakipler yapmaz; şimdiye kadar yayımlanan 380,8 t/s SPEKÜLASYONLA ölçülmüştü. Spekülasyon olmadan (`--speculative none`, aynı zincir, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram, b=1'de llama.cpp ve vLLM'nin ardından **üçüncü**. Enerjide llama.cpp'nin önünde kalır (net 0,700'e karşı 0,601 J/token). b=12'de spekülasyon hiçbir zaman etkin değildir (`lot_max=2` koruması): bu hücre zaten eşit koşullardaydı.

² 23/09, aynı oturum, aynı HTTP istemcisi (`acvram serve` ve `vllm serve`'e karşı `banc-llamacpp-16-09.py`), her kol etrafında açıkça yerleştirilmiş `-lgc 2700`, alternatif hücreler A V V A, kol başına ≥ 5 parti, sapma yalnızca 2 σ'nın ötesinde bildiriliyor (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (kod çözmede w13, açılmış dikkat azaltma): sapma −%1,6, **2 σ'nın altında: verim eşitliği**. J/token'de **vLLM %7,0 önde kalır** (2 σ'nın ötesinde). 0.6.37 ile aynı protokol −%4,7 veriyordu.

³ ² ile aynı oturum ve aynı protokol, her iki tarafta da spekülasyon olmadan: acvram 312,3'e karşı vLLM 284,8 — **acvram verimde %9,7 önde** (2 σ'nın ötesinde); J/token: **eşitlik** (sapma %0,04, 2 σ'nın altında).

⁴ 23/09, llama.cpp'ye karşı aynı protokol (revue/poste2-piece72-llamacpp-b1-23-09.md), yeniden yazılmış yönlendirmeyle (+%5,6) acvram 0.6.37: acvram 310,8'e karşı llama.cpp 329,9 t/s — **llama.cpp verimde %5,8 önde, acvram J/token'de %13,4 önde** (0,598'e karşı 0,691).

Günün verimleri (post 1030, tasarruf rejimi `-lgc 2700`, hizmette hat; açgözlü örnekleme CUDA grafiğinde yakalandı, 0.6.35'in varsayılanı). acvram'ın b=12'si resmi mühürlü bir hücredir (6 iç içe geçmiş pencerenin medyanı, pencere başına saat).

> **Düzeltme (23/09/2026).** Şimdiye kadar yayımlanan vLLM karşılaştırması (b=12: 1 782'ye karşı 1 634 t/s; b=1: 290,6), HTTP üzerinden ölçülen acvram'ı **çevrimdışı** ölçülen vLLM'ye (`LLM().generate()`) karşı koyuyordu, ve 22/09 düzeltmesi vLLM hücresinin `vllm serve` üzerinden geçtiğini yanlışlıkla iddia ediyordu. 23/09: her ikisi için aynı HTTP istemcisi, ve her ikisi için yerleştirilmiş `-lgc` (acvram başlangıçta kendisininkini yerleştirir, `vllm serve` yapmaz: bu önlem olmadan vLLM ~2650'ye karşı ~2930 MHz'de çalışıyordu). ² notundaki sonuç: b=12'de vLLM %9,1 önde.

14/09 sabahı acvram aynı hücrede 630 t/s ve 0,619 J/token'daydı: kazanımlar Blackwell'in yerel FP4 MMA'sından (`mma.sync … kind::mxf4nvf4`, bf16'nın ×7,9 katı), toplu iş kovası başına gruplanmış GEMM'de MoE'den, tek bir çekirdekte yönlendirmeden (adım başına 3.677 → 1.517 başlatma) ve izdüşümler için tensor çekirdekleri üzerinde dar bir GEMM'den gelir. Her sayının `acvram-memoire/revue/`'de ölçümden önce mühürlü tahmini, aracı ve rejimi ile bir notu vardır — rejimsiz bir sayı yayımlanmaz.

acvram'ın önde olduğu yerler: vLLM'nin yalnızca FP8'de sunduğu native NVFP4 sm_120'deki MLA modelleri (GLM-4.7-Flash) (b=1: hizmette 165,35 t/s); VRAM'e sığmayan modeller. Tek dizi kod çözme bunlardan biri değildir: spekülasyon olmadan, acvram orada vLLM'nin %9,7 önünde (not ³), verimde llama.cpp'nin %5,8 gerisinde ama enerjide onun %13,4 önünde (not ⁴). Büyük bir toplu işte, VRAM'e sığan bir MoE üzerinde, vLLM b=12'de verim eşitliğinde (1 995,1'e karşı 2 027,0 t/s, 2 σ'nın altında, not ²) ama %7,0 daha az J/token tutar; acvram orada 1.540 t/s'den (0.6.34) 1.995'e (0.6.38) ilerledi.

---

<a id="etat"></a>

## Durum

Sürüm 0.6.38. Her şey 5090'da çalışıyor: `sm_120a` (native FP4) ve `sm_86` için derlenmiş CUDA çekirdekleri, CUDA grafikleri, NVFP4/INT8/INT4 nicemleme, HTTP sunucusu. Koruyucular yerinde: kart çalışma oturumlarına görünmez (boş `CUDA_VISIBLE_DEVICES`) ve yalnızca `outils/carte.sh` onu, kilit altında, seferde bir ölçüm için sunar; bir gözlemci kilit dışındaki her erişimi günlüğe kaydeder; birden fazla kartı kapsayan veya 10 s'den kısa bir enerji ölçümü geçersiz kılınır; bozulmuş rejimde yüklenen bir model bunu belirtir ve bir düelloya girmez.

4.107 test (`pytest --collect-only -q`, işlemcide bir dakika; GPU testleri yalnızca `carte.sh` altında çalışır). İşin takibi: `acvram-memoire/` (kurallar, dizin, defterler, birkaç yüz notun incelemesi).

---

<a id="credits"></a>

## Katkılar

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, Apache-2.0 lisansı altında: `acvram/kernels/marlin_port/`, [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE)'da tam dosya dosya atıf ile Marlin çekirdeklerini (MoE ve yoğun) taşır.
- **NVIDIA** — CUDA, Blackwell'in FP4 tensor çekirdekleri (`sm_120`) ve bu projenin bağımlı olduğu kütüphaneler.
- **PyTorch** — tensör motoru ve C++/CUDA uzantıları.

Bağımsız proje, ASUS, NVIDIA veya vLLM projesi ile bağlantılı değildir.

---

<a id="licence"></a>

## Lisans

Bu depodaki kod için [GPL-3.0 veya sonraki](../LICENSE). `acvram/kernels/marlin_port/`, [vLLM](https://github.com/vllm-project/vllm) v0.29.0'dan taşınan kodu içerir (`marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size` çekirdekleri), Apache-2.0 lisansı altında: her dosya orijinal başlığını korur, lisans metni `LICENSE-vllm`'de bulunur, ve dosya listesi, kaynak commit ve değişiklikler [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE)'da bulunur.

---

<a id="soutien"></a>

## Projeyi destekleyin

acvram'ın geliştirilmesi kişisel donanım üzerinde yürütülmektedir. Proje size faydalı olduysa:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Offrir%20un%20café&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Çeviriler: [TRADUIRE.md](TRADUIRE.md) (Fransızca; katkı kılavuzu henüz çevrilmedi).
