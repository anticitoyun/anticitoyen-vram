<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Destek: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

OpenAI API'si ile uyumlu, belleği bir hiyerarşi olarak ele alan ve her GPU'ya
silikonunun en iyi okuduğu sayısal biçimi veren bir çıkarım geçidi.

Belirli bir makine için tasarlandı:

| | |
|---|---|
| İşlemci | Intel Core i9-14900K (8 P çekirdek + 16 E çekirdek) |
| Anakart | ASUS ROG Maximus Z790 Dark Hero |
| Bellek | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistem | Ubuntu 26.04 LTS (CUDA 13); iki kart da PCIe x8/x8'de, 400 W / 275 W ile sınırlı |

## İki fikir

**GPU başına bir biçim.** RTX 5090'da FP4 tensör çekirdekleri var; RTX 3080
Ti'de yok, FP8 de yok. İkisini ortak bir biçime hizalamak 5090'ı boşa
harcardı. Dönüştürücü bu yüzden *aynı modeli iki kez* yazar, her hedefin
gerçekten kullanabildiği biçimde:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| ağırlıklar | **NVFP4** — E2M1 + her 16'da FP8 E4M3 ölçek | **INT4** — uint4 + her 128'de fp16 ölçek ve sıfır |
| ağırlık başına bit | 4,50 | 4,16 |
| BF16'ya göre | ×3,56 daha küçük | ×3,85 daha küçük |
| hesaplama modu | FP4 tensör çekirdekleri | çekirdekte FP16'ya dekuantize, FP16 tensör çekirdekleri |
| KV önbelleği | INT8 | INT8 |

Ağırlık başına 4,5 bit ile 32 GB VRAM yaklaşık **56 milyar parametre**
barındırır; BF16'da bu 16 milyardır. İki kartta bu, RAM'e dokunmadan önce
yaklaşık **78 milyar yerleşik parametre** eder.

**Bellek bir hiyerarşidir, duvar değil.** Üç kat; planlayıcı modelin sığmasını
ummak yerine her katın neye mal olduğunu ölçer:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Hızlı başlangıç

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Ardından herhangi bir OpenAI istemcisi bağlanır:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Bonjour"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="inutilise")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Bonjour"}])
```

## `acvram plan` ne söyler

Planlayıcı, herhangi bir indirmeden önce çalıştırılmayı hak eder. Bir
modelin bu makinede kullanılabilir olup olmadığını belirleyen sorulara yanıt
verir:

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

Sığan ilk yapılandırmayı tutmak yerine yapılandırma uzayını tarar ve iki
kararı, dile getirilmeyi hak edecek kadar sezgiye aykırıdır:

* **Bir model tek başına 5090'a sığdığında 3080 Ti'yi kullanmaz.** Bir boru
  hattının dilimleri seri çalışır: 1790 GB/s'lik bir hatta 912 GB/s'lik bir
  aşama eklemek tek akışlı çözümlemeyi yavaşlatır. `--gpus all` ile
  zorlanabilir.
* **Ağırlıkları VRAM'de tutmak için KV önbelleğini küçültür.** Önbelleğe
  verilen her gigabayt, PCIe veri yoluna itilen bir gigabayt ağırlıktır ve
  PCIe üzerinden bir ağırlık okumak VRAM'den okumanın yaklaşık otuz katına mal
  olur. Yukarıdaki 70B'de tek başına bu denge 2,3'ten 17,8 jeton/s'ye götürür.

## Hızlı gitmek

Dört eniyileme; her biri yalnızca kronometreyle değil, bir denklik kanıtıyla
doğrulanmıştır: yanıtı değiştiren bir eniyileme bir hatadır.

### Spekülatif çözümleme (`--speculative`)

1 boyutlu yığınla bir jeton çözümlemek bellekle sınırlıdır: makine tek bir
jeton üretmek için tüm etkin ağırlıkları okur. Önerilen K jetonu doğrulamak
aynı ağırlıkları **yalnızca bir kez** okur. İki öneren:

* `ngram` (varsayılan) — geçerli soneki bağlamda daha önce arar ve ardından
  geleni önerir. Hiçbir şeye mal olmaz, model gerektirmez. Çıktı girdiyi
  kopyaladığında kârlıdır: kod düzenleme, RAG, özetleme.
* `draft` — ikinci bir aygıtta küçük bir model. Bu donanımda o aygıt,
  planlayıcının 5090'a sığan her model için bilerek boşta bıraktığı
  RTX 3080 Ti'dir.

Kabul kesindir, yaklaşık değil: bir öneri `min(1, p/q)` olasılığıyla kabul
edilir ve bir ret, `p - q`'nun normalleştirilmiş pozitif kısmından yeniden
örnekler. Bilerek kötü kalibre edilmiş bir taslağa karşı 40 000 çekilişte
ölçüldüğünde, yayılan dağılım hedeften 0,002 toplam varyasyon içinde kalır —
spekülasyon hız satın alır, asla farklı bir yanıt değil.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Önek önbelleği (varsayılan olarak etkin)

Bloklar, jeton dilimlerinin *zincirlenmiş* özetiyle adreslenir: bir sistem
komutunu paylaşan iki istek onun bloklarını paylaşır ve ikincisinin bunları
yeniden hesaplaması gerekmez. Zincirleme vazgeçilmezdir: farklı bir bağlamdaki
aynı on altı jeton aynı anahtar ve değerleri içermez ve yalnızca dilimi
özetlemek bir dizinin önbelleğini bir diğerine sunardı.

İçeriği tanımlanabilir kalan serbest bırakılmış bir blok, boş listesi yerine
bir LRU kuyruğuna girer: önbellek böylece istekler arasında, sunabileceği bir
tahsisi asla reddetmeden hayatta kalır.

### Ana bilgisayar katında hesaplama (`--host-exec`)

Ağırlıkları RAM'de bulunan bir katman GPU'ya kopyalanabilir ya da yerinde
hesaplanabilir. İki yol da bellekle sınırlıdır ve aynı baytları okur: daha
hızlı olan, veri yolu daha geniş olandır — PCIe 5.0 x16 yaklaşık 54 GB/s,
çift kanallı DDR5 yaklaşık 70 GB/s verir — ve yerinde hesaplamak ayrıca GPU'yu
bir kopyayı bekletmek yerine serbest bırakır.

Bu yalnızca işlemci 4 bit paketlenmiş ağırlıkları doğrudan okuyorsa değer.
Bu yüzden AVX2 yollu küçük bir C++ çekirdeği (`acvram_cpu.cpp`, ctypes ile
yüklenir, Python başlıkları ya da ninja olmadan). **Skaler** yedek dalında
bile `dequantize() @ x`'i INT4'te 1,44, NVFP4'te 3,21 kat geçer; çünkü
ikincisi önce tüm matrisin 32 bitlik bir kopyasını yazar.

Mistral-Large-123B'de planlayıcının tahmini 1,35'ten 2,42 jeton/s'ye çıkar.

### Karma hassasiyet (`--snr-floor`, varsayılan olarak kapalı)

Dönüştürücü her tensör için katman çıkışındaki sinyal/gürültü oranını ölçer
ve `--snr-floor` altına düşenleri, tensörlerin %15'i ve bir fiyat tavanı
(`--promotion-cout-max`, eklenen mebibayt) sınırı içinde daha geniş bir
biçime yükseltebilir.

Taban **varsayılan olarak sıfırdır**: hiçbir şey yükseltilmez. Çözümleme
bellek bant genişliğiyle sınırlıdır ve `Huihui-Qwen3.8-27B` üzerindeki ölçüm
kararı verir — 25 dB taban, %2,0 karışıklık (16 383 jetonluk derlemde 42,591'e
karşı 43,447) için %13,4 bellek ve %10,6 verim (16,02 ve 46,2'ye karşı
18,50 GiB ve 41,8 j/s) götürür. Kalite hızdan önemliyse `--snr-floor 25` eski
davranışı geri getirir.

### Ve `acvram eval`

Sinyal/gürültü oranı ve logit kosinüsü yaklaşıklardır.
`acvram eval DİZİN [DİZİN ...]` kayan pencereyle karışıklığı ölçer; böylece
bir biçim seçimi kanıtla karara bağlanır:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP uç noktaları

| uç nokta | notlar |
|---|---|
| `POST /v1/chat/completions` | SSE akışı ya da tek yanıt; modelin sohbet şablonunu kullanır |
| `POST /v1/completions` | metin ya da jeton kimlikleri olarak istem |
| `POST /v1/embeddings` | ortalanmış son gizli durumlar, L2 normalleştirilmiş, `dimensions` gözetilir |
| `GET /v1/models` | artı bir `acvram` bloğu: biçimler, aygıtlar, KV önbellek kapasitesi |
| `GET /health`, `GET /metrics` | çözümleme verimi, KV bloklarının doluluğu |

Bu yanıtlardaki alan adları İngilizce kalır: bu OpenAI protokolüdür ve
çevrilmeleri mevcut tüm istemcileri bozardı.

## Sayılar nereden geliyor

Yukarıda anılan her değer bu depodaki kod tarafından üretilir ve `pytest` ile
doğrulanır. Ölçümler referans çekirdeklerle işlemci üzerinde yapıldı:

| biçim | bit/ağırlık | ağırlık SNR'si | BF16'ya karşı logit kosinüsü |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Bu ölçümlerden çıkan iki bulgu varsayılanları değiştirdi:

* **Hadamard döndürmesi INT4'e yardım eder, NVFP4'e etmez.** INT4'ün 128'lik
  grupları yalıtılmış bir aykırı kanalı soğuramaz; bu yüzden uç değerleri
  yaymak, etkinleştirme başına n log n bir dönüşüme değer. NVFP4'ün 16'lık
  blokları zaten kendi ölçeğini taşır. Bu yüzden `--hadamard auto` bunu
  yalnızca INT4'e uygular.
* **KV önbelleği için INT8, FP8 E4M3'ü geçer**: aynı boyutta 44 dB'ye karşı
  32 dB; çünkü (jeton, baş) başına bir ölçek, FP8'in üs bitleri harcadığı
  dinamik aralığı zaten sağlar. Bu yüzden iki kart da, 5090 FP8 yapabilse
  bile INT8 KV önbelleği kullanır.

## Belgeler

* [`REPRISE.md`](../REPRISE.md) — **projeyi başka bir makinede sürdürmek**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — parçalar nasıl bir araya gelir
* [`docs/MATERIEL.md`](MATERIEL.md) — bu belirli makineyi ayarlamak
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **yapılmamış olanlar**, önce okuyun
* [`CONVENTIONS.md`](../CONVENTIONS.md) — kod üzerinde çalışma kuralları (dil, biçem, push öncesi denetimler)

## Ölçülen sonuçlar (22/09/2026, 400 W'ta RTX 5090, enerji sayacında ≥ 20 s rejim)

NVFP4 (uzmanlar) + INT8 (dikkat, baş) olarak Qwen3-Coder-30B-A3B, tüm
motorlar için aynı protokol (`outils/`, tek kart, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| çözümleme, 12 dizi | **1 634 j/s** | 1 782 j/s | — |
| çözümleme, 1 dizi | **380,8 j/s** | 290,6 j/s | 323,6 j/s |
| prefill pp2048 | **22 707** jeton/s | 21 054 | 8 671 (TabbyAPI, geri çekildi) |

Günün verimi (tezgâh 1030, eko rejim `-lgc 2700`, hizmetteki ardışık düzen; açgözlü örnekleme CUDA grafiğinde yakalanmış, 0.6.35'ten beri varsayılan). b=12, resmî mühürlü bir hücredir (6 iç içe geçmiş pencerenin medyanı, pencere başına saat frekansı).

> **Erratum (22/09/2026).** 0.6.35'in ilk yayını, «vLLM'in +1,84 % önünde» ifadesini 21/09 tarihli 1 596 j/s'lik bir vLLM referansından çıkarıyordu; bu referans bir **çevrimdışı üretimden** (`LLM().generate()`) geliyordu ve **bir sunucuyla kıyaslanamaz**: sürekli zamanlama yok, `acvram serve` yolu değil. 22/09'da, `acvram serve` ile aynı kart ve aynı yol üzerinde **`vllm serve`** (HTTP) karşısında dönüşümlü bir A/V hücresiyle (A1 V1 A2 V2 A3 V3) düzeltildi: vLLM medyanı **1 782 j/s**. Kıyaslanabilir ölçümde **acvram (1 634 j/s), b=12'de vLLM'in yaklaşık 8 % GERİSİNDE**, önünde değil. Eşit saat frekansında J/jeton yeniden ölçümde kalıyor.

14/09 sabahı acvram aynı hücrede 630 j/s ve 0,619 J/jeton'daydı: kazanımlar
Blackwell'in yerli FP4 MMA'sından (`mma.sync … kind::mxf4nvf4`, bf16'ya göre
×7,9), yığın kovası başına gruplanmış GEMM olarak MoE'den, tek çekirdekli
yönlendirmeden (adım başına 3 677 → 1 517 başlatma) ve izdüşümler için dar bir
tensör çekirdeği GEMM'inden gelir. Her rakamın `acvram-memoire/revue/` içinde,
ölçümden önce mühürlenmiş tahmini, aracı ve rejimiyle bir notu vardır —
rejimsiz bir rakam yayımlanmaz.

acvram'ın önde olduğu yer: vLLM'in yalnızca FP8'de sunduğu, yerli sm_120
NVFP4'teki MLA modelleri (GLM-4.7-Flash, b=1: hizmette 165,35 j/s); VRAM'e
sığmayan modeller; ve tek dizilik çözümleme (b=1: vLLM için 290,6'ya karşı
380,8 j/s). Büyük yığında ise, VRAM'e sığan bir MoE üzerinde vLLM b=12'de önde
kalır (1 782'ye karşı 1 634 j/s, bkz. erratum); acvram burada ilerledi (0.6.34'te
1 540 → 1 634) ama öne geçmedi. Enerjideki fark yeniden ölçülecek.

## Durum

Sürüm 0.6.35. Her şey 5090'da çalışır: `sm_120a` (yerli FP4) ve `sm_86` için
derlenmiş CUDA çekirdekleri, CUDA grafları, NVFP4/INT8/INT4 kuantizasyonu,
HTTP sunucusu. Korkuluklar yerinde: kart çalışma oturumlarına görünmez
(`CUDA_VISIBLE_DEVICES` boş) ve yalnızca `outils/carte.sh` onu kilit altında
tek seferde bir ölçüme ödünç verir; bir bekçi kilit dışındaki her erişimi
günlükler; birden fazla kartı kapsayan ya da 10 s'den kısa bir enerji ölçümü
geçersiz sayılır; bozulmuş rejimde yüklenen bir model bunu söyler ve düelloya
girmez.

640 test (`pytest -q`, işlemcide bir dakika; GPU testleri yalnızca
`carte.sh` altında çalışır). İş takibi: `acvram-memoire/` (kurallar,
dizin, defterler, 180 notluk inceleme).

## Destek

acvram kişisel donanım üzerinde geliştirilmektedir. Proje size yararlıysa:
**Destek: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Lisans

GPL-3.0 veya sonrası.
