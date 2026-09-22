<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Támogatás: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Az OpenAI API-val kompatibilis következtetési átjáró, amely a memóriát
hierarchiaként kezeli, és minden GPU-nak azt a számformátumot adja, amelyet a
szilíciuma a legjobban olvas.

Egy meghatározott gépre tervezve:

| | |
|---|---|
| Processzor | Intel Core i9-14900K (8 P-mag + 16 E-mag) |
| Alaplap | ASUS ROG Maximus Z790 Dark Hero |
| Memória | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Rendszer | Ubuntu 26.04 LTS (CUDA 13); mindkét kártya PCIe x8/x8-on, 400 W / 275 W-ra korlátozva |

## A két ötlet

**GPU-nként egy formátum.** Az RTX 5090-nek FP4 tensormagjai vannak; az
RTX 3080 Ti-nek nincsenek, és FP8-a sincs. A kettő közös formátumra hozása
elpazarolná az 5090-et. A konverter ezért *kétszer írja ki ugyanazt a
modellt*, abban a formátumban, amelyet az adott cél valóban ki tud használni:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| súlyok | **NVFP4** — E2M1 + FP8 E4M3 skála 16-onként | **INT4** — uint4 + fp16 skála és nullpont 128-anként |
| bit súlyonként | 4,50 | 4,16 |
| BF16-hoz képest | ×3,56 kisebb | ×3,85 kisebb |
| számítási mód | FP4 tensormagok | a kernelben FP16-ra dekvantálva, FP16 tensormagok |
| KV-gyorsítótár | INT8 | INT8 |

32 GB VRAM 4,5 bit/súly mellett körülbelül **56 milliárd paramétert**
tartalmaz, szemben a BF16 16 milliárdjával. A két kártyán ez nagyjából
**78 milliárd rezidens paraméter**, még mielőtt a rendszermemóriához
nyúlnánk.

**A memória hierarchia, nem fal.** Három szint, és a tervező megméri, mibe
kerül mindegyik, ahelyett hogy reménykedne, hogy a modell befér:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Gyors indulás

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Bármely OpenAI-kliens rácsatlakozik ezután:

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

## Mit mond az `acvram plan`

A tervezőt érdemes minden letöltés előtt lefuttatni. Azokra a kérdésekre
válaszol, amelyek eldöntik, használható-e egy modell ezen a gépen:

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

A konfigurációs teret járja be, ahelyett hogy az első beférőt tartaná meg, és
két döntése elég ellentmondásos ahhoz, hogy ki kelljen mondani:

* **Kihasználatlanul hagyja a 3080 Ti-t**, ha egy modell egyedül az 5090-re
  is befér. Egy csővezeték szeletei sorban futnak: egy 912 GB/s-os fokozat
  hozzáadása egy 1790 GB/s-os csővezetékhez lelassítja az egyfolyamú
  dekódolást. A `--gpus all` kapcsolóval kényszeríthető.
* **Zsugorítja a KV-gyorsítótárat, hogy a súlyok a VRAM-ban maradjanak.**
  Minden a gyorsítótárnak adott gigabájt egy gigabájtnyi súly, amely a PCIe
  buszra szorul, és egy súly PCIe-n át olvasása körülbelül harmincszor annyiba
  kerül, mint a VRAM-ból. A fenti 70B-nél önmagában ez a döntés 2,3-ról 17,8
  token/s-ra visz.

## Gyorsan haladni

Négy optimalizálás, mindegyiket ekvivalenciabizonyíték igazolja, nem csak
stopperóra: egy optimalizálás, amely megváltoztatja a választ, hiba.

### Spekulatív dekódolás (`--speculative`)

Egy token dekódolása 1-es kötegmérettel memóriakorlátos: a gép minden aktív
súlyt beolvas egyetlen token előállításához. K javasolt token ellenőrzése
ugyanezeket a súlyokat **egyetlenegyszer** olvassa. Két javaslattevő:

* `ngram` (alapértelmezett) — az aktuális utótagot keresi korábban a
  kontextusban, és azt javasolja, ami utána következett. Semmibe sem kerül,
  nem igényel modellt. Kifizetődő, ha a kimenet a bemenetet másolja:
  kódszerkesztés, RAG, összefoglalás.
* `draft` — egy kis modell egy második eszközön. Ezen a gépen ez az eszköz az
  RTX 3080 Ti, amelyet a tervező szándékosan tétlenül hagy minden 5090-re
  beférő modellhez.

Az elfogadás pontos, nem közelítő: egy javaslatot `min(1, p/q)`
valószínűséggel fogad el, elutasításkor pedig a `p - q` normalizált pozitív
részéből mintavételez újra. 40 000 húzáson mérve egy szándékosan rosszul
kalibrált piszkozattal szemben a kibocsátott eloszlás 0,002 teljes
variáción belül marad a céltól — a spekuláció sebességet vesz, soha nem más
választ.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Előtag-gyorsítótár (alapértelmezés szerint aktív)

A blokkokat tokenszeletük *láncolt* hash-e címzi: két kérés, amely közös
rendszerutasításon osztozik, annak blokkjain is osztozik, és a másodiknak már
nem kell előre kiszámolnia őket. A láncolás nélkülözhetetlen: ugyanaz a
tizenhat token más kontextusban nem ugyanazokat a kulcsokat és értékeket
tartalmazza, és csak a szelet hash-elése az egyik szekvencia gyorsítótárát
szolgálná ki egy másiknak.

A felszabadított, de még azonosítható tartalmú blokk LRU-sorba kerül a szabad
lista helyett: a gyorsítótár így túléli a kéréseket anélkül, hogy valaha is
elutasítana egy kiszolgálható foglalást.

### Számítás a gazdaszinten (`--host-exec`)

Egy réteg, amelynek súlyai a RAM-ban vannak, átmásolható a GPU-ra vagy
helyben kiszámítható. Mindkét út memóriakorlátos és ugyanazokat a bájtokat
olvassa: a gyorsabb az, amelyiknek szélesebb a busza — a PCIe 5.0 x16 kb.
54 GB/s-ot ad, a kétcsatornás DDR5 kb. 70 GB/s-ot —, és a helyben számítás
ráadásul szabadon hagyja a GPU-t, ahelyett hogy másolásra várakoztatná.

Ez csak akkor éri meg, ha a processzor közvetlenül olvassa a 4 bitesre
csomagolt súlyokat. Innen egy kis C++ kernel AVX2 úttal (`acvram_cpu.cpp`,
ctypes-szal betöltve, Python-fejlécek és ninja nélkül). Még a **skalár**
tartalék ágán is 1,44-szeresen veri a `dequantize() @ x`-et INT4-ben és
3,21-szeresen NVFP4-ben, mert az utóbbi előbb az egész mátrix 32 bites
másolatát írja ki.

A Mistral-Large-123B-n a tervező becslése 1,35-ről 2,42 token/s-ra nő.

### Vegyes pontosság (`--snr-floor`, alapértelmezés szerint kikapcsolva)

A konverter minden tenzorra méri a jel-zaj viszonyt a réteg kimenetén, és a
`--snr-floor` alá esőket szélesebb formátumba léptetheti, legfeljebb a
tenzorok 15 %-áig és egy árplafonig (`--promotion-cout-max`, hozzáadott
mebibájtban).

A küszöb **alapértelmezés szerint nulla**: semmi sem lép elő. A dekódolást a
memória-sávszélesség korlátozza, és a `Huihui-Qwen3.8-27B` mérése dönt — egy
25 dB-es küszöb 13,4 % memóriába és 10,6 % átbocsátásba kerül (18,50 GiB és
41,8 t/s szemben 16,02-vel és 46,2-vel) 2,0 % perplexitásért (42,591 szemben
43,447-tel, 16 383 tokenes korpusz). A `--snr-floor 25` visszaállítja a régi
viselkedést, ha a minőség fontosabb a sebességnél.

### És az `acvram eval`

A jel-zaj viszony és a logit-koszinusz közelítések.
Az `acvram eval KÖNYVTÁR [KÖNYVTÁR ...]` csúszóablakos perplexitást mér, hogy
egy formátumválasztás bizonyítékok alapján dőljön el:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP végpontok

| végpont | megjegyzések |
|---|---|
| `POST /v1/chat/completions` | SSE-folyam vagy egyetlen válasz; a modell beszélgetési sablonját használja |
| `POST /v1/completions` | prompt szövegként vagy tokenazonosítókként |
| `POST /v1/embeddings` | átlagolt végső rejtett állapotok, L2-normalizálva, a `dimensions` tiszteletben tartva |
| `GET /v1/models` | plusz egy `acvram` blokk: formátumok, eszközök, KV-gyorsítótár kapacitása |
| `GET /health`, `GET /metrics` | dekódolási átbocsátás, KV-blokkok kihasználtsága |

E válaszok mezőnevei angolul maradnak: ez az OpenAI-protokoll, és a
lefordításuk minden meglévő klienst elrontana.

## Honnan jönnek a számok

Minden fent idézett értéket e tároló kódja állít elő és a `pytest`
ellenőriz. A mérések processzoron, a referencia-kernelekkel készültek:

| formátum | bit/súly | súlyok SNR-je | logit-koszinusz vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

E mérések két megállapítása megváltoztatta az alapértelmezéseket:

* **A Hadamard-forgatás az INT4-nek segít, az NVFP4-nek nem.** Az INT4
  128-as csoportjai nem tudnak elnyelni egy elszigetelt kiugró csatornát, így
  a szélsőértékek szétterítése megér egy n log n transzformációt
  aktivációnként. Az NVFP4 16-os blokkjai már saját skálát hordoznak. Innen a
  `--hadamard auto`, amely csak az INT4-re alkalmazza.
* **Az INT8 veri az FP8 E4M3-at a KV-gyorsítótárban**, 44 dB szemben 32
  dB-lel azonos méretnél, mert egy (token, fej) szerinti skála már megadja azt
  a dinamikatartományt, amelyre az FP8 exponensbiteket költ. Ezért mindkét
  kártya INT8 KV-gyorsítótárat használ, noha az 5090 tudna FP8-at.

## Dokumentáció

* [`REPRISE.md`](../REPRISE.md) — **a projekt folytatása egy másik gépen**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — hogyan állnak össze a részek
* [`docs/MATERIEL.md`](MATERIEL.md) — ennek a konkrét gépnek a beállítása
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **ami nincs kész**, ezt olvasd először
* [`CONVENTIONS.md`](../CONVENTIONS.md) — a kód munkakonvenciói (nyelv, stílus, ellenőrzések push előtt)

## Mért eredmények (2026. 09. 22., RTX 5090 400 W-on, ≥ 20 s-os üzem az energiamérőn)

Qwen3-Coder-30B-A3B NVFP4-ben (szakértők) + INT8-ban (figyelem, fej), azonos
protokoll minden motorra (`outils/`, egy kártya, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekódolás, 12 szekvencia | **1 634 t/s** | 1 782 t/s | — |
| dekódolás, 1 szekvencia | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, visszavonva) |

A napi átbocsátások (1030-as állomás, `-lgc 2700` takarékos üzem, szolgálatban
lévő futószalag; a CUDA-gráfba rögzített mohó mintavételezés, a 0.6.35
alapértelmezése). A b=12 hivatalos, lepecsételt cella (6 összefésült ablak
mediánja, ablakonkénti órajel).

> **Erratum (2026. 09. 22.).** A 0.6.35 első közleménye a « +1,84%-kal a vLLM előtt » állítást egy 09. 21-i, 1 596 t/s-os vLLM-referenciából vonta le, amely offline generálásból (`LLM().generate()`) származott, és nem összevethető egy szerverrel: nincs folytonos ütemezés, nincs meg az `acvram serve` útvonala. Kijavítva 09. 22-én egy váltakozó A/V-cellával (A1 V1 A2 V2 A3 V3) a `vllm serve` (HTTP) ellenében, ugyanazon a kártyán és ugyanazon az útvonalon, mint az `acvram serve`: a vLLM mediánja 1 782 t/s. Összevethető mérésen az acvram (1 634 t/s) b=12-nél mintegy 8%-kal a vLLM MÖGÖTT van, nem előtte. A J/token azonos órajelen továbbra is újramérés alatt áll.

09. 14. reggelén az acvram ugyanebben a cellában 630 t/s-on és 0,619
J/tokenen állt: a nyereség a Blackwell natív FP4 MMA-jából
(`mma.sync … kind::mxf4nvf4`, ×7,9 a bf16-hoz képest), a kötegvödrönként
csoportosított GEMM-es MoE-ból, az egyetlen kernelben végzett útválasztásból
(3 677 → 1 517 indítás lépésenként) és a vetítésekhez használt keskeny
tensormagos GEMM-ből jön. Minden számnak megvan a jegyzete az
`acvram-memoire/revue/` alatt a mérés előtt lepecsételt előrejelzéssel, a
műszerrel és annak üzemével — üzem nélküli szám nem jelenik meg.

Ahol az acvram elöl jár: MLA-modellek (GLM-4.7-Flash) natív sm_120 NVFP4-ben,
amelyeket a vLLM csak FP8-ban szolgál ki (b=1: 165,35 t/s szolgálatban); a
VRAM-ba nem férő modellek; és az egyetlen szekvencia dekódolása (b=1: 380,8 t/s a
vLLM 290,6-ával szemben). Nagy kötegnél viszont, egy VRAM-ba férő MoE-n, a vLLM
b=12-nél elöl marad (1 782 az 1 634 t/s-sal szemben, vö. erratum); az acvram itt
előrelépett (1 540 a 0.6.34-ben → 1 634), de nem került elé. Az energiában
mutatkozó eltérést újra kell mérni.

## Állapot

0.6.35-es verzió. Minden az 5090-en fut: `sm_120a`-ra (natív FP4) és
`sm_86`-ra fordított CUDA-kernelek, CUDA-gráfok, NVFP4/INT8/INT4 kvantálás,
HTTP-szerver. Korlátok a helyükön: a kártya láthatatlan a munkameneteknek
(`CUDA_VISIBLE_DEVICES` üres), és csak az `outils/carte.sh` adja kölcsön,
zár alatt, egyszerre egy mérésnek; egy őr naplóz minden záron kívüli
hozzáférést; az egynél több kártyát lefedő vagy 10 s-nál rövidebb energiamérés
érvénytelen; a lefokozott üzemben betöltött modell ezt kimondja, és nem száll
párbajba.

640 teszt (`pytest -q`, egy perc processzoron; a GPU-tesztek csak a `carte.sh`
alatt futnak). Munkakövetés: `acvram-memoire/` (szabályok, névtár, füzetek,
180 jegyzet áttekintése).

## Támogatás

Az acvram fejlesztése személyes hardveren folyik. Ha a projekt hasznos
Önnek: **Támogatás: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licenc

GPL-3.0 vagy újabb.
