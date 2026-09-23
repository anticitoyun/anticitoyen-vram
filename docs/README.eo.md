<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Subteni: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Inferenca kluzo kongrua kun la API de OpenAI, kiu traktas la memoron kiel
hierarkion kaj donas al ĉiu GPU la nombran formaton, kiun ĝia silicio plej
bone legas.

Konceptita por unu preciza maŝino:

| | |
|---|---|
| Procesoro | Intel Core i9-14900K (8 P-kernoj + 16 E-kernoj) |
| Ĉefplato | ASUS ROG Maximus Z790 Dark Hero |
| Memoro | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistemo | Ubuntu 26.04 LTS (CUDA 13); ambaŭ kartoj ĉe PCIe x8/x8, limigitaj al 400 W / 275 W |

## La du ideoj

**Unu formato por ĉiu GPU.** La RTX 5090 havas FP4-tensorkernojn; la RTX
3080 Ti ne havas ilin, kaj ankaŭ ne FP8. Alĝustigi ambaŭ al komuna formato
malŝparus la 5090. La konvertilo do skribas *dufoje la saman modelon*, en la
formato, kiun ĉiu celo vere kapablas ekspluati:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pezoj | **NVFP4** — E2M1 + skalo FP8 E4M3 ĉiun 16 | **INT4** — uint4 + skalo kaj nulo fp16 ĉiun 128 |
| bitoj por pezo | 4,50 | 4,16 |
| kompare kun BF16 | ×3,56 pli malgranda | ×3,85 pli malgranda |
| kalkulmaniero | FP4-tensorkernoj | malkvantigita al FP16 en la kerno, FP16-tensorkernoj |
| KV-kaŝmemoro | INT8 | INT8 |

32 GB da VRAM je 4,5 bitoj por pezo enhavas ĉirkaŭ **56 miliardojn da
parametroj**, kontraŭ 16 miliardoj en BF16. Sur ambaŭ kartoj tio faras
proksimume **78 miliardojn da loĝantaj parametroj** eĉ antaŭ ol tuŝi la
ĉefmemoron.

**La memoro estas hierarkio, ne muro.** Tri etaĝoj, kaj la planilo mezuras,
kiom ĉiu kostas, anstataŭ esperi, ke la modelo enkonveniĝos:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Rapida komenco

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Iu ajn OpenAI-kliento poste konektiĝas:

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

## Kion diras `acvram plan`

La planilo meritas esti lanĉita antaŭ ĉia elŝuto. Ĝi respondas la demandojn,
kiuj decidas, ĉu modelo estas uzebla sur ĉi tiu maŝino:

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

Ĝi esploras la spacon de agordoj anstataŭ konservi la unuan, kiu enkonvenas,
kaj du el ĝiaj decidoj estas sufiĉe kontraŭintuiciaj por meriti esti diritaj:

* **Ĝi lasas la 3080 Ti neuzata**, kiam modelo enkonvenas sur la sola 5090.
  La tranĉoj de dukto plenumiĝas serie: aldoni etapon je 912 GB/s en dukton
  je 1790 GB/s malrapidigas la unufluan malkodadon. Oni devigas per
  `--gpus all`.
* **Ĝi malgrandigas la KV-kaŝmemoron por teni la pezojn en VRAM.** Ĉiu
  gigabajto donita al la kaŝmemoro estas gigabajto da pezoj forpuŝita sur la
  buson PCIe, kaj legi pezon tra PCIe kostas ĉirkaŭ tridekoble tion, kion ĝi
  kostas el VRAM. Sur la supra 70B, ĉi tiu sola arbitracio pasigas de 2,3 al
  17,8 ĵetonoj/s.

## Rapidi

Kvar optimumigoj, ĉiu kontrolita per ekvivalenteca pruvo kaj ne nur per
kronometro: optimumigo, kiu ŝanĝas la respondon, estas cimo.

### Spekulativa malkodado (`--speculative`)

Malkodi unu ĵetonon kun aro de grando 1 estas limigita de la memoro: la
maŝino legas ĉiujn aktivajn pezojn por produkti unu solan ĵetonon. Kontroli K
proponitajn ĵetonojn legas tiujn samajn pezojn **unu solan fojon**. Du
proponantoj:

* `ngram` (defaŭlte) — serĉas la nunan sufikson pli frue en la kunteksto kaj
  proponas tion, kio sekvis. Kostas nenion, postulas neniun modelon. Rentas,
  kiam la eligo kopias la enigon: kodredaktado, RAG, resumo.
* `draft` — malgranda modelo sur dua aparato. Sur ĉi tiu maŝinaro tiu aparato
  estas la RTX 3080 Ti, kiun la planilo intence lasas senokupa por ĉiu modelo,
  kiu enkonvenas sur la 5090.

La akcepto estas ekzakta, ne proksimuma: propono estas akceptita kun
probablo `min(1, p/q)` kaj rifuzo respecimenas en la normigita pozitiva parto
de `p - q`. Mezurite sur 40 000 tiroj kontraŭ intence mise kalibrita
malneto, la eligita distribuo restas je 0,002 da totala variado de la celo —
la spekulado aĉetas rapidon, neniam alian respondon.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefiksa kaŝmemoro (aktiva defaŭlte)

La blokoj estas adresitaj per la *ĉenita* haketo de sia ĵetontranĉo: du
petoj, kiuj kunhavas sisteman instrukcion, kunhavas ĝiajn blokojn, kaj la dua
ne plu devas antaŭkalkuli ilin. La ĉenado estas nemalhavebla: la samaj dek ses
ĵetonoj en alia kunteksto ne enhavas la samajn ŝlosilojn kaj valorojn, kaj
haketi nur la tranĉon servus la kaŝmemoron de unu sekvenco al alia.

Liberigita bloko, kies enhavo restas identigebla, iras en LRU-vicon anstataŭ
en la liston de liberaj blokoj: la kaŝmemoro tiel postvivas inter la petoj
neniam rifuzante asignon, kiun ĝi povus servi.

### Kalkulado sur la gastiga etaĝo (`--host-exec`)

Tavolo, kies pezoj loĝas en RAM, povas esti kopiita al la GPU aŭ kalkulita
surloke. Ambaŭ vojoj estas limigitaj de la memoro kaj legas la samajn bajtojn:
la pli rapida estas tiu kun la pli larĝa buso — la PCIe 5.0 x16 donas ĉirkaŭ
54 GB/s, la DDR5 en duobla kanalo ĉirkaŭ 70 GB/s — kaj kalkuli surloke krome
lasas la GPU libera anstataŭ igi ĝin atendi kopion.

Tio valoras nur se la procesoro rekte legas la 4-bite pakitajn pezojn. De tie
malgranda C++-kerno kun AVX2-vojo (`acvram_cpu.cpp`, ŝargita per ctypes, sen
Python-kaplinioj nek ninja). Eĉ sur sia **skalara** rezerva branĉo, ĝi
superas `dequantize() @ x` per faktoro 1,44 en INT4 kaj 3,21 en NVFP4, ĉar ĉi
tiu lasta unue skribas 32-bitan kopion de la tuta matrico.

Sur Mistral-Large-123B, la takso de la planilo pasas de 1,35 al
2,42 ĵetonoj/s.

### Miksita precizeco (`--snr-floor`, malŝaltita defaŭlte)

La konvertilo mezuras la rilaton signalo/bruo ĉe la eligo de ĉiu tavolo por
ĉiu tensoro kaj povas promocii al pli larĝa formato tiujn, kiuj falas sub
`--snr-floor`, en la limo de 15 % de la tensoroj kaj de plafona prezo
(`--promotion-cout-max`, en aldonitaj mebibajtoj).

La planko valoras **nul defaŭlte**: nenio estas promociita. La malkodado
estas limigita de la memora bendolarĝo, kaj la mezuro sur `Huihui-Qwen3.8-27B`
decidas — planko de 25 dB kostas 13,4 % da memoro kaj 10,6 % da trafluo
(18,50 GiB kaj 41,8 ĵ/s kontraŭ 16,02 kaj 46,2) por 2,0 % da perplekseco
(42,591 kontraŭ 43,447, korpuso de 16 383 ĵetonoj). `--snr-floor 25` restarigas
la malnovan konduton, kiam la kvalito superas la rapidon.

### Kaj `acvram eval`

La rilato signalo/bruo kaj la kosinuso de la logitoj estas proksimumoj.
`acvram eval DOS [DOS ...]` mezuras la perpleksecon per glita fenestro, por ke
elekto de formato estu decidita per pruvoj:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-enirpunktoj

| enirpunkto | notoj |
|---|---|
| `POST /v1/chat/completions` | SSE-fluo aŭ unuopa respondo; uzas la konversacian ŝablonon de la modelo |
| `POST /v1/completions` | instigo kiel teksto aŭ kiel ĵetonaj identigiloj |
| `POST /v1/embeddings` | finaj kaŝitaj statoj averaĝigitaj, L2-normigitaj, `dimensions` respektata |
| `GET /v1/models` | plus bloko `acvram`: formatoj, aparatoj, kapacito de la KV-kaŝmemoro |
| `GET /health`, `GET /metrics` | malkoda trafluo, okupiteco de la KV-blokoj |

La kampnomoj de ĉi tiuj respondoj restas en la angla: tio estas la
OpenAI-protokolo, kaj traduki ilin rompus ĉiujn ekzistantajn klientojn.

## De kie venas la ciferoj

Ĉiu supre citita valoro estas produktita de kodo de ĉi tiu deponejo kaj
kontrolita de `pytest`. Mezuroj faritaj sur procesoro kun la referencaj
kernoj:

| formato | bitoj/pezo | SNR de la pezoj | kosinuso de la logitoj kontraŭ BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Du konstatoj el ĉi tiuj mezuroj ŝanĝis la defaŭltajn valorojn:

* **Hadamard-rotacio helpas la INT4 kaj ne la NVFP4.** La 128-grupoj de INT4
  ne povas sorbi izolitan eksterordinaran kanalon, tiel ke disvastigi la
  ekstremajn valorojn valoras n log n transformon por aktivigo. La 16-blokoj
  de NVFP4 jam portas sian propran skalon. De tie `--hadamard auto`, kiu
  aplikas ĝin nur al INT4.
* **INT8 superas FP8 E4M3 por la KV-kaŝmemoro**, 44 dB kontraŭ 32 dB je sama
  grando, ĉar skalo por (ĵetono, kapo) jam provizas la dinamikan amplekson,
  por kiu FP8 elspezas eksponentajn bitojn. Ambaŭ kartoj do uzas KV-kaŝmemoron
  en INT8, kvankam la 5090 scius fari FP8.

## Dokumentaro

* [`REPRISE.md`](../REPRISE.md) — **repreni la projekton sur alia maŝino**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — kiel la pecoj kunmetiĝas
* [`docs/MATERIEL.md`](MATERIEL.md) — agordi ĉi tiun precizan maŝinon
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **kio ne estas farita**, legenda unue
* [`CONVENTIONS.md`](../CONVENTIONS.md) — laborkonvencioj pri la kodo (lingvo, stilo, kontroloj antaŭ puŝo)

## Mezuritaj rezultoj (22/09/2026, RTX 5090 je 400 W, reĝimo ≥ 20 s ĉe la energimezurilo)

Qwen3-Coder-30B-A3B en NVFP4 (spertuloj) + INT8 (atento, kapo), sama
protokolo por ĉiuj motoroj (`outils/`, unu karto, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| malkodado 12 sekvencoj | **1 634 ĵ/s** | 1 782 ĵ/s | — |
| malkodado 1 sekvenco | **380,8 ĵ/s** | 290,6 ĵ/s | 323,6 ĵ/s |
| prefill pp2048 | **22 707 ĵetonoj/s** | 21 054 | 8 671 (TabbyAPI, retirita) |

Trafluoj de la tago (posteno 1030, ŝparreĝimo `-lgc 2700`, dukto en servo;
avida specimenado kaptita en la CUDA-grafo, defaŭlto de 0.6.35). La b=12 estas
oficiala sigelita ĉelo (mediano de 6 interplektitaj fenestroj, horloĝo laŭ
fenestro).

> **Erratum (22/09/2026).** La unua publikigo de 0.6.35 eltiris « +1,84 % antaŭ vLLM » el vLLM-referenco de 1 596 ĵ/s de la 21/09, kiu venis de senreta generado (`LLM().generate()`), ne komparebla kun servilo: sen kontinua planado, sen la vojo de `acvram serve`. Korektita la 22/09 per alterna A/V-ĉelo (A1 V1 A2 V2 A3 V3) kontraŭ `vllm serve` (HTTP), sama karto kaj sama vojo kiel `acvram serve`: vLLM mediano 1 782 ĵ/s. Je komparebla mezuro, acvram (1 634 ĵ/s) estas MALANTAŬ vLLM je ĉirkaŭ 8 % ĉe b=12, ne antaŭe. La J/ĵetono je egala horloĝo restas remezurata.

La 14/09 matene acvram estis je 630 ĵ/s kaj 0,619 J/ĵetono sur la sama ĉelo:
la gajnoj venas de la indiĝena FP4-MMA de Blackwell (`mma.sync …
kind::mxf4nvf4`, ×7,9 super bf16), de la MoE en GEMM grupigita laŭ arsitelo,
de enkursigo en unu sola kerno (3 677 → 1 517 lanĉoj por paŝo) kaj de mallarĝa
GEMM sur tensorkernoj por la projekcioj. Ĉiu cifero havas sian noton en
`acvram-memoire/revue/` kun la antaŭdiro sigelita antaŭ la mezuro, la
instrumento kaj ĝia reĝimo — cifero sen reĝimo ne estas publikigita.

Kie acvram antaŭas: MLA-modeloj (GLM-4.7-Flash) en indiĝena sm_120-NVFP4,
kiujn vLLM servas nur en FP8 (b=1: 165,35 ĵ/s en servo); la modeloj, kiuj ne
enkonvenas en VRAM; kaj la malkodado je unuopa sekvenco (b=1: 380,8 ĵ/s kontraŭ
290,6 por vLLM). Sed je granda lot, sur MoE kiu enkonvenas en VRAM, vLLM restas
antaŭe ĉe b=12 (1 782 kontraŭ 1 634 ĵ/s, kp. erratum); acvram tie progresis
(1 540 en 0.6.34 → 1 634) sen preterpasi. La diferenco en energio estas
remezurenda.

## Stato

Versio 0.6.35. Ĉio funkcias sur la 5090: CUDA-kernoj kompilitaj por `sm_120a`
(indiĝena FP4) kaj `sm_86`, CUDA-grafoj, NVFP4/INT8/INT4-kvantigo,
HTTP-servilo. Sekurbariloj surloke: la karto estas nevidebla por la
laborsesioj (`CUDA_VISIBLE_DEVICES` malplena) kaj nur `outils/carte.sh`
pruntedonas ĝin, sub ŝloso, al unu mezuro samtempe; gardisto protokolas ĉiun
aliron ekster la ŝloso; energimezuro kovranta pli ol unu karton aŭ malpli ol
10 s estas nuligita; modelo ŝargita en degradita reĝimo diras tion kaj ne
eniras duelon.

640 testoj (`pytest -q`, unu minuto sur procesoro; la GPU-testoj ruliĝas nur
sub `carte.sh`). Sekvado de la laboro: `acvram-memoire/` (reguloj, adresaro,
kajeroj, revuo de 180 notoj).

## Subteni

La evoluigo de acvram okazas sur persona aparataro. Se la projekto utilas al
vi: **Subteni: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Permesilo

GPL-3.0 aŭ posta.
