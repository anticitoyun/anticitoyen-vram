<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Donar suport: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Una passarel·la d'inferència compatible amb l'API d'OpenAI, que tracta la
memòria com una jerarquia i dóna a cada GPU el format numèric que el seu
silici llegeix millor.

Dissenyada per a una màquina concreta:

| | |
|---|---|
| Processador | Intel Core i9-14900K (8 nuclis P + 16 nuclis E) |
| Placa base | ASUS ROG Maximus Z790 Dark Hero |
| Memòria | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13); les dues targetes en PCIe x8/x8, limitades a 400 W / 275 W |

## Les dues idees

**Un format per GPU.** La RTX 5090 té tensor cores FP4; la RTX 3080 Ti no en
té, ni tampoc FP8. Alinear-les en un format comú malbarataria la 5090. El
convertidor escriu per tant *dues vegades el mateix model*, en el format que
cada destinació sap explotar de debò:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesos | **NVFP4** — E2M1 + escala FP8 E4M3 cada 16 | **INT4** — uint4 + escala i zero fp16 cada 128 |
| bits per pes | 4,50 | 4,16 |
| davant del BF16 | ×3,56 més petit | ×3,85 més petit |
| mode de càlcul | tensor cores FP4 | desquantitzat a FP16 dins el nucli, tensor cores FP16 |
| memòria cau KV | INT8 | INT8 |

32 GB de VRAM a 4,5 bits per pes contenen uns **56 mil milions de
paràmetres**, contra 16 mil milions en BF16. Entre les dues targetes, això fa
aproximadament **78 mil milions de paràmetres residents** abans i tot de tocar
la memòria RAM.

**La memòria és una jerarquia, no un mur.** Tres nivells, i el planificador
mesura què costa cadascun en lloc d'esperar que el model hi càpiga:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Inici ràpid

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Qualsevol client OpenAI s'hi connecta després:

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

## Què diu `acvram plan`

El planificador mereix executar-se abans de qualsevol descàrrega. Respon a
les preguntes que decideixen si un model és utilitzable en aquesta màquina:

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

Explora l'espai de configuracions en lloc de quedar-se amb la primera que hi
cap, i dues de les seves decisions són prou contraintuïtives per merèixer
enunciar-se:

* **Deixa la 3080 Ti sense fer servir** quan un model cap només a la 5090.
  Els trams d'un pipeline s'executen en sèrie: afegir una etapa a 912 GB/s en
  un pipeline a 1790 GB/s alenteix la descodificació d'un sol flux. Es força
  amb `--gpus all`.
* **Redueix la memòria cau KV per mantenir els pesos a la VRAM.** Cada
  gigabyte donat a la memòria cau és un gigabyte de pesos empès al bus PCIe, i
  llegir un pes pel PCIe costa unes trenta vegades el que costa des de la VRAM.
  Al 70B de dalt, només aquest arbitratge passa de 2,3 a 17,8 tokens/s.

## Anar de pressa

Quatre optimitzacions, cadascuna verificada per una prova d'equivalència i no
només per un cronòmetre: una optimització que canvia la resposta és un error.

### Descodificació especulativa (`--speculative`)

Descodificar un token amb un lot de mida 1 està limitat per la memòria: la
màquina llegeix tots els pesos actius per produir un sol token. Verificar K
tokens proposats llegeix aquests mateixos pesos **una sola vegada**. Dos
proposants:

* `ngram` (per defecte) — cerca el sufix actual més enrere en el context i
  proposa el que seguia. No costa res, no demana cap model. Rendible quan la
  sortida copia l'entrada: edició de codi, RAG, resum.
* `draft` — un model petit en un segon dispositiu. En aquest equip aquest
  dispositiu és la RTX 3080 Ti, que el planificador deixa voluntàriament
  ociosa per a qualsevol model que càpiga a la 5090.

L'acceptació és exacta, no aproximada: una proposta s'accepta amb probabilitat
`min(1, p/q)` i un rebuig remostreja a la part positiva normalitzada de
`p - q`. Mesurat sobre 40 000 extraccions davant d'un esborrany deliberadament
mal calibrat, la distribució emesa es queda a 0,002 de variació total de
l'objectiu — l'especulació compra velocitat, mai una resposta diferent.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Memòria cau de prefix (activa per defecte)

Els blocs s'adrecen pel hash *encadenat* del seu tram de tokens: dues
peticions que comparteixen una consigna de sistema comparteixen els seus
blocs, i la segona ja no els ha de precalcular. L'encadenament és
indispensable: els mateixos setze tokens en un context diferent no contenen
les mateixes claus i valors, i fer el hash només del tram serviria la memòria
cau d'una seqüència a una altra.

Un bloc alliberat el contingut del qual continua sent identificable passa a
una cua LRU en lloc de la llista de blocs lliures: la memòria cau sobreviu
així entre peticions sense refusar mai una assignació que hauria pogut servir.

### Càlcul al nivell amfitrió (`--host-exec`)

Una capa els pesos de la qual resideixen a la RAM pot copiar-se a la GPU o
calcular-se in situ. Els dos camins estan limitats per la memòria i llegeixen
els mateixos bytes: el més ràpid és el del bus més ample — el PCIe 5.0 x16
dóna uns 54 GB/s, la DDR5 en doble canal uns 70 GB/s — i calcular in situ deixa
a més la GPU lliure en lloc de fer-la esperar una còpia.

Això només val si el processador llegeix directament els pesos empaquetats a
4 bits. D'aquí un petit nucli C++ amb un camí AVX2 (`acvram_cpu.cpp`, carregat
per ctypes, sense capçaleres Python ni ninja). Fins i tot en la seva branca
**escalar** de reserva, supera `dequantize() @ x` per un factor 1,44 en INT4 i
3,21 en NVFP4, perquè aquest últim escriu primer una còpia de 32 bits de tota
la matriu.

En Mistral-Large-123B, l'estimació del planificador passa d'1,35 a
2,42 tokens/s.

### Precisió mixta (`--snr-floor`, desactivada per defecte)

El convertidor mesura la relació senyal/soroll a la sortida de cada capa per a
cada tensor i pot promoure a un format més ample els que cauen sota
`--snr-floor`, amb un límit del 15 % dels tensors i un preu màxim
(`--promotion-cout-max`, en mebibytes afegits).

El llindar val **zero per defecte**: no es promou res. La descodificació està
limitada per l'amplada de banda de memòria, i la mesura en `Huihui-Qwen3.8-27B`
ho decideix — un llindar de 25 dB costa un 13,4 % de memòria i un 10,6 % de
cabal (18,50 GiB i 41,8 t/s contra 16,02 i 46,2) per un 2,0 % de perplexitat
(42,591 contra 43,447, corpus de 16 383 tokens). `--snr-floor 25` restableix
l'antic comportament quan la qualitat prima sobre la velocitat.

### I `acvram eval`

La relació senyal/soroll i el cosinus dels logits són aproximacions.
`acvram eval DIR [DIR ...]` mesura la perplexitat per finestra lliscant,
perquè una elecció de format es decideixi amb proves:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Punts d'entrada HTTP

| punt d'entrada | notes |
|---|---|
| `POST /v1/chat/completions` | flux SSE o resposta única; fa servir la plantilla de conversa del model |
| `POST /v1/completions` | prompt en text o en identificadors de tokens |
| `POST /v1/embeddings` | estats ocults finals mitjanats, normalitzats L2, `dimensions` respectat |
| `GET /v1/models` | més un bloc `acvram`: formats, dispositius, capacitat de la memòria cau KV |
| `GET /health`, `GET /metrics` | cabal de descodificació, ocupació dels blocs KV |

Els noms de camp d'aquestes respostes es queden en anglès: és el protocol
OpenAI, i traduir-los trencaria tots els clients existents.

## D'on surten les xifres

Cada valor citat més amunt el produeix codi d'aquest dipòsit i el verifica
`pytest`. Mesures fetes en processador amb els nuclis de referència:

| format | bits/pes | SNR dels pesos | cosinus dels logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dues constatacions sorgides d'aquestes mesures han canviat els valors per
defecte:

* **Una rotació de Hadamard ajuda l'INT4 i no el NVFP4.** Els grups de 128 de
  l'INT4 no poden absorbir un canal aberrant aïllat, de manera que repartir els
  valors extrems val una transformada en n log n per activació. Els blocs de 16
  del NVFP4 ja porten la seva pròpia escala. D'aquí `--hadamard auto`, que
  només l'aplica a l'INT4.
* **L'INT8 supera el FP8 E4M3 per a la memòria cau KV**, 44 dB contra 32 dB a
  mida idèntica, perquè una escala per (token, cap) ja proporciona el rang
  dinàmic en què el FP8 gasta bits d'exponent. Les dues targetes fan servir
  per tant una memòria cau KV en INT8, encara que la 5090 sabria fer FP8.

## Documentació

* [`REPRISE.md`](../REPRISE.md) — **reprendre el projecte en una altra màquina**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — com s'encaixen les peces
* [`docs/MATERIEL.md`](MATERIEL.md) — ajustar aquesta màquina concreta
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **el que no està fet**, a llegir primer
* [`CONVENTIONS.md`](../CONVENTIONS.md) — convencions de treball sobre el codi (llengua, estil, controls abans d'empènyer)

## Resultats mesurats (22/09/2026, RTX 5090 a 400 W, règim ≥ 20 s al comptador d'energia)

Qwen3-Coder-30B-A3B en NVFP4 (experts) + INT8 (atenció, cap), mateix
protocol per a tots els motors (`outils/`, una targeta, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| descodificació 12 seqüències | **1 634 t/s** | 1 782 t/s | — |
| descodificació 1 seqüència | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, retirat) |

Cabal del dia (estació 1030, règim eco `-lgc 2700`, pipeline en servei;
mostreig cobejós capturat al graf CUDA, activat per defecte a 0.6.35). El b=12
és la cel·la oficial segellada (mediana de 6 finestres intercalades).

> **Fe d'errates (22/09/2026).** La primera publicació de 0.6.35 treia
> «+1,84 % davant de vLLM» d'una referència vLLM de 1 596 t/s del 21/09 que
> venia d'una **generació fora de línia** (`LLM().generate()`), **no comparable
> amb un servidor**: sense planificació contínua, sense el camí d'`acvram
> serve`. Corregit el 22/09 amb una cel·la alternada A/V (A1 V1 A2 V2 A3 V3)
> contra **`vllm serve`** (HTTP), la mateixa targeta i el mateix camí que
> `acvram serve`: vLLM mediana **1 782 t/s**. A mesura comparable, **acvram
> (1 634 t/s) va DARRERE de vLLM aproximadament un 8 % a b=12**, no davant. El
> J/token a rellotge igual encara s'està tornant a mesurar.

El matí del 14/09 acvram era a 630 t/s i 0,619 J/token a la mateixa cel·la:
els guanys vénen de la MMA FP4 nativa de Blackwell (`mma.sync …
kind::mxf4nvf4`, ×7,9 sobre el bf16), del MoE en GEMM agrupada per galleda de
lot, d'un encaminament en un sol nucli (3 677 → 1 517 llançaments per pas) i
d'una GEMM estreta en tensor cores per a les projeccions. Cada xifra té la
seva nota a `acvram-memoire/revue/` amb la predicció segellada abans de la
mesura, l'instrument i el seu règim — una xifra sense règim no es publica.

On acvram va al davant: models MLA (GLM-4.7-Flash) en NVFP4 natiu sm_120, que
vLLM només serveix en FP8 (b=1: 165,35 t/s en servei); els models que no
caben a la VRAM; i la descodificació amb seqüència única (b=1: 380,8 t/s contra
290,6 de vLLM). En canvi, amb lot gran, sobre un MoE que cap a la VRAM, vLLM es
manté al davant a b=12 (1 782 contra 1 634 t/s, vegeu la fe d'errates); acvram
hi ha progressat (1 540 a 0.6.34 → 1 634) sense passar al davant. La diferència
en energia s'ha de tornar a mesurar.

## Estat

Versió 0.6.35. Tot funciona a la 5090: nuclis CUDA compilats per a `sm_120a`
(FP4 natiu) i `sm_86`, grafs CUDA, quantització NVFP4/INT8/INT4, servidor
HTTP. Salvaguardes en marxa: la targeta és invisible per a les sessions de
treball (`CUDA_VISIBLE_DEVICES` buit) i només `outils/carte.sh` la presta,
sota forrellat, a una mesura alhora; un vigilant registra tot accés fora del
forrellat; una mesura d'energia que cobreixi més d'una targeta o menys de 10 s
queda invalidada; un model carregat en règim degradat ho diu i no entra en un
duel.

640 tests (`pytest -q`, un minut en processador; els tests GPU només corren
sota `carte.sh`). Seguiment de la feina: `acvram-memoire/` (regles,
directori, quaderns, revisió de 180 notes).

## Donar suport

El desenvolupament d'acvram es fa amb material personal. Si el projecte us és
útil: **Donar suport: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Llicència

GPL-3.0 o posterior.
