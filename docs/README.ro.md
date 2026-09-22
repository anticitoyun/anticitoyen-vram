<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Susține: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

O poartă de inferență compatibilă cu API-ul OpenAI, care tratează memoria ca o
ierarhie și dă fiecărui GPU formatul numeric pe care siliciul său îl citește
cel mai bine.

Concepută pentru o mașină precisă:

| | |
|---|---|
| Procesor | Intel Core i9-14900K (8 nuclee P + 16 nuclee E) |
| Placă de bază | ASUS ROG Maximus Z790 Dark Hero |
| Memorie | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistem | Ubuntu 26.04 LTS (CUDA 13); ambele plăci pe PCIe x8/x8, limitate la 400 W / 275 W |

## Cele două idei

**Un format pentru fiecare GPU.** RTX 5090 are tensor cores FP4; RTX 3080 Ti
nu are, și nici FP8. A le alinia pe un format comun ar irosi 5090.
Convertorul scrie deci *același model de două ori*, în formatul pe care fiecare
destinație știe cu adevărat să îl exploateze:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| ponderi | **NVFP4** — E2M1 + scară FP8 E4M3 la fiecare 16 | **INT4** — uint4 + scară și zero fp16 la fiecare 128 |
| biți pe pondere | 4,50 | 4,16 |
| față de BF16 | ×3,56 mai mic | ×3,85 mai mic |
| mod de calcul | tensor cores FP4 | decuantizat în FP16 în kernel, tensor cores FP16 |
| cache KV | INT8 | INT8 |

32 GB de VRAM la 4,5 biți pe pondere conțin aproximativ **56 de miliarde de
parametri**, față de 16 miliarde în BF16. Pe cele două plăci, asta înseamnă
aproximativ **78 de miliarde de parametri rezidenți** înainte chiar de a
atinge memoria RAM.

**Memoria este o ierarhie, nu un zid.** Trei etaje, iar planificatorul măsoară
cât costă fiecare în loc să spere că modelul încape:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Pornire rapidă

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Orice client OpenAI se conectează apoi:

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

## Ce spune `acvram plan`

Planificatorul merită rulat înaintea oricărei descărcări. Răspunde la
întrebările care decid dacă un model este utilizabil pe această mașină:

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

Explorează spațiul configurațiilor în loc să o rețină pe prima care încape,
iar două dintre deciziile sale sunt suficient de contraintuitive pentru a
merita enunțate:

* **Lasă 3080 Ti nefolosită** când un model încape doar pe 5090. Tranșele
  unui pipeline se execută în serie: a adăuga o etapă la 912 GB/s într-un
  pipeline la 1790 GB/s încetinește decodarea cu un singur flux. Se forțează
  cu `--gpus all`.
* **Micșorează cache-ul KV pentru a păstra ponderile în VRAM.** Fiecare
  gigabyte dat cache-ului este un gigabyte de ponderi împins pe magistrala
  PCIe, iar citirea unei ponderi prin PCIe costă de vreo treizeci de ori cât
  costă din VRAM. Pe 70B de mai sus, doar acest arbitraj trece de la 2,3 la
  17,8 tokeni/s.

## A merge repede

Patru optimizări, fiecare verificată printr-o dovadă de echivalență și nu doar
cu un cronometru: o optimizare care schimbă răspunsul este un defect.

### Decodare speculativă (`--speculative`)

A decoda un token cu un lot de mărime 1 este limitat de memorie: mașina citește
toate ponderile active pentru a produce un singur token. A verifica K tokeni
propuși citește aceleași ponderi **o singură dată**. Doi propunători:

* `ngram` (implicit) — caută sufixul curent mai devreme în context și propune
  ce urma. Nu costă nimic, nu cere niciun model. Rentabil când ieșirea copiază
  intrarea: editare de cod, RAG, rezumat.
* `draft` — un model mic pe un al doilea dispozitiv. Pe acest rig, dispozitivul
  este RTX 3080 Ti, pe care planificatorul o lasă intenționat inactivă pentru
  orice model care încape pe 5090.

Acceptarea este exactă, nu aproximativă: o propunere este acceptată cu
probabilitatea `min(1, p/q)`, iar un refuz reeșantionează din partea pozitivă
normalizată a lui `p - q`. Măsurat pe 40 000 de extrageri față de o ciornă
deliberat prost calibrată, distribuția emisă rămâne la 0,002 variație totală
de țintă — speculația cumpără viteză, niciodată un alt răspuns.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache de prefix (activ implicit)

Blocurile sunt adresate prin hash-ul *înlănțuit* al tranșei lor de tokeni:
două cereri care împart o instrucțiune de sistem împart blocurile ei, iar a
doua nu mai trebuie să le precalculeze. Înlănțuirea este indispensabilă:
aceiași șaisprezece tokeni într-un context diferit nu conțin aceleași chei și
valori, iar a hash-ui doar tranșa ar servi cache-ul unei secvențe alteia.

Un bloc eliberat al cărui conținut rămâne identificabil intră într-o coadă LRU
în loc de lista blocurilor libere: cache-ul supraviețuiește astfel între cereri
fără a refuza vreodată o alocare pe care ar fi putut-o servi.

### Calcul pe etajul gazdă (`--host-exec`)

Un strat ale cărui ponderi stau în RAM poate fi copiat pe GPU sau calculat pe
loc. Ambele căi sunt limitate de memorie și citesc aceiași octeți: cea mai
rapidă este cea cu magistrala mai largă — PCIe 5.0 x16 dă circa 54 GB/s, DDR5
pe canal dublu circa 70 GB/s — iar calculul pe loc lasă în plus GPU-ul liber în
loc să îl facă să aștepte o copie.

Asta merită doar dacă procesorul citește direct ponderile împachetate pe
4 biți. De aici un mic kernel C++ cu o cale AVX2 (`acvram_cpu.cpp`, încărcat
prin ctypes, fără antete Python și fără ninja). Chiar și pe ramura sa
**scalară** de rezervă, bate `dequantize() @ x` cu un factor de 1,44 în INT4 și
3,21 în NVFP4, pentru că acesta din urmă scrie mai întâi o copie pe 32 de biți
a întregii matrice.

Pe Mistral-Large-123B, estimarea planificatorului trece de la 1,35 la
2,42 tokeni/s.

### Precizie mixtă (`--snr-floor`, dezactivată implicit)

Convertorul măsoară raportul semnal/zgomot la ieșirea fiecărui strat pentru
fiecare tensor și îi poate promova către un format mai larg pe cei care cad
sub `--snr-floor`, în limita a 15 % din tensori și a unui preț plafon
(`--promotion-cout-max`, în mebiocteți adăugați).

Pragul este **zero implicit**: nimic nu este promovat. Decodarea este limitată
de lățimea de bandă a memoriei, iar măsura pe `Huihui-Qwen3.8-27B` tranșează —
un prag de 25 dB costă 13,4 % memorie și 10,6 % debit (18,50 GiB și 41,8 t/s
față de 16,02 și 46,2) pentru 2,0 % perplexitate (42,591 față de 43,447,
corpus de 16 383 tokeni). `--snr-floor 25` restabilește vechiul comportament
când calitatea primează asupra vitezei.

### Și `acvram eval`

Raportul semnal/zgomot și cosinusul logiților sunt aproximări.
`acvram eval DIR [DIR ...]` măsoară perplexitatea cu fereastră glisantă,
pentru ca o alegere de format să fie tranșată pe dovezi:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Puncte de intrare HTTP

| punct de intrare | note |
|---|---|
| `POST /v1/chat/completions` | flux SSE sau răspuns unic; folosește șablonul de conversație al modelului |
| `POST /v1/completions` | prompt ca text sau ca identificatori de tokeni |
| `POST /v1/embeddings` | stări ascunse finale mediate, normalizate L2, `dimensions` respectat |
| `GET /v1/models` | plus un bloc `acvram`: formate, dispozitive, capacitatea cache-ului KV |
| `GET /health`, `GET /metrics` | debit de decodare, ocuparea blocurilor KV |

Numele câmpurilor din aceste răspunsuri rămân în engleză: este protocolul
OpenAI, iar traducerea lor ar strica toți clienții existenți.

## De unde vin cifrele

Fiecare valoare citată mai sus este produsă de cod din acest depozit și
verificată cu `pytest`. Măsurători făcute pe procesor cu kernelurile de
referință:

| format | biți/pondere | SNR al ponderilor | cosinusul logiților vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Două constatări din aceste măsurători au schimbat valorile implicite:

* **O rotație Hadamard ajută INT4 și nu NVFP4.** Grupurile de 128 ale INT4 nu
  pot absorbi un canal aberant izolat, astfel că împrăștierea valorilor
  extreme merită o transformată în n log n pe activare. Blocurile de 16 ale
  NVFP4 își poartă deja propria scară. De aici `--hadamard auto`, care o
  aplică doar la INT4.
* **INT8 bate FP8 E4M3 pentru cache-ul KV**, 44 dB față de 32 dB la mărime
  identică, pentru că o scară pe (token, cap) oferă deja plaja dinamică pe
  care FP8 cheltuie biți de exponent. Ambele plăci folosesc deci un cache KV
  în INT8, chiar dacă 5090 ar ști să facă FP8.

## Documentație

* [`REPRISE.md`](../REPRISE.md) — **reluarea proiectului pe o altă mașină**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — cum se asamblează piesele
* [`docs/MATERIEL.md`](MATERIEL.md) — reglarea acestei mașini precise
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **ce nu este făcut**, de citit primul
* [`CONVENTIONS.md`](../CONVENTIONS.md) — convenții de lucru asupra codului (limbă, stil, controale înainte de push)

## Rezultate măsurate (22/09/2026, RTX 5090 la 400 W, regim ≥ 20 s la contorul de energie)

Qwen3-Coder-30B-A3B în NVFP4 (experți) + INT8 (atenție, cap), același
protocol pentru toate motoarele (`outils/`, o placă, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| decodare 12 secvențe | **1 625,5 t/s** | 1 596,1 t/s | — |
| decodare 1 secvență | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokeni/s** | 21 054 | 8 671 (TabbyAPI, retras) |

Debite din ziua curentă (post 1030, regim eco `-lgc 2700`, pipeline în
serviciu; eșantionare greedy capturată în graful CUDA, implicită din 0.6.35).
b=12 este o celulă oficială sigilată (mediana a 6 ferestre întrețesute, ceas pe
fereastră). Valoarea vLLM 1 596,1 este referința înghețată din 21/09 (vLLM
neruleat din nou în acea zi): diferența +1,84 % este valabilă la referință egală,
nu ca remăsurare a ambelor în aceeași dimineață. J/token la ceas egal față de
cele trei motoare este în curs de remăsurare
(`outils/gpu/mesure/banc-4moteurs.py`) — o cifră fără regim nu se publică.

În dimineața de 14/09 acvram era la 630 t/s și 0,619 J/token pe aceeași
celulă: câștigurile vin din MMA FP4 nativă a lui Blackwell (`mma.sync …
kind::mxf4nvf4`, ×7,9 față de bf16), din MoE în GEMM grupată pe găleată de
lot, dintr-o rutare într-un singur kernel (3 677 → 1 517 lansări pe pas) și
dintr-un GEMM îngust pe tensor cores pentru proiecții. Fiecare cifră are nota
sa în `acvram-memoire/revue/` cu predicția sigilată înainte de măsurare,
instrumentul și regimul său — o cifră fără regim nu se publică.

Unde acvram este în față: modelele MLA (GLM-4.7-Flash) în NVFP4 nativ sm_120,
pe care vLLM le servește doar în FP8 (b=1: 165,35 t/s în serviciu); modelele care
nu încap în VRAM; și, din 0.6.35, decodarea cu lot mare a unui MoE care încape în
VRAM — b=12 crește de la 1 540 (0.6.34) la 1 625,5 t/s, adică +1,84 % înaintea
referinței vLLM înghețate (1 596,1). Diferența rămâne strânsă și la referință
înghețată; diferența în energie este de remăsurat.

## Stare

Versiunea 0.6.35. Totul rulează pe 5090: kerneluri CUDA compilate pentru
`sm_120a` (FP4 nativ) și `sm_86`, grafuri CUDA, cuantizare NVFP4/INT8/INT4,
server HTTP. Garduri de protecție la locul lor: placa este invizibilă pentru
sesiunile de lucru (`CUDA_VISIBLE_DEVICES` gol) și doar `outils/carte.sh` o
împrumută, sub zăvor, unei singure măsurători odată; un paznic jurnalizează
orice acces în afara zăvorului; o măsurătoare de energie care acoperă mai mult
de o placă sau mai puțin de 10 s este invalidată; un model încărcat în regim
degradat o spune și nu intră într-un duel.

640 de teste (`pytest -q`, un minut pe procesor; testele GPU rulează doar sub
`carte.sh`). Urmărirea lucrului: `acvram-memoire/` (reguli, anuar, caiete,
revizie de 180 de note).

## Susține

Dezvoltarea acvram se face pe echipament personal. Dacă proiectul vă este
util: **Susține: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licență

GPL-3.0 sau ulterioară.
