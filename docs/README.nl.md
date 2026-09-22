<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Steunen: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Een inferentie-gateway die compatibel is met de OpenAI-API, het geheugen als
een hiërarchie behandelt en elke GPU het numerieke formaat geeft dat zijn
silicium het best leest.

Ontworpen voor één specifieke machine:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-kernen + 16 E-kernen) |
| Moederbord | ASUS ROG Maximus Z790 Dark Hero |
| Geheugen | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Systeem | Ubuntu 26.04 LTS (CUDA 13); beide kaarten op PCIe x8/x8, begrensd op 400 W / 275 W |

## De twee ideeën

**Eén formaat per GPU.** De RTX 5090 heeft FP4-tensorcores; de RTX 3080 Ti
heeft die niet, en ook geen FP8. Beide op een gemeenschappelijk formaat zetten
zou de 5090 verspillen. De converter schrijft daarom *hetzelfde model twee
keer*, in het formaat dat elke bestemming werkelijk kan benutten:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| gewichten | **NVFP4** — E2M1 + FP8 E4M3-schaal per 16 | **INT4** — uint4 + fp16-schaal en -nulpunt per 128 |
| bits per gewicht | 4,50 | 4,16 |
| tegenover BF16 | ×3,56 kleiner | ×3,85 kleiner |
| rekenmodus | FP4-tensorcores | in de kernel gedequantiseerd naar FP16, FP16-tensorcores |
| KV-cache | INT8 | INT8 |

32 GB VRAM bij 4,5 bits per gewicht bevat ongeveer **56 miljard parameters**,
tegen 16 miljard in BF16. Over beide kaarten is dat ruwweg **78 miljard
residente parameters** nog voordat het werkgeheugen wordt aangesproken.

**Geheugen is een hiërarchie, geen muur.** Drie lagen, en de planner meet wat
elke laag kost in plaats van te hopen dat het model past:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Snel van start

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Elke OpenAI-client kan er vervolgens op aansluiten:

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

## Wat `acvram plan` zegt

De planner is het waard om vóór elke download te draaien. Hij beantwoordt de
vragen die bepalen of een model op deze machine bruikbaar is:

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

Hij verkent de configuratieruimte in plaats van de eerste passende oplossing
te nemen, en twee van zijn beslissingen zijn contra-intuïtief genoeg om ze uit
te spreken:

* **Hij laat de 3080 Ti ongebruikt** wanneer een model alleen op de 5090
  past. De delen van een pijplijn draaien in serie: een trap van 912 GB/s
  toevoegen aan een pijplijn van 1790 GB/s vertraagt het decoderen van één
  stroom. Forceren kan met `--gpus all`.
* **Hij verkleint de KV-cache om de gewichten in VRAM te houden.** Elke
  gigabyte voor de cache is een gigabyte gewichten die naar de PCIe-bus wordt
  verdrongen, en een gewicht via PCIe lezen kost ongeveer dertig keer zoveel
  als vanuit VRAM. Bij de 70B hierboven brengt alleen deze afweging 2,3 → 17,8
  tokens/s.

## Snel gaan

Vier optimalisaties, elk geverifieerd met een equivalentiebewijs en niet
alleen met een stopwatch: een optimalisatie die het antwoord verandert, is een
bug.

### Speculatief decoderen (`--speculative`)

Eén token decoderen met een batch van 1 is geheugengebonden: de machine leest
alle actieve gewichten om één enkel token te produceren. K voorgestelde tokens
verifiëren leest diezelfde gewichten **één keer**. Twee voorstellers:

* `ngram` (standaard) — zoekt het huidige suffix eerder in de context en
  stelt voor wat erop volgde. Kost niets, heeft geen model nodig. Loont
  wanneer de uitvoer de invoer overneemt: code bewerken, RAG, samenvatten.
* `draft` — een klein model op een tweede apparaat. Op dit rig is dat de
  RTX 3080 Ti, die de planner bewust inactief laat voor elk model dat op de
  5090 past.

De acceptatie is exact, niet benaderend: een voorstel wordt aanvaard met kans
`min(1, p/q)` en een afwijzing herbemonstert uit het genormaliseerde positieve
deel van `p - q`. Gemeten over 40 000 trekkingen tegen een opzettelijk slecht
gekalibreerde ontwerpversie blijft de uitgezonden verdeling binnen 0,002
totale variatie van het doel — speculatie koopt snelheid, nooit een ander
antwoord.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefixcache (standaard actief)

Blokken worden geadresseerd via de *geketende* hash van hun tokenreeks: twee
verzoeken die een systeeminstructie delen, delen haar blokken, en het tweede
hoeft ze niet meer voor te berekenen. Het ketenen is onmisbaar: dezelfde
zestien tokens in een andere context bevatten niet dezelfde sleutels en
waarden, en alleen de reeks hashen zou de cache van de ene sequentie aan een
andere uitleveren.

Een vrijgegeven blok waarvan de inhoud herkenbaar blijft, gaat naar een
LRU-wachtrij in plaats van naar de vrije lijst: zo overleeft de cache tussen
verzoeken zonder ooit een toewijzing te weigeren die hij had kunnen bedienen.

### Rekenen op de hostlaag (`--host-exec`)

Een laag waarvan de gewichten in RAM staan, kan naar de GPU worden gekopieerd
of ter plaatse worden berekend. Beide paden zijn geheugengebonden en lezen
dezelfde bytes: het snelste is dat met de breedste bus — PCIe 5.0 x16 geeft
ongeveer 54 GB/s, DDR5 in dual channel ongeveer 70 GB/s — en ter plaatse
rekenen laat bovendien de GPU vrij in plaats van hem op een kopie te laten
wachten.

Dat loont alleen als de processor de 4-bits ingepakte gewichten rechtstreeks
leest. Vandaar een kleine C++-kernel met een AVX2-pad (`acvram_cpu.cpp`,
geladen via ctypes, zonder Python-headers of ninja). Zelfs op zijn **scalaire**
terugvaltak verslaat hij `dequantize() @ x` met een factor 1,44 in INT4 en 3,21
in NVFP4, omdat die laatste eerst een 32-bits kopie van de hele matrix
schrijft.

Bij Mistral-Large-123B gaat de schatting van de planner van 1,35 naar
2,42 tokens/s.

### Gemengde precisie (`--snr-floor`, standaard uit)

De converter meet de signaal-ruisverhouding aan de uitgang van elke laag voor
elke tensor en kan de tensoren die onder `--snr-floor` vallen naar een breder
formaat promoveren, binnen een grens van 15 % van de tensoren en een
prijsplafond (`--promotion-cout-max`, in toegevoegde mebibytes).

De drempel is **standaard nul**: niets wordt gepromoveerd. Decoderen is
begrensd door de geheugenbandbreedte, en de meting op `Huihui-Qwen3.8-27B`
beslist — een drempel van 25 dB kost 13,4 % geheugen en 10,6 % doorvoer
(18,50 GiB en 41,8 t/s tegenover 16,02 en 46,2) voor 2,0 % perplexiteit
(42,591 tegenover 43,447, corpus van 16 383 tokens). `--snr-floor 25` herstelt
het oude gedrag wanneer kwaliteit boven snelheid gaat.

### En `acvram eval`

De signaal-ruisverhouding en de logit-cosinus zijn benaderingen.
`acvram eval MAP [MAP ...]` meet de perplexiteit met een schuivend venster,
zodat een formaatkeuze op bewijs wordt beslist:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-eindpunten

| eindpunt | opmerkingen |
|---|---|
| `POST /v1/chat/completions` | SSE-stroom of enkel antwoord; gebruikt het chatsjabloon van het model |
| `POST /v1/completions` | prompt als tekst of als token-id's |
| `POST /v1/embeddings` | gemiddelde laatste verborgen toestanden, L2-genormaliseerd, `dimensions` gerespecteerd |
| `GET /v1/models` | plus een blok `acvram`: formaten, apparaten, capaciteit van de KV-cache |
| `GET /health`, `GET /metrics` | decodeerdoorvoer, bezetting van de KV-blokken |

De veldnamen van deze antwoorden blijven Engels: het is het OpenAI-protocol,
en ze vertalen zou elke bestaande client breken.

## Waar de cijfers vandaan komen

Elke hierboven genoemde waarde wordt geproduceerd door code uit deze
repository en gecontroleerd met `pytest`. Metingen op de processor met de
referentiekernels:

| formaat | bits/gewicht | SNR van de gewichten | logit-cosinus vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Twee bevindingen uit deze metingen hebben de standaardwaarden veranderd:

* **Een Hadamard-rotatie helpt INT4 en niet NVFP4.** De groepen van 128 van
  INT4 kunnen een geïsoleerd uitschieterkanaal niet opvangen, zodat het
  spreiden van de extreme waarden een n log n-transformatie per activatie
  waard is. De blokken van 16 van NVFP4 dragen al hun eigen schaal. Vandaar
  `--hadamard auto`, dat het alleen op INT4 toepast.
* **INT8 verslaat FP8 E4M3 voor de KV-cache**, 44 dB tegenover 32 dB bij
  gelijke grootte, omdat een schaal per (token, kop) al het dynamisch bereik
  levert waar FP8 exponentbits aan besteedt. Beide kaarten gebruiken daarom een
  KV-cache in INT8, ook al zou de 5090 FP8 kunnen.

## Documentatie

* [`REPRISE.md`](../REPRISE.md) — **het project op een andere machine hervatten**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — hoe de onderdelen in elkaar passen
* [`docs/MATERIEL.md`](MATERIEL.md) — deze specifieke machine afstellen
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **wat niet gedaan is**, eerst lezen
* [`CONVENTIONS.md`](../CONVENTIONS.md) — werkafspraken voor de code (taal, stijl, controles vóór het pushen)

## Gemeten resultaten (22/09/2026, RTX 5090 op 400 W, regime ≥ 20 s op de energiemeter)

Qwen3-Coder-30B-A3B in NVFP4 (experts) + INT8 (attentie, kop), hetzelfde
protocol voor alle engines (`outils/`, één kaart, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| decoderen, 12 sequenties | **1 625,5 t/s** | 1 596,1 t/s | — |
| decoderen, 1 sequentie | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, ingetrokken) |

Doorvoer van de dag (werkstation 1030, eco-regime `-lgc 2700`, pipeline in
bedrijf; greedy sampling vastgelegd in de CUDA-graaf, standaard sinds 0.6.35).
b=12 is een verzegelde officiële cel (mediaan van 6 verweven vensters, klok per
venster). De vLLM-waarde 1 596,1 is de bevroren referentie van 21/09 (vLLM die
dag niet opnieuw uitgevoerd): het verschil +1,84 % geldt bij gelijke referentie,
niet als hermeting van beide op dezelfde ochtend. De J/token bij gelijke klok
tegen de drie engines wordt opnieuw gemeten
(`outils/gpu/mesure/banc-4moteurs.py`) — een getal zonder regime wordt niet
gepubliceerd.

Op de ochtend van 14/09 zat acvram op 630 t/s en 0,619 J/token in dezelfde
cel: de winst komt van Blackwells native FP4-MMA (`mma.sync …
kind::mxf4nvf4`, ×7,9 ten opzichte van bf16), van MoE als gegroepeerde GEMM
per batch-emmer, van routering in één kernel (3 677 → 1 517 starts per stap) en
van een smalle tensorcore-GEMM voor de projecties. Elk cijfer heeft zijn notitie
in `acvram-memoire/revue/` met de vóór de meting verzegelde voorspelling, het
instrument en zijn regime — een cijfer zonder regime wordt niet gepubliceerd.

Waar acvram voorop loopt: MLA-modellen (GLM-4.7-Flash) in native sm_120-NVFP4,
die vLLM alleen in FP8 bedient (b=1: 165,35 t/s in bedrijf); modellen die niet in
VRAM passen; en, sinds 0.6.35, het decoderen met grote batch van een MoE dat in
VRAM past — b=12 gaat van 1 540 (0.6.34) naar 1 625,5 t/s, oftewel +1,84 % vóór
de bevroren vLLM-referentie (1 596,1). Het verschil blijft klein en bij bevroren
referentie; het energieverschil moet opnieuw worden gemeten.

## Stand van zaken

Versie 0.6.35. Alles draait op de 5090: CUDA-kernels gecompileerd voor
`sm_120a` (native FP4) en `sm_86`, CUDA-grafen, NVFP4/INT8/INT4-kwantisatie,
HTTP-server. Vangrails aanwezig: de kaart is onzichtbaar voor werksessies
(`CUDA_VISIBLE_DEVICES` leeg) en alleen `outils/carte.sh` leent haar, onder
vergrendeling, aan één meting tegelijk; een wachter logt elke toegang buiten de
vergrendeling; een energiemeting over meer dan één kaart of korter dan 10 s
wordt ongeldig verklaard; een model dat in gedegradeerd regime is geladen, zegt
dat en doet niet mee aan een duel.

640 tests (`pytest -q`, één minuut op de processor; GPU-tests draaien alleen
onder `carte.sh`). Werkopvolging: `acvram-memoire/` (regels, register,
schriften, review van 180 notities).

## Steunen

acvram wordt ontwikkeld op persoonlijke hardware. Als het project u van nut
is: **Steunen: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licentie

GPL-3.0 of later.
