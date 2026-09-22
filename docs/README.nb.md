<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Støtt: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

En inferensportal kompatibel med OpenAI-API-et, som behandler minnet som et
hierarki og gir hver GPU det tallformatet silisiumet dens leser best.

Laget for én bestemt maskin:

| | |
|---|---|
| Prosessor | Intel Core i9-14900K (8 P-kjerner + 16 E-kjerner) |
| Hovedkort | ASUS ROG Maximus Z790 Dark Hero |
| Minne | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); begge kortene på PCIe x8/x8, begrenset til 400 W / 275 W |

## De to ideene

**Ett format per GPU.** RTX 5090 har FP4-tensorkjerner; RTX 3080 Ti har ingen,
og heller ikke FP8. Å legge begge på et felles format ville sløse bort 5090.
Konverteren skriver derfor *den samme modellen to ganger*, i det formatet hver
destinasjon faktisk kan utnytte:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| vekter | **NVFP4** — E2M1 + FP8 E4M3-skala per 16 | **INT4** — uint4 + fp16-skala og -nullpunkt per 128 |
| bit per vekt | 4,50 | 4,16 |
| mot BF16 | ×3,56 mindre | ×3,85 mindre |
| beregningsmodus | FP4-tensorkjerner | dekvantisert til FP16 i kjernen, FP16-tensorkjerner |
| KV-hurtigbuffer | INT8 | INT8 |

32 GB VRAM ved 4,5 bit per vekt rommer omtrent **56 milliarder parametere**,
mot 16 milliarder i BF16. På begge kortene blir det omtrent **78 milliarder
residente parametere** før arbeidsminnet i det hele tatt røres.

**Minnet er et hierarki, ikke en mur.** Tre nivåer, og planleggeren måler hva
hvert av dem koster i stedet for å håpe at modellen får plass:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Hurtigstart

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Enhver OpenAI-klient kan deretter koble seg til:

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

## Hva `acvram plan` sier

Planleggeren fortjener å kjøres før enhver nedlasting. Den svarer på
spørsmålene som avgjør om en modell er brukbar på denne maskinen:

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

Den utforsker konfigurasjonsrommet i stedet for å beholde den første som
passer, og to av beslutningene dens er kontraintuitive nok til at de fortjener
å sies rett ut:

* **Den lar 3080 Ti stå ubrukt** når en modell får plass på 5090 alene.
  Skivene i en pipeline kjøres i serie: å legge til et trinn på 912 GB/s i en
  pipeline på 1790 GB/s bremser enkeltstrøms-dekoding. Det kan tvinges med
  `--gpus all`.
* **Den krymper KV-hurtigbufferen for å holde vektene i VRAM.** Hver gigabyte
  gitt til hurtigbufferen er en gigabyte vekter dyttet ut på PCIe-bussen, og å
  lese en vekt over PCIe koster rundt tretti ganger det den koster fra VRAM.
  På 70B-en ovenfor flytter denne ene avveiningen alene tallet fra 2,3 til
  17,8 token/s.

## Å være rask

Fire optimaliseringer, hver bekreftet av et ekvivalensbevis og ikke bare av en
stoppeklokke: en optimalisering som endrer svaret, er en feil.

### Spekulativ dekoding (`--speculative`)

Å dekode ett token med en batch av størrelse 1 er minnebegrenset: maskinen
leser alle aktive vekter for å produsere ett eneste token. Å verifisere K
foreslåtte token leser de samme vektene **én gang**. To forslagsstillere:

* `ngram` (standard) — leter etter det gjeldende suffikset tidligere i
  konteksten og foreslår det som fulgte. Koster ingenting, krever ingen
  modell. Lønnsomt når utdata kopierer inndata: koderedigering, RAG,
  oppsummering.
* `draft` — en liten modell på en annen enhet. På denne riggen er den enheten
  RTX 3080 Ti, som planleggeren med vilje lar stå ledig for enhver modell som
  får plass på 5090.

Aksepten er eksakt, ikke tilnærmet: et forslag aksepteres med sannsynlighet
`min(1, p/q)`, og et avslag trekker på nytt fra den normaliserte positive
delen av `p - q`. Målt over 40 000 trekninger mot et bevisst dårlig kalibrert
utkast holder den utsendte fordelingen seg innenfor 0,002 total variasjon fra
målet — spekulasjon kjøper fart, aldri et annet svar.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefiks-hurtigbuffer (aktiv som standard)

Blokker adresseres via den *lenkede* hashen av tokenutsnittet sitt: to
forespørsler som deler en systeminstruks, deler blokkene dens, og den andre
trenger ikke lenger å forhåndsberegne dem. Lenkingen er uunnværlig: de samme
seksten tokenene i en annen kontekst inneholder ikke de samme nøklene og
verdiene, og å hashe bare utsnittet ville servere én sekvens' hurtigbuffer
til en annen.

En frigitt blokk hvis innhold fortsatt kan identifiseres, går inn i en LRU-kø
i stedet for frilisten: hurtigbufferen overlever dermed mellom forespørsler
uten noen gang å avslå en tildeling den kunne ha betjent.

### Beregning på vertsnivået (`--host-exec`)

Et lag med vekter i RAM kan kopieres til GPU-en eller beregnes på stedet. Begge
veiene er minnebegrensede og leser de samme bytene: den raskeste er den med
den bredeste bussen — PCIe 5.0 x16 gir rundt 54 GB/s, DDR5 i dual channel
rundt 70 GB/s — og å beregne på stedet lar dessuten GPU-en være ledig i stedet
for å la den vente på en kopi.

Det lønner seg bare hvis prosessoren leser de 4-bit-pakkede vektene direkte.
Derav en liten C++-kjerne med en AVX2-sti (`acvram_cpu.cpp`, lastet via
ctypes, uten Python-headere eller ninja). Selv på sin **skalare** reservegren
slår den `dequantize() @ x` med en faktor 1,44 i INT4 og 3,21 i NVFP4, fordi
sistnevnte først skriver en 32-bits kopi av hele matrisen.

På Mistral-Large-123B går planleggerens estimat fra 1,35 til 2,42 token/s.

### Blandet presisjon (`--snr-floor`, av som standard)

Konverteren måler signal-støy-forholdet ved utgangen av hvert lag for hver
tensor og kan forfremme dem som faller under `--snr-floor` til et bredere
format, innenfor en grense på 15 % av tensorene og et pristak
(`--promotion-cout-max`, i tilførte mebibyte).

Gulvet er **null som standard**: ingenting forfremmes. Dekoding er begrenset
av minnebåndbredden, og målingen på `Huihui-Qwen3.8-27B` avgjør — et gulv på
25 dB koster 13,4 % minne og 10,6 % gjennomstrømning (18,50 GiB og 41,8 t/s mot
16,02 og 46,2) for 2,0 % perpleksitet (42,591 mot 43,447, korpus på 16 383
token). `--snr-floor 25` gjenoppretter den gamle oppførselen når kvalitet går
foran fart.

### Og `acvram eval`

Signal-støy-forholdet og logit-cosinus er tilnærminger.
`acvram eval MAPPE [MAPPE ...]` måler perpleksiteten med glidende vindu, slik
at et formatvalg avgjøres på bevis:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-endepunkter

| endepunkt | merknader |
|---|---|
| `POST /v1/chat/completions` | SSE-strøm eller enkeltsvar; bruker modellens samtalemal |
| `POST /v1/completions` | prompt som tekst eller som token-id-er |
| `POST /v1/embeddings` | gjennomsnitt av siste skjulte tilstander, L2-normalisert, `dimensions` respektert |
| `GET /v1/models` | pluss en `acvram`-blokk: formater, enheter, KV-hurtigbufferens kapasitet |
| `GET /health`, `GET /metrics` | dekodingsgjennomstrømning, belegg av KV-blokker |

Feltnavnene i disse svarene forblir på engelsk: det er OpenAI-protokollen, og
å oversette dem ville ødelegge alle eksisterende klienter.

## Hvor tallene kommer fra

Hver verdi sitert ovenfor produseres av kode i dette depotet og kontrolleres
av `pytest`. Målinger gjort på prosessoren med referansekjernene:

| format | bit/vekt | vektenes SNR | logit-cosinus vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

To funn fra disse målingene har endret standardverdiene:

* **En Hadamard-rotasjon hjelper INT4 og ikke NVFP4.** INT4s grupper på 128
  kan ikke absorbere en isolert avvikende kanal, så å spre de ekstreme
  verdiene er verdt en n log n-transformasjon per aktivering. NVFP4s blokker
  på 16 bærer allerede sin egen skala. Derav `--hadamard auto`, som bare bruker
  den på INT4.
* **INT8 slår FP8 E4M3 for KV-hurtigbufferen**, 44 dB mot 32 dB ved samme
  størrelse, fordi en skala per (token, hode) allerede gir det dynamiske
  området FP8 bruker eksponentbit på. Begge kortene bruker derfor en
  KV-hurtigbuffer i INT8, selv om 5090 kunne kjørt FP8.

## Dokumentasjon

* [`REPRISE.md`](../REPRISE.md) — **gjenoppta prosjektet på en annen maskin**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — hvordan delene henger sammen
* [`docs/MATERIEL.md`](MATERIEL.md) — innstilling av akkurat denne maskinen
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **det som ikke er gjort**, les først
* [`CONVENTIONS.md`](../CONVENTIONS.md) — arbeidskonvensjoner for koden (språk, stil, kontroller før push)

## Målte resultater (22/09/2026, RTX 5090 ved 400 W, regime ≥ 20 s på energimåleren)

Qwen3-Coder-30B-A3B i NVFP4 (eksperter) + INT8 (attention, hode), samme
protokoll for alle motorer (`outils/`, ett kort, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekoding, 12 sekvenser | **1 625,5 t/s** | 1 596,1 t/s | — |
| dekoding, 1 sekvens | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, trukket) |

Gjennomstrømning for dagen (arbeidsstasjon 1030, øko-regime `-lgc 2700`,
pipeline i drift; grådig sampling fanget i CUDA-grafen, standard fra 0.6.35).
b=12 er en forseglet offisiell celle (median av 6 flettede vinduer, klokke per
vindu). vLLM-verdien 1 596,1 er den frosne referansen fra 21/09 (vLLM ikke kjørt
på nytt den dagen): avviket +1,84 % gjelder ved lik referanse, ikke som ny måling
av begge samme morgen. J/token ved lik klokke mot de tre motorene måles på nytt
(`outils/gpu/mesure/banc-4moteurs.py`) — et tall uten regime publiseres ikke.

Om morgenen 14/09 lå acvram på 630 t/s og 0,619 J/token i samme celle:
gevinstene kommer fra Blackwells innebygde FP4-MMA (`mma.sync …
kind::mxf4nvf4`, ×7,9 over bf16), fra MoE som gruppert GEMM per batch-bøtte,
fra ruting i én enkelt kjerne (3 677 → 1 517 starter per steg) og fra en smal
tensorkjerne-GEMM for projeksjonene. Hvert tall har sitt notat i
`acvram-memoire/revue/` med prediksjonen forseglet før målingen, instrumentet
og regimet dets — et tall uten regime publiseres ikke.

Der acvram ligger foran: MLA-modeller (GLM-4.7-Flash) i innebygd
sm_120-NVFP4, som vLLM bare serverer i FP8 (b=1: 165,35 t/s i drift); modeller
som ikke får plass i VRAM; og, fra 0.6.35, dekoding med stor batch av en MoE som
får plass i VRAM — b=12 går fra 1 540 (0.6.34) til 1 625,5 t/s, altså +1,84 %
foran den frosne vLLM-referansen (1 596,1). Avviket er fortsatt lite og ved
frosen referanse; energiavviket må måles på nytt.

## Status

Versjon 0.6.35. Alt kjører på 5090: CUDA-kjerner kompilert for `sm_120a`
(innebygd FP4) og `sm_86`, CUDA-grafer, NVFP4/INT8/INT4-kvantisering,
HTTP-server. Rekkverk på plass: kortet er usynlig for arbeidsøkter
(`CUDA_VISIBLE_DEVICES` tom), og bare `outils/carte.sh` låner det ut, under
lås, til én måling om gangen; en vakt logger all tilgang utenfor låsen; en
energimåling som dekker mer enn ett kort eller under 10 s, ugyldiggjøres; en
modell lastet i degradert modus sier det og deltar ikke i en duell.

640 tester (`pytest -q`, ett minutt på prosessoren; GPU-tester kjører bare
under `carte.sh`). Arbeidsoppfølging: `acvram-memoire/` (regler, register,
hefter, gjennomgang av 180 notater).

## Støtt

acvram utvikles på privat maskinvare. Hvis prosjektet er nyttig for deg:
**Støtt: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Lisens

GPL-3.0 eller senere.
