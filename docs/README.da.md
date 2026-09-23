<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Støt: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

En inferens-gateway, der er kompatibel med OpenAI-API'et, behandler
hukommelsen som et hierarki og giver hver GPU det talformat, dens silicium
læser bedst.

Designet til én bestemt maskine:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-kerner + 16 E-kerner) |
| Bundkort | ASUS ROG Maximus Z790 Dark Hero |
| Hukommelse | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); begge kort på PCIe x8/x8, begrænset til 400 W / 275 W |

## De to idéer

**Ét format pr. GPU.** RTX 5090 har FP4-tensorkerner; RTX 3080 Ti har ingen,
og heller ikke FP8. At tvinge begge på et fælles format ville spilde 5090'en.
Konverteren skriver derfor *den samme model to gange*, i det format hver
destination reelt kan udnytte:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| vægte | **NVFP4** — E2M1 + FP8 E4M3-skala for hver 16 | **INT4** — uint4 + fp16-skala og -nulpunkt for hver 128 |
| bit pr. vægt | 4,50 | 4,16 |
| i forhold til BF16 | ×3,56 mindre | ×3,85 mindre |
| beregningstilstand | FP4-tensorkerner | dekvantiseret til FP16 i kernen, FP16-tensorkerner |
| KV-cache | INT8 | INT8 |

32 GB VRAM ved 4,5 bit pr. vægt rummer omkring **56 milliarder parametre**,
mod 16 milliarder i BF16. På begge kort giver det cirka **78 milliarder
residente parametre**, før arbejdshukommelsen overhovedet røres.

**Hukommelsen er et hierarki, ikke en mur.** Tre lag, og planlæggeren måler,
hvad hvert lag koster, i stedet for at håbe, at modellen passer:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Hurtig start

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Enhver OpenAI-klient kan derefter kobles på:

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

## Hvad `acvram plan` siger

Planlæggeren fortjener at blive kørt før enhver download. Den besvarer de
spørgsmål, der afgør, om en model kan bruges på denne maskine:

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

Den udforsker konfigurationsrummet i stedet for at tage den første, der
passer, og to af dens beslutninger er kontraintuitive nok til at fortjene at
blive sagt højt:

* **Den lader 3080 Ti stå ubrugt**, når en model passer på 5090 alene.
  Skiverne i en pipeline kører i serie: at tilføje et trin på 912 GB/s til en
  pipeline på 1790 GB/s gør enkeltstrøms-dekodning langsommere. Det kan
  tvinges med `--gpus all`.
* **Den krymper KV-cachen for at holde vægtene i VRAM.** Hver gigabyte givet
  til cachen er en gigabyte vægte skubbet ud på PCIe-bussen, og at læse en vægt
  over PCIe koster omkring tredive gange så meget som fra VRAM. På 70B'eren
  ovenfor flytter denne ene afvejning alene tallet fra 2,3 til 17,8 tokens/s.

## At være hurtig

Fire optimeringer, hver bekræftet af et ækvivalensbevis og ikke kun af et
stopur: en optimering, der ændrer svaret, er en fejl.

### Spekulativ dekodning (`--speculative`)

At dekode ét token med en batch af størrelse 1 er hukommelsesbegrænset:
maskinen læser alle aktive vægte for at producere ét enkelt token. At
verificere K foreslåede tokens læser de samme vægte **én gang**. To forslagsstillere:

* `ngram` (standard) — leder efter det aktuelle suffiks tidligere i konteksten
  og foreslår det, der fulgte. Koster intet, kræver ingen model. Betaler sig,
  når outputtet kopierer inputtet: koderedigering, RAG, resumé.
* `draft` — en lille model på en anden enhed. På dette rig er den enhed
  RTX 3080 Ti, som planlæggeren bevidst lader stå ledig for enhver model, der
  passer på 5090.

Accepten er eksakt, ikke tilnærmet: et forslag accepteres med sandsynligheden
`min(1, p/q)`, og et afslag gensampler fra den normaliserede positive del af
`p - q`. Målt over 40 000 trækninger mod et bevidst dårligt kalibreret udkast
holder den udsendte fordeling sig inden for 0,002 total variation fra målet —
spekulation køber hastighed, aldrig et andet svar.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Præfiks-cache (aktiv som standard)

Blokke adresseres via den *kædede* hash af deres token-udsnit: to
forespørgsler, der deler en systeminstruktion, deler dens blokke, og den
anden behøver ikke længere at forudberegne dem. Kædningen er uundværlig: de
samme seksten tokens i en anden kontekst indeholder ikke de samme nøgler og
værdier, og at hashe udsnittet alene ville servere én sekvens' cache til en
anden.

En frigivet blok, hvis indhold stadig kan identificeres, ryger i en LRU-kø i
stedet for frilisten: cachen overlever dermed mellem forespørgsler uden
nogensinde at afvise en tildeling, den kunne have betjent.

### Beregning på værtslaget (`--host-exec`)

Et lag, hvis vægte ligger i RAM, kan kopieres til GPU'en eller beregnes på
stedet. Begge veje er hukommelsesbegrænsede og læser de samme bytes: den
hurtigste er den med den bredeste bus — PCIe 5.0 x16 giver omkring 54 GB/s,
DDR5 i dual channel omkring 70 GB/s — og at beregne på stedet lader desuden
GPU'en være fri i stedet for at lade den vente på en kopi.

Det betaler sig kun, hvis processoren læser de 4-bit-pakkede vægte direkte.
Deraf en lille C++-kerne med en AVX2-sti (`acvram_cpu.cpp`, indlæst via
ctypes, uden Python-headere eller ninja). Selv på sin **skalære**
fallback-gren slår den `dequantize() @ x` med en faktor 1,44 i INT4 og 3,21 i
NVFP4, fordi sidstnævnte først skriver en 32-bit-kopi af hele matricen.

På Mistral-Large-123B går planlæggerens estimat fra 1,35 til 2,42 tokens/s.

### Blandet præcision (`--snr-floor`, slået fra som standard)

Konverteren måler signal-støj-forholdet ved hvert lags udgang for hver tensor
og kan forfremme dem, der falder under `--snr-floor`, til et bredere format,
inden for en grænse på 15 % af tensorerne og et prisloft
(`--promotion-cout-max`, i tilføjede mebibyte).

Gulvet er **nul som standard**: intet forfremmes. Dekodning er begrænset af
hukommelsesbåndbredden, og målingen på `Huihui-Qwen3.8-27B` afgør det — et
gulv på 25 dB koster 13,4 % hukommelse og 10,6 % gennemløb (18,50 GiB og
41,8 t/s mod 16,02 og 46,2) for 2,0 % perpleksitet (42,591 mod 43,447, korpus
på 16 383 tokens). `--snr-floor 25` genopretter den gamle adfærd, når kvalitet
går forud for hastighed.

### Og `acvram eval`

Signal-støj-forholdet og logit-cosinus er tilnærmelser.
`acvram eval MAPPE [MAPPE ...]` måler perpleksiteten med glidende vindue, så
et formatvalg afgøres på beviser:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-endepunkter

| endepunkt | noter |
|---|---|
| `POST /v1/chat/completions` | SSE-strøm eller enkelt svar; bruger modellens chat-skabelon |
| `POST /v1/completions` | prompt som tekst eller som token-id'er |
| `POST /v1/embeddings` | gennemsnit af sidste skjulte tilstande, L2-normaliseret, `dimensions` respekteret |
| `GET /v1/models` | plus en `acvram`-blok: formater, enheder, KV-cachens kapacitet |
| `GET /health`, `GET /metrics` | dekodningsgennemløb, belægning af KV-blokke |

Feltnavnene i disse svar forbliver på engelsk: det er OpenAI-protokollen, og
at oversætte dem ville ødelægge alle eksisterende klienter.

## Hvor tallene kommer fra

Hver værdi citeret ovenfor er produceret af kode i dette repository og
kontrolleret af `pytest`. Målinger foretaget på processoren med
referencekernerne:

| format | bit/vægt | vægtenes SNR | logit-cosinus vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

To iagttagelser fra disse målinger har ændret standardværdierne:

* **En Hadamard-rotation hjælper INT4 og ikke NVFP4.** INT4's grupper på 128
  kan ikke opsuge en isoleret afvigende kanal, så at sprede de ekstreme værdier
  er en n log n-transformation pr. aktivering værd. NVFP4's blokke på 16 bærer
  allerede deres egen skala. Deraf `--hadamard auto`, som kun anvender den på
  INT4.
* **INT8 slår FP8 E4M3 til KV-cachen**, 44 dB mod 32 dB ved samme størrelse,
  fordi en skala pr. (token, hoved) allerede leverer det dynamiske område, som
  FP8 bruger eksponentbit på. Begge kort bruger derfor en KV-cache i INT8,
  selvom 5090 kunne køre FP8.

## Dokumentation

* [`REPRISE.md`](../REPRISE.md) — **genoptag projektet på en anden maskine**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — hvordan delene passer sammen
* [`docs/MATERIEL.md`](MATERIEL.md) — indstilling af netop denne maskine
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **det, der ikke er gjort**, læs først
* [`CONVENTIONS.md`](../CONVENTIONS.md) — arbejdskonventioner for koden (sprog, stil, kontroller før push)

## Målte resultater (22/09/2026, RTX 5090 ved 400 W, regime ≥ 20 s på energimåleren)

Qwen3-Coder-30B-A3B i NVFP4 (eksperter) + INT8 (attention, hoved), samme
protokol for alle motorer (`outils/`, ét kort, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekodning, 12 sekvenser | **1 634 t/s** | 1 782 t/s | — |
| dekodning, 1 sekvens | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, trukket tilbage) |

Dagens gennemløb (station 1030, øko-regime `-lgc 2700`, pipeline i drift;
grådig sampling fanget i CUDA-grafen, slået til som standard i 0.6.35). b=12 er
den officielle forseglede celle (median af 6 sammenflettede vinduer).

> **Rettelse (22/09/2026).** Den første udgivelse af 0.6.35 udledte
> «+1,84 % foran vLLM» af en vLLM-reference på 1 596 t/s fra 21/09, der
> stammede fra en **offline-generering** (`LLM().generate()`), **ikke
> sammenlignelig med en server**: ingen løbende planlægning, ikke stien i
> `acvram serve`. Rettet den 22/09 med en vekslende A/V-celle (A1 V1 A2 V2 A3
> V3) mod **`vllm serve`** (HTTP), samme kort og samme sti som `acvram serve`:
> vLLM median **1 782 t/s**. Ved sammenlignelig måling er **acvram (1 634 t/s)
> BAG vLLM med omkring 8 % ved b=12**, ikke foran. J/token ved samme klokke
> måles stadig igen.

Om morgenen den 14/09 lå acvram på 630 t/s og 0,619 J/token i samme celle:
gevinsterne kommer fra Blackwells native FP4-MMA (`mma.sync …
kind::mxf4nvf4`, ×7,9 over bf16), fra MoE som grupperet GEMM pr. batch-spand,
fra routing i én enkelt kerne (3 677 → 1 517 starter pr. trin) og fra en smal
tensorkerne-GEMM til projektionerne. Hvert tal har sin note i
`acvram-memoire/revue/` med forudsigelsen forseglet før målingen, instrumentet
og dets regime — et tal uden regime offentliggøres ikke.

Hvor acvram er foran: MLA-modeller (GLM-4.7-Flash) i native sm_120-NVFP4, som
vLLM kun serverer i FP8 (b=1: 165,35 t/s i drift); modeller, der ikke passer
i VRAM; og enkeltsekvens-dekodning (b=1: 380,8 t/s mod 290,6 for vLLM). Ved stor
batch derimod, på en MoE, der passer i VRAM, forbliver vLLM foran ved b=12
(1 782 mod 1 634 t/s, jf. rettelsen); acvram er her gået frem (1 540 i 0.6.34 →
1 634) uden at komme foran. Energiforskellen skal måles igen.

## Status

Version 0.6.35. Alt kører på 5090: CUDA-kerner kompileret til `sm_120a`
(native FP4) og `sm_86`, CUDA-grafer, NVFP4/INT8/INT4-kvantisering,
HTTP-server. Værn på plads: kortet er usynligt for arbejdssessioner
(`CUDA_VISIBLE_DEVICES` tom), og kun `outils/carte.sh` udlåner det, under lås,
til én måling ad gangen; en vagt logger enhver adgang uden for låsen; en
energimåling, der dækker mere end ét kort eller under 10 s, ugyldiggøres; en
model indlæst i degraderet tilstand siger det og deltager ikke i en duel.

640 tests (`pytest -q`, ét minut på processoren; GPU-tests kører kun under
`carte.sh`). Arbejdsopfølgning: `acvram-memoire/` (regler, register, hæfter,
gennemgang af 180 noter).

## Støt

acvram udvikles på privat hardware. Hvis projektet er nyttigt for dig:
**Støt: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licens

GPL-3.0 eller senere.
