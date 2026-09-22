<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Stöd: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

En inferens-gateway kompatibel med OpenAI-API:et, som behandlar minnet som en
hierarki och ger varje GPU det talformat dess kisel läser bäst.

Utformad för en bestämd maskin:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-kärnor + 16 E-kärnor) |
| Moderkort | ASUS ROG Maximus Z790 Dark Hero |
| Minne | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); båda korten på PCIe x8/x8, begränsade till 400 W / 275 W |

## De två idéerna

**Ett format per GPU.** RTX 5090 har FP4-tensorkärnor; RTX 3080 Ti har inga,
och inte heller FP8. Att lägga båda på ett gemensamt format skulle slösa bort
5090:n. Konverteraren skriver därför *samma modell två gånger*, i det format
varje destination verkligen kan utnyttja:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| vikter | **NVFP4** — E2M1 + FP8 E4M3-skala var 16:e | **INT4** — uint4 + fp16-skala och -nollpunkt var 128:e |
| bitar per vikt | 4,50 | 4,16 |
| jämfört med BF16 | ×3,56 mindre | ×3,85 mindre |
| beräkningsläge | FP4-tensorkärnor | dekvantiserad till FP16 i kärnan, FP16-tensorkärnor |
| KV-cache | INT8 | INT8 |

32 GB VRAM vid 4,5 bitar per vikt rymmer ungefär **56 miljarder parametrar**,
mot 16 miljarder i BF16. Över båda korten blir det cirka **78 miljarder
residenta parametrar** innan arbetsminnet ens rörs.

**Minnet är en hierarki, inte en mur.** Tre nivåer, och planeraren mäter vad
var och en kostar i stället för att hoppas att modellen får plats:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Snabbstart

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Vilken OpenAI-klient som helst kan sedan kopplas in:

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

## Vad `acvram plan` säger

Planeraren är värd att köra före varje nedladdning. Den besvarar frågorna som
avgör om en modell är användbar på den här maskinen:

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

Den utforskar konfigurationsrymden i stället för att behålla den första som
får plats, och två av dess beslut är tillräckligt kontraintuitiva för att
förtjäna att uttalas:

* **Den lämnar 3080 Ti oanvänd** när en modell får plats på 5090 ensam.
  Skivorna i en pipeline körs i serie: att lägga till ett steg på 912 GB/s i
  en pipeline på 1790 GB/s bromsar avkodningen av en enskild ström. Det kan
  tvingas fram med `--gpus all`.
* **Den krymper KV-cachen för att behålla vikterna i VRAM.** Varje gigabyte
  som ges till cachen är en gigabyte vikter som trängs ut på PCIe-bussen, och
  att läsa en vikt över PCIe kostar ungefär trettio gånger vad det kostar från
  VRAM. På 70B-modellen ovan flyttar enbart denna avvägning siffran från 2,3
  till 17,8 token/s.

## Att gå fort

Fyra optimeringar, var och en verifierad med ett ekvivalensbevis och inte bara
med ett tidtagarur: en optimering som ändrar svaret är en bugg.

### Spekulativ avkodning (`--speculative`)

Att avkoda ett token med en batch av storlek 1 är minnesbegränsat: maskinen
läser alla aktiva vikter för att producera ett enda token. Att verifiera K
föreslagna token läser samma vikter **en enda gång**. Två förslagsställare:

* `ngram` (standard) — letar efter det aktuella suffixet tidigare i kontexten
  och föreslår det som följde. Kostar inget, kräver ingen modell. Lönsamt när
  utdata kopierar indata: kodredigering, RAG, sammanfattning.
* `draft` — en liten modell på en andra enhet. På denna rigg är den enheten
  RTX 3080 Ti, som planeraren avsiktligt lämnar sysslolös för varje modell som
  får plats på 5090.

Accepterandet är exakt, inte approximativt: ett förslag accepteras med
sannolikheten `min(1, p/q)` och ett avslag omsamplar ur den normaliserade
positiva delen av `p - q`. Mätt över 40 000 dragningar mot ett avsiktligt
felkalibrerat utkast håller sig den utsända fördelningen inom 0,002 total
variation från målet — spekulation köper hastighet, aldrig ett annat svar.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefixcache (aktiv som standard)

Block adresseras via den *kedjade* hashen av sitt tokenavsnitt: två
förfrågningar som delar en systeminstruktion delar dess block, och den andra
behöver inte längre förberäkna dem. Kedjningen är oumbärlig: samma sexton
token i en annan kontext innehåller inte samma nycklar och värden, och att
hasha enbart avsnittet skulle servera en sekvens cache till en annan.

Ett frigjort block vars innehåll förblir identifierbart hamnar i en LRU-kö i
stället för i frilistan: cachen överlever därmed mellan förfrågningar utan att
någonsin neka en tilldelning den kunde ha betjänat.

### Beräkning på värdnivån (`--host-exec`)

Ett lager vars vikter ligger i RAM kan kopieras till GPU:n eller beräknas på
plats. Båda vägarna är minnesbegränsade och läser samma byte: den snabbaste
är den med den bredaste bussen — PCIe 5.0 x16 ger ungefär 54 GB/s, DDR5 i dual
channel ungefär 70 GB/s — och att beräkna på plats lämnar dessutom GPU:n fri i
stället för att låta den vänta på en kopia.

Det lönar sig bara om processorn läser de 4-bitspackade vikterna direkt.
Därav en liten C++-kärna med en AVX2-väg (`acvram_cpu.cpp`, laddad via ctypes,
utan Python-headers eller ninja). Även på sin **skalära** reservgren slår den
`dequantize() @ x` med en faktor 1,44 i INT4 och 3,21 i NVFP4, eftersom den
senare först skriver en 32-bitarskopia av hela matrisen.

På Mistral-Large-123B går planerarens uppskattning från 1,35 till
2,42 token/s.

### Blandad precision (`--snr-floor`, avstängd som standard)

Konverteraren mäter signal-brus-förhållandet vid varje lagers utgång för
varje tensor och kan befordra dem som faller under `--snr-floor` till ett
bredare format, inom en gräns på 15 % av tensorerna och ett pristak
(`--promotion-cout-max`, i tillagda mebibyte).

Golvet är **noll som standard**: inget befordras. Avkodningen begränsas av
minnesbandbredden, och mätningen på `Huihui-Qwen3.8-27B` avgör — ett golv på
25 dB kostar 13,4 % minne och 10,6 % genomströmning (18,50 GiB och 41,8 t/s
mot 16,02 och 46,2) för 2,0 % perplexitet (42,591 mot 43,447, korpus på
16 383 token). `--snr-floor 25` återställer det gamla beteendet när kvalitet
går före hastighet.

### Och `acvram eval`

Signal-brus-förhållandet och logit-cosinus är approximationer.
`acvram eval KAT [KAT ...]` mäter perplexiteten med glidande fönster, så att
ett formatval avgörs på bevis:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-slutpunkter

| slutpunkt | anteckningar |
|---|---|
| `POST /v1/chat/completions` | SSE-ström eller enskilt svar; använder modellens chattmall |
| `POST /v1/completions` | prompt som text eller som token-id:n |
| `POST /v1/embeddings` | medelvärde av sista dolda tillstånd, L2-normaliserat, `dimensions` respekteras |
| `GET /v1/models` | plus ett `acvram`-block: format, enheter, KV-cachens kapacitet |
| `GET /health`, `GET /metrics` | avkodningsgenomströmning, beläggning av KV-block |

Fältnamnen i dessa svar förblir på engelska: det är OpenAI-protokollet, och
att översätta dem skulle förstöra alla befintliga klienter.

## Varifrån siffrorna kommer

Varje värde som citeras ovan produceras av kod i detta repository och
kontrolleras av `pytest`. Mätningar gjorda på processorn med referenskärnorna:

| format | bitar/vikt | vikternas SNR | logit-cosinus vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Två iakttagelser från dessa mätningar har ändrat standardvärdena:

* **En Hadamard-rotation hjälper INT4 men inte NVFP4.** INT4:s grupper om 128
  kan inte absorbera en isolerad avvikande kanal, så att sprida ut de extrema
  värdena är värt en n log n-transform per aktivering. NVFP4:s block om 16 bär
  redan sin egen skala. Därav `--hadamard auto`, som bara tillämpar den på
  INT4.
* **INT8 slår FP8 E4M3 för KV-cachen**, 44 dB mot 32 dB vid samma storlek,
  eftersom en skala per (token, huvud) redan ger det dynamiska omfång som FP8
  lägger exponentbitar på. Båda korten använder därför en KV-cache i INT8,
  även om 5090 skulle kunna köra FP8.

## Dokumentation

* [`REPRISE.md`](../REPRISE.md) — **återuppta projektet på en annan maskin**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — hur delarna hänger ihop
* [`docs/MATERIEL.md`](MATERIEL.md) — ställa in just denna maskin
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **det som inte är gjort**, läs först
* [`CONVENTIONS.md`](../CONVENTIONS.md) — arbetskonventioner för koden (språk, stil, kontroller före push)

## Uppmätta resultat (22/09/2026, RTX 5090 vid 400 W, regim ≥ 20 s på energimätaren)

Qwen3-Coder-30B-A3B i NVFP4 (experter) + INT8 (attention, huvud), samma
protokoll för alla motorer (`outils/`, ett kort, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| avkodning, 12 sekvenser | **1 634 t/s** | 1 782 t/s | — |
| avkodning, 1 sekvens | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707** token/s | 21 054 | 8 671 (TabbyAPI, tillbakadraget) |

Genomströmning för dagen (rigg 1030, eko-regim `-lgc 2700`, pipeline i drift; girig sampling infångad i CUDA-grafen, standard från 0.6.35). b=12 är en officiell förseglad cell (median av 6 inflätade fönster, klockfrekvens per fönster).

> **Erratum (22/09/2026).** Den första publiceringen av 0.6.35 drog «+1,84 % före vLLM» ur en vLLM-referens på 1 596 t/s från 21/09 som kom från en **offline-generering** (`LLM().generate()`), **inte jämförbar med en server**: ingen kontinuerlig schemaläggning, inte vägen för `acvram serve`. Rättat den 22/09 med en alternerande cell A/V (A1 V1 A2 V2 A3 V3) mot **`vllm serve`** (HTTP), samma kort och samma väg som `acvram serve`: vLLM-median **1 782 t/s**. Vid jämförbar mätning ligger **acvram (1 634 t/s) EFTER vLLM med omkring 8 % vid b=12**, inte före. J/token vid lika klockfrekvens återstår att mäta om.

På morgonen den 14/09 låg acvram på 630 t/s och 0,619 J/token i samma cell:
vinsterna kommer från Blackwells inbyggda FP4-MMA (`mma.sync …
kind::mxf4nvf4`, ×7,9 över bf16), från MoE som grupperad GEMM per batchhink,
från routning i en enda kärna (3 677 → 1 517 starter per steg) och från en
smal tensorkärne-GEMM för projektionerna. Varje siffra har sin anteckning i
`acvram-memoire/revue/` med förutsägelsen förseglad före mätningen,
instrumentet och dess regim — en siffra utan regim publiceras inte.

Där acvram ligger före: MLA-modeller (GLM-4.7-Flash) i inbyggd
sm_120-NVFP4, som vLLM bara serverar i FP8 (b=1: 165,35 t/s i drift); modeller
som inte får plats i VRAM; och avkodning av en enda sekvens (b=1: 380,8 t/s mot
290,6 för vLLM). Vid stor batch däremot, på en MoE som får plats i VRAM, ligger
vLLM fortfarande före vid b=12 (1 782 mot 1 634 t/s, jfr erratum); acvram har där
gått framåt (1 540 i 0.6.34 → 1 634) utan att gå om. Skillnaden i energi
återstår att mäta om.

## Läge

Version 0.6.35. Allt körs på 5090: CUDA-kärnor kompilerade för `sm_120a`
(inbyggd FP4) och `sm_86`, CUDA-grafer, NVFP4/INT8/INT4-kvantisering,
HTTP-server. Skyddsräcken på plats: kortet är osynligt för arbetssessioner
(`CUDA_VISIBLE_DEVICES` tom) och endast `outils/carte.sh` lånar ut det, under
lås, till en mätning i taget; en väktare loggar varje åtkomst utanför låset;
en energimätning som täcker mer än ett kort eller under 10 s ogiltigförklaras;
en modell laddad i degraderat läge säger det och deltar inte i en duell.

640 tester (`pytest -q`, en minut på processorn; GPU-tester körs bara under
`carte.sh`). Arbetsuppföljning: `acvram-memoire/` (regler, register, häften,
granskning av 180 anteckningar).

## Stöd

acvram utvecklas på privat hårdvara. Om projektet är till nytta för dig:
**Stöd: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licens

GPL-3.0 eller senare.
