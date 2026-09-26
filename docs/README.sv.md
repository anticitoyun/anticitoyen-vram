<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-st%C3%B6d-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

En OpenAI-API-kompatibel inferensgateway som behandlar minnet som en hierarki, ger varje GPU det numeriska format dess kisel läser bäst, och optimerar varje token i joule lika mycket som i sekunder.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · **🇸🇪 Svenska** · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Jämförelse av genomströmning och energi mot vLLM och llama.cpp" width="720"></p>

---

## Innehåll

- [Två idéer](#idees)
- [Snabbstart](#demarrage)
- [Installation](#installer)
- [Vad `acvram plan` säger](#plan)
- [Gå snabbt](#optimisations)
- [HTTP-slutpunkter](#http)
- [Var siffrorna kommer från](#chiffres)
- [Dokumentation](#documentation)
- [Uppmätta resultat](#resultats)
- [Status](#etat)
- [Tack till](#credits)
- [Licens](#licence)
- [Stöd projektet](#soutien)

---

<a id="idees"></a>

## Två idéer

Byggd för en bestämd maskin:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-kärnor + 16 E-kärnor) |
| Moderkort | ASUS ROG Maximus Z790 Dark Hero |
| Minne | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13) ; båda korten på PCIe x8/x8, begränsade till 400 W / 275 W |

**Ett format per GPU.** RTX 5090 har FP4-tensorkärnor; RTX 3080 Ti har det inte, och inte FP8 heller. Att tvinga båda till ett gemensamt format skulle förstöra 5090:ans potential. Konverteraren skriver därför *samma modell två gånger*, i det format varje destination faktiskt kan utnyttja:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| vikter | **NVFP4** — E2M1 + FP8 E4M3-skala var 16:e | **INT4** — uint4 + fp16-skala och nollpunkt var 128:e |
| bitar per vikt | 4,50 | 4,16 |
| jämfört med BF16 | ×3,56 mindre | ×3,85 mindre |
| beräkningsläge | FP4-tensorkärnor | avkvantiserad till FP16 i kärnan, FP16-tensorkärnor |
| KV-cache | INT8 | INT8 |

32 GB VRAM vid 4,5 bitar per vikt rymmer omkring **56 miljarder parametrar**, mot 16 miljarder i BF16. På båda korten tillsammans ger det ungefär **78 miljarder residenta parametrar** innan man ens rör vid arbetsminnet.

**Minnet är en hierarki, inte en vägg.** Tre nivåer, och schemaläggaren mäter vad varje nivå kostar istället för att hoppas att modellen får plats:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Snabbstart

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Vilken OpenAI-klient som helst kan sedan anslutas:

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

---

<a id="installer"></a>

## Installation

Från källkoden (alla plattformar):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Eller via paket, en fil bifogad till varje [GitHub-release](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanal | Fil bifogad till release | Kommando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (namn genererade av `rpmbuild`, inte fasta) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (eller `rpmbuild --rebuild *.src.rpm` utifrån `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Verifiera den nedladdade filen mot summorna som bifogas utgåvan innan installation (`SHA256SUMS`, publiceras när alla andra filer finns på plats):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip publiceras inte som paket (inget byggt wheel): `pip install -e '.[dev]'` installerar från en klon av källkoden, precis som `./install.sh`.

---

<a id="plan"></a>

## Vad `acvram plan` säger

Schemaläggaren förtjänar att köras innan någon nedladdning. Den svarar på frågorna som avgör om en modell går att använda på denna maskin:

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

Den utforskar konfigurationsutrymmet istället för att nöja sig med den första som får plats, och två av dess beslut är motintuitiva nog för att förtjäna att nämnas:

* **Den lämnar 3080 Ti oanvänt** när en modell får plats på 5090 ensam. Ett pipelinens steg körs seriellt: att lägga till ett steg på 912 GB/s i en pipeline på 1790 GB/s bromsar enströms-avkodningen. Man tvingar fram det med `--gpus all`.
* **Den krymper KV-cachen för att hålla vikterna i VRAM.** Varje gigabyte som ges till cachen är ett gigabyte vikter som skjuts ut på PCIe-bussen, och att läsa en vikt via PCIe kostar omkring trettio gånger vad det kostar från VRAM. På 70B-modellen ovan gör just detta val skillnaden mellan 2,3 och 17,8 tokens/s.

---

<a id="optimisations"></a>

## Gå snabbt

Fyra optimeringar, var och en verifierad med ett likvärdighetsbevis och inte bara ett stoppur: en optimering som ändrar svaret är en bugg.

De täta NVFP4-linjärlagren i täta modeller går som standard via Marlin-layouten (+57 till +90 % genomströmning vid b = 8, TTFT +2 till +4 ms enligt revue/poste6-piece147-verdict-24-09.md; reservläge `ACVRAM_PROJ_MARLIN=0`, se [CHANGELOG.md](../CHANGELOG.md)).

### Spekulativ avkodning (`--speculative`)

Att avkoda en token med batchstorlek 1 är minnesbegränsat: maskinen läser alla aktiva vikter för att producera en enda token. Att verifiera K föreslagna tokens läser samma vikter **bara en gång**. Två föreslagare:

* `ngram` (standard) — söker det aktuella suffixet tidigare i kontexten och föreslår vad som följde då. Kostar ingenting, kräver ingen modell. Lönsamt när utdata återger indata: kodredigering, RAG, sammanfattning.
* `draft` — en liten modell på en andra enhet. På denna rigg är den enheten RTX 3080 Ti, som schemaläggaren medvetet lämnar sysslolös för varje modell som får plats på 5090.

`mtp` (modellens `nextn`-huvud) och `auto` finns också; inte lönsamma som det ser ut nu och inte aktiverade som standard — se `docs/ARCHITECTURE.md`.

Acceptansen är exakt, inte approximerad: ett förslag accepteras med sannolikheten `min(1, p/q)`, och ett avslag samplar om i den normaliserade positiva delen av `p - q`. Uppmätt på 40 000 dragningar mot ett medvetet dåligt kalibrerat utkast förblir den utsända fördelningen inom 0,002 total variation från målet — spekulationen köper hastighet, aldrig ett annat svar.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefixcache (aktiv som standard)

Block adresseras via *kedjad* hash av deras tokentråd: två förfrågningar som delar en systeminstruktion delar dess block, och den andra behöver inte längre förberäkna dem. Kedjningen är nödvändig: samma sexton tokens i en annan kontext innehåller inte samma nycklar och värden, och att hasha bara tråden skulle betjäna en sekvens' cache till en annan.

Ett frigjort block vars innehåll fortfarande går att identifiera går med i en LRU-kö snarare än listan över lediga block: cachen överlever så mellan förfrågningar utan att någonsin neka en allokering den kunde ha betjänat.

### Beräkning på värdnivån (`--host-exec`)

Ett lager vars vikter ligger i RAM kan kopieras till GPU:n eller beräknas på plats. Båda vägarna är minnesbegränsade och läser samma bytes: den snabbaste är den vars buss är bredast — PCIe 5.0 x16 ger omkring 54 GB/s, dubbelkanals-DDR5 omkring 70 GB/s — och att beräkna på plats lämnar dessutom GPU:n fri istället för att låta den vänta på en kopia.

Detta är bara värt det om processorn läser de 4-bitspackade vikterna direkt. Därav en liten C++-kärna med en AVX2-väg (`acvram_cpu.cpp`, laddad via ctypes, utan Python-headers eller ninja). Även på sin **skalära** reservgren slår den `dequantize() @ x` med en faktor 1,44 i INT4 och 3,21 i NVFP4, eftersom den senare först skriver en 32-bitars kopia av hela matrisen.

På Mistral-Large-123B går schemaläggarens uppskattning från 1,35 till 2,42 tokens/s.

### Blandad precision (`--snr-floor`, avstängd som standard)

Konverteraren mäter signal-brusförhållandet vid lagrets utgång för varje tensor och kan höja de som hamnar under `--snr-floor` till ett bredare format, inom gränsen 15 % av tensorerna och ett kostnadstak (`--promotion-cout-max`, i mebibyte tillagt).

Golvet är **noll som standard**: inget höjs. Avkodningen är minnesbandbreddsbegränsad, och mätningen på `Huihui-Qwen3.8-27B` avgör saken — ett golv på 25 dB kostar 13,4 % minne och 10,6 % genomströmning (18,50 GiB och 41,8 t/s mot 16,02 och 46,2) för 2,0 % perplexitet (42,591 mot 43,447, korpus på 16 383 tokens). `--snr-floor 25` återställer det gamla beteendet när kvalitet väger mer än hastighet.

### Och `acvram eval`

Signal-brusförhållandet och logit-cosinus är approximationer. `acvram eval REP [REP ...]` mäter perplexiteten i ett glidande fönster, så att ett formatval avgörs med bevis:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-slutpunkter

| slutpunkt | anteckningar |
|---|---|
| `POST /v1/chat/completions` | SSE-ström eller enstaka svar; använder modellens konversationsmall |
| `POST /v1/completions` | text- eller token-ID-instruktion |
| `POST /v1/embeddings` | genomsnittliga slutliga dolda tillstånd, L2-normaliserade, `dimensions` respekterad |
| `GET /v1/models` | plus ett `acvram`-block: format, enheter, KV-cachens kapacitet |
| `GET /health`, `GET /metrics` | avkodningsgenomströmning, KV-blockens beläggning |

Fältnamnen i dessa svar förblir på engelska: det är OpenAI-protokollet, och att översätta dem skulle bryta alla befintliga klienter.

---

<a id="chiffres"></a>

## Var siffrorna kommer från

Varje värde citerat ovan produceras av kod från detta repo och verifieras med `pytest`. Mätningar gjorda på processor med referenskärnorna:

| format | bitar/vikt | vikternas SNR | logit-cosinus mot BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Två slutsatser från dessa mätningar har ändrat standardvärdena:

* **En Hadamard-rotation hjälper INT4 men inte NVFP4.** INT4:ans grupper om 128 kan inte absorbera en isolerad avvikande kanal, så att sprida ut extremvärdena är värt en n log n-transform per aktivering. NVFP4:ans block om 16 bär redan sin egen skala. Därav `--hadamard auto`, som bara tillämpar den på INT4.
* **INT8 slår FP8 E4M3 för KV-cachen**, 44 dB mot 32 dB vid samma storlek, eftersom en skala per (token, huvud) redan ger den dynamiska omfång som FP8 lägger exponentbitar på. Båda korten använder därför en INT8 KV-cache, även om 5090:an skulle kunna göra FP8. Ett `k8v4`-format (värden i INT4, −22 % cache-bytes) finns som alternativ, **okvalificerat** — se `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentation

| Dokument | Innehåll |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **återuppta projektet på en annan maskin** (franska) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | hur delarna sitter ihop |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | ren NVFP4 eller attention+GDN i int8 per kanal, på en Gated DeltaNet-hybrid |
| [`docs/MATERIEL.md`](MATERIEL.md) | ställa in denna specifika maskin |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **vad som inte är klart**, läs detta först |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | arbetskonventioner för koden (språk, stil, kontroller innan push) |

---

<a id="resultats"></a>

## Uppmätta resultat (22/09/2026, RTX 5090 vid 400 W, mätfönster ≥ 20 s på energimätaren)

Qwen3-Coder-30B-A3B i NVFP4 (experter) + INT8 (attention, huvud), samma protokoll för alla motorer (`outils/`, ett kort, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| avkodning 12 sekvenser | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| avkodning 1 sekvens | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, borttagen) |

¹ Erratum 22/09: `serve` spekulerar som standard (`--speculative ngram`, cli.py), konkurrenterna inte; de 380,8 t/s som publicerats hittills mättes MED spekulation. Utan spekulation (`--speculative none`, samma kedja, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram är **trea** vid b=1, efter llama.cpp och vLLM. I energi ligger den fortfarande före llama.cpp (0,601 mot 0,700 J/token netto). Vid b=12 är spekulationen aldrig aktiv (skydd `lot_max=2`): den cellen var redan på lika villkor.

² 23/09, samma session, samma HTTP-klient (`banc-llamacpp-16-09.py` mot `acvram serve` och `vllm serve`), `-lgc 2700` satt explicit runt varje arm, celler växlade A V V A, ≥ 5 batcher per arm, avvikelse deklarerad endast bortom 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 vid avkodning, utrullad attention-reduktion): avvikelse −1,6 %, **under 2 σ: lika genomströmning**. I J/token ligger **vLLM fortfarande före med 7,0 %** (bortom 2 σ). Med 0.6.37 gav samma protokoll −4,7 %.

³ Samma session och protokoll som ², utan spekulation på båda sidor: acvram 312,3 mot vLLM 284,8 — **acvram före med 9,7 % i genomströmning** (bortom 2 σ); J/token: **lika** (avvikelse 0,04 %, under 2 σ).

⁴ 23/09, samma protokoll mot llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 med omskrivet routing (+5,6 %): acvram 310,8 mot llama.cpp 329,9 t/s — **llama.cpp före med 5,8 % i genomströmning, acvram före med 13,4 % i J/token** (0,598 mot 0,691).

Dagens genomströmning (station 1030, ekoläge `-lgc 2700`, pipeline i drift; glupsk sampling infångad i CUDA-grafen, standard sedan 0.6.35). b=12 för acvram är en officiellt förseglad cell (median av 6 samtida fönster, klocka per fönster).

> **Erratum (23/09/2026).** Den vLLM-jämförelse som publicerats hittills (b=12: 1 782 mot 1 634 t/s; b=1: 290,6) ställde acvram mätt via HTTP mot vLLM mätt **offline** (`LLM().generate()`), och erratumet från 22/09 hävdade felaktigt att vLLM-cellen gick via `vllm serve`. Den 23/09: samma HTTP-klient för båda, och `-lgc` satt för båda (acvram sätter sitt eget vid start, `vllm serve` gör det inte: utan denna försiktighetsåtgärd körde vLLM på ~2 930 MHz mot ~2 650). Resultat i not ²: vLLM före med 9,1 % vid b=12.

Den 14/09 på morgonen låg acvram på 630 t/s och 0,619 J/token på samma cell: vinsterna kommer från Blackwells nativa FP4-MMA (`mma.sync … kind::mxf4nvf4`, ×7,9 mot bf16), MoE i grupperad GEMM per batchhink, routing i en enda kärna (3 677 → 1 517 anrop per steg) och en smal GEMM på tensorkärnor för projektionerna. Varje siffra har sin anteckning i `acvram-memoire/revue/` med förseglad förutsägelse innan mätningen, instrumentet och dess läge — en siffra utan ett läge publiceras inte.

Där acvram ligger före: MLA-modeller (GLM-4.7-Flash) i nativ NVFP4 sm_120, som vLLM bara betjänar i FP8 (b=1: 165,35 t/s i drift); modeller som inte får plats i VRAM. Enstaka sekvens-avkodning är inte en av dem: utan spekulation ligger acvram där före vLLM med 9,7 % (not ³), efter llama.cpp med 5,8 % i genomströmning men före den med 13,4 % i energi (not ⁴). Vid stor batch, på en MoE som får plats i VRAM, ligger vLLM lika i genomströmning vid b=12 (1 995,1 mot 2 027,0 t/s, under 2 σ, not ²) men behåller 7,0 % lägre J/token; acvram har där gått från 1 540 t/s (0.6.34) till 1 995 (0.6.38).

---

<a id="etat"></a>

## Status

Version 0.6.38. Allt körs på 5090:an: CUDA-kärnor kompilerade för `sm_120a` (nativt FP4) och `sm_86`, CUDA-grafer, NVFP4/INT8/INT4-kvantisering, HTTP-server. Skyddsmekanismer på plats: kortet är osynligt för arbetssessioner (`CUDA_VISIBLE_DEVICES` tom) och endast `outils/carte.sh` lånar ut det, under lås, till en mätning i taget; en väktare loggar all åtkomst utanför låset; en energimätning som täcker mer än ett kort eller mindre än 10 s ogiltigförklaras; en modell laddad i degraderat läge säger det och deltar inte i en duell.

4 107 tester (`pytest --collect-only -q`, en minut på processor; GPU-testerna körs bara under `carte.sh`). Uppföljning av arbetet: `acvram-memoire/` (regler, katalog, anteckningsböcker, genomgång av flera hundra anteckningar).

---

<a id="credits"></a>

## Tack till

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, under Apache-2.0-licens: `acvram/kernels/marlin_port/` bär dess Marlin-kärnor (MoE och täta), med fullständig attribution fil för fil i [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, Blackwells FP4-tensorkärnor (`sm_120`) och biblioteken detta projekt bygger på.
- **PyTorch** — tensormotor och C++/CUDA-tillägg.

Oberoende projekt, inte anslutet till ASUS, NVIDIA eller vLLM-projektet.

---

<a id="licence"></a>

## Licens

[GPL-3.0 eller senare](../LICENSE) för koden i detta repo. `acvram/kernels/marlin_port/` innehåller kod portad från [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kärnorna `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), under Apache-2.0-licens: varje fil behåller sin ursprungliga header, licensen finns i `LICENSE-vllm` och filistan, ursprungscommit och ändringarna finns i [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Stöd projektet

Utvecklingen av acvram sker på privat hårdvara. Om projektet är till nytta för dig:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Bjud%20p%C3%A5%20kaffe&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Översättningar: [TRADUIRE.md](TRADUIRE.md) (franska; projektets bidragsguide är ännu inte översatt).
