<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-støtt-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

En OpenAI-API-kompatibel inferensportal som behandler minnet som et hierarki, gir hver GPU det tallformatet silisiumet dens leser best, og optimaliserer hvert token i joule like mye som i sekunder.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · **🇳🇴 Norsk bokmål** · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Sammenligning av gjennomstrømning og energi mot vLLM og llama.cpp" width="720"></p>

---

## Innhold

- [De to ideene](#idees)
- [Hurtigstart](#demarrage)
- [Installasjon](#installer)
- [Hva `acvram plan` sier](#plan)
- [Å være rask](#optimisations)
- [HTTP-endepunkter](#http)
- [Hvor tallene kommer fra](#chiffres)
- [Dokumentasjon](#documentation)
- [Målte resultater](#resultats)
- [Status](#etat)
- [Kreditering](#credits)
- [Lisens](#licence)
- [Støtt prosjektet](#soutien)

---

<a id="idees"></a>

## De to ideene

Laget for én bestemt maskin:

| | |
|---|---|
| Prosessor | Intel Core i9-14900K (8 P-kjerner + 16 E-kjerner) |
| Hovedkort | ASUS ROG Maximus Z790 Dark Hero |
| Minne | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); begge kortene på PCIe x8/x8, begrenset til 400 W / 275 W |

**Ett format per GPU.** RTX 5090 har FP4-tensorkjerner; RTX 3080 Ti har ingen, og heller ikke FP8. Å legge begge på et felles format ville sløse bort 5090. Konverteren skriver derfor *den samme modellen to ganger*, i det formatet hver destinasjon faktisk kan utnytte:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| vekter | **NVFP4** — E2M1 + FP8 E4M3-skala per 16 | **INT4** — uint4 + fp16-skala og -nullpunkt per 128 |
| bit per vekt | 4,50 | 4,16 |
| mot BF16 | ×3,56 mindre | ×3,85 mindre |
| beregningsmodus | FP4-tensorkjerner | dekvantisert til FP16 i kjernen, FP16-tensorkjerner |
| KV-hurtigbuffer | INT8 | INT8 |

32 GB VRAM ved 4,5 bit per vekt rommer omtrent **56 milliarder parametere**, mot 16 milliarder i BF16. På begge kortene blir det omtrent **78 milliarder residente parametere** før arbeidsminnet i det hele tatt røres.

**Minnet er et hierarki, ikke en mur.** Tre nivåer, og planleggeren måler hva hvert av dem koster i stedet for å håpe at modellen får plass:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

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

---

<a id="installer"></a>

## Installasjon

Fra kildekoden (alle plattformer):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Eller som pakke, en fil vedlagt hver [GitHub-utgivelse](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanal | Fil vedlagt utgivelsen | Kommando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (navn generert av `rpmbuild`, ikke faste) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (eller `rpmbuild --rebuild *.src.rpm` fra `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Før installasjon, verifiser den nedlastede filen mot summene vedlagt utgivelsen (`SHA256SUMS`, publisert når alle andre filer er til stede):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip publiseres ikke som pakke (ingen bygget wheel): `pip install -e '.[dev]'` installerer fra en klone av kilden, akkurat som `./install.sh`.

---

<a id="plan"></a>

## Hva `acvram plan` sier

Planleggeren fortjener å kjøres før enhver nedlasting. Den svarer på spørsmålene som avgjør om en modell er brukbar på denne maskinen:

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

Den utforsker konfigurasjonsrommet i stedet for å beholde den første som passer, og to av beslutningene dens er kontraintuitive nok til at de fortjener å sies rett ut:

* **Den lar 3080 Ti stå ubrukt** når en modell får plass på 5090 alene. Skivene i en pipeline kjøres i serie: å legge til et trinn på 912 GB/s i en pipeline på 1790 GB/s bremser enkeltstrøms-dekoding. Det kan tvinges med `--gpus all`.
* **Den krymper KV-hurtigbufferen for å holde vektene i VRAM.** Hver gigabyte gitt til hurtigbufferen er en gigabyte vekter dyttet ut på PCIe-bussen, og å lese en vekt over PCIe koster rundt tretti ganger det den koster fra VRAM. På 70B-en ovenfor flytter denne ene avveiningen alene tallet fra 2,3 til 17,8 token/s.

---

<a id="optimisations"></a>

## Å være rask

Fire optimaliseringer, hver bekreftet av et ekvivalensbevis og ikke bare av en stoppeklokke: en optimalisering som endrer svaret, er en feil.

Tette NVFP4-lag i tette modeller går som standard gjennom Marlin-oppsettet (+57 til +90 % gjennomstrømning ved b = 8, TTFT +2 til +4 ms, jf. revue/poste6-piece147-verdict-24-09.md; reserveløsning `ACVRAM_PROJ_MARLIN=0`, se [CHANGELOG.md](../CHANGELOG.md)).

### Spekulativ dekoding (`--speculative`)

Å dekode ett token med en batch av størrelse 1 er minnebegrenset: maskinen leser alle aktive vekter for å produsere ett eneste token. Å verifisere K foreslåtte token leser de samme vektene **én gang**. To forslagsstillere:

* `ngram` (standard) — leter etter det gjeldende suffikset tidligere i konteksten og foreslår det som fulgte. Koster ingenting, krever ingen modell. Lønnsomt når utdata kopierer inndata: koderedigering, RAG, oppsummering.
* `draft` — en liten modell på en annen enhet. På denne riggen er den enheten RTX 3080 Ti, som planleggeren med vilje lar stå ledig for enhver modell som får plass på 5090.

`mtp` (modellens `nextn`-hode) og `auto` finnes også; ikke lønnsomme som de er og ikke aktivert som standard — se `docs/ARCHITECTURE.md`.

Aksepten er eksakt, ikke tilnærmet: et forslag aksepteres med sannsynlighet `min(1, p/q)`, og et avslag trekker på nytt fra den normaliserte positive delen av `p - q`. Målt over 40 000 trekninger mot et bevisst dårlig kalibrert utkast holder den utsendte fordelingen seg innenfor 0,002 total variasjon fra målet — spekulasjon kjøper fart, aldri et annet svar.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefiks-hurtigbuffer (aktiv som standard)

Blokker adresseres via den *lenkede* hashen av tokenutsnittet sitt: to forespørsler som deler en systeminstruks, deler blokkene dens, og den andre trenger ikke lenger å forhåndsberegne dem. Lenkingen er uunnværlig: de samme seksten tokenene i en annen kontekst inneholder ikke de samme nøklene og verdiene, og å hashe bare utsnittet ville servere én sekvens' hurtigbuffer til en annen.

En frigitt blokk hvis innhold fortsatt kan identifiseres, går inn i en LRU-kø i stedet for frilisten: hurtigbufferen overlever dermed mellom forespørsler uten noen gang å avslå en tildeling den kunne ha betjent.

### Beregning på vertsnivået (`--host-exec`)

Et lag med vekter i RAM kan kopieres til GPU-en eller beregnes på stedet. Begge veiene er minnebegrensede og leser de samme bytene: den raskeste er den med den bredeste bussen — PCIe 5.0 x16 gir rundt 54 GB/s, DDR5 i dual channel rundt 70 GB/s — og å beregne på stedet lar dessuten GPU-en være ledig i stedet for å la den vente på en kopi.

Det lønner seg bare hvis prosessoren leser de 4-bit-pakkede vektene direkte. Derav en liten C++-kjerne med en AVX2-sti (`acvram_cpu.cpp`, lastet via ctypes, uten Python-headere eller ninja). Selv på sin **skalare** reservegren slår den `dequantize() @ x` med en faktor 1,44 i INT4 og 3,21 i NVFP4, fordi sistnevnte først skriver en 32-bits kopi av hele matrisen.

På Mistral-Large-123B går planleggerens estimat fra 1,35 til 2,42 token/s.

### Blandet presisjon (`--snr-floor`, av som standard)

Konverteren måler signal-støy-forholdet ved utgangen av hvert lag for hver tensor og kan forfremme dem som faller under `--snr-floor` til et bredere format, innenfor en grense på 15 % av tensorene og et pristak (`--promotion-cout-max`, i tilførte mebibyte).

Gulvet er **null som standard**: ingenting forfremmes. Dekoding er begrenset av minnebåndbredden, og målingen på `Huihui-Qwen3.8-27B` avgjør — et gulv på 25 dB koster 13,4 % minne og 10,6 % gjennomstrømning (18,50 GiB og 41,8 t/s mot 16,02 og 46,2) for 2,0 % perpleksitet (42,591 mot 43,447, korpus på 16 383 token). `--snr-floor 25` gjenoppretter den gamle oppførselen når kvalitet går foran fart.

### Og `acvram eval`

Signal-støy-forholdet og logit-cosinus er tilnærminger. `acvram eval REP [REP ...]` måler perpleksiteten med glidende vindu, slik at et formatvalg avgjøres på bevis:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-endepunkter

| endepunkt | merknader |
|---|---|
| `POST /v1/chat/completions` | SSE-strøm eller enkeltsvar; bruker modellens samtalemal |
| `POST /v1/completions` | prompt som tekst eller som token-id-er |
| `POST /v1/embeddings` | gjennomsnitt av siste skjulte tilstander, L2-normalisert, `dimensions` respektert |
| `GET /v1/models` | pluss en `acvram`-blokk: formater, enheter, KV-hurtigbufferens kapasitet |
| `GET /health`, `GET /metrics` | dekodingsgjennomstrømning, belegg av KV-blokker |

Feltnavnene i disse svarene forblir på engelsk: det er OpenAI-protokollen, og å oversette dem ville ødelegge alle eksisterende klienter.

---

<a id="chiffres"></a>

## Hvor tallene kommer fra

Hver verdi sitert ovenfor produseres av kode i dette depotet og kontrolleres av `pytest`. Målinger gjort på prosessoren med referansekjernene:

| format | bit/vekt | vektenes SNR | logit-cosinus vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

To funn fra disse målingene har endret standardverdiene:

* **En Hadamard-rotasjon hjelper INT4 og ikke NVFP4.** INT4s grupper på 128 kan ikke absorbere en isolert avvikende kanal, så å spre de ekstreme verdiene er verdt en n log n-transformasjon per aktivering. NVFP4s blokker på 16 bærer allerede sin egen skala. Derav `--hadamard auto`, som bare bruker den på INT4.
* **INT8 slår FP8 E4M3 for KV-hurtigbufferen**, 44 dB mot 32 dB ved samme størrelse, fordi en skala per (token, hode) allerede gir det dynamiske området FP8 bruker eksponentbit på. Begge kortene bruker derfor en KV-hurtigbuffer i INT8, selv om 5090 kunne kjørt FP8. Et `k8v4`-format (verdier i INT4, −22 % hurtigbufferbyte) finnes som alternativ, **ikke kvalifisert** — se `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentasjon

| Dokument | Innhold |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **gjenoppta prosjektet på en annen maskin** |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | hvordan delene henger sammen |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | ren NVFP4 eller attention+GDN i int8 per kanal, på en Gated DeltaNet-hybrid |
| [`docs/MATERIEL.md`](MATERIEL.md) | innstilling av akkurat denne maskinen |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **det som ikke er gjort**, les først |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | arbeidskonvensjoner for koden (språk, stil, kontroller før push) |

---

<a id="resultats"></a>

## Målte resultater (22/09/2026, RTX 5090 ved 400 W, regime ≥ 20 s på energimåleren)

Qwen3-Coder-30B-A3B i NVFP4 (eksperter) + INT8 (attention, hode), samme protokoll for alle motorer (`outils/`, ett kort, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekoding, 12 sekvenser | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekoding, 1 sekvens | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, trukket) |

¹ Erratum av 22/09 : `serve` spekulerer som standard (`--speculative ngram`, cli.py), konkurrentene ikke; 380,8 t/s publisert tidligere ble målt MED spekulasjon. Uten spekulasjon (`--speculative none`, samme kjede, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram er **tredje** ved b=1, bak llama.cpp og vLLM. I energi ligger den fortsatt foran llama.cpp (0,601 mot 0,700 J/token netto). Ved b=12 er spekulasjon aldri aktiv (vakt `lot_max=2`): denne cellen var allerede på like vilkår.

² 23/09, samme sesjon, samme HTTP-klient (`banc-llamacpp-16-09.py` mot `acvram serve` og `vllm serve`), `-lgc 2700` satt eksplisitt rundt hver arm, celler vekslet A V V A, ≥ 5 batcher per arm, avvik erklært bare over 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 ved dekoding, avrullet attention-reduksjon): avvik −1,6 %, **under 2 σ: likt gjennomstrømning**. I J/token er **vLLM fortsatt foran med 7,0 %** (over 2 σ). Med 0.6.37 gav samme protokoll −4,7 %.

³ Samme sesjon og protokoll som ², uten spekulasjon på begge sider: acvram 312,3 mot vLLM 284,8 — **acvram foran med 9,7 % i gjennomstrømning** (over 2 σ); J/token: **likt** (avvik 0,04 %, under 2 σ).

⁴ 23/09, samme protokoll mot llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 med omskrevet ruting (+5,6 %): acvram 310,8 mot llama.cpp 329,9 t/s — **llama.cpp foran med 5,8 % i gjennomstrømning, acvram foran med 13,4 % i J/token** (0,598 mot 0,691).

Gjennomstrømning for dagen (arbeidsstasjon 1030, øko-regime `-lgc 2700`, pipeline i drift; grådig sampling fanget i CUDA-grafen, standard fra 0.6.35). b=12-cellen for acvram er en forseglet offisiell celle (median av 6 flettede vinduer, klokke per vindu).

> **Erratum (23/09/2026).** Den tidligere publiserte vLLM-sammenligningen (b=12: 1 782 mot 1 634 t/s; b=1: 290,6) stilte acvram målt over HTTP mot vLLM målt **offline** (`LLM().generate()`), og erratumet av 22/09 hevdet feilaktig at vLLM-cellen gikk via `vllm serve`. 23/09: samme HTTP-klient for begge, og `-lgc` satt for begge (acvram setter sin ved oppstart, `vllm serve` ikke: uten dette forbeholdet kjørte vLLM på ~2 930 MHz mot ~2 650). Resultat i note ²: vLLM foran med 9,1 % ved b=12.

Om morgenen 14/09 lå acvram på 630 t/s og 0,619 J/token i samme celle: gevinstene kommer fra Blackwells innebygde FP4-MMA (`mma.sync … kind::mxf4nvf4`, ×7,9 over bf16), fra MoE som gruppert GEMM per batch-bøtte, fra ruting i én enkelt kjerne (3 677 → 1 517 starter per steg) og fra en smal tensorkjerne-GEMM for projeksjonene. Hvert tall har sitt notat i `acvram-memoire/revue/` med prediksjonen forseglet før målingen, instrumentet og regimet dets — et tall uten regime publiseres ikke.

Der acvram ligger foran: MLA-modeller (GLM-4.7-Flash) i innebygd sm_120-NVFP4, som vLLM bare serverer i FP8 (b=1: 165,35 t/s i drift); modeller som ikke får plass i VRAM. Dekoding med én sekvens er ikke blant dem: uten spekulasjon ligger acvram der foran vLLM med 9,7 % (note ³), bak llama.cpp med 5,8 % i gjennomstrømning men foran den med 13,4 % i energi (note ⁴). Ved stor batch, på en MoE som får plass i VRAM, er vLLM likt i gjennomstrømning ved b=12 (1 995,1 mot 2 027,0 t/s, under 2 σ, note ²) men holder 7,0 % mindre J/token; acvram har der gått fra 1 540 t/s (0.6.34) til 1 995 (0.6.38).

---

<a id="etat"></a>

## Status

Versjon 0.6.38. Alt kjører på 5090: CUDA-kjerner kompilert for `sm_120a` (innebygd FP4) og `sm_86`, CUDA-grafer, NVFP4/INT8/INT4-kvantisering, HTTP-server. Rekkverk på plass: kortet er usynlig for arbeidsøkter (`CUDA_VISIBLE_DEVICES` tom), og bare `outils/carte.sh` låner det ut, under lås, til én måling om gangen; en vakt logger all tilgang utenfor låsen; en energimåling som dekker mer enn ett kort eller under 10 s, ugyldiggjøres; en modell lastet i degradert modus sier det og deltar ikke i en duell.

4 107 tester (`pytest --collect-only -q`, ett minutt på prosessoren; GPU-tester kjører bare under `carte.sh`). Arbeidsoppfølging: `acvram-memoire/` (regler, register, hefter, gjennomgang av flere hundre notater).

---

<a id="credits"></a>

## Kreditering

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, under Apache-2.0-lisens: `acvram/kernels/marlin_port/` bærer Marlin-kjernene (MoE og tett), med full fil-for-fil-tilskrivelse i [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, Blackwells FP4-tensorkjerner (`sm_120`) og bibliotekene dette prosjektet er avhengig av.
- **PyTorch** — tensormotor og C++/CUDA-utvidelser.

Uavhengig prosjekt, ikke tilknyttet ASUS, NVIDIA eller vLLM-prosjektet.

---

<a id="licence"></a>

## Lisens

[GPL-3.0 eller senere](../LICENSE) for koden i dette depotet. `acvram/kernels/marlin_port/` inneholder kode portert fra [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kjernene `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), under Apache-2.0-lisens: hver fil bevarer sitt opprinnelige toppfelt, lisensen står i `LICENSE-vllm`, og fillisten, opprinnelseskommiten og endringene står i [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Støtt prosjektet

Utviklingen av acvram skjer på privat maskinvare. Hvis prosjektet er nyttig for deg:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Kjøp%20meg%20en%20kaffe&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Oversettelser: [TRADUIRE.md](TRADUIRE.md) (fransk; prosjektets bidragsguide er ikke oversatt ennå).
