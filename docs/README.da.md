<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

En OpenAI-API-kompatibel inferensgateway, der behandler hukommelse som et hierarki, giver hver GPU det numeriske format dens silicium bedst kan læse, og optimerer hvert token i joule såvel som i sekunder.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · **🇩🇰 Dansk** · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Sammenligning af gennemløb og energiforbrug mod vLLM og llama.cpp" width="720"></p>

---

## Indhold

- [To ideer](#idees)
- [Kom hurtigt i gang](#demarrage)
- [Installation](#installer)
- [Hvad `acvram plan` siger](#plan)
- [Gå hurtigere](#optimisations)
- [HTTP-endepunkter](#http)
- [Hvor tallene kommer fra](#chiffres)
- [Dokumentation](#documentation)
- [Målte resultater](#resultats)
- [Status](#etat)
- [Kreditering](#credits)
- [Licens](#licence)
- [Støt projektet](#soutien)

---

<a id="idees"></a>

## To ideer

Designet til en bestemt maskine:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-kerner + 16 E-kerner) |
| Bundkort | ASUS ROG Maximus Z790 Dark Hero |
| Hukommelse | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); begge kort i PCIe x8/x8, begrænset til 400 W / 275 W |

**Ét format pr. GPU.** RTX 5090 har FP4-tensorkerner; RTX 3080 Ti har ingen, og heller ikke FP8. At bruge samme format til begge ville spilde 5090'ens potentiale. Konverteren skriver derfor *samme model to gange*, i det format hver destination reelt kan udnytte:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| vægte | **NVFP4** — E2M1 + FP8 E4M3-skala hver 16 | **INT4** — uint4 + fp16-skala og nulpunkt hver 128 |
| bit pr. vægt | 4,50 | 4,16 |
| vs. BF16 | ×3,56 mindre | ×3,85 mindre |
| beregningstilstand | FP4-tensorkerner | dekvantiseret til FP16 i kernen, FP16-tensorkerner |
| KV-cache | INT8 | INT8 |

32 GB VRAM ved 4,5 bit pr. vægt indeholder omkring **56 milliarder parametre**, mod 16 milliarder i BF16. På begge kort giver det tilsammen omkring **78 milliarder resident parametre**, endnu inden hovedhukommelsen tages i brug.

**Hukommelse er et hierarki, ikke en mur.** Tre niveauer, og planlæggeren måler hvad hver enkelt koster i stedet for at gætte på, at modellen passer:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Kom hurtigt i gang

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Enhver OpenAI-klient kan derefter koble sig til:

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

Fra kildekoden (alle platforme):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Eller som pakke, en fil vedhæftet hver [GitHub-udgivelse](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanal | Fil vedhæftet udgivelsen | Kommando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (navne genereret af `rpmbuild`, ikke faste) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (eller `rpmbuild --rebuild *.src.rpm` ud fra `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpak` | `flatpak install acvram-<version>.flatpak` |

Pip udgives ikke som pakke (ingen bygget wheel): `pip install -e '.[dev]'` installerer fra en klon af kilden, ligesom `./install.sh`.

---

<a id="plan"></a>

## Hvad `acvram plan` siger

Planlæggeren er værd at køre, før noget som helst downloades. Den besvarer de spørgsmål, der afgør, om en model kan bruges på denne maskine:

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

Den udforsker konfigurationsrummet i stedet for at vælge den første, der passer, og to af dens beslutninger er kontraintuitive nok til at forklare:

* **Den lader 3080 Ti stå ubrugt**, når en model passer på 5090 alene. Trinene i en pipeline udføres i serie: at tilføje et trin på 912 GB/s til en pipeline på 1790 GB/s bremser enkelt-strøms-dekodning. Man kan tvinge det med `--gpus all`.
* **Den skærer ned på KV-cachen for at holde vægtene i VRAM.** Hver gigabyte givet til cachen er en gigabyte vægte, der skubbes ud på PCIe-bussen, og at læse en vægt via PCIe koster omkring tredive gange så meget som fra VRAM. På 70B-modellen ovenfor får denne afvejning alene gennemløbet til at gå fra 2,3 til 17,8 tokens/s.

---

<a id="optimisations"></a>

## Gå hurtigere

Fire optimeringer, hver verificeret med et ækvivalensbevis og ikke bare et stopur: en optimering, der ændrer svaret, er en fejl.

Tætte modellers NVFP4-lineære lag går som standard gennem Marlin-layoutet (+57 til +90 % gennemløb ved b = 8, TTFT +2 til +4 ms, jf. revue/poste6-piece147-verdict-24-09.md; reserveløsning `ACVRAM_PROJ_MARLIN=0`, se [CHANGELOG.md](../CHANGELOG.md)).

### Spekulativ dekodning (`--speculative`)

At dekode ét token med en batch på 1 er hukommelsesbegrænset: maskinen læser alle aktive vægte for at producere ét eneste token. At verificere K foreslåede tokens læser disse samme vægte **kun én gang**. To forslagsstillere:

* `ngram` (standard) — søger den nuværende slutning tidligere i konteksten og foreslår det, der fulgte. Kostar ingenting, kræver ingen model. Rentabel når outputtet kopierer input: kodeeditering, RAG, opsummering.
* `draft` — en lille model på en anden enhed. På dette rig er den enhed RTX 3080 Ti, som planlæggeren bevidst lader stå ledig for enhver model, der passer på 5090.

`mtp` (modellens `nextn`-hoved) og `auto` findes også; ikke rentable som de er og ikke aktiveret som standard — se `docs/ARCHITECTURE.md`.

Accepten er eksakt, ikke tilnærmet: et forslag accepteres med sandsynligheden `min(1, p/q)`, og et afslag genudtager fra den normaliserede positive del af `p - q`. Målt over 40 000 træk mod et bevidst dårligt kalibreret udkast forbliver den udsendte fordeling inden for 0,002 total variation fra målet — spekulation køber hastighed, aldrig et andet svar.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Præfikscache (aktiv som standard)

Blokke adresseres via den *kædede* hash af deres tokensegment: to forespørgsler, der deler en systemprompt, deler dens blokke, og den anden behøver ikke genberegne dem. Kædningen er afgørende: samme seksten tokens i en anden kontekst indeholder ikke de samme nøgler og værdier, og at hashe kun segmentet ville betjene en sekvens' cache til en anden.

En frigivet blok, hvis indhold stadig er genkendeligt, går ind i en LRU-kø i stedet for listen over fri blokke: cachen overlever dermed mellem forespørgsler uden nogensinde at afvise en allokering, den kunne have betjent.

### Beregning på værtsniveau (`--host-exec`)

Et lag, hvis vægte ligger i RAM, kan kopieres til GPU'en eller beregnes på stedet. Begge veje er hukommelsesbegrænsede og læser de samme bytes: den hurtigste er den med den bredeste bus — PCIe 5.0 x16 giver omkring 54 GB/s, dual-channel DDR5 omkring 70 GB/s — og at beregne på stedet efterlader desuden GPU'en fri i stedet for at lade den vente på en kopiering.

Det er kun værd noget, hvis processoren læser de 4-bit-pakkede vægte direkte. Deraf en lille C++-kerne med en AVX2-vej (`acvram_cpu.cpp`, indlæst via ctypes, uden Python-headers eller ninja). Selv på sin **skalære** reservevej slår den `dequantize() @ x` med en faktor 1,44 for INT4 og 3,21 for NVFP4, fordi sidstnævnte først skriver en 32-bit-kopi af hele matricen.

På Mistral-Large-123B går planlæggerens estimat fra 1,35 til 2,42 tokens/s.

### Blandet præcision (`--snr-floor`, slået fra som standard)

Konverteren måler signal/støj-forholdet ved lagets output for hver tensor og kan opgradere til et bredere format dem, der falder under `--snr-floor`, inden for en grænse på 15 % af tensorerne og en maksimalpris (`--promotion-cout-max`, i tilføjede mebibyte).

Grænsen er **nul som standard**: intet opgraderes. Dekodning er hukommelsesbåndbreddebegrænset, og målingen på `Huihui-Qwen3.8-27B` afgør sagen — en grænse på 25 dB koster 13,4 % hukommelse og 10,6 % gennemløb (18,50 GiB og 41,8 t/s mod 16,02 og 46,2) for 2,0 % perplexitet (42,591 mod 43,447, korpus på 16 383 tokens). `--snr-floor 25` genindfører den gamle adfærd, når kvalitet vægter mere end hastighed.

### Og `acvram eval`

Signal/støj-forholdet og logit-cosinus er tilnærmelser. `acvram eval REP [REP ...]` måler perplexitet med et glidende vindue, så et formatvalg kan afgøres på beviser:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-endepunkter

| endepunkt | noter |
|---|---|
| `POST /v1/chat/completions` | SSE-strøm eller enkelt svar; bruger modellens samtaleskabelon |
| `POST /v1/completions` | prompt som tekst eller token-id'er |
| `POST /v1/embeddings` | gennemsnitlige endelige skjulte tilstande, L2-normaliseret, `dimensions` respekteret |
| `GET /v1/models` | plus en `acvram`-blok: formater, enheder, KV-cachekapacitet |
| `GET /health`, `GET /metrics` | dekodningsgennemløb, KV-blokbelægning |

Feltnavnene i disse svar forbliver på engelsk: det er OpenAI-protokollen, og at oversætte dem ville bryde alle eksisterende klienter.

---

<a id="chiffres"></a>

## Hvor tallene kommer fra

Hver værdi nævnt ovenfor er produceret af kode fra dette repository og verificeret med `pytest`. Målinger foretaget på processor med referencekerner:

| format | bit/vægt | vægtenes SNR | logit-cosinus vs. BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

To konstateringer fra disse målinger har ændret standardværdierne:

* **En Hadamard-rotation hjælper INT4, men ikke NVFP4.** INT4's grupper af 128 kan ikke absorbere en isoleret afvigende kanal, så det er en n log n-transformation pr. aktivering værd at udjævne ekstremværdierne. NVFP4's blokke af 16 bærer allerede deres egen skala. Deraf `--hadamard auto`, som kun anvender det på INT4.
* **INT8 slår FP8 E4M3 til KV-cachen**, 44 dB mod 32 dB ved samme størrelse, fordi en skala pr. (token, hoved) allerede giver det dynamiske område, som FP8 bruger eksponentbit på. Begge kort bruger derfor en INT8 KV-cache, selv om 5090 kunne køre FP8. Et `k8v4`-format (værdier i INT4, −22 % cache-bytes) findes som en mulighed, **ikke kvalificeret** — se `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentation

| Dokument | Indhold |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **at genoptage projektet på en anden maskine** (fransk) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | hvordan delene passer sammen |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | ren NVFP4 eller attention+GDN i int8 pr. kanal, på en Gated DeltaNet-hybrid |
| [`docs/MATERIEL.md`](MATERIEL.md) | at tilpasse denne bestemte maskine |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **hvad der ikke er gjort endnu**, læs dette først |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | arbejdskonventioner for koden (sprog, stil, kontroller før push) |

---

<a id="resultats"></a>

## Målte resultater (22/09/2026, RTX 5090 ved 400 W, ≥ 20 s vindue på energimåleren)

Qwen3-Coder-30B-A3B i NVFP4 (experts) + INT8 (opmærksomhed, hoved), samme protokol for alle motorer (`outils/`, ét kort, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekodning, 12 sekvenser | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekodning, 1 sekvens | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, fjernet) |

¹ Erratum af 22/09: `serve` spekulerer som standard (`--speculative ngram`, cli.py), konkurrenterne gør ikke; de 380,8 t/s, der hidtil er offentliggjort, blev målt MED spekulation. Uden spekulation (`--speculative none`, samme kæde, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram er **tredje** ved b=1, efter llama.cpp og vLLM. I energi ligger den stadig foran llama.cpp (0,601 mod 0,700 J/token netto). Ved b=12 er spekulation aldrig aktiv (grænsen `lot_max=2`): denne celle var allerede på lige fod.

² 23/09, samme session, samme HTTP-klient (`banc-llamacpp-16-09.py` mod `acvram serve` og `vllm serve`), `-lgc 2700` sat eksplicit rundt om hver arm, celler alterneret A V V A, ≥ 5 batches pr. arm, afvigelse erklæret kun over 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 ved dekodning, udrullet opmærksomhedsreduktion): afvigelse −1,6 %, **under 2 σ: lige gennemløb**. I J/token er **vLLM stadig foran med 7,0 %** (over 2 σ). Med 0.6.37 gav samme protokol −4,7 %.

³ Samme session og protokol som ², uden spekulation på begge sider: acvram 312,3 mod vLLM 284,8 — **acvram foran med 9,7 % i gennemløb** (over 2 σ); J/token: **lige** (afvigelse 0,04 %, under 2 σ).

⁴ 23/09, samme protokol mod llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 med omskrevet routing (+5,6 %): acvram 310,8 mod llama.cpp 329,9 t/s — **llama.cpp foran med 5,8 % i gennemløb, acvram foran med 13,4 % i J/token** (0,598 mod 0,691).

Dagens gennemløb (post 1030, øko-regime `-lgc 2700`, pipeline i drift; grådig sampling fanget i CUDA-grafen, standard siden 0.6.35). acvram-b=12 er en officielt forseglet celle (median af 6 sammenflettede vinduer, klokfrekvens pr. vindue).

> **Erratum (23/09/2026).** Den hidtil offentliggjorte vLLM-sammenligning (b=12: 1 782 mod 1 634 t/s; b=1: 290,6) satte acvram målt via HTTP op mod vLLM målt **offline** (`LLM().generate()`), og erratumet fra 22/09 påstod fejlagtigt, at vLLM-cellen gik via `vllm serve`. Den 23/09: samme HTTP-klient for begge, og `-lgc` sat for begge (acvram sætter sin egen ved opstart, `vllm serve` gør ikke: uden denne forholdsregel kørte vLLM ved ~2 930 MHz mod ~2 650). Resultat i note ²: vLLM foran med 9,1 % ved b=12.

Om morgenen den 14/09 lå acvram på 630 t/s og 0,619 J/token på samme celle: gevinsterne kommer fra Blackwells native FP4-MMA (`mma.sync … kind::mxf4nvf4`, ×7,9 mod bf16), MoE i grupperet GEMM pr. batch-bucket, routing i en enkelt kerne (3 677 → 1 517 lanceringer pr. skridt) og en smal GEMM på tensorkerner til projektionerne. Hvert tal har sin note i `acvram-memoire/revue/` med forudsigelsen forseglet før målingen, instrumentet og dets regime — et tal uden et regime bliver ikke offentliggjort.

Hvor acvram er foran: MLA-modeller (GLM-4.7-Flash) i nativ NVFP4 sm_120, som vLLM kun betjener i FP8 (b=1: 165,35 t/s i drift); modeller der ikke passer i VRAM. Enkelt-sekvens-dekodning er ikke en af dem: uden spekulation er acvram der foran vLLM med 9,7 % (note ³), bagefter llama.cpp med 5,8 % i gennemløb men foran den med 13,4 % i energi (note ⁴). Ved stor batch, på en MoE der passer i VRAM, er vLLM på lige gennemløb ved b=12 (1 995,1 mod 2 027,0 t/s, under 2 σ, note ²) men holder 7,0 % mindre J/token; acvram er der gået fra 1 540 t/s (0.6.34) til 1 995 (0.6.38).

---

<a id="etat"></a>

## Status

Version 0.6.38. Alt kører på 5090'en: CUDA-kerner kompileret til `sm_120a` (nativ FP4) og `sm_86`, CUDA-grafer, NVFP4/INT8/INT4-kvantisering, HTTP-server. Sikkerhedsforanstaltninger på plads: kortet er usynligt for arbejdssessioner (`CUDA_VISIBLE_DEVICES` tom), og kun `outils/carte.sh` udlåner det, under lås, til én måling ad gangen; en vagt logfører al adgang uden lås; en energimåling der dækker mere end ét kort eller mindre end 10 s bliver ugyldiggjort; en model indlæst i degraderet regime siger det og deltager ikke i en duel.

4 107 tests (`pytest --collect-only -q`, ét minut på processor; GPU-tests køres kun under `carte.sh`). Arbejdsopfølgning: `acvram-memoire/` (regler, register, notesbøger, gennemgang af flere hundrede noter).

---

<a id="credits"></a>

## Kreditering

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, under Apache-2.0-licens: `acvram/kernels/marlin_port/` bærer dens Marlin-kerner (MoE og tæt), med fuld fil-for-fil-attribution i [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, Blackwells FP4-tensorkerner (`sm_120`), og de biblioteker projektet afhænger af.
- **PyTorch** — tensormotor og C++/CUDA-udvidelser.

Uafhængigt projekt, ikke tilknyttet ASUS, NVIDIA eller vLLM-projektet.

---

<a id="licence"></a>

## Licens

[GPL-3.0 eller senere](../LICENSE) for koden i dette repository. `acvram/kernels/marlin_port/` indeholder kode porteret fra [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kernerne `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), under Apache-2.0-licens: hver fil bevarer sin oprindelige header, licensteksten er i `LICENSE-vllm`, og fillisten, oprindelseskommittet og ændringerne er i [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Støt projektet

Udviklingen af acvram sker på personligt hardware. Hvis projektet er nyttigt for dig:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Giv%20en%20kop%20kaffe&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Oversættelser: [TRADUIRE.md](TRADUIRE.md) (fransk; projektets bidragsguide er endnu ikke oversat).
