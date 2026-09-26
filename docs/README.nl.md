<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-steunen-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Een met de OpenAI-API compatibele inferentie-gateway die geheugen als een hiërarchie behandelt, elke GPU het numerieke formaat geeft dat zijn silicium het beste kan lezen, en elk token evenveel optimaliseert in joules als in seconden.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · **🇳🇱 Nederlands** · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Doorvoer- en energievergelijking met vLLM en llama.cpp" width="720"></p>

---

## Inhoud

- [Twee ideeën](#idees)
- [Snel starten](#demarrage)
- [Installeren](#installer)
- [Wat `acvram plan` zegt](#plan)
- [Sneller gaan](#optimisations)
- [HTTP-eindpunten](#http)
- [Waar de cijfers vandaan komen](#chiffres)
- [Documentatie](#documentation)
- [Gemeten resultaten](#resultats)
- [Status](#etat)
- [Credits](#credits)
- [Licentie](#licence)
- [Het project steunen](#soutien)

---

<a id="idees"></a>

## Twee ideeën

Ontworpen voor één specifieke machine:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-kernen + 16 E-kernen) |
| Moederbord | ASUS ROG Maximus Z790 Dark Hero |
| Geheugen | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Systeem | Ubuntu 26.04 LTS (CUDA 13) ; beide kaarten op PCIe x8/x8, begrensd op 400 W / 275 W |

**Één formaat per GPU.** De RTX 5090 heeft FP4-tensorkernen; de RTX 3080 Ti niet, en ook geen FP8. Beide op één gemeenschappelijk formaat afstemmen zou de 5090 verspillen. De converter schrijft dus *tweemaal hetzelfde model*, in het formaat dat elke bestemming werkelijk kan benutten:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| gewichten | **NVFP4** — E2M1 + FP8-E4M3-schaal per 16 | **INT4** — uint4 + fp16-schaal en -nulpunt per 128 |
| bits per gewicht | 4,50 | 4,16 |
| tegenover BF16 | ×3,56 kleiner | ×3,85 kleiner |
| rekenmodus | FP4-tensorkernen | gedekwantiseerd naar FP16 in de kernel, FP16-tensorkernen |
| KV-cache | INT8 | INT8 |

32 GB VRAM bij 4,5 bits per gewicht bevat ongeveer **56 miljard parameters**, tegen 16 miljard in BF16. Op beide kaarten samen geeft dat ongeveer **78 miljard residente parameters** nog vóór het hoofdgeheugen wordt aangeraakt.

**Geheugen is een hiërarchie, geen muur.** Drie niveaus, en de planner meet wat elk niveau kost in plaats van te hopen dat het model erin past:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Snel starten

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Elke OpenAI-client sluit er vervolgens op aan:

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

## Installeren

Vanuit de broncode (alle platforms):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Of via een pakket, een bestand toegevoegd aan elke [GitHub-release](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanaal | Bestand toegevoegd aan de release | Commando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (namen gegenereerd door `rpmbuild`, niet vast) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (of `rpmbuild --rebuild *.src.rpm` vanuit de `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpak` | `flatpak install acvram-<version>.flatpak` |

Pip wordt niet als pakket gepubliceerd (geen gebouwde wheel): `pip install -e '.[dev]'` installeert vanuit een kloon van de broncode, net als `./install.sh`.

---

<a id="plan"></a>

## Wat `acvram plan` zegt

De planner is de moeite waard om vóór elke download te starten. Hij beantwoordt de vragen die bepalen of een model op deze machine bruikbaar is:

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

Hij doorzoekt de ruimte van configuraties in plaats van de eerste die past te nemen, en twee van zijn beslissingen zijn contra-intuïtief genoeg om te vermelden:

* **Hij laat de 3080 Ti ongebruikt** wanneer een model op de 5090 alleen past. De etappes van een pijplijn draaien serieel: een stap van 912 GB/s toevoegen aan een pijplijn van 1790 GB/s vertraagt het decoderen met één stroom. Afdwingen met `--gpus all`.
* **Hij verkleint de KV-cache om de gewichten in VRAM te houden.** Elke gigabyte die aan de cache wordt gegeven, is een gigabyte gewichten die naar de PCIe-bus wordt verdrongen, en een gewicht via PCIe lezen kost ongeveer dertig keer zoveel als vanuit VRAM. Op de 70B hierboven brengt alleen deze afweging het van 2,3 naar 17,8 tokens/s.

---

<a id="optimisations"></a>

## Sneller gaan

Vier optimalisaties, elk gecontroleerd met een equivalentiebewijs en niet alleen met een stopwatch: een optimalisatie die het antwoord verandert, is een bug.

De NVFP4-lineairen van dichte modellen gaan standaard via de Marlin-layout (+57 tot +90% doorvoer bij b = 8, TTFT +2 tot +4 ms volgens revue/poste6-piece147-verdict-24-09.md; terugval `ACVRAM_PROJ_MARLIN=0`, zie [CHANGELOG.md](../CHANGELOG.md)).

### Speculatief decoderen (`--speculative`)

Een token decoderen met een lot van grootte 1 wordt door het geheugen beperkt: de machine leest alle actieve gewichten om één token te produceren. K voorgestelde tokens verifiëren leest diezelfde gewichten **maar één keer**. Twee voorstellers:

* `ngram` (standaard) — zoekt het huidige suffix eerder in de context en stelt voor wat erop volgde. Kost niets, vraagt geen model. Rendabel wanneer de uitvoer de invoer overneemt: codebewerking, RAG, samenvatting.
* `draft` — een klein model op een tweede apparaat. Op deze rig is dat apparaat de RTX 3080 Ti, die de planner voor elk model dat op de 5090 past bewust ongebruikt laat.

`mtp` (`nextn`-kop van het model) en `auto` bestaan ook; op dit moment niet rendabel en niet standaard geactiveerd — zie `docs/ARCHITECTURE.md`.

De acceptatie is exact, niet benaderend: een voorstel wordt aanvaard met kans `min(1, p/q)` en een weigering herbemonstert in het genormaliseerde positieve deel van `p - q`. Gemeten over 40 000 trekkingen tegen een opzettelijk slecht gekalibreerd concept, blijft de uitgezonden verdeling op 0,002 totale variatie van het doel — speculatie koopt snelheid, nooit een ander antwoord.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefix-cache (standaard actief)

Blokken worden geadresseerd via de *geketende* hash van hun tokensegment: twee verzoeken die een systeeminstructie delen, delen ook diens blokken, en het tweede hoeft ze niet meer opnieuw te berekenen. De ketening is onmisbaar: dezelfde zestien tokens in een andere context bevatten niet dezelfde keys en values, en alleen het segment hashen zou de cache van de ene sequentie aan een andere bedienen.

Een vrijgegeven blok waarvan de inhoud herkenbaar blijft, komt in een LRU-wachtrij terecht in plaats van in de lijst met vrije blokken: zo overleeft de cache tussen verzoeken zonder ooit een toewijzing te weigeren die hij had kunnen bedienen.

### Berekening op het host-niveau (`--host-exec`)

Een laag waarvan de gewichten in RAM staan, kan naar de GPU worden gekopieerd of ter plekke worden berekend. Beide paden worden door het geheugen beperkt en lezen dezelfde bytes: het snelste is dat met de breedste bus — PCIe 5.0 x16 geeft ongeveer 54 GB/s, DDR5 in dual-channel ongeveer 70 GB/s — en ter plekke berekenen laat de GPU bovendien vrij in plaats van te wachten op een kopie.

Dit loont alleen als de processor de op 4 bits verpakte gewichten rechtstreeks leest. Vandaar een kleine C++-kernel met een AVX2-pad (`acvram_cpu.cpp`, geladen via ctypes, zonder Python-headers of ninja). Zelfs op zijn **scalaire** terugvaltak verslaat hij `dequantize() @ x` met een factor 1,44 in INT4 en 3,21 in NVFP4, omdat laatstgenoemde eerst een 32-bits kopie van de volledige matrix schrijft.

Op Mistral-Large-123B gaat de schatting van de planner van 1,35 naar 2,42 tokens/s.

### Gemengde precisie (`--snr-floor`, standaard uit)

De converter meet de signaal-ruisverhouding aan de uitgang van elke laag voor elke tensor en kan degene die onder `--snr-floor` vallen promoveren naar een breder formaat, tot maximaal 15% van de tensoren en een kostenplafond (`--promotion-cout-max`, in extra mebibytes).

De drempel staat **standaard op nul**: er wordt niets gepromoveerd. Het decoderen wordt beperkt door de geheugenbandbreedte, en de meting op `Huihui-Qwen3.8-27B` beslecht het — een drempel van 25 dB kost 13,4% geheugen en 10,6% doorvoer (18,50 GiB en 41,8 t/s tegen 16,02 en 46,2) voor 2,0% perplexiteit (42,591 tegen 43,447, corpus van 16 383 tokens). `--snr-floor 25` herstelt het oude gedrag wanneer kwaliteit voorrang heeft op snelheid.

### En `acvram eval`

De signaal-ruisverhouding en de cosinus van de logits zijn benaderingen. `acvram eval REP [REP ...]` meet de perplexiteit met een schuivend venster, zodat een formaatkeuze op bewijzen wordt beslist:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-eindpunten

| eindpunt | opmerkingen |
|---|---|
| `POST /v1/chat/completions` | SSE-stream of enkel antwoord; gebruikt het conversatiesjabloon van het model |
| `POST /v1/completions` | prompt als tekst of als token-ID's |
| `POST /v1/embeddings` | gemiddelde laatste verborgen toestanden, L2-genormaliseerd, `dimensions` gerespecteerd |
| `GET /v1/models` | plus een `acvram`-blok: formaten, apparaten, KV-cachecapaciteit |
| `GET /health`, `GET /metrics` | decodeer-doorvoer, KV-blokbezetting |

De veldnamen van deze antwoorden blijven in het Engels: dat is het OpenAI-protocol, en ze vertalen zou alle bestaande clients breken.

---

<a id="chiffres"></a>

## Waar de cijfers vandaan komen

Elke hierboven aangehaalde waarde wordt geproduceerd door code uit deze repository en gecontroleerd met `pytest`. Metingen op de processor met de referentiekernels:

| formaat | bits/gewicht | SNR van de gewichten | cosinus van de logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Twee vaststellingen uit deze metingen hebben de standaardwaarden veranderd:

* **Een Hadamard-rotatie helpt INT4 en niet NVFP4.** De groepen van 128 van INT4 kunnen geen geïsoleerd afwijkend kanaal absorberen, zodat het uitspreiden van extreme waarden een n log n-transformatie per activatie waard is. De blokken van 16 van NVFP4 dragen al hun eigen schaal. Vandaar `--hadamard auto`, dat het alleen op INT4 toepast.
* **INT8 verslaat FP8 E4M3 voor de KV-cache**, 44 dB tegen 32 dB bij gelijke grootte, omdat een schaal per (token, kop) al het dynamisch bereik levert waarvoor FP8 exponentbits uitgeeft. Beide kaarten gebruiken dus een KV-cache in INT8, ook al zou de 5090 FP8 kunnen doen. Een formaat `k8v4` (waarden in INT4, −22% cachebytes) bestaat als optie, **niet gekwalificeerd** — zie `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentatie

| Document | Inhoud |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **het project op een andere machine hervatten** (Frans) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | hoe de onderdelen samenkomen |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | puur NVFP4 of attention+GDN in int8 per kanaal, op een Gated DeltaNet-hybride |
| [`docs/MATERIEL.md`](MATERIEL.md) | deze specifieke machine afstellen |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **wat nog niet gedaan is**, lees dit eerst |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | werkconventies voor de code (taal, stijl, controles voor het pushen) |

---

<a id="resultats"></a>

## Gemeten resultaten (22/09/2026, RTX 5090 bij 400 W, regime ≥ 20 s op de energiemeter)

Qwen3-Coder-30B-A3B in NVFP4 (experts) + INT8 (attentie, kop), zelfde protocol voor alle motoren (`outils/`, één kaart, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decodering 12 sequenties | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| decodering 1 sequentie | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, verwijderd) |

¹ Erratum van 22/09: `serve` speculeert standaard (`--speculative ngram`, cli.py), de concurrenten niet; de eerder gepubliceerde 380,8 t/s was gemeten MET speculatie. Zonder speculatie (`--speculative none`, dezelfde keten, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram is **derde** bij b=1, achter llama.cpp en vLLM. In energie blijft het voor op llama.cpp (0,601 tegen 0,700 J/token netto). Bij b=12 is speculatie nooit actief (grens `lot_max=2`): die cel stond dus al op gelijke voet.

² 23/09, dezelfde sessie, dezelfde HTTP-client (`banc-llamacpp-16-09.py` tegen `acvram serve` en `vllm serve`), `-lgc 2700` expliciet ingesteld rond elke arm, afgewisselde cellen A V V A, ≥ 5 loten per arm, verschil pas vermeld voorbij 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 bij decodering, uitgerolde attentiereductie): verschil −1,6%, **onder 2 σ: gelijke doorvoer**. In J/token blijft **vLLM 7,0% voor** (voorbij 2 σ). Met 0.6.37 gaf hetzelfde protocol −4,7%.

³ Zelfde sessie en protocol als ², zonder speculatie aan beide zijden: acvram 312,3 tegen vLLM 284,8 — **acvram 9,7% voor in doorvoer** (voorbij 2 σ); J/token: **gelijk** (verschil 0,04%, onder 2 σ).

⁴ 23/09, zelfde protocol tegen llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 met herschreven routering (+5,6%): acvram 310,8 tegen llama.cpp 329,9 t/s — **llama.cpp 5,8% voor in doorvoer, acvram 13,4% voor in J/token** (0,598 tegen 0,691).

Doorvoer van de dag (post 1030, eco-regime `-lgc 2700`, pijplijn in dienst; gulzige bemonstering vastgelegd in de CUDA-graaf, standaard sinds 0.6.35). De b=12 van acvram is een officieel verzegelde cel (mediaan van 6 verweven vensters, klok per venster).

> **Erratum (23/09/2026).** De eerder gepubliceerde vLLM-vergelijking (b=12: 1 782 tegen 1 634 t/s; b=1: 290,6) stelde acvram gemeten via HTTP tegenover vLLM gemeten **offline** (`LLM().generate()`), en het erratum van 22/09 beweerde ten onrechte dat de vLLM-cel via `vllm serve` liep. Op 23/09: dezelfde HTTP-client voor beide, en `-lgc` ingesteld voor beide (acvram stelt zijn eigen klok in bij het starten, `vllm serve` niet: zonder deze voorzorg draaide vLLM op ~2 930 MHz tegen ~2 650). Resultaat in noot ²: vLLM 9,1% voor bij b=12.

Op de ochtend van 14/09 stond acvram op 630 t/s en 0,619 J/token op dezelfde cel: de winst komt van de native FP4-MMA van Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 tegenover bf16), van MoE in gegroepeerde GEMM per lot-emmer, van routering in één enkele kernel (3 677 → 1 517 aanroepen per stap) en van een smalle GEMM op tensorkernen voor de projecties. Elk cijfer heeft zijn noot in `acvram-memoire/revue/` met de voorspelling verzegeld vóór de meting, het instrument en zijn regime — een cijfer zonder regime wordt niet gepubliceerd.

Waar acvram voorop loopt: MLA-modellen (GLM-4.7-Flash) in native NVFP4 op sm_120, die vLLM alleen in FP8 bedient (b=1: 165,35 t/s in dienst); modellen die niet in VRAM passen. Decoderen met één sequentie hoort daar niet bij: zonder speculatie ligt acvram er 9,7% voor op vLLM (noot ³), 5,8% achter op llama.cpp in doorvoer maar 13,4% voor in energie (noot ⁴). Bij een groot lot, op een MoE die in VRAM past, is vLLM gelijk in doorvoer bij b=12 (1 995,1 tegen 2 027,0 t/s, onder 2 σ, noot ²) maar houdt 7,0% minder J/token over; acvram is er gegaan van 1 540 t/s (0.6.34) naar 1 995 (0.6.38).

---

<a id="etat"></a>

## Status

Versie 0.6.38. Alles draait op de 5090: CUDA-kernels gecompileerd voor `sm_120a` (native FP4) en `sm_86`, CUDA-graven, NVFP4/INT8/INT4-kwantisering, HTTP-server. Waarborgen aanwezig: de kaart is onzichtbaar voor werksessies (`CUDA_VISIBLE_DEVICES` leeg) en alleen `outils/carte.sh` leent hem, onder slot, aan één meting op een moment; een wachter registreert elke toegang buiten het slot; een energiemeting die meer dan één kaart beslaat of minder dan 10 s duurt, wordt ongeldig verklaard; een model dat in verslechterd regime is geladen, meldt dit en neemt niet deel aan een duel.

4 107 tests (`pytest --collect-only -q`, één minuut op de processor; de GPU-tests draaien alleen onder `carte.sh`). Voortgang van het werk: `acvram-memoire/` (regels, register, notitieboekjes, overzicht van meerdere honderden notities).

---

<a id="credits"></a>

## Credits

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, onder Apache-2.0-licentie: `acvram/kernels/marlin_port/` draagt zijn Marlin-kernels (MoE en dicht), met volledige toeschrijving per bestand in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, de FP4-tensorkernen van Blackwell (`sm_120`) en de bibliotheken waarvan dit project afhankelijk is.
- **PyTorch** — tensor-engine en C++/CUDA-extensies.

Onafhankelijk project, niet gelieerd aan ASUS, NVIDIA of het vLLM-project.

---

<a id="licence"></a>

## Licentie

[GPL-3.0 of later](../LICENSE) voor de code in deze repository. `acvram/kernels/marlin_port/` bevat code overgedragen van [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kernels `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), onder Apache-2.0-licentie: elk bestand behoudt zijn oorspronkelijke koptekst, de licentie staat in `LICENSE-vllm` en de bestandenlijst, het oorspronkelijke commit en de wijzigingen staan in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Het project steunen

De ontwikkeling van acvram gebeurt op persoonlijke hardware. Als het project je van pas komt:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Trakteer%20op%20een%20koffie&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Vertalingen: [TRADUIRE.md](TRADUIRE.md) (Frans; de bijdragegids van het project is nog niet vertaald).
