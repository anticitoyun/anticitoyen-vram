<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-soutenir-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Konkludo-pordego kongrua kun la OpenAI API, kiu traktas la memoron kiel hierarkion, donas al ĉiu GPU la nombran formaton, kiun ĝia siliko plej bone legas, kaj optimumigas ĉiun ĵetonon same en ĵulioj kiel en sekundoj.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · **🌐 Esperanto** · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Komparo de trafluo kaj energio kontraŭ vLLM kaj llama.cpp" width="720"></p>

---

## Enhavo

- [La du ideoj](#idees)
- [Rapida ekstarto](#demarrage)
- [Instali](#installer)
- [Kion diras `acvram plan`](#plan)
- [Rapidiĝi](#optimisations)
- [HTTP-alirpunktoj](#http)
- [De kie venas la nombroj](#chiffres)
- [Dokumentado](#documentation)
- [Mezuritaj rezultoj](#resultats)
- [Stato](#etat)
- [Dankoj](#credits)
- [Permesilo](#licence)
- [Subteni la projekton](#soutien)

---

<a id="idees"></a>

## La du ideoj

Desegnita por specifa maŝino:

| | |
|---|---|
| Procesoro | Intel Core i9-14900K (8 P-kernoj + 16 E-kernoj) |
| Bazokarto | ASUS ROG Maximus Z790 Dark Hero |
| Memoro | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistemo | Ubuntu 26.04 LTS (CUDA 13); la du kartoj en PCIe x8/x8, limigitaj je 400 W / 275 W |

**Unu formato po GPU.** La RTX 5090 havas FP4-tensor-kernojn; la RTX 3080 Ti ne havas ilin, nek FP8. Vicigi ambaŭ al komuna formato malŝparus la 5090. La konvertilo do skribas *la saman modelon dufoje*, en la formato, kiun ĉiu celo efektive povas ekspluati:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pezoj | **NVFP4** — E2M1 + skalo FP8 E4M3 ĉiun 16 | **INT4** — uint4 + skalo kaj nulo fp16 ĉiun 128 |
| bitoj po pezo | 4,50 | 4,16 |
| kontraŭ BF16 | ×3,56 pli malgranda | ×3,85 pli malgranda |
| kalkula reĝimo | FP4-tensor-kernoj | malkvantigita al FP16 en la kerno, FP16-tensor-kernoj |
| KV-kaŝmemoro | INT8 | INT8 |

32 Go da VRAM je 4,5 bitoj po pezo enhavas ĉirkaŭ **56 miliardojn da parametroj**, kontraŭ 16 miliardoj en BF16. Sur la du kartoj kune, tio donas ĉirkaŭ **78 miliardojn da loĝantaj parametroj** eĉ antaŭ ol tuŝi la ĉefmemoron.

**La memoro estas hierarkio, ne muro.** Tri etaĝoj, kaj la planilo mezuras kiom kostas ĉiu anstataŭ esperi, ke la modelo enkaĝiĝos:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Rapida ekstarto

```bash
./install.sh                       # virtuala medio + torch cu128 + acvram
acvram doctor                      # ĉu ĉi tiu maŝino pretas, kaj por kio
acvram detect                      # kio efektive ekzistas ĉi tie

acvram plan  ~/modeles/Qwen3-32B                    # kien irus ĉiu tavolo
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Ajna OpenAI-kliento poste konektiĝas:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Saluton"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="inutilise")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Saluton"}])
```

---

<a id="installer"></a>

## Instali

El la fonto (ĉiuj platformoj):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Aŭ per pako, dosiero alligita al ĉiu [GitHub-eldono](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanalo | Dosiero alligita al la eldono | Komando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nomoj generitaj de `rpmbuild`, ne fiksaj) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (aŭ `rpmbuild --rebuild *.src.rpm` el la `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

`.flatpakref` ĉiam instalas la plej lastan publikigitan version de la deponejo.

Antaŭ instalado, kontrolu la elŝutitan dosieron kontraŭ la sumoj aldonitaj al la eldono (`SHA256SUMS`, publikigita post kiam ĉiuj aliaj dosieroj ĉeestas):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip ne estas publikigita kiel pako (neniu konstruita wheel): `pip install -e '.[dev]'` instalas el klono de la fonto, same kiel `./install.sh`.

---

<a id="plan"></a>

## Kion diras `acvram plan`

La planilo meritas esti lanĉita antaŭ ĉia elŝuto. Ĝi respondas la demandojn, kiuj decidas ĉu modelo estas uzebla sur ĉi tiu maŝino:

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

Ĝi esploras la spacon de konfiguroj anstataŭ konservi la unuan taŭgan, kaj du el ĝiaj decidoj sufiĉe kontraŭintuiciaj por meriti klarigon:

* **Ĝi lasas la 3080 Ti neuzata** kiam modelo enkaĝiĝas nur en la 5090. La ŝtupoj de kondukilo (pipeline) plenumiĝas sinsekve: aldoni ŝtupon je 912 Go/s en kondukilon je 1790 Go/s malrapidigas la malkodigon de sola fluo. Devigata per `--gpus all`.
* **Ĝi malgrandigas la KV-kaŝmemoron por konservi la pezojn en VRAM.** Ĉiu gibibajto donita al la kaŝmemoro estas gibibajto da pezoj forpuŝita al la PCIe-buso, kaj legi pezon per PCIe kostas ĉirkaŭ tridek fojojn tiom kiom el VRAM. Sur la 70B supre, nur ĉi tiu elekto pligrandigas de 2,3 al 17,8 ĵetonoj/s.

---

<a id="optimisations"></a>

## Rapidiĝi

Kvar optimumigoj, ĉiu kontrolita per ekvivalenteca pruvo kaj ne nur per kronometro: optimumigo, kiu ŝanĝas la respondon, estas cimo.

La NVFP4-lineoj de densaj modeloj defaŭlte trapasas la Marlin-aranĝon (+57 ĝis +90 % da trafluo je b = 8, TTFT +2 ĝis +4 ms laŭ revue/poste6-piece147-verdict-24-09.md; retropaŝo `ACVRAM_PROJ_MARLIN=0`, vidu [CHANGELOG.md](../CHANGELOG.md)).

### Spekulativa malkodigo (`--speculative`)

Malkodigi unu ĵetonon per aro de grandeco 1 estas limigita de memoro: la maŝino legas ĉiujn aktivajn pezojn por produkti nur unu ĵetonon. Kontroli K proponitajn ĵetonojn legas la samajn pezojn **nur unufoje**. Du proponantoj:

* `ngram` (defaŭlte) — serĉas la nunan sufikson pli frue en la kunteksto kaj proponas tion, kio sekvis. Kostas nenion, postulas neniun modelon. Rentinda kiam la eligo kopias la enigon: koda redaktado, RAG, resumo.
* `draft` — malgranda modelo sur dua aparato. Sur ĉi tiu ilaro, tiu aparato estas la RTX 3080 Ti, kiun la planilo intence lasas senokupa por ĉiu modelo, kiu enkaĝiĝas en la 5090.

`mtp` (kapo `nextn` de la modelo) kaj `auto` ankaŭ ekzistas; nerentindaj nuntempe kaj ne defaŭlte ŝaltitaj — vidu `docs/ARCHITECTURE.md`.

La akcepto estas preciza, ne proksimuma: propono estas akceptata kun probableco `min(1, p/q)`, kaj rifuzo reprovas en la normigita pozitiva parto de `p - q`. Mezurita sur 40 000 tirado kontraŭ intence malbone kalibrita malneto, la eligita distribuo restas ĉe 0,002 da tuta varioo de la celo — la spekulado aĉetas rapidon, neniam malsaman respondon.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefiksa kaŝmemoro (defaŭlte aktiva)

Blokoj estas adresataj per la *ĉenita* haketvaloro de sia ĵetontranĉo: du petoj, kiuj kunhavas sisteman instrukcion, kunhavas ĝiajn blokojn, kaj la dua ne plu devas antaŭkalkuli ilin. La ĉenado estas nemalhavebla: la samaj dek ses ĵetonoj en malsama kunteksto ne enhavas la samajn ŝlosilojn kaj valorojn, kaj haketi nur la tranĉon servus la kaŝmemoron de unu sekvenco al alia.

Malokupigita bloko, kies enhavo restas rekonebla, aliĝas al LRU-atendovico anstataŭ al la listo de liberaj blokoj: la kaŝmemoro tiel postvivas inter petoj, neniam rifuzante disponigon, kiun ĝi povus estinti servi.

### Kalkulo ĉe la gastiga etaĝo (`--host-exec`)

Tavolo, kies pezoj loĝas en la ĉefmemoro, povas esti kopiita al la GPU aŭ kalkulita surloke. Ambaŭ vojoj estas limigitaj de memoro kaj legas la samajn bajtojn: la plej rapida estas tiu kun la plej larĝa buso — PCIe 5.0 x16 donas ĉirkaŭ 54 Go/s, DDR5 en duobla kanalo ĉirkaŭ 70 Go/s — kaj kalkuli surloke krome liberigas la GPU-n anstataŭ igi ĝin atendi kopion.

Tio valoras nur se la procesoro legas rekte la pezojn pakigitajn en 4 bitoj. Do malgranda C++-kerno kun AVX2-vojo (`acvram_cpu.cpp`, ŝargita per ctypes, sen Python-kapoj nek ninja). Eĉ sur ĝia **skalara** retropaŝa branĉo, ĝi venkas `dequantize() @ x` per faktoro 1,44 en INT4 kaj 3,21 en NVFP4, ĉar la lasta unue skribas 32-bitan kopion de la tuta matrico.

Sur Mistral-Large-123B, la takso de la planilo pliiĝas de 1,35 al 2,42 ĵetonoj/s.

### Miksita precizeco (`--snr-floor`, defaŭlte malŝaltita)

La konvertilo mezuras la rilaton signalo/bruo ĉe la eligo de ĉiu tavolo por ĉiu tensoro kaj povas altigi al pli larĝa formato tiujn, kiuj falas sub `--snr-floor`, limigite al 15 % de la tensoroj kaj maksimuma prezo (`--promotion-cout-max`, en aldonitaj mebibajtoj).

La sojlo estas **nulo defaŭlte**: nenio estas altigata. La malkodigo estas limigita de memortrafluo, kaj la mezuro sur `Huihui-Qwen3.8-27B` decidas — sojlo de 25 dB kostas 13,4 % da memoro kaj 10,6 % da trafluo (18,50 Gio kaj 41,8 t/s kontraŭ 16,02 kaj 46,2) por 2,0 % da perplekseco (42,591 kontraŭ 43,447, korpuso de 16 383 ĵetonoj). `--snr-floor 25` restarigas la malnovan konduton kiam kvalito antaŭas rapidon.

### Kaj `acvram eval`

La rilato signalo/bruo kaj la kosinuso de la logitoj estas proksimumoj. `acvram eval REP [REP ...]` mezuras la perpleksecon per rulanta fenestro, por ke elekto de formato decidiĝu per pruvoj:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-alirpunktoj

| alirpunkto | notoj |
|---|---|
| `POST /v1/chat/completions` | SSE-fluo aŭ unuopa respondo; uzas la konversacian ŝablonon de la modelo |
| `POST /v1/completions` | instigo kiel teksto aŭ kiel ĵetonidentigiloj |
| `POST /v1/embeddings` | mezumigitaj finaj kaŝitaj statoj, L2-normigitaj, `dimensions` respektata |
| `GET /v1/models` | plus bloko `acvram`: formatoj, aparatoj, kapacito de KV-kaŝmemoro |
| `GET /health`, `GET /metrics` | malkodiga trafluo, okupateco de KV-blokoj |

La kampaj nomoj de ĉi tiuj respondoj restas en la angla: temas pri la OpenAI-protokolo, kaj traduki ilin rompus ĉiujn ekzistantajn klientojn.

---

<a id="chiffres"></a>

## De kie venas la nombroj

Ĉiu supre citita valoro estas produktita de kodo de ĉi tiu deponejo kaj kontrolita per `pytest`. Mezuroj faritaj sur procesoro kun la referencaj kernoj:

| formato | bitoj/pezo | SNR de la pezoj | kosinuso de la logitoj kontraŭ BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Du konstatoj el ĉi tiuj mezuroj ŝanĝis la defaŭltajn valorojn:

* **Hadamard-turno helpas la INT4-on kaj ne la NVFP4-on.** La grupoj de 128 de INT4 ne povas sorbi izolitan kanalan ekstremaĵon, do disvastigi la ekstremajn valorojn valoras transformon de ordo n log n po aktivigo. La blokoj de 16 de NVFP4 jam portas sian propran skalon. Do `--hadamard auto`, kiu aplikas ĝin nur al INT4.
* **INT8 venkas FP8 E4M3 por la KV-kaŝmemoro**, 44 dB kontraŭ 32 dB je sama grandeco, ĉar skalo po (ĵetono, kapo) jam provizas la dinamikan gamon, por kiu FP8 elspezas eksponentajn bitojn. La du kartoj do uzas KV-kaŝmemoron en INT8, eĉ se la 5090 povus fari FP8. Formato `k8v4` (valoroj en INT4, −22 % da kaŝmemoraj bajtoj) ekzistas kiel opcio, **nekvalifikita** — vidu `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentado

| Dokumento | Enhavo |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **daŭrigi la projekton sur alia maŝino** (france) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | kiel la pecoj kuniĝas |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | pura NVFP4 aŭ atento+GDN en int8 po kanalo, sur Gated DeltaNet-hibrido |
| [`docs/MATERIEL.md`](MATERIEL.md) | agordi ĉi tiun specifan maŝinon |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **kio ankoraŭ ne farita**, legu ĝin unue |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | laborkonvencioj pri la kodo (lingvo, stilo, kontroloj antaŭ puŝo) |

---

<a id="resultats"></a>

## Mezuritaj rezultoj (22/09/2026, RTX 5090 je 400 W, reĝimo ≥ 20 s ĉe la energimezurilo)

Qwen3-Coder-30B-A3B en NVFP4 (spertuloj) + INT8 (atento, kapo), sama protokolo por ĉiuj motoroj (`outils/`, unu karto, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| malkodigo, 12 sekvencoj | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| malkodigo, 1 sekvenco | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| antaŭplenigo pp2048 | **22 707 ĵetonoj/s** | 21 054 | 8 671 (TabbyAPI, forigita) |

¹ Erato de la 22/09: `serve` defaŭlte spekulas (`--speculative ngram`, cli.py), la konkurantoj ne; la ĝis nun publikigita 380,8 t/s estis mezurita KUN spekulado. Sen spekulado (`--speculative none`, sama ĉeno, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram estas **tria** ĉe b=1, malantaŭ llama.cpp kaj vLLM. En energio ĝi restas antaŭ llama.cpp (0,601 kontraŭ 0,700 J/ĵetono neta). Ĉe b=12 la spekulado neniam estas aktiva (gardo `lot_max=2`): tiu ĉi ĉelo jam estis je egalaj armiloj.

² 23/09, sama seanco, sama HTTP-kliento (`banc-llamacpp-16-09.py` kontraŭ `acvram serve` kaj `vllm serve`), `-lgc 2700` eksplicite metita ĉirkaŭ ĉiu brako, alternaj ĉeloj A V V A, ≥ 5 aroj po brako, devio deklarita nur preter 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 ĉe la malkodigo, malvolvita atento-redukto): devio −1,6 %, **sub 2 σ: egaleco de trafluo**. En J/ĵetono, **vLLM restas antaŭe je 7,0 %** (preter 2 σ). Kun 0.6.37 la sama protokolo donis −4,7 %.

³ Sama seanco kaj sama protokolo kiel ², sen spekulado ambaŭflanke: acvram 312,3 kontraŭ vLLM 284,8 — **acvram antaŭe je 9,7 % en trafluo** (preter 2 σ); J/ĵetono: **egaleco** (devio 0,04 %, sub 2 σ).

⁴ 23/09, sama protokolo kontraŭ llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 kun la reskribita vojigo (+5,6 %): acvram 310,8 kontraŭ llama.cpp 329,9 t/s — **llama.cpp antaŭe je 5,8 % en trafluo, acvram antaŭe je 13,4 % en J/ĵetono** (0,598 kontraŭ 0,691).

Traflutoj de la tago (posteno 1030, ŝpara reĝimo `-lgc 2700`, kondukilo en servo; avida specimenado kaptita en la CUDA-grafo, defaŭlto de 0.6.35). La b=12 de acvram estas oficiala sigelita ĉelo (mediano de 6 interplektitaj fenestroj, horloĝo po fenestro).

> **Erato (23/09/2026).** La ĝis nun publikigita komparo kun vLLM (b=12: 1 782 kontraŭ 1 634 t/s; b=1: 290,6) kontraŭmetis acvram-on mezuritan per HTTP al vLLM mezurita **eksterrete** (`LLM().generate()`), kaj la erato de la 22/09 erare asertis, ke la vLLM-ĉelo trapasas `vllm serve`-on. La 23/09: sama HTTP-kliento por ambaŭ, kaj `-lgc` metita por ambaŭ (acvram metas la sian ĉe la lanĉo, `vllm serve` ne: sen tiu antaŭzorgo vLLM funkciis je ~2 930 MHz kontraŭ ~2 650). Rezulto en noto ²: vLLM antaŭe je 9,1 % ĉe b=12.

La matenon de 14/09 acvram estis je 630 t/s kaj 0,619 J/ĵetono sur la sama ĉelo: la gajnoj venas de la denaska FP4-MMA de Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 kontraŭ la bf16), de la MoE en grupigita GEMM po sitelo de aro, de vojigo en unusola kerno (3 677 → 1 517 lanĉoj po paŝo) kaj de mallarĝa GEMM sur tensor-kernoj por la projekcioj. Ĉiu nombro havas sian noton en `acvram-memoire/revue/` kun la sigelita antaŭdiro antaŭ la mezuro, la instrumento kaj ĝia reĝimo — nombro sen reĝimo ne estas publikigata.

Kie acvram estas antaŭe: MLA-modeloj (GLM-4.7-Flash) en denaska NVFP4 sm_120, kiujn vLLM servas nur en FP8 (b=1: 165,35 t/s en servo); modeloj, kiuj ne enkaĝiĝas en VRAM. La malkodigo de sola sekvenco ne apartenas al tio: sen spekulado, acvram tie antaŭas vLLM je 9,7 % (noto ³), malantaŭas llama.cpp je 5,8 % en trafluo sed antaŭas ĝin je 13,4 % en energio (noto ⁴). Ĉe granda aro, sur MoE, kiu enkaĝiĝas en VRAM, vLLM havas egalecon de trafluo ĉe b=12 (1 995,1 kontraŭ 2 027,0 t/s, sub 2 σ, noto ²) sed konservas 7,0 % malpli da J/ĵetono; acvram tie progresis de 1 540 t/s (0.6.34) al 1 995 (0.6.38).

---

<a id="etat"></a>

## Stato

Versio 0.6.38. Ĉio funkcias sur la 5090: CUDA-kernoj kompilitaj por `sm_120a` (denaska FP4) kaj `sm_86`, CUDA-grafoj, kvantigo NVFP4/INT8/INT4, HTTP-servilo. Gardoj enloke: la karto estas nevidebla al laborsesioj (`CUDA_VISIBLE_DEVICES` malplena) kaj nur `outils/carte.sh` disponigas ĝin, sub ŝlosilo, po unu mezuro samtempe; observanto protokolas ĉiun aliron ekster la ŝlosilo; energimezuro kovranta pli ol unu karton aŭ malpli ol 10 s estas nuligata; modelo ŝargita en degradita reĝimo diras tion kaj ne eniras duelon.

4 107 testoj (`pytest --collect-only -q`, unu minuto sur procesoro; la GPU-testoj funkcias nur sub `carte.sh`). Sekvado de la laboro: `acvram-memoire/` (reguloj, listo, kajeroj, revizio de pluraj centoj da notoj).

---

<a id="credits"></a>

## Dankoj

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, sub permesilo Apache-2.0: `acvram/kernels/marlin_port/` portas ĝiajn Marlin-kernojn (MoE kaj densa), kun kompleta atribuo dosier-post-dosiere en [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, la FP4-tensor-kernoj de Blackwell (`sm_120`) kaj la bibliotekoj de kiuj dependas ĉi tiu projekto.
- **PyTorch** — tensora motoro kaj C++/CUDA-etendaĵoj.

Sendependa projekto, ne filiigita kun ASUS, NVIDIA nek kun la projekto vLLM.

---

<a id="licence"></a>

## Permesilo

[GPL-3.0 aŭ posta](../LICENSE) por la kodo de ĉi tiu deponejo. `acvram/kernels/marlin_port/` enhavas kodon portitan de [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kernoj `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), sub permesilo Apache-2.0: ĉiu dosiero konservas sian originan kapon, la permesila teksto estas en `LICENSE-vllm`, kaj la dosierlisto, la origina kompilo kaj la modifoj estas en [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Subteni la projekton

La disvolvado de acvram estas farata sur persona aparataro. Se la projekto utilas al vi:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Offrir%20un%20café&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Tradukoj: [TRADUIRE.md](TRADUIRE.md) (france; la gvidilo por kontribuoj ankoraŭ ne tradukita).
