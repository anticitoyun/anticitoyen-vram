<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-unterst%C3%BCtzen-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Ein OpenAI-API-kompatibles Inference-Gateway, das Speicher als Hierarchie behandelt, jeder GPU das Zahlenformat gibt, das ihr Silizium am besten lesen kann, und jedes Token ebenso in Joule wie in Sekunden optimiert.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · **🇩🇪 Deutsch** · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Durchsatz- und Energievergleich gegen vLLM und llama.cpp" width="720"></p>

---

## Inhalt

- [Zwei Ideen](#idees)
- [Schnellstart](#demarrage)
- [Installation](#installer)
- [Was `acvram plan` sagt](#plan)
- [Schneller werden](#optimisations)
- [HTTP-Endpunkte](#http)
- [Woher die Zahlen kommen](#chiffres)
- [Dokumentation](#documentation)
- [Gemessene Ergebnisse](#resultats)
- [Status](#etat)
- [Danksagungen](#credits)
- [Lizenz](#licence)
- [Projekt unterstützen](#soutien)

---

<a id="idees"></a>

## Zwei Ideen

Für eine bestimmte Maschine konzipiert:

| | |
|---|---|
| Prozessor | Intel Core i9-14900K (8 P-Kerne + 16 E-Kerne) |
| Mainboard | ASUS ROG Maximus Z790 Dark Hero |
| Speicher | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); beide Karten in PCIe x8/x8, gedrosselt auf 400 W / 275 W |

**Ein Format pro GPU.** Die RTX 5090 besitzt FP4-Tensor-Cores; die RTX 3080 Ti hat weder diese noch FP8. Beide auf ein gemeinsames Format zu zwingen würde die 5090 verschwenden. Der Konverter schreibt daher *dasselbe Modell zweimal*, jeweils im Format, das das Zielgerät wirklich nutzen kann:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| Gewichte | **NVFP4** — E2M1 + FP8-E4M3-Skala alle 16 | **INT4** — uint4 + fp16-Skala und -Nullpunkt alle 128 |
| Bits pro Gewicht | 4,50 | 4,16 |
| gegenüber BF16 | ×3,56 kleiner | ×3,85 kleiner |
| Rechenmodus | FP4-Tensor-Cores | im Kernel nach FP16 dequantisiert, FP16-Tensor-Cores |
| KV-Cache | INT8 | INT8 |

32 GB VRAM bei 4,5 Bit pro Gewicht fassen etwa **56 Milliarden Parameter**, gegenüber 16 Milliarden in BF16. Auf beiden Karten zusammen sind das ungefähr **78 Milliarden residente Parameter**, noch bevor der Arbeitsspeicher überhaupt berührt wird.

**Speicher ist eine Hierarchie, keine Wand.** Drei Stufen, und der Planer misst, was jede kostet, statt zu hoffen, dass das Modell hineinpasst:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Schnellstart

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Jeder OpenAI-Client kann sich anschließend verbinden:

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

Aus dem Quellcode (alle Plattformen):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Oder per Paket, eine Datei angehängt an jedes [GitHub-Release](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanal | Der Release angehängte Datei | Befehl |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (von `rpmbuild` generierte Namen, nicht fest) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (oder `rpmbuild --rebuild *.src.rpm` ausgehend von der `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Eine `.flatpakref` installiert immer die zuletzt veröffentlichte Version des Repositorys.

Vor der Installation die heruntergeladene Datei gegen die dem Release beigefügten Prüfsummen verifizieren (`SHA256SUMS`, veröffentlicht sobald alle anderen Dateien vorhanden sind):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip wird nicht als Paket veröffentlicht (kein gebautes Wheel): `pip install -e '.[dev]'` installiert aus einem Quell-Klon, wie `./install.sh`.

---

<a id="plan"></a>

## Was `acvram plan` sagt

Der Planer lohnt sich vor jedem Download. Er beantwortet die Fragen, die entscheiden, ob ein Modell auf dieser Maschine nutzbar ist:

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

Er durchsucht den Konfigurationsraum, statt die erste passende Lösung zu übernehmen, und zwei seiner Entscheidungen sind gegenintuitiv genug, um es wert zu sein, genannt zu werden:

* **Er lässt die 3080 Ti ungenutzt**, wenn ein Modell allein auf der 5090 passt. Die Stufen einer Pipeline laufen seriell: eine Stufe mit 912 GB/s in eine 1790-GB/s-Pipeline einzufügen verlangsamt das Single-Stream-Decoding. Erzwungen mit `--gpus all`.
* **Er verkleinert den KV-Cache, um die Gewichte im VRAM zu halten.** Jedes dem Cache gegebene Gigabyte ist ein Gigabyte Gewichte, das auf den PCIe-Bus verdrängt wird, und ein Gewicht über PCIe zu lesen kostet etwa das Dreißigfache dessen, was es aus dem VRAM kostet. Beim obigen 70B-Modell bringt diese eine Abwägung 2,3 auf 17,8 Token/s.

---

<a id="optimisations"></a>

## Schneller werden

Vier Optimierungen, jede durch einen Äquivalenznachweis belegt, nicht nur durch eine Stoppuhr: eine Optimierung, die die Antwort verändert, ist ein Bug.

Die NVFP4-Linearschichten dichter Modelle laufen standardmäßig über das Marlin-Layout (+57 bis +90 % Durchsatz bei b = 8, TTFT +2 bis +4 ms laut revue/poste6-piece147-verdict-24-09.md; Rückfall `ACVRAM_PROJ_MARLIN=0`, siehe [CHANGELOG.md](../CHANGELOG.md)).

### Spekulatives Decoding (`--speculative`)

Ein Token mit Losgröße 1 zu decodieren ist speichergebunden: die Maschine liest alle aktiven Gewichte, um ein einziges Token zu erzeugen. K vorgeschlagene Token zu prüfen liest dieselben Gewichte **nur einmal**. Zwei Vorschlagsgeber:

* `ngram` (Standard) — sucht das aktuelle Suffix früher im Kontext und schlägt vor, was danach kam. Kostet nichts, benötigt kein Modell. Lohnt sich, wenn die Ausgabe die Eingabe wiederholt: Code-Bearbeitung, RAG, Zusammenfassung.
* `draft` — ein kleines Modell auf einem zweiten Gerät. Auf diesem Rig ist dieses Gerät die RTX 3080 Ti, die der Planer für jedes auf der 5090 passende Modell bewusst untätig lässt.

`mtp` (`nextn`-Kopf des Modells) und `auto` existieren ebenfalls; derzeit nicht rentabel und nicht standardmäßig aktiviert — siehe `docs/ARCHITECTURE.md`.

Die Annahme ist exakt, nicht approximiert: ein Vorschlag wird mit Wahrscheinlichkeit `min(1, p/q)` angenommen, und eine Ablehnung samplet erneut aus dem normalisierten positiven Anteil von `p - q`. Gemessen über 40 000 Ziehungen gegen einen absichtlich schlecht kalibrierten Entwurf bleibt die ausgegebene Verteilung bei 0,002 totaler Variation zum Ziel — Spekulation kauft Geschwindigkeit, niemals eine andere Antwort.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Präfix-Cache (standardmäßig aktiv)

Blöcke werden über den *verketteten* Hash ihres Token-Abschnitts adressiert: zwei Anfragen, die eine System-Vorgabe teilen, teilen sich deren Blöcke, und die zweite muss sie nicht mehr neu berechnen. Die Verkettung ist unerlässlich: dieselben sechzehn Token in einem anderen Kontext enthalten nicht dieselben Schlüssel und Werte, und nur den Abschnitt zu hashen würde den Cache einer Sequenz für eine andere bedienen.

Ein freigegebener Block, dessen Inhalt weiterhin identifizierbar ist, wandert in eine LRU-Warteschlange statt in die Liste freier Blöcke: so überlebt der Cache zwischen Anfragen, ohne je eine Zuweisung abzulehnen, die er hätte bedienen können.

### Host-Stufen-Berechnung (`--host-exec`)

Eine Schicht, deren Gewichte im RAM liegen, kann zur GPU kopiert oder an Ort und Stelle berechnet werden. Beide Wege sind speichergebunden und lesen dieselben Bytes: der schnellere ist der mit dem breiteren Bus — PCIe 5.0 x16 liefert etwa 54 GB/s, DDR5 im Dual-Channel etwa 70 GB/s — und die Berechnung vor Ort lässt die GPU zudem frei, statt sie auf eine Kopie warten zu lassen.

Das lohnt sich nur, wenn der Prozessor die auf 4 Bit gepackten Gewichte direkt lesen kann. Daher ein kleiner C++-Kernel mit AVX2-Pfad (`acvram_cpu.cpp`, über ctypes geladen, ohne Python-Header oder ninja). Selbst auf seinem **skalaren** Rückfallzweig schlägt er `dequantize() @ x` um den Faktor 1,44 bei INT4 und 3,21 bei NVFP4, weil letzteres zuerst eine vollständige 32-Bit-Kopie der ganzen Matrix schreibt.

Bei Mistral-Large-123B steigt die Schätzung des Planers von 1,35 auf 2,42 Token/s.

### Gemischte Präzision (`--snr-floor`, standardmäßig aus)

Der Konverter misst das Signal-Rausch-Verhältnis am Schichtausgang für jeden Tensor und kann jene, die unter `--snr-floor` fallen, auf ein breiteres Format anheben, begrenzt auf 15 % der Tensoren und einen Kostendeckel (`--promotion-cout-max`, in zusätzlichen MiB).

Die Untergrenze liegt standardmäßig bei **null**: nichts wird angehoben. Das Decoding ist speicherbandbreitengebunden, und die Messung an `Huihui-Qwen3.8-27B` entscheidet — eine Grenze von 25 dB kostet 13,4 % Speicher und 10,6 % Durchsatz (18,50 GiB und 41,8 T/s gegenüber 16,02 und 46,2) für 2,0 % Perplexität (42,591 gegenüber 43,447, Korpus von 16 383 Token). `--snr-floor 25` stellt das alte Verhalten wieder her, wenn Qualität wichtiger ist als Geschwindigkeit.

### Und `acvram eval`

Signal-Rausch-Verhältnis und Logit-Kosinus sind Näherungen. `acvram eval REP [REP ...]` misst die Perplexität per Sliding Window, damit sich eine Formatwahl anhand von Belegen entscheidet:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-Endpunkte

| Endpunkt | Anmerkungen |
|---|---|
| `POST /v1/chat/completions` | SSE-Stream oder Einzelantwort; nutzt das Chat-Template des Modells |
| `POST /v1/completions` | Prompt als Text oder Token-IDs |
| `POST /v1/embeddings` | gemittelte finale versteckte Zustände, L2-normalisiert, `dimensions` beachtet |
| `GET /v1/models` | plus ein `acvram`-Block: Formate, Geräte, KV-Cache-Kapazität |
| `GET /health`, `GET /metrics` | Decoding-Durchsatz, Belegung der KV-Blöcke |

Die Feldnamen dieser Antworten bleiben Englisch: das ist das OpenAI-Protokoll, und sie zu übersetzen würde alle bestehenden Clients brechen.

---

<a id="chiffres"></a>

## Woher die Zahlen kommen

Jeder oben genannte Wert wird von Code dieses Repositorys erzeugt und durch `pytest` verifiziert. Messungen auf dem Prozessor mit Referenz-Kernels:

| Format | Bit/Gewicht | SNR der Gewichte | Logit-Kosinus vs. BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Zwei Erkenntnisse aus diesen Messungen haben die Standardwerte verändert:

* **Eine Hadamard-Rotation hilft INT4, nicht NVFP4.** Die 128er-Gruppen von INT4 können einen isolierten Ausreißer-Kanal nicht absorbieren, sodass das Verteilen der Extremwerte eine n-log-n-Transformation pro Aktivierung wert ist. Die 16er-Blöcke von NVFP4 tragen bereits ihre eigene Skala. Daher `--hadamard auto`, das sie nur auf INT4 anwendet.
* **INT8 schlägt FP8 E4M3 für den KV-Cache**, 44 dB gegenüber 32 dB bei gleicher Größe, weil eine Skala pro (Token, Kopf) bereits den Dynamikbereich liefert, für den FP8 Exponentenbits ausgibt. Beide Karten nutzen daher einen INT8-KV-Cache, obwohl die 5090 auch FP8 könnte. Ein `k8v4`-Format (Werte in INT4, −22 % Cache-Bytes) existiert als Option, **nicht qualifiziert** — siehe `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **das Projekt auf einer anderen Maschine fortsetzen** (Französisch) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | wie die Teile zusammenpassen |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | reines NVFP4 oder Attention+GDN in Int8 pro Kanal, auf einem Gated-DeltaNet-Hybrid |
| [`docs/MATERIEL.md`](MATERIEL.md) | diese spezifische Maschine einstellen |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **was noch nicht fertig ist**, zuerst lesen |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | Arbeitskonventionen für den Code (Sprache, Stil, Prüfungen vor dem Push) |

---

<a id="resultats"></a>

## Gemessene Ergebnisse (22.09.2026, RTX 5090 bei 400 W, Fenster ≥ 20 s am Energiemesser)

Qwen3-Coder-30B-A3B in NVFP4 (Experten) + INT8 (Attention, Kopf), gleiches Protokoll für alle Engines (`outils/`, eine Karte, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| Decoding, 12 Sequenzen | 1 995,1 T/s ² | 2 027,0 T/s ² | — |
| Decoding, 1 Sequenz | 312,3 T/s ³ ⁴ | 284,8 T/s ³ | **329,9 T/s** ⁴ |
| Prefill pp2048 | **22 707 Token/s** | 21 054 | 8 671 (TabbyAPI, entfernt) |

¹ Erratum vom 22.09: `serve` spekuliert standardmäßig (`--speculative ngram`, cli.py), die Konkurrenten nicht; die bisher veröffentlichten 380,8 T/s wurden MIT Spekulation gemessen. Ohne Spekulation (`--speculative none`, gleiche Kette, revue/poste2-piece44-speculation-none-22-09.md): 283,6 T/s — acvram ist bei b=1 **Dritter**, hinter llama.cpp und vLLM. Bei der Energie liegt es weiterhin vor llama.cpp (0,601 gegenüber 0,700 J/Token netto). Bei b=12 ist Spekulation nie aktiv (Schutz `lot_max=2`): diese Zelle war bereits mit gleichen Waffen.

² 23.09, gleiche Sitzung, gleicher HTTP-Client (`banc-llamacpp-16-09.py` gegen `acvram serve` und `vllm serve`), `-lgc 2700` explizit für jeden Arm gesetzt, alternierende Zellen A V V A, ≥ 5 Läufe pro Arm, Abweichung erst über 2 σ deklariert (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 beim Decoding, entrollte Attention-Reduktion): Abweichung −1,6 %, **unter 2 σ: Durchsatz-Gleichstand**. Bei J/Token bleibt **vLLM um 7,0 % vorn** (über 2 σ). Mit 0.6.37 ergab dasselbe Protokoll −4,7 %.

³ Gleiche Sitzung und gleiches Protokoll wie ², ohne Spekulation auf beiden Seiten: acvram 312,3 gegenüber vLLM 284,8 — **acvram um 9,7 % im Durchsatz vorn** (über 2 σ); J/Token: **Gleichstand** (Abweichung 0,04 %, unter 2 σ).

⁴ 23.09, gleiches Protokoll gegen llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 mit neu geschriebenem Routing (+5,6 %): acvram 310,8 gegenüber llama.cpp 329,9 T/s — **llama.cpp um 5,8 % im Durchsatz vorn, acvram um 13,4 % bei J/Token vorn** (0,598 gegenüber 0,691).

Tagesdurchsätze (Rig 1030, Eco-Modus `-lgc 2700`, Pipeline im Dienst; gieriges Sampling im CUDA-Graph erfasst, Standard seit 0.6.35). Das b=12 von acvram ist eine versiegelte offizielle Zelle (Median aus 6 verschachtelten Fenstern, Takt pro Fenster).

> **Erratum (23.09.2026).** Der bisher veröffentlichte vLLM-Vergleich (b=12: 1 782 gegenüber 1 634 T/s; b=1: 290,6) stellte acvram, gemessen per HTTP, gegen vLLM, gemessen **offline** (`LLM().generate()`), gegenüber, und das Erratum vom 22.09 behauptete fälschlich, die vLLM-Zelle liefe über `vllm serve`. Am 23.09: gleicher HTTP-Client für beide, und `-lgc` für beide gesetzt (acvram setzt seinen beim Start, `vllm serve` nicht: ohne diese Vorsichtsmaßnahme lief vLLM mit ~2 930 MHz gegenüber ~2 650). Ergebnis in Anmerkung ²: vLLM bei b=12 um 9,1 % vorn.

Am Morgen des 14.09 lag acvram bei 630 T/s und 0,619 J/Token auf derselben Zelle: die Gewinne kommen von der nativen FP4-MMA von Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 gegenüber BF16), dem MoE als gruppiertem GEMM pro Losbucket, einem Routing in einem einzigen Kernel (3 677 → 1 517 Aufrufe pro Schritt) und einem schmalen GEMM auf Tensor-Cores für die Projektionen. Jede Zahl hat ihre Notiz in `acvram-memoire/revue/` mit der vor der Messung versiegelten Vorhersage, dem Instrument und seinem Regime — eine Zahl ohne Regime wird nicht veröffentlicht.

Wo acvram vorn liegt: MLA-Modelle (GLM-4.7-Flash) in nativem NVFP4 auf sm_120, das vLLM nur in FP8 bedient (b=1: 165,35 T/s im Dienst); Modelle, die nicht in den VRAM passen. Das Decoding einer einzelnen Sequenz gehört nicht dazu: ohne Spekulation liegt acvram dort um 9,7 % vor vLLM (Anmerkung ³), 5,8 % hinter llama.cpp im Durchsatz, aber 13,4 % vor ihm bei der Energie (Anmerkung ⁴). Bei großen Losen, auf einem MoE, das in den VRAM passt, liegt vLLM bei b=12 im Durchsatz gleich (1 995,1 gegenüber 2 027,0 T/s, unter 2 σ, Anmerkung ²), behält aber 7,0 % weniger J/Token; acvram ist dort von 1 540 T/s (0.6.34) auf 1 995 (0.6.38) gestiegen.

---

<a id="etat"></a>

## Status

Version 0.6.38. Alles läuft auf der 5090: CUDA-Kernel kompiliert für `sm_120a` (natives FP4) und `sm_86`, CUDA-Graphen, NVFP4/INT8/INT4-Quantisierung, HTTP-Server. Schutzmechanismen vorhanden: die Karte ist für Arbeitssitzungen unsichtbar (`CUDA_VISIBLE_DEVICES` leer) und nur `outils/carte.sh` gibt sie, unter Verriegelung, jeweils einer Messung frei; ein Wächter protokolliert jeden Zugriff außerhalb der Verriegelung; eine Energiemessung über mehr als eine Karte oder unter 10 s wird verworfen; ein im degradierten Modus geladenes Modell sagt es und tritt in kein Duell ein.

4 107 Tests (`pytest --collect-only -q`, eine Minute auf dem Prozessor; die GPU-Tests laufen nur unter `carte.sh`). Verlauf der Arbeit: `acvram-memoire/` (Regeln, Verzeichnis, Notizbücher, Durchsicht mehrerer hundert Notizen).

---

<a id="credits"></a>

## Danksagungen

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, unter Apache-2.0-Lizenz: `acvram/kernels/marlin_port/` portiert dessen Marlin-Kernel (MoE und dicht), mit vollständiger dateiweiser Zuschreibung in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, die FP4-Tensor-Cores von Blackwell (`sm_120`) und die Bibliotheken, von denen dieses Projekt abhängt.
- **PyTorch** — Tensor-Engine und C++/CUDA-Erweiterungen.

Unabhängiges Projekt, nicht mit ASUS, NVIDIA oder dem vLLM-Projekt verbunden.

---

<a id="licence"></a>

## Lizenz

[GPL-3.0-or-later](../LICENSE) für den Code dieses Repositorys. `acvram/kernels/marlin_port/` enthält portierten Code von [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (Kernel `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), unter Apache-2.0-Lizenz: jede Datei behält ihren Original-Header, der Lizenztext steht in `LICENSE-vllm`, und die Dateiliste, der Ursprungscommit und die Änderungen stehen in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Projekt unterstützen

acvram wird auf privater Hardware entwickelt. Wenn Ihnen das Projekt nützlich ist:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Kaffee%20spendieren&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Übersetzungen: [TRADUIRE.md](TRADUIRE.md) (Französisch; der Beitragsleitfaden des Projekts ist noch nicht übersetzt).
