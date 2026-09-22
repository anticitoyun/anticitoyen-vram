<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Unterstützen: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Ein zur OpenAI-API kompatibles Inferenz-Gateway, das den Speicher als
Hierarchie behandelt und jeder GPU das Zahlenformat gibt, das ihr Silizium am
besten liest.

Entworfen für eine ganz bestimmte Maschine:

| | |
|---|---|
| Prozessor | Intel Core i9-14900K (8 P-Kerne + 16 E-Kerne) |
| Hauptplatine | ASUS ROG Maximus Z790 Dark Hero |
| Arbeitsspeicher | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); beide Karten an PCIe x8/x8, gedrosselt auf 400 W / 275 W |

## Die zwei Ideen

**Ein Format pro GPU.** Die RTX 5090 besitzt FP4-Tensor-Cores; die RTX 3080 Ti
hat keine, und auch kein FP8. Beide auf ein gemeinsames Format zu bringen,
würde die 5090 verschwenden. Der Konverter schreibt daher *dasselbe Modell
zweimal*, in dem Format, das jedes Ziel tatsächlich ausnutzen kann:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| Gewichte | **NVFP4** — E2M1 + FP8-E4M3-Skala alle 16 | **INT4** — uint4 + fp16-Skala und -Nullpunkt alle 128 |
| Bits pro Gewicht | 4,50 | 4,16 |
| gegenüber BF16 | ×3,56 kleiner | ×3,85 kleiner |
| Rechenmodus | FP4-Tensor-Cores | im Kernel nach FP16 dequantisiert, FP16-Tensor-Cores |
| KV-Cache | INT8 | INT8 |

32 GB VRAM bei 4,5 Bit pro Gewicht fassen etwa **56 Milliarden Parameter**,
gegenüber 16 Milliarden in BF16. Über beide Karten sind das ungefähr
**78 Milliarden residente Parameter**, noch bevor der Arbeitsspeicher
angerührt wird.

**Der Speicher ist eine Hierarchie, keine Mauer.** Drei Stufen, und der Planer
misst, was jede kostet, statt zu hoffen, dass das Modell passt:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Schnellstart

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Jeder OpenAI-Client verbindet sich anschließend damit:

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

## Was `acvram plan` sagt

Der Planer lohnt sich vor jedem Download. Er beantwortet die Fragen, die
entscheiden, ob ein Modell auf dieser Maschine brauchbar ist:

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

Er durchsucht den Konfigurationsraum, statt die erste passende Lösung zu
nehmen, und zwei seiner Entscheidungen sind kontraintuitiv genug, um sie
auszusprechen:

* **Er lässt die 3080 Ti ungenutzt**, wenn ein Modell allein auf die 5090
  passt. Die Abschnitte einer Pipeline laufen seriell: eine Stufe mit 912 GB/s
  in eine Pipeline mit 1790 GB/s einzufügen, bremst das Einzelstrom-Decoding.
  Erzwingen lässt es sich mit `--gpus all`.
* **Er verkleinert den KV-Cache, um die Gewichte im VRAM zu halten.** Jedes
  Gigabyte für den Cache ist ein Gigabyte Gewichte, das auf den PCIe-Bus
  verdrängt wird, und ein Gewicht über PCIe zu lesen kostet etwa dreißigmal so
  viel wie aus dem VRAM. Beim 70B oben bringt allein diese Abwägung 2,3 → 17,8
  Token/s.

## Schnell sein

Vier Optimierungen, jede durch einen Äquivalenzbeweis geprüft und nicht nur
mit der Stoppuhr: eine Optimierung, die die Antwort ändert, ist ein Fehler.

### Spekulatives Decoding (`--speculative`)

Ein Token mit Batchgröße 1 zu decodieren ist speicherlimitiert: die Maschine
liest alle aktiven Gewichte, um ein einziges Token zu erzeugen. K
vorgeschlagene Token zu prüfen liest dieselben Gewichte **ein einziges Mal**.
Zwei Vorschlagsgeber:

* `ngram` (Standard) — sucht das aktuelle Suffix weiter vorn im Kontext und
  schlägt vor, was darauf folgte. Kostet nichts, braucht kein Modell. Lohnt
  sich, wenn die Ausgabe die Eingabe abschreibt: Code-Bearbeitung, RAG,
  Zusammenfassung.
* `draft` — ein kleines Modell auf einem zweiten Gerät. Auf diesem Rig ist das
  die RTX 3080 Ti, die der Planer für jedes auf die 5090 passende Modell
  absichtlich untätig lässt.

Die Annahme ist exakt, nicht näherungsweise: ein Vorschlag wird mit
Wahrscheinlichkeit `min(1, p/q)` angenommen, und eine Ablehnung zieht neu aus
dem normierten positiven Teil von `p - q`. Gemessen über 40 000 Ziehungen gegen
einen absichtlich schlecht kalibrierten Entwurf bleibt die ausgegebene
Verteilung innerhalb von 0,002 Totalvariation des Ziels — Spekulation kauft
Geschwindigkeit, nie eine andere Antwort.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Präfix-Cache (standardmäßig aktiv)

Blöcke werden über den *verketteten* Hash ihres Token-Abschnitts adressiert:
zwei Anfragen mit derselben Systemanweisung teilen sich deren Blöcke, und die
zweite muss sie nicht mehr vorberechnen. Die Verkettung ist unverzichtbar:
dieselben sechzehn Token in einem anderen Kontext enthalten nicht dieselben
Schlüssel und Werte, und nur den Abschnitt zu hashen würde den Cache einer
Sequenz an eine andere ausliefern.

Ein freigegebener Block, dessen Inhalt identifizierbar bleibt, wandert in eine
LRU-Warteschlange statt in die Freiliste: so überlebt der Cache zwischen
Anfragen, ohne je eine Zuteilung abzulehnen, die er hätte bedienen können.

### Rechnen auf der Host-Stufe (`--host-exec`)

Eine Schicht, deren Gewichte im RAM liegen, kann zur GPU kopiert oder an Ort
und Stelle berechnet werden. Beide Wege sind speicherlimitiert und lesen
dieselben Bytes: schneller ist der mit dem breiteren Bus — PCIe 5.0 x16
liefert etwa 54 GB/s, DDR5 im Dual-Channel etwa 70 GB/s — und das Rechnen vor
Ort lässt die GPU zudem frei, statt sie auf eine Kopie warten zu lassen.

Das lohnt nur, wenn der Prozessor die 4-Bit-gepackten Gewichte direkt liest.
Daher ein kleiner C++-Kernel mit AVX2-Pfad (`acvram_cpu.cpp`, per ctypes
geladen, ohne Python-Header und ohne ninja). Selbst auf seinem **skalaren**
Rückfallzweig schlägt er `dequantize() @ x` um den Faktor 1,44 in INT4 und
3,21 in NVFP4, weil letzteres zuerst eine 32-Bit-Kopie der ganzen Matrix
schreibt.

Bei Mistral-Large-123B steigt die Schätzung des Planers von 1,35 auf
2,42 Token/s.

### Gemischte Genauigkeit (`--snr-floor`, standardmäßig aus)

Der Konverter misst am Ausgang jeder Schicht den Signal-Rausch-Abstand jedes
Tensors und kann diejenigen, die unter `--snr-floor` fallen, in ein breiteres
Format befördern, begrenzt auf 15 % der Tensoren und eine Preisobergrenze
(`--promotion-cout-max`, in hinzugefügten Mebibyte).

Die Schwelle ist **standardmäßig null**: nichts wird befördert. Das Decoding
ist durch die Speicherbandbreite begrenzt, und die Messung an
`Huihui-Qwen3.8-27B` entscheidet — eine Schwelle von 25 dB kostet 13,4 %
Speicher und 10,6 % Durchsatz (18,50 GiB und 41,8 t/s gegenüber 16,02 und
46,2) für 2,0 % Perplexität (42,591 gegenüber 43,447, Korpus von 16 383
Token). `--snr-floor 25` stellt das alte Verhalten wieder her, wenn Qualität
vor Geschwindigkeit geht.

### Und `acvram eval`

Signal-Rausch-Abstand und Logit-Kosinus sind Näherungen.
`acvram eval VERZ [VERZ ...]` misst die Perplexität im gleitenden Fenster,
damit eine Formatwahl auf Belegen beruht:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-Endpunkte

| Endpunkt | Anmerkungen |
|---|---|
| `POST /v1/chat/completions` | SSE-Strom oder Einzelantwort; nutzt die Chat-Vorlage des Modells |
| `POST /v1/completions` | Prompt als Text oder als Token-IDs |
| `POST /v1/embeddings` | gemittelte letzte verborgene Zustände, L2-normiert, `dimensions` beachtet |
| `GET /v1/models` | plus ein Block `acvram`: Formate, Geräte, Kapazität des KV-Caches |
| `GET /health`, `GET /metrics` | Decoding-Durchsatz, Belegung der KV-Blöcke |

Die Feldnamen dieser Antworten bleiben englisch: es ist das OpenAI-Protokoll,
und sie zu übersetzen würde jeden bestehenden Client brechen.

## Woher die Zahlen kommen

Jeder oben genannte Wert wird von Code dieses Repositories erzeugt und mit
`pytest` geprüft. Messungen auf dem Prozessor mit den Referenz-Kernels:

| Format | Bits/Gewicht | SNR der Gewichte | Logit-Kosinus vs. BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Zwei Befunde aus diesen Messungen haben die Standardwerte geändert:

* **Eine Hadamard-Rotation hilft INT4 und nicht NVFP4.** Die 128er-Gruppen
  von INT4 können einen einzelnen Ausreißerkanal nicht auffangen, sodass das
  Verteilen der Extremwerte eine n-log-n-Transformation pro Aktivierung wert
  ist. Die 16er-Blöcke von NVFP4 tragen bereits ihre eigene Skala. Daher
  `--hadamard auto`, das sie nur auf INT4 anwendet.
* **INT8 schlägt FP8 E4M3 beim KV-Cache**, 44 dB gegenüber 32 dB bei
  gleicher Größe, weil eine Skala pro (Token, Kopf) bereits den Dynamikbereich
  liefert, für den FP8 Exponentenbits ausgibt. Beide Karten nutzen daher einen
  INT8-KV-Cache, obwohl die 5090 FP8 könnte.

## Dokumentation

* [`REPRISE.md`](../REPRISE.md) — **das Projekt auf einer anderen Maschine wieder aufnehmen**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — wie die Teile zusammenpassen
* [`docs/MATERIEL.md`](MATERIEL.md) — diese konkrete Maschine einstellen
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **was nicht erledigt ist**, zuerst lesen
* [`CONVENTIONS.md`](../CONVENTIONS.md) — Arbeitskonventionen für den Code (Sprache, Stil, Prüfungen vor dem Push)

## Gemessene Ergebnisse (22.09.2026, RTX 5090 bei 400 W, Regime ≥ 20 s am Energiezähler)

Qwen3-Coder-30B-A3B in NVFP4 (Experten) + INT8 (Attention, Kopf), gleiches
Protokoll für alle Engines (`outils/`, eine Karte, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| Decoding, 12 Sequenzen | **1 634 t/s** | 1 782 t/s | — |
| Decoding, 1 Sequenz | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| Prefill pp2048 | **22 707 Token/s** | 21 054 | 8 671 (TabbyAPI, zurückgezogen) |

Durchsatz des Tages (Station 1030, Eco-Regime `-lgc 2700`, Pipeline im Dienst;
gieriges Sampling im CUDA-Graphen erfasst, in 0.6.35 standardmäßig aktiviert).
Das b=12 ist die offizielle versiegelte Zelle (Median aus 6 verschachtelten
Fenstern).

> **Erratum (22.09.2026).** Die erste Veröffentlichung von 0.6.35 leitete
> „+1,84 % vor vLLM“ aus einer vLLM-Referenz von 1 596 t/s vom 21.09. ab, die
> aus einer **Offline-Generierung** (`LLM().generate()`) stammte, **nicht mit
> einem Server vergleichbar**: kein kontinuierliches Scheduling, nicht der Pfad
> von `acvram serve`. Am 22.09. mit einer alternierenden A/V-Zelle (A1 V1 A2 V2
> A3 V3) gegen **`vllm serve`** (HTTP) korrigiert, dieselbe Karte und derselbe
> Pfad wie `acvram serve`: vLLM Median **1 782 t/s**. Bei vergleichbarer Messung
> liegt **acvram (1 634 t/s) rund 8 % HINTER vLLM bei b=12**, nicht davor. Das
> J/Token bei gleicher Uhr wird noch neu gemessen.

Am Morgen des 14.09. lag acvram in derselben Zelle bei 630 t/s und
0,619 J/Token: die Gewinne stammen von Blackwells nativer FP4-MMA
(`mma.sync … kind::mxf4nvf4`, ×7,9 gegenüber bf16), vom MoE als gruppierter
GEMM pro Batch-Eimer, von einem Routing in einem einzigen Kernel (3 677 → 1 517
Starts pro Schritt) und von einer schmalen Tensor-Core-GEMM für die
Projektionen. Jede Zahl hat ihre Notiz in `acvram-memoire/revue/` mit der vor
der Messung versiegelten Vorhersage, dem Instrument und seinem Regime — eine
Zahl ohne Regime wird nicht veröffentlicht.

Wo acvram vorn liegt: MLA-Modelle (GLM-4.7-Flash) in nativem sm_120-NVFP4,
die vLLM nur in FP8 bedient (b=1: 165,35 t/s im Dienst); Modelle, die nicht
in den VRAM passen; und das Decoding mit Einzelsequenz (b=1: 380,8 t/s gegenüber
290,6 bei vLLM). Bei großem Batch dagegen, auf einem MoE, das in den VRAM passt,
bleibt vLLM bei b=12 vorn (1 782 gegenüber 1 634 t/s, siehe Erratum); acvram hat
sich hier verbessert (1 540 in 0.6.34 → 1 634), ohne vorbeizuziehen. Der
Energieabstand ist neu zu messen.

## Stand

Version 0.6.35. Alles läuft auf der 5090: für `sm_120a` (natives FP4) und
`sm_86` kompilierte CUDA-Kernel, CUDA-Graphen, NVFP4/INT8/INT4-Quantisierung,
HTTP-Server. Leitplanken vorhanden: die Karte ist für Arbeitssitzungen
unsichtbar (`CUDA_VISIBLE_DEVICES` leer), und nur `outils/carte.sh` leiht sie
unter Sperre jeweils einer Messung; ein Wächter protokolliert jeden Zugriff
außerhalb der Sperre; eine Energiemessung über mehr als eine Karte oder unter
10 s wird verworfen; ein im degradierten Regime geladenes Modell sagt es und
tritt in kein Duell.

640 Tests (`pytest -q`, eine Minute auf dem Prozessor; GPU-Tests laufen nur
unter `carte.sh`). Arbeitsverfolgung: `acvram-memoire/` (Regeln,
Verzeichnis, Hefte, Review mit 180 Notizen).

## Unterstützen

acvram wird auf privater Hardware entwickelt. Wenn Ihnen das Projekt nützt:
**Unterstützen: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Lizenz

GPL-3.0 oder später.
