<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-suport-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Una passarel·la d'inferència compatible amb l'API d'OpenAI, que tracta la memòria com una jerarquia, dona a cada GPU el format numèric que el seu silici sap llegir millor, i optimitza cada testimoni en joules tant com en segons.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · **🇪🇸 Català** · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Comparativa de rendiment i energia contra vLLM i llama.cpp" width="720"></p>

---

## Sumari

- [Les dues idees](#idees)
- [Inici ràpid](#demarrage)
- [Instal·lació](#installer)
- [Què diu `acvram plan`](#plan)
- [Anar ràpid](#optimisations)
- [Punts d'entrada HTTP](#http)
- [D'on surten les xifres](#chiffres)
- [Documentació](#documentation)
- [Resultats mesurats](#resultats)
- [Estat](#etat)
- [Crèdits](#credits)
- [Llicència](#licence)
- [Donar suport al projecte](#soutien)

---

<a id="idees"></a>

## Les dues idees

Dissenyada per a una màquina precisa:

| | |
|---|---|
| Processador | Intel Core i9-14900K (8 nuclis P + 16 nuclis E) |
| Placa base | ASUS ROG Maximus Z790 Dark Hero |
| Memòria | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13); les dues targetes en PCIe x8/x8, limitades a 400 W / 275 W |

**Un format per GPU.** La RTX 5090 té tensor cores FP4; la RTX 3080 Ti no en té, ni tampoc FP8. Alinear-les totes dues en un format comú faria malbé la 5090. El convertidor escriu doncs *dues vegades el mateix model*, en el format que cada destinació sap explotar realment:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesos | **NVFP4** — E2M1 + escala FP8 E4M3 cada 16 | **INT4** — uint4 + escala i zero fp16 cada 128 |
| bits per pes | 4,50 | 4,16 |
| enfront del BF16 | ×3,56 més petit | ×3,85 més petit |
| mode de càlcul | tensor cores FP4 | desquantificat a FP16 al nucli, tensor cores FP16 |
| memòria cau KV | INT8 | INT8 |

32 Go de VRAM a 4,5 bits per pes contenen aproximadament **56 mil milions de paràmetres**, contra 16 mil milions en BF16. A les dues targetes, això fa aproximadament **78 mil milions de paràmetres residents** abans fins i tot de tocar la memòria RAM.

**La memòria és una jerarquia, no un mur.** Tres nivells, i el planificador mesura què costa cadascun en lloc d'esperar que el model hi càpiga:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Inici ràpid

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Qualsevol client d'OpenAI s'hi pot connectar després:

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

## Instal·lació

Des del codi font (totes les plataformes):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

O per paquet, un fitxer adjunt a cada [versió de GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Canal | Fitxer adjunt a la versió | Ordre |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (noms generats per `rpmbuild`, no fixos) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (o `rpmbuild --rebuild *.src.rpm` des del `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpak` | `flatpak install acvram-<version>.flatpak` |

Pip no es publica com a paquet (no hi ha wheel construïda): `pip install -e '.[dev]'` instal·la des d'un clon del codi font, igual que `./install.sh`.

---

<a id="plan"></a>

## Què diu `acvram plan`

El planificador es mereix ser executat abans de qualsevol descàrrega. Respon a les preguntes que decideixen si un model és utilitzable en aquesta màquina:

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

Explora l'espai de configuracions en lloc de quedar-se amb la primera que hi càpiga, i dues de les seves decisions són prou contraintuïtives com per merèixer ser explicades:

* **Deixa la 3080 Ti sense utilitzar** quan un model càpiga només amb la 5090. Els nivells d'un pipeline s'executen en sèrie: afegir una etapa a 912 Go/s en un pipeline a 1790 Go/s alenteix la decodificació d'un sol flux. Es força amb `--gpus all`.
* **Redueix la memòria cau KV per mantenir els pesos a la VRAM.** Cada gigabyte donat a la memòria cau és un gigabyte de pesos empès cap al bus PCIe, i llegir un pes pel PCIe costa aproximadament trenta vegades el que costa des de la VRAM. Sobre el 70B anterior, aquest sol compromís fa passar de 2,3 a 17,8 testimonis/s.

---

<a id="optimisations"></a>

## Anar ràpid

Quatre optimitzacions, cada una verificada per una prova d'equivalència i no només per un cronòmetre: una optimització que canvia la resposta és un error.

Els lineals NVFP4 dels models densos passen per defecte per la disposició Marlin (+57 a +90 % de rendiment a b = 8, TTFT +2 a +4 ms segons revue/poste6-piece147-verdict-24-09.md; recurs `ACVRAM_PROJ_MARLIN=0`, vegeu [CHANGELOG.md](../CHANGELOG.md)).

### Decodificació especulativa (`--speculative`)

Decodificar un testimoni amb un lot de mida 1 està limitat per la memòria: la màquina llegeix tots els pesos actius per produir un sol testimoni. Verificar K testimonis proposats llegeix aquests mateixos pesos **una sola vegada**. Dos proposadors:

* `ngram` (per defecte) — cerca el sufix actual més enrere en el context i proposa el que el seguia. No costa res, no requereix cap model. Rendible quan la sortida copia l'entrada: edició de codi, RAG, resum.
* `draft` — un petit model en un segon dispositiu. En aquest muntatge, aquest dispositiu és la RTX 3080 Ti, que el planificador deixa voluntàriament inactiva per a qualsevol model que càpiga a la 5090.

`mtp` (capçal `nextn` del model) i `auto` també existeixen; no rendibles tal com estan i no activats per defecte — vegeu `docs/ARCHITECTURE.md`.

L'acceptació és exacta, no aproximada: una proposta s'accepta amb la probabilitat `min(1, p/q)` i un rebuig remostreja en la part positiva normalitzada de `p - q`. Mesurat sobre 40 000 tirades davant d'un esborrany voluntàriament mal calibrat, la distribució emesa es manté a 0,002 de variació total de l'objectiu — l'especulació compra velocitat, mai una resposta diferent.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Memòria cau de prefix (activa per defecte)

Els blocs s'adrecen pel hash *encadenat* del seu tram de testimonis: dues sol·licituds que comparteixen una consigna de sistema comparteixen els seus blocs, i la segona ja no els ha de precalcular. L'encadenament és indispensable: els mateixos setze testimonis en un context diferent no contenen les mateixes claus i valors, i fer hash només del tram serviria la memòria cau d'una seqüència a una altra.

Un bloc alliberat el contingut del qual segueix sent identificable s'incorpora a una cua LRU en lloc de la llista de blocs lliures: la memòria cau sobreviu així entre sol·licituds sense mai refusar una assignació que hauria pogut servir.

### Càlcul del nivell hoste (`--host-exec`)

Una capa amb els pesos residint a la RAM es pot copiar a la GPU o calcular in situ. Els dos camins estan limitats per la memòria i llegeixen els mateixos bytes: el més ràpid és el del bus més ample — el PCIe 5.0 x16 dona aproximadament 54 Go/s, la DDR5 en doble canal aproximadament 70 Go/s — i calcular in situ deixa a més la GPU lliure en lloc de fer-la esperar una còpia.

Això només val si el processador llegeix directament els pesos empaquetats en 4 bits. D'aquí un petit nucli C++ amb un camí AVX2 (`acvram_cpu.cpp`, carregat per ctypes, sense capçaleres de Python ni ninja). Fins i tot en la seva branca **escalar** de recurs, supera `dequantize() @ x` per un factor de 1,44 en INT4 i 3,21 en NVFP4, perquè aquest últim escriu primer una còpia de 32 bits de tota la matriu.

Sobre Mistral-Large-123B, l'estimació del planificador passa de 1,35 a 2,42 testimonis/s.

### Precisió mixta (`--snr-floor`, apagada per defecte)

El convertidor mesura la relació senyal/soroll a la sortida de cada capa per a cada tensor i pot promocionar cap a un format més ampli els que caiguin per sota de `--snr-floor`, dins del límit del 15 % dels tensors i d'un preu topall (`--promotion-cout-max`, en mebibytes afegits).

El llindar val **zero per defecte**: no es promociona res. La decodificació està limitada per l'ample de banda de memòria, i la mesura sobre `Huihui-Qwen3.8-27B` ho decideix — un llindar de 25 dB costa un 13,4 % de memòria i un 10,6 % de rendiment (18,50 Gio i 41,8 t/s contra 16,02 i 46,2) per un 2,0 % de perplexitat (42,591 contra 43,447, corpus de 16 383 testimonis). `--snr-floor 25` restableix l'antic comportament quan la qualitat prima sobre la velocitat.

### I `acvram eval`

La relació senyal/soroll i el cosinus dels logits són aproximacions. `acvram eval REP [REP ...]` mesura la perplexitat per finestra lliscant, perquè una elecció de format es decideixi amb proves:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Punts d'entrada HTTP

| punt d'entrada | notes |
|---|---|
| `POST /v1/chat/completions` | flux SSE o resposta única; utilitza la plantilla de conversa del model |
| `POST /v1/completions` | consigna en text o en identificadors de testimonis |
| `POST /v1/embeddings` | estats amagats finals mitjanats, normalitzats L2, `dimensions` respectat |
| `GET /v1/models` | més un bloc `acvram`: formats, dispositius, capacitat de la memòria cau KV |
| `GET /health`, `GET /metrics` | rendiment de decodificació, ocupació dels blocs KV |

Els noms de camp d'aquestes respostes es mantenen en anglès: és el protocol d'OpenAI, i traduir-los trencaria tots els clients existents.

---

<a id="chiffres"></a>

## D'on surten les xifres

Cada valor citat més amunt és produït per codi d'aquest dipòsit i verificat per `pytest`. Mesures fetes en processador amb els nuclis de referència:

| format | bits/pes | SNR dels pesos | cosinus dels logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dues constatacions sorgides d'aquestes mesures han canviat els valors per defecte:

* **Una rotació de Hadamard ajuda l'INT4 i no el NVFP4.** Els grups de 128 de l'INT4 no poden absorbir un canal aberrant isolat, de manera que escampar els valors extrems val una transformada en n log n per activació. Els blocs de 16 del NVFP4 ja porten la seva pròpia escala. D'aquí `--hadamard auto`, que només l'aplica a l'INT4.
* **L'INT8 supera el FP8 E4M3 per a la memòria cau KV**, 44 dB contra 32 dB a mida idèntica, perquè una escala per (testimoni, capçal) ja proporciona el rang dinàmic pel qual el FP8 gasta bits d'exponent. Les dues targetes utilitzen doncs una memòria cau KV en INT8, encara que la 5090 sabria fer FP8. Un format `k8v4` (valors en INT4, −22 % de bytes de memòria cau) existeix com a opció, **no qualificat** — vegeu `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentació

| Document | Contingut |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **represa del projecte en una altra màquina** (francès) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | com s'ensamblen les peces |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 pur o atenció+GDN en int8 per canal, sobre un híbrid Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | ajustar aquesta màquina precisa |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **el que encara no està fet**, per llegir primer |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | convencions de treball sobre el codi (llengua, estil, controls abans de pujar) |

---

<a id="resultats"></a>

## Resultats mesurats (22/09/2026, RTX 5090 a 400 W, règim ≥ 20 s al comptador d'energia)

Qwen3-Coder-30B-A3B en NVFP4 (experts) + INT8 (atenció, capçal), mateix protocol per a tots els motors (`outils/`, una targeta, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decodificació 12 seqüències | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| decodificació 1 seqüència | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 testimonis/s** | 21 054 | 8 671 (TabbyAPI, retirat) |

¹ Errata del 22/09: `serve` especula per defecte (`--speculative ngram`, cli.py), els competidors no; el 380,8 t/s publicat fins ara era mesurat AMB especulació. Sense especulació (`--speculative none`, mateixa cadena, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram és **tercer** a b=1, darrere de llama.cpp i vLLM. En energia continua per davant de llama.cpp (0,601 contra 0,700 J/testimoni net). A b=12 l'especulació mai no és activa (protecció `lot_max=2`): aquesta cel·la ja estava a armes iguals.

² 23/09, mateixa sessió, mateix client HTTP (`banc-llamacpp-16-09.py` contra `acvram serve` i `vllm serve`), `-lgc 2700` posat explícitament al voltant de cada braç, cel·les alternades A V V A, ≥ 5 lots per braç, diferència declarada només més enllà de 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 en decodificació, reducció d'atenció desenrotllada): diferència −1,6 %, **sota 2 σ: igualtat de rendiment**. En J/testimoni, **vLLM continua per davant en un 7,0 %** (més enllà de 2 σ). Amb 0.6.37 el mateix protocol donava −4,7 %.

³ Mateixa sessió i mateix protocol que ², sense especulació en tots dos costats: acvram 312,3 contra vLLM 284,8 — **acvram per davant en un 9,7 % de rendiment** (més enllà de 2 σ); J/testimoni: **igualtat** (diferència 0,04 %, sota 2 σ).

⁴ 23/09, mateix protocol contra llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 amb l'encaminament reescrit (+5,6 %): acvram 310,8 contra llama.cpp 329,9 t/s — **llama.cpp per davant en un 5,8 % de rendiment, acvram per davant en un 13,4 % en J/testimoni** (0,598 contra 0,691).

Rendiments del dia (lloc 1030, règim eco `-lgc 2700`, pipeline en servei; mostreig cobdiciós capturat al graf CUDA, per defecte des de 0.6.35). El b=12 d'acvram és una cel·la oficial segellada (mediana de 6 finestres intercalades, rellotge per finestra).

> **Errata (23/09/2026).** La comparativa amb vLLM publicada fins ara (b=12: 1 782 contra 1 634 t/s; b=1: 290,6) confrontava acvram mesurat en HTTP amb vLLM mesurat **fora de línia** (`LLM().generate()`), i l'errata del 22/09 afirmava erròniament que la cel·la vLLM passava per `vllm serve`. El 23/09: mateix client HTTP per a tots dos, i `-lgc` posat per a tots dos (acvram posa el seu propi a l'inici, `vllm serve` no: sense aquesta precaució vLLM anava a ~2 930 MHz contra ~2 650). Resultat a la nota ²: vLLM per davant en un 9,1 % a b=12.

El 14/09 al matí acvram estava a 630 t/s i 0,619 J/testimoni sobre la mateixa cel·la: els guanys venen de la MMA FP4 nativa de Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 sobre el bf16), del MoE en GEMM agrupada per pot de lot, d'un encaminament en un sol nucli (3 677 → 1 517 llançaments per pas) i d'un GEMM estret sobre tensor cores per a les projeccions. Cada xifra té la seva nota a `acvram-memoire/revue/` amb la predicció segellada abans de la mesura, l'instrument i el seu règim — una xifra sense règim no es publica.

On acvram va per davant: models MLA (GLM-4.7-Flash) en NVFP4 natiu sm_120, que vLLM només serveix en FP8 (b=1: 165,35 t/s en servei); els models que no càpiguen en VRAM. La decodificació de seqüència única no en forma part: sense especulació, acvram hi va per davant de vLLM en un 9,7 % (nota ³), darrere de llama.cpp en un 5,8 % de rendiment però per davant en un 13,4 % en energia (nota ⁴). A gran lot, sobre un MoE que càpiga a la VRAM, vLLM està en igualtat de rendiment a b=12 (1 995,1 contra 2 027,0 t/s, sota 2 σ, nota ²) però manté un 7,0 % menys de J/testimoni; acvram hi ha progressat de 1 540 t/s (0.6.34) a 1 995 (0.6.38).

---

<a id="etat"></a>

## Estat

Versió 0.6.38. Tot funciona a la 5090: nuclis CUDA compilats per a `sm_120a` (FP4 natiu) i `sm_86`, grafs CUDA, quantificació NVFP4/INT8/INT4, servidor HTTP. Salvaguardes en marxa: la targeta és invisible per a les sessions de treball (`CUDA_VISIBLE_DEVICES` buit) i només `outils/carte.sh` la presta, sota clau, a una mesura alhora; un vigilant registra qualsevol accés fora de clau; una mesura d'energia que cobreix més d'una targeta o menys de 10 s s'invalida; un model carregat en règim degradat ho diu i no entra en un duel.

4 107 proves (`pytest --collect-only -q`, un minut en processador; les proves GPU només s'executen sota `carte.sh`). Seguiment del treball: `acvram-memoire/` (regles, directori, quaderns, revisió de diverses centenes de notes).

---

<a id="credits"></a>

## Crèdits

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, sota llicència Apache-2.0: `acvram/kernels/marlin_port/` en porta els nuclis Marlin (MoE i dens), amb atribució completa fitxer per fitxer a [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, els tensor cores FP4 de Blackwell (`sm_120`) i les biblioteques de les quals depèn aquest projecte.
- **PyTorch** — motor tensorial i extensions C++/CUDA.

Projecte independent, no afiliat a ASUS, NVIDIA ni al projecte vLLM.

---

<a id="licence"></a>

## Llicència

[GPL-3.0 o posterior](../LICENSE) per al codi d'aquest dipòsit. `acvram/kernels/marlin_port/` conté codi portat de [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (nuclis `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), sota llicència Apache-2.0: cada fitxer manté la seva capçalera original, la llicència es troba a `LICENSE-vllm` i la llista de fitxers, el commit d'origen i les modificacions es troben a [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Donar suport al projecte

El desenvolupament d'acvram es duu a terme amb maquinari personal. Si el projecte us és útil:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Convidar%20un%20cafè&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Traduccions: [TRADUIRE.md](TRADUIRE.md) (francès; la guia de contribució del projecte encara no està traduïda).
