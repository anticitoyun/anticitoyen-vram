<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-supporta-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Un gateway di inferenza compatibile con l'API OpenAI, che tratta la memoria come una gerarchia, offre a ogni GPU il formato numerico che il suo silicio sa leggere meglio, e ottimizza ogni token in joule tanto quanto in secondi.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · **🇮🇹 Italiano** · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Confronto di throughput ed energia con vLLM e llama.cpp" width="720"></p>

---

## Sommario

- [Le due idee](#idees)
- [Avvio rapido](#demarrage)
- [Installazione](#installer)
- [Cosa dice `acvram plan`](#plan)
- [Andare veloci](#optimisations)
- [Endpoint HTTP](#http)
- [Da dove vengono i numeri](#chiffres)
- [Documentazione](#documentation)
- [Risultati misurati](#resultats)
- [Stato](#etat)
- [Crediti](#credits)
- [Licenza](#licence)
- [Sostenere il progetto](#soutien)

---

<a id="idees"></a>

## Le due idee

Pensata per una macchina precisa:

| | |
|---|---|
| Processore | Intel Core i9-14900K (8 core P + 16 core E) |
| Scheda madre | ASUS ROG Maximus Z790 Dark Hero |
| Memoria | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13) ; le due schede in PCIe x8/x8, limitate a 400 W / 275 W |

**Un formato per GPU.** La RTX 5090 possiede tensor core FP4; la RTX 3080 Ti non ne ha, e non ha nemmeno l'FP8. Allineare le due su un formato comune sprecherebbe la 5090. Il convertitore scrive quindi *due volte lo stesso modello*, nel formato che ogni destinazione sa realmente sfruttare:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesi | **NVFP4** — E2M1 + scala FP8 E4M3 ogni 16 | **INT4** — uint4 + scala e zero fp16 ogni 128 |
| bit per peso | 4,50 | 4,16 |
| rispetto al BF16 | ×3,56 più piccolo | ×3,85 più piccolo |
| modalità di calcolo | tensor core FP4 | dequantizzato in FP16 nel kernel, tensor core FP16 |
| cache KV | INT8 | INT8 |

32 Go di VRAM a 4,5 bit per peso contengono circa **56 miliardi di parametri**, contro 16 miliardi in BF16. Sulle due schede, ciò equivale approssimativamente a **78 miliardi di parametri residenti** prima ancora di toccare la memoria RAM.

**La memoria è una gerarchia, non un muro.** Tre livelli, e lo scheduler misura ciò che ognuno costa invece di sperare che il modello ci stia:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Avvio rapido

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Qualsiasi client OpenAI ci si collega poi:

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

## Installazione

Dal codice sorgente (tutte le piattaforme):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Oppure tramite pacchetto, un file allegato a ogni [release GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Canale | File allegato alla release | Comando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nomi generati da `rpmbuild`, non fissi) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (oppure `rpmbuild --rebuild *.src.rpm` a partire dal `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Prima di installare, verifica il file scaricato rispetto alle somme allegate alla release (`SHA256SUMS`, pubblicata una volta presenti tutti gli altri file):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip non è pubblicato come pacchetto (nessuna wheel compilata): `pip install -e '.[dev]'` installa da un clone del sorgente, come `./install.sh`.

---

<a id="plan"></a>

## Cosa dice `acvram plan`

Lo scheduler merita di essere lanciato prima di ogni download. Risponde alle domande che decidono se un modello è utilizzabile su questa macchina:

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

Esplora lo spazio delle configurazioni invece di accontentarsi della prima che ci sta, e due delle sue decisioni sono sufficientemente controintuitive da meritare di essere enunciate:

* **Lascia la 3080 Ti inutilizzata** quando un modello sta sulla sola 5090. Le tranche di una pipeline si eseguono in serie: aggiungere una fase a 912 Go/s in una pipeline a 1790 Go/s rallenta la decodifica a flusso singolo. Si forza con `--gpus all`.
* **Restringe la cache KV per mantenere i pesi in VRAM.** Ogni gigabyte dato alla cache è un gigabyte di pesi respinto sul bus PCIe, e leggere un peso via PCIe costa circa trenta volte quanto costa dalla VRAM. Sul 70B qui sopra, questo solo compromesso fa passare da 2,3 a 17,8 token/s.

---

<a id="optimisations"></a>

## Andare veloci

Quattro ottimizzazioni, ciascuna verificata da una prova di equivalenza e non solo da un cronometro: un'ottimizzazione che cambia la risposta è un bug.

I lineari NVFP4 dei modelli densi passano per default dalla disposizione Marlin (+57 a +90% di throughput a b = 8, TTFT +2 a +4 ms secondo revue/poste6-piece147-verdict-24-09.md; fallback `ACVRAM_PROJ_MARLIN=0`, vedi [CHANGELOG.md](../CHANGELOG.md)).

### Decodifica speculativa (`--speculative`)

Decodificare un token con un lotto di dimensione 1 è limitato dalla memoria: la macchina legge tutti i pesi attivi per produrre un solo token. Verificare K token proposti legge questi stessi pesi **una sola volta**. Due proponenti:

* `ngram` (default) — cerca il suffisso corrente più indietro nel contesto e propone ciò che seguiva. Non costa nulla, non richiede alcun modello. Vantaggioso quando l'output ricopia l'input: editing di codice, RAG, riassunto.
* `draft` — un piccolo modello su un secondo dispositivo. Su questo rig, questo dispositivo è la RTX 3080 Ti, che lo scheduler lascia volontariamente inattiva per ogni modello che sta sulla 5090.

`mtp` (testa `nextn` del modello) e `auto` esistono anch'essi; non vantaggiosi allo stato attuale e non attivati per default — vedi `docs/ARCHITECTURE.md`.

L'accettazione è esatta, non approssimata: una proposta è accettata con probabilità `min(1, p/q)` e un rifiuto ricampiona nella parte positiva normalizzata di `p - q`. Misurato su 40 000 estrazioni contro una bozza volutamente mal calibrata, la distribuzione emessa resta a 0,002 di variazione totale dal target — la speculazione compra velocità, mai una risposta diversa.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache di prefisso (attiva per default)

I blocchi sono indirizzati dall'hash *concatenato* della loro tranche di token: due richieste che condividono una consegna di sistema condividono i suoi blocchi, e la seconda non deve più precalcolarli. La concatenazione è indispensabile: gli stessi sedici token in un contesto diverso non contengono le stesse chiavi e valori, e hashare la sola tranche servirebbe la cache di una sequenza a un'altra.

Un blocco liberato il cui contenuto resta identificabile si unisce a una coda LRU piuttosto che alla lista dei blocchi liberi: la cache sopravvive così tra le richieste senza mai rifiutare un'allocazione che avrebbe potuto servire.

### Calcolo del livello host (`--host-exec`)

Uno strato i cui pesi risiedono in RAM può essere copiato verso la GPU o calcolato sul posto. I due percorsi sono limitati dalla memoria e leggono gli stessi byte: il più rapido è quello con il bus più largo — il PCIe 5.0 x16 dà circa 54 Go/s, la DDR5 dual channel circa 70 Go/s — e calcolare sul posto lascia inoltre la GPU libera invece di farla attendere una copia.

Questo vale solo se il processore legge direttamente i pesi impacchettati su 4 bit. Da qui un piccolo kernel C++ con un percorso AVX2 (`acvram_cpu.cpp`, caricato via ctypes, senza header Python né ninja). Anche sul suo ramo **scalare** di fallback, batte `dequantize() @ x` di un fattore 1,44 in INT4 e 3,21 in NVFP4, perché quest'ultimo scrive prima una copia a 32 bit dell'intera matrice.

Su Mistral-Large-123B, la stima dello scheduler passa da 1,35 a 2,42 token/s.

### Precisione mista (`--snr-floor`, disattivata per default)

Il convertitore misura il rapporto segnale/rumore in uscita di strato per ogni tensore e può promuovere verso un formato più ampio quelli che scendono sotto `--snr-floor`, nel limite del 15% dei tensori e di un prezzo massimo (`--promotion-cout-max`, in mebibyte aggiunti).

La soglia vale **zero per default**: nulla viene promosso. La decodifica è limitata dalla banda di memoria, e la misura su `Huihui-Qwen3.8-27B` decide — una soglia di 25 dB costa 13,4% di memoria e 10,6% di throughput (18,50 Gio e 41,8 t/s contro 16,02 e 46,2) per 2,0% di perplessità (42,591 contro 43,447, corpus di 16 383 token). `--snr-floor 25` ripristina il vecchio comportamento quando la qualità conta più della velocità.

### E `acvram eval`

Il rapporto segnale/rumore e il coseno dei logit sono approssimazioni. `acvram eval REP [REP ...]` misura la perplessità con finestra scorrevole, perché una scelta di formato si decida su prove:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Endpoint HTTP

| endpoint | note |
|---|---|
| `POST /v1/chat/completions` | flusso SSE o risposta unica; usa il template di conversazione del modello |
| `POST /v1/completions` | prompt in testo o in id di token |
| `POST /v1/embeddings` | stati nascosti finali mediati, normalizzati L2, `dimensions` rispettato |
| `GET /v1/models` | più un blocco `acvram`: formati, dispositivi, capacità della cache KV |
| `GET /health`, `GET /metrics` | throughput di decodifica, occupazione dei blocchi KV |

I nomi dei campi di queste risposte restano in inglese: è il protocollo OpenAI, e tradurli romperebbe tutti i client esistenti.

---

<a id="chiffres"></a>

## Da dove vengono i numeri

Ogni valore citato sopra è prodotto da codice di questo repository e verificato da `pytest`. Misure fatte su processore con i kernel di riferimento:

| formato | bit/peso | SNR dei pesi | coseno dei logit vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Due riscontri da queste misure hanno cambiato i valori di default:

* **Una rotazione di Hadamard aiuta l'INT4 e non il NVFP4.** I gruppi di 128 dell'INT4 non possono assorbire un canale anomalo isolato, per cui distribuire i valori estremi vale una trasformata in n log n per attivazione. I blocchi di 16 del NVFP4 portano già la propria scala. Da qui `--hadamard auto`, che la applica solo all'INT4.
* **L'INT8 batte l'FP8 E4M3 per la cache KV**, 44 dB contro 32 dB a parità di dimensione, perché una scala per (token, testa) fornisce già l'intervallo dinamico per il quale l'FP8 spende bit di esponente. Le due schede usano quindi una cache KV in INT8, anche se la 5090 saprebbe fare FP8. Un formato `k8v4` (valori in INT4, −22% di byte di cache) esiste come opzione, **non qualificato** — vedi `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentazione

| Documento | Contenuto |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **riprendere il progetto su un'altra macchina** (francese) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | come le parti si assemblano |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 puro oppure attention+GDN in int8 per canale, su un ibrido Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | configurare questa macchina precisa |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **ciò che non è ancora fatto**, da leggere per primo |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | convenzioni di lavoro sul codice (lingua, stile, controlli prima del push) |

---

<a id="resultats"></a>

## Risultati misurati (22/09/2026, RTX 5090 a 400 W, regime ≥ 20 s al contatore di energia)

Qwen3-Coder-30B-A3B in NVFP4 (esperti) + INT8 (attenzione, testa), stesso protocollo per tutti i motori (`outils/`, una scheda, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decodifica 12 sequenze | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| decodifica 1 sequenza | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, ritirato) |

¹ Errata del 22/09: `serve` specula per default (`--speculative ngram`, cli.py), i concorrenti no; il 380,8 t/s pubblicato finora era misurato CON speculazione. Senza speculazione (`--speculative none`, stessa catena, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram è **terzo** a b=1, dietro llama.cpp e vLLM. In energia resta davanti a llama.cpp (0,601 contro 0,700 J/token netti). A b=12 la speculazione non è mai attiva (guardia `lot_max=2`): questa cella era già ad armi uguali.

² 23/09, stessa sessione, stesso client HTTP (`banc-llamacpp-16-09.py` contro `acvram serve` e `vllm serve`), `-lgc 2700` posto esplicitamente attorno a ogni braccio, celle alternate A V V A, ≥ 5 lotti per braccio, scarto dichiarato solo oltre 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 alla decodifica, riduzione di attenzione srotolata): scarto −1,6%, **sotto 2 σ: parità di throughput**. In J/token, **vLLM resta davanti del 7,0%** (oltre 2 σ). Con 0.6.37 lo stesso protocollo dava −4,7%.

³ Stessa sessione e stesso protocollo di ², senza speculazione da entrambi i lati: acvram 312,3 contro vLLM 284,8 — **acvram davanti del 9,7% in throughput** (oltre 2 σ); J/token: **parità** (scarto 0,04%, sotto 2 σ).

⁴ 23/09, stesso protocollo contro llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 con il routing riscritto (+5,6%): acvram 310,8 contro llama.cpp 329,9 t/s — **llama.cpp davanti del 5,8% in throughput, acvram davanti del 13,4% in J/token** (0,598 contro 0,691).

Throughput del giorno (postazione 1030, regime eco `-lgc 2700`, pipeline in servizio; campionamento greedy catturato nel grafo CUDA, default di 0.6.35). Il b=12 acvram è una cella ufficiale sigillata (mediana di 6 finestre intercalate, clock per finestra).

> **Errata (23/09/2026).** Il confronto vLLM pubblicato finora (b=12: 1 782 contro 1 634 t/s; b=1: 290,6) opponeva acvram misurato in HTTP a vLLM misurato **offline** (`LLM().generate()`), e l'errata del 22/09 affermava a torto che la cella vLLM passasse per `vllm serve`. Il 23/09: stesso client HTTP per entrambi, e `-lgc` posto per entrambi (acvram pone il suo all'avvio, `vllm serve` no: senza questa precauzione vLLM girava a ~2 930 MHz contro ~2 650). Risultato in nota ²: vLLM davanti del 9,1% a b=12.

Il 14/09 al mattino acvram era a 630 t/s e 0,619 J/token sulla stessa cella: i guadagni vengono dalla MMA FP4 nativa di Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 sul bf16), dal MoE in GEMM raggruppata per godet di lotto, da un routing in un solo kernel (3 677 → 1 517 lanci per passo) e da un GEMM stretto su tensor core per le proiezioni. Ogni numero ha la sua nota in `acvram-memoire/revue/` con la previsione sigillata prima della misura, lo strumento e il suo regime — un numero senza regime non è pubblicato.

Dove acvram è avanti: modelli MLA (GLM-4.7-Flash) in NVFP4 nativo sm_120, che vLLM serve solo in FP8 (b=1: 165,35 t/s in servizio); i modelli che non stanno in VRAM. La decodifica a sequenza unica non ne fa parte: senza speculazione, acvram è lì davanti a vLLM del 9,7% (nota ³), dietro a llama.cpp del 5,8% in throughput ma davanti in energia del 13,4% (nota ⁴). A grande lotto, su un MoE che sta in VRAM, vLLM è alla pari in throughput a b=12 (1 995,1 contro 2 027,0 t/s, sotto 2 σ, nota ²) ma mantiene 7,0% di J/token in meno; acvram lì è progredito da 1 540 t/s (0.6.34) a 1 995 (0.6.38).

---

<a id="etat"></a>

## Stato

Versione 0.6.38. Tutto funziona sulla 5090: kernel CUDA compilati per `sm_120a` (FP4 nativo) e `sm_86`, grafi CUDA, quantizzazione NVFP4/INT8/INT4, server HTTP. Salvaguardie in atto: la scheda è invisibile alle sessioni di lavoro (`CUDA_VISIBLE_DEVICES` vuoto) e solo `outils/carte.sh` la presta, sotto lucchetto, a una misura alla volta; una sentinella registra ogni accesso fuori lucchetto; una misura di energia che coinvolge più di una scheda o meno di 10 s è invalidata; un modello caricato in regime degradato lo dichiara e non entra in un duello.

4 107 test (`pytest --collect-only -q`, un minuto su processore; i test GPU girano solo sotto `carte.sh`). Tracciamento del lavoro: `acvram-memoire/` (regole, indice, quaderni, revisione di diverse centinaia di note).

---

<a id="credits"></a>

## Crediti

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, sotto licenza Apache-2.0: `acvram/kernels/marlin_port/` ne porta i kernel Marlin (MoE e denso), con attribuzione completa file per file in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, i tensor core FP4 di Blackwell (`sm_120`) e le librerie da cui questo progetto dipende.
- **PyTorch** — motore tensoriale ed estensioni C++/CUDA.

Progetto indipendente, non affiliato ad ASUS, NVIDIA né al progetto vLLM.

---

<a id="licence"></a>

## Licenza

[GPL-3.0 o successiva](../LICENSE) per il codice di questo repository. `acvram/kernels/marlin_port/` contiene codice portato da [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kernel `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), sotto licenza Apache-2.0: ogni file mantiene la propria intestazione originale, la licenza è in `LICENSE-vllm` e l'elenco dei file, il commit d'origine e le modifiche sono in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Sostenere il progetto

Lo sviluppo di acvram è condotto su hardware personale. Se il progetto ti è utile:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Offri%20un%20caffè&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Traduzioni: [TRADUIRE.md](TRADUIRE.md) (francese; la guida ai contributi del progetto non è ancora tradotta).
