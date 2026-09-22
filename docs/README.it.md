<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Sostenere: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Un gateway di inferenza compatibile con l'API OpenAI, che tratta la memoria
come una gerarchia e dà a ogni GPU il formato numerico che il suo silicio legge
meglio.

Progettato per una macchina precisa:

| | |
|---|---|
| Processore | Intel Core i9-14900K (8 core P + 16 core E) |
| Scheda madre | ASUS ROG Maximus Z790 Dark Hero |
| Memoria | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13); entrambe le schede in PCIe x8/x8, limitate a 400 W / 275 W |

## Le due idee

**Un formato per GPU.** La RTX 5090 ha tensor core FP4; la RTX 3080 Ti non ne
ha, e non ha nemmeno l'FP8. Allineare le due su un formato comune sprecherebbe
la 5090. Il convertitore scrive quindi *due volte lo stesso modello*, nel
formato che ogni destinazione sa davvero sfruttare:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesi | **NVFP4** — E2M1 + scala FP8 E4M3 ogni 16 | **INT4** — uint4 + scala e zero fp16 ogni 128 |
| bit per peso | 4,50 | 4,16 |
| rispetto a BF16 | ×3,56 più piccolo | ×3,85 più piccolo |
| modalità di calcolo | tensor core FP4 | dequantizzato in FP16 nel kernel, tensor core FP16 |
| cache KV | INT8 | INT8 |

32 GB di VRAM a 4,5 bit per peso contengono circa **56 miliardi di
parametri**, contro 16 miliardi in BF16. Sulle due schede, sono
approssimativamente **78 miliardi di parametri residenti** prima ancora di
toccare la RAM.

**La memoria è una gerarchia, non un muro.** Tre livelli, e il pianificatore
misura quanto costa ciascuno invece di sperare che il modello ci stia:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Avvio rapido

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Qualsiasi client OpenAI si collega poi:

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

## Cosa dice `acvram plan`

Il pianificatore merita di essere lanciato prima di qualsiasi download.
Risponde alle domande che decidono se un modello è utilizzabile su questa
macchina:

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

Esplora lo spazio delle configurazioni invece di tenere la prima che ci sta,
e due delle sue decisioni sono abbastanza controintuitive da meritare di
essere enunciate:

* **Lascia la 3080 Ti inutilizzata** quando un modello ci sta sulla sola
  5090. Le fette di una pipeline si eseguono in serie: aggiungere uno stadio a
  912 GB/s in una pipeline a 1790 GB/s rallenta la decodifica a flusso
  singolo. Si forza con `--gpus all`.
* **Restringe la cache KV per tenere i pesi in VRAM.** Ogni gigabyte dato alla
  cache è un gigabyte di pesi respinto sul bus PCIe, e leggere un peso via PCIe
  costa circa trenta volte quanto costa dalla VRAM. Sul 70B qui sopra, questo
  solo arbitrato fa passare da 2,3 a 17,8 token/s.

## Andare veloci

Quattro ottimizzazioni, ciascuna verificata da una prova di equivalenza e non
solo da un cronometro: un'ottimizzazione che cambia la risposta è un bug.

### Decodifica speculativa (`--speculative`)

Decodificare un token con un lotto di dimensione 1 è limitato dalla memoria:
la macchina legge tutti i pesi attivi per produrre un solo token. Verificare K
token proposti legge quegli stessi pesi **una sola volta**. Due proponenti:

* `ngram` (predefinito) — cerca il suffisso corrente più indietro nel contesto
  e propone ciò che seguiva. Non costa nulla, non richiede alcun modello.
  Conviene quando l'uscita ricopia l'ingresso: modifica di codice, RAG,
  riassunto.
* `draft` — un piccolo modello su un secondo dispositivo. Su questa macchina
  quel dispositivo è la RTX 3080 Ti, che il pianificatore lascia
  volutamente inattiva per ogni modello che ci sta sulla 5090.

L'accettazione è esatta, non approssimata: una proposta è accettata con
probabilità `min(1, p/q)` e un rifiuto ricampiona nella parte positiva
normalizzata di `p - q`. Misurato su 40 000 estrazioni contro una bozza
volutamente mal calibrata, la distribuzione emessa resta a 0,002 di
variazione totale dall'obiettivo — la speculazione compra velocità, mai una
risposta diversa.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache dei prefissi (attiva per impostazione predefinita)

I blocchi sono indirizzati dall'hash *concatenato* della loro fetta di token:
due richieste che condividono un'istruzione di sistema condividono i suoi
blocchi, e la seconda non deve più precalcolarli. La concatenazione è
indispensabile: gli stessi sedici token in un contesto diverso non contengono
le stesse chiavi e valori, e fare l'hash della sola fetta servirebbe la cache
di una sequenza a un'altra.

Un blocco liberato il cui contenuto resta identificabile passa in una coda LRU
anziché nella lista dei blocchi liberi: la cache sopravvive così tra le
richieste senza mai rifiutare un'allocazione che avrebbe potuto servire.

### Calcolo sul livello host (`--host-exec`)

Uno strato i cui pesi risiedono in RAM può essere copiato sulla GPU o
calcolato sul posto. Entrambi i percorsi sono limitati dalla memoria e leggono
gli stessi byte: il più veloce è quello con il bus più largo — il PCIe 5.0 x16
dà circa 54 GB/s, la DDR5 a doppio canale circa 70 GB/s — e calcolare sul posto
lascia inoltre la GPU libera invece di farla aspettare una copia.

Questo vale solo se il processore legge direttamente i pesi impacchettati a
4 bit. Da qui un piccolo kernel C++ con un percorso AVX2 (`acvram_cpu.cpp`,
caricato via ctypes, senza header Python né ninja). Persino sul suo ramo
**scalare** di ripiego, batte `dequantize() @ x` di un fattore 1,44 in INT4 e
3,21 in NVFP4, perché quest'ultimo scrive prima una copia a 32 bit di tutta la
matrice.

Su Mistral-Large-123B, la stima del pianificatore passa da 1,35 a
2,42 token/s.

### Precisione mista (`--snr-floor`, disattivata per impostazione predefinita)

Il convertitore misura il rapporto segnale/rumore in uscita da ogni strato per
ogni tensore e può promuovere a un formato più ampio quelli che scendono sotto
`--snr-floor`, nel limite del 15 % dei tensori e di un prezzo massimo
(`--promotion-cout-max`, in mebibyte aggiunti).

La soglia vale **zero per impostazione predefinita**: nulla viene promosso.
La decodifica è limitata dalla banda di memoria, e la misura su
`Huihui-Qwen3.8-27B` decide — una soglia di 25 dB costa il 13,4 % di memoria e
il 10,6 % di portata (18,50 GiB e 41,8 t/s contro 16,02 e 46,2) per il 2,0 % di
perplessità (42,591 contro 43,447, corpus di 16 383 token). `--snr-floor 25`
ripristina il vecchio comportamento quando la qualità conta più della
velocità.

### E `acvram eval`

Il rapporto segnale/rumore e il coseno dei logit sono approssimazioni.
`acvram eval DIR [DIR ...]` misura la perplessità a finestra scorrevole,
perché una scelta di formato si decida sulle prove:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Endpoint HTTP

| endpoint | note |
|---|---|
| `POST /v1/chat/completions` | flusso SSE o risposta unica; usa il template di conversazione del modello |
| `POST /v1/completions` | prompt in testo o in identificatori di token |
| `POST /v1/embeddings` | stati nascosti finali mediati, normalizzati L2, `dimensions` rispettato |
| `GET /v1/models` | più un blocco `acvram`: formati, dispositivi, capacità della cache KV |
| `GET /health`, `GET /metrics` | portata di decodifica, occupazione dei blocchi KV |

I nomi dei campi di queste risposte restano in inglese: è il protocollo
OpenAI, e tradurli romperebbe tutti i client esistenti.

## Da dove vengono i numeri

Ogni valore citato sopra è prodotto da codice di questo repository e
verificato da `pytest`. Misure fatte su processore con i kernel di
riferimento:

| formato | bit/peso | SNR dei pesi | coseno dei logit vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Due constatazioni tratte da queste misure hanno cambiato i valori
predefiniti:

* **Una rotazione di Hadamard aiuta l'INT4 e non l'NVFP4.** I gruppi di 128
  dell'INT4 non possono assorbire un canale anomalo isolato, sicché distribuire
  i valori estremi vale una trasformata in n log n per attivazione. I blocchi
  di 16 dell'NVFP4 portano già la propria scala. Da qui `--hadamard auto`, che
  la applica solo all'INT4.
* **L'INT8 batte l'FP8 E4M3 per la cache KV**, 44 dB contro 32 dB a
  dimensione identica, perché una scala per (token, testa) fornisce già la
  gamma dinamica per cui l'FP8 spende bit di esponente. Le due schede usano
  quindi una cache KV in INT8, anche se la 5090 saprebbe fare FP8.

## Documentazione

* [`REPRISE.md`](../REPRISE.md) — **riprendere il progetto su un'altra macchina**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — come i pezzi si assemblano
* [`docs/MATERIEL.md`](MATERIEL.md) — regolare questa macchina precisa
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **ciò che non è fatto**, da leggere per primo
* [`CONVENTIONS.md`](../CONVENTIONS.md) — convenzioni di lavoro sul codice (lingua, stile, controlli prima del push)

## Risultati misurati (22/09/2026, RTX 5090 a 400 W, regime ≥ 20 s al contatore di energia)

Qwen3-Coder-30B-A3B in NVFP4 (esperti) + INT8 (attenzione, testa), stesso
protocollo per tutti i motori (`outils/`, una scheda, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| decodifica 12 sequenze | **1 634 t/s** | 1 782 t/s | — |
| decodifica 1 sequenza | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, ritirato) |

Portate del giorno (postazione 1030, regime eco `-lgc 2700`, pipeline in
servizio; campionamento greedy catturato nel grafo CUDA, predefinito dalla
0.6.35). Il b=12 è una cella ufficiale sigillata (mediana di 6 finestre
interlacciate, clock per finestra).

> **Erratum (22/09/2026).** La prima pubblicazione di 0.6.35 traeva « +1,84 %
> davanti a vLLM » da un riferimento vLLM di 1 596 t/s del 21/09 che veniva da
> una **generazione offline** (`LLM().generate()`), **non comparabile con un
> server**: nessuna schedulazione continua, non il percorso di `acvram serve`.
> Corretto il 22/09 con una cella alternata A/V (A1 V1 A2 V2 A3 V3) contro
> **`vllm serve`** (HTTP), stessa scheda e stesso percorso di `acvram serve`:
> vLLM mediana **1 782 t/s**. A misura comparabile, **acvram (1 634 t/s) è
> DIETRO vLLM di circa l'8 % a b=12**, non davanti. Il J/token a clock uguale
> resta in rimisurazione.

La mattina del 14/09 acvram era a 630 t/s e 0,619 J/token sulla stessa cella:
i guadagni vengono dalla MMA FP4 nativa di Blackwell
(`mma.sync … kind::mxf4nvf4`, ×7,9 sul bf16), dal MoE in GEMM raggruppata per
secchio di lotto, da un instradamento in un solo kernel (3 677 → 1 517 lanci
per passo) e da una GEMM stretta su tensor core per le proiezioni. Ogni cifra
ha la sua nota in `acvram-memoire/revue/` con la previsione sigillata prima
della misura, lo strumento e il suo regime — una cifra senza regime non si
pubblica.

Dove acvram è avanti: modelli MLA (GLM-4.7-Flash) in NVFP4 nativo sm_120, che
vLLM serve solo in FP8 (b=1: 165,35 t/s in servizio); i modelli che non ci
stanno in VRAM; e la decodifica a sequenza singola (b=1: 380,8 t/s contro 290,6
per vLLM). A lotto grande invece, su un MoE che ci sta in VRAM, vLLM resta
davanti a b=12 (1 782 contro 1 634 t/s, cfr. erratum); acvram è progredito
(1 540 in 0.6.34 → 1 634) senza passare davanti. Lo scarto in energia è da
rimisurare.

## Stato

Versione 0.6.35. Tutto gira sulla 5090: kernel CUDA compilati per `sm_120a`
(FP4 nativo) e `sm_86`, grafi CUDA, quantizzazione NVFP4/INT8/INT4, server
HTTP. Protezioni in essere: la scheda è invisibile alle sessioni di lavoro
(`CUDA_VISIBLE_DEVICES` vuoto) e solo `outils/carte.sh` la presta, sotto
lucchetto, a una misura alla volta; un guardiano registra ogni accesso fuori
dal lucchetto; una misura di energia che copra più di una scheda o meno di
10 s è invalidata; un modello caricato in regime degradato lo dice e non entra
in un duello.

640 test (`pytest -q`, un minuto su processore; i test GPU girano solo sotto
`carte.sh`). Tracciamento del lavoro: `acvram-memoire/` (regole, elenco,
quaderni, revisione di 180 note).

## Sostenere

Lo sviluppo di acvram è condotto su hardware personale. Se il progetto vi è
utile: **Sostenere: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licenza

GPL-3.0 o successiva.
