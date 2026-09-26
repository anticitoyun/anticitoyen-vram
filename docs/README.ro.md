<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-sustine-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Un gateway de inferență compatibil cu API-ul OpenAI, care tratează memoria ca o hierarhie, dă fiecărui GPU formatul numeric pe care siliciul său îl citește cel mai bine și optimizează fiecare jeton în jouli tot atât cât în secunde.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · **🇷🇴 Română** · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Comparație de debit și energie față de vLLM și llama.cpp" width="720"></p>

---

## Cuprins

- [Cele două idei](#idees)
- [Start rapid](#demarrage)
- [Instalare](#installer)
- [Ce spune `acvram plan`](#plan)
- [Mai rapid](#optimisations)
- [Puncte finale HTTP](#http)
- [De unde vin cifrele](#chiffres)
- [Documentație](#documentation)
- [Rezultate măsurate](#resultats)
- [Stare](#etat)
- [Credite](#credits)
- [Licență](#licence)
- [Susține proiectul](#soutien)

---

<a id="idees"></a>

## Cele două idei

Conceput pentru o mașină precisă:

| | |
|---|---|
| Procesor | Intel Core i9-14900K (8 nuclee P + 16 nuclee E) |
| Placă de bază | ASUS ROG Maximus Z790 Dark Hero |
| Memorie | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistem | Ubuntu 26.04 LTS (CUDA 13); ambele plăci în PCIe x8/x8, limitate la 400 W / 275 W |

**Un format per GPU.** RTX 5090 are tensor cores FP4; RTX 3080 Ti nu are, și nici FP8. Alinierea celor două la un format comun ar risipi 5090. Convertorul scrie deci *același model de două ori*, în formatul pe care fiecare destinație îl poate exploata cu adevărat:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| ponderi | **NVFP4** — E2M1 + scală FP8 E4M3 la fiecare 16 | **INT4** — uint4 + scală și zero fp16 la fiecare 128 |
| biți per pondere | 4,50 | 4,16 |
| față de BF16 | ×3,56 mai mic | ×3,85 mai mic |
| mod de calcul | tensor cores FP4 | dequantizat în FP16 în nucleu, tensor cores FP16 |
| cache KV | INT8 | INT8 |

32 Go de VRAM la 4,5 biți per pondere conțin aproximativ **56 de miliarde de parametri**, față de 16 miliarde în BF16. Pe ambele plăci, aceasta face aproximativ **78 de miliarde de parametri rezidenți** înainte de a atinge memoria RAM.

**Memoria este o hierarhie, nu un zid.** Trei etaje, iar planificatorul măsoară costul fiecăruia în loc să spere că modelul încape:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Start rapid

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Orice client OpenAI se conectează apoi:

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

## Instalare

Din sursă (toate platformele):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Sau pe pachet, un fișier atașat fiecărei [versiuni GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Canal | Fișier atașat versiunii | Comandă |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nume generate de `rpmbuild`, nefixe) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (sau `rpmbuild --rebuild *.src.rpm` din `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Înainte de instalare, verificați fișierul descărcat față de sumele atașate versiunii (`SHA256SUMS`, publicată după ce toate celelalte fișiere sunt prezente):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip nu este publicat ca pachet (nicio roată construită): `pip install -e '.[dev]'` instalează dintr-o clonă a sursei, la fel ca `./install.sh`.

---

<a id="plan"></a>

## Ce spune `acvram plan`

Planificatorul merită lansat înainte de orice descărcare. Răspunde la întrebările care decid dacă un model este utilizabil pe această mașină:

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

Explorează spațiul configurațiilor în loc să rețină prima care încape, iar două dintre deciziile sale sunt destul de contra-intuitive pentru a merita menționate:

* **Lasă 3080 Ti neutilizată** când un model încape doar pe 5090. Etapele unui pipeline se execută în serie: adăugarea unei etape la 912 Go/s într-un pipeline la 1790 Go/s încetinește decodarea cu un singur flux. Se forțează cu `--gpus all`.
* **Micșorează cache-ul KV pentru a păstra ponderile în VRAM.** Fiecare gigaoctet dat cache-ului este un gigaoctet de ponderi respins pe magistrala PCIe, iar citirea unei ponderi prin PCIe costă aproximativ de treizeci de ori mai mult decât din VRAM. Pe 70B de mai sus, doar acest compromis face să treacă de la 2,3 la 17,8 jetoane/s.

---

<a id="optimisations"></a>

## Mai rapid

Patru optimizări, fiecare verificată printr-o dovadă de echivalență și nu doar printr-un cronometru: o optimizare care schimbă răspunsul este un bug.

Linearele NVFP4 ale modelelor dense trec implicit prin dispunerea Marlin (+57 la +90% debit la b = 8, TTFT +2 la +4 ms conform revue/poste6-piece147-verdict-24-09.md; revenire `ACVRAM_PROJ_MARLIN=0`, vezi [CHANGELOG.md](../CHANGELOG.md)).

### Decodare speculativă (`--speculative`)

Decodarea unui jeton cu un lot de mărimea 1 este limitată de memorie: mașina citește toate ponderile active pentru a produce un singur jeton. Verificarea a K jetoane propuse citește aceleași ponderi **o singură dată**. Doi propunători:

* `ngram` (implicit) — caută sufixul curent mai devreme în context și propune ce urma. Nu costă nimic, nu cere niciun model. Rentabil când răspunsul recopiază intrarea: editare de cod, RAG, rezumat.
* `draft` — un model mic pe un al doilea dispozitiv. Pe acest rig, acest dispozitiv este RTX 3080 Ti, pe care planificatorul o lasă voit inactivă pentru orice model care încape pe 5090.

`mtp` (capul `nextn` al modelului) și `auto` există de asemenea; nerentabile în starea actuală și neactivate implicit — vezi `docs/ARCHITECTURE.md`.

Acceptarea este exactă, nu aproximativă: o propunere este acceptată cu probabilitatea `min(1, p/q)` și un refuz reeșantionează în partea pozitivă normalizată a `p - q`. Măsurată pe 40 000 de trageri față de un draft voit mal calibrat, distribuția emisă rămâne la 0,002 variație totală față de țintă — speculația cumpără viteză, niciodată un răspuns diferit.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache de prefix (activ implicit)

Blocurile sunt adresate prin hash-ul *înlănțuit* al secțiunii lor de jetoane: două cereri care partajează o instrucțiune de sistem partajează blocurile ei, și a doua nu mai trebuie să le precalculeze. Înlănțuirea este indispensabilă: aceleași șaisprezece jetoane într-un context diferit nu conțin aceleași chei și valori, și hash-uirea doar a secțiunii ar servi cache-ul unei secvențe altei secvențe.

Un bloc liberat al cărui conținut rămâne identificabil se alătură unei cozi LRU în loc de lista blocurilor libere: cache-ul supraviețuiește astfel între cereri fără a refuza vreodată o alocare pe care ar fi putut-o servi.

### Calcul pe etajul host (`--host-exec`)

Un strat ale cărui ponderi rezidă în RAM poate fi copiat spre GPU sau calculat pe loc. Ambele căi sunt limitate de memorie și citesc aceiași octeți: cea mai rapidă este cea a cărei magistrală este mai largă — PCIe 5.0 x16 oferă aproximativ 54 Go/s, DDR5 dual-channel aproximativ 70 Go/s — și calculul pe loc lasă în plus GPU-ul liber în loc să-l facă să aștepte o copie.

Aceasta este valabil doar dacă procesorul citește direct ponderile împachetate pe 4 biți. De aici un mic nucleu C++ cu o cale AVX2 (`acvram_cpu.cpp`, încărcat prin ctypes, fără header-e Python nici ninja). Chiar și pe ramura sa **scalară** de fallback, bate `dequantize() @ x` cu un factor de 1,44 în INT4 și 3,21 în NVFP4, pentru că acesta din urmă scrie mai întâi o copie pe 32 de biți a întregii matrice.

Pe Mistral-Large-123B, estimarea planificatorului trece de la 1,35 la 2,42 jetoane/s.

### Precizie mixtă (`--snr-floor`, dezactivată implicit)

Convertorul măsoară raportul semnal/zgomot la ieșirea stratului pentru fiecare tensor și poate promova spre un format mai larg pe cei care cad sub `--snr-floor`, în limita a 15% din tensori și a unui preț plafon (`--promotion-cout-max`, în mebioctets adăugați).

Plafonul valorează **zero implicit**: nimic nu este promovat. Decodarea este limitată de banda de memorie, iar măsurarea pe `Huihui-Qwen3.8-27B` decide — un plafon de 25 dB costă 13,4% memorie și 10,6% debit (18,50 Gio și 41,8 t/s față de 16,02 și 46,2) pentru 2,0% perplexitate (42,591 față de 43,447, corpus de 16 383 jetoane). `--snr-floor 25` restabilește vechiul comportament când calitatea primează asupra vitezei.

### Și `acvram eval`

Raportul semnal/zgomot și cosinusul logiturilor sunt aproximări. `acvram eval REP [REP ...]` măsoară perplexitatea prin fereastră glisantă, pentru ca o alegere de format să se decidă pe dovezi:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Puncte finale HTTP

| punct final | note |
|---|---|
| `POST /v1/chat/completions` | flux SSE sau răspuns unic; folosește șablonul de conversație al modelului |
| `POST /v1/completions` | invitație în text sau în identificatori de jetoane |
| `POST /v1/embeddings` | stări ascunse finale mediate, normalizate L2, `dimensions` respectat |
| `GET /v1/models` | plus un bloc `acvram`: formate, dispozitive, capacitate cache KV |
| `GET /health`, `GET /metrics` | debit de decodare, ocupare bloc KV |

Numele câmpurilor acestor răspunsuri rămân în engleză: este protocolul OpenAI, și traducerea lor ar rupe toți clienții existenți.

---

<a id="chiffres"></a>

## De unde vin cifrele

Fiecare valoare citată mai sus este produsă de codul acestui depozit și verificată prin `pytest`. Măsurători făcute pe procesor cu nucleele de referință:

| format | biți/pondere | SNR al ponderilor | cosinus al logiturilor vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Două constatări din aceste măsurători au schimbat valorile implicite:

* **O rotație Hadamard ajută INT4 și nu NVFP4.** Grupurile de 128 ale INT4 nu pot absorbi un canal aberant izolat, astfel încât întinderea valorilor extreme merită o transformare în n log n per activare. Blocurile de 16 ale NVFP4 poartă deja propria lor scală. De aici `--hadamard auto`, care o aplică doar la INT4.
* **INT8 bate FP8 E4M3 pentru cache-ul KV**, 44 dB față de 32 dB la mărime identică, pentru că o scală per (jeton, cap) furnizează deja plaja dinamică pentru care FP8 cheltuiește biți de exponent. Ambele plăci folosesc deci un cache KV în INT8, chiar dacă 5090 ar putea face FP8. Un format `k8v4` (valori în INT4, −22% octeți de cache) există ca opțiune, **necalificat** — vezi `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentație

| Document | Conținut |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **reluarea proiectului pe altă mașină** (franceză) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | cum se asamblează piesele |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 pur sau attention+GDN în int8 pe canal, pe un hibrid Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | reglarea acestei mașini precise |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **ce nu este încă făcut**, de citit primul |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | convenții de lucru pe cod (limbă, stil, verificări înainte de a împinge) |

---

<a id="resultats"></a>

## Rezultate măsurate (22/09/2026, RTX 5090 la 400 W, regim ≥ 20 s la contorul de energie)

Qwen3-Coder-30B-A3B în NVFP4 (experți) + INT8 (atenție, cap), același protocol pentru toate motoarele (`outils/`, o placă, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decodare 12 secvențe | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| decodare 1 secvență | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 jetoane/s** | 21 054 | 8 671 (TabbyAPI, retras) |

¹ Erată din 22/09: `serve` speculează implicit (`--speculative ngram`, cli.py), concurenții nu; 380,8 t/s publicat până acum a fost măsurat CU speculație. Fără speculație (`--speculative none`, același lanț, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram este **al treilea** la b=1, în urma lui llama.cpp și vLLM. În energie rămâne înaintea lui llama.cpp (0,601 față de 0,700 J/jeton net). La b=12 speculația nu este niciodată activă (gardă `lot_max=2`): această celulă era deja la arme egale.

² 23/09, aceeași sesiune, același client HTTP (`banc-llamacpp-16-09.py` față de `acvram serve` și `vllm serve`), `-lgc 2700` pus explicit în jurul fiecărui braț, celule alternate A V V A, ≥ 5 loturi per braț, abatere declarată doar dincolo de 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 la decodare, reducere de atenție derulată): abatere −1,6%, **sub 2 σ: egalitate de debit**. În J/jeton, **vLLM rămâne înainte cu 7,0%** (dincolo de 2 σ). Cu 0.6.37 același protocol dădea −4,7%.

³ Aceeași sesiune și același protocol ca ², fără speculație pe ambele părți: acvram 312,3 față de vLLM 284,8 — **acvram înainte cu 9,7% în debit** (dincolo de 2 σ); J/jeton: **egalitate** (abatere 0,04%, sub 2 σ).

⁴ 23/09, același protocol față de llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 cu rutarea rescrisă (+5,6%): acvram 310,8 față de llama.cpp 329,9 t/s — **llama.cpp înainte cu 5,8% în debit, acvram înainte cu 13,4% în J/jeton** (0,598 față de 0,691).

Debituri ale zilei (post 1030, regim eco `-lgc 2700`, pipeline în serviciu; eșantionare glutonă capturată în graful CUDA, implicit din 0.6.35). b=12 acvram este o celulă oficială sigilată (mediana a 6 ferestre intercalate, ceas per fereastră).

> **Erată (23/09/2026).** Comparativul vLLM publicat până acum (b=12: 1 782 față de 1 634 t/s; b=1: 290,6) opunea acvram măsurat în HTTP față de vLLM măsurat **offline** (`LLM().generate()`), iar erata din 22/09 afirma eronat că celula vLLM trecea prin `vllm serve`. Pe 23/09: același client HTTP pentru ambele, și `-lgc` pus pentru ambele (acvram îl pune la pornire, `vllm serve` nu: fără această precauție vLLM funcționa la ~2 930 MHz față de ~2 650). Rezultat în nota ²: vLLM înainte cu 9,1% la b=12.

Pe 14/09 dimineața acvram era la 630 t/s și 0,619 J/jeton pe aceeași celulă: câștigurile vin din MMA FP4 nativă a Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 față de bf16), din MoE în GEMM grupată per godet de lot, dintr-o rutare într-un singur nucleu (3 677 → 1 517 lansări per pas) și dintr-un GEMM îngust pe tensor cores pentru proiecții. Fiecare cifră are nota sa în `acvram-memoire/revue/` cu predicția sigilată înainte de măsurare, instrumentul și regimul său — o cifră fără regim nu este publicată.

Unde acvram este înainte: modele MLA (GLM-4.7-Flash) în NVFP4 nativ sm_120, pe care vLLM îl servește doar în FP8 (b=1: 165,35 t/s în serviciu); modelele care nu încap în VRAM. Decodarea cu secvență unică nu face parte: fără speculație, acvram este acolo înainte de vLLM cu 9,7% (nota ³), în urma lui llama.cpp cu 5,8% în debit dar înaintea lui cu 13,4% în energie (nota ⁴). La lot mare, pe un MoE care încape în VRAM, vLLM este la egalitate de debit la b=12 (1 995,1 față de 2 027,0 t/s, sub 2 σ, nota ²) dar păstrează 7,0% mai puțin J/jeton; acvram a progresat aici de la 1 540 t/s (0.6.34) la 1 995 (0.6.38).

---

<a id="etat"></a>

## Stare

Versiunea 0.6.38. Tot funcționează pe 5090: nuclee CUDA compilate pentru `sm_120a` (FP4 nativ) și `sm_86`, grafuri CUDA, cuantizare NVFP4/INT8/INT4, server HTTP. Garanții în vigoare: placa este invizibilă sesiunilor de lucru (`CUDA_VISIBLE_DEVICES` gol) și doar `outils/carte.sh` o împrumută, sub blocaj, unei singure măsurători odată; un supraveghetor înregistrează orice acces fără blocaj; o măsurare de energie care acoperă mai mult de o placă sau mai puțin de 10 s este invalidată; un model încărcat în regim degradat o spune și nu intră într-un duel.

4 107 teste (`pytest --collect-only -q`, un minut pe procesor; testele GPU rulează doar sub `carte.sh`). Urmărirea lucrului: `acvram-memoire/` (regulament, director, carnete, revistă a câteva sute de note).

---

<a id="credits"></a>

## Credite

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, sub licență Apache-2.0: `acvram/kernels/marlin_port/` îi poartă nucleele Marlin (MoE și dens), cu atribuire completă fișier cu fișier în [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, tensor cores FP4 ale Blackwell (`sm_120`) și bibliotecile de care depinde acest proiect.
- **PyTorch** — motor tensorial și extensii C++/CUDA.

Proiect independent, neafiliat cu ASUS, NVIDIA nici cu proiectul vLLM.

---

<a id="licence"></a>

## Licență

[GPL-3.0 sau ulterioară](../LICENSE) pentru codul acestui depozit. `acvram/kernels/marlin_port/` conține cod portat din [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (nuclee `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), sub licență Apache-2.0: fiecare fișier își păstrează antetul original, licența este în `LICENSE-vllm` iar lista fișierelor, commit-ul de origine și modificările sunt în [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Susține proiectul

Dezvoltarea lui acvram este dusă pe hardware personal. Dacă proiectul vă este util:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Ofera%20o%20cafea&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Traduceri: [TRADUIRE.md](TRADUIRE.md) (franceză; ghidul de contribuție al proiectului nu este încă tradus).
