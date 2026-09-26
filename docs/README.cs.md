<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-podpořit-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Brána pro inferenci kompatibilní s API OpenAI, která zachází s pamětí jako s hierarchií, dává každému GPU číselný formát, který jeho křemík čte nejlépe, a optimalizuje každý token v joulech stejně jako v sekundách.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · **🇨🇿 Čeština** · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Srovnání propustnosti a energie s vLLM a llama.cpp" width="720"></p>

---

## Obsah

- [Dvě myšlenky](#idees)
- [Rychlý start](#demarrage)
- [Instalace](#installer)
- [Co říká `acvram plan`](#plan)
- [Zrychlení](#optimisations)
- [HTTP koncové body](#http)
- [Odkud pocházejí čísla](#chiffres)
- [Dokumentace](#documentation)
- [Naměřené výsledky](#resultats)
- [Stav](#etat)
- [Poděkování](#credits)
- [Licence](#licence)
- [Podpořte projekt](#soutien)

---

<a id="idees"></a>

## Dvě myšlenky

Navrženo pro konkrétní stroj:

| | |
|---|---|
| Procesor | Intel Core i9-14900K (8 jader P + 16 jader E) |
| Základní deska | ASUS ROG Maximus Z790 Dark Hero |
| Paměť | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Systém | Ubuntu 26.04 LTS (CUDA 13); obě karty v PCIe x8/x8, omezené na 400 W / 275 W |

**Jeden formát na GPU.** RTX 5090 má tenzorová jádra FP4; RTX 3080 Ti nemá ani ta, ani FP8. Sladění obou na společný formát by promarnilo potenciál 5090. Konvertor proto zapíše *tentýž model dvakrát*, ve formátu, který každý cíl dokáže skutečně využít:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| váhy | **NVFP4** — E2M1 + škála FP8 E4M3 každých 16 | **INT4** — uint4 + škála a nula fp16 každých 128 |
| bitů na váhu | 4,50 | 4,16 |
| oproti BF16 | ×3,56 menší | ×3,85 menší |
| režim výpočtu | tenzorová jádra FP4 | dekvantizace na FP16 v jádru, tenzorová jádra FP16 |
| KV cache | INT8 | INT8 |

32 GB VRAM při 4,5 bitu na váhu pojme přibližně **56 miliard parametrů**, oproti 16 miliardám v BF16. Na obou kartách dohromady to dává přibližně **78 miliard rezidentních parametrů** ještě před sáhnutím do paměti hostitele.

**Paměť je hierarchie, ne zeď.** Tři úrovně, a plánovač měří, co každá skutečně stojí, místo aby doufal, že se model vejde:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
DDR5 hostitele 96 GB  omezeno PCIe nebo DDR
```

---

<a id="demarrage"></a>

## Rychlý start

```bash
./install.sh                       # virtuální prostředí + torch cu128 + acvram
acvram doctor                      # je tento stroj připraven, a na co
acvram detect                      # co tu skutečně je

acvram plan  ~/modely/Qwen3-32B                     # kam by šla každá vrstva
acvram convert ~/modely/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Jakýkoli klient OpenAI se pak hned připojí:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Ahoj"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="nepoužito")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Ahoj"}])
```

---

<a id="installer"></a>

## Instalace

Ze zdrojového kódu (všechny platformy):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Nebo z balíčku – ke každému [vydání na GitHubu](https://github.com/anticitoyun/anticitoyen-vram/releases/latest) je přiložen jeden soubor:

| Kanál | Soubor přiložený k vydání | Příkaz |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (názvy generuje `rpmbuild`, nejsou pevně dané) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (nebo `rpmbuild --rebuild *.src.rpm` ze souboru `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Před instalací ověřte stažený soubor podle součtů připojených k release (`SHA256SUMS`, zveřejněné až po přítomnosti všech ostatních souborů):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip není publikován jako balíček (nesestavuje se žádný wheel): `pip install -e '.[dev]'` instaluje z klonu zdrojového kódu, stejně jako `./install.sh`.

---

<a id="plan"></a>

## Co říká `acvram plan`

Plánovač si zaslouží spuštění před každým stažením. Odpovídá na otázky, které rozhodují, zda je model na tomto stroji vůbec použitelný:

```
$ acvram plan ~/modely/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

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

Prohledává prostor konfigurací místo toho, aby se zastavil u první, která se vejde, a dvě jeho rozhodnutí jsou dost neintuitivní na to, aby stálo za to je vyslovit:

* **Nechává 3080 Ti nevyužitou**, když se model vejde na samotnou 5090. Fáze pipeline běží sériově: přidání fáze 912 GB/s do pipeline 1790 GB/s zpomalí dekódování jednoho proudu. Vynutí se to přes `--gpus all`.
* **Zmenšuje KV cache, aby udržel váhy ve VRAM.** Každý gigabajt daný cache je gigabajt vah odsunutých na sběrnici PCIe, a čtení váhy přes PCIe stojí zhruba třicetkrát víc, než co stojí z VRAM. Na 70B výše tento jediný kompromis zvedne výkon z 2,3 na 17,8 tokenu/s.

---

<a id="optimisations"></a>

## Zrychlení

Čtyři optimalizace, každá ověřená důkazem ekvivalence, ne jen stopkami: optimalizace, která mění odpověď, je chyba.

Lineární vrstvy NVFP4 hustých modelů standardně procházejí uspořádáním Marlin (+57 až +90 % propustnosti při b = 8, TTFT +2 až +4 ms podle revue/poste6-piece147-verdict-24-09.md; návrat `ACVRAM_PROJ_MARLIN=0`, viz [CHANGELOG.md](../CHANGELOG.md)).

### Spekulativní dekódování (`--speculative`)

Dekódování jednoho tokenu s dávkou velikosti 1 je omezeno pamětí: stroj čte všechny aktivní váhy, aby vyprodukoval jediný token. Ověření K navržených tokenů čte tytéž váhy **jen jednou**. Dva navrhovatelé:

* `ngram` (výchozí) — hledá aktuální příponu dříve v kontextu a navrhuje, co po ní následovalo. Nic nestojí, nevyžaduje žádný model. Vyplatí se, když výstup opisuje vstup: úprava kódu, RAG, shrnutí.
* `draft` — malý model na druhém zařízení. Na této sestavě je tímto zařízením RTX 3080 Ti, kterou plánovač záměrně nechává nečinnou pro každý model vejde se na 5090.

Existují také `mtp` (hlava `nextn` modelu) a `auto`; za současného stavu se nevyplatí a nejsou ve výchozím nastavení zapnuty — viz `docs/ARCHITECTURE.md`.

Přijetí je přesné, ne přibližné: návrh je přijat s pravděpodobností `min(1, p/q)` a zamítnutí znovu vzorkuje z normalizované kladné části `p - q`. Měřeno na 40 000 taženích proti záměrně špatně kalibrovanému draftu zůstává vyzařované rozdělení v mezích 0,002 celkové variace cíle — spekulace kupuje rychlost, nikdy jinou odpověď.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefixová cache (aktivní ve výchozím nastavení)

Bloky jsou adresovány *řetězeným* hashem svého úseku tokenů: dva požadavky sdílející systémový pokyn sdílejí jeho bloky, a druhý je už nemusí předpočítat. Řetězení je nezbytné: stejných šestnáct tokenů v jiném kontextu neobsahuje stejné klíče a hodnoty, a hashování samotného úseku by obsloužilo cache jedné sekvence požadavkem jiné.

Uvolněný blok, jehož obsah zůstává identifikovatelný, se připojí do fronty LRU místo do seznamu volných bloků: cache tak přežívá mezi požadavky, aniž by kdy odmítla alokaci, kterou mohla obsloužit.

### Výpočet na úrovni hostitele (`--host-exec`)

Vrstva, jejíž váhy sídlí v RAM, může být zkopírována na GPU nebo vypočtena na místě. Obě cesty jsou omezeny pamětí a čtou stejné bajty: rychlejší je ta, jejíž sběrnice je širší — PCIe 5.0 x16 dává asi 54 GB/s, dvoukanálová DDR5 asi 70 GB/s — a výpočet na místě navíc nechává GPU volné místo aby čekalo na kopii.

To se vyplatí jen tehdy, čte-li procesor přímo váhy zabalené do 4 bitů. Odtud malé jádro v C++ s cestou AVX2 (`acvram_cpu.cpp`, načítané přes ctypes, bez hlaviček Pythonu nebo ninja). I na své záložní **skalární** větvi poráží `dequantize() @ x` faktorem 1,44 v INT4 a 3,21 v NVFP4, protože ten druhý nejprve zapíše 32bitovou kopii celé matice.

Na Mistral-Large-123B stoupá odhad plánovače z 1,35 na 2,42 tokenu/s.

### Smíšená přesnost (`--snr-floor`, ve výchozím nastavení vypnuto)

Konvertor měří poměr signálu k šumu na výstupu vrstvy pro každý tenzor a může povýšit na širší formát ty, které klesnou pod `--snr-floor`, v mezích 15 % tenzorů a cenového stropu (`--promotion-cout-max`, v přidaných mebibajtech).

Práh je **ve výchozím nastavení nula**: nic se nepovyšuje. Dekódování je omezeno šířkou paměťové sběrnice, a měření na `Huihui-Qwen3.8-27B` to rozhoduje — práh 25 dB stojí 13,4 % paměti a 10,6 % propustnosti (18,50 GiB a 41,8 t/s oproti 16,02 a 46,2) za 2,0 % perplexity (42,591 oproti 43,447, korpus 16 383 tokenů). `--snr-floor 25` obnoví staré chování, když kvalita má přednost před rychlostí.

### A `acvram eval`

Poměr signálu k šumu a kosinus logitů jsou aproximace. `acvram eval REP [REP ...]` měří perplexitu v klouzavém okně, aby volbu formátu rozhodly důkazy:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP koncové body

| koncový bod | poznámky |
|---|---|
| `POST /v1/chat/completions` | proud SSE nebo jediná odpověď; používá konverzační šablonu modelu |
| `POST /v1/completions` | pokyn jako text nebo jako identifikátory tokenů |
| `POST /v1/embeddings` | zprůměrované finální skryté stavy, L2 normalizované, `dimensions` respektováno |
| `GET /v1/models` | plus blok `acvram`: formáty, zařízení, kapacita KV cache |
| `GET /health`, `GET /metrics` | propustnost dekódování, obsazenost bloků KV |

Názvy polí těchto odpovědí zůstávají anglicky: je to protokol OpenAI a jejich překlad by rozbil všechny existující klienty.

---

<a id="chiffres"></a>

## Odkud pocházejí čísla

Každá výše uvedená hodnota vzniká z kódu tohoto repozitáře a je ověřena `pytest`em. Měření provedena na procesoru s referenčními jádry:

| formát | bitů/váhu | SNR vah | kosinus logitů vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dva poznatky z těchto měření změnily výchozí hodnoty:

* **Hadamardova rotace pomáhá INT4, ne NVFP4.** Skupiny po 128 v INT4 nemohou pohltit izolovaný odlehlý kanál, takže rozprostření extrémních hodnot stojí za transformaci n log n na aktivaci. Bloky po 16 v NVFP4 už mají vlastní škálu. Odtud `--hadamard auto`, které ji aplikuje jen na INT4.
* **INT8 poráží FP8 E4M3 pro KV cache**, 44 dB oproti 32 dB při stejné velikosti, protože škála na (token, hlavu) už poskytuje dynamický rozsah, za který FP8 utrácí bity exponentu. Obě karty proto používají KV cache v INT8, i když by 5090 uměla FP8. Formát `k8v4` (hodnoty v INT4, −22 % bajtů cache) existuje jako volba, **nekvalifikovaný** — viz `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentace

| Dokument | Obsah |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **obnovení projektu na jiném stroji** (francouzsky) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | jak do sebe zapadají jednotlivé díly |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | čisté NVFP4, nebo attention+GDN v int8 po kanálech, na hybridu Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | ladění tohoto konkrétního stroje |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **co ještě není hotovo**, přečíst jako první |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | pracovní konvence pro kód (jazyk, styl, kontroly před pushnutím) |

---

<a id="resultats"></a>

## Naměřené výsledky (22. 9. 2026, RTX 5090 při 400 W, okno ≥ 20 s na měřiči energie)

Qwen3-Coder-30B-A3B v NVFP4 (experti) + INT8 (pozornost, hlava), stejný protokol pro všechny enginy (`outils/`, jedna karta, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekódování 12 sekvencí | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekódování 1 sekvence | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokenů/s** | 21 054 | 8 671 (TabbyAPI, staženo) |

¹ Erratum z 22. 9.: `serve` ve výchozím nastavení spekuluje (`--speculative ngram`, cli.py), konkurenti ne; dosud zveřejněných 380,8 t/s bylo měřeno SE SPEKULACÍ. Bez spekulace (`--speculative none`, stejný řetězec, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram je při b=1 **třetí**, za llama.cpp a vLLM. V energii zůstává před llama.cpp (0,601 oproti 0,700 J/token netto). Při b=12 spekulace nikdy není aktivní (pojistka `lot_max=2`): tato buňka už byla za rovných podmínek.

² 23. 9., stejné sezení, stejný HTTP klient (`banc-llamacpp-16-09.py` proti `acvram serve` a `vllm serve`), `-lgc 2700` explicitně nastaveno kolem každé větve, buňky střídány A V V A, ≥ 5 dávek na větev, rozdíl hlášen jen nad 2σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 při dekódování, rozvinutá redukce pozornosti): rozdíl −1,6 %, **pod 2σ: parita propustnosti**. V J/token **vLLM zůstává vpředu o 7,0 %** (nad 2σ). S 0.6.37 stejný protokol dával −4,7 %.

³ Stejné sezení a protokol jako ², bez spekulace na obou stranách: acvram 312,3 oproti vLLM 284,8 — **acvram vpředu o 9,7 % v propustnosti** (nad 2σ); J/token: **parita** (rozdíl 0,04 %, pod 2σ).

⁴ 23. 9., stejný protokol proti llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 s přepsaným routingem (+5,6 %): acvram 310,8 oproti llama.cpp 329,9 t/s — **llama.cpp vpředu o 5,8 % v propustnosti, acvram vpředu o 13,4 % v J/token** (0,598 oproti 0,691).

Dnešní propustnosti (stanice 1030, eco režim `-lgc 2700`, pipeline v provozu; hladové vzorkování zachycené v CUDA grafu, výchozí od 0.6.35). Hodnota acvram při b=12 je oficiální zapečetěná buňka (medián ze 6 prokládaných oken, hodinky na okno).

> **Erratum (23. 9. 2026).** Dosud zveřejněné srovnání s vLLM (b=12: 1 782 oproti 1 634 t/s; b=1: 290,6) stavělo acvram měřené přes HTTP proti vLLM měřenému **offline** (`LLM().generate()`), a erratum z 22. 9. mylně tvrdilo, že buňka vLLM prochází přes `vllm serve`. 23. 9.: stejný HTTP klient pro obojí, a `-lgc` nastaveno pro obojí (acvram si své nastavuje při startu, `vllm serve` ne: bez tohoto opatření vLLM běžel na ~2930 MHz oproti ~2650). Výsledek v poznámce ²: vLLM vpředu o 9,1 % při b=12.

Ráno 14. 9. byl acvram na 630 t/s a 0,619 J/token na stejné buňce: zisky pocházejí z nativního FP4 MMA Blackwellu (`mma.sync … kind::mxf4nvf4`, ×7,9 oproti bf16), MoE v seskupeném GEMM podle kbelíku dávky, routingu v jediném jádru (3677 → 1517 spuštění na krok) a úzkého GEMM na tenzorových jádrech pro projekce. Každé číslo má svou poznámku v `acvram-memoire/revue/` s předpovědí zapečetěnou před měřením, nástrojem a jeho režimem — číslo bez režimu se nezveřejňuje.

Kde je acvram vpředu: modely MLA (GLM-4.7-Flash) v nativním sm_120 NVFP4, které vLLM obsluhuje jen ve FP8 (b=1: 165,35 t/s v provozu); modely, které se nevejdou do VRAM. Dekódování jediné sekvence mezi ně nepatří: bez spekulace je tam acvram před vLLM o 9,7 % (poznámka ³), za llama.cpp o 5,8 % v propustnosti, ale před ním o 13,4 % v energii (poznámka ⁴). Při velké dávce, na MoE, který se vejde do VRAM, má vLLM paritu propustnosti při b=12 (1 995,1 oproti 2 027,0 t/s, pod 2σ, poznámka ²), ale drží náskok 7,0 % v J/token; acvram tam postoupil z 1 540 t/s (0.6.34) na 1 995 (0.6.38).

---

<a id="etat"></a>

## Stav

Verze 0.6.38. Vše běží na 5090: jádra CUDA zkompilovaná pro `sm_120a` (nativní FP4) a `sm_86`, CUDA grafy, kvantizace NVFP4/INT8/INT4, HTTP server. Pojistky na místě: karta je neviditelná pro pracovní sezení (`CUDA_VISIBLE_DEVICES` prázdné) a jen `outils/carte.sh` ji zapůjčí, pod zámkem, jednomu měření naráz; hlídač loguje každý přístup mimo zámek; měření energie pokrývající více než jednu kartu nebo trvající méně než 10 s je zneplatněno; model načtený v degradovaném režimu to řekne a nevstoupí do souboje.

4 107 testů (`pytest --collect-only -q`, minuta na procesoru; GPU testy běží jen pod `carte.sh`). Sledování práce: `acvram-memoire/` (pravidla, adresář, sešity, přehled několika set poznámek).

---

<a id="credits"></a>

## Poděkování

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, pod licencí Apache-2.0: `acvram/kernels/marlin_port/` přenáší jeho jádra Marlin (MoE i husté), s úplnou atribucí soubor po souboru v [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, tenzorová jádra FP4 Blackwellu (`sm_120`) a knihovny, na kterých tento projekt závisí.
- **PyTorch** — tenzorový engine a rozšíření C++/CUDA.

Nezávislý projekt, nespojený s ASUS, NVIDIA ani projektem vLLM.

---

<a id="licence"></a>

## Licence

[GPL-3.0 nebo novější](../LICENSE) pro kód tohoto repozitáře. `acvram/kernels/marlin_port/` obsahuje kód přenesený z [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (jádra `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), pod licencí Apache-2.0: každý soubor si zachovává svou původní hlavičku, text licence je v `LICENSE-vllm`, a seznam souborů, výchozí commit a úpravy jsou v [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Podpořte projekt

Vývoj acvram probíhá na soukromém hardwaru. Pokud je pro vás projekt užitečný:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Darujte%20kávu&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Překlady: [TRADUIRE.md](TRADUIRE.md) (francouzsky; průvodce přispěním do projektu zatím není přeložen).
