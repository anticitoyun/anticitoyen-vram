<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Podpořit: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Inferenční brána kompatibilní s API OpenAI, která zachází s pamětí jako s
hierarchií a dává každé GPU číselný formát, který její křemík čte nejlépe.

Navržena pro jeden konkrétní stroj:

| | |
|---|---|
| Procesor | Intel Core i9-14900K (8 jader P + 16 jader E) |
| Základní deska | ASUS ROG Maximus Z790 Dark Hero |
| Paměť | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Systém | Ubuntu 26.04 LTS (CUDA 13); obě karty na PCIe x8/x8, omezené na 400 W / 275 W |

## Dvě myšlenky

**Jeden formát na GPU.** RTX 5090 má tensorová jádra FP4; RTX 3080 Ti je
nemá, a nemá ani FP8. Srovnat obě na společný formát by 5090 promrhalo.
Převodník proto zapisuje *tentýž model dvakrát*, ve formátu, který každý cíl
skutečně umí využít:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| váhy | **NVFP4** — E2M1 + měřítko FP8 E4M3 každých 16 | **INT4** — uint4 + měřítko a nula fp16 každých 128 |
| bitů na váhu | 4,50 | 4,16 |
| oproti BF16 | ×3,56 menší | ×3,85 menší |
| režim výpočtu | tensorová jádra FP4 | dekvantizováno na FP16 v jádře, tensorová jádra FP16 |
| KV cache | INT8 | INT8 |

32 GB VRAM při 4,5 bitu na váhu pojme zhruba **56 miliard parametrů**, oproti
16 miliardám v BF16. Na obou kartách to dělá přibližně **78 miliard
rezidentních parametrů**, ještě než se sáhne na operační paměť.

**Paměť je hierarchie, ne zeď.** Tři patra, a plánovač měří, kolik které
stojí, místo aby doufal, že se model vejde:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Rychlý start

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Poté se připojí libovolný klient OpenAI:

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

## Co říká `acvram plan`

Plánovač stojí za spuštění před jakýmkoli stahováním. Odpovídá na otázky,
které rozhodují, zda je model na tomto stroji použitelný:

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

Prohledává prostor konfigurací, místo aby si nechal první, která se vejde, a
dvě z jeho rozhodnutí jsou dost neintuitivní na to, aby si zasloužila
vyslovit:

* **Nechává 3080 Ti nevyužitou**, když se model vejde na samotnou 5090. Části
  pipeline běží sériově: přidání stupně 912 GB/s do pipeline 1790 GB/s
  zpomaluje dekódování jednoho proudu. Vynutí se pomocí `--gpus all`.
* **Zmenšuje KV cache, aby váhy zůstaly ve VRAM.** Každý gigabajt daný cache
  je gigabajt vah vytlačený na sběrnici PCIe a čtení váhy přes PCIe stojí
  zhruba třicetkrát tolik co z VRAM. U 70B výše jen toto rozhodnutí posune z
  2,3 na 17,8 tokenů/s.

## Rychlost

Čtyři optimalizace, každá ověřená důkazem ekvivalence, a ne jen stopkami:
optimalizace, která mění odpověď, je chyba.

### Spekulativní dekódování (`--speculative`)

Dekódování jednoho tokenu s dávkou velikosti 1 je omezeno pamětí: stroj čte
všechny aktivní váhy, aby vyrobil jediný token. Ověření K navržených tokenů
čte tytéž váhy **jen jednou**. Dva navrhovatelé:

* `ngram` (výchozí) — hledá aktuální příponu dříve v kontextu a navrhuje, co
  následovalo. Nic nestojí, nevyžaduje žádný model. Vyplatí se, když výstup
  opisuje vstup: editace kódu, RAG, shrnutí.
* `draft` — malý model na druhém zařízení. Na této sestavě je tím zařízením
  RTX 3080 Ti, kterou plánovač záměrně nechává nečinnou pro každý model, jenž
  se vejde na 5090.

Přijetí je přesné, ne přibližné: návrh je přijat s pravděpodobností
`min(1, p/q)` a odmítnutí znovu vzorkuje z normalizované kladné části
`p - q`. Měřeno na 40 000 tazích proti záměrně špatně kalibrovanému náčrtu
zůstává vydávané rozdělení do 0,002 totální variace od cíle — spekulace
kupuje rychlost, nikdy jinou odpověď.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache prefixů (ve výchozím stavu aktivní)

Bloky jsou adresovány *řetězeným* hashem svého úseku tokenů: dva požadavky
sdílející systémový pokyn sdílejí jeho bloky a druhý je už nemusí
předpočítávat. Řetězení je nezbytné: stejných šestnáct tokenů v jiném
kontextu neobsahuje stejné klíče a hodnoty a hashování samotného úseku by
podstrčilo cache jedné sekvence druhé.

Uvolněný blok, jehož obsah zůstává rozpoznatelný, jde do fronty LRU místo do
seznamu volných bloků: cache tak přežívá mezi požadavky, aniž by kdy odmítla
alokaci, kterou mohla obsloužit.

### Výpočet na hostitelském patře (`--host-exec`)

Vrstva, jejíž váhy sídlí v RAM, může být zkopírována na GPU nebo spočítána na
místě. Obě cesty jsou omezené pamětí a čtou tytéž bajty: rychlejší je ta se
širší sběrnicí — PCIe 5.0 x16 dává asi 54 GB/s, DDR5 ve dvoukanálu asi
70 GB/s — a výpočet na místě navíc nechává GPU volnou, místo aby čekala na
kopii.

To se vyplatí jen tehdy, když procesor čte 4bitově zabalené váhy přímo. Odtud
malé jádro v C++ s cestou AVX2 (`acvram_cpu.cpp`, načtené přes ctypes, bez
hlaviček Pythonu a bez ninja). I ve své **skalární** záložní větvi poráží
`dequantize() @ x` faktorem 1,44 v INT4 a 3,21 v NVFP4, protože to druhé
nejprve zapíše 32bitovou kopii celé matice.

U Mistral-Large-123B stoupá odhad plánovače z 1,35 na 2,42 tokenů/s.

### Smíšená přesnost (`--snr-floor`, ve výchozím stavu vypnutá)

Převodník měří odstup signálu od šumu na výstupu každé vrstvy pro každý tenzor
a může povýšit do širšího formátu ty, které klesnou pod `--snr-floor`, v mezích
15 % tenzorů a cenového stropu (`--promotion-cout-max`, v přidaných
mebibajtech).

Práh je **ve výchozím stavu nula**: nic se nepovyšuje. Dekódování je omezeno
propustností paměti a měření na `Huihui-Qwen3.8-27B` rozhoduje — práh 25 dB
stojí 13,4 % paměti a 10,6 % propustnosti (18,50 GiB a 41,8 t/s oproti 16,02 a
46,2) za 2,0 % perplexity (42,591 oproti 43,447, korpus 16 383 tokenů).
`--snr-floor 25` obnoví staré chování, když má kvalita přednost před
rychlostí.

### A `acvram eval`

Odstup signálu od šumu a kosinus logitů jsou aproximace.
`acvram eval ADR [ADR ...]` měří perplexitu s klouzavým oknem, aby se volba
formátu rozhodovala na důkazech:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Koncové body HTTP

| koncový bod | poznámky |
|---|---|
| `POST /v1/chat/completions` | proud SSE nebo jediná odpověď; používá konverzační šablonu modelu |
| `POST /v1/completions` | prompt jako text nebo jako identifikátory tokenů |
| `POST /v1/embeddings` | zprůměrované konečné skryté stavy, normalizované L2, `dimensions` respektováno |
| `GET /v1/models` | plus blok `acvram`: formáty, zařízení, kapacita KV cache |
| `GET /health`, `GET /metrics` | propustnost dekódování, obsazenost bloků KV |

Názvy polí těchto odpovědí zůstávají anglicky: je to protokol OpenAI a jejich
překlad by rozbil všechny existující klienty.

## Odkud čísla pocházejí

Každou výše uvedenou hodnotu vytváří kód tohoto repozitáře a ověřuje
`pytest`. Měření provedena na procesoru s referenčními jádry:

| formát | bitů/váhu | SNR vah | kosinus logitů vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dvě zjištění z těchto měření změnila výchozí hodnoty:

* **Hadamardova rotace pomáhá INT4 a ne NVFP4.** Skupiny po 128 u INT4
  nedokážou pohltit osamocený odlehlý kanál, takže rozprostření extrémních
  hodnot stojí za transformaci n log n na aktivaci. Bloky po 16 u NVFP4 už
  nesou vlastní měřítko. Odtud `--hadamard auto`, které ji použije jen na
  INT4.
* **INT8 poráží FP8 E4M3 pro KV cache**, 44 dB oproti 32 dB při stejné
  velikosti, protože měřítko na (token, hlavu) už poskytuje dynamický rozsah,
  na který FP8 utrácí bity exponentu. Obě karty proto používají KV cache v
  INT8, i když by 5090 FP8 uměla.

## Dokumentace

* [`REPRISE.md`](../REPRISE.md) — **převzetí projektu na jiném stroji**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — jak do sebe díly zapadají
* [`docs/MATERIEL.md`](MATERIEL.md) — ladění tohoto konkrétního stroje
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **co není hotové**, číst jako první
* [`CONVENTIONS.md`](../CONVENTIONS.md) — pracovní konvence pro kód (jazyk, styl, kontroly před pushem)

## Naměřené výsledky (22. 9. 2026, RTX 5090 při 400 W, režim ≥ 20 s na měřiči energie)

Qwen3-Coder-30B-A3B v NVFP4 (experti) + INT8 (attention, hlava), stejný
protokol pro všechny motory (`outils/`, jedna karta, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekódování 12 sekvencí | **1 634 t/s** | 1 782 t/s | — |
| dekódování 1 sekvence | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokenů/s** | 21 054 | 8 671 (TabbyAPI, staženo) |

Propustnost dne (stanice 1030, úsporný režim `-lgc 2700`, pipeline v provozu;
hladové vzorkování zachycené v grafu CUDA, ve verzi 0.6.35 zapnuté ve výchozím
nastavení). b=12 je oficiální zapečetěná buňka (medián ze 6 prokládaných oken).

> **Erratum (22. 9. 2026).** První zveřejnění verze 0.6.35 odvozovalo
> „+1,84 % před vLLM“ z referenční hodnoty vLLM 1 596 t/s z 21. 9., která
> pocházela z **offline generování** (`LLM().generate()`), **nesrovnatelného se
> serverem**: bez průběžného plánování, bez cesty `acvram serve`. Opraveno
> 22. 9. střídavou buňkou A/V (A1 V1 A2 V2 A3 V3) proti **`vllm serve`** (HTTP),
> stejná karta a stejná cesta jako `acvram serve`: vLLM medián **1 782 t/s**.
> Při srovnatelném měření je **acvram (1 634 t/s) ZA vLLM přibližně o 8 % při
> b=12**, ne před ním. J/token při stejných hodinách se stále znovu měří.

Ráno 14. 9. byl acvram ve stejné buňce na 630 t/s a 0,619 J/token: zisky
pocházejí z nativního FP4 MMA Blackwellu (`mma.sync … kind::mxf4nvf4`, ×7,9
oproti bf16), z MoE jako seskupeného GEMM po kbelících dávky, ze směrování v
jediném jádře (3 677 → 1 517 spuštění na krok) a z úzkého GEMM na
tensorových jádrech pro projekce. Každé číslo má svou poznámku v
`acvram-memoire/revue/` s předpovědí zapečetěnou před měřením, nástrojem a
jeho režimem — číslo bez režimu se nezveřejňuje.

Kde je acvram napřed: modely MLA (GLM-4.7-Flash) v nativním NVFP4 sm_120,
které vLLM obsluhuje jen v FP8 (b=1: 165,35 t/s v provozu); modely, které se
nevejdou do VRAM; a dekódování jediné sekvence (b=1: 380,8 t/s oproti 290,6 u
vLLM). U velkých dávek naopak, u MoE, který se do VRAM vejde, zůstává vLLM
napřed při b=12 (1 782 oproti 1 634 t/s, viz erratum); acvram se zde zlepšil
(1 540 v 0.6.34 → 1 634), aniž by se dostal napřed. Rozdíl v energii je třeba
znovu změřit.

## Stav

Verze 0.6.35. Vše běží na 5090: jádra CUDA zkompilovaná pro `sm_120a`
(nativní FP4) a `sm_86`, grafy CUDA, kvantizace NVFP4/INT8/INT4, HTTP server.
Zábrany na místě: karta je pro pracovní relace neviditelná
(`CUDA_VISIBLE_DEVICES` prázdné) a jen `outils/carte.sh` ji pod zámkem
půjčuje jednomu měření najednou; hlídač loguje každý přístup mimo zámek;
měření energie pokrývající více než jednu kartu nebo kratší než 10 s je
neplatné; model načtený v degradovaném režimu to řekne a do souboje
nevstupuje.

640 testů (`pytest -q`, minuta na procesoru; testy GPU běží jen pod
`carte.sh`). Sledování práce: `acvram-memoire/` (pravidla, adresář, sešity,
revize 180 poznámek).

## Podpořit

Vývoj acvramu probíhá na osobním hardwaru. Pokud je vám projekt užitečný:
**Podpořit: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licence

GPL-3.0 nebo novější.
