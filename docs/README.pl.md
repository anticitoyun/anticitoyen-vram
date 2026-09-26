<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licencja GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-wesprzyj-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Brama wnioskowania zgodna z API OpenAI, która traktuje pamięć jako hierarchię, daje każdemu GPU format liczbowy najlepiej dopasowany do jego krzemu i optymalizuje każdy token w dżulach tak samo jak w sekundach.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · **🇵🇱 Polski** · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Porównanie przepustowości i energii z vLLM i llama.cpp" width="720"></p>

---

## Spis treści

- [Dwie idee](#idees)
- [Szybki start](#demarrage)
- [Instalacja](#installer)
- [Co mówi `acvram plan`](#plan)
- [Przyspieszanie](#optimisations)
- [Punkty końcowe HTTP](#http)
- [Skąd biorą się liczby](#chiffres)
- [Dokumentacja](#documentation)
- [Zmierzone wyniki](#resultats)
- [Stan projektu](#etat)
- [Podziękowania](#credits)
- [Licencja](#licence)
- [Wesprzyj projekt](#soutien)

---

<a id="idees"></a>

## Dwie idee

Zaprojektowana dla konkretnej maszyny:

| | |
|---|---|
| Procesor | Intel Core i9-14900K (8 rdzeni P + 16 rdzeni E) |
| Płyta główna | ASUS ROG Maximus Z790 Dark Hero |
| Pamięć | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); obie karty w PCIe x8/x8, limit mocy 400 W / 275 W |

**Jeden format na GPU.** RTX 5090 ma rdzenie tensorowe FP4; RTX 3080 Ti nie ma ani ich, ani FP8. Wyrównanie obu do wspólnego formatu zmarnowałoby potencjał 5090. Konwerter zapisuje więc *ten sam model dwukrotnie*, w formacie, który każdy cel naprawdę potrafi wykorzystać:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| wagi | **NVFP4** — E2M1 + skala FP8 E4M3 co 16 | **INT4** — uint4 + skala i zero fp16 co 128 |
| bitów na wagę | 4,50 | 4,16 |
| względem BF16 | ×3,56 mniej | ×3,85 mniej |
| tryb obliczeń | rdzenie tensorowe FP4 | dekwantyzacja do FP16 w jądrze, rdzenie tensorowe FP16 |
| pamięć podręczna KV | INT8 | INT8 |

32 GB VRAM przy 4,5 bita na wagę mieści około **56 miliardów parametrów**, wobec 16 miliardów w BF16. Na obu kartach razem daje to w przybliżeniu **78 miliardów parametrów rezydentnych** jeszcze przed sięgnięciem do pamięci hosta.

**Pamięć to hierarchia, nie ściana.** Trzy poziomy, a planista mierzy, ile kosztuje każdy z nich, zamiast liczyć na to, że model się zmieści:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
DDR5 hosta   96 GB   ograniczone przez PCIe lub DDR
```

---

<a id="demarrage"></a>

## Szybki start

```bash
./install.sh                       # środowisko wirtualne + torch cu128 + acvram
acvram doctor                      # czy ta maszyna jest gotowa, i do czego
acvram detect                      # co tu naprawdę jest

acvram plan  ~/modele/Qwen3-32B                     # gdzie trafiłaby każda warstwa
acvram convert ~/modele/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Dowolny klient OpenAI podłącza się od razu:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Cześć"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="nieużywany")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Cześć"}])
```

---

<a id="installer"></a>

## Instalacja

Ze źródeł (dowolna platforma):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Lub z pakietu — jeden plik dołączony do każdego [wydania GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanał | Plik dołączony do wydania | Polecenie |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nazwy generuje `rpmbuild`, nie są stałe) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (lub `rpmbuild --rebuild *.src.rpm` z pliku `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Przed instalacją zweryfikuj pobrany plik względem sum dołączonych do wydania (`SHA256SUMS`, publikowanych po pojawieniu się wszystkich pozostałych plików):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip nie jest publikowany jako pakiet (nie jest budowany plik wheel): `pip install -e '.[dev]'` instaluje z klonu źródeł, tak samo jak `./install.sh`.

---

<a id="plan"></a>

## Co mówi `acvram plan`

Warto uruchomić planistę przed każdym pobraniem. Odpowiada na pytania, które decydują, czy model da się w ogóle wykorzystać na tej maszynie:

```
$ acvram plan ~/modele/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

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

Planista przeszukuje przestrzeń konfiguracji zamiast zatrzymywać się na pierwszej, która się mieści, a dwie jego decyzje są na tyle nieoczywiste, że warto je wypowiedzieć:

* **Zostawia 3080 Ti bezczynną**, gdy model mieści się na samej 5090. Etapy potoku wykonują się szeregowo: dodanie etapu 912 GB/s do potoku 1790 GB/s spowalnia dekodowanie jednego strumienia. Wymusza się to opcją `--gpus all`.
* **Zmniejsza pamięć podręczną KV, żeby utrzymać wagi w VRAM.** Każdy gigabajt oddany pamięci podręcznej to gigabajt wag zepchniętych na magistralę PCIe, a odczyt wagi przez PCIe kosztuje około trzydzieści razy więcej niż z VRAM. Na modelu 70B powyżej sam ten kompromis podnosi wynik z 2,3 do 17,8 tokena/s.

---

<a id="optimisations"></a>

## Przyspieszanie

Cztery optymalizacje, każda zweryfikowana dowodem równoważności, nie tylko stoperem: optymalizacja, która zmienia odpowiedź, jest błędem.

Warstwy liniowe NVFP4 modeli gęstych domyślnie przechodzą przez układ Marlin (+57 do +90% przepustowości przy b = 8, TTFT +2 do +4 ms wg revue/poste6-piece147-verdict-24-09.md; odwrót `ACVRAM_PROJ_MARLIN=0`, patrz [CHANGELOG.md](../CHANGELOG.md)).

### Dekodowanie spekulatywne (`--speculative`)

Dekodowanie jednego tokena przy wsadzie o rozmiarze 1 jest ograniczone pamięcią: maszyna czyta wszystkie aktywne wagi, by wyprodukować jeden token. Sprawdzenie K zaproponowanych tokenów czyta te same wagi **tylko raz**. Dwaj proponenci:

* `ngram` (domyślny) — szuka bieżącego sufiksu wcześniej w kontekście i proponuje to, co po nim następowało. Nic nie kosztuje, nie wymaga żadnego modelu. Opłacalny, gdy wyjście przepisuje wejście: edycja kodu, RAG, streszczenia.
* `draft` — mały model na drugim urządzeniu. W tym zestawie tym urządzeniem jest RTX 3080 Ti, którą planista celowo zostawia bezczynną dla każdego modelu mieszczącego się na 5090.

Istnieją też `mtp` (głowica `nextn` modelu) i `auto`; nieopłacalne w obecnym stanie i nieaktywne domyślnie — patrz `docs/ARCHITECTURE.md`.

Akceptacja jest dokładna, nie przybliżona: propozycja jest akceptowana z prawdopodobieństwem `min(1, p/q)`, a odrzucenie próbkuje ponownie z znormalizowanej dodatniej części `p - q`. Zmierzone na 40 000 losowaniach wobec celowo źle skalibrowanego szkicu, wyemitowany rozkład pozostaje w granicach 0,002 całkowitej wariacji celu — spekulacja kupuje szybkość, nigdy inną odpowiedź.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Pamięć podręczna prefiksu (aktywna domyślnie)

Bloki są adresowane *łańcuchowym* skrótem swojego wycinka tokenów: dwa żądania współdzielące polecenie systemowe współdzielą jego bloki, a drugie nie musi ich już wstępnie obliczać. Łańcuchowanie jest niezbędne: te same szesnaście tokenów w innym kontekście nie mają tych samych kluczy i wartości, a haszowanie samego wycinka obsłużyłoby pamięć podręczną jednej sekwencji cudzym żądaniem.

Zwolniony blok, którego zawartość pozostaje identyfikowalna, dołącza do kolejki LRU zamiast do listy wolnych bloków: pamięć podręczna przeżywa w ten sposób między żądaniami, nigdy nie odmawiając alokacji, którą mogłaby obsłużyć.

### Obliczenia na poziomie hosta (`--host-exec`)

Warstwa, której wagi znajdują się w RAM, może zostać skopiowana na GPU albo obliczona na miejscu. Obie ścieżki są ograniczone pamięcią i czytają te same bajty: szybsza jest ta, której magistrala jest szersza — PCIe 5.0 x16 daje około 54 GB/s, DDR5 w trybie dwukanałowym około 70 GB/s — a obliczanie na miejscu dodatkowo zostawia GPU wolne zamiast kazać mu czekać na kopię.

Opłaca się to tylko wtedy, gdy procesor czyta bezpośrednio wagi spakowane na 4 bitach. Stąd mały rdzeń C++ ze ścieżką AVX2 (`acvram_cpu.cpp`, ładowany przez ctypes, bez nagłówków Pythona ani ninja). Nawet na swojej awaryjnej gałęzi **skalarnej** bije `dequantize() @ x` o czynnik 1,44 w INT4 i 3,21 w NVFP4, ponieważ ten ostatni najpierw zapisuje 32-bitową kopię całej macierzy.

Na Mistral-Large-123B szacunek planisty wzrasta z 1,35 do 2,42 tokena/s.

### Precyzja mieszana (`--snr-floor`, domyślnie wyłączona)

Konwerter mierzy stosunek sygnału do szumu na wyjściu warstwy dla każdego tensora i może promować do szerszego formatu te, które spadną poniżej `--snr-floor`, w granicach 15% tensorów i pułapu ceny (`--promotion-cout-max`, w dodanych mebibajtach).

Próg wynosi **zero domyślnie**: nic nie jest promowane. Dekodowanie jest ograniczone przepustowością pamięci, a pomiar na `Huihui-Qwen3.8-27B` rozstrzyga — próg 25 dB kosztuje 13,4% pamięci i 10,6% przepustowości (18,50 GiB i 41,8 t/s wobec 16,02 i 46,2) za 2,0% perplexity (42,591 wobec 43,447, korpus 16 383 tokenów). `--snr-floor 25` przywraca dawne zachowanie, gdy jakość jest ważniejsza niż szybkość.

### I `acvram eval`

Stosunek sygnału do szumu i cosinus logitów to przybliżenia. `acvram eval REP [REP ...]` mierzy perplexity w oknie przesuwnym, żeby wybór formatu rozstrzygały dowody:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Punkty końcowe HTTP

| punkt końcowy | uwagi |
|---|---|
| `POST /v1/chat/completions` | strumień SSE lub pojedyncza odpowiedź; używa szablonu konwersacji modelu |
| `POST /v1/completions` | polecenie jako tekst lub jako identyfikatory tokenów |
| `POST /v1/embeddings` | uśrednione końcowe stany ukryte, znormalizowane L2, `dimensions` respektowane |
| `GET /v1/models` | plus blok `acvram`: formaty, urządzenia, pojemność pamięci podręcznej KV |
| `GET /health`, `GET /metrics` | przepustowość dekodowania, zajętość bloków KV |

Nazwy pól tych odpowiedzi pozostają w języku angielskim: to protokół OpenAI, a ich tłumaczenie zepsułoby wszystkich istniejących klientów.

---

<a id="chiffres"></a>

## Skąd biorą się liczby

Każda przytoczona powyżej wartość pochodzi z kodu tego repozytorium i jest sprawdzana przez `pytest`. Pomiary wykonane na procesorze z jądrami referencyjnymi:

| format | bitów/wagę | SNR wag | cosinus logitów vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dwa wnioski z tych pomiarów zmieniły wartości domyślne:

* **Rotacja Hadamarda pomaga INT4, a nie NVFP4.** Grupy po 128 w INT4 nie mogą wchłonąć odosobnionego kanału odstającego, więc rozłożenie wartości ekstremalnych jest warte transformaty n log n na aktywację. Bloki po 16 w NVFP4 mają już własną skalę. Stąd `--hadamard auto`, które stosuje ją tylko do INT4.
* **INT8 bije FP8 E4M3 dla pamięci podręcznej KV**, 44 dB wobec 32 dB przy tym samym rozmiarze, ponieważ skala na (token, głowicę) dostarcza już zakres dynamiczny, na który FP8 wydaje bity wykładnika. Obie karty używają więc pamięci podręcznej KV w INT8, mimo że 5090 potrafiłaby FP8. Format `k8v4` (wartości w INT4, −22% bajtów pamięci podręcznej) istnieje jako opcja, **niezakwalifikowany** — patrz `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentacja

| Dokument | Zawartość |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **wznowienie projektu na innej maszynie** (po francusku) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | jak łączą się elementy |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | czyste NVFP4 albo attention+GDN w int8 na kanał, na hybrydzie Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | dostrojenie tej konkretnej maszyny |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **co jeszcze nie jest zrobione**, przeczytać w pierwszej kolejności |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | konwencje pracy nad kodem (język, styl, kontrole przed wypchnięciem) |

---

<a id="resultats"></a>

## Zmierzone wyniki (22.09.2026, RTX 5090 przy 400 W, okno ≥ 20 s na liczniku energii)

Qwen3-Coder-30B-A3B w NVFP4 (eksperci) + INT8 (uwaga, głowica), ten sam protokół dla wszystkich silników (`outils/`, jedna karta, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekodowanie 12 sekwencji | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekodowanie 1 sekwencji | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokenów/s** | 21 054 | 8 671 (TabbyAPI, wycofany) |

¹ Erratum z 22.09: `serve` domyślnie spekuluje (`--speculative ngram`, cli.py), konkurenci nie; opublikowane dotąd 380,8 t/s było mierzone Z SPEKULACJĄ. Bez spekulacji (`--speculative none`, ten sam łańcuch, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram jest **trzeci** przy b=1, za llama.cpp i vLLM. W energii pozostaje przed llama.cpp (0,601 wobec 0,700 J/token netto). Przy b=12 spekulacja nigdy nie jest aktywna (zabezpieczenie `lot_max=2`): ta komórka była już na równych zasadach.

² 23.09, ta sama sesja, ten sam klient HTTP (`banc-llamacpp-16-09.py` wobec `acvram serve` i `vllm serve`), `-lgc 2700` ustawione jawnie wokół każdego ramienia, komórki naprzemienne A V V A, ≥ 5 partii na ramię, różnica zgłaszana tylko powyżej 2σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 przy dekodowaniu, rozwinięta redukcja uwagi): różnica −1,6%, **poniżej 2σ: parytet przepustowości**. W J/token **vLLM pozostaje przed o 7,0%** (powyżej 2σ). Z 0.6.37 ten sam protokół dawał −4,7%.

³ Ta sama sesja i protokół co ², bez spekulacji po obu stronach: acvram 312,3 wobec vLLM 284,8 — **acvram przed o 9,7% w przepustowości** (powyżej 2σ); J/token: **parytet** (różnica 0,04%, poniżej 2σ).

⁴ 23.09, ten sam protokół wobec llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 z przepisanym routingiem (+5,6%): acvram 310,8 wobec llama.cpp 329,9 t/s — **llama.cpp przed o 5,8% w przepustowości, acvram przed o 13,4% w J/token** (0,598 wobec 0,691).

Dzisiejsze przepustowości (stanowisko 1030, tryb eco `-lgc 2700`, potok w usłudze; zachłanne próbkowanie przechwycone w grafie CUDA, domyślne od 0.6.35). Wynik acvram przy b=12 to oficjalna zapieczętowana komórka (mediana z 6 przeplatanych okien, zegar na okno).

> **Erratum (23.09.2026).** Opublikowane dotąd porównanie z vLLM (b=12: 1 782 wobec 1 634 t/s; b=1: 290,6) zestawiało acvram mierzone przez HTTP z vLLM mierzonym **w trybie offline** (`LLM().generate()`), a erratum z 22.09 błędnie twierdziło, że komórka vLLM przechodziła przez `vllm serve`. 23.09: ten sam klient HTTP dla obu, i `-lgc` ustawione dla obu (acvram ustawia swój przy starcie, `vllm serve` nie: bez tego zabezpieczenia vLLM działał przy ~2930 MHz wobec ~2650). Wynik w przypisie ²: vLLM przed o 9,1% przy b=12.

Rano 14.09 acvram był na 630 t/s i 0,619 J/token na tej samej komórce: zyski pochodzą z natywnego MMA FP4 Blackwella (`mma.sync … kind::mxf4nvf4`, ×7,9 wobec bf16), MoE w zgrupowanym GEMM na kubełek partii, routingu w jednym jądrze (3677 → 1517 uruchomień na krok) i wąskiego GEMM na rdzeniach tensorowych dla projekcji. Każda liczba ma swoją notatkę w `acvram-memoire/revue/` z prognozą zapieczętowaną przed pomiarem, instrumentem i jego trybem — liczba bez trybu nie jest publikowana.

Gdzie acvram jest przed: modele MLA (GLM-4.7-Flash) w natywnym NVFP4 sm_120, które vLLM obsługuje tylko w FP8 (b=1: 165,35 t/s w usłudze); modele, które nie mieszczą się w VRAM. Dekodowanie pojedynczej sekwencji do nich nie należy: bez spekulacji acvram jest tam przed vLLM o 9,7% (przypis ³), za llama.cpp o 5,8% w przepustowości, ale przed nim o 13,4% w energii (przypis ⁴). Przy dużej partii, na MoE, który mieści się w VRAM, vLLM ma parytet przepustowości przy b=12 (1 995,1 wobec 2 027,0 t/s, poniżej 2σ, przypis ²), ale utrzymuje przewagę 7,0% w J/token; acvram progresował tam z 1 540 t/s (0.6.34) do 1 995 (0.6.38).

---

<a id="etat"></a>

## Stan projektu

Wersja 0.6.38. Wszystko działa na 5090: jądra CUDA skompilowane dla `sm_120a` (natywne FP4) i `sm_86`, grafy CUDA, kwantyzacja NVFP4/INT8/INT4, serwer HTTP. Zabezpieczenia na miejscu: karta jest niewidoczna dla sesji roboczych (`CUDA_VISIBLE_DEVICES` puste), a tylko `outils/carte.sh` udostępnia ją, pod blokadą, jednemu pomiarowi naraz; obserwator loguje każdy dostęp poza blokadą; pomiar energii obejmujący więcej niż jedną kartę lub trwający poniżej 10 s jest unieważniany; model załadowany w trybie zdegradowanym mówi to i nie wchodzi do pojedynku.

4 107 testów (`pytest --collect-only -q`, minuta na procesorze; testy GPU działają tylko pod `carte.sh`). Śledzenie pracy: `acvram-memoire/` (zasady, katalog, notatniki, przegląd kilkuset notatek).

---

<a id="credits"></a>

## Podziękowania

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, na licencji Apache-2.0: `acvram/kernels/marlin_port/` przenosi jego jądra Marlin (MoE i gęste), z pełną atrybucją plik po pliku w [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, rdzenie tensorowe FP4 Blackwella (`sm_120`) i biblioteki, od których zależy ten projekt.
- **PyTorch** — silnik tensorowy i rozszerzenia C++/CUDA.

Projekt niezależny, niezwiązany z ASUS, NVIDIA ani z projektem vLLM.

---

<a id="licence"></a>

## Licencja

[GPL-3.0 lub nowsza](../LICENSE) dla kodu tego repozytorium. `acvram/kernels/marlin_port/` zawiera kod przeniesiony z [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (jądra `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), na licencji Apache-2.0: każdy plik zachowuje swój oryginalny nagłówek, tekst licencji znajduje się w `LICENSE-vllm`, a lista plików, commit źródłowy i modyfikacje są w [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Wesprzyj projekt

Rozwój acvram prowadzony jest na prywatnym sprzęcie. Jeśli projekt jest dla ciebie przydatny:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Postaw%20kawę&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Tłumaczenia: [TRADUIRE.md](TRADUIRE.md) (po francusku; przewodnik po wkładzie do projektu nie jest jeszcze przetłumaczony).
