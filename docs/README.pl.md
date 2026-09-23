<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Wesprzyj: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Brama inferencyjna zgodna z API OpenAI, która traktuje pamięć jak hierarchię
i daje każdemu GPU format liczbowy, który jego krzem odczytuje najlepiej.

Zaprojektowana dla jednej konkretnej maszyny:

| | |
|---|---|
| Procesor | Intel Core i9-14900K (8 rdzeni P + 16 rdzeni E) |
| Płyta główna | ASUS ROG Maximus Z790 Dark Hero |
| Pamięć | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); obie karty na PCIe x8/x8, ograniczone do 400 W / 275 W |

## Dwa pomysły

**Jeden format na GPU.** RTX 5090 ma rdzenie tensorowe FP4; RTX 3080 Ti ich
nie ma, nie ma też FP8. Sprowadzenie obu do wspólnego formatu zmarnowałoby
5090. Konwerter zapisuje więc *ten sam model dwa razy*, w formacie, który dany
cel potrafi naprawdę wykorzystać:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| wagi | **NVFP4** — E2M1 + skala FP8 E4M3 co 16 | **INT4** — uint4 + skala i zero fp16 co 128 |
| bity na wagę | 4,50 | 4,16 |
| względem BF16 | ×3,56 mniejszy | ×3,85 mniejszy |
| tryb obliczeń | rdzenie tensorowe FP4 | dekwantyzacja do FP16 w jądrze, rdzenie tensorowe FP16 |
| pamięć podręczna KV | INT8 | INT8 |

32 GB VRAM przy 4,5 bita na wagę mieści około **56 miliardów parametrów**,
wobec 16 miliardów w BF16. Na obu kartach daje to w przybliżeniu
**78 miliardów parametrów rezydentnych**, zanim w ogóle sięgnie się po RAM.

**Pamięć to hierarchia, nie ściana.** Trzy poziomy, a planista mierzy, ile
kosztuje każdy z nich, zamiast liczyć, że model się zmieści:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Szybki start

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Następnie podłącza się dowolny klient OpenAI:

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

## Co mówi `acvram plan`

Planistę warto uruchomić przed każdym pobraniem. Odpowiada na pytania, które
rozstrzygają, czy model nadaje się do użytku na tej maszynie:

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

Przeszukuje przestrzeń konfiguracji, zamiast zatrzymać się na pierwszej,
która się mieści, a dwie z jego decyzji są na tyle sprzeczne z intuicją, że
warto je wypowiedzieć:

* **Zostawia 3080 Ti nieużywaną**, gdy model mieści się na samej 5090.
  Segmenty potoku wykonują się szeregowo: dodanie etapu 912 GB/s do potoku
  1790 GB/s spowalnia dekodowanie pojedynczego strumienia. Wymusza się je
  przez `--gpus all`.
* **Zmniejsza pamięć podręczną KV, by zatrzymać wagi w VRAM.** Każdy gigabajt
  oddany pamięci podręcznej to gigabajt wag wypchnięty na magistralę PCIe, a
  odczyt wagi przez PCIe kosztuje około trzydzieści razy więcej niż z VRAM. Na
  powyższym 70B sam ten kompromis przenosi z 2,3 na 17,8 tokenów/s.

## Szybkość

Cztery optymalizacje, każda zweryfikowana dowodem równoważności, a nie tylko
stoperem: optymalizacja, która zmienia odpowiedź, jest błędem.

### Dekodowanie spekulatywne (`--speculative`)

Dekodowanie jednego tokenu z partią rozmiaru 1 jest ograniczone pamięcią:
maszyna czyta wszystkie aktywne wagi, by wytworzyć jeden token. Weryfikacja K
proponowanych tokenów czyta te same wagi **tylko raz**. Dwaj proponujący:

* `ngram` (domyślny) — szuka bieżącego sufiksu wcześniej w kontekście i
  proponuje to, co po nim następowało. Nic nie kosztuje, nie wymaga modelu.
  Opłaca się, gdy wyjście przepisuje wejście: edycja kodu, RAG, streszczenie.
* `draft` — mały model na drugim urządzeniu. W tym zestawie tym urządzeniem
  jest RTX 3080 Ti, którą planista celowo pozostawia bezczynną dla każdego
  modelu mieszczącego się na 5090.

Akceptacja jest dokładna, nie przybliżona: propozycja jest przyjmowana z
prawdopodobieństwem `min(1, p/q)`, a odrzucenie losuje ponownie z
znormalizowanej dodatniej części `p - q`. Zmierzone na 40 000 losowań wobec
celowo źle skalibrowanego szkicu: emitowany rozkład pozostaje w odległości
0,002 wariacji całkowitej od celu — spekulacja kupuje szybkość, nigdy inną
odpowiedź.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Pamięć podręczna prefiksów (domyślnie włączona)

Bloki są adresowane *łańcuchowym* skrótem swojego wycinka tokenów: dwa
zapytania dzielące instrukcję systemową dzielą jej bloki, a drugie nie musi
już ich wyliczać. Łańcuchowanie jest niezbędne: te same szesnaście tokenów w
innym kontekście nie zawiera tych samych kluczy i wartości, a skrót samego
wycinka podałby pamięć podręczną jednej sekwencji innej.

Zwolniony blok, którego zawartość pozostaje rozpoznawalna, trafia do kolejki
LRU zamiast na listę wolnych bloków: pamięć podręczna przeżywa tak między
zapytaniami, nigdy nie odmawiając alokacji, którą mogłaby obsłużyć.

### Obliczenia na poziomie hosta (`--host-exec`)

Warstwa, której wagi rezydują w RAM, może być skopiowana na GPU albo
policzona na miejscu. Obie ścieżki są ograniczone pamięcią i czytają te same
bajty: szybsza jest ta z szerszą magistralą — PCIe 5.0 x16 daje około
54 GB/s, DDR5 w trybie dwukanałowym około 70 GB/s — a liczenie na miejscu
zostawia ponadto GPU wolne, zamiast kazać mu czekać na kopię.

To się opłaca tylko wtedy, gdy procesor czyta bezpośrednio wagi spakowane do
4 bitów. Stąd małe jądro C++ ze ścieżką AVX2 (`acvram_cpu.cpp`, ładowane
przez ctypes, bez nagłówków Pythona i bez ninja). Nawet w swojej **skalarnej**
gałęzi zapasowej bije `dequantize() @ x` 1,44 raza w INT4 i 3,21 raza w
NVFP4, bo to ostatnie najpierw zapisuje 32-bitową kopię całej macierzy.

Na Mistral-Large-123B szacunek planisty rośnie z 1,35 do 2,42 tokenów/s.

### Mieszana precyzja (`--snr-floor`, domyślnie wyłączona)

Konwerter mierzy stosunek sygnału do szumu na wyjściu każdej warstwy dla
każdego tensora i może awansować do szerszego formatu te, które spadają
poniżej `--snr-floor`, w granicach 15 % tensorów i pułapu cenowego
(`--promotion-cout-max`, w dodanych mebibajtach).

Próg wynosi **domyślnie zero**: nic nie jest awansowane. Dekodowanie
ogranicza przepustowość pamięci, a pomiar na `Huihui-Qwen3.8-27B` rozstrzyga —
próg 25 dB kosztuje 13,4 % pamięci i 10,6 % przepustowości (18,50 GiB i
41,8 t/s wobec 16,02 i 46,2) za 2,0 % perpleksji (42,591 wobec 43,447, korpus
16 383 tokenów). `--snr-floor 25` przywraca dawne zachowanie, gdy jakość jest
ważniejsza od szybkości.

### I `acvram eval`

Stosunek sygnału do szumu i kosinus logitów to przybliżenia.
`acvram eval KAT [KAT ...]` mierzy perpleksję w oknie przesuwnym, by wybór
formatu rozstrzygać na dowodach:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Punkty końcowe HTTP

| punkt końcowy | uwagi |
|---|---|
| `POST /v1/chat/completions` | strumień SSE lub pojedyncza odpowiedź; używa szablonu rozmowy modelu |
| `POST /v1/completions` | prompt jako tekst lub identyfikatory tokenów |
| `POST /v1/embeddings` | uśrednione końcowe stany ukryte, znormalizowane L2, `dimensions` respektowane |
| `GET /v1/models` | plus blok `acvram`: formaty, urządzenia, pojemność pamięci podręcznej KV |
| `GET /health`, `GET /metrics` | przepustowość dekodowania, zajętość bloków KV |

Nazwy pól w tych odpowiedziach pozostają angielskie: to protokół OpenAI, a
ich tłumaczenie zepsułoby wszystkich istniejących klientów.

## Skąd biorą się liczby

Każda wartość przytoczona powyżej pochodzi z kodu tego repozytorium i jest
sprawdzana przez `pytest`. Pomiary wykonane na procesorze z jądrami
referencyjnymi:

| format | bity/wagę | SNR wag | kosinus logitów vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dwa spostrzeżenia z tych pomiarów zmieniły wartości domyślne:

* **Obrót Hadamarda pomaga INT4, a nie NVFP4.** Grupy po 128 w INT4 nie
  potrafią wchłonąć pojedynczego odstającego kanału, więc rozproszenie
  wartości skrajnych jest warte transformaty n log n na aktywację. Bloki po 16
  w NVFP4 niosą już własną skalę. Stąd `--hadamard auto`, który stosuje go
  tylko do INT4.
* **INT8 bije FP8 E4M3 w pamięci podręcznej KV**, 44 dB wobec 32 dB przy tym
  samym rozmiarze, bo skala na (token, głowicę) daje już zakres dynamiczny, na
  który FP8 wydaje bity wykładnika. Obie karty używają więc pamięci
  podręcznej KV w INT8, choć 5090 umiałaby FP8.

## Dokumentacja

* [`REPRISE.md`](../REPRISE.md) — **wznowienie projektu na innej maszynie**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — jak części pasują do siebie
* [`docs/MATERIEL.md`](MATERIEL.md) — strojenie tej konkretnej maszyny
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **czego nie zrobiono**, czytać najpierw
* [`CONVENTIONS.md`](../CONVENTIONS.md) — konwencje pracy nad kodem (język, styl, kontrole przed wypchnięciem)

## Zmierzone wyniki (22.09.2026, RTX 5090 przy 400 W, reżim ≥ 20 s na liczniku energii)

Qwen3-Coder-30B-A3B w NVFP4 (eksperci) + INT8 (uwaga, głowica), ten sam
protokół dla wszystkich silników (`outils/`, jedna karta, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekodowanie 12 sekwencji | **1 634 t/s** | 1 782 t/s | — |
| dekodowanie 1 sekwencji | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokenów/s** | 21 054 | 8 671 (TabbyAPI, wycofane) |

Przepustowość dnia (stanowisko 1030, reżim eko `-lgc 2700`, potok w działaniu;
próbkowanie zachłanne ujęte w grafie CUDA, domyślne od 0.6.35). b=12 to
zapieczętowana oficjalna komórka (mediana z 6 przeplatanych okien, zegar na
okno).

> **Erratum (22.09.2026).** Pierwsza publikacja 0.6.35 wyciągała « +1,84 %
> przed vLLM » z odniesienia vLLM 1 596 t/s z 21.09, które pochodziło z
> **generacji offline** (`LLM().generate()`), **nieporównywalnej z serwerem**:
> brak ciągłego szeregowania, nie ścieżka `acvram serve`. Poprawione 22.09
> naprzemienną komórką A/V (A1 V1 A2 V2 A3 V3) wobec **`vllm serve`** (HTTP), ta
> sama karta i ta sama ścieżka co `acvram serve`: mediana vLLM **1 782 t/s**.
> Przy porównywalnym pomiarze **acvram (1 634 t/s) jest ZA vLLM o około 8 % przy
> b=12**, nie przed. J/token przy równym zegarze pozostaje w ponownym pomiarze.

Rankiem 14.09 acvram miał w tej samej komórce 630 t/s i 0,619 J/token: zyski
pochodzą z natywnego MMA FP4 Blackwella (`mma.sync … kind::mxf4nvf4`, ×7,9
względem bf16), z MoE jako GEMM grupowanego według kubełka partii, z
routingu w jednym jądrze (3 677 → 1 517 uruchomień na krok) i z wąskiego GEMM
na rdzeniach tensorowych dla projekcji. Każda liczba ma swoją notatkę w
`acvram-memoire/revue/` z prognozą zapieczętowaną przed pomiarem,
instrumentem i jego reżimem — liczby bez reżimu się nie publikuje.

Gdzie acvram jest z przodu: modele MLA (GLM-4.7-Flash) w natywnym NVFP4
sm_120, które vLLM serwuje tylko w FP8 (b=1: 165,35 t/s w działaniu); modele
niemieszczące się w VRAM; oraz dekodowanie pojedynczej sekwencji (b=1: 380,8 t/s
wobec 290,6 dla vLLM). Przy dużej partii natomiast, na MoE mieszczącym się w
VRAM, vLLM pozostaje z przodu przy b=12 (1 782 wobec 1 634 t/s, por. erratum);
acvram poczynił postęp (1 540 w 0.6.34 → 1 634), nie wychodząc na prowadzenie.
Różnicę w energii trzeba zmierzyć ponownie.

## Stan

Wersja 0.6.35. Wszystko działa na 5090: jądra CUDA skompilowane dla `sm_120a`
(natywne FP4) i `sm_86`, grafy CUDA, kwantyzacja NVFP4/INT8/INT4, serwer
HTTP. Zabezpieczenia na miejscu: karta jest niewidoczna dla sesji roboczych
(`CUDA_VISIBLE_DEVICES` puste) i tylko `outils/carte.sh` wypożycza ją, pod
blokadą, jednemu pomiarowi naraz; strażnik loguje każdy dostęp poza blokadą;
pomiar energii obejmujący więcej niż jedną kartę lub krótszy niż 10 s jest
unieważniany; model załadowany w trybie zdegradowanym mówi o tym i nie
staje do pojedynku.

640 testów (`pytest -q`, minuta na procesorze; testy GPU uruchamiają się
tylko pod `carte.sh`). Śledzenie pracy: `acvram-memoire/` (reguły, spis,
zeszyty, przegląd 180 notatek).

## Wesprzyj

acvram jest rozwijany na prywatnym sprzęcie. Jeśli projekt jest dla Ciebie
użyteczny: **Wesprzyj: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licencja

GPL-3.0 lub nowsza.
