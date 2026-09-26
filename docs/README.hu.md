<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="GPL-3.0-or-later licenc"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-t%C3%A1mogatom-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Egy OpenAI API-kompatibilis következtetési átjáró, amely a memóriát hierarchiaként kezeli, minden GPU-nak azt a számformátumot adja, amit a szilíciuma a legjobban ért, és minden tokent joule-ban optimalizál, nem csak másodpercben.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · **🇭🇺 Magyar** · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Átviteli sebesség és energiafogyasztás összehasonlítása a vLLM-mel és a llama.cpp-vel" width="720"></p>

---

## Tartalom

- [A két alapötlet](#idees)
- [Gyors kezdés](#demarrage)
- [Telepítés](#installer)
- [Mit mond az `acvram plan`](#plan)
- [Gyorsítás](#optimisations)
- [HTTP végpontok](#http)
- [Honnan jönnek a számok](#chiffres)
- [Dokumentáció](#documentation)
- [Mért eredmények](#resultats)
- [Állapot](#etat)
- [Köszönet](#credits)
- [Licenc](#licence)
- [Támogasd a projektet](#soutien)

---

<a id="idees"></a>

## A két alapötlet

Egy konkrét géphez tervezve:

| | |
|---|---|
| Processzor | Intel Core i9-14900K (8 P-mag + 16 E-mag) |
| Alaplap | ASUS ROG Maximus Z790 Dark Hero |
| Memória | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Rendszer | Ubuntu 26.04 LTS (CUDA 13); mindkét kártya PCIe x8/x8-on, 400 W / 275 W-ra korlátozva |

**Egy formátum GPU-nként.** Az RTX 5090-nek van FP4 tenzormagja; az RTX 3080 Ti-nek se ilyen, se FP8-ja nincs. A kettőt közös formátumra igazítani az 5090-et fosztaná meg a potenciáljától. A konverter tehát *kétszer írja ki ugyanazt a modellt*, olyan formátumban, amit az adott célgép valóban ki tud használni:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| súlyok | **NVFP4** — E2M1 + FP8 E4M3 skála 16-onként | **INT4** — uint4 + fp16 skála és nulla 128-anként |
| bit/súly | 4,50 | 4,16 |
| BF16-hoz képest | ×3,56-szor kisebb | ×3,85-ször kisebb |
| számítási mód | FP4 tenzormagok | dekvantálás FP16-ra a kernelben, FP16 tenzormagok |
| KV gyorsítótár | INT8 | INT8 |

32 GB VRAM 4,5 bit/súly mellett kb. **56 milliárd paramétert** fér el, szemben a BF16-ban elférő 16 milliárddal. A két kártyán együtt ez nagyjából **78 milliárd rezidens paramétert** jelent, még mielőtt a gazdagép memóriájához nyúlnánk.

**A memória hierarchia, nem fal.** Három szint, és a tervező megméri, mibe kerül mindegyik, ahelyett hogy reménykedne, hogy a modell beleférjen:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
Gazda DDR5   96 GB   PCIe vagy DDR korlátozza
```

---

<a id="demarrage"></a>

## Gyors kezdés

```bash
./install.sh                       # virtuális környezet + torch cu128 + acvram
acvram doctor                      # készen áll-e ez a gép, és mire
acvram detect                      # mi van itt valójában

acvram plan  ~/modellek/Qwen3-32B                   # melyik réteg hova kerülne
acvram convert ~/modellek/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Bármelyik OpenAI-kliens rögtön csatlakozhat:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Szia"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="nem_hasznalt")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Szia"}])
```

---

<a id="installer"></a>

## Telepítés

Forrásból (minden platformon):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Vagy csomagból: minden [GitHub-kiadáshoz](https://github.com/anticitoyun/anticitoyen-vram/releases/latest) egy-egy fájl van csatolva:

| Csatorna | A kiadáshoz csatolt fájl | Parancs |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (a neveket az `rpmbuild` állítja elő, nem rögzítettek) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (vagy `rpmbuild --rebuild *.src.rpm` a `.src.rpm` fájlból) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Telepítés előtt ellenőrizze a letöltött fájlt a kiadáshoz csatolt összegek alapján (`SHA256SUMS`, az összes többi fájl megléte után kerül közzétételre):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip csomagként nem érhető el (nem készül wheel): a `pip install -e '.[dev]'` egy forrásklónból telepít, akárcsak a `./install.sh`.

---

<a id="plan"></a>

## Mit mond az `acvram plan`

A tervezőt érdemes minden letöltés előtt lefuttatni. Olyan kérdésekre válaszol, amelyek eldöntik, hogy egy modell egyáltalán használható-e ezen a gépen:

```
$ acvram plan ~/modellek/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

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

A konfigurációs teret bejárja ahelyett, hogy megállna az elsőnél, ami elfér, és két döntése elég ellentmondásos ahhoz, hogy külön kiemeljük:

* **Az 3080 Ti-t kihasználatlanul hagyja**, ha egy modell elfér önmagában az 5090-en. Egy pipeline lépései sorban futnak: egy 912 GB/s-os lépés hozzáadása egy 1790 GB/s-os pipeline-hoz lassítja az egyszálú dekódolást. A `--gpus all` kapcsolóval kényszeríthető.
* **Csökkenti a KV gyorsítótárat, hogy a súlyok VRAM-ban maradjanak.** Minden gigabájt, amit a gyorsítótárnak adunk, egy gigabájt súly, amit a PCIe buszra tolunk vissza, egy súly PCIe-n keresztüli olvasása pedig kb. harmincszor annyiba kerül, mint VRAM-ból. A fenti 70B-n ez az egyetlen döntés 2,3-ról 17,8 token/s-ra emeli a sebességet.

---

<a id="optimisations"></a>

## Gyorsítás

Négy optimalizáció, mindegyiket ekvivalenciabizonyítás igazolja, nem csak stopperóra: egy optimalizáció, amely megváltoztatja a választ, hiba.

A sűrű modellek NVFP4 lineáris rétegei alapértelmezetten a Marlin elrendezésen mennek keresztül (+57–90% átviteli sebesség b = 8 mellett, TTFT +2–4 ms a revue/poste6-piece147-verdict-24-09.md szerint; visszaesés `ACVRAM_PROJ_MARLIN=0`, lásd [CHANGELOG.md](../CHANGELOG.md)).

### Spekulatív dekódolás (`--speculative`)

Egyetlen token dekódolása 1 méretű köteggel memóriakorlátos: a gép minden aktív súlyt beolvas, hogy egyetlen tokent állítson elő. K javasolt token ellenőrzése ugyanazokat a súlyokat **csak egyszer** olvassa. Két javasló:

* `ngram` (alapértelmezett) — a jelenlegi utótagot keresi korábban a kontextusban, és azt javasolja, ami utána következett. Semmibe sem kerül, nem igényel modellt. Megéri, ha a kimenet visszaadja a bemenetet: kódszerkesztés, RAG, összefoglalás.
* `draft` — egy kis modell egy második eszközön. Ezen a gépen ez az eszköz az RTX 3080 Ti, amit a tervező szándékosan tétlenül hagy minden olyan modellnél, amely elfér az 5090-en.

Létezik még `mtp` (a modell `nextn` feje) és `auto` is; jelen állapotukban nem érik meg, és alapértelmezetten nincsenek bekapcsolva — lásd `docs/ARCHITECTURE.md`.

Az elfogadás pontos, nem közelítő: egy javaslatot `min(1, p/q)` valószínűséggel fogadunk el, egy elutasítás pedig a `p - q` normalizált pozitív részéből mintavételez újra. 40 000 húzáson mérve, egy szándékosan rosszul kalibrált draft ellenében, a kibocsátott eloszlás a cél teljes variációjának 0,002-én belül marad — a spekuláció sebességet vásárol, sosem más választ.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Előtag-gyorsítótár (alapértelmezetten bekapcsolva)

A blokkokat a token-szeletük *láncolt* hasításával címezzük: két kérés, amely osztozik egy rendszerüzeneten, osztozik annak blokkjain is, és a másodiknak már nem kell újraszámolnia őket. A láncolás elengedhetetlen: ugyanaz a tizenhat token más kontextusban nem ugyanazokat a kulcsokat és értékeket tartalmazza, és a puszta szelet hasítása egy másik sorozat kérésének szolgálná ki a gyorsítótárat.

Egy felszabadított blokk, amelynek tartalma még azonosítható, egy LRU sorba kerül a szabad blokkok listája helyett: a gyorsítótár így túléli a kéréseket anélkül, hogy valaha visszautasítana egy allokációt, amit ki tudott volna szolgálni.

### Gazdagép-szintű számítás (`--host-exec`)

Egy réteg, amelynek súlyai RAM-ban laknak, átmásolható a GPU-ra, vagy helyben kiszámítható. Mindkét út memóriakorlátos, és ugyanazokat a bájtokat olvassa: a gyorsabb az, amelyiknek szélesebb a busza — a PCIe 5.0 x16 kb. 54 GB/s-ot ad, a kétcsatornás DDR5 kb. 70 GB/s-ot —, és a helyben számolás emellett szabadon hagyja a GPU-t, ahelyett hogy egy másolásra várakoztatná.

Ez csak akkor éri meg, ha a processzor közvetlenül olvassa a 4 biten csomagolt súlyokat. Innen ered egy kis C++ kernel AVX2 útvonallal (`acvram_cpu.cpp`, ctypes-szal betöltve, Python fejlécek vagy ninja nélkül). Még a tartalék **skaláris** ágán is 1,44-szeresen veri a `dequantize() @ x`-et INT4-ben és 3,21-szeresen NVFP4-ben, mert az utóbbi előbb egy 32 bites másolatot ír ki az egész mátrixról.

A Mistral-Large-123B-n a tervező becslése 1,35-ről 2,42 token/s-ra emelkedik.

### Vegyes pontosság (`--snr-floor`, alapértelmezetten kikapcsolva)

A konverter minden tenzorra megméri a réteg kimenetének jel-zaj arányát, és a `--snr-floor` alá esőket szélesebb formátumra léptetheti, a tenzorok 15%-ának korlátjáig és egy ártapláig (`--promotion-cout-max`, hozzáadott mebibájtban).

A küszöb **alapértelmezetten nulla**: semmi sem lép szintet. A dekódolás memóriasávszélesség-korlátos, és a `Huihui-Qwen3.8-27B`-n végzett mérés eldönti — egy 25 dB-es küszöb 13,4% memóriába és 10,6% sebességbe kerül (18,50 GiB és 41,8 t/s a 16,02 és 46,2 helyett) 2,0% perplexitásért (42,591 a 43,447 helyett, 16 383 tokenes korpusz). A `--snr-floor 25` visszaállítja a régi viselkedést, amikor a minőség fontosabb a sebességnél.

### És az `acvram eval`

A jel-zaj arány és a logitok koszinusza közelítések. Az `acvram eval REP [REP ...]` csúszóablakos perplexitást mér, hogy a formátumválasztást bizonyítékok döntsék el:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP végpontok

| végpont | megjegyzés |
|---|---|
| `POST /v1/chat/completions` | SSE folyam vagy egyetlen válasz; a modell társalgási sablonját használja |
| `POST /v1/completions` | a prompt szövegként vagy token-azonosítóként |
| `POST /v1/embeddings` | átlagolt végső rejtett állapotok, L2-normalizálva, `dimensions` figyelembe véve |
| `GET /v1/models` | plusz egy `acvram` blokk: formátumok, eszközök, KV gyorsítótár kapacitása |
| `GET /health`, `GET /metrics` | dekódolási sebesség, KV blokkok foglaltsága |

Ezeknek a válaszoknak a mezőnevei angolul maradnak: ez az OpenAI protokoll, és a lefordításuk minden meglévő klienst elrontana.

---

<a id="chiffres"></a>

## Honnan jönnek a számok

Minden fent idézett érték ennek a tárolónak a kódjából származik, és `pytest`-tel ellenőrzött. A méréseket processzoron végeztük, a referencia kernelekkel:

| formátum | bit/súly | súlyok SNR-je | logit-koszinusz vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

E mérésekből két megállapítás változtatta meg az alapértelmezett értékeket:

* **A Hadamard-forgatás az INT4-nek segít, az NVFP4-nek nem.** Az INT4 128-as csoportjai nem tudnak elnyelni egy elszigetelt kiugró csatornát, ezért a szélsőértékek szétterítése megéri az aktivációnkénti n log n transzformációt. Az NVFP4 16-os blokkjainak már megvan a saját skálájuk. Innen a `--hadamard auto`, amely csak az INT4-re alkalmazza.
* **Az INT8 veri az FP8 E4M3-at a KV gyorsítótárnál**, 44 dB a 32 dB-lel szemben azonos méret mellett, mert a (token, fej) szerinti skála már megadja azt a dinamikatartományt, amelyre az FP8 exponensbiteket költ. Mindkét kártya tehát INT8 KV gyorsítótárat használ, még ha az 5090 tudna is FP8-at. Egy `k8v4` formátum (INT4 értékek, −22% gyorsítótár-bájt) opcióként létezik, **nem minősítve** — lásd `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentáció

| Dokumentum | Tartalom |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **a projekt folytatása egy másik gépen** (franciául) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | hogyan illeszkednek össze a darabok |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | tiszta NVFP4 vagy attention+GDN csatornánkénti int8-ban, egy Gated DeltaNet hibriden |
| [`docs/MATERIEL.md`](MATERIEL.md) | ennek a konkrét gépnek a beállítása |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **ami még nincs kész**, ezt olvasd el először |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | a kódon való munka konvenciói (nyelv, stílus, ellenőrzések push előtt) |

---

<a id="resultats"></a>

## Mért eredmények (2026.09.22, RTX 5090, 400 W-on, ≥ 20 s ablak az energiamérőn)

Qwen3-Coder-30B-A3B NVFP4-ben (szakértők) + INT8-ban (figyelem, fej), ugyanaz a protokoll minden motorra (`outils/`, egy kártya, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekódolás, 12 szekvencia | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekódolás, 1 szekvencia | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, visszavonva) |

¹ 09.22-i helyesbítés: a `serve` alapértelmezetten spekulál (`--speculative ngram`, cli.py), a versenytársak nem; az eddig közölt 380,8 t/s SPEKULÁCIÓVAL mérve készült. Spekuláció nélkül (`--speculative none`, ugyanaz a lánc, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — az acvram **harmadik** b=1-nél, a llama.cpp és a vLLM mögött. Energiában megelőzi a llama.cpp-t (0,601 a 0,700 nettó J/token helyett). b=12-nél a spekuláció soha nincs aktív (a `lot_max=2` védelem miatt): ez a cella már egyenlő eséllyel indult.

² 09.23., ugyanaz az ülés, ugyanaz a HTTP kliens (`banc-llamacpp-16-09.py` az `acvram serve` és a `vllm serve` ellen), `-lgc 2700` explicit módon beállítva mindkét oldalon, felváltva mért cellák A V V A, ≥ 5 tétel oldalanként, eltérés csak 2σ felett jelentve (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 dekódoláskor, letekert figyelem-redukció): eltérés −1,6%, **2σ alatt: átviteli sebesség paritás**. J/tokenben **a vLLM 7,0%-kal vezet** (2σ felett). A 0.6.37-tel ugyanez a protokoll −4,7%-ot adott.

³ Ugyanaz az ülés és protokoll mint ²-nél, spekuláció nélkül mindkét oldalon: acvram 312,3 a vLLM 284,8-cal szemben — **az acvram 9,7%-kal vezet átviteli sebességben** (2σ felett); J/token: **paritás** (0,04% eltérés, 2σ alatt).

⁴ 09.23., ugyanaz a protokoll a llama.cpp ellen (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 újraírt útválasztással (+5,6%): acvram 310,8 a llama.cpp 329,9 t/s-jával szemben — **a llama.cpp 5,8%-kal vezet átviteli sebességben, az acvram 13,4%-kal vezet J/tokenben** (0,598 a 0,691 helyett).

A mai átviteli sebességek (1030-as állomás, `-lgc 2700` eco mód, pipeline üzemben; mohó mintavétel a CUDA gráfba rögzítve, alapértelmezett 0.6.35 óta). Az acvram b=12-es értéke egy hivatalos, lezárt cella (6 egymásba fésült ablak mediánja, óra ablakonként).

> **Helyesbítés (2026.09.23.).** Az eddig közölt vLLM-összehasonlítás (b=12: 1 782 a 1 634 t/s-szal szemben; b=1: 290,6) HTTP-n mért acvram-ot állított szembe **offline** mért vLLM-mel (`LLM().generate()`), és a 09.22-i helyesbítés tévesen állította, hogy a vLLM-cella `vllm serve`-n keresztül ment. 09.23.: ugyanaz a HTTP kliens mindkettőnél, és `-lgc` beállítva mindkettőnél (az acvram indításkor beállítja a sajátját, a `vllm serve` nem: enélkül a vLLM ~2930 MHz-en futott a ~2650 helyett). Eredmény a ² jegyzetben: a vLLM 9,1%-kal vezet b=12-nél.

09.14-én reggel az acvram 630 t/s-en és 0,619 J/tokenen állt ugyanezen a cellán: a nyereség a Blackwell natív FP4 MMA-jából (`mma.sync … kind::mxf4nvf4`, ×7,9 a bf16-hoz képest), a kötegenkénti vödrökbe csoportosított GEMM-es MoE-ból, az egyetlen kernelbe tömörített útválasztásból (3677 → 1517 indítás lépésenként) és a projekciókhoz használt szűk, tenzormagos GEMM-ből ered. Minden számnak megvan a saját jegyzete az `acvram-memoire/revue/`-ban, a mérés előtt lezárt előrejelzéssel, a műszerrel és annak módjával — mód nélküli szám nem kerül közlésre.

Ahol az acvram vezet: MLA modellek (GLM-4.7-Flash) natív sm_120 NVFP4-ben, amelyeket a vLLM csak FP8-ban szolgál ki (b=1: 165,35 t/s üzemben); a VRAM-ba nem férő modellek. Az egyszekvenciás dekódolás nem tartozik ide: spekuláció nélkül az acvram ott 9,7%-kal vezet a vLLM előtt (³ jegyzet), 5,8%-kal marad le a llama.cpp mögött átviteli sebességben, de 13,4%-kal vezeti energiában (⁴ jegyzet). Nagy kötegméretnél, egy VRAM-ba férő MoE-n, a vLLM átviteli sebesség paritáson áll b=12-nél (1 995,1 a 2 027,0 t/s-szal szemben, 2σ alatt, ² jegyzet), de megtart 7,0% J/token előnyt; az acvram ott 1 540 t/s-ről (0.6.34) 1 995-re (0.6.38) fejlődött.

---

<a id="etat"></a>

## Állapot

0.6.38-as verzió. Minden az 5090-en fut: `sm_120a`-ra (natív FP4) és `sm_86`-ra fordított CUDA kernelek, CUDA gráfok, NVFP4/INT8/INT4 kvantálás, HTTP szerver. Védőkorlátok a helyükön: a kártya láthatatlan a munkamenetek számára (`CUDA_VISIBLE_DEVICES` üres), és csak az `outils/carte.sh` kölcsönzi, zár alatt, egyszerre egy mérésnek; egy megfigyelő naplóz minden zár nélküli hozzáférést; egy energiamérés, amely több kártyát fed le, vagy 10 s alatt tart, érvénytelenítésre kerül; egy degradált módban betöltött modell ezt jelzi, és nem lép be egy párbajba.

4107 teszt (`pytest --collect-only -q`, egy perc processzoron; a GPU-tesztek csak `carte.sh` alatt futnak). A munka nyomon követése: `acvram-memoire/` (szabályok, jegyzék, füzetek, több száz jegyzet áttekintése).

---

<a id="credits"></a>

## Köszönet

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, Apache-2.0 licenc alatt: az `acvram/kernels/marlin_port/` az ő Marlin kerneljeit (MoE és sűrű) hordozza tovább, teljes, fájlonkénti forrásmegjelöléssel az [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE) fájlban.
- **NVIDIA** — CUDA, a Blackwell FP4 tenzormagjai (`sm_120`) és a könyvtárak, amelyektől ez a projekt függ.
- **PyTorch** — tenzormotor és C++/CUDA kiterjesztések.

Független projekt, nem áll kapcsolatban az ASUS-szal, az NVIDIA-val vagy a vLLM projekttel.

---

<a id="licence"></a>

## Licenc

[GPL-3.0 vagy újabb](../LICENSE) e tároló kódjára. Az `acvram/kernels/marlin_port/` a [vLLM](https://github.com/vllm-project/vllm) v0.29.0-ból átemelt kódot tartalmaz (`marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size` kernelek), Apache-2.0 licenc alatt: minden fájl megtartja eredeti fejlécét, a licencszöveg a `LICENSE-vllm`-ben található, a fájllista, az eredeti commit és a módosítások pedig az [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE) fájlban.

---

<a id="soutien"></a>

## Támogasd a projektet

Az acvram fejlesztése magánhardveren zajlik. Ha hasznos számodra a projekt:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Hívj%20meg%20egy%20kávéra&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Fordítások: [TRADUIRE.md](TRADUIRE.md) (franciául; a hozzájárulási útmutató még nincs lefordítva).
