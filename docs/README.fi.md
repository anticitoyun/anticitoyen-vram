<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Tue: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

OpenAI-rajapinnan kanssa yhteensopiva päättelyväylä, joka käsittelee muistia
hierarkiana ja antaa jokaiselle näytönohjaimelle sen lukuformaatin, jota sen
piiri lukee parhaiten.

Suunniteltu yhdelle tietylle koneelle:

| | |
|---|---|
| Suoritin | Intel Core i9-14900K (8 P-ydintä + 16 E-ydintä) |
| Emolevy | ASUS ROG Maximus Z790 Dark Hero |
| Muisti | 96 Gt DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Gt — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Gt — Ampere, `sm_86` |
| Järjestelmä | Ubuntu 26.04 LTS (CUDA 13); molemmat kortit PCIe x8/x8 -väylässä, rajoitettu 400 W / 275 W |

## Kaksi ideaa

**Yksi formaatti per GPU.** RTX 5090:ssä on FP4-tensoriytimet; RTX 3080 Ti:ssä
ei ole, eikä FP8:aa myöskään. Molempien pakottaminen yhteiseen formaattiin
hukkaisi 5090:n. Muunnin kirjoittaa siis *saman mallin kahdesti*, siinä
formaatissa, jota kukin kohde osaa todella hyödyntää:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| painot | **NVFP4** — E2M1 + FP8 E4M3 -skaala joka 16. | **INT4** — uint4 + fp16-skaala ja -nollakohta joka 128. |
| bittiä per paino | 4,50 | 4,16 |
| suhteessa BF16:een | ×3,56 pienempi | ×3,85 pienempi |
| laskentatapa | FP4-tensoriytimet | dekvantisoidaan FP16:ksi ytimessä, FP16-tensoriytimet |
| KV-välimuisti | INT8 | INT8 |

32 Gt VRAM-muistia 4,5 bitillä per paino sisältää noin **56 miljardia
parametria**, kun BF16:ssa mahtuu 16 miljardia. Molemmilla korteilla se
tekee noin **78 miljardia residenttiä parametria** ennen kuin keskusmuistiin
edes kosketaan.

**Muisti on hierarkia, ei seinä.** Kolme tasoa, ja suunnittelija mittaa,
mitä kukin maksaa, sen sijaan että toivoisi mallin mahtuvan:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Pika-aloitus

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Mikä tahansa OpenAI-asiakas kytkeytyy sen jälkeen:

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

## Mitä `acvram plan` kertoo

Suunnittelija kannattaa ajaa ennen mitään latausta. Se vastaa kysymyksiin,
jotka ratkaisevat, onko malli käyttökelpoinen tällä koneella:

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

Se tutkii kokoonpanoavaruuden sen sijaan, että pitäisi ensimmäisen sopivan,
ja kaksi sen päätöksistä on riittävän epäintuitiivisia, että ne on syytä
sanoa ääneen:

* **Se jättää 3080 Ti:n käyttämättä**, kun malli mahtuu pelkkään 5090:een.
  Liukuhihnan viipaleet suoritetaan peräkkäin: 912 Gt/s -vaiheen lisääminen
  1790 Gt/s -liukuhihnaan hidastaa yhden virran dekoodausta. Pakotetaan
  valitsimella `--gpus all`.
* **Se kutistaa KV-välimuistia pitääkseen painot VRAM-muistissa.** Jokainen
  välimuistille annettu gigatavu on gigatavu painoja, jotka työnnetään
  PCIe-väylälle, ja painon lukeminen PCIe:n yli maksaa noin kolmekymmentä
  kertaa sen, mitä VRAM-muistista lukeminen. Yllä olevalla 70B:llä pelkkä tämä
  punninta vie 2,3:sta 17,8 tokeniin/s.

## Nopeasti

Neljä optimointia, jokainen varmennettu ekvivalenssitodistuksella eikä vain
sekuntikellolla: optimointi, joka muuttaa vastausta, on virhe.

### Spekulatiivinen dekoodaus (`--speculative`)

Yhden tokenin dekoodaus eräkoolla 1 on muistirajoitteista: kone lukee kaikki
aktiiviset painot tuottaakseen yhden ainoan tokenin. K ehdotetun tokenin
tarkistus lukee samat painot **vain kerran**. Kaksi ehdottajaa:

* `ngram` (oletus) — etsii nykyisen loppuliitteen aiemmasta kontekstista ja
  ehdottaa sitä, mikä seurasi. Ei maksa mitään, ei vaadi mallia. Kannattaa,
  kun tuloste kopioi syötettä: koodin muokkaus, RAG, tiivistys.
* `draft` — pieni malli toisella laitteella. Tällä kokoonpanolla se laite on
  RTX 3080 Ti, jonka suunnittelija jättää tarkoituksella joutilaaksi kaikille
  5090:een mahtuville malleille.

Hyväksyntä on tarkka, ei likimääräinen: ehdotus hyväksytään todennäköisyydellä
`min(1, p/q)`, ja hylkäys uudelleenotostaa `p - q`:n normalisoidusta
positiivisesta osasta. 40 000 arvonnalla tahallaan huonosti kalibroitua
luonnosta vastaan mitattuna tuotettu jakauma pysyy 0,002 kokonaisvariaation
sisällä kohteesta — spekulaatio ostaa nopeutta, ei koskaan eri vastausta.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Etuliitevälimuisti (oletuksena päällä)

Lohkot osoitetaan niiden tokeniviipaleen *ketjutetulla* tiivisteellä: kaksi
pyyntöä, jotka jakavat järjestelmäohjeen, jakavat sen lohkot, eikä
jälkimmäisen tarvitse enää esilaskea niitä. Ketjutus on välttämätön: samat
kuusitoista tokenia eri kontekstissa eivät sisällä samoja avaimia ja arvoja,
ja pelkän viipaleen tiivistäminen tarjoilisi yhden sekvenssin välimuistin
toiselle.

Vapautettu lohko, jonka sisältö on yhä tunnistettavissa, siirtyy LRU-jonoon
vapaiden lohkojen listan sijaan: välimuisti säilyy näin pyyntöjen välillä
kieltäytymättä koskaan varauksesta, jonka se olisi voinut palvella.

### Laskenta isäntätasolla (`--host-exec`)

Kerros, jonka painot ovat keskusmuistissa, voidaan kopioida GPU:lle tai
laskea paikallaan. Molemmat polut ovat muistirajoitteisia ja lukevat samat
tavut: nopeampi on se, jolla on leveämpi väylä — PCIe 5.0 x16 antaa noin
54 Gt/s, kaksikanavainen DDR5 noin 70 Gt/s — ja paikallaan laskeminen jättää
lisäksi GPU:n vapaaksi sen sijaan, että se odottaisi kopiota.

Tämä kannattaa vain, jos suoritin lukee 4-bittisiksi pakatut painot suoraan.
Siitä pieni C++-ydin AVX2-polulla (`acvram_cpu.cpp`, ladataan ctypesillä,
ilman Python-otsikoita tai ninjaa). Jopa **skalaarisella** varahaarallaan se
voittaa `dequantize() @ x`:n kertoimella 1,44 INT4:ssä ja 3,21 NVFP4:ssä, koska
jälkimmäinen kirjoittaa ensin 32-bittisen kopion koko matriisista.

Mistral-Large-123B:llä suunnittelijan arvio nousee 1,35:stä 2,42 tokeniin/s.

### Sekatarkkuus (`--snr-floor`, oletuksena pois)

Muunnin mittaa signaali-kohinasuhteen jokaisen kerroksen ulostulossa
jokaiselle tensorille ja voi ylentää leveämpään formaattiin ne, jotka jäävät
alle `--snr-floor`-rajan, enintään 15 % tensoreista ja hintakaton rajoissa
(`--promotion-cout-max`, lisättyinä mebitavuina).

Lattia on **oletuksena nolla**: mitään ei ylennetä. Dekoodausta rajoittaa
muistikaista, ja mittaus mallilla `Huihui-Qwen3.8-27B` ratkaisee — 25 dB:n
lattia maksaa 13,4 % muistia ja 10,6 % läpimenoa (18,50 GiB ja 41,8 t/s vastaan
16,02 ja 46,2) 2,0 %:n perpleksiteetistä (42,591 vastaan 43,447, 16 383 tokenin
korpus). `--snr-floor 25` palauttaa vanhan käytöksen, kun laatu menee nopeuden
edelle.

### Ja `acvram eval`

Signaali-kohinasuhde ja logit-kosini ovat likiarvoja.
`acvram eval HAK [HAK ...]` mittaa perpleksiteetin liukuvalla ikkunalla, jotta
formaattivalinta ratkaistaan todisteilla:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP-päätepisteet

| päätepiste | huomautukset |
|---|---|
| `POST /v1/chat/completions` | SSE-virta tai yksittäinen vastaus; käyttää mallin keskustelumallia |
| `POST /v1/completions` | kehote tekstinä tai tokenitunnisteina |
| `POST /v1/embeddings` | viimeisten piilotilojen keskiarvo, L2-normalisoitu, `dimensions` huomioidaan |
| `GET /v1/models` | lisäksi `acvram`-lohko: formaatit, laitteet, KV-välimuistin kapasiteetti |
| `GET /health`, `GET /metrics` | dekoodauksen läpimeno, KV-lohkojen täyttöaste |

Näiden vastausten kenttänimet pysyvät englanniksi: kyseessä on
OpenAI-protokolla, ja niiden kääntäminen rikkoisi kaikki olemassa olevat
asiakkaat.

## Mistä luvut tulevat

Jokainen yllä mainittu arvo on tämän arkiston koodin tuottama ja `pytest`in
tarkistama. Mittaukset tehty suorittimella viiteytimillä:

| formaatti | bittiä/paino | painojen SNR | logit-kosini vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Kaksi näistä mittauksista saatua havaintoa muutti oletusarvoja:

* **Hadamard-rotaatio auttaa INT4:ää mutta ei NVFP4:ää.** INT4:n 128:n ryhmät
  eivät pysty vaimentamaan yksittäistä poikkeavaa kanavaa, joten ääriarvojen
  levittäminen on n log n -muunnoksen arvoinen aktivointia kohden. NVFP4:n
  16:n lohkot kantavat jo oman skaalansa. Siitä `--hadamard auto`, joka
  soveltaa sitä vain INT4:ään.
* **INT8 voittaa FP8 E4M3:n KV-välimuistissa**, 44 dB vastaan 32 dB samalla
  koolla, koska skaala per (token, pää) tarjoaa jo sen dynaamisen alueen, johon
  FP8 kuluttaa eksponenttibittejä. Molemmat kortit käyttävät siis INT8-
  KV-välimuistia, vaikka 5090 osaisi FP8:aa.

## Dokumentaatio

* [`REPRISE.md`](../REPRISE.md) — **projektin jatkaminen toisella koneella**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — miten osat sopivat yhteen
* [`docs/MATERIEL.md`](MATERIEL.md) — juuri tämän koneen säätäminen
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **mitä ei ole tehty**, lue ensin
* [`CONVENTIONS.md`](../CONVENTIONS.md) — koodin työskentelykäytännöt (kieli, tyyli, tarkistukset ennen pushia)

## Mitatut tulokset (22.9.2026, RTX 5090 400 W:ssa, ≥ 20 s:n ajo energiamittarilla)

Qwen3-Coder-30B-A3B NVFP4:nä (asiantuntijat) + INT8:na (huomio, pää), sama
protokolla kaikille moottoreille (`outils/`, yksi kortti, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekoodaus, 12 sekvenssiä | **1 625,5 t/s** | 1 596,1 t/s | — |
| dekoodaus, 1 sekvenssi | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokenia/s** | 21 054 | 8 671 (TabbyAPI, vedetty pois) |

Päivän läpimenot (asema 1030, säästöajotila `-lgc 2700`, liukuhihna palvelussa;
ahne otanta talletettu CUDA-graafiin, version 0.6.35 oletus). b=12 on virallinen
sinetöity solu (6 lomitetun ikkunan mediaani, kello ikkunaa kohden). vLLM-arvo
1 596,1 on 21.9. jäädytetty viite (vLLM:ää ei ajettu uudelleen sinä päivänä):
ero +1,84 % pätee yhtäläisellä viitteellä, ei molempien uusintamittauksena
samana aamuna. J/token samalla kellotaajuudella kolmea moottoria vastaan on yhä
uusintamittauksessa (`outils/gpu/mesure/banc-4moteurs.py`) — lukua ilman ajotilaa
ei julkaista.

Aamulla 14.9. acvram oli samassa solussa 630 t/s ja 0,619 J/token: parannukset
tulevat Blackwellin natiivista FP4-MMA:sta (`mma.sync … kind::mxf4nvf4`, ×7,9
bf16:een nähden), MoE:stä ryhmiteltynä GEMM:nä erä-ämpäriä kohden,
yhden ytimen reitityksestä (3 677 → 1 517 käynnistystä per askel) ja kapeasta
tensoriydin-GEMM:stä projektioille. Jokaisella luvulla on muistiinpanonsa
hakemistossa `acvram-memoire/revue/` ennen mittausta sinetöityine
ennusteineen, instrumentteineen ja ajotiloineen — lukua ilman ajotilaa ei
julkaista.

Missä acvram on edellä: MLA-mallit (GLM-4.7-Flash) natiivina sm_120-NVFP4:nä,
joita vLLM tarjoilee vain FP8:na (b=1: 165,35 t/s palvelussa); mallit, jotka
eivät mahdu VRAM-muistiin; ja, versiosta 0.6.35 lähtien, VRAM-muistiin mahtuvan
MoE:n suurten erien dekoodaus — b=12 nousee arvosta 1 540 (0.6.34) arvoon
1 625,5 t/s, eli +1,84 % jäädytetyn vLLM-viitteen (1 596,1) edellä. Ero pysyy
kapeana ja jäädytetyllä viitteellä; energiaero on mitattava uudelleen.

## Tila

Versio 0.6.35. Kaikki pyörii 5090:llä: CUDA-ytimet käännetty `sm_120a`:lle
(natiivi FP4) ja `sm_86`:lle, CUDA-graafit, NVFP4/INT8/INT4-kvantisointi,
HTTP-palvelin. Suojakaiteet paikoillaan: kortti on näkymätön työistunnoille
(`CUDA_VISIBLE_DEVICES` tyhjä) ja vain `outils/carte.sh` lainaa sen lukon
alla yhdelle mittaukselle kerrallaan; vartija kirjaa jokaisen lukon
ulkopuolisen käytön; energiamittaus, joka kattaa useamman kuin yhden kortin
tai alle 10 s, mitätöidään; heikennetyssä tilassa ladattu malli sanoo sen
eikä osallistu kaksintaisteluun.

640 testiä (`pytest -q`, minuutti suorittimella; GPU-testit ajetaan vain
`carte.sh`:n alla). Työn seuranta: `acvram-memoire/` (säännöt, hakemisto,
vihot, 180 muistiinpanon katselmus).

## Tue

acvramia kehitetään henkilökohtaisella laitteistolla. Jos projekti on sinulle
hyödyllinen: **Tue: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Lisenssi

GPL-3.0 tai uudempi.
