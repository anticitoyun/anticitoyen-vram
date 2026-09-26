<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="GPL-3.0-or-later-lisenssi"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-tue-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

OpenAI-API-yhteensopiva päättelyportti, joka kohtelee muistia hierarkiana, antaa jokaiselle GPU:lle sen piirin parhaiten lukeman lukumuodon ja optimoi jokaisen tokenin jouleina yhtä lailla kuin sekunteina.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · **🇫🇮 Suomi** · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Läpäisyn ja energiankulutuksen vertailu vLLM:ään ja llama.cpp:hen" width="720"></p>

---

## Sisällys

- [Kaksi ajatusta](#idees)
- [Pika-aloitus](#demarrage)
- [Asennus](#installer)
- [Mitä `acvram plan` kertoo](#plan)
- [Nopeuttaminen](#optimisations)
- [HTTP-päätepisteet](#http)
- [Mistä luvut tulevat](#chiffres)
- [Dokumentaatio](#documentation)
- [Mitatut tulokset](#resultats)
- [Tila](#etat)
- [Kiitokset](#credits)
- [Lisenssi](#licence)
- [Tue projektia](#soutien)

---

<a id="idees"></a>

## Kaksi ajatusta

Suunniteltu tietylle koneelle:

| | |
|---|---|
| Suoritin | Intel Core i9-14900K (8 P-ydintä + 16 E-ydintä) |
| Emolevy | ASUS ROG Maximus Z790 Dark Hero |
| Muisti | 96 Gt DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Gt — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Gt — Ampere, `sm_86` |
| Järjestelmä | Ubuntu 26.04 LTS (CUDA 13); molemmat kortit PCIe x8/x8:ssa, rajoitettu 400 W / 275 W |

**Yksi muoto per GPU.** RTX 5090:ssä on FP4-tensoriytimet; RTX 3080 Ti:ssä ei ole niitä eikä FP8:aa. Molempien sovittaminen yhteiseen muotoon hukkaisi 5090:n potentiaalin. Muunnin kirjoittaa siis *saman mallin kahdesti*, muodossa, jota kukin kohde todella pystyy hyödyntämään:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| painot | **NVFP4** — E2M1 + FP8 E4M3 -skaala joka 16. | **INT4** — uint4 + fp16-skaala ja nolla joka 128. |
| bittiä per paino | 4,50 | 4,16 |
| verrattuna BF16:een | ×3,56 pienempi | ×3,85 pienempi |
| laskentatila | FP4-tensoriytimet | dekvantisoidaan FP16:ksi ytimessä, FP16-tensoriytimet |
| KV-välimuisti | INT8 | INT8 |

32 Gt VRAM:ia 4,5 bitillä per paino mahduttaa noin **56 miljardia parametria**, kun BF16:ssa vastaava on 16 miljardia. Molemmilla korteilla yhteensä tämä tekee noin **78 miljardia residenttiä parametria** ennen kuin isäntämuistiin edes koskaan.

**Muisti on hierarkia, ei seinä.** Kolme tasoa, ja suunnittelija mittaa, mitä kukin todella maksaa, sen sijaan että toivoisi mallin mahtuvan:

```
RTX 5090     32 Gt   ~1790 Gt/s     NVFP4
RTX 3080 Ti  12 Gt    ~912 Gt/s     INT4
Isännän DDR5   96 Gt   PCIe:n tai DDR:n rajoittama
```

---

<a id="demarrage"></a>

## Pika-aloitus

```bash
./install.sh                       # virtuaaliympäristö + torch cu128 + acvram
acvram doctor                      # onko tämä kone valmis, ja mihin
acvram detect                      # mitä täällä todella on

acvram plan  ~/mallit/Qwen3-32B                     # minne kukin kerros menisi
acvram convert ~/mallit/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Mikä tahansa OpenAI-asiakas kytkeytyy sitten heti:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Hei"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="ei_kaytossa")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Hei"}])
```

---

<a id="installer"></a>

## Asennus

Lähdekoodista (kaikki alustat):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Tai pakettina, jolloin kukin tiedosto on liitetty jokaiseen [GitHub-julkaisuun](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kanava | Julkaisuun liitetty tiedosto | Komento |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nimet muodostaa `rpmbuild`, eivätkä ne ole kiinteitä) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (tai `rpmbuild --rebuild *.src.rpm` lähtien `.src.rpm`-tiedostosta) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Ennen asennusta, tarkista ladattu tiedosto julkaisuun liitettyjä tarkistussummia vasten (`SHA256SUMS`, julkaistaan kun kaikki muut tiedostot ovat läsnä):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip-pakettia ei julkaista (valmista wheel-pakettia ei rakenneta): `pip install -e '.[dev]'` asentaa lähdekoodin kloonista, samoin kuin `./install.sh`.

---

<a id="plan"></a>

## Mitä `acvram plan` kertoo

Suunnittelija kannattaa ajaa ennen jokaista latausta. Se vastaa kysymyksiin, jotka ratkaisevat, onko malli ylipäätään käyttökelpoinen tällä koneella:

```
$ acvram plan ~/mallit/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

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

Se tutkii konfiguraatioavaruuden sen sijaan, että pysähtyisi ensimmäiseen, joka mahtuu, ja kaksi sen päätöksistä ovat riittävän vastaintuitiivisia ansaitakseen maininnan:

* **Se jättää 3080 Ti:n käyttämättä**, kun malli mahtuu pelkälle 5090:lle. Putkilinjan vaiheet suoritetaan sarjassa: 912 Gt/s -vaiheen lisääminen 1790 Gt/s -putkeen hidastaa yksivirtaista dekoodausta. Pakotetaan `--gpus all`:lla.
* **Se pienentää KV-välimuistia pitääkseen painot VRAM:ssa.** Jokainen välimuistille annettu gigatavu on gigatavu painoja, jotka työnnetään PCIe-väylälle, ja painon lukeminen PCIe:n kautta maksaa noin kolmekymmentä kertaa sen, mitä se maksaisi VRAM:sta. Yllä olevalla 70B:llä tämä yksi kompromissi nostaa nopeuden 2,3:sta 17,8 tokeniin/s.

---

<a id="optimisations"></a>

## Nopeuttaminen

Neljä optimointia, joista jokainen on varmennettu ekvivalenssitodistuksella eikä pelkällä sekuntikellolla: optimointi, joka muuttaa vastauksen, on bugi.

Tiheiden mallien NVFP4-lineaarikerrokset kulkevat oletuksena Marlin-asettelun kautta (+57–90 % läpäisyä b = 8:lla, TTFT +2–4 ms lähteen revue/poste6-piece147-verdict-24-09.md mukaan; paluu `ACVRAM_PROJ_MARLIN=0`, ks. [CHANGELOG.md](../CHANGELOG.md)).

### Spekulatiivinen dekoodaus (`--speculative`)

Yhden tokenin dekoodaus eräkoolla 1 on muistirajoitteista: kone lukee kaikki aktiiviset painot tuottaakseen vain yhden tokenin. K:n ehdotetun tokenin tarkistaminen lukee samat painot **vain kerran**. Kaksi ehdottajaa:

* `ngram` (oletus) — etsii nykyistä loppuliitettä aiemmin kontekstista ja ehdottaa sitä, mikä sitä seurasi. Ei maksa mitään, ei vaadi mallia. Kannattaa, kun tuloste toistaa syötteen: koodin muokkaus, RAG, tiivistäminen.
* `draft` — pieni malli toisella laitteella. Tässä kokoonpanossa tuo laite on RTX 3080 Ti, jonka suunnittelija jättää tarkoituksella joutilaaksi jokaiselle mallille, joka mahtuu 5090:lle.

Myös `mtp` (mallin `nextn`-pää) ja `auto` ovat olemassa; eivät kannattavia nykyisellään eivätkä oletuksena päällä — ks. `docs/ARCHITECTURE.md`.

Hyväksyntä on tarkka, ei likimääräinen: ehdotus hyväksytään todennäköisyydellä `min(1, p/q)`, ja hylkäys uudelleennäytteistää `p - q`:n normalisoidusta positiivisesta osasta. Mitattuna 40 000 nostolla tarkoituksella huonosti kalibroitua luonnosta vastaan, tuotettu jakauma pysyy 0,002:n sisällä kohteen kokonaisvariaatiosta — spekulaatio ostaa nopeutta, ei koskaan erilaista vastausta.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Etuliitevälimuisti (päällä oletuksena)

Lohkot osoitetaan token-viipaleensa *ketjutetulla* tiivisteellä: kaksi pyyntöä, jotka jakavat järjestelmäkehotteen, jakavat myös sen lohkot, eikä toisen enää tarvitse esilaskea niitä. Ketjutus on välttämätöntä: samat kuusitoista tokenia eri kontekstissa eivät sisällä samoja avaimia ja arvoja, ja pelkän viipaleen tiivistäminen palvelisi yhden sekvenssin välimuistia toisen pyynnöllä.

Vapautettu lohko, jonka sisältö on yhä tunnistettavissa, liittyy LRU-jonoon vapaiden lohkojen listan sijaan: välimuisti selviää näin pyyntöjen välillä koskaan kieltäytymättä varauksesta, jonka se olisi voinut palvella.

### Isäntätason laskenta (`--host-exec`)

Kerros, jonka painot asuvat RAM:ssa, voidaan kopioida GPU:lle tai laskea paikan päällä. Molemmat reitit ovat muistirajoitteisia ja lukevat samat tavut: nopeampi on se, jonka väylä on leveämpi — PCIe 5.0 x16 antaa noin 54 Gt/s, kaksikanavainen DDR5 noin 70 Gt/s — ja paikallaan laskeminen jättää lisäksi GPU:n vapaaksi sen sijaan, että se odottaisi kopiota.

Tämä kannattaa vain, jos suoritin lukee suoraan 4 bittiin pakatut painot. Siitä pieni C++-ydin AVX2-polulla (`acvram_cpu.cpp`, ladattu ctypesillä, ilman Python-otsikoita tai ninjaa). Jopa varajärjestelmänsä **skalaarihaarassa** se voittaa `dequantize() @ x`:n kertoimella 1,44 INT4:ssä ja 3,21 NVFP4:ssä, koska jälkimmäinen kirjoittaa ensin 32-bittisen kopion koko matriisista.

Mistral-Large-123B:llä suunnittelijan arvio nousee 1,35:stä 2,42 tokeniin/s.

### Sekatarkkuus (`--snr-floor`, pois päältä oletuksena)

Muunnin mittaa signaali-kohinasuhteen kerroksen ulostulossa jokaiselle tensorille ja voi ylentää leveämpään muotoon ne, jotka jäävät `--snr-floor`-arvon alle, enintään 15 %:iin tensoreista ja hintakattoon asti (`--promotion-cout-max`, lisättyinä mebitavuina).

Kynnys on **oletuksena nolla**: mitään ei ylennetä. Dekoodaus on muistikaistanleveysrajoitteista, ja mittaus `Huihui-Qwen3.8-27B`:llä ratkaisee asian — 25 dB:n kynnys maksaa 13,4 % muistia ja 10,6 % läpäisyä (18,50 GiB ja 41,8 t/s vs. 16,02 ja 46,2) 2,0 %:n perpleksiteetistä (42,591 vs. 43,447, 16 383 tokenin korpus). `--snr-floor 25` palauttaa vanhan käytöksen, kun laatu menee nopeuden edelle.

### Ja `acvram eval`

Signaali-kohinasuhde ja logittien kosini ovat approksimaatioita. `acvram eval REP [REP ...]` mittaa perpleksiteetin liukuvalla ikkunalla, jotta muotovalinta ratkaistaan todisteilla:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## HTTP-päätepisteet

| päätepiste | huomiot |
|---|---|
| `POST /v1/chat/completions` | SSE-virta tai yksittäinen vastaus; käyttää mallin keskustelumallia |
| `POST /v1/completions` | kehote tekstinä tai token-tunnisteina |
| `POST /v1/embeddings` | keskiarvoistetut lopulliset piilotilat, L2-normalisoitu, `dimensions` huomioitu |
| `GET /v1/models` | plus `acvram`-lohko: muodot, laitteet, KV-välimuistin kapasiteetti |
| `GET /health`, `GET /metrics` | dekoodausnopeus, KV-lohkojen käyttöaste |

Näiden vastausten kenttänimet pysyvät englanniksi: kyseessä on OpenAI-protokolla, ja niiden kääntäminen rikkoisi kaikki olemassa olevat asiakkaat.

---

<a id="chiffres"></a>

## Mistä luvut tulevat

Jokainen yllä mainittu arvo tulee tämän arkiston koodista ja on `pytest`in varmentama. Mittaukset tehty suorittimella referenssiytimillä:

| muoto | bittiä/paino | painojen SNR | logit-kosini vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Kaksi näistä mittauksista tehtyä havaintoa muuttivat oletusarvoja:

* **Hadamard-rotaatio auttaa INT4:ää mutta ei NVFP4:ää.** INT4:n 128:n ryhmät eivät pysty imemään eristynyttä poikkeavaa kanavaa, joten ääriarvojen levittäminen kannattaa n log n -muunnoksen verran per aktivointi. NVFP4:n 16:n lohkoilla on jo oma skaalansa. Siitä `--hadamard auto`, joka soveltaa sitä vain INT4:ään.
* **INT8 voittaa FP8 E4M3:n KV-välimuistissa**, 44 dB vs. 32 dB samalla koolla, koska (token, pää) -kohtainen skaala tarjoaa jo sen dynamiikka-alueen, johon FP8 kuluttaa eksponenttibittejä. Molemmat kortit käyttävät siis INT8-KV-välimuistia, vaikka 5090 osaisikin FP8:aa. `k8v4`-muoto (arvot INT4:ssä, −22 % välimuistitavuja) on olemassa vaihtoehtona, **kelpuuttamattomana** — ks. `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentaatio

| Asiakirja | Sisältö |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **projektin jatkaminen toisella koneella** (ranskaksi) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | miten palaset sopivat yhteen |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | puhdas NVFP4 tai attention+GDN kanavakohtaisena int8:na, Gated DeltaNet -hybridissä |
| [`docs/MATERIEL.md`](MATERIEL.md) | tämän tietyn koneen säätäminen |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **mitä ei ole vielä tehty**, lue tämä ensin |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | koodityön käytännöt (kieli, tyyli, tarkistukset ennen pushia) |

---

<a id="resultats"></a>

## Mitatut tulokset (22.9.2026, RTX 5090, 400 W:ssa, ≥ 20 s ikkuna energiamittarilla)

Qwen3-Coder-30B-A3B NVFP4:ssä (asiantuntijat) + INT8:ssa (huomio, pää), sama protokolla kaikille moottoreille (`outils/`, yksi kortti, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekoodaus, 12 sekvenssiä | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekoodaus, 1 sekvenssi | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| esitäyttö pp2048 | **22 707 tokenia/s** | 21 054 | 8 671 (TabbyAPI, poistettu) |

¹ 22.9. oikaisu: `serve` spekuloi oletuksena (`--speculative ngram`, cli.py), kilpailijat eivät; tähän asti julkaistu 380,8 t/s oli mitattu SPEKULAATION KANSSA. Ilman spekulaatiota (`--speculative none`, sama ketju, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram on **kolmas** b=1:llä, llama.cpp:n ja vLLM:n jäljessä. Energiassa se pysyy llama.cpp:n edellä (0,601 vs. 0,700 J/token netto). b=12:lla spekulaatio ei ole koskaan aktiivinen (`lot_max=2`-suoja): tuo solu oli jo tasavertainen.

² 23.9., sama istunto, sama HTTP-asiakas (`banc-llamacpp-16-09.py` `acvram serve`- ja `vllm serve`-palvelimia vastaan), `-lgc 2700` asetettu eksplisiittisesti molempien ympärille, solut vuorotellen A V V A, ≥ 5 erää per haara, ero ilmoitettu vain yli 2σ:n (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 dekoodauksessa, purettu huomion redusointi): ero −1,6 %, **alle 2σ: läpäisyn pariteetti**. J/tokenissa **vLLM pysyy edellä 7,0 %:lla** (yli 2σ). Versiolla 0.6.37 sama protokolla antoi −4,7 %.

³ Sama istunto ja protokolla kuin ²:ssa, ilman spekulaatiota kummallakaan puolella: acvram 312,3 vs. vLLM 284,8 — **acvram edellä 9,7 %:lla läpäisyssä** (yli 2σ); J/token: **pariteetti** (0,04 %:n ero, alle 2σ).

⁴ 23.9., sama protokolla llama.cpp:tä vastaan (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 uudelleenkirjoitetulla reitityksellä (+5,6 %): acvram 310,8 vs. llama.cpp 329,9 t/s — **llama.cpp edellä 5,8 %:lla läpäisyssä, acvram edellä 13,4 %:lla J/tokenissa** (0,598 vs. 0,691).

Päivän läpäisyt (asema 1030, eco-tila `-lgc 2700`, putki käytössä; ahne näytteistys otettu CUDA-graafiin, oletus versiosta 0.6.35 lähtien). acvram-moottorin b=12-arvo on virallinen sinetöity solu (mediaani 6:sta lomitetusta ikkunasta, kello ikkunaa kohti).

> **Oikaisu (23.9.2026).** Tähän asti julkaistu vLLM-vertailu (b=12: 1 782 vs. 1 634 t/s; b=1: 290,6) asetti HTTP:llä mitatun acvram-moottorin vastakkain **offline**-mitatun vLLM:n kanssa (`LLM().generate()`), ja 22.9. oikaisu väitti virheellisesti, että vLLM-solu kulki `vllm serve`:n kautta. 23.9.: sama HTTP-asiakas molemmille, ja `-lgc` asetettu molemmille (acvram asettaa omansa käynnistyksessä, `vllm serve` ei: ilman tätä varotoimea vLLM pyöri ~2930 MHz:llä ~2650:n sijaan). Tulos huomautuksessa ²: vLLM edellä 9,1 %:lla b=12:lla.

Aamulla 14.9. acvram oli 630 t/s:ssä ja 0,619 J/tokenissa samalla solulla: hyödyt tulevat Blackwellin natiivista FP4 MMA:sta (`mma.sync … kind::mxf4nvf4`, ×7,9 bf16:een verrattuna), erän ämpäreittäin ryhmitellystä GEMM-MoE:sta, yhteen ytimeen tiivistetystä reitityksestä (3677 → 1517 käynnistystä per askel) ja kapeasta tensoriydin-GEMM:stä projektioille. Jokaisella luvulla on oma muistiinpanonsa `acvram-memoire/revue/`-hakemistossa, mittausta edeltävällä sinetöidyllä ennusteella, instrumentilla ja sen tilalla — lukua ilman tilaa ei julkaista.

Missä acvram on edellä: MLA-mallit (GLM-4.7-Flash) natiivissa sm_120 NVFP4:ssä, joita vLLM tarjoilee vain FP8:ssa (b=1: 165,35 t/s käytössä); mallit, jotka eivät mahdu VRAM:iin. Yhden sekvenssin dekoodaus ei kuulu näihin: ilman spekulaatiota acvram on siinä edellä vLLM:ää 9,7 %:lla (huomautus ³), jäljessä llama.cpp:stä 5,8 %:lla läpäisyssä mutta edellä sitä 13,4 %:lla energiassa (huomautus ⁴). Suurella erällä, VRAM:iin mahtuvalla MoE:lla, vLLM on läpäisyn pariteetissa b=12:lla (1 995,1 vs. 2 027,0 t/s, alle 2σ, huomautus ²) mutta säilyttää 7,0 %:n J/token-edun; acvram on edennyt siinä 1 540 t/s:stä (0.6.34) 1 995:een (0.6.38).

---

<a id="etat"></a>

## Tila

Versio 0.6.38. Kaikki toimii 5090:llä: CUDA-ytimet käännetty `sm_120a`:lle (natiivi FP4) ja `sm_86`:lle, CUDA-graafit, NVFP4/INT8/INT4-kvantisointi, HTTP-palvelin. Suojaukset paikoillaan: kortti on näkymätön työistunnoille (`CUDA_VISIBLE_DEVICES` tyhjä), ja vain `outils/carte.sh` lainaa sen, lukittuna, yhdelle mittaukselle kerrallaan; vahti kirjaa lokiin jokaisen lukituksen ulkopuolisen käytön; useamman kortin kattava tai alle 10 s kestävä energiamittaus mitätöidään; heikentyneessä tilassa ladattu malli sanoo niin eikä osallistu kaksintaisteluun.

4 107 testiä (`pytest --collect-only -q`, minuutti suorittimella; GPU-testit ajetaan vain `carte.sh`:n alla). Työn seuranta: `acvram-memoire/` (säännöt, hakemisto, muistikirjat, satojen muistiinpanojen katsaus).

---

<a id="credits"></a>

## Kiitokset

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, Apache-2.0-lisenssillä: `acvram/kernels/marlin_port/` siirtää sen Marlin-ytimet (MoE ja tiheä), täydellä tiedosto­kohtaisella lähdemerkinnällä tiedostossa [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, Blackwellin FP4-tensoriytimet (`sm_120`) ja kirjastot, joista tämä projekti riippuu.
- **PyTorch** — tensorimoottori ja C++/CUDA-laajennukset.

Riippumaton projekti, ei sidoksissa ASUS:iin, NVIDIAan eikä vLLM-projektiin.

---

<a id="licence"></a>

## Lisenssi

[GPL-3.0 tai uudempi](../LICENSE) tämän arkiston koodille. `acvram/kernels/marlin_port/` sisältää koodia, joka on siirretty [vLLM](https://github.com/vllm-project/vllm) v0.29.0:sta (`marlin_moe_wna16`-, `gptq_marlin_repack`- ja `moe_align_block_size`-ytimet), Apache-2.0-lisenssillä: jokainen tiedosto säilyttää alkuperäisen otsikkonsa, lisenssiteksti on tiedostossa `LICENSE-vllm`, ja tiedostolista, alkuperäinen commit ja muutokset ovat tiedostossa [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Tue projektia

acvram-moottorin kehitys tapahtuu henkilökohtaisella laitteistolla. Jos projektista on sinulle hyötyä:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Tarjoa%20kahvi&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Käännökset: [TRADUIRE.md](TRADUIRE.md) (ranskaksi; osallistumisopasta ei ole vielä käännetty).
