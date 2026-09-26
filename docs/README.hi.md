<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

OpenAI API के साथ संगत एक इन्फ़रेंस गेटवे, जो मेमोरी को एक पदानुक्रम के रूप में देखता है, हर GPU को वह संख्यात्मक फ़ॉर्मैट देता है जिसे उसका सिलिकॉन सबसे अच्छी तरह पढ़ता है, और हर टोकन को सेकंड जितना ही जूल में भी अनुकूलित करता है।

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · **🇮🇳 हिन्दी** · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Throughput and energy comparison against vLLM and llama.cpp" width="720"></p>

---

## विषय-सूची

- [दो विचार](#idees)
- [त्वरित शुरुआत](#demarrage)
- [इंस्टॉल करें](#installer)
- [`acvram plan` क्या बताता है](#plan)
- [तेज़ी से चलना](#optimisations)
- [HTTP एंडपॉइंट](#http)
- [आँकड़े कहाँ से आते हैं](#chiffres)
- [दस्तावेज़ीकरण](#documentation)
- [मापे गए परिणाम](#resultats)
- [स्थिति](#etat)
- [आभार](#credits)
- [लाइसेंस](#licence)
- [परियोजना का समर्थन करें](#soutien)

---

<a id="idees"></a>

## दो विचार

एक विशिष्ट मशीन के लिए बनाया गया:

| | |
|---|---|
| प्रोसेसर | Intel Core i9-14900K (8 P-कोर + 16 E-कोर) |
| मदरबोर्ड | ASUS ROG Maximus Z790 Dark Hero |
| मेमोरी | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| सिस्टम | Ubuntu 26.04 LTS (CUDA 13); दोनों कार्ड PCIe x8/x8 पर, पावर सीमा 400 W / 275 W |

**हर GPU के लिए एक फ़ॉर्मैट।** RTX 5090 में FP4 tensor cores हैं; RTX 3080 Ti में न तो ये हैं और न ही FP8। दोनों को एक साझा फ़ॉर्मैट पर लाने से 5090 की क्षमता व्यर्थ जाएगी। इसलिए कन्वर्टर *एक ही मॉडल को दो बार* लिखता है, उस फ़ॉर्मैट में जिसका हर गंतव्य वास्तव में उपयोग कर सकता है:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| वेट | **NVFP4** — E2M1 + हर 16 पर FP8 E4M3 स्केल | **INT4** — uint4 + हर 128 पर fp16 स्केल और ज़ीरो-पॉइंट |
| प्रति वेट बिट | 4.50 | 4.16 |
| BF16 की तुलना में | ×3.56 छोटा | ×3.85 छोटा |
| गणना मोड | FP4 tensor cores | कर्नेल के भीतर FP16 में डीक्वांटाइज़, FP16 tensor cores |
| KV कैश | INT8 | INT8 |

4.5 बिट प्रति वेट पर 32 GB VRAM में लगभग **56 अरब पैरामीटर** समा जाते हैं, जबकि BF16 में केवल 16 अरब। दोनों कार्डों को मिलाकर, होस्ट मेमोरी को छुए बिना ही लगभग **78 अरब रेज़िडेंट पैरामीटर** हो जाते हैं।

**मेमोरी एक पदानुक्रम है, दीवार नहीं।** तीन स्तर हैं, और प्लानर यह आशा करने के बजाय कि मॉडल समा जाएगा, मापता है कि हर स्तर की वास्तविक लागत क्या है:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
होस्ट DDR5   96 GB   PCIe या DDR द्वारा सीमित
```

---

<a id="demarrage"></a>

## त्वरित शुरुआत

```bash
./install.sh                       # वर्चुअल एनवायरनमेंट + torch cu128 + acvram
acvram doctor                      # क्या यह मशीन तैयार है, और किस काम के लिए
acvram detect                      # यहाँ वास्तव में क्या मौजूद है

acvram plan  ~/models/Qwen3-32B                     # हर लेयर कहाँ जाएगी
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

कोई भी OpenAI क्लाइंट तुरंत जुड़ जाता है:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"नमस्ते"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="अप्रयुक्त")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "नमस्ते"}])
```

---

<a id="installer"></a>

## इंस्टॉल करें

सोर्स से (सभी प्लेटफ़ॉर्म पर):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

या पैकेज के ज़रिए, हर [GitHub release](https://github.com/anticitoyun/anticitoyen-vram/releases/latest) के साथ एक फ़ाइल संलग्न होती है:

| चैनल | release के साथ संलग्न फ़ाइल | कमांड |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (नाम `rpmbuild` द्वारा बनाए जाते हैं, तय नहीं होते) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (या `rpmbuild --rebuild *.src.rpm`, जो `.src.rpm` से पैकेज बनाता है) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

एक `.flatpakref` हमेशा रिपॉज़िटरी का नवीनतम प्रकाशित संस्करण इंस्टॉल करता है।

इंस्टॉल करने से पहले, डाउनलोड की गई फ़ाइल को रिलीज़ से जुड़े योगों के विरुद्ध सत्यापित करें (`SHA256SUMS`, बाकी सभी फ़ाइलों के मौजूद होने के बाद प्रकाशित):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip पैकेज के रूप में प्रकाशित नहीं है (कोई wheel नहीं बनाया जाता): `pip install -e '.[dev]'` सोर्स के क्लोन से इंस्टॉल करता है, ठीक `./install.sh` की तरह।

---

<a id="plan"></a>

## `acvram plan` क्या बताता है

कोई भी डाउनलोड शुरू करने से पहले प्लानर चलाना सार्थक है। यह उन प्रश्नों का उत्तर देता है जो तय करते हैं कि कोई मॉडल इस मशीन पर उपयोग योग्य है भी या नहीं:

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  tier    format      capacity      weights       KV  slice
  cuda:0  nvfp4        30.3 GiB    25.5 GiB   4.5 GiB  layers 0-58
  cuda:1  int4_awq     10.9 GiB     8.7 GiB   1.6 GiB  layers 59-79
  cpu     nvfp4        74.8 GiB     3.4 GiB      0 B   -

  total weights      37.6 GiB
  read per token     35.1 GiB
  KV per token       162.5 KiB  -> 39,843 tokens cached
  MLP on host RAM    55-58

  decode estimate    17.8 tokens/s  (batch of 1)
  prefill estimate   847 tokens/s
```

यह पहले समा जाने वाले कॉन्फ़िगरेशन पर रुकने के बजाय पूरे कॉन्फ़िगरेशन-क्षेत्र को खंगालता है, और इसके दो निर्णय इतने प्रति-अंतर्ज्ञानी हैं कि उन्हें स्पष्ट रूप से बताना उचित है:

* **यह 3080 Ti को निष्क्रिय छोड़ देता है** जब कोई मॉडल अकेले 5090 पर समा जाता है। पाइपलाइन के चरण क्रम से चलते हैं: 1790 GB/s की पाइपलाइन में 912 GB/s का चरण जोड़ने से सिंगल-स्ट्रीम डिकोड धीमा हो जाता है। इसे `--gpus all` से बाध्य किया जा सकता है।
* **यह वेट को VRAM में रखने के लिए KV कैश को छोटा कर देता है।** कैश को दिया गया हर गीगाबाइट, PCIe बस पर धकेला गया एक गीगाबाइट वेट है, और PCIe के ज़रिए एक वेट पढ़ने की लागत VRAM से पढ़ने की तुलना में लगभग तीस गुना होती है। ऊपर के 70B पर, केवल यही एक समझौता डिकोड को 2.3 से 17.8 टोकन/s तक ले जाता है।

---

<a id="optimisations"></a>

## तेज़ी से चलना

चार ऑप्टिमाइज़ेशन, हर एक केवल स्टॉपवॉच से नहीं बल्कि समतुल्यता के प्रमाण से सत्यापित: जो ऑप्टिमाइज़ेशन उत्तर बदल दे, वह एक बग है।

डेंस मॉडलों के NVFP4 लीनियर डिफ़ॉल्ट रूप से Marlin लेआउट से होकर गुज़रते हैं (b = 8 पर +57 से +90% थ्रूपुट, TTFT +2 से +4 ms, revue/poste6-piece147-verdict-24-09.md के अनुसार; फ़ॉलबैक `ACVRAM_PROJ_MARLIN=0`, देखें [CHANGELOG.md](../CHANGELOG.md))।

### स्पेक्युलेटिव डिकोडिंग (`--speculative`)

आकार 1 के बैच के साथ एक टोकन डिकोड करना मेमोरी-बाउंड है: मशीन एक अकेला टोकन बनाने के लिए हर सक्रिय वेट पढ़ती है। K प्रस्तावित टोकनों की जाँच उन्हीं वेट को **सिर्फ़ एक बार** पढ़ती है। दो प्रस्तावक:

* `ngram` (डिफ़ॉल्ट) — वर्तमान प्रत्यय (suffix) को संदर्भ में पहले खोजता है और उसके बाद जो आया था उसे प्रस्तावित करता है। इसकी कोई लागत नहीं, किसी मॉडल की ज़रूरत नहीं। तब फ़ायदेमंद जब आउटपुट इनपुट को दोहराता है: कोड संपादन, RAG, सारांश।
* `draft` — दूसरे डिवाइस पर एक छोटा मॉडल। इस रिग पर वह डिवाइस RTX 3080 Ti है, जिसे प्लानर 5090 पर समा जाने वाले हर मॉडल के लिए जान-बूझकर निष्क्रिय छोड़ता है।

`mtp` (मॉडल का `nextn` हेड) और `auto` भी मौजूद हैं; वर्तमान रूप में लाभदायक नहीं और डिफ़ॉल्ट रूप से सक्षम नहीं — देखें `docs/ARCHITECTURE.md`।

स्वीकृति सटीक है, अनुमानित नहीं: किसी प्रस्ताव को `min(1, p/q)` प्रायिकता से स्वीकार किया जाता है, और अस्वीकृति होने पर `p - q` के सामान्यीकृत धनात्मक भाग से फिर से सैंपल लिया जाता है। जान-बूझकर ग़लत कैलिब्रेट किए गए ड्राफ़्ट के विरुद्ध 40,000 ड्रॉ पर मापने पर, उत्सर्जित वितरण लक्ष्य से 0.002 कुल विचलन (total variation) के भीतर रहता है — स्पेक्युलेशन गति ख़रीदता है, कभी कोई अलग उत्तर नहीं।

```
टॉय मॉडल, greedy, k=4         चरण   टोकन/चरण   आउटपुट
  बिना स्पेक्युलेशन               23        1.00   संदर्भ
  n-ग्राम                         13        1.77   समान
  draft (= लक्ष्य)                 5        4.60   समान
```

### प्रीफ़िक्स कैश (डिफ़ॉल्ट रूप से चालू)

ब्लॉकों को उनके टोकन-खंड के *शृंखलित* (chained) हैश से संबोधित किया जाता है: एक ही सिस्टम प्रॉम्प्ट साझा करने वाले दो अनुरोध उसके ब्लॉक भी साझा करते हैं, और दूसरे अनुरोध को उनकी पूर्व-गणना फिर से नहीं करनी पड़ती। शृंखलन अनिवार्य है: किसी भिन्न संदर्भ में वही सोलह टोकन समान keys और values नहीं रखते, और केवल खंड का हैश लेने से एक अनुक्रम का कैश दूसरे अनुक्रम को परोस दिया जाता।

मुक्त किया गया ऐसा ब्लॉक जिसकी सामग्री पहचानने योग्य बनी रहती है, मुक्त-ब्लॉक सूची के बजाय एक LRU कतार में शामिल हो जाता है: इस तरह कैश अनुरोधों के बीच बचा रहता है, और कभी ऐसा आवंटन अस्वीकार नहीं करता जिसे वह पूरा कर सकता था।

### होस्ट-स्तर पर गणना (`--host-exec`)

जिस लेयर के वेट RAM में रहते हैं, उसे GPU पर कॉपी किया जा सकता है या वहीं उसकी गणना की जा सकती है। दोनों रास्ते मेमोरी-बाउंड हैं और समान बाइट पढ़ते हैं: तेज़ रास्ता वह है जिसकी बस अधिक चौड़ी है — PCIe 5.0 x16 लगभग 54 GB/s देता है, डुअल-चैनल DDR5 लगभग 70 GB/s — और वहीं गणना करने से GPU भी किसी कॉपी की प्रतीक्षा करने के बजाय मुक्त रहता है।

यह तभी लाभदायक है जब CPU पैक किए गए 4-बिट वेट को सीधे पढ़े। इसीलिए AVX2 पथ वाला एक छोटा C++ कर्नेल है (`acvram_cpu.cpp`, ctypes के ज़रिए लोड, Python हेडर या ninja के बिना)। अपनी **स्केलर** फ़ॉलबैक शाखा पर भी यह `dequantize() @ x` को INT4 में 1.44 और NVFP4 में 3.21 गुना से पीछे छोड़ देता है, क्योंकि बाद वाला पहले पूरे मैट्रिक्स की 32-बिट कॉपी लिखता है।

Mistral-Large-123B पर प्लानर का अनुमान 1.35 से बढ़कर 2.42 टोकन/s हो जाता है।

### मिश्रित प्रिसिज़न (`--snr-floor`, डिफ़ॉल्ट रूप से बंद)

कन्वर्टर हर टेंसर के लिए लेयर-आउटपुट पर सिग्नल-टू-नॉइज़ अनुपात मापता है और `--snr-floor` से नीचे गिरने वाले टेंसरों को अधिक चौड़े फ़ॉर्मैट में प्रमोट कर सकता है, अधिकतम 15% टेंसरों और एक मूल्य-सीमा (`--promotion-cout-max`, जोड़े गए मेबीबाइट में) के भीतर।

यह फ़्लोर **डिफ़ॉल्ट रूप से शून्य** है: कुछ भी प्रमोट नहीं होता। डिकोडिंग मेमोरी-बैंडविड्थ से सीमित है, और `Huihui-Qwen3.8-27B` पर माप इसे तय कर देता है — 25 dB का फ़्लोर 2.0% परप्लेक्सिटी (42.591 बनाम 43.447, 16,383 टोकन का कॉर्पस) के बदले 13.4% मेमोरी और 10.6% थ्रूपुट की क़ीमत लेता है (18.50 GiB और 41.8 t/s बनाम 16.02 और 46.2)। जब गुणवत्ता गति से अधिक मायने रखती हो, तो `--snr-floor 25` पुराना व्यवहार लौटा देता है।

### और `acvram eval`

सिग्नल-टू-नॉइज़ अनुपात और लॉजिट कोसाइन समानता केवल सन्निकटन हैं। `acvram eval REP [REP ...]` स्लाइडिंग विंडो पर परप्लेक्सिटी मापता है, ताकि फ़ॉर्मैट का चुनाव प्रमाण के आधार पर तय हो:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  model                    ppl     bpp        size     tokens
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## HTTP एंडपॉइंट

| एंडपॉइंट | टिप्पणियाँ |
|---|---|
| `POST /v1/chat/completions` | SSE स्ट्रीम या एकल प्रतिक्रिया; मॉडल के चैट टेम्पलेट का उपयोग करता है |
| `POST /v1/completions` | प्रॉम्प्ट टेक्स्ट के रूप में या टोकन id के रूप में |
| `POST /v1/embeddings` | अंतिम hidden states का मीन-पूलिंग, L2-सामान्यीकृत, `dimensions` का पालन |
| `GET /v1/models` | साथ में एक `acvram` ब्लॉक: फ़ॉर्मैट, डिवाइस, KV कैश क्षमता |
| `GET /health`, `GET /metrics` | डिकोड थ्रूपुट, KV ब्लॉक अधिभोग |

इन प्रतिक्रियाओं के फ़ील्ड नाम अंग्रेज़ी में ही रहते हैं: यह OpenAI प्रोटोकॉल है, और उनका अनुवाद करने से हर मौजूदा क्लाइंट टूट जाएगा।

---

<a id="chiffres"></a>

## आँकड़े कहाँ से आते हैं

ऊपर उद्धृत हर मान इस रिपॉज़िटरी के कोड से आता है और `pytest` द्वारा जाँचा जाता है। संदर्भ कर्नेलों के साथ CPU पर लिए गए माप:

| फ़ॉर्मैट | बिट/वेट | वेट SNR | BF16 की तुलना में लॉजिट कोसाइन |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

इन मापों से निकले दो निष्कर्षों ने डिफ़ॉल्ट मान बदल दिए:

* **Hadamard रोटेशन INT4 की मदद करता है, NVFP4 की नहीं।** INT4 के 128 के समूह किसी अकेले आउटलायर चैनल को सोख नहीं सकते, इसलिए चरम मानों को फैलाना प्रति एक्टिवेशन एक n log n ट्रांसफ़ॉर्म के लायक है। NVFP4 के 16 के ब्लॉक पहले से ही अपना स्केल रखते हैं। इसीलिए `--hadamard auto`, जो इसे केवल INT4 पर लागू करता है।
* **KV कैश के लिए INT8, FP8 E4M3 से बेहतर है**, समान आकार पर 44 dB बनाम 32 dB, क्योंकि प्रति-(टोकन, हेड) स्केल पहले से ही वह डायनामिक रेंज दे देता है जिसके लिए FP8 एक्सपोनेंट बिट ख़र्च करता है। इसलिए दोनों कार्ड INT8 KV कैश का उपयोग करते हैं, भले ही 5090 FP8 कर सकती है। एक `k8v4` फ़ॉर्मैट (INT4 में values, कैश बाइट में −22%) विकल्प के रूप में मौजूद है, **अभी प्रमाणित नहीं** — देखें `docs/ARCHITECTURE.md`।

---

<a id="documentation"></a>

## दस्तावेज़ीकरण

| दस्तावेज़ | सामग्री |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **किसी दूसरी मशीन पर परियोजना फिर से शुरू करना** (फ़्रेंच में) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | हिस्से आपस में कैसे जुड़ते हैं |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | शुद्ध NVFP4 या प्रति-चैनल int8 में attention+GDN, एक Gated DeltaNet हाइब्रिड पर |
| [`docs/MATERIEL.md`](MATERIEL.md) | इस विशेष मशीन को ट्यून करना |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **क्या अभी पूरा नहीं हुआ है**, इसे सबसे पहले पढ़ें |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | कोड पर काम करने की परिपाटियाँ (भाषा, शैली, पुश करने से पहले की जाँचें) |

---

<a id="resultats"></a>

## मापे गए परिणाम (2026-09-22, 400 W पर RTX 5090, ऊर्जा मीटर पर ≥ 20 s की विंडो)

Qwen3-Coder-30B-A3B, NVFP4 (एक्सपर्ट) + INT8 (अटेंशन, हेड) में, हर इंजन के लिए एक ही प्रोटोकॉल (`outils/`, एक कार्ड, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| डिकोड, 12 अनुक्रम | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| डिकोड, 1 अनुक्रम | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| प्रीफ़िल pp2048 | **22,707 टोकन/s** | 21,054 | 8,671 (TabbyAPI, अब हटाया गया) |

¹ 2026-09-22 का शुद्धिपत्र: `serve` डिफ़ॉल्ट रूप से स्पेक्युलेट करता है (`--speculative ngram`, cli.py), प्रतिस्पर्धी नहीं करते; अब तक प्रकाशित 380.8 t/s स्पेक्युलेशन **के साथ** मापा गया था। बिना स्पेक्युलेशन के (`--speculative none`, वही शृंखला, revue/poste2-piece44-speculation-none-22-09.md): 283.6 t/s — b=1 पर acvram **तीसरे** स्थान पर है, llama.cpp और vLLM के पीछे। ऊर्जा में यह llama.cpp से आगे रहता है (0.601 बनाम 0.700 J/टोकन नेट)। b=12 पर स्पेक्युलेशन कभी सक्रिय नहीं होता (गार्ड `lot_max=2`): वह सेल पहले से ही बराबरी की शर्तों पर था।

² 2026-09-23, वही सत्र, वही HTTP क्लाइंट (`banc-llamacpp-16-09.py`, `acvram serve` और `vllm serve` के विरुद्ध), हर आर्म के इर्द-गिर्द `-lgc 2700` स्पष्ट रूप से सेट, सेल बारी-बारी से A V V A, हर आर्म पर ≥ 5 बैच, अंतर केवल 2σ से अधिक होने पर ही घोषित (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md)। acvram 0.6.38 (डिकोड पर w13, अनरोल किया गया अटेंशन रिडक्शन): अंतर −1.6%, **2σ से कम: थ्रूपुट में बराबरी**। J/टोकन में **vLLM 7.0% से आगे रहता है** (2σ से अधिक)। 0.6.37 के साथ इसी प्रोटोकॉल ने −4.7% दिया था।

³ नोट ² जैसा ही सत्र और प्रोटोकॉल, दोनों ओर बिना स्पेक्युलेशन: acvram 312.3 बनाम vLLM 284.8 — **थ्रूपुट में acvram 9.7% आगे** (2σ से अधिक); J/टोकन: **बराबरी** (0.04% अंतर, 2σ से कम)।

⁴ 2026-09-23, llama.cpp के विरुद्ध वही प्रोटोकॉल (revue/poste2-piece72-llamacpp-b1-23-09.md), पुनर्लिखित राउटर के साथ acvram 0.6.37 (+5.6%): acvram 310.8 बनाम llama.cpp 329.9 t/s — **थ्रूपुट में llama.cpp 5.8% आगे, J/टोकन में acvram 13.4% आगे** (0.598 बनाम 0.691)।

आज के थ्रूपुट आँकड़े (स्टेशन 1030, इको मोड `-lgc 2700`, पाइपलाइन सेवा में; CUDA ग्राफ़ में कैप्चर की गई greedy सैंपलिंग, 0.6.35 से डिफ़ॉल्ट)। b=12 पर acvram का आँकड़ा एक आधिकारिक सील्ड सेल है (6 अंतर्गुंथित विंडो की माध्यिका, हर विंडो पर क्लॉक पढ़ी गई)।

> **शुद्धिपत्र (2026-09-23)।** अब तक प्रकाशित vLLM तुलना (b=12: 1,782 बनाम 1,634 t/s; b=1: 290.6) में HTTP पर मापे गए acvram की तुलना **ऑफ़लाइन** (`LLM().generate()`) मापे गए vLLM से की गई थी, और 2026-09-22 के शुद्धिपत्र ने ग़लत दावा किया था कि vLLM सेल `vllm serve` से होकर गया था। 2026-09-23 को: दोनों के लिए वही HTTP क्लाइंट, और दोनों के लिए `-lgc` सेट (acvram स्टार्टअप पर अपना मान सेट करता है, `vllm serve` नहीं करता: इस सावधानी के बिना vLLM ~2,930 MHz पर चल रहा था, बनाम ~2,650)। नोट ² में परिणाम: b=12 पर vLLM 9.1% आगे।

2026-09-14 की सुबह acvram उसी सेल पर 630 t/s और 0.619 J/टोकन पर था: लाभ Blackwell के नेटिव FP4 MMA (`mma.sync … kind::mxf4nvf4`, bf16 पर ×7.9) से, बैच-बकेट के अनुसार grouped-GEMM MoE से, सिंगल-कर्नेल रूटिंग (प्रति स्टेप 3,677 → 1,517 लॉन्च) से, और प्रोजेक्शनों के लिए एक संकरे tensor-core GEMM से आते हैं। हर आँकड़े का अपना नोट `acvram-memoire/revue/` में है, जिसमें माप से पहले सील की गई भविष्यवाणी, उपकरण और उसका रेजीम दर्ज है — बिना रेजीम वाला आँकड़ा प्रकाशित नहीं किया जाता।

जहाँ acvram आगे है: sm_120 पर नेटिव NVFP4 में MLA मॉडल (GLM-4.7-Flash), जिन्हें vLLM केवल FP8 में परोसता है (b=1: सेवा में 165.35 t/s); और वे मॉडल जो VRAM में नहीं समाते। एकल-अनुक्रम डिकोड इनमें शामिल नहीं है: बिना स्पेक्युलेशन के, वहाँ acvram vLLM से 9.7% आगे है (नोट ³), थ्रूपुट में llama.cpp से 5.8% पीछे लेकिन ऊर्जा में उससे 13.4% आगे (नोट ⁴)। बड़े बैच पर, VRAM में समाने वाले MoE मॉडल पर, b=12 पर vLLM थ्रूपुट में बराबरी पर है (1,995.1 बनाम 2,027.0 t/s, 2σ से कम, नोट ²) लेकिन J/टोकन में 7.0% की बढ़त बनाए रखता है; वहाँ acvram 1,540 t/s (0.6.34) से बढ़कर 1,995 (0.6.38) तक पहुँचा है।

---

<a id="etat"></a>

## स्थिति

संस्करण 0.6.38। सब कुछ 5090 पर चलता है: `sm_120a` (नेटिव FP4) और `sm_86` के लिए कंपाइल किए गए CUDA कर्नेल, CUDA ग्राफ़, NVFP4/INT8/INT4 क्वांटाइज़ेशन, HTTP सर्वर। लागू सुरक्षा-उपाय: कार्ड कार्य-सत्रों के लिए अदृश्य है (`CUDA_VISIBLE_DEVICES` खाली) और केवल `outils/carte.sh` उसे, लॉक के तहत, एक समय में एक माप के लिए उधार देता है; एक निगरानी प्रक्रिया लॉक के बाहर हर पहुँच को लॉग करती है; एक से अधिक कार्ड पर फैला या 10 s से छोटा ऊर्जा माप अमान्य कर दिया जाता है; डिग्रेडेड रेजीम में लोड हुआ मॉडल यह बता देता है और किसी मुक़ाबले में शामिल नहीं होता।

4,107 परीक्षण (`pytest --collect-only -q`, CPU पर एक मिनट; GPU परीक्षण केवल `carte.sh` के तहत चलते हैं)। कार्य का लेखा-जोखा: `acvram-memoire/` (नियम, निर्देशिका, नोटबुक, कई सौ समीक्षा नोट)।

---

<a id="credits"></a>

## आभार

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, Apache-2.0 लाइसेंस: `acvram/kernels/marlin_port/` इसके Marlin कर्नेलों (MoE और डेंस) को पोर्ट करता है, और फ़ाइल-दर-फ़ाइल पूरा श्रेय [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE) में दर्ज है।
- **NVIDIA** — CUDA, Blackwell के FP4 tensor cores (`sm_120`), और वे लाइब्रेरी जिन पर यह परियोजना निर्भर है।
- **PyTorch** — टेंसर इंजन और C++/CUDA एक्सटेंशन।

स्वतंत्र परियोजना, ASUS, NVIDIA या vLLM परियोजना से संबद्ध नहीं।

---

<a id="licence"></a>

## लाइसेंस

इस रिपॉज़िटरी के कोड के लिए [GPL-3.0-or-later](../LICENSE)। `acvram/kernels/marlin_port/` में [vLLM](https://github.com/vllm-project/vllm) v0.29.0 से पोर्ट किया गया कोड है (`marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size` कर्नेल), Apache-2.0 लाइसेंस के तहत: हर फ़ाइल अपना मूल हेडर बनाए रखती है, लाइसेंस का पाठ `LICENSE-vllm` में है, और फ़ाइलों की सूची, मूल commit और संशोधन [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE) में हैं।

---

<a id="soutien"></a>

## परियोजना का समर्थन करें

acvram निजी हार्डवेयर पर विकसित किया जाता है। अगर यह परियोजना आपके लिए उपयोगी है:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

अनुवाद: [TRADUIRE.md](TRADUIRE.md) (फ़्रेंच में; परियोजना की योगदान मार्गदर्शिका का अभी अनुवाद नहीं हुआ है)।
