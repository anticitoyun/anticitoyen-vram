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

بوابة استدلال متوافقة مع واجهة OpenAI البرمجية، تعامل الذاكرة على أنها تسلسل هرمي، وتمنح كل GPU الصيغة الرقمية التي يقرؤها عتاده على أفضل وجه، وتُحسِّن كل رمز بالجول بقدر ما تُحسِّنه بالثواني.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · **🇸🇦 العربية** · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<div dir="rtl">

<p align="center"><img src="captures/resultats-22-09.png" alt="Throughput and energy comparison against vLLM and llama.cpp" width="720"></p>

---

## المحتويات

- [فكرتان](#idees)
- [البدء السريع](#demarrage)
- [التثبيت](#installer)
- [ماذا يقول `acvram plan`](#plan)
- [الذهاب بسرعة](#optimisations)
- [نقاط الدخول HTTP](#http)
- [من أين تأتي الأرقام](#chiffres)
- [التوثيق](#documentation)
- [النتائج المقيسة](#resultats)
- [الحالة](#etat)
- [الشكر والإسهامات](#credits)
- [الترخيص](#licence)
- [ادعم المشروع](#soutien)

---

<a id="idees"></a>

## فكرتان

صُمِّم لجهاز بعينه:

| | |
|---|---|
| المعالج | Intel Core i9-14900K (‏8 أنوية P + ‏16 نواة E) |
| اللوحة الأم | ASUS ROG Maximus Z790 Dark Hero |
| الذاكرة | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC، ‏32 GB — Blackwell، `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti، ‏12 GB — Ampere، `sm_86` |
| النظام | Ubuntu 26.04 LTS ‏(CUDA 13)؛ البطاقتان على PCIe x8/x8، بسقف قدرة 400 W / 275 W |

**صيغة لكل GPU.** تمتلك RTX 5090 أنوية موتّرات (tensor cores) بصيغة FP4؛ أما RTX 3080 Ti فلا تملكها، ولا تملك FP8 أيضًا. ومواءمة البطاقتين على صيغة مشتركة تُهدر قدرات الـ 5090. لذلك يكتب المحوِّل *النموذج نفسه مرتين*، بالصيغة التي تستطيع كل وجهة استغلالها فعلًا:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| الأوزان | **NVFP4** — E2M1 + مقياس FP8 E4M3 كل 16 | **INT4** — uint4 + مقياس ونقطة صفر fp16 كل 128 |
| بت لكل وزن | 4.50 | 4.16 |
| مقارنةً بـ BF16 | أصغر بـ ×3.56 | أصغر بـ ×3.85 |
| نمط الحساب | أنوية موتّرات FP4 | فك التكميم إلى FP16 داخل النواة، أنوية موتّرات FP16 |
| ذاكرة KV المؤقتة | INT8 | INT8 |

تتسع ذاكرة VRAM بسعة 32 GB، بمعدل 4.5 بت لكل وزن، لنحو **56 مليار معامل**، مقابل 16 مليارًا بصيغة BF16. وعلى البطاقتين معًا يصل ذلك إلى قرابة **78 مليار معامل مقيم** قبل المساس بذاكرة المضيف أصلًا.

**الذاكرة تسلسل هرمي، لا جدار.** ثلاث طبقات، والمخطِّط يقيس ما تكلّفه كل واحدة فعلًا بدل أن يأمل أن يتسع النموذج:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
Host DDR5    96 GB   limited by PCIe or DDR
```

---

<a id="demarrage"></a>

## البدء السريع

```bash
./install.sh                       # virtual environment + torch cu128 + acvram
acvram doctor                      # is this machine ready, and for what
acvram detect                      # what is actually here

acvram plan  ~/models/Qwen3-32B                     # where each layer would go
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

يتصل به أي عميل OpenAI مباشرةً:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"مرحبًا"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "مرحبًا"}])
```

---

<a id="installer"></a>

## التثبيت

من المصدر (جميع المنصات):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

أو عبر حزمة، وهي ملف مرفق بكل [إصدار على GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| القناة | الملف المرفق بالإصدار | الأمر |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (أسماء يولّدها `rpmbuild`، وهي غير ثابتة) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (أو `rpmbuild --rebuild *.src.rpm` انطلاقًا من `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

قبل التثبيت، تحقق من الملف الذي نزّلته مقابل المجاميع المرفقة بالإصدار (`SHA256SUMS`، تُنشر بعد وجود جميع الملفات الأخرى):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

لا تتوفر حزمة منشورة عبر Pip (لم تُبنَ أي حزمة wheel): الأمر `pip install -e '.[dev]'` يثبّت من نسخة مستنسخة من المصدر، تمامًا مثل `./install.sh`.

---

<a id="plan"></a>

## ماذا يقول `acvram plan`

يستحق المخطِّط أن يُشغَّل قبل أي تنزيل. فهو يجيب عن الأسئلة التي تحسم ما إذا كان النموذج صالحًا للاستخدام على هذا الجهاز أصلًا:

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

يستكشف المخطِّط فضاء الإعدادات بدل أن يكتفي بأول إعداد يتسع، واثنان من قراراته مخالفان للحدس بما يكفي ليستحقا التوضيح:

* **يترك الـ 3080 Ti دون استخدام** حين يتسع النموذج على الـ 5090 وحدها. فمراحل خط المعالجة (pipeline) تُنفَّذ على التوالي: وإضافة مرحلة بسرعة 912 GB/s إلى خط يعمل بسرعة 1790 GB/s تُبطئ فك الترميز أحادي التدفق. يمكن فرض استخدامها بـ `--gpus all`.
* **يُقلِّص ذاكرة KV المؤقتة لإبقاء الأوزان في VRAM.** كل غيغابايت يُمنح للذاكرة المؤقتة هو غيغابايت من الأوزان يُدفع إلى ناقل PCIe، وقراءة وزن عبر PCIe تكلّف نحو ثلاثين ضعف ما تكلّفه من VRAM. على نموذج 70B أعلاه، ترفع هذه المقايضة وحدها سرعة فك الترميز من 2.3 إلى 17.8 رمزًا/ث.

---

<a id="optimisations"></a>

## الذهاب بسرعة

أربعة تحسينات، كلٌّ منها مُتحقَّق منه ببرهان تكافؤ لا بساعة توقيت فحسب: فالتحسين الذي يغيّر الإجابة خطأ برمجي.

تمرّ الطبقات الخطية NVFP4 في النماذج الكثيفة افتراضيًا عبر ترتيب Marlin (من +57 إلى +90% في الإنتاجية عند b = 8، وTTFT من +2 إلى +4 ms وفق revue/poste6-piece147-verdict-24-09.md؛ البديل الاحتياطي `ACVRAM_PROJ_MARLIN=0`، انظر [CHANGELOG.md](../CHANGELOG.md)).

### فك الترميز التخميني (`--speculative`)

فك ترميز رمز واحد بدفعة حجمها 1 محدود بالذاكرة: إذ يقرأ الجهاز كل الأوزان النشطة لإنتاج رمز واحد. أما التحقق من K رموز مقترحة فيقرأ الأوزان نفسها **مرة واحدة فقط**. ثمة مقترِحان:

* `ngram` (الافتراضي) — يبحث عن اللاحقة الحالية في موضع سابق من السياق ويقترح ما تلاها. لا يكلّف شيئًا ولا يحتاج إلى أي نموذج. يؤتي ثماره حين يكرّر الخرجُ الدخلَ: تحرير الشيفرة، وRAG، والتلخيص.
* `draft` — نموذج صغير على جهاز ثانٍ. في هذه المنصة يكون ذلك الجهاز هو RTX 3080 Ti، التي يتركها المخطِّط عمدًا خاملة لأي نموذج يتسع على الـ 5090.

يوجد أيضًا `mtp` (رأس `nextn` في النموذج) و`auto`؛ غير مجديين في حالتهما الراهنة وغير مفعّلين افتراضيًا — انظر `docs/ARCHITECTURE.md`.

القبول دقيق لا تقريبي: يُقبل الاقتراح باحتمال `min(1, p/q)`، وعند الرفض يُعاد أخذ العينة من الجزء الموجب المُطبَّع من `p - q`. وبالقياس على 40,000 سحبة مقابل نموذج مسودة أُسيئت معايرته عمدًا، يبقى التوزيع الصادر ضمن 0.002 من التباين الكلي عن التوزيع المستهدف — التخمين يشتري السرعة، ولا يشتري إجابة مختلفة أبدًا.

```
toy model, greedy, k=4        steps   tokens/step   output
  no speculation                 23          1.00   reference
  n-grams                        13          1.77   identical
  draft (= target)                5          4.60   identical
```

### ذاكرة البادئات المؤقتة (مفعّلة افتراضيًا)

تُعنوَن الكتل بالتجزئة *المتسلسلة* لشريحة رموزها: فالطلبان اللذان يتشاركان موجِّه النظام يتشاركان كتله، ولا يعود الثاني بحاجة إلى حسابها مسبقًا. والتسلسل ضروري: فالرموز الستة عشر نفسها في سياق مختلف لا تحمل المفاتيح والقيم نفسها، وتجزئة الشريحة وحدها كانت ستقدّم الذاكرة المؤقتة لتسلسلٍ ما إلى تسلسلٍ آخر.

الكتلة المحرَّرة التي يظل محتواها قابلًا للتعرّف تنضم إلى طابور LRU بدل قائمة الكتل الحرة: وبهذا تبقى الذاكرة المؤقتة بين الطلبات دون أن ترفض أبدًا تخصيصًا كان بوسعها تلبيته.

### الحساب في طبقة المضيف (`--host-exec`)

الطبقة التي تقيم أوزانها في RAM يمكن نسخها إلى GPU أو حسابها في مكانها. وكلا المسارين محدود بالذاكرة ويقرأ البايتات نفسها: والأسرع هو صاحب الناقل الأعرض — يوفّر PCIe 5.0 x16 نحو 54 GB/s، وDDR5 ثنائية القناة نحو 70 GB/s — كما أن الحساب في المكان يترك GPU حرًا بدل أن يجعله ينتظر نسخة.

ولا يجدي هذا إلا إذا قرأ المعالج الأوزانَ المحزومة بـ 4 بت مباشرةً. ومن هنا نواة C++ صغيرة بمسار AVX2 ‏(`acvram_cpu.cpp`، تُحمَّل عبر ctypes، دون ترويسات Python ولا ninja). وحتى على فرعها الاحتياطي **العددي (scalar)**، تتفوّق على `dequantize() @ x` بعامل 1.44 في INT4 و3.21 في NVFP4، لأن الأخير يكتب أولًا نسخة 32 بت من المصفوفة كلها.

على Mistral-Large-123B، يرتفع تقدير المخطِّط من 1.35 إلى 2.42 رمزًا/ث.

### الدقة المختلطة (`--snr-floor`، معطَّلة افتراضيًا)

يقيس المحوِّل نسبة الإشارة إلى الضجيج عند خرج الطبقة لكل موتّر، ويمكنه ترقية ما يقع منها تحت `--snr-floor` إلى صيغة أوسع، في حدود 15% من الموتّرات وسقف للكلفة (`--promotion-cout-max`، بالميبيبايت المضافة).

الحد الأدنى **صفر افتراضيًا**: لا يُرقّى شيء. ففك الترميز محدود بعرض نطاق الذاكرة، والقياس على `Huihui-Qwen3.8-27B` يحسم الأمر — حد أدنى قدره 25 dB يكلّف 13.4% من الذاكرة و10.6% من الإنتاجية (18.50 GiB و41.8 t/s مقابل 16.02 و46.2) مقابل 2.0% في الحيرة (perplexity) ‏(42.591 مقابل 43.447، مدوّنة من 16,383 رمزًا). يعيد `--snr-floor 25` السلوك القديم حين تكون الجودة أهم من السرعة.

### و`acvram eval`

نسبة الإشارة إلى الضجيج وتشابه جيب التمام للـ logits تقريبات. أما `acvram eval REP [REP ...]` فيقيس الحيرة بنافذة منزلقة، ليُحسم اختيار الصيغة بالأدلة:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  model                    ppl     bpp        size     tokens
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## نقاط الدخول HTTP

| نقطة الدخول | ملاحظات |
|---|---|
| `POST /v1/chat/completions` | تدفق SSE أو استجابة واحدة؛ يستخدم قالب المحادثة الخاص بالنموذج |
| `POST /v1/completions` | الموجِّه نصًا أو معرّفات رموز |
| `POST /v1/embeddings` | متوسط الحالات المخفية النهائية، مُطبَّع L2، مع احترام `dimensions` |
| `GET /v1/models` | إضافة إلى كتلة `acvram`: الصيغ، والأجهزة، وسعة ذاكرة KV المؤقتة |
| `GET /health`، `GET /metrics` | إنتاجية فك الترميز، وإشغال كتل KV |

تبقى أسماء الحقول في هذه الاستجابات بالإنجليزية: فهذا بروتوكول OpenAI، وترجمتها ستُعطّل كل العملاء الموجودين.

---

<a id="chiffres"></a>

## من أين تأتي الأرقام

كل قيمة وردت أعلاه ناتجة عن شيفرة في هذا المستودع ومُتحقَّق منها بـ `pytest`. قياسات أُجريت على المعالج بالنوى المرجعية:

| الصيغة | بت/وزن | SNR الأوزان | جيب تمام الـ logits مقابل BF16 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

استنتاجان من هذه القياسات غيّرا القيم الافتراضية:

* **دوران Hadamard يفيد INT4 ولا يفيد NVFP4.** فمجموعات INT4 ذات الـ 128 عنصرًا لا تستطيع امتصاص قناة شاذة منعزلة، لذا فإن توزيع القيم المتطرفة يستحق تحويلًا بكلفة n log n لكل تنشيط. أما كتل NVFP4 ذات الـ 16 عنصرًا فتحمل مقياسها الخاص أصلًا. ومن هنا `--hadamard auto`، الذي لا يطبّقه إلا على INT4.
* **INT8 يتفوّق على FP8 E4M3 في ذاكرة KV المؤقتة**، ‏44 dB مقابل 32 dB بالحجم نفسه، لأن مقياسًا لكل (رمز، رأس) يوفّر أصلًا المدى الديناميكي الذي تنفق عليه FP8 بتات الأس. لذا تستخدم البطاقتان ذاكرة KV مؤقتة بصيغة INT8، مع أن الـ 5090 قادرة على FP8. وتوجد صيغة `k8v4` (قيم بصيغة INT4، ‏−22% من بايتات الذاكرة المؤقتة) كخيار، **غير مؤهَّل** — انظر `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## التوثيق

| المستند | المحتوى |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **استئناف المشروع على جهاز آخر** (بالفرنسية) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | كيف تترابط الأجزاء |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 نقي أو الانتباه+GDN بصيغة int8 لكل قناة، على هجين Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | ضبط هذا الجهاز بعينه |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **ما لم يُنجز بعد**، اقرأه أولًا |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | أعراف العمل على الشيفرة (اللغة، والأسلوب، والفحوص قبل الدفع) |

---

<a id="resultats"></a>

## النتائج المقيسة (2026-09-22، RTX 5090 عند 400 W، نافذة ≥ 20 ث على عدّاد الطاقة)

Qwen3-Coder-30B-A3B بصيغة NVFP4 (الخبراء) + INT8 (الانتباه، الرأس)، بالبروتوكول نفسه لكل المحركات (`outils/`، بطاقة واحدة، `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| فك الترميز، 12 تسلسلًا | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| فك الترميز، تسلسل 1 | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| prefill pp2048 | **22,707 رمزًا/ث** | 21,054 | 8,671 (TabbyAPI، متقاعد) |

¹ تصويب 2026-09-22: يقوم `serve` بالتخمين افتراضيًا (`--speculative ngram`، cli.py)، بينما لا يفعل المنافسون ذلك؛ والرقم 380.8 t/s المنشور حتى الآن قيس مع التخمين. ومن دون تخمين (`--speculative none`، السلسلة نفسها، revue/poste2-piece44-speculation-none-22-09.md): ‏283.6 t/s — أي أن acvram **ثالث** عند b=1، خلف llama.cpp وvLLM. وفي الطاقة يبقى متقدمًا على llama.cpp ‏(0.601 مقابل 0.700 J/رمز صافٍ). وعند b=12 لا يكون التخمين نشطًا أبدًا (الحارس `lot_max=2`): فتلك الخانة كانت أصلًا على قدم المساواة.

² 2026-09-23، الجلسة نفسها، وعميل HTTP نفسه (`banc-llamacpp-16-09.py` مقابل `acvram serve` و`vllm serve`)، مع ضبط `-lgc 2700` صراحةً حول كل ذراع، وخانات متناوبة A V V A، و≥ 5 دفعات لكل ذراع، ولا يُعلن الفارق إلا فوق 2σ ‏(revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 عند فك الترميز، واختزال انتباه مفرود): الفارق −1.6%، **تحت 2σ: تعادل في الإنتاجية**. أما في J/رمز، **فيبقى vLLM متقدمًا بنسبة 7.0%** (فوق 2σ). ومع 0.6.37 أعطى البروتوكول نفسه −4.7%.

³ الجلسة والبروتوكول نفساهما كما في ²، دون تخمين لدى الطرفين: acvram ‏312.3 مقابل vLLM ‏284.8 — **acvram متقدم بنسبة 9.7% في الإنتاجية** (فوق 2σ)؛ J/رمز: **تعادل** (فارق 0.04%، تحت 2σ).

⁴ 2026-09-23، البروتوكول نفسه مقابل llama.cpp ‏(revue/poste2-piece72-llamacpp-b1-23-09.md)، acvram 0.6.37 مع الموجِّه (router) المُعاد كتابته (+5.6%): acvram ‏310.8 مقابل llama.cpp ‏329.9 t/s — **llama.cpp متقدم بنسبة 5.8% في الإنتاجية، وacvram متقدم بنسبة 13.4% في J/رمز** (0.598 مقابل 0.691).

أرقام الإنتاجية الحالية (المحطة 1030، النظام الاقتصادي `-lgc 2700`، خط المعالجة في الخدمة؛ أخذ العينات الجشع ملتقَط داخل رسم CUDA البياني، وهو الافتراضي منذ 0.6.35). ورقم acvram عند b=12 خانة رسمية مختومة (وسيط 6 نوافذ متداخلة، مع قراءة الساعة لكل نافذة).

> **تصويب (2026-09-23).** كانت مقارنة vLLM المنشورة حتى الآن (b=12: ‏1,782 مقابل 1,634 t/s؛ b=1: ‏290.6) تقابل acvram مقيسًا عبر HTTP بـ vLLM مقيسًا **دون اتصال** (`LLM().generate()`)، وقد زعم تصويب 2026-09-22 خطأً أن خانة vLLM مرّت عبر `vllm serve`. وفي 2026-09-23: عميل HTTP نفسه للطرفين، و`-lgc` مضبوط للطرفين (acvram يضبط قيمته بنفسه عند الإقلاع، و`vllm serve` لا يفعل: ومن دون هذا الاحتياط كان vLLM يعمل عند ~2,930 MHz مقابل ~2,650). النتيجة في الملاحظة ²: vLLM متقدم بنسبة 9.1% عند b=12.

صباح 2026-09-14 كان acvram عند 630 t/s و0.619 J/رمز على الخانة نفسها: وتأتي المكاسب من MMA بصيغة FP4 الأصلية في Blackwell ‏(`mma.sync … kind::mxf4nvf4`، ×7.9 مقارنةً بـ bf16)، ومن MoE بعمليات GEMM مجمّعة لكل فئة دفعات، ومن توجيه بنواة واحدة (3,677 → 1,517 إطلاقًا لكل خطوة)، ومن GEMM ضيّق على أنوية الموتّرات للإسقاطات. لكل رقم ملاحظته في `acvram-memoire/revue/` مع التنبؤ المختوم قبل القياس، والأداة، ونظام التشغيل الخاص بها — والرقم الذي لا نظام له لا يُنشر.

حيث يتقدم acvram: نماذج MLA ‏(GLM-4.7-Flash) بصيغة NVFP4 الأصلية على sm_120، التي لا يقدّمها vLLM إلا بصيغة FP8 ‏(b=1: ‏165.35 t/s في الخدمة)؛ والنماذج التي لا تتسع في VRAM. وليس فك الترميز بتسلسل واحد من بينها: فمن دون تخمين، يتقدم acvram فيه على vLLM بنسبة 9.7% (الملاحظة ³)، ويتأخر عن llama.cpp بنسبة 5.8% في الإنتاجية لكنه يتقدم عليه بنسبة 13.4% في الطاقة (الملاحظة ⁴). وعند الدفعات الكبيرة، على نموذج MoE يتسع في VRAM، يتعادل vLLM في الإنتاجية عند b=12 ‏(1,995.1 مقابل 2,027.0 t/s، تحت 2σ، الملاحظة ²) لكنه يحتفظ بأفضلية 7.0% في J/رمز؛ وقد تقدّم acvram هناك من 1,540 t/s ‏(0.6.34) إلى 1,995 ‏(0.6.38).

---

<a id="etat"></a>

## الحالة

الإصدار 0.6.38. كل شيء يعمل على الـ 5090: نوى CUDA مُصرَّفة لـ `sm_120a` (FP4 أصلي) و`sm_86`، ورسوم CUDA البيانية، والتكميم NVFP4/INT8/INT4، وخادم HTTP. الضمانات القائمة: البطاقة غير مرئية لجلسات العمل (`CUDA_VISIBLE_DEVICES` فارغ) ولا يُعيرها إلا `outils/carte.sh`، تحت قفل، لقياس واحد في كل مرة؛ ومراقب يسجّل أي وصول خارج القفل؛ وأي قياس طاقة يشمل أكثر من بطاقة أو يقل عن 10 ث يُلغى؛ والنموذج المحمَّل في وضع متدهور يُصرّح بذلك ولا يدخل في مبارزة.

4,107 اختبارًا (`pytest --collect-only -q`، دقيقة واحدة على المعالج؛ ولا تعمل اختبارات GPU إلا تحت `carte.sh`). سجل العمل: `acvram-memoire/` (القواعد، والدليل، والدفاتر، ومراجعة من عدة مئات من الملاحظات).

---

<a id="credits"></a>

## الشكر والإسهامات

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0، بترخيص Apache-2.0: ينقل `acvram/kernels/marlin_port/` نوى Marlin الخاصة به (MoE والكثيفة)، مع إسناد كامل ملفًا بملف في [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — ‏CUDA، وأنوية الموتّرات FP4 في Blackwell ‏(`sm_120`)، والمكتبات التي يعتمد عليها هذا المشروع.
- **PyTorch** — محرك الموتّرات وامتدادات C++/CUDA.

مشروع مستقل، غير تابع لـ ASUS ولا NVIDIA ولا لمشروع vLLM.

---

<a id="licence"></a>

## الترخيص

[GPL-3.0-or-later](../LICENSE) للشيفرة في هذا المستودع. يحتوي `acvram/kernels/marlin_port/` على شيفرة منقولة من [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (النوى `marlin_moe_wna16` و`gptq_marlin_repack` و`moe_align_block_size`)، بترخيص Apache-2.0: يحتفظ كل ملف بترويسته الأصلية، ونص الترخيص موجود في `LICENSE-vllm`، وقائمة الملفات والإيداع الأصلي والتعديلات موجودة في [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## ادعم المشروع

يُطوَّر acvram على عتاد شخصي. إن كان المشروع مفيدًا لك:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

الترجمات: [TRADUIRE.md](TRADUIRE.md) (بالفرنسية؛ دليل المساهمة في المشروع لم يُترجم بعد).

</div>
