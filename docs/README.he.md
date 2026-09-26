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

שער הסקה (inference gateway) תואם ל-API של OpenAI, שמתייחס לזיכרון כאל היררכיה, נותן לכל GPU את הפורמט המספרי שהסיליקון שלו קורא בצורה הטובה ביותר, וממטב כל טוקן בג'אולים לא פחות מאשר בשניות.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · **🇮🇱 עברית** · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<div dir="rtl">

<p align="center"><img src="captures/resultats-22-09.png" alt="Throughput and energy comparison against vLLM and llama.cpp" width="720"></p>

---

## תוכן העניינים

- [שני רעיונות](#idees)
- [התחלה מהירה](#demarrage)
- [התקנה](#installer)
- [מה אומר `acvram plan`](#plan)
- [להאיץ](#optimisations)
- [נקודות קצה HTTP](#http)
- [מאין באים המספרים](#chiffres)
- [תיעוד](#documentation)
- [תוצאות מדודות](#resultats)
- [מצב](#etat)
- [קרדיטים](#credits)
- [רישיון](#licence)
- [תמיכה בפרויקט](#soutien)

---

<a id="idees"></a>

## שני רעיונות

נבנה עבור מכונה אחת מסוימת:

| | |
|---|---|
| מעבד | Intel Core i9-14900K (8 ליבות P + 16 ליבות E) |
| לוח אם | ASUS ROG Maximus Z790 Dark Hero |
| זיכרון | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| מערכת | Ubuntu 26.04 LTS (CUDA 13); שני הכרטיסים ב-PCIe x8/x8, עם הגבלת הספק של 400 W / 275 W |

**פורמט אחד לכל GPU.** ל-RTX 5090 יש ליבות tensor של FP4; ל-RTX 3080 Ti אין אותן, וגם לא FP8. יישור שניהם לפורמט משותף היה מבזבז את ה-5090. לכן הממיר כותב *את אותו מודל פעמיים*, בפורמט שכל יעד יודע לנצל בפועל:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| משקלים | **NVFP4** — E2M1 + קנה מידה FP8 E4M3 כל 16 | **INT4** — uint4 + קנה מידה ונקודת אפס fp16 כל 128 |
| ביטים למשקל | 4.50 | 4.16 |
| לעומת BF16 | קטן פי 3.56 | קטן פי 3.85 |
| מצב חישוב | ליבות tensor של FP4 | דה-קוונטיזציה ל-FP16 בתוך הקרנל, ליבות tensor של FP16 |
| מטמון KV | INT8 | INT8 |

‏32 GB של VRAM ב-4.5 ביטים למשקל מכילים כ-**56 מיליארד פרמטרים**, לעומת 16 מיליארד ב-BF16. על פני שני הכרטיסים, מדובר בכ-**78 מיליארד פרמטרים שוכני זיכרון** עוד לפני שנוגעים בזיכרון המארח בכלל.

**הזיכרון הוא היררכיה, לא קיר.** שלוש שכבות, והמתכנן מודד כמה כל אחת מהן עולה בפועל, במקום לקוות שהמודל ייכנס:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
Host DDR5    96 GB   limited by PCIe or DDR
```

---

<a id="demarrage"></a>

## התחלה מהירה

```bash
./install.sh                       # סביבה וירטואלית + torch cu128 + acvram
acvram doctor                      # האם המכונה הזו מוכנה, ולמה
acvram detect                      # מה באמת יש כאן

acvram plan  ~/models/Qwen3-32B                     # לאן הייתה הולכת כל שכבה
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

כל לקוח OpenAI מתחבר מיד:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"שלום"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="לא-בשימוש")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "שלום"}])
```

---

<a id="installer"></a>

## התקנה

מקוד המקור (בכל הפלטפורמות):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

או כחבילה, קובץ המצורף לכל [מהדורת GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| ערוץ | הקובץ המצורף למהדורה | פקודה |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (השמות נוצרים על ידי `rpmbuild` ואינם קבועים) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (או `rpmbuild --rebuild *.src.rpm` מתוך קובץ ה-`.src.rpm`) |
| Flatpak | `acvram-<version>.flatpak` | `flatpak install acvram-<version>.flatpak` |

Pip אינו מתפרסם כחבילה (לא נבנה קובץ wheel): `pip install -e '.[dev]'` מתקין מתוך שכפול של קוד המקור, בדומה ל-`./install.sh`.

---

<a id="plan"></a>

## מה אומר `acvram plan`

כדאי להריץ את המתכנן לפני כל הורדה. הוא עונה על השאלות שמכריעות אם מודל שמיש בכלל על המכונה הזו:

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

הוא סורק את מרחב התצורות במקום לעצור בראשונה שנכנסת, ושתיים מההחלטות שלו מנוגדות לאינטואיציה במידה שמצדיקה לפרט אותן:

* **הוא משאיר את ה-3080 Ti בטלה** כשמודל נכנס ל-5090 לבדה. השלבים של צינור (pipeline) רצים בטור: הוספת שלב של 912 GB/s לצינור של 1790 GB/s מאטה את הפענוח בזרם יחיד. אפשר לכפות זאת עם `--gpus all`.
* **הוא מקטין את מטמון ה-KV כדי להשאיר את המשקלים ב-VRAM.** כל ג'יגה-בייט שניתן למטמון הוא ג'יגה-בייט של משקלים שנדחק אל אפיק ה-PCIe, וקריאת משקל דרך PCIe עולה בערך פי שלושים מקריאתו מ-VRAM. במודל ה-70B שלמעלה, הפשרה הזו לבדה מעלה את הפענוח מ-2.3 ל-17.8 טוקנים/s.

---

<a id="optimisations"></a>

## להאיץ

ארבע אופטימיזציות, כל אחת מאומתת בהוכחת שקילות ולא רק בשעון עצר: אופטימיזציה שמשנה את התשובה היא באג.

השכבות הליניאריות ב-NVFP4 של מודלים צפופים עוברות כברירת מחדל דרך פריסת Marlin (תוספת של 57% עד 90% בתפוקה ב-b = 8, ‏TTFT גדל ב-2 עד 4 ms, לפי revue/poste6-piece147-verdict-24-09.md; מסלול חלופי `ACVRAM_PROJ_MARLIN=0`, ראו [CHANGELOG.md](../CHANGELOG.md)).

### פענוח ספקולטיבי (`--speculative`)

פענוח של טוקן אחד באצווה בגודל 1 מוגבל על ידי הזיכרון: המכונה קוראת כל משקל פעיל כדי להפיק טוקן יחיד. בדיקה של K טוקנים מוצעים קוראת את אותם משקלים **פעם אחת בלבד**. שני מציעים:

* `ngram` (ברירת מחדל) — מחפש את הסיומת הנוכחית מוקדם יותר בהקשר ומציע את מה שבא אחריה. אינו עולה דבר ואינו דורש מודל. משתלם כשהפלט משכפל את הקלט: עריכת קוד, RAG, סיכום.
* `draft` — מודל קטן על התקן שני. במערך הזה ההתקן הוא ה-RTX 3080 Ti, שהמתכנן משאיר בטלה במכוון לכל מודל שנכנס ל-5090.

גם `mtp` (ראש ה-`nextn` של המודל) ו-`auto` קיימים; הם אינם משתלמים במצבם הנוכחי ואינם מופעלים כברירת מחדל — ראו `docs/ARCHITECTURE.md`.

הקבלה מדויקת, לא מקורבת: הצעה מתקבלת בהסתברות `min(1, p/q)`, ודחייה דוגמת מחדש מהחלק החיובי המנורמל של `p - q`. במדידה על פני 40,000 הגרלות מול טיוטה שכוילה שלא כהלכה במכוון, ההתפלגות הנפלטת נשארת במרחק וריאציה כוללת של עד 0.002 מהיעד — הספקולציה קונה מהירות, לעולם לא תשובה אחרת.

```
toy model, greedy, k=4        steps   tokens/step   output
  no speculation                 23          1.00   reference
  n-grams                        13          1.77   identical
  draft (= target)                5          4.60   identical
```

### מטמון קידומות (מופעל כברירת מחדל)

הבלוקים ממוענים לפי הגיבוב *המשורשר* של פרוסת הטוקנים שלהם: שתי בקשות שחולקות הנחיית מערכת חולקות גם את הבלוקים שלה, והשנייה כבר אינה צריכה לחשב אותם מראש. השרשור הכרחי: אותם שישה-עשר טוקנים בהקשר אחר אינם מחזיקים אותם מפתחות וערכים, וגיבוב של הפרוסה לבדה היה מגיש את המטמון של רצף אחד לרצף אחר.

בלוק משוחרר שתוכנו נשאר מזוהה מצטרף לתור LRU ולא לרשימת הבלוקים הפנויים: כך המטמון שורד בין בקשות בלי לסרב אי-פעם להקצאה שהיה יכול לספק.

### חישוב בשכבת המארח (`--host-exec`)

שכבה שמשקליה שוכנים ב-RAM יכולה להיות מועתקת ל-GPU או מחושבת במקומה. שני המסלולים מוגבלים על ידי הזיכרון וקוראים את אותם בתים: המהיר מביניהם הוא זה שהאפיק שלו רחב יותר — PCIe 5.0 x16 נותן כ-54 GB/s, ‏DDR5 בערוץ כפול כ-70 GB/s — וחישוב במקום גם משאיר את ה-GPU פנוי, במקום לגרום לו להמתין להעתקה.

זה משתלם רק אם המעבד קורא ישירות את המשקלים הארוזים ב-4 ביטים. מכאן קרנל C++ קטן עם מסלול AVX2 (`acvram_cpu.cpp`, נטען דרך ctypes, בלי כותרות Python ובלי ninja). אפילו בענף הגיבוי **הסקלרי** שלו, הוא עוקף את `dequantize() @ x` בפקטור 1.44 ב-INT4 ו-3.21 ב-NVFP4, כי האחרון כותב תחילה עותק של 32 ביט של המטריצה כולה.

ב-Mistral-Large-123B, הערכת המתכנן עולה מ-1.35 ל-2.42 טוקנים/s.

### דיוק מעורב (`--snr-floor`, כבוי כברירת מחדל)

הממיר מודד את יחס האות לרעש ביציאת השכבה עבור כל טנזור, ויכול לקדם לפורמט רחב יותר כל טנזור שיורד מתחת ל-`--snr-floor`, בגבול של 15% מהטנזורים ובתקרת מחיר (`--promotion-cout-max`, במביבייטים נוספים).

הרצפה היא **אפס כברירת מחדל**: שום דבר אינו מקודם. הפענוח מוגבל על ידי רוחב הפס של הזיכרון, והמדידה על `Huihui-Qwen3.8-27B` מכריעה — רצפה של 25 dB עולה 13.4% בזיכרון ו-10.6% בתפוקה (18.50 GiB ו-41.8 t/s לעומת 16.02 ו-46.2) תמורת 2.0% בפרפלקסיה (42.591 לעומת 43.447, קורפוס של 16,383 טוקנים). `--snr-floor 25` מחזיר את ההתנהגות הקודמת כשהאיכות חשובה יותר מהמהירות.

### ו-`acvram eval`

יחס האות לרעש ודמיון הקוסינוס של הלוג'יטים הם קירובים. `acvram eval REP [REP ...]` מודד פרפלקסיה בחלון מחליק, כך שבחירת פורמט מוכרעת על סמך ראיות:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  model                    ppl     bpp        size     tokens
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## נקודות קצה HTTP

| נקודת קצה | הערות |
|---|---|
| `POST /v1/chat/completions` | זרם SSE או תגובה יחידה; משתמש בתבנית הצ'אט של המודל |
| `POST /v1/completions` | הנחיה כטקסט או כמזהי טוקנים |
| `POST /v1/embeddings` | ממוצע של המצבים החבויים הסופיים, מנורמל L2, הפרמטר `dimensions` נתמך |
| `GET /v1/models` | ובנוסף בלוק `acvram`: פורמטים, התקנים, קיבולת מטמון ה-KV |
| `GET /health`, `GET /metrics` | תפוקת פענוח, תפוסת בלוקי KV |

שמות השדות בתגובות אלה נשארים באנגלית: זהו פרוטוקול OpenAI, ותרגומם היה שובר כל לקוח קיים.

---

<a id="chiffres"></a>

## מאין באים המספרים

כל ערך שצוטט לעיל מופק על ידי קוד במאגר הזה ונבדק על ידי `pytest`. המדידות בוצעו על המעבד עם קרנלי הייחוס:

| פורמט | ביטים/משקל | SNR של המשקלים | קוסינוס הלוג'יטים לעומת BF16 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

שני ממצאים מהמדידות האלה שינו את ברירות המחדל:

* **סיבוב Hadamard עוזר ל-INT4 אך לא ל-NVFP4.** הקבוצות בנות 128 של INT4 אינן יכולות לספוג ערוץ חריג בודד, ולכן פיזור הערכים הקיצוניים שווה טרנספורמציה בעלות n log n לכל אקטיבציה. הבלוקים בני 16 של NVFP4 כבר נושאים קנה מידה משלהם. מכאן `--hadamard auto`, שמחיל אותו רק על INT4.
* **INT8 עדיף על FP8 E4M3 למטמון ה-KV**, ‏44 dB לעומת 32 dB באותו גודל, כי קנה מידה לכל (טוקן, ראש) כבר מספק את הטווח הדינמי שעליו FP8 מוציא ביטי מעריך. לכן שני הכרטיסים משתמשים במטמון KV ב-INT8, אף שה-5090 הייתה יכולה לעבוד ב-FP8. פורמט `k8v4` (ערכים ב-INT4, ‏−22% בבתים של המטמון) קיים כאפשרות, **לא מוסמך** — ראו `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## תיעוד

| מסמך | תוכן |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **המשך הפרויקט על מכונה אחרת** (בצרפתית) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | איך החלקים מתחברים זה לזה |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 טהור או attention+GDN ב-int8 לכל ערוץ, על היברידי Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | כוונון המכונה המסוימת הזו |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **מה שעוד לא נעשה**, יש לקרוא זאת ראשון |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | מוסכמות העבודה על הקוד (שפה, סגנון, בדיקות לפני דחיפה) |

---

<a id="resultats"></a>

## תוצאות מדודות (22/09/2026, ‏RTX 5090 ב-400 W, חלון של ≥ 20 s במד האנרגיה)

‏Qwen3-Coder-30B-A3B ב-NVFP4 (מומחים) + INT8 (קשב, ראש), אותו פרוטוקול לכל מנוע (`outils/`, כרטיס אחד, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| פענוח, 12 רצפים | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| פענוח, רצף 1 | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| prefill pp2048 | **22,707 טוקנים/s** | 21,054 | 8,671 (TabbyAPI, הוצא משימוש) |

¹ תיקון מ-22/09/2026: `serve` מבצע ספקולציה כברירת מחדל (`--speculative ngram`, ‏cli.py), והמתחרים לא; הנתון 380.8 t/s שפורסם עד כה נמדד עם ספקולציה. בלי ספקולציה (`--speculative none`, אותה שרשרת, revue/poste2-piece44-speculation-none-22-09.md): ‏283.6 t/s — acvram **שלישי** ב-b=1, אחרי llama.cpp ו-vLLM. באנרגיה הוא נשאר לפני llama.cpp ‏(0.601 לעומת 0.700 J/טוקן נטו). ב-b=12 הספקולציה לעולם אינה פעילה (שומר `lot_max=2`): התא הזה כבר היה בתנאים שווים.

² ‏23/09/2026, אותו מושב, אותו לקוח HTTP ‏(`banc-llamacpp-16-09.py` מול `acvram serve` ו-`vllm serve`), ‏`-lgc 2700` נקבע במפורש סביב כל זרוע, תאים לסירוגין A V V A, ‏≥ 5 אצוות לכל זרוע, פער מדווח רק מעבר ל-2σ ‏(revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). ‏acvram 0.6.38 (w13 בפענוח, רדוקציית קשב פרושה): פער של −1.6%, **מתחת ל-2σ: שוויון בתפוקה**. ב-J/טוקן, **vLLM נשאר לפני ב-7.0%** (מעבר ל-2σ). עם 0.6.37 אותו פרוטוקול נתן −4.7%.

³ אותו מושב ואותו פרוטוקול כמו ב-², בלי ספקולציה בשני הצדדים: acvram ‏312.3 לעומת vLLM ‏284.8 — **acvram לפני ב-9.7% בתפוקה** (מעבר ל-2σ); ב-J/טוקן: **שוויון** (פער של 0.04%, מתחת ל-2σ).

⁴ ‏23/09/2026, אותו פרוטוקול מול llama.cpp ‏(revue/poste2-piece72-llamacpp-b1-23-09.md), ‏acvram 0.6.37 עם הנתב שנכתב מחדש (+5.6%): acvram ‏310.8 לעומת llama.cpp ‏329.9 t/s — **llama.cpp לפני ב-5.8% בתפוקה, acvram לפני ב-13.4% ב-J/טוקן** (0.598 לעומת 0.691).

נתוני התפוקה של היום (תחנה 1030, משטר חסכוני `-lgc 2700`, צינור בשירות; דגימה חמדנית שנלכדה בגרף ה-CUDA, ברירת מחדל מאז 0.6.35). נתון ה-b=12 של acvram הוא תא רשמי חתום (חציון של 6 חלונות משולבים, קריאת שעון בכל חלון).

> **תיקון (23/09/2026).** ההשוואה ל-vLLM שפורסמה עד כה (b=12: ‏1,782 לעומת 1,634 t/s; ‏b=1: ‏290.6) העמידה את acvram שנמדד דרך HTTP מול vLLM שנמדד **במצב לא מקוון** (`LLM().generate()`), והתיקון מ-22/09 טען בטעות שתא ה-vLLM עבר דרך `vllm serve`. ב-23/09: אותו לקוח HTTP לשניהם, ו-`-lgc` נקבע לשניהם (acvram קובע את שלו בעת ההפעלה, `vllm serve` לא: בלי אמצעי הזהירות הזה vLLM רץ בכ-2,930 MHz לעומת כ-2,650). התוצאה בהערה ²: ‏vLLM לפני ב-9.1% ב-b=12.

בבוקר 14/09/2026 עמד acvram על 630 t/s ו-0.619 J/טוקן באותו תא: הרווחים נובעים מה-MMA הטבעי של FP4 ב-Blackwell (`mma.sync … kind::mxf4nvf4`, ‏×7.9 לעומת bf16), מ-MoE ב-GEMM מקובץ לפי דלי אצווה, מניתוב בקרנל יחיד (3,677 → 1,517 הפעלות לכל צעד), ומ-GEMM צר על ליבות tensor עבור ההטלות. לכל נתון יש הערה ב-`acvram-memoire/revue/` עם התחזית שנחתמה לפני המדידה, המכשיר והמשטר שלו — נתון ללא משטר אינו מתפרסם.

היכן acvram מוביל: מודלי MLA ‏(GLM-4.7-Flash) ב-NVFP4 טבעי ל-sm_120, ש-vLLM מגיש רק ב-FP8 ‏(b=1: ‏165.35 t/s בשירות); מודלים שאינם נכנסים ל-VRAM. פענוח ברצף יחיד אינו אחד מהם: בלי ספקולציה, acvram מקדים שם את vLLM ב-9.7% (הערה ³), ומפגר אחרי llama.cpp ב-5.8% בתפוקה אך מקדים אותו ב-13.4% באנרגיה (הערה ⁴). באצווה גדולה, במודל MoE שנכנס ל-VRAM, ‏vLLM בשוויון תפוקה ב-b=12 ‏(1,995.1 לעומת 2,027.0 t/s, מתחת ל-2σ, הערה ²) אך שומר על יתרון של 7.0% ב-J/טוקן; acvram התקדם שם מ-1,540 t/s ‏(0.6.34) ל-1,995 ‏(0.6.38).

---

<a id="etat"></a>

## מצב

גרסה 0.6.38. הכול רץ על ה-5090: קרנלי CUDA מהודרים ל-`sm_120a` (FP4 טבעי) ול-`sm_86`, גרפי CUDA, כימות NVFP4/INT8/INT4, שרת HTTP. מעקות בטיחות קיימים: הכרטיס בלתי נראה לסשנים של עבודה (`CUDA_VISIBLE_DEVICES` ריק) ורק `outils/carte.sh` משאיל אותו, תחת נעילה, למדידה אחת בכל פעם; משגיח רושם ביומן כל גישה מחוץ לנעילה; מדידת אנרגיה שמשתרעת על יותר מכרטיס אחד או על פחות מ-10 s נפסלת; מודל שנטען במשטר פגום מצהיר על כך ואינו נכנס לדו-קרב.

‏4,107 בדיקות (`pytest --collect-only -q`, דקה אחת על המעבד; בדיקות GPU רצות רק תחת `carte.sh`). יומן העבודה: `acvram-memoire/` (כללים, מדריך, מחברות, כמה מאות הערות סקירה).

---

<a id="credits"></a>

## קרדיטים

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, ברישיון Apache-2.0: ‏`acvram/kernels/marlin_port/` מעביר (port) את קרנלי ה-Marlin שלו (MoE וצפופים), עם ייחוס מלא קובץ אחר קובץ ב-[`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — ‏CUDA, ליבות ה-tensor של FP4 ב-Blackwell (`sm_120`), והספריות שהפרויקט הזה תלוי בהן.
- **PyTorch** — מנוע טנזורים והרחבות C++/CUDA.

פרויקט עצמאי, שאינו קשור ל-ASUS, ל-NVIDIA או לפרויקט vLLM.

---

<a id="licence"></a>

## רישיון

[GPL-3.0-or-later](../LICENSE) עבור הקוד במאגר הזה. `acvram/kernels/marlin_port/` מכיל קוד שהועבר מ-[vLLM](https://github.com/vllm-project/vllm) v0.29.0 (הקרנלים `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), תחת רישיון Apache-2.0: כל קובץ שומר על הכותרת המקורית שלו, טקסט הרישיון נמצא ב-`LICENSE-vllm`, ורשימת הקבצים, קומיט המקור והשינויים נמצאים ב-[`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## תמיכה בפרויקט

‏acvram מפותח על חומרה אישית. אם הפרויקט מועיל לך:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

תרגומים: [TRADUIRE.md](TRADUIRE.md) (בצרפתית; מדריך התרומה של הפרויקט עדיין לא תורגם).

</div>
