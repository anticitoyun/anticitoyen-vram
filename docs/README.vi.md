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

Một cổng suy luận tương thích với API OpenAI, coi bộ nhớ như một hệ phân cấp, trao cho mỗi GPU định dạng số mà silicon của nó đọc tốt nhất, và tối ưu từng token theo joule cũng như theo giây.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · **🇻🇳 Tiếng Việt** · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="So sánh thông lượng và năng lượng với vLLM và llama.cpp" width="720"></p>

---

## Mục lục

- [Hai ý tưởng](#idees)
- [Bắt đầu nhanh](#demarrage)
- [Cài đặt](#installer)
- [`acvram plan` cho biết gì](#plan)
- [Chạy nhanh](#optimisations)
- [Các điểm cuối HTTP](#http)
- [Các con số đến từ đâu](#chiffres)
- [Tài liệu](#documentation)
- [Kết quả đo được](#resultats)
- [Tình trạng](#etat)
- [Ghi công](#credits)
- [Giấy phép](#licence)
- [Ủng hộ dự án](#soutien)

---

<a id="idees"></a>

## Hai ý tưởng

Được xây dựng cho một cỗ máy cụ thể:

| | |
|---|---|
| CPU | Intel Core i9-14900K (8 nhân P + 16 nhân E) |
| Bo mạch chủ | ASUS ROG Maximus Z790 Dark Hero |
| Bộ nhớ | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Hệ thống | Ubuntu 26.04 LTS (CUDA 13); cả hai card chạy PCIe x8/x8, giới hạn công suất 400 W / 275 W |

**Mỗi GPU một định dạng.** RTX 5090 có tensor core FP4; RTX 3080 Ti không có loại này, và cũng không có FP8. Đưa cả hai về một định dạng chung sẽ lãng phí 5090. Vì vậy bộ chuyển đổi ghi *cùng một mô hình hai lần*, theo định dạng mà mỗi đích đến thực sự khai thác được:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| trọng số | **NVFP4** — E2M1 + hệ số tỉ lệ FP8 E4M3 cho mỗi 16 phần tử | **INT4** — uint4 + hệ số tỉ lệ và điểm không fp16 cho mỗi 128 phần tử |
| số bit mỗi trọng số | 4.50 | 4.16 |
| so với BF16 | nhỏ hơn ×3.56 | nhỏ hơn ×3.85 |
| chế độ tính toán | tensor core FP4 | giải lượng tử hóa sang FP16 ngay trong kernel, tensor core FP16 |
| bộ đệm KV | INT8 | INT8 |

32 GB VRAM ở mức 4.5 bit mỗi trọng số chứa được khoảng **56 tỷ tham số**, so với 16 tỷ ở BF16. Trên cả hai card, con số đó xấp xỉ **78 tỷ tham số thường trú** trước khi phải đụng đến bộ nhớ host.

**Bộ nhớ là một hệ phân cấp, không phải một bức tường.** Có ba tầng, và bộ lập kế hoạch đo chi phí thực tế của từng tầng thay vì hy vọng mô hình sẽ vừa:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
DDR5 host    96 GB   giới hạn bởi PCIe hoặc DDR
```

---

<a id="demarrage"></a>

## Bắt đầu nhanh

```bash
./install.sh                       # môi trường ảo + torch cu128 + acvram
acvram doctor                      # máy này đã sẵn sàng chưa, và cho việc gì
acvram detect                      # thực sự có gì ở đây

acvram plan  ~/models/Qwen3-32B                     # mỗi lớp sẽ được đặt ở đâu
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Mọi client OpenAI đều kết nối được ngay:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Xin chào"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="không dùng")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Xin chào"}])
```

---

<a id="installer"></a>

## Cài đặt

Từ mã nguồn (mọi nền tảng):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Hoặc qua gói cài đặt, mỗi gói là một tệp đính kèm trong từng [bản phát hành trên GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Kênh | Tệp đính kèm trong bản phát hành | Lệnh |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (tên do `rpmbuild` tạo ra, không cố định) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (hoặc `rpmbuild --rebuild *.src.rpm` từ tệp `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Một `.flatpakref` luôn cài đặt phiên bản mới nhất được công bố của kho lưu trữ.

Trước khi cài đặt, hãy xác minh tệp đã tải xuống với các tổng kiểm tra đính kèm bản phát hành (`SHA256SUMS`, được phát hành sau khi tất cả các tệp khác đã có mặt):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip không được phát hành dưới dạng gói (không có bản wheel dựng sẵn): `pip install -e '.[dev]'` cài đặt từ một bản sao mã nguồn, giống như `./install.sh`.

---

<a id="plan"></a>

## `acvram plan` cho biết gì

Bộ lập kế hoạch đáng được chạy trước mọi lần tải xuống. Nó trả lời những câu hỏi quyết định liệu một mô hình có dùng được trên máy này hay không:

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  tầng    định dạng   dung lượng    trọng số      KV  phần
  cuda:0  nvfp4        30.3 GiB    25.5 GiB   4.5 GiB  lớp 0-58
  cuda:1  int4_awq     10.9 GiB     8.7 GiB   1.6 GiB  lớp 59-79
  cpu     nvfp4        74.8 GiB     3.4 GiB      0 B   -

  tổng trọng số      37.6 GiB
  đọc mỗi token      35.1 GiB
  KV mỗi token       162.5 KiB  -> 39,843 token trong bộ đệm
  MLP trên RAM host  55-58

  ước tính giải mã   17.8 token/s  (lô cỡ 1)
  ước tính prefill   847 token/s
```

Nó khám phá không gian cấu hình thay vì dừng lại ở cấu hình đầu tiên vừa khít, và có hai quyết định của nó đủ phản trực giác để cần nói rõ:

* **Nó để 3080 Ti nhàn rỗi** khi mô hình vừa với riêng 5090. Các giai đoạn của một pipeline chạy nối tiếp: thêm một giai đoạn 912 GB/s vào một pipeline 1790 GB/s sẽ làm chậm quá trình giải mã một luồng. Có thể ép dùng bằng `--gpus all`.
* **Nó thu nhỏ bộ đệm KV để giữ trọng số trong VRAM.** Mỗi gigabyte dành cho bộ đệm là một gigabyte trọng số bị đẩy ra bus PCIe, và đọc một trọng số qua PCIe tốn khoảng ba mươi lần so với đọc từ VRAM. Với mô hình 70B ở trên, riêng sự đánh đổi này đã đưa tốc độ giải mã từ 2.3 lên 17.8 token/s.

---

<a id="optimisations"></a>

## Chạy nhanh

Bốn tối ưu hóa, mỗi cái đều được kiểm chứng bằng một chứng minh tương đương chứ không chỉ bằng đồng hồ bấm giờ: một tối ưu hóa làm thay đổi câu trả lời là một lỗi.

Các lớp tuyến tính NVFP4 của mô hình dense mặc định đi qua bố cục Marlin (+57 đến +90% thông lượng ở b = 8, TTFT +2 đến +4 ms theo revue/poste6-piece147-verdict-24-09.md; phương án dự phòng `ACVRAM_PROJ_MARLIN=0`, xem [CHANGELOG.md](../CHANGELOG.md)).

### Giải mã suy đoán (`--speculative`)

Giải mã một token với lô cỡ 1 bị giới hạn bởi bộ nhớ: máy phải đọc mọi trọng số đang hoạt động chỉ để tạo ra một token duy nhất. Kiểm tra K token được đề xuất chỉ đọc chính những trọng số đó **một lần duy nhất**. Có hai bộ đề xuất:

* `ngram` (mặc định) — tìm hậu tố hiện tại ở phần trước của ngữ cảnh và đề xuất những gì từng theo sau nó. Không tốn chi phí, không cần mô hình nào. Có lợi khi đầu ra lặp lại đầu vào: chỉnh sửa mã, RAG, tóm tắt.
* `draft` — một mô hình nhỏ trên thiết bị thứ hai. Trên dàn máy này, thiết bị đó là RTX 3080 Ti, card mà bộ lập kế hoạch cố ý để nhàn rỗi với mọi mô hình vừa với 5090.

`mtp` (đầu `nextn` của mô hình) và `auto` cũng có; ở trạng thái hiện tại chúng chưa đáng dùng và không được bật mặc định — xem `docs/ARCHITECTURE.md`.

Việc chấp nhận là chính xác, không phải xấp xỉ: một đề xuất được chấp nhận với xác suất `min(1, p/q)`, và một lần từ chối sẽ lấy mẫu lại từ phần dương đã chuẩn hóa của `p - q`. Đo trên 40,000 lần rút mẫu với một mô hình nháp cố ý hiệu chỉnh sai, phân phối phát ra vẫn nằm trong khoảng 0.002 biến phân toàn phần so với phân phối đích — suy đoán mua được tốc độ, không bao giờ đổi lấy một câu trả lời khác.

```
mô hình mẫu, tham lam, k=4    bước    token/bước    đầu ra
  không suy đoán                 23          1.00   tham chiếu
  n-gram                         13          1.77   giống hệt
  nháp (= đích)                   5          4.60   giống hệt
```

### Bộ đệm tiền tố (bật mặc định)

Các khối được định địa chỉ bằng giá trị băm *nối chuỗi* của đoạn token tương ứng: hai yêu cầu dùng chung một system prompt sẽ dùng chung các khối của nó, và yêu cầu thứ hai không còn phải tính trước chúng nữa. Việc nối chuỗi là thiết yếu: cùng mười sáu token nằm trong một ngữ cảnh khác không mang cùng các khóa và giá trị, và nếu chỉ băm riêng đoạn token thì bộ đệm của chuỗi này sẽ bị dùng cho một chuỗi khác.

Một khối đã giải phóng mà nội dung vẫn còn nhận dạng được sẽ vào một hàng đợi LRU thay vì danh sách khối trống: nhờ vậy bộ đệm tồn tại qua các yêu cầu mà không bao giờ từ chối một lần cấp phát lẽ ra nó có thể đáp ứng.

### Tính toán ở tầng host (`--host-exec`)

Một lớp có trọng số nằm trong RAM có thể được sao chép sang GPU hoặc được tính ngay tại chỗ. Cả hai cách đều bị giới hạn bởi bộ nhớ và đọc cùng những byte đó: cách nhanh hơn là cách có bus rộng hơn — PCIe 5.0 x16 cho khoảng 54 GB/s, DDR5 kênh đôi khoảng 70 GB/s — và tính tại chỗ còn để GPU rảnh thay vì bắt nó chờ một lần sao chép.

Điều này chỉ có lợi nếu CPU đọc trực tiếp các trọng số 4 bit đã đóng gói. Vì thế có một kernel C++ nhỏ với nhánh AVX2 (`acvram_cpu.cpp`, nạp qua ctypes, không cần header Python hay ninja). Ngay cả trên nhánh dự phòng **vô hướng**, nó vẫn nhanh hơn `dequantize() @ x` 1.44 lần với INT4 và 3.21 lần với NVFP4, vì cách sau trước tiên phải ghi ra một bản sao 32 bit của toàn bộ ma trận.

Với Mistral-Large-123B, ước tính của bộ lập kế hoạch tăng từ 1.35 lên 2.42 token/s.

### Độ chính xác hỗn hợp (`--snr-floor`, tắt mặc định)

Bộ chuyển đổi đo tỉ số tín hiệu trên nhiễu ở đầu ra của lớp cho từng tensor và có thể nâng những tensor rơi xuống dưới `--snr-floor` lên một định dạng rộng hơn, trong giới hạn 15% số tensor và một mức giá trần (`--promotion-cout-max`, tính bằng số mebibyte tăng thêm).

Ngưỡng sàn **mặc định bằng không**: không có gì được nâng. Giải mã bị giới hạn bởi băng thông bộ nhớ, và phép đo trên `Huihui-Qwen3.8-27B` đã phân định rõ — một ngưỡng sàn 25 dB tốn thêm 13.4% bộ nhớ và mất 10.6% thông lượng (18.50 GiB và 41.8 t/s so với 16.02 và 46.2) để đổi lấy 2.0% perplexity (42.591 so với 43.447, kho ngữ liệu 16,383 token). `--snr-floor 25` khôi phục hành vi cũ khi chất lượng quan trọng hơn tốc độ.

### Và `acvram eval`

Tỉ số tín hiệu trên nhiễu và độ tương đồng cosin của logit chỉ là những phép xấp xỉ. `acvram eval REP [REP ...]` đo perplexity theo cửa sổ trượt, để việc chọn định dạng được quyết định dựa trên bằng chứng:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  mô hình                  ppl     bpp  kích thước     token
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## Các điểm cuối HTTP

| điểm cuối | ghi chú |
|---|---|
| `POST /v1/chat/completions` | luồng SSE hoặc phản hồi đơn; dùng chat template của mô hình |
| `POST /v1/completions` | prompt dạng văn bản hoặc dạng danh sách id token |
| `POST /v1/embeddings` | trung bình các trạng thái ẩn cuối cùng, chuẩn hóa L2, hỗ trợ `dimensions` |
| `GET /v1/models` | kèm một khối `acvram`: định dạng, thiết bị, dung lượng bộ đệm KV |
| `GET /health`, `GET /metrics` | thông lượng giải mã, mức chiếm dụng khối KV |

Tên các trường trong những phản hồi này vẫn giữ tiếng Anh: đó là giao thức OpenAI, và dịch chúng sẽ làm hỏng mọi client hiện có.

---

<a id="chiffres"></a>

## Các con số đến từ đâu

Mọi giá trị nêu ở trên đều do mã trong kho này tạo ra và được `pytest` kiểm tra. Các phép đo được thực hiện trên CPU với các kernel tham chiếu:

| định dạng | bit/trọng số | SNR trọng số | cosin logit so với BF16 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

Hai phát hiện từ các phép đo này đã làm thay đổi các giá trị mặc định:

* **Phép quay Hadamard giúp INT4 nhưng không giúp NVFP4.** Các nhóm 128 của INT4 không thể hấp thụ một kênh ngoại lai đơn lẻ, nên việc dàn trải các giá trị cực trị đáng với cái giá một phép biến đổi n log n cho mỗi activation. Các khối 16 của NVFP4 vốn đã mang hệ số tỉ lệ riêng. Vì thế có `--hadamard auto`, chỉ áp dụng phép quay cho INT4.
* **INT8 thắng FP8 E4M3 cho bộ đệm KV**, 44 dB so với 32 dB ở cùng kích thước, vì một hệ số tỉ lệ theo từng cặp (token, head) đã cung cấp sẵn dải động mà FP8 phải tiêu tốn bit số mũ để có. Do đó cả hai card đều dùng bộ đệm KV INT8, dù 5090 có thể chạy FP8. Một định dạng `k8v4` (giá trị INT4, −22% số byte bộ đệm) có sẵn dưới dạng tùy chọn, **chưa được kiểm định** — xem `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **tiếp tục dự án trên một máy khác** (tiếng Pháp) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | các thành phần khớp với nhau như thế nào |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 thuần hoặc attention+GDN ở int8 theo từng kênh, trên một hybrid Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | tinh chỉnh chính cỗ máy này |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **những gì chưa làm xong**, hãy đọc trước tiên |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | quy ước làm việc với mã (ngôn ngữ, phong cách, kiểm tra trước khi push) |

---

<a id="resultats"></a>

## Kết quả đo được (22/09/2026, RTX 5090 ở 400 W, cửa sổ ≥ 20 s trên máy đo năng lượng)

Qwen3-Coder-30B-A3B ở NVFP4 (expert) + INT8 (attention, head), cùng một quy trình cho mọi engine (`outils/`, một card, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| giải mã, 12 chuỗi | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| giải mã, 1 chuỗi | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| prefill pp2048 | **22,707 token/s** | 21,054 | 8,671 (TabbyAPI, đã ngừng dùng) |

¹ Đính chính ngày 22/09/2026: `serve` mặc định dùng suy đoán (`--speculative ngram`, cli.py), còn các đối thủ thì không; con số 380.8 t/s công bố cho đến nay được đo CÓ suy đoán. Không suy đoán (`--speculative none`, cùng chuỗi xử lý, revue/poste2-piece44-speculation-none-22-09.md): 283.6 t/s — acvram đứng **thứ ba** ở b=1, sau llama.cpp và vLLM. Về năng lượng, nó vẫn dẫn trước llama.cpp (0.601 so với 0.700 J/token ròng). Ở b=12, suy đoán không bao giờ được kích hoạt (điều kiện chặn `lot_max=2`): ô đó vốn đã được so sánh ngang bằng.

² Ngày 23/09/2026, cùng phiên, cùng client HTTP (`banc-llamacpp-16-09.py` gọi tới `acvram serve` và `vllm serve`), `-lgc 2700` được đặt tường minh quanh mỗi nhánh, các ô xen kẽ theo thứ tự A V V A, ≥ 5 lô mỗi nhánh, chỉ báo cáo chênh lệch khi vượt quá 2σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 khi giải mã, phép rút gọn attention được trải vòng lặp): chênh lệch −1.6%, **dưới 2σ: ngang bằng về thông lượng**. Về J/token, **vLLM vẫn dẫn trước 7.0%** (vượt 2σ). Với 0.6.37, cùng quy trình cho ra −4.7%.

³ Cùng phiên và cùng quy trình như ², không suy đoán ở cả hai phía: acvram 312.3 so với vLLM 284.8 — **acvram dẫn trước 9.7% về thông lượng** (vượt 2σ); J/token: **ngang bằng** (chênh lệch 0.04%, dưới 2σ).

⁴ Ngày 23/09/2026, cùng quy trình so với llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 với bộ định tuyến được viết lại (+5.6%): acvram 310.8 so với llama.cpp 329.9 t/s — **llama.cpp dẫn trước 5.8% về thông lượng, acvram dẫn trước 13.4% về J/token** (0.598 so với 0.691).

Thông lượng hôm nay (máy 1030, chế độ tiết kiệm `-lgc 2700`, pipeline đang vận hành; lấy mẫu tham lam được ghi vào CUDA graph, mặc định từ 0.6.35). Con số b=12 của acvram là một ô chính thức đã được niêm phong (trung vị của 6 cửa sổ xen kẽ, xung nhịp được đọc theo từng cửa sổ).

> **Đính chính (23/09/2026).** Phép so sánh với vLLM công bố cho đến nay (b=12: 1,782 so với 1,634 t/s; b=1: 290.6) đã đặt acvram đo qua HTTP cạnh vLLM đo **ngoại tuyến** (`LLM().generate()`), và bản đính chính ngày 22/09/2026 đã khẳng định sai rằng ô vLLM đi qua `vllm serve`. Ngày 23/09/2026: cùng một client HTTP cho cả hai, và `-lgc` được đặt cho cả hai (acvram tự đặt khi khởi động, `vllm serve` thì không: thiếu biện pháp này, vLLM chạy ở ~2,930 MHz so với ~2,650). Kết quả ở ghi chú ²: vLLM dẫn trước 9.1% ở b=12.

Sáng ngày 14/09/2026, acvram đạt 630 t/s và 0.619 J/token trên cùng ô đó: các cải thiện đến từ MMA FP4 gốc của Blackwell (`mma.sync … kind::mxf4nvf4`, ×7.9 so với bf16), MoE bằng GEMM nhóm theo từng nhóm kích thước lô, định tuyến trong một kernel duy nhất (3,677 → 1,517 lần khởi chạy mỗi bước), và một GEMM hẹp trên tensor core cho các phép chiếu. Mỗi con số đều có ghi chú riêng trong `acvram-memoire/revue/` với dự đoán được niêm phong trước khi đo, công cụ đo và chế độ đo của nó — một con số không có chế độ đo thì không được công bố.

Nơi acvram dẫn đầu: các mô hình MLA (GLM-4.7-Flash) ở NVFP4 gốc sm_120, thứ mà vLLM chỉ phục vụ ở FP8 (b=1: 165.35 t/s khi vận hành); các mô hình không vừa VRAM. Giải mã một chuỗi không nằm trong số đó: không suy đoán, acvram dẫn trước vLLM 9.7% (ghi chú ³), kém llama.cpp 5.8% về thông lượng nhưng dẫn trước nó 13.4% về năng lượng (ghi chú ⁴). Ở lô lớn, trên một mô hình MoE vừa VRAM, vLLM ngang bằng về thông lượng ở b=12 (1,995.1 so với 2,027.0 t/s, dưới 2σ, ghi chú ²) nhưng vẫn giữ lợi thế 7.0% về J/token; acvram đã tiến bộ ở đó từ 1,540 t/s (0.6.34) lên 1,995 (0.6.38).

---

<a id="etat"></a>

## Tình trạng

Phiên bản 0.6.38. Mọi thứ đều chạy trên 5090: kernel CUDA biên dịch cho `sm_120a` (FP4 gốc) và `sm_86`, CUDA graph, lượng tử hóa NVFP4/INT8/INT4, máy chủ HTTP. Các rào chắn đã có: card vô hình với các phiên làm việc (`CUDA_VISIBLE_DEVICES` rỗng) và chỉ `outils/carte.sh` cho mượn nó, dưới khóa, cho từng phép đo một; một tiến trình giám sát ghi nhật ký mọi truy cập ngoài khóa; một phép đo năng lượng trải trên nhiều hơn một card hoặc ngắn hơn 10 s sẽ bị vô hiệu; một mô hình được nạp trong chế độ suy giảm sẽ tự báo như vậy và không tham gia đối đầu.

4,107 bài kiểm thử (`pytest --collect-only -q`, một phút trên CPU; các bài kiểm thử GPU chỉ chạy dưới `carte.sh`). Nhật ký công việc: `acvram-memoire/` (quy tắc, danh bạ, sổ ghi chép, vài trăm ghi chú rà soát).

---

<a id="credits"></a>

## Ghi công

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, giấy phép Apache-2.0: `acvram/kernels/marlin_port/` port lại các kernel Marlin của nó (MoE và dense), với ghi công đầy đủ theo từng tệp trong [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, tensor core FP4 của Blackwell (`sm_120`), và các thư viện mà dự án này phụ thuộc vào.
- **PyTorch** — engine tensor và các phần mở rộng C++/CUDA.

Dự án độc lập, không liên kết với ASUS, NVIDIA hay dự án vLLM.

---

<a id="licence"></a>

## Giấy phép

[GPL-3.0-or-later](../LICENSE) cho mã trong kho này. `acvram/kernels/marlin_port/` chứa mã được port từ [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (các kernel `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), theo giấy phép Apache-2.0: mỗi tệp giữ nguyên phần đầu gốc, văn bản giấy phép nằm trong `LICENSE-vllm`, còn danh sách tệp, commit gốc và các sửa đổi nằm trong [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Ủng hộ dự án

acvram được phát triển trên phần cứng cá nhân. Nếu dự án hữu ích với bạn:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Bản dịch: [TRADUIRE.md](TRADUIRE.md) (tiếng Pháp; hướng dẫn đóng góp của dự án chưa được dịch).
