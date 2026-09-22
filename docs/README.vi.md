<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Ủng hộ: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Một cổng suy luận tương thích với API OpenAI, coi bộ nhớ như một hệ thống
phân cấp và trao cho mỗi GPU định dạng số mà silicon của nó đọc tốt nhất.

Được thiết kế cho một máy cụ thể:

| | |
|---|---|
| Bộ xử lý | Intel Core i9-14900K (8 nhân P + 16 nhân E) |
| Bo mạch chủ | ASUS ROG Maximus Z790 Dark Hero |
| Bộ nhớ | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Hệ thống | Ubuntu 26.04 LTS (CUDA 13); cả hai card ở PCIe x8/x8, giới hạn 400 W / 275 W |

## Hai ý tưởng

**Mỗi GPU một định dạng.** RTX 5090 có nhân tensor FP4; RTX 3080 Ti không có,
cũng không có FP8. Ép cả hai về một định dạng chung sẽ lãng phí 5090. Vì thế
bộ chuyển đổi ghi *cùng một mô hình hai lần*, ở định dạng mà mỗi đích thực
sự khai thác được:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| trọng số | **NVFP4** — E2M1 + thang FP8 E4M3 mỗi 16 | **INT4** — uint4 + thang và điểm không fp16 mỗi 128 |
| bit mỗi trọng số | 4,50 | 4,16 |
| so với BF16 | nhỏ hơn ×3,56 | nhỏ hơn ×3,85 |
| chế độ tính toán | nhân tensor FP4 | giải lượng tử về FP16 trong kernel, nhân tensor FP16 |
| bộ đệm KV | INT8 | INT8 |

32 GB VRAM ở 4,5 bit mỗi trọng số chứa khoảng **56 tỷ tham số**, so với
16 tỷ ở BF16. Trên cả hai card, đó là xấp xỉ **78 tỷ tham số thường trú**
trước cả khi chạm tới RAM.

**Bộ nhớ là một hệ thống phân cấp, không phải bức tường.** Ba tầng, và bộ
lập kế hoạch đo chi phí của từng tầng thay vì hy vọng mô hình vừa:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Bắt đầu nhanh

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Sau đó bất kỳ client OpenAI nào cũng kết nối được:

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

## `acvram plan` nói gì

Bộ lập kế hoạch đáng được chạy trước mọi lần tải về. Nó trả lời những câu hỏi
quyết định một mô hình có dùng được trên máy này hay không:

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

Nó khám phá không gian cấu hình thay vì giữ cấu hình đầu tiên vừa, và hai
quyết định của nó đủ trái trực giác để cần nói rõ:

* **Nó để 3080 Ti không dùng** khi một mô hình vừa trên riêng 5090. Các
  lát của một pipeline chạy nối tiếp: thêm một chặng 912 GB/s vào pipeline
  1790 GB/s làm chậm giải mã đơn luồng. Ép buộc bằng `--gpus all`.
* **Nó thu nhỏ bộ đệm KV để giữ trọng số trong VRAM.** Mỗi gigabyte trao cho
  bộ đệm là một gigabyte trọng số bị đẩy ra bus PCIe, và đọc một trọng số qua
  PCIe tốn khoảng ba mươi lần so với từ VRAM. Trên 70B ở trên, chỉ riêng
  sự cân nhắc này đưa từ 2,3 lên 17,8 token/s.

## Đi nhanh

Bốn tối ưu, mỗi cái được kiểm chứng bằng chứng minh tương đương chứ không chỉ
bằng đồng hồ bấm giờ: một tối ưu làm thay đổi câu trả lời là một lỗi.

### Giải mã suy đoán (`--speculative`)

Giải mã một token với lô kích thước 1 bị giới hạn bởi bộ nhớ: máy đọc toàn
bộ trọng số đang hoạt động để tạo ra một token duy nhất. Xác minh K token đề
xuất chỉ đọc các trọng số ấy **một lần**. Hai bộ đề xuất:

* `ngram` (mặc định) — tìm hậu tố hiện tại ở phần trước của ngữ cảnh và đề
  xuất phần theo sau. Không tốn gì, không cần mô hình. Có lợi khi đầu ra sao
  chép đầu vào: sửa mã, RAG, tóm tắt.
* `draft` — một mô hình nhỏ trên thiết bị thứ hai. Trên dàn máy này, thiết
  bị đó là RTX 3080 Ti, được bộ lập kế hoạch cố ý để rảnh cho mọi mô hình
  vừa trên 5090.

Việc chấp nhận là chính xác, không xấp xỉ: một đề xuất được chấp nhận với xác
suất `min(1, p/q)` và một lần từ chối lấy mẫu lại từ phần dương chuẩn hóa của
`p - q`. Đo trên 40 000 lần rút so với một bản nháp cố ý hiệu chỉnh sai, phân
phối phát ra vẫn trong 0,002 biến thiên toàn phần so với đích — suy đoán mua
tốc độ, không bao giờ mua một câu trả lời khác.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Bộ đệm tiền tố (bật mặc định)

Các khối được định địa chỉ bằng băm *xâu chuỗi* của lát token của chúng: hai
yêu cầu dùng chung một chỉ dẫn hệ thống dùng chung các khối của nó, và yêu
cầu thứ hai không phải tính trước chúng nữa. Xâu chuỗi là bắt buộc: cùng mười
sáu token trong ngữ cảnh khác không chứa cùng khóa và giá trị, và chỉ băm lát
sẽ đưa bộ đệm của chuỗi này cho chuỗi khác.

Một khối được giải phóng mà nội dung vẫn nhận diện được sẽ vào hàng đợi LRU
thay vì danh sách khối trống: bộ đệm nhờ đó tồn tại qua các yêu cầu mà không
bao giờ từ chối một cấp phát mà nó có thể phục vụ.

### Tính toán ở tầng máy chủ (`--host-exec`)

Một lớp có trọng số nằm trong RAM có thể được sao chép lên GPU hoặc tính tại
chỗ. Cả hai đường đều bị giới hạn bộ nhớ và đọc cùng các byte: đường nhanh
hơn là đường có bus rộng hơn — PCIe 5.0 x16 cho khoảng 54 GB/s, DDR5 hai
kênh khoảng 70 GB/s — và tính tại chỗ còn để GPU rảnh thay vì bắt nó chờ một
bản sao.

Điều này chỉ đáng khi bộ xử lý đọc trực tiếp trọng số đóng gói 4 bit. Vì
thế có một kernel C++ nhỏ với đường AVX2 (`acvram_cpu.cpp`, nạp qua ctypes,
không cần header Python hay ninja). Ngay cả ở nhánh dự phòng **vô hướng**, nó
vượt `dequantize() @ x` với hệ số 1,44 ở INT4 và 3,21 ở NVFP4, vì cái sau ghi
trước một bản sao 32 bit của toàn bộ ma trận.

Trên Mistral-Large-123B, ước lượng của bộ lập kế hoạch tăng từ 1,35 lên
2,42 token/s.

### Độ chính xác hỗn hợp (`--snr-floor`, tắt mặc định)

Bộ chuyển đổi đo tỷ số tín hiệu/nhiễu ở đầu ra mỗi lớp cho từng tensor và có
thể nâng lên định dạng rộng hơn những tensor rơi dưới `--snr-floor`, trong
giới hạn 15 % số tensor và một trần giá (`--promotion-cout-max`, tính bằng
mebibyte thêm vào).

Ngưỡng **mặc định bằng không**: không gì được nâng. Giải mã bị giới hạn bởi
băng thông bộ nhớ, và phép đo trên `Huihui-Qwen3.8-27B` quyết định — ngưỡng
25 dB tốn 13,4 % bộ nhớ và 10,6 % thông lượng (18,50 GiB và 41,8 t/s so với
16,02 và 46,2) để đổi lấy 2,0 % perplexity (42,591 so với 43,447, kho ngữ
liệu 16 383 token). `--snr-floor 25` khôi phục hành vi cũ khi chất lượng quan
trọng hơn tốc độ.

### Và `acvram eval`

Tỷ số tín hiệu/nhiễu và cosin của logit là các xấp xỉ.
`acvram eval THƯMỤC [THƯMỤC ...]` đo perplexity bằng cửa sổ trượt, để việc
chọn định dạng được quyết định dựa trên bằng chứng:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Điểm cuối HTTP

| điểm cuối | ghi chú |
|---|---|
| `POST /v1/chat/completions` | luồng SSE hoặc phản hồi đơn; dùng mẫu hội thoại của mô hình |
| `POST /v1/completions` | prompt dạng văn bản hoặc id token |
| `POST /v1/embeddings` | trạng thái ẩn cuối lấy trung bình, chuẩn hóa L2, tôn trọng `dimensions` |
| `GET /v1/models` | cộng thêm khối `acvram`: định dạng, thiết bị, dung lượng bộ đệm KV |
| `GET /health`, `GET /metrics` | thông lượng giải mã, mức chiếm dụng khối KV |

Tên trường trong các phản hồi này giữ nguyên tiếng Anh: đó là giao thức
OpenAI, và dịch chúng sẽ làm hỏng mọi client hiện có.

## Các con số từ đâu ra

Mọi giá trị trích dẫn ở trên đều do mã trong kho này tạo ra và được `pytest`
kiểm tra. Đo trên bộ xử lý với các kernel tham chiếu:

| định dạng | bit/trọng số | SNR trọng số | cosin logit so với BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Hai nhận xét từ các phép đo này đã thay đổi giá trị mặc định:

* **Phép xoay Hadamard giúp INT4 chứ không giúp NVFP4.** Các nhóm 128 của
  INT4 không hấp thụ được một kênh ngoại lai đơn lẻ, nên trải các giá trị cực
  trị đáng giá một phép biến đổi n log n mỗi lần kích hoạt. Các khối 16 của
  NVFP4 đã mang thang riêng. Vì thế `--hadamard auto` chỉ áp dụng cho INT4.
* **INT8 thắng FP8 E4M3 cho bộ đệm KV**, 44 dB so với 32 dB ở cùng kích
  thước, vì một thang theo (token, đầu) đã cung cấp dải động mà FP8 tiêu bit
  số mũ cho nó. Do đó cả hai card dùng bộ đệm KV INT8, dù 5090 có thể làm FP8.

## Tài liệu

* [`REPRISE.md`](../REPRISE.md) — **tiếp tục dự án trên máy khác**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — các phần ghép với nhau thế nào
* [`docs/MATERIEL.md`](MATERIEL.md) — tinh chỉnh chính máy này
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **những gì chưa làm**, đọc trước
* [`CONVENTIONS.md`](../CONVENTIONS.md) — quy ước làm việc với mã (ngôn ngữ, phong cách, kiểm tra trước khi push)

## Kết quả đo được (22/09/2026, RTX 5090 ở 400 W, chế độ ≥ 20 s trên đồng hồ năng lượng)

Qwen3-Coder-30B-A3B ở NVFP4 (chuyên gia) + INT8 (chú ý, đầu), cùng giao
thức cho mọi engine (`outils/`, một card, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| giải mã 12 chuỗi | **1 625,5 t/s** | 1 596,1 t/s | — |
| giải mã 1 chuỗi | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707** token/s | 21 054 | 8 671 (TabbyAPI, đã rút) |

Thông lượng trong ngày (giàn 1030, chế độ tiết kiệm `-lgc 2700`, pipeline đang phục vụ; lấy mẫu tham lam được bắt trong đồ thị CUDA, mặc định từ 0.6.35). b=12 là một ô chính thức đã niêm phong (trung vị của 6 cửa sổ đan xen, xung nhịp theo cửa sổ). Giá trị vLLM 1 596,1 là tham chiếu đóng băng từ 21/09 (vLLM không chạy lại ngày đó): chênh +1,84 % có giá trị ở tham chiếu ngang nhau, không phải như đo lại cả hai cùng một sáng. J/token ở cùng xung nhịp so với ba engine vẫn đang được đo lại (`outils/gpu/mesure/banc-4moteurs.py`) — một con số không có chế độ thì không công bố.

Sáng 14/09 acvram ở 630 t/s và 0,619 J/token trên cùng ô: mức tăng đến từ
MMA FP4 gốc của Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 so với bf16),
từ MoE dưới dạng GEMM gộp theo gầu lô, từ định tuyến trong một kernel duy
nhất (3 677 → 1 517 lần khởi chạy mỗi bước) và từ một GEMM hẹp trên nhân
tensor cho các phép chiếu. Mỗi con số có ghi chú trong `acvram-memoire/revue/`
với dự đoán được niêm phong trước khi đo, dụng cụ và chế độ của nó — một con
số không có chế độ thì không được công bố.

Nơi acvram dẫn trước: các mô hình MLA (GLM-4.7-Flash) ở NVFP4 gốc sm_120
(b=1: 165,35 t/s đang phục vụ), mà vLLM chỉ phục vụ ở FP8; các mô hình không
vừa VRAM; và, từ 0.6.35, giải mã lô lớn của một MoE vừa VRAM — b=12 tăng từ
1 540 (0.6.34) lên 1 625,5 t/s, tức +1,84 % dẫn trước tham chiếu vLLM đóng băng
(1 596,1). Chênh lệch vẫn hẹp và ở tham chiếu đóng băng; chênh lệch về năng
lượng còn phải đo lại.

## Trạng thái

Phiên bản 0.6.35. Mọi thứ chạy trên 5090: kernel CUDA biên dịch cho `sm_120a`
(FP4 gốc) và `sm_86`, đồ thị CUDA, lượng tử hóa NVFP4/INT8/INT4, máy chủ
HTTP. Rào chắn đã có: card vô hình với các phiên làm việc
(`CUDA_VISIBLE_DEVICES` trống) và chỉ `outils/carte.sh` cho mượn nó, dưới
khóa, cho một phép đo mỗi lần; một người canh ghi lại mọi truy cập ngoài
khóa; một phép đo năng lượng bao trùm hơn một card hoặc dưới 10 s bị vô hiệu;
một mô hình nạp ở chế độ suy giảm nói ra điều đó và không tham gia đấu tay
đôi.

640 bài kiểm thử (`pytest -q`, một phút trên bộ xử lý; kiểm thử GPU chỉ chạy
dưới `carte.sh`). Theo dõi công việc: `acvram-memoire/` (quy tắc, danh bạ,
sổ tay, rà soát 180 ghi chú).

## Ủng hộ

acvram được phát triển trên phần cứng cá nhân. Nếu dự án hữu ích với bạn:
**Ủng hộ: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Giấy phép

GPL-3.0 hoặc mới hơn.
