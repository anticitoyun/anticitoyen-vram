<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> 支持: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

一个与 OpenAI API 兼容的推理网关，把内存当作层级来处理，并给每块 GPU 它的硅片
读得最好的数值格式。

为一台特定的机器而设计：

| | |
|---|---|
| 处理器 | Intel Core i9-14900K（8 个 P 核 + 16 个 E 核） |
| 主板 | ASUS ROG Maximus Z790 Dark Hero |
| 内存 | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC，32 GB — Blackwell，`sm_120` |
| GPU 1 | ASUS RTX 3080 Ti，12 GB — Ampere，`sm_86` |
| 系统 | Ubuntu 26.04 LTS（CUDA 13）；两块卡都在 PCIe x8/x8，限功 400 W / 275 W |

## 两个想法

**每块 GPU 一种格式。** RTX 5090 有 FP4 张量核心；RTX 3080 Ti 没有，也没有
FP8。把两者对齐到一种公共格式会浪费 5090。因此转换器把*同一个模型写两遍*，
分别用每个目标真正能利用的格式：

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| 权重 | **NVFP4** — E2M1 + 每 16 个一个 FP8 E4M3 尺度 | **INT4** — uint4 + 每 128 个一组 fp16 尺度和零点 |
| 每权重比特 | 4,50 | 4,16 |
| 相对 BF16 | 小 ×3,56 | 小 ×3,85 |
| 计算方式 | FP4 张量核心 | 在核函数内反量化为 FP16，FP16 张量核心 |
| KV 缓存 | INT8 | INT8 |

每权重 4,5 比特时，32 GB 显存约可容纳 **560 亿参数**，BF16 则只有 160 亿。两块卡
加起来，在触碰主内存之前就有大约 **780 亿常驻参数**。

**内存是层级，不是墙。** 三个层级，规划器测量每一层的代价，而不是指望模型
刚好装下：

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## 快速开始

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

之后任何 OpenAI 客户端都能接上：

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

## `acvram plan` 说了什么

规划器值得在任何下载之前运行。它回答那些决定一个模型在这台机器上是否可用的
问题：

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

它探索整个配置空间，而不是保留第一个装得下的配置；它的两个决定足够反直觉，
值得说明：

* **当模型能单独装进 5090 时，它让 3080 Ti 闲置。** 流水线的各段是串行执行
  的：在 1790 GB/s 的流水线里加一个 912 GB/s 的阶段，会拖慢单流解码。可用
  `--gpus all` 强制。
* **它缩小 KV 缓存以把权重留在显存里。** 给缓存的每 1 GB，就是被挤到 PCIe
  总线上的 1 GB 权重，而通过 PCIe 读一个权重的代价约是从显存读的三十倍。在
  上面的 70B 上，仅这一项权衡就从 2,3 提到 17,8 token/s。

## 跑得快

四项优化，每一项都经过等价性证明的验证，而不只是秒表：改变答案的优化就是
缺陷。

### 投机解码 (`--speculative`)

以批大小 1 解码一个 token 受内存限制：机器为了产出一个 token 要读完所有活跃
权重。验证 K 个候选 token 只需把这些权重**读一次**。两种提议器：

* `ngram`（默认）— 在上下文前面查找当前后缀，并提议其后曾出现的内容。零
  成本，不需要模型。当输出复制输入时划算：代码编辑、RAG、摘要。
* `draft` — 第二台设备上的小模型。在这台机器上，该设备是 RTX 3080 Ti，规划
  器对任何能装进 5090 的模型都刻意让它闲置。

接受是精确的，不是近似的：候选以概率 `min(1, p/q)` 被接受，拒绝时从 `p - q`
归一化后的正部分重新采样。对一个故意校准不良的草稿模型做 40 000 次抽样测量，
输出分布与目标的总变差保持在 0,002 以内 — 投机换来速度，从不换来不同的答案。

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### 前缀缓存（默认开启）

块按其 token 片段的*链式*哈希寻址：共享同一系统提示的两个请求共享它的块，
第二个请求无需再预计算。链式是必需的：同样十六个 token 在不同上下文中不含
相同的键和值，只对片段做哈希会把一个序列的缓存端给另一个序列。

被释放但内容仍可识别的块进入 LRU 队列而非空闲列表：缓存因此跨请求存活，且
从不拒绝一个它本可满足的分配。

### 主机层计算 (`--host-exec`)

权重驻留在内存中的层可以拷到 GPU，也可以就地计算。两条路都受内存限制且读
同样的字节：更快的是总线更宽的那条 — PCIe 5.0 x16 约 54 GB/s，双通道 DDR5 约
70 GB/s — 而就地计算还让 GPU 保持空闲，而不是等待一次拷贝。

这只有在处理器直接读取 4 位打包权重时才划算。于是有了一个带 AVX2 路径的小
C++ 核函数（`acvram_cpu.cpp`，通过 ctypes 加载，不需要 Python 头文件也不需要
ninja）。即使在它的**标量**回退分支上，它也以 INT4 下 1,44 倍、NVFP4 下 3,21
倍的优势胜过 `dequantize() @ x`，因为后者要先写出整个矩阵的 32 位副本。

在 Mistral-Large-123B 上，规划器的估算从 1,35 提高到 2,42 token/s。

### 混合精度（`--snr-floor`，默认关闭）

转换器测量每个张量在层输出处的信噪比，可把低于 `--snr-floor` 的张量提升到更
宽的格式，上限为 15 % 的张量与一个价格上限（`--promotion-cout-max`，以增加的
MiB 计）。

阈值**默认为零**：什么都不提升。解码受内存带宽限制，在 `Huihui-Qwen3.8-27B` 上的
测量给出定论 — 25 dB 的阈值要付出 13,4 % 内存和 10,6 % 吞吐（18,50 GiB 与
41,8 t/s，对比 16,02 与 46,2），换来 2,0 % 的困惑度（42,591 对 43,447，16 383
token 语料）。当质量优先于速度时，`--snr-floor 25` 恢复旧行为。

### 以及 `acvram eval`

信噪比与 logits 余弦只是近似。`acvram eval DIR [DIR ...]` 用滑动窗口测量困惑
度，让格式的选择基于证据：

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP 端点

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | SSE 流或单次响应；使用模型的对话模板 |
| `POST /v1/completions` | 文本或 token id 形式的提示 |
| `POST /v1/embeddings` | 最终隐藏状态取平均，L2 归一化，遵守 `dimensions` |
| `GET /v1/models` | 附带一个 `acvram` 块：格式、设备、KV 缓存容量 |
| `GET /health`, `GET /metrics` | 解码吞吐、KV 块占用 |

这些响应的字段名保持英文：这是 OpenAI 协议，翻译它们会弄坏所有现有客户端。

## 数字从哪里来

上面引用的每个值都由本仓库的代码产生并经 `pytest` 检验。使用参考核函数在
处理器上测得：

| 格式 | 比特/权重 | 权重信噪比 | 对 BF16 的 logits 余弦 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

这些测量得出的两个结论改变了默认值：

* **哈达玛旋转帮助 INT4 而不帮助 NVFP4。** INT4 的 128 元组无法吸收孤立的
  离群通道，因此摊开极端值值得每次激活做一次 n log n 变换。NVFP4 的 16 元块
  已自带尺度。于是 `--hadamard auto` 只对 INT4 施加。
* **对 KV 缓存而言 INT8 胜过 FP8 E4M3**，同等大小下 44 dB 对 32 dB，因为按
  （token，头）的尺度已经提供了 FP8 用指数位换来的动态范围。因此两块卡都用
  INT8 的 KV 缓存，尽管 5090 能做 FP8。

## 文档

* [`REPRISE.md`](../REPRISE.md) — **在另一台机器上恢复项目**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — 各部分如何组合
* [`docs/MATERIEL.md`](MATERIEL.md) — 调校这台特定的机器
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **尚未完成的事**，请先读
* [`CONVENTIONS.md`](../CONVENTIONS.md) — 代码工作约定（语言、风格、推送前的检查）

## 实测结果（2026/09/22，RTX 5090 于 400 W，电能表上 ≥ 20 s 的工况）

Qwen3-Coder-30B-A3B 采用 NVFP4（专家）+ INT8（注意力、头），所有引擎同一协议
（`outils/`，单卡，`energie.py`）：

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| 解码 12 个序列 | **1 625,5 t/s** | 1 596,1 t/s | — |
| 解码 1 个序列 | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707** token/s | 21 054 | 8 671（TabbyAPI，已撤回） |

当日吞吐（1030 号机，节能工况 `-lgc 2700`，服务中的流水线；贪心采样已捕获进 CUDA 图，自 0.6.35 起为默认）。b=12 是官方封存单元格（6 个交错窗口的中位数，逐窗口频率）。vLLM 1 596,1 是 21/09 的冻结基准（当天未重跑 vLLM）：+1,84 % 的差距在基准相同的前提下成立，而非同一上午对两者重新测量。相同频率下对三个引擎的 J/token 仍在重新测量（`outils/gpu/mesure/banc-4moteurs.py`）——没有工况的数字不公布。

09/14 早上 acvram 在同一单元格是 630 t/s 和 0,619 J/token：增益来自 Blackwell
原生 FP4 MMA（`mma.sync … kind::mxf4nvf4`，相对 bf16 ×7,9）、按批次桶分组的
GEMM 形式 MoE、单核函数路由（每步 3 677 → 1 517 次启动）以及投影用的窄张量核心
GEMM。每个数字在 `acvram-memoire/revue/` 都有其记录，附测量前封存的预测、
仪器及其工况 — 没有工况的数字不发布。

acvram 领先之处：sm_120 原生 NVFP4 的 MLA 模型（GLM-4.7-Flash，服务中 b=1：165,35 t/s，vLLM 只能以 FP8
提供）；装不进显存的模型；以及自 0.6.35 起，装得进显存的 MoE 的大批次解码——b=12
从 1 540（0.6.34）升至 1 625,5 t/s，即领先冻结的 vLLM 基准（1 596,1）+1,84 %。差距
仍然很窄且在冻结基准下成立；能耗上的差距尚待重新测量。

## 状态

版本 0.6.35。一切都在 5090 上运行：为 `sm_120a`（原生 FP4）和 `sm_86` 编译的
CUDA 核函数、CUDA 图、NVFP4/INT8/INT4 量化、HTTP 服务器。护栏已就位：显卡对
工作会话不可见（`CUDA_VISIBLE_DEVICES` 为空），只有 `outils/carte.sh` 在锁下一次
借给一个测量；守望者记录锁外的每次访问；覆盖多于一块卡或少于 10 s 的能耗测量
无效；以降级工况加载的模型会声明这一点且不参加对决。

640 个测试（`pytest -q`，处理器上一分钟；GPU 测试只在 `carte.sh` 下运行）。
工作跟踪：`acvram-memoire/`（规则、名录、笔记本、180 条评审记录）。

## 支持

acvram 在个人硬件上开发。如果这个项目对你有用：
**支持: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**。

## 许可证

GPL-3.0 或更新版本。
