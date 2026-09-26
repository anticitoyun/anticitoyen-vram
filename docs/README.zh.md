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

一个兼容 OpenAI API 的推理网关：它把内存视为一个层级结构，为每块 GPU 选用其硅片最擅长读取的数值格式，并且像优化秒数一样，以焦耳为单位优化每一个 token。

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · **🇨🇳 中文**

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="与 vLLM 和 llama.cpp 的吞吐量与能耗对比" width="720"></p>

---

## 目录

- [两个核心思想](#idees)
- [快速上手](#demarrage)
- [安装](#installer)
- [`acvram plan` 告诉你什么](#plan)
- [追求速度](#optimisations)
- [HTTP 端点](#http)
- [数字从何而来](#chiffres)
- [文档](#documentation)
- [实测结果](#resultats)
- [现状](#etat)
- [致谢](#credits)
- [许可证](#licence)
- [支持本项目](#soutien)

---

<a id="idees"></a>

## 两个核心思想

专为一台特定的机器打造：

| | |
|---|---|
| 处理器 | Intel Core i9-14900K（8 个 P 核 + 16 个 E 核） |
| 主板 | ASUS ROG Maximus Z790 Dark Hero |
| 内存 | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC，32 GB — Blackwell，`sm_120` |
| GPU 1 | ASUS RTX 3080 Ti，12 GB — Ampere，`sm_86` |
| 系统 | Ubuntu 26.04 LTS（CUDA 13）；两块显卡均为 PCIe x8/x8，功耗上限分别为 400 W / 275 W |

**每块 GPU 一种格式。** RTX 5090 拥有 FP4 张量核心；RTX 3080 Ti 既没有 FP4，也没有 FP8。让两者统一使用同一种格式会浪费 5090 的能力。因此转换器会把*同一个模型写两遍*，分别采用每个目标设备真正能够发挥的格式：

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| 权重 | **NVFP4** — E2M1 + 每 16 个一组的 FP8 E4M3 缩放因子 | **INT4** — uint4 + 每 128 个一组的 fp16 缩放因子与零点 |
| 每权重比特数 | 4.50 | 4.16 |
| 相对 BF16 | 缩小 ×3.56 | 缩小 ×3.85 |
| 计算方式 | FP4 张量核心 | 在内核中反量化为 FP16，使用 FP16 张量核心 |
| KV 缓存 | INT8 | INT8 |

32 GB 显存在每权重 4.5 比特的情况下可容纳约 **560 亿参数**，而 BF16 只能容纳 160 亿。两块显卡合计，在完全不动用主机内存之前，就能常驻约 **780 亿参数**。

**内存是一个层级结构，而不是一堵墙。** 共三个层级，规划器会测量每一层的实际代价，而不是寄希望于模型恰好装得下：

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
主机 DDR5    96 GB   受 PCIe 或 DDR 限制
```

---

<a id="demarrage"></a>

## 快速上手

```bash
./install.sh                       # 虚拟环境 + torch cu128 + acvram
acvram doctor                      # 这台机器是否就绪，能胜任什么
acvram detect                      # 这里实际有哪些硬件

acvram plan  ~/models/Qwen3-32B                     # 每一层会被放到哪里
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

任何 OpenAI 客户端都可以直接接入：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "你好"}])
```

---

<a id="installer"></a>

## 安装

从源码安装（适用于所有平台）：

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

也可以通过软件包安装，每个 [GitHub 发布版本](https://github.com/anticitoyun/anticitoyen-vram/releases/latest)都附有对应文件：

| 渠道 | 发布版本附带的文件 | 命令 |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz`（PKGBUILD + .SRCINFO） | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm`（文件名由 `rpmbuild` 生成，并不固定） | `sudo rpm -i acvram-<version>-1.*.noarch.rpm`（也可以用 `rpmbuild --rebuild *.src.rpm` 从 `.src.rpm` 重新构建） |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

安装前，请对照发行版所附的校验和验证下载的文件(`SHA256SUMS`，在其他所有文件都就绪后发布)：

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip 上没有发布本软件包（未构建 wheel）：`pip install -e '.[dev]'` 从源码克隆安装，与 `./install.sh` 相同。

---

<a id="plan"></a>

## `acvram plan` 告诉你什么

规划器值得在任何下载之前先运行一次。它回答的正是决定一个模型能否在这台机器上使用的那些问题：

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  层级    格式        容量          权重          KV  切片
  cuda:0  nvfp4        30.3 GiB    25.5 GiB   4.5 GiB  层 0-58
  cuda:1  int4_awq     10.9 GiB     8.7 GiB   1.6 GiB  层 59-79
  cpu     nvfp4        74.8 GiB     3.4 GiB      0 B   -

  权重总量           37.6 GiB
  每 token 读取      35.1 GiB
  每 token KV        162.5 KiB  -> 可缓存 39,843 个 token
  位于主机内存的 MLP 55-58

  解码估计           17.8 token/s  （批大小为 1）
  预填充估计         847 token/s
```

它会探索整个配置空间，而不是停在第一个装得下的方案上。其中有两项决策足够反直觉，值得明确说明：

* **当模型单靠 5090 就能装下时，它会让 3080 Ti 闲置。** 流水线的各个阶段是串行执行的：在 1790 GB/s 的流水线中加入一个 912 GB/s 的阶段，会拖慢单流解码。可以用 `--gpus all` 强制启用。
* **它会缩小 KV 缓存，以便把权重留在显存中。** 分给缓存的每一 GB，都是被挤到 PCIe 总线上的一 GB 权重，而通过 PCIe 读取一个权重的代价大约是从显存读取的三十倍。以上面的 70B 为例，仅这一项取舍就让解码速度从 2.3 提升到 17.8 token/s。

---

<a id="optimisations"></a>

## 追求速度

四项优化，每一项都经过等价性证明验证，而不只是靠秒表计时：改变了答案的优化就是 bug。

稠密模型的 NVFP4 线性层默认采用 Marlin 布局（b = 8 时吞吐量提升 57% 到 90%，TTFT 增加 2 到 4 ms，依据 revue/poste6-piece147-verdict-24-09.md；回退方式 `ACVRAM_PROJ_MARLIN=0`，参见 [CHANGELOG.md](../CHANGELOG.md)）。

### 投机解码（`--speculative`）

以批大小 1 解码一个 token 受限于内存：机器要读取全部活跃权重才能生成一个 token。而验证 K 个候选 token 时，同样这些权重**只需读取一次**。提供两种提议器：

* `ngram`（默认）— 在上下文中更早的位置查找当前后缀，并提议其后曾出现的内容。没有任何开销，也不需要模型。当输出大量复述输入时收益明显：代码编辑、RAG、摘要。
* `draft` — 在第二块设备上运行的小模型。在这台机器上，这块设备就是 RTX 3080 Ti，而对于任何能装进 5090 的模型，规划器都会有意让它闲置。

`mtp`（模型的 `nextn` 头）和 `auto` 也存在；就目前而言并不划算，默认不启用 — 参见 `docs/ARCHITECTURE.md`。

接受判定是精确的，而非近似：一个提议以概率 `min(1, p/q)` 被接受，被拒绝时则从 `p - q` 归一化后的正部中重新采样。针对一个刻意失准的草稿模型进行 40,000 次抽样测量，输出分布与目标分布的总变差距离始终在 0.002 以内 — 投机换来的是速度，而绝不是一个不同的答案。

```
玩具模型，贪心，k=4           步数   token/步      输出
  不投机                         23          1.00   基准
  n-gram                         13          1.77   一致
  draft（= 目标）                 5          4.60   一致
```

### 前缀缓存（默认开启）

每个块以其 token 片段的*链式*哈希寻址：两个共享同一系统提示词的请求会共享其对应的块，第二个请求不必再重新预计算。链式哈希必不可少：同样的十六个 token 处在不同的上下文中，其键和值并不相同；若只对片段本身做哈希，就会把一个序列的缓存错误地提供给另一个序列。

一个被释放但内容仍可识别的块，会进入 LRU 队列而不是空闲块列表：缓存由此得以在请求之间保留下来，同时绝不会拒绝一次它本可以满足的分配。

### 主机层计算（`--host-exec`）

权重位于内存中的层，既可以复制到 GPU 上，也可以就地计算。两条路径都受限于内存，读取的字节也完全相同：哪条路径的总线更宽，哪条就更快 — PCIe 5.0 x16 约为 54 GB/s，双通道 DDR5 约为 70 GB/s。此外，就地计算还能让 GPU 腾出手来，而不是让它等待一次复制。

只有当 CPU 直接读取打包的 4 比特权重时，这样做才划算。因此有了一个带 AVX2 路径的小型 C++ 内核（`acvram_cpu.cpp`，通过 ctypes 加载，无需 Python 头文件或 ninja）。即使在其**标量**回退分支上，它也比 `dequantize() @ x` 快：INT4 下快 1.44 倍，NVFP4 下快 3.21 倍，因为后者要先写出整个矩阵的 32 位副本。

在 Mistral-Large-123B 上，规划器的估计值从 1.35 提升到 2.42 token/s。

### 混合精度（`--snr-floor`，默认关闭）

转换器会为每个张量测量层输出的信噪比，并可将低于 `--snr-floor` 的张量提升为更宽的格式，上限为张量总数的 15%，同时受价格上限约束（`--promotion-cout-max`，以新增的 MiB 计）。

该下限**默认为零**：不会提升任何张量。解码受限于内存带宽，而在 `Huihui-Qwen3.8-27B` 上的测量给出了定论 — 25 dB 的下限要付出 13.4% 的内存和 10.6% 的吞吐量（18.50 GiB、41.8 t/s，对比 16.02 与 46.2），只换来 2.0% 的困惑度改善（42.591 对比 43.447，16,383 个 token 的语料）。当质量比速度更重要时，`--snr-floor 25` 可恢复旧的行为。

### 还有 `acvram eval`

信噪比与 logit 余弦相似度都只是近似指标。`acvram eval REP [REP ...]` 测量滑动窗口困惑度，让格式的选择建立在证据之上：

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  模型                     ppl     bpp        大小   token 数
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## HTTP 端点

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | SSE 流式或单次响应；使用模型自带的对话模板 |
| `POST /v1/completions` | 提示词可以是文本，也可以是 token id |
| `POST /v1/embeddings` | 对最终隐藏状态做均值池化并 L2 归一化，支持 `dimensions` |
| `GET /v1/models` | 另附一个 `acvram` 块：格式、设备、KV 缓存容量 |
| `GET /health`, `GET /metrics` | 解码吞吐量、KV 块占用率 |

这些响应中的字段名保持英文：这是 OpenAI 协议，翻译它们会破坏所有现有客户端。

---

<a id="chiffres"></a>

## 数字从何而来

上文引用的每一个数值都由本仓库中的代码产生，并经过 `pytest` 验证。以下测量在 CPU 上使用参考内核完成：

| 格式 | 比特/权重 | 权重 SNR | 相对 BF16 的 logit 余弦 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

这些测量得出的两个结论改变了默认设置：

* **Hadamard 旋转对 INT4 有帮助，对 NVFP4 则没有。** INT4 以 128 个为一组，无法吸收孤立的离群通道，因此把极端值分散开来，值得为每个激活付出一次 n log n 的变换。NVFP4 以 16 个为一块，每块本身已带有独立的缩放因子。因此采用 `--hadamard auto`，只对 INT4 应用该旋转。
* **在 KV 缓存上，INT8 胜过 FP8 E4M3**：同等大小下为 44 dB 对 32 dB，因为按（token，头）设置的缩放因子已经提供了动态范围，而 FP8 却要为此花费指数位。因此两块显卡都使用 INT8 KV 缓存，尽管 5090 支持 FP8。另有一种 `k8v4` 格式（值用 INT4，缓存字节减少 22%）作为选项存在，但**尚未经过验证** — 参见 `docs/ARCHITECTURE.md`。

---

<a id="documentation"></a>

## 文档

| 文档 | 内容 |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **在另一台机器上接手本项目**（法文） |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | 各个部分如何组合在一起 |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | 纯 NVFP4 或按通道 int8 的 attention+GDN，基于 Gated DeltaNet 混合架构 |
| [`docs/MATERIEL.md`](MATERIEL.md) | 针对这台特定机器的调校 |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **尚未完成的工作**，请先阅读 |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | 代码的工作约定（语言、风格、推送前的检查） |

---

<a id="resultats"></a>

## 实测结果（2026-09-22，RTX 5090 功耗 400 W，能耗计测量窗口 ≥ 20 s）

Qwen3-Coder-30B-A3B，NVFP4（专家）+ INT8（注意力、输出头），所有引擎采用同一测量流程（`outils/`，单卡，`energie.py`）：

| | acvram | vLLM 0.29（`vllm serve`） | llama.cpp（sm_120） |
|---|---|---|---|
| 解码，12 个序列 | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| 解码，1 个序列 | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| 预填充 pp2048 | **22,707 token/s** | 21,054 | 8,671（TabbyAPI，已停用） |

¹ 2026-09-22 勘误：`serve` 默认开启投机（`--speculative ngram`，cli.py），而竞争对手没有；此前公布的 380.8 t/s 是在开启投机的情况下测得的。关闭投机后（`--speculative none`，同一链路，revue/poste2-piece44-speculation-none-22-09.md）：283.6 t/s — acvram 在 b=1 时位列**第三**，落后于 llama.cpp 和 vLLM。在能耗上它仍领先于 llama.cpp（净值 0.601 对 0.700 J/token）。在 b=12 时投机从不启用（保护条件 `lot_max=2`）：该单元格本来就是在公平条件下比较的。

² 2026-09-23，同一场次、同一 HTTP 客户端（`banc-llamacpp-16-09.py` 分别对 `acvram serve` 和 `vllm serve`），在每一组测量前后显式设定 `-lgc 2700`，单元格按 A V V A 交替，每组 ≥ 5 批，只有超过 2σ 的差距才会报告（revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md）。acvram 0.6.38（解码时使用 w13，展开的注意力归约）：差距 −1.6%，**低于 2σ：吞吐量持平**。在 J/token 上，**vLLM 仍领先 7.0%**（超过 2σ）。使用 0.6.37 时，同一流程得到的是 −4.7%。

³ 与 ² 同一场次、同一流程，双方均不投机：acvram 312.3 对 vLLM 284.8 — **acvram 吞吐量领先 9.7%**（超过 2σ）；J/token：**持平**（差距 0.04%，低于 2σ）。

⁴ 2026-09-23，以同一流程对比 llama.cpp（revue/poste2-piece72-llamacpp-b1-23-09.md），acvram 0.6.37 使用重写后的路由（+5.6%）：acvram 310.8 对 llama.cpp 329.9 t/s — **llama.cpp 吞吐量领先 5.8%，acvram 在 J/token 上领先 13.4%**（0.598 对 0.691）。

当日吞吐量数据（1030 号工位，节能模式 `-lgc 2700`，流水线处于服务状态；贪心采样被捕获进 CUDA 图，自 0.6.35 起为默认）。acvram 的 b=12 数据是一个正式封存的单元格（6 个交错窗口的中位数，每个窗口单独读取时钟）。

> **勘误（2026-09-23）。** 此前公布的 vLLM 对比（b=12：1,782 对 1,634 t/s；b=1：290.6）是把通过 HTTP 测得的 acvram 与**离线**测得的 vLLM（`LLM().generate()`）相比较，而 2026-09-22 的勘误错误地声称 vLLM 的单元格走的是 `vllm serve`。2026-09-23：双方使用同一 HTTP 客户端，并且都设定了 `-lgc`（acvram 在启动时会自行设定，`vllm serve` 不会：若不采取这一预防措施，vLLM 运行在约 2,930 MHz，而非约 2,650 MHz）。结果见注 ²：在 b=12 时 vLLM 领先 9.1%。

2026-09-14 上午，acvram 在同一单元格上的成绩为 630 t/s 和 0.619 J/token：提升来自 Blackwell 原生的 FP4 MMA（`mma.sync … kind::mxf4nvf4`，相对 bf16 为 ×7.9）、按批次桶分组的 GEMM MoE、单内核路由（每步启动次数从 3,677 降到 1,517），以及用于投影层的窄型张量核心 GEMM。每一个数字在 `acvram-memoire/revue/` 中都有对应的记录，包括测量前封存的预测、所用仪器及其测量条件 — 没有注明测量条件的数字不予发布。

acvram 领先的场景：以 sm_120 原生 NVFP4 运行的 MLA 模型（GLM-4.7-Flash），而 vLLM 只能以 FP8 提供服务（b=1：服务状态下 165.35 t/s）；以及装不进显存的模型。单序列解码不在其中：在不投机的情况下，acvram 在这方面领先 vLLM 9.7%（注 ³），吞吐量落后 llama.cpp 5.8%，但能耗上领先它 13.4%（注 ⁴）。在大批次下，对于能装进显存的 MoE 模型，vLLM 在 b=12 时吞吐量持平（1,995.1 对 2,027.0 t/s，低于 2σ，注 ²），但在 J/token 上保持 7.0% 的优势；acvram 在这一场景下已从 1,540 t/s（0.6.34）提升到 1,995（0.6.38）。

---

<a id="etat"></a>

## 现状

版本 0.6.38。一切都在 5090 上运行：为 `sm_120a`（原生 FP4）和 `sm_86` 编译的 CUDA 内核、CUDA 图、NVFP4/INT8/INT4 量化、HTTP 服务器。已就位的防护措施：显卡对工作会话不可见（`CUDA_VISIBLE_DEVICES` 为空），只有 `outils/carte.sh` 会在加锁的情况下把它借给一次测量，且同一时间只借给一次；一个监视程序会记录锁之外的任何访问；跨越多块显卡或时长不足 10 s 的能耗测量会被判为无效；以降级状态加载的模型会明确说明这一点，并且不参与对比测试。

4,107 个测试（`pytest --collect-only -q`，在 CPU 上约需一分钟；GPU 测试只在 `carte.sh` 下运行）。工作日志：`acvram-memoire/`（规则、名录、笔记本，以及数百篇评审记录）。

---

<a id="credits"></a>

## 致谢

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0，Apache-2.0 许可证：`acvram/kernels/marlin_port/` 移植了其 Marlin 内核（MoE 与稠密），逐文件的完整署名见 [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE)。
- **NVIDIA** — CUDA、Blackwell 的 FP4 张量核心（`sm_120`），以及本项目所依赖的各种库。
- **PyTorch** — 张量引擎与 C++/CUDA 扩展。

独立项目，与 ASUS、NVIDIA 及 vLLM 项目均无关联。

---

<a id="licence"></a>

## 许可证

本仓库中的代码采用 [GPL-3.0-or-later](../LICENSE)。`acvram/kernels/marlin_port/` 包含从 [vLLM](https://github.com/vllm-project/vllm) v0.29.0 移植的代码（`marlin_moe_wna16`、`gptq_marlin_repack`、`moe_align_block_size` 内核），采用 Apache-2.0 许可证：每个文件都保留了原始文件头，许可证文本位于 `LICENSE-vllm`，文件清单、来源提交和修改内容见 [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE)。

---

<a id="soutien"></a>

## 支持本项目

acvram 在个人硬件上开发。如果本项目对你有用：

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

翻译：[TRADUIRE.md](TRADUIRE.md)（法文；本项目的贡献指南尚未翻译）。
