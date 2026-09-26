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

OpenAI API 互換の推論ゲートウェイ。メモリを階層として扱い、各 GPU にそのシリコンが最も得意とする数値フォーマットを与え、すべてのトークンを秒と同じだけジュールでも最適化する。

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · **🇯🇵 日本語** · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="vLLM および llama.cpp とのスループット・エネルギー比較" width="720"></p>

---

## 目次

- [二つの考え方](#idees)
- [クイックスタート](#demarrage)
- [インストール](#installer)
- [`acvram plan` が示すこと](#plan)
- [高速化](#optimisations)
- [HTTP エンドポイント](#http)
- [数値の出どころ](#chiffres)
- [ドキュメント](#documentation)
- [実測結果](#resultats)
- [現状](#etat)
- [クレジット](#credits)
- [ライセンス](#licence)
- [プロジェクトを支援する](#soutien)

---

<a id="idees"></a>

## 二つの考え方

特定の一台のマシンのために設計されている：

| | |
|---|---|
| CPU | Intel Core i9-14900K（P コア 8 基 + E コア 16 基） |
| マザーボード | ASUS ROG Maximus Z790 Dark Hero |
| メモリ | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC、32 GB — Blackwell、`sm_120` |
| GPU 1 | ASUS RTX 3080 Ti、12 GB — Ampere、`sm_86` |
| システム | Ubuntu 26.04 LTS（CUDA 13）。二枚とも PCIe x8/x8、電力制限 400 W / 275 W |

**GPU ごとに一つのフォーマット。** RTX 5090 は FP4 Tensor コアを備えるが、RTX 3080 Ti にはそれがなく、FP8 もない。両者を共通のフォーマットに揃えれば 5090 を無駄にしてしまう。そこでコンバーターは*同じモデルを二回*、それぞれの配置先が実際に活用できるフォーマットで書き出す：

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| 重み | **NVFP4** — E2M1 + 16 個ごとの FP8 E4M3 スケール | **INT4** — uint4 + 128 個ごとの fp16 スケールとゼロ点 |
| 重みあたりのビット数 | 4.50 | 4.16 |
| BF16 比 | ×3.56 縮小 | ×3.85 縮小 |
| 演算方式 | FP4 Tensor コア | カーネル内で FP16 に逆量子化し、FP16 Tensor コアで演算 |
| KV キャッシュ | INT8 | INT8 |

32 GB の VRAM に重みあたり 4.5 ビットで載せると、約 **560 億パラメータ**が収まる。BF16 では 160 億にとどまる。二枚合わせれば、ホストメモリに一切触れる前の段階で約 **780 億の常駐パラメータ**になる。

**メモリは壁ではなく階層である。** 階層は三段あり、プランナーはモデルが収まることを期待するのではなく、各段が実際にどれだけのコストになるかを計測する：

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
ホスト DDR5  96 GB   PCIe または DDR が上限
```

---

<a id="demarrage"></a>

## クイックスタート

```bash
./install.sh                       # 仮想環境 + torch cu128 + acvram
acvram doctor                      # このマシンは準備できているか、何に向いているか
acvram detect                      # 実際に何が搭載されているか

acvram plan  ~/models/Qwen3-32B                     # 各レイヤーがどこに置かれるか
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

どの OpenAI クライアントもそのまま接続できる：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"こんにちは"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "こんにちは"}])
```

---

<a id="installer"></a>

## インストール

ソースから（全プラットフォーム共通）：

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

またはパッケージから。各 [GitHub リリース](https://github.com/anticitoyun/anticitoyen-vram/releases/latest) にファイルが一つずつ添付されています：

| チャネル | リリースの添付ファイル | コマンド |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm`（ファイル名は `rpmbuild` が生成するため固定されません） | `sudo rpm -i acvram-<version>-1.*.noarch.rpm`（または `rpmbuild --rebuild *.src.rpm` で `.src.rpm` からビルド） |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

インストール前に、リリースに添付されたチェックサム(`SHA256SUMS`、他のすべてのファイルが揃った後に公開)に対してダウンロードしたファイルを検証してください:

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip はパッケージとして公開されていません（ビルド済みの wheel はありません）。`pip install -e '.[dev]'` は `./install.sh` と同じく、ソースのクローンからインストールします。

---

<a id="plan"></a>

## `acvram plan` が示すこと

プランナーは、何かをダウンロードする前にぜひ実行しておきたい。モデルがこのマシンでそもそも使えるかどうかを決める問いに答えてくれる：

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  階層    形式             容量        重み       KV  区間
  cuda:0  nvfp4        30.3 GiB    25.5 GiB   4.5 GiB  レイヤー 0-58
  cuda:1  int4_awq     10.9 GiB     8.7 GiB   1.6 GiB  レイヤー 59-79
  cpu     nvfp4        74.8 GiB     3.4 GiB      0 B   -

  重みの合計           37.6 GiB
  読み出し/トークン    35.1 GiB
  KV/トークン          162.5 KiB  -> 39,843 トークンをキャッシュ
  ホスト RAM の MLP    55-58

  デコード推定         17.8 トークン/s  (バッチ 1)
  プリフィル推定       847 トークン/s
```

プランナーは最初に収まった構成で止まらず、構成空間を探索する。その判断のうち二つは直感に反するので、はっきり述べておく：

* **モデルが 5090 単体に収まるときは、3080 Ti を使わずに置いておく。** パイプラインの各段は直列に実行されるため、1790 GB/s のパイプラインに 912 GB/s の段を加えると、単一ストリームのデコードが遅くなる。`--gpus all` で強制できる。
* **重みを VRAM に残すために KV キャッシュを縮める。** キャッシュに割り当てた容量は、そのまま PCIe バスへ追い出される重みの容量にほかならない。そして PCIe 越しに重みを読むコストは、VRAM から読む場合のおよそ三十倍である。上の 70B では、このトレードオフ一つだけでデコードが 2.3 から 17.8 トークン/s に向上する。

---

<a id="optimisations"></a>

## 高速化

四つの最適化があり、いずれもストップウォッチだけでなく等価性の証明によって検証されている。答えを変えてしまう最適化はバグである。

密モデルの NVFP4 線形層は、デフォルトで Marlin レイアウトを通る（b = 8 でスループット +57〜+90%、TTFT は +2〜+4 ms。根拠は revue/poste6-piece147-verdict-24-09.md。フォールバックは `ACVRAM_PROJ_MARLIN=0`。[CHANGELOG.md](../CHANGELOG.md) を参照）。

### 投機的デコード（`--speculative`）

バッチサイズ 1 でトークンをひとつデコードする処理はメモリ律速である。たったひとつのトークンを生成するために、マシンはアクティブな重みをすべて読む。提案された K 個のトークンを検証する場合も、同じ重みを**一度だけ**読めば済む。提案器は二つある：

* `ngram`（デフォルト）— 現在の接尾辞を文脈のより前の部分から探し、その後に続いていたものを提案する。コストはかからず、モデルも不要。出力が入力をなぞる場面、つまりコード編集、RAG、要約で効果を発揮する。
* `draft` — 二台目のデバイス上で動く小さなモデル。このマシンではそのデバイスは RTX 3080 Ti であり、5090 に収まるモデルであればプランナーがあえて遊ばせている GPU である。

`mtp`（モデルの `nextn` ヘッド）と `auto` も存在するが、現状では採算が合わず、デフォルトでは有効になっていない — `docs/ARCHITECTURE.md` を参照。

受理は近似ではなく厳密である。提案は確率 `min(1, p/q)` で受理され、棄却された場合は `p - q` の正の部分を正規化した分布から再サンプリングする。意図的に較正を狂わせたドラフトを相手に 40,000 回の抽出で測定したところ、出力される分布はターゲットから全変動距離 0.002 以内に収まった — 投機で得られるのは速度であって、別の答えではない。

```
トイモデル、貪欲法、k=4      ステップ  トークン/ステップ  出力
  投機なし                       23          1.00   基準
  n-gram                         13          1.77   同一
  ドラフト (= ターゲット)         5          4.60   同一
```

### プレフィックスキャッシュ（デフォルトで有効）

ブロックは、そのトークン区間の*連鎖*ハッシュによってアドレス付けされる。システムプロンプトを共有する二つのリクエストはそのブロックも共有し、二番目のリクエストはそれらを改めて事前計算する必要がない。連鎖は欠かせない。同じ十六トークンでも文脈が異なればキーと値は同じにならず、区間だけをハッシュすると、あるシーケンスのキャッシュを別のシーケンスに渡してしまう。

解放されたブロックのうち、内容がまだ識別できるものは、空きブロックのリストではなく LRU キューに入る。こうしてキャッシュはリクエストをまたいで生き残り、しかも応じられたはずの割り当てを拒むことは決してない。

### ホスト階層での演算（`--host-exec`）

重みが RAM にあるレイヤーは、GPU へコピーすることも、その場で計算することもできる。どちらの経路もメモリ律速で、読み出すバイトも同じである。速いのはバスが広いほうだ — PCIe 5.0 x16 はおよそ 54 GB/s、デュアルチャネルの DDR5 はおよそ 70 GB/s — そのうえ、その場で計算すれば、GPU をコピー待ちにさせずに空けておける。

これが効果を持つのは、CPU がパックされた 4 ビットの重みを直接読む場合に限られる。そのため、AVX2 パスを持つ小さな C++ カーネルを用意した（`acvram_cpu.cpp`、ctypes で読み込み、Python ヘッダーも ninja も不要）。フォールバックの**スカラー**分岐でさえ、`dequantize() @ x` を INT4 で 1.44 倍、NVFP4 で 3.21 倍上回る。後者はまず行列全体の 32 ビットのコピーを書き出すからである。

Mistral-Large-123B では、プランナーの推定値が 1.35 から 2.42 トークン/s に上がる。

### 混合精度（`--snr-floor`、デフォルトで無効）

コンバーターはテンソルごとにレイヤー出力の信号対雑音比を測定し、`--snr-floor` を下回るものをより広いフォーマットへ昇格させることができる。ただし上限は、テンソル全体の 15% と、コストの上限（`--promotion-cout-max`、追加されるメビバイト数で指定）である。

下限は**デフォルトでゼロ**であり、何も昇格されない。デコードはメモリ帯域律速であり、`Huihui-Qwen3.8-27B` での測定が決着をつけている — 25 dB の下限はメモリを 13.4%、スループットを 10.6% 犠牲にし（18.50 GiB・41.8 t/s 対 16.02・46.2）、その見返りはパープレキシティの 2.0% にすぎない（42.591 対 43.447、16,383 トークンのコーパス）。速度より品質を優先したいときは、`--snr-floor 25` で以前の挙動に戻せる。

### そして `acvram eval`

信号対雑音比とロジットのコサイン類似度は近似にすぎない。`acvram eval REP [REP ...]` はスライディングウィンドウでパープレキシティを測定するので、フォーマットの選択を実証に基づいて決められる：

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  モデル                   ppl     bpp      サイズ   トークン
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## HTTP エンドポイント

| エンドポイント | 備考 |
|---|---|
| `POST /v1/chat/completions` | SSE ストリームまたは単一レスポンス。モデルのチャットテンプレートを使用 |
| `POST /v1/completions` | プロンプトはテキストでもトークン ID でも可 |
| `POST /v1/embeddings` | 最終隠れ状態の平均プーリング、L2 正規化、`dimensions` に対応 |
| `GET /v1/models` | 加えて `acvram` ブロック：フォーマット、デバイス、KV キャッシュ容量 |
| `GET /health`, `GET /metrics` | デコードのスループット、KV ブロックの占有率 |

これらのレスポンスのフィールド名は英語のままである。これは OpenAI のプロトコルであり、翻訳すれば既存のクライアントがすべて動かなくなる。

---

<a id="chiffres"></a>

## 数値の出どころ

上に挙げた値はすべて、このリポジトリのコードが出したものであり、`pytest` で検証されている。測定はリファレンスカーネルを用いて CPU 上で行った：

| フォーマット | ビット/重み | 重みの SNR | BF16 に対するロジットのコサイン |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

これらの測定から得られた二つの知見が、デフォルト値を変えた：

* **Hadamard 回転は INT4 には効くが、NVFP4 には効かない。** INT4 の 128 個単位のグループは孤立した外れ値チャネルを吸収できないため、極端な値を分散させることには、活性化ごとに n log n の変換を払うだけの価値がある。NVFP4 の 16 個単位のブロックは、すでにそれぞれ固有のスケールを持っている。そこで `--hadamard auto` は INT4 にのみこれを適用する。
* **KV キャッシュでは INT8 が FP8 E4M3 に勝る。** 同じサイズで 44 dB 対 32 dB である。(トークン, ヘッド) ごとのスケールが、FP8 が指数ビットを費やして得ているダイナミックレンジをすでに与えているからだ。そのため、5090 は FP8 を扱えるにもかかわらず、両カードとも INT8 の KV キャッシュを使う。`k8v4` フォーマット（値を INT4 化、キャッシュのバイト数 −22%）はオプションとして存在するが、**未認定**である — `docs/ARCHITECTURE.md` を参照。

---

<a id="documentation"></a>

## ドキュメント

| ドキュメント | 内容 |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **別のマシンでプロジェクトを引き継ぐ**（フランス語） |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | 各部品がどう組み合わさっているか |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | 純粋なNVFP4か、チャネルごとint8のattention+GDNか、Gated DeltaNetハイブリッド上で |
| [`docs/MATERIEL.md`](MATERIEL.md) | この特定のマシンのチューニング |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **まだ済んでいないこと**。最初にこれを読むこと |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | コードの作業規約（言語、スタイル、プッシュ前のチェック） |

---

<a id="resultats"></a>

## 実測結果（2026-09-22、RTX 5090 を 400 W で運用、エネルギー計で ≥ 20 s の計測窓）

Qwen3-Coder-30B-A3B を NVFP4（エキスパート）+ INT8（アテンション、ヘッド）で使用し、すべてのエンジンで同じプロトコルを適用（`outils/`、カード一枚、`energie.py`）：

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| デコード、12 シーケンス | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| デコード、1 シーケンス | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| プリフィル pp2048 | **22,707 トークン/s** | 21,054 | 8,671（TabbyAPI、使用終了） |

¹ 2026-09-22 の訂正：`serve` はデフォルトで投機を行う（`--speculative ngram`、cli.py）が、競合は行わない。これまで公表していた 380.8 t/s は投機ありで測定したものだった。投機なし（`--speculative none`、同じチェーン、revue/poste2-piece44-speculation-none-22-09.md）では 283.6 t/s — acvram は b=1 で llama.cpp と vLLM に次ぐ **三位**である。エネルギーでは llama.cpp より優位を保っている（正味 0.601 対 0.700 J/トークン）。b=12 では投機は一切働かない（ガード `lot_max=2`）ため、このセルはもともと同条件での比較だった。

² 2026-09-23、同一セッション、同一の HTTP クライアント（`banc-llamacpp-16-09.py` を `acvram serve` と `vllm serve` に対して使用）、各アームの前後で `-lgc 2700` を明示的に設定、セルは A V V A の順に交互に実施、アームごとに ≥ 5 バッチ、差は 2σ を超えた場合のみ報告（revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md）。acvram 0.6.38（デコード時の w13、アテンション縮約のループ展開）：差は −1.6%、**2σ 未満：スループットは同等**。J/トークンでは **vLLM が 7.0% 優位のまま**（2σ 超）。0.6.37 では同じプロトコルで −4.7% だった。

³ ² と同じセッション・同じプロトコルで、双方とも投機なし：acvram 312.3 対 vLLM 284.8 — **スループットで acvram が 9.7% 優位**（2σ 超）。J/トークン：**同等**（差 0.04%、2σ 未満）。

⁴ 2026-09-23、llama.cpp に対して同じプロトコル（revue/poste2-piece72-llamacpp-b1-23-09.md）、ルーターを書き直した acvram 0.6.37（+5.6%）：acvram 310.8 対 llama.cpp 329.9 t/s — **スループットでは llama.cpp が 5.8% 優位、J/トークンでは acvram が 13.4% 優位**（0.598 対 0.691）。

本日のスループット値（ステーション 1030、エコ設定 `-lgc 2700`、パイプライン稼働中。貪欲サンプリングを CUDA グラフに取り込んだ状態で、0.6.35 以降のデフォルト）。acvram の b=12 の値は、公式の封印済みセル（交互に配置した 6 つの計測窓の中央値、クロックは窓ごとに読み取り）である。

> **訂正（2026-09-23）。** これまで公表していた vLLM との比較（b=12：1,782 対 1,634 t/s、b=1：290.6）は、HTTP 経由で測定した acvram と、**オフライン**（`LLM().generate()`）で測定した vLLM を比べたものであり、さらに 2026-09-22 の訂正では、vLLM のセルが `vllm serve` を経由していたと誤って記していた。2026-09-23：両者に同じ HTTP クライアントを使い、`-lgc` も両者に設定した（acvram は起動時に自ら設定するが、`vllm serve` は設定しない。この配慮がなければ、~2,650 に対して vLLM は ~2,930 MHz で動作していた）。結果は注 ² のとおり：b=12 で vLLM が 9.1% 優位。

2026-09-14 の朝、同じセルで acvram は 630 t/s、0.619 J/トークンだった。向上をもたらしたのは、Blackwell ネイティブの FP4 MMA（`mma.sync … kind::mxf4nvf4`、bf16 比 ×7.9）、バッチバケットごとのグループ化 GEMM による MoE、単一カーネルでのルーティング（ステップごとの起動回数 3,677 → 1,517）、そして射影向けの幅の狭い Tensor コア GEMM である。すべての数値には `acvram-memoire/revue/` にノートがあり、測定前に封印した予測、計測器、そしてその動作条件が記されている — 動作条件の記されていない数値は公表しない。

acvram が優位な領域：sm_120 ネイティブ NVFP4 での MLA モデル（GLM-4.7-Flash）。vLLM はこれを FP8 でしか提供しない（b=1：稼働時 165.35 t/s）。そして VRAM に収まらないモデル。単一シーケンスのデコードはここに含まれない。投機なしの場合、acvram はそこで vLLM を 9.7% 上回り（注 ³）、llama.cpp にはスループットで 5.8% 劣るものの、エネルギーでは 13.4% 上回る（注 ⁴）。大きなバッチでは、VRAM に収まる MoE モデルにおいて、vLLM は b=12 でスループット同等（1,995.1 対 2,027.0 t/s、2σ 未満、注 ²）だが、J/トークンでは 7.0% の優位を保っている。acvram はそこで 1,540 t/s（0.6.34）から 1,995（0.6.38）まで伸びた。

---

<a id="etat"></a>

## 現状

バージョン 0.6.38。すべてが 5090 で動作する：`sm_120a`（ネイティブ FP4）と `sm_86` 向けにコンパイルした CUDA カーネル、CUDA グラフ、NVFP4/INT8/INT4 量子化、HTTP サーバー。安全策も整っている：カードは作業セッションからは見えず（`CUDA_VISIBLE_DEVICES` は空）、`outils/carte.sh` だけがロックのもとで、一度に一つの測定にだけカードを貸し出す。監視プロセスがロック外のアクセスをすべて記録する。複数のカードにまたがる、または 10 s 未満のエネルギー測定は無効とする。劣化した状態で読み込まれたモデルはその旨を表明し、対決には加わらない。

4,107 件のテスト（`pytest --collect-only -q`、CPU で一分。GPU テストは `carte.sh` の管理下でのみ実行）。作業記録：`acvram-memoire/`（規則、名簿、作業ノート、数百件に及ぶレビューノート）。

---

<a id="credits"></a>

## クレジット

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0、Apache-2.0 ライセンス：`acvram/kernels/marlin_port/` はその Marlin カーネル（MoE 向けおよび密モデル向け）を移植したもので、ファイルごとの完全な帰属表示は [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE) にある。
- **NVIDIA** — CUDA、Blackwell の FP4 Tensor コア（`sm_120`）、および本プロジェクトが依存するライブラリ。
- **PyTorch** — テンソルエンジンと C++/CUDA 拡張。

独立したプロジェクトであり、ASUS、NVIDIA、vLLM プロジェクトのいずれとも提携関係はない。

---

<a id="licence"></a>

## ライセンス

このリポジトリのコードは [GPL-3.0-or-later](../LICENSE)。`acvram/kernels/marlin_port/` には、[vLLM](https://github.com/vllm-project/vllm) v0.29.0 から移植したコード（`marlin_moe_wna16`、`gptq_marlin_repack`、`moe_align_block_size` の各カーネル）が Apache-2.0 ライセンスのもとで含まれている。各ファイルは元のヘッダーを保持しており、ライセンス本文は `LICENSE-vllm` に、ファイル一覧・元のコミット・変更点は [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE) にある。

---

<a id="soutien"></a>

## プロジェクトを支援する

acvram は個人所有のハードウェアで開発されている。このプロジェクトが役に立ったなら：

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

翻訳について：[TRADUIRE.md](TRADUIRE.md)（フランス語。プロジェクトのコントリビューションガイドはまだ翻訳されていない）。
