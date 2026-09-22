<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> 支援する: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

OpenAI API 互換の推論ゲートウェイ。メモリを階層として扱い、各 GPU に、その
シリコンが最もよく読める数値形式を与える。

特定の一台のマシンのために設計されている:

| | |
|---|---|
| プロセッサ | Intel Core i9-14900K（P コア 8 + E コア 16） |
| マザーボード | ASUS ROG Maximus Z790 Dark Hero |
| メモリ | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| システム | Ubuntu 26.04 LTS（CUDA 13）; 両カードとも PCIe x8/x8、400 W / 275 W に制限 |

## 二つの考え方

**GPU ごとに一つの形式。** RTX 5090 には FP4 テンソルコアがある。RTX 3080 Ti
にはなく、FP8 もない。両者を共通形式に揃えれば 5090 を無駄にする。そこで
変換器は、*同じモデルを二度*、各宛先が実際に活用できる形式で書き出す:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| 重み | **NVFP4** — E2M1 + 16 個ごとの FP8 E4M3 スケール | **INT4** — uint4 + 128 個ごとの fp16 スケールとゼロ点 |
| 重みあたりビット | 4,50 | 4,16 |
| BF16 比 | ×3,56 小さい | ×3,85 小さい |
| 計算モード | FP4 テンソルコア | カーネル内で FP16 に逆量子化、FP16 テンソルコア |
| KV キャッシュ | INT8 | INT8 |

重みあたり 4,5 ビットなら 32 GB の VRAM に約 **560 億パラメータ**が収まる。
BF16 では 160 億だ。二枚のカードを合わせると、メインメモリに触れる前に
およそ **780 億の常駐パラメータ**になる。

**メモリは壁ではなく階層である。** 三段構えで、プランナーはモデルが収まる
ことを祈るのではなく、各段のコストを測る:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## クイックスタート

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

あとは任意の OpenAI クライアントが接続できる:

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

## `acvram plan` が語ること

プランナーはどんなダウンロードよりも先に実行する価値がある。あるモデルが
このマシンで使えるかを決める問いに答えてくれる:

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

最初に収まった構成を採るのではなく構成空間を探索し、その判断のうち二つは
直観に反するので明記しておく:

* **モデルが 5090 一枚に収まるなら 3080 Ti を使わない。** パイプラインの
  各区間は直列に走る。1790 GB/s のパイプラインに 912 GB/s の段を足せば、
  単一ストリームのデコードは遅くなる。`--gpus all` で強制できる。
* **重みを VRAM に留めるために KV キャッシュを縮める。** キャッシュに渡す
  1 GB は、PCIe バスへ押し出される重み 1 GB であり、PCIe 越しの重みの読み
  出しは VRAM からのおよそ三十倍かかる。上の 70B では、この一つの判断だけで
  2,3 から 17,8 トークン/秒に上がる。

## 速く走る

四つの最適化。いずれもストップウォッチではなく等価性の証明で検証されて
いる。答えを変える最適化はバグである。

### 投機的デコード (`--speculative`)

バッチサイズ 1 で一トークンをデコードする処理はメモリ律速だ。マシンは一
トークンを出すために全アクティブ重みを読む。提案された K トークンの検証
は、同じ重みを**一度だけ**読む。二つの提案器:

* `ngram`（既定）— 現在の接尾辞をコンテキストの前方に探し、その後に続いた
  ものを提案する。コストゼロ、モデル不要。出力が入力を写す場面で効く:
  コード編集、RAG、要約。
* `draft` — 第二のデバイス上の小さなモデル。この構成ではそれは RTX 3080 Ti
  で、5090 に収まるモデルに対してプランナーが意図的に遊ばせている。

受理は近似ではなく厳密だ。提案は確率 `min(1, p/q)` で受理され、棄却時は
`p - q` の正規化した正の部分から再サンプリングする。意図的に較正の悪い
ドラフトに対して 40 000 回の抽選で測ると、出力分布は目標から全変動 0,002
以内に留まる。投機は速度を買うのであって、別の答えを買うことは決してない。

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### プレフィックスキャッシュ（既定で有効）

ブロックは、そのトークン区間の*連鎖*ハッシュでアドレスされる。システム
プロンプトを共有する二つの要求はそのブロックを共有し、二つ目はそれを再計算
しなくてよい。連鎖は不可欠だ。同じ十六トークンでも文脈が違えば同じキーと
値を持たず、区間だけをハッシュすれば、あるシーケンスのキャッシュを別の
シーケンスに渡してしまう。

解放されたが内容がまだ識別できるブロックは、空きリストではなく LRU キュー
に入る。こうしてキャッシュは要求をまたいで生き残り、応えられたはずの割り
当てを拒むことはない。

### ホスト段での計算 (`--host-exec`)

重みが RAM にある層は、GPU にコピーするか、その場で計算するかを選べる。
どちらもメモリ律速で同じバイトを読む。速いのはバスの広い方だ。PCIe 5.0 x16
は約 54 GB/s、デュアルチャネル DDR5 は約 70 GB/s。しかもその場で計算すれば、
GPU をコピー待ちにせず空けておける。

これが成立するのは、プロセッサが 4 ビット詰めの重みを直接読める場合だけ
だ。そこで AVX2 パスを持つ小さな C++ カーネル（`acvram_cpu.cpp`、ctypes で
読み込み、Python ヘッダも ninja も不要）がある。**スカラー**のフォールバック
枝でさえ `dequantize() @ x` を INT4 で 1,44 倍、NVFP4 で 3,21 倍上回る。後者は
まず行列全体の 32 ビットコピーを書き出すからだ。

Mistral-Large-123B では、プランナーの見積もりが 1,35 から 2,42 トークン/秒に
なる。

### 混合精度（`--snr-floor`、既定でオフ）

変換器は各テンソルについて層出力の信号対雑音比を測り、`--snr-floor` を下回る
ものをより広い形式に昇格できる。上限はテンソルの 15 % と価格上限
（`--promotion-cout-max`、追加 MiB）。

閾値は**既定でゼロ**。何も昇格しない。デコードはメモリ帯域律速で、
`Huihui-Qwen3.8-27B` での測定が決着をつける。25 dB の閾値は、パープレキシティ
2,0 %（16 383 トークンのコーパスで 42,591 対 43,447）のために、メモリ 13,4 %
とスループット 10,6 %（16,02 と 46,2 に対し 18,50 GiB と 41,8 t/s）を払う。
品質が速度に優先するときは `--snr-floor 25` で従来の挙動に戻る。

### そして `acvram eval`

信号対雑音比とロジットの余弦は近似にすぎない。`acvram eval DIR [DIR ...]`
はスライディングウィンドウでパープレキシティを測り、形式の選択を証拠で
決められるようにする:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP エンドポイント

| エンドポイント | 備考 |
|---|---|
| `POST /v1/chat/completions` | SSE ストリームまたは単一応答; モデルのチャットテンプレートを使用 |
| `POST /v1/completions` | テキストまたはトークン ID のプロンプト |
| `POST /v1/embeddings` | 最終隠れ状態の平均、L2 正規化、`dimensions` を尊重 |
| `GET /v1/models` | 加えて `acvram` ブロック: 形式、デバイス、KV キャッシュ容量 |
| `GET /health`, `GET /metrics` | デコードスループット、KV ブロックの占有率 |

これらの応答のフィールド名は英語のままだ。OpenAI プロトコルであり、訳せば
既存クライアントがすべて壊れる。

## 数値の出所

上で引いた値はすべてこのリポジトリのコードが生成し、`pytest` で検証している。
参照カーネルを用いた CPU 上の測定:

| 形式 | ビット/重み | 重みの SNR | BF16 に対するロジット余弦 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

これらの測定から得た二つの知見が既定値を変えた:

* **アダマール回転は INT4 に効き、NVFP4 には効かない。** INT4 の 128 個
  グループは孤立した外れ値チャネルを吸収できないため、極端な値を散らす
  ことは活性化ごとの n log n 変換に見合う。NVFP4 の 16 個ブロックはすでに
  自前のスケールを持つ。そこで `--hadamard auto` は INT4 にだけ適用する。
* **KV キャッシュでは INT8 が FP8 E4M3 に勝つ。** 同サイズで 44 dB 対 32 dB。
  （トークン、ヘッド）ごとのスケールが、FP8 が指数ビットを費やすダイナミック
  レンジをすでに与えるからだ。5090 が FP8 を扱えても、両カードは INT8 の
  KV キャッシュを使う。

## ドキュメント

* [`REPRISE.md`](../REPRISE.md) — **別のマシンでプロジェクトを再開する**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — 部品がどう組み合わさるか
* [`docs/MATERIEL.md`](MATERIEL.md) — この特定のマシンの調整
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **未完了のこと**、最初に読む
* [`CONVENTIONS.md`](../CONVENTIONS.md) — コード作業の規約（言語、スタイル、push 前の確認）

## 測定結果（2026/09/22、RTX 5090 を 400 W で、電力計で ≥ 20 秒の運転）

Qwen3-Coder-30B-A3B を NVFP4（エキスパート）+ INT8（アテンション、ヘッド）で、
全エンジン同一プロトコル（`outils/`、カード一枚、`energie.py`）:

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| デコード 12 シーケンス | **1 625,5 t/s** | 1 596,1 t/s | — |
| デコード 1 シーケンス | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 トークン/秒** | 21 054 | 8 671（TabbyAPI、撤回） |

当日のスループット（拠点 1030、エコ運転 `-lgc 2700`、稼働中パイプライン。
貪欲サンプリングを CUDA グラフに取り込み、0.6.35 の既定）。b=12 は封印済みの
公式セル（インターリーブした 6 ウィンドウの中央値、ウィンドウごとのクロック）。
vLLM の 1 596,1 は 09/21 の凍結参照（当日 vLLM は再実行せず）で、差の +1,84 % は
同一参照での値であり、同じ朝に両者を再測定したものではない。3 エンジンに
対するクロック同一の J/トークンは再測定中
（`outils/gpu/mesure/banc-4moteurs.py`）。運転条件のない数値は公開しない。

09/14 の朝、同じセルで acvram は 630 t/s、0,619 J/トークンだった。伸びは
Blackwell ネイティブの FP4 MMA（`mma.sync … kind::mxf4nvf4`、bf16 比 ×7,9）、
バッチ桶ごとのグループ化 GEMM としての MoE、単一カーネルでのルーティング
（1 ステップあたり 3 677 → 1 517 回の起動）、射影向けのテンソルコア上の狭い
GEMM から来ている。各数値には `acvram-memoire/revue/` に、測定前に封印した
予測、計測器、その運転条件を記したノートがある。運転条件のない数値は公開
しない。

acvram が先行する領域: sm_120 ネイティブ NVFP4 の MLA モデル（GLM-4.7-Flash。
vLLM は FP8 でしか提供しない。b=1: 稼働中 165,35 t/s）、VRAM に収まらない
モデル、そして 0.6.35 以降は VRAM に収まる MoE の大バッチデコード — b=12 は
1 540（0.6.34）から 1 625,5 t/s になり、凍結した vLLM 参照（1 596,1）を +1,84 %
上回る。差は依然として僅かで、参照は凍結値。エネルギーの差は再測定が必要。

## 状態

バージョン 0.6.35。すべて 5090 上で動く。`sm_120a`（ネイティブ FP4）と `sm_86`
向けにコンパイルした CUDA カーネル、CUDA グラフ、NVFP4/INT8/INT4 量子化、
HTTP サーバー。ガードレールは設置済み。カードは作業セッションから不可視
（`CUDA_VISIBLE_DEVICES` は空）で、`outils/carte.sh` だけがロックの下で一度に
一つの測定に貸し出す。監視役がロック外のアクセスをすべて記録する。二枚
以上のカードにまたがる、あるいは 10 秒未満のエネルギー測定は無効。劣化
モードで読み込まれたモデルはそう申告し、対決に参加しない。

640 テスト（`pytest -q`、CPU で一分。GPU テストは `carte.sh` の下でのみ動く）。
作業の追跡: `acvram-memoire/`（規則、名簿、ノート、180 件のレビュー）。

## 支援する

acvram は個人の機材で開発されている。役に立ったなら:
**支援する: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**。

## ライセンス

GPL-3.0 以降。
