<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> 후원하기: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

OpenAI API와 호환되는 추론 게이트웨이. 메모리를 계층으로 다루고, 각 GPU에
그 실리콘이 가장 잘 읽는 수치 형식을 부여한다.

특정한 한 대의 머신을 위해 설계되었다:

| | |
|---|---|
| 프로세서 | Intel Core i9-14900K (P 코어 8 + E 코어 16) |
| 메인보드 | ASUS ROG Maximus Z790 Dark Hero |
| 메모리 | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| 시스템 | Ubuntu 26.04 LTS (CUDA 13); 두 카드 모두 PCIe x8/x8, 400 W / 275 W로 제한 |

## 두 가지 아이디어

**GPU마다 하나의 형식.** RTX 5090에는 FP4 텐서 코어가 있다. RTX 3080 Ti에는
없고 FP8도 없다. 둘을 공통 형식에 맞추면 5090을 낭비하게 된다. 그래서
변환기는 *같은 모델을 두 번*, 각 대상이 실제로 활용할 수 있는 형식으로
기록한다:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| 가중치 | **NVFP4** — E2M1 + 16개마다 FP8 E4M3 스케일 | **INT4** — uint4 + 128개마다 fp16 스케일과 영점 |
| 가중치당 비트 | 4,50 | 4,16 |
| BF16 대비 | ×3,56 작음 | ×3,85 작음 |
| 연산 방식 | FP4 텐서 코어 | 커널 안에서 FP16으로 역양자화, FP16 텐서 코어 |
| KV 캐시 | INT8 | INT8 |

가중치당 4,5비트로 32 GB VRAM에 약 **560억 개의 파라미터**가 들어간다. BF16으로는
160억 개다. 두 카드를 합치면 메인 메모리에 손대기도 전에 약 **780억 개의 상주
파라미터**가 된다.

**메모리는 벽이 아니라 계층이다.** 세 단계이며, 플래너는 모델이 들어가기를
바라는 대신 각 단계의 비용을 측정한다:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## 빠른 시작

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

이후 어떤 OpenAI 클라이언트든 연결할 수 있다:

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

## `acvram plan`이 말해 주는 것

플래너는 어떤 다운로드보다 먼저 실행할 가치가 있다. 어떤 모델이 이 머신에서
쓸 만한지를 결정하는 질문에 답한다:

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

들어가는 첫 구성을 고르는 대신 구성 공간을 탐색하며, 그 결정 중 두 가지는
직관에 반하므로 명시해 둔다:

* **모델이 5090 한 장에 들어가면 3080 Ti를 쓰지 않는다.** 파이프라인의 각
  구간은 직렬로 실행된다: 1790 GB/s 파이프라인에 912 GB/s 단계를 더하면
  단일 스트림 디코딩이 느려진다. `--gpus all`로 강제할 수 있다.
* **가중치를 VRAM에 남기기 위해 KV 캐시를 줄인다.** 캐시에 주는 1 GB는
  PCIe 버스로 밀려나는 가중치 1 GB이고, PCIe로 가중치 하나를 읽는 비용은
  VRAM에서 읽는 비용의 약 서른 배다. 위의 70B에서는 이 하나의 절충만으로
  2,3에서 17,8 토큰/초로 올라간다.

## 빠르게 가기

네 가지 최적화. 각각 스톱워치가 아니라 동치 증명으로 검증되었다: 답을
바꾸는 최적화는 버그다.

### 추측 디코딩 (`--speculative`)

배치 크기 1로 토큰 하나를 디코딩하는 일은 메모리 제한이다: 머신은 토큰 하나를
만들기 위해 모든 활성 가중치를 읽는다. 제안된 K개의 토큰을 검증하면 같은
가중치를 **한 번만** 읽는다. 두 가지 제안기:

* `ngram`(기본값) — 현재 접미사를 컨텍스트 앞부분에서 찾아 그 뒤에 왔던 것을
  제안한다. 비용이 없고 모델도 필요 없다. 출력이 입력을 베끼는 경우에
  이득이다: 코드 편집, RAG, 요약.
* `draft` — 두 번째 장치의 작은 모델. 이 장비에서 그 장치는 RTX 3080 Ti이며,
  플래너는 5090에 들어가는 모든 모델에 대해 의도적으로 이를 놀린다.

수락은 근사가 아니라 정확하다: 제안은 확률 `min(1, p/q)`로 수락되고, 기각
시에는 `p - q`의 정규화된 양의 부분에서 다시 표집한다. 의도적으로 보정을
망친 초안을 상대로 40 000회 추첨을 측정하면, 출력 분포는 목표로부터 전변동
0,002 이내에 머문다 — 추측은 속도를 사지, 결코 다른 답을 사지 않는다.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### 접두사 캐시 (기본 활성)

블록은 토큰 구간의 *연쇄* 해시로 주소가 지정된다: 시스템 지시를 공유하는 두
요청은 그 블록을 공유하고, 두 번째 요청은 이를 다시 계산할 필요가 없다.
연쇄는 필수다: 다른 문맥의 같은 열여섯 토큰은 같은 키와 값을 갖지 않으며,
구간만 해시하면 한 시퀀스의 캐시를 다른 시퀀스에 넘기게 된다.

해제되었지만 내용을 여전히 식별할 수 있는 블록은 빈 목록 대신 LRU 큐로 간다:
캐시는 이렇게 요청 사이에 살아남되, 응할 수 있었던 할당을 결코 거부하지
않는다.

### 호스트 단계에서의 계산 (`--host-exec`)

가중치가 RAM에 있는 층은 GPU로 복사하거나 제자리에서 계산할 수 있다. 두 경로
모두 메모리 제한이며 같은 바이트를 읽는다: 더 빠른 쪽은 버스가 더 넓은 쪽이다
— PCIe 5.0 x16은 약 54 GB/s, 듀얼 채널 DDR5는 약 70 GB/s — 그리고 제자리 계산은
GPU를 복사 대기시키지 않고 비워 둔다.

이는 프로세서가 4비트로 묶인 가중치를 직접 읽을 때만 값어치가 있다. 그래서
AVX2 경로를 가진 작은 C++ 커널이 있다(`acvram_cpu.cpp`, ctypes로 로드, Python
헤더나 ninja 불필요). **스칼라** 대체 분기에서조차 `dequantize() @ x`를 INT4에서
1,44배, NVFP4에서 3,21배 앞선다. 후자는 먼저 행렬 전체의 32비트 사본을 쓰기
때문이다.

Mistral-Large-123B에서 플래너의 추정치는 1,35에서 2,42 토큰/초로 오른다.

### 혼합 정밀도 (`--snr-floor`, 기본 꺼짐)

변환기는 각 텐서에 대해 층 출력의 신호 대 잡음비를 측정하고, `--snr-floor`
아래로 떨어지는 텐서를 더 넓은 형식으로 승격할 수 있다. 상한은 텐서의 15 %와
가격 한도(`--promotion-cout-max`, 추가 MiB).

문턱은 **기본값 0**: 아무것도 승격되지 않는다. 디코딩은 메모리 대역폭 제한이며,
`Huihui-Qwen3.8-27B`에서의 측정이 결론을 낸다 — 25 dB 문턱은 퍼플렉시티 2,0 %
(16 383 토큰 코퍼스에서 42,591 대 43,447)를 위해 메모리 13,4 %와 처리량
10,6 %(16,02와 46,2 대비 18,50 GiB와 41,8 t/s)를 치른다. 품질이 속도보다
중요할 때는 `--snr-floor 25`가 이전 동작을 되살린다.

### 그리고 `acvram eval`

신호 대 잡음비와 로짓 코사인은 근사다. `acvram eval DIR [DIR ...]`은 슬라이딩
윈도로 퍼플렉시티를 측정해 형식 선택이 증거로 결정되게 한다:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP 엔드포인트

| 엔드포인트 | 비고 |
|---|---|
| `POST /v1/chat/completions` | SSE 스트림 또는 단일 응답; 모델의 채팅 템플릿 사용 |
| `POST /v1/completions` | 텍스트 또는 토큰 ID 형태의 프롬프트 |
| `POST /v1/embeddings` | 최종 은닉 상태 평균, L2 정규화, `dimensions` 준수 |
| `GET /v1/models` | 추가로 `acvram` 블록: 형식, 장치, KV 캐시 용량 |
| `GET /health`, `GET /metrics` | 디코딩 처리량, KV 블록 점유율 |

이 응답들의 필드 이름은 영어로 남는다: OpenAI 프로토콜이며, 번역하면 기존
클라이언트가 모두 깨진다.

## 숫자의 출처

위에 인용된 모든 값은 이 저장소의 코드가 생성하고 `pytest`로 검증한다.
참조 커널로 프로세서에서 수행한 측정:

| 형식 | 비트/가중치 | 가중치 SNR | BF16 대비 로짓 코사인 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

이 측정에서 얻은 두 가지 발견이 기본값을 바꿨다:

* **아다마르 회전은 INT4에는 도움이 되고 NVFP4에는 그렇지 않다.** INT4의 128개
  그룹은 고립된 이상치 채널을 흡수하지 못하므로, 극단값을 흩뜨리는 일은
  활성화당 n log n 변환의 값어치가 있다. NVFP4의 16개 블록은 이미 자체 스케일을
  지닌다. 그래서 `--hadamard auto`는 INT4에만 적용한다.
* **KV 캐시에서는 INT8이 FP8 E4M3를 이긴다.** 같은 크기에서 44 dB 대 32 dB.
  (토큰, 헤드)별 스케일이 FP8이 지수 비트를 쓰는 동적 범위를 이미 제공하기
  때문이다. 그래서 5090이 FP8을 할 수 있어도 두 카드 모두 INT8 KV 캐시를 쓴다.

## 문서

* [`REPRISE.md`](../REPRISE.md) — **다른 머신에서 프로젝트 재개하기**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — 부품이 어떻게 맞물리는가
* [`docs/MATERIEL.md`](MATERIEL.md) — 바로 이 머신의 조정
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **아직 안 된 것**, 먼저 읽을 것
* [`CONVENTIONS.md`](../CONVENTIONS.md) — 코드 작업 규약(언어, 스타일, push 전 점검)

## 측정 결과 (2026/09/22, RTX 5090 400 W, 전력계 기준 ≥ 20초 운전)

Qwen3-Coder-30B-A3B를 NVFP4(전문가) + INT8(어텐션, 헤드)로, 모든 엔진에 동일한
프로토콜(`outils/`, 카드 한 장, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| 디코딩 12 시퀀스 | **1 625,5 t/s** | 1 596,1 t/s | — |
| 디코딩 1 시퀀스 | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 토큰/초** | 21 054 | 8 671 (TabbyAPI, 철회) |

당일 처리량(거점 1030, 에코 운전 `-lgc 2700`, 서비스 중 파이프라인; 그리디
샘플링을 CUDA 그래프에 담았고 0.6.35의 기본값). b=12는 봉인된 공식 셀(인터리브
한 6개 창의 중앙값, 창별 클럭)이다. vLLM 1 596,1은 09/21의 동결 기준(당일 vLLM
재실행 안 함)이며, 격차 +1,84 %는 동일 기준에서의 값이지 같은 아침에 둘을 재측정
한 것이 아니다. 세 엔진에 대한 동일 클럭 J/토큰은 재측정 중이다
(`outils/gpu/mesure/banc-4moteurs.py`) — 운전 조건 없는 수치는 공개하지 않는다.

09/14 아침 같은 셀에서 acvram은 630 t/s, 0,619 J/토큰이었다: 이득은 Blackwell의
네이티브 FP4 MMA(`mma.sync … kind::mxf4nvf4`, bf16 대비 ×7,9), 배치 버킷별 그룹
GEMM으로서의 MoE, 단일 커널 라우팅(스텝당 3 677 → 1 517회 실행), 투영용 텐서
코어 좁은 GEMM에서 온다. 각 수치는 `acvram-memoire/revue/`에 측정 전 봉인된
예측, 계측기, 그 운전 조건을 적은 노트를 가진다 — 운전 조건 없는 수치는
공개하지 않는다.

acvram이 앞서는 곳: sm_120 네이티브 NVFP4의 MLA 모델(GLM-4.7-Flash, vLLM은
FP8로만 제공, b=1: 서비스 중 165,35 t/s), VRAM에 들어가지 않는 모델, 그리고
0.6.35부터는 VRAM에 들어가는 MoE의 대배치 디코딩 — b=12는 1 540(0.6.34)에서
1 625,5 t/s가 되어 동결된 vLLM 기준(1 596,1)을 +1,84 % 앞선다. 격차는 여전히
좁고 기준은 동결값이다; 에너지 격차는 재측정이 필요하다.

## 상태

버전 0.6.35. 모든 것이 5090에서 돈다: `sm_120a`(네이티브 FP4)와 `sm_86`용으로
컴파일한 CUDA 커널, CUDA 그래프, NVFP4/INT8/INT4 양자화, HTTP 서버. 안전장치
가동 중: 카드는 작업 세션에 보이지 않고(`CUDA_VISIBLE_DEVICES` 비움),
`outils/carte.sh`만이 잠금 아래 한 번에 하나의 측정에 빌려 준다; 감시자가 잠금
밖의 모든 접근을 기록한다; 카드 두 장 이상에 걸치거나 10초 미만인 에너지
측정은 무효; 저하 모드로 로드된 모델은 그 사실을 알리고 대결에 들어가지
않는다.

640개 테스트(`pytest -q`, 프로세서에서 1분; GPU 테스트는 `carte.sh` 아래에서만
실행). 작업 추적: `acvram-memoire/`(규칙, 명부, 노트, 180개 노트 리뷰).

## 후원하기

acvram은 개인 장비로 개발된다. 프로젝트가 도움이 되었다면:
**후원하기: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## 라이선스

GPL-3.0 이상.
