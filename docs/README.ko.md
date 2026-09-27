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

OpenAI API와 호환되는 추론 게이트웨이입니다. 메모리를 하나의 계층 구조로 다루고, 각 GPU에 그 실리콘이 가장 잘 읽는 수치 형식을 부여하며, 모든 토큰을 초 단위만큼이나 줄(J) 단위로도 최적화합니다.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · **🇰🇷 한국어** · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Throughput and energy comparison against vLLM and llama.cpp" width="720"></p>

---

## 목차

- [두 가지 아이디어](#idees)
- [빠른 시작](#demarrage)
- [설치](#installer)
- [`acvram plan`이 알려 주는 것](#plan)
- [속도를 내는 법](#optimisations)
- [HTTP 엔드포인트](#http)
- [수치의 출처](#chiffres)
- [문서](#documentation)
- [측정 결과](#resultats)
- [현황](#etat)
- [크레딧](#credits)
- [라이선스](#licence)
- [프로젝트 후원](#soutien)

---

<a id="idees"></a>

## 두 가지 아이디어

특정한 한 대의 머신을 위해 설계되었습니다:

| | |
|---|---|
| CPU | Intel Core i9-14900K (P 코어 8개 + E 코어 16개) |
| 메인보드 | ASUS ROG Maximus Z790 Dark Hero |
| 메모리 | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| 시스템 | Ubuntu 26.04 LTS (CUDA 13); 두 카드 모두 PCIe x8/x8, 전력 제한 400 W / 275 W |

**GPU마다 하나의 형식.** RTX 5090에는 FP4 텐서 코어가 있지만, RTX 3080 Ti에는 FP4 텐서 코어도 FP8도 없습니다. 두 카드를 공통 형식에 맞추면 5090을 낭비하게 됩니다. 그래서 변환기는 *같은 모델을 두 번* 기록하되, 각 대상이 실제로 활용할 수 있는 형식으로 씁니다:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| 가중치 | **NVFP4** — E2M1 + 16개마다 FP8 E4M3 스케일 | **INT4** — uint4 + 128개마다 fp16 스케일과 영점 |
| 가중치당 비트 | 4.50 | 4.16 |
| BF16 대비 | ×3.56 작음 | ×3.85 작음 |
| 연산 방식 | FP4 텐서 코어 | 커널 내부에서 FP16으로 역양자화, FP16 텐서 코어 |
| KV 캐시 | INT8 | INT8 |

가중치당 4.5비트라면 32 GB VRAM에 약 **560억 개의 파라미터**가 들어갑니다. BF16에서는 160억 개입니다. 두 카드를 합치면 호스트 메모리에 손대기도 전에 약 **780억 개의 파라미터가 상주**합니다.

**메모리는 벽이 아니라 계층입니다.** 계층은 세 단계이며, 플래너는 모델이 들어가기를 바라는 대신 각 단계가 실제로 치르는 비용을 측정합니다:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
호스트 DDR5  96 GB   PCIe 또는 DDR에 의해 제한
```

---

<a id="demarrage"></a>

## 빠른 시작

```bash
./install.sh                       # 가상 환경 + torch cu128 + acvram
acvram doctor                      # 이 머신이 준비되었는지, 무엇에 쓸 수 있는지
acvram detect                      # 실제로 무엇이 있는지

acvram plan  ~/models/Qwen3-32B                     # 각 레이어가 어디에 배치될지
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

어떤 OpenAI 클라이언트든 곧바로 연결됩니다:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"안녕하세요"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="미사용")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "안녕하세요"}])
```

---

<a id="installer"></a>

## 설치

소스에서 설치(모든 플랫폼):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

또는 패키지로 설치할 수 있으며, 각 [GitHub 릴리스](https://github.com/anticitoyun/anticitoyen-vram/releases/latest)에 파일이 하나씩 첨부되어 있습니다:

| 채널 | 릴리스 첨부 파일 | 명령 |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (파일 이름은 `rpmbuild`가 생성하며 고정되어 있지 않음) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (또는 `rpmbuild --rebuild *.src.rpm`으로 `.src.rpm`에서 빌드) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

`.flatpakref`는 항상 저장소의 최신 게시된 버전을 설치합니다.

설치하기 전에, 릴리스에 첨부된 체크섬(`SHA256SUMS`, 다른 모든 파일이 존재한 후에 게시됨)으로 다운로드한 파일을 검증하십시오:

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip 패키지로는 배포하지 않습니다(빌드된 휠 없음). `pip install -e '.[dev]'`는 `./install.sh`와 마찬가지로 소스 클론에서 설치합니다.

---

<a id="plan"></a>

## `acvram plan`이 알려 주는 것

플래너는 무엇이든 다운로드하기 전에 실행해 볼 만합니다. 어떤 모델을 이 머신에서 쓸 수 있는지를 가르는 질문들에 답해 줍니다:

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  계층    형식        용량          가중치        KV  구간
  cuda:0  nvfp4        30.3 GiB    25.5 GiB   4.5 GiB  레이어 0-58
  cuda:1  int4_awq     10.9 GiB     8.7 GiB   1.6 GiB  레이어 59-79
  cpu     nvfp4        74.8 GiB     3.4 GiB      0 B   -

  가중치 합계        37.6 GiB
  토큰당 읽기량      35.1 GiB
  토큰당 KV          162.5 KiB  -> 39,843 토큰 캐시
  호스트 RAM의 MLP   55-58

  디코드 추정치      17.8 토큰/s  (배치 1)
  프리필 추정치      847 토큰/s
```

플래너는 처음으로 들어맞는 구성에서 멈추지 않고 구성 공간 전체를 탐색하며, 그 결정 가운데 두 가지는 직관에 반하는 만큼 따로 설명할 가치가 있습니다:

* **모델이 5090 한 장에 들어가면 3080 Ti를 쉬게 둡니다.** 파이프라인의 구간은 직렬로 실행됩니다. 1790 GB/s 파이프라인에 912 GB/s 단계를 추가하면 단일 스트림 디코드가 느려집니다. `--gpus all`로 강제할 수 있습니다.
* **가중치를 VRAM에 두기 위해 KV 캐시를 줄입니다.** 캐시에 내준 용량은 그대로 PCIe 버스로 밀려나는 가중치의 용량이며, PCIe로 가중치를 읽는 비용은 VRAM에서 읽는 비용의 약 서른 배입니다. 위의 70B 모델에서는 이 절충 하나만으로 디코드가 2.3에서 17.8 토큰/s로 올라갑니다.

---

<a id="optimisations"></a>

## 속도를 내는 법

네 가지 최적화가 있으며, 각각은 스톱워치만이 아니라 동치성 증명으로 검증됩니다. 답을 바꾸는 최적화는 버그입니다.

밀집(dense) 모델의 NVFP4 선형 계층은 기본적으로 Marlin 레이아웃을 거칩니다(revue/poste6-piece147-verdict-24-09.md 기준 b = 8에서 처리량 +57~+90%, TTFT +2~+4 ms; 되돌리려면 `ACVRAM_PROJ_MARLIN=0`, [CHANGELOG.md](../CHANGELOG.md) 참조).

### 추측 디코딩 (`--speculative`)

배치 크기 1로 토큰 하나를 디코드하는 작업은 메모리에 묶여 있습니다. 머신은 토큰 하나를 만들기 위해 활성 가중치를 모두 읽습니다. 제안된 토큰 K개를 검증할 때는 같은 가중치를 **단 한 번만** 읽습니다. 제안기는 두 가지입니다:

* `ngram`(기본값) — 현재 접미사를 컨텍스트 앞부분에서 찾아 그 뒤에 이어졌던 내용을 제안합니다. 비용이 들지 않고 모델도 필요 없습니다. 출력이 입력을 되풀이할 때 효과를 봅니다: 코드 편집, RAG, 요약.
* `draft` — 두 번째 장치에 올린 작은 모델. 이 장비에서 그 장치는 RTX 3080 Ti이며, 플래너는 5090에 들어가는 모델이라면 이 카드를 의도적으로 쉬게 둡니다.

`mtp`(모델의 `nextn` 헤드)와 `auto`도 있지만, 현재 상태로는 이득이 없어 기본으로 활성화되어 있지 않습니다 — `docs/ARCHITECTURE.md` 참조.

수락은 근사가 아니라 정확합니다. 제안은 확률 `min(1, p/q)`로 수락되고, 거부되면 `p - q`의 양의 부분을 정규화한 분포에서 다시 샘플링합니다. 일부러 잘못 보정한 초안 모델을 상대로 40,000회 추출해 측정한 결과, 방출된 분포는 목표 분포와의 총변동 거리 0.002 이내에 머뭅니다 — 추측이 사 오는 것은 속도일 뿐, 결코 다른 답이 아닙니다.

```
장난감 모델, 그리디, k=4      스텝      토큰/스텝   출력
  추측 없음                      23          1.00   기준
  n-그램                         13          1.77   동일
  초안 (= 목표)                   5          4.60   동일
```

### 접두사 캐시 (기본 활성)

블록은 해당 토큰 구간의 *연쇄* 해시로 주소가 매겨집니다. 시스템 프롬프트를 공유하는 두 요청은 그 블록들도 공유하며, 두 번째 요청은 더 이상 이를 미리 계산할 필요가 없습니다. 연쇄는 필수입니다. 같은 열여섯 개의 토큰이라도 다른 컨텍스트에서는 같은 키와 값을 담지 않으므로, 구간만 해시하면 한 시퀀스의 캐시를 다른 시퀀스에 내주게 됩니다.

해제된 블록 가운데 내용을 여전히 식별할 수 있는 블록은 빈 블록 목록이 아니라 LRU 큐에 들어갑니다. 이렇게 해서 캐시는 요청과 요청 사이에서도 살아남으며, 처리할 수 있었던 할당을 거부하는 일은 결코 없습니다.

### 호스트 계층 연산 (`--host-exec`)

가중치가 RAM에 있는 레이어는 GPU로 복사할 수도, 그 자리에서 연산할 수도 있습니다. 두 경로 모두 메모리에 묶여 있고 같은 바이트를 읽습니다. 더 빠른 쪽은 버스가 더 넓은 쪽입니다 — PCIe 5.0 x16은 약 54 GB/s, 듀얼 채널 DDR5는 약 70 GB/s — 게다가 그 자리에서 연산하면 GPU를 복사 대기에 묶어 두지 않고 자유롭게 남겨 둡니다.

이는 CPU가 패킹된 4비트 가중치를 직접 읽을 때만 이득이 됩니다. 그래서 AVX2 경로를 갖춘 작은 C++ 커널을 두었습니다(`acvram_cpu.cpp`, ctypes로 로드, Python 헤더나 ninja 불필요). **스칼라** 폴백 분기에서조차 `dequantize() @ x`보다 INT4에서 1.44배, NVFP4에서 3.21배 빠른데, 후자는 먼저 행렬 전체의 32비트 사본을 쓰기 때문입니다.

Mistral-Large-123B에서 플래너의 추정치는 1.35에서 2.42 토큰/s로 올라갑니다.

### 혼합 정밀도 (`--snr-floor`, 기본 비활성)

변환기는 모든 텐서에 대해 레이어 출력의 신호 대 잡음비를 측정하고, `--snr-floor` 아래로 떨어지는 텐서를 더 넓은 형식으로 승격할 수 있습니다. 단, 전체 텐서의 15%라는 상한과 가격 상한(`--promotion-cout-max`, 추가되는 메비바이트 단위) 안에서입니다.

하한은 **기본값이 영**이므로 아무것도 승격되지 않습니다. 디코딩은 메모리 대역폭에 묶여 있으며, `Huihui-Qwen3.8-27B`에서의 측정이 결론을 내려 줍니다 — 25 dB 하한은 메모리 13.4%와 처리량 10.6%(18.50 GiB와 41.8 t/s 대 16.02와 46.2)를 대가로 퍼플렉서티를 2.0% 개선합니다(42.591 대 43.447, 16,383토큰 코퍼스). 속도보다 품질이 중요할 때는 `--snr-floor 25`로 예전 동작을 되돌릴 수 있습니다.

### 그리고 `acvram eval`

신호 대 잡음비와 로짓 코사인 유사도는 근사치입니다. `acvram eval REP [REP ...]`는 슬라이딩 윈도 퍼플렉서티를 측정하므로, 형식 선택을 증거로 결정할 수 있습니다:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  모델                     ppl     bpp        크기       토큰
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## HTTP 엔드포인트

| 엔드포인트 | 비고 |
|---|---|
| `POST /v1/chat/completions` | SSE 스트림 또는 단일 응답; 모델의 채팅 템플릿 사용 |
| `POST /v1/completions` | 텍스트 또는 토큰 ID로 된 프롬프트 |
| `POST /v1/embeddings` | 최종 은닉 상태의 평균 풀링, L2 정규화, `dimensions` 지원 |
| `GET /v1/models` | 추가로 `acvram` 블록 제공: 형식, 장치, KV 캐시 용량 |
| `GET /health`, `GET /metrics` | 디코드 처리량, KV 블록 점유율 |

이 응답들의 필드 이름은 영어로 유지됩니다. OpenAI 프로토콜이기 때문이며, 번역하면 기존의 모든 클라이언트가 동작하지 않게 됩니다.

---

<a id="chiffres"></a>

## 수치의 출처

위에 인용한 모든 값은 이 저장소의 코드에서 나오며 `pytest`로 검증됩니다. 참조 커널을 사용해 CPU에서 측정한 값입니다:

| 형식 | 비트/가중치 | 가중치 SNR | BF16 대비 로짓 코사인 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

이 측정에서 얻은 두 가지 발견이 기본값을 바꾸었습니다:

* **Hadamard 회전은 INT4에는 도움이 되지만 NVFP4에는 그렇지 않습니다.** INT4의 128개 단위 그룹은 고립된 이상치 채널을 흡수하지 못하므로, 극단값을 퍼뜨리는 것은 활성화마다 n log n 변환을 치를 만한 가치가 있습니다. NVFP4의 16개 단위 블록은 이미 자체 스케일을 지닙니다. 그래서 `--hadamard auto`는 INT4에만 이를 적용합니다.
* **KV 캐시에서는 INT8이 FP8 E4M3를 이깁니다.** 같은 크기에서 44 dB 대 32 dB인데, (토큰, 헤드)별 스케일이 FP8이 지수 비트를 들여 얻는 동적 범위를 이미 제공하기 때문입니다. 따라서 5090이 FP8을 처리할 수 있음에도 두 카드 모두 INT8 KV 캐시를 사용합니다. `k8v4` 형식(값은 INT4, 캐시 바이트 −22%)이 옵션으로 있지만 **검증되지 않았습니다** — `docs/ARCHITECTURE.md` 참조.

---

<a id="documentation"></a>

## 문서

| 문서 | 내용 |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **다른 머신에서 프로젝트 이어받기** (프랑스어) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | 구성 요소들이 맞물리는 방식 |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | 순수 NVFP4 또는 채널별 int8의 attention+GDN, Gated DeltaNet 하이브리드에서 |
| [`docs/MATERIEL.md`](MATERIEL.md) | 이 특정 머신의 튜닝 |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **아직 끝나지 않은 것**, 가장 먼저 읽을 것 |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | 코드 작업 규칙 (언어, 스타일, 푸시 전 점검) |

---

<a id="resultats"></a>

## 측정 결과 (2026-09-22, RTX 5090, 400 W, 에너지 계측기 기준 ≥ 20 s 구간)

NVFP4(전문가) + INT8(어텐션, 헤드)로 변환한 Qwen3-Coder-30B-A3B, 모든 엔진에 동일한 프로토콜(`outils/`, 카드 한 장, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| 디코드, 12 시퀀스 | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| 디코드, 1 시퀀스 | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| 프리필 pp2048 | **22,707 토큰/s** | 21,054 | 8,671 (TabbyAPI, 퇴역) |

¹ 2026-09-22 정오표: `serve`는 기본적으로 추측을 사용하지만(`--speculative ngram`, cli.py) 경쟁 엔진들은 그렇지 않습니다. 지금까지 공개된 380.8 t/s는 추측을 켠 상태로 측정한 값이었습니다. 추측 없이(`--speculative none`, 동일 체인, revue/poste2-piece44-speculation-none-22-09.md) 측정하면 283.6 t/s — acvram은 b=1에서 llama.cpp와 vLLM에 이어 **세 번째**입니다. 에너지에서는 여전히 llama.cpp보다 앞섭니다(순 0.601 대 0.700 J/토큰). b=12에서는 추측이 결코 활성화되지 않으므로(가드 `lot_max=2`) 이 셀은 이미 동등한 조건이었습니다.

² 2026-09-23, 같은 세션, 같은 HTTP 클라이언트(`acvram serve`와 `vllm serve`를 상대로 `banc-llamacpp-16-09.py`), 각 비교군 앞뒤로 `-lgc 2700`을 명시적으로 설정, 셀은 A V V A 순으로 교대, 비교군마다 ≥ 5개 배치, 2σ를 넘을 때만 차이로 보고(revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38(디코드 시 w13, 펼친 어텐션 리덕션): 차이 −1.6%, **2σ 미만: 처리량 동등**. J/토큰에서는 **vLLM이 7.0% 앞섭니다**(2σ 초과). 0.6.37에서는 같은 프로토콜로 −4.7%였습니다.

³ ²와 같은 세션과 프로토콜, 양쪽 모두 추측 없음: acvram 312.3 대 vLLM 284.8 — **처리량에서 acvram이 9.7% 앞섭니다**(2σ 초과). J/토큰: **동등**(차이 0.04%, 2σ 미만).

⁴ 2026-09-23, llama.cpp를 상대로 같은 프로토콜(revue/poste2-piece72-llamacpp-b1-23-09.md), 라우터를 다시 작성한 acvram 0.6.37(+5.6%): acvram 310.8 대 llama.cpp 329.9 t/s — **처리량에서는 llama.cpp가 5.8% 앞서고, J/토큰에서는 acvram이 13.4% 앞섭니다**(0.598 대 0.691).

오늘의 처리량 수치입니다(스테이션 1030, 절전 모드 `-lgc 2700`, 파이프라인 운용 중; CUDA 그래프에 캡처된 그리디 샘플링, 0.6.35부터 기본값). b=12 acvram 수치는 공식 봉인 셀입니다(교차 배치한 6개 윈도의 중앙값, 윈도마다 클록 판독).

> **정오표 (2026-09-23).** 지금까지 공개된 vLLM 비교(b=12: 1,782 대 1,634 t/s; b=1: 290.6)는 HTTP로 측정한 acvram과 **오프라인**(`LLM().generate()`)으로 측정한 vLLM을 맞붙인 것이었으며, 2026-09-22 정오표는 vLLM 셀이 `vllm serve`를 거쳤다고 잘못 주장했습니다. 2026-09-23: 양쪽 모두 같은 HTTP 클라이언트를 쓰고, 양쪽 모두 `-lgc`를 설정했습니다(acvram은 시작 시 스스로 설정하지만 `vllm serve`는 그렇지 않습니다. 이 예방 조치가 없으면 vLLM은 ~2,650 MHz가 아닌 ~2,930 MHz로 동작했습니다). 결과는 주석 ²: b=12에서 vLLM이 9.1% 앞섭니다.

2026-09-14 아침, acvram은 같은 셀에서 630 t/s와 0.619 J/토큰이었습니다. 향상은 Blackwell의 네이티브 FP4 MMA(`mma.sync … kind::mxf4nvf4`, bf16 대비 ×7.9), 배치 버킷별 grouped-GEMM MoE, 단일 커널 라우팅(스텝당 실행 횟수 3,677 → 1,517), 프로젝션용 좁은 텐서 코어 GEMM에서 나왔습니다. 모든 수치에는 `acvram-memoire/revue/`에 해당 노트가 있으며, 측정 전에 봉인한 예측, 계측 도구, 그 측정 조건이 담겨 있습니다 — 측정 조건이 없는 수치는 공개하지 않습니다.

acvram이 앞서는 영역: sm_120 네이티브 NVFP4의 MLA 모델(GLM-4.7-Flash) — vLLM은 이를 FP8로만 서빙합니다(b=1: 운용 중 165.35 t/s) — 그리고 VRAM에 들어가지 않는 모델. 단일 시퀀스 디코드는 여기에 속하지 않습니다. 추측 없이 acvram은 이 영역에서 vLLM보다 9.7% 앞서고(주석 ³), llama.cpp보다 처리량은 5.8% 뒤처지지만 에너지는 13.4% 앞섭니다(주석 ⁴). 대형 배치에서, VRAM에 들어가는 MoE 모델의 경우 vLLM은 b=12에서 처리량이 동등하지만(1,995.1 대 2,027.0 t/s, 2σ 미만, 주석 ²) J/토큰에서 7.0% 우위를 유지합니다. acvram은 이 영역에서 1,540 t/s(0.6.34)에서 1,995(0.6.38)로 향상되었습니다.

---

<a id="etat"></a>

## 현황

버전 0.6.38. 모든 것이 5090에서 동작합니다: `sm_120a`(네이티브 FP4)와 `sm_86`용으로 컴파일된 CUDA 커널, CUDA 그래프, NVFP4/INT8/INT4 양자화, HTTP 서버. 안전장치도 갖추었습니다. 카드는 작업 세션에 보이지 않으며(`CUDA_VISIBLE_DEVICES` 비어 있음) 오직 `outils/carte.sh`만이 잠금 아래에서 한 번에 한 측정에만 카드를 빌려줍니다. 감시자는 잠금 밖의 모든 접근을 기록하고, 둘 이상의 카드에 걸치거나 10 s 미만인 에너지 측정은 무효 처리되며, 성능 저하 상태로 로드된 모델은 그 사실을 알리고 대결에 참여하지 않습니다.

테스트 4,107개(`pytest --collect-only -q`, CPU에서 일 분; GPU 테스트는 `carte.sh` 아래에서만 실행). 작업 기록: `acvram-memoire/`(규칙, 명부, 작업 수첩, 수백 건의 리뷰 노트).

---

<a id="credits"></a>

## 크레딧

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, Apache-2.0 라이선스: `acvram/kernels/marlin_port/`는 그 Marlin 커널(MoE 및 dense)을 이식했으며, 파일별 전체 출처 표기는 [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE)에 있습니다.
- **NVIDIA** — CUDA, Blackwell의 FP4 텐서 코어(`sm_120`), 그리고 이 프로젝트가 의존하는 라이브러리들.
- **PyTorch** — 텐서 엔진과 C++/CUDA 확장.

독립 프로젝트이며, ASUS, NVIDIA, vLLM 프로젝트와 제휴 관계가 없습니다.

---

<a id="licence"></a>

## 라이선스

이 저장소의 코드는 [GPL-3.0-or-later](../LICENSE)로 배포됩니다. `acvram/kernels/marlin_port/`에는 [vLLM](https://github.com/vllm-project/vllm) v0.29.0에서 이식한 코드(`marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size` 커널)가 Apache-2.0 라이선스로 포함되어 있습니다. 각 파일은 원래의 헤더를 유지하고, 라이선스 전문은 `LICENSE-vllm`에, 파일 목록·원본 커밋·수정 사항은 [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE)에 있습니다.

---

<a id="soutien"></a>

## 프로젝트 후원

acvram은 개인 하드웨어로 개발됩니다. 이 프로젝트가 유용하다면:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

번역: [TRADUIRE.md](TRADUIRE.md) (프랑스어이며, 프로젝트의 기여 가이드는 아직 번역되지 않았습니다).
