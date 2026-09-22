<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Apoiar: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Um gateway de inferência compatível com a API OpenAI, que trata a memória como
uma hierarquia e dá a cada GPU o formato numérico que o seu silício lê melhor.

Concebido para uma máquina precisa:

| | |
|---|---|
| Processador | Intel Core i9-14900K (8 núcleos P + 16 núcleos E) |
| Placa-mãe | ASUS ROG Maximus Z790 Dark Hero |
| Memória | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13); as duas placas em PCIe x8/x8, limitadas a 400 W / 275 W |

## As duas ideias

**Um formato por GPU.** A RTX 5090 possui tensor cores FP4; a RTX 3080 Ti não
os tem, nem tem FP8. Alinhar as duas num formato comum desperdiçaria a 5090. O
conversor escreve portanto *duas vezes o mesmo modelo*, no formato que cada
destino sabe realmente explorar:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesos | **NVFP4** — E2M1 + escala FP8 E4M3 a cada 16 | **INT4** — uint4 + escala e zero fp16 a cada 128 |
| bits por peso | 4,50 | 4,16 |
| face ao BF16 | ×3,56 mais pequeno | ×3,85 mais pequeno |
| modo de cálculo | tensor cores FP4 | desquantizado em FP16 no kernel, tensor cores FP16 |
| cache KV | INT8 | INT8 |

32 GB de VRAM a 4,5 bits por peso contêm cerca de **56 mil milhões de
parâmetros**, contra 16 mil milhões em BF16. Nas duas placas, isso dá
aproximadamente **78 mil milhões de parâmetros residentes** antes sequer de
tocar na memória RAM.

**A memória é uma hierarquia, não um muro.** Três andares, e o planificador
mede o que cada um custa em vez de esperar que o modelo caiba:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Arranque rápido

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Qualquer cliente OpenAI liga-se depois:

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

## O que diz `acvram plan`

O planificador merece ser lançado antes de qualquer transferência. Responde às
perguntas que decidem se um modelo é utilizável nesta máquina:

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

Explora o espaço das configurações em vez de ficar com a primeira que cabe, e
duas das suas decisões são suficientemente contraintuitivas para merecerem ser
enunciadas:

* **Deixa a 3080 Ti sem uso** quando um modelo cabe só na 5090. As fatias de
  um pipeline executam-se em série: acrescentar um andar a 912 GB/s num
  pipeline a 1790 GB/s abranda a descodificação de fluxo único. Força-se com
  `--gpus all`.
* **Encolhe a cache KV para manter os pesos em VRAM.** Cada gigabyte dado à
  cache é um gigabyte de pesos empurrado para o barramento PCIe, e ler um peso
  pelo PCIe custa cerca de trinta vezes o que custa a partir da VRAM. No 70B
  acima, só esta arbitragem faz passar de 2,3 para 17,8 tokens/s.

## Ir depressa

Quatro otimizações, cada uma verificada por uma prova de equivalência e não
apenas por um cronómetro: uma otimização que muda a resposta é um erro.

### Descodificação especulativa (`--speculative`)

Descodificar um token com um lote de tamanho 1 é limitado pela memória: a
máquina lê todos os pesos ativos para produzir um único token. Verificar K
tokens propostos lê esses mesmos pesos **uma só vez**. Dois proponentes:

* `ngram` (por omissão) — procura o sufixo atual mais atrás no contexto e
  propõe o que se seguia. Não custa nada, não pede nenhum modelo. Rentável
  quando a saída copia a entrada: edição de código, RAG, resumo.
* `draft` — um pequeno modelo num segundo dispositivo. Nesta máquina esse
  dispositivo é a RTX 3080 Ti, que o planificador deixa voluntariamente ociosa
  para qualquer modelo que caiba na 5090.

A aceitação é exata, não aproximada: uma proposta é aceite com probabilidade
`min(1, p/q)` e uma rejeição reamostra na parte positiva normalizada de
`p - q`. Medido em 40 000 tiragens contra um rascunho deliberadamente mal
calibrado, a distribuição emitida fica a 0,002 de variação total do alvo — a
especulação compra velocidade, nunca uma resposta diferente.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache de prefixo (ativa por omissão)

Os blocos são endereçados pelo hash *encadeado* da sua fatia de tokens: dois
pedidos que partilham uma instrução de sistema partilham os seus blocos, e o
segundo já não tem de os pré-calcular. O encadeamento é indispensável: os
mesmos dezasseis tokens num contexto diferente não contêm as mesmas chaves e
valores, e fazer o hash só da fatia serviria a cache de uma sequência a outra.

Um bloco libertado cujo conteúdo continua identificável junta-se a uma fila
LRU em vez da lista de blocos livres: a cache sobrevive assim entre pedidos sem
nunca recusar uma alocação que poderia ter servido.

### Cálculo no andar anfitrião (`--host-exec`)

Uma camada cujos pesos residem em RAM pode ser copiada para a GPU ou calculada
no local. Os dois caminhos são limitados pela memória e leem os mesmos bytes:
o mais rápido é o do barramento mais largo — o PCIe 5.0 x16 dá cerca de
54 GB/s, a DDR5 em canal duplo cerca de 70 GB/s — e calcular no local deixa
além disso a GPU livre em vez de a fazer esperar por uma cópia.

Isto só vale se o processador ler diretamente os pesos empacotados em 4 bits.
Daí um pequeno kernel C++ com um caminho AVX2 (`acvram_cpu.cpp`, carregado por
ctypes, sem cabeçalhos Python nem ninja). Mesmo no seu ramo **escalar** de
recurso, bate `dequantize() @ x` por um fator de 1,44 em INT4 e 3,21 em NVFP4,
porque este último escreve primeiro uma cópia de 32 bits de toda a matriz.

No Mistral-Large-123B, a estimativa do planificador passa de 1,35 para
2,42 tokens/s.

### Precisão mista (`--snr-floor`, desligada por omissão)

O conversor mede a relação sinal/ruído à saída de cada camada para cada tensor
e pode promover para um formato mais largo os que caem abaixo de
`--snr-floor`, no limite de 15 % dos tensores e de um preço máximo
(`--promotion-cout-max`, em mebibytes acrescentados).

O limiar vale **zero por omissão**: nada é promovido. A descodificação é
limitada pela largura de banda de memória, e a medida em `Huihui-Qwen3.8-27B`
decide — um limiar de 25 dB custa 13,4 % de memória e 10,6 % de débito
(18,50 GiB e 41,8 t/s contra 16,02 e 46,2) por 2,0 % de perplexidade (42,591
contra 43,447, corpus de 16 383 tokens). `--snr-floor 25` repõe o antigo
comportamento quando a qualidade prima sobre a velocidade.

### E `acvram eval`

A relação sinal/ruído e o cosseno dos logits são aproximações.
`acvram eval DIR [DIR ...]` mede a perplexidade por janela deslizante, para
que uma escolha de formato se decida com provas:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Pontos de entrada HTTP

| ponto de entrada | notas |
|---|---|
| `POST /v1/chat/completions` | fluxo SSE ou resposta única; usa o modelo de conversa do modelo |
| `POST /v1/completions` | prompt em texto ou em identificadores de tokens |
| `POST /v1/embeddings` | estados ocultos finais em média, normalizados L2, `dimensions` respeitado |
| `GET /v1/models` | mais um bloco `acvram`: formatos, dispositivos, capacidade da cache KV |
| `GET /health`, `GET /metrics` | débito de descodificação, ocupação dos blocos KV |

Os nomes dos campos destas respostas ficam em inglês: é o protocolo OpenAI, e
traduzi-los quebraria todos os clientes existentes.

## De onde vêm os números

Cada valor citado acima é produzido por código deste repositório e verificado
por `pytest`. Medidas feitas em processador com os kernels de referência:

| formato | bits/peso | SNR dos pesos | cosseno dos logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Duas constatações destas medidas mudaram os valores por omissão:

* **Uma rotação de Hadamard ajuda o INT4 e não o NVFP4.** Os grupos de 128 do
  INT4 não conseguem absorver um canal aberrante isolado, pelo que espalhar os
  valores extremos vale uma transformada em n log n por ativação. Os blocos de
  16 do NVFP4 já trazem a sua própria escala. Daí `--hadamard auto`, que só a
  aplica ao INT4.
* **O INT8 bate o FP8 E4M3 para a cache KV**, 44 dB contra 32 dB a tamanho
  idêntico, porque uma escala por (token, cabeça) já fornece a gama dinâmica em
  que o FP8 gasta bits de expoente. As duas placas usam portanto uma cache KV
  em INT8, mesmo que a 5090 soubesse fazer FP8.

## Documentação

* [`REPRISE.md`](../REPRISE.md) — **retomar o projeto noutra máquina**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — como as peças se encaixam
* [`docs/MATERIEL.md`](MATERIEL.md) — afinar esta máquina precisa
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **o que não está feito**, a ler primeiro
* [`CONVENTIONS.md`](../CONVENTIONS.md) — convenções de trabalho no código (língua, estilo, controlos antes de enviar)

## Resultados medidos (22/09/2026, RTX 5090 a 400 W, regime ≥ 20 s no contador de energia)

Qwen3-Coder-30B-A3B em NVFP4 (peritos) + INT8 (atenção, cabeça), mesmo
protocolo para todos os motores (`outils/`, uma placa, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| descodificação 12 sequências | **1 625,5 t/s** | 1 596,1 t/s | — |
| descodificação 1 sequência | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, retirado) |

Débitos do dia (posto 1030, regime eco `-lgc 2700`, pipeline em serviço;
amostragem gulosa capturada no grafo CUDA, predefinição da 0.6.35). O b=12 é uma
célula oficial selada (mediana de 6 janelas intercaladas, relógio por janela). O
valor vLLM 1 596,1 é a referência congelada de 21/09 (vLLM não reexecutado nesse
dia): a diferença +1,84 % vale a referência igual, não como remedição de ambos na
mesma manhã. O J/token a relógio igual contra os três motores continua em
remedição (`outils/gpu/mesure/banc-4moteurs.py`) — um valor sem regime não se
publica.

Na manhã de 14/09 o acvram estava a 630 t/s e 0,619 J/token na mesma célula:
os ganhos vêm da MMA FP4 nativa de Blackwell (`mma.sync … kind::mxf4nvf4`,
×7,9 sobre o bf16), do MoE em GEMM agrupada por balde de lote, de um
encaminhamento num único kernel (3 677 → 1 517 lançamentos por passo) e de uma
GEMM estreita em tensor cores para as projeções. Cada número tem a sua nota em
`acvram-memoire/revue/` com a previsão selada antes da medida, o instrumento e
o seu regime — um número sem regime não é publicado.

Onde o acvram está à frente: modelos MLA (GLM-4.7-Flash) em NVFP4 nativo
sm_120, que o vLLM só serve em FP8 (b=1: 165,35 t/s em serviço); os modelos que
não cabem em VRAM; e, desde a 0.6.35, a descodificação de lote grande de um MoE
que cabe em VRAM — o b=12 passa de 1 540 (0.6.34) a 1 625,5 t/s, ou seja +1,84 %
à frente da referência vLLM congelada (1 596,1). A diferença mantém-se estreita e
a referência congelada; a diferença em energia está por remedir.

## Estado

Versão 0.6.35. Tudo corre na 5090: kernels CUDA compilados para `sm_120a` (FP4
nativo) e `sm_86`, grafos CUDA, quantização NVFP4/INT8/INT4, servidor HTTP.
Salvaguardas no lugar: a placa é invisível às sessões de trabalho
(`CUDA_VISIBLE_DEVICES` vazio) e só `outils/carte.sh` a empresta, sob
fechadura, a uma medida de cada vez; um vigia regista todo o acesso fora da
fechadura; uma medida de energia que cubra mais de uma placa ou menos de 10 s
é invalidada; um modelo carregado em regime degradado di-lo e não entra num
duelo.

640 testes (`pytest -q`, um minuto em processador; os testes GPU só correm
sob `carte.sh`). Acompanhamento do trabalho: `acvram-memoire/` (regras,
diretório, cadernos, revisão de 180 notas).

## Apoiar

O desenvolvimento do acvram é feito em material pessoal. Se o projeto lhe é
útil: **Apoiar: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licença

GPL-3.0 ou posterior.
