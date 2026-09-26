<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-apoiar-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Um gateway de inferência compatível com a API OpenAI, que trata a memória como uma hierarquia, dá a cada GPU o formato numérico que seu silício lê melhor, e otimiza cada token em joules tanto quanto em segundos.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · **🇵🇹 Português** · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Comparativo de desempenho e energia contra vLLM e llama.cpp" width="720"></p>

---

## Sumário

- [As duas ideias](#idees)
- [Início rápido](#demarrage)
- [Instalação](#installer)
- [O que `acvram plan` diz](#plan)
- [Ir mais rápido](#optimisations)
- [Pontos de entrada HTTP](#http)
- [De onde vêm os números](#chiffres)
- [Documentação](#documentation)
- [Resultados medidos](#resultats)
- [Estado](#etat)
- [Créditos](#credits)
- [Licença](#licence)
- [Apoiar o projeto](#soutien)

---

<a id="idees"></a>

## As duas ideias

Concebida para uma máquina precisa:

| | |
|---|---|
| Processador | Intel Core i9-14900K (8 núcleos P + 16 núcleos E) |
| Placa-mãe | ASUS ROG Maximus Z790 Dark Hero |
| Memória | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13) ; as duas placas em PCIe x8/x8, limitadas a 400 W / 275 W |

**Um formato por GPU.** A RTX 5090 possui tensor cores FP4 ; a RTX 3080 Ti não tem, e também não tem FP8. Alinhar as duas num formato comum desperdiçaria a 5090. O conversor escreve portanto *duas vezes o mesmo modelo*, no formato que cada destino sabe realmente explorar :

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesos | **NVFP4** — E2M1 + escala FP8 E4M3 a cada 16 | **INT4** — uint4 + escala e zero fp16 a cada 128 |
| bits por peso | 4,50 | 4,16 |
| face ao BF16 | ×3,56 menor | ×3,85 menor |
| modo de cálculo | tensor cores FP4 | desquantizado em FP16 no núcleo, tensor cores FP16 |
| cache KV | INT8 | INT8 |

32 Go de VRAM a 4,5 bits por peso contêm aproximadamente **56 mil milhões de parâmetros**, contra 16 mil milhões em BF16. Nas duas placas, isso dá aproximadamente **78 mil milhões de parâmetros residentes** antes mesmo de tocar na memória RAM.

**A memória é uma hierarquia, não um muro.** Três andares, e o planejador mede o que cada um custa em vez de esperar que o modelo caiba :

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Início rápido

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Qualquer cliente OpenAI se conecta depois :

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

---

<a id="installer"></a>

## Instalação

A partir do código-fonte (todas as plataformas):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Ou por pacote, um arquivo anexado a cada [release do GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Canal | Arquivo anexado à release | Comando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nomes gerados pelo `rpmbuild`, não fixos) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (ou `rpmbuild --rebuild *.src.rpm` a partir do `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

Antes de instalar, verifique o ficheiro transferido em relação às somas anexadas à release (`SHA256SUMS`, publicada assim que todos os outros ficheiros estejam presentes):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

O Pip não é publicado como pacote (sem wheel compilada): `pip install -e '.[dev]'` instala a partir de um clone do código-fonte, como `./install.sh`.

---

<a id="plan"></a>

## O que `acvram plan` diz

O planejador merece ser executado antes de qualquer download. Ele responde às perguntas que decidem se um modelo é utilizável nesta máquina :

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

Ele explora o espaço das configurações em vez de reter a primeira que cabe, e duas de suas decisões são bastante contraintuitivas para merecer ser enunciadas :

* **Deixa a 3080 Ti sem uso** quando um modelo cabe apenas na 5090. As fatias de um pipeline são executadas em série : adicionar uma etapa a 912 Go/s num pipeline a 1790 Go/s desacelera a decodificação de fluxo único. Força-se com `--gpus all`.
* **Reduz o cache KV para manter os pesos em VRAM.** Cada gigabyte dado ao cache é um gigabyte de pesos empurrado para o barramento PCIe, e ler um peso pelo PCIe custa aproximadamente trinta vezes o que custa a partir da VRAM. No 70B acima, apenas esse ajuste faz passar de 2,3 para 17,8 tokens/s.

---

<a id="optimisations"></a>

## Ir mais rápido

Quatro otimizações, cada uma verificada por uma prova de equivalência e não apenas por um cronômetro : uma otimização que muda a resposta é um bug.

Os lineares NVFP4 dos modelos densos passam por padrão pela disposição Marlin (+57 a +90% de vazão em b = 8, TTFT +2 a +4 ms conforme revue/poste6-piece147-verdict-24-09.md ; fallback `ACVRAM_PROJ_MARLIN=0`, ver [CHANGELOG.md](../CHANGELOG.md)).

### Decodificação especulativa (`--speculative`)

Decodificar um token com um lote de tamanho 1 é limitado pela memória : a máquina lê todos os pesos ativos para produzir um único token. Verificar K tokens propostos lê esses mesmos pesos **uma única vez**. Dois propositores :

* `ngram` (padrão) — busca o sufixo atual mais adiante no contexto e propõe o que seguia. Não custa nada, não requer nenhum modelo. Rentável quando a saída copia a entrada : edição de código, RAG, resumo.
* `draft` — um pequeno modelo em um segundo dispositivo. Neste rig, esse dispositivo é a RTX 3080 Ti, que o planejador deixa voluntariamente ociosa para qualquer modelo que caiba na 5090.

`mtp` (cabeça `nextn` do modelo) e `auto` também existem ; não rentáveis no estado atual e não ativados por padrão — ver `docs/ARCHITECTURE.md`.

A aceitação é exata, não aproximada : uma proposta é aceita com probabilidade `min(1, p/q)` e uma rejeição reamostra na parte positiva normalizada de `p - q`. Medido em 40 000 sorteios contra um rascunho deliberadamente mal calibrado, a distribuição emitida permanece a 0,002 de variação total do alvo — a especulação compra velocidade, nunca uma resposta diferente.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache de prefixo (ativo por padrão)

Os blocos são endereçados pelo hash *encadeado* de sua fatia de tokens : duas requisições que compartilham uma instrução de sistema compartilham seus blocos, e a segunda não precisa mais pré-calculá-los. O encadeamento é indispensável : os mesmos dezesseis tokens em um contexto diferente não contêm as mesmas chaves e valores, e fazer hash apenas da fatia serviria o cache de uma sequência a outra.

Um bloco liberado cujo conteúdo permanece identificável entra numa fila LRU em vez da lista de blocos livres : o cache sobrevive assim entre as requisições sem nunca recusar uma alocação que poderia ter servido.

### Cálculo do andar do host (`--host-exec`)

Uma camada cujos pesos residem em RAM pode ser copiada para a GPU ou calculada no lugar. Os dois caminhos são limitados pela memória e leem os mesmos bytes : o mais rápido é aquele cujo barramento é mais largo — o PCIe 5.0 x16 dá aproximadamente 54 Go/s, a DDR5 em canal duplo aproximadamente 70 Go/s — e calcular no lugar deixa além disso a GPU livre em vez de fazê-la esperar uma cópia.

Isso só vale se o processador ler diretamente os pesos empacotados em 4 bits. Daí um pequeno núcleo C++ com um caminho AVX2 (`acvram_cpu.cpp`, carregado por ctypes, sem cabeçalhos Python nem ninja). Mesmo em seu ramo **escalar** de fallback, ele supera `dequantize() @ x` por um fator de 1,44 em INT4 e 3,21 em NVFP4, porque este último escreve primeiro uma cópia de 32 bits de toda a matriz.

No Mistral-Large-123B, a estimativa do planejador passa de 1,35 para 2,42 tokens/s.

### Precisão mista (`--snr-floor`, desligada por padrão)

O conversor mede a relação sinal/ruído na saída de cada camada para cada tensor e pode promover para um formato mais amplo aqueles que caem abaixo de `--snr-floor`, no limite de 15% dos tensores e de um preço máximo (`--promotion-cout-max`, em mebibytes adicionados).

O piso vale **zero por padrão** : nada é promovido. A decodificação é limitada pela largura de banda de memória, e a medição em `Huihui-Qwen3.8-27B` decide — um piso de 25 dB custa 13,4% de memória e 10,6% de vazão (18,50 Gio e 41,8 t/s contra 16,02 e 46,2) por 2,0% de perplexidade (42,591 contra 43,447, corpus de 16 383 tokens). `--snr-floor 25` restaura o antigo comportamento quando a qualidade importa mais que a velocidade.

### E `acvram eval`

A relação sinal/ruído e o cosseno dos logits são aproximações. `acvram eval REP [REP ...]` mede a perplexidade por janela deslizante, para que uma escolha de formato se decida com provas :

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Pontos de entrada HTTP

| ponto de entrada | notas |
|---|---|
| `POST /v1/chat/completions` | fluxo SSE ou resposta única ; usa o template de conversa do modelo |
| `POST /v1/completions` | instrução em texto ou em identificadores de tokens |
| `POST /v1/embeddings` | estados ocultos finais em média, normalizados L2, `dimensions` respeitado |
| `GET /v1/models` | mais um bloco `acvram` : formatos, dispositivos, capacidade do cache KV |
| `GET /health`, `GET /metrics` | vazão de decodificação, ocupação dos blocos KV |

Os nomes de campos dessas respostas permanecem em inglês : é o protocolo OpenAI, e traduzi-los romperia todos os clientes existentes.

---

<a id="chiffres"></a>

## De onde vêm os números

Cada valor citado acima é produzido por código deste repositório e verificado por `pytest`. Medições feitas em processador com os núcleos de referência :

| formato | bits/peso | SNR dos pesos | cosseno dos logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Duas constatações dessas medições mudaram os valores padrão :

* **Uma rotação de Hadamard ajuda o INT4 e não o NVFP4.** Os grupos de 128 do INT4 não conseguem absorver um canal aberrante isolado, de modo que espalhar os valores extremos vale uma transformada em n log n por ativação. Os blocos de 16 do NVFP4 já trazem sua própria escala. Daí `--hadamard auto`, que só a aplica ao INT4.
* **O INT8 vence o FP8 E4M3 para o cache KV**, 44 dB contra 32 dB em tamanho idêntico, porque uma escala por (token, cabeça) já fornece a faixa dinâmica para a qual o FP8 gasta bits de expoente. As duas placas usam portanto um cache KV em INT8, mesmo que a 5090 soubesse fazer FP8. Um formato `k8v4` (valores em INT4, −22% de bytes de cache) existe como opção, **não qualificado** — ver `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentação

| Documento | Conteúdo |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **retomar o projeto em outra máquina** (French) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | como as peças se encaixam |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 puro ou attention+GDN em int8 por canal, num híbrido Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | ajustar esta máquina precisa |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **o que ainda não está feito**, leia isso primeiro |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | convenções de trabalho no código (idioma, estilo, verificações antes de enviar) |

---

<a id="resultats"></a>

## Resultados medidos (22/09/2026, RTX 5090 a 400 W, regime ≥ 20 s no medidor de energia)

Qwen3-Coder-30B-A3B em NVFP4 (especialistas) + INT8 (atenção, cabeça), mesmo protocolo para todos os motores (`outils/`, uma placa, `energie.py`) :

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decodificação 12 sequências | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| decodificação 1 sequência | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, removido) |

¹ Errata de 22/09 : `serve` especula por padrão (`--speculative ngram`, cli.py), os concorrentes não ; o 380,8 t/s publicado até aqui foi medido COM especulação. Sem especulação (`--speculative none`, mesma cadeia, revue/poste2-piece44-speculation-none-22-09.md) : 283,6 t/s — acvram é **terceiro** em b=1, atrás de llama.cpp e vLLM. Em energia permanece na frente de llama.cpp (0,601 contra 0,700 J/token líquido). Em b=12 a especulação nunca está ativa (guarda `lot_max=2`) : essa célula já estava em armas iguais.

² 23/09, mesma sessão, mesmo cliente HTTP (`banc-llamacpp-16-09.py` contra `acvram serve` e `vllm serve`), `-lgc 2700` posto explicitamente em torno de cada braço, células alternadas A V V A, ≥ 5 lotes por braço, diferença declarada apenas além de 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 na decodificação, redução de atenção desenrolada) : diferença −1,6% , **sob 2 σ : igualdade de vazão**. Em J/token, **vLLM permanece na frente por 7,0%** (além de 2 σ). Com 0.6.37 o mesmo protocolo dava −4,7%.

³ Mesma sessão e mesmo protocolo que ², sem especulação de ambos os lados : acvram 312,3 contra vLLM 284,8 — **acvram na frente por 9,7% em vazão** (além de 2 σ) ; J/token : **igualdade** (diferença 0,04%, sob 2 σ).

⁴ 23/09, mesmo protocolo contra llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 com o roteamento reescrito (+5,6%) : acvram 310,8 contra llama.cpp 329,9 t/s — **llama.cpp na frente por 5,8% em vazão, acvram na frente por 13,4% em J/token** (0,598 contra 0,691).

Vazões do dia (posto 1030, regime econômico `-lgc 2700`, pipeline em serviço ; amostragem gulosa capturada no grafo CUDA, padrão do 0.6.35). O b=12 acvram é uma célula oficial selada (mediana de 6 janelas intercaladas, clock por janela).

> **Errata (23/09/2026).** O comparativo vLLM publicado até aqui (b=12 : 1 782 contra 1 634 t/s ; b=1 : 290,6) opunha acvram medido em HTTP a vLLM medido **offline** (`LLM().generate()`), e a errata de 22/09 afirmava erroneamente que a célula vLLM passava por `vllm serve`. Em 23/09 : mesmo cliente HTTP para os dois, e `-lgc` posto para os dois (acvram põe o seu ao iniciar, `vllm serve` não : sem essa precaução vLLM rodava a ~2 930 MHz contra ~2 650). Resultado na nota ² : vLLM na frente por 9,1% em b=12.

Em 14/09 de manhã acvram estava a 630 t/s e 0,619 J/token na mesma célula : os ganhos vêm da MMA FP4 nativa da Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 sobre o bf16), do MoE em GEMM agrupada por godê de lote, de um roteamento em um único núcleo (3 677 → 1 517 lançamentos por passo) e de um GEMM estreito em tensor cores para as projeções. Cada número tem sua nota em `acvram-memoire/revue/` com a previsão selada antes da medição, o instrumento e seu regime — um número sem regime não é publicado.

Onde acvram está na frente : modelos MLA (GLM-4.7-Flash) em NVFP4 nativo sm_120, que vLLM só serve em FP8 (b=1 : 165,35 t/s em serviço) ; os modelos que não cabem em VRAM. A decodificação de sequência única não faz parte disso : sem especulação, acvram está na frente do vLLM por 9,7% (nota ³), atrás do llama.cpp por 5,8% em vazão mas na frente dele por 13,4% em energia (nota ⁴). Em lote grande, num MoE que cabe em VRAM, vLLM está em igualdade de vazão em b=12 (1 995,1 contra 2 027,0 t/s, sob 2 σ, nota ²) mas mantém 7,0% menos de J/token ; acvram progrediu de 1 540 t/s (0.6.34) para 1 995 (0.6.38).

---

<a id="etat"></a>

## Estado

Versão 0.6.38. Tudo roda na 5090 : núcleos CUDA compilados para `sm_120a` (FP4 nativo) e `sm_86`, grafos CUDA, quantização NVFP4/INT8/INT4, servidor HTTP. Salvaguardas em vigor : a placa é invisível às sessões de trabalho (`CUDA_VISIBLE_DEVICES` vazio) e apenas `outils/carte.sh` a empresta, sob trava, a uma medição por vez ; um vigia registra todo acesso fora da trava ; uma medição de energia cobrindo mais de uma placa ou menos de 10 s é invalidada ; um modelo carregado em regime degradado o diz e não entra num duelo.

4 107 testes (`pytest --collect-only -q`, um minuto em processador ; os testes GPU só rodam sob `carte.sh`). Acompanhamento do trabalho : `acvram-memoire/` (regras, diretório, cadernos, revisão de várias centenas de notas).

---

<a id="credits"></a>

## Créditos

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, sob licença Apache-2.0 : `acvram/kernels/marlin_port/` traz seus núcleos Marlin (MoE e denso), com atribuição completa arquivo por arquivo em [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, os tensor cores FP4 da Blackwell (`sm_120`) e as bibliotecas das quais este projeto depende.
- **PyTorch** — motor tensorial e extensões C++/CUDA.

Projeto independente, não afiliado à ASUS, NVIDIA nem ao projeto vLLM.

---

<a id="licence"></a>

## Licença

[GPL-3.0 ou posterior](../LICENSE) para o código deste repositório. `acvram/kernels/marlin_port/` contém código portado do [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (núcleos `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), sob licença Apache-2.0 : cada arquivo mantém seu cabeçalho original, a licença está em `LICENSE-vllm` e a lista de arquivos, o commit de origem e as modificações estão em [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Apoiar o projeto

O desenvolvimento do acvram é feito em hardware pessoal. Se o projeto lhe é útil :

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Oferecer%20um%20café&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Traduções : [TRADUIRE.md](TRADUIRE.md) (French; the project's contribution guide is not yet translated).
