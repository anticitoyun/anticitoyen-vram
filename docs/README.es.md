<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-apoyar-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Una pasarela de inferencia compatible con la API de OpenAI, que trata la memoria como una jerarquía, da a cada GPU el formato numérico que su silicio sabe leer mejor, y optimiza cada token tanto en julios como en segundos.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · **🇪🇸 Español** · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Comparativa de rendimiento y energía frente a vLLM y llama.cpp" width="720"></p>

---

## Índice

- [Las dos ideas](#idees)
- [Inicio rápido](#demarrage)
- [Instalación](#installer)
- [Lo que dice `acvram plan`](#plan)
- [Ir rápido](#optimisations)
- [Puntos de entrada HTTP](#http)
- [De dónde vienen las cifras](#chiffres)
- [Documentación](#documentation)
- [Resultados medidos](#resultats)
- [Estado](#etat)
- [Créditos](#credits)
- [Licencia](#licence)
- [Apoyar el proyecto](#soutien)

---

<a id="idees"></a>

## Las dos ideas

Diseñada para una máquina concreta:

| | |
|---|---|
| Procesador | Intel Core i9-14900K (8 núcleos P + 16 núcleos E) |
| Placa base | ASUS ROG Maximus Z790 Dark Hero |
| Memoria | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13); ambas tarjetas en PCIe x8/x8, limitadas a 400 W / 275 W |

**Un formato por GPU.** La RTX 5090 tiene tensor cores FP4; la RTX 3080 Ti no los tiene, ni tampoco FP8. Alinear ambas en un formato común desperdiciaría la 5090. Por eso el conversor escribe *el mismo modelo dos veces*, en el formato que cada destino puede realmente explotar:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesos | **NVFP4** — E2M1 + escala FP8 E4M3 cada 16 | **INT4** — uint4 + escala y punto cero fp16 cada 128 |
| bits por peso | 4,50 | 4,16 |
| frente a BF16 | ×3,56 más pequeño | ×3,85 más pequeño |
| modo de cálculo | tensor cores FP4 | desquantizado a FP16 dentro del núcleo, tensor cores FP16 |
| caché KV | INT8 | INT8 |

32 GB de VRAM a 4,5 bits por peso contienen aproximadamente **56 mil millones de parámetros**, frente a 16 mil millones en BF16. Entre ambas tarjetas, eso da aproximadamente **78 mil millones de parámetros residentes** antes de tocar siquiera la memoria del sistema.

**La memoria es una jerarquía, no un muro.** Tres niveles, y el planificador mide lo que cuesta cada uno en lugar de esperar que el modelo quepa:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Inicio rápido

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Cualquier cliente OpenAI se conecta de inmediato:

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

## Instalación

Desde el código fuente (todas las plataformas):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

O por paquete, un archivo adjunto a cada [versión de GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Canal | Archivo adjunto a la versión | Comando |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nombres generados por `rpmbuild`, no fijos) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (o `rpmbuild --rebuild *.src.rpm` desde el `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpak` | `flatpak install acvram-<version>.flatpak` |

Pip no se publica como paquete (no hay wheel construida): `pip install -e '.[dev]'` instala desde un clon del código fuente, igual que `./install.sh`.

---

<a id="plan"></a>

## Lo que dice `acvram plan`

El planificador merece ejecutarse antes de cualquier descarga. Responde a las preguntas que decidirán si un modelo es siquiera utilizable en esta máquina:

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

Explora el espacio de configuraciones en lugar de detenerse en la primera que cabe, y dos de sus decisiones son lo bastante contraintuitivas como para merecer explicación:

* **Deja la 3080 Ti sin usar** cuando un modelo cabe en la sola 5090. Las etapas de una tubería se ejecutan en serie: añadir una etapa a 912 GB/s en una tubería a 1790 GB/s ralentiza la decodificación de un solo flujo. Se fuerza con `--gpus all`.
* **Reduce la caché KV para mantener los pesos en VRAM.** Cada gigabyte dado a la caché es un gigabyte de pesos empujado al bus PCIe, y leer un peso por PCIe cuesta aproximadamente treinta veces lo que cuesta desde la VRAM. En el 70B de arriba, esta única decisión hace pasar de 2,3 a 17,8 tokens/s.

---

<a id="optimisations"></a>

## Ir rápido

Cuatro optimizaciones, cada una verificada por una prueba de equivalencia y no solo por un cronómetro: una optimización que cambia la respuesta es un error.

Los lineales NVFP4 de los modelos densos pasan por defecto por la disposición Marlin (+57 a +90 % de rendimiento a b = 8, TTFT +2 a +4 ms según revue/poste6-piece147-verdict-24-09.md; repliegue `ACVRAM_PROJ_MARLIN=0`, ver [CHANGELOG.md](../CHANGELOG.md)).

### Decodificación especulativa (`--speculative`)

Decodificar un token con un lote de tamaño 1 está limitado por la memoria: la máquina lee todos los pesos activos para producir un solo token. Verificar K tokens propuestos lee esos mismos pesos **una sola vez**. Dos proponentes:

* `ngram` (por defecto) — busca el sufijo actual más atrás en el contexto y propone lo que le seguía. No cuesta nada, no necesita ningún modelo. Rentable cuando la salida repite la entrada: edición de código, RAG, resumen.
* `draft` — un modelo pequeño en un segundo dispositivo. En este equipo, ese dispositivo es la RTX 3080 Ti, que el planificador deja voluntariamente ociosa para todo modelo que cabe en la 5090.

`mtp` (cabeza `nextn` del modelo) y `auto` también existen; no rentables tal cual y no activados por defecto — ver `docs/ARCHITECTURE.md`.

La aceptación es exacta, no aproximada: una propuesta se acepta con probabilidad `min(1, p/q)`, y un rechazo remuestrea en la parte positiva normalizada de `p - q`. Medido sobre 40 000 sorteos frente a un borrador deliberadamente mal calibrado, la distribución emitida se mantiene a 0,002 de variación total del objetivo — la especulación compra velocidad, nunca una respuesta diferente.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Caché de prefijo (activa por defecto)

Los bloques se direccionan por el hash *encadenado* de su tramo de tokens: dos solicitudes que comparten una instrucción de sistema comparten sus bloques, y la segunda ya no tiene que precalcularlos. El encadenamiento es indispensable: los mismos dieciséis tokens en un contexto diferente no contienen las mismas claves y valores, y hashear solo el tramo serviría la caché de una secuencia a otra.

Un bloque liberado cuyo contenido sigue siendo identificable se une a una cola LRU en lugar de la lista de bloques libres: la caché sobrevive así entre solicitudes sin negar nunca una asignación que podría haber servido.

### Cómputo en el nivel del host (`--host-exec`)

Una capa cuyos pesos residen en RAM puede copiarse a la GPU o calcularse en su lugar. Ambos caminos están limitados por la memoria y leen los mismos bytes: el más rápido es aquel cuyo bus es más ancho — el PCIe 5.0 x16 da aproximadamente 54 GB/s, la DDR5 de doble canal aproximadamente 70 GB/s — y calcular en su lugar además deja la GPU libre en lugar de hacerla esperar una copia.

Esto solo vale la pena si el procesador lee directamente los pesos empaquetados en 4 bits. De ahí un pequeño núcleo C++ con una vía AVX2 (`acvram_cpu.cpp`, cargado por ctypes, sin cabeceras Python ni ninja). Incluso en su rama **escalar** de repliegue, supera a `dequantize() @ x` por un factor de 1,44 en INT4 y 3,21 en NVFP4, porque este último escribe primero una copia de 32 bits de toda la matriz.

En Mistral-Large-123B, la estimación del planificador pasa de 1,35 a 2,42 tokens/s.

### Precisión mixta (`--snr-floor`, desactivada por defecto)

El conversor mide la relación señal/ruido a la salida de capa para cada tensor y puede promover a un formato más amplio aquellos que caen por debajo de `--snr-floor`, dentro de un límite del 15 % de los tensores y un precio tope (`--promotion-cout-max`, en mebibytes añadidos).

El umbral vale **cero por defecto**: nada se promueve. La decodificación está limitada por el ancho de banda de memoria, y la medición sobre `Huihui-Qwen3.8-27B` lo decide — un umbral de 25 dB cuesta 13,4 % de memoria y 10,6 % de rendimiento (18,50 GiB y 41,8 t/s frente a 16,02 y 46,2) por 2,0 % de perplejidad (42,591 frente a 43,447, corpus de 16 383 tokens). `--snr-floor 25` restablece el comportamiento anterior cuando la calidad importa más que la velocidad.

### Y `acvram eval`

La relación señal/ruido y el coseno de los logits son aproximaciones. `acvram eval REP [REP ...]` mide la perplejidad por ventana deslizante, para que una elección de formato se decida con pruebas:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Puntos de entrada HTTP

| punto de entrada | notas |
|---|---|
| `POST /v1/chat/completions` | flujo SSE o respuesta única; usa la plantilla de conversación del modelo |
| `POST /v1/completions` | prompt en texto o en identificadores de tokens |
| `POST /v1/embeddings` | estados ocultos finales promediados, normalizados L2, `dimensions` respetado |
| `GET /v1/models` | más un bloque `acvram`: formatos, dispositivos, capacidad de la caché KV |
| `GET /health`, `GET /metrics` | rendimiento de decodificación, ocupación de bloques KV |

Los nombres de los campos de estas respuestas permanecen en inglés: es el protocolo de OpenAI, y traducirlos rompería todos los clientes existentes.

---

<a id="chiffres"></a>

## De dónde vienen las cifras

Cada valor citado arriba es producido por código de este repositorio y verificado por `pytest`. Mediciones hechas en procesador con los núcleos de referencia:

| formato | bits/peso | SNR de los pesos | coseno de los logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dos hallazgos de estas mediciones cambiaron los valores por defecto:

* **Una rotación de Hadamard ayuda a INT4 y no a NVFP4.** Los grupos de 128 de INT4 no pueden absorber un canal aberrante aislado, así que repartir los valores extremos vale una transformada en n log n por activación. Los bloques de 16 de NVFP4 ya llevan su propia escala. De ahí `--hadamard auto`, que solo la aplica a INT4.
* **INT8 supera a FP8 E4M3 para la caché KV**, 44 dB frente a 32 dB al mismo tamaño, porque una escala por (token, cabeza) ya proporciona el rango dinámico en el que FP8 gasta bits de exponente. Por eso ambas tarjetas usan una caché KV en INT8, aunque la 5090 podría hacer FP8. Un formato `k8v4` (valores en INT4, −22 % de bytes de caché) existe como opción, **sin calificar** — ver `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentación

| Documento | Contenido |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **reanudar el proyecto en otra máquina** (francés) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | cómo se ensamblan las piezas |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 puro o atención+GDN en int8 por canal, sobre un híbrido Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | ajustar esta máquina concreta |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **lo que no está hecho**, léase primero |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | convenciones de trabajo sobre el código (idioma, estilo, comprobaciones antes de subir) |

---

<a id="resultats"></a>

## Resultados medidos (22/09/2026, RTX 5090 a 400 W, régimen ≥ 20 s en el medidor de energía)

Qwen3-Coder-30B-A3B en NVFP4 (expertos) + INT8 (atención, cabeza), mismo protocolo para todos los motores (`outils/`, una tarjeta, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decodificación 12 secuencias | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| decodificación 1 secuencia | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, retirado) |

¹ Fe de erratas del 22/09: `serve` especula por defecto (`--speculative ngram`, cli.py), los competidores no; el 380,8 t/s publicado hasta ahora se midió CON especulación. Sin especulación (`--speculative none`, misma cadena, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram es **tercero** a b=1, tras llama.cpp y vLLM. En energía sigue por delante de llama.cpp (0,601 frente a 0,700 J/token neto). A b=12 la especulación nunca está activa (guarda `lot_max=2`): esa celda ya estaba en igualdad de condiciones.

² 23/09, misma sesión, mismo cliente HTTP (`banc-llamacpp-16-09.py` contra `acvram serve` y `vllm serve`), `-lgc 2700` fijado explícitamente en torno a cada brazo, celdas alternadas A V V A, ≥ 5 lotes por brazo, diferencia declarada solo más allá de 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 en decodificación, reducción de atención desenrollada): diferencia −1,6 %, **por debajo de 2 σ: paridad de rendimiento**. En J/token, **vLLM sigue por delante en un 7,0 %** (más allá de 2 σ). Con 0.6.37 el mismo protocolo daba −4,7 %.

³ Misma sesión y protocolo que ², sin especulación en ambos lados: acvram 312,3 frente a vLLM 284,8 — **acvram por delante en un 9,7 % de rendimiento** (más allá de 2 σ); J/token: **paridad** (diferencia 0,04 %, por debajo de 2 σ).

⁴ 23/09, mismo protocolo frente a llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 con el enrutamiento reescrito (+5,6 %): acvram 310,8 frente a llama.cpp 329,9 t/s — **llama.cpp por delante en un 5,8 % de rendimiento, acvram por delante en un 13,4 % en J/token** (0,598 frente a 0,691).

Cifras de rendimiento del día (puesto 1030, régimen eco `-lgc 2700`, tubería en servicio; muestreo voraz capturado en el grafo CUDA, por defecto desde 0.6.35). La celda b=12 de acvram es una celda oficial sellada (mediana de 6 ventanas intercaladas, reloj por ventana).

> **Fe de erratas (23/09/2026).** La comparativa con vLLM publicada hasta ahora (b=12: 1 782 frente a 1 634 t/s; b=1: 290,6) enfrentaba a acvram medido por HTTP con vLLM medido **fuera de línea** (`LLM().generate()`), y la fe de erratas del 22/09 afirmaba erróneamente que la celda de vLLM pasaba por `vllm serve`. El 23/09: mismo cliente HTTP para ambos, y `-lgc` fijado para ambos (acvram fija el suyo al arrancar, `vllm serve` no: sin esta precaución vLLM funcionaba a ~2 930 MHz frente a ~2 650). Resultado en la nota ²: vLLM por delante en un 9,1 % a b=12.

La mañana del 14/09 acvram estaba a 630 t/s y 0,619 J/token en la misma celda: las ganancias vienen de la MMA FP4 nativa de Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 sobre bf16), del MoE en GEMM agrupada por cubo de lote, de un enrutamiento en un solo núcleo (3 677 → 1 517 lanzamientos por paso) y de un GEMM estrecho sobre tensor cores para las proyecciones. Cada cifra tiene su nota en `acvram-memoire/revue/` con la predicción sellada antes de la medición, el instrumento y su régimen — una cifra sin régimen no se publica.

Donde acvram va por delante: modelos MLA (GLM-4.7-Flash) en NVFP4 nativo sm_120, que vLLM solo sirve en FP8 (b=1: 165,35 t/s en servicio); los modelos que no caben en VRAM. La decodificación de secuencia única no forma parte de ello: sin especulación, acvram va por delante de vLLM ahí en un 9,7 % (nota ³), por detrás de llama.cpp en un 5,8 % de rendimiento pero por delante de él en un 13,4 % en energía (nota ⁴). En lote grande, sobre un MoE que cabe en VRAM, vLLM está en paridad de rendimiento a b=12 (1 995,1 frente a 2 027,0 t/s, por debajo de 2 σ, nota ²) pero mantiene un 7,0 % menos de J/token; acvram progresó ahí de 1 540 t/s (0.6.34) a 1 995 (0.6.38).

---

<a id="etat"></a>

## Estado

Versión 0.6.38. Todo funciona en la 5090: núcleos CUDA compilados para `sm_120a` (FP4 nativo) y `sm_86`, grafos CUDA, cuantización NVFP4/INT8/INT4, servidor HTTP. Salvaguardas en su lugar: la tarjeta es invisible para las sesiones de trabajo (`CUDA_VISIBLE_DEVICES` vacío) y solo `outils/carte.sh` la presta, bajo cerrojo, a una medición a la vez; un vigía registra cualquier acceso fuera del cerrojo; una medición de energía que abarque más de una tarjeta o menos de 10 s se invalida; un modelo cargado en régimen degradado lo dice y no entra en un duelo.

4 107 pruebas (`pytest --collect-only -q`, un minuto en procesador; las pruebas de GPU solo se ejecutan bajo `carte.sh`). Seguimiento del trabajo: `acvram-memoire/` (reglas, directorio, cuadernos, revisión de varios cientos de notas).

---

<a id="credits"></a>

## Créditos

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, bajo licencia Apache-2.0: `acvram/kernels/marlin_port/` porta sus núcleos Marlin (MoE y densos), con atribución completa archivo por archivo en [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, los tensor cores FP4 de Blackwell (`sm_120`) y las bibliotecas de las que depende este proyecto.
- **PyTorch** — motor tensorial y extensiones C++/CUDA.

Proyecto independiente, no afiliado a ASUS, NVIDIA ni al proyecto vLLM.

---

<a id="licence"></a>

## Licencia

[GPL-3.0 o posterior](../LICENSE) para el código de este repositorio. `acvram/kernels/marlin_port/` contiene código portado de [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (núcleos `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), bajo licencia Apache-2.0: cada archivo conserva su cabecera original, el texto de la licencia está en `LICENSE-vllm`, y la lista de archivos, el commit de origen y las modificaciones están en [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Apoyar el proyecto

El desarrollo de acvram se lleva a cabo con hardware personal. Si el proyecto le resulta útil:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Invitame%20un%20cafe&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Traducciones: [TRADUIRE.md](TRADUIRE.md) (francés; la guía de contribución del proyecto aún no está traducida).
