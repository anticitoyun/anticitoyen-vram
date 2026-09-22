<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Apoyar: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Una pasarela de inferencia compatible con la API de OpenAI, que trata la
memoria como una jerarquía y da a cada GPU el formato numérico que su silicio
lee mejor.

Diseñada para una máquina concreta:

| | |
|---|---|
| Procesador | Intel Core i9-14900K (8 núcleos P + 16 núcleos E) |
| Placa base | ASUS ROG Maximus Z790 Dark Hero |
| Memoria | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistema | Ubuntu 26.04 LTS (CUDA 13); ambas tarjetas en PCIe x8/x8, limitadas a 400 W / 275 W |

## Las dos ideas

**Un formato por GPU.** La RTX 5090 tiene tensor cores FP4; la RTX 3080 Ti no
los tiene, ni tampoco FP8. Alinear ambas en un formato común desperdiciaría la
5090. El conversor escribe por tanto *dos veces el mismo modelo*, en el formato
que cada destino sabe explotar de verdad:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| pesos | **NVFP4** — E2M1 + escala FP8 E4M3 cada 16 | **INT4** — uint4 + escala y cero fp16 cada 128 |
| bits por peso | 4,50 | 4,16 |
| frente a BF16 | ×3,56 más pequeño | ×3,85 más pequeño |
| modo de cálculo | tensor cores FP4 | descuantizado a FP16 en el núcleo, tensor cores FP16 |
| caché KV | INT8 | INT8 |

32 GB de VRAM a 4,5 bits por peso contienen unos **56 mil millones de
parámetros**, frente a 16 mil millones en BF16. Entre las dos tarjetas, eso
son aproximadamente **78 mil millones de parámetros residentes** antes
siquiera de tocar la memoria RAM.

**La memoria es una jerarquía, no un muro.** Tres niveles, y el planificador
mide lo que cuesta cada uno en lugar de esperar que el modelo quepa:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Inicio rápido

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Cualquier cliente OpenAI se conecta después:

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

## Lo que dice `acvram plan`

Merece la pena lanzar el planificador antes de cualquier descarga. Responde a
las preguntas que deciden si un modelo es utilizable en esta máquina:

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

Explora el espacio de configuraciones en lugar de quedarse con la primera que
cabe, y dos de sus decisiones son lo bastante contraintuitivas como para
enunciarlas:

* **Deja la 3080 Ti sin usar** cuando un modelo cabe solo en la 5090. Los
  tramos de un pipeline se ejecutan en serie: añadir una etapa a 912 GB/s en un
  pipeline a 1790 GB/s ralentiza la decodificación de un solo flujo. Se fuerza
  con `--gpus all`.
* **Reduce la caché KV para mantener los pesos en VRAM.** Cada gigabyte dado a
  la caché es un gigabyte de pesos empujado al bus PCIe, y leer un peso por
  PCIe cuesta unas treinta veces lo que cuesta desde la VRAM. En el 70B de
  arriba, solo este arbitraje pasa de 2,3 a 17,8 tokens/s.

## Ir rápido

Cuatro optimizaciones, cada una verificada por una prueba de equivalencia y
no solo por un cronómetro: una optimización que cambia la respuesta es un
error.

### Decodificación especulativa (`--speculative`)

Decodificar un token con un lote de tamaño 1 está limitado por la memoria: la
máquina lee todos los pesos activos para producir un solo token. Verificar K
tokens propuestos lee esos mismos pesos **una sola vez**. Dos proponentes:

* `ngram` (por defecto) — busca el sufijo actual más atrás en el contexto y
  propone lo que seguía. No cuesta nada, no necesita ningún modelo. Rentable
  cuando la salida copia la entrada: edición de código, RAG, resumen.
* `draft` — un modelo pequeño en un segundo dispositivo. En este equipo ese
  dispositivo es la RTX 3080 Ti, que el planificador deja voluntariamente
  ociosa para todo modelo que cabe en la 5090.

La aceptación es exacta, no aproximada: una propuesta se acepta con
probabilidad `min(1, p/q)` y un rechazo remuestrea en la parte positiva
normalizada de `p - q`. Medido sobre 40 000 extracciones frente a un borrador
deliberadamente mal calibrado, la distribución emitida se queda a 0,002 de
variación total del objetivo — la especulación compra velocidad, nunca una
respuesta distinta.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Caché de prefijo (activa por defecto)

Los bloques se direccionan por el hash *encadenado* de su tramo de tokens: dos
peticiones que comparten una consigna de sistema comparten sus bloques, y la
segunda ya no tiene que precalcularlos. El encadenamiento es indispensable: los
mismos dieciséis tokens en un contexto distinto no contienen las mismas claves
y valores, y hashear solo el tramo serviría la caché de una secuencia a otra.

Un bloque liberado cuyo contenido sigue siendo identificable pasa a una cola
LRU en lugar de a la lista de bloques libres: la caché sobrevive así entre
peticiones sin rechazar nunca una asignación que podría haber servido.

### Cálculo en el nivel anfitrión (`--host-exec`)

Una capa cuyos pesos residen en RAM puede copiarse a la GPU o calcularse in
situ. Ambos caminos están limitados por la memoria y leen los mismos bytes: el
más rápido es el del bus más ancho — el PCIe 5.0 x16 da unos 54 GB/s, la DDR5
en doble canal unos 70 GB/s — y calcular in situ deja además la GPU libre en
vez de hacerla esperar una copia.

Esto solo vale si el procesador lee directamente los pesos empaquetados en
4 bits. De ahí un pequeño núcleo C++ con un camino AVX2 (`acvram_cpu.cpp`,
cargado por ctypes, sin cabeceras Python ni ninja). Incluso en su rama
**escalar** de respaldo, supera a `dequantize() @ x` por un factor 1,44 en
INT4 y 3,21 en NVFP4, porque este último escribe primero una copia de 32 bits
de toda la matriz.

En Mistral-Large-123B, la estimación del planificador pasa de 1,35 a
2,42 tokens/s.

### Precisión mixta (`--snr-floor`, desactivada por defecto)

El conversor mide la relación señal/ruido a la salida de cada capa para cada
tensor y puede promover a un formato más ancho los que caen por debajo de
`--snr-floor`, con un límite del 15 % de los tensores y un precio máximo
(`--promotion-cout-max`, en mebibytes añadidos).

El umbral vale **cero por defecto**: no se promueve nada. La decodificación
está limitada por el ancho de banda de memoria, y la medida en
`Huihui-Qwen3.8-27B` lo zanja — un umbral de 25 dB cuesta un 13,4 % de memoria
y un 10,6 % de caudal (18,50 GiB y 41,8 t/s frente a 16,02 y 46,2) por un 2,0 %
de perplejidad (42,591 frente a 43,447, corpus de 16 383 tokens).
`--snr-floor 25` restablece el comportamiento antiguo cuando la calidad prima
sobre la velocidad.

### Y `acvram eval`

La relación señal/ruido y el coseno de los logits son aproximaciones.
`acvram eval DIR [DIR ...]` mide la perplejidad por ventana deslizante, para
que una elección de formato se decida con pruebas:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Puntos de entrada HTTP

| punto de entrada | notas |
|---|---|
| `POST /v1/chat/completions` | flujo SSE o respuesta única; usa la plantilla de conversación del modelo |
| `POST /v1/completions` | prompt en texto o en identificadores de tokens |
| `POST /v1/embeddings` | estados ocultos finales promediados, normalizados L2, `dimensions` respetado |
| `GET /v1/models` | más un bloque `acvram`: formatos, dispositivos, capacidad de la caché KV |
| `GET /health`, `GET /metrics` | caudal de decodificación, ocupación de los bloques KV |

Los nombres de campo de estas respuestas se quedan en inglés: es el protocolo
OpenAI, y traducirlos rompería todos los clientes existentes.

## De dónde salen las cifras

Cada valor citado arriba lo produce código de este repositorio y lo verifica
`pytest`. Medidas hechas en procesador con los núcleos de referencia:

| formato | bits/peso | SNR de los pesos | coseno de los logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dos constataciones de estas medidas cambiaron los valores por defecto:

* **Una rotación de Hadamard ayuda al INT4 y no al NVFP4.** Los grupos de 128
  del INT4 no pueden absorber un canal aberrante aislado, así que repartir los
  valores extremos vale una transformada en n log n por activación. Los bloques
  de 16 del NVFP4 ya llevan su propia escala. De ahí `--hadamard auto`, que solo
  lo aplica al INT4.
* **El INT8 supera al FP8 E4M3 para la caché KV**, 44 dB frente a 32 dB a
  tamaño idéntico, porque una escala por (token, cabeza) ya proporciona el
  rango dinámico en el que el FP8 gasta bits de exponente. Las dos tarjetas
  usan por tanto una caché KV en INT8, aunque la 5090 sabría hacer FP8.

## Documentación

* [`REPRISE.md`](../REPRISE.md) — **retomar el proyecto en otra máquina**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — cómo encajan las piezas
* [`docs/MATERIEL.md`](MATERIEL.md) — ajustar esta máquina concreta
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **lo que no está hecho**, léase primero
* [`CONVENTIONS.md`](../CONVENTIONS.md) — convenciones de trabajo sobre el código (idioma, estilo, controles antes de empujar)

## Resultados medidos (22/09/2026, RTX 5090 a 400 W, régimen ≥ 20 s en el contador de energía)

Qwen3-Coder-30B-A3B en NVFP4 (expertos) + INT8 (atención, cabeza), mismo
protocolo para todos los motores (`outils/`, una tarjeta, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| decodificación 12 secuencias | **1 625,5 t/s** | 1 596,1 t/s | — |
| decodificación 1 secuencia | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 tokens/s** | 21 054 | 8 671 (TabbyAPI, retirado) |

Caudales del día (puesto 1030, régimen eco `-lgc 2700`, canal en servicio;
muestreo voraz capturado en el grafo CUDA, valor por defecto de 0.6.35). El
b=12 es una celda oficial sellada (mediana de 6 ventanas intercaladas, reloj por
ventana). El valor vLLM 1 596,1 es la referencia congelada del 21/09 (vLLM no
reejecutado ese día): la diferencia +1,84 % vale a referencia igual, no como
nueva medida de ambos la misma mañana. El J/token a reloj igual frente a los tres
motores sigue en remedición (`outils/gpu/mesure/banc-4moteurs.py`) — una cifra
sin régimen no se publica.

La mañana del 14/09 acvram estaba a 630 t/s y 0,619 J/token en la misma
celda: las ganancias vienen de la MMA FP4 nativa de Blackwell
(`mma.sync … kind::mxf4nvf4`, ×7,9 sobre bf16), del MoE en GEMM agrupada por
cubo de lote, de un enrutado en un solo núcleo (3 677 → 1 517 lanzamientos
por paso) y de una GEMM estrecha en tensor cores para las proyecciones. Cada
cifra tiene su nota en `acvram-memoire/revue/` con la predicción sellada antes
de la medida, el instrumento y su régimen — una cifra sin régimen no se
publica.

Donde acvram va por delante: modelos MLA (GLM-4.7-Flash) en NVFP4 nativo
sm_120, que vLLM solo sirve en FP8 (b=1: 165,35 t/s en servicio); los modelos
que no caben en VRAM; y, desde 0.6.35, la decodificación con lote grande de un
MoE que cabe en VRAM — b=12 pasa de 1 540 (0.6.34) a 1 625,5 t/s, es decir
+1,84 % por delante de la referencia vLLM congelada (1 596,1). La diferencia
sigue siendo estrecha y a referencia congelada; la diferencia en energía está por
volver a medir.

## Estado

Versión 0.6.35. Todo funciona en la 5090: núcleos CUDA compilados para
`sm_120a` (FP4 nativo) y `sm_86`, grafos CUDA, cuantización NVFP4/INT8/INT4,
servidor HTTP. Salvaguardas en su sitio: la tarjeta es invisible para las
sesiones de trabajo (`CUDA_VISIBLE_DEVICES` vacío) y solo `outils/carte.sh` la
presta, bajo cerrojo, a una medida a la vez; un vigilante registra todo acceso
fuera del cerrojo; una medida de energía que cubra más de una tarjeta o menos
de 10 s se invalida; un modelo cargado en régimen degradado lo dice y no entra
en un duelo.

640 tests (`pytest -q`, un minuto en procesador; los tests GPU solo corren
bajo `carte.sh`). Seguimiento del trabajo: `acvram-memoire/` (reglas,
directorio, cuadernos, revisión de 180 notas).

## Apoyar

El desarrollo de acvram se hace con material personal. Si el proyecto le es
útil: **Apoyar: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Licencia

GPL-3.0 o posterior.
