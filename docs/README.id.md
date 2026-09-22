<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Dukung: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

Gerbang inferensi yang kompatibel dengan API OpenAI, yang memperlakukan
memori sebagai hierarki dan memberi setiap GPU format numerik yang paling
baik dibaca oleh silikonnya.

Dirancang untuk satu mesin tertentu:

| | |
|---|---|
| Prosesor | Intel Core i9-14900K (8 inti P + 16 inti E) |
| Papan induk | ASUS ROG Maximus Z790 Dark Hero |
| Memori | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| Sistem | Ubuntu 26.04 LTS (CUDA 13); kedua kartu di PCIe x8/x8, dibatasi 400 W / 275 W |

## Dua gagasan

**Satu format per GPU.** RTX 5090 memiliki tensor core FP4; RTX 3080 Ti tidak
memilikinya, dan juga tidak memiliki FP8. Menyamakan keduanya pada format
bersama akan menyia-nyiakan 5090. Karena itu konverter menulis *model yang
sama dua kali*, dalam format yang benar-benar bisa dimanfaatkan setiap tujuan:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| bobot | **NVFP4** — E2M1 + skala FP8 E4M3 setiap 16 | **INT4** — uint4 + skala dan nol fp16 setiap 128 |
| bit per bobot | 4,50 | 4,16 |
| dibanding BF16 | ×3,56 lebih kecil | ×3,85 lebih kecil |
| mode komputasi | tensor core FP4 | didekuantisasi ke FP16 di dalam kernel, tensor core FP16 |
| cache KV | INT8 | INT8 |

32 GB VRAM pada 4,5 bit per bobot menampung sekitar **56 miliar parameter**,
dibanding 16 miliar dalam BF16. Pada kedua kartu, itu kira-kira **78 miliar
parameter residen** bahkan sebelum menyentuh RAM.

**Memori adalah hierarki, bukan tembok.** Tiga tingkat, dan perencana
mengukur biaya masing-masing alih-alih berharap modelnya muat:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Mulai cepat

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Klien OpenAI mana pun kemudian dapat tersambung:

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

## Apa kata `acvram plan`

Perencana layak dijalankan sebelum unduhan apa pun. Ia menjawab pertanyaan
yang menentukan apakah sebuah model dapat dipakai di mesin ini:

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

Ia menjelajahi ruang konfigurasi alih-alih mengambil yang pertama muat, dan
dua keputusannya cukup berlawanan dengan intuisi sehingga layak diucapkan:

* **Ia membiarkan 3080 Ti tak terpakai** saat sebuah model muat di 5090
  saja. Irisan sebuah pipeline berjalan berurutan: menambahkan tahap 912 GB/s
  ke pipeline 1790 GB/s memperlambat dekode aliran tunggal. Dipaksa dengan
  `--gpus all`.
* **Ia menyusutkan cache KV agar bobot tetap di VRAM.** Setiap gigabita yang
  diberikan ke cache adalah satu gigabita bobot yang terdorong ke bus PCIe,
  dan membaca satu bobot lewat PCIe berbiaya sekitar tiga puluh kali lipat
  dibanding dari VRAM. Pada 70B di atas, pertukaran ini saja membawa dari 2,3
  ke 17,8 token/s.

## Melaju cepat

Empat optimasi, masing-masing diverifikasi dengan bukti kesetaraan dan bukan
sekadar stopwatch: optimasi yang mengubah jawaban adalah bug.

### Dekode spekulatif (`--speculative`)

Mendekode satu token dengan batch berukuran 1 dibatasi memori: mesin membaca
semua bobot aktif untuk menghasilkan satu token saja. Memverifikasi K token
yang diusulkan membaca bobot yang sama **sekali saja**. Dua pengusul:

* `ngram` (bawaan) — mencari sufiks saat ini lebih awal dalam konteks dan
  mengusulkan yang mengikutinya. Tidak berbiaya, tidak butuh model.
  Menguntungkan saat keluaran menyalin masukan: penyuntingan kode, RAG,
  ringkasan.
* `draft` — model kecil pada perangkat kedua. Di rig ini perangkat itu adalah
  RTX 3080 Ti, yang sengaja dibiarkan menganggur oleh perencana untuk setiap
  model yang muat di 5090.

Penerimaan bersifat eksak, bukan hampiran: sebuah usulan diterima dengan
peluang `min(1, p/q)` dan penolakan mengambil sampel ulang dari bagian positif
ternormalisasi `p - q`. Diukur pada 40 000 undian terhadap draf yang sengaja
dikalibrasi buruk, distribusi yang dipancarkan tetap dalam 0,002 variasi total
dari target — spekulasi membeli kecepatan, tidak pernah jawaban yang berbeda.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache prefiks (aktif secara bawaan)

Blok dialamatkan oleh hash *berantai* dari irisan tokennya: dua permintaan
yang berbagi instruksi sistem berbagi bloknya, dan yang kedua tidak perlu
lagi menghitungnya di muka. Perantaian mutlak diperlukan: enam belas token
yang sama dalam konteks berbeda tidak memuat kunci dan nilai yang sama, dan
meng-hash irisan saja akan menyajikan cache satu urutan ke urutan lain.

Blok yang dibebaskan namun isinya masih dapat dikenali masuk ke antrean LRU
alih-alih daftar blok bebas: cache dengan demikian bertahan antar permintaan
tanpa pernah menolak alokasi yang bisa dilayaninya.

### Komputasi di tingkat host (`--host-exec`)

Lapisan yang bobotnya berada di RAM dapat disalin ke GPU atau dihitung di
tempat. Kedua jalur dibatasi memori dan membaca bita yang sama: yang lebih
cepat adalah yang busnya lebih lebar — PCIe 5.0 x16 memberi sekitar 54 GB/s,
DDR5 kanal ganda sekitar 70 GB/s — dan menghitung di tempat juga membiarkan
GPU bebas alih-alih membuatnya menunggu salinan.

Ini hanya sepadan jika prosesor membaca bobot terkemas 4 bit secara langsung.
Karena itu ada kernel C++ kecil dengan jalur AVX2 (`acvram_cpu.cpp`, dimuat
lewat ctypes, tanpa header Python maupun ninja). Bahkan pada cabang cadangan
**skalarnya**, ia mengalahkan `dequantize() @ x` dengan faktor 1,44 pada INT4
dan 3,21 pada NVFP4, karena yang terakhir menulis dulu salinan 32 bit dari
seluruh matriks.

Pada Mistral-Large-123B, perkiraan perencana naik dari 1,35 ke 2,42 token/s.

### Presisi campuran (`--snr-floor`, nonaktif secara bawaan)

Konverter mengukur rasio sinyal/derau di keluaran setiap lapisan untuk setiap
tensor dan dapat mempromosikan ke format yang lebih lebar tensor yang jatuh di
bawah `--snr-floor`, dalam batas 15 % tensor dan plafon harga
(`--promotion-cout-max`, dalam mebibita yang ditambahkan).

Ambangnya **nol secara bawaan**: tidak ada yang dipromosikan. Dekode dibatasi
oleh lebar pita memori, dan pengukuran pada `Huihui-Qwen3.8-27B` memutuskan —
ambang 25 dB berbiaya 13,4 % memori dan 10,6 % throughput (18,50 GiB dan
41,8 t/s dibanding 16,02 dan 46,2) demi 2,0 % perpleksitas (42,591 dibanding
43,447, korpus 16 383 token). `--snr-floor 25` mengembalikan perilaku lama
saat kualitas lebih penting daripada kecepatan.

### Dan `acvram eval`

Rasio sinyal/derau dan kosinus logit adalah hampiran.
`acvram eval DIR [DIR ...]` mengukur perpleksitas dengan jendela geser, agar
pilihan format diputuskan berdasarkan bukti:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Titik akhir HTTP

| titik akhir | catatan |
|---|---|
| `POST /v1/chat/completions` | aliran SSE atau respons tunggal; memakai templat percakapan model |
| `POST /v1/completions` | prompt berupa teks atau id token |
| `POST /v1/embeddings` | keadaan tersembunyi akhir yang dirata-rata, dinormalisasi L2, `dimensions` dihormati |
| `GET /v1/models` | ditambah blok `acvram`: format, perangkat, kapasitas cache KV |
| `GET /health`, `GET /metrics` | throughput dekode, hunian blok KV |

Nama bidang dalam respons ini tetap dalam bahasa Inggris: ini protokol
OpenAI, dan menerjemahkannya akan merusak semua klien yang ada.

## Dari mana angka-angka berasal

Setiap nilai yang dikutip di atas dihasilkan oleh kode di repositori ini dan
diverifikasi oleh `pytest`. Pengukuran dilakukan di prosesor dengan kernel
referensi:

| format | bit/bobot | SNR bobot | kosinus logit vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dua temuan dari pengukuran ini mengubah nilai bawaan:

* **Rotasi Hadamard membantu INT4 dan bukan NVFP4.** Kelompok 128 pada INT4
  tidak dapat menyerap satu kanal pencilan terisolasi, sehingga menyebarkan
  nilai ekstrem sepadan dengan transformasi n log n per aktivasi. Blok 16 pada
  NVFP4 sudah membawa skalanya sendiri. Karena itu `--hadamard auto`, yang
  hanya menerapkannya pada INT4.
* **INT8 mengalahkan FP8 E4M3 untuk cache KV**, 44 dB dibanding 32 dB pada
  ukuran identik, karena skala per (token, kepala) sudah menyediakan rentang
  dinamis yang untuknya FP8 menghabiskan bit eksponen. Kedua kartu karena itu
  memakai cache KV dalam INT8, meski 5090 bisa FP8.

## Dokumentasi

* [`REPRISE.md`](../REPRISE.md) — **melanjutkan proyek di mesin lain**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — bagaimana bagian-bagian tersusun
* [`docs/MATERIEL.md`](MATERIEL.md) — menyetel mesin tertentu ini
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **yang belum selesai**, baca dulu
* [`CONVENTIONS.md`](../CONVENTIONS.md) — konvensi kerja pada kode (bahasa, gaya, pemeriksaan sebelum push)

## Hasil terukur (22/09/2026, RTX 5090 pada 400 W, rezim ≥ 20 s pada meteran energi)

Qwen3-Coder-30B-A3B dalam NVFP4 (pakar) + INT8 (atensi, kepala), protokol
yang sama untuk semua mesin (`outils/`, satu kartu, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| dekode 12 urutan | **1 625,5 t/s** | 1 596,1 t/s | — |
| dekode 1 urutan | **380,8 t/s** | 290,6 t/s | 323,6 t/s |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, ditarik) |

Throughput hari ini (stasiun 1030, rezim hemat `-lgc 2700`, pipeline dalam
layanan; sampling serakah yang ditangkap dalam graf CUDA, bawaan 0.6.35). b=12
adalah sel resmi tersegel (median dari 6 jendela terjalin, clock per jendela).
Nilai vLLM 1 596,1 adalah rujukan beku 21/09 (vLLM tidak dijalankan ulang hari
itu): selisih +1,84% berlaku pada rujukan setara, bukan sebagai pengukuran ulang
keduanya di pagi yang sama. J/token pada clock yang sama terhadap ketiga mesin
masih diukur ulang (`outils/gpu/mesure/banc-4moteurs.py`) — angka tanpa rezim
tidak dipublikasikan.

Pagi 14/09 acvram berada di 630 t/s dan 0,619 J/token pada sel yang sama:
perolehan datang dari MMA FP4 asli Blackwell (`mma.sync … kind::mxf4nvf4`,
×7,9 atas bf16), dari MoE sebagai GEMM terkelompok per ember batch, dari
perutean dalam satu kernel (3 677 → 1 517 peluncuran per langkah), dan dari
GEMM sempit pada tensor core untuk proyeksi. Setiap angka punya catatannya di
`acvram-memoire/revue/` dengan prediksi yang disegel sebelum pengukuran,
instrumen, dan rezimnya — angka tanpa rezim tidak dipublikasikan.

Di mana acvram unggul: model MLA (GLM-4.7-Flash) dalam NVFP4 asli sm_120, yang
hanya dilayani vLLM dalam FP8 (b=1: 165,35 t/s dalam layanan); model yang tidak
muat di VRAM; dan, sejak 0.6.35, dekode batch besar dari MoE yang muat di VRAM —
b=12 naik dari 1 540 (0.6.34) ke 1 625,5 t/s, yaitu +1,84% di depan rujukan vLLM
beku (1 596,1). Selisihnya tetap sempit dan pada rujukan beku; selisih energi
masih harus diukur ulang.

## Status

Versi 0.6.35. Semuanya berjalan di 5090: kernel CUDA dikompilasi untuk
`sm_120a` (FP4 asli) dan `sm_86`, graf CUDA, kuantisasi NVFP4/INT8/INT4,
server HTTP. Pagar pengaman terpasang: kartu tak terlihat oleh sesi kerja
(`CUDA_VISIBLE_DEVICES` kosong) dan hanya `outils/carte.sh` yang
meminjamkannya, di bawah kunci, ke satu pengukuran pada satu waktu; penjaga
mencatat setiap akses di luar kunci; pengukuran energi yang mencakup lebih
dari satu kartu atau kurang dari 10 s dibatalkan; model yang dimuat dalam
rezim terdegradasi menyatakannya dan tidak ikut duel.

640 pengujian (`pytest -q`, satu menit di prosesor; pengujian GPU hanya
berjalan di bawah `carte.sh`). Pelacakan kerja: `acvram-memoire/` (aturan,
direktori, buku catatan, tinjauan 180 catatan).

## Dukung

acvram dikembangkan pada perangkat keras pribadi. Jika proyek ini berguna
bagi Anda: **Dukung: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## Lisensi

GPL-3.0 atau yang lebih baru.
