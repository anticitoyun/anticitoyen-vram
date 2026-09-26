<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-soutenir-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Gerbang inferensi yang kompatibel dengan API OpenAI, yang memperlakukan memori sebagai hierarki, memberi setiap GPU format numerik yang paling baik dibaca oleh siliconnya, dan mengoptimalkan setiap token dalam joule sama seriusnya dengan dalam detik.

<div align="center">

[🇫🇷 Français](../README.md) · [🇬🇧 English](README.en.md) · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · **🇮🇩 Bahasa Indonesia** · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Perbandingan throughput dan energi dengan vLLM dan llama.cpp" width="720"></p>

---

## Daftar isi

- [Dua gagasan](#idees)
- [Mulai cepat](#demarrage)
- [Instalasi](#installer)
- [Apa yang dikatakan `acvram plan`](#plan)
- [Menjadi cepat](#optimisations)
- [Titik akses HTTP](#http)
- [Dari mana angka-angka ini berasal](#chiffres)
- [Dokumentasi](#documentation)
- [Hasil yang diukur](#resultats)
- [Status](#etat)
- [Kredit](#credits)
- [Lisensi](#licence)
- [Dukung proyek ini](#soutien)

---

<a id="idees"></a>

## Dua gagasan

Dirancang untuk satu mesin spesifik:

| | |
|---|---|
| Prosesor | Intel Core i9-14900K (8 inti P + 16 inti E) |
| Motherboard | ASUS ROG Maximus Z790 Dark Hero |
| Memori | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Sistem | Ubuntu 26.04 LTS (CUDA 13); kedua kartu di PCIe x8/x8, dibatasi 400 W / 275 W |

**Satu format per GPU.** RTX 5090 memiliki tensor core FP4; RTX 3080 Ti tidak memilikinya, begitu pula FP8. Menyelaraskan keduanya ke format yang sama akan menyia-nyiakan 5090. Konverter karena itu menulis *model yang sama dua kali*, dalam format yang benar-benar bisa dimanfaatkan setiap tujuan:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| bobot | **NVFP4** — E2M1 + skala FP8 E4M3 setiap 16 | **INT4** — uint4 + skala dan nol fp16 setiap 128 |
| bit per bobot | 4,50 | 4,16 |
| dibanding BF16 | ×3,56 lebih kecil | ×3,85 lebih kecil |
| mode komputasi | tensor core FP4 | didekuantisasi ke FP16 di dalam kernel, tensor core FP16 |
| cache KV | INT8 | INT8 |

32 Go VRAM pada 4,5 bit per bobot memuat sekitar **56 miliar parameter**, dibanding 16 miliar dalam BF16. Pada kedua kartu bersama-sama, ini menghasilkan sekitar **78 miliar parameter residen** bahkan sebelum menyentuh RAM.

**Memori adalah hierarki, bukan tembok.** Tiga lapisan, dan penjadwal mengukur biaya masing-masing alih-alih berharap model akan muat:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

---

<a id="demarrage"></a>

## Mulai cepat

```bash
./install.sh                       # lingkungan virtual + torch cu128 + acvram
acvram doctor                      # apakah mesin ini siap, dan untuk apa
acvram detect                      # apa yang sebenarnya ada di sini

acvram plan  ~/modeles/Qwen3-32B                    # ke mana setiap lapisan akan pergi
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Klien OpenAI apa pun kemudian terhubung:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Halo"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="inutilise")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Halo"}])
```

---

<a id="installer"></a>

## Instalasi

Dari sumber (semua platform):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Atau lewat paket, satu berkas terlampir pada setiap [rilis GitHub](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Saluran | Berkas terlampir pada rilis | Perintah |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (nama yang dihasilkan oleh `rpmbuild`, tidak tetap) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (atau `rpmbuild --rebuild *.src.rpm` dari `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpak` | `flatpak install acvram-<version>.flatpak` |

Pip tidak dipublikasikan sebagai paket (tidak ada wheel yang dibangun): `pip install -e '.[dev]'` menginstal dari klon sumber, sama seperti `./install.sh`.

---

<a id="plan"></a>

## Apa yang dikatakan `acvram plan`

Penjadwal pantas dijalankan sebelum unduhan apa pun. Ia menjawab pertanyaan-pertanyaan yang menentukan apakah sebuah model dapat digunakan pada mesin ini:

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

Ia menjelajahi ruang konfigurasi alih-alih berhenti pada yang pertama yang muat, dan dua keputusannya cukup kontra-intuitif untuk pantas dijelaskan:

* **Ia membiarkan 3080 Ti tidak terpakai** ketika sebuah model muat hanya pada 5090. Tahapan sebuah pipeline berjalan secara berurutan: menambahkan tahap pada 912 Go/s ke dalam pipeline pada 1790 Go/s memperlambat dekode aliran tunggal. Dipaksakan dengan `--gpus all`.
* **Ia mengecilkan cache KV untuk menjaga bobot tetap di VRAM.** Setiap gibibyte yang diberikan ke cache adalah satu gibibyte bobot yang terdorong ke bus PCIe, dan membaca satu bobot lewat PCIe berbiaya sekitar tiga puluh kali lipat dibanding dari VRAM. Pada 70B di atas, hanya pilihan ini yang menaikkan dari 2,3 menjadi 17,8 token/s.

---

<a id="optimisations"></a>

## Menjadi cepat

Empat optimisasi, masing-masing diverifikasi dengan bukti ekuivalensi dan bukan sekadar stopwatch: optimisasi yang mengubah jawaban adalah bug.

Lapisan linier NVFP4 model padat secara bawaan melewati tata letak Marlin (+57 hingga +90% throughput pada b = 8, TTFT +2 hingga +4 ms menurut revue/poste6-piece147-verdict-24-09.md; fallback `ACVRAM_PROJ_MARLIN=0`, lihat [CHANGELOG.md](../CHANGELOG.md)).

### Dekode spekulatif (`--speculative`)

Mendekode satu token dengan batch berukuran 1 dibatasi oleh memori: mesin membaca semua bobot aktif untuk menghasilkan hanya satu token. Memverifikasi K token yang diusulkan membaca bobot yang sama **hanya sekali**. Dua pengusul:

* `ngram` (bawaan) — mencari akhiran saat ini lebih awal dalam konteks dan mengusulkan apa yang mengikutinya. Tidak berbiaya apa pun, tidak memerlukan model apa pun. Menguntungkan ketika keluaran menyalin masukan: penyuntingan kode, RAG, peringkasan.
* `draft` — model kecil pada perangkat kedua. Pada rig ini, perangkat tersebut adalah RTX 3080 Ti, yang sengaja dibiarkan menganggur oleh penjadwal untuk model apa pun yang muat pada 5090.

`mtp` (kepala `nextn` model) dan `auto` juga ada; belum menguntungkan saat ini dan tidak diaktifkan secara bawaan — lihat `docs/ARCHITECTURE.md`.

Penerimaan bersifat eksak, bukan pendekatan: sebuah usulan diterima dengan probabilitas `min(1, p/q)` dan penolakan mengambil sampel ulang dari bagian positif ternormalisasi dari `p - q`. Diukur pada 40.000 penarikan terhadap draft yang sengaja dikalibrasi buruk, distribusi yang dihasilkan tetap berada pada 0,002 variasi total dari target — spekulasi membeli kecepatan, tidak pernah jawaban yang berbeda.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache prefiks (aktif secara bawaan)

Blok dialamatkan oleh hash *berantai* dari potongan tokennya: dua permintaan yang berbagi instruksi sistem berbagi bloknya, dan yang kedua tidak lagi perlu menghitungnya terlebih dahulu. Perantaian ini sangat diperlukan: enam belas token yang sama dalam konteks berbeda tidak berisi kunci dan nilai yang sama, dan mem-hash hanya potongan tersebut akan melayani cache satu urutan ke urutan lain.

Blok yang dibebaskan yang isinya tetap dapat dikenali bergabung dengan antrean LRU alih-alih daftar blok bebas: cache dengan demikian bertahan di antara permintaan tanpa pernah menolak alokasi yang sebenarnya bisa dilayaninya.

### Komputasi lapisan host (`--host-exec`)

Sebuah lapisan yang bobotnya berada di RAM dapat disalin ke GPU atau dihitung di tempat. Kedua jalur dibatasi oleh memori dan membaca byte yang sama: yang tercepat adalah yang bus-nya paling lebar — PCIe 5.0 x16 memberikan sekitar 54 Go/s, DDR5 dual-channel sekitar 70 Go/s — dan menghitung di tempat juga membebaskan GPU alih-alih membuatnya menunggu salinan.

Ini hanya berharga jika prosesor membaca langsung bobot yang dikemas dalam 4 bit. Dari situ muncul kernel C++ kecil dengan jalur AVX2 (`acvram_cpu.cpp`, dimuat lewat ctypes, tanpa header Python maupun ninja). Bahkan pada cabang fallback **skalarnya**, ia mengalahkan `dequantize() @ x` dengan faktor 1,44 pada INT4 dan 3,21 pada NVFP4, karena yang terakhir ini terlebih dahulu menulis salinan 32-bit dari seluruh matriks.

Pada Mistral-Large-123B, estimasi penjadwal naik dari 1,35 menjadi 2,42 token/s.

### Presisi campuran (`--snr-floor`, dimatikan secara bawaan)

Konverter mengukur rasio sinyal terhadap derau pada keluaran setiap lapisan untuk setiap tensor dan dapat mempromosikan ke format yang lebih lebar bagi yang jatuh di bawah `--snr-floor`, dibatasi hingga 15% dari tensor dan harga maksimum (`--promotion-cout-max`, dalam mebibyte tambahan).

Ambang batas **nol secara bawaan**: tidak ada yang dipromosikan. Dekode dibatasi oleh bandwidth memori, dan pengukuran pada `Huihui-Qwen3.8-27B` memutuskan — ambang batas 25 dB berbiaya 13,4% memori dan 10,6% throughput (18,50 Gio dan 41,8 t/s dibanding 16,02 dan 46,2) demi 2,0% perpleksitas (42,591 dibanding 43,447, korpus 16 383 token). `--snr-floor 25` memulihkan perilaku lama ketika kualitas lebih diutamakan daripada kecepatan.

### Dan `acvram eval`

Rasio sinyal terhadap derau dan kosinus logit hanyalah pendekatan. `acvram eval REP [REP ...]` mengukur perpleksitas dengan jendela bergulir, agar pilihan format diputuskan dengan bukti:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

---

<a id="http"></a>

## Titik akses HTTP

| titik akses | catatan |
|---|---|
| `POST /v1/chat/completions` | aliran SSE atau respons tunggal; menggunakan templat percakapan model |
| `POST /v1/completions` | prompt sebagai teks atau sebagai ID token |
| `POST /v1/embeddings` | status tersembunyi akhir yang dirata-ratakan, dinormalisasi L2, `dimensions` dihormati |
| `GET /v1/models` | ditambah blok `acvram`: format, perangkat, kapasitas cache KV |
| `GET /health`, `GET /metrics` | throughput dekode, keterisian blok KV |

Nama field respons ini tetap dalam bahasa Inggris: ini adalah protokol OpenAI, dan menerjemahkannya akan merusak semua klien yang ada.

---

<a id="chiffres"></a>

## Dari mana angka-angka ini berasal

Setiap nilai yang dikutip di atas dihasilkan oleh kode dari repositori ini dan diverifikasi dengan `pytest`. Pengukuran dilakukan pada prosesor dengan kernel referensi:

| format | bit/bobot | SNR bobot | kosinus logit dibanding BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Dua temuan dari pengukuran ini mengubah nilai-nilai bawaan:

* **Rotasi Hadamard membantu INT4 dan tidak membantu NVFP4.** Grup-grup 128 milik INT4 tidak dapat menyerap satu nilai ekstrem kanal yang terisolasi, sehingga menyebarkan nilai-nilai ekstrem sepadan dengan transformasi berorde n log n per aktivasi. Blok-blok 16 milik NVFP4 sudah membawa skalanya sendiri. Dari situ `--hadamard auto`, yang hanya menerapkannya pada INT4.
* **INT8 mengalahkan FP8 E4M3 untuk cache KV**, 44 dB dibanding 32 dB pada ukuran yang sama, karena satu skala per (token, head) sudah memberikan rentang dinamis yang untuknya FP8 menghabiskan bit eksponen. Kedua kartu karena itu menggunakan cache KV dalam INT8, meskipun 5090 bisa saja menggunakan FP8. Format `k8v4` (nilai dalam INT4, −22% byte cache) tersedia sebagai opsi, **belum terkualifikasi** — lihat `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **melanjutkan proyek di mesin lain** (bahasa Prancis) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | bagaimana bagian-bagian saling terpasang |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | NVFP4 murni atau attention+GDN dalam int8 per kanal, pada hibrida Gated DeltaNet |
| [`docs/MATERIEL.md`](MATERIEL.md) | menyetel mesin spesifik ini |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **apa yang belum dikerjakan**, baca ini dahulu |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | konvensi kerja pada kode (bahasa, gaya, pemeriksaan sebelum push) |

---

<a id="resultats"></a>

## Hasil yang diukur (22/09/2026, RTX 5090 pada 400 W, rezim ≥ 20 s pada pengukur energi)

Qwen3-Coder-30B-A3B dalam NVFP4 (expert) + INT8 (atensi, head), protokol yang sama untuk semua mesin (`outils/`, satu kartu, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| dekode 12 urutan | 1 995,1 t/s ² | 2 027,0 t/s ² | — |
| dekode 1 urutan | 312,3 t/s ³ ⁴ | 284,8 t/s ³ | **329,9 t/s** ⁴ |
| prefill pp2048 | **22 707 token/s** | 21 054 | 8 671 (TabbyAPI, dihapus) |

¹ Ralat 22/09: `serve` berspekulasi secara bawaan (`--speculative ngram`, cli.py), para pesaing tidak; angka 380,8 t/s yang dipublikasikan sejauh ini diukur DENGAN spekulasi. Tanpa spekulasi (`--speculative none`, rantai yang sama, revue/poste2-piece44-speculation-none-22-09.md): 283,6 t/s — acvram berada di **posisi ketiga** pada b=1, di belakang llama.cpp dan vLLM. Dalam energi ia tetap di depan llama.cpp (0,601 dibanding 0,700 J/token neto). Pada b=12 spekulasi tidak pernah aktif (pengaman `lot_max=2`): sel ini sudah berada dalam kondisi setara sejak awal.

² 23/09, sesi yang sama, klien HTTP yang sama (`banc-llamacpp-16-09.py` terhadap `acvram serve` dan `vllm serve`), `-lgc 2700` dipasang secara eksplisit di sekitar setiap lengan, sel bergantian A V V A, ≥ 5 batch per lengan, deviasi dinyatakan hanya di luar 2 σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 pada dekode, reduksi atensi yang digulung terbuka): deviasi −1,6%, **di bawah 2 σ: kesetaraan throughput**. Dalam J/token, **vLLM tetap unggul 7,0%** (di luar 2 σ). Dengan 0.6.37 protokol yang sama memberikan −4,7%.

³ Sesi dan protokol yang sama seperti ², tanpa spekulasi di kedua sisi: acvram 312,3 dibanding vLLM 284,8 — **acvram unggul 9,7% dalam throughput** (di luar 2 σ); J/token: **kesetaraan** (deviasi 0,04%, di bawah 2 σ).

⁴ 23/09, protokol yang sama terhadap llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 dengan routing yang ditulis ulang (+5,6%): acvram 310,8 dibanding llama.cpp 329,9 t/s — **llama.cpp unggul 5,8% dalam throughput, acvram unggul 13,4% dalam J/token** (0,598 dibanding 0,691).

Throughput hari ini (posting 1030, rezim hemat `-lgc 2700`, pipeline sedang beroperasi; sampling greedy ditangkap dalam graf CUDA, bawaan dari 0.6.35). b=12 acvram adalah sel resmi yang disegel (median dari 6 jendela yang saling diselang-seling, jam per jendela).

> **Ralat (23/09/2026).** Perbandingan dengan vLLM yang dipublikasikan sejauh ini (b=12: 1 782 dibanding 1 634 t/s; b=1: 290,6) mempertentangkan acvram yang diukur lewat HTTP dengan vLLM yang diukur **offline** (`LLM().generate()`), dan ralat 22/09 secara keliru menyatakan bahwa sel vLLM melewati `vllm serve`. Pada 23/09: klien HTTP yang sama untuk keduanya, dan `-lgc` dipasang untuk keduanya (acvram memasang miliknya saat startup, `vllm serve` tidak: tanpa langkah pencegahan ini vLLM berjalan pada ~2.930 MHz dibanding ~2.650). Hasil dalam catatan ²: vLLM unggul 9,1% pada b=12.

Pada pagi 14/09 acvram berada pada 630 t/s dan 0,619 J/token pada sel yang sama: keuntungan berasal dari MMA FP4 native Blackwell (`mma.sync … kind::mxf4nvf4`, ×7,9 dibanding bf16), dari MoE dalam GEMM berkelompok per ember batch, dari routing dalam satu kernel tunggal (3 677 → 1 517 peluncuran per langkah) dan dari GEMM sempit pada tensor core untuk proyeksi. Setiap angka memiliki catatannya di `acvram-memoire/revue/` dengan prediksi tersegel sebelum pengukuran, instrumen dan rezimnya — angka tanpa rezim tidak dipublikasikan.

Di mana acvram unggul: model MLA (GLM-4.7-Flash) dalam NVFP4 native sm_120, yang hanya dilayani vLLM dalam FP8 (b=1: 165,35 t/s dalam operasi); model yang tidak muat di VRAM. Dekode urutan tunggal bukan bagian dari itu: tanpa spekulasi, acvram di sana unggul dari vLLM sebesar 9,7% (catatan ³), tertinggal dari llama.cpp sebesar 5,8% dalam throughput tetapi unggul darinya sebesar 13,4% dalam energi (catatan ⁴). Pada batch besar, pada MoE yang muat di VRAM, vLLM setara dalam throughput pada b=12 (1 995,1 dibanding 2 027,0 t/s, di bawah 2 σ, catatan ²) tetapi mempertahankan 7,0% lebih sedikit J/token; acvram di sana berkembang dari 1 540 t/s (0.6.34) menjadi 1 995 (0.6.38).

---

<a id="etat"></a>

## Status

Versi 0.6.38. Semuanya berjalan pada 5090: kernel CUDA dikompilasi untuk `sm_120a` (FP4 native) dan `sm_86`, graf CUDA, kuantisasi NVFP4/INT8/INT4, server HTTP. Pengaman terpasang: kartu tidak terlihat oleh sesi kerja (`CUDA_VISIBLE_DEVICES` kosong) dan hanya `outils/carte.sh` yang menyediakannya, di bawah kunci, untuk satu pengukuran sekaligus; seorang pengawas mencatat setiap akses di luar kunci; pengukuran energi yang mencakup lebih dari satu kartu atau kurang dari 10 s dibatalkan; sebuah model yang dimuat dalam rezim terdegradasi menyatakan hal itu dan tidak masuk ke dalam duel.

4 107 tes (`pytest --collect-only -q`, satu menit pada prosesor; tes GPU hanya berjalan di bawah `carte.sh`). Pelacakan pekerjaan: `acvram-memoire/` (aturan, direktori, buku catatan, tinjauan atas beberapa ratus catatan).

---

<a id="credits"></a>

## Kredit

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, di bawah lisensi Apache-2.0: `acvram/kernels/marlin_port/` membawa kernel Marlin-nya (MoE dan padat), dengan atribusi lengkap berkas demi berkas di [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, tensor core FP4 milik Blackwell (`sm_120`) dan pustaka-pustaka yang menjadi ketergantungan proyek ini.
- **PyTorch** — mesin tensor dan ekstensi C++/CUDA.

Proyek independen, tidak berafiliasi dengan ASUS, NVIDIA, maupun proyek vLLM.

---

<a id="licence"></a>

## Lisensi

[GPL-3.0 atau yang lebih baru](../LICENSE) untuk kode dalam repositori ini. `acvram/kernels/marlin_port/` berisi kode yang dipindahkan dari [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (kernel `marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size`), di bawah lisensi Apache-2.0: setiap berkas menyimpan headernya yang asli, teks lisensi berada di `LICENSE-vllm`, dan daftar berkas, commit asal serta modifikasinya berada di [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Dukung proyek ini

Pengembangan acvram dilakukan pada perangkat keras pribadi. Jika proyek ini berguna bagi Anda:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Offrir%20un%20café&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Terjemahan: [TRADUIRE.md](TRADUIRE.md) (bahasa Prancis; panduan kontribusi belum diterjemahkan).
