#!/bin/bash
# acvram contre llama.cpp, meme source Q4_K_M, ABBA.
#   acvram  Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4   16,5 Go
#   gguf    Qwen3-Coder-30B-A3B-Instruct-Q4_K_M            17,3 Go
# L'ecart de taille (5 %) est un BIAIS EN NOTRE FAVEUR sur un decodage limite
# par la bande passante : 5 % d'octets en moins valent 5 % de debit en plus.
# Les deux tailles sont ecrites a cote du resultat pour qu'on puisse soustraire.
# Binaire llama.cpp : le CUDA de Jan (b9967/linux-cuda-13), PAS les builds de
# /mnt/AI_GENERATOR/llamacpp qui sont en Vulkan.
# Cache KV : q8_0 cote llama.cpp, int8 cote acvram — 8 bits des deux cotes.
set -u
S="$(dirname "$0")"
G=/mnt/4TO_SATACMR_2022/Modeles/models_gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
A=$("$(dirname "$0")/../../racine_modeles.py")/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
RACINE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/../../.." && pwd))
PY=${ACVRAM_PY:-$RACINE/../../anticitoyen-vram/.venv/bin/python}
CLE=llamacpp-9c1f4c1e6f2a4d0f
ESSAIS=7
CONC=${CONC:-1}

libre() {
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 800 ] || { echo "REFUS : carte non libre ($u Mio)"; exit 1; }
}
tuer() {
  for p in 8080 8090; do
    pid=$(ss -tlnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
  sleep 8
}

manche_llama() {
  libre
  CACHE_KV=q8_0 llamacpp-serveur "$G" 8192 >/dev/null 2>&1
  for _ in $(seq 1 150); do
    curl -s -m 2 -H "Authorization: Bearer $CLE" http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && break
    sleep 2
  done
  # Le BINAIRE doit se declarer. Le 170 contre 122 t/s de la 3080 Ti reste
  # indecidable faute de savoir quelles architectures son build portait :
  # un chiffre dont on ne peut plus retrouver les conditions est perdu.
  if [ -z "${INFO_PUBLIEE:-}" ]; then
    echo "binaire llama.cpp : $(curl -s -m 5 -H "Authorization: Bearer $CLE" \
      http://127.0.0.1:8080/props 2>/dev/null | head -c 400 | tr -d "\n")"
    INFO_PUBLIEE=1
  fi
  printf 'llamacpp\t'
  $PY "$S/duel-moteurs.py" http://127.0.0.1:8080/v1/chat/completions "$CLE" \
      "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" $ESSAIS $CONC
  tuer
}
manche_acvram() {
  libre
  SPECULATIF=none acvram-serveur "$A" 8192 >/dev/null 2>&1
  printf 'acvram  \t'
  $PY "$S/duel-moteurs.py" http://127.0.0.1:8090/v1/chat/completions x \
      "Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4" $ESSAIS $CONC
  tuer
}

tuer
echo "modele : Qwen3-Coder-30B-A3B-Instruct · source Q4_K_M des deux cotes"
echo "poids  : acvram 16,5 Go · gguf 17,3 Go (+5 % pour llama.cpp)"
echo "cache  : q8_0 (llama.cpp) / int8 (acvram) · contexte 8192 · carte 0 epinglee · speculation DESACTIVEE des deux cotes"
echo "binaire: llama-server CUDA de Jan b9967/linux-cuda-13"
manche_acvram
manche_llama
manche_llama
manche_acvram
echo "corpus : /mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw  sha256 $(sha256sum /mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw | cut -c1-16)  invite ~350 mots de texte reel, extrait different par essai, MEME invite des deux cotes"
echo "carte  : $(nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem,power.limit --format=csv,noheader)"
echo "FIN-DUEL2"
