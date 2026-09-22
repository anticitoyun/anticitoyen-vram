#!/usr/bin/env bash
# Un pas de decodage b=1 de llama.cpp (llama-server, binaire LM Studio de Katy/Laure,
# Coder-30B Q4_K_M) sous ncu : meme instrument que outils/ncu_instr_par_octet.sh
# (sudoers ncu, dram__bytes_op_read, sm__inst_executed, cache chaud par defaut).
# Pas de plage NVTX possible : invite d'UN jeton (le prefill est alors un pas b=1
# comme les autres) et n_predict=PAS-1, tout est profile, l'agregateur divise par PAS.
# llama-server est lance PAR ncu dans un bash root qui le tue lui-meme apres la
# requete (sudoers n'autorise que ncu, pas kill).
#   outils/carte.sh outils/ncu_llamacpp_b1.sh [PAS=3]
set -euo pipefail
PAS=${1:-3}
ICI=$(dirname "$(readlink -f "$0")")
NCU_SORTIE=${NCU_SORTIE:-/tmp/ncu-acvram}; mkdir -p "$NCU_SORTIE"
BIN_DIR=${LLAMACPP_LMSTUDIO_BIN:-$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.22.0}
GGUF=${GGUF:-/mnt/4TO_SATACMR_2022/Modeles/models_gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}
PORT=${PORT:-8091}
OUT=$NCU_SORTIE/ncu-ipo-llamacpp.csv
HOME_NCU=${HOME_NCU:-$NCU_SORTIE/home-ncu}; mkdir -p "$HOME_NCU"
cat > "$NCU_SORTIE/llamacpp-cible.sh" <<CIBLE
#!/bin/bash
export LD_LIBRARY_PATH=$BIN_DIR:/usr/local/lib/ollama/cuda_v12 CUDA_VISIBLE_DEVICES=0 HOME=$HOME_NCU
$BIN_DIR/llama-server -m "$GGUF" --host 127.0.0.1 --port $PORT -np 1 -c 4096 -ngl 999 --no-warmup > "$NCU_SORTIE/llamacpp-serveur.log" 2>&1 &
pid=\$!
for i in \$(seq 1 600); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 1; done
curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \\
  -d '{"prompt": [1000], "n_predict": $((PAS-1)), "temperature": 0.0, "cache_prompt": false, "ignore_eos": true, "id_slot": 0}' > "$NCU_SORTIE/llamacpp-reponse.json"
kill \$pid; wait \$pid 2>/dev/null; exit 0
CIBLE
chmod +x "$NCU_SORTIE/llamacpp-cible.sh"
sudo -n /usr/local/cuda/bin/ncu --csv --target-processes all --clock-control none --cache-control "${NCU_CACHE:-none}" \
  --metrics "${NCU_METRIQUES:-gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__cycles_elapsed.avg.per_second}" \
  "$NCU_SORTIE/llamacpp-cible.sh" > "$OUT" 2>"$NCU_SORTIE/ncu-ipo-llamacpp.err" || true
grep -c '^"[0-9]' "$OUT" || { echo "ECHEC : rien profile, voir $NCU_SORTIE/ncu-ipo-llamacpp.err"; tail -5 "$NCU_SORTIE/ncu-ipo-llamacpp.err"; exit 1; }
python3 "$ICI/ncu_instr_par_octet_agrege.py" "$OUT" "$PAS" | tee "$NCU_SORTIE/ncu-ipo-llamacpp.txt"
