"""Micro-banc GEMV bf16, préparé sur demande de chef (9/09), NON EXÉCUTÉ —
la 5090 est à poste2. À lancer une fois la carte rendue.

Compare, à forme identique (hidden 5120, un jeton, bf16), le chemin réel
d'acvram (`torch.nn.functional.linear`, ce que `PlainTensor`/`QuantLinear`
appellent pour un poids non quantifié — voir poste4.md §"Chemin de lecture
des poids") contre un GEMV minimal écrit à la main (Triton, un bloc par
ligne de sortie, réduction en registres — même idée structurelle que
`mmvf.cu` de llama.cpp, sans en être une traduction).

Usage prévu, SOUS LES MÊMES INSTRUMENTS QUE poste2 (à aligner avec elle avant
de conclure quoi que ce soit — cf. sa demande « mêmes options, sinon les
deux mesures ne se parlent pas ») :

    # temps seul, sanity check avant tout profilage lourd :
    python3 microbanc_gemv_bf16.py

    # trafic DRAM réel, sous ncu (métriques standard à confirmer/aligner
    # avec celles de poste2 — placeholders ci-dessous) :
    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
        --kernel-name-base demangled \
        python3 microbanc_gemv_bf16.py --ncu

`--ncu` réduit à UN seul passage par variante (ncu réinstrumente déjà
chaque lancement de noyau ; empiler 5 passages ne fait que multiplier le
temps de capture sans rien ajouter à la mesure).
"""
import argparse
import time

import torch

try:
    import triton
    import triton.language as tl
    _TRITON = True
except ImportError:
    _TRITON = False


HIDDEN = 5120        # forme du comparatif : à recaler si chef/poste2
                      # utilisent une autre dimension pour Qwen2.5-Coder-14B
                      # (à vérifier dans le manifeste avant de publier un
                      # chiffre — ne pas supposer 5120 est correct sans lire)
PASSAGES = 5          # le premier jeté, comme partout dans le dossier


if _TRITON:
    @triton.jit
    def _gemv_bf16_kernel(x_ptr, w_ptr, y_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
        """Un programme Triton par ligne de sortie — même idée que
        `mmvf.cu` (un bloc CUDA par ligne), pas une copie de son code."""
        row = tl.program_id(0)
        acc = tl.zeros((), dtype=tl.float32)
        w_row = w_ptr + row * K
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_row + offs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x * w)
        tl.store(y_ptr + row, acc)

    def gemv_triton(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x: [K] bf16, w: [N, K] bf16 -> y: [N] fp32 (accumulation, comme
        mmvf.cu accumule en fp32 quel que soit T)."""
        n, k = w.shape
        y = torch.empty(n, device=x.device, dtype=torch.float32)
        block_k = 256 if k >= 256 else triton.next_power_of_2(k)
        _gemv_bf16_kernel[(n,)](x, w, y, K=k, BLOCK_K=block_k)
        return y


def bench(fn, *args, passages=PASSAGES):
    torch.cuda.synchronize()
    temps = []
    for i in range(passages):
        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        temps.append(time.perf_counter() - t0)
    # le premier jeté : coût unique déjà établi ailleurs dans le dossier
    utiles = temps[1:] if len(temps) > 1 else temps
    return out, temps, sum(utiles) / len(utiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--ncu", action="store_true",
                     help="un seul passage par variante, pour tourner sous ncu")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("PAS DE GPU ICI — script préparé, non exécutable en l'état "
              "(attendu : la carte est à poste2)")
        return

    dev = torch.device("cuda:0")   # 5090 — ACVRAM_PLAN_FIGE/CUDA_VISIBLE_DEVICES=0
                                    # obligatoire, cf. le piège du 9/09 sur sm_86
    torch.manual_seed(0)
    x = torch.randn(a.hidden, device=dev, dtype=torch.bfloat16)
    w = torch.randn(a.hidden, a.hidden, device=dev, dtype=torch.bfloat16)
    passages = 1 if a.ncu else PASSAGES

    print(f"forme : hidden={a.hidden}, un jeton, bf16, {passages} passage(s)")

    y1, t1, moy1 = bench(lambda: torch.nn.functional.linear(x, w),
                         passages=passages)
    print(f"F.linear (chemin acvram reel)      : {moy1*1e6:8.1f} us/appel  "
          f"passages={['%.1f'%(t*1e6) for t in t1]}")

    if _TRITON:
        y2, t2, moy2 = bench(lambda: gemv_triton(x, w), passages=passages)
        print(f"GEMV Triton (bloc par ligne)        : {moy2*1e6:8.1f} us/appel  "
              f"passages={['%.1f'%(t*1e6) for t in t2]}")
        ecart = (y1.float() - y2).abs()
        print(f"écart max vs F.linear : {ecart.max().item():.4f} "
              f"(accumulation fp32 des deux côtés, un écart non nul est "
              f"attendu par l'ordre de sommation — PAS un bogue en soi, "
              f"cf. la règle du dossier sur les divergences bf16/fp32)")
        if not a.ncu:
            print(f"rapport temps F.linear / Triton : {moy1/moy2:.2f}x")
    else:
        print("triton absent de cet interpreteur — installer dans le venv "
              "acvram avant de lancer, ou lancer avec le python qui a "
              "triton 3.7.1 (verifie le 9/09, hors venv acvram)")


if __name__ == "__main__":
    main()
