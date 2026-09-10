"""Chiffrer le PLANCHER du harnais, avant toute mesure de noyau.

Un montage a rendu ~49,5 us pour un noyau qui ecrit trois flottants et sort.
C'est un plancher d'instrument, pas un temps de noyau — et il emporte les
« 94 % de cout fixe » qu'on en avait tires, un plancher etant par construction
independant du travail.

LA CAUSE N'EST PAS CELLE QU'ON CROYAIT. Le diagnostic propose etait « sans
doute une synchronisation par appel » ; verification faite, les deux montages
fautifs faisaient DEJA une seule synchronisation pour 300 appels, evenements
CUDA autour du lot. Le correctif propose etait donc deja en place. La cause
reste inconnue — raison de plus pour CHIFFRER le plancher plutot que de le
deduire.

METHODE (poste1) : K iterations de FMA chainees, meme grille, meme harnais.
Le temps doit devenir lineaire en K ; LE COUDE DONNE LE PLANCHER, sans
hypothese sur sa cause.

On applique l'echelle aux DEUX harnais — lot amorti et appel isole — pour
retenir celui dont le plancher est le plus bas, au lieu de le supposer.
"""
import sys, time, torch
from acvram import kernels

ext = kernels.get_extension()
if ext is None or not hasattr(ext, "banc_fma"):
    print("REFUS : banc_fma absent du binaire — recompiler")
    sys.exit(1)

GX, GY, GZ, THREADS = 1, 32, 8, 128        # grille voisine de paged_attn
KS = [1, 10, 100, 1000, 10000, 100000]

def lot(K, n=300):
    """Harnais A : evenements autour du LOT, une seule synchronisation."""
    for _ in range(20):
        ext.banc_fma(GX, GY, GZ, THREADS, K)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        ext.banc_fma(GX, GY, GZ, THREADS, K)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / n

def isole(K, n=51):
    """Harnais B : un appel entre deux evenements, synchronisation a chaque."""
    for _ in range(20):
        ext.banc_fma(GX, GY, GZ, THREADS, K)
    torch.cuda.synchronize()
    v = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        ext.banc_fma(GX, GY, GZ, THREADS, K)
        e1.record()
        torch.cuda.synchronize()
        v.append(e0.elapsed_time(e1) * 1000.0)
    v.sort()
    return v[len(v) // 2]

print(f"# grille {GX}x{GY}x{GZ} = {GX*GY*GZ} blocs de {THREADS} fils")
print(f"{'K':>8} {'lot amorti':>12} {'appel isole':>12}")
for K in KS:
    print(f"{K:8d} {lot(K):10.2f} us {isole(K):10.2f} us")
print("# le COUDE de chaque colonne donne le plancher de ce harnais.")
print("# un temps sous ~5x le plancher se publie « sous le plancher », "
      "jamais comme une valeur.")
