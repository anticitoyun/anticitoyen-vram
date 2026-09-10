"""Charge GPU CONNUE, pour eprouver la fragilite d'un test sous contention.

Deux grandeurs reglees, pas « un truc qui tourne » :
    --gio     memoire occupee, allouee d'un bloc et gardee
    --calcul  fraction de temps passee a calculer (0 = memoire seule)

Sans cela on ne saurait pas ce qu'on accuse : un test peut tomber par manque
de VRAM, par contention de calcul, ou par les deux — et le remede n'est pas le
meme. La charge s'annonce sur stdout et se retire proprement a l'arret.
"""
import argparse, os, signal, sys, time, torch

p = argparse.ArgumentParser()
p.add_argument("--gio", type=float, default=20.0)
p.add_argument("--calcul", type=float, default=0.5)
p.add_argument("--duree", type=float, default=600.0)
a = p.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("REFUS : pas de carte")

n = int(a.gio * (1 << 30) / 4)
bloc = torch.empty(n, dtype=torch.float32, device="cuda")
libre, total = torch.cuda.mem_get_info()
print(f"charge : {a.gio:.1f} Gio pris · {libre/(1<<30):.1f} Gio libres sur "
      f"{total/(1<<30):.1f} · calcul {a.calcul:.0%}", flush=True)

# Le calcul occupe la carte sans allouer davantage : deux matrices fixes.
m = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
fin = time.time() + a.duree
arret = {"v": False}
signal.signal(signal.SIGTERM, lambda *_: arret.__setitem__("v", True))
signal.signal(signal.SIGINT, lambda *_: arret.__setitem__("v", True))
while time.time() < fin and not arret["v"]:
    t0 = time.time()
    if a.calcul > 0:
        for _ in range(20):
            m = torch.nn.functional.relu(m @ m.T)[:4096, :4096].contiguous()
        torch.cuda.synchronize()
    dt = time.time() - t0
    if a.calcul < 1.0 and dt > 0:
        time.sleep(dt * (1 - a.calcul) / max(a.calcul, 1e-3))
print("charge retiree", flush=True)
