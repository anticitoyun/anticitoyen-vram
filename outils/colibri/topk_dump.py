"""Lecteur du fichier TOPK_DUMP de colibri (outils/colibri/topk-dump.patch, bd 076) et contrôle de cohérence avec la TF-NLL.

Format (petit-boutiste) : "CTK1", int32 version=1, int32 K, int32 vocab, int32 n_positions ; puis par position :
int32 pos, int32 id_ref, float32 logprob_ref, K × (int32 id, float32 logprob) triés par logprob décroissant.

  topk_dump.py FICHIER [TF_NLL]   → résumé ; avec TF_NLL (la ligne « TF-NLL: x nats/token » de qwen36), rc 1 si
                                    |−moyenne(logprob_ref) − TF_NLL| > 1e-4 (la TF-NLL imprimée a 4 décimales).
"""
import struct
import sys

import numpy as np

TOLERANCE_NLL = 1e-4


def lire(chemin: str) -> dict:
    with open(chemin, "rb") as f:
        brut = f.read()
    if len(brut) < 20 or brut[:4] != b"CTK1":
        raise ValueError(f"{chemin} : pas un fichier TOPK_DUMP (en-tête « CTK1 » absent)")
    version, k, vocab, n = struct.unpack_from("<4i", brut, 4)
    if version != 1:
        raise ValueError(f"{chemin} : version {version} inconnue")
    # paires (id, logprob) ENTRELACÉES, comme topk_dump_ecrire les écrit — pas deux tableaux
    enreg = np.dtype([("pos", "<i4"), ("ref", "<i4"), ("lp_ref", "<f4"), ("paires", [("id", "<i4"), ("lp", "<f4")], (k,))])
    corps = brut[20:]
    if len(corps) != n * enreg.itemsize:
        # un moteur tué en cours d'écriture laisse un fichier court : il ne se lit jamais comme un fichier complet
        raise ValueError(f"{chemin} : {len(corps)} octets pour {n} positions de {enreg.itemsize} (fichier tronqué ?)")
    t = np.frombuffer(corps, dtype=enreg)
    ids, lps = t["paires"]["id"], t["paires"]["lp"]
    # Le format promet un ordre décroissant et des positions consécutives : un écrivain fautif est refusé ici, pas plus loin
    if n and (np.any(np.diff(lps, axis=1) > 0) or np.any(np.diff(t["pos"]) != 1)):
        raise ValueError(f"{chemin} : top-K non trié ou positions non consécutives")
    return {"k": k, "vocab": vocab, "n": n, "pos": t["pos"], "ref": t["ref"], "lp_ref": t["lp_ref"],
            "ids": ids, "lps": lps}


def nll(d: dict) -> float:
    return float(-np.mean(d["lp_ref"].astype(np.float64)))


def verifier(d: dict, tf_nll: float) -> bool:
    return abs(nll(d) - tf_nll) <= TOLERANCE_NLL


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    d = lire(sys.argv[1])
    print(f"K={d['k']} vocab={d['vocab']} positions={d['n']} (pos {d['pos'][0]}..{d['pos'][-1]}) NLL={nll(d):.6f}")
    if len(sys.argv) == 3:
        ok = verifier(d, float(sys.argv[2]))
        print(("CONTRÔLE tenu" if ok else "CONTRÔLE FAUX") + f" : −moyenne(logprob_ref) {nll(d):.6f} contre TF-NLL {sys.argv[2]}")
        sys.exit(0 if ok else 1)
