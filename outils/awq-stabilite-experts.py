"""Pièce 25 (a) : l échelle AWQ d un expert MoE est-elle stable en fonction du
nombre de jetons que le corpus lui a routés ? Courbe et seuil de stabilité
MESURÉS — aujourd hui `MIN_ECHANTILLONS_AWQ = 8` (convert.py) est un seuil
écrit sans mesure, et 13,5 % des experts du 30B-VL n y arrivent pas.

Méthode : statistiques collectées sur des préfixes croissants du corpus
(`--jetons 100,1000,10000` : nombre total de jetons de calibration), la
plus grande sert de référence ; pour un échantillon d experts (stratifié par
nombre de jetons routés dans la référence), l échelle AWQ `s_n` obtenue avec
les stats du préfixe n est comparée à `s_ref` : écart relatif
||s_n − s_ref|| / ||s_ref|| (par expert), et erreur de sortie relative de la
quantification (`search_channel_scales` rend les deux). Sortie : par
préfixe, la courbe écart médian / p90 en fonction du nombre de jetons routés
à l expert (classes 1-7, 8-31, 32-127, 128-511, ≥ 512), et le SEUIL DE
STABILITÉ = la plus petite classe dont l écart médian ≤ `--tolerance`
(défaut 0,10, écrit avant la mesure), avec la part des experts sous ce seuil.

Ce qui rend « faux » : écart médian > tolérance même à ≥ 512 jetons (la
recherche AWQ n est pas stable du tout, le repli médiane n a pas de sens) ;
ou écart ≤ tolérance dès 1-7 jetons (MIN_ECHANTILLONS_AWQ = 8 jette des
statistiques utilisables, le repli est inutile).

Usage (carte, Manon ; le 30B sur processeur est trop lent pour 10 000 jetons) :
  outils/carte.sh python outils/awq-stabilite-experts.py MODELE_SOURCE --jetons 100,1000,10000
        [--experts 240] [--calib-file F] [--device cuda:0] [--json sortie.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

CLASSES = [(1, 7), (8, 31), (32, 127), (128, 511), (512, 10 ** 9)]
RE_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def classe_de(n: int) -> str:
    for a, b in CLASSES:
        if a <= n <= b:
            return f"{a}-{b}" if b < 10 ** 9 else f">={a}"
    return "0"


def decouper(calib: list[list[int]], jetons: int) -> list[list[int]]:
    """Le préfixe du corpus tokenisé qui totalise `jetons` (séquences entières, la dernière coupée)."""
    out, total = [], 0
    for seq in calib:
        if total >= jetons:
            break
        reste = jetons - total
        out.append(seq[:reste] if len(seq) > reste else seq)
        total += len(out[-1])
    return out


class LecteurPoids:
    """Le poids `nn.Linear` d un expert nommé `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`,
    quelle que soit la disposition des shards : (1) par expert, tel quel ;
    (2) sous `model.language_model.` (enrobage multimodal) ; (3) blob hub
    ≥ 5 `experts.gate_up_proj` [E, H, 2I] / `experts.down_proj` [E, I, H],
    découpé par `convert._expert_depuis_blob` (la même définition que le
    convertisseur et la collecte : même poids sous le même nom). None si
    aucune des trois — l appelant compte et refuse à zéro."""
    def __init__(self, dossier: str):
        from safetensors import safe_open
        self._open = safe_open
        self.dossier = dossier
        self.index = {}
        for fn in sorted(os.listdir(dossier)):
            if fn.endswith(".safetensors"):
                with safe_open(os.path.join(dossier, fn), framework="pt", device="cpu") as fh:
                    for k in fh.keys():
                        self.index[k] = fn
        self._blobs: dict = {}

    def _lire(self, cle: str):
        with self._open(os.path.join(self.dossier, self.index[cle]), framework="pt", device="cpu") as fh:
            return fh.get_tensor(cle)

    def charger(self, nom: str):
        m = RE_EXPERT.match(nom)
        if not m:
            return None
        for cle in (nom, nom.replace("model.", "model.language_model.", 1)):
            if cle in self.index:
                return self._lire(cle)
        from acvram.quant.convert import _expert_depuis_blob
        couche, e, proj = m.group(1), int(m.group(2)), m.group(3)
        blob_nom = "gate_up_proj" if proj in ("gate_proj", "up_proj") else "down_proj"
        for prefixe in ("model.", "model.language_model."):
            cle = f"{prefixe}layers.{couche}.mlp.experts.{blob_nom}"
            if cle in self.index:
                if cle not in self._blobs:
                    self._blobs = {cle: self._lire(cle)}          # un blob à la fois en mémoire (jusqu à 200 Mo)
                return _expert_depuis_blob(proj, self._blobs[cle][e])
        return None


def echelles(poids: "torch.Tensor", st, fmt: str, group_size: int, quant_act: bool):
    from acvram.quant.calibrate import search_channel_scales
    scaler, err = search_channel_scales(poids, st, fmt, group_size=group_size, n_grid=20,
                                        quantize_activation_nvfp4=quant_act)
    import torch
    s = scaler.scale if scaler.scale is not None else torch.ones(poids.shape[1], dtype=torch.float32)
    return s.to(torch.float32).cpu(), float(err)


def mesurer(a) -> dict:
    import torch
    from safetensors import safe_open
    from acvram.engine.config import load_model_spec
    from acvram.quant.collect import collect_activation_stats, load_calib_ids
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(a.model, os.path.basename(a.model.rstrip("/")))
    tok = load_tokenizer(a.model)
    if tok is None:
        sys.exit("ÉCHEC / CAUSE : pas de tokenizer.json / SUITE : --model")
    tailles = sorted(int(x) for x in a.jetons.split(","))
    n_seqs = -(-tailles[-1] // a.calib_len)
    calib = load_calib_ids(tok, a.calib_file, n_seqs, a.calib_len, spec.vocab_size)
    stats_par_taille = {}
    for n in tailles:
        t0 = time.time()
        stats_par_taille[n] = collect_activation_stats(a.model, spec, decouper(calib, n), device=a.device)
        print(f"[stabilite] stats sur {n} jetons : {len(stats_par_taille[n])} tenseurs en {time.time() - t0:.0f} s", flush=True)
    ref = stats_par_taille[tailles[-1]]
    experts_ref = {k: v for k, v in ref.items() if RE_EXPERT.match(k) and v is not None and v.n_samples >= 1}
    # échantillon stratifié par classe de jetons routés dans la référence
    par_classe: dict[str, list[str]] = {}
    for k, v in experts_ref.items():
        par_classe.setdefault(classe_de(v.n_samples), []).append(k)
    rng = random.Random(0)
    retenus = []
    par_cl = max(1, a.experts // max(1, len(par_classe)))
    for cl, noms in par_classe.items():
        rng.shuffle(noms)
        retenus += noms[:par_cl]
    # poids des experts retenus, lus aux shards par `charger_poids` (par expert,
    # préfixe multimodal, ou blob hub ≥ 5 découpé — 22/09 : 0 expert lu sur
    # 17 235 parce que la source déquantifiée porte `experts.gate_up_proj` [E, H, 2I])
    lecteur = LecteurPoids(a.model)
    lignes = []
    non_lus = []
    for i, nom in enumerate(retenus):
        w = lecteur.charger(nom)
        if w is None:
            non_lus.append(nom)
            continue
        w = w.to(a.device)
        s_ref, err_ref = echelles(w, ref[nom], "nvfp4", a.group_size, True)
        ligne = {"expert": nom, "n_ref": int(ref[nom].n_samples), "classe_ref": classe_de(int(ref[nom].n_samples)),
                 "err_ref": err_ref, "par_taille": {}}
        for n in tailles[:-1]:
            st = stats_par_taille[n].get(nom)
            n_n = int(st.n_samples) if st is not None else 0
            if st is None or n_n < 1:
                ligne["par_taille"][n] = {"n": n_n, "ecart": None, "err": None}
                continue
            s_n, err_n = echelles(w, st, "nvfp4", a.group_size, True)
            ecart = float((s_n - s_ref).norm() / s_ref.norm().clamp(min=1e-12))
            ligne["par_taille"][n] = {"n": n_n, "classe": classe_de(n_n), "ecart": ecart, "err": err_n}
        lignes.append(ligne)
        if (i + 1) % 40 == 0:
            print(f"[stabilite] {i + 1}/{len(retenus)} experts", flush=True)
    if not lignes:
        sys.exit(f"INVALIDE : 0 expert lu sur {len(retenus)} retenus (premier : {non_lus[0] if non_lus else '?'}) — "
                 f"disposition des poids non reconnue par LecteurPoids ; classes vues : {sorted(par_classe)}")
    return {"modele": a.model, "jetons": tailles, "tolerance": a.tolerance, "experts_mesures": len(lignes),
            "experts_non_lus": len(non_lus), "experts_total_ref": len(experts_ref),
            "par_classe_ref": {cl: len(v) for cl, v in par_classe.items()},
            "lignes": lignes, **courbe(lignes, a.tolerance)}


def courbe(lignes: list, tolerance: float) -> dict:
    """Par CLASSE de jetons routés (dans le préfixe), médiane et p90 de l écart
    à l échelle de référence, tous préfixes confondus ; seuil de stabilité =
    plus petite classe dont la médiane ≤ tolérance."""
    import statistics
    par_classe: dict[str, list[float]] = {}
    for l in lignes:
        for n, r in l["par_taille"].items():
            if r["ecart"] is not None:
                par_classe.setdefault(r["classe"], []).append(r["ecart"])
    ordre = [f"{a}-{b}" if b < 10 ** 9 else f">={a}" for a, b in CLASSES]
    table = {}
    for cl in ordre:
        v = sorted(par_classe.get(cl, []))
        if v:
            table[cl] = {"n": len(v), "ecart_mediane": round(statistics.median(v), 4),
                         "ecart_p90": round(v[int(0.9 * (len(v) - 1))], 4)}
    seuil = next((cl for cl in ordre if cl in table and table[cl]["ecart_mediane"] <= tolerance), None)
    if not table:
        verdict = "INVALIDE (aucune classe mesurée)"
    elif seuil is None:
        verdict = "RÉFUTÉ : l échelle n est stable dans aucune classe — la recherche AWQ n est pas stable, repli médiane sans objet"
    elif seuil == ordre[0]:
        verdict = "RÉFUTÉ : stable dès 1-7 jetons — MIN_ECHANTILLONS_AWQ = 8 jette des statistiques utilisables"
    else:
        verdict = f"TENU : seuil de stabilité mesuré = {seuil} jetons routés (médiane ≤ {tolerance})"
    return {"courbe": table, "seuil_stabilite": seuil, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--jetons", default="100,1000,10000")
    ap.add_argument("--experts", type=int, default=240)
    ap.add_argument("--calib-file")
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = mesurer(a)
    if a.json:
        sortie = {k: v for k, v in r.items() if k != "lignes"}
        sortie["lignes"] = r["lignes"]
        json.dump(sortie, open(a.json, "w"), indent=1)
    print(f"[stabilite] {r['experts_mesures']} experts sur {r['experts_total_ref']} ; tolérance {r['tolerance']}")
    for cl, t in r["courbe"].items():
        print(f"  {cl:>8s} jetons : écart médian {t['ecart_mediane']:.3f}  p90 {t['ecart_p90']:.3f}  (n = {t['n']})")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
