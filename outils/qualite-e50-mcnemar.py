#!/usr/bin/env python3
"""e50.3 § 4 (poste4, 01/10) -- McNemar ITEM PAR ITEM entre deux passes du témoin, à la place
de la garde de cohérence simple (écart de S > 0,05) de `campagne-qualite-e50.py`.

Pourquoi la garde simple ne suffit pas : un écart de S peut rester sous le seuil alors que
l'instrument a dérivé item par item (ex. 15 items justes→faux ET 15 faux→justes entre deux
passes : S identique, 30 items ont pourtant changé de verdict chacun). Le delta de S global ne
peut PAS détecter ce cas par construction (les deux sens se compensent dans la moyenne) ; un
McNemar item par item le détecte toujours, puisqu'il compte les discordances dans CHAQUE sens
séparément plutôt que leur somme. Voir `tests/test_qualite_e50_mcnemar.py::
test_detecte_une_derive_masquee_par_un_S_identique` pour un cas construit où S(a) == S(b) mais
McNemar rend p << 0,05 (la garde simple laisserait passer, McNemar arrête).

Usage (bibliothèque, appelée par campagne-qualite-e50.py) -- le nom de fichier porte des tirets
(convention e50.3, comme `qualite-e50-bareme.py`) donc jamais `import qualite_e50_mcnemar` :
    import importlib.util
    _s = importlib.util.spec_from_file_location("qe50_mcnemar", "outils/qualite-e50-mcnemar.py")
    mcnemar_lib = importlib.util.module_from_spec(_s); _s.loader.exec_module(mcnemar_lib)
"""
import json
import math
import os
import subprocess
import tempfile


def _lire_jsonl(chemin):
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                yield json.loads(ligne)


def corrects_mmlu_gsm8k(chemin_jsonl, cle_metrique="exact_match"):
    """Un sample lm-eval `--log_samples` porte sa métrique calculée directement dans le jsonl
    (ex. "exact_match": 1.0 ou 0.0, posée par le paquet au moment du scoring) -- pas besoin de
    refaire le filtre ici, juste de le lire. Renvoie {doc_id: correct (bool)}."""
    out = {}
    for item in _lire_jsonl(chemin_jsonl):
        doc_id = item.get("doc_id")
        if doc_id is None:
            continue
        out[doc_id] = bool(item.get(cle_metrique))
    return out


def corrects_humaneval(chemin_jsonl):
    """pass@1 PAR ITEM, exécuté sous bac à sable (jamais nu, § 2 bis -- même garantie que
    `qualite-e50-humaneval-score.py`, qui ne rend que l'agrégat ; ce module garde le détail par
    doc_id, nécessaire à McNemar). Renvoie {doc_id: réussi (bool)}."""
    ici = os.path.dirname(os.path.abspath(__file__))
    bac_a_sable = os.path.join(ici, "bac-a-sable-humaneval.sh")
    out = {}
    for item in _lire_jsonl(chemin_jsonl):
        doc_id = item.get("doc_id")
        if doc_id is None:
            continue
        doc = item.get("doc", item)
        completion = (item.get("filtered_resps") or item.get("resps") or [[""]])[0]
        if isinstance(completion, list):
            completion = completion[0]
        entry_point = doc.get("entry_point", "")
        programme = f"{doc['prompt']}{completion}\n\n{doc['test']}\n\ncheck({entry_point})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(programme)
            chemin_py = fh.name
        try:
            r = subprocess.run(["bash", bac_a_sable, chemin_py], capture_output=True, text=True, timeout=20)
            out[doc_id] = (r.returncode == 0)
        finally:
            os.unlink(chemin_py)
    return out


def mcnemar(a, b):
    """(discordants_b, discordants_c, p) sur les doc_id COMMUNS à `a` et `b` (deux {doc_id:
    bool}). b = juste en a, faux en b ; c = faux en a, juste en b. Test exact binomial (b+c
    essais, b succès, p=0,5 sous H0 "pas de dérive directionnelle") -- équivalent au McNemar
    exact pour les petits effectifs de la méthode (170 items, b+c rarement > 30) ; pas de
    dépendance à scipy (absent de ce poste, aucun `.venv-panel`)."""
    communs = set(a) & set(b)
    disc_b = sum(1 for d in communs if a[d] and not b[d])
    disc_c = sum(1 for d in communs if not a[d] and b[d])
    n = disc_b + disc_c
    if n == 0:
        return disc_b, disc_c, 1.0
    k = min(disc_b, disc_c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return disc_b, disc_c, min(p, 1.0)


def mcnemar_global(corrects_a_par_tache: dict, corrects_b_par_tache: dict):
    """Fusionne TOUTES les tâches (MMLU×3 + GSM8K + HumanEval) en un seul test McNemar global
    (méthode § 4 : « McNemar p > 0,05 sur les 5 tâches », lu ici comme un seul test sur
    l'ensemble des 170 items -- une tâche à part aurait trop peu d'items (30-40) pour qu'un test
    exact soit discriminant). Préfixe chaque doc_id par le nom de tâche pour éviter toute
    collision d'ids entre tâches."""
    a, b = {}, {}
    for tache, corrects in corrects_a_par_tache.items():
        a.update({f"{tache}:{d}": v for d, v in corrects.items()})
    for tache, corrects in corrects_b_par_tache.items():
        b.update({f"{tache}:{d}": v for d, v in corrects.items()})
    return mcnemar(a, b)


TAUX_DISCORDANCE_MAX = 0.10   # méthode §4/E1 (poste6, scellé) : « réponses identiques ≥ 90 % »


def temoin_tenu(corrects_a_par_tache: dict, corrects_b_par_tache: dict):
    """Verdict du témoin (méthode § 4, E1) : TENU seulement si p > 0,05 ET le taux de
    discordance reste ≤ 10 % (E1 : « réponses identiques ≥ 90 % »). Les deux conditions sont
    nécessaires : p seul rate une dérive SYMÉTRIQUE (autant de items justes→faux que
    faux→justes -- p proche de 1 par construction, voir test_qualite_e50_mcnemar.py) ; le taux
    de discordance seul raterait une dérive directionnelle à faible effectif. Renvoie
    (tenu: bool, discordants_b, discordants_c, p, taux_discordance)."""
    a, b = {}, {}
    for tache, corrects in corrects_a_par_tache.items():
        a.update({f"{tache}:{d}": v for d, v in corrects.items()})
    for tache, corrects in corrects_b_par_tache.items():
        b.update({f"{tache}:{d}": v for d, v in corrects.items()})
    disc_b, disc_c, p = mcnemar(a, b)
    n_communs = len(set(a) & set(b))
    taux = (disc_b + disc_c) / n_communs if n_communs else 1.0
    tenu = (p > 0.05) and (taux <= TAUX_DISCORDANCE_MAX)
    return tenu, disc_b, disc_c, p, taux


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: qualite-e50-mcnemar.py <samples_a.jsonl> <samples_b.jsonl>", file=sys.stderr)
        raise SystemExit(64)
    a = corrects_mmlu_gsm8k(sys.argv[1])
    b = corrects_mmlu_gsm8k(sys.argv[2])
    disc_b, disc_c, p = mcnemar(a, b)
    print(json.dumps({"discordants_b": disc_b, "discordants_c": disc_c, "p": p}))
