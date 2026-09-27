#!/usr/bin/env python3
"""Pièce 261 : fusionne les `results_*.json` de lm-eval (un pour MMLU, un pour GSM8K — deux
invocations séparées dans `panel-taches.sh` car `--limit` est global à une invocation, et les
deux groupes veulent des limites différentes) en un résumé : score par tâche, moyenne, pire
tâche. Le score retenu par tâche est la métrique PRINCIPALE de sa `metric_list` (celle marquée
`higher_is_better`, en écartant les `_stderr`) — jamais une métrique choisie au hasard parmi
celles rapportées.

Usage : panel-taches-resume.py <results_mmlu.json> <results_gsm8k.json> <sortie.json>
"""
import json
import sys


def score_principal(nom_tache, valeurs, configs):
    """Cherche, dans les métriques rapportées pour une tâche, celle qui correspond à sa
    `metric_list`/`filter_list` (config lm-eval) — écarte les `_stderr` et toute clé non
    numérique. Une clé de résultat est `métrique,filtre` (ex. `exact_match,strict-match`) :
    GARDÉE ENTIÈRE (pas seulement la métrique) — GSM8K a deux filtres du MÊME `exact_match`
    (`strict-match`, `flexible-extract`) qui peuvent diverger ; les fusionner en une seule clé
    `exact_match` ferait perdre l'un des deux au hasard de l'ordre d'itération. Le filtre
    retenu est le PREMIER déclaré dans `filter_list` de la config (convention lm-eval : GSM8K
    déclare `strict-match` en premier, c'est le score officiel du leaderboard) ; `acc` reste
    préférée pour les tâches multiple_choice (pas de `filter_list` déclarée, `acc,none` seul)."""
    cfg = configs.get(nom_tache, {})
    filtres_config = [f["name"] for f in cfg.get("filter_list", [])]
    metriques_config = [m["metric"] for m in cfg.get("metric_list", [])]
    candidats = {k: v for k, v in valeurs.items()
                 if not k.endswith("_stderr,none") and isinstance(v, (int, float))}
    for m in (["acc"] + metriques_config):
        for f in (filtres_config or ["none"]):
            cle = f"{m},{f}"
            if cle in candidats:
                return cle, candidats[cle]
    if candidats:
        cle = next(iter(candidats))
        return cle, candidats[cle]
    raise ValueError(f"{nom_tache} : aucune métrique numérique trouvée dans {valeurs}")


def charger(chemin):
    d = json.load(open(chemin, encoding="utf-8"))
    return d.get("results", {}), d.get("configs", {})


def main():
    chemin_mmlu, chemin_gsm8k, sortie = sys.argv[1], sys.argv[2], sys.argv[3]
    resultats_m, configs_m = charger(chemin_mmlu)
    resultats_g, configs_g = charger(chemin_gsm8k)
    resultats = {**resultats_m, **resultats_g}
    configs = {**configs_m, **configs_g}

    par_tache = {}
    for nom, valeurs in resultats.items():
        metrique, score = score_principal(nom, valeurs, configs)
        par_tache[nom] = {"metrique": metrique, "score": round(float(score), 4)}

    if not par_tache:
        raise SystemExit("REFUS : aucune tâche dans les résultats fusionnés")

    scores = [v["score"] for v in par_tache.values()]
    pire_tache = min(par_tache, key=lambda t: par_tache[t]["score"])
    r = {"taches": par_tache, "moyenne": round(sum(scores) / len(scores), 4),
         "pire_tache": pire_tache, "pire_score": par_tache[pire_tache]["score"],
         "n_taches": len(par_tache)}
    json.dump(r, open(sortie, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULTAT_PANEL", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
