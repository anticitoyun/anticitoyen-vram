"""Garde-fou partagé pour les bancs : refuser de mesurer un moteur dégradé
sans le savoir (bead runner, 14/09 soir -- chef, après qu'un deuxième
chargement dans le même processus a silencieusement coupé les graphes CUDA
et réparti le modèle sur 2 cartes pendant une mesure).

Usage dans un banc, juste après `Engine(...)` (et après `warm_graphs()` si
le banc en fait un -- sinon `piles_ok` reste "?" et n'est pas jugé) :

    from regime import exiger_regime_nominal      # sibling de outils/, pas un paquet
    exiger_regime_nominal(engine)
"""
import sys


def exiger_regime_nominal(engine, autoriser_piles_inconnues: bool = True) -> None:
    """Lève `RuntimeError` si le régime n'est pas nominal, avec le régime
    imprimé -- jamais un refus muet. `autoriser_piles_inconnues=True` (par
    défaut) : un banc qui n'a pas encore fait tourner de pas GPU par couche
    MoE (pas de `warm_graphs()`/`step()` avant l'appel) verrait `piles_ok`
    a "?" pour chaque couche -- ce n'est pas une dégradation, juste une
    vérification pas encore faite. Poser `False` pour l'exiger prouvé "oui".
    """
    r = engine.regime()
    fautes = []
    if not r["graphes"]:
        fautes.append(f"graphes désactivés ({r['graphes_raison']})")
    if r["couches_exilees"]:
        fautes.append(f"{r['couches_exilees']}/{r['couches_total']} couches exilées")
    if r["experts_exiles"]:
        fautes.append(f"{r['experts_exiles']}/{r['experts_total']} experts exilés")
    if r["piles_ok"] is False:
        fautes.append("pile(s) d'experts hétérogène(s), repli eager")
    elif r["piles_ok"] is None and not autoriser_piles_inconnues:
        fautes.append("piles_ok non vérifié (pas de pas GPU avant l'appel)")
    if len(r["cartes"]) > 1:
        fautes.append(f"modèle réparti sur {len(r['cartes'])} cartes : {r['cartes']}")

    if fautes:
        print(f"[regime] {engine.regime_ligne()}", file=sys.stderr, flush=True)
        raise RuntimeError(
            "régime dégradé, mesure refusée : " + " ; ".join(fautes))
