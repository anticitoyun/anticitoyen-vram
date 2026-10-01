#!/usr/bin/env python3
"""e50.3 § 3 (poste6, scellé AVANT toute mesure — ne se resserre jamais après) — barème étoiles
↔ score composite S = (acc_MMLU90 + acc_GSM8K40 + pass1_HumanEval40) / 3.

`format`, écrit en premier (prioritaire sur tout calcul de S) : ≥ 50 % de `[invalid]` sur MMLU
(90 items) ou ≥ 50 % de réponses vides toutes tâches confondues — l'instrument n'a pas lu le
modèle (gabarit, `</think>` jamais fermé, fenêtre), ce n'est jamais une étoile.

Chaque palier a sa garde (aucune tâche sous un plancher) ; une garde manquée descend d'UNE
étoile (pas directement en « format »)."""

PALIERS = [
    (0.80, "★★★★★", 0.60),
    (0.65, "★★★★", 0.40),
    (0.50, "★★★", None),
    (0.35, "★★", None),
    (0.0, "★", None),
]
ORDRE_ETOILES = [p[1] for p in PALIERS]


def etoile(S, acc_mmlu, acc_gsm8k, pass1_humaneval, invalid_mmlu=0, n_mmlu=90,
          reponses_vides=0, n_total=170):
    """`S` et les trois exactitudes dans [0, 1]. Rend la chaîne d'étoiles (ou « format »)."""
    if n_mmlu and invalid_mmlu / n_mmlu >= 0.5:
        return "format"
    if n_total and reponses_vides / n_total >= 0.5:
        return "format"
    exactitudes = (acc_mmlu, acc_gsm8k, pass1_humaneval)
    for i, (seuil, etoiles, plancher) in enumerate(PALIERS):
        if S >= seuil:
            if plancher is not None and min(exactitudes) < plancher:
                return ORDRE_ETOILES[min(i + 1, len(ORDRE_ETOILES) - 1)]
            return etoiles
    return "★"   # inatteignable (PALIERS couvre [0, +inf)), gardé pour la lisibilité


if __name__ == "__main__":
    import sys
    S, mmlu, gsm8k, he = (float(x) for x in sys.argv[1:5])
    print(etoile(S, mmlu, gsm8k, he))
