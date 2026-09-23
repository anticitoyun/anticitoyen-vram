# Pièce 44, dispersion b=1 (client corrigé) — décroissante-puis-plateau — 22/09 (Manon)

* instrument : `chaine-energie-4moteurs.sh --bras "acvram:1:4"`, sha `b4930600` (client `banc-llamacpp-16-09.py` corrigé : `courtes.sort()` n'efface plus l'ordre chronologique)
* mesuré : `passes_courtes_jetons_s` dans l'ordre chronologique : **[353,9 ; 336,9 ; 333,6 ; 273,3 ; 272,2 ; 272,1 ; 272,4]** — forme **décroissante puis plateau** (chute nette sur les 4 premières sondes, stable ensuite à ≈272 t/s). Charge hôte stable (`load1` 4,38→4,14, aucune tendance marquée).
* lecture (table Océane, `d76cbb14`) : **décroissante-puis-plateau → capture de graphe à un franchissement de godet** (128/256/512 jetons) — pas un plafond `MAX_GRAPHS` (qui donnerait croissante) ni un effet de charge hôte seul (qui donnerait sans tendance, or `load1` est stable ici).
* **limite instrumentale nommée** : `captures`, `replays`, `len(graphs)` (moteur, `graphs.py:292-293`) ne sont exposés ni par `/metrics` (`EngineStats.to_dict()` ne les porte pas) ni par une ligne de régime — seuls des prints verbeux par-rejeu existent, gardés derrière `ACVRAM_TRACE_ENTREES=1` (bruit per-token, pas des compteurs avant/après). Je n'ai donc PAS pu relever ces trois compteurs comme demandé — seule la forme de la série et `load1` sont publiés.
* verdict : **forme décroissante-puis-plateau confirmée**, cohérente avec une capture de graphe déclenchée au franchissement d'un godet de contexte pendant les premières sondes, stable une fois la capture faite.
* durée : ~2 min de carte

## Suite (en pause)
Compteurs captures/replays/len(graphs) à exposer par un accès léger (à Océane) si la pièce 44 doit être creusée davantage. Groupe en pause : reconversion alpha2 et chrono 30B-VL non entamés.
