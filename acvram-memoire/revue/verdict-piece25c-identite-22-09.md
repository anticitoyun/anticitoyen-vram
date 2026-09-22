# Pièce 25 (c) identite — NON JUGÉ, corpus n'atteint pas les experts sans stats — 22/09 (poste2)

* instrument : `outils/gpu/mesure/invites-experts-sans-stats.py selectionner`, alias Qwen3-VL-30B-A3B-abl-nvfp4-vision (régime identite, `experts_sans_stats_liste` rafraîchi ce matin), corpus construit à partir de wiki-gptq.txt (250 paragraphes ≥90 mots, `scratchpad/piece25c-corpus-22-09.txt`, préparation locale — pas un corpus scellé existant)
* commit : main à jour (worktree poste2-w-21-09)
* régime : prefill b=1, graphes off, 200 invites notées, `ACVRAM_TRACE_ROUTAGE`
* scellé : alarme si score max des ciblées < 2 × part attendue (0,135) = 0,27 → « non jugé »
* mesuré : part attendue 0,135 ; **scores ciblées [0,067 ; 0,057 ; 0,057 ; 0,052 ; 0,05 ; 0,048 ; 0,047 ; 0,046]** — le meilleur (0,067) reste sous 0,27 ; témoins [0,006-0,006] cohérents (bien en dessous)
* verdict : **NON JUGÉ (alarme déclenchée)** — le corpus Wikipedia générique que j'ai préparé ne route pas assez vers les experts sans stats pour discriminer ciblées/témoins de façon significative. La chaîne decode-pas (16 invites) n'a pas été lancée : son résultat serait de toute façon non exploitable sous cette alarme, carte économisée.
* durée : ~4 min de carte (200 prefills de trace)

## Suite
Il faut un corpus qui route davantage vers les experts rares (vocabulaire spécialisé/rare plutôt que du texte encyclopédique courant) — à construire par qui connaît mieux la distribution de routage (poste1) ou reprendre le corpus de calibration lui-même (moins généraliste). PPL A/B 31B ensuite (file de la chef).
