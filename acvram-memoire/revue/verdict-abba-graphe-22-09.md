# ABBA sampler b=12 Coder : A(lent)/B(graphe) — INTERMÉDIAIRE, B gagne +1,24 % — 22/09 (poste2)

* instrument : `scratchpad/poste4-b12-21-09/chaine-sampler-abba.sh` corrigé (A=défaut, B=`ACVRAM_SAMPLER_GRAPHE=1`, remplace l'ancien `ACVRAM_SAMPLER_LENT` qui comparait par erreur lent contre lent) ; worktree `poste2-w-21-09`, répertoire `sorties-abba/` absent au 1er essai du soir (aucun rapport avec un permission menu `rm`, corrigé par `mkdir -p`)
* commit : 3a925976
* régime : Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c, b=12, `--max-model-len 2304`, ordre A0/B0 (blanc) puis A1 B1 B2 A2 A3 B3 B4 A4
* scellé (poste1) : ABBA t/s B/A ≥ 1,000 (le levier ne doit pas coûter) ; rejet si |horloge_medA−horloge_medB|/horloge_medA > 3 %
* mesuré : A = 1515,9/1533,2/1546,1/1528,9 (méd 1531,1) ; B = 1556,1/1542,4/1570,4/1544,1 (méd 1550,1) ; **ratio B/A = 1,0124** (+1,24 %) ; `ecart_horloge_pct = 0,0` mais **`horloge_med` identique (210 MHz) sur les 8 fenêtres** — valeur non plausible au regard des t/s sains (1500-1570), limite d'instrument de calcul de médiane sur la sortie multi-lignes `nvidia-smi`, à corriger avant de s'y fier pour un futur rejet (ici elle ne fausse rien : identique des deux côtés, ne motive ni rejet ni faux TENU)
* verdict : script = **INTERMEDIAIRE** ; B ≥ A tenu (+1,24 % > seuil ≥1,000), cohérent avec la frontière (2) TENUE (+53,95 µs/pas, prédiction +0,6 à +1,0 %, mesuré légèrement au-dessus mais dans le bruit de 4 paires/bras). Pas de coût mesuré pour `sampler=graphe`.
* durée : 10 min (03:10:06-03:20:04)

## Suite
Scellé E ensuite (chaîne déjà écrite). Le calcul `horloge_med` de `chaine-sampler-abba.sh` (fonction `horloge()`, sortie `nvidia-smi --query-gpu=clocks.sm`) est à revoir par qui le maintient — rendu 210 constant, probablement un artefact de tri sur une sortie multi-lignes inattendue, jamais nommé jusqu'ici car il n'a pas encore fait rejeter un verdict de bonne foi.
