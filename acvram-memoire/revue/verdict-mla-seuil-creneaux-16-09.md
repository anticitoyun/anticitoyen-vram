# Verdict — seuil des créneaux : FAUX pour tout b = SLOTS (1, 4, 5, 8, 12) SOUS GRAPHES, JUSTE en eager (b=4 : 27/28, b=1 : 7/7) → le défaut est dans la capture/rejeu du chemin à créneaux (GraphRunner), pas dans `static_load` ni dans l'attention

- **instrument** : `arbitre-prefill-mla-16-09.py` (prefill W4A16 rejoue prompt 128 + k jetons du bras, logit comparé au décodage à k ; k ∈ {1,2,4,8,16,32,63}) ; ties à `TIES_N_SEQ = b` ; sorties `scratchpad/slots-seuil-16-09/`
- **commit** : main **dcb09ba** (travail/poste3 f7d8afb) ; régime W4A16 prefill et décodage, `-k48`, `ACVRAM_HYBRID_SLOTS = b` à chaque fois (règle § 10)
- **régime** : graphes actifs (ties) ; contrôle **eager** (`TIES_EAGER=1`) à b=4/SLOTS=4 et b=1/SLOTS=1
- **scellé** (poste7 § 10) : faux dès 5 → `static_load`/magasin au-delà de 4 ; juste à 5 et 8, faux à 12 → godet 16 / `_SID_REMBOURRAGE`. (2) b=1 et b=4 du duel publiés seulement s'ils passent
- **mesuré** (top-1 prefill = décodage, sous graphes) : **b=1 : 5/7 · b=4 : 19/28 (68 %) · b=5 : 26/35 · b=8 : 38/56 · b=12 : 54/84 (64 %)** ; k=2 : 0/1, 2/4, 3/5, 2/8, 3/12 ; |Δ| médian 4-6. **Eager : b=4 27/28 (|Δ| ≤ 4,1), b=1 7/7 (|Δ| ≤ 2,0)**. Pour mémoire, tout-eager à b=12 (SLOTS=4 → `godet_hybride` rend None) : 81/84
- **verdict** : **aucun des deux motifs de poste7** — c'est faux **dès b=1**, pour tout nombre de créneaux, **seulement sous graphes** ; le chemin à créneaux est juste en eager. (2) **b=1 et b=4 du duel ne passent pas** (5/7, 19/28) : non publiés. Le duel entier (b=1, 4, 12 : `certifie` à SLOTS=b, graphes) tournait dans ce régime

## 1. Le motif temporel, pour poste4 (fichier:ligne à confirmer, pas une piste vague)
- k=1 (1er pas de décodage après le prefill) : 7-11/12 juste ; **k=2 : 0-3/12** ; k ≥ 32 : 10-12/12. Donc le premier pas lit correctement les 128 lignes chargées par `static_load` (`mla.py:345-355`) ; c'est à partir du **deuxième** pas que le rejeu ne voit plus l'état juste — la ligne écrite au pas 1 (`_ecrit_ligne`/`mla_ecrit_latent`, position `len`) ou `len` lui-même ne sont pas ceux que le graphe relit. Puis l'erreur se dilue (les lignes écrites par le décodage dominent).
- En eager, exactement le même code Python (`decode_static`, `decode_static_batch_complet`) est juste → ce qui diffère sous graphes : ce que le graphe a **capturé** au warm-up et relit tel quel — `GraphRunner._bind_hybrid` (`graphs.py:512-537`), la clé de graphe et le palier `godet_mla(len) + MLA_BUCKET` (`graphs.py:536`, `mla.py:89`) contre le `len` réel après `static_load`, et `_mla_lot` (`model.py:1708-1726`, table `ptrs`/`len_ptrs` construite et mise en cache par `data_ptr`). Le test de poste4 : après `static_bind` d'une vraie séquence, comparer `st["len"]`, `st["cache"].data_ptr()` et le palier utilisés par le graphe rejoué avec ceux du créneau — et refaire cet arbitre : eager = graphes exigé.
- Le prompt d'un jeton ne montre presque rien (PPL décodage +0,3 %) : c'est pourquoi toutes les PPL de décodage du jour étaient aveugles, comme les ties sur invites synthétiques.

## 2. Conséquences
- **Duel prise A** : les trois lots (b=1, 4, 12) sont à refaire après correctif — ou, en attendant, mesurables **en eager à créneaux** si poste7 veut un chiffre juste tout de suite (plus lent, mais vrai).
- Les nsys 43,9 → 17,2 ms (graphes, SLOTS=12) : temps d'un rejeu numériquement faux ; à remesurer après correctif — le gain des commits MLA 1-3 sera à reprouver dans le même passage.
- Règle (§ 4 bis, proposée en plus) : **eager = graphes** par cet arbitre pour tout chemin à créneaux, avant toute capture publiée.
