# Banc unified_attention a/b/c — INSTRUMENT FAUX au critère scellé ; lectures : b=12 sans levier, b=1 −28 % pour le noyau vLLM — 23/09 (poste1)

* **instrument** : `outils/gpu/mesure/banc-attn-unifie.py`. L2 froid (48 caches distincts), 48 appels rejoués en un graphe, mur/48 (médiane de 30 rejeux) ; en plus, la somme des noyaux CUPTI en eager, comparable à nsys. Référence fp64 lue du même cache.
  * (a) vLLM 0.29.0 tel que servi : fp8 par tenseur, disposition LBNHC, 16 segments.
  * (b) le même noyau en INT8_PER_TOKEN_HEAD sur notre cache.
  * (c) notre `paged_attention(compact)` au défaut, réduction déroulée à 8 warps.
* **commit** : cf11fecc (le banc imprime les chemins des modules et la garde d'arbre). Prise 1 à ffe85761, invalide : la garde « godet recopié ≠ bucket_blocks » a arrêté le bras (c) ; le log est gardé.
* **régime** : -lgc 2700, horloge moyenne sous charge 2 688 MHz. Au début et à la fin de la prise, seul llama-server 4627 tourne. Triton 3.7.1 pour (a) et (b), 3.8.0 pour (c) : confondu nommé, non levé.
* **scellé** : `scratchpad/poste1-unifie-23-09/scelle.md`, commit 2fecd6d5, écrit avant la prise.
* **mesuré** (µs par couche, mur de graphe ; noyaux CUPTI entre parenthèses) :

| cellule | (a) vLLM fp8 | (b) vLLM int8 sur notre cache | (c) acvram | prédit (a) / (b) / (c) |
|---|---|---|---|---|
| **b=12 ctx 768** | **14,42** (15,06) | **16,58** (16,58) | **14,11** (14,29) | 11,4-13,9 / 12,0-14,5 / 14,0-15,0 |
| **b=1 ctx 768** | 5,95 (5,64) | **5,82** (5,52) | **8,07** (7,90) | 5-8 / 5-9 / 7-9 |
| b=12 ctx 320 | 9,15 | 9,44 | 9,77 | — |
| b=12 ctx 1 216 | 19,63 | 22,08 | 20,82 | — |

  Justesse, erreur relative max contre fp64 : (a) 3,4e-3, (b) 4,4e-3, (c) 2,0e-3. Tous les bras sont sous 2⁻⁸, aucun n'est cassé. (b) fait deux fois l'erreur de (c) : c'est le P en bf16, comme prévu.

* **verdict** :
  1. **Instrument faux au critère scellé.** (a) donne 14,42 µs à b=12 ctx 768, hors de la bande 11,4-13,9 fixée d'après la trace servie de la p91 (12,66). L'écart au trace grandit avec le contexte : +8 % à ctx 320, +14 % à 768, +30 % à 1 216 (ce dernier point vient d'un extrapolé). **Je ne scelle ni GO ni RÉFUTÉ.**
  2. **Lectures non scellées.** Toutes vont dans le même sens à b=12 :
     * le noyau vLLM n'est pas plus rapide que le nôtre à formes et cache égaux ;
     * sur notre int8, il est même 2,5 µs plus lent, soit 17 % ((b) contre (c)) ;
     * et (b) − (a) = 2,2 µs ≥ 1,5 : c'est le cas « format ».

     Si l'instrument est juste, **le port ne gagne rien à b=12**. L'écart servi d'attention de la p91 (+0,145 ms/pas) ne viendrait alors pas de la forme du noyau, mais de ce que le banc ne reproduit pas du service vLLM.
  3. **b=1, lecture inattendue.** Le noyau vLLM sur notre cache fait 5,82 µs contre 8,07 pour le nôtre, soit −28 % (−2,25 µs/couche, ≈ −0,11 ms/pas, ≈ 3,4 % du pas b=1). J'avais prédit « 0 à +2 %, régression possible » : c'est l'inverse de ma prédiction.
     * Mécanisme probable : 64 programmes de 3 tuiles de 16 jetons chez vLLM, contre 48 programmes d'une tuile de 64 chez nous. À b=1, c'est la latence par programme qui compte.
     * La même réserve d'instrument s'applique ici.
  4. **Hypothèses sur l'écart banc/trace**, à trancher avant toute décision :
     * H1 : dans la trace vLLM, l'indice du pas dans le lot n'est pas le contexte (p91 : ctx = 256 + i, mais la trace vLLM s'arrête vers 1 060) ;
     * H2 : dans le service vLLM, les séquences partagent physiquement les blocs du préfixe commun (cache de préfixe activé), ce que le banc ne reproduit pas (blocs disjoints). À noter : H2 prédit un gain absolu constant, alors que l'écart mesuré grandit avec le contexte, ce qui plaide plutôt pour H1 ;
     * H3 : la disposition entrelacée K|V et le fp8 se comportent autrement en service.

     Le contrôle qui tranche : lancer (a) sous nsys avec des longueurs connues et comparer à la trace p91 de même contexte ; puis relire les invites du client vLLM (préfixe partagé ou non).
* **ce que cela ordonne (au chef)** : ne pas porter le noyau pour b=12 tant que (1) n'est pas levé. Il existe un levier b=1 candidat : des tuiles de 16 jetons et des segments fixes tirés de `slen`, limités aux petits lots. Il reste à confirmer au même banc, une fois l'instrument réparé.
* **durée** : prévue 1 h de script + 5 min de carte ; tenue ≈ 45 min, carte 2 prises de 6 s (16:03:41-46, 16:04:22-28) entre deux prises de poste2 (verrou libre depuis 15:38).
