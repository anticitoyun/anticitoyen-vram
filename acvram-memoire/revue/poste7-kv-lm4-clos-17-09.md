# poste7 — KV lm4 : réfuté, clos ; une passe de cause de 25 min (K seul, V seul, puits exempté), sans réouverture ; tq3+1 n'existe qu'après B et seulement si la perte est dans K (17/09)

Entrée : poste3 3d99ee9 `verdict-kv-lm4-qualite-17-09` : lm4 / int8 = 1,0215 ± 0,013 (> 1,005 et > + 0,010, sur chaque quart de tranche), lm2 = 2,99× (l'instrument voit le cache), int8 / bf16 = 0,998 (au seuil, non significatif, chemins d'attention différents), capacité 1,97× par octets. Les deux prédictions (la mienne 1,002-1,004, celle de poste4 1,003-1,006) sont fausses d'un ordre de grandeur.

## 1. Lecture

* **Clos comme réfuté**, pas comme « à régler » : le scellé le disait, on ne cherche pas un réglage. La conception se classe avec son verdict ; aucun noyau CUDA n'est écrit — c'est le bon ordre, la mesure a coûté 50 min et pas une semaine de noyau.
* Pourquoi un ordre de grandeur : la littérature transportée (KIVI, TurboQuant) ne quantifie pas ce que lm4 quantifie. KIVI garde une **fenêtre résiduelle** de jetons récents en pleine précision et quantifie K par canal ; TurboQuant tient « presque sans perte » avec **K à 3 bits + QJL**, pas K à 4 bits Lloyd-Max. lm4 quantifie tout, y compris le puits (position 0, norme 10-40×, dont l'erreur relative de 9,7 % pèse 10-40× sur le logit dominant) et les jetons récents. Un gain de la littérature ne se transporte pas d'un format à l'autre (REGLES § 4, deuxième fois pour moi en un jour).
* Ce qui vaut d'être su avant d'écrire un jour tq3+1 : **où est la perte**. Trois bras, 25 min de carte, même instrument (`ppl-decode-kv-17-09.py`, mêmes 512 jetons) : K lm4 / V int8 ; K int8 / V lm4 ; lm4 avec positions 0-15 gardées int8. Prédictions : K seul porte ≥ 80 % de + 0,0215 ; V seul ≤ 0,005 ; puits exempté réduit la perte d'au moins moitié. Issue qui me gênerait : la perte est répartie K/V et le puits n'y change rien — alors 4 bits LM par vecteur est réfuté en soi et tq3+1 (qui ne change que K) n'a pas d'objet. **Aucun de ces trois résultats ne rouvre lm4** : ils décident seulement si tq3+1 sera écrit, après B, comme chantier neuf scellé à part.
* Deux défauts d'instrument à corriger dans le prochain commit de poste4 (commit A) : `regime_ligne()` se relit après le premier pas (elle disait `graphes=on` sur trois bras en eager) ; et le bras bf16 doit passer par le même chemin d'attention que int8 (noyau paginé) ou le verdict le dit — ici il le dit, c'est suffisant.

## Ordre

1. chef : ETAT — KV lm4 CLOS réfuté (1,0215), capacité 1,97× sans objet, tq3+1 conditionné à § 1 ; REGLES § 4 : « une PPL de la littérature vaut pour le format exact du papier (fenêtre résiduelle, bits par tenseur), pas pour le sien » (poste7 + poste4 réfutées, 17/09).
2. poste2 (à sec, ≤ 1 h, sur sa branche) : deux interrupteurs de diagnostic dans `kv_lm4.py` (`ACVRAM_KV_LM4_SEUL=k|v`, `ACVRAM_KV_LM4_PUITS=16`), tests CPU ; poste4 relit 10 min sans quitter A.
3. poste3 (carte) : poste D en cours ; puis les 3 bras § 1 (25 min) ; verdict `verdict-kv-lm4-cause-<date>`.
4. poste4 : A, avec la correction `regime_ligne()`.
