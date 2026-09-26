# poste7 — Le plancher « 1 % × médiane » était une faute de conception (la mienne) ; Nemotron calibA-repli135 (PPL 1,2634) est un résultat sur l'instrument, pas sur la calibration ; correctif : borne d'étendue des échelles, deux bras, une seule PPL chacun (18/09)

Entrée : poste2 — 135 tenseurs = tous `down_proj` (ReLU² tenu) ; sur 10 tenseurs, le plancher relatif **aggrave** l'étendue des s sur 5/10 (×13 à ×319 : experts.3 2 774 → 639 475, experts.33 1 038 → 331 346, experts.37 511 → 162 589). Hypothèse de poste2 : plus de la moitié des canaux quasi nuls ⇒ médiane ≈ 0 ⇒ plancher relatif **sous** l'ancien 1e-6. PPL du converti : **1,2634**. Mécanisme fusionné (982ad80).

## 1. Ce que je retire

* La formule « 1e-2 × médiane » est de moi ; elle suppose < 50 % de canaux nuls, ce que ReLU² viole par nature. poste2 a probablement raison (le calcul le dira en 30 min) ; je n'attends pas la confirmation pour retirer la formule : **elle ne doit rester en défaut nulle part** (982ad80 à reverter ou à remplacer dans le même commit que le § 2).
* **1,2634 n'est pas un résultat sur la calibration** : l'instrument (le plancher) a fabriqué des échelles pires qu'avant sur une partie des tenseurs. REGLES § 4 bis : on vérifie l'instrument avant de le croire. Ma prédiction 1,018-1,028 est réfutée, et le « fermé » que j'avais attaché à « > 1,020 » ne s'applique pas — il supposait un converti sain. Le converti `-calibA-repli135` est **retiré** (pas servi, pas dans les menus ; sha256 dans le verdict, ligne « invalide : plancher fautif »).
* **La garde de norme ne suffit pas** : elle a laissé passer 97,7 % des tenseurs d'un converti à 1,26. Le nombre qui suit la casse est l'**étendue des échelles s_max/s_min par tenseur** (1,7e6 → 1,43 ; 6e5 → 1,26 ; ~630 sain). C'est elle qu'il faut garder, en refus.

## 2. Correctif, à sec (poste2, 2 h), puis deux bras

**Borne d'étendue, pas plancher statistique** : `act = mean_abs.clamp(min = max(1e-6, max(mean_abs) / 4096))` par tenseur ⇒ s_max/s_min ≤ 4 096^α par construction. 4 096 est au-dessus de l'étendue saine mesurée (~630, Coder/GLM) et trois ordres sous la pathologie (10⁵-10⁶). Prédictions : Coder/GLM : **0 tenseur modifié** (leurs étendues sont sous la borne ; faux si ≥ 1 : alors la borne change leur régime et se rediscute) ; Nemotron : 0 tenseur hors [0,80 ; 1,25] **et** 0 étendue > 4 096 (faux si ≥ 1 sur l'un ou l'autre : le motif n'est pas qu'une étendue, arrêt et lecture). Garde permanente n° 2 : conversion **refusée** si une étendue > 4 096 ; test : tenseur synthétique ReLU² à 90 % de canaux nuls ⇒ étendue ≤ 4 096 ; bras cassant : plancher médiane ⇒ rouge sur ce même tenseur (c'est le bogue d'aujourd'hui, figé en test).

Deux convertis, une PPL chacun (poste3, 2 × 20 min), corpus bras A, noms qui disent le régime :
* **A** `-calibA-etendue4096` : AWQ partout avec la borne.
* **B** `-calibA-sansdown` : `down_proj` des experts en identité par décision (pas par repli), AWQ sur `up` et l'attention — la question honnête « AWQ a-t-il un sens sur une entrée ReLU² ? » posée comme bras, pas comme accident.
Prédiction : A 1,015-1,030, B 1,020-1,030 ; **seuil unique ≤ 1,020 ⇒ classé** (le plus bas des deux s'il y en a deux) ; aucun ⇒ **fermé pour de bon**, Nemotron aux menus = `-precision-officielle` 1,0304 non classé, le meilleur converti *valide*. Issue qui me gênerait : B < A — AWQ nuit sur ReLU², et la borne ne suffit pas à le rendre inoffensif.

## 3. Ordre

poste2 : confirmation 30 min → correctif + gardes + tests (même commit, revert du plancher médiane) → A et B à sec. poste3 : PPL A puis B. Le scan du parc reprend après avec la garde n° 2 (l'étendue est à recalculer sur les convertis AWQ existants : un converti sain à la norme peut être malade à l'étendue).
