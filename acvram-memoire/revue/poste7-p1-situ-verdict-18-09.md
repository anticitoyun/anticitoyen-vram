# poste7 — P1 in situ : prefill +63 % (15 987 j/s) et b=12 1 239 t/s · 0,303 J TENUS, GLM +18 % ; b=1 −1,2 % et ppl-decode-kv +0,008 FAUX → rien en défaut, et le poste qui lâche est nommé : le GEMV de décodage lisant la disposition Marlin porte un BIAIS, pas un bruit ; poste4 le reprend, trois hypothèses, un juge (18/09)

Entrée : poste3 3c9f0c8 `verdict-p1-situ-complet-18-09` — régime NOMINAL, `experts_layout=marlin`, 15/15, capture 1/2/8/16 (3,53/4,96/6,33/8,46 ms). Tenus : prefill Coder **15 987 j/s** σ 34 (≥ 15 700), PPL prefill 1,0155 (Δ +0,0007) ; b=12 harnais égal **1 239 t/s · 0,303 J brut** (≥ 1 092, ≤ 0,357 — **J sous llama.cpp 0,306 pour la première fois**) ; GLM prefill 5 502 (+18 %), PPL GLM 1,0120 (Δ −0,003, B0 à rejouer avant de l'écrire). Faux : b=1 350,9 < 355,4 (−1,2 %) ; **ppl-decode-kv 5,5506 contre 5,5426 (+0,008 > ± 0,002)**.

## 1. Lecture

* Le prefill Marlin est juste (Δ +0,0007) ; le décodage par le GEMV (b) ne l'est pas : +0,008 sur 1 024 pas × 48 couches est un **biais systématique**, pas un ordre de somme (un ordre de somme donne du bruit centré, ≤ 1e-3). Le banc l'a laissé passer parce que « 0 hors 2⁻⁷ relatif par ligne » ne voit pas un biais de 1e-3 signé : **un critère par comptage ne détecte pas un biais ; il faut la moyenne signée de Δ** (REGLES § 4 bis : témoin négatif — le critère aurait dû rendre faux sur un noyau biaisé construit exprès).
* Règle appliquée par poste3 : rien en défaut — juste, la disposition est unique, le prefill ne se prend pas sans le GEMV.

## 2. Ordre poste4 — trois hypothèses, dans l'ordre, une à la fois, à sec puis 5 min de carte

1. **Précision du produit ou de l'accumulation** (hfma2 bf16 sur le chemin de la somme, v1 accumule en fp32) : contrôle = biais moyen signé de (b) contre fp32 sur routage réel rejoué (`ACVRAM_TRACE_ROUTAGE`), par couche ; v1 ≈ 0 attendu, (b) ≠ 0 ⇒ cause ; remède : produit et accumulation fp32 comme v1 (registres à relire par `-Xptxas -v` ; la perte de vitesse est le prix, à remesurer).
2. **Échelles étagées** ≠ E4M3 d'origine sur une fraction des blocs (le contrôle « 0 / 589 824 » portait les échelles du prefill, pas les tables du GEMV) : compter, attendu 0.
3. Fantômes / bornes de tuile à M = 1 (b=1 seul) : bras cassant existant.
**Juge unique, écrit avant** : le banc gagne un critère de biais — |moyenne(Δ)| ≤ 1e-4 × moyenne|y| par couche, chaque chemin contre fp32, et un témoin négatif (noyau à hfma2 forcé) qui DOIT le faire rendre faux ; puis in situ (poste3, 20 min) : ppl-decode-kv = 5,5426 ± 0,002 ET b=1 ≥ 355,4 ET b=12 ≥ 1 092 / J ≤ 0,357 (les tenus se remesurent, ils ne s'héritent pas). Tenu ⇒ `GEMV_LAYOUT=marlin` + `PREFILL_GROUPED=marlin` passent en défaut ensemble.
Issue nommée si PPL tient mais b=1 reste dans [0,95 ; 0,97) × 366 : ce n'est pas à moi de l'adopter — ligne utilisateur : « prefill +63 %, b=12 +10 % t/s et J sous llama.cpp, au prix de −1 à −3 % à b=1 : adopter oui/non ? », cellule b=1 republiée telle quelle si oui.

## 3. Ce qui est acquis même si (b) tarde

Le port Marlin prefill est validé (PPL, 15 987, GLM 5 502) ; ce qui manque est le décodage sur la même disposition. Aucune cellule du comparatif n'est éditée avant le défaut ; le verdict entre dans INDEX tel quel (tenus et faux nommés).

## Ordre

* poste4 : § 2, critère de biais + témoin négatif d'abord, puis la cause ; verdict à sec avec la couche et le signe du biais.
* poste3 : GLM B0 rejoué (PPL 1,0120 vs B0, 20 min) dans la même fenêtre que l'in situ suivant ; P2 GLM verdict.
* chef : ETAT — P1 : prefill tenu, décodage (b) biaisé, rien en défaut ; REGLES § 4 bis ligne « un critère par comptage ne voit pas un biais : moyenne signée + témoin négatif ».

## 4. Addendum (poste3 ddea1ce) : (b) ≡ v1 au bit sur 5 couches puis 1 ulp bf16 à la couche 5, amplifié de façon chaotique ; le témoin graphes/eager de v1 diverge pareil (−0,015 toléré depuis le 14/09) — mon scellé ± 0,002 sur UNE cellule était faux par construction pour un noyau non bit-exact

* Ce que la bissection dit : ni tampon, ni biais moyen (instrument par couche : 1e-9, témoin rouge) — une non-identité d'ordre d'accumulation fp32 qui bascule un arrondi bf16, et 43 couches l'amplifient. C'est exactement le mécanisme de l'écart graphes/eager que le régime par défaut porte déjà (naturel : −0,015 sur la même cellule). **Réfuté sur moi** : j'ai scellé ± 0,002 sans avoir mesuré le témoin (v1 capturé / v1 eager) dans le même instrument — REGLES § 3, « un scellé ne descend pas sous 2× l'écart du témoin », écrit le 16/09, refait aujourd'hui. Et « biais systématique » (§ 1) était une conclusion tirée de trois signes positifs : 1/8 de hasard, pas une preuve.
* Un noyau arithmétiquement équivalent mais non bit-exact ne se juge pas sur une cellule : **le juge est statistique**, sur ≥ 8 fenêtres, contre le témoin qui porte le même défaut (graphes/eager de v1).

### Protocole (poste3, ~15 min de carte), écrit avant

* 8 fenêtres de `ppl-decode-kv` : 3 tranches privées (décalages 0 / 24 576 / 49 152) × préfixes {256, 2 048} + 2 fenêtres wiki-gptq (préfixe 256, 2 048) ; 1 024 pas décodés chacune ; cache Triton chaud ; régime en tête.
* Trois bras par fenêtre : **A** = v1 eager, **B** = v1 capturé, **C** = (b) capturé. Δ_témoin = B − A ; Δ_b = C − B.
* Scellé : (i) |moyenne(Δ_b)| ≤ 2 × écart-type(Δ_b) / √8 (pas de signe constant au-delà du bruit), ET (ii) |moyenne(Δ_b)| ≤ |moyenne(Δ_témoin)| (pas pire que l'écart graphes/eager déjà porté par le défaut), ET (iii) aucune fenêtre avec Δ_b > 0,020. Tenu ⇒ (b) équivalent au grain du moteur, défaut des deux régimes ; faux sur (i) ou (ii) ⇒ (b) porte quelque chose de plus que l'ordre de somme, poste4 aligne l'ordre de réduction sur v1 (bit-exact, coût de vitesse à remesurer) ; (iii) seul ⇒ on note la fenêtre et on cherche le préfixe.
* Prédiction : tenu — moyenne(Δ_b) dans ± 0,004, |Δ_témoin| ≈ 0,010-0,015 ; issue qui me gênerait : 8/8 signes positifs (1/256) — alors (ii) peut tenir et (i) non, et c'est (i) qui commande.
* b=1 350,9 : poste vitesse séparé, jugé après (ligne utilisateur si dans [0,95 ; 0,97) × 366).

## 5. Verdict statistique (poste3 ee417ab) : (i) tenu (moy Δ_b −0,0086, 2 sd/√8 = 0,022), (ii) tenu (|Δ_témoin| 0,021), (iii) « aucune fenêtre > 0,020 » : le témoin v1 lui-même le viole sur 3/8 fenêtres (−0,066 à +0,206) — (iii) est un critère que la référence échoue contre elle-même, donc invalide par REGLES § 7 (17/09), pas « desserré après »

* (b) est équivalent au grain du moteur : signe non constant (3+/5−), moyenne sous v1 sur le privé ; le « biais » était trois fenêtres wiki. Réfuté sur moi : moy −0,009 hors ± 0,004, |Δ_témoin| 0,021 au-dessus de 0,010-0,015 — et (iii) posé sans témoin, **troisième fois aujourd'hui**. Règle mécanique pour moi désormais : **tout seuil sur une statistique d'ordre (max, min, « aucune fenêtre ») se mesure d'abord sur le témoin contre lui-même, dans la même note, avant d'être écrit** ; sinon il n'est pas écrit.
* Fait nouveau, à part et sérieux : l'écart graphes/eager de v1 va jusqu'à **+0,206** sur une fenêtre (tr2-p256) — ce n'est plus « 1-2 ulp bf16 » (14/09). Bead nommé « écart graphes/eager par fenêtre », poste1 après P1 : deux exemplaires de tr2-p256, puis fantômes/bornes du godet à b=1 (REGLES § 3 : capture aux godets). Il ne bloque pas (b), qui porte le même écart que v1.

### Décision

* Qualité : (b) équivalent, tenu. Vitesse : prefill 15 987 tenu, b=12 1 239 / 0,303 tenu, **b=1 350,9 = 0,959 × 366, dans [0,95 ; 0,97)** → comme écrit § 2, **ligne utilisateur** : « Adopter P1 (prefill Coder +63 %, GLM +18 %, b=12 +10 % t/s et J sous llama.cpp pour la première fois) au prix de −1,2 % à b=1 (366 → 351) — oui / non ? » Oui ⇒ `GEMV_LAYOUT=marlin` + `PREFILL_GROUPED=marlin` en défaut ensemble, cellules republiées telles quelles (b=1 351 étiquetée), comparatif édité ; non ⇒ témoin, rien n'est perdu.
* Poste b=1 ouvert quoi qu'il arrive (poste4, après le oui) : profil du pas b=1 (b) vs v1 sous graphes — banc 0,909× et in situ 0,988× : ce qui manque est hors du noyau (colle, lancements ?) ; scellé ≥ 366 au harnais égal.

## Ordre

* chef : ligne utilisateur ci-dessus ; ETAT ; REGLES § 3 « statistique d'ordre → témoin contre lui-même d'abord » ; bead graphes/eager +0,206 (poste1, après P1).
* poste3 : rien avant la réponse ; poste4 : idem, puis profil b=1.

## 6. OUI utilisateur (18/09) : P1 adopté — ordre de bascule

1. poste4 : défauts `ACVRAM_GEMV_LAYOUT=marlin` et `ACVRAM_PREFILL_GROUPED=marlin` dans `regime.py` (témoins `naturel`/`groupe` gardés), aucune variable à poser ; tests d'équivalence déjà dans les commits (règle 9).
2. poste3, 20 min : **contrôle au défaut sans variable** (comme RPW/XREG le 18/09) — `regime_noyaux` prouve les deux défauts, `hors_defaut={}` ; cellules harnais égal republiées telles quelles depuis les verdicts : Coder b=1 **350,9**, b=12 **1 239 / 0,303**, prefill **15 987**, GLM prefill **5 502** ; PPL inchangées (1,0155 prefill Coder, GLM 1,0120 non revendiqué comme gain).
3. chef, comparatif : colonne source ; **revendication contre la plus haute passe llama.cpp** (1 065,7 t/s · 0,278 J brut) : b=12 t/s **+16 %**, J brut **0,303 contre 0,278 : encore derrière de 9 %** — « J sous llama.cpp » ne s'écrit que contre leur moyenne, pas contre leur meilleure passe ; b=1 351 contre 344 (+2 %, contexte 1 024 ; à 2 048 à remesurer) ; prefill 15 987 contre 15 717 (**+1,7 %, parité tenue de peu**). Retrait daté des lignes « prefill −37 % ».
4. Suite, une chose à la fois chez poste4 : **P2 Coder** (chemin `_int_mm`, converti `-qkvo-i8c`, cible ≈ 20 000 j/s) avant le poste b=1 ; poste3 en parallèle : profil du pas b=1 (b) vs v1 sous graphes (20 min) pour nommer le poste avant que poste4 y touche.
