# poste7 — S2 rejoué 18/19 : le feu vert 0.6.34 attend une chauffe qui clampe et ne ment pas (21/09, 08 h 00)

Lu : `verdict-s2-contexte-rejeu-20-09` (poste2, 26,2 min, main 645f87df), `poste1-scelle-qualite-nvfp4-31b.md` (095df3e9), message chef 05 h 17 (main 50ea396d).

## 1. S2 — deux défauts, pas un

* **GLM k48** : chauffe tenue 32768/32768 puis 500 réel à la requête (318 demandés / 298 libres = déficit 6,3 %). La chauffe envoie `[1]×N` : même jeton, même KV homogène, pas la fragmentation d'une requête réelle. Une garantie fausse 1/19 est un régime faux porté par un nom (`ctx_tenu=oui`, REGLES § 6) : **0.6.34 ne part pas avec cette ligne**. « Au mieux, pas garanti » publié = le mot « tenu » retiré de la ligne, pas gardé avec une note.
* **Les 8 refus-chauffe** (i8c 32768→15360, GLM ×3 →31744, Kimi →15360, Ornith →15360, Qwen3.5 →31744, Qwen3.8 →31744) sont honnêtes mais **régressent 0.6.33** : hier ces alias servaient (500 près de la limite), aujourd'hui ils ne chargent plus. Un alias refusé à son contexte par défaut n'est pas livrable.

## 2. Décision — la chauffe clampe, la ligne dit le chiffre, la colonne suit la mesure

* poste1, à sec ≤ 1 h : (a) séquence de chauffe pseudo-aléatoire sur le vocabulaire, graine fixe (0), longueur = ctx demandé ; (b) réserve : `ctx_tenu` = plus grand ctx (pas de 1024) dont la chauffe laisse ≥ max(5 %, 64 Mio) libres après la passe ; (c) **clamp au lieu de refus** : `ctx_tenu=15360(demandé 32768)` sur la ligne de régime, 400 nommé au-delà ; refus (`ContexteNonTenu` rc 2) seulement si ctx_tenu < 4096 ou sur `--ctx-strict` ; (d) tests cassants : chauffe [1]×N vs aléatoire sur un jouet CPU → l'aléatoire consomme plus, sinon le test est faux ; clamp → ligne exacte.
* Prédiction écrite : k48 → `ctx_tenu=31744` (comme calibA/base/Grande), 0 × 500 à ctx_tenu−64 ; les 8 clampés servent à leur valeur mesurée.
* poste9 : colonne `ctx` du TSV = valeur `ctx_tenu` du verdict pour les 8 + k48 (instrument = verdict, colonne = lecture) ; `262144` du 31B vision corrigé. Une colonne ≠ ligne servie = faux, contrôle par `verifier-contexte.py`.
* poste2, **prise ≤ 8 min** avant le feu vert : k48 + i8c + Ornith + Kimi (les quatre plus grands écarts) à ctx_tenu−64 → 200, ctx_tenu+64 → 400 nommé ; scellé **0 × 500 sur 4**. Rejeu 19/19 (26 min) après le `dpkg`, comme verdict de 0.6.34 installé — le code ne change pas entre les deux.
* Si k48 casse encore après (a)+(b) : mécanisme inconnu → chantier ; k48 clampé à 31744 dans la colonne (chiffre mesuré), 0.6.34 part avec la ligne exacte. La colonne porte le régime, pas la vigilance.
* Feu vert 0.6.34 = § 1b 4/4 + scellé 4/4 ci-dessus + rejeu GLM b=1 (charge < 1).

## 3. P3 (3) — référence

Pas de 30B bf16 sur disque ; notre nvfp4 30B est converti **depuis l'AWQ**. Le témoin AWQ déquantifié mesure donc exactement notre étape (nvfp4 sur AWQ) : c'est la clause P3 (3) « conversion fidèle à sa source », pas « qualité contre l'officiel ». Ordre : P3 (3) contre l'AWQ déquantifié maintenant, fiche : « source AWQ int4 ; nvfp4 vs bf16 officiel non mesuré ». Le bf16 officiel (≈ 60 Go, débit de l'utilisateur, nvme3) = question à l'utilisateur en clôture, non bloquante, utile seulement si le 30B devient un alias servi au quotidien.

## 4. Scellé 31B (poste1 05 h 28) — accepté avec deux amendements

* Faute construite = l'alias heretic srcQ4_K_M-nvfp4 (déjà connu pire), **pas** d'option de conversion ×1,05 à écrire ; il doit sortir de « tenue » sur le texte, sinon le seuil ne contrôle rien.
* Prédiction poste7, écrite avant : texte KL max ≈ 2 nat (témoin du 20/09 : 2,12) → **DÉFAUT attendu** sur la clause KL max, bande à nommer ; si la bande fautive est unique et la première variable (KV bf16) la referme, le coût est le KV, pas nvfp4. T (acvram bf16 étagé vs HF) reste la porte : sans T tenu, aucun chiffre. Rang : après D (30B), inchangé.

## Ordre

1. poste1 : § 2 (a)-(d), pointeur avec sha ; puis instruments P3 (3) contre l'AWQ déquantifié.
2. poste9 (trou de 10 min en cours : § 1a + § 3a) : ensuite colonnes `ctx` = ctx_tenu ×9, 262144 corrigé, `verifier-contexte.py` 0 FAUX.
3. poste2 : § 1b maintenant ; puis prise ≤ 8 min (4 alias, scellé 0 × 500) sur le correctif d'poste1 ; puis feu vert GLM b=1 charge < 1 → dpkg ; puis n = 60 ; 19/19 rejoué après l'installation.
4. chef : ETAT — S2 18/19, deux défauts nommés, feu vert conditionné ; question bf16 30B (60 Go) à l'utilisateur en clôture avec C9 119B.
