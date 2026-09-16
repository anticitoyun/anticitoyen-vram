# poste7 — Hadamard GLM réfuté : clos ; la calibration à 230 jetons est le chantier (17/09)

Entrée : `verdict-ppl-hadamard-17-09` (poste3, 148d3a4, main a902245). Scellés de `poste7-hadamard-16-09` § 4 : réfutés — W4A16 tourné = non tourné ± 0,002 → mesuré +0,013 (1,042 vs 1,029) ; temps +0,3 ms de FWHT → mesuré +1,6 ms (22,6 vs 21,0). Réfuté reste réfuté.

## 1. Hadamard sur GLM : clos, pas suspendu

* Le PPL W4A4 MMA=1 n'a pas été mesuré, et n'a plus à l'être : la précondition (rotation neutre en W4A16) est tombée, et le temps (+1,6 ms, +7,6 % du pas) tombe seul le scellé ≤ 19,0 quelle que soit la calibration. Conséquence écrite le 16/09, appliquée : **W4A16 définitif sur GLM, prefill ×6 assumé dans la table à cinq.**
* Fait à garder pour un prochain modèle : une FWHT de 512 sur 12 lignes × 8 experts par couche coûte des µs, pas 1,6 ms — c'est un coût d'implémentation (lancements hors noyau), pas de la rotation. Si Hadamard revient un jour, la FWHT vit dans le noyau. Personne n'y travaille maintenant.
* Cause du +0,013 non établie : rotation intrinsèquement défavorable au NVFP4 par blocs de 16 (les poids tournés perdent la structure que l'échelle de bloc exploite), ou AWQ sur poids tournés qui suradapte davantage. Bras qui tranche, optionnel (§ 3.3).

## 2. Le vrai résultat de poste3 : l'inversion croît avec l'adaptation

    RTN 0,015 → AWQ 0,032 → AWQ + rotation 0,057   (écart privé − public, méd × bf16)

Lecture de poste3 : calibration sur « l'anglais par défaut ». Lu dans le code, c'est pire : `collect.py:48-64` — **six phrases** (2 anglais, 2 français sans accents, 1 Python, 1 SQL), ~230 jetons, **répétées** 16 fois pour remplir `--calib-seqs 16` (`collect.py:77`). L'AWQ de tous nos convertis `-k48` cherche son alpha sur les statistiques de six phrases. **Ce n'est pas d'abord la langue, c'est la taille** ; la langue est la seconde hypothèse. Le public passe SOUS bf16 (0,986) : un signe de suradaptation, pas de qualité.

Ce défaut touche le converti nominal 0.6.x, pas Hadamard — c'est lui qu'on teste.

## 3. Test décisif, sur la recette NOMINALE (`-k48`, sans rotation), trois bras à sec

| bras | calibration | jetons | sépare |
|---|---|---|---|
| A | anglais général, disjoint de wikitext (Gutenberg) | ≥ 16 k (32 × 512) | la taille |
| B | mixte : 1/3 français accentué (Gutenberg), 1/3 anglais (même source que A), 1/3 code du dépôt `acvram/` | ≥ 16 k | la langue, à taille égale |
| C (optionnel, dernier) | `-sansawq` + rotation 512 | — | la cause du +0,013 (§ 1) |

Interdits : `revue/*.md` et wikitext dans toute calibration (REGLES § 3) ; sha256 du fichier de calibration dans chaque verdict. Prédictions, écrites avant : écart `-k48` actuel 0,032. A ≤ 0,020 → la taille suffit ; A ≈ 0,032 et B ≤ 0,020 → la langue ; A et B ≥ 0,028 → la calibration n'explique pas l'inversion, le privé est intrinsèquement plus dur pour la quantification, et le scellé privé ≤ 1,010 se rediscute avec l'utilisateur. Issue qui me gênerait : A améliore le public encore plus et le privé pas du tout. C : privé ≈ 1,021 → rotation neutre en RTN, le +0,013 venait de l'AWQ ; ≈ 1,034 → rotation nuisible aux blocs de 16, Hadamard mort pour tout W4A16 NVFP4.

## 4. Métrique scellée : moyenne géométrique, plus la médiane

Deux références bf16 diffèrent de 1,1 % en médiane, 0,1 % en géométrique (poste3, § Bornes). REGLES § 3 : un scellé ne descend pas sous 2 × l'écart du témoin → sur la médiane, rien sous 0,022 n'est décidable. **Dès maintenant, le scellé porte sur la PPL pondérée par jeton (exp de la NLL moyenne = géométrique)**, seuil inchangé ≤ 1,010 ; la médiane reste une colonne. Les verdicts passés ne sont pas relus (réfutés sous les deux références).

## Ordre

1. poste2 (à sec, 3 × ~1 h) : bras A puis B sur `-k48`, corpus construits et sha256 publiés avant la conversion ; C en dernier si le temps le permet. Verdicts : `verdict-calib-k48-{A,B,C}`.
2. poste3 (carte, 3 × 20 min) : PPL privé + public, géométrique ET médiane, sur chaque bras ; scellés § 3.
3. poste4 : rien sur Hadamard. Le +1,6 ms est noté, pas assigné.
4. chef : Hadamard GLM clos dans ETAT ; `collect.py:48-77` cité comme défaut d'inventaire (six phrases répétées) ; porter à l'utilisateur que le converti nominal 0.6.x est calibré sur 230 jetons et qu'un correctif de corpus par défaut suivra le bras gagnant.
