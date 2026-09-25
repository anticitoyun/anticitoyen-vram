# Scellé — pièce 190 (ordre chef) : bilan chiffré A (74bcdd07, 24/09 soir) → B (main + 175b fusionnée)

Écrit AVANT toute mesure, avant même le sha de B (chef doit le donner). Prédiction construite à
partir des cinq pièces déjà fusionnées/mesurées entre A et B : 172 (poste5, déquant partagé),
175 (poste6, GDN α/β concat/triton), 176 (poste1, GDN qkv‖gate pile int8), 179 (poste5, B' int8 +
admission deux pas), 182 (poste1, z sans cast + attention GQA).

## Ce que chaque pièce a réellement mesuré (relu avant de prédire, REGLES §3/poste1 25/09)

| pièce | ce qui change | instrument | delta mesuré |
|---|---|---|---|
| 172 | déquant partagé par la boucle par séquence | forward isolé (diag) | Qwen3.8 −12,5 % (365,1→319,4 ms) ; TTFT servi tenu (0 passe nulle), pas de régression |
| 175 | GDN α/β en un appel (concat/triton) | pas_gpu médian, 300 pas, mixte | b=8 −12,6 % (concat) / −10,9 % (triton) |
| 176 | GDN qkv‖gate en une pile int8 | ABBA certifie-b12, mixte | b=8 B/A 0,9967 (−0,33 %, prédit −0,5 à −2,5 % : FAUX sur l'ampleur, sans régression) ; b=1 0,9863 (−1,37 %) |
| 179 | B' int8 + admission en deux pas | en-processus (8×L) + banc mixte | en-processus L=92 −51,5 %, L=120 −48,6 % ; **banc mixte B/A −0,8 % (prédit +5 à +8 % : FAUX — l'ampleur en-processus ne survit PAS au banc servi, cause nommée par `eng179`)** |
| 182 | z sans cast, attention GQA | banc chat + en-processus | z : +0,40 % méd. (légèrement PLUS lent, sous le seuil) ; GQA : gain en-processus (−0,18 à −0,42 ms), NEUTRE au banc (< +0,2 %) |

**Leçon retenue avant de prédire (179, la plus proche de mon propre instrument banc/service) :**
un gain en-processus (forward isolé, pas_gpu médian) de 10-50 % ne se traduit PAS au même ordre de
grandeur sur un banc chat servi — 179 en est la preuve directe (prédit +5-8 %, mesuré −0,8 %). Je
prédis donc sur le banc/service une fraction SEULEMENT des gains en-processus cités ci-dessus, pas
leur somme.

## Prédiction chiffrée par modèle, avant mesure

Toutes les pièces sources sont mesurées sur **Qwen3.8** (nvfp4, 172) et **mixte** (i8c, 175/176/179/182,
sauf 172 qui couvre les deux). **Aucune des cinq ne mentionne gemma31** — la prédiction pour ce
modèle est donc la plus faible, presque une extrapolation à vide.

* **Qwen3.8-27B-nvfp4** — b=8 débit : **+1 à +4 %** (seule 172 s'applique directement, forward
  −12,5 % dilué comme en 179 ; b=1 : **+0 à +2 %** (moins de leviers testés à b=1 sur ce modèle) ;
  J/jeton : suit le débit au signe près (même puissance, plus de jetons/s), **−1 à −4 %** ; TTFT
  2048 : **quasi inchangé (± 2 %)**, les cinq pièces touchent le décodage, pas le préfill.
* **Qwen3.8-27B-unsloth-mixte-i8c** — b=8 débit : **+2 à +6 %** (172+175+176+179+182 s'appliquent
  tous, dilution attendue comme en 179 : je ne prends qu'une fraction des gains en-processus,
  jamais leur somme brute qui donnerait +15-25 %) ; b=1 : **+1 à +4 %** (176 montre un gain PLUS
  fort à b=1 qu'à b=8 sur cette seule pièce, contre-intuitif, à confirmer ou réfuter ici) ; J/jeton :
  **−2 à −6 %** ; TTFT 2048 : **quasi inchangé (± 2 %)**, même raison qu'au-dessus.
* **gemma31** (nvfp4, alias non touché par les cinq pièces sources) — b=1/b=8 débit, J/jeton, TTFT :
  **NEUTRE, 0 ± 2 %** — aucun mécanisme identifié qui le concernerait. Falsificateur direct : si
  gemma31 bouge de plus de 2 %, c'est qu'une des cinq pièces touche un chemin partagé (déquant,
  GQA, admission) plus large que ce que son propre verdict décrivait — à nommer, pas à ignorer.
* **Banc chat b=8 du mixte** (le plus proche de l'instrument de 179, donc le plus fiable pour
  juger la dilution) : **+1 à +3 %** de débit servi, **pas les +5-8 % que 179 avait lui-même
  prédits et ratés**. Si le banc montre ≥ +5 %, c'est que quelque chose d'AUTRE que la simple
  somme des cinq pièces joue (peut-être leur composition, testée nulle part isolément) — bonne
  nouvelle à vérifier, pas à publier sans regarder pourquoi.

## Issue qui me gênerait

Un gain nul ou une régression sur le banc mixte malgré cinq pièces individuellement TENUES —
répéterait exactement le motif de 179 (banc ≠ somme des instruments isolés) mais cette fois sur le
cumul de CINQ pièces au lieu d'une seule, ce qui rendrait la méthode de prédiction par sommation
inutilisable pour les prochains bilans de nuit (162 et ceux qui suivront).

## Reste avant de mesurer

Attente du sha de B (main + 175b fusionnée) donné par chef. Prises `poste2-p190-*`, ABBA, < 15 min
chacune, charge hôte relevée par la 189 (`ACVRAM_TYPE=mesure` par défaut, `~/.cache/acvram/charge/`).
