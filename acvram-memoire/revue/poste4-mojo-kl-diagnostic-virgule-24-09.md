# Diagnostic — virgule pleine chasse MAX (ordre chef, deux hypothèses)

* (1) entrées identiques : PAR CONSTRUCTION, pas juste vérifié — `kl_reference.py` construit
  `prompt_ids = ids(HF) + cibles(HF)` une fois par invite (depuis le dump `.pt`) et envoie la MÊME liste
  Python aux deux moteurs, en ids déjà tokenisés (`prompt`: array d'entiers, aucun gabarit rejoué côté
  serveur). Hypothèse (1) exclue par construction, pas seulement par mesure — rien à corriger côté entrée.
* (2) logit de `,` contre `，` à entrées égales, position n_prefix+3 (après « Okay », 5/5 invites) :

  | invite | id `,` =11 (HF logprob) | id `，`=3837 (HF logprob) | rang HF de `,` | `，` chez MAX (top-7) |
  |---|---|---|---|---|
  | anglais | ≈0 (−6e-7) | −20,375 | 1er | 1er (−0,586), `,` absent du top-7 |
  | code | 0,0 | −22,125 | 1er | (même motif) |
  | court | 0,0 | −21,875 | 1er | (même motif) |
  | systeme | 0,0 | −22,25 | 1er | (même motif) |
  | tours | 0,0 | −22,75 | 1er | (même motif) |

  HF est quasi certain de `,` (probabilité ≈1, `，` à e−9/e−10 de masse). MAX place `，` en tête (−0,586,
  soit ≈56 % de masse) et `,` hors de son propre top-7 — **inversion du classement à entrées égales,
  systématique sur 5/5 invites, toujours à la même position syntaxique** (juste après le jeton « Okay » du
  raisonnement `<think>`).

## Verdict (écrit tel quel, ordre chef)

**Écart de calcul réel**, pas un artefact de gabarit ni d'instrument. Le point commun aux 5 invites (même
position relative : après « Okay », premier jeton de ponctuation du raisonnement) suggère un biais du
tokeniseur ou du sampler de MAX vers la ponctuation CJK à cet endroit précis (poids de sortie mal indexé
pour un octet de ponctuation ASCII bas rang ? confusion top-p/BPE sur un jeton à un seul caractère ?) —
hypothèse non vérifiée plus loin, hors de portée d'un diagnostic à l'instrument : c'est un comportement du
moteur MAX/Mojo lui-même, pas quelque chose que ce squelette peut corriger. Pas de nouvelle prise KL avant
un ordre : rejouer au même seuil ne changerait rien tant que la cause (dans MAX, pas ici) n'est pas traitée.
