# evp — justesse de gpt-oss-20b dans acvram (format F) : TENU sur 11 invites sur 16

poste1, 02/10 12 h 41. Pièce bd evp, branche `poste1-gpt-oss`. Scellé : `scratchpad/poste1-evp-01-10/scelle.md` (f0f2ea708).

## Instrument

* Référence : transformers sur processeur, source MXFP4 déquantifiée
  (`sources_hf_temporaire/gpt-oss-20b-mxfp4`). C'est la bonne ancre : gpt-oss-20b est MXFP4 natif, il n'a jamais été servi
  ni publié en bf16 (avis duck.ai, `revue/poste4-duckai-02-10.md`, 09b5092cb). Pour chaque invite, 32 pas
  d'argmax forcé ; les logits sont gardés (`dumps/`, sha256 dans `dumps.sha256` et `dumps-b.sha256`).
* Mesuré : alias `gpt-oss-20b-srcmxfp4-nvfp4` (experts NVFP4 obtenus EXACTEMENT depuis le MXFP4, puits bf16), mêmes
  ids, `kl_oss.py acvram`, sous `carte.sh` (ACVRAM_POSTE=poste1, carte 0 vide avant, 12:40 → 12:41).
* Sortie : `scratchpad/poste1-evp-01-10/kl-F.json` (5 125 o, sha256 5843af01a924…).

## Couverture : 11 invites sur 16 (352 positions sur 512)

Le scellé prévoyait 16 invites. La phase HF sur processeur s'est arrêtée deux fois sur son plafond :
* 01/10 : 3 600 s, 8 invites (0, 10-16), 7 min 40 s par invite ;
* 02/10 : 4 500 s, 3 invites seulement (1, 2, 3), soit environ 25 min par invite (processeur partagé avec la campagne
  e50.2). Les deux plafonds ont rendu rc 124.

Il manque 4, 7, 8, 9 et 17. Le verdict porte sur 11 invites, pas sur 16.

## Résultat

| | prédit (scellé) | seuil | mesuré | |
|---|---|---|---|---|
| J1 top-1 | ≥ 99 % | ≥ 97 % | **98,58 %** (347/352) | tenu ; prédiction FAUSSE par le bas |
| J2 KL moyenne | 0,002-0,01 nat | ≤ 0,02 | **0,00121** | tenu ; prédiction fausse, mais du bon côté |
| J3 pire invite | — | ≤ 0,10 | **0,00369** (invite15) | tenu |
| KL max sur un pas | — | — | 0,0188 (invite15) | |

Par invite : KL moyenne entre 0,00046 et 0,00369. Top-1 32/32 sur 6 invites, 31/32 sur 5 (0, 2, 10, 11, 13).

**Verdict : format F TENU (J1, J2, J3) sur 11 invites.** Le chargement de gpt-oss (YaRN, puits, SwiGLU bornée, biais) est
juste au niveau de précision du bf16.

## Lecture

* Les 5 écarts de top-1 coexistent avec un KL par pas ≤ 0,019. Ce sont des quasi-égalités entre les deux premiers
  candidats, pas une divergence. La référence elle-même en compte : sur les invites 13, 14 et 15, l'argmax forcé de HF
  s'écarte de son propre generate (31/32). L'issue gênante nommée avant la mesure (« 13-15 peuvent coûter du top-1 sans
  faute du code ») ne s'est produite que sur l'invite 13.
* Ma prédiction de top-1 était trop haute. Avec un vocabulaire de 201 k et des logits bf16, une égalité sur ~70 pas
  suffit à faire tomber un pas sur 32 ; 98,6 % est le plancher du bruit bf16, pas un signal.

## Reste

* Les invites 4, 7, 8, 9 et 17 : environ 2 h de processeur hors carte (plafond 7 200 s), puis 1 min de carte. Utile
  seulement si l'on veut publier 16/16. Le verdict ne peut pas basculer : il faudrait une KL moyenne de plus de 0,10
  sur l'une d'elles, soit 27 fois le pire des 11 invites mesurées.
* La suite de la pièce : format S (q/k/v/o + lm_head int8) jugé contre F ; vitesse (chemins groupés avec biais, puits
  dans le noyau paginé) ; 120b (exil).
