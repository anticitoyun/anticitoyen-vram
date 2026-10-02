# S1 bis rejouée sur fusion-2 — préfill par morceaux sous témoin reprise : court TENU, dense FAUX (0,043 pour un seuil de 0,008, inchangé depuis hier), long NON JUGEABLE (exil inégal entre les bras, ma faute de protocole)

instrument : `scratchpad/poste6-s1-bis/carte-s1-bis.sh` → `outils/gpu/mesure/prise-s1-morceaux-kv31b.sh` (un processus par bras, REQUETES=2 sur A1 = témoin reprise) + `outils/gpu/mesure/s1-morceaux-comparer.py` (REGLES § 4 : P3 = Δ(B, A1) ≤ 2 × Δ(reprise)) ; glouton 32 jetons, logprobs=10, sans spéculation ; texte généré jamais lu
commit : poste6-gemma-anneau 94aa34699 (= fusion-2 62396fcba + scellé), HEAD asserté par chaque prise, arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, 15 Mio occupés au départ et à la fin ; carte 1 : llama-server permanent seul ; gemma-4-31B 4sur6 kv=int8 ; Devstral-24B ; bras A sous `ACVRAM_PREFILL_MORCEAU_AU_DELA=0`, bras B `ACVRAM_PREFILL_MORCEAU=4096` ; chaîne longue sous `ACVRAM_KV_ANNEAU=0`
scellé : `poste6-s1-morceaux-scelle-carte-bis-01-10.md`, addendum du 02/10 (Q1'-Q5'), poussé avant la prise (94aa34699)
mesuré : 9 prises, 25-116 s chacune, 15:11:34 → 15:19:43 puis 15:20:31 → 15:20:56 (dernier bras rejoué après une prise de chef intercalée)
verdict : **court TENU** (B/A1 0,067 ≤ 2 × témoin 0,193) ; **dense FAUX** (B/A1 0,0428 > 2 × témoin 0,00403) ; **long NON JUGEABLE** (P5 : 31, 31 et 16 MLP exilés selon le bras — B/A1 0,312 pour un seuil de 0,126 ne compare pas que les morceaux). Q4' : ma prédiction « la cause d'hier a disparu » est FAUSSE pour Devstral.
durée : 8 min 34 de carte tenue (prévu ≈ 13, plafond 20), en deux temps

## Mesuré

| chaîne | régime des bras | témoin A1/A2 | témoin reprise | B / A1 | seuil 2 × | |
|---|---|---|---|---|---|---|
| court, gemma 7 953 jetons, 8 192 | A : NOMINAL, 0/60, seul tenant, table de plongements en RAM hôte ; B : NOMINAL, 0/60, `(morceaux@4096)`, table sur la carte | Δ 0 sur 341 valeurs | 0,193 (32 ids égaux) | **0,067** (position 0) | 0,386 | **tenu** (P1-P5) |
| long, gemma 17 859 jetons, 20 480, anneau interdit | A : DÉGRADÉ **31/60** exilés (réserve 6,32 Gio) ; B : DÉGRADÉ **16/60** (réserve 3,60 Gio) | Δ 0 sur 352 | 0,0631 | 0,312 | 0,126 | **non jugeable** : P5 FAUX, exil inégal |
| dense, Devstral 7 865 jetons, 10 240 | A et B : NOMINAL, 0/40, graphes actifs | Δ 0 sur 352 | 0,00403 | **0,0428** | 0,00806 | **FAUX** (5,3 × le seuil) |

Hier (f634d5016, morceaux relisant leurs K/V int8 du cache) : Devstral 0,041 pour un témoin de 0,004 ; gemma court sans
coupe 0,088 ; gemma long 0,436 (témoin 0,027).

## Lecture

* **Devstral : rien n'a bougé** (0,0428 contre 0,041). Les K/V bf16 transitoires de d19 n'ont donc PAS réduit l'écart
  morceaux / seul tenant : la relecture int8 n'en était pas la cause sur ce modèle. Mon addendum l'affirmait (« la cause
  du Q4 FAUX d'hier n'est plus là ») et son critère de réfutation (« un écart ≥ celui d'hier ») est atteint. Reste
  l'explication de REGLES § 4 : le SDPA et cuBLAS réduisent selon la longueur de l'appel, un morceau n'a pas celle du
  seul tenant — et le témoin reprise de Devstral est minuscule (0,004), dix fois sous cet écart.
* **gemma court tient parce que son témoin est grand** (0,193 : la reprise relit des K/V int8), pas parce que les morceaux
  y seraient plus fidèles — 0,067 est du même ordre que Devstral.
* **Long : ma faute de protocole.** `ACVRAM_PREFILL_MORCEAU_AU_DELA=0`, posé sur les bras A pour les garder d'un seul
  tenant, est aussi lu par le planificateur : sans morceaux possibles la réserve passe de 3,60 à 6,32 Gio et l'exil de 16
  à 31 MLP. Les bras n'ont plus le même régime (le comparateur le dit : « comparaison contaminée »). J'avais simulé à sec
  le plan du bras B, pas celui du bras A. À 20 480 sur cette carte, je ne vois pas de chaîne longue propre : KV plein
  = exil qui dépend du bras ; anneau = pas de cache de préfixe, donc pas de témoin reprise.
* Même mécanisme, sans conséquence, sur le court : le bras A sort la table de plongements (pas de plafond possible), le
  bras B la garde ; 0 MLP exilé des deux côtés, et les ids table hôte / table carte sont égaux (prise de 13:02).
* Durées : court 47 / 37 / 33 s ; long 90 / 82 / 54 s ; dense 116 / 28 / 25 s. Le premier bras Devstral a relu le disque
  dur (109 s avant « prêt » pour 15,1 Gio) : mon préchauffage de 14:58 avait été évincé du cache de pages entre-temps.
* **Interruption** : le bras dense B a été refusé à 15:19:43 — une boucle de chef a pris le verrou dans la seconde
  laissée libre entre deux bras ; rejoué à 15:20:31 après sa prise (48 s d'écart avec A2, même commit, même régime).
  Défaut de mon script (verrou repris bras par bras) ; il porte désormais une garde de chaîne, non jouée.

## Reste

Le critère § 4 n'est pas tenu sur un dense : les morceaux forcés restent un opt-in. Cela ne dit rien de plus sur d19
(« morceaux au-delà du tenu »), déjà jugé par sa garde de qualité (KL 2,9e-4, top-1 99,2 %), décision de chef du 01/10.
Chaîne longue : à redéfinir avant tout rejeu (simuler à sec le plan de CHAQUE bras), ou à abandonner.
