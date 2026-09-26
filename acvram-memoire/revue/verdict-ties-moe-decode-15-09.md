# Verdict — ties du chemin MoE-MMA au décodage (5) et chiffre certifié b=12 moyen

poste3, 15/09/2026, 12:33-12:40, une prise de carte (verrou ties-poste3, 388 s).
Ordre : poste7 [`poste7-poste3-15-09.md`](poste7-poste3-15-09.md). Protocole et
prédictions scellées avant : [`protocole-ties-moe-decode-15-09.md`](protocole-ties-moe-decode-15-09.md)
(poste3 aab592f). Code : main b516eb4 (chemin (5) de poste4, 92beffe).

## En-tête de mesure (REGLES §3)

    instrument   ties : logits f32 des top-2 à chaque `_emit` ; certifié :
                 energie.py, compteur NVML TotalEnergyConsumption
    cartes       [0] seule (exposée par carte.sh, champ `cartes` des JSON)
    fenêtre      certifié : 29-31 s par cellule (une ronde b=12 de 1 788
                 jetons, ctx 2048, invite 256), repos 30 s avant
    plafond      400 W relevé dans chaque processus ; horloge libre
    mode         moyen ; bras PROUVÉ dans le processus : `model._MOE_DECODE_MMA`
                 relu et écrit au JSON (A False, B True, bt 16)
    carte        vide au début et à la fin (`--query-compute-apps`), aucune
                 alarme « processus ont changé »
    données      scratchpad/ties-moe-15-09/{ties-A,ties-B}.json,
                 certifie-moyen-20s-{A1,B1,A2,B2}.json

## 1. Ties — RÉFUTÉ

12 invites × 256 jetons greedy, b=12 tenu (lot 12 à chaque pas), graphes.
Sept séquences (s0 s2 s4 s5 s7 s8 s9) identiques sur 256 jetons. Cinq
divergent, toutes avant 64 jetons (seuil 4/12 dépassé) :

    seq  jeton   bras A top-2 (val)                  bras B top-2 (val)                 écart B
    s1     2     220/198  (12,25 / 12,19)            198/220  (12,25 / 12,06)           0,19
    s3     2     522/7926 (15,25 / 15,06)            7926/522 (15,06 / 14,75)           0,31  > 0,30
    s11    1     198/47   (10,00 / 9,38)             47/198   (9,50 / 9,25)             0,25
    s10    1     62/198   (10,31 / 9,69)             220/62   (8,94 / 8,88)             0,06  top-1 de B absent du top-2 de A ; −1,4 de magnitude
    s6    17     104813/105918 (33,25 / 30,50)       32100/104720 (27,75 / 27,00)       0,75  candidats différents ; A franc (2,75) ; −5,5 de magnitude

Pire écart B 0,75 > 0,30 ; 5/12 > 4/12. **Ce ne sont pas des ties.** s6 et
s10 ne sont pas deux candidats qui permutent à un souffle : les logits de B
sont décalés de 1,4 à 5,5 et le vainqueur de A n'est plus dans le top-2 de B.

**Cause, lue dans le code (pas supposée).** Bras A = `_forward_grouped`
(`model.py:1054-1070`) : GEMV groupée, « activation bf16 lue telle quelle »
— W4A16. Bras B = `_forward_grouped_mma` (`model.py:1029` et `:1033`) :
`ext.nvfp4_quant_act` sur l'entrée `xs` ET sur l'activation intermédiaire —
W4A4 (E2M1 bloc 16 + UE4M3). Le chemin (5) ne change pas l'ordre des sommes,
il **quantifie les activations du MoE en 4 bits au décodage** (le prefill
le fait déjà : `model.py:954-958`). Un contrôle de ties ne peut pas juger un
changement de régime de quantification ; ce qui le juge, c'est la PPL de
poste2 (≤ +1 %) ou une comparaison de logits à tolérance A4 posée d'avance.

## 2. Témoin de faute — ABSENT

`ACVRAM_MOE_DECODE_MMA_BT=8` **plante avant de diverger** :
`RuntimeError: GEMM groupee MMA : bt dans {16, 32, 64, 128}` (la grille
refuse la tuile). Par la règle de poste7, le contrôle « ne garde rien » : il
n'a pas de témoin qui rende « réfuté » sur une faute connue. Le verdict 1
ci-dessus est donc un **écart mesuré**, pas un contrôle validé — il suffit à
maintenir (5) coupé, il ne suffirait pas à le valider. Témoin possible sans
carte supplémentaire longue : bras B avec un `nvfp4_quant_act` remplacé par
une quantification volontairement fausse (échelle ×2) sous garde d'env — à
écrire par poste4 dans (5), pas par moi dans un scratch.

## 3. Chiffre certifié b=12 moyen — RÉFUTÉ sur le pas, tenu sur l'énergie

    bras  MMA   pas ms   t/s      J/jeton  W moy   sm moy MHz
    A1    0     17,365   630,31   0,6195   390,5   2605
    B1    1     16,248   673,65   0,5371   361,8   2917
    A2    0     17,369   630,18   0,6213   391,5   2592
    B2    1     16,250   673,60   0,5403   364,0   2915

    gain B/A   pas −6,4 %   t/s +6,9 %   J −13,2 %   W −27 W

Reproductibilité ABAB : A à 0,004 ms près, B à 0,002 ms près.
Seuil de poste7 : réfuté si gain < 8 % en ms **ou** < 12 % en J → **RÉFUTÉ**
(−6,4 % en ms) ; l'énergie tient (−13,2 %). Alarme publiée d'avance et
déclenchée : A = 17,37 ms, hors de 15,9 ± 0,5 (poste4, fenêtres 22 s) —
mon montage (rondes ctx 2048, KV qui croît de 256 à 2 044, prefill compris
dans la fenêtre) n'est pas le sien ; je ne compare que B/A. Mécanisme
visible : B sort du plafond 400 W (362 W, horloge 2 915 MHz contre 2 600
bridée en A) — le gain d'énergie vient pour partie de l'horloge libérée,
et le gain de pas est plus petit que le −13,6 % de poste4.

## Trois états

    1. ties          RÉFUTÉ  (écart 0,75 ; 5/12) — cause : A16 → A4, pas un tie
    2. témoin        ABSENT  (bt8 refusé par la grille)
    3. certifié      RÉFUTÉ sur le pas (−6,4 % < 8 %), énergie tenue (−13,2 %)

Conséquence (poste7 § 1) : (5) reste coupé (`model.py:1290` défaut « 0 »),
retour à poste4 avec pas et logits ci-dessus ; le passage par défaut
attend une PPL du chemin A4 au décodage et un témoin de faute qui rende
« réfuté ».
