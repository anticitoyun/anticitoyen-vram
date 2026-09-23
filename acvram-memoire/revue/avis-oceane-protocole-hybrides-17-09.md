# Avis — relecture du protocole hybrides avant fenêtre carte (Océane, 17/09)

Ordonné par Sage (`sage-priorite-apres-campagne-17-09.md` §2, poussé 6abba3f) :
revoir la cohérence des scellés et le montage de mesure avant que la fenêtre
carte (Laure, Qwen3.8-27B calibA) ne s'ouvre. À sec, aucun GPU touché ;
lecture seule des verdicts déjà commités (revue/*.md, origin/main).

## 1. Les quatre scellés retracés à leur source

| scellé (priorité) | source du chiffre | vérifié |
|---|---|---|
| b=12 : 97 → ≥ 400 t/s | `verdict-palier2-qwen38-27b-17-09.md` : acvram b=12 **96,7 t/s** (srcexl3), calibA **96,8 t/s** (`protocole-trtllm-qwen38-17-09.md:4`) | cohérent — même chiffre aux deux convertis, la récurrence torch domine indépendamment du format source |
| prefill ≥ 5× la voie torch | palier2 : acvram prefill **2 482 j/s** (srcexl3), calibA **2 446 j/s** | cohérent, mais **le chiffre n'est pas écrit dans le protocole hybrides** — seul le facteur ×5 y figure |
| PPL ± 0,002 (équivalence torch/fla) | protocole §2 point 3, comparaison interne au même converti | cohérent tel qu'écrit, à condition que les deux bras (torch, fla) tournent sur `calibA`, jamais l'un sur calibA et l'autre sur srcexl3 |
| b=1 : 67 → ≥ 90 t/s | **67,0 t/s est le chiffre srcexl3** (`verdict-palier2-qwen38-27b-17-09.md`) ; le converti réellement embarqué dans la fenêtre est **calibA, mesuré à 63,8 t/s** (`verdict-qwen38-calibA-17-09.md`) | **baseline erroné** — voir §2 |

## 2. Le baseline b=1 cite le mauvais converti

Le protocole hybrides écrit « b=1 Qwen3.8 67 → ≥ 90 » et annonce la fenêtre
carte sur `Qwen3.8-27B-nvfp4-calibA` (§2 point 4 : « Qwen3.8-27B calibA,
mêmes instruments que palier 2 »). Mais 67,0 t/s est le chiffre du converti
`srcexl3_6_00bpw-nvfp4` (palier 2, double quantifié EXL3→NVFP4) ; calibA
seul (single quantifié depuis bf16, calibré GDN) mesure **63,8 t/s**
(`verdict-qwen38-calibA-17-09.md`, Laure 27f2e8d), et ce chiffre est déjà
expliqué : bridage de puissance (399-400 W dans les deux bras), pas un
écart de moteur — « la différence entre convertis est sous le bruit d'un
bras bridé » (même verdict).

Le b=12 tient (96,7 vs 96,8, écart négligeable, cf. §1) parce que la
récurrence GDN torch domine à ce lot et efface la différence de source ; le
b=1 ne tient pas, précisément parce que b=1 est déjà **en butée de
puissance**, pas en butée de noyau — écart de 4,8 % entre deux mesures du
même modèle nominal. Le régime n'est pas porté par le nom : « 67 » et
« calibA » désignent deux mesures différentes qu'on est en train de
confondre dans un même scellé (REGLES §4, « un chiffre exact hors de son
régime »).

**Corriger avant la fenêtre :** baseline b=1 = **63,8 t/s** (calibA), pas
67. Garder ≥ 90 comme cible si Sage le souhaite, mais le dire en connaissance
d'un départ à 63,8, pas 67 — l'écart réel à couvrir est de +41 %, pas +34 %.

## 3. b=1 est en butée de puissance — le scellé ≥ 90 n'est démontrable que si la fenêtre publie les watts

`verdict-qwen38-calibA-17-09.md` classe explicitement la cellule b=1 comme
« plafonnée par la puissance, pas par le moteur ». Un noyau fla plus rapide
en instructions/octet PEUT lever ce plafond (mécanisme déjà établi dans ce
dépôt : `MECANISMES.md` « Les instructions coûtent des watts, pas
seulement du temps », Laurine 14/09 — un noyau qui fait moins
d'instructions par octet DRAM peut tenir un débit plus haut sous le même
plafond de 400 W). Mais rien dans le protocole hybrides ne demande de
publier la puissance (W) à côté du t/s pour la cellule b=1 fla. Sans elle,
un ≥ 90 t/s obtenu **parce que la carte n'a pas throttlé cette fois** (bras
froid, horloge libre transitoire) serait indistinguable d'un vrai gain de
noyau — c'est exactement l'instrument corrélé à la variable étudiée que
REGLES §4 met en garde. **Demander : chaque JSON b=1 de la fenêtre fla
publie W et horloge SM, comme la cellule calibA de référence.**

## 4. `regime_ligne()` ne porte pas encore le régime GDN

`verdict-palier2-qwen38-27b-17-09.md` le note lui-même en fin d'en-tête :
« GDN = torch de référence transformers (`gdn.py:30`)... `regime_ligne()`
ne l'imprime pas (à ajouter) ». Le protocole hybrides (§2 point 2) prévoit
bien `ACVRAM_GDN=fla|torch` porté par `regime_ligne()`, mais c'est une
tâche de code (étape 2, avant la fenêtre), pas encore un fait vérifié dans
un JSON. **Rendre le contrôle impossible à sauter (REGLES §3)** : avant
d'ouvrir la fenêtre, vérifier par un JSON réel (même à sec, un tenseur
jouet suffit) que `regime_ligne()` imprime bien `ACVRAM_GDN=fla` ou
`=torch` — sinon aucune cellule de la campagne ne pourra distinguer après
coup quel chemin a tourné, et un résultat manqué (§ scellé faux) resterait
inexplicable.

## 5. Contrôles déjà propres, rien à corriger

* **Préfixe** : corpus Qwen sans préfixe (`add_bos_token=false`,
  `protocole-palier2-qwen38-27b-17-09.md`), pas de biais type
  `[gMASK]<sop>` manquant — ce piège est spécifique à GLM, sans objet ici.
* **PPL d'équivalence torch/fla (± 0,002)** : comparaison interne au même
  converti, correctement isolée du scellé de classement (≤ 1,02 × bf16,
  qui reste hors d'atteinte pour calibA — 1,0253, non classé — et n'est
  pas ce que la fenêtre hybrides cherche à changer).
* **Corpus, régime NARROW_GEMM/MOE_MMA, budget KV** : identiques palier 2,
  déjà scellés et publiés avec sha256/en-tête complet.

## Consequence — a transmettre a Jerome

1. Corriger le baseline b=1 du protocole/scellé : 63,8 t/s (calibA), pas
   67,0 (srcexl3).
2. Ajouter W et horloge SM aux JSON b=1 de la fenêtre fla, pour distinguer
   gain de noyau et absence de throttle.
3. Verifier `regime_ligne()` porte `ACVRAM_GDN` AVANT la fenetre (test a
   sec suffit), pas seulement code ecrit.
4. Ecrire le prefill baseline en clair dans le protocole (2 446-2 482 j/s
   selon convertis), pas seulement « ≥ 5× ».

Rien d'autre trouve de faux dans le montage ; b=12 et prefill sont bien
ancres, seul b=1 portait un biais de regime.
