# Verdict — re-PPL alpha-commun après correctif quant_act : RÉFUTÉ (1,098 ; témoin ×43) ; site fautif = `act/s_d` à k_act = 8, k_x = 4 seul rend 1,019

- **instrument** : `acvram eval` (ligne de poste2 : `wiki-gptq.txt`, window/stride 2048, min-context 256, 4 fenêtres, `--json`), NOMINAL MMA=1, `ACVRAM_QA_COMPTE=1` ; référence bf16 8,1427 ; sorties `scratchpad/reppl-alpha-commun-16-09/eval-*.{out,err}`
- **commit** : travail/poste3 **ea3dd4a** (code `acvram/` = main e84a43d, correctif 01c48ef fusionné, t-qa 210/210) ; extension compilée hors verrou, empreinte du source contrôlée au chargement
- **régime** : une carte, 30,85 Gio libres au chargement ; chemin MMA confirmé par les compteurs (337 641 472 blocs quantifiés sur les 4 fenêtres, identiques sur les 8 passes)
- **scellé** (poste7 `pile-correctif` § 1.5 iii, § 6.3) : B alpha-commun ≤ 1,010, compteurs mis à zéro = 0 et saturés = 0 ; A témoin experts sans AWQ ≤ 1,022. Moi : B 1,009 ± 0,002, A 1,020 ± 0,002, compteurs 0/0
- **mesuré** : B (k 4/8) **8,9443 = 1,0984**, saturés 55 984 (0,0166 %) ; A (k 4/8) **351,00 = 43,1**, saturés 48 289 ; à k = 0/0 : B 8,3941 (= poste2 8,394), A 8,3076 (= 8,3076)
- **verdict** : **RÉFUTÉ**, les deux bras, mes deux prédictions aussi. La fusion (division dans le noyau, main e84a43d) est correcte : à k = 0/0 les deux PPL d'avant se rejouent au 10⁻⁴. Le défaut est dans le chemin 2ᵏ, et l'attribution par site le localise (§ 1). Retour poste4, rien d'autre sur la carte avant.

## 1. Attribution par site (un k à la fois, même ligne, 6 passes de 6 s)

```
bras  k_x  k_act   PPL       ratio    mis à zéro         saturés            s (4 fen.)
B     0    0       8,3941    1,0309   221 728 (0,066 %)  0                  6,1
B     4    0       8,3004    1,0194   221 993            34 286 (0,010 %)   6,5
B     0    8       9,0324    1,1093   0                  22 342 (0,007 %)   6,7
B     4    8       8,9443    1,0984   0                  55 984 (0,017 %)   26,5
A     0    0       8,3076    1,0203   334 170 (0,099 %)  0                  5,7
A     4    0       8,4162    1,0336   333 885            0                  6,1
A     0    8     369,80     45,4      0                  49 116 (0,015 %)   6,0
A     4    8     351,00     43,1      0                  48 289             24,9
```
- **k_x = 4 seul** : B passe de 1,031 à **1,019** — les blocs de `x/s` mis à zéro étaient bien le coût AWQ (poste7 § 1.1) ; il reste 34 286 blocs saturés (amax(x/s) > 168, invisibles sur 16 jetons) et 1,019 > 1,010. Sur A (sans table) +1,3 % : sans division, `x` n'avait rien à gagner et la saturation coûte.
- **k_act = 8 seul** : B 1,109, **A 45,4**. Deux faits distincts : (i) 22-49 k blocs saturés → `act/s_d` a des amax > 10,5 dans une fenêtre réelle, le max 0,289 d'poste1 sur 16 jetons a raté la queue d'un facteur ≥ 36 — le k du § 6.2 (« 36× le max observé ») est exactement mangé ; (ii) **A ×45 n'est pas une saturation** (même ordre de blocs saturés que B, PPL ×5 pire) : sur le site down **sans table** le facteur 2⁸ n'est pas compensé quelque part — asymétrie avec le site x sans table (A k_x = 4 → 8,42, sain). À lire : `model.py:1054/1177` (`awq_d` None) contre `:1049`, et le noyau côté table nulle ; hypothèse non vérifiée : `_gs_mma_cache` clé `id(gs)` (`:945`) partagée entre piles.
- **k = (4, 8) est 4× plus lent** (25-26 s contre 6 s pour 4 fenêtres) alors que chaque k seul ne l'est pas : à mesurer par poste4 avant le pas b=12, sinon B/B∅ ≤ 1,008× ne tiendra pas.

## 2. Conséquence pour la file (ETAT.md)
Bloc 2 rendu RÉFUTÉ ; blocs 3-5 (reconversion poste2, pas b=12, duel) attendent le correctif du site down. Rien ne bouge côté 0.6.6 (rondes Coder mesurées ce matin à MIN_T=5 sans W4A4 GLM). Le compteur global a suffi pour l'attribution ; un compteur **par site** (x / act) éviterait les 6 passes la prochaine fois — une ligne dans `_qa_compteurs`.
