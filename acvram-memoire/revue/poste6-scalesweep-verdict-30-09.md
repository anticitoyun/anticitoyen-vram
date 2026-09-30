# ScaleSweep (arXiv 2606.07618) — échelle de bloc NVFP4 par balayage E4M3 à la conversion : prototype et mesure CPU (poste6, 30/09, branche poste6-scalesweep, à sec)

instrument : `outils/scalesweep-blocs-30-09.py` (CPU, 8 fils, `CUDA_VISIBLE_DEVICES=""`), `acvram.quant.nvfp4` échelles `balayage` / `balayage-w` (option explicite, défaut max6 inchangé)
commit : prédiction et prototype dans CE commit (en-tête relu par `--commit`) ; base origin/main 51e9f2d48
régime : à sec, aucune prise de carte ; charge hôte relevée dans l'en-tête de chaque sortie (1,7 au départ, llama-server 5 614 Mio hors verrou, verrou libre)
scellé : `scratchpad/poste6-scalesweep-30-09/*.json` (hors dépôt), totaux recopiés ci-dessous
mesuré : MSE et WMSE (importance = act_scale², proxy nommé) contre AbsMax (max6) et 4sur6 ; échelles E4M3 sous-normales ; échelles « écrasées » par Marlin (critère ya1) ; temps
verdict : PROMETTEUR — P1-P7 tenus : balayage MSE 0,735 × max6 et 0,875 × 4sur6 sur les DEUX modèles ; WMSE 0,687 × max6 (balayage-w) ; sous-normales 0,97× ; aucune écrasée par ligne ; 8-15× le temps CPU de max6. La perplexité sur carte tranche (hors pièce)
durée : prévu 0 min de carte / tenu 0

## L'article, en cinq lignes (lu à sec, PDF 30/05/2026, Lin & Wan, PKU)
* NVFP4 = E2M1 par bloc de 16 × échelle E4M3 par bloc × échelle globale fp32. L'initialisation courante des échelles est AbsMax
  (s_base = amax/6) ou « 4/6 » (Cook et al. 2026 : notre `4sur6`, deux candidats). Un E4M3 positif n'a que 126 valeurs : on peut
  TOUTES les essayer par bloc — ScaleSweep restreint la fenêtre et prend celle de moindre (W)MSE.
* Bornes prouvées (bloc de 16) : au-delà de 12/7 s_base, s/2 fait toujours mieux (lemme 4.1) → +7 motifs binaires ; en dessous
  de 4/5 s_base jamais optimal en MSE (lemme 4.3) → −3 motifs ; WMSE : borne basse empirique s_base/2 → −8 motifs.
* Global : S = amax/(6·448) comme chez nous ; la figure 2 montre que S change peu (NMSE 7,96-8,00 ‰ AbsMax, 6,16 ‰ ScaleSweep : −23 %).
* WMSE = Σ_j diag(XᵀX)_j (w_j − ŵ_j)² (approximation diagonale du hessien, calibration nécessaire).
* Résultats article (Llama-3.1-8B, Qwen3-4B/8B, RTN et GPTQ) : +0,5 à +3 points de moyenne sur 5 bancs selon l'agressivité
  (W, WA, WAKV, WAKVQ) ; surcoût de conversion « négligeable » sur GPU (table 7).

## Prototype (`acvram/quant/nvfp4.py`, HORS défaut)
`ECHELLES += balayage, balayage-w` ; `_balayer` : par tranche de 512 lignes, motifs base−3…+7 (−8 pour la WMSE), bornés 0x01…0x7E,
bloc nul inchangé, strictement mieux que s_base pour bouger ; `importance` [entrées] pour la WMSE ; stats `balayes`, `sous_normales`.
`tests/test_nvfp4_balayage.py` : référence en boucle explicite au bit, jamais pire que max6 ni 4sur6 (le candidat amax/4 est dans
la fenêtre), WMSE jamais pire, max6/4sur6 inchangés au bit. Mêmes format, noyaux et manifeste : rien ne change pour le moteur.

## Prédiction et seuils — écrits AVANT la mesure
Aveu : l'instrument a été essayé à blanc sur 6 tenseurs (couche 0, experts 0-1 du MoE, 9 M poids) avant d'écrire ceci ; il y
rendait balayage MSE 0,739 × max6 (4sur6 : 0,842), sous-normales 381 contre 450. Les prédictions ci-dessous portent sur les
sélections complètes (dense : couches 0, 12, 24, 36, 47, 35 tenseurs ; MoE : couches 0, 16, 32, 47, 128 experts, 1 552 tenseurs).
Sur poids gaussiens (test) : balayage 0,73 × max6, 0,87 × 4sur6 ; balayage-w WMSE 0,67 × max6.

| # | grandeur | prédiction | seuil « prometteur » / alarme |
|---|---|---|---|
| P1 | MSE balayage / max6 (dense et MoE) | 0,70-0,85 | prometteur si ≤ 0,90 sur les DEUX modèles |
| P2 | MSE balayage / 4sur6 | 0,85-0,95 | prometteur si ≤ 0,96 (sinon 4sur6 suffit, pièce Q1) |
| P3 | WMSE balayage-w / max6 (dense, proxy act_scale²) | 0,60-0,80 ; et balayage-w / balayage ≤ 0,95 en WMSE | proxy jugé utile si balayage-w bat balayage en WMSE d'au moins 5 % |
| P4 | sous-normales E4M3, balayage / max6 | 0,80-1,10 (la fenêtre descend au plus de ×0,7) | alarme si > 1,20 |
| P5 | sous-normales, balayage-w / max6 | 1,0-1,6 (borne basse s_base/2) | alarme si > 1,6 |
| P6 | écrasées Marlin (piles de 128 experts, facteur commun et par ligne), balayage / max6 | 0,8-1,1 ; par ligne : 0 → 0 | alarme (lien ya1) si par ligne > 0 alors que max6 = 0, ou commun > 1,2× |
| P7 | temps CPU balayage / max6 par tenseur | 8-14× (11 candidats) ; balayage-w 12-20× | acceptable si la conversion GPU reste < 2× (à voir plus tard, hors pièce) |

Issues nommées avant : (a) P1 tenu mais P2 non → le gain vient du candidat amax/4 déjà pris par 4sur6, ScaleSweep n'apporte rien
de plus ; (b) P3 non tenu → le proxy act_scale² (échelle AWQ, pas E[x²]) ne porte pas d'information, la WMSE reste à faire avec une
vraie calibration ; (c) P4-P6 en alarme → le balayage fabrique des sous-normales/écrasées, donc des refus Marlin (ya1) : à écarter
ou à borner ; (d) tout tenu → « prometteur », la perplexité sur carte tranchera (le MSE des poids n'est pas la PPL, table 4 de
l'article le dit elle-même) ; (e) l'échantillon (5 + 4 couches) n'est pas le modèle : les totaux sont pondérés par le nombre de poids
et les extrêmes (couche 0, dernière) sont dedans exprès.

## Résultats (mesure 12 h 00-12 h 03, commit 2f3044f57, charge 1,2 au départ → 8 pendant, 8 fils ; JSON dans le scratchpad de session)
Dense Qwen2.5-Coder-14B-Instruct (35 tenseurs, 1 376 M poids, 116 s ; importance = act_scale² du converti AWQ « pur ») ; MoE Qwen3-Coder-30B-A3B-Instruct
(1 552 tenseurs dont 128 experts × 3 × 4 couches, 2 416 M poids, 71 s ; pas d'act_scale → pas de WMSE). Le bf16 d'ORIGINE est requantifié
(pas le poids mis à l'échelle AWQ du converti) : on compare des règles d'échelle, pas des convertis.

| grandeur | dense | MoE | prédit | tenu |
|---|---|---|---|---|
| P1 MSE balayage / max6 | **0,7346** (par tenseur 0,724-0,760) | **0,7339** | 0,70-0,85 (≤ 0,90) | oui |
| P2 MSE balayage / 4sur6 | **0,875** (0,872-0,886) | **0,875** | 0,85-0,95 (≤ 0,96) | oui |
| P3 WMSE balayage-w / max6 ; balayage-w / balayage | **0,687** ; **0,926** (par tenseur 0,71-0,99, k_proj L0 ≈ 1) | — | 0,60-0,80 ; ≤ 0,95 | oui |
| MSE de balayage-w / max6 (prix de la WMSE) | 0,770 (+4,8 % sur balayage) | — | non prédit | — |
| P4 sous-normales balayage / max6 | 0 / 0 (aucune : AWQ + g par tenseur) | **0,967** (132 042 / 136 551 ; 1,45 % des 9,4 M blocs) | 0,80-1,10 | oui |
| P5 sous-normales balayage-w / max6 | 0 / 0 — non mesurable ici | — | 1,0-1,6 | sans objet |
| P6 écrasées Marlin, piles de 128 experts | — | commun 0,967× (= sous-normales, une pour une) ; **par ligne 0 → 0** (12 piles) | 0,8-1,1 ; 0 → 0 | oui |
| P7 temps CPU / max6 | balayage 7,6× ; balayage-w 11,0× | balayage 14,5× (petits tenseurs, surcoût fixe) | 8-14× ; 12-20× | oui (MoE 14,5 vs 14) |
| blocs ayant quitté s_base | 67 % (57,7 M / 86,0 M) | 67 % | — | — |

4sur6 seul vaut 0,840 × max6 sur les deux modèles (pièce Q1 confirmée) ; ScaleSweep enlève encore 12,5 % de MSE par-dessus — le gain ne vient
donc pas du seul candidat amax/4 (issue (a) écartée). Le proxy act_scale² porte de l'information (issue (b) écartée : −7,4 % de WMSE contre
balayage, jusqu'à −29 % sur down_proj L47), mais ce n'est pas diag(XᵀX) : la WMSE vraie attend une calibration. Aucune sous-normale ni écrasée
créée (issue (c) écartée) : la fenêtre descend au plus de ×0,7 et le balayage en retire même 3 %. Reste l'issue (d) : le MSE des poids n'est pas
la PPL ; l'article gagne +0,5 à +3 points de bancs pour −23 % de NMSE, nous avons −26,5 %.

## Suite (hors pièce, sur carte, à l'ordre de chef)
Convertir un alias avec `--echelle balayage` (à brancher dans le CLI de conversion : `regler_echelle` existe, l'option n'est pas exposée) et
mesurer la PPL appariée contre max6 et 4sur6 (méthode bootstrap de la 138) ; prédiction à écrire alors. Le format ne change pas : mêmes noyaux,
même manifeste, Marlin inchangé (par ligne 0 écrasée).
