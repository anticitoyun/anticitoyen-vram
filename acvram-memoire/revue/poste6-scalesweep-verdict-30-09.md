# ScaleSweep (arXiv 2606.07618) — échelle de bloc NVFP4 par balayage E4M3 à la conversion : prototype et mesure CPU (poste6, 30/09, branche poste6-scalesweep, à sec)

instrument : `outils/scalesweep-blocs-30-09.py` (CPU, 8 fils, `CUDA_VISIBLE_DEVICES=""`), `acvram.quant.nvfp4` échelles `balayage` / `balayage-w` (option explicite, défaut max6 inchangé)
commit : prédiction et prototype dans CE commit (en-tête relu par `--commit`) ; base origin/main 51e9f2d48
régime : à sec, aucune prise de carte ; charge hôte relevée dans l'en-tête de chaque sortie (1,7 au départ, llama-server 5 614 Mio hors verrou, verrou libre)
scellé : `scratchpad/poste6-scalesweep-30-09/*.json` (hors dépôt), totaux recopiés ci-dessous
mesuré : MSE et WMSE (importance = act_scale², proxy nommé) contre AbsMax (max6) et 4sur6 ; échelles E4M3 sous-normales ; échelles « écrasées » par Marlin (critère ya1) ; temps
verdict : (à remplir après la mesure — voir « Résultats »)
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
rendait balayage MSE 0,739 × max6 (4sur6 : 0,842), sous-normales 381 contre 449. Les prédictions ci-dessous portent sur les
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

## Résultats
(à venir)
