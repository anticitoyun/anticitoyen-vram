# Verdict — re-tampon 0.6.6 narrow OFF : 13,60 ms / 0,494 J ; cellule narrow 3 × 3 : moyenne 0,9984 tenue, condition par tranche fausse sur les trois → INDÉTERMINÉ à la lettre

- **instrument** : `certifie-b12-15-09.py` (rondes ctx 2048 / 256 / ≥ 20 s, `energie.py` b11b8a3 + chrono hôte, écart ≤ 0,02 s) ; `outils/ppl-narrow-b12-coder30b.py` + `PPL_DECALAGE` (c0a8b36), plan `max_concurrent_seqs=12`, `kv_max_tokens` 25 344 ≥ 24 576, 24 564 jetons notés sur les 9 bras ; JSON `scratchpad/retampon-cellule-narrow-16-09/`
- **commit** : travail/laure figé **c0a8b36** (code `acvram/` = main b06e337), une prise de carte 07:24-07:35
- **régime** : une carte, -pl 400, horloge libre 2 880-2 890 MHz, 34-51 °C, bridage puissance signalé sur les 4 rondes (comme l'officiel 66c7532) ; Qwen3-Coder-30B-A3B-nvfp4, MMA=1, MIN_T défaut **5** (officiel : 9)
- **scellé** : (1) Sage § 1 : A ≈ 14,0 ms / 0,51 J ; moi : A ∈ [13,7 ; 14,4] sinon bissection avant publication. (2) Sage § 3 : moyenne B/A ∈ 1,000 ± 0,002 ET par tranche |B/A − 1| ≤ 1,5 × |A-g/A-e − 1| ; réfuté si moyenne > 1,002 ou tranche > 1,004
- **mesuré** : (1) A 13,584 / 13,609 ms · 0,4935 / 0,4950 J · 806 t/s ; B 11,383 / 11,381 · 0,4114 / 0,4117 · 962 t/s ; B/A −16,3 % ms, −16,7 % J. (2) B/A = 1,002286 · 0,994605 · 0,998309 ; moyenne **0,998400** ; témoins 0,00143 · 0,00165 · 0,00010
- **verdict** : (1) re-tampon OFF = **13,60 ms / 0,494 J / 806 t/s** (ON : 11,38 / 0,412 / 962) — mon intervalle réfuté, cause en § 1 ; (2) aucune condition de réfutation atteinte, condition « tenu » fausse sur 3/3 tranches → **INDÉTERMINÉ**, Sage arbitre (§ 2)

## 1. Re-tampon : −2,6 % sous l'officiel, pas un re-tampon à l'identique

L'officiel A (13,956 / 0,508, 66c7532) et ce A (13,60 / 0,494) diffèrent par un seul défaut : MIN_T 9 → 5 (b94d74d). J'avais scellé « sans effet à b=12 » : faux, car les rondes incluent la queue (lot 12 → 1 en fin de ronde), où les lots 5-8 passent désormais en MMA (cellule b=5 : ms −0,3 %, J −4,1 % ; b=6 : −10,5 %). Le B bouge du même −2,9 % (11,72 → 11,38), et B/A reste −16 % : l'écart narrow est reproduit, le niveau absolu est celui de main d'aujourd'hui.
Bissection à une variable (MIN_T=9 contre 5, narrow OFF, même arbre, 4 rondes 9/5/9/5) : **M9 13,815 / 13,785 ms · 0,5026 / 0,5022 J ; M5 13,621 / 13,622 · 0,4956 / 0,4967** → MIN_T=5 vaut −1,3 % ms, −1,2 % J sur les rondes (la queue), et M5 reproduit A à 0,01 ms. Reste **−1,2 % entre 66c7532 et b06e337 à MIN_T égal** (13,97 → 13,80), non attribué : `acvram/` a bougé sur model.py, acvram_kernels.cu, convert.py (3b44846, 908e926, 944fc9d, 7a48bb6 : AWQ gate/up, table d'unité, témoin GLM) — un de ces commits touche le pas Coder, ou le jour (T 41-55 °C contre 34-48). Je ne le publie pas comme gain ; bissection sur 4 commits si Sage la veut (10 min).
**Chiffre à publier** : 0.6.6 narrow OFF, MIN_T=5 : 13,60 ms / 806 t/s / 0,494 J (témoin ON 11,38 / 962 / 0,412). Pas de bridage horloge ; le drapeau « bridage puissance » (396-398 W sous -pl 400) est le même régime que l'officiel.

## 2. Cellule 3 × 3 : ce que les trois tranches disent

```
tranche   A-graphes   B-graphes   A-eager     B/A        |B/A−1|  témoin   borne 1,5×  tenu
0         8,498852    8,518279    8,511008    1,002286   0,00229  0,00143  0,00214     non (+0,00015)
24576     9,991444    9,937541    9,975014    0,994605   0,00539  0,00165  0,00247     non (2,2× la borne)
49152     8,760416    8,745605    8,759524    0,998309   0,00169  0,00010  0,00015     non (témoin ≈ 0)
moyenne                                        0,998400   ∈ 1,000 ± 0,002 : tenu
```
- Tranche 0 = Manon à 10⁻⁶ sur les trois bras : **déterminisme prouvé**, elle compte (Jérôme).
- Preuve de chemin : `lancements_narrow_gemm` 0 / 1 551 / 0 sur chaque tranche (1 551 = 8 passes de capture × 193, `verdict-ppl-narrow-b12` § 2), `pas_t_le_32` 2047/2047.
- Lecture : narrow bouge la PPL de +0,23 / −0,54 / −0,17 % selon le texte, **signes non concordants**, moyenne favorable — pas un coût systématique (ma prédiction 2 tenue : « réfuté si les trois > 1,001 de même signe »). Mais l'amplitude est 1,6 à 17 × celle de graphes/eager : narrow n'est pas « du bruit de l'ordre des sommes », c'est une autre arithmétique (int8 étroit), dont l'effet est plus grand que la borne choisie. Sur la tranche 49 152 le témoin vaut 0,0001 : la borne 1,5 × devient 0,00015, inatteignable pour tout écart (prédiction 3, dite d'avance) — cette condition ne peut pas rendre « tenu » quand graphes et eager coïncident presque.
- Je ne tranche pas : « tenu » exige les deux conditions, l'une est tenue, l'autre fausse trois fois ; « réfuté » n'est atteint par aucune tranche ni par la moyenne. Question pour Sage : la condition par tranche visait-elle |B/A − 1| ≤ 0,004 (alors tenu 2/3, tranche 24 576 hors borne dans le sens favorable) ou bien 1,5 × un témoin qui peut être nul ?

Ma prédiction 1 (tranche 0 identique) tenue ; prédiction re-tampon réfutée (§ 1).
