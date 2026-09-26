# Verdict — pièce 147 L3' : dépaquetage Marlin → bf16 au débit de `nvfp4_dequant` (poste6, 24/09)

instrument : cellule TTFT de la 145 (`scratchpad/poste6-p145-24-09/prise.sh <gemma|qwen38> 1`, serve + `ttft-service-p145.py`, invites distinctes de 512 / 2 048 / 4 096, `--no-prefix-cache`, ABBA par relance A B B A A), preuve `/metrics` `depaquetage {cuda: n, triton: 0}` ; profil torch `scratchpad/poste6-p147-24-09/profil-prefill.py` (v1, 11 h 56) ; tests `tests/test_depaqueter_cuda_p147.py`
commit : v1 9e7024cd / e5d550cd (mesurée 11 h 23-11 h 42), **v2 96bd7ca7** (mesurée sur 3f070a49 = v2 + notes 155, 13 h 37-13 h 56) ; extension recompilée sous verrou (163 s, 166 s) ; HEAD asserté rc 65 ; `acvram.__file__` = travail/poste6
régime : identique à la 145 (b=1, `-lgc 2700` posé, préfill au plafond 400 W dans les deux bras, SM lue 2 416-2 440, cpu-safe=off 100/100, compute-apps début = fin) ; A = défaut ; B = PROJ_MARLIN unique + v2 + dépaquetage **CUDA v2** (`depaqueter_marlin_kernel`, `ACVRAM_DEPAQUETAGE=auto` → cuda, 27 264 / 31 008 appels comptés, 0 Triton)
scellé : `revue/poste6-piece147-dossier-prefill-marlin-24-09.md` § « Scellé L3' » (9e7024cd, avant compilation) — TENU si |Δ TTFT B − A| ≤ 10 ms à 2 048 et 4 096, ≤ 15 ms à 512, prises gemma 1 et qwen38 1 ; au bit contre Triton et torch, bras cassant, défaut non touché
mesuré : **au bit** : 105 passed (11 formes × cuda = triton = torch à `torch.equal`, vue par colonne, vue de pile sans copie, bras cassant ≠ référence, défaut sans appel). **v1** (un fil par (tuile, 128 k), tampon de transposition, copies `.contiguous()` des vues) : gemma Δ +33,1 / +32,0 / +34,0 ms, Qwen3.8 +28,3 / +26,5 / +26,8 → FAUX ; profil : `depaqueter_marlin_kernel` 64,8 ms contre `nvfp4_dequant_kernel` 47,4 (410 appels), copies +7,6 ms. **v2** (un fil par (colonne, tuile), lignes entières en deux uint4, vues à pas libre sans copie) : **gemma Δ +3,2 / +1,9 / +3,9 ms (B/A 1,012 / 1,002 / 1,002), Qwen3.8 +2,8 / +2,1 / +2,0 ms (1,012 / 1,003 / 1,001)** ; J par préfill Δ ≤ +1,5 J (gemma), −0,2 à +4,4 J (Qwen3.8) ; dispersion par lot ≤ 0,5 %
verdict : **TENU** (v2) — le surcoût constant de PROJ_MARLIN au préfill passe de +47 / +40 ms à **+2 à +4 ms** à toute longueur ; B = A à 1,2 % près à 512 jetons et à 0,2 % à 2 048 et 4 096 ; le préfill n'est plus un argument dans la décision « PROJ_MARLIN par défaut »
durée : compilations 163 + 166 s ; tests 3 × ≤ 4 s ; prises v1 421 + 401 s, v2 443 + 426 s ; profil 2 × 40 s ; trois prises refusées (rc 65) par des commits pendant la file — faute consignée

## Chiffres v2 (médianes ; A 3 lots, B 2 lots, même prise)
| modèle | L | TTFT A | TTFT B | Δ ms | B/A | J A | J B |
|---|---|---|---|---|---|---|---|
| gemma4 31B | 512 | 262,2 | 265,4 | +3,2 | 1,012 | 84,0 | 85,1 |
| gemma4 31B | 2 048 | 950,5 | 952,4 | +1,9 | 1,002 | 304,1 | 305,6 |
| gemma4 31B | 4 096 | 2 025,2 | 2 029,1 | +3,9 | 1,002 | 648,6 | 650,2 |
| Qwen3.8-27B | 512 | 228,9 | 231,8 | +2,8 | 1,012 | 73,5 | 74,1 |
| Qwen3.8-27B | 2 048 | 755,5 | 757,7 | +2,1 | 1,003 | 242,4 | 242,2 |
| Qwen3.8-27B | 4 096 | 1 439,8 | 1 441,8 | +2,0 | 1,001 | 458,1 | 462,5 |

## Lecture
* Le +47 ms de la 145 se décomposait en : dépaquetage Triton à mi-bande (+17 ms contre `nvfp4_dequant`), tampon et écriture par
  segments de 32 o (v1 encore +17), copies `.contiguous()` des vues q/k/v (+7,6 ms, 366 copies), reste (+3-4 ms : trois GEMM par
  pile au lieu d'une, `.contiguous()` de x). La v2 supprime les trois premiers ; le reste (+2-4 ms) est le prix des vues séparées.
* Les deux bras payent toujours ≈ 48 ms par requête de matérialisation bf16 (72 Go de trafic sur gemma) : L2 (GEMM W4A16 sans
  matérialisation) reste la pièce ± 1 ulp à part, pour les DEUX dispositions — la veille (sm120_gemm) montre un W4A16 tensoriel à
  déquantification en registres ; s'il arrondit chaque poids en bf16 comme `nvfp4_dequant` avant la MMA, il tombe dans la classe
  « ordre de sommation » admise en opt-in, ce que la GEMM Marlin (échelle après la somme) ne fait pas.
* Trouvaille annexe (pièce 155) : 361 copies HtoD paginées = 70 ms par préfill de 512 dans les DEUX bras.
* Faute de conduite : commits (155) pendant une file de prises → HEAD ≠ ATTENDU, prises refusées deux fois (13 h 06, 13 h 36) ;
  règle prise : aucun commit sur la branche tant qu'une prise de cette branche est en file.

## Pour chef
Code opt-in (`ACVRAM_PROJ_MARLIN=1` seul le déclenche), défaut inchangé au bit ; `ACVRAM_DEPAQUETAGE` documenté ; à fusionner avec
la décision PROJ_MARLIN. Si la décision est « défaut » : le TTFT servi ne bouge pas (≤ +4 ms), les gains de décodage des 129/142
restent seuls en balance.

## Erratum 14 h 1x — la « trouvaille annexe » HtoD est fausse
Les 361 copies HtoD (70 ms) venaient de la construction de l'Engine que `profil-prefill.py` refaisait à chaque préfill, pas du
forward (4 copies, 0,0 ms — `revue/poste6-piece155-verdict-24-09.md`). Le verdict 147 (Δ TTFT B − A) n'en dépend pas : les
deux bras portaient le même artefact et il était hors de la fenêtre de la cellule servie.
