# AWQ par expert dans la pile MoE — 15/09 (poste4)

Ordre : `revue/poste7-glm-awq-pile-15-09.md` (poste7). Branche `poste4` a82bc28…fb40fe9.

## Ce qui est câblé
- Chargeur (`model.py:_try_build_stacks`) : tables `awq[nom]` [E, K_in] bf16, 1 là où l'expert n'a
  pas d'échelle ; refus (repli boucle) seulement pour Hadamard ou gate ≠ up.
- Décodage MMA : `moe_route_pack(…, awq)` divise la ligne (jeton, expert) par `s[e]` en bf16 juste
  avant `nvfp4_quant_act`, rend `e_sorted` ; `moe_act(…, awq, e_sorted)` divise l'activation avant
  down. Témoin torch (`ACVRAM_MOE_ROUTE_PACK=0`) fait de même.
- Décodage GEMV (`_forward_grouped`) : `x_g = x[tok] / s[e]`, `act / s_down[e]`.
- Prefill groupé (`_forward_prefill_grouped`) : idem — absent au premier jet, la sortie aurait
  changé en silence dès un expert à échelle.

## Équivalence (tests/test_moe_awq_pile.py, 7 cas, carte)
- GEMV : pile == boucle par expert à ≤ 1 ulp (du max |ref|).
- MMA et prefill (W4A4 contre la boucle W4A16) : écart normalisé avec table = 0,97 × l'écart sans
  table contre la boucle sans échelle ; témoin sans table > 2 × (88 ulp).
- `moe_route_pack` avec table : bit à bit égal au calcul torch.
- Bogue trouvé au premier jet : `moe_act_kernel` ignorait `awq` (écart MMA 0,234 → 0,104 après).

## Coût (Coder-30B, aucune échelle réelle : `ACVRAM_MOE_AWQ_TEMOIN=1` = tables de 1, chemin exécuté)
Prédiction scellée ≤ +0,10 ms/pas. ABAB b=12, 1 787 pas, 25 s, 400 W :

| bras | pas ms | J/jeton | W |
|---|---|---|---|
| A1 / A2 (sans table) | 13,926 / 13,958 | 0,5077 / 0,5087 | 399 |
| B1 / B2 (tables) | 14,056 / 14,073 | 0,5123 / 0,5125 | 399 |

**+0,12 ms (+0,9 %), 2,5 µs/couche** — au-dessus des 2 µs de poste7 ; profil GPU d'un pas +0,07 ms
(route_pack +0,5 µs, moe_act +0,2 µs par couche), le reste au plafond de puissance.

## Garde d'unité (poste7 § 8) et scellé Coder 0,00 ± 0,03 ms — RÉFUTÉ tel que scellé
Table = 1 partout (échelle absente écrite en identité, poste2 2205709) → `awq[nom] = None`, produit
sauté (`ACVRAM_MOE_AWQ_TEMOIN=1` construit puis saute ; `=2` force le produit, témoin du +0,12).
ABAB b=12 (b12-unite-*.json), même chemin de calcul dans les deux bras :

| bras | pas ms | J/jeton |
|---|---|---|
| A1 / A2 (rien) | 13,948 / 13,976 | 0,508 / 0,510 |
| B1 / B2 (tables construites puis sautées) | **14,210** / 14,044 | 0,516 / 0,512 |

Δ = +0,26 et +0,07 ms : le scellé (0,00 ± 0,03, réfuté > 0,05) est réfuté, mais les deux B
s'écartent de 0,17 ms entre eux — la ronde ABAB de 25 s ne résout pas ±0,03. Seule différence
réelle : 34 Mo de tables allouées au chargement puis libérées (placement mémoire ?). À trancher
par plus de bras ou par un profil GPU pur (qui donnait +0,00 par construction). Pas de garde à
ajouter tant que l'instrument ne sait pas dire « faux » à ce niveau.

## Instrument : +1 s dans `energie.py` (b11b8a3)
Première passe : +0,56 ms, et cinq bras à 25,04 / 26,04 s exactement. `Energie.__exit__` attendait
le `sleep(periode)` du fil de sonde avant de lire `time.time()` : durée, ms/pas, jetons/s, W moyens
portaient +[0, 1) s (jusqu'à 4 % sur 25 s) ; les joules sont justes. Corrigé (relevé avant le join),
test qui casse avec la faute. **Tous les ms/pas certifiés par ce chemin depuis sa création portent
ce biais** ; les A/B à moins de ±0,5 ms sont à relire.

## Non fait / à poste7
- Scellés PPL 0,998 ± 0,002 (poste2, GLM avec AWQ experts) et pas GLM b=12 ≤ 1,03× : pas de modèle
  GLM converti avec échelles par expert sous la main ici.
- A = 13,93 ms sur l'arbre fusionné (main c4a7106) contre 12,80 officiel v0.6.5 : à re-tamponner
  (77b5def le prévoyait), ce n'est pas l'AWQ (A n'a pas de table).
