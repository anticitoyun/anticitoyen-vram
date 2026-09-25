# Verdict — pièce 172 : B' au défaut, poids déquantifié partagé par la boucle par séquence (poste5, 25/09) — TENU

* **instrument** : `scratchpad/poste5-p172-25-09/prise.sh` (tests + 2 bras cassants ; `diag172.py` sur deux modèles ;
  TTFT servi par `ttft-charge.py` de la 165) → `prise-1.txt`, `prise-ttft.txt`, `diag-*.json`, `ttft.jsonl`
* **commit** : f0795b95 · **régime** : défaut (B' = `ACVRAM_DEPAQ_PARTAGE=1`), -lgc 2700 ; TTFT : 20/20 passes
  `repli_eager=0`, A porte `DEPAQ_PARTAGE=0` à sa ligne de régime, 0 passe nulle ; Qwen3.5-35B : Marlin refusé (157)
* **scellé** : `revue/poste5-piece172-scelle-25-09.md` (avant) · **durée** : tests 00:55-00:58:28, diag → 00:59:2x,
  TTFT → 01:13:26

## Au bit
* Tests : 171 verts (dont `test_depaquetage_partage_172.py` : naturel et Marlin, C1 et C2, 1 fabrication + 7
  réutilisations, témoin « GEMM groupée ≠ » vert ; couche GDN au bit, LOT=1 ≠). Bras cassants : GEMM groupée dans la
  couche → ROUGE ; partage perdu → ROUGE (5/5).
* Modèles : logits fp32 **égaux au bit** entre A et B', C1 et C2, sur Qwen3.8 ET Qwen3.5-35B ; bras cassant B' + LOT=1
  ≠ A sur les deux. Réutilisations par passage : 1 680 (Qwen3.8 : 48 couches × 5 linéaires × 7), 1 050 (Qwen3.5-35B).

## Vitesse (prédictions écrites avant)
| modèle | forward C1 (A → B') | forward C2 | prédit | TTFT servi C1 (8 simultanées) | TTFT C2 | prédit |
|---|---|---|---|---|---|---|
| Qwen3.8 | 365,1 → 319,4 ms **−12,5 %** | 417,4 → 370,4 **−11,3 %** | −10 à −15 | 429,2 → 389,8 **−9,2 %** | 464,6 → 427,0 **−8,1 %** | −9 à −13 / −7 à −11 |
| Qwen3.5-35B | 191,3 → 184,1 **−3,8 %** | 192,6 → 185,6 **−3,6 %** | −3 à −8 | 213,8 → 207,1 **−3,1 %** | 216,8 → 209,5 **−3,4 %** | −2 à −6 |

Toutes les prédictions sont tenues. Forward : A B' B' A en processus, médiane de 5 par passe ; TTFT : 5 passes par bras,
σ ≤ 2,3 ms.

## Verdict
* **TENU** : au bit (tests, bras cassants, logits de deux modèles), gain de préfill −11 à −13 % sur Qwen3.8 et −3,6 à
  −3,8 % sur Qwen3.5-35B, TTFT servi −8 à −9 % et −3 %. Environ la moitié du gain de LOT=1, sans rien changer à la sortie.
* **Limite** : le coût mémoire annoncé (≈ +230 Mo de pic, le temps d'une couche) n'est PAS mesuré en différence. `pic_Mo`
  cumule A et B' dans un même processus (27,0-27,5 Go sur Qwen3.8). Aucun OOM ni refus de capacité dans les 20 passes
  servies.
* Reste à faire avant le push : la suite complète sous verrou.

### Suite complète (f81483c1 contre la base ea85b7e7, 01:14:03-01:29:58, sous mon verrou)
HEAD 0 échec, 2 765 verts ; base 0 échec, 2 764 verts ; aucun échec propre. **TENU.**

## Addendum (25/09 01 h 5x, AVANT la mesure) — le pic de B' est-il dans le compte du planificateur ? (question de chef)
Lecture : la réserve de préfill (`config.py:355` `activations_prefill_bytes`, via `loader.py:1887` `_reserve_prefill`
→ `loader.py:1173` `_marge_carte`) comptait UNE matrice déquantifiée, la plus grosse (Qwen3.8 : MLP 17 408 × 5 120 ×
2 = 178 Mio). Sous B', une couche GDN garde ses cinq poids ensemble (232 Mio) : **+54 Mio hors compte**, et même pour
une seule séquence, où il n'y a rien à partager. Corrigé avant la mesure : (1) portée fermée pour une séquence
(`couches.py`) ; (2) la réserve prend max(plus grosse matrice, somme des poids d'une couche linéaire)
(`config.py`, `poids_bf16_couche_lineaire_bytes`). Tests : `test_reserve_depaq_172.py`, qui casse si le terme est retiré,
et `test_une_seule_sequence_ne_garde_rien`.
Prédiction (`pic.py`, pic − alloué avant, Qwen3.8 à max_model_len 8 192) : 1 × 8 192 : B' − A = 0 (portée fermée) ;
ancien comportement forcé : +0 à +232 Mio ; 8 × 1 024 : B' − A entre 0 et +232 Mio. Les activations du MLP dominent
probablement le pic, auquel cas les deux différences sont nulles.
**1re mesure (241d617c, 01:4x) : OOM dès le bras A (B' COUPÉ)** sur 1 × 8 192 jetons, en processus
(`load_model(max_model_len=8192, max_concurrent_seqs=8)`) : fla `chunk_gated_delta_rule` (`wy_fast.py:269`,
`w = k.new_empty(B, T, HV, K)`) demande 192 Mio et il en reste 152. Ce n'est pas B' : le préfill par défaut de 8 192
jetons ne tient pas dans ce chargement. Les tampons fp32 du préfill par blocs de fla (T × têtes v × 128 × 4 o, plusieurs
par couche) ne figurent pas dans `activations_prefill_bytes`. Tests : 20 verts. Mesure refaite aux longueurs qui passent
(1 × 4 096, 8 × 512), prédiction inchangée.
**2e mesure (02:1x) : OOM encore dès A, à 1 × 4 096** (11 064 Mio libres après chargement ; `gdn.py:253`, fla chunk). Le
`model(batch)` nu n'est donc pas le chemin servi pour les longues invites : l'OOM à 8 192 de la 1re mesure ne se
transpose PAS au service. Il reste inexpliqué, et je ne l'affirme pas comme défaut du service. Nouvel instrument, écrit
avant la 3e mesure : `pic2.py`, le moteur `Engine` comme `acvram serve`, cas 1 × 8 000 et 8 × 1 000, A B' A B'
basculés à chaud, OOM rattrapé et rapporté par bras. Prédiction inchangée.
**3e mesure (2dd9c0b0, 02:1x-02:22:36, moteur servi)** : 11 056 Mio libres après le moteur ; 1 × 8 000 jetons : pic
3 570,1 Mio, A = B' à 0,1 Mio près (A, B', A, B') ; 8 × 1 000 : 522,5 Mio, A = B'. Aucun OOM par le moteur : l'OOM des
mesures 1-2 tient au `model(batch)` nu. Doute, écrit avant la 4e mesure : 522 Mio pour 8 × 1 000, contre 3 570 pour
1 × 8 000, suggère que le moteur préfille ces 8 requêtes séquence par séquence ; B' n'y serait alors pas PRIS, et
« A = B' » ne dirait rien. 4e mesure : compteur de réutilisations relevé, cas 8 × 1 000 et 8 × 78 (celui du TTFT).
**4e mesure (moteur servi, compteur relevé, → 02:38:00)** :

| cas | pic A | pic B' | B' − A | réutilisations sous B' |
|---|---|---|---|---|
| 1 × 8 000 (3e mesure) | 3 570,1 Mio | 3 570,1 | 0 | — (une séquence : portée fermée) |
| 8 × 1 000 | 522,5 | 522,5 | 0 | **0** : le moteur préfille ces 8 requêtes séparément, B' n'est pas pris |
| 8 × 78 | 1 438,9 | 1 470,0 | **+31,1 Mio** | 1 680 (pris) |

**Réponse à chef** : le surcoût mesuré de B' est de +31 Mio, et seulement quand il est pris (préfill groupé de courtes
invites). Le plus long préfill servi (une invite) n'est pas touché. Il est désormais COMPTÉ : `config.py:395`,
`activations_prefill_bytes` prend max(plus grosse matrice, `poids_bf16_couche_lineaire_bytes` `config.py:397`), soit
+54 Mio de réserve sur Qwen3.8 (232 − 178), qui couvre les 31 mesurés. Cette réserve passe par `loader.py:1887`
`_reserve_prefill`, puis par `loader.py:1173` `_marge_carte`. Portée fermée pour une séquence : `couches.py:150`.
Tests : `tests/test_reserve_depaq_172.py` (casse si le terme est retiré), `test_une_seule_sequence_ne_garde_rien`.
Constat à part, non imputable à B' : `model(batch)` nu tombe en OOM dès 4 096 jetons dans ce chargement, alors que le
moteur tient 8 000. Cause non cherchée.
