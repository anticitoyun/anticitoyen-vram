# Levier 3 bis — ce qui reste dans le pas servi (chemin `gemv_marlin`, 654 lancements) : la mesure d abord, un seul levier à latence, et MMA_MARLIN reste opt-in (22/09, poste1, à sec)

* sources : `verdict-nsys-familles-22-09` (poste2 : Σ 6,749 ms, 654 lancements/pas, mur 7,083), lecture du code après le correctif d instrument `poste1-chemin-moe` 30870b12 (la ligne dit le chemin atteint), `verdict-m2-b12-21-09` (étroit servi : qkv 6,7 µs à 1,57 To/s, o 8,1 µs à 1,03), `poste7-c17-faux-ferme-20-09` (MMA_MARLIN), `poste4-scelle-b12-21-09` § 3 (routeur en un nœud : 14,8 µs contre 6,0, réfuté), regime.py:169 (ROPE_KV réfuté a3f1b7e)

## 1. Le pas servi, tel que la table le donne, ramené par couche (13,6 lancements)
| poste | ms/pas | lancements | par couche | statut |
|---|---|---|---|---|
| experts `gemv_marlin` gate·up + down | 3,937 | 96 | 2 | 94 % du plancher à D = 42 (M2) — calcul, pas latence |
| projections étroites int8 | **1,335** | 145 | 3 (q, kv, o) | **0,906 Go/pas à 0,68 To/s effectifs contre 1,55 possibles : la plus grande marge, 0,75 ms** ; M2 ne mesurait que qkv 6,7 + o 8,1 = 14,8 µs/couche, la table dit 27,8 : **à séparer par forme avant tout mot** |
| attention compacte | 0,459 | 48 | 1 | à ctx 256+ ; croît avec la longueur |
| routeur : GEMM cutlass + `_route_fusee` | 0,132 + 0,310 | 48 + 48 | 2 | **déjà le routeur compact** (`route_logits_fusee` = le MÊME cuBLAS puis `_route_fusee` à un warp) ; le nœud unique a été essayé : 14,8 µs contre 6,0 → **pas de levier** |
| normes (add_norm) | 0,241 | 97 | 2 | résidu déjà fusionné |
| rope + kv_write | 0,234 | 96 | 2 | fusion 3a `ROPE_KV` **réfutée** a3f1b7e (corrompt sous graphe) : ≤ 0,1 ms si le défaut est nommé un jour |
| moe_act | 0,052 | 48 | 1 | 1,1 µs : latence, rien à gagner |
| glue torch + copies | 0,049 | 28 | — | le levier 3 tel que conçu n avait plus d objet |
Somme : 6,749 ; mur 7,083 (trou 0,334 sous nsys, 0,025 sous frontiere-pas : le profileur pèse).

## 2. Le seul levier qui reste, et sa mesure d abord (poste2, 0 min de carte : la trace existe)
`familles-noyaux.py graphe_cuda_gpu_trace.csv --detail proj_etroites_int8` (ajouté, 30870b12 : ms/pas, lancements et µs par lancement, par nom + grille) sur la trace du 22/09 — **prédit** : trois formes ; q [4096 × 2048] ≈ 8,4 Mo, kv [1024 × 2048] ≈ 2,1 Mo, o [2048 × 4096] ≈ 8,4 Mo ; si q et o sont à ~10 µs (0,84 To/s) et kv à ~6 µs, la marge est **q + o : 2 × (10 − 5,4) µs × 48 ≈ 0,44 ms = 6,5 %** ; si les trois sont à 1,3-1,5 To/s, la table de poste2 compte autre chose dans la famille (regex `_etroit`/`splitK`/`gemvx` à relire) et la marge est ≤ 0,13 (o seul, M2).
Mécanisme (si la marge est là) : `_etroit_reduit_kernel` à M = 12, split-K par tranches (`gemm_etroit.py:176-179` : `tranches = min(ng, 2·SM/tuiles_n)`) — pour N = 4096 (q) les tuiles_n saturent et tranches → 1 : un seul programme par tuile lit K = 2048 en série ; pour o, K = 4096 avec N = 2048 : mieux découpé. Levier : forcer ≥ 2 tranches quand N/BN ≥ SM (q), au prix d une réduction déjà dans le noyau (compact) — **au bit ?** NON par construction : l ordre de sommation des tranches change (fp32) → équivalence à ± 1 ulp bf16 par noyau sur entrées réelles + PPL avec SE (REGLES § 3, noyau de décodage), pas « au bit ». Plafond 0,44 ms ; prédit −0,25 à −0,40 ms (−3,7 à −6 %) ; réfuté si `--detail` montre q et o ≥ 1,3 To/s, ou gain < 0,1 ms au banc à sec du noyau (≤ 1 min, `banc-horloge-decodage`), ou > 1 ulp.

## 3. `ACVRAM_MOE_DECODE_MMA_MARLIN=1` (chemin mma-a4 sur disposition unique) : reste opt-in, Marlin défaut pour de bon
`poste7-c17-faux-ferme-20-09` : (a) plus lent à l unité servie (u = 45 : ×1,234 ; gagne seulement à u ≤ 27) ; (b) **change la sortie** — `ppl-decode-kv` lot 12 +2,2 %, 9/11 témoins dégradés, « défaut réel du chemin sur routage réel, non expliqué » ; (d) certifie b=12 +25 %. Trois raisons indépendantes, dont une de qualité : le chemin mma-a4 de décodage ne peut pas devenir le défaut sans nommer (b) — et la ligne de régime le dira désormais s il est atteint. Conséquence pour B et C : leurs noyaux ne serviraient que sous cet opt-in ; ils restent en branche locale, sans commit.

## 4. Verdict sur la suite du chantier b=12
Après leviers 1 + 2 : 1 624 t/s, +1,8 % devant vLLM. Le pas est à 94 % du plancher sur les experts (58 %), et les seuls postes à latence restants (moe_act, glue) valent < 0,1 ms. Deux pistes seulement, toutes deux de **calcul** : (i) les étroites (§ 2, ≤ 0,44 ms, mesure d abord, pas au bit), (ii) l attention à longueur réelle (0,46 à 256 ; à 2 048 ?) — à mesurer par la même trace sur ctx 2 048 avant d y penser. Tout le reste (routeur, rope_kv, MMA_MARLIN) a déjà été essayé et réfuté : on ne le retente pas.
