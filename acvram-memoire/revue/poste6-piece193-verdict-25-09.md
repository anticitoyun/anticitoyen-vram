# Verdict — pièce 193 : k/v int8 étroit à 57-69 % (185) — chemin servi, cause, et ce que la 185 n'avait pas vu (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p193-25-09/banc.py` (`_etroit_reduit_kernel` sur poids int8 G=128 = la vue g128 servie ;
  `occ.poids_int8`, `lat.chrono_froid`, graphe, 60 rejeux, médiane ; **49 copies** de k/v = 262 Mo en rotation, 9 d'o/out = 283 Mo) ;
  log `scratchpad/poste6-p193-25-09/banc.log`.
* **commit** : a077a2bc8 (poste6-193 = origin/main 0c54811f2 + scellé + banc).
* **régime** : carte 0, horloge LIBRE (225 → 2 580 MHz relevés ; comparaison en % du plancher 1,55 To/s et à l'intérieur de la
  prise), 4 warps / 3 étages, verrou poste6-p193-banc, compute-apps début = fin (1 python de service, 686 Mio).
* **scellé** : `scratchpad/poste6-p193-25-09/scelle.md` (a077a2bc8, avant la prise).
* **mesuré** : k/v 1024 × 5120, M = 8 (idéal 3,46 µs) ; o/out 5120 × 6144 (idéal 20,8 µs) ; M = 1 k/v.
* **verdict** : (1) q‖k‖v est DÉJÀ SERVI empilé : rien à fusionner, aucun ABBA ; (2) sur la forme isolée, cause = grille
  (`decouper_k` vise 2 programmes/SM → tranches trop fines) + un plancher de ≈ 2,9 µs par appel ; (3) **o/out, forme SERVIE
  (64 appels/pas) : T=4 −2,3 µs (−9,1 %) sur T=5 servi** — quantification en vagues, falsificateur (c) DÉCLENCHÉ, hors bit.
* **durée** : prévu ≤ 4 min ; tenu 163 s (`tenue=163s`), après 442 s de file (une 1re prise à vide : `.venv/bin/python` absent
  du worktree, carte tenue 1 s après 27 min de file — le venv partagé vit dans l'arbre principal).

## 1. Ce qui est servi (à sec, contrôle : le compte du nsys 173)
Alias mixte-i8c : TOUS les int8 sont par canal (échelles [N, 1], G = K ; safetensors). `loader.py:938` → `Attention.fuse`
(`attention.py:159`) → `stack_int8_linears` (`layers.py:1350`, conditions tenues, plan cuda:0 × 64) ; `_proj` (`attention.py:272`)
sert la pile à t ≤ 256, la voie 3b aussi (`:366`). **Appel servi : 14336 × 5120** (q 12288 avec porte + k + v), 224 tuiles × 2
tranches = 448 programmes (185 b : 50,77 µs, 93 %). Compte 173 : 193 = 16 × (qkv + o) + 48 GDN × 3 (avant 176) + 8 MLP × 2 + tête.
**Les lignes k, v ET q (6144) de la 185 ne sont jamais servies à 2 ≤ b ≤ 16.** À b = 1, pas de vue g128 (`kernels/__init__.py:1366`)
→ `eligible` refuse G = 5120 → `ext.int8_gemv` CUDA (`:1397`) : la ligne M = 1 de la 185 (Triton) n'est pas le noyau de b = 1.

## 2. Pourquoi 1024 × 5120 est lent (forme non servie ; grille servie = 16 tuiles × 20 tranches, gpt 2)
| réglage (M = 8) | prog. | µs | % | lecture |
|---|---|---|---|---|
| T=1 gpt 40 | 16 | 28,01 | 12,4 | 16 SM : 12 Go/s chacun — H_prog vraie, pire que prédit (10-16) |
| T=2 / T=4 / T=5 | 32 / 64 / 80 | 15,18 / 9,12 / 8,04 | 23 / 38 / 43 | monte avec les programmes |
| T=8 gpt 5 | 128 | 6,53 | 53 | |
| **T=10 gpt 4** | 160 | **6,36** | **54,4** | meilleur ; BN32_T10 (320 prog.) 6,36 aussi : au-delà de 160, c'est gpt qui compte |
| T=20 gpt 2 (**servi**) | 320 | 7,87 | 44,0 | −1,51 µs disponible, seuil 0,5 dépassé : `decouper_k` mal réglé ici |
| T=40 gpt 1 | 640 | 10,58 | 32,7 | H_gpt vraie (+34 %, prédit ≥ +10 %) |
| BN16, T 5/10/20 | 320-1280 | 7,28-9,26 | 37-48 | tuile de 16 colonnes : toujours pire |
M = 1 : Triton T servi 7,16 ; `int8_gemv` par canal (servi à b = 1) 6,40 (M = 8 : 14,06, non servi). Falsificateurs (a) ≤ 4,0 µs et
(b) T=1 ≤ 7 µs NON déclenchés : même à la meilleure grille il reste ≈ 2,9 µs (rampe + réduction) sur un appel de 3,5 µs d'octets.
**Écart avec la 185** : 7,87 ici contre 6,03 (57 %) et 69 % à M = 1. La 185 tournait « 12 poids en rotation (> L2) » : 12 × 5,36 Mo
= 64 Mo < 96 Mo de L2 (5090) → ses lignes k/v étaient PARTIELLEMENT CHAUDES ; à froid la forme vaut 44 %. Ses lignes q/o/GDN
(≥ 31 Mo × 12) étaient froides et tiennent (o/out 25,56 ; ici 25,22).

## 3. o/out 5120 × 6144, forme SERVIE (o × 16 + out GDN × 48 = 64 appels/pas ; pour poste1, 185 c)
| T (gpt) | 1 (48) | 2 (24) | 3 (16) | **4 (12)** | 5 (10) servi | 6 (8) | 8 (6) | 12 (4) | 16 (3) | 24 (2) | 48 (1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| programmes | 80 | 160 | 240 | 320 | 400 | 480 | 640 | 960 | 1280 | 1920 | 3840 |
| µs | 34,56 | 24,41 | 23,26 | **22,92** | 25,22 | 23,72 | 27,57 | 26,35 | 26,89 | 29,45 | 31,90 |
| % plancher | 60 | 85 | 89 | **90,6** | 82,3 | 87,6 | 75 | 79 | 77 | 70 | 65 |
Lecture : 400 programmes = 2,35 vagues de 170 → la 3e vague à 35 % (modèle ⌈p⌉/p = 1,277) ; 320 (1,88) et 480 (2,82) = 1,063 :
**T = 5 est un creux de quantification**, T = 4 et 6 l'évitent ; au-delà, gpt ≤ 6 paie les programmes fins. C'est le modèle
max/moy de la 185 b, cette fois SANS changer BN (la 185 b perdait le gain dans le coût par octet de BN 32). Gain brut si T = 4
servi : 2,3 µs × 64 = **−0,147 ms/pas** (0,74 % des 19,90 ms de la 173) ; down 5120 × 17408 (8 appels, 400 prog.) : même creux
probable, non mesuré. **Hors bit** (l'ordre des sommes fp32 change avec T) → opt-in ± 1 ulp au mieux, ou réécriture de
`decouper_k` avec un test d'équivalence par tranche… impossible au bit : décision chef / poste1 (185 c, H = charge du SM le plus
chargé — confirmée ici pour o/out). Prédiction à vérifier par la 185 c : qkv 14336 (448 prog. = 2,64 vagues, modèle 1,14 → T = 3 :
672 = 3,95 vagues, 1,01) et GDN qkv‖gate 16384 (512 = 3,01 vagues, 1,33 → T = 3 : 1,11), si gpt reste ≥ 8.

## Suite
Rien à servir dans cette pièce. À chef : (i) 185 § k/v à requalifier (chaud partiel) ; (ii) T = 4 sur o/out : pièce 185 c
(poste1) ou opt-in ± 1 ulp ; (iii) le plancher ≈ 2,9 µs des petits appels reste (rampe/réduction), hors de ma pièce.
