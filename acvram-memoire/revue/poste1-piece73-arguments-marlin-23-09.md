# Pièce 73 — nos arguments Marlin contre ceux de vLLM 0.29 : **aucun écart**, la piste des réglages est réfutée à sec — 23/09 (poste1)

## Comparaison argument par argument (à sec, 0 min de carte)

Sources lues : notre `acvram/kernels/marlin_port/__init__.py:415-437` et
`bindings.cpp:12-25` ; vLLM installé,
`site-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:135-162`
et `:325-339`, `quantization/utils/marlin_utils.py:408-420`.

| argument | nous | vLLM 0.29 | écart |
|---|---|---|---|
| `moe_block_size` | `choisir_block_size` : `for b in [8,16,32,48,64]: if M*top_k/E/b < 0.9: break` | **la même boucle, au mot près** (`marlin_moe.py:333-335`) | **aucun** |
| `use_atomic_add` | `False` | `False` (`:159`) | aucun |
| `use_fp32_reduce` | `True` | `True` (`:160`) | aucun |
| `is_zp_float` | `False` | `False` (`:161`) | aucun |
| `is_k_full` | `True` | `is_k_full` (vrai hors act-order) | aucun |
| `thread_k` / `thread_n` / `blocks_per_sm` | `−1, −1, −1` | **non passés** → les mêmes défauts du binding | aucun |
| `workspace` | `espace_travail(blocs_par_sm=4)` = sms × 4 | `marlin_make_workspace_new(device, 4)` = sms × 4 | aucun |
| `b_q_type` | `kFE2M1f` (nvfp4) | idem | aucun |
| **disposition** | gate, up, down : **3 GEMM** | gate‖up interfoliés (`size_n = w13_num_shards * N`), down : **2 GEMM** | **le seul** |

**Et la prémisse tient** : leur modèle est
`Qwen3-Coder-30B-A3B-Instruct-FP4-a16`, `config.json` →
`{"num_bits": 4, "type": "float", "group_size": 16, "input_activations": null}`
— c'est **notre format exact** (nvfp4, échelles tous les 16, activations bf16),
servi par **le noyau dont notre port est la copie**. La comparaison est donc
légitime, et « leurs réglages » n'existent pas comme levier : **la pièce 73
telle que formulée est réfutée à sec, sans une seconde de carte.**

## Ce que la trace de vLLM dit, et le désaccord qu'il faut trancher

`scratchpad/poste5-p59-23-09/familles-vllm.txt` : `marlin_moe_wna16::Marlin`
**96,0 lancements/pas à 26,6 µs** = 2,551 ms/pas = **53,1 µs/couche**, soit
exactement 2 lancements par couche. Nous : 3 lancements à 22,0 = 65,5 µs/couche.

Si nos trois lancements coûtent à peu près le même prix, alors notre `down`
(22,0) est **plus rapide** que le leur et tout l'écart est sur gate+up : 44 chez
nous contre 26,6 pour le même travail, soit **−17 µs/couche = −0,84 ms/pas** à
gagner par la fusion. Or le banc w13 de poste5 (pièce 71 bis) ne rend que
**−4,0 µs/couche**. Les deux ne peuvent pas être vrais. **Ce désaccord est la
vraie question de la pièce**, et il se tranche par une décomposition de NOS
lancements par forme — ce que ni la p63 ni la p71 bis ne donnent.

## Prédiction et issues, écrites AVANT la prise

Instrument : `frontiere-pas.py 12 60` sous nsys puis
`familles-noyaux.py --detail experts_marlin` sur l'alias servi — le détail est
par forme de grille, donc il sépare gate, up et down. ≤ 5 min, aucun code touché.

| issue | condition | ce qu'elle rend |
|---|---|---|
| **I1 — la fusion vaut le détour** | gate et up ≈ 20-23 µs chacun (somme ≥ 40) et down ≤ 26 | l'écart est bien le 3ᵉ lancement ; le banc de la 71 bis sous-estime d'un facteur ≈ 4 et il faut savoir pourquoi avant d'engager la demi-journée |
| **I2 — la fusion ne vaut rien** | gate + up ≤ 30 µs (le gros est dans `down`) | 71 bis confirmée, l'écart est ailleurs, et la piste w13 se ferme |
| **I3 — nos lancements ne sont pas 3** | le compte par pas n'est pas 144 | la prémisse « 3 contre 2 » est fausse et tout le raisonnement tombe |

* **prédiction nominale : I1**, gate ≈ up ≈ 21 µs, down ≈ 23, total ≈ 65.
* **seuil de chef** : une piste ne vaut d'être poursuivie qu'à **≥ 0,3 ms/pas**
  équivalent. I1 donnerait ≈ 0,8, I2 donnerait ≈ 0,19 — sous le seuil, donc
  réfutée.
* **ce qui me gênerait** : I2 confirme poste5 et invalide ma lecture de la
  trace vLLM, après que j'ai écrit qu'elle valait 17 µs. Je le nomme d'avance.
* **alarme** : si le total mesuré s'écarte de plus de 15 % des 65,5 µs/couche
  de la p63, je ne mesure pas le même régime (godet, `MOE_TENSOR_MIN_T`,
  horloge) et je le dis avant toute conclusion.

## Mesure — 23/09 07 h 10, une prise de 249 s (tenue=249s, horloge médiane 2 692 MHz)

* instrument : `frontiere-pas.py 12 60` sous nsys, `familles-noyaux --detail
  experts_marlin`, alias `Qwen3-Coder-30B-A3B-nvfp4`, commit 2d4a6445 ;
  `experts_layout=marlin`, `chemin_moe=…+tensor(b≥8)`, 105 pas jugés ;
  compute-apps début = fin ; journaux `scratchpad/poste1-p73-23-09/`.
* **alarme non déclenchée** : 64,6 µs/couche contre les 65,5 de la p63 (−1,4 %),
  mur 6,106 ms contre 6,198 — même régime.

| | lancements/pas | µs par lancement | ms/pas |
|---|---|---|---|
| nous, `marlin_moe_wna16::Marlin` **[510x1]** | **144** (3/couche) | **21,5** | **3,106** |
| vLLM, même noyau (p59) | 96 (2/couche) | 26,6 | 2,551 |

* **I3 écartée** : 144 lancements, donc bien 3 par couche.
* **I1 et I2 ne se départagent pas comme je l'avais prévu, et c'est instructif** :
  les trois lancements sont **indiscernables** — une seule forme de grille,
  `[510x1]` = 170 SM × 3, le noyau étant persistant, sa grille ne dépend pas de
  la forme. gate, up et down coûtent donc **21,5 µs chacun**. Ma prédiction
  « gate ≈ up ≈ 21, down ≈ 23, la fusion vaut 0,8 ms/pas » est **réfutée** sur
  sa conclusion : voir ci-dessous.

## Ce que l'écart est vraiment, et ce qu'il n'est pas

**Il n'est pas dans les arguments** (comparés un à un, aucun écart) et **il
n'est pas dans le nombre de lancements seul**. À octets égaux :

* poids lus par couche, des deux côtés : 33,4 experts distincts × (768×2048 +
  768×2048 + 2048×768) × 0,5625 o/poids ≈ **88,6 Mo** ;
* **nous : 88,6 Mo / 64,6 µs = 1,37 To/s · vLLM : 88,6 / 53,1 = 1,67 To/s.**

Leur GEMM de `2N` atteint un meilleur débit que nos deux GEMM de `N` sur les
mêmes octets — et **1,67 To/s est au-dessus du plancher de bande que nous avons
mesuré le 22/09** (1,52-1,55 To/s, pièce 41).

Extrapolation du banc de poste5 (71 bis : 69,1 → 65,1, soit −5,8 %) au service :
64,6 → ≈ 60,9 µs/couche, **gain ≈ 0,18 ms/pas — sous le seuil de 0,3**, et il
resterait **≈ 0,38 ms/pas inexpliqués** contre vLLM. Son banc n'est pas en
cause : je l'ai relu (`banc-moe-w13.py:17-18`), il concatène bien les poids et
lance **une** GEMM de `2I`. **La fusion w13 est donc confirmée réfutée**, et
l'écart est ailleurs : dans ce que le noyau fait des mêmes octets.

## L'alarme que je dois nommer, et qui pourrait tout expliquer

Les 88,6 Mo supposent que **leur routage touche autant d'experts distincts que
le nôtre** (33,4 par couche, mesuré chez nous seulement, `poste5-p60`). Ce
nombre n'a **jamais été mesuré côté vLLM**. S'il était de 28 au lieu de 33,4,
ils liraient 16 % d'octets en moins et leur « meilleur débit » serait une
illusion d'instrument — l'écart s'expliquerait sans rien changer au noyau.
**Contrôle qui tranche, 0 min de carte** : compter les experts distincts par
couche dans une cellule vLLM (même invites, même graine) et comparer à 33,4 ;
faux si l'écart dépasse 5 %.

## Verdict

**Piste des réglages : RÉFUTÉE à sec.** Piste du 3ᵉ lancement : **réfutée par la
mesure** (0,18 ms/pas, sous le seuil). Ce qui reste — 0,555 ms/pas, l'essentiel
des 9,1 % — est **un écart de débit du même noyau sur les mêmes octets**, donc
exactement ce que la **pièce 48** devait lire (taux d'émission, attentes
mémoire, occupation) et qui est bloqué par `ERR_NVGPUCTRPERM`. Je ne le
remplacerai pas par une hypothèse : la 73 ferme ses deux pistes et rend la
question à la 48, après le contrôle des experts distincts ci-dessus.
