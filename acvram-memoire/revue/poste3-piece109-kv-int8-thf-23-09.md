# Pièce 109 — KV int8 servi diverge du jumeau (anticitoyen-vram-thf) — EN COURS, pause du groupe

Ordre (chef) : `acvram_kernels.cu:4217/4220/4223` calcule `m/127.f` (division approchée sous
`--use_fast_math`) puis `x·(1/sc)`, alors que le jumeau `kvcache.py` fait `x/scale` (IEEE).
Correctif : `__fdiv_rn` et division directe, comme la pièce 104. Puis : test au bit noyau =
jumeau qui casse sur l'ancien code, puis KL b=1 ≤ 0,74 et lot mêlé sur carte, en fin de file,
seuils scellés avant. **Pause du groupe demandée avant la fin de la validation : livré dans
l'état, testé partiellement, la suite reste ouverte.**

## Fait

- `acvram/kernels/acvram_kernels.cu` : deux correctifs, PAS un seul —
  1. `kv_write_int8_kernel` (`:4190`, chemin par jeton servi par défaut pour K **et** V) :
     `red[0] = fmaxf(__fdiv_rn(m, 127.f), 1e-8f)` puis `__float2int_rn(__fdiv_rn(x, sc))`
     (plus de `inv = 1.f/sc` multiplié).
  2. `kvc_quant_par_jeton` (`:4415`, trouvée en lisant : COPIE du même calcul pour V du chemin
     canal C5-b — portait le même défaut, non nommé par chef mais du même ressort). Sa
     voisine `kvc_quant_bloc_canal` (K du chemin canal) était déjà juste (`__fdiv_rn`, commentée
     comme telle) : l'incohérence entre les deux fonctions voisines est le même genre de faute
     que le ticket signale.
- `tests/test_kv_write_int8_thf_carte.py` : deux tests, sur carte.
  - `test_ecriture_int8_par_jeton_au_bit_contre_le_jumeau` (un tirage, graine 1) : **VERT**,
    K et V au bit contre `PagedKVCache._quantize`, échelles au bit.
  - `test_plusieurs_tirages_toujours_au_bit` (5 graines, 100-104) : **ROUGE, un seul cas** —
    graine 103, V seulement (K identique sur les 5) : `assert False` sur `torch.equal`.

## Pas fait — le désaccord n'est PAS expliqué, à rouvrir

Deux mesures contradictoires sur EXACTEMENT le même tirage (graine 103, mêmes k/v vérifiés
identiques par construction — `torch.manual_seed` déterministe) :
- Dans la boucle des 5 graines (4 `PagedKVCache` créés avant lui) : **V diverge**.
- Isolé, dans un script neuf, seed 103 posée directement (`scratchpad/poste3-thf-23-09/diag.py`,
  rejoué deux fois, y compris une seconde fois DANS le même run que le test rouge, sans
  recompilation) : **0 écart, échelles égales**.

Donc le désaccord n'est pas dans la formule (le correctif tient à l'isolement) : quelque
chose lié à L'ÉTAT ACCUMULÉ par les créations/destructions précédentes de `PagedKVCache`
dans la boucle change le résultat du MÊME calcul sur les MÊMES données. Hypothèses nommées,
aucune vérifiée :
1. **Réutilisation mémoire de l'allocateur CUDA** (cache PyTorch) : les 3 instances
   précédentes libérées avant la 4e pourraient laisser une adresse partagée dans un état
   qui influence un noyau qui, lui, ÉCRIT tout — improbable mais pas exclu (pointeurs
   non alignés différemment ?).
2. **Un noyau voisin de l'extension** (compilation JIT, lancement) laisse un état de flux
   CUDA ou un `cudaStreamSynchronize` implicite différent selon l'historique du process.
3. Script de diagnostic supplémentaire écrit mais NON JOUÉ (pause du groupe) :
   `scratchpad/poste3-thf-23-09/diag2.py` — rejoue la boucle et capture le détail de l'écart
   dans SON PROPRE contexte (pas un script à part), rejoue l'écriture deux fois de suite sur
   les mêmes données (teste la reproductibilité intra-process), compare k/v de la boucle à
   ceux d'un tirage isolé (déjà confirmés identiques par construction, à re-vérifier par le
   code plutôt que par le raisonnement).
4. Pas exclu : un défaut PRÉEXISTANT (avant ce correctif) déjà présent dans le chemin V —
   le test C5-b (`test_kv_canal_c5b_carte.py:8`) tolérait déjà « V … à ± 1 code » comme un
   fait accepté, jamais expliqué non plus.

**Le correctif de code (division `__fdiv_rn`) reste posé** : il est correct par lecture (même
arithmétique que le jumeau et que le C5-b déjà juste) et TENU sur le test à un seul tirage ;
mais le test à cinq tirages n'est PAS vert, donc **la sortie par défaut ne doit PAS changer**
tant que ce désaccord n'est pas expliqué — aucune KL ni lot mêlé n'a été mesuré (jamais
atteint le tour de file).

## Reste, dans l'ordre, à la reprise

1. Jouer `diag2.py` (déjà écrit, jamais lancé) pour voir le détail de l'écart DANS son
   contexte reproductible.
2. Si confirmé lié à l'allocateur/état du process : isoler avec
   `torch.cuda.empty_cache()` entre les itérations de la boucle, ou par un test à processus
   séparé par graine (`subprocess`), pour trancher entre allocateur et autre chose.
3. Une fois expliqué et le test à 5 tirages vert : KL b=1 ≤ 0,74 et lot mêlé (REGLES § 1),
   seuils à écrire avant la mesure, en fin de file carte.
4. Tant que 1-3 ne sont pas faits : le correctif reste un candidat lu et partiellement
   prouvé, PAS un changement de défaut servi.
