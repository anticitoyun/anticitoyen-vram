# ggrun — lanceur de placement MoE pour llama.cpp (lu le 12/09/2026)

Source : <https://github.com/raketenkater/ggrun> (Go, MIT, ~570 commits, actif),
`docs/optimizer-theory.md`. Ce n'est **pas un moteur** : il lit la disposition
des tenseurs d'un GGUF, inventorie VRAM/RAM/PCIe, et génère la ligne
`llama-server` (`-ot`, `-ngl`, `--tensor-split`, `-c`, `-b/-ub`, `-np`,
`--cache-type-k/v`, `--kv-offload`, mmap). Chiffres annoncés (3090+3060+4070,
128 Go RAM, ctx 32k) : Qwen 27B 37,4 t/s, Qwen 122B 22,9 t/s, MiniMax-M3 5,59 t/s.

## Ce qu'il pose que nous ne posons pas

| dimension | ggrun | acvram (`loader._reajuster_plan`, manifeste) |
|---|---|---|
| plan | deux phases : **Fit** (contraintes explicites : ctx, KV, slots, batch) puis **Performance** (classe la résidence roomy/tight/non-résident, ≤ 1 challenger A/B) | plan figé à la conversion ; réajustement descend/remonte les MLP selon `capacité − marge` |
| coût | `T = T_backbone_GPU + T_expert_GPU + T_expert_CPU + T_transfer` ; `p_touch(B) = 1 − (1 − k/E)^B` pour B jetons | aucune fonction de coût ; exil = falaise (−70 % dès la première couche, revue *l-exil-d-une-couche-est-une-falaise*) |
| mesures | inventaire statique **mesuré** : bande VRAM, PCIe négocié, matrice P2P, bande DRAM, NUMA, cgroup | `acvram-topology.json` : capacités déclarées, pas mesurées |
| preuve | « preuve d'allocation » par candidat avec niveaux : statique / allouée / observée / rejetée | `verifier_place` (aveugle à la mémoire épinglée jusqu'au patch) |
| recherche | bornée : base + 3 replis + 1 challenger mesuré | aucune |

## Limites reconnues par l'auteur (utiles pour nous)

- Routage MoE supposé **uniforme indépendant** — nous savons (sac à dos,
  Pearson −0,992) que la PPL suit les octets, pas le routage ; la question
  ouverte reste le *coût* par expert, où ggrun facture l'expert hôte au débit
  DRAM, pas au flux PCIe complet par jeton.
- Coût de lien = plafond PCIe négocié, non mesuré.
- Cache d'experts : brouillon seulement (mise en page decode-only, correction
  asynchrone absentes) — même chantier que notre « cache d'experts promis ».
- `-ot` désactive le pipeline-parallel ; layer-split est un chemin **sériel**.

## Ce qui est transposable à acvram

1. **Fonction de coût explicite** avant de descendre une couche : un
   `T_transfer` chiffré par la bande mesurée aurait prédit la falaise.
2. **Preuve d'allocation** avec niveau (statique/allouée/observée) — remplace
   un `verifier_place` qui ne peut rendre « faux ».
3. **Inventaire mesuré** (VRAM, DRAM, PCIe) au premier lancement, stocké dans
   `acvram-topology.json`, plutôt que déclaré.
4. **Challenger unique A/B** : une alternative mesurée, pas une grille.

Non transposable : tout ce qui génère une ligne llama.cpp ; ggrun pilote un
moteur qu'il ne modifie pas, nous modifions le nôtre.

Prochain geste possible (à qui prendra le sujet) : lire
`docs/fitting-the-hardware.md` et `docs/compute-preflight-plan.md` pour la
méthode de mesure de la bande DRAM/PCIe, et la comparer à `outils/gpu/mesure`.
