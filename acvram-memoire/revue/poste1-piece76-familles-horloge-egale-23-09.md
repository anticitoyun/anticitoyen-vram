# Pièce 76 — vLLM retracé en service à -lgc 2700 ; familles de noyaux à horloge égale contre notre trace — 23/09 (poste1)

* **instrument** : chaîne `scratchpad/poste1-p76-23-09/chaine.sh` (celle de la p59, même client `decode-vllm-17-09.py`, b = 12, fenêtre 5 s, + -lgc 2700 et échantillonneur d'horloge 200 ms) ; `outils/gpu/mesure/familles-comparees.py` (taxonomie commune, **durées de noyaux seules**, pas = 48 marqueurs par couche, fenêtres établies ≤ 1,3 × mur médian) contre notre trace p73 (`scratchpad/poste1-p73-23-09/graphe_cuda_gpu_trace.csv`)
* **commit** : 730b7075 (arbre de la prise) ; notre trace : 2d4a6445 (p73) ; vLLM 0.29.0, `/opt/ia/vLLM/.venv`
* **régime** : **vLLM 2 639 MHz** (moyenne NVML de la fenêtre du client ; médiane de l'échantillonneur sous charge 2 692) ; **acvram 2 692 MHz** (médiane p73) ; compute-apps début = fin = llama-server 4627 ; 851/903 fenêtres vLLM gardées, 93/108 acvram
* **scellé** : aucune prédiction propre à la 76 écrite avant la prise (**faute**, REGLES § 3). Seules les deux issues de la « suite proposée » de la 75 l'étaient : Marlin MoE vLLM remonté vers 58-62 µs/couche → surtout l'horloge ; resté à ~53 → contexte de service
* **mesuré** (ms/pas, noyaux seuls) :

| famille | acvram (2 692 MHz) | vLLM (2 639 MHz) | écart | lancements A / V |
|---|---|---|---|---|
| moe_gemm (Marlin MoE) | 3,045 | 2,729 | **+0,316** | 142 / 96 |
| proj_dense (q/k/v/o) | 1,025 (int8 étroites) | 0,787 (Marlin fp4) | **+0,238** | 98 / 96 |
| normes + rope/kv + autres ¹ | 0,446 | 0,416 | +0,030 | 193 / 198 |
| routeur_gemm | 0,163 | 0,170 | −0,007 | 92 / 96 |
| moe_glue | 0,349 | 0,418 | −0,069 | 188 / 240 |
| échantillonnage | 0,000 ² | 0,049 | −0,049 | 0 / 3 |
| attention | 0,435 | 0,538 | −0,103 | 48 / 98 |
| tête | 0,201 (int8) | 0,378 (bf16) | −0,177 | 1 / 1 |
| **noyaux** | **5,709** | **5,526** | **+0,183** | |

  ¹ vLLM fond q/k-norme + rope dans `triton_red_fused_2` (47/pas, « autres ») : rope_kv seul (+0,154) est un artefact de taxonomie ; les trois familles réunies sont à égalité. ² notre échantillonnage est capturé dans le graphe sous d'autres noms (p39 : 49,5 µs), rangés ailleurs.
* **verdict** :
  1. **Horloge** : le Marlin MoE de vLLM passe de 53,1 µs/couche (p59, 2 905 MHz) à **56,9** (2 639 MHz). C'est entre les deux issues écrites : l'horloge explique **3,8 µs** des 9,3 µs d'écart entre leur service et le banc froid (62,4 à la 75) ; il reste ~5,5 µs (9 %) de contexte de service chez eux.
  2. **À horloge égale, nos noyaux ne coûtent que +0,183 ms/pas de plus (+3,3 %)**. C'est le tiers de l'écart de la 64 : 6,551 contre 6,003 ms/pas servis, **0,548 ms/pas**.
  3. **Les deux tiers restants (~0,36 ms/pas) sont hors noyaux**. Cette part est obtenue par soustraction, **pas** par les parts hôte de nsys : mur servi sans nsys (p64) − noyaux de la trace = **0,842 ms/pas chez nous contre 0,477 chez vLLM**.
  4. Où sont nos +0,183 : **MoE +0,316** (dont 0,22 = nos 3 lancements contre leur w13 fusionné, mesuré au banc de la 75 ; ~0,1 de contexte) et **projections +0,238** (int8 étroites contre leur Marlin fp4). Ils sont compensés par la tête (−0,177), l'attention (−0,103), la glue MoE (−0,069) et l'échantillonnage (−0,049).
* **durée** : prévue ≤ 5 min ; tenue 3 s (échec : `$PWD` évalué après `cd`) + ≈ 1 min 10 (prise), fin 08:56:46

## Ce que cela ordonne

* **Premier levier, et le plus gros : les 0,36 ms/pas hors noyaux du service** (0,842 contre 0,477). Ni un noyau ni
  la 48 ne les touchent. Ils se cherchent entre deux rejeux de graphe : ordonnanceur, préparation du pas,
  retour HTTP. Méthode : chronos hôte du moteur (pas nsys), mur de pas en service contre en `Engine` direct (p73 :
  trou 0,38 déjà en direct).
* **Second : les projections** (+0,238). L'alias à projections nvfp4 de la pièce 42 (prédit −0,38 à −0,46 ms/pas,
  qualité tenue 5/5) est la voie écrite ; son débit reste à mesurer.
* **Troisième : w13 fusionné** (0,22 au banc), sous le seuil de 0,3 ms/pas pris seul.
* La 48 (ncu) n'est plus sur le chemin critique de cet écart.
