# Choisir entre NVFP4 pur et attention+GDN en int8 par canal (Qwen3.8-27B et hybrides GDN)

Sur un modèle hybride Gated DeltaNet (attention pleine + couches GDN, comme Qwen3.8-27B),
`acvram quant convert` sait promouvoir les projections d'attention **et** les cinq projections
du GatedDeltaNet (qkv/gate/alpha/beta/out) en int8 symétrique par canal, au lieu du NVFP4
par défaut. Ce document compare les deux formats et donne la commande qui produit le second.

## La commande de conversion

```
acvram convert SOURCE -o SORTIE \
  --promotion-classes q_proj,k_proj,v_proj,o_proj,linear_attn.qkv,linear_attn.gate,linear_attn.alpha,linear_attn.beta,linear_attn.out \
  --max-promotions 1.0 --snr-floor 99 --attn-qkvo-int8-canal --gdn-int8-canal --no-awq
```

Trois précisions qui coûtent cher si on les oublie (pièce 153, deux échecs avant cette forme) :

* `--attn-qkvo-int8-canal` et `--gdn-int8-canal` ne *promeuvent* rien tout seuls — ils décident
  seulement du régime (par canal, pas par groupe de 128) d'une projection **déjà** promue int8.
  Sans `--snr-floor` élevé et `--promotion-classes` nommant explicitement ces projections, le
  convertisseur les laisse en NVFP4 et les deux drapeaux tournent à vide (avertissement
  « SANS EFFET » au manifeste — `acvram/quant/convert.py:446`, contrôlé par
  `tests/test_gdn_int8_canal_153.py`).
* `--snr-floor 99` force la promotion de toute projection nommée dans `--promotion-classes`
  (un plancher qu'aucun SNR mesuré ne dépasse) ; `--max-promotions 1.0` lève le plafond de 15 %
  de tenseurs promus, sinon la conversion sature avant d'avoir couvert les 48 couches GDN.
* `--no-awq` : simple arrondi, pas de recherche d'échelle AWQ — la comparaison ci-dessous porte
  sur le format de stockage (canal vs groupe), pas sur un calibrage plus fin.

Vérifié à sec (`--dry-run`, 26/09/2026) via la CLI réelle (`acvram.cli.main`, pas
`convert_checkpoint` en direct) sur un mini-checkpoint hybride GDN+attention de la même famille
(gabarit de `tests/test_gdn_int8_canal_conversion_reelle_167.py`, layer_types
linear_attention+full_attention) — le Qwen3.8-27B-bf16 réel (55 Gio) a été essayé mais a chargé
la machine partagée pendant ~45 min sous calibration CPU même confinée (`nice`/`ionice`/`taskset`),
arrêté sur ordre de chef ; le mini-checkpoint prouve le même mécanisme (les 5 projections GDN
promues int8, RC=0, aucun fragment `.safetensors` écrit) en 0,1 s sans toucher la carte. Script :
`scratchpad/poste4-p224-26-09/dry-run-tiny.py`. `tests/test_cli_convert_i8c_224.py` fait casser
la CI si un des drapeaux cités disparaît ou change de type dans `acvram/cli.py`.

## Le tableau

| | NVFP4 pur (102) | attn+GDN int8-canal (153) |
|---|---|---|
| Taille disque | 16,09 Gio | 19,94 Gio (**+24 %**, pièce 153) |
| KL(HF bf16 ‖ moteur), dernière position, L=2048 | 0,0513 (pièce 174, défaut) | 0,0155 (pièce 153d) — **≈3,3× plus fidèle** |
| argmax = HF (mêmes fenêtres) | oui | oui (pièce 153d, 3/3 fenêtres) |
| Débit b=1, graphes=on | référence (80,9 t/s) | **−14,7 %** (70,5 t/s, pièce 153d) |
| Débit b=8, graphes=on | référence (591,5 t/s) | **−13,1 %** (522,9 t/s, pièce 153d) |
| Capacité KV à ctx 32 768 (B=8) | 119 440 jetons (−7,4 % vs avant-201) | 9/64 couches exilées pour tenir, chargé (pièce 201) |
| Coût VRAM `warm_graphs` (B=8, CTX=2048) | 1 880 Mio (dépasse la marge 1 536 Mio d'alors) | 1 720 Mio — les deux couverts depuis par la marge GDN à 3 072 Mio (pièce 212) |
| PPL, fenêtre 2048/2048, wiki-gptq | non mesurée dans cette série | 7,2157 (pièce 153b) |

Toutes les mesures 153/153d/174 portent sur le même modèle source
(`Qwen3.8-27B-bf16`) et le même corpus/fenêtres ; 201 et 212 sont mesurées à B=8/CTX=2048 sur
le parc, pas spécifiquement recalculées pour ce tableau.

## Quand choisir lequel

* **NVFP4 pur** : la vitesse prime, la marge VRAM est serrée (moins de couches à exiler, capacité
  KV pleine), et une fidélité déjà correcte (KL ≈0,05, argmax intact) suffit à l'usage visé.
* **attn+GDN int8-canal** : la fidélité prime sur ~14 % de débit — un chatbot ou une tâche
  sensible aux petites divergences de logits, sur une carte avec assez de VRAM pour absorber
  +24 % de disque et l'exil de couches à grand contexte (pièce 201). Coûte aussi le disque et la
  bande passante mémoire du modèle plus lourd (le KV lui-même reste en int8 des deux côtés,
  inchangé par ce choix).

Aucune des deux mesures de fidélité n'est un défaut qualité au sens des scellés antérieurs (KL très
en-dessous de la référence Coder ≈0,52-0,74) : le choix est un compromis vitesse/fidélité/VRAM, pas
un chemin cassé à éviter.
