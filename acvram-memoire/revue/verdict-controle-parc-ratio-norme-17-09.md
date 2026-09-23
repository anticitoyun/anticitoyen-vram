# Verdict — Contrôle ratio de norme sur Coder et GLM classés : sains, garde permanente ajoutée au convertisseur

Manon, 17/09. Suite de `verdict-nemotron35-calibration-awq-17-09.md` :
Nemotron calibA (1,4301 vs 1,0304 sans calibration) venait d'une échelle
AWQ tirée d'une statistique bruitée sur des experts peu routés (act_scale
étalée sur 1,68e6×). Sage : « le bras A garantit » n'est pas un contrôle —
Coder classé n'a pas été converti avec bras A (corpus par défaut,
`collect.py:48-77`, ~14 jetons distincts/expert sur 128, dizaines d'experts
sous 8). Deux gestes : contrôler Coder+GLM sans reconvertir, et ajouter une
garde structurelle au convertisseur.

## 1. Garde permanente (commit `2a68aa6`)

`quantize_with_calibration` (`acvram/quant/calibrate.py`) calcule
`ratio_norme = ‖poids reconstruit‖ / ‖poids source‖` pour chaque tenseur
quantifié — distinct de `w_rel_err` (une erreur élément par élément reste
petite même quand la norme globale s'effondre sur quelques canaux
seulement, motif exact de Nemotron : 27-29 dB de `w_snr_db` propre mais
ratio 0,29-0,69 après rééchelonnage). `convert_checkpoint`
(`acvram/quant/convert.py`) **refuse la conversion** (`ValueError`) si un
seul tenseur sort de `[0,80 ; 1,25]`, et écrit `ratio_norme_min`/`max` au
manifeste. Protège l'EFFET quelle que soit la CAUSE — le seuil
`MIN_ECHANTILLONS_AWQ=8` (commit précédent) ne protégeait que la cause
supposée (statistique bruitée par manque de jetons).

Trois tests, témoins cassants vérifiés (`tests/test_awq_seuil_
echantillons.py`) : seuil d'échantillons, motif varié au-dessus du seuil,
et la garde de norme déclenchée par un motif dégénéré INDÉPENDAMMENT du
nombre d'échantillons (100 échantillons, un canal à 5000× la baseline).

## 2. Contrôle à sec : Coder et GLM sains

Ratio de norme mesuré tenseur par tenseur (dequantification NVFP4 +
rééchelonnage act_scale) contre la source bf16, sur tous les tenseurs
d'experts (`up_proj`/`down_proj`/`gate_proj`).

| modèle | tenseurs vérifiés | moyenne | min | max | malades [0,80;1,25] |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B-nvfp4 (corpus par défaut) | 18432 | 0,9982 | 0,9970 | 1,0787 | **0** |
| GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA (bras A) | 9024 | 0,9998 | 0,9716 | 1,0086 | **0** |

Prédiction Sage : Coder ≥1 malade (faux : 0) — **RÉFUTÉE**. GLM 0 malade
(faux : ≥1) — **TENUE**. Aucune reconversion nécessaire pour l'un ou
l'autre ; les deux restent classés tels qu'ils sont sur disque.

Lecture retenue (Sage) : un compte d'échantillons bas seul ne fabrique pas
un motif d'échelle extrême. Il faut en plus un motif d'ACTIVATION
spécifique — un canal quasi jamais vu côte à côte d'un canal heurté une
fois par une valeur extrême, comme mesuré sur Nemotron (`act_scale` de
0,0001 à 177 sur un même tenseur). Le corpus par défaut de Coder, malgré
des experts peu échantillonnés, n'a pas produit ce motif bimodal — les
statistiques y restent apparemment plus régulières.

Tenseur Coder le plus éloigné de 1 (hors zone saine 0,95-1,05, PAS malade,
aucune action requise, nommé sur demande de Sage) :

```
model.layers.1.mlp.experts.87.down_proj.weight   1,0787
model.layers.1.mlp.experts.27.down_proj.weight   1,0715
model.layers.1.mlp.experts.1.down_proj.weight    1,0554
model.layers.1.mlp.experts.65.down_proj.weight   1,0229
model.layers.1.mlp.experts.100.down_proj.weight  1,0049
```

Les cinq sortent tous de la couche 1 (`down_proj`) — au-dessus de 1,0
(reconstruction légèrement AMPLIFIÉE, pas atténuée comme sur les malades
de Nemotron), dans la bande saine élargie mais hors de la bande étroite
0,95-1,05.

## sha256 (pour mémoire, convertis inchangés par ce contrôle)

```
Qwen3-Coder-30B-A3B-nvfp4/
  acvram-00000.safetensors : 85d3faccd9bf6a653dc372b0fe4a4077f1518f960126099d1e58ed9acde78113
  acvram-00001.safetensors : 545472b28ea70ea794a9e1697ddb3a3534270716494834b1542025bd30651937
  acvram-00002.safetensors : f3af6ed47c5789f0b6b64020f2c45f32877f164505c5e1ec33081cf3f4856c0a
  acvram-00003.safetensors : cad10d8acb22f2d176504a875f414b4f0941bbd8bc82e885bcf94e2f768d5ebd
  acvram-00004.safetensors : 1ebf7fe84f32125ea901a5a7fe9dc05dbb7f776e4f1600e18cc154d7743e0973
  acvram_manifest.json     : e2183573a9ad5e59cdcc379c7d38072246185b06ee09420dfc49b4387b56ee28

GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA/
  acvram-00000.safetensors : 805ab513e904d3c61de9718c79a7d24cb82cc6a55f39a34d7cb26f96ea8f2c65
  acvram-00001.safetensors : 41bcd25effcfa61fc1a2a6655cfabef24d6363a75c1fe40c720c6d1596df14c3
  acvram-00002.safetensors : 8f84d465b55431507b043f3cd822a1fea84730f0f0bc5c375e5deebb502689ac
  acvram-00003.safetensors : 4149d82c887521060b89ed8088725b0361f6a4221c64cc555f4e2778e5f6c27a
  acvram-00004.safetensors : b40292e72d81b1bdd3829389355a4e2798a329776523edfe8b1a13e318c93f92
  acvram_manifest.json     : 387507838ca52b3a00ac7d969dcbe106d40123b5c52cb7b58bd875dcf3f91f60
```

Confirmé (Sage) : le manifeste GLM ci-dessus ne porte ni device, ni sha256
de corpus, ni commit du convertisseur — antérieur au commit `5bce2cc`
(REGLES §4). Pas une régression : ce champ n'existe que pour les
conversions faites APRÈS ce commit. Le converti `-verif17-09` (en cours,
§3) le portera.

## 3. Reconversion GLM `-verif17-09` : terminée, expert partagé NON affecté (promotion int8)

Durée totale 22625,1 s (6h17), contre 81,7 min pour l'original seul sur
CPU — contention confirmée (Nemotron en file + suite pytest complète en
parallèle jusqu'à leur propre fin, ~3h de recouvrement). 1er fragment à
19h03:03 (délai 166,9 min), le reste a suivi sans second blocage une fois
la suite pytest terminée. Conversion réussie, garde ratio_norme jamais
déclenchée (aucun refus).

**Diff octet à octet, résultat surprenant** : les 141 tenseurs
`mlp.shared_expert.*` sont en format **int8** dans les DEUX convertis
(promotion `mixed_precision=auto`, SNR NVFP4 brut sous le plancher 25 dB),
`has_act_scale=False` et `out_snr_db` **identiques à 2 décimales près**
dans les deux versions (ex. couche 1 down_proj : 44,36 dB dans les deux).
**L'AWQ est délibérément désactivé sur les tenseurs int8** (retiré depuis
`sage-organisation-16-09.md`/Océane : compense une erreur ~16× plus petite
pour le même coût) — le correctif de clé de l'expert partagé
(`0e4ef38`) ne pouvait donc rien changer ici : la statistique qu'il
corrige n'est simplement jamais consultée pour un tenseur promu en int8.

**Conclusion révisée** : le code était bien vulnérable (branche MLA
inconditionnelle, comme rapporté), mais sur CE converti GLM précis,
l'expert partagé est protégé par la promotion int8, pas par le
correctif — la casse théorique n'a eu AUCUN effet pratique sur les
chiffres déjà publiés (privé 1,0143, public 1,0281). Témoin de contrôle
(un expert routé, format nvfp4, censé être insensible au correctif
expert-partagé) : écart de reconstruction de 10,7 % entre les deux
convertis malgré un `out_snr_db` identique à 2 décimales -- la recherche
AWQ (grille à 20 points) n'est PAS bit-exacte d'une exécution à l'autre
(réductions flottantes multi-thread non associatives), même si la
QUALITÉ retenue l'est. Le diff octet à octet demandé n'est donc
concluant que par les MÉTADONNÉES (format, has_act_scale, SNR), pas par
l'égalité des octets bruts -- à signaler pour toute future demande de
diff « byte-exact » sur cette pile.

## Suite

- Item 2 (Sage) : scan du parc en fond (CPU) — script prêt
  (`scan_parc_ratio_norme.py`, un rapport JSON par modèle sous
  `scratchpad/rapports-ratio-norme/`), retardé pour ne pas ajouter de
  contention CPU pendant que GLM tourne seul dessus.
- Nemotron calibA (avec le seuil + la garde) : en file GPU, après la GEMV
  experts de Laure, fenêtre carte.sh 60 min (le timeout 30 min était la
  cause de l'échec exit 3 précédent).
