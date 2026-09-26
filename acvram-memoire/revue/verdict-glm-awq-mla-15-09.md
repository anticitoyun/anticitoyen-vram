# Verdict — AWQ sur un MLA (GLM-4.7-Flash), et marge de plan trop fine (P6)

poste2, 15/09 soir. Suite du signalement `calibration indisponible
('model.layers.0.self_attn.q_proj.weight')` : corrigé par poste1
(`poste1-11` 96f8434 puis 731bf4e, fusionné main `63a50cd`) —
résolution par structure (`spec.est_mla`) au lieu d'un nom de tenseur
en dur, résilience par couche.

## Mesure gratuite demandée par chef : ce que vaut AWQ sur un MLA

Même modèle (GLM-4.7-Flash-srcbf16-nvfp4), même régime d'éval (window
2048/2048, min-context 256, wiki-gptq.txt), deux conversions distinctes,
chacune dans son propre processus (pas de double-chargement) :

| conversion | AWQ | PPL |
|---|---|---:|
| `-sansawq` | non (calibration indisponible avant le correctif) | 8,4415 |
| (nom principal, reconverti apres le correctif) | oui (8704 tenseurs calibrés) | 8,1275 |

**Ratio AWQ/sans-AWQ = 0,9628 (−3,72 %).** AWQ apporte un gain de PPL
réel et net sur ce MLA une fois la calibration réparée — cohérent avec
son rôle habituel (mise à l'échelle par canal avant quantification),
rien de spécifique à MLA ne l'invalidait : le bogue était dans le
COLLECTEUR de statistiques (nom de tenseur en dur), pas dans la méthode
AWQ elle-même.

## P6 (poste7/poste1) : la marge de plan de la conversion bf16 était trop fine

En tentant la PPL de contrôle vs bf16 pur (`GLM-4.7-Flash-srcbf16-bf16`,
56,8 Gio), deux OOM CUDA reproductibles (fenêtre 2048 puis 256, même
échec) :

    31,27 Gio en usage sur 31,36 Gio total, 4 Mio libres

Le plan de placement de cette conversion :

    etage   format   capacite   poids      KV      tranche
    cuda:0  bf16     30,1 Gio   27,2 Gio   1,8 Gio  couches 0-46

27,2 + 1,8 = 29,0 Gio planifiés sur une capacité annoncée de 30,1 Gio —
1,1 Gio de marge théorique. **L'overhead réel (contexte CUDA, allocateur
PyTorch, fragmentation) mange cette marge en entier** avant même que la
plus petite fenêtre d'évaluation (256 jetons) ne puisse allouer son
propre KV. Ce n'est pas un bogue de calcul (le plan est arithmétiquement
correct), c'est une marge de sécurité insuffisante pour un modèle dont
les poids seuls (27,2 Gio) frôlent la capacité annoncée de la carte
(30,1 Gio) — **27,2 Gio de poids bf16 pour un modèle dont le fichier
source pèse 58,2 Gio (donc ~29,1 G paramètres, comptés en 2 octets/param)
veut dire que le planificateur avait DÉJÀ exilé environ la moitié du
modèle vers l'hôte, sans le dire dans le journal** (le journal
n'affiche que ce qui reste sur `cuda:0`, jamais explicitement ce qui a
été exilé) — un plan qui exile la moitié sans le nommer complique le
diagnostic de sa propre marge.

Reconversion tentée avec `--host-exec stream` pour une marge réelle :
**plan identique au bit près** (27,2 Gio de poids sur cuda:0, couches
0-46) — le flag n'a eu AUCUN effet sur ce modèle. **Bogue tiering à
remonter à poste1, pas corrigé ce soir** : `--host-exec` ne semble agir
que sur des `layer_types` spécifiques (nemotron_h/mamba dans
`acvram/memory/tiering.py`), pas sur le placement générique d'un MoE
dense comme GLM — une seule ligne à vérifier dans `tiering.py`, pas
creusé plus loin (hors périmètre de ce soir, chef confirme).

Contournement retenu (chef) : `outils/glm-ppl-bf16-hf.py`, HF
`transformers` direct avec `device_map="auto"` (accelerate, installé ce
soir dans le venv vLLM — absent avant), même régime/encodage que
`acvram eval`. Contourne le planificateur acvram entièrement plutôt que
de le réparer.

## Suite

PPL absolue vs bf16 (juge final ≤×1,01) : lancée via
`outils/glm-ppl-bf16-hf.py`. Rendu séparément.
