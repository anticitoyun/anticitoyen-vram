# Sage — Clé `mlp.shared` ≠ `mlp.shared_expert` : portée rétroactive — les cellules mesurées restent vraies, une conclusion est affaiblie, et un contrôle à sec de 15 min dit quels convertis sont touchés (17/09)

Entrée : Manon 72ace16 (8e393ef, 0e4ef38) — la calibration AWQ attachait les statistiques de l'expert partagé sous l'attribut Python `mlp.shared.*` (`model.py:679`), `calibrate.py` les cherchait sous `mlp.shared_expert.*` : **l'expert partagé n'a jamais été calibré, il était arrondi au plus proche (RTN), silencieusement**. AWQ est le défaut de `acvram convert` (`cli.py:522`), donc chaque conversion d'un modèle à expert partagé passée par ce chemin porte le défaut.

## 1. Ce que ça ne change pas

**Aucune cellule publiée n'est fausse.** Une PPL est un fait sur un fichier (sha256) mesuré par un instrument : GLM `-k48-calibA` 1,0150 géo *est* la PPL de ce converti. On ne retire rien des menus ni du comparatif. Coder (Qwen3-MoE, `n_shared_experts` absent) n'a pas d'expert partagé : intact. Qwen3.8 : dense, intact. Nemotron : converti après le correctif.

## 2. Ce que ça affaiblit

* `sage-calibration-verdict-17-09` : « sur GLM le privé ne bouge pas ± 0,004 quelle que soit la calibration » — mesuré avec l'expert partagé en RTN **dans tous les bras** (A, B, RTN, hadamard). L'expert partagé de GLM (1 × 1 536, traité par chaque jeton, ≈ 1/9 du calcul MLP) n'a jamais eu sa chance ; la conclusion « la calibration n'est pas le levier » n'est démontrée que pour les experts routés.
* Tout « AWQ contre `--no-awq` » sur un modèle à expert partagé comparait deux bras identiques sur ce tenseur.
* MLA n'exclut pas : `MoEBlock` est le même pour GLM (`model.py:658`), l'attention n'y est pour rien — sauf si le chemin de collecte GLM attachait déjà la bonne clé, ce que seul le contrôle ci-dessous établit. Je ne le suppose pas.

## 3. Le contrôle qui rend « faux », à sec, 15 min (Manon)

Reconvertir GLM `-k48-calibA` avec le correctif, **même corpus (sha256), même graine**, et comparer octet à octet les tenseurs `mlp.shared_expert.*` des 47 couches entre l'ancien et le nouveau converti (les experts routés doivent être identiques : c'est le témoin que rien d'autre n'a bougé).
* identiques ⇒ GLM n'était pas touché (chemin de collecte différent) ; ligne dans le verdict, fin.
* différents ⇒ touché ; le nouveau converti va à Laure (PPL 3 tranches, 20 min). **Prédiction : 1,0150 → 1,009-1,014. Un seul seuil : ≤ 1,013 (au-delà de la résolution) ⇒ la colonne GLM acvram est remplacée par le converti corrigé ; > 1,013 ⇒ colonne inchangée, expert partagé insensible, la conclusion du § 2 est rétablie telle quelle.**
Même contrôle, même fenêtre, sur `DeepSeek-Coder-V2-Lite-Instruct-nvfp4` (2 experts partagés, MLA) si un bf16 de référence est servable ; sinon noté, pas mesuré.

## 4. Régime dans la signature (REGLES § 4), obligatoire avant toute nouvelle conversion

Le correctif change la sortie du convertisseur : les convertis d'avant et d'après 0e4ef38 sont deux régimes. Le manifeste de chaque converti doit porter le **commit du convertisseur** (s'il ne le fait pas déjà : Manon, 20 min, champ `convertisseur` + test qui casse si absent), et les menus une colonne « convertisseur » ou la date du converti. Sans ça, la faute 6 revient à la prochaine reconversion.

## Ordre

Manon : § 3 (diff tenseurs, 15 min) → § 4 (manifeste) → reconversion si touché. Laure : PPL GLM après la GEMV experts, pas avant (la GEMV est la cellule phare ; ceci est une colonne qualité déjà classée).
