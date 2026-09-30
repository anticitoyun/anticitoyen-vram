# 90q — GUI kimi-modeles : seuil_complet 65 536 recopié de kimi-modele → une seule source, même méthode que ph1 (poste6, 30/09, branche poste6-90q, à sec)

instrument : pytest à sec (`CUDA_VISIBLE_DEVICES=""`, venv du dépôt principal), xvfb pour la GUI ; aucun modèle, aucune carte
commit : 3e87be573 (base, poste6-ph1 = origin/main + ph1) ; correctif dans ce commit (`parc/bin/kimi-modele`, `parc/bin/kimi-modeles`, `parc/lib/menu_modeles/config.py`, tests)
régime : à sec, aucune prise de carte (arbre principal gelé : rien tiré dans main)
scellé : rien — pièce de justesse ; contrôles écrits avant (C1-C4) et rendus
mesuré : rien de chiffré ; 65 tests verts (lanceurs, GUI, menus, contexte)
verdict : copie retirée — kimi-modeles lit `kimi-modele --seuils` ; KIMI_MCP_CTX_MIN défini UNE fois en tête, préchargement acvram lié ; deux cassures vérifiées
durée : 0 min de carte / 0 tenu

## Cause (avant correctif)
* `parc/bin/kimi-modeles:55` : `"seuil_complet": 65536` et `:57` `"seuil_minimum": None`, recopiés de `kimi-modele:110,115` (poste3-gui 29/09, 4df0adeb6).
* `parc/bin/kimi-modele` : `${KIMI_MCP_CTX_MIN:-65536}` écrit DEUX fois (:115, :117) et `CTX_CLIENT_MIN=65536` en dur au préchargement acvram (:173) —
  trois endroits pour un seul seuil, la GUI en faisait un quatrième. Pas de bogue visible aujourd'hui (les quatre valent 65 536) : défaut latent, celui de ph1.

## Correctif
* `kimi-modele` : `KIMI_MCP_CTX_MIN="${KIMI_MCP_CTX_MIN:-65536}"` en tête (env toujours honoré, une définition) ; `kimi_local` l'utilise ; préchargement
  `CTX_CLIENT_MIN=$KIMI_MCP_CTX_MIN` (servi à cette fenêtre, kimi_local garde les MCP — c'est le sens de la demande haute) ; `--seuils` imprime
  `seuil_complet=<KIMI_MCP_CTX_MIN>` et `seuil_minimum=` VIDE (aucun plancher de refus générique), avant l'eval python du parc.
* `menu_modeles.config.seuils_lanceur` : valeur vide → `None` (pas de seuil), sinon entier. `kimi-modeles` : `**seuils_lanceur("kimi-modele")`, plus aucun chiffre ni `None` codé.

## Contrôles (écrits avant, rendus)
| # | contrôle | rendrait faux si… | rendu |
|---|---|---|---|
| C1 | `kimi-modele --seuils` sous HOME vide → `65536` / vide ; avec `KIMI_MCP_CTX_MIN=40000` → `40000` | l'env n'était plus honoré, ou l'option passait après l'eval du parc | ✔ |
| C2 | une seule occurrence de `65536` dans kimi-modele | une seconde définition ou un préchargement en dur subsistait | 1 ✔ |
| C3 | la GUI rend `{65536, None}` par `seuils_lanceur("kimi-modele")` | vide mal lu (ValueError → `{0, None}`) | ✔ |
| C4 | GUI xvfb kimi : 1 000 → réduit « 65 536 » jamais refus ; 70 000 → rien | seuil GUI ≠ lanceur | ✔ (tests existants, inchangés) |

Cassures jouées à la main puis restaurées : `"seuil_complet": 65536` remis dans kimi-modeles → 1 rouge ; `CTX_CLIENT_MIN=65536` remis en dur → 1 rouge (C2).
Piège rencontré : le remplacement global `${KIMI_MCP_CTX_MIN:-65536}` → `$KIMI_MCP_CTX_MIN` a aussi mangé la définition en tête (« variable sans liaison » sous
`set -u`) ; et le lanceur SANS `--seuils` appelé avec `--seuils` ouvre le menu interactif et pend — d'où un premier essai de 10 min à tuer (pgrep). Toujours `timeout 5`.

## Reste
* Rien. CHANGELOG/README à la prochaine version (ph1 + 90q ensemble : « seuils des GUI lus dans les lanceurs »).
