# Les 33 groupes qui fusionnent sans arbitrage

Recensé le 9/09/2026 sur les 110 convertis de `models_acvram`, échelles
comparées **au bit près** (`np.array_equal`, zéro tolérance).

## Statut

**ACQUIS · GAIN SOUS LA RÉSOLUTION · NON MESURÉ**

Ces trois mots comptent et sont écrits **avant** toute mesure, pas après :

* **acquis** — les `act_scale` des projections d'un même groupe y sont
  identiques bit à bit, donc la fusion est **exactement** équivalente aux
  appels séparés. Aucune approximation, aucun arbitrage de qualité, aucune
  perplexité à mesurer. Le correctif du 9/09 (`eb4c099`) les autorise en
  faisant porter le scaler commun par la pile ;
* **sous la résolution** — 33 groupes sur 385 concernés, soit 8,6 %, sur une
  borne de gain d'environ 3 % : **~0,26 % attendu**, contre un seuil de
  détection du banc de **0,40 %** ;
* **non mesuré** — et il ne faut pas le mesurer isolément. Une campagne
  dessus rendrait un nul, et un nul se lit comme « ça ne rapporte rien ».

## Liste nominative

| modèle | gate/up | q/k/v |
|---|---|---|
| `Qwen3B-pipeline-nvfp4` | 2, 4, 26, 27, 29, 30, 31, 32, 34 | 19 |
| `deepseek-coder-6.7b-nvfp4` | 0, 2, 4, 15, 24, 25 | — |
| `Qwen2.5-Coder-3B-nvfp4` | 2, 31, 33, 34 | 9, 13, 19, 28 |
| `Qwen2.5-Coder-7B-nvfp4` | 0 | 5, 8, 18 |
| `Mistral-Nemo-12B-Heretic-nvfp4` | 1, 3 | — |
| `Qwen3-14B-nvfp4` | 7, 11 | — |
| `Nemo-12B-Claude-nvfp4` | 3 | — |

**25 gate/up + 8 q/k/v = 33 groupes.** Les **352 autres** ont des échelles
différentes et dépendent de l'arbitrage « AWQ par groupe ».

## Ce que le classement par nombre de groupes ne dit pas

`Qwen3B-pipeline-nvfp4` en a le plus (10) et pèse **2,25 Gio** : ses groupes
sont petits et le gain absolu y est minuscule. **Le classement par nombre de
groupes n'est pas le classement par gain** — la performance est dentelée en
taille, et le coût d'un GEMV à ces largeurs n'a pas été mesuré.

## Comment ce recensement a été rendu croyable

Le comparateur a été **validé avant d'être cru** : sur des triplets fabriqués
égaux et différents, et sur le témoin `Qwen2.5-Coder-14B-pur-nvfp4` où il rend
zéro identique — le même zéro qu'une mesure indépendante du moteur.

Sans cette validation, un zéro partout n'aurait pas été distinguable d'un test
aveugle. Trois mesures de la même journée ont échoué faute de cette
précaution, dont une qui a rendu +0,00 dB sur huit tenseurs en comparant
`None` à `None`.
