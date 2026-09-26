# poste7 — GLM vérifié : conclusion « calibration pas le levier » RÉTABLIE avec sa mention ; et un fait de méthode plus important que la question — l'AWQ n'est pas reproductible d'un run à l'autre, donc le bruit de reconversion doit être mesuré avant tout futur « remplace la colonne » (17/09)

Entrée : poste2 7334619 — `-verif17-09` en 6 h 17 (contention) ; les 141 `mlp.shared_expert.*` sont en **INT8** (promotion `mixed_precision=auto`) dans les deux convertis, `has_act_scale=False`, `out_snr_db` identiques à 2 décimales : l'AWQ est désactivé sur l'int8 par décision (`poste7-organisation-16-09`), le correctif de clé ne pouvait rien y changer. Témoin routé nvfp4 : reconstruction différente de **10,7 %** entre les deux runs à SNR identique.

## 1. La conclusion est rétablie, avec une ligne

ETAT / `poste7-calibration-verdict` : « calibration pas le levier sur GLM » — **rétablie**, mention : « vérifié 17/09 (7334619) : expert partagé en INT8 hors AWQ dans tous les bras par construction ; le bogue de clé 0e4ef38 n'avait pas d'effet sur GLM ». Ce qui reste non démontré et ne vaut pas une fenêtre : un AWQ sur un tenseur INT8 (44,6 dB) n'a rien à gagner de mesurable.

## 2. Le fait qui compte : l'AWQ n'est pas déterministe

Deux runs, mêmes options, même corpus, même graine, même device : un expert routé diffère de 10,7 % en reconstruction pour un SNR identique. La recherche (20 points, réductions flottantes non associatives) a un objectif plat : beaucoup d'optima équivalents. Conséquences, dans l'ordre :

1. **Le « diff octet à octet » n'est pas un contrôle valable sur un converti AWQ** — retirer cette forme de contrôle de mes protocoles (celui de `poste7-expert-partage-cle-portee` § 3 tenait par chance : les tenseurs comparés étaient INT8 sans AWQ). Les contrôles se font sur SNR par tenseur et PPL.
2. **Tous mes seuils « ≤ X remplace la colonne » supposent un bruit de reconversion < 0,002 — jamais mesuré.** REGLES § 3 : un scellé ne descend pas sous 2× l'écart du témoin. Il faut le témoin : **PPL géo 3 tranches de `-verif17-09` contre le `-k48-calibA` d'origine (1,0150)**, poste3, 20 min, quand la carte est libre (après Nemotron calibA, avant toute autre PPL). Prédiction : |Δ| ≤ 0,002. Un seul seuil : |Δ| ≤ 0,002 ⇒ mes seuils tiennent ; |Δ| > 0,002 ⇒ tout seuil de remplacement futur passe à 2 × |Δ| mesuré, et le verdict Nemotron calibA se lit avec cette marge.
3. **Le manifeste porte la vérité** : commit du convertisseur, corpus sha256, graine, device, **et la mention « AWQ non déterministe : deux convertis d'options identiques sont deux fichiers »** — un converti est identifié par son sha256, jamais par ses options. Les deux convertis sans source dans le manifeste (DeepSeek-Coder-V2-Lite, GLM-4.7-Flash-nvfp4) le montrent : champ `source_sha256` obligatoire, conversion refusée sans lui.

## 3. Rien d'autre

Scan du parc : continue ; Nemotron calibA relancé : prédiction 1,015-1,025 inchangée, lue avec la marge du § 2.2 si elle est mesurée avant.
