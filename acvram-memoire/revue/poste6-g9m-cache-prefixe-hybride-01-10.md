# g9m — cache de préfixe à 0 jeton sur gemma-4-31B : `est_hybride` = « a des layer_types », pas « a un état récurrent » (poste6, 01/10, diagnostic sans carte)

instrument : lecture runner.py + reproduction sur le jouet de la CI (processeur, cache de préfixe ON, même invite de 300 jetons deux fois, `est_hybride` forcé)
commit : 60e3a8e86 (aucun changement de code : diagnostic)
régime : à sec
scellé : prédiction avant la reproduction — « est_hybride vrai sans état récurrent → 0 jeton servi ; faux → > 0 »
mesuré : `est_hybride=False` : cached_prompt_tokens 288, hit_rate 0,480 ; `est_hybride=True` : 0, 0,000, aucun instantané (`_insta` vide) — tenu
verdict : cause nommée. runner.py:703 `self.est_hybride = bool(spec.layer_types)` ; à l'admission (runner.py:1341-1352) un hybride plafonne l'appariement du préfixe à la plus grande frontière photographiée (`_frontiere_disponible`), et `_photographier` (runner.py:1559) ne range rien sans état récurrent (`gdn_states` vide → `instant` vide → return) : frontière 0 → `matched = []` → l'invite entière est recalculée à chaque requête. gemma-4 (couches `sliding_attention`/`full_attention`, `couches_recurrentes` = 0) est traité comme Qwen3-Next
durée : 0 min de carte

## Effets du prédicat, et donc du correctif (`est_hybride = spec.couches_recurrentes > 0`, config.py:300)
Sur tout modèle à `layer_types` sans récurrence (gemma-3/4, Llama 4…), trois comportements changent, tous observés ce matin sur gemma-4-31B :
1. le cache de préfixe sert enfin (0 → N jetons ; S1 : 0 sur une requête identique, Devstral 19 %) ;
2. plus de coupe du préfill à la frontière d'instantané (régime `prefill=…(coupé@256)` disparaît ; la passe en deux morceaux que la pièce S1 a
   dû couvrir n'a plus lieu) ;
3. décodage : runner.py:2158 (`hyb` → `_plain_decode` dès que > 1 séquence ou sans graphe) et graphes.py:55 (chauffe spéculative k+1) ne
   s'appliquent plus à gemma — décodage spéculatif par lot comme un dense. C'est le point qui touche le débit (b=12) : à mesurer, pas à supposer.
Correctif d'une ligne + test (jouet, `layer_types` posés sans type récurrent → cache servi) prêts ; non commités : les effets 2-3 changent la
sortie servie et le débit de gemma — décision du chef, mesure b=12 gemma avant/après par poste2 si (3) est retenu.
