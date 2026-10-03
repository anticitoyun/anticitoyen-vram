# Produits cuBLAS bf16 du préfill, carte : flash confirmé (B1), `down_proj` (K = 32 768) dépend bien du découpage (18,6 %), les six produits rendus indépendants de M au bit (B3, B5) — mais le moteur dépend ENCORE du découpage sous `etroite` et sous `tranches`, et les deux bras coûtent trop (+2,32 % et +7,44 % à M = 4 096) : aucun bras retenu, défaut inchangé

instrument : `scratchpad/poste6-bf16/carte-cublas.sh` (garde de chaîne `poste6-cublas-chaine`, 11 bras `service`) — `tests/test_reduction_bf16.py` sur carte (E0-E2, B5 au test), `prise-cublas.py` (B1 `can_use_flash_attention(debug=True)`, B2-B6 produits nus aux cinq formes de Devstral, C2 / C4 débit moteur ABBA 6 passes, modes basculés dans le processus), chaîne dense S1 sous `etroite`, `tranches`, `reduite`, comparateur même candidat ; journal `scratchpad/poste6-bf16/carte-cublas.log` (non suivi)
commit : poste6-gemma-anneau c89fb1e6a (main 4f39deccf fusionné), arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, 15 Mio avant et après, aucun autre poste entre deux bras ; `llama-server` étranger sur la carte 1 (5,6 Gio) ; Devstral-24B `srcawq-nvfp4`, NOMINAL, 0 exil, 7 865 jetons, 32 décodés ; torch 2.14.0+cu130
scellé : `poste6-bf16-prefill-cublas-scelle-03-10.md` (B1-B6, C1-C6, décision écrite avant)
mesuré : 03/10 03:20:39 → 03:26:11, une fenêtre
verdict : **aucun bras retenu.** `etroite` redéfinie (six sites exacts) : B / A1 n'est pas au bit (C1 FAUX : premier jeton basculé, Δ 0,047 — à peine moins que `reduite`, 0,061) et coûte +2,32 % à 4 096 (C2, seuil 2 %). `tranches` : B / A1 mêmes ids mais Δ 0,216 sur 319 valeurs (C3 FAUX) et +3,20 / +7,44 % à 2 048 / 4 096 (C4 FAUX) — les GEMM larges perdent × 1,17-1,28 en blocs de 1 024 (B6, ma prédiction fausse). **Le fait neuf** : quand les six produits cuBLAS des poids sont rendus indépendants de M au bit (B3, B5 : 0 partout, y compris `down_proj` qui dépendait de M à 18,6 %), le moteur dépend encore du découpage — et le seul tenant sous `etroite` est au bit de `reduite` (C5 : Δ 0) alors que le drapeau global le changeait (C3 du 02/10). Ce qui reste est donc un produit cuBLAS bf16 HORS des six sites, sensible au drapeau global, dépendant de M, non nommé — candidat : l'attention du moteur sur le chemin « math » malgré B1 (flash admissible n'est pas flash pris), à établir en forçant le backend. Défaut `reduite` inchangé, `etroite` et `tranches` opt-in.
durée : 5 min 32 s de carte (prévu ≈ 9 min)

## Prédit / mesuré

| # | prédit | mesuré | |
|---|---|---|---|
| C0 garde de chaîne | tenue | 03:20:39 → 03:26:11, 11 bras, aucun autre poste | tenu |
| B1 flash admis (seul tenant 7 865 ; morceau 3 769 / 7 865), GQA 32 / 8, d 128, sans masque | True, True | **True, True** | tenu — (a) confirmé pour l'admissibilité |
| B2 dépendance à M, drapeau retiré, 7 865 contre 4 096 + 3 769 | q, gate / up : 0 · o : 0-5 % · **down > 0 (10-40 %)** · k / v 15-31 % | q 0 · **k / v 31,46 %** · o 0 · gate / up 0 · **down 18,55 %** | tenu |
| B3 idem, drapeau posé | 0 partout | 0 partout | tenu |
| B4 chrono exacte / réduite, M = 4 096 | down × 1,2-2,0 · gate / up × 1,00-1,10 · q, o × 1,00-1,15 · k / v × 1,13 | **down × 1,22** (7 624 / 6 264 µs) · **gate / up × 1,11** (6 897 / 6 237) · q, o × 1,00 · k / v × 1,13 ; à M ≤ 2 048 : × 1,00 partout sauf k / v 512 (× 1,07) | tenu (gate / up à la limite haute) |
| B5 tranches 1 024 sous drapeau retiré, dépendance à M | 0 partout | 0 partout (k / v et down au test aussi : 0 sur 8,05 M et 40,3 M éléments) | tenu |
| B6 chrono tranches / réduite | × 1,00-1,06 (larges × 1,00-1,03) | 512, 1 024 : × 1,00 ; 2 048 : q × 1,12, gate / up × 1,17, down × 0,96 ; **4 096 : q × 1,17, gate / up × 1,28, down × 1,22**, k / v × 1,15, o × 1,02 | **FAUX** : cuBLAS à 1 024 lignes est bien moins efficace qu'à 4 096 sur les GEMM larges |
| **C1 dense S1, `etroite` (six sites), B / A1** | Δ 0, ids égaux | **ids divergents dès la position 0, Δ 0,0466** sur 10 valeurs (≤ 2 × témoin 0,0533 : « tenu » au sens de REGLES § 4, pas au bit) ; `reduite` : 0,0613 | **FAUX** |
| **C2 débit `etroite` contre `reduite`** | 4 096 : +1,8 à +2,6 % · 2 048 : +0,4 à +1,0 | 512 : +0,04 % · 1 024 : +0,03 % · 2 048 : −0,01 % · **4 096 : +2,32 %** (1 157,6 contre 1 131,3 ms, étendues disjointes) | > 2 % : **non retenue** (prédit) |
| **C3 dense S1, `tranches`, B / A1** | Δ 0, ids égaux | ids ÉGAUX (sha eadeaf5a…), mais **Δ 0,216 sur 319 valeurs** (seuil 0,105) | **FAUX** |
| **C4 débit `tranches` contre `reduite`** | 4 096 : +0,2 à +1,5 % · 2 048 : +0,0 à +0,8 | 512 : −0,05 % · 1 024 : +0,01 % · **2 048 : +3,20 %** · **4 096 : +7,44 %** (1 221,6 contre 1 137,0 ms) | **FAUX**, non retenue |
| C5 seul tenant A1 `etroite` / `tranches` contre `reduite` | `etroite` : non prédit ici ; `tranches` : ids différents possibles | `etroite` : **ids égaux, Δ 0 sur 320** · `tranches` : ids différents dès le jeton 22, Δ 0,142 sur 229 | dit |
| C6 découpage non aligné (reprise sur cache de préfixe) sous `tranches` | B ≠ A1 attendu | reprise contre A1 : premier jeton basculé, Δ 0,0526 | tenu (limite confirmée) |

## Ce que la fenêtre apprend

1. **Les six produits cuBLAS des poids ne sont pas toute la cause, ni sous `etroite` ni sous `tranches`.** B3 et B5 les
   rendent indépendants de M au bit, et le moteur diverge encore entre B et A1 (C1 : bascule du premier jeton ; C3 :
   mêmes ids mais logprobs à 0,216). Or le drapeau global rend B = A1 au bit (C1 du 02/10). Il existe donc un produit cuBLAS
   bf16 **hors des six sites**, lu par le drapeau global, dépendant de M. Il est aussi ce qui changeait le seul tenant sous
   le drapeau global (C3 du 02/10) alors que `etroite` ne le change pas (C5 : Δ 0) — la cause de « C3 faux » du 02/10
   est le même produit inconnu, pas k / v.
2. **Candidat, non mesuré** : l'attention sur le chemin « math » (`bmm` cuBLAS : scores M = lignes de requêtes, sortie
   N = 128 avec K = longueur des clés — la forme même qui coupe la réduction). B1 ne dit que l'admissibilité de flash pour
   des tenseurs nus ; le moteur peut tomber en « math » par un détail de disposition (q / k / v transposés depuis
   [T, têtes, d], `_kv_transitoires`, masque dense d'un autre chemin que `bas_droite`) ou au décodage
   (`batched_decode_attention`, masque explicite → pas de flash : M = 1, K = 7 865 : réduction coupée, et les logprobs
   des 32 jetons décodés en dépendent — ce qui expliquerait C3 : mêmes ids, logprobs différents). À établir sur carte en
   forçant `torch.nn.attention.sdpa_kernel([FLASH_ATTENTION])` autour d'un préfill (erreur = pas flash) et en comptant
   les noyaux au profileur ; une minute de carte, aucun modèle de plus.
3. **`down_proj` (K = 32 768) dépendait de M à 18,6 %** sous le défaut — jamais vu avant parce que H4 l'avait en int8 sur
   la couche 0. Les +2,40 % du drapeau global se décomposent : gate / up × 1,11 et down × 1,22 à M = 4 096 (B4), k / v
   × 1,13 ; q et o rien. Au moteur, `etroite` (six sites) coûte +2,32 % : le drapeau global ne coûtait donc presque rien
   ailleurs — l'inconnu de 1. est bon marché sous réduction exacte.
4. **Les tranches de 1 024 sont une mauvaise idée pour cuBLAS** : × 1,17-1,28 sur q et gate / up à 4 096 (B6), +7,44 % au
   moteur. Le noyau choisi à 1 024 lignes n'est pas celui de 4 096 en plus petit. Bras clos.
5. **C1 « tenu » au sens de REGLES § 4 et pourtant faux au bit** : 0,0466 ≤ 2 × 0,0267 parce que le témoin reprise bascule
   lui-même sur cette invite en quasi-égalité ; l'ordre du chef (« la sortie ne dépend plus du découpage ») demande le bit,
   pas le seuil — je juge au bit, comme le scellé l'écrit.

## Décision appliquée et ce qui reste au chef

* Aucun bras retenu ; `ACVRAM_BF16_REDUCTION` : `reduite` (défaut), `exacte`, `etroite`, `tranches` opt-in. Je propose de
  **retirer `tranches`** (plus cher et pas au bit : rien à garder) et de garder `etroite` (six sites, gratuite sous 4 096,
  utile à qui veut k / v et down indépendants de M sans le drapeau global) — à toi.
* La suite qui découle des chiffres, à sceller avant : nommer le produit restant (point 2 : flash forcé au préfill et au
  décodage, profileur). S'il est l'attention en « math » : le levier n'est plus cuBLAS mais le backend (flash au préfill
  par le biais bas-droite déjà là ; au décodage, un noyau sans masque dense) — et le drapeau global devient inutile.
* La garde de qualité au modèle reste due avant toute bascule du défaut ; rien ici ne la remplace.
