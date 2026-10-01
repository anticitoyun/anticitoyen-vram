# lic — test décisif sans carte : un seul tenant qui relit ses propres K/V quantifiés égale-t-il les morceaux au bit ? NON sous int8 (poste6, 01/10, branche poste6-reserve-attention)

instrument : jouet de la CI (`converted`, 4 couches) sur processeur, `tests/test_prefill_morceaux_kv2.py::test_relire_ses_kv_fait_du_seul_tenant_un_morceaux_au_bit`, logits du dernier jeton d'invite (400 jetons), morceaux de 128, `torch.equal`
commit : 73d0d97be + ce commit (témoin `ACVRAM_PREFILL_RELIRE_KV`, test)
régime : à sec (`CUDA_VISIBLE_DEVICES=""`), venv du dépôt principal ; aucune carte
scellé : prédiction écrite dans la docstring avant la mesure — « K/V relus des deux côtés → au bit ; FAUX si Δ ≠ 0 : le chemin porte une différence propre »
mesuré : bf16 : morceaux − seul tenant 0 ; K/V relus des deux côtés 0. int8 : morceaux − seul tenant 6,3e-3 ; **K/V relus des deux côtés 4,3e-3**
verdict : **RÉFUTÉ, mécanisme nommé** — à entrées identiques (q, K/V relus égaux au bit, vérifié), le noyau d'attention rend pour UNE ligne (253) un résultat qui dépend de la LONGUEUR TOTALE DES CLÉS de l'appel (256 pour le morceau, 400 pour le seul tenant) : réduction par blocs de clés (softmax en ligne), pas le masque, pas le format — l'int8 ne fait qu'exposer une frontière d'arrondi (1 ulp bf16, 3,05e-5). « Au bit à toute longueur » (30/09) était une mesure heureuse. Cliquet : `xfail(strict=True)` sur le cas int8
durée : 0 min de carte

## Le témoin
`attention.py:_prefill` : `if cache is not None and (offset > 0 or _RELIRE_KV)` → `cache.gather` aussi pour un seul tenant (le cache est écrit avant
l'attention dans les deux chemins, attention.py:347-352). Hors défaut, `ACVRAM_PREFILL_RELIRE_KV=1`, déclaré `prefill=…(relu)` au régime, dans
`regime.VARIABLES` et la liste du CLI.

## Ce que cela exclut et ce qui reste
* Exclu : la granularité d'écriture des échelles int8 — `_quantize` est par jeton et par tête (`amax(dim=-1)`, kvcache.py:492) : écrire 400 jetons
  d'un coup ou 128 + 128 + 144 donne les mêmes octets. Exclu aussi : la longueur de réduction de l'attention (bf16 au bit à toute longueur, mesuré).
* Reste à ouvrir (bissection à sec, par couche puis par ligne, caches ET états) : pourquoi, sous int8, les lignes 0-127 d'un seul tenant relu ne
  donnent pas les mêmes K/V au cache que le morceau 1 relu — même entrée, même dequantification. Suspects, le plus probable d'abord :
  (1) `gather` d'une longueur non multiple de bloc : le dernier bloc partiel (400 = 25 × 16 exact ici, mais 128 + 144…) ; (2) un état du cache lu
  par `gather` avant d'être totalement écrit pour les positions du morceau courant (ordre write/gather par tête ou par bloc) ; (3) V :
  `storage_dim_v`/`echelles_v` ou le chemin `_dequantize_v` différent de K.
* Sur carte (S1 du 01/10) : gemma Δ lp 0,60 / 0,088, Devstral 0,041 — même classe de cause probable, plus les noyaux GPU (paged, q_offset) : à
  reprendre seulement après la bissection à sec.

## Bissection (ordre chef 01/10) — premier élément qui diffère, puis le mécanisme
* Par couche et par module (hooks, jouet int8, relecture des deux côtés) : **couche 0, sortie de l'attention, ligne 253, Δ 3,05e-5** — une seule
  ligne sur 400 ; caches K/V/échelles de la couche 0 identiques ; tout le reste (121 lignes à la couche 1, une position de cache) en découle.
* Les appels `attention()` de la couche 0 (espion) : morceau [128, 256) reçoit q, K, V ÉGAUX AU BIT à ceux du seul tenant (tranches), même
  `scale`, `n_rep` 4, `window` 0 ; sa sortie diffère à la ligne 253 ; rejouer la fonction sur les tenseurs du seul tenant découpés pareil
  reproduit l'écart (3,05e-5) ; rejouer l'appel morceau à l'identique rend l'identique (déterministe).
* Le noyau, ligne 253 : `is_causal` à 400 = masque booléen à 400 = seul tenant ; masque booléen à 256 = `causal_lower_right` à 256 = morceau.
  Clés REMPLIES de zéros masqués au-delà de 256 : longueur totale 272, 320, 384, 400 → égal au seul tenant ; 256, 288, 512 → différent.
  Donc : le SDPA processeur réduit les clés par blocs et le découpage dépend de la longueur totale de l'appel (softmax en ligne, ordre des
  sommes partielles) ; chaque variante reste à 2,4e-4 du fp32. Le GPU (flash, paginé) a la même propriété, amplifiée par la profondeur (40-60
  couches) et par le cache int8 (S1 : Devstral 0,041, gemma 0,088).
* Conséquence : « au bit » n'est PAS un objectif atteignable pour un préfill par morceaux avec un noyau d'attention par blocs — la bonne
  exigence est celle de la reprise après le cache de préfixe, déjà acceptée : équivalence sous témoin (KL/logprobs ≤ 2 × le témoin), à sceller.
  Reste à part : gemma 0,60 avec cache (frontière 7 936) contre 0,088 sans — un bras `ACVRAM_INSTA_PAS` grand, sur carte, le dira.

## Reste
`ACVRAM_PREFILL_MORCEAU` reste OFF ; pas d'anneau tant que chef n'a pas scellé l'exigence (témoin reprise, 2 ×) ; scripts de bissection
non commités (scratch, reproductibles par les lignes ci-dessus).
