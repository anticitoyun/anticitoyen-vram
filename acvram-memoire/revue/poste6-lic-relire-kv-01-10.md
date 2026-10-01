# lic — test décisif sans carte : un seul tenant qui relit ses propres K/V quantifiés égale-t-il les morceaux au bit ? NON sous int8 (poste6, 01/10, branche poste6-reserve-attention)

instrument : jouet de la CI (`converted`, 4 couches) sur processeur, `tests/test_prefill_morceaux_kv2.py::test_relire_ses_kv_fait_du_seul_tenant_un_morceaux_au_bit`, logits du dernier jeton d'invite (400 jetons), morceaux de 128, `torch.equal`
commit : 73d0d97be + ce commit (témoin `ACVRAM_PREFILL_RELIRE_KV`, test)
régime : à sec (`CUDA_VISIBLE_DEVICES=""`), venv du dépôt principal ; aucune carte
scellé : prédiction écrite dans la docstring avant la mesure — « K/V relus des deux côtés → au bit ; FAUX si Δ ≠ 0 : le chemin porte une différence propre »
mesuré : bf16 : morceaux − seul tenant 0 ; K/V relus des deux côtés 0. int8 : morceaux − seul tenant 6,3e-3 ; **K/V relus des deux côtés 4,3e-3**
verdict : **RÉFUTÉ** — la différence morceaux/seul tenant n'est pas le seul format de lecture : à format égal et relecture égale, le chemin par morceaux diffère encore sous int8 (0 sous bf16). Cliquet : `xfail(strict=True)` sur le cas int8
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

## Reste
Bissection (pièce lic, suite) ; `ACVRAM_PREFILL_MORCEAU` reste OFF ; pas d'anneau.
