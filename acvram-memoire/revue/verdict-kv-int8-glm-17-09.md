# Verdict — piste « KV int8 » pour l'inversion privé/public d'acvram GLM : sans objet par construction

instrument : lecture du code et des manifestes, à sec (aucune prise) · commit : arbre laure cbea64e (= main) · régime : GLM `-k48`, PPL prefill (`acvram eval`) — c'est là que l'inversion a été mesurée (`verdict-ppl-refonte-17-09` : privé 1,0289, public 0,9972)
scellé (Jérôme/Sage) : PPL privée avec KV bf16, |privé − public| ≤ 0,01
mesuré : **il n'y a pas de KV int8 dans la PPL GLM d'acvram.** (a) GLM est MLA sur ses 47 couches : aucun cache paginé (`model.caches` vide, `config.py:272-280`), donc le `kv_format: int8` du plan (`tiers[0]`) ne s'applique à rien ; le latent MLA est en **bf16** (`mla.py:355-357`, uint8/fp8 seulement sous `ACVRAM_MLA_LATENT_FP8=1`, OFF par défaut et non posé dans mes prises). (b) La PPL prefill calcule l'attention sur la séquence complète : elle n'écrit ni ne relit un cache quantifié. La mesure demandée « KV bf16 » est donc la mesure déjà faite : **1,0289 privé / 0,9972 public, écart 0,032** — scellé réfuté par construction, sans nouvelle prise.
verdict : **KV innocenté comme l'AWQ. Ce qui reste et distingue GLM de Coder (Coder : privé 1,027 / public 1,018, pas d'inversion) n'est ni le KV ni l'AWQ ; et l'inversion est partagée par `-sansawq` (0,015) et `-avant-alpha` (0,021).**

## Pistes qui restent, par ordre de coût
1. Les experts NVFP4 (9 024 tenseurs, g16) sur un registre français dense en nombres : llama.cpp Q4_K_M (imatrix unsloth, mélange non publié) va dans l'autre sens (privé 1,015 < public 1,027) — le classement dépend du corpus sur lequel la quantification a été réglée, même sans « calibration » nommée (RTN NVFP4 ≠ Q4_K avec imatrix). Test : le converti Hadamard de Manon (rotation, moins dépendant de la distribution) sur privé ET public — c'est la prochaine mesure de toute façon.
2. Les 190 tenseurs d'attention int8 et les 2 nvfp4 (GLM) contre 192 int8 (Coder) : à isoler par un converti GLM à attention bf16 (`keep_sensitive_16bit` étendu à toute l'attention) — Manon, à sec.
3. Le bruit : ±0,004 par tranche à 12 288 cibles ; 0,032 est huit fois au-dessus, ce n'est pas du bruit.

## Ce que ça change
Rien à la table : les chiffres tiennent. Le prochain verdict GLM est la PPL Hadamard (prefill + décodage 12/12, préfixe compris, privé + public, médiane), dès livraison de Manon.
