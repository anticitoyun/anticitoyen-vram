# kv31b levier 2, étape 1 — préfill de l'attention par morceaux (`ACVRAM_PREFILL_MORCEAU`, défaut OFF), via `forward_tranches`, au bit sur processeur (poste6, 30/09, branche poste6-prefill-morceaux, à sec)

instrument : jouet de la CI (`converted`, 4 couches, 512 jetons) sur processeur, modèle bf16, cache KV bf16 puis int8 ; logits du dernier jeton d'invite capturés à `_emit` ; `torch.equal`
commit : ce commit, sur poste6-reserve-tranches ; `acvram/engine/runner.py`, `tests/test_prefill_morceaux_kv2.py`
régime : à sec
scellé : la règle du bit (chef, duck.ai Q1 30/09 soir : vLLM n'est PAS au bit — softmax en ligne sur des morceaux de KV ; chez nous, découpe par LIGNES DE REQUÊTE, chaque ligne réduisant sur le même KV, dans le même ordre, avec le même noyau)
mesuré : 5 longueurs (17, 257, 300, 400, 511) par morceaux de 128 : logits IDENTIQUES au bit au seul tenant (cache bf16) ; int8 : écart 6-9e-3 (K/V relus quantifiés, comme une reprise après le cache de préfixe) ; repli séquentiel au bit ; 7 tests verts, cassure
verdict : LIVRÉ, défaut OFF, déclaré au régime « prefill=…(morceaux@N) », compteur `prefill_morceaux` ; au bit sous UN chemin MLP ; plancher de lignes 128 et « même côté du seuil de fusion que le seul tenant », dits quand ils refusent
durée : 0 min de carte

## Mécanique
`_prefill_morceaux(seq, fin)` : bornes `_limites_morceaux` (multiples du morceau, déplacées hors des images par `_eviter_coupe_image`, dernier
morceau ≥ plancher sinon fondu dans le précédent), un lot `_build_batch(limite=…)` par morceau (ce que la reprise après le cache de préfixe
produit déjà : `q_offset` > 0, K/V relus par la table de blocs), tous passés COUCHE PAR COUCHE par `forward_tranches` (284 b : à chaque couche
le morceau k écrit ses K/V, le morceau k+1 les relit) ; si le modèle refuse les tranches (images, deepstack, MTP) : `forward` morceau par morceau.
Hors périmètre : récurrence linéaire (`couches_recurrentes` > 0 : état séquentiel, instantanés) — un seul tenant, comme avant.

## Ce que le jouet a appris (chiffres)
| cas | écart max (logits) | cause |
|---|---|---|
| modèle fp32, cache bf16, morceaux 32 | 7e-4 | le CACHE arrondit les K/V relus ; en bf16 partout : 0 |
| bf16, morceaux 32, 64, 100 jetons | 0 | — |
| bf16, 257 jetons, morceaux 32 / 48 / 64 / 96 | 8e-3 / 2,5e-3 / 5,9e-4 / 5,9e-4 (chemins alignés) | chemins à petit M (≤ 32 : attention.py:101, model.py:341 ; 64-96 : non élucidé, dit) |
| bf16, 257 jetons, morceaux 128 / 160 | 0 | — |
| seuil de fusion gate/up (256) : MLP fusionné ≤ 256 lignes vs séparé au-delà | 1,8e-4 par ligne (module isolé) | `attention.py:636` — d'où « même côté du seuil » |
| seuil abaissé à 64 (seul tenant ET morceaux non fusionnés, jouet) | 4,5e-3 | le chemin non fusionné du jouet dépend des lignes : NON élucidé, dit ; la preuve de service (morceaux 4 096, seul tenant 32 768, tous non fusionnés) est à faire sur carte (S1 du scellé de conception) |
SDPA processeur seul (is_causal, biais bas-droite, masque, math) : exact par lignes à toute longueur (mesuré) — l'attention n'est pas en cause.
Règles retenues : morceau ≥ 128 lignes (`_MORCEAU_LIGNES_MIN`, réglage plus bas ignoré et dit) ; pour une invite > `SEUIL_FUSION`, morceau > `SEUIL_FUSION`
(sinon un seul tenant, dit une fois) ; dernier morceau ≥ ce plancher (fondu sinon). Test au bit sous fusion imposée (10⁶) : le jouet ne peut pas
tenir 32 768 ; sous le seuil de service il n'est pas au bit — la preuve « non fusionné contre non fusionné » revient à la carte.

## Reste
Preuve carte S1 (logits au bit, gemma-4-31B, morceaux 4 096 contre seul tenant sur un prompt > 5 fenêtres) avant l'anneau ; élucider le chemin
non fusionné à petit M du jouet (pièce à part) ; duck.ai Q2 (llama.cpp : anneau strict et repli en préremplissage complet ; vLLM : pas de préfixe
partagé hors fenêtre) versé au scellé de conception.
