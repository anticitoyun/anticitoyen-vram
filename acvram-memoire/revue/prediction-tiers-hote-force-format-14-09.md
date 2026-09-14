# Prédiction scellée — le tiers hôte doit lire `force_format`

poste1, 14/09/2026, avant mesure.

## Cause (lue, pas mesurée)

`build_tiers` (acvram/memory/tiering.py:346-376) : le tiers GPU lit
`opts.force_format or (caps.weight_format ...)` (ligne 350) mais le tiers
hôte lisait `tiers[0].weight_format if tiers else "int4_awq"` (ligne 372) —
sans GPU visible (`rig.gpus` vide), `tiers` est encore vide à ce point, donc
le repli est TOUJOURS `"int4_awq"`, quel que soit `--format`. C'est
précisément le cas de `mini-acvram3` (`CUDA_VISIBLE_DEVICES=""`,
`--host-exec cpu`, `--format bf16 --no-awq`) : manifeste vérifié, 207/223
tenseurs en `int4_awq` malgré la demande explicite de bf16.

## Prédiction

Avec le correctif (`opts.force_format` lu en premier, comme le tiers GPU),
`build_tiers(Rig(gpus=[], host=HostMemory(total=64*2**30,
available=60*2**30)), PlannerOptions(force_format="bf16",
allow_host_tier=True))` rend un tiers `"cpu"` avec `weight_format="bf16"`.

**Seuil de réfutation** : `weight_format` différent de `"bf16"` dans ce
cas — le correctif serait sans effet ou mal placé.
