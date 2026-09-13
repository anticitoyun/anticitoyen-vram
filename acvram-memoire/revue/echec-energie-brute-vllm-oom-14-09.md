# Échec — vLLM ne charge pas Qwen3-Coder-30B-A3B-nvfp4 sur cette carte (item 3, bloqué)

poste3, 14/09/2026. Suite à
[`protocole-energie-brute-faible-lot-14-09.md`](protocole-energie-brute-faible-lot-14-09.md).
Le bras acvram est fait (ci-dessous) ; **le bras vLLM échoue systématiquement
au chargement, avant toute mesure**, trois tentatives, trois causes
écartées.

## Ce qui marche : acvram, brut, b=1..4

    b    tok/s    J/jeton BRUT   watts_repos
    1    226,83   1,4395         68,2
    2    326,15   1,2287         76,2
    3    425,93   1,0011         73,8
    4    540,86   0,8039         74,0

Carte exclusive, EOS neutralisé aux deux moteurs (prévu), main d'aujourd'hui.
Fichier complet : `scratchpad/resultat-energie-brute-acvram-14-09.json`.

## L'échec vLLM

Trois lancements, `carte.sh` exclusif à chaque fois, carte à 15 Mio avant
chaque tentative (vérifié) :

    tentative 1 : gpu_memory_utilization=0.85 (config du duel de poste2)
                  → OOM au chargement, 30,44 Gio alloués, 234 Mio libres
    tentative 2 : gpu_memory_utilization=0.78 (réduit)
                  → OOM IDENTIQUE au chiffre près : 30,44 Gio, 234 Mio
    tentative 3 : + compilation_config cudagraph_capture_sizes=[1,2,3,4],
                  max_num_seqs=4 (réduire la capture de graphes)
                  → OOM IDENTIQUE au chiffre près : 30,44 Gio, 234 Mio

**Les trois leviers essayés n'ont eu AUCUN effet sur le point d'échec** —
le nombre exact d'octets alloués au moment du crash est identique aux
trois tentatives. Ceci montre que le crash survient **pendant le
chargement des poids eux-mêmes**, avant que `gpu_memory_utilization` (qui
ne dimensionne que le cache KV après chargement) ou `cudagraph_capture_sizes`
(capturé après chargement) n'aient leur mot à dire. Trois hypothèses
plausibles écartées par l'expérience — je n'en tente pas une quatrième
sans piste nouvelle (règle du 8/09, 3 hypothèses maximum).

## Ce qui ne colle pas

poste2 a mesuré ce même modèle avec vLLM avec succès le 14/09
(`audit-a2-duel-vllm-14-09.md`, 1 198,4 t/s à b=12) — la même config
`gpu_memory_utilization=0.85` a donc chargé chez elle. La carte annonce
aujourd'hui 31,36 Gio de capacité totale utilisable par PyTorch (contre
32 607 Mio nominaux par `nvidia-smi`, écart normal de ~500 Mio réservés
driver) — marge de 234 Mio au moment du crash, donc à la limite. Sans
accès à sa configuration exacte au moment de sa mesure (pilote, version
vLLM, éventuel `PYTORCH_CUDA_ALLOC_CONF`), je ne peux pas dire si quelque
chose a changé côté machine ou si sa marge était simplement plus large ce
jour-là.

## bd

Item 3 (énergie brute, volet vLLM) **bloqué, pas abandonné** — le volet
acvram est publiable seul, mais la comparaison demandée (acvram VS vLLM)
ne l'est pas sans le second bras. Je rapporte l'échec plutôt que de
fabriquer un chiffre manquant (règle 10). Piste à essayer ensuite (pas
tentée ici, en dehors du périmètre des trois essais) :
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (suggéré par le message
d'erreur lui-même) ou vérifier la config exacte du duel de poste2
(version vLLM, pilote) au moment où elle a réussi.
