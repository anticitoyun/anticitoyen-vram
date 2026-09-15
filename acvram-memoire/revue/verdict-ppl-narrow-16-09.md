# Verdict — PPL teacher-forcing, voyant b=1 de `narrow_gemm` : PASSÉ (ratio 1,000000)

poste2, 16/09. Ordre poste7 §6 (`poste7-reprise-15-09-b.md`, main `77b5def`),
transmis par chef, tâche confiée par poste3 dans
`protocole-narrow-voyants-15-09.md` : « poste2 fait la PPL teacher-forcing
(1,000 ± 0,002) ». Réfuté si le ratio B/A s'écarte de plus de 0,002 de 1,000.

**Prédiction avant mesure** : `_NARROW_MIN` = 2 par défaut
(`kernels/__init__.py:502`) → à `t=1` par pas (teacher forcing, batch=1) le
chemin étroit n'est JAMAIS pris, quel que soit `ACVRAM_NARROW_GEMM` — les
deux bras doivent tomber sur le MÊME chemin (GEMV), donc PPL strictement
identique. Réfuté si les PPL diffèrent (alors soit la garde ne fait pas ce
qu'elle dit, soit `ACVRAM_NARROW_GEMM` fuit dans un autre chemin que prévu).

## En-tête de mesure

Instrument : `outils/ppl-decode-mma-coder30b.py` (teacher forcing, KV
continu, jeton forcé du corpus, comptage `pas_t_le_32` en preuve du chemin
de décodage réellement emprunté) — modifié ce soir pour ASSERT
`ACVRAM_NARROW_GEMM` posé avant import et le publier dans la ligne
RESULTAT (même piège que `ACVRAM_MOE_DECODE_MMA`, déjà gardé). Une seule
carte (`carte.sh`, `CUDA_VISIBLE_DEVICES=0`). Pas de plafond de puissance
ni d'horloge à publier : ceci est un contrôle de COHÉRENCE (la sortie
change-t-elle ?), pas une mesure de débit/énergie — REGLES §3 ne s'y
applique que pour l'instrument et la carte, déjà donnés.

Modèle : `Qwen3-Coder-30B-A3B-nvfp4`, corpus `wiki-gptq.txt`, 8191 jetons
notés, `pas_t_le_32=8191/8191` (chemin décodage `_forward_grouped_mma`
bien emprunté à chaque pas, `t=1` ≤ 32).

## Résultat

```
A (ACVRAM_NARROW_GEMM=0) : ppl=8.7501 acvram_moe_decode_mma=1 acvram_narrow_gemm=0
B (ACVRAM_NARROW_GEMM=1) : ppl=8.7501 acvram_moe_decode_mma=1 acvram_narrow_gemm=1
```

**Ratio B/A = 1,000000. PASSÉ**, exactement comme prédit — à `b=1` la garde
`_NARROW_MIN=2` empêche `narrow_gemm` d'être pris, les deux bras exécutent
le même code, la PPL est bit-identique.

**Ce voyant NE JUGE PAS s3@3** (`verdict-narrow-voyants-15-09.md` § (2),
14 ulp hors ex-aequo, sous le plancher témoin 88 ulp) : cette divergence a
été observée à `b=12`, où `narrow_gemm` s'active réellement (le script
teacher-forcing utilisé ici est `max_batch_size=1` par construction — le
même script que `a10b987`). Un jugement PPL de cette divergence précise
demanderait une mesure teacher-forcing à `b=12`, non tentée ce soir (script
à écrire, hors du "~20 min" annoncé). Signalé à chef.

## Suite

Rendu à chef : ma prise libère poste3. Aucune reconversion GLM relancée —
en attente de la décision de poste7 sur `gate=up` dans la pile AWQ
(`verdict-glm-alpha-commun-experts-15-09.md`, PPL réfutée 1,03086).
