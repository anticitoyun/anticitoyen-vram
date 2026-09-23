# Verdict — noyau MMA réel sur les experts, Qwen3-Coder-30B-A3B-nvfp4

Manon, 13/09/2026. Suite de `revue/prediction-moe-experts-a4.md` (addendum
« mesure du noyau RÉEL, pas d'un fake-quant ») — le fake-quant par hooks
s'étant révélé inerte à `window=2048` (`_forward_prefill_grouped` ne passe
jamais par `_grouped`), la mesure porte directement sur `ACVRAM_MOE_MMA=1`,
le vrai noyau W4A4 (`nvfp4_gemm_grouped_mma`), pas une simulation.

## Régime

`outils/moe-mma-reel-qwen3-coder.py`, un processus par régime (le flag est
figé à l'import de `model.py`), sous `systemd-run --user ... outils/carte.sh`
(échappe au garde-fou MemFree du superviseur — cf. Laurine/Jérôme). Corpus
wiki-gptq.txt (sha vérifié), fenêtre 2048/2048, min_context 0. Seuil scellé :
PPL mma-reel ≤ PPL a16 mesuré × 1,01 (revue/prediction-moe-experts-a4.md).

**Aucun étalon PPL archivé pour ce dossier avant cette campagne** — le
témoin a16 est mesuré ici, dans la même campagne.

## Mesure

| régime    | PPL                | débit    |
|-----------|--------------------|----------|
| a16       | 9,121772082952349  | 2052 j/s |
| mma-reel  | 9,20563488694055   | 7454 j/s |

Écart : **+0,919 %**. Seuil : 9,213. **RESPECTÉ.**

Vérification de mécanisme (contre l'inertie déjà rencontrée avec le
fake-quant) : débit nettement différent entre régimes (7454 contre
2052 j/s) — le chemin MMA est bien emprunté, pas silencieusement contourné.
Le débit absolu (2052/7454 j/s) est plus bas que les bancs chauds de
Laurine (8600/10500 j/s) : chargement froid dans ce script, pas un régime
comparable — sans conséquence sur le rapport PPL, qui ne dépend pas du
débit.

## Contre la prédiction

Prédiction scellée : +0,4 % à +1,0 %, réussite probable. **Mesuré : +0,919 %,
dans la fourchette, proche de la borne haute.** Confirmé.

## Décision

`ACVRAM_MOE_MMA` passe à **1 par défaut** (`acvram/engine/model.py:1095`) —
gate/up/down des experts MoE routés en W4A4 natif (E2M1 bloc 16, échelle
UE4M3), routeur/attention/expert partagé restent A16. `ACVRAM_MOE_MMA=0`
reste disponible pour revenir au chemin déquant+GEMM bf16.

## Limite assumée

Mesure en un seul passage par régime (pas de jumelles), comme les campagnes
précédentes de ce chantier. L'écart (+0,919 %) est proche du seuil (+1 %) —
une seconde passe est recommandée avant de considérer la question tranchée
au-delà du doute raisonnable, en particulier si un changement matériel ou
logiciel ultérieur touche ce chemin.

## Ce qui reste

Piste « lissage homogène par bloc » (notée par Jérôme comme repli si la
mesure avait échoué) sans objet ici — la mesure passe. Reste ouvert : la
piste (a) A4 gate/up seul + down en A16 n'a pas été mesurée séparément
(inutile, le régime complet passe déjà).
