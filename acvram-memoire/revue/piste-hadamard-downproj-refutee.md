# Piste 2 (duck.ai 12/09) réfutée : Hadamard aveugle sur down_proj/NVFP4

Manon, 13/09/2026, à sec. Consigne de Jérôme : « Hadamard sur w2 — option
de conversion, test synthétique d'orthogonalité et de non-régression du
SNR, sans carte. »

## Ce qui était proposé

`revue/duck-manon-12-09.md`, piste 2 : appliquer une rotation de Hadamard
sur `down_proj` avant quantification NVFP4, motivée par Bridging Gap
(arXiv:2509.23202) qui documente des outliers massifs à l'entrée de
down_proj (valeurs >1000-1400 sur Llama-2-7B). L'intuition : la même
rotation qui aide déjà l'INT4-AWQ (mode `auto` existant,
`TensorRouter.wants_hadamard`) devrait aider le NVFP4.

## Ce qui a été mesuré

`tests/test_hadamard_downproj_nvfp4.py`, sur le motif déjà validé par le
témoin existant (`test_quant.py::test_hadamard_helps_int4_on_outlier_channels`,
colonnes d'entrée ×8 tous les 64 canaux, pas un motif choisi pour faire
mentir la piste) :

* **INT4-AWQ (groupe 128)** : la rotation aide toujours, +7,3 dB — confirmé.
* **NVFP4 (bloc 16)** : la rotation **dégrade** le SNR, −0,38 dB, mesuré
  stable sur six graines (écart-type < 0,02 dB).

À un motif plus extrême (colonnes ×1000), la rotation devient destructrice
pour **les deux** formats (NVFP4 −2,2 dB, INT4-AWQ 38,9 → 0,4 dB) : régime
non retenu, l'effet de magnitude y écrase l'effet de structure de bloc qui
intéresse cette piste.

## Explication qualitative

Le bloc de 16 du NVFP4 porte déjà une échelle E4M3 propre à *chaque* groupe
de 16 colonnes : un outlier confiné à 16 colonnes ne pénalise que cette
échelle locale, à coût quasi nul pour le reste du tenseur. Le groupe de 128
de l'INT4-AWQ est assez large pour qu'étaler un outlier sur ses 128
colonnes reste une amélioration nette. Mais la rotation de Hadamard telle
qu'implémentée (`largest_pow2_divisor` prend le plus grand diviseur de la
dimension **entière**, pas du bloc de quantification) étale l'énergie de
l'outlier sur **tout le tenseur** — perdant la localisation que le bloc de
16 offrait déjà gratuitement, sans la compensation que le groupe de 128
offre à l'INT4-AWQ.

## Décision

**Aucune option livrée.** Ajouter `hadamard_downproj_nvfp4` sur la foi de
la piste 2 aurait expédié une fonctionnalité que son propre test unitaire
réfute — contraire à la règle 9 (« une optimisation qui change la sortie
est un bogue ») et à la règle 10 (« un échec est un résultat »)
de `REGLES.md`. Le code de `convert.py`/`cli.py` n'est pas modifié ;
seul le test de falsification reste, pour qu'une session future ne
retente pas la même idée sans preuve.

## Ce qui resterait à tester, hors de ce chantier

DuQuant (arXiv:2406.01721) propose une rotation **et permutation ciblées**,
pas une rotation aveugle sur tout le tenseur — ce test ne le couvre pas.
Une rotation confinée au bloc de 16 lui-même (au lieu du plus grand
diviseur de la dimension entière) pourrait aussi se comporter différemment
— non mesurée ici, changerait `hadamard_transform` elle-même, hors
périmètre d'une option de conversion.
