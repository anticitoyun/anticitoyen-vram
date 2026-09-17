# Verdict — 1a, part de blocs divergents sans code de magnitude 6

instrument : outils/verdict-modelopt-bloc-amax.py (a sec, CUDA_VISIBLE_DEVICES="")
commit : 441eb27 (branche poste1-11)
regime : CPU, aucun GPU touche
scelle : poste7-convertisseur-formats-16-09.md §6 — ≈0 % -> (b) instrument
  desaligne ; > 20 % -> (a) echelle cherchee, PPL <= actuelle - 0,004
mesure : 39242232/39242232 blocs divergents SANS code de magnitude 6 (100,00 %)
verdict : (a) — ModelOpt calibre son echelle de bloc, pas un amax/6 brut

## Detail par tenseur

| tenseur | blocs divergents | sans code=6 |
|---|---|---|
| lm_head.weight | 33702083/79462400 | 100,00 % |
| gate_proj (couche 0) | 2771424/5570560 | 100,00 % |
| down_proj (couche 0) | 2768725/5570560 | 100,00 % |

## Controle (temoin qui doit reagir, REGLES §4)

Sur le meme tenseur (down_proj couche 0), blocs CONCORDANTS (notre
requant == ModelOpt) : 2801835/2801835 portent un code de magnitude 6
(100 %), aucun sans. Separation nette 100/0 entre concordant et
divergent — pas un artefact de decodage des codes (bit 0x07, ordre
pack_e2m1) : les concordants la valident, les divergents la refutent.

Exemples de blocs divergents (echelle e4m3 ModelOpt vs nous, meme
tenseur) : 14,0↔9,0 · 16,0↔11,0 · 20,0↔13,0 · 14,0↔9,0 · 20,0↔13,0 —
ModelOpt choisit systematiquement une echelle PLUS GRANDE que
amax_bloc/6, ce qui empeche mecaniquement tout code d'atteindre la
magnitude 6. Coherent avec une echelle protegee contre l'aberrant
(clipping), pas un amax brut.

## Consequence

Le trou 1 reste un vrai chantier pour poste4 (passage direct, §6 1b) :
pas de raccourci --no-awq + garde manifeste. Ajouter la recherche
d'echelle de bloc dans quantize_nvfp4 (nvfp4.py:216), scelle PPL <=
conversion actuelle - 0,004 sur 3 tranches (deja pose par 1b).
