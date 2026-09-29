# 3zo — sortir-version étape 7 : doctor Flatpak sous carte.sh, refus 67 à sec (poste6, 29/09, branche poste6-3zo, sans GPU)

Ordre chef (bd anticitoyen-vram-3zo, P2). Défaut vu à la sortie 0.7.13 : l'étape 7 lançait `outils/verifier-release.sh $V`
hors carte.sh ; le shell du chef porte `CUDA_VISIBLE_DEVICES=""` (à sec, REGLES § 1) → le bac à sable Flatpak ne voyait aucun
GPU → « FAUX flatpak : Marlin non chargé depuis le précompilé » (rc 71) sur une release bonne (rejouée à la main sous carte.sh
avec `--flatpak-doctor` : TENU). Un défaut d'instrument lu comme un défaut de release — REGLES § 4 bis.

## Correctif
* `outils/sortir-version.sh` étape 7 : deux bras. `verifier-release.sh $V --flatpak-installer` hors carte (≈ 3 Go tirés,
  tous les autres contrôles), puis `ACVRAM_NOM=sortir-version-$V outils/carte.sh outils/verifier-release.sh $V --flatpak-doctor`
  — carte.sh rétablit `CUDA_VISIBLE_DEVICES=$ACVRAM_CARTE` pour la commande (carte.sh:259). `--simule` trace les deux lignes.
  Les deux refus restent en 71, message distinct (« bras installation » / « bras doctor sous carte.sh »).
* `outils/verifier-release.sh` : **code 67**, avant tout téléchargement, si `CUDA_VISIBLE_DEVICES` est défini vide alors que
  le doctor doit tourner (défaut ou `--flatpak-doctor`) ; `--sans-flatpak` et `--flatpak-installer` passent à sec. Le message
  nomme la cause et le geste (carte.sh). Codes existants : 64 usage, 65 lien mort, 66 version Flatpak, 1 FAUX.

## Contrôle (prédit avant : test rouge sans correctif, vert avec ; 259/281/285/285b/295 inchangés)
`tests/test_sortir_version_3zo.py`, 8 tests, témoins carte.sh / verifier-release.sh qui journalisent leurs arguments et
`CUDA_VISIBLE_DEVICES` (aucune prise de carte réelle) :
* `--simule` montre `outils/carte.sh outils/verifier-release.sh vX --flatpak-doctor` ;
* `--depuis 7` avec `CUDA_VISIBLE_DEVICES=""` : journal = installer `CVD=` (vide), carte.sh, doctor `CVD=0` ;
* défaut du bras doctor → 71 ; verifier-release à sec → 67 (défaut et `--flatpak-doctor`), pas 67 pour `--sans-flatpak`,
  `--flatpak-installer`, ni carte visible.

| état | 3zo | voisins (281, 285, 285b, 295, 259) |
|---|---|---|
| sans correctif (stash) | 5 échecs / 3 passés (les 3 : contre-épreuves) | — |
| avec correctif | 8/8 | 37/37 |

Non prouvé ici : une vraie sortie (étape 7 réelle sur une release publiée) — à voir à la 0.7.14 ; le doctor réel sous
carte.sh l'a déjà été à la main sur la 0.7.13 (chef).
