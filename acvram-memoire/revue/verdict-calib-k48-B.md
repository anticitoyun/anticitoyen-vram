# Verdict — bras B (calibration mixte FR/EN/code ≥16k jetons) : conversion faite, NOMINAL, PPL à poste3

poste2, 17/09. Suite du bras A (`verdict-calib-k48-A.md`) : bras B isole
l'effet de la LANGUE à taille égale (16 384 jetons dans les deux bras).

## Corpus, sha256 publié avant la conversion

Mixte, 1/3 français accentué + 1/3 anglais (même source que A) + 1/3 code
réel du dépôt `acvram/` :
- Français : *Le Comte de Monte-Cristo*, tome I (Alexandre Dumas, Project
  Gutenberg #17989), 260 000 caractères, texte accentué vérifié
  (19 088 caractères accentués sur l'extrait complet).
- Anglais : *Pride and Prejudice* (Gutenberg #1342), 260 000 caractères —
  même source que le bras A.
- Code : 12 fichiers `.py` réels du dépôt `acvram/`, 323 558 caractères
  (aucun `revue/*.md`, REGLES §3).

```
sha256 (bras-B-mixte.txt) : 6a96eda740147ac2687c91ba9fbe63321b4a13d13a324e41b31dc9e24aa1adb7
```

Vérifié avec le tokenizer GLM et `load_calib_ids(..., n_seqs=32,
seq_len=512)` avant la conversion : **32 séquences × 512 jetons = 16 384
jetons exacts**, identique au bras A. Spot-check des frontières de
séquences (0, 10, 11, 21, 22, 31) confirme la progression FR → EN → code
dans l'ordre attendu.

## En-tête de mesure

`--quant-device cpu`, à sec (pas de carte réservée). `acvram.__file__`
vérifié = `.../anticitoyen-vram/acvram/__init__.py`, commit `f78a093`.
Source `GLM-4.7-Flash-bf16`, sortie `GLM-4.7-Flash-srcbf16-nvfp4-k48-
calibB`. Mêmes options que A sauf `--calib-file`.

## Résultat de conversion

```
tenseurs 9751, sortie 18,4 Gio (x3,16), SNR sortie moyen 21,4 dB
201 experts sans statistique de calibration (identique au bras A)
durée 4935,3 s (≈ 82,3 min, CPU)
```

Même nombre d'experts sans statistique que le bras A (201, contre 558 sur
`-k48`/230 jetons) : cohérent avec l'hypothèse de poste7 — c'est la TAILLE
du corpus qui détermine combien d'experts sont vus, pas sa composition
linguistique. SNR moyen quasi identique (21,4 vs 21,5 dB sur A).

## Régime

`acvram serve --regime` : **NOMINAL**, graphes=on, **10 godets réellement
capturés**, couches_exilées=0/47, experts_exilés=0/2944, piles_ok=True.

## Suite

Rendu à poste3 pour la PPL (privé + public, géométrique ET médiane, préfixe
`[gMASK]<sop>`, scellé § 3 de `poste7-hadamard-verdict-17-09.md` : comparer
à l'écart du bras A pour trancher taille vs langue).

**Pause générale du circuit demandée par l'utilisateur (relayée par
chef) après ce bras** : le bras C (optionnel, `-sansawq` + rotation
Hadamard 512) n'est PAS lancé.
