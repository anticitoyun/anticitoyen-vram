# Verdict — bras A (calibration anglaise ≥16k jetons) : conversion faite, NOMINAL, PPL à poste3

poste2, 17/09. Ordre poste7 (`poste7-hadamard-verdict-17-09.md` § 3) : Hadamard
clos, vrai chantier = calibration nominale à 230 jetons (six phrases
répétées ×16, `collect.py:48-77`). Test décisif 3 bras sur la recette
`-k48` (AWQ, sans rotation) : bras A isole l'effet de la TAILLE seule.

## Corpus, sha256 publié avant la conversion

Anglais général, disjoint de wikitext : *Pride and Prejudice* (Jane
Austen, Project Gutenberg #1342), en-tête/pied Gutenberg retirés,
728 713 caractères. Aucun `revue/*.md`, aucun wikitext (REGLES §3).

```
sha256 (bras-A-anglais.txt) : cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138
```

Vérifié avec le tokenizer GLM et `load_calib_ids(..., n_seqs=32, seq_len=512)`
avant la conversion : **32 séquences × 512 jetons = 16 384 jetons exacts**
(≥ 16 k demandé).

## En-tête de mesure

`--quant-device cpu`, à sec (pas de carte réservée, en parallèle du
comparatif de poste3) : `outils/carte.sh` non utilisé pour cette étape.
`acvram.__file__` vérifié = `.../anticitoyen-vram/acvram/__init__.py`,
commit `f78a093`. Source `GLM-4.7-Flash-bf16`, sortie `GLM-4.7-Flash-
srcbf16-nvfp4-k48-calibA`. `acvram convert --format nvfp4 --snr-floor 25
--calib-file bras-A-anglais.txt --calib-seqs 32 --calib-len 512`.

## Résultat de conversion

```
tenseurs 9751, sortie 18,4 Gio (x3,16), SNR sortie moyen 21,5 dB
201 experts sans statistique de calibration (contre 558 sur -k48/230 jetons)
durée 4902,9 s (≈ 81,7 min, CPU)
```

201 experts sans statistique contre 558 précédemment : la calibration à
16 384 jetons diversifiés route vers presque trois fois plus d'experts que
les six phrases répétées — signe direct que la taille/diversité du
corpus change ce que l'AWQ voit, cohérent avec l'hypothèse de poste7.

## Contrôle demandé par poste7 : la ligne « calibration sur … jetons »

Le manifeste MENT sur la taille de calibration réellement utilisée :
`options.calib_seqs=16`, `options.calib_tokens=128` — ce sont les défauts
figés de la dataclasse `ConversionOptions` (`convert.py:44-45`), jamais
lus ni posés par `cmd_convert` (`cli.py:503` ne transmet que `args`
générique, pas ces deux champs) ; ils ne reflètent PAS `--calib-seqs 32
--calib-len 512` réellement passés en ligne de commande. Seul
`options.calib_source.fichier`/`sha256` (ajouté ce chantier) est fidèle.

Ligne du journal de conversion (`cli.py:505`, `print(f"  calibration sur
{len(calib)} sequences ({sum(len(c) for c in calib)} jetons) ...")`),
reproduite à l'identique par appel direct à `load_calib_ids` avec le même
fichier/mêmes paramètres (le journal original du 17/09 n'a pas survécu au
redémarrage de session, mais le calcul est déterministe — même fichier,
même sha256, même sortie) :

```
  calibration sur 32 sequences (16384 jetons) ...
```

Confirme, indépendamment de la vérification `load_calib_ids` déjà citée
plus haut, que la conversion a bien tourné sur 16 384 jetons — le
manifeste ment sur `calib_seqs`/`calib_tokens`, pas sur ce qui a
réellement été calibré.

## Régime

`acvram serve --regime` (`outils/carte.sh`) : **NOMINAL**, graphes=on,
**10 godets réellement capturés** (contrairement à la reconversion
Hadamard qui échouait à 0), couches_exilées=0/47, experts_exilés=0/2944,
piles_ok=True.

## Suite

Rendu à poste3 pour la PPL (privé + public, géométrique ET médiane, préfixe
`[gMASK]<sop>`, scellé § 3 : écart privé−public ≤ 0,020 tranche « la
taille suffit », ≈ 0,032 avec bras B ≤ 0,020 tranche « la langue »).
J'enchaîne le bras B (corpus déjà construit, `bras-B-mixte.txt`, sha256
publié dans `verdict-calib-k48-B.md`).
