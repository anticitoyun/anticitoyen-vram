# poste7 — Calibration tranchée : le privé n'y est pas sensible ; le « 16 × 128 » est un défaut de manifeste, pas de conversion ; chantier clos (17/09)

Entrée : `verdict-bras-calibration-17-09` (poste3, 804fbc9, main 3e05155). Scellés de `poste7-hadamard-verdict-17-09` § 3 : A ≤ 0,020 → tenu (−0,013 géo, −0,003 méd) ; B ≤ 0,020 → réfuté (+0,019). Juge géométrique (§ 4) : inchangé.

## 1. Le piège n'existe pas : A a bien lu 16 384 jetons

`calib_seqs: 16` et `calib_tokens: 128` du manifeste sont les **défauts de classe** de `ConvertOptions` (`convert.py:44-45`), écrits par `asdict(opts)` (`convert.py:1075`), **jamais posés depuis les arguments et jamais lus** : `load_calib_ids` (`collect.py:67-90`) reçoit `args.calib_seqs` / `args.calib_len` directement (`cli.py:503-504`, défauts CLI 16 / 512 en `cli.py:837-838`). poste2 a passé 32 / 512 et vérifié 32 × 512 = 16 384 avant de convertir (`verdict-calib-k48-A` l. 18-20) ; 201 experts sans statistique contre 558 le corrobore. C'est le cas « origine d'une valeur » de MECANISMES : un défaut de classe qui a l'apparence d'une lecture. Contrôle qui peut me donner tort : la ligne `calibration sur N sequences (M jetons)` (`cli.py:505`) du journal de conversion de poste2 — si elle dit 2 048, (a) s'impose et cette note tombe.

## 2. Ce que les cinq convertis disent

| converti | géo privé | ce qui change vs défaut |
|---|---|---|
| `-k48` défaut | 1,0156 | — (230 jetons, six phrases) |
| `-k48-calibA` | 1,0150 | 16 384 jetons, prose anglaise |
| `-sansawq` | 1,0166 | plus d'AWQ du tout |
| `-k48-calibB` | 1,0244 | 1/3 FR littéraire + 1/3 code |
| `-hadamard512` | 1,0289 | rotation (clos) |

Bruit ± 0,004 : taille ×70, texte, AWQ ou RTN — **le privé ne bouge pas**. L'inversion privé/public était un artefact du **public** : les six phrases de `collect.py:48-64` (registre wiki) valaient 1,004 à wikitext ; A, disjoint, donne 1,028. Le « candidat RTN experts g16 » d'ETAT n'explique pas une inversion qui n'existe plus : 1,015-1,017 est le plancher du format sur ce corpus, et il se publie. B : français littéraire et code nuisent (+0,009, 27/36 fenêtres) — fermé, cause non cherchée.

## 3. Décision : (b), chantier clos, avec deux correctifs à sec

* Colonne GLM acvram de la table à cinq = **`-k48-calibA`** : privé 1,015 géo classé, public 1,028, W4A16, 21,0 ms — calibration disjointe et sha256 publié (REGLES § 3). Le défaut classe aussi (1,016) ; A l'est sans le bonus wikitext.
* Corpus par défaut : `bras-A-anglais.txt` (Gutenberg #1342, domaine public, 738 Ko) entre dans le dépôt à la place de `DEFAULT_CALIB_TEXT`, avec `--calib-seqs 32 --calib-len 512` par défaut. Prédiction : le prochain `-k48` nominal reproduit A à ± 0,004 sur les deux corpus ; se vérifie sur le prochain converti GLM demandé de toute façon, pas de passe dédiée.
* Manifeste : `calib_seqs` / `calib_tokens` disparaissent ou portent ce que `load_calib_ids` a rendu (séquences, jetons lus) dans `calib_source` ; le test casse si le défaut de classe revient.
* ETAT : « acvram GLM réfuté 1,029 » (médiane, `poste7-comparatif` § 11) est remplacé par « classé 1,015 géo, médiane 1,024 en colonne » — juge scellé le 17/09 avant ces mesures, pas rouvert. Coder (1,027 géo) ne se relit pas.
* Pas de bras A-16k (il existe), pas de bras C, Hadamard reste clos.

## Ordre

1. chef : ETAT — chantier calibration CLOS ; cause de l'inversion = six phrases favorisant wikitext (ligne « candidat RTN » retirée) ; colonne GLM acvram = `-k48-calibA` 1,015 géo ; porter à l'utilisateur : le converti nominal 0.6.x classe déjà sur le privé, le correctif de corpus par défaut ne change que la colonne publique (1,004 → ≈ 1,028).
2. poste2 (à sec, 1 ligne) : citer dans `verdict-calib-k48-A` la ligne `calibration sur … jetons` de son journal de conversion ; puis EXL3 Coder + GLM (`poste7-comparatif` § 7), carte libre.
3. poste4 (à sec, 1 h, un commit) : § 3 — corpus par défaut bras-A + défauts 32 / 512 (`cli.py:837-838`) + manifeste = compte rendu par `load_calib_ids`, test qui casse si le défaut revient ; ensuite 1b passage direct (porte du comparatif).
4. poste3 : rien de plus sur la calibration ; prochaine passe = PPL des EXL3 quand poste2 les rend.
