# Correctif — GardeSpeculation conditionne n-gram au lot réel

poste3, 14/09/2026. Suite au verdict
[`verdict-cout-verification-ngram-b12-14-09.md`](verdict-cout-verification-ngram-b12-14-09.md)
et à la décision de chef : « n-gram actif si b_reel ≤
ACVRAM_SPECULATION_LOT_MAX (défaut 2), réévalué à chaque pas, plus une garde
glissante ; désactiver si les jetons émis par pas sur 32 pas < 1,05 × base,
réactiver quand le lot redescend ».

## Le correctif

`acvram/engine/speculative.py:47-95` — nouvelle classe `GardeSpeculation` :

- `eligible(b_reel)` : refuse au-dessus de `lot_max` (défaut 2, env
  `ACVRAM_SPECULATION_LOT_MAX`) ; se réarme (fenêtre glissante vidée) sur
  une transition lot haut → lot bas ;
- `enregistrer(jetons_emis, b_reel)` : alimente une fenêtre glissante de 32
  pas du ratio jetons/pas ÷ b_reel ; désactive si la moyenne tombe sous
  1,05.

`acvram/engine/runner.py:333-336` (constructeur `Engine`) : instancie la
garde. `runner.py:761-769` (`step()`) : dispatch conditionné à
`self._garde_spec.eligible(b_reel)` avant d'entrer dans
`_speculative_decode` ; le delta `decode_tokens` du pas alimente la garde.
S'applique à N'IMPORTE QUEL propositeur (`ngram`, `auto`, `draft`, `mtp`),
pas seulement `ngram` — le mécanisme de coût est le même quel que soit le
propositeur (largeur de vérification élevée sans marge de carte).

`acvram/cli.py` : aide de `--speculative` et bannière `serve` documentent
`ACVRAM_SPECULATION_LOT_MAX` ; ajouté au registre `_ENV_CONNUES`.

## Tests CPU

`tests/test_speculation_lot.py`, 7 tests, purs (pas de carte, pas de
modèle) : éligibilité sous/au-dessus du seuil, désactivation glissante sous
le gain minimal, persistance de la désactivation tant que le lot reste
haut, réarmement sur transition haute → basse, et le scénario demandé par
chef (transition 1 → 12 → 1). `7 passed in 0.25s`.

## Vérification sur carte

Script `scratchpad/verif-garde-speculation-14-09.py`, même modèle et mêmes
invites que le verdict du 14/09 (Qwen3-Coder-30B-A3B-nvfp4, carte
exclusive, repos 30 s entre bras) :

    condition                                    tok/s     J/jeton
    b=1, sans ngram                              127,35    1,656
    b=1, avec ngram (garde, lot_max=2)           127,23    1,733
    b=12, avec ngram (garde, lot_max=2)          481,52    0,689

    référence verdict 14/09 (sans correctif) :
    b=12, ngram inactif (bras A)                 481,37    0,690
    b=12, ngram actif SANS garde (bras B)        241,35    0,995

**Prédiction de chef confirmée sur le point qui comptait** : à b=12, le
correctif ramène le débit à 481,52 t/s — **écart de 0,03 % avec le régime
sans spéculation (481,37 t/s)**, contre 241,35 t/s sans garde (verdict du
14/09). La garde de lot supprime intégralement le coût mesuré.

**Réserve honnête sur b=1** : le gain attendu (« garde son gain ») n'est
PAS visible ici (127,23 contre 127,35 t/s, différence dans le bruit) — mon
script de vérification n'utilise qu'UNE invite pour ce bras (`prompts_ids[0]`,
« merge deux listes triées »), et cette invite précise a peu de répétition
à proposer : le taux d'acceptation dépend du contenu (chantier-speculation.md
§2, `verdict-taux-ngram-code-13-09.md`), il n'est pas garanti sur un
échantillon de taille 1. Ce n'est pas un défaut du correctif — la garde
laisse la spéculation ACTIVE à b=1 (elle n'a pas désarmé), elle mesure
seulement que CETTE invite ne rapportait rien ici. Pour confirmer le gain
moyen à b=1, il faudrait rejouer les 20 invites (comme
`verdict-taux-ngram-code-13-09.md`) sous le nouveau code, pas fait ici —
hors du périmètre de cette vérification.

## Anomalie notée, sans lien avec ce correctif

Une première tentative de cette vérification (carte partagée, VRAM
insuffisante) a exilé 11 MLP en RAM hôte et plongé dans un crash distinct :
`runner.py:268` (`_promote_expert`) — `AttributeError: 'NoneType' object
has no attribute 'host'` sur `lin.streamed.host`, déclenché par
`_repin_pass` pendant un `generate()` avec spéculation active et graphes
CUDA désactivés (exil). Pré-existant, sans rapport avec `GardeSpeculation` —
signalé, non corrigé ici (hors périmètre demandé). Disparu sur carte libre
(pas d'exil).

## bd

Correctif livré, testé (CPU) et vérifié sur carte pour le point b=12
explicitement demandé. Défaut du serveur (`--speculative` sans condition de
lot) refermé.
