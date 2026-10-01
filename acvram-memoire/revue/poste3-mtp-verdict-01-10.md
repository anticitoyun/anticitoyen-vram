# Verdict — MTP contre n-gram contre aucun, acvram-qwen3.8-27b-nvfp4 (poste3, 01/10)

Scellé : `acvram-memoire/revue/poste3-mtp-scelle-30-09.md` (prédiction, seuils, protocole ABCCBA,
§1-§14 — diagnostic et correctifs de la journée, décalage de 2 jetons et préfill en plusieurs
passes qui écrasait `_mtp_prefill` au lieu de l'étendre).

## Bras court (invite ~78 jetons, `--max-model-len 4096`), commit fd3121e81

12 passes ABCCBA sous `carte.sh`, conditions de validité de la règle 9 tenues :
**IDENTITE B=A : OUI, IDENTITE C=A : OUI** (jetons gloutons identiques au bit, température 0,
même invite) — la spéculation, quand elle produit un résultat, ne change jamais la sortie servie.

Médianes `jetons_s` : A(none)=79,5 · B(mtp)=59,3 · C(ngram)=80,2-80,3.

**B/A ≈ 0,746 — RÉFUTÉE** (seuil du scellé §4 : B/A < 0,95 → hypothèse arXiv 2609.35188
réfutée pour notre implémentation). La marge est large (0,746 contre le seuil 0,95), pas une
frontière à discuter.

MTP s'engage réellement cette fois (correctifs du jour : `_nourrir_mtp_prefill` dans
`_consommer`, garde `st.length==0`) : `accepted_tokens=689`, `proposed_tokens=3220` sur
l'ensemble du bras B, **acceptance ≈ 21,4 %** — à comparer aux 35,8 % de l'essai court isolé
avant l'ABCCBA (1 serveur, 3 requêtes) : l'acceptation dépend de l'invite et/ou du nombre de
requêtes servies, pas une constante du mécanisme. C (ngram) : `proposed=128`,
`accepted=92`, acceptance ≈ 71,9 %.

**Cause du débit en baisse malgré l'acceptation réelle : COMPATIBLE avec la note
`docs/ARCHITECTURE.md` (chaque jeton brouillon traverse `lm_head` en entier, hors graphe CUDA,
plein coût de lancement par brouillon) — PAS MESURÉE ici.** Un `nsys` dédié serait nécessaire
pour l'établir ; non fait aujourd'hui (hors budget, carte reprise par la suite).

Régime : `[acvram] régime NOMINAL — … speculation=mtp(on,lot_max=2)` (B) / `ngram(on,lot_max=2)`
(C) / `off` (A), pris du journal serveur à chaque passe (pas du client, §9 du scellé).

## Bras long (invite 16 440 jetons, `--max-model-len 20480`) — NON MESURÉ, refusé par sa garde

Passe A : `jetons_s=72,4`. Passe B : `/metrics` après coup donne `proposed_tokens=0` →
**REFUS automatique** par la condition de validité de `prise-abccba.sh` (règle 9), avant la passe
C. Carte rendue proprement (158 s tenue, confirmée au journal `carte.sh`), pause e50.2 retirée
par le trap (filet de sécurité ajouté le jour même).

**Cause identifiée, pas mesurée en vrai** : à 16 440 jetons (très au-delà du seuil
`_pas_insta`=256), le préfill se coupe en plusieurs passes (frontière d'instantané,
`runner.py` ~1930-1948). Avant le correctif (2) de ce commit, `model.py:_sortie` écrivait
`_mtp_prefill` par AFFECTATION à chaque appel — un préfill en plusieurs passes ne gardait que la
DERNIÈRE, trop courte pour amorcer la tête sur l'invite entière. Correctif à sec (sans carte) :
concatène quand la passe continue la même séquence au point où la précédente s'est arrêtée
(même seq.id, position de départ = taille déjà accumulée), repart à zéro sinon. Jouet bout en
bout (préfill en deux passes contiguës, amorçage réussi directement) : 4 cas, cassent sans le
correctif, 29/29 au total, aucune régression (62 passed/5 skipped sur les suites spéculatives +
mla_glue + d19).

**Pas rejoué sur la carte** : budget du jour épuisé (+4,3), la réfutation du bras court est déjà
nette et large. Bead ouvert pour la suite : rejouer le bras long MTP seulement si le coût par
jeton brouillon baisse (lm_head ramené dans le graphe CUDA) — sinon la conclusion du bras court
(réfutée) vaudrait probablement aussi en long, et la carte serait dépensée pour confirmer ce
qu'on sait déjà.

## Conclusion

**Hypothèse arXiv 2609.35188 RÉFUTÉE pour acvram-qwen3.8-27b-nvfp4, en court contexte, avec
marge large (B/A=0,746).** MTP accepte réellement des jetons (21,4-35,8 % selon la charge) mais
le débit net baisse — compatible avec le coût `lm_head` hors graphe par brouillon décrit dans
`docs/ARCHITECTURE.md`, cause non mesurée directement (pas de `nsys` aujourd'hui). Pas de défaut
`--speculative mtp` à activer. Le bras long reste à mesurer, mais seulement si le correctif
lm_head-dans-le-graphe est fait d'abord (sinon la carte confirmerait probablement la même
réfutation sans rien apprendre de plus).

verdict: acvram-memoire/revue/poste3-mtp-verdict-01-10.md — bras court ABCCBA, identités tenues (règle 9), B/A=0,746 RÉFUTÉE (seuil 0,95, marge large), acceptance B=21,4%/C=71,9%, cause (lm_head hors graphe) compatible avec ARCHITECTURE.md mais non mesurée ; bras long refusé par sa garde (proposed_tokens=0, préfill multi-passes écrasait _mtp_prefill) — correctif à sec fait (concaténation si même séquence contiguë), pas rejoué (budget), bead ouvert pour rejeu conditionné au lm_head-dans-le-graphe
