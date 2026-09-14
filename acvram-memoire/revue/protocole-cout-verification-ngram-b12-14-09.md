# Protocole + prédiction — coût de vérification n-gram à b=12

poste3, 14/09/2026. Suite à chef : « (2) le coût de vérification n-gram à
b=12 (dernière inconnue de A4) : ACVRAM avec ngram actif vs inactif, 12
séquences, t/s + J — c'est un défaut du serveur qui peut coûter 31 %. »
Ferme la question laissée ouverte par
[`chantier-speculation.md`](chantier-speculation.md) §2 (« LE LOT FAIT DEJA
MIEUX — à BQ=12 la carte est remplie sans spéculer », jamais mesuré) et par
`verdict-taux-ngram-code-13-09.md` (taux réel 1,6137, régime Coder-30B —
**taux différent, sans rapport avec le 1,105-1,163 de `chantier-speculation.md`
§2, mesuré sur un modèle EXL3 tiers, régime non apparié**, règle 6).

## Hypothèse

À b=12, le lot occupe déjà la carte à largeur 1 par séquence. Activer
`ngram` fait vérifier jusqu'à `spec_k+1` positions par séquence et par pas —
la largeur réelle du noyau d'attention/GEMV monte, alors que la carte n'a
plus de marge libre pour l'absorber gratuitement (contrairement à b=1, où
la vérification remplit un noyau autrement sous-occupé). **Je prédis que
`ngram` actif est PLUS LENT que `ngram` inactif à b=12**, pas plus rapide —
inversion du signe qu'on aurait à b=1.

## Montage

    modèle      Qwen3-Coder-30B-A3B-nvfp4 (régime des campagnes d'poste1/
                poste2 sur le routage d'experts — même famille que le taux
                1,6137 mesuré)
    slots       12 (ACVRAM_HYBRID_SLOTS=12), lot RÉELLEMENT concurrent
                (add_request + step(), pas 12 generate() séquentiels —
                piège du 13/09, bead x0s)
    invites     12 des 20 invites de code réelles de
                `scratchpad/generer-sorties-ngram-coder30b.py` (pas de
                jetons synthétiques : le taux d'acceptation dépend du
                contenu, un n-gramme sur du bruit ne représente rien)
    n_jetons    128 par séquence (assez pour amortir le préremplissage,
                court pour rester dans une fenêtre de carte raisonnable)
    bras A      `speculator=None` (ngram inactif)
    bras B      `speculator=NGramProposer()`, `spec_k=4` (défaut CLI,
                cf. `acvram/cli.py:828` `--speculative ngram`)
    mesure      t/s agrégé (jetons décodés / mur), énergie via
                `outils/gpu/mesure/energie.py` (`Energie` + `repos(30.0)`
                avant/après chaque bras — pas de mesure d'énergie sous 30 s
                de repos, cf. correctif du 14/09)
    carte       exclusive, verrou `carte.sh` tenu du chargement à la fin,
                `ACVRAM_TYPE=mesure`, pas d'interlaçage (consigne chef du
                14/09 relayée par poste4)

## Seuil de décision

    t/s(B) / t/s(A) ≥ 1,0     ngram ne coûte rien à b=12 : question fermée,
                              rien à changer au défaut
    0,90 ≤ ratio < 1,0        coût réel mais mineur : documenter, ne pas
                              changer le défaut serveur
    ratio < 0,90              coût significatif : le défaut `auto` (qui
                              active ngram sans tête MTP) est un DÉFAUT DU
                              SERVEUR au sens de chef — proposer de le
                              désactiver par défaut à b≥N (N à déterminer)

## Ce qui réfuterait l'hypothèse

Si `ratio ≥ 1,0` (B au moins aussi rapide que A), l'hypothèse d'occupation
saturée qui ne peut pas absorber la vérification sans coût est fausse pour
ce régime — je le rapporterai tel quel, sans reformuler la prédiction après
coup (règle du 8/09, `estimation-figee-modifiee-apres-coup.md`).
