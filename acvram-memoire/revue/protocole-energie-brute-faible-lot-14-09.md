# Protocole + prédiction — énergie BRUTE b=1-4, acvram vs vLLM

poste3, 14/09/2026. Suite à chef, mesure 2a de poste7 : « énergie BRUTE
b=1-4 Coder-30B, acvram vs vLLM, mêmes séquences, J/jeton total (repos
compris : nous 17 W au repos, vLLM 64 W) — c'est le créneau à
revendiquer. »

## Ce que « brute » change par rapport au duel du 14/09

Le duel décodage b=12 de poste2 (`audit-a2-duel-vllm-14-09.md`) rapporte du
**net** : `joules_net = brut - repos_moyen × durée` — soustrait la
puissance de repos du moteur AVANT de comparer, ce qui masque
précisément l'écart de repos entre moteurs. Si acvram tourne à 17 W au
repos contre 64 W pour vLLM (chiffre de chef, non revérifié ici), le
NET annule cet écart par construction — **la métrique qui compte pour un
service à faible trafic (des requêtes espacées, le GPU repasse au repos
entre deux) est le BRUT total : temps × puissance réellement consommée,
repos inclus**, pas seulement le travail actif.

## Prédiction

À b=12 (duel existant), le travail actif domine largement le repos : le
brut et le net convergent, d'où le classement vLLM > acvram inchangé
(×2,11 débit, ×2,97 efficacité NETTE). **À b=1-4, la part de temps où le
moteur est concrètement en train de calculer (par opposition à attendre le
prochain jeton, la synchronisation hôte, l'ordonnanceur) est plus faible en
proportion** — je m'attends à ce que l'écart de repos (17 contre 64 W)
pèse assez pour RÉDUIRE l'écart d'efficacité brute par rapport au ×2,97
net à b=12, potentiellement au point de RAPPROCHER ou d'INVERSER le
classement en J/jeton BRUT à b=1. Seuil de décision :

    ratio brut(acvram)/brut(vLLM) < ratio net(acvram)/net(vLLM) à b=12 (2,97)
        → confirmé : le brut resserre l'écart à faible lot, créneau réel
    ratio brut ≈ ratio net (± 10 %)
        → réfuté : le repos ne pèse pas assez pour changer la conclusion
    acvram bat vLLM en J/jeton BRUT à un b quelconque de 1 à 4
        → le créneau existe littéralement, pas seulement en tendance

## Montage

    modèle     Qwen3-Coder-30B-A3B-nvfp4 (mêmes séquences aux deux moteurs :
               invites synthétiques identiques à `banc-horloge-decodage.py`
               / `banc_decode_vllm.py`, même fonction `invite(k,n)`)
    b          1, 2, 3, 4 (boucle)
    ctx        2048, N_JETONS 200, EOS neutralisé aux deux moteurs
               (`ignore_eos=True` côté vLLM déjà en place ; `eng._eos =
               set()` côté acvram, cohérent avec la leçon du 14/09 —
               sinon un des deux moteurs pourrait finir plus tôt à petit
               lot et fausser la comparaison)
    énergie    `outils/gpu/mesure/energie.py`, PROCESSUS SÉPARÉS (acvram
               puis vLLM, jamais concurrents), carte exclusive tenue tout
               du long (`outils/carte.sh`, `ACVRAM_TYPE=mesure`)
    métrique   BRUT = `e.joules` tel quel (PAS de soustraction du repos) ÷
               jetons décodés ; le repos mesuré (`base.moyenne`) est publié
               À CÔTÉ, pour montrer l'écart 17/64 W plutôt que l'effacer
    code       `main` d'aujourd'hui (anticitoyen-vram), comme pour le
               chiffre officiel du point précédent

## Réserve

Le chiffre « 17 W / 64 W » vient de chef, pas revérifié dans ce
protocole — la mesure de repos AVANT chaque bras (incluse dans le script)
sert justement à le confirmer ou le corriger en même temps que le brut.
