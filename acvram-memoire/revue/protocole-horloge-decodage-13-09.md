# Protocole + prédiction — horloge SM verrouillée au décodage, Coder-30B b=12

poste3, 13/09/2026, à sec (lecture seule + calcul, sans carte). Suite à poste7,
[`poste7-veille-internet-13-09.md`](poste7-veille-internet-13-09.md) §1 :

> Le bridage 400 W est probablement inerte au décodage, et l'horloge SM est
> le vrai levier (arXiv 2605.11999) : sur H200, le décodage tire 137-300 W
> sur 700 W, aucun cap ne se déclenche ; le verrouillage d'horloge SM
> « Pareto-domine » le cap et récupère jusqu'à 32 % de l'énergie de décodage
> à débit quasi égal. Confirmé indépendamment par 2501.08219 : décodage =
> 77-91 % du temps, insensible à la fréquence, 2842 → 180 MHz = −42 %
> d'énergie pour +1-6 % de latence.

## 1. Ce que notre carte fait aujourd'hui, lu sans hypothèse

    nvidia-smi -i 0 --query-gpu=clocks.max.sm,clocks.sm,power.limit,power.default_limit
    3135 MHz (max) / 2745 MHz (au repos, ~102 W) / 400 W (limite actuelle) / 600 W (défaut usine)

**Le plafond de puissance est déjà abaissé à 400 W** (bridage matériel ou
logiciel antérieur, non documenté ici — à vérifier séparément). C'est
exactement le régime que 2605.11999 décrit : un cap de puissance qui ne se
déclenche jamais au décodage (137-300 W mesurés chez eux, largement sous
400 W), donc **inerte par construction** — le levier utile est ailleurs :
l'horloge, pas le plafond.

## 2. Prédiction, scellée avant la mesure

GLM-4.7 étant hybride et déjà sous investigation séparée (bead 6wa, 92,80 ms
de pas à `b_reel=12`, `mla_*` sous-occupé), je prédis sur **Coder-30B**
(MoE dense, Qwen3-Coder-30B-A3B, `qwen3moe`, non hybride — c'est le modèle
demandé) que le décodage à `b=12` est **plus proche du régime « latence de
lancement/mémoire » que du régime « calcul SM saturé »** — cohérent avec ce
que ce dépôt a déjà établi partout ailleurs (paged_attn sous-occupé,
GEMV borné par le lancement, MLA sous-occupé). Si c'est le cas, réduire
l'horloge SM devrait coûter peu de débit tant que la fréquence reste
au-dessus du seuil où les noyaux mémoire-bornés cessent d'être cachés par le
pipeline.

    palier    perte de débit attendue     gain de J/jeton attendu
    2400 MHz  ≤  5 %                       ≥ 12 %
    2100 MHz  ≤ 12 %                       ≥ 20 %
    1800 MHz  ≤ 20 %                       ≥ 25 %
    1500 MHz  franchit un coude : perte de débit qui accélère
              plus vite que la puissance ne baisse — j/jeton
              plafonne ou remonte à ce palier

**Ce qui me réfuterait** : un débit qui décroît proportionnellement à
l'horloge dès 2400 MHz (perte ≥ 15 % dès le premier palier) — cela dirait que
le décodage Coder-30B est réellement compute-bound sur cette carte, contraire
à tout ce que ce dépôt a mesuré jusqu'ici pour des noyaux de décodage à petit
lot, et il faudrait revoir le diagnostic « sous-occupé partout ».
**Deuxième réfutation possible**, indépendante de la première : si J/jeton
**ne s'améliore à AUCUN palier** (le gain de puissance est entièrement
absorbé par la perte de débit), la thèse de poste7 ne se transporte pas de
H200/2842→180 MHz à notre carte/régime — à dire tel quel, pas à maquiller.

## 3. Protocole

    modèle     Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
    décodage   slots=12, ctx=2048, 200 jetons décodés par palier, un seul passage
               (pas de répétition ABBA dans cette première passe — le coût
               carte d'un aller-retour sudo par palier est déjà élevé ;
               si un palier est ambigu, le refaire spécifiquement)
    prefill    UNE mesure, horloge par défaut SEULEMENT (pas de balayage —
               décision de chef : le prefill n'est pas la question posée
               ici, et compte-rendu séparé demandé)
    paliers    défaut (aucun verrou), 2400, 2100, 1800, 1500 MHz
    outil      outils/gpu/mesure/energie.py (Energie, compteur NVML monotone,
               PAS une moyenne de nvidia-smi --query power.draw)
    verrou     outils/carte.sh enveloppe CHAQUE commande, y compris le
               verrouillage d'horloge lui-même (ACVRAM_TYPE=etat) — un
               changement d'état de la carte compte comme la mesure elle-même
    contrôle   `e.resume()["horloge_min"/"horloge_max"]` doit confirmer le
               palier annoncé (à ±30 MHz près) ; sinon la fenêtre est à rejeter
    baseline   `repos()` pris APRÈS chaque fenêtre, même durée — jamais avant
               (dérive documentée dans `energie.py`)
    métriques  t/s agrégé (décodage), J/jeton net (énergie moins la ligne de
               base au repos, à l'horloge courante)

**Le sudo n'est pas exécuté par cette session.** `outils/gpu/mesure/banc-horloge-decodage.sh`
affiche la commande exacte à chaque palier (`sudo ... nvidia-smi -lgc F,F`)
et attend une confirmation avant de lancer la mesure — préparé, pas joué.
Réinitialisation de l'horloge (`nvidia-smi -rgc`) obligatoire en fin de
campagne, également affichée et attendue.

## 4. Ce qui ferait adopter `ACVRAM_HORLOGE_DECODAGE`

Le palier retenu est celui qui maximise le gain de J/jeton **sous** une perte
de débit ≤ 10 % (seuil arbitraire mais nommé, pour ne pas choisir après coup
le palier qui arrange la conclusion). **Si ce gain dépasse 15 %** (seuil de
chef), le geste devient un réglage `ACVRAM_HORLOGE_DECODAGE` — sinon il
reste une note de mesure, pas un réglage produit.

## 5. Reste à faire

`outils/gpu/mesure/banc-horloge-decodage.py` (mesure, un palier à la fois,
décodage et prefill) et `.sh` (orchestration, paliers + confirmations
utilisateur) sont écrits et prêts. **La campagne elle-même attend qu'un
humain tape les commandes `sudo`** — cette session ne les exécute pas.
