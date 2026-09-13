# Protocole + prédiction — horloge mémoire × SM, décodage b=12 (duck.ai chef, item 1)

poste3, 14/09/2026, à sec (carte occupée par poste4 puis poste1). Suite au
duel du jour : vLLM 0,202 J/jeton contre nos 0,601 à b=12 (poste2) — l'énergie
est notre plus gros retard, l'horloge mémoire est le levier cité par le
rapport GitHub 07/2026 relayé par le tour duck.ai de chef (-26 % W en
path-tracing à `-lmc 15000`, fréquence physique inchangée).

## 1. Réserve à écrire avant tout test — le chiffre source ne transporte pas tel quel

    nvidia-smi -i 0 --query-gpu=clocks.max.memory --format=csv
    14001 MHz

**Notre carte plafonne à 14001 MHz, sous les 15000 MHz du protocole
demandé.** Le rapport GitHub cité portait sur du path-tracing (charge très
différente du décodage LLM) et ne précise pas le modèle de carte — `15000`
n'est probablement pas une valeur universelle mais celle qui convenait à
LEUR matériel. **Je teste `-lmc 15000,15000` littéralement comme demandé**
(pour documenter le refus/l'écrêtage, pas en espérant qu'il passe), puis je
mène le balayage sur des valeurs qui existent réellement sur cette carte :
`{stock, 14000, 12000, 10000}` — la seconde valeur du protocole de chef,
proche du max réel, sert de témoin « verrou actif mais non contraignant »
plutôt que d'un vrai levier.

## 2. Protocole

    modèle       Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4 (même que le duel)
    régime       décodage, b=12, ctx=2048 (même protocole que banc-horloge-decodage.py)
    axes         mémoire ∈ {stock (aucun verrou), 14000, 12000, 10000} MHz
                 SM      ∈ {défaut (aucun verrou), 2100} MHz
                 8 cellules (stock+défaut = témoin nul, sans verrou d'aucune sorte)
    outil        outils/gpu/mesure/energie.py (compteur NVML monotone),
                 repos() ≥ 30 s (corrigé le 13/09, cf. verification-energie-py-14-09.md)
    verrous      `-lgc` (SM) et `-lmc` (mémoire) posés/déposés séparément,
                 CHACUN sous carte.sh (ACVRAM_TYPE=etat) — le test d'acceptation
                 (`-lmc 15000`) et chaque palier du balayage
    garde        `-rmc` ET `-rgc` en fin de campagne ET en trap (EXIT/INT/TERM),
                 même patron que `outils/gpu/mesure/banc-horloge-decodage.sh` —
                 étendu pour couvrir les DEUX horloges, pas seulement SM
    contrôle     power.limit revérifié à 400 W après chaque reset (comme le 13/09)
    lancement    systemd-run (le superviseur de session tue sur MemFree brut,
                 cf. `outils/carte.sh:76-84`) — le patron qui a fonctionné le 13/09

## 3. Prédiction, scellée avant la mesure

Le décodage à petit batch (même b=12) est déjà établi dans ce dépôt comme
**latence-borné plutôt que bande-passante-borné** presque partout où on l'a
mesuré (MLA, paged_attn, GEMV — cf. `campagne horloge SM du 13/09`, où
réduire l'horloge SM coûtait 1,6 à 2,3× plus de débit que prévu, signe que
le calcul EST sensible à la fréquence, contrairement à l'hypothèse de
départ). **La mémoire est un axe différent** : si le décodage est
réellement borné par la lecture des poids (26 Go lus par pas à b=1, moins
par jeton à b=12 amorti), réduire l'horloge mémoire devrait coûter du débit
de façon plus marquée que l'horloge SM ne l'a fait — **je prédis un coude
plus tôt sur l'axe mémoire que sur l'axe SM** :

    mémoire   perte de débit attendue      gain J/jeton attendu
    14000     ≤ 3 % (proche du max, presque un témoin)   ≥ 3 %
    12000     10-20 %                                     10-18 %
    10000     25-40 % (coude probable ici, pas après)     15-25 % (plafonne ou régresse)

**Combinaison mémoire × SM** : je prédis que les deux effets sont
**quasi-additifs en perte de débit mais PAS additifs en gain d'énergie** —
verrouiller les deux horloges à la fois ne double pas le gain J/jeton du
13/09 (2100 MHz seul, +19,3 %) plus celui de la mémoire seule ; je prédis un
gain combiné **inférieur** à la somme des deux gains isolés, parce que les
deux leviers agissent sur le MÊME goulot (la latence globale du pas), pas
sur deux postes indépendants.

**Ce qui me réfuterait** : un débit qui ne bouge presque pas jusqu'à
10000 MHz (le décodage ne serait alors pas du tout borné par la mémoire à
b=12, contrairement à ce que l'écart avec vLLM suggère) ; ou un gain
combiné supérieur à la somme des deux gains isolés (additivité franche,
contraire à l'hypothèse d'un goulot partagé).

## 4. Ce qui reste à faire

Test d'acceptation `-lmc 15000,15000` et balayage complet : **sur carte**,
après poste4 (banc MMA2 décodage) et poste1 (masquage fantômes) — les deux
prévenues, en attente de leur signal.
