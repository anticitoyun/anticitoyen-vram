# Dispersion acvram b=1 (sd 13,1 % > 10 %) — lecture à sec, 22/09

Demande poste2 : lire les fenêtres brutes acvram b=1 de `sorties-energie/` (passages 3
et 5) et dire d'où vient la dispersion. **Limite rencontrée** : `fenetre-*.log` est
réécrit au même nom à chaque passage de la chaîne (pas de suffixe par passage) — seul
le DERNIER passage écrit avant l'arrêt de la carte est encore lisible. Les fichiers
`chaine-2.log` à `chaine-8.log` du même dossier (mêmes horodatages, 12:12-15:15)
montrent sept lancements successifs de `chaine-energie-4moteurs.sh` le 22/09 ;
`fenetre-acvram-b1-1.log` a été écrit entre `chaine-7.log` et `chaine-8.log`
(horodatages de fichier), donc il appartient au **dernier passage (8e lancement,
probablement le « passage 5 » compté par poste2 en excluant les échecs précoces)**.
Le passage 3 n'a laissé aucune trace récupérable : son `fenetre-acvram-b1-1.log` a
été écrasé par les passages suivants. Cette lecture ne porte donc que sur UN passage,
pas deux — nommé plutôt que deviné.

## Ce que dit le RESULTAT brut du passage disponible

```
passes_courtes_jetons_s: [274.6, 274.7, 275.3, 275.6, 348.6, 349.0, 370.9]
horloge_min/moy/max: 2647 / 2662 / 2692 MHz
temp_max: 46 °C
bridages: aucun
charge_avant: {load1: 0.97, load5: 1.17, processus_charges: []}
charge_apres: {load1: 1.04, load5: 1.17, processus_charges: []}
```

Moyenne des 7 passes courtes ≈ 309,8 t/s, écart-type/moyenne ≈ 13 % — cohérent avec
le rejet publié.

## Lecture

Les quatre premières passes courtes (274,6 à 275,6 t/s) sont resserrées entre elles
(< 0,4 % d'écart), puis les trois dernières (348,6 à 370,9 t/s) forment un second
palier, lui aussi resserré (< 6 %) mais nettement plus rapide. La dispersion n'est
donc pas du bruit aléatoire fenêtre par fenêtre : c'est une **marche en escalier en
deux paliers**, lent puis rapide, à l'intérieur d'une seule fenêtre de mesure de
20,9 s.

Ce que les autres champs EXCLUENT :
- **Horloge** : 2647-2692 MHz, écart ≈ 1,7 % seulement — ne peut pas expliquer un
  saut de débit de 35 % (274 → 371 t/s).
- **Bridage** : `aucun`.
- **Charge étrangère** : `processus_charges: []` avant et après, `load1`/`load5`
  quasi stables — rien détecté par la garde de charge.
- **Thermique** : `temp_max 46 °C`, très loin d'un seuil de throttle ; de plus un
  throttle thermique ralentirait la fin de fenêtre, pas le début — c'est l'inverse
  qui est observé ici (lent puis rapide).

Ce que ça laisse : un **régime transitoire de démarrage** propre à b=1 (une seule
séquence, donc rien pour amortir un warm-up que le protocole absorbe déjà à b=12 par
la moyenne sur 12 flux). `attendre_port` déclare le serveur acvram prêt dès le premier
`/v1/models` répondant ; rien ne garantit que le premier lot décodé après ce point
tourne déjà à la vitesse stationnaire (allocateur CUDA, autotune de noyau,
réchauffement du planificateur MoE/attention, etc. — hypothèses non départagées ici,
à sec). La fenêtre de mesure de 20,9 s capture donc une partie de ce warm-up : b=1
n'a qu'UN passage par protocole (pas de paire à moyenner comme en b=12), donc rien
n'amortit ce transitoire côté protocole.

## Conséquence proposée (pas encore codée, à trancher par la chef/poste2)

Le rejet actuel (sd > 10 %) est correct au sens du protocole : cette fenêtre ne doit
pas être publiée telle quelle. Deux pistes pour obtenir un point b=1 valide :
1. Rejouer b=1 avec un délai de stabilisation après `attendre_port` (ex. une requête
   de chauffe jetée avant le chronométrage), comme le fait déjà `chaine-cellule-b12-
   vraie-vllm.sh` pour vLLM (`llm.generate(chauffe, ...)` avant la fenêtre mesurée).
2. Rejouer b=1 plusieurs fois (via `--bras`, voir commit suivant) et ne garder que
   les fenêtres qui passent le seuil sd, en acceptant qu'une partie soit perdue au
   warm-up.

## Mise à jour 22/09 18 h 3x (poste4, à sec) — la chauffe réfutée, ET un bogue de lecture trouvé

**Réfuté (poste2 0ba60ca4)** : chauffe posée (`jetons_chauffe` publié, 256 vus sur les
3 nouvelles fenêtres), sd reste 12-13 % > 10 % sur les 3. Ma prédiction « sd < 5 % » ne
tient pas.

**Découverte en relisant le code, pas seulement les chiffres** : `passes_courtes_jetons_s`
est publié APRÈS un `courtes.sort()` (`scratchpad/banc-llamacpp-16-09.py`, avant
correctif ci-dessous). Tout ce que « marche en deux paliers » ci-dessus affirmait sur
la FORME temporelle (lent puis rapide) était donc lu sur un tableau **trié par valeur**,
nécessairement croissant quel que soit l'ordre réel des 7 sondes — l'observation n'était
pas fausse sur les valeurs, mais la lecture « warm-up chronologique » n'était jamais
vérifiable avec ces données. Corrigé (sha ci-dessous) : `passes_courtes_jetons_s` publie
maintenant l'ordre CHRONOLOGIQUE réel, `meilleure_passe_jetons_s` calculé par `max()`
plutôt que par le dernier élément trié (même valeur, méthode indépendante de l'ordre).

**Ce qui reste vrai indépendamment du tri** (l'AMPLITUDE de la dispersion, contrairement
à sa FORME, n'est pas affectée par `sort()`) : comparé sur le même client, la même
fenêtre, le même jour — acvram b=1 : min 274,6, max 370,9 t/s (écart 35 %) ; llama.cpp
b=1 : min 320,9, max 322,1 (écart 0,4 %) ; trtllm b=1 : min 46,1, max 46,6 (écart 1 %).
La dispersion est spécifique à acvram (le même client HTTP mesure les trois moteurs),
pas un artefact de cadence du client — confirmé aussi en lisant `decode()` : à b=1,
`lot()` est une requête HTTP séquentielle unique (`ThreadPoolExecutor(max_workers=1)`),
les 7 sondes s'enchaînent en boucle Python sans `sleep` ni délai imposé entre elles.
Autre fait qui déplace la question : les 7 sondes courtes tournent APRÈS la fenêtre
principale de 20 s (déjà chaude, 8192 jetons décodés à 392 t/s agrégé stable) et après
la chauffe — un warm-up général du serveur ne peut donc pas expliquer une éventuelle
lenteur initiale des sondes, puisque le serveur a déjà tourné ~26 s avant la première
sonde courte.

**Hypothèse suivante** (à départager par le prochain relevé, maintenant lisible dans
l'ordre) : la dispersion est propre au CHEMIN des requêtes courtes (128 jetons, contre
1 024 pour la fenêtre principale) chez acvram spécifiquement — pas un warm-up général
du serveur, pas la cadence du client. Trois formes possibles, à distinguer par la
FORME de la série maintenant publiée dans l'ordre :
- **croissante** (les premières sondes plus lentes) → chemin froid propre à CETTE
  taille de requête, indépendant de l'état déjà chaud du serveur — tester alors
  chauffe 20 s ou fenêtre b=1 de 40 s (pistes de poste2) ;
- **dispersée sans tendance** → pas un warm-up du tout ; chercher une gigue de
  planification/réseau côté serveur ou client, indépendante de b (tester alors client
  à 2 requêtes en vol pour voir si ça change la variance, autre piste de poste2) ;
- **périodique/alternante** → signature d'un mécanisme cyclique interne (ex.
  recapture de graphe, libération VRAM périodique) — hors des trois pistes de poste2,
  nommé si observé.

**Test minimal proposé, avant de choisir entre chauffe 20 s / fenêtre 40 s / 2
requêtes en vol** : une seule fenêtre b=1 de plus (déjà avec chauffe, code corrigé) et
lire `passes_courtes_jetons_s` dans l'ordre publié — la forme observée départage
directement les trois hypothèses ci-dessus sans changer le protocole ni la carte au-delà
de cette unique fenêtre.
