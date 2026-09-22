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
