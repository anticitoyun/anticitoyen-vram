# Documentation : ce qui trancherait, ce qui ne servirait à rien

Part d'Océane dans le chantier documentation 5090 / 3080 Ti (9/09/2026).
Question posée : quelle documentation fermerait une question ouverte, laquelle
n'en fermerait aucune.

## Le critère

Un document ne ferme une question que s'il **prédit un nombre que notre mesure
aurait pu démentir**. Sans cela il commente, il ne tranche pas.

D'où le critère d'acceptation de tout le chantier, à appliquer par chacun :

> **Chaque document rassemblé arrive avec une ligne nommant le chiffre du
> dossier qu'il pourrait contredire.** Sans cette ligne, c'est un livre, pas une
> pièce.

Et l'ordre de valeur, qui n'est pas intuitif : les documents les plus utiles ne
portent pas sur les causes que nous cherchons, mais sur les **instruments de nos
éliminations**. Nous avons éliminé cinq causes du −4,7 % ; chaque élimination
repose sur un compteur dont nous n'avons documenté qu'une partie de la
sémantique. C'est là qu'une contradiction est la plus probable, et une
contradiction vaut la journée.

## Les trois à chercher, dans cet ordre

### 1. NVML : échantillonnage et répétabilité de `power.draw`

Référence de l'API NVML pour `nvmlDeviceGetPowerUsage`, les champs de puissance
moyennée (`nvmlDeviceGetFieldValues`, `POWER_INSTANT` / `POWER_AVERAGE`), la
période d'échantillonnage, et ce que la lecture inclut (carte entière ou puce
seule, mémoire comprise ou non). Documentation DCGM des mêmes champs.

**Chiffre visé : les −4,7 % d'énergie.** Nous cherchons la cause d'un écart sans
avoir établi que notre instrument le résout. Si la répétabilité du relevé est du
même ordre, la cause n'est pas introuvable : elle est inexistante, et cinq
éliminations ont été dépensées sur du bruit.

Précision à ne pas rater dans la lecture : c'est la **répétabilité** qui décide,
pas la justesse absolue. Un biais constant s'annule dans un rapport entre deux
moteurs mesurés sur le même instrument ; une dispersion d'échantillonnage, non.
Et une fenêtre d'intégration plus longue que nos passages fait lire à deux
moteurs la même moyenne glissante.

C'est le seul document du chantier qui peut **contredire** un chiffre publié
plutôt que l'expliquer. Il passe avant tout le reste.

### 2. Sémantique complète des bits de `clocks_event_reasons`

Liste exhaustive des bits, leur signification et leur précédence (référence
NVML / nvidia-smi, pas un billet de blog).

**Chiffre visé : « fréquences éliminées ».** Nous avons découvert par accident
que le bit 0 est `GpuIdle`. Nous lisons un champ de bits dont nous avons deviné
une partie de la sémantique, et nous en avons tiré une élimination. Si un bit
que nous ignorons signale un plafonnement dans nos traces existantes,
l'élimination tombe — et sans nouvelle mesure, les traces sont déjà là.

### 3. Granularité en M des instructions matricielles

Tables de formes du PTX ISA (`mma`, `tcgen05` sur les familles concernées) et
tables de tuiles CUTLASS pour `sm_120`.

**Chiffre visé : le croisement de fusion entre 171 et 512 jetons.** Une
granularité matérielle en M prédit un croisement à un **multiple de tuile** ;
nos deux bornes y tombent ou n'y tombent pas, et c'est décidable.

**Mais à ne chercher qu'après une vérification de dix minutes** : si le noyau
fusionné est le nôtre, la tuile est **celle que nous avons choisie**, elle est
dans notre configuration de lancement, et aucune documentation constructeur ne
répondra. Le nom déformé du noyau effectivement lancé (`nsys`, `cuobjdump`)
porte ses dimensions. **Regarder ce que nous lançons avant de chercher ce que le
matériel offre.** Nous venons de reprocher au seuil de 8 d'être une constante non
mesurée ; 256 le serait aussi, et 171 encore plus.

## Ce qu'il ne sert à rien de rassembler

* **Le livre blanc d'architecture Blackwell et toute fiche produit.** TFLOPS
  crête, « tensor cores de 5ᵉ génération », bande passante affichée : aucun
  contenu falsifiable à notre résolution. Aucune mesure du dossier n'aurait pu
  en sortir différente.
* **La documentation Ampere / 3080 Ti au-delà de l'existant.** La conclusion est
  fermée par le SASS : ni FP4 ni conversion E2M1. Il n'y a plus de décision qui
  en dépende.
* **Le guide de programmation CUDA rassemblé comme corpus.** C'est une référence
  à consulter par symbole, jamais un document à collecter.
* **Tout ce qui porte sur des fonctions que nous n'employons pas** : NVLink,
  MIG, informatique confidentielle, DPX, mémoire distribuée entre blocs si nous
  ne lançons pas de grappes.
* **Les bancs et billets tiers sur la 5090.** Ce ne sont pas des
  spécifications : ils donnent un chiffre à égaler, pas une prédiction à
  éprouver. Un chiffre à égaler oriente les mesures suivantes vers sa
  confirmation.
* **Toute documentation sur le TTFT.** Le plancher de 73 ms contre 33 est un
  coût de notre code, parfaitement déterministe (73, 72, 73, 73). Aucune
  spécification constructeur ne le concerne. Cette question se ferme par
  profilage, jamais par lecture.
