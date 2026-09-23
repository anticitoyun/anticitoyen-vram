# Découplage fréquence cœur / fréquence mémoire — ce qui est possible, ce qui trancherait

Demandé par Jérôme le 10/09/2026, suite à une piste d'un avis extérieur (à
vérifier, pas un fait établi — la source contient des erreurs factuelles
ailleurs, voir mise en garde de Jérôme). **Rien exécuté sur la carte pour ce
protocole** — établi par lecture de documentation et une vérification de
permission qui a mal tourné (voir l'incident signalé séparément, corrigé
dans la minute, sans effet sur une mesure en cours).

## 1. Ce que la carte permet, vérifié sur cette machine

Pilote 595.84, RTX 5090 (`sm_120`) :

* **`nvidia-smi --applications-clocks` (l'ancienne interface) est
  DÉPRÉCIÉE** sur ce pilote — ne pas s'appuyer dessus.
* **`-lgc`/`--lock-gpu-clocks=min,max`** (verrouille la fréquence cœur),
  **`-lmc`/`--lock-memory-clocks=min,max`** (fréquence mémoire),
  **`-lmcd`** (verrou mémoire différé, appliqué au prochain init GPU),
  **`-rgc`** (réinitialisation) — tous existent dans cette version.
* **Aucun droit manquant** : `sudo -n nvidia-smi -lgc ...` s'exécute sans
  mot de passe sur cette machine (sudoers pré-autorisé). `-rgc` aussi.
  **Vérifié par accident** (incident signalé séparément) — la commande n'a
  pas échoué en test de permission, elle a réellement pris effet.
* **Fréquences supportées** (`-q -d SUPPORTED_CLOCKS`) : mémoire annoncée à
  une seule valeur haute (14001 MHz — GDDR7 ne semble pas offrir de paliers
  intermédiaires pilotables comme le cœur) ; cœur en paliers fins de
  7-8 MHz de ~2100 à 3135 MHz (plage complète non énumérée ici, capturée
  partiellement).
* **Courbe tension/fréquence** : non explorée — nécessiterait
  `nvidia-settings` ou un outil tiers (non vérifié disponible sur cette
  machine), hors du périmètre `nvidia-smi` seul.
* La carte est déjà bridée en puissance à **400 W** (mémoire du 8/09), ce
  qui interagit avec tout verrouillage de fréquence : au-delà d'un certain
  point, c'est le plafond de puissance qui limite, pas la fréquence
  demandée.

## 2. Le protocole qui trancherait

**Modèle et contexte fixés, un seul paramètre varié : la fréquence cœur.**
Mémoire verrouillée à sa fréquence max (14001 MHz, pas de paliers à
balayer) pour isoler l'effet du cœur seul — c'est la variable que la piste
propose de sous-fréquencer, pas la mémoire.

    pour chaque frequence_coeur dans [palier bas ... 3135 MHz, ~6-8 points] :
        nvidia-smi -lmc 14001,14001 -lgc F,F   (verrou double, une seule variable)
        chauffe (palier thermique, cf. protocole habituel)
        N pas de decodage, meme modele, meme contexte, meme lot (mono-flux,
          celui de notre duel, pas un lot batch)
        releve : debit (tok/s) ET puissance (W, moyenne sur la fenetre
          mesuree — nvidia-smi --query-gpu=power.draw en boucle courte,
          ou DCGM si disponible)
        calcule : jetons/kJ = debit / (puissance_W / 1000)
        nvidia-smi -rgc -rmc   (reinitialiser avant le point suivant —
          jamais enchainer un verrou sur un autre sans repasser par defaut)

**La quantité qui décide est jetons/kJ, jamais le débit seul.** Tracer les
deux courbes (débit vs fréquence, jetons/kJ vs fréquence) sur le même
graphique : si jetons/kJ passe par un maximum à une fréquence intermédiaire,
la piste tient ; si jetons/kJ est monotone croissant avec la fréquence
(l'énergie totale baisse moins vite que le débit ne baisse), elle tombe.

**Point de départ pour le palier bas** : ne pas descendre sous la fréquence
où `Clocks Event Reasons` signale un `HW Slowdown` ou un comportement
instable — quelques points d'essai avant le balayage complet pour trouver
la plage utile, plutôt que balayer à l'aveugle jusqu'à l'instabilité.

## 3. Le confondant, écrit avant de mesurer

**Baisser la fréquence cœur baisse aussi le débit — ce n'est pas une
réserve, c'est la prémisse même du test.** Le gain n'existe QUE si l'énergie
baisse plus vite que le débit, c'est-à-dire si la puissance instantanée à
fréquence cœur réduite chute proportionnellement plus que le débit ne
chute. Trois issues, écrites d'avance :

* **jetons/kJ monte quand la fréquence cœur baisse, jusqu'à un point puis
  redescend** → la piste tient, il existe une fréquence optimale
  différente du défaut, gain sans toucher au code. C'est l'issue que la
  piste prédit.
* **jetons/kJ est plat ou monotone décroissant quand on baisse la
  fréquence** → la piste tombe : soit le plancher de puissance (les 400 W)
  n'est jamais atteint au régime mono-flux qu'on mesure (donc rien à
  regagner en énergie en dessous de ce plafond), soit la puissance mémoire
  domine déjà la puissance totale et la fréquence cœur n'y change presque
  rien.
* **Un résultat intermédiaire — jetons/kJ monte un peu puis stagne bien
  avant la fréquence par défaut** → dit que le gain existe mais est petit,
  et qu'il faut le chiffrer avant de décider si ça vaut la complexité
  opérationnelle (verrouiller les fréquences à chaque démarrage de
  service).

**Second confondant, à surveiller pendant la mesure, pas après** : à
fréquence cœur réduite, un mono-flux en décodage (donc déjà mémoire-borné,
par nos propres mesures : 6,1× et 8,6× la borne mémoire théorique) pourrait
voir son débit inchangé sur une large plage avant de chuter — c'est
précisément l'hypothèse qui rend la piste plausible. Mais si `paged_attn`
ou une autre étape a une composante de calcul non négligeable (nos propres
17,5% de lancements d'attention sur ce card, `PA_CHUNK` variable), la
chute de débit pourrait commencer plus tôt que prévu. **Relever aussi le
temps par noyau (nsys), pas seulement le débit agrégé**, pour savoir SI la
chute de débit vient du cœur ou d'ailleurs — sinon un plateau surprenant se
lira comme un résultat alors qu'il pourrait être un artefact de la même
famille que ceux retirés ce soir.

## Ce qui n'est PAS fait ici

Aucune mesure exécutée. Aucun verrou appliqué délibérément. Le protocole
attend une carte réservée via `outils/carte.sh`, un modèle choisi (proposer
un modèle DENSE de taille moyenne, cohérent avec le duel principal, pas un
MoE — pour ne pas mélanger deux variables), et le feu vert de Jérôme sur le
choix du modèle et de la plage de fréquences avant tout balayage réel.
