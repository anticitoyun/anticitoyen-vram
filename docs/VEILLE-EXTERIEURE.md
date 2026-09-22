# Veille extérieure

Projets voisins suivis pour ce qu'ils apportent — ou pour ce qu'ils laissent
de côté et qui reste donc à faire ici. Une fiche par projet, datée.

## NVIDIA Personal AI Router (PAIR) — relevé le 3 septembre 2026

<https://github.com/NVIDIA/Personal-AI-Router> · Apache-2.0 · Go ·
créé le 2 juillet 2026, dernier envoi le 28 août 2026 · 4 étoiles.

> « Router that virtually distributes inference across connected devices in the
> home. »

### Ce que c'est

Un **routeur d'inférence de réseau local**, pas un moteur. Il découvre les
machines participantes du réseau, pilote les moteurs installés dessus, et
présente aux applications deux façades de compatibilité : **API Ollama** et
**API OpenAI**. Chaque requête indépendante part vers un nœud éligible selon
trois critères — moteur disponible, modèle disponible, charge courante.

| | |
|---|---|
| systèmes | Windows 11, Linux, macOS (x64 et arm64 ; Windows/ARM expérimental) |
| paquets | `.exe`, `.deb`, `.dmg` ; sinon compilation depuis les sources |
| moteurs pilotés | **Ollama et LM Studio uniquement** |
| interface | application de bureau (recommandée) ou interface terminal |
| nœuds | Windows, Linux et macOS peuvent être appairés entre eux |

### Ce que ce n'est pas — le point important pour nous

Le dépôt le dit explicitement : PAIR **ne met pas la VRAM en commun**, ne
fusionne pas plusieurs GPU en un GPU logique, ne découpe pas un modèle entre
machines et ne scinde pas une requête en vol. Il aiguille des requêtes
**entières** vers des machines **entières**. C'est l'inverse exact du problème
que traite acvram (faire tenir un modèle sur la carte qu'on a, au format le plus
compact qui garde la qualité) : les deux ne se concurrencent pas, ils
s'empilent.

Sa politique d'ordonnancement est, de son propre aveu, sommaire : une seule
règle, combinant la file d'attente et un signal lissé d'utilisation GPU. Elle
**ignore** le modèle de GPU, la mémoire libre, le fait qu'un modèle soit déjà
chaud, et le coût prévisible d'une requête — les auteurs notent que cela
convient à un parc homogène, mal à un parc mixte. Notre parc (5090 32 Gio +
3080 Ti 12 Gio) est exactement le cas mixte.

### Ce qu'on en tire

1. **La façade double Ollama + OpenAI** est le bon choix d'interface : nos
   clients (menus `kimi-modeles`, `claude-modeles`) parlent déjà OpenAI ; le
   dialecte Ollama ouvrirait les applications qui ne connaissent que lui.
2. **Le pilotage de moteurs tiers est restreint à Ollama et LM Studio** : notre
   serveur (port 8090, OpenAI-compatible) n'est pas branchable en l'état.
   Adosser acvram à PAIR demanderait soit un adaptateur côté PAIR, soit de se
   faire passer pour un point d'entrée Ollama — c'est peu de travail et ce
   serait la voie d'une contribution amont.
3. **Leur angle mort est notre matière** : mémoire libre, chaleur du modèle,
   coût estimé d'une requête sont précisément ce que `memory/tiering.py` et le
   plan de placement calculent déjà par carte. Un ordonnanceur sensible à ces
   signaux est un chantier possible ici, et une contribution possible là-bas.
4. **Non retenu pour l'instant** : PAIR suppose plusieurs machines. Nos deux
   cartes sont dans le même boîtier ; l'ordonnancement entre elles se joue en
   processus, pas sur le réseau.
