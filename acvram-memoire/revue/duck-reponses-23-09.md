# Réponses brutes duck.ai — 23/09/2026

poste4 · Questions Q(15) et Q(16) · 3 modèles demandés, 2 complétés (session interrompue par pause groupe)

---

## Modèle 1 — GPT-5.6 Luna (raisonnement activé)

Discussion : « Optimisation latence graphe CUDA »

### Q(15) — Latence nœuds graphe CUDA b=1

**Contexte envoyé :** Graphe CUDA de ~600 nœuds par pas de décodage à b=1 : ~0,22 µs de latence nœud à nœud, soit 132 µs/pas. Sur sm_120 (RTX 5090). Question : Quelles techniques réduisent cette latence sans changer la sortie : Programmatic Dependent Launch dans un graphe, fusion de nœuds, noyau persistant (« megakernel »), nœuds conditionnels ? Gain attendu, et ce qui le réfuterait.

**Réponse Luna (extraits JS — offsets 51500–63000, reste bloqué par filtre extension) :**

> À 0,22 µs par dépendance, les 600 nœuds représentent bien environ 600 × 0,22 ≃ 132 µs. La première question est toutefois de déterminer si ces 132 µs sont réellement sur le chemin critique GPU, ou s'ils incluent des temps de noyaux, des bulles de disponibilité des SM, des accès mémoire et des effets de mesure. Un graphe CUDA récent peut déjà réduire le coût de lancement CPU à une valeur presque constante ; le coût résiduel observé ici ressemble donc plutôt à du scheduling et de la synchronisation entre nœuds côté GPU qu'à un simple coût d'appel CPU. NVIDIA mesure typiquement une amélioration de l'ordre de 60 ns par nœud sur le temps « launch-to-completion » de graphes linéaires, ce qui est inférieur à vos 220 ns mais du même ordre de grandeur.

Tableau (extrait) :
| Technique | Mécanisme | Gain |
|---|---|---|
| PDL | Le kernel secondaire peut commencer avant la fin complète du primaire (cudaTriggerProgrammaticLaunchCompletion / cudaGridDependencySynchronize). Disponible cc ≥ 9.0, donc sm_120 supporté. Gain ≈ min(travail indépendant du successeur, queue disponible SM). | Si chaque nœud suivant consomme immédiatement toutes les sorties du précédent, le gain est quasi nul. |
| Fusion de nœuds | Réduit le nombre de frontières ; si 10–20 nœuds fusionnés ne réduisent pas le temps malgré disparition des frontières, les 0,22 µs ne sont probablement pas le coût dominant. | |
| Megakernel | La seule option qui peut supprimer presque toutes les frontières. Peut rester résident, faire circuler les intermédiaires en registres/shared memory, utiliser une file ou un graphe de tâches côté device, remplacer 600 lancements internes par une boucle locale. Le gain maximal naïf ≈ totalité des 132 µs, mais non réaliste : synchronisations entre étapes deviennent barrières, compteurs, drapeaux ou événements device — coût déplacé, pas supprimé. | |
| Nœuds conditionnels | Réduisent les lancements conditionnels inutiles. | |

Note : les offsets 63000–70000 étaient bloqués par le filtre d'extension ; contenu partiel.

---

### Q(16) — NVFP4 W4A16 Gemma 4 31B KL hors gabarit

**Contexte envoyé :** Modèle : Gemma 4 31B quantifié NVFP4 W4A16 (poids FP4 E2M1, bloc 16, échelle E4M3, échelle globale FP32). Problème : sur 5 invites de test, 2 produisent une divergence KL > 1,0 entre le modèle bf16 de référence et le quantifié. L'erreur est répartie environ 45 % couches d'attention, 55 % couches MLP. Contrainte : on dispose d'au plus 1 heure de GPU. Question : quelles techniques permettent de récupérer ces 2 invites sans dépasser cette contrainte, et ce qui réfuterait chaque hypothèse ?

**Réponse Luna (extraits JS — offsets 73000–75094, reste bloqué) :**

> Étape 4 : ajouter FP8 progressivement — Ajouter d'abord 5 %, 10 %, puis 20 % des couches les plus sensibles. Si le gain KL sature rapidement, conserver un modèle majoritairement NVFP4. Si les gains sont distribués uniformément, cela indique que NVFP4 W4A16 est simplement trop agressif pour ce checkpoint sans QAT.

**Réponse directe à la contrainte « moins d'une heure » :**
- **Oui, très probablement : mixte FP8/NVFP4.** C'est le choix le plus sûr.
- **Oui, possiblement : QuaRot fixe**, mais son bénéfice pour NVFP4 est incertain.
- **Oui, possiblement : GPTQ NVFP4-aware**, avec 128–256 séquences et une implémentation CUDA efficace ; c'est le meilleur candidat si vous voulez conserver presque tout en NVFP4.
- **Non, probablement pas : SpinQuant complet** sur un 31B avec rotations apprises, sauf pipeline fortement réduit ou déjà optimisé.

**Ordre concret :**
1. FP8/NVFP4 sensible par couche
2. GPTQ NVFP4-aware + échelles optimisées
3. GPTQ NVFP4-aware sur les seules couches restées difficiles
4. QuaRot fixe comme test diagnostique
5. SpinQuant seulement si les quatre premiers essais échouent

> Si GPTQ NVFP4 + échelles optimisées et 20–30 % de couches FP8 échouent encore, cela réfuterait l'hypothèse d'un simple mauvais calibrage PTQ. Il faudrait alors envisager que ce checkpoint Gemma 4 31B nécessite un QAT NVFP4/W4A16, plutôt qu'une récupération purement post-entraînement. Des checkpoints QAT Gemma 4 W4A16 ont précisément été publiés pour absorber la perturbation de quantification pendant une phase d'entraînement, ce que GPTQ et les rotations ne peuvent pas toujours reproduire.

Sources citées : ScaleSweep (arxiv.org), GPTQ (mbrenndoerfer.com), Rotation pre-processing (amd.com), SpinQuant (arxiv.org, github.com), Gemma 4 QAT (spheron.network).

---

## Modèle 2 — gpt-oss 120B

Discussion : « Contexte : Graphe CUDA de ~600 nœuds… »

### Q(15) — Latence nœuds graphe CUDA b=1

**Réponse gpt-oss (complète, JS offsets 51500–55202) :**

**Techniques de réduction de latence d'un graphe CUDA (~600 nœuds, 0,22 µs / nœud → 132 µs/pas)**

| Technique | Comment ça agit sur la latence | Gain attendu (≈) | Pourquoi ça pourrait ne pas fonctionner / limites |
|---|---|---|---|
| Programmatic Dependent Launch (PDL) | En insérant explicitement des dépendances « stream-to-stream », le runtime peut lancer les nœuds dès que leurs pré-requis sont prêts, évitant le barrier global du graphe. | –5 % → −7 µs (≈ 125 µs) si le graphe a beaucoup de chemins parallèles non exploités. | Si le graphe est déjà fortement séquentiel, le PDL n'ajoute que du surcoût d'enregistrement et ne crée pas de parallélisme supplémentaire. |
| Fusion de nœuds (coalescence de kernels) | Réduit le nombre total de launches ; chaque lancement coûte ≈ 0,2 µs d'overhead. Fusionner 2 → 3 nœuds diminue le nombre de lancements de 600 à ~400-300, donc moins d'overhead. | 10–15 % → −12 µs à −20 µs (≈ 112–120 µs). | La fusion augmente la taille du kernel ; si cela dépasse la limite de registre/partage, le occupancy chute et le gain d'overhead est compensé par un temps d'exécution plus long. |
| Kernel persistant / « megakernel » | Un seul kernel persistant exécute tous les « steps » en boucle, éliminant complètement le coût de lancement. Les nœuds sont remplacés par des branches ou du dynamic parallelism interne. | 20–30 % → −25 µs à −40 µs (≈ 92–107 µs). | Nécessite une logique de contrôle très fine (barrières, atomics). Sur sm_120, les latences de synchronisation interne peuvent devenir le facteur limitant, surtout si le kernel persistant doit attendre des synchronisations fréquentes, ou utilisation de dynamic parallelism qui introduit son propre overhead sur sm_120. |
| Nœuds conditionnels | Si 10 % des nœuds sont souvent désactivés, le coût de leurs lancements disparaît. | 3–5 % → −4 µs à −7 µs (≈ 125–128 µs). | Le coût du calcul de la predicate et la possible divergence dans le kernel persistant peuvent annuler le gain si la condition est très volatile. |

**Priorisation pratique (RTX 5090, sm_120) :**
1. Fusion de nœuds – Gains faciles à quantifier, surtout si les kernels sont petits et peu dépendants.
2. Kernel persistant – Le plus gros gain potentiel, à condition de maîtriser la synchronisation interne.
3. Programmatic Dependent Launch – Utile lorsqu'il y a de nombreuses dépendances cachées qui empêchent le parallélisme maximal.
4. Nœuds conditionnels – À ajouter en dernier, seulement si la logique de branchement est déjà connue et stable.

**Ce qui pourrait réfuter chaque technique :**
- PDL : si le graphe est déjà optimalement parallélisé, le surcoût de la création de dépendances explicites (~0,05 µs / nœud) peut éliminer le gain.
- Fusion : dépassement des limites de registre/partage, augmentation du temps d'exécution du kernel fusionné, ou besoin de shared memory qui force le scheduler à réduire le nombre de blocs simultanés.
- Megakernel : mauvaise scalabilité du kernel persistant, besoin de synchronisations fréquentes, ou utilisation de dynamic parallelism qui introduit son propre overhead sur sm_120.
- Conditionnels : la charge de travail devient très non-uniforme, entraînant de la divergence et des pertes d'occupancy, voire un coût de calcul de la predicate supérieur au coût d'un simple lancement.

**Estimation globale :** En combinant fusion de nœuds + kernel persistant (en évitant les dépassements de ressources), on peut raisonnablement viser ≈ 25 % de réduction, passant de 132 µs à ≈ 100 µs par pas. L'ajout d'un PDL bien ciblé peut pousser le gain à ≈ 30 % (≈ 90 µs). Les nœuds conditionnels offrent un bonus supplémentaire uniquement dans les scénarios où les branches sont très prévisibles.

---

### Q(16) — NVFP4 W4A16 Gemma 4 31B KL hors gabarit

**Non posée** — pause groupe reçue avant envoi.

---

## Modèle 3 — Gemma 4 31B

**Non interrogé** — pause groupe reçue avant ouverture de la discussion.

---

## Note de session

Pause groupe ordonnée par chef (23/09/2026) avant la fin du cycle complet. Q(16) gpt-oss et Gemma 4 31B (Q15+Q16) restent à faire à la reprise. La réponse Luna Q(15) est partielle (filtrage extension sur offsets 63000-70000).
