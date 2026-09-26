# poste7 — trois modes d'énergie (économique / moyen / maximal) : oui, chiffrés sur ce qu'on a mesuré (14/09)

Sources : `verdict-horloge-decodage-13-09.md:11-16` (poste3, Coder-30B b=12, base 568,6 t/s / 0,601 J net) ; campagne 20 s (0234a7d : 630,6 t/s / 0,619 J brut, 390 W, 19,0 ms/pas) ; instr-par-octet (1 pas non bridé : 15,66 ms) ; `energie-brute-faible-lot` (repos chaud 68-76 W, froid 17 W) ; `nvidia-smi` : plafond appliqué 400 W, défaut carte 600 W. Aucun réglage d'énergie n'existe dans le moteur (`hardware/profiles.py:51` ne sert qu'au tiering).

**Réponse : oui.** Un mode = un réglage de carte (plafond de puissance, horloge) **plus** deux leviers moteur qui n'ont besoin d'aucun droit. Ce qui a été refusé le 13/09 (« −20 % de débit ») l'était comme **défaut** ; comme **choix explicite de l'utilisateur**, c'est précisément un mode. Prédictions sur Coder-30B b=12, 20 s, référence = mode moyen :

| mode | réglage | prédit (t/s, J/jeton) | réfuté si | coût |
|---|---|---|---|---|
| **maximal** | plafond 600 W (défaut de la carte), horloges libres | **+15-20 % t/s** (pas 19,0 → ~15,7 ms, le pas mesuré non bridé), **J/jeton ±5 %** — le plafond coûte du temps, pas des joules | t/s < +8 % ou J > +15 % | +150 W, chaleur, bruit ; `sudo` |
| **moyen** (défaut actuel) | 400 W, horloges libres | 630,6 t/s / 0,619 J | — | — |
| **économique** | plafond 300 W **ou** `-lgc` 2 100 MHz | poste3 13/09 : 2 100 MHz → **t/s −20 %, J −19 %** ; 1 800 → −31 %, −23 % ; coude à 1 500 (J remonte). Prédit à 300 W auto-régulé : J −15 à −20 %, t/s −15 à −25 % | J < −10 % | débit ; `sudo` |

**Deux leviers moteur, sans droits, qui pèsent plus que l'horloge pour un service à faible trafic :**
1. **Plancher de repos** : 68-76 W chaud contre 17 W froid ; à b=1 le repos vaut 0,30 J/jeton sur 1,43 (21 %). Cause à établir par mesure, pas à deviner : attente active de la boucle de service, ou contexte CUDA maintenu en P0. Mesure : W au repos, serveur chargé et bloqué sur la prise, contre carte vide ; si ≤ 30 W est atteignable, **−0,17 J/jeton à b=1 (−12 %)** sans toucher au débit. Réfuté si le repos bloqué reste ≥ 60 W.
2. **Fenêtre de regroupement** (attendre 10-50 ms pour former un lot) : 1,43 J/jeton à b=1 → 0,619 à b=12, **−57 % par jeton** quand le trafic le permet ; coût : +fenêtre de latence. Réfuté si, à ≥ 2 requêtes concurrentes, J/jeton ne baisse pas d'au moins 20 %.

**Contraintes d'implémentation, toutes tirées des règles :** `-pl`/`-lgc` exigent la racine — ici `sudoers` pour `nvidia-smi` seul, donc `acvram serve --energie eco|moyen|max` appelle `sudo -n nvidia-smi` et **remet `-rgc` / `-pl 400` à la sortie et sur signal** (REGLES : « toujours remettre -rgc ») ; le réglage est **global à la carte** — refus sans le verrou `carte.sh`, et `acvram doctor` dit si le mode est disponible ; **le mode est porté par le nom** de tout dossier de mesure (règle 4). Test d'acceptation qui casse : un banc fixe à 20 s par mode, `eco` doit rendre J ≤ 0,90 × `moyen` et `max` t/s ≥ 1,08 × `moyen`, sinon le mode ment.

**Ce qui n'est pas un mode** : la MMA MoE au décodage (−30 % de J à puissance égale) améliore les trois ; réduire `top_k` des experts change la sortie — pas sans PPL publiée, et pas sous ce nom.

**Correction 14/09 nuit (chef, main 1103825)** : la 5090 n'accepte un plafond qu'entre **400 et 600 W** (« should be between 400.00 W and 600.00 W », vérifié, `-pl 400` remis) — le « REFUS » de poste3 n'était pas `sudo`, c'était la carte. **Le mode économique ne passe donc pas par le plafond** : il passe par l'horloge (`-lgc 2 100` : J −19 %, t/s −20 %, mesuré le 13/09 ; `-rgc` à la sortie) et par les deux leviers moteur (plancher de repos, fenêtre de regroupement). Le mode maximal (`-pl 600`) reste tel quel. Tableau ci-dessus à lire avec cette ligne ; prédiction eco inchangée en J (−15 à −20 %), instrument changé.
