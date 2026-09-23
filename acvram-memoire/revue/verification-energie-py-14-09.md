# Vérification `energie.py` (point 2, duck.ai Jérôme 14/09) — à sec

Laure, en attendant la carte pour A8. Question posée : `outils/gpu/mesure/energie.py`
lit-il `power.draw` (moyenne 1 s) ou `power.draw.instant` (25 ms) ?

## Réponse — `power.draw`, déjà correct

`energie.py:95-96` :

    def puissance_w(self, h) -> float:
        v = self._u32(self.l.nvmlDeviceGetPowerUsage, h)

**`nvmlDeviceGetPowerUsage`** est l'API classique de NVML — c'est exactement
celle que `nvidia-smi --query-gpu=power.draw` interroge. Vérifié dans la
documentation officielle du champ, lue à l'instant
(`nvidia-smi --help-query-gpu`) :

> **"power.draw"** — The last measured power draw for the entire board, in
> watts. **On Ampere or newer devices, returns average power draw over
> 1 sec.** On older devices, returns instantaneous power draw.

La RTX 5090 (Blackwell) est postérieure à Ampere → **moyenne sur 1 s**.
`power.draw.instant` (25 ms, champ distinct, confirmé existant et
interrogeable séparément : `nvidia-smi --query-gpu=power.draw.instant`)
correspond à une API NVML plus récente (`nvmlDeviceGetFieldValues` avec un
champ instantané dédié), **jamais appelée dans ce dépôt**.

**Conclusion : pas de correctif nécessaire sur ce point précis.** `energie.py`
lit déjà la valeur lissée, pas le signal bruité à 25 ms.

## Ce que la mesure d'énergie n'utilise PAS pour son chiffre principal — mieux que prévu

`self.joules` (`energie.py:186-193`) vient de
`nvmlDeviceGetTotalEnergyConsumption` — **un compteur matériel monotone**,
pas une intégration de puissances échantillonnées. C'est plus robuste que
« intégrer par horodatage » : aucune hypothèse sur la régularité de
l'échantillonnage n'entre dans ce chiffre, la différence de deux lectures
EST l'énergie (déjà documenté dans le fichier lui-même, `energie.py:8-13`).
`self.moyenne` (la puissance publiée dans `resume()["watts"]`) est
`self.joules / self.duree` — dérivée du même compteur fiable, pas de
`statistics.mean(self.puissances)`.

**Conséquence à noter** : la liste `self.puissances` (échantillons
`puissance_w` pris toutes les `periode` secondes, `energie.py:183-186`)
**n'est utilisée dans aucune métrique publiée** — ni `joules`, ni `moyenne`,
ni `resume()`. Elle existe mais ne sert aujourd'hui qu'à... rien de mesuré
(seuls `horloge_min/max` et `temp_max`, dérivés d'autres listes du même
thread, sont publiés). Pas un bug, mais un angle mort : si quelqu'un
utilise un jour `e.puissances` en pensant lire un signal fiable, c'est un
échantillonnage à `power.draw` (1 s), pas `.instant` — cohérent avec le
reste, juste inemployé pour l'instant.

## Ce qui reste un vrai point d'amélioration — la fenêtre de repos

`repos(secondes: float = 10.0, ...)` (`energie.py:279`) — **10 s par
défaut**, sous le seuil de 30 s que Jérôme demande. Pire : mes propres
campagnes récentes l'ont appelée encore plus court —
`banc-horloge-decodage.py` : `repos(secondes=8.0)` (décodage) et
`repos(secondes=5.0)` (prefill). Le docstring du fichier dit déjà
« de préférence de même durée que la fenêtre mesurée » — mais rien n'impose
un plancher, et une fenêtre de repos courte donne une ligne de base plus
bruitée, qui se propage ensuite dans `joules_net = joules - moyenne_repos ×
durée` : un bruit sur `moyenne_repos` est multiplié par la durée de la
fenêtre mesurée, potentiellement grand.

**Proposition** : ajouter un plancher explicite dans `repos()` — refuser ou
avertir si `secondes < 30`, plutôt que d'attendre que chaque appelant s'en
souvienne. Je ne l'implémente pas ici (à sec, pas de test à faire tourner
avant la carte) ; je le signale pour que la prochaine campagne (la mienne
incluse, si je refais un balayage d'horloge) corrige ses propres appels.

## Ce qui n'est pas dans ce document

Pas de mesure sur carte ici — uniquement une lecture de code et de la
documentation officielle des champs `nvidia-smi`. La vérification
« power.draw vs .instant » ne nécessitait pas de carte, contrairement à ce
que la question aurait pu laisser supposer.
