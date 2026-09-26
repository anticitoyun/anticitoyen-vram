# poste7 — Pause : oui ; trois restes à sec ou à 5 min de carte, aucun ne retient le circuit ; point de reprise nommé (18/09)

Entrée : chef — GUI (4 ajouts, 0.6.12) et chantier GEMV RPW/XREG clos, pairs idle, l'utilisateur demande si c'est fini.

## 1. Fini pour aujourd'hui

Cellule phare Coder b=12 : 997,8 → 1 198 t/s bridé (+20 %), 0,400 → 0,334 J/jeton (−16 %), 1 312 nu, bit-exact, régime prouvé au défaut (`verdict-coder-b12-defaut-18-09`, main 23903b5). REGLES porte les trois lignes du jour (ptxas, `sudo -n ncu`, pip hors fenêtre — vérifié par grep). Pause confirmée.

## 2. Restes — non reçus à cette heure, à faire à la reprise, dans cet ordre (aucun ne dépend d'un autre)

| reste | qui | coût | source |
|---|---|---|---|
| contrôle `j_par_jeton_10s` ± 10 % de `energie.py` sur 10 s (le seul qui touche la carte) | poste3 | 5 min carte | `poste7-metrics-energie-fenetre-18-09` |
| canal massif GLM : 5/5 tenseurs ≤ 3 canaux > 100× médiane, même canal d'un converti à l'autre | poste2 | 15 min, rapports existants | `poste7-glm-etendue-canal-saillant-18-09` § 3 |
| `regime_ligne()` porte les versions torch/triton/fla (absentes, grep `regime.py`) + test | chef ou poste4 | 30 min à sec | même note § 5 |

REPRISE : piste `__launch_bounds__(256,4)` gateup avec les chiffres de poste3 (105,2 µs / 96 registres / 2 blocs), et `xreg=tout` témoin — à vérifier qu'elle y est.

## 3. Point de reprise (après les restes)

Priorité 1 inchangée depuis `poste7-priorite-apres-campagne-17-09` : **hybrides** (Qwen3.8 GDN 97 t/s → scellé ≥ 400, Nemotron Mamba2), fla en défaut déjà (`regime.py:127`) — ETAT dit le chantier clos avec gains réels ; ce qui reste à trancher à la reprise est le **plafond GEMV dense NVFP4 O(b)** (93 % du pas hybride) : c'est le même noyau-famille que ce qu'on vient de fermer, et rpw/xreg s'y appliquent-ils ? Question chiffrée pour poste3 au retour, 10 min : Qwen3.8 b=12 au défaut d'aujourd'hui contre la cellule publiée — si les défauts du jour n'y changent rien (< 3 %), le dense a son propre profil à écrire avant tout geste.

## Ordre

* chef : ETAT ≤ 40 lignes avec § 1 et le tableau § 2 ; pause. Rien d'autre.
