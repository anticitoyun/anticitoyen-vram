# Sage — AWQ sur experts peu routés : « le corpus garantit » n'est pas un contrôle ; le contrôle est un rapport de norme par tenseur, à sec, sur Coder et GLM d'abord, puis tout le parc (17/09)

Entrée : Manon 2a68aa6 — 6/346 `experts.*.down_proj` de Nemotron calibA à ratio de norme 0,29-0,69 (act_scale étendu 2,4e5-1,7e6× contre ~630× sain) : `search_channel_scales` clampe chaque valeur, jamais l'étendue entre canaux ; `experts_sans_stats` ne comptait que `n_samples == 0`. Correctif : `MIN_ECHANTILLONS_AWQ = 8`, identité en dessous. Expert partagé écarté (70 tenseurs byte-identiques).

## 1. Réponse : vérifier, pas supposer

Le bras A (16 k jetons) réduit la probabilité d'un expert sous-échantillonné, il ne l'annule pas — et **Coder n'a pas été converti avec le bras A** : son converti classé (`Qwen3-Coder-30B-A3B-nvfp4`, srcQ4_K_M) date du corpus par défaut de `collect.py:48-77`, six phrases ≈ 230 jetons répétées 16× : à 8 actifs sur 128, c'est ~14 jetons *distincts* par expert en moyenne, donc des dizaines d'experts sous 8. Qu'il soit classé à 1,0148 dit seulement que ses experts malades, s'il en a, sont peu routés sur le privé aussi — sur Nemotron, 6 experts malades ont suffi pour 1,43. Un converti classé avec des tenseurs corrompus est une bombe à retardement : le premier corpus qui route dedans la déclenche.

## 2. Le contrôle, à sec, sans reconversion (Manon, ~20 min par modèle)

Pour chaque tenseur d'expert (gate/up/down) du converti : **ratio ‖W_nvfp4 déquantifié‖ / ‖W_source‖** (source bf16 ou GGUF déquantifié, tous deux sur disque). Sain : 0,95-1,05 (l'arrondi NVFP4 conserve la norme à ~1 %). **Un seul seuil : ratio hors [0,80 ; 1,25] ⇒ tenseur malade.** Sortie : liste (couche, expert, projection, ratio) + histogramme.
* Coder classé — **prédiction : ≥ 1 tenseur malade** (corpus de six phrases). Faux : 0.
* GLM `-k48-calibA` (bras A, 64 experts) — **prédiction : 0**. Faux : ≥ 1.
* Le même script sur tout le parc converti avec AWQ (à sec, CPU, en tâche de fond, plusieurs heures) : un rapport par converti dans le verdict, pas une ligne par tenseur dans les menus.

## 3. Si Coder a des tenseurs malades

Reconversion Coder avec correctif + **bras A** (à sec, ~1 h), même chaîne que GLM : PPL 3 tranches (Laure, 20 min, après la GEMV experts) — prédiction 1,0148 → 1,010-1,014 ; **seuil unique ≤ 1,013 ⇒ remplace la colonne Coder** (et redevient la référence de toutes les cellules vitesse : à remesurer b=12 et b=1 sur le nouveau converti dans la même fenêtre, sinon table incohérente — 20 min de plus) ; > 1,013 ⇒ colonne inchangée, convertis malades retirés du disque quand même (règle 9 : on ne sert pas un tenseur faux, même s'il ne se voit pas).

## 4. Deux tests permanents (Laurine ou Manon, dans le commit du correctif s'ils n'y sont pas)

* `test_awq_experts_peu_routes` : expert à 3 échantillons ⇒ échelles = identité ; bras cassant : `MIN_ECHANTILLONS_AWQ = 0` ⇒ rouge sur un motif extrême synthétique.
* **Garde à la conversion** : ratio de norme calculé pour chaque tenseur quantifié, conversion **refusée** (pas avertie) si un ratio sort de [0,80 ; 1,25] — c'est le contrôle du § 2 rendu automatique, et c'est lui qui aurait vu 1,43 avant la carte. Manifeste : `ratio_norme_min/max` par converti.

## Ordre

Manon : § 2 Coder + GLM (40 min) → § 4 garde → parc en fond → § 3 si Coder malade. Reconversions Nemotron/GLM en file : inchangées. Laure : GEMV experts d'abord, puis PPL GLM, puis Coder si § 3.
