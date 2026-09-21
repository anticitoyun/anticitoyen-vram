# Levier 1 — échantillonnage glouton capturé dans le graphe CUDA (conception à sec, 21/09, poste1)

* instrument : `outils/gpu/mesure/frontiere-pas.py` (9d4e221c, non joué) — juge principal ; `certifie-b12` ABBA — juge secondaire
* commit de référence (témoin) : 665eeacc (`sampler=lent`, `pipeline=1`)
* régime : b=12 Coder nvfp4, ctx 2048, invite 256, éco 2700, graphes + pipeline ; b=1 GLM pour l équivalence
* scellé : § 5 ; code seulement après la frontière mesurée et sur go de la chef

## 1. Ce que fait le pas aujourd hui (lu, pas supposé)
Le graphe (`graphs.py:818` `step()` = `decode_fixed`) rend les **logits fp32** (`model.py:3292 _tete`, sortie fp32 par défaut).
Puis, hors graphe, `_sample_only` (runner.py:1670) → `_sample_lent` (sampler.py:170, défaut en service) : `logits.to(float32)` (no-op sur fp32), `argmax`, `gather`, `logsumexp`, soustraction — **4 lancements hôte**, ~55 µs carte (banc 4e9c76ff) ; puis `_consommer` (runner.py:1700) : `tokens.tolist()` **et** `logprobs.tolist()` — **deux rapatriements synchrones**. Le pipeline lance le rejeu n+1 (`_pipeline_suite`) AVANT de consommer n : les tenseurs de sample sont des allocations neuves, donc sûrs.

## 2. Ce qui entre dans le graphe
`step()` rend `(logits, sortie)` où `sortie` est un tampon statique int64 `[2, b_max·ql]` écrit dans la capture :
`l32 = logits.to(float32)` · `ids = l32.argmax(-1)` · `lp = l32.gather(1, ids[:, None]).squeeze(-1) − logsumexp(l32, −1)` · `sortie[0] = ids` · `sortie[1] = lp.view(int32).to(int64)` (bits fp32 transportés tels quels, dépaquetés côté hôte par `struct`/`view`).
**Même suite d opérations, mêmes noyaux torch que `_sample_lent` ligne 190-193** : ids et logprobs au bit par construction (argmax = premier indice du max dans les deux cas, logsumexp identique sur les mêmes fp32). `entry["out"]` (logits) reste rendu : chemin spéculatif, logprobs top-k, pénalités inchangés.

## 3. Ce qui reste hôte
* Éligibilité par lot, calculée à l amorçage et à chaque recomposition (déjà les moments où le pipeline repose l état) : `not besoin_historique(params)` — exactement la branche que `_sample_lent` prend ligne 190. Sinon (température > 0, pénalités, historique) : `_sample_only` sur `entry["out"]`, inchangé.
* `_sample_only` éligible : **un** `clone()` du tampon `sortie[:, :b_reel·ql]` (1 lancement au lieu de 4) — obligatoire, pas une marge : le rejeu n+1 réécrit le tampon statique avant que n soit consommé (§ 1). Un tampon épinglé écrit par le graphe éviterait ce lancement mais ferait de la lecture hôte une course de 7,8 ms contre le rejeu suivant : refusé (invariant > marge).
* `_consommer` : **un** `.tolist()` sur le clone `[2, n]` (un rapatriement au lieu de deux), dépaquetage ids / logprobs en python.
* Régime imprimé : `sampler=lent|lot|graphe` sur la ligne (`regime_ligne`, runner.py:786), opt-in `ACVRAM_SAMPLER_GRAPHE=1` jusqu au verdict ; `pipeline=0` ou eager (`preparer` → False) : chemin d aujourd hui, jamais le tampon.

## 4. Invariants et tests (même commit que le code)
1. **ids au bit b=1 et b=12** contre 665eeacc, test carte existant (`test carte b=1/12`, 030f0ab9) rejoué sous `ACVRAM_SAMPLER_GRAPHE=1` : ids ET logprobs égaux (`torch.equal`), 3 graines, lot mêlé glouton/échantillonné (les lignes échantillonnées passent par l ancien chemin sur les mêmes logits).
2. Test à sec (`ACVRAM_GRAPHS_EAGER`, `entry["step"]`) : le tampon `sortie` vaut `_sample_lent(entry["out"])` au bit, y compris les fantômes du godet (lignes > b_reel ignorées, jamais lues).
3. Test cassant : sans le `clone()`, un rejeu enfilé avant la lecture doit rendre les ids du pas suivant — le test le force (deux `rejouer_suivant` puis lecture) et doit rendre FAUX si le clone disparaît.
4. Test `regime_ligne` : `sampler=graphe` seulement si graphes actifs ET opt-in ; le dump `ACVRAM_DUMP_MOE` et le chemin spéculatif ne voient aucun changement (ids égaux sur le 2B CPU, à sec).

## 5. Gain plafonné, prédiction, seuil (écrits avant la mesure)
Plafond : ce qui disparaît est **au plus** 55 µs carte (échantillon) + les lancements hôte de 4 noyaux + 1 rapatriement — soit ≤ 55 + ~40 + ~15 ≈ **110 µs sur un pas de ≈ 7 800 µs = 1,4 %** ; rien d autre ne bouge (le graphe garde ses 7,3 ms). Le gain réel dépend de la frontière mesurée :
* si `trou_gpu` ≥ 150 µs (carte oisive, hôte limitant) : la carte gagne 55 → ~35 µs sur l échantillon (noyaux enchaînés dans le graphe, sans latence de lancement) et l hôte ~55 µs (3 lancements + 1 rapatriement en moins) → **prédit +0,6 à +1,0 % de t/s à b=12** (1 551 → 1 560-1 567), J/jeton −0,5 à −1 % ;
* si `trou_gpu` < 100 µs (carte déjà saturée par le graphe) : gain = seulement la part carte, ≤ 20 µs → **≤ +0,3 %**, levier inutile seul, on le dit.
Réfuté si : frontiere-pas sous `ACVRAM_SAMPLER_GRAPHE=1` ne baisse pas `echantillon + trou_gpu` d au moins **30 µs** (médiane, 300 pas pleins) ; ou ids/logprobs non égaux au bit (défaut, pas un réglage) ; ou ABBA t/s B/A < 1,000 (le levier coûte).
Alarme : un gain > 1,4 % mesure autre chose (ordre A/B, horloge) — je le dis avant de le publier.

## 6. Protocole
1. `frontiere-pas.py` A (lent) puis B (graphe), même processus impossible (opt-in à l import) → deux processus, A B B A, 300 pas pleins chacun, ≤ 3 min chaque : juge principal = `echantillon`, `echant_hote`, `trou_gpu`, `consommer` (µs, résolution < 5 µs).
2. `certifie-b12` ABBA 20 s × ≥ 8 paires (dispersion intra-B 50 t/s connue : 8 paires résolvent ≈ ± 8 t/s sur la différence), horloge SM médiane par fenêtre, rejet > 3 % d écart d horloge ; J/jeton contre le témoin de la même fenêtre.
3. Équivalence carte (§ 4.1) AVANT toute cellule. Rien de ceci ne se joue avant la frontière de poste2.

## 7. Amendement (relecture Vibe R3.6, verdict 3.6 de qr.md) — quatre pièges = quatre tests du commit du code
1. **Godet ≠ tampon** : le tampon `sortie` est dimensionné par godet capturé (`b_godet · ql`), la lecture par `b_reel · ql` ; test : lot réel 5 dans le godet 8 → 5 ids, les 3 fantômes jamais lus, et un godet 16 après un godet 8 ne rend pas les ids du 8.
2. **Séquence inéligible dans un lot éligible** : éligibilité par lot = `all(not besoin_historique)` ET aucune `is_done` ; une seule ligne à température > 0 ou à pénalité → lot entier par l ancien chemin sur `entry["out"]` ; test : lot mêlé (11 gloutons + 1 échantillonné) → ids au bit avec `_sample_lent` sur les 12 lignes, et la ligne finie du pas précédent est ignorée (fantôme du planificateur, `_consommer`).
3. **Course clone / rejeu** : le clone est enfilé sur LE MÊME flux, après les noyaux du graphe et avant le rejeu n+1 — l ordre du flux garantit la lecture avant l écrasement (invariant de flux, pas une marge) ; test : deux `rejouer_suivant` enfilés puis lecture du clone du premier → ids du premier, et sans clone → ids du second (doit rendre faux).
4. **Ordre ids puis logprobs** : un seul tampon `[2, n]` écrit par le graphe, un seul événement enregistré après le clone → il n existe aucun état où les ids sont lisibles et les logprobs pas encore ; test : `event.synchronize()` puis égalité des deux moitiés avec `_sample_lent` sur 100 pas.
**Alternative chiffrée — double tampon par parité du pas, sans clone** : un graphe capturé écrit à des adresses fixes, donc « deux tampons » = **deux captures par clé de godet** (parité dans la clé) : + une capture par godet (40-130 ms chacune, ~24 godets → +1-3 s de chauffe, pool partagé donc pas de mémoire en plus) ; la course disparaît par l invariant « au plus un pas en vol » du pipeline — un invariant **global** qu un futur pipeline à deux pas en vol casserait en silence. Le clone supprime la course par l ordre du flux, invariant **local** à la fonction, pour ≈ 10 µs D2D (≤ 0,13 % du pas). Jugé sur la suppression de la course : les deux la suppriment ; le clone la suppriment là où on la lit. **Retenu : clone** ; le double tampon reste nommé comme repli si la mesure de frontière montre que 10 µs comptent (elle ne le montrera pas : résolution 5 µs, gain visé ≥ 30).
