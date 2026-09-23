# Chantier C9-M1 — taux de succès d'experts sur trace réelle : instrument à sec livré, la trace Coder reste à prendre (Manon, 1 h carte)

Objectif (Sage, `sage-c9-119b-cache-experts-19-09` § 2, M1) : décider si un cache d'experts EXISTE avant d'écrire du code — `h_pin(C)`, `h_lru(C)`, C ∈ {16, 32, 48, 64, 96}, apprises/chauffées sur la première moitié des jetons, jugées sur la seconde ; experts distincts par pas.

Prédiction scellée (Sage, recopiée) : Coder b=1 `h_lru(64)` = 0,78 ± 0,08 (13/09) ; 119B `h_lru(43 = E/3)` 0,55 ± 0,10. Seuil : Δh = h(E/2) − 0,5 **< 0,10 → pas de cache apprenant** (exil par couche seul + 3080 Ti) ; **≥ 0,25 → cache engagé** ; entre : Sage tranche.

## Ce qui existe déjà (fichier:ligne)

* `acvram/memory/trace_routage.py` : journal texte `<jeton> <couche> <experts>` écrit par `noter()` (model.py:1833, un test de booléen hors trace) ; `taux_de_succes` (pool partagé, LRU/LFU, :149), `taux_de_succes_par_couche` (:199), `taux_de_succes_pin` (:250, appris 1re moitié, jugé 2e).
* `outils/trace-taux-succes-experts.py` : prise de trace (invites réelles, `ACVRAM_DISABLE_CUDA_GRAPHS=1`).
* **Défaut trouvé et corrigé** : `ACVRAM_TRACE_ROUTAGE` était lu par DEUX mécanismes — le journal texte et, depuis le 18/09, model.py:2196 (`torch.save(_ROUTAGES, chemin)` à la sortie, pour `banc-marlin-decode --routages`) : le `.pt` écrasait le journal texte d'une trace M1 à la fin du processus. Renommé `ACVRAM_TRACE_ROUTAGE_PT` (HORS_REGIME, liste de garde, outil mis à jour). Sans cela, la trace de Manon aurait été détruite à la sortie.

## Livré ce soir (à sec)

* `pas_de_decodage()` : un pas = rafale de lignes de même couche à rangs croissants (à b=12, `noter` écrit B lignes d'un coup ; la couche suivante réécrit les mêmes rangs) ; `distincts_par_pas()` : distincts/pas par couche (moyenne, médiane, max, lot moyen, recouvrement = 1 − distincts/demandes) — c'est ce qui se paie en octets PCIe, pas la somme des top_k.
* `taux_de_succes_lru_juge(chemin, C, entrainement, par_pas)` : LRU par couche chauffée sur la 1re moitié, jugée sur la 2e (mêmes jetons que `pin`), un expert demandé par plusieurs jetons du même pas = une demande.
* `rapport_m1(chemin, capacites, entrainement, nb_experts)` : h_pin/h_lru par C et par couche, distincts par pas, h(E/2), Δh, verdict Sage. `outils/rapport-m1-cache-experts.py trace.txt --experts 128 [--json]`.
* Tests (`tests/test_trace_routage.py`, 16 verts) : segmentation des pas, recouvrement (témoin b=1 = 0), LRU jugée (capacité 2 → 100 %, 1 → 0 %), par_pas (2 contre 6 demandes), rapport (routage concentré → « cache engagé » ; uniforme → non).
* Fumée sur une trace synthétique 48 couches × 400 jetons (routage à 80 % sur 24 experts chauds/couche) : h_pin(64) 0,889, h_lru(64) 0,891, Δh +0,39 — l'instrument rend ce qu'on lui donne ; **aucun chiffre réel**.

## Reste / non vérifié

* La trace Coder réelle (≥ 20 requêtes, ≥ 50 000 jetons, b=1 puis b=12, `ACVRAM_TRACE_ROUTAGE=<nom portant le régime>.txt`, graphes off) : Manon, sous verrou. Puis `outils/rapport-m1-cache-experts.py trace.txt --experts 128`.
* 119B : dès que le converti charge (Jérôme télécharge) — E=128 top-4, `--experts 128`.
* Non vérifié : le format d'un journal à b=12 en service réel (rangs `base + i`, model.py/trace_routage.py:104-112 lus, pas exercés sur carte).
