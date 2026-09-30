# kv31b — correctif de la réserve de préfill : terme d'attention par blocs de lignes, pic mesuré par la chauffe (poste6, 30/09 soir, branche poste6-reserve-attention, à sec — TESTS ÉCRITS, NON JOUÉS : aucun pytest pendant la campagne de poste2)

instrument : chiffres de la preuve carte du jour (journal de la prise 16:31 : 6,59 Gio libres avant la passe de chauffe à 20 480, 1,72 après → pic ≈ 4,87 Gio ; formule 2,70) ; réplique gemma-4-31B à sec
commit : 3de55de7e (sur 301b60449) ; `config.py` (`activations_prefill_bytes`), `loader.py` (`enregistrer_chauffe`, `lire_chauffe`, `_exces_mesure`), `contexte.py` (mesure du pic), `regime.py`/`cli.py` (ACVRAM_CHAUFFE_CACHE hors régime), `tests/test_reserve_attention_kv31b.py`
régime : à sec ; les 3 tests seront joués demain avant tout push définitif (règle de la nuit : aucun pytest tant que la campagne p275 tourne, fin prévue 22 h 50-04 h 50)
scellé : prédictions ci-dessous, écrites avant les tests et avant la carte
verdict : (en attente des tests, puis de la carte)
durée : 0 min de carte

## Ce qui a été retenu (chef, duck.ai 30/09 nuit : profile_run de vLLM saute l'attention, 150 Mio fixes → pas une référence ; notre chauffe passe la vraie attention)
* La FORMULE garde la main et gagne le terme manquant : scores/masque d'attention par blocs de lignes (layers.py cqy, ≥ 1 024 lignes) en fp32 :
  `têtes × min(T, 1 024) × T × 4` = 128 Kio/jeton sur gemma-4-31B (32 têtes) ; en bf16 (64 Kio) il ne couvrirait pas les 111 Kio/jeton mesurés.
  Hypothèse nommée (fp32) — c'est la chauffe qui la confirmera par modèle.
* La CHAUFFE mesure le pic (`torch.cuda.max_memory_allocated` autour de la passe tenue, alloué et non réservé), le dit au journal
  (« pic transitoire du préfill X Gio à N jetons (Y Kio/jeton ; formule Z) — excès E enregistré »), l'écrit dans `~/.cache/acvram/chauffe/<modèle>.json` ;
  au chargement suivant `_reserve_prefill` ajoute max(0, pic − formule)/jeton × T et le dit — la mesure vérifie la formule, ne la remplace pas.
* llama.cpp (Q2) : tampon dimensionné au pire cas puis échec nommé = notre « fenêtre qui tient » + refus nommé.

## Prédictions (avant tests, avant carte)
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| T1 | formule(20 480, plafond 5 120) avec le terme | 5,2 Gio ≥ pic mesuré 4,87 ; ≤ 1,5 × | < 4,87 (terme trop court) ou > 7,3 (trop large) |
| T2 | fichier de mesure : excès enregistré = (pic − formule)/jetons, ajouté à la réserve du chargement suivant, rien si pic ≤ formule | tenu | — |
| T3 | réplique 30,7 Gio libres, 31 744 : refus, fenêtre qui tient | 24 576-28 672 (31 744 sans le terme, démenti par la chauffe) | hors de la plage |
| C1 (carte, demain) | gemma-4-31B à 31 744 : refus + « fenêtre qui tient » ≈ 27 000 ; relance à N | chauffe prouve ≥ N (KV dimensionné pour N) et journal « pic … formule … excès 0 » | chauffe clampe sous N (formule encore courte : l'excès enregistré dit de combien) |
| C2 (carte) | Devstral-24B à 32 768 (témoin a5v) | inchangé : plafond posé, chauffe ≥ 32 768 | clamp → le terme fp32 sur-réserve un modèle qui tenait : à revoir |
Issues : (a) le terme sur-réserve les petits modèles (Devstral C2) → passer au bf16 ou au mesuré ; (b) `max_memory_allocated` sous-estime le pic
réservé (fragmentation) : l'excès enregistré serait trop petit — comparer aussi au réservé (mem_get_info) ; (c) plafond MLP différent entre la mesure
et le chargement suivant : l'excès par jeton reste à peu près valable (l'attention domine), dit dans le fichier.

## Reste
Jouer les 3 tests (demain matin, à sec) ; C1-C2 sur carte dans la fenêtre avec S1 de l'étape 1 ; suite complète avant fusion (chef).
