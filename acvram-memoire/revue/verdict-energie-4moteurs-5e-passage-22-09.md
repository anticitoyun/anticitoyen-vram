# Banc énergie 4 moteurs, 5ᵉ passage (sha 8cd5ef07) — les 4 moteurs mesurent, llamacpp seul TENU au protocole strict — 22/09 (poste2)

* instrument : `chaine-energie-4moteurs.sh` sha `8cd5ef07` (garde de capture poste1 + nettoyage poste4 fusionnés, main `2b83f1a3`/`77bc7991`), après preuve isolée réussie (vLLM+trtllm b=12, verdict séparé) et mode service déclaré (~35 min)
* commit : `fec21ba0`
* mesuré (`protocole25.tsv`, 12 fenêtres, **aucun hang cette fois** — les 4 moteurs ont chargé et mesuré, 1re fois de la journée) :

| moteur | fenêtres valides (protocole) | j/jeton médian (valides) | t/s médian (valides) |
|---|---|---|---|
| **llamacpp** | **2/3** | **0,2055** | **1053,9** |
| vLLM | 1/3 (b=1 seul) | 0,8088 | 305,4 |
| acvram | 0/3 | — | — |
| trtllm | 0/3 | — | — |

* **acvram, chiffres bruts cohérents avec la prédiction malgré le rejet protocole** : b=12 1650,9/1603,9 t/s, 0,1943/0,1989 J/jeton net — **dans la fourchette prédite 0,19-0,21 J** ; b=1 392,1 t/s, 0,4482 J/jeton — **dans la fourchette prédite 0,40-0,48 J**. Rejetés uniquement pour bridage puissance (b=12×2) et sd>10% (b=1) — **le critère « bridage accepté avec watts_moy » annoncé par le groupe n'est toujours pas actif dans `resumer-energie-4moteurs.py`** (colonne `watts_moy` désormais affichée, 247,6-400,5 W, mais ne change pas la décision `valide`) — 3ᵉ constat identique, à vérifier où vit réellement ce correctif.
* **vLLM b=12 chiffres bruts plausibles** (1858,3/1903,6 t/s, 0,1639/0,1616 J/jeton net) mais rejetés pour bridage, même raison qu'acvram.
* **trtllm** : b=12 T1 mesuré (2036,4 t/s, 0,1539 J/jeton net) mais rejeté bridage ; **T2 absent** (« serveur jamais prêt », **3ᵉ occurrence du même motif récurrent** — timeout au 2ᵉ chargement trtllm, vu aux 3ᵉ et 5ᵉ passages, jamais diagnostiqué) ; **b=1 signal frappant** : 46,3 t/s seulement (contre 305-392 t/s pour les 3 autres moteurs à b=1) et 7,2645 J/jeton net (×15-35 les autres) — **cohérent avec l'effondrement trtllm à b=1 déjà nommé par poste3** (`verdict-cellules-trtllm-22-09` : b=1 46 t/s contre acvram 390) — confirmation indépendante par un instrument différent. Fenêtre invalidée en plus pour puissance moyenne 400,5 W > plafond 400 W (fenêtre trop courte pour le limiteur, artefact de moyennage, pas une erreur de mesure).
* verdict : **llamacpp seul TENU au protocole strict** (3ᵉ mesure indépendante convergente : 0,2055/0,2064/0,2157 J/jeton, 1054/1044/1023 t/s selon les passages). Les 4 moteurs ont désormais des chiffres bruts publiables, mais 3 restent sous le seuil de rejet protocole pour des raisons distinctes et nommées (bridage puissance à 400 W dès que le débit est élevé — structurel sur ce poste, pas un défaut d'instrument ; T2 timeout récurrent ; trtllm b=1 réellement dégradé, pas un artefact).
* durée : ~39 min de carte (15:15-15:54, mode service tenu)

## Suite
Le plafond 400 W du poste rend le protocole strict quasi impossible à satisfaire pour tout moteur qui approche le débit maximal (acvram, vLLM, trtllm y arrivent tous les trois à b=12) — à trancher par le groupe : assouplir réellement le critère bridage (le correctif annoncé mais absent), ou accepter que seul llamacpp (plus lent, jamais au plafond) puisse être « TENU » sur ce protocole. T2 timeout à diagnostiquer si un 6ᵉ passage est voulu. Carte rendue à poste3.
