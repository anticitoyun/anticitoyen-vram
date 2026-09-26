# 194 (b2) — verdict (poste1, 25/09) : β‖α sur un second flux, mixte b=8 +2,20 % (z 8,3), J/jeton −1,9 %, au bit ; témoins nuls tenus

* instrument : banc chat de la 102 (`scratchpad/poste2-piece102-etalon-hf-24-09/banc-chat-openai.py`, `energie.py` par
  PYTHONPATH : ses `sys.path` « ~/… » ne se développent plus depuis la purge), `prise-abba.sh` (A B B A A B B A A B, serveur neuf
  par passe, fenêtre 20 s, arrêt au premier banc échoué) ; `outils/gpu/mesure/capture-godets.py` ; tests `tests/test_gdn_ab_flux_194.py`
* commit : 6370670b (poste1-194 = origin/main f3197f0b + 194 b2), ATTENDU vérifié par les scripts
* régime : A `ACVRAM_GDN_AB_FLUX=0`, B `=1` (ligne /metrics : `abflux` en B, absent en A, 20/20 passes), repli_eager=0 partout,
  -lgc 2700 (horloge moyenne 2 631 à b=8), ECO=off, cpu-safe 100 début/fin ; bridage puissance (400 W) dans les passes b=8 des deux bras
* scellé : `scratchpad/poste1-p194-25-09/scelle.md` § prises servies (162d389e, avant les ABBA)
* mesuré : ci-dessous ; capture godets {1, 2, 8, 16} en B 4/4 ; tests 74/74 (73 carte sous verrou + à sec)
* verdict : mixte b=8 TENU (≥ +1,5 %, z ≥ 2) mais SOUS ma fourchette (+2,5 à +4,5) ; mixte b=1 TENU ; nvfp4 b=8 et b=1 témoins nuls TENUS
* durée : prévu ≤ 22 + 18 min ; tenu 12:29:13-12:47:42 et 12:53:58-13:12:23 ; ABBA nuls de 12:27 (banc sans `energie`) écartés, `echec-energie/`

| cellule | A t/s (médiane) | B t/s | B/A | z (Welch) | J/jeton net A → B |
|---|---|---|---|---|---|
| mixte b=8 | 399,2 | **408,0** | **+2,20 %** | 8,3 | 0,8018 → 0,7866 (**−1,9 %**) |
| mixte b=1 | 65,1 (5/5) | 65,4 (5/5) | +0,46 % | séparés | 4,879 → 4,881 (+0,06 %) |
| Qwen3.8-27B-nvfp4 b=8 (flux inerte) | 554,4 | 554,3 | −0,02 % | −0,4 | 0,5513 → 0,5514 |
| Qwen3.8-27B-nvfp4 b=1 (flux inerte) | 79,4 | 79,4 | 0,00 % | — | 3,646 → 3,651 |

Lecture : +2,2 % au banc chat ≈ −0,45 ms/pas servi (préfill et service compris), contre −0,58 au banc noyau : mon alarme (« le
servi peut gagner plus ») allait dans le mauvais sens. b=1 : β‖α en concat cuBLAS 3,9 µs, gain ≈ +0,5 % comme borné. Témoins nuls
à 0,02 % : la séance et l'instrument ne bougent pas.

## Proposition (décision : chef)
`ACVRAM_GDN_AB_FLUX=1` AU DÉFAUT : au bit par construction et prouvé (test qui casse quand la jointure manque), capture 4/4,
gain servi +2,2 % b=8 / +0,5 % b=1, J/jeton −1,9 % b=8, inerte hors β‖α bf16. Bascule = `"0"` → `"1"` dans `gdn.py` et
`regime.py`, le test `test_la_variable_est_opt_in` inversé dans le même commit.
