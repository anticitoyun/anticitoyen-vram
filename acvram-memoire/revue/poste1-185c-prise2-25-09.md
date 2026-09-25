# 185 c prise 2 — verdict (poste1, 25/09) : grille compacte des segments GDN au bit, −1,24 µs/appel (seuil 1,5) → (c) clos ; 185 c sans code au bit

* instrument : `scratchpad/poste1-p185c-25-09/banc-segments.py` (pile int8 [10240 ‖ 6144, 5120], `_segments` (10240, 6144),
  L2 froid, graphe, 60 rejeux), `prise2.sh` → `segments.jsonl`, `prise2.txt`
* commit : 87fda7b7 (poste1-185c), ATTENDU vérifié par `prise2.sh`
* régime : carte 0 seule, -lgc 2700, b = 8, défaut (4 warps / 3 étages), cpu-safe 100 début/fin, compute-apps début = fin (4242 seul)
* scellé : `scratchpad/poste1-p185c-25-09/scelle.md` § prise 2 (87fda7b7, après la prise 1, avant celle-ci)
* mesuré : V0 servi 60,26 µs ; C1 59,02 ; C2 60,70 ; C3 62,10 ; au bit 3/3 + servi = deux appels
* verdict : au bit TENU ; gain FAUX sous le seuil (C1 −1,24 µs < 1,5 ; 0,060 ms/pas < 0,07) ; C2 (vides retirés) FAUX, +0,44
* durée : prévu ≤ 3 min ; tenu 11:22:00-11:22:03, après 1 444 s d'attente du verrou

Lecture : les 160 programmes vides ne coûtent rien (C2 ≥ V0) ; seul l'ORDRE compte, 3,1 µs entre plus longs d'abord (C1) et
plus courts d'abord (C3) — la queue de la dernière vague. C1 est dans ma fourchette (−1 à −3) mais sous le seuil scellé : pas de code.

## Bilan 185 c (ordre de chef : diagnostic du déséquilibre, épilogue déroulé, grille compacte)
| levier | au bit | gain | décision |
|---|---|---|---|
| (a) H, charge du SM le plus chargé | — | R 0,799 indécise ; 2 prog/SM exacts −5 % à octets +6 % | levier = partition K → hors bit (± 1 ulp opt-in) ; tranches : 193 poste6 |
| (b) épilogue déroulé | oui | −0,5 à −1,3 µs (plus lent) | clos |
| (c) grille compacte des segments | oui | −1,24 µs/appel, 0,06 ms/pas | clos (seuil 1,5) |
Reste du coût fixe : plancher 3,3-4,0 µs par appel (V4, prise 1) × 193 appels ≈ 0,65-0,75 ms/pas — se prend en
lançant moins (fusion d'appels, noyau persistant), pas en réordonnant un appel.
