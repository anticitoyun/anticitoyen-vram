# Verdict — juge statistique de (b) contre le témoin graphes/eager (8 fenêtres × 3 bras) : **(i) tenu, (ii) tenu, (iii) FAUX sur une fenêtre (tr0-p256 : Δ_b = +0,043)** → pas de défaut ce soir ; « (iii) seul : on note et on cherche le préfixe » (Sage) ; Δ_b n'a pas de signe constant (3 +, 5 −), moy −0,0086

instrument : `scratchpad/juge-stat-b-18-09/chaine.sh` — `ppl-decode-kv-17-09.py` (corpus par `PPL_DECODE_CORPUS`), 1 024 pas notés, 8 fenêtres : tranches privées Coder 0/1/2 × préfixe {256, 2 048} + wiki-gptq × {256, 2 048} ; bras A v1 eager, B v1 capturé, C (b) capturé (disposition unique) ; 24 runs, 23:05-23:20, régime classé, charge 1-2, arbre 7dff172
scellé (Sage) : (i) |moy Δ_b| ≤ 2·sd/√8 ; (ii) |moy Δ_b| ≤ |moy Δ_témoin| ; (iii) aucune fenêtre Δ_b > 0,020 ; les trois → défaut des deux régimes ; (i) ou (ii) faux → alignement bit-exact ; (iii) seul → noter, chercher le préfixe ; prédiction Sage : tenu, moy Δ_b ± 0,004, |moy Δ_témoin| 0,010-0,015
mesuré (A / B / C ; Δ_témoin = B − A ; Δ_b = C − B) :
| fenêtre | A v1 eager | B v1 capturé | C (b) capturé | Δ_témoin | Δ_b |
|---|---|---|---|---|---|
| tr0-p256 | 16,5935 | 16,5934 | 16,6367 | −0,0001 | **+0,0433** |
| tr0-p2048 | 9,5157 | 9,5312 | 9,4862 | +0,0155 | −0,0450 |
| tr1-p256 | 17,1291 | 17,0629 | 17,0591 | −0,0662 | −0,0038 |
| tr1-p2048 | 11,0912 | 11,1381 | 11,1107 | +0,0469 | −0,0274 |
| tr2-p256 | 17,9041 | 18,1105 | 18,0758 | **+0,2064** | −0,0347 |
| tr2-p2048 | 11,1338 | 11,1140 | 11,0839 | −0,0198 | −0,0301 |
| wiki-p256 | 5,5444 | 5,5426 | 5,5506 | −0,0018 | +0,0080 |
| wiki-p2048 | 8,4581 | 8,4475 | 8,4681 | −0,0106 | +0,0206 |
n = 8 : **moy Δ_b = −0,0086, sd = 0,0309, 2·sd/√8 = 0,0218** ; **moy Δ_témoin = +0,0213** (de −0,066 à +0,206 : le témoin graphes/eager est bien plus dispersé sur le privé que sur wiki) ; **max Δ_b = +0,0433** (tr0-p256) ; wiki-p2048 aussi > 0,020 (+0,0206, à la limite)
verdict : **(i) TENU** (0,0086 ≤ 0,0218), **(ii) TENU** (0,0086 ≤ 0,0213), **(iii) FAUX** — deux fenêtres au-dessus de 0,020 (tr0-p256 +0,043 ; wiki-p2048 +0,021) → **pas de défaut** ; suite prescrite : noter et chercher le préfixe (les deux dépassements sont à préfixes différents, 256 et 2 048, sur deux corpus : rien ne désigne un préfixe) ; prédiction Sage : moy Δ_b −0,009 (hors ± 0,004 de peu), |moy Δ_témoin| 0,021 (au-dessus de 0,010-0,015) — le témoin lui-même dépasse 0,020 sur 3 fenêtres sur 8 (0,047, 0,066, 0,206) : **la condition (iii) telle qu'écrite est plus sévère pour (b) que ce que v1 se permet contre lui-même** ; à Sage : soit (iii) se lit relativement au témoin (|Δ_b| ≤ max|Δ_témoin| : tenu, 0,045 < 0,206), soit elle reste absolue et (b) échoue là où v1 échouerait aussi

## Lecture
- Le signe de Δ_b n'est pas constant (3 +, 5 −) : le « biais systématique » du verdict in situ était un artefact de 3 fenêtres wiki ; sur le privé (b) est même en moyenne sous v1 (−0,0086) — dispersion, pas biais.
- La dispersion graphes/eager sur les tranches privées (+0,206 sur tr2-p256) est un fait nouveau et plus gros que tout ce que (b) apporte : à consigner à part (REGLES §3), il concerne v1 et la cellule ppl-decode-kv en général — toute cellule ppl-decode-kv sur le privé porte ± 0,05-0,2 selon le mode ; le 5,5426 « témoin » wiki n'était représentatif que de wiki.
- La (iii) a arrêté le défaut ; le poste b=1 (350,9) reste à juger séparément (vitesse) — et il faut le trancher avant tout défaut de toute façon.
