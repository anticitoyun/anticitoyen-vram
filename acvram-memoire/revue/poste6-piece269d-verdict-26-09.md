# Pièce 269 d — guet d'admission coupé pour les alias vision : verdict de confirmation (TENU) — 0.7.4

poste6, 26/09/2026, décision chef sur la mesure 276 (`poste6-piece276-verdict-26-09.md` § 2 : images FAUX, +10,2 ms de
mur, +9,6 de TTFT p50, +25 au p95, le guet au plafond de 20 ms à chaque tour). Branche poste6-269d, code **d0c66f46b**
(main 6342eda9b + 269 d), sous carte.sh (ACVRAM_NOM=poste6-269d, carte obtenue à 16:36:32 après 536 s ; une première
mise en file s'était abandonnée à 1 800 s derrière poste2-p275 puis poste1-p274).

**Quoi.** `EngineService.guet_actif()` = `_GUET_ADMISSION and not vision_servie()`, lu une fois par réveil du fil aux deux
sites de la fenêtre (porte à une requête en file, boucle) ; régime : « 269 d : sans effet pour un alias vision », défaut 1
et opt-out `=0` inchangés ; test cassant `test_269d_alias_vision_le_guet_est_coupe` (fabrique avec manifeste `vision: oui`
→ pas d'attente ; sans → la fenêtre ; `guet_actif` False/True) ; 0.7.4 (versions, metainfo, tables, CHANGELOG avec les
chiffres 276 et le non couvert).

**Tests sous carte.sh** (16:36 → 16:39) : 165 verts, 1 xfail connu (guet 269 b/c/d, 179, défauts, alias, 223, régime, cli, 268).

**Confirmation** — Qwen3-VL-2B-Instruct-bf16-vision, b = 4, une image PNG 448×448 par requête, A0 B1 B1 A0 (16:39:36 →
16:40:11), 7 tours à 4 + 5 solo par bras, instruments 262. Scellé (écrit dans `prise-269d.sh` avant la prise) : B = A ± 1 ms
(mur médian, TTFT p50), 0 tour de l'issue nommée, fenêtre B = fenêtre A, solo ± 1.

| | A = guet 0 | B = guet 1, coupé par 269 d | 276 (avant) | verdict |
|---|---|---|---|---|
| tours à un pas / profil | 0/14, [1, 3] | 0/14, [1, 3] | 12/14 (un pas de 4 après 20 ms) | identique à A |
| fenêtre (trace), médiane / max | 5,52 / 5,53 | **5,54 / 5,57** | 20,23 / 20,69 | **tenu** |
| mur médian | 108,2 | **107,0 (−1,2)** | 118,3 (+10,2) | **tenu** |
| TTFT p50 / p95 par requête | 104,0 / 112,1 | **105,0 (+1,0) / 110,2 (−1,9)** | 114,0 / 138,0 | **tenu, p50 à la borne** |
| issue nommée (fen ≥ 15 ms + pas de 1) | 0/14 | **0/14** | 12/14 | **tenu** |
| solo, médiane de 10 tours | 30,8 | 31,1 (+0,3) | 31,9 | tenu |

B est redevenu A : mêmes pas, même fenêtre à la dizaine de microsecondes, mur −1,2, p95 −1,9 ; le p50 par requête est à +1,0
exactement sur la borne du scellé — je le dis tel quel, la borne est tenue, pas dépassée, et le mur comme le p95 vont dans
l'autre sens (bruit de tour, pas un coût du chemin `guet_actif()` : un getattr et un `dict.get` par réveil). La régression
vision de la 0.7.3 est fermée ; le texte garde le défaut 1 (276 § 1).

**Décision.** Livrée en 0.7.4 (à chef : suite complète et tag). Non couvert, comme au 276 : alias vision servant du texte
seul (guet coupé pour lui aussi ; coût texte à b = 4 : 3 tours à deux pas sur 14), plusieurs images par requête, b = 12
avec images, 2e modèle (274). Fichiers : `scratchpad/poste6-p269d-26-09/` (client, analyses, prise), traces
`~/.cache/acvram/dumps-269d/` (hors git).
