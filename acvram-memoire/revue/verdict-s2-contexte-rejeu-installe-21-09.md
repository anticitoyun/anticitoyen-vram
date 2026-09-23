# S2-contexte rejoué sur le paquet installé (0.6.34) — 21-22/09 (Manon)

* instrument : `scratchpad/s2-contexte-20-09/chaine.sh` (3 prises), worktree `manon-665eeacc-21-09` pinné exactement au commit du `.deb` installé (`665eeacc`), `PYA -m acvram.cli serve` — équivalent au binaire installé, « le code ne change pas entre les deux »
* commit : 665eeacc (arbre modifié : 11 fichiers non suivis, images-20 seulement) ; dpkg `acvram 0.6.34`
* régime : `outils/carte.sh` par prise (16 alias texte en 2 lots + 3 vision), chauffe de contexte au chargement + garde budget KV
* scellé : `ctx−64` acceptée (1 jeton décodé) ; `ctx+64` → 400 nommé ; 0 × 500 silencieux sur 19
* mesuré : **12 TENU** / 5 DEFAUT-CHAUFFE nommés (GLM-k48-calibA, GLM-Flash base, GLM-Grande, Qwen3.5-35B, Qwen3.8-27B-calibA) / **2 DEFAUT chauffe-clampée** (Qwen3-Coder-i8c ctx_tenu=11264<15360, connu et tranché par la Maîtresse ; Kimi-Linear ctx_tenu=14336<15360, même mécanisme, nouveau cas) — **0 REFUS budget KV** et **0 « 500 malgré chauffe tenue »**
* verdict : **MIEUX que la référence 20/09** (`verdict-s2-contexte-rejeu-20-09` : 9 TENU / 8 DEFAUT-CHAUFFE / 1 REFUS budget KV / 1 DEFAUT réel GLM k48 OOM) — `gemma-4-31B-it-nvfp4-vision` (ctx=262144 colonne fautive le 20/09) sert maintenant `ctx=4096` et **TENU** : colonne corrigée entre-temps ; **GLM k48 ne reproduit pas son 500 OOM du 20/09** aujourd'hui (TENU). 0 × 500 silencieux tenu sur les 19, comme scellé.
* durée : prise 1 = 8 alias (17:56-18:56), prise 2 = 8 alias (18:56-19:11), prise 3 = 3 vision (01:48-01:50, après la reconversion 31B libérant la carte) ; carte propre à chaque prise (seul PID 4453 étranger permanent)

## Suite
Pièce (3) de la Maîtresse close. Kimi-Linear (chauffe clampée à 14336 sur 15360 demandé) à nommer par Océane comme les autres DEFAUT-CHAUFFE — même mécanisme, pas de diagnostic supplémentaire ici.
