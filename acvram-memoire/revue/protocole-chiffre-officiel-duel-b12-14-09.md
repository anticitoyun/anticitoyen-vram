# Protocole + prédiction — chiffre officiel du duel b=12 et l'écart 568/717/826

Laure, 14/09/2026. Suite à Jérôme : « le 568,6 t/s du duel vient de ton
banc-horloge (Engine direct, OK), mais il date d'avant le masquage
d'Océane et les correctifs du jour ; le banc de Laurine sur le même moteur
donne 826 j/s au pas, Océane 717. Trois bancs, trois chiffres : il faut UN
chiffre officiel. » Deux volets : (1) recalculer le chiffre défaut sur
`main` d'aujourd'hui (3b8f173+) ; (2) expliquer l'écart avec
`banc_decodage_moe` (826).

## Ce que la lecture du code montre AVANT mesure

`outils/gpu/mesure/banc-horloge-decodage.py` (source du 568,6 historique)
ne neutralise l'EOS nulle part — grep sur `_eos` : aucune occurrence.
`outils/banc_decodage_moe.py:34` : `eng._eos = set()`, explicite,
commentaire absent mais l'effet est clair : aucune séquence ne peut
terminer avant `max_tokens`. Sans cette neutralisation, un lot de 12
séquences à température 0 sur des invites SYNTHÉTIQUES (jetons pseudo-
aléatoires, `invite(k)` dans les deux scripts) peut faire émettre EOS à des
moments différents selon la séquence — dès qu'une séquence finit avant les
200 pas mesurés, le lot réel tombe sous 12 pour le reste de la fenêtre
(« traîne de fin de lot », l'hypothèse de Jérôme), ce qui réduit le débit
agrégé rapporté sur toute la fenêtre.

## Prédiction

1. La cellule « défaut » rejouée sur `main` d'aujourd'hui donnera un
   chiffre PROCHE de 568,6 (± bruit thermique/bridage, pas un écart de
   régime) — main n'a pas changé le chemin décodage b=12 lui-même depuis
   cette mesure (les correctifs du jour touchent MoE MMA, cp.async,
   masquage fantômes : aucun n'est actif par défaut ou ne change GEMV,
   sous réserve de vérification des flags par défaut au moment de la
   mesure).
2. La MÊME configuration (modèle, montage, script) avec `eng._eos = set()`
   ajouté donnera un chiffre NETTEMENT plus proche de 826 que de 568,6 —
   si l'écart mesuré dépasse un facteur ~1,3 entre les deux bras EOS
   actif/neutralisé, l'hypothèse de la traîne est confirmée comme
   explication PRINCIPALE de 568 vs 826 sur ce point précis.

## Ce qui réfuterait

Si le bras EOS-neutralisé reste proche de 568,6 (écart < 10 %), l'EOS
n'explique pas l'écart et il faut chercher ailleurs (graphes CUDA on/off,
`enable_prefix_cache`, `max_model_len`, modèle exact — `banc_decodage_moe.py`
utilise `Qwen3-Coder-30B-A3B-nvfp4` sans suffixe, à vérifier identique au
modèle du banc-horloge). Je le rapporterai tel quel.

## Montage

    modèle    Qwen3-Coder-30B-A3B-nvfp4 (identique aux deux scripts source)
    script    copie de banc-horloge-decodage.py, palier "defaut", un bras
              inchangé, un bras avec eng._eos = set() ajouté juste après
              la construction de l'Engine (avant warm_graphs)
    carte     exclusive, ACVRAM_TYPE=mesure, repos 30 s (energie.py)
    code      `main` d'aujourd'hui (anticitoyen-vram, 3b8f173 ou plus
              récent au moment de la mesure) — PAS le worktree laure, pour
              répondre explicitement à « sur main du jour »
