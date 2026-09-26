# Pièce 87 bis — σ intra-serveur, ABBA L/N à b=12 (poste2, 23/09)

instrument : `scratchpad/banc-llamacpp-16-09.py` decode, `BANC_SLOTS` avec répétitions de 12 (5 par
  occurrence) pour obtenir des fenêtres de 10 s indépendantes au lieu d'une seule ; `/metrics` toutes
  les 2 s ; `scratchpad/poste2-p87bis-23-09/prise.sh` — écrit uniquement dans son propre dossier
commit : 7cbd8aa5
régime : -lgc 2700, plafond 400 W, max-batch 12, max-model-len 2 304, défaut MAX_GRAPHS=64,
  alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c
scellé : ABBA L1,N1,N2,L2 (2 occurrences par bras, serveur redémarré à chaque occurrence, L après
  ramp 1,2,4,8) ; 5 fenêtres b=12 par occurrence → 10 mesures indépendantes par bras ; critère ±3 %
  entre moyennes L et N ; σ (échantillon) publié par bras (`scratchpad/poste2-p87bis-23-09/scelle.md`,
  avant la prise)
mesuré : L (n=10) moyenne 1 775,8 t/s, σ 71,3 (CV 4,01 %) — L1 [1797,4 1669,5 1800,5 1802,3 1835,9],
  L2 [1693,0 1676,5 1790,8 1812,0 1879,6]. N (n=10) moyenne 1 799,9 t/s, σ 55,0 (CV 3,06 %) —
  N1 [1725,5 1811,9 1803,5 1843,8 1808,2], N2 [1776,2 1700,9 1890,9 1827,8 1810,1]. repli_eager=0
  aux 4 occurrences ; graphes_nombre 22 (L1, L2) contre 13 (N1, N2), cohérent avec 87 (22 vs 12).
verdict : critère TENU cette fois — écart des moyennes L vs N = −1,34 %, dans la bande ±3 %. σ mesuré
  (4,01 % / 3,06 %, CV 3-4 %) confirme l'ordre de grandeur donné par poste1 (4,5 %) : le −6,72 % de la
  pièce 87 (une seule fenêtre de 10 s par bras, donc un seul lot effectif) était compatible avec le bruit
  intra-serveur, PAS un effet du service à lots successifs. La pièce 87 est donc invalidée par sa propre
  méthode (un seul tirage ne suffisait pas) ; le reste (a) de la pièce 85 (L sous N à b=12, même séance)
  N'EST PLUS établi — à rouvrir seulement si un écart réapparaît à ≥ 5 lots par bras. repli_eager=0 aux
  quatre occurrences (partie du critère, tenue dans les deux pièces).
durée : prévu ≤ 30 min / tenu 968 s (16 min 08, carte obtenue sans attente, journal `tenue=`)
