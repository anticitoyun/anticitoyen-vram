# Scellé d19 qualité — au-delà du tenu sur Devstral-24B : ΔNLL, KL et top-1 contre la passe dense, PPL à 65 536 (poste6, 01/10, écrit AVANT la carte, ordre chef : option (a) servie étiquetée `morceaux>N` si la qualité tient, sinon (c) 400)

Branche poste6-d19 (f141ed8d8 + `forward_tranches(return_hidden)` pour l'instrument, test J4 au bit sur le jouet). Verdicts relus de la même
famille : `poste1-d19-verdict-27-09` (prise 5 : MoE/GDN par tranches contre seul tenant, corpus 5909d27, 2 × 16 384 — PPL 12,8661 → 12,8659,
ΔNLL −0,000019 ± 0,000871, KL moyen 7,3e-3, KL max 2,80, top-1 95,96 % : FAUX au seuil 99 %, servi quand même par décision du chef, étiqueté) ;
`poste6-d19-verdict-01-10` (C2 : Δ logprob 0,043 à la position 0, témoin reprise 0,004, relecture 0,090, masque 0,0215) ; S1 bis (0,041) ; lic.

## Instrument — `scratchpad/d19-qualite.py` (d'après `nll-kl.py` d'poste1, même tête `_tete` + `_logits_finaux` par tranches de 512 lignes)
* A : passe dense d'un seul tenant, `model(batch, return_hidden=True)` (le chemin de `acvram eval`) ; états bf16 enregistrés.
* B : le chemin servi au-delà du tenu — lots de D19_MORCEAU = 4 096 jetons (`morceau=(j, n, L, True)`) par `forward_tranches(…, return_hidden=True)` :
  K/V transitoires + biais bas-droite (défaut) ; variantes `ACVRAM_PREFILL_BIAIS_MORCEAUX=0` (masque dense) et `ACVRAM_PREFILL_TRANSITOIRES=0`
  (relecture int8, l'ancien S1). Sur le jouet (test J4) B = A au bit ; sur carte flash ≠ math (lic).
* Corpus REGLES § 3 (`revue/*.md` à 5909d27, 556 264 jetons Devstral), fenêtres L ∈ {16 384, 24 576} (≤ tenu 25 600 : A existe) aux offsets
  0 et 20 000 → 2 × (L − 1) positions appariées ; P : B seul à 65 536, offsets 0 et 100 000, PPL et NLL par tranche de 8 192 positions.
* Alias `acvram-devstral-24b-srcawq-nvfp4`, Engine direct (cuda:0, kv int8, max_concurrent_seqs 1), un processus par mode sous carte.sh
  (`scratchpad/poste6-d19-qualite.sh`), pause coopérative e50.2, nvidia-smi au début et à la fin.

## Prédictions (seuils dérivés d'poste1 d19 — KL ≤ 1e-2 FAUX, |ΔNLL| ≤ 2 SE, top-1 ≥ 99 % — et de C2 : Δ 0,043 sur UNE position quasi-égale, l'écart moyen est plus petit)
| # | contrôle | prédit | TENU si | FAUX si (→ option (c), 400 au-delà du tenu) |
|---|---|---|---|---|
| Q1 | B défaut contre A, L = 16 384 et 24 576 (chacune 2 fenêtres) | ΔNLL dans ± 1 SE ; KL moyen **2e-3 à 8e-3** nats ; top-1 **97-99 %** ; KL max ≤ 3 | KL moyen ≤ 1e-2 ET top-1 ≥ 97 % ET \|ΔNLL\| ≤ 2 SE (ou ≤ 0,2 % de la NLL) aux deux L | KL moyen > 2e-2 OU top-1 < 95 % OU (\|ΔNLL\| > 2 SE ET > 0,2 %) à l'une des deux L |
| Q2 | B masque dense (biais off) contre A, 16 384 | KL ≈ ½ de Q1 (C2 : 0,0215 contre 0,043), top-1 ≥ Q1 | KL_masque ≤ KL_défaut | KL_masque > KL_défaut : le biais ne coûte rien, le garder sans débat |
| Q3 | B relecture int8 (transitoires off) contre A, 16 384 | KL ≈ 2 × Q1 (C2 : 0,090), top-1 ≤ Q1 | KL_relecture ≥ KL_défaut | KL_relecture < KL_défaut : les transitoires n'aident pas à l'échelle du corpus |
| Q4 | P : B à 65 536, PPL (offset 0) contre PPL_A à 24 576 (offset 0) | PPL_B(65 536) ≤ PPL_A(24 576) (contexte plus long) ; NLL par tranche de 8 192 sans montée au-delà de 24 576 | ≤ × 1,00 | > × 1,02, ou tranche [57 344, 65 536) > tranche [16 384, 24 576) × 1,10 (dérive des morceaux) |
| Q5 | durée | ≤ 14 min (7 chargements de Devstral depuis le disque chaud, ≈ 40 s chacun) | ≤ 16 | > 20 |
Issues nommées : (i) Q1 FAUX → (c) : le serveur répond 400 au-delà du tenu comme avant, la pièce garde la réserve honnête et les morceaux
opt-in ; (ii) Q1 tenu de justesse (top-1 95-97 %) → zone grise comme le d19 d'poste1 (95,96 %) : décision du chef, chiffres publiés tels quels ;
(iii) KL max grand (> 3) sur quelques positions quasi-égales (C2 : −3,190/−3,194) : la moyenne juge, le max est dit ; (iv) P à 65 536 sans A :
la PPL seule ne prouve pas l'exactitude, elle exclut une dérive grossière — dit ; (v) la tête par tranches de 512 lignes est la même pour A et B,
seuls les états diffèrent ; (vi) Engine direct ≠ serveur (pas de chauffe, pas de graphes) : le chemin morceaux est le même code
(`forward_tranches`, transitoires, biais), le seuil est forcé par le script, pas par la chauffe.
