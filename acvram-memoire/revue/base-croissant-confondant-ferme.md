# `base_croissant` : le confondant « on promeut les mal quantifiés » est fermé

11/09/2026. Bras témoin construit avant la mesure (chef, 10/09) : trier par
SNR de base **croissant** en ignorant le coût, c'est-à-dire « les mal quantifiés
d'abord », et rien d'autre. Si une clé raffinée bat le défaut parce qu'elle
promeut simplement les tenseurs les plus dégradés, ce bras le fait aussi.

## Résultat — deux exemplaires, PPL identique

```
                    PPL       écart / 5,4141
B (ordre inverse)   5,4482    +0,63 %     ← meilleur
A (snr, défaut)     5,4918    +1,43 %
base_croissant      5,4981    +1,55 %     ← pire  (ex1 = ex2 = 5,4981)
```

Régime : Llama-2-7b, budget 6,00 Gio, wiki-gptq.txt (sha e529227…),
2048/2048/min_context 0, étalon 5,4141. `.venv/bin/python` (le défaut
`/usr/bin/python3` n'a pas `tokenizers`). 173 promus /225 candidats, dépense
5,9971 Gio, ordre du manifeste `snr_de_base_croissant_sans_cout`.

**Le confondant est fermé.** Promouvoir les pires-SNR d'abord ne reproduit pas
le gain de B — c'est le **pire** des trois, sous A même. L'avantage de B n'est
donc pas « on promeut les mal quantifiés ». Ce que B capture est autre chose que
la dégradation de base.

## Ce que cela ouvre — et ce n'est pas un bras parmi quatre

**B (ordre inverse de `gain_db/coût`) bat A (ordre `gain_db/coût`).** Donc
`gain_db` **ne prédit pas la PPL**, ou la prédit à l'envers, sur la queue où le
glouton puise. La question suivante n'est plus « quel bras » mais **quel agrégat
par tenseur prédit la PPL** — `agregat-qui-predit-la-perplexite.py`. C'est le
successeur direct de ce résultat, pas un item de liste.

## La réserve des 98 tenseurs sans `out_ref_norm` — dissoute

L'avertissement « 98 tenseurs sans échelle, la simulation `absolu` les
écartera » était trompeur. Les 98 sont **tous des non-candidats** :

```
65  normes (bf16, non quantifiées → pas de calibration, pas d'échelle)
32  rotary_emb.inv_freq (tables RoPE, pas des projections linéaires)
 1  embed_tokens (bf16)
```

Les **225 tenseurs linéaires quantifiés** (173 int8 + 52 nvfp4) — c'est-à-dire
tous les candidats du sac à dos — portent `out_ref_norm` **sans exception**.
`absolu` ne perd donc aucun candidat. La condition « savoir lesquels avant d'en
tirer un classement » est remplie : aucun.
