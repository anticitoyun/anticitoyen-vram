# d19 qualité, carte — au-delà du tenu sur Devstral-24B : KL 2,9e-4 nats, top-1 99,2 %, ΔNLL dans 1 SE, PPL à 65 536 sans dérive — garde qualité TENUE, option (a) servie (poste6, 01/10 17:09-17:14, branche poste6-d19)

instrument : `scratchpad/poste6-d19-qualite.sh` → `scratchpad/d19-qualite.py` (nll-kl.py d'poste1 adapté : B = lots-morceaux de 4 096 par `forward_tranches(return_hidden=True)`, le chemin servi au-delà du tenu ; même tête `_tete` + `_logits_finaux` par 512 lignes), un processus par mode sous carte.sh (TYPE=mesure)
commit : poste6-d19 618d5f128 (HEAD asserté, provenance `…/poste6-d19/acvram/__init__.py`)
régime : Engine direct cuda:0, Devstral `acvram-devstral-24b-srcawq-nvfp4`, kv int8, max_concurrent_seqs 1 ; corpus REGLES § 3 `revue/*.md` à 5909d27 (sha256 relevé dans chaque journal), 556 264 jetons Devstral ; fenêtres aux offsets 0 et 20 000 (A/B), 0 et 100 000 (P) ; pause coopérative e50.2 17:00:50 → 17:14:05 (marque après 8 min 45 : alias en cours) ; nvidia-smi : 4436 (3080 Ti) seul au départ et à la fin
scellé : `poste6-d19-qualite-scelle-01-10.md` (Q1-Q5)
mesuré : 17:09:35 → 17:14:05, **4 min 30 de carte** (7 chargements chauds ≈ 20 s)
verdict : **Q1 TENU** (B défaut contre A : L = 16 384, n = 32 766 : PPL 10,0599 → 10,0582, ΔNLL −0,000171 ± 0,000143 (1,2 SE), KL moyen **2,86e-4**, KL max 0,95, top-1 **99,167 %** ; L = 24 576, n = 49 150 : PPL 9,0096 → 9,0100, ΔNLL +0,000051 ± 0,000123 (0,4 SE), KL moyen **2,66e-4**, KL max 0,106, top-1 **99,192 %**) — prédit KL 2e-3 à 8e-3 et top-1 97-99 % : mesuré 10 × plus petit et au-dessus, hors fourchette du bon côté, dit ; **Q2 FAUX** (masque dense : KL 2,78e-4, top-1 99,084 %, ΔNLL −0,000066 ± 0,000142 — égal au défaut dans le bruit, pas « ≈ ½ » : le biais bas-droite ne coûte rien à l'échelle du corpus, il reste, et il rend 8 Gio) ; **Q3 TENU** (relecture int8 : KL 4,44e-4 = 1,55 × le défaut, top-1 98,843 %, ΔNLL +0,000004 ± 0,000183 — les transitoires aident, moins que le × 2 prédit) ; **Q4 TENU** (B seul à 65 536, 2 fenêtres, n = 131 070 : PPL 6,5351 contre PPL_A 9,0096 à 24 576 (× 0,73) ; NLL par tranche de 8 192 sur la fenêtre 0 : 11,51 · 10,48 · 7,98 · 8,98 · 6,88 · 7,58 · 7,24 · 6,29 — [57 344, 65 536) = 6,29 < 7,98 × 1,10 : aucune dérive au-delà de 24 576) ; **Q5 TENU** (4 min 30)
durée : 4 min 30 de carte
provenance du code (bd jdp) : ligne `# provenance …/poste6-d19/acvram/__init__.py` au journal `d19-qualite.log`, `sys.path[0] = ACVRAM_ARBRE` dans l'instrument

## Chiffres (B contre A, positions appariées)
| bras | L | n | PPL_A → PPL_B | ΔNLL ± SE | KL moyen | KL max | top-1 |
|---|---|---|---|---|---|---|---|
| B défaut (transitoires + biais) | 16 384 | 32 766 | 10,0599 → 10,0582 | −0,000171 ± 0,000143 | **2,86e-4** | 0,95 | **99,167 %** |
| B défaut | 24 576 | 49 150 | 9,0096 → 9,0100 | +0,000051 ± 0,000123 | **2,66e-4** | 0,106 | **99,192 %** |
| B masque dense (biais off) | 16 384 | 32 766 | 10,0599 → 10,0592 | −0,000066 ± 0,000142 | 2,78e-4 | 0,87 | 99,084 % |
| B relecture int8 (transitoires off) | 16 384 | 32 766 | 10,0599 → 10,0599 | +0,000004 ± 0,000183 | 4,44e-4 | 1,05 | 98,843 % |
| d19 d'poste1 (35B, GDN + MoE par tranches) pour l'échelle | 16 384 | 32 766 | 12,8661 → 12,8659 | −0,000019 ± 0,000871 | 7,3e-3 | 2,80 | 95,96 % |
P (B seul, 65 536, offsets 0 et 100 000) : PPL 6,5351 ; tranches de 8 192 (fenêtre 0) : 11,51 · 10,48 · 7,98 · 8,98 · 6,88 · 7,58 · 7,24 · 6,29.

## Lecture
* **La garde qualité tient, largement** : KL moyen 2,7-2,9e-4 nats (seuil 1e-2, prédit 2e-3-8e-3), top-1 99,2 % (seuil 97, prédit 97-99), ΔNLL
  dans l'erreur-type aux deux longueurs. Le Δ de 0,043 mesuré par C2 est UNE position quasi-égale (−3,190/−3,194) ; sur 82 000 positions
  appariées l'écart moyen est 100 fois plus petit. Les morceaux d'attention sur un dense sont 25 × plus propres que les tranches MoE/GDN
  du 35B (KL 7,3e-3, top-1 96 %) — attendu : pas de routage top-8 qui bascule.
* **Biais bas-droite gardé** : masque dense et biais sont égaux dans le bruit (2,78e-4 contre 2,86e-4) ; le biais rend 8 Gio de scores à 65 536
  (C1 n'est passé qu'avec lui). **Transitoires gardés** : la relecture int8 coûte +55 % de KL et −0,3 point de top-1.
* À 65 536 (sans passe dense possible), la PPL descend avec le contexte (6,54 contre 9,01 à 24 576) et la NLL par tranche ne remonte pas après
  la frontière du seul tenant : pas de dérive des morceaux ; ce n'est pas une preuve d'exactitude (dit au scellé), c'est l'exclusion d'une
  dérive grossière.
* Note poste4 (duckai-01-10-d) : vLLM assume l'absence de garantie sur les logprobs en chunked prefill ; `CausalBias` bas-droite → flash
  `is_causal`, GQA = math ou flash — cohérent avec le pic 4,58 Gio de C1 (rien de matérialisé).

## Reste
* Fusion de poste6-d19 par chef (option (a) : servi au-delà du tenu, étiqueté `morceaux>N`) ; CHANGELOG 0.7.18 : « Devstral-24B (dense sans
  fenêtre) tient 65 536 : préfill de l'attention par morceaux au-delà du tenu d'un seul tenant, K/V bf16 transitoires, réserve honnête ; au-delà
  du tenu KL 2,9e-4 / top-1 99,2 % contre la passe dense ; sous le tenu au bit ».
* kimi-modele : Devstral peut garder les MCP (≥ 65 536) — règle `KIMI_MCP_CTX_MIN` déjà en place (poste1 t5e) : rien à changer, à vérifier
  au prochain lancement du menu.
* gemma-4-31B à 64 k : levier 2 (KV en anneau), scellé de conception du 30/09.
* Sorties : `scratchpad/d19-qualite/` (journaux par mode, états A en .pt), `d19-qualite.log`.
