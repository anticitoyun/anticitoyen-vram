# kv31b levier 1 — scellé de la preuve sur carte : réserve de préfill plafonnée, chauffe d'un dense à 32 768 (poste6, 30/09, écrit AVANT la prise)

Feu chef (30/09) : fenêtre après poste1, ≤ 15 min, ACVRAM_POSTE=poste6, carte seule. Code : branche poste6-reserve-tranches
(levier 1 11de7765e + scellé de conception e8e816f58 ; le paquet 0.7.16 ne porte PAS le levier 1 → le moteur tourne depuis l'arbre
du worktree, `PYTHONPATH`, venv du dépôt principal). Script : `outils/gpu/mesure/prise-reserve-plafonnee-kv31b.sh` (HEAD asserté,
relevés nvidia-smi début/fin, journal du moteur lu par ses lignes `[acvram]` seulement — § 6 : l'alias gemma-4-31b `4sur6-vision` est censuré).

## Modèle et régime
* `acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4` (19,0 Gio, 60 couches, dense), `--max-model-len 32768`, speculation ngram, graphes ON.
* **Carte seule** = aucun autre processus de calcul sur la 5090 (le llama-server 4436 hors verrou occupait 5,6 Gio pendant l'edz : s'il est
  encore là, la prise se fait à ≈ 28,1 Gio libres et la prédiction devient celle de la colonne « 28,1 »). Relevé `nvidia-smi
  --query-compute-apps` au début ET à la fin ; libre lu dans la ligne « borné par la VRAM libre ».
* Témoin (si le temps le permet, sinon reporté) : Devstral-24B à 32 768 — a5v seul posait déjà le plafond (poste5 294) : inchangé.

## Prédiction et seuils (réplique à sec, verdict poste6-reserve-tranches)
| # | grandeur | carte seule (≈ 33,6 libres) | 28,1 (llama-server présent) | FAUX si |
|---|---|---|---|---|
| S1 | ligne `MLP dense par tranches au-delà de 4096 jetons : … 5,0x Gio au lieu de 10,7x` | présente | présente (relance à 31 744 après refus à 32 768) | absente à carte seule |
| S2 | MLP exilés (`plan réajusté … MLP … RAM hôte`) | 25-40 sur 60 (réplique : 33) | refus à 32 768 puis 55-60 à 31 744 | 60/60 à carte seule, ou 0 |
| S3 | `budget KV … borné` : KV ≥ 15,1 Gio, poids résidents 10-13 Gio | oui | — | KV < 15,1 → refus |
| S4 | chauffe : `ctx_tenu=32768` au régime, aucun OOM | tenu | 31 744 tenu | OOM ou refus |
| S5 | une complétion de 8 jetons répond (rc 200) | oui | oui | 5xx |
| S6 | durée totale prise | ≤ 12 min (chargement 19 Gio + chauffe 32 k) | idem | > 15 min : carte.sh tue |
Issues nommées : (a) le plafond est posé mais la chauffe échoue en OOM → la réserve plafonnée sous-estime les activations réelles
(le seuil MLP `_MLP_SEUIL` n'était pas posé, ou une matrice déquantifiée hors compte) : réserve à revoir, levier 1 NON tenu ;
(b) 60/60 exilés à carte seule → l'exil n'a pas suivi la réserve (autre défaut, à chiffrer) ; (c) carte non seule → prédiction 28,1 ;
(d) débit de décodage non mesuré ici (25-40 MLP exilés = falaise de l'exil, connue : le levier 2 y répond).
