# Porte KL étape 1 acvram_mojo — verdict (question b, contrat §3)

* instrument : dump HF bf16 (`transformers` direct, poids source, 8 jetons gloutons/invite), KL@7 (borne
  MAX : `logprobs` max 7) teacher-forcée via `/v1/completions echo=true`, contre acvram serve et MAX serve
* commit : `7ce7dc99` (branche poste4, étape 1 fusionnée) ; cpu-safe=off (bridage utilisateur retiré à
  00:54:35, max_perf_pct 100 aux deux bouts de cette prise, aucun chevauchement avec la fenêtre bridée)
* régime : `carte.sh` type mesure, une carte, acvram puis MAX en série (contention VRAM constatée si
  chargés ensemble)
* scellé : seuil 0,74 chacun (`acvram-memoire/revue/poste4-mojo-kl-prediction-24-09.md`, écrit avant mesure)
* mesuré : **acvram KL_max = 0,00149** (5/5 invites ≤ seuil, prédiction ≤0,10 tenue) ; **MAX KL_max = 20,01**
  (5/5 invites très au-dessus du seuil, prédiction 0,05-0,74 RÉFUTÉE)
* verdict : **porte KL NON tenue pour MAX** — mais la divergence n'est PAS diffuse : 7/8 positions par
  invite sont au bit près identiques à HF bf16 (KL ≈ 0 exact, `top1_moteur == attendu`), UNE seule position
  (après « Okay », jeton attendu `,`) porte tout le KL — MAX y choisit la virgule PLEINE CHASSE `，` au lieu
  de `,` (5/5 invites, même position dans la séquence de pensée). Pas un bruit de noyau diffus : un point de
  divergence net, probablement un artefact du sampler/format de sortie de MAX sur ce jeton précis (à
  confirmer). Étape (2) du contrat (cellule débit/énergie, conditionnée à « si tenu ») **ne se lance pas**.
  Deux bogues d'instrument trouvés et corrigés en route (documentés dans `outils/kl_reference.py`) :
  clés `top_logprobs` d'acvram = texte décodé (pas des ids), et `top_logprobs[i]` de MAX prédit le jeton
  i+1 (décalage −1, absent chez acvram) — les deux auto-détectés/corrigés, pas codés en dur pour un moteur.
* durée : ~9 relances (bogues d'instrument successifs : `accelerate`, `Encoding` tokenizers,
  `--no-enable-overlap-scheduler --force --enable-echo` requis par MAX pour les logprobs, décalage d'index) ;
  prise finale ≤ 10 min, files d'attente carte 1-16 min à chaque tentative (poste5/poste3 en parallèle)
