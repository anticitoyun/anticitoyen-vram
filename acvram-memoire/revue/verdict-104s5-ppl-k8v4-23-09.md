# Verdict — 104 § 5 point 2 : PPL de décodage KV k8v4 contre int8 (prise B) — 23/09 23 h 4x (poste1)

* **instrument** : `outils/gpu/mesure/ppl-decode-kv.py` (fichier suivi), 36 tranches `scratchpad/corpus-prive/tranches-36`, préfixe 8 192 + 512 notés, un chargement par bras ; agrégat par `scratchpad/poste1-p104s5-23-09/prise-s5b.sh`
* **commit** : 1fa96cd3 (poste1-mtp, origin/main fusionné, garde HEAD rc 65 dans le script)
* **régime** : Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c, bras A `ACVRAM_KV_FORMAT=int8` (servi), bras B `k8v4` ; horloge libre (PPL : grandeur numérique, pas de temps)
* **scellé** : `scratchpad/poste1-p104s5-23-09/scelle.md` point 2, écrit avant : géo(k8v4/int8) + 2 SE ≤ +0,30 %, géo prédite +0,10 à +0,25 %
* **mesuré** : n = 36, géo **+0,211 %**, SE 0,138 %, géo + 2 SE **+0,489 %** ; 22/36 tranches positives, médiane +0,14 %, étendue −1,64 % (t29) à +2,40 % (t7) ; sans la pire tranche, géo +0,152 %
* **verdict** : **FAUX au scellé** (+0,489 > +0,30). La géo tombe dans la prédiction, mais l'instrument ne résout pas le seuil : 2 SE = 0,276 %, il faudrait une géo ≤ +0,024 % pour passer. k8v4 **n'est pas un défaut** ; replis dans l'ordre du dossier (§ 4 veille arXiv) : (1) puits d'attention gardés int8, (2) V avec point zéro, (3) Hadamard par tête + INT4, (4) E2M1
* **durée** : prévue ≤ 25 min, tenue=404 s (`carte.sh`, journal 23:41:17) ; compute-apps début = fin (llama-server 4627 seul)

Ordre changé (A puis B demandé) : à 23 h 34 la charge était à 6,36 pour 8 cœurs (conversion de poste4, CPU, sans carte) ; la frontière de la prise A mesure un pas en ms, d'où B d'abord (PPL insensible à la charge). A attend une charge < nproc/2.

Remarque d'instrument : σ par tranche ≈ 0,83 % → résoudre ± 0,15 % à 2 SE demanderait ≈ 120 tranches. Un scellé à +0,30 % sur 36 tranches ne pouvait passer que pour une géo quasi nulle ; décider sur ce même critère, ou sur une campagne plus longue, relève du chef.
