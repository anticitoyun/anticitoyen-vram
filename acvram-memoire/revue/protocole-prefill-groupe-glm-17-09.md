# Protocole — contrôle GLM avant défaut `groupe` : GLM `-k48-calibA` prefill 2048 et PPL privée, `groupe` contre `grouped_mm`
instrument : `prefill-glm-acvram-15-09.py w4a16 2048` ; `ppl-acvram-17-09.py` privé 3 tranches GLM (`corpus-encode-brut-glm/prive-tr*.txt`, préfixe `[gMASK]<sop>` par `encode_brut`) — `scratchpad/prefill-groupe-glm-17-09/`.
commit : arbre laure (= main, B0 83c24e1) ; régime classé, MLA ; référence GLM prefill w4a16 : 4 422 j/s (`verdict-llamacpp-glm-16-09`, prise B).
scellé (Sage/Jérôme) : prefill ≥ 5 000 j/s ET PPL ± 0,002 contre témoin. Moi : groupe 5 000-5 600 (Coder + 14,8 % ; GLM 46 couches MoE de 64 experts actifs 4+1, K = 2048, N = 1536 : tuiles Triton moins remplies), témoin 4 300-4 600, PPL ± 0,001.
falsification : groupe < témoin → les tuiles fixes perdent sur les experts de GLM (M plus petit : 4 experts/jeton sur 64), défaut refusé ; PPL > ± 0,002 → non recevable.
