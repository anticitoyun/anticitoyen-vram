# Verdict — prise A : 110 sur carte, contrôle MTP brut/norme (105), KL b=1 k8v4 (104 § 5.1), frontière ctx 8 k (104 § 5.3) — 23/09 23 h 4x (poste1)

* **instrument** : `scratchpad/poste1-p104s5-23-09/prise-s5a.sh` (garde HEAD rc 65) — `tests/test_fusion_partielle_numerique.py`, `scratchpad/poste1-p105-23-09/accept-mtp.py`, `kl-b.py` (KL_B=1, 5 dumps HF), `outils/gpu/mesure/frontiere-pas.py`
* **commit** : 362c9cfb (poste1-mtp)
* **régime** : Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c (KL, frontière) ; MTP : alias de la 105 ; -lgc 2700 posé pour la frontière seulement ; charge 5,3 au départ (conversion de poste4 finie), 2,7 avant la frontière
* **scellé** : `scratchpad/poste1-p104s5-23-09/scelle.md` (écrit avant)
* **mesuré** :
  * 110 sur carte, sans la variable : **4 passed**.
  * MTP : sorties identiques à la génération sans spéculation dans les deux bras (5/5) ; a(brut) **0,3208**, a(norme) **0,3811** ; t_s 65,7 → 52,9 (brut) / 57,6 (norme).
  * KL b=1 : int8 kl_max 0,5193, 5/5 ; k8v4 kl_max par invite 0,0094 / 0,1185 / 0,5193 / 0,1849 / 0,2326 (5/5 ≤ 0,74) ; ΔKL_max(k8v4 − int8) = +0,0021 / 0 / 0 / **+0,1173** / 0.
  * Frontière : **rc 1 dans les deux bras** — OOM au préfill de 12 × 8 192 (`moe.py:958 _forward_prefill_grouped`, 1 Gio demandé, 12 Mio libres ; llama-server 8081 résident 5,6 Gio). Aucun pas mesuré.
* **verdict** :
  * 110 : **tenu** au bit sur carte.
  * MTP : **hypothèse TENUE** au seuil scellé (0,3811 ≥ 0,3208 + 0,05 = 0,3708 ; marge 0,010, mince). Suite scellée : `ACVRAM_MTP_ETAT=auto` par convention (Qwen3.5 → norme, DeepSeek → brut, test_mtp_qwen35). Brouillon en graphe NON (a < 0,45).
  * 104 § 5.1 KL b=1 : **FAUX** — invite 3 : ΔKL +0,1173 > 0,10. Avec la PPL FAUSSE de la prise B, k8v4 seul est deux fois refusé.
  * 104 § 5.3 frontière : **non mesurée** (instrument hors régime : préfill d'un bloc en OOM). Correctif proposé, sans toucher la grandeur mesurée (pas de DÉCODAGE) : `ACVRAM_BUDGET_JETONS=2048` (préfill par tranches, runner.py:1385), mêmes deux bras, une prise de 5 min après la 123.
* **durée** : prévue ≤ 15 min, tenue=132 s (journal carte 23:44:17) ; compute-apps début = fin (llama-server 4627 seul)

Contrôle de configuration : le bras k8v4 imprime `kv=k8v4` 5/5 (int8 : `kv=int8` 5/5) — la variable a pris. ΔKL = 0 sur 3 invites = kl_max inchangé à 10⁻⁴ (arrondi du JSON), pas un bras inerte.

**ERRATUM 24/09 02 h 3x (poste1)** : llama-server (PID 4627) tourne sur la RTX 3080 Ti (GPU 1), pas sur la 5090 (`nvidia-smi --query-compute-apps=gpu_uuid`). L'OOM et le lot jamais plein à ctx 8 k viennent de NOTRE processus (31,32 Gio sur la 5090 : poids + KV de 12 × 8 704 + activations du préfill), pas d'un voisin. La consigne « ne pas toucher au llama-server » reste juste, mais sa cause était fausse : la frontière à 8 k, b=12 ne tient pas sur 32 Go avec ce plan mémoire, llama-server ou non.
