# Banc énergie 4 moteurs, 3ᵉ passage (sha 20ec9212, correctifs poste4) — llamacpp confirmé TENU, 3 défauts encore ouverts — 22/09 (poste2)

* instrument : `chaine-energie-4moteurs.sh` sha `20ec9212` (BANC_MOTEUR=acvram forcé, bridage puissance censé être accepté), `resumer-energie-4moteurs.py` même sha
* commit : main à jour au moment de la mesure (`c840a9b2`)
* mesuré (`protocole25.tsv`) :

| moteur | fenêtres valides | j/jeton médian | t/s médian |
|---|---|---|---|
| **llamacpp** | **2/3** | **0,2064** | **1043,5** |
| acvram | 0/3 | — | — |
| vLLM | 0/3 | — | — |
| trtllm | 0/3 | — | — |

* **llamacpp CONFIRMÉ TENU** (2ᵉ mesure indépendante, cohérente avec le passage précédent : 0,2064 contre 0,2157 J/jeton, 1043,5 contre 1022,6 t/s — écart <5 %).
* **3 défauts encore ouverts, aucun corrigé (consigne : pas de correctif à chaud sans accord)** :
  1. **acvram, critère bridage PAS assoupli malgré l'annonce** : `resumer-energie-4moteurs.py:48-49,71-72` (`throttle(d)` teste `bridages != "aucun"`, sans exception `watts_moy` visible) — les 3 fenêtres acvram (b12×2, b1×1) rejetées identiquement au passage précédent, malgré le message du groupe annonçant « bridage puissance accepté avec watts_moy » sur ce sha. À vérifier par poste4 : le correctif n'est peut-être pas dans ce fichier/cette version.
  2. **vLLM, NOUVEAU symptôme** (le fix `BANC_MOTEUR=acvram` est bien appliqué, `moteur: "acvram"` confirmé dans le RESULTAT) : `n_jetons_decodes=0` malgré 4090 lots en 20 s, `watts=26,4` (niveau repos), `horloge_moy=1026` (pas de charge réelle), `bridages: "aucun"`. Les requêtes réussissent sans erreur mais reviennent vides quasi instantanément — signal compatible avec `ignore_eos` non honoré par vLLM sur ce checkpoint NVFP4 (arrêt immédiat sur EOS malgré la demande).
  3. **trtllm, NOUVEAU crash** : `banc-llamacpp-16-09.py:107` (`client.get(f"{HOTE}/metrics", timeout=5.0).json().get("cartes")`) → `AttributeError: 'list' object has no attribute 'get'` — l'endpoint `/metrics` de `trtllm-serve` renvoie une **liste** JSON, pas un objet, incompatible avec ce parseur (`RESULTAT absent/invalide` sur les 3 fenêtres trtllm).
* verdict : **llamacpp seul publiable** (2 mesures indépendantes convergentes). acvram/vLLM/trtllm toujours non mesurés, chacun pour une cause désormais précisément nommée et distincte.
* durée : 3ᵉ passage 13:40-14:00 (20 min, dans le budget annoncé)

## Suite
Trois points distincts pour poste4/le groupe : (1) vérifier où vit le correctif bridage annoncé (absent de `resumer-energie-4moteurs.py` à ce sha) ; (2) vLLM `ignore_eos` à investiguer côté serveur (pas côté client, déjà corrigé) ; (3) `banc-llamacpp-16-09.py:107` à rendre tolérant à une réponse `/metrics` en liste (ou à l'ignorer si non-dict) pour trtllm. Carte rendue à poste3.
