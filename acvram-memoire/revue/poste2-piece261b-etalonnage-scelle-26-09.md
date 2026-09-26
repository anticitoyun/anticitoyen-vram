# Pièce 261b — scellé AVANT mesure (poste2, 26/09, ordre chef) : étalonnage de `panel-taches.sh`

La 237c a rendu un panel « NON CONCLUANT » à des scores incompatibles avec un modèle réel :
MMLU 18-33 % (le hasard à 4 choix = 25 %), GSM8K 0/50 sur un Qwen3-Coder-30B. Diagnostic de
chef, retenu tel quel : **l'instrument était faux**, pas le modèle — `local-completions` en
5-shot BRUT (sans gabarit de conversation) sur un modèle réglé « Instruct » ne mesure rien
d'utilisable.

## Correctifs (avant toute mesure)

1. **`local-chat-completions` + `--apply_chat_template`** au lieu de `local-completions` en
   5-shot brut : le gabarit de conversation du modèle est appliqué par le serveur (déjà
   supporté, `/v1/chat/completions`).
2. **MMLU passe aux variantes `*_generative`** : `local-chat-completions` ne supporte PAS
   `loglikelihood` (`openai_completions.py:LocalChatCompletion.loglikelihood` lève
   `NotImplementedError` — vérifié en lisant le code, pas supposé) ; seules les tâches
   `generate_until` sont compatibles avec un gabarit de conversation dans `lm-eval`.
3. **`--log_samples` conservé** (jamais supprimé par le script, sous `<sortie>.echantillons/`)
   pour l'inspection humaine demandée.

## Prédiction (avant mesure)

* **MMLU generative, avec gabarit** : je prédis un score **nettement au-dessus du hasard**
  (> 45 %) sur les trois tâches — un Qwen3-Coder-30B-A3B-Instruct correctement sollicité
  répond normalement bien au-dessus de 25 % sur du MMLU généraliste, même en formule
  générative (plus dure que le loglikelihood, qui note la SEULE lettre sans exiger que le
  modèle la produise lui-même en premier).
* **GSM8K, avec gabarit** : je prédis **> 0/50**, probablement entre 20 et 60 % — un modèle de
  cette taille avec capacités de raisonnement de base résout une fraction notable de GSM8K
  5-shot, même sans chaîne de pensée explicite dans ce gabarit brut de tâche.
* **Falsificateur** : si MMLU reste ≤ 30 % OU GSM8K reste à 0 après ces deux correctifs,
  l'instrument est ENCORE faux (ou le protocole few-shot + gabarit de conversation reste mal
  assorti dans `lm-eval` lui-même — cas connu, pas rare) — retour à chef avant tout autre
  chiffre publié.

## Étalonnage contre référence (llama.cpp, même GGUF, mêmes questions)

Référence : `/mnt/4TO_SATACMR_2022/Modeles/models_gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M/
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, servi par `llama-server` (chat template natif du
GGUF, détecté automatiquement), même panel, même graine, mêmes limites.

* **Écart toléré (avant mesure)** : |score_acvram − score_llamacpp| ≤ la largeur de l'IC de
  Wilson à 95 % de la référence (n=100 MMLU, n=50 GSM8K — cf. 237c, 9 à 27 points selon le
  score et la tâche) — c'est-à-dire : les IC des deux moteurs SE RECOUVRENT. Quantification
  différente (NVFP4 contre Q4_K_M GGUF, ≈ 4-4,5 bits/poids les deux) : un écart de qualité de
  cet ordre est attendu par construction, pas un défaut d'instrument.
* **Falsificateur d'étalonnage** : si acvram est HORS de l'IC de llama.cpp alors que les DEUX
  scores restent proches du hasard (échec structurel des deux côtés) → suspecter le PANEL
  (prompt/extraction), pas un des deux moteurs. Si acvram seul est bas et llama.cpp normal →
  suspecter acvram (ou son gabarit de conversation spécifiquement).

## Vérification manuelle (5 échantillons, écrits en clair dans le verdict)

Pour au moins une tâche MMLU et GSM8K, 5 échantillons choisis dans les premiers de
`--log_samples` : réponse BRUTE du serveur (`resps`), réponse EXTRAITE (après le filtre de la
tâche), réponse ATTENDUE (`target`) — recopiés tels quels, pas résumés.

## McNemar (P1 contre P0), préparé mais PAS exécuté dans cette pièce

Test proposé pour départager P1/P0 une fois l'instrument étalonné : sur les MÊMES questions
(même graine, même limite), compter les paires discordantes (P0 correct/P1 faux) contre
(P0 faux/P1 correct) par tâche, statistique de McNemar avec correction de continuité (petits
effectifs). Script à écrire à la 237d si chef le demande — hors périmètre de cette pièce
(étalonner l'instrument, pas encore relancer P0/P1).

## Durée prévue

Test rapide (n=10/tâche) pour confirmer le correctif avant la mesure complète : ≤ 10 min.
Mesure complète (acvram n=100/50 + llama.cpp n=100/50) : ≤ 30 min sous `carte.sh` pour la
partie acvram (llama.cpp hors carte, CPU/3080Ti selon dispo, à vérifier avant lancement).
