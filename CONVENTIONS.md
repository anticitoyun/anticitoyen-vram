# Travailler sur acvram

Repères pour qui reprend ce code. Lisez d'abord [`REPRISE.md`](REPRISE.md) :
il contient l'état du projet, les invariants, les pièges déjà rencontrés et les
décisions prises. Ce fichier-ci ne couvre que les conventions de travail.

## Langue

**Tout le dépôt est en français.** Commentaires, docstrings, documentation,
aide en ligne de commande, messages d'erreur, messages de commit.

Restent en anglais les seuls identifiants qui constituent un contrat externe,
parce que les traduire romprait l'interopérabilité :

* champs de l'API OpenAI — `choices`, `finish_reason`, `prompt_tokens` ;
* clés des configurations Hugging Face — `num_hidden_layers`, `rope_theta` ;
* méthodes et attributs PyTorch — `forward`, `state_dict`, `dtype` ;
* noms de tenseurs dans les safetensors — `model.layers.0.self_attn.q_proj.weight` ;
* noms de formats — `nvfp4`, `int4_awq`, servis par `/v1/models`.

Les noms de variables et de fonctions internes sont en anglais lorsqu'ils
prolongent une de ces conventions, en français partout où le choix est libre.

## Commandes

```bash
./install.sh                        # ou : pip install -e '.[dev]'
pytest -q                           # processeur uniquement, ~20 s
acvram doctor                       # vérification de l'environnement
ACVRAM_VERBOSE_BUILD=1 python -c "from acvram import kernels; print(kernels.build_info())"
ACVRAM_DISABLE_KERNELS=1 pytest -q  # force le chemin de référence
ACVRAM_DISABLE_CPU_KERNELS=1 ...    # idem pour les noyaux processeur
ACVRAM_TRACEBACK=1 acvram ...       # traces complètes depuis le CLI
```

## Style

* Les commentaires expliquent le **pourquoi**, et méritent d'être écrits partout
  où une décision paraît arbitraire — le choix du format de cache KV et le refus
  du planificateur d'utiliser le second GPU sont tous deux contre-intuitifs, et
  tous deux commentés.
* Les chiffres cités dans la documentation proviennent de code de ce dépôt. Si
  vous changez une valeur par défaut à cause d'une mesure, mettez la mesure dans
  un test.
* Une optimisation qui change la sortie est un bogue : ajoutez son test
  d'équivalence dans le même commit.

## Mesurer un correctif

Trois règles tirées d'erreurs réelles, chacune ayant coûté une conclusion
fausse :

* **L'« avant » se produit par `git checkout <rev> -- <fichier>`, jamais par
  `git stash`.** Sur un arbre propre, `git stash push` ne remise rien et le
  correctif se compare à lui-même — sans erreur, avec un résultat plausible.
* **Vérifier par le contenu du fichier que la bonne version est en place**
  (`grep -c` sur une ligne du correctif) avant et après restauration. Un état
  d'arbre supposé n'est pas un état d'arbre.
* **Un contrôle négatif ne vaut que flanqué d'un contrôle positif.** Un
  dispositif qui rend « identique » à tous les coups ne mesure peut-être rien :
  il faut avoir montré, sur un cas de différence connue, qu'il sait dire
  « différent ».

Et vérifier que la mesure **exerce** le correctif : un correctif du chemin
processeur ne s'exerce que si des couches sont réellement en mémoire hôte, un
correctif de conversion ne s'exerce pas en réévaluant un modèle déjà converti.

## Avant de pousser

```bash
pytest -q                                   # aucune régression
git diff --cached | grep -nE 'glpat-|ghp_'  # aucun secret
```

Le jeton GitLab vit chiffré dans `~/.config/acvram/gitlab-token.gpg` et n'entre
jamais dans le dépôt ni dans `.git/config` ; voir la section 4 de
[`REPRISE.md`](REPRISE.md).
