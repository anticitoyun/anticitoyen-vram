# Contributing — Contribuer

## English

### Contributions welcome

acvram is a local inference engine for large language models on consumer NVIDIA GPUs (RTX 5090 `sm_120`,
RTX 3080 Ti `sm_86`). There is a lot left to build: kernels, model families, quantization formats, the menus of the
companion `parc` package. Bug reports, measurements on other GPUs, ideas and pull requests are all welcome.

For quick feedback on an idea, open an [issue](https://github.com/anticitoyun/anticitoyen-vram/issues). For
security problems, do **not** open an issue: follow [`SECURITY.md`](SECURITY.md).

### Contribution workflow

The project uses the fork-and-pull model.

1. In your fork, create a branch with a meaningful name.
2. Make your change, meeting the [quality standards](#quality-standards) below.
3. Open a pull request against `main`.
4. A maintainer reviews it. Address every comment; amend your commits and force-push your branch rather than
   stacking "fix review" commits.
5. Once approved, a maintainer applies it.

**How changes reach `main`.** The public repository is a filtered copy of the maintainers' repository: an accepted
pull request is applied there, the full test suite runs on the real GPUs, then the public history is regenerated and
published as a fast-forward. Your commits therefore appear with a new identifier and a neutral author field; your
credit is kept by name in the commit message and in [`CHANGELOG.md`](CHANGELOG.md).

### Request for comments

To get feedback on a direction before writing the whole thing, open a pull request whose title starts with `[RFC]`,
based on the current code. Write down in the pull request what was concluded.

### Quality standards

The repository is written in **French**: comments, docstrings, documentation, CLI help, error messages and commit
messages. Identifiers that are an external contract stay in English (OpenAI API fields, Hugging Face config keys,
PyTorch methods, safetensors tensor names, format names). If French is a barrier, write in English and a maintainer
will translate — the content matters more than the language.

Before opening a pull request:

```bash
pip install -e '.[dev]'
pytest -q                           # CPU only; GPU tests skip without a GPU
ACVRAM_DISABLE_KERNELS=1 pytest -q  # the reference path must pass too
acvram doctor                       # environment check
```

Your contribution must meet these standards:

- **One logical change per commit**, with a descriptive message (title ≤ 72 characters, then the why).
- **Every commit passes the test suite.** New behaviour comes with a test that **fails without your change**; a test
  that cannot fail is not a test.
- **An optimization that changes the output is a bug.** Any change to a kernel, a numeric path or a default ships
  with its equivalence test in the same commit (bit-exact where the old path was bit-exact, or a stated tolerance and
  why), and that test must break if the fault is reintroduced.
- **No silent fallback.** A path selected by a variable, backend or kernel either runs or says loudly why it does
  not. A kernel declares the architectures and block/group sizes it can read and refuses the others.
- **Every `ACVRAM_*` environment variable is declared**: in `acvram/cli.py` (`VARIABLES_LUES`) and in
  `acvram/regime.py` (in `VARIABLES`, or in `HORS_REGIME` if it only observes, compiles or names a file). A new or
  changed default is recorded in `CHANGELOG.md`. The tests enforce this.
- **Numbers in the documentation come from code in this repository.** If you change a default because of a
  measurement, put the measurement in a test.
- **Performance claims come with their regime**: GPU, driver, CUDA and torch versions, model, context, batch, and
  the `[régime]` line printed by acvram. Write the predicted number and the threshold *before* measuring, and report
  the result even if it contradicts the prediction. Improvements outside hot paths are unlikely to be accepted.
- **Comments explain why**, especially where a decision looks arbitrary.
- **No personal data** in tracked files: no home paths (`/home/<name>`), e-mail addresses or tokens.
- **Explain your pull request**: the reasoning behind each change and the testing done (CPU, GPU model, what ran).
- **You answer for every line you submit**, whatever tools helped you write it.

### Developer Certificate of Origin

acvram is released under the [GNU General Public License v3.0](LICENSE). To make sure every contribution is
correctly attributed and licensed, each commit must carry a `Signed-off-by` line, by which you agree to the
Developer Certificate of Origin 1.1 (<https://developercertificate.org/>):

```
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the
    best of my knowledge, is covered under an appropriate open
    source license and I have the right under that license to
    submit that work with modifications, whether created in whole
    or in part by me, under the same open source license (unless
    I am permitted to submit under a different license), as
    indicated in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including
    all personal information I submit with it, including my
    sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s)
    involved.
```

Use `git commit -s`. A stable pseudonym is accepted, as long as it identifies you consistently across your
contributions. Forgot it? `git commit --amend -s`.

---

## Français

### Les contributions sont les bienvenues

acvram est un moteur d'inférence local pour grands modèles de langage sur GPU NVIDIA grand public (RTX 5090
`sm_120`, RTX 3080 Ti `sm_86`). Il reste beaucoup à construire : noyaux, familles de modèles, formats de
quantification, menus du paquet compagnon `parc`. Rapports de bogue, mesures sur d'autres GPU, idées et demandes de
fusion sont les bienvenus.

Pour un avis rapide sur une idée, ouvrez un [ticket](https://github.com/anticitoyun/anticitoyen-vram/issues). Pour un
problème de sécurité, n'ouvrez **pas** de ticket : suivez [`SECURITY.md`](SECURITY.md).

### Déroulement

Le projet suit le modèle « fork and pull ».

1. Dans votre fork, créez une branche au nom parlant.
2. Faites votre changement en respectant les [exigences de qualité](#exigences-de-qualité) ci-dessous.
3. Ouvrez une demande de fusion (pull request) vers `main`.
4. Une personne de l'équipe la relit. Répondez à chaque remarque ; modifiez vos commits et repoussez votre branche
   (push forcé) plutôt que d'empiler des commits « correction de relecture ».
5. Une fois acceptée, elle est appliquée par l'équipe.

**Comment un changement arrive sur `main`.** Le dépôt public est une copie filtrée du dépôt de l'équipe : une demande
acceptée y est appliquée, la suite complète tourne sur les vrais GPU, puis l'historique public est régénéré et publié
en avance rapide. Vos commits y apparaissent donc avec un nouvel identifiant et un auteur neutre ; le crédit vous est
conservé nommément dans le message du commit et dans [`CHANGELOG.md`](CHANGELOG.md).

### Demande de commentaires

Pour un avis sur une direction avant de tout écrire, ouvrez une demande de fusion dont le titre commence par `[RFC]`,
bâtie sur le code actuel. Notez dans la demande ce qui a été conclu.

### Exigences de qualité

Le dépôt est rédigé **en français** : commentaires, docstrings, documentation, aide du CLI, messages d'erreur et
messages de commit. Restent en anglais les identifiants qui forment un contrat externe (champs de l'API OpenAI, clés
des configurations Hugging Face, méthodes PyTorch, noms de tenseurs safetensors, noms de formats). Si le français
vous bloque, écrivez en anglais : l'équipe traduira — le fond compte plus que la langue.

Avant d'ouvrir une demande :

```bash
pip install -e '.[dev]'
pytest -q                           # processeur seul ; les tests GPU se sautent sans GPU
ACVRAM_DISABLE_KERNELS=1 pytest -q  # le chemin de référence doit passer aussi
acvram doctor                       # vérification de l'environnement
```

Votre contribution doit respecter ces règles :

- **Un changement logique par commit**, avec un message parlant (titre ≤ 72 caractères, puis le pourquoi).
- **Chaque commit passe la suite de tests.** Un comportement neuf arrive avec un test qui **échoue sans votre
  changement** ; un test qui ne peut pas échouer n'est pas un test.
- **Une optimisation qui change la sortie est un bogue.** Tout changement de noyau, de chemin numérique ou de défaut
  part avec son test d'équivalence dans le même commit (au bit là où l'ancien chemin l'était, sinon une tolérance
  énoncée et justifiée), et ce test doit casser si l'on réintroduit la faute.
- **Aucun repli silencieux.** Un chemin choisi par une variable, un backend ou un noyau tourne, ou dit haut et fort
  pourquoi il ne tourne pas. Un noyau déclare les architectures et les tailles de bloc ou de groupe qu'il sait lire,
  et refuse les autres.
- **Toute variable d'environnement `ACVRAM_*` est déclarée** : dans `acvram/cli.py` (`VARIABLES_LUES`) et dans
  `acvram/regime.py` (dans `VARIABLES`, ou dans `HORS_REGIME` si elle ne fait qu'observer, compiler ou nommer un
  fichier). Un défaut neuf ou modifié est inscrit au `CHANGELOG.md`. Les tests y veillent.
- **Les chiffres de la documentation viennent du code de ce dépôt.** Si vous changez un défaut à cause d'une mesure,
  mettez la mesure dans un test.
- **Une affirmation de performance vient avec son régime** : GPU, pilote, versions de CUDA et de torch, modèle,
  contexte, lot, et la ligne `[régime]` imprimée par acvram. Écrivez le chiffre prédit et le seuil *avant* de mesurer,
  et publiez le résultat même s'il contredit la prédiction. Un gain hors des chemins chauds a peu de chances d'être
  retenu.
- **Les commentaires expliquent le pourquoi**, surtout là où une décision paraît arbitraire.
- **Aucune donnée personnelle** dans les fichiers suivis : ni chemin personnel (`/home/<nom>`), ni courriel, ni jeton.
- **Expliquez votre demande** : la raison de chaque changement et les essais faits (processeur, modèle de GPU, ce
  qui a tourné).
- **Vous répondez de chaque ligne soumise**, quels que soient les outils qui vous ont aidé à l'écrire.

### Certificat d'origine du développeur

acvram est publié sous la [Licence publique générale GNU v3.0](LICENSE). Pour que chaque contribution soit
correctement attribuée et placée sous licence, chaque commit porte une ligne `Signed-off-by`, par laquelle vous
acceptez le Developer Certificate of Origin 1.1 (<https://developercertificate.org/>), reproduit en anglais dans la
section ci-dessus : ce texte fait foi dans sa langue d'origine.

Utilisez `git commit -s`. Un pseudonyme stable est accepté, pourvu qu'il vous identifie de façon constante d'une
contribution à l'autre. Oubli ? `git commit --amend -s`.
