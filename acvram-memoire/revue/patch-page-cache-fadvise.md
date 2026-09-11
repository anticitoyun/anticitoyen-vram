# Proposition : rendre le page cache des safetensors après chargement

poste2, 8 septembre 2026. **Non écrit dans le code** — une vérification
conditionne l'application, elle est nommée plus bas.

## Le fait, mesuré pendant que le témoin chargeait

    MemFree       3,1 Go     ← ce que lisait le superviseur qui a tué à 20h50
    MemAvailable  40,0 Go    ← réellement disponible
    Cached        86,4 Go    ← dont ~57 Go de safetensors du témoin
    PSI full      0,00

`MemFree` compte ce qui n'est utilisé **par rien**. Le page cache est utilisé
et **entièrement récupérable** : le témoin pèse 57 Go sur disque, lus en mmap
par `safe_open` (`loader.py:48`), et fait chuter `MemFree` d'autant sans
retirer un octet à quiconque. Onze minutes de mesure réussie perdues sur ce
compteur — le troisième de la famille `Mem*` à tromper le circuit en une
soirée, dans un sens à 19h45, dans l'autre à 20h50.

## La proposition

Ce page cache **ne sert à rien après le chargement** : les poids sont déjà
copiés dans `plat` épinglé ou en VRAM, les fichiers ne seront jamais relus.

Après chargement complet : fermer les handles `self._open` puis, sur chaque
fichier, `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)`.

Deux effets d'un coup : le faux positif du superviseur disparaît **à la
source** (et non par correction d'un superviseur qu'on ne contrôle pas), et la
machine récupère ~57 Go de cache mort sans attendre une pression pour les
réclamer.

## Précondition VIOLÉE en l'état — vérifié par poste1, confirmé ici

`QuantLinear.to_device(streamed=True)` construit `StreamedWeight` depuis
`dict(self.qweight.state_dict())` **sans libérer `self.qweight`** : la source
reste vivante comme gabarit, ses tenseurs maintiennent les pages du mmap
mappées et référencées, et `DONTNEED` ne peut donc rien réclamer.

Remède identifié par poste1 : `_rehydrate` ne lit du gabarit que des
**scalaires** (`shape`, `padded_in`, `group_size`, `format`, `block`), aucune
donnée tensorielle — le contenu du gabarit peut donc être libéré après
`_emballer`, en gardant l'objet comme porteur de métadonnées.

**Condition préalable** : `nbytes` (`layers.py:357`) lit `self.qweight.nbytes`
et doit d'abord passer sur `plat.numel()`. Le raffinement (c) devient une
condition.

## Le couplage qui rend ce remède sûr, et qui n'est écrit nulle part

**Relevé indépendant des usages du gabarit.** Deux fonctions lisent des
**données** tensorielles et non des scalaires — `stack_nvfp4_linears` (805) et
`stack_int8_linears` (856), qui font `torch.cat([t.qweight for t in ts])`. Et
elles sont appelées **après** le streaming : `to_device(streamed=…)` aux lignes
296-347 de `loader.py`, `m.fuse()` aux lignes 753-755.

L'ordre est donc défavorable, et vider les gabarits devrait casser la fusion.
**Il ne la casse pas, pour une seule raison** : les deux fonctions refusent
explicitement les couches streamées —

    if any(l.bias is not None or l.scaler is not None
           or l.streamed is not None for l in lins):
        return None                      # layers.py:823 et 865

Une couche streamée n'est jamais fusionnée, donc son gabarit n'est jamais lu
pour ses données.

**À écrire à côté de la garde, parce que le couplage est invisible** : si l'on
autorise un jour la fusion des couches streamées — optimisation plausible, une
GEMV au lieu de trois sur les couches exilées — **le remède `fadvise` devient
silencieusement faux**, et la fusion lira des gabarits vidés. Deux correctifs
justes chacun, dont l'un dépend d'une garde de l'autre sans que rien ne le
dise.

## Ce qu'il reste à vérifier avant d'appliquer


**Aucun poids ne doit rester une vue sur le mmap.** Si un tenseur pointe
encore dans le fichier, `fadvise` ne rendra rien et le correctif viserait faux
— ou pire, provoquerait des relectures disque en plein décodage.

Indice favorable : `loader.py:269` copie explicitement la table d'embedding
« pour ne pas rester un mmap », donc le cas est connu du code et traité au
moins une fois. Il reste à établir qu'il l'est **partout**.

Relevé à faire quand la carte se libère : après `load_model`, vérifier que
`Cached` chute d'environ la taille du modèle une fois les handles fermés et
`fadvise` appelé, et que la perplexité reste identique au même modèle chargé
sans `fadvise`. Si `Cached` ne bouge pas, des poids sont encore mappés.
