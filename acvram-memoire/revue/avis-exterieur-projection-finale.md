# Avis extérieur — le coût de la projection finale (question 3)

Source : GPT-5.6 avec recherche, duck.ai, 10/09. **Séparation affirmé / vérifié / à
vérifier.**

## LA DISTINCTION QUI TRANCHE, et que je n'avais pas faite

La réponse ouvre sur une séparation que ma question mélangeait :

- **réduire le nombre de LIGNES de vocabulaire lues** réduit le coût fondamental de
  bande passante de `lm_head` ;
- **éviter de matérialiser le tenseur de logits** réduit l'allocation, la communication
  et le coût d'échantillonnage, **mais n'évite pas de lire toutes les lignes de
  `lm_head`**.

C'est décisif pour nous parce que nous sommes **mémoire-bornés** : seule la première
catégorie compte. Le tableau rendu, avec le statut de production affirmé :

    (a) vocabulaire parallele avec reduction   trafic PAR DEVICE      partiellement
                                                                      implante ; la
                                                                      forme a sampler
                                                                      reduit n'est pas
                                                                      un defaut de
                                                                      production
    (b) echantillonner sans materialiser       materialisation et     partiellement
        tous les logits                        communication des      implante ; « ne
                                               logits, parfois la     supprime GENERA-
                                               memoire de sortie      LEMENT pas la
                                                                      lecture de toutes
                                                                      les lignes »
    (c) vocabulaire elague ou adaptatif        les LIGNES reellement  surtout recherche
                                               lues                   et prototypes
    (d) tete a precision inferieure            « reduit DIRECTEMENT   implante dans
                                               les octets lus :       certains moteurs
                                               bf16 -> fp8/int8 la    et formats
                                               moitie, int4 le quart »
    (e) decodage speculatif                    amortit le modele      largement
                                               cible, y compris sa    implante
                                               tete

## CE QUE ÇA ÉLIMINE — et cela corrige la piste extérieure d'origine

**Le « Split-V sur le vocabulaire » proposé plus tôt suppose plusieurs cartes.** La
réponse dit « trafic **par device** » : partitionner le vocabulaire répartit les octets
entre GPU, elle ne les réduit pas. **Sur une seule carte, (a) ne gagne rien en régime
mémoire-borné** — les mêmes 1,45 Gio sont lus, simplement par des SM différents. Et (b)
ne les réduit pas davantage. Nous sommes mono-carte : **deux des trois pistes
extérieures tombent sur ce seul point.**

Restent (d) et (e) pour la bande passante, et (c) qui n'est pas prêt.

## VÉRIFIÉ DANS NOTRE CODE — (d) est déjà fait, et mon 5,6 % ne vaut que pour bf16 pur

`acvram/quant/convert.py:212` :

    if name.startswith("lm_head"):
        fmt = (self.opts.lm_head_format
               or self._layer_fmt.get(self.spec.num_layers - 1, "int4_awq"))
        # la tete projette sur 151 936 classes, a 3,25 bits ses logits ne classent plus
        return "int8" if fmt == "q3n" else fmt

**La tête est déjà quantifiée au format de la dernière couche**, `int4_awq` par défaut,
avec un plancher explicite : `q3n` est promu en `int8` parce qu'à 3,25 bits les logits
ne classent plus. Donc l'option (d) — celle que l'avis extérieur donne comme la seule à
réduire *directement* les octets — **est en place depuis longtemps**, avec sa réserve de
qualité déjà écrite dans le code.

**Conséquence sur mon propre chiffre** : les 1,45 Gio et les 5,6 % du pas valent pour
`Qwen2.5-Coder-14B-bf16-pur`, le modèle du duel, qui n'est **pas** un converti. Sur un
converti `nvfp4`, la tête pèse environ le quart, soit ~0,36 Gio et ~1,4 % du pas. **Le
poste que je proposais de mesurer est donc quatre fois plus petit que je ne l'ai
annoncé sur les modèles que nous servons réellement.** J'ai transporté un chiffre du
modèle bf16 de référence vers le régime de production — la faute du dénominateur
emprunté, cette fois sur le numérateur.

## CE QU'IL RESTE, et l'ordre que j'en tire

1. **Rien à faire sur `lm_head`** : (d) est fait, (a) et (b) sont sans objet en
   mono-carte, (c) n'est pas prêt. Le Split-V sort de la liste des pistes.
2. **(e) rejoint la spéculation**, déjà en file pour deux autres raisons — occupation et
   énergie. Elle gagne ici une troisième : elle amortit la tête. Trois motifs
   indépendants pour le même chantier.
3. **À vérifier avant de s'appuyer sur ceci** : le statut « implanté dans certains
   moteurs » de (d) n'est adossé à aucun nom de fichier ; et « largement implanté » pour
   (e) est plausible mais non vérifié. Aucune de ces deux affirmations n'a besoin d'être
   vraie pour que la conclusion tienne, puisqu'elle repose sur notre propre code.
