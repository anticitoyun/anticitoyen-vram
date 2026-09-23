# Verdict — AWQ par expert dans la pile : régime NOMINAL, mais PPL REFUTEE (pire que sans AWQ)

Manon, 15/09 soir. Suite au moteur de Laurine (main `5239f27`, `_try_build_stacks`
accepte l'échelle AWQ par expert `[E,K]`) : reconversion de
`GLM-4.7-Flash-srcbf16-nvfp4` avec mon correctif `2205709` (AWQ rétabli sur les
experts, identité EXPLICITE quand la recherche s'effondre) — mais `model.py:816`
exige en plus que `gate_proj` et `up_proj` d'un même expert partagent EXACTEMENT
la même table, sans quoi la pile de cette projection retombe en boucle. J'ai donc
ajouté `_precalculer_alpha_commun_experts` (même mécanisme qu'A7, rendu
OBLIGATOIRE pour les experts, pas derrière `--alpha-commun-gate-up`) : sur 3008
paires gate/up (47 couches × 64 experts), 2818 partagent une échelle AWQ réelle,
190 sont forcées à l'identité explicite.

## Résultat

```
régime NOMINAL — graphes=on couches_exilées=0/47 experts_exilés=0/2944
piles_ok=True cartes=['cuda:0'] chemin_moe=mma
graphes CUDA : actifs (décodage), 10 godets capturés d'avance
```

**Réussite d'ingénierie complète : la pile groupée se construit sur TOUTES les
couches, avec de vraies échelles AWQ, régime NOMINAL, graphes capturés.**

**PPL 8,394 — ratio 1,03086 vs bf16 (8,1427). RÉFUTÉ**, et PIRE que la variante
« jamais d'AWQ pour un expert » de ce soir (1,02025, `verdict-glm-noawq-
experts-15-09.md`) — pire encore que sans forcer aucune échelle commune.

## Hypothèse, pas encore vérifiée

Forcer gate_proj ET up_proj d'un même expert à partager UNE échelle (au lieu de
chacun sa propre recherche AWQ indépendante) est un compromis : `search_
channel_scales_commun` optimise une métrique jointe, potentiellement moins bonne
pour l'un ET l'autre que deux recherches indépendantes. Sur 2944 experts × 2
projections, ce compromis semble coûter plus que le gain d'avoir une échelle
réelle du tout — l'exact inverse de ce qu'A7 avait mesuré sur les couches
denses (« coûte rien de mesurable »). Pourquoi le mécanisme se comporte si
différemment à cette échelle (3008 paires contre quelques dizaines chez A7) :
non investigué ce soir, ordre de pause reçu.

## Trois convertis GLM connus ce soir, aucun aux deux exigences

| converti | échelle AWQ experts | régime | PPL | ratio |
|---|---|---|---|---:|
| avant tout correctif | oui, hétérogène (pas de contrainte gate=up) | DÉGRADÉ | 8,1275 | 0,99813 |
| aa963a5 (jamais d'AWQ) | non | NOMINAL | 8,3076 | 1,02025 |
| ce soir (alpha commun forcé) | oui, gate=up forcé | NOMINAL | 8,394 | 1,03086 |

## Suite

Pas de duel ce soir sur ce converti (PPL hors seuil, pire que les deux essais
précédents). Le dossier `GLM-4.7-Flash-srcbf16-nvfp4-avant-alpha-experts-fix`
(régime NOMINAL, PPL 1,02025, aa963a5) reste le meilleur candidat mesuré à ce
jour si un duel doit avoir lieu avant que ce désaccord soit résolu. Piste à
explorer (non tentée ce soir) : ne forcer l'échelle commune QUE lorsque les
deux recherches indépendantes convergent déjà proches (éviter le compromis
quand il coûte trop), ou lever la contrainte gate=up côté moteur plutôt que
côté conversion. Rendu à Jérôme/Sage.
