# Équivalence CPU 2 couches GLM-4.7-Flash vs HF — note courte

poste2, 14/09/2026 soir. Détail complet, tableau des 16 positions et
discussion : `revue/verdict-equivalence-glm-2couches-14-09.md`. Cette
note résume pour la suite.

## Protocole

Seuil scellé par poste7 (`revue/poste7-lancement-14-09.md` §2) : max
|Δlogit| ≤ 5·10⁻² **et** cosinus ≥ 0,999 sur les logits de 16 positions
(invite commune, teacher-forcée), forward bf16 des couches 0-1 (dense +
première MoE) sur un mini-répertoire extrait (2,71 Gio, 223 tenseurs :
embed/norm/lm_head + couches 0 et 1 en entier, 192/192 tenseurs des 64
experts vérifiés présents). acvram (`outils/equivalence-glm-2couches.py`,
conversion bf16 sans AWQ + `Engine`) contre HF `transformers` 5.17
(`glm4_moe_lite` natif, venv vLLM). Sur le correctif MLA d'poste1 fusionné
(`16bac2f`, bead anticitoyen-vram-992).

## Résultat

**RÉFUTÉ**, reproduit sur deux passages (un GPU par accident, un CPU
propre) : pire |Δlogit| ≈ 5,0 (seuil 0,05, ×100), pire cosinus ≈ 0,95
(seuil 0,999). Aucune des 16 positions ne passe le critère Δlogit.

Le journal du passage GPU citait `« gate_proj : formats de
quantification mélangés entre experts (ni tout NVFP4, ni tout INT4) »`
(régime DÉGRADÉ) sur une conversion bf16 pure sans quantification — **ce
message est un artefact du chemin GPU** (le contrôle d'homogénéité de
pile ne s'exécute pas du tout côté CPU) : le passage CPU, sans ce
message, donne le même écart. Il ne cause pas l'écart, il ne l'explique
pas non plus.

## Conséquence

Pas de conversion GLM-4.7-Flash ce soir. Diagnostic par couche pris en
charge par poste1 (couche 0 dense+MLA seule, puis routage MoE GLM :
sigmoid + bias + expert partagé).

## Données

`revue/donnees-equivalence-glm-14-09/` : logits bruts des deux modèles
(16×154880, passage CPU).
