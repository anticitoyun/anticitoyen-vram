# SPDX-FileCopyrightText: 2026 Anticitoyen
# SPDX-License-Identifier: Apache-2.0
"""Contrat de la table d'adresses par expert (bead anticitoyen-vram-pds,
point (2) — poste7 §4, `revue/poste7-cache-experts-13-09.md`).

Ce fichier ne construit PAS encore la table (ça attend l'intégration avec le
noyau MMA de poste4) : il fixe le contrat que le noyau consomme et que le
Python remplit, pour que les deux côtés avancent sans se coordonner à chaque
ligne. Écrit par poste1 le 13/09/2026, envoyé à poste4 le même jour.

Contrat
-------

Pour une couche MoE à `E` experts, une projection (gate, up, ou down) est
servie par UNE table `[E]` int64 sur device — trois tables par couche, une
par projection, pas une table par couche : `qw`, `bscale` et `gscales`
(`kernels/acvram_kernels.cu:632-650`, noyau `nvfp4_gemv_grouped_kernel`) sont
aujourd'hui TROIS buffers distincts par projection, de tailles différentes
(`[E,M,K/2]`, `[E,M,K/16]`, `[E]`) ; une seule adresse par expert ne peut pas
retrouver les trois sans qu'ils soient déjà contigus, ce qu'ils ne sont pas.
Recomposer trois buffers séparés en un seul blob par expert est un
changement de mise en page plus large que ce geste — reporté, pas ignoré.

Donc, PAR PROJECTION (gate/up/down), DEUX tables `[E]` int64 :

    table_qw[e]     -- adresse OCTET du bloc `qw`     de l'expert `e`
    table_bscale[e] -- adresse OCTET du bloc `bscale` de l'expert `e`

`gscales` reste un petit tableau `[E]` float TOUJOURS résident en VRAM
(512 octets pour E=128) : le déplacer n'économise rien et compliquerait
tout pour rien — jamais dans la table.

Ce que porte chaque entrée :
    - Expert VRAM-résident : `data_ptr()` de sa tranche dans le buffer
      contigu habituel (le comportement actuel, sans changement de calcul).
    - Expert froid (RAM hôte) : l'adresse DEVICE zéro-copie d'un tampon
      épinglé (`cudaHostGetDevicePointer` sur un tenseur `.pin_memory()`),
      PAS l'adresse hôte — le noyau ne doit jamais recevoir un pointeur qu'il
      ne peut pas déréférencer depuis la carte.
    - JAMAIS 0. Un zéro est indiscernable d'un pointeur nul déréférencé par
      erreur ; une entrée non encore décidée n'entre pas dans la table tant
      qu'elle n'a pas une adresse réelle — `verifier_table` (ci-dessous) le
      fait respecter.

Où vit la table : sur device (`torch.int64`, `[E]`), un tenseur PERSISTANT
par (couche, projection) — même contrainte que le compteur de routage
(`MoEBlock._compter_routage`) : alloué une fois hors capture, puis modifié en
place. Mise à jour SEULEMENT hors pas (REPIN, `memory/repin.py`), jamais
pendant un pas de décodage : la capture d'un graphe CUDA fige les adresses
qu'il a vues, une table qui change de valeur pendant un pas capturé serait
relue à la prochaine capture, pas au prochain jeton.

À vérifier avant d'écrire le premier octet de noyau (pas supposé) :
qu'un tenseur `.pin_memory()` de PyTorch (allocateur CUDA par défaut) rend
bien un pointeur device utilisable via `cudaHostGetDevicePointer` sans
option d'allocation supplémentaire — dépend de l'UVA et du mappage
`cudaDeviceMapHost`, pas garanti sans un test dédié. Si non : il faudra une
petite extension C++ qui alloue avec `cudaHostAllocMapped` explicitement.
"""

from __future__ import annotations

import torch

__all__ = ["verifier_table"]


def verifier_table(table: torch.Tensor) -> None:
    """Lève si une entrée de la table est 0 (contrat : jamais 0, voir
    ci-dessus). Fonction de contrôle, pas de construction — à appeler après
    toute construction ou mise à jour, avant que le noyau ne la lise."""
    if bool((table == 0).any().item()):
        mauvais = (table == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"table d'adresses : entrée(s) nulle(s) à l'index "
                         f"{mauvais[:5]}{'…' if len(mauvais) > 5 else ''} — "
                         f"jamais 0, voir la docstring du module")
