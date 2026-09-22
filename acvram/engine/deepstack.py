"""Jetons image dans le LM : `disperser_images` (traits de la tour à la place des plongements, par intersection avec
le morceau), `niveaux_deepstack` et `ajouter_deepstack` (Qwen3-VL : niveaux ajoutés aux lignes image après les
couches 0..k−1, sur la somme x + delta). Déplacement PUR depuis `engine/model.py` (scission 21/09, module 2/5),
corps octet pour octet ; `model` réexporte les trois noms."""

from __future__ import annotations

from typing import Optional

import torch

from .lot import ForwardBatch

__all__ = ["disperser_images", "niveaux_deepstack", "ajouter_deepstack"]


def disperser_images(x: torch.Tensor, batch: ForwardBatch) -> torch.Tensor:
    """Remplace, ligne à ligne, l'embedding des jetons image par les traits de
    la tour de vision (multimodal P1, un seul site d'embedding).

    La séquence i occupe les lignes [r, r + query_lens[i]) de ``x`` pour les
    positions absolues [seq_lens[i] − query_lens[i], seq_lens[i]) ; une plage
    [debut, fin) n'est recopiée que sur son intersection avec ce morceau —
    le prefill par morceaux, ou un préfixe servi par le cache, ne voient
    qu'une partie de l'image. Formes vérifiées, jamais un décalage muet.
    """
    if len(batch.images) != len(batch.query_lens):
        raise ValueError(f"images : {len(batch.images)} entrées pour "
                         f"{len(batch.query_lens)} séquences")
    r = 0
    for i, qlen in enumerate(batch.query_lens):
        offset = batch.seq_lens[i] - qlen
        for d, f, e in (batch.images[i] or []):
            d, f = int(d), int(f)
            if e.ndim != 2 or e.shape[0] != f - d or e.shape[1] != x.shape[1]:
                raise ValueError(f"embeds image [{d}, {f}) : forme {tuple(e.shape)}, "
                                 f"attendu ({f - d}, {x.shape[1]})")
            a, b = max(d, offset), min(f, offset + qlen)
            if a < b:
                x[r + a - offset: r + b - offset] = e[a - d: b - d].to(x.device, x.dtype)
        r += qlen
    return x


def niveaux_deepstack(batch: ForwardBatch) -> int:
    """Nombre de niveaux deepstack portés par le lot (0 sans) ; toutes les
    entrées en portent le même nombre, sinon erreur nommée."""
    if batch.deepstack is None:
        return 0
    if len(batch.deepstack) != len(batch.query_lens):
        raise ValueError(f"deepstack : {len(batch.deepstack)} entrées pour "
                         f"{len(batch.query_lens)} séquences")
    ks = {int(n.shape[0]) for entrees in batch.deepstack for _, _, n in (entrees or [])}
    if len(ks) > 1:
        raise ValueError(f"deepstack : nombres de niveaux différents dans le lot {sorted(ks)}")
    return ks.pop() if ks else 0


def ajouter_deepstack(x: torch.Tensor, delta: Optional[torch.Tensor],
                      batch: ForwardBatch, k: int):
    """Ajoute le niveau ``k`` des traits deepstack aux lignes image, après la
    couche ``k`` du LM — la référence est ``Qwen3VLTextModel._deepstack_process`` :
    ``hidden[mask] = hidden[mask] + niveaux[k]`` en bf16, sur l'état de sortie
    de la couche (résidu + MLP déjà sommés).

    Résidu différé (x, delta) : l'état vrai est ``x + delta`` et l'ajout se fait
    sur CETTE SOMME, jamais sur ``x`` seul — bf16((x + v) + delta) ≠
    bf16((x + delta) + v), l'addition bf16 n'est pas associative. Aux seules
    lignes image : ``x ← (x + delta) + v`` et ``delta ← 0`` ; la couche suivante
    fait ``add_norm(x, delta)`` = bf16(fp32(x) + 0) = x, au bit, sans changer
    de noyau ; les lignes texte ne sont pas touchées. Rend (x, delta).
    Même règle d'intersection avec le morceau que `disperser_images`."""
    r = 0
    for i, qlen in enumerate(batch.query_lens):
        offset = batch.seq_lens[i] - qlen
        for d, f, niv in (batch.deepstack[i] or []):
            d, f = int(d), int(f)
            if niv.ndim != 3 or niv.shape[1] != f - d or niv.shape[2] != x.shape[1]:
                raise ValueError(f"deepstack [{d}, {f}) : forme {tuple(niv.shape)}, "
                                 f"attendu (n_niveaux, {f - d}, {x.shape[1]})")
            a, b = max(d, offset), min(f, offset + qlen)
            if a < b:
                lignes = slice(r + a - offset, r + b - offset)
                v = niv[k, a - d: b - d].to(x.device, x.dtype)
                if delta is None:
                    x[lignes] = x[lignes] + v
                else:
                    x[lignes] = (x[lignes] + delta[lignes]) + v
                    delta[lignes] = 0
        r += qlen
    return x, delta
