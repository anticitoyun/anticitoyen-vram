"""Le lot d une passe avant (`ForwardBatch`) : prefill ou décodage, jetons, positions, tables de blocs, images,
deepstack. Déplacement PUR depuis `engine/model.py` (scission 21/09, module 1/5), corps octet pour octet ;
`model` le réexporte, aucun importeur ne change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..memory.kvcache import bucket_blocks

__all__ = ["ForwardBatch"]


@dataclass

class ForwardBatch:
    """Une étape de travail, prefill ou décodage.

    ``seq_lens`` est la longueur de contexte totale de chaque séquence *après*
    ajout des jetons de ce lot : c'est ce dont l'attention a besoin pour savoir
    quelle quantité d'histoire lire.
    """

    tokens: torch.Tensor              # [total_tokens] flattened across sequences
    positions: torch.Tensor           # [total_tokens]
    seq_lens: list[int]
    query_lens: list[int]
    block_tables: list[torch.Tensor]
    slot_mapping: torch.Tensor        # [total_tokens]
    is_prefill: bool
    seq_ids: list[int] = None         # identités des séquences (états GDN)
    gdn_store: dict = None            # {layer_idx: {seq_id: état récurrent}}
    # Multimodal P1 (sage-go-multimodal-organisation-20-09 § 2) : par séquence,
    # None ou liste de (debut, fin, embeds bf16 [fin − debut, hidden]) en
    # positions absolues de l'invite ; None pour tout le lot = chemin texte
    # inchangé au bit. Rempli au prefill seulement (jamais au décodage).
    images: Optional[list] = None
    # M-RoPE (Qwen3-VL, engine/mrope, contrat sage-go-qwen3vl-parallele-20-09 § 2) :
    # positions à trois axes [3, total_tokens] (t, h, w) d'un PREFILL sur un
    # modèle à mrope_section — None = RoPE 1-D sur ``positions`` (texte, modèle
    # sans mrope : rien ne change). Le décodage reste 1-D : ``positions`` y porte
    # déjà « position 1-D + rope_delta » de la séquence (runner._build_batch),
    # ce que ``positions_on`` rend tel quel et que les graphes rejouent sans
    # rien savoir.
    positions_3d: Optional[torch.Tensor] = None
    # ``rope_delta`` par séquence (= rope_deltas de transformers, ≤ 0 ; 0 pour
    # le texte) — informatif : déjà appliqué dans ``positions`` au décodage.
    rope_delta: Optional[list] = None
    # Qwen3-VL deepstack (sage-go-qwen3vl-parallele-20-09 § 2, pièce c) : par
    # séquence, None ou liste de (debut, fin, niveaux bf16 [n_niveaux, fin − debut,
    # hidden]) ; le niveau k est AJOUTÉ après la couche k du LM aux seules lignes
    # image. None pour tout le lot = aucun tenseur touché. Prefill seulement.
    deepstack: Optional[list] = None

    def images_de(self, i: int) -> list:
        """Plages (debut, fin) de la séquence i, [] sans image."""
        if self.images is None or self.images[i] is None:
            return []
        return [(d, f) for d, f, *_ in self.images[i]]

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    # Les index d'une etape (positions, slots) naissent sur l'hote et sont lus
    # par chaque couche. Les transferer a chaque couche coutait ~0,7 ms par
    # appel — plus que le calcul de la couche elle-meme. On les copie donc une
    # fois par peripherique et par etape ; le lot est reconstruit a chaque
    # etape, le cache ne peut pas devenir obsolete.
    def positions_on(self, device: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_pos_cache", {})
        t = cache.get(device)
        if t is None:
            t = cache[device] = self.positions.to(device, non_blocking=True)
        return t

    def positions_rope_on(self, device: torch.device) -> torch.Tensor:
        """Les positions que le RoPE lit : ``positions_3d`` [3, t] au prefill
        M-RoPE, sinon ``positions_on`` (1-D, delta compris au décodage)."""
        if self.positions_3d is None:
            return self.positions_on(device)
        cache = self.__dict__.setdefault("_pos3_cache", {})
        t = cache.get(device)
        if t is None:
            t = cache[device] = self.positions_3d.to(device, non_blocking=True)
        return t

    def fixed_decode_views(self, device: torch.device
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Table de blocs complétée au godet et longueurs, en tenseurs.

        C'est la forme sous laquelle le décodage pur consomme le lot — la même
        pour le chemin eager et pour le graphe CUDA, précisément pour que le
        second reproduise le premier au bit près. Mémorisé par périphérique,
        comme les index d'étape.
        """
        cache = self.__dict__.setdefault("_fixed_cache", {})
        got = cache.get(device)
        if got is None:
            n = bucket_blocks(max(t.shape[0] for t in self.block_tables))
            tables = torch.zeros(len(self.block_tables), n, dtype=torch.long)
            for i, t in enumerate(self.block_tables):
                tables[i, : t.shape[0]] = t
            got = cache[device] = (
                tables.to(device, non_blocking=True),
                torch.tensor(self.seq_lens, dtype=torch.long).to(
                    device, non_blocking=True))
        return got

    def slots_on(self, device: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_slot_cache", {})
        t = cache.get(device)
        if t is None:
            t = cache[device] = self.slot_mapping.to(device, non_blocking=True)
        return t

    @property
    def is_decode(self) -> bool:
        return not self.is_prefill

    @property
    def q_offsets(self) -> list[int]:
        """Position absolue à laquelle commence le bloc de requêtes de chaque séquence."""
        return [s - q for s, q in zip(self.seq_lens, self.query_lens)]

    def last_token_indices(self) -> torch.Tensor:
        out, pos = [], 0
        for qlen in self.query_lens:
            pos += qlen
            out.append(pos - 1)
        return torch.tensor(out, dtype=torch.long)

    def all_token_indices(self) -> torch.Tensor:
        return torch.arange(self.tokens.shape[0], dtype=torch.long)
