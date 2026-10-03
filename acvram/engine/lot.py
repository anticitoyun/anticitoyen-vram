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
    # Multimodal P1 (poste7-go-multimodal-organisation-20-09 § 2) : par séquence,
    # None ou liste de (debut, fin, embeds bf16 [fin − debut, hidden]) en
    # positions absolues de l'invite ; None pour tout le lot = chemin texte
    # inchangé au bit. Rempli au prefill seulement (jamais au décodage).
    images: Optional[list] = None
    # M-RoPE (Qwen3-VL, engine/mrope, contrat poste7-go-qwen3vl-parallele-20-09 § 2) :
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
    # Qwen3-VL deepstack (poste7-go-qwen3vl-parallele-20-09 § 2, pièce c) : par
    # séquence, None ou liste de (debut, fin, niveaux bf16 [n_niveaux, fin − debut,
    # hidden]) ; le niveau k est AJOUTÉ après la couche k du LM aux seules lignes
    # image. None pour tout le lot = aucun tenseur touché. Prefill seulement.
    deepstack: Optional[list] = None
    # d19 (poste6 01/10) : lot d'un préfill par morceaux — (rang, total, fin, transitoires) ; `transitoires` vrai quand la passe
    # est couche-majeure (forward_tranches) : l'attention garde alors en bf16 les K/V des morceaux précédents de la couche
    morceau: Optional[tuple] = None
    # levier 2 (poste6 01/10) : créneau d'anneau de chaque séquence (−1 : aucun) et R ; les couches à fenêtre en dérivent leurs tables
    # et emplacements (`tables_fenetre`, `slots_fenetre`, `fixed_decode_views_fenetre`) — rien ne change pour les couches pleines
    anneau: Optional[list] = None
    R: int = 0

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

    def creneaux_on(self, device: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_cren_cache", {})
        t = cache.get(device)
        if t is None:
            t = cache[device] = torch.tensor(self.anneau or [-1] * self.batch_size, dtype=torch.long).to(device, non_blocking=True)
        return t

    def tables_fenetre(self, i: int, device: torch.device) -> torch.Tensor:
        """Table d'anneau de la séquence i (même longueur logique que sa table pleine)."""
        from ..memory.kvcache import tables_anneau
        cache = self.__dict__.setdefault("_tfen_cache", {})
        t = cache.get((i, device))
        if t is None:
            c = torch.tensor([self.anneau[i]], dtype=torch.long, device=device)
            t = cache[(i, device)] = tables_anneau(c, int(self.block_tables[i].shape[0]), self.R)[0]
        return t

    def slots_fenetre(self, device: torch.device) -> torch.Tensor:
        """Emplacements d'anneau des jetons du lot ; au préfill, seuls les (R − 1) derniers blocs de chaque séquence sont écrits
        (les autres ne seront jamais relus : −1, ignoré par `write`) — deux positions d'un même scatter ne visent jamais le même bloc."""
        from ..memory.kvcache import slots_anneau, BLOCK_SIZE as _bs
        cache = self.__dict__.setdefault("_sfen_cache", {})
        t = cache.get(device)
        if t is None:
            cren, fins = [], []
            for i, ql in enumerate(self.query_lens):
                cren += [self.anneau[i]] * ql; fins += [self.seq_lens[i]] * ql
            cren_t = torch.tensor(cren, dtype=torch.long); fins_t = torch.tensor(fins, dtype=torch.long)
            pos = self.positions.to(torch.long)
            if self.is_prefill:
                cren_t = torch.where(pos >= fins_t - (self.R - 1) * _bs, cren_t, torch.full_like(cren_t, -1))
            t = cache[device] = slots_anneau(pos, cren_t, self.R, _bs).to(device, non_blocking=True)
        return t

    def fixed_decode_views_fenetre(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Comme `fixed_decode_views`, tables d'anneau (même godet, mêmes longueurs)."""
        from ..memory.kvcache import tables_anneau
        cache = self.__dict__.setdefault("_fixed_fen_cache", {})
        got = cache.get(device)
        if got is None:
            tables, lens = self.fixed_decode_views(device)
            got = cache[device] = (tables_anneau(self.creneaux_on(device), int(tables.shape[1]), self.R), lens)
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
