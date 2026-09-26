"""build_tiers doit respecter --format (force_format) meme sans GPU visible.

Bogue reel (14/09/2026) : le tiers hote retombait TOUJOURS sur "int4_awq"
quand aucun GPU n'etait detecte, ignorant --format silencieusement. Cassait
toute conversion CPU-only depuis que les sessions exportent
CUDA_VISIBLE_DEVICES="" par defaut (revue/prediction-tiers-hote-force-
format-14-09.md).
"""

from acvram.hardware.detect import HostMemory, Rig
from acvram.memory.tiering import PlannerOptions, build_tiers


def _rig_sans_gpu() -> Rig:
    return Rig(gpus=[], host=HostMemory(total=64 * 2**30, available=60 * 2**30))


def test_tiers_hote_respecte_force_format_sans_gpu():
    tiers = build_tiers(_rig_sans_gpu(),
                        PlannerOptions(force_format="bf16", allow_host_tier=True))
    hote = next(t for t in tiers if t.kind == "host")
    assert hote.weight_format == "bf16"


def test_tiers_hote_sans_force_format_garde_le_repli():
    """Sans --format explicite, le repli int4_awq (economie RAM) reste
    inchange : ce test casse si le correctif est mal place et force bf16 en
    permanence, pas seulement quand demande."""
    tiers = build_tiers(_rig_sans_gpu(),
                        PlannerOptions(force_format=None, allow_host_tier=True))
    hote = next(t for t in tiers if t.kind == "host")
    assert hote.weight_format == "int4_awq"
