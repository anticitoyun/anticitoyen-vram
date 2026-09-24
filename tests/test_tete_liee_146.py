"""Pièce 146 : `_tete_liee` mesure la VRAM libre APRÈS avoir rendu au pilote la réserve inutilisée de l'allocateur.
Sans cela (gemma4 31B, KV agrandi, 5,48 Gio réservés non alloués), la tête restait en bf16 et chaque appel la
convertissait en fp32 (5,25 Gio) : OOM à la chauffe. Le test rend la VRAM « libre » seulement après empty_cache."""
import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


@carte
def test_reserve_rendue_avant_de_mesurer(monkeypatch):
    from acvram.engine import loader
    monkeypatch.setattr(loader, "_TETE_LIEE", "int8")
    etat = {"vide": False}
    vrai = torch.cuda.mem_get_info
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: etat.update(vide=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda d=None: vrai(d) if etat["vide"] else (0, vrai(d)[1]))
    embed = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    w = loader._tete_liee(embed)
    assert getattr(w, "format", "") == "int8", f"tête laissée en {getattr(w, 'format', type(w).__name__)}"
