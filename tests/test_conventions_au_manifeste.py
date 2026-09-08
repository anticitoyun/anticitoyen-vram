"""Le manifeste transporte-t-il ce que le chargeur attend ?

Le 8/09/2026 : `config.json` portait `gdn_a_log_negexp = True`, le manifeste ne
le portait PAS, et `load_model` reconstruit la spec depuis le MANIFESTE. La
branche qui retransforme `a_log` ne s'executait donc jamais en service, le
facteur de decroissance des couches recurrentes etait transforme deux fois, et
l'ecart contre llama.cpp valait onze pour cent au lieu d'un.

Ce qui a rendu le defaut difficile : `load_model_spec(dossier)` lit
`config.json` et porte bien le drapeau. Verifier CELUI-LA et conclure que le
service est correct — ce que j'ai fait — ne dit rien du chemin reel. **Lire le
code ne remplace pas mesurer ce qu'il produit.**

Ces tests ferment la classe entiere « le manifeste ne transporte pas ce que le
chargeur attend », plutot que ce seul cas.
"""
import json

import pytest
import torch

from acvram.engine.config import ModelSpec

MINIMAL = dict(name="essai", architecture="llama", hidden_size=64,
               intermediate_size=128, num_layers=2, num_attention_heads=4,
               num_key_value_heads=4, vocab_size=100,
               max_position_embeddings=512)


def test_les_cles_utiles_sont_declarees():
    """La liste existe et n'est pas vide : c'est le contrat."""
    assert ModelSpec.CLES_BRUTES_UTILES
    assert "gdn_a_log_negexp" in ModelSpec.CLES_BRUTES_UTILES


def test_to_dict_transporte_les_cles_utiles():
    """`to_dict` jette `raw`, mais doit en preserver ce que le chargeur lit."""
    spec = ModelSpec(**MINIMAL, raw={"gdn_a_log_negexp": True,
                                     "sans_interet": 42})
    d = spec.to_dict()
    assert d.get("gdn_a_log_negexp") is True, \
        "la convention ne survit pas a la conversion : le chargeur ne la verra pas"
    assert "sans_interet" not in d, "seules les cles utiles voyagent"
    assert "raw" not in d


def test_une_cle_absente_ne_fabrique_pas_de_valeur():
    """Absente de `raw`, elle doit rester absente — pas prendre une valeur par
    defaut qui deciderait a la place de la conversion."""
    d = ModelSpec(**MINIMAL, raw={}).to_dict()
    assert "gdn_a_log_negexp" not in d


@pytest.mark.parametrize("stocke", [-0.337890625, -0.00384521484375, -304.0])
def test_la_transformation_est_son_propre_inverse(stocke):
    """`log(-x)` puis `-exp(.)` doit redonner x, sur toute la plage observee.

    C'est le calcul que fait le chargeur quand le drapeau est vrai. Les valeurs
    viennent des modeles reels : de -0,0038 a -304.
    """
    t = torch.tensor([stocke], dtype=torch.float32)
    charge = torch.log(torch.clamp(-t, min=1e-12))
    assert torch.allclose(-charge.exp(), t, rtol=1e-5), \
        f"{stocke} ne revient pas sur lui-meme"


def test_le_manifeste_dun_modele_converti_porte_la_convention(tmp_path):
    """Bout en bout : ce qu'un convertisseur ecrit, un chargeur doit le relire.

    On simule le trajet complet — spec avec la convention, serialisation au
    manifeste, relecture comme le fait `load_model` — parce que c'est
    exactement le maillon qui avait cede.
    """
    spec = ModelSpec(**MINIMAL, raw={"gdn_a_log_negexp": True})
    manifeste = {"model": spec.to_dict()}
    (tmp_path / "acvram_manifest.json").write_text(json.dumps(manifeste))

    relu = json.loads((tmp_path / "acvram_manifest.json").read_text())
    reconstruit = ModelSpec(**{k: v for k, v in relu["model"].items()
                               if k in ModelSpec.__dataclass_fields__})
    # `gdn_a_log_negexp` n'est pas un champ de la dataclasse : le chargeur doit
    # le retrouver dans le manifeste, pas dans la spec reconstruite.
    assert "gdn_a_log_negexp" not in ModelSpec.__dataclass_fields__
    assert relu["model"]["gdn_a_log_negexp"] is True, \
        "la convention n'a pas traverse le manifeste"
    assert reconstruit.name == "essai"
