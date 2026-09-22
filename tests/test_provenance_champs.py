"""Un champ absent de la source ne doit pas être écrit comme s'il était mesuré.

Contexte (audit du 10/09, corrigé le 11/09) : 4085 valeurs sont écrites dans
les manifestes sans qu'aucune source les fournisse, parce que
`load_model_spec` substitue un défaut à l'extraction (`cfg.get(cle, DEFAUT)`).
Seule la CLASSE 3 est un mensonge — un défaut *devinable et faux* qui a
l'apparence d'une mesure :

    hidden_activation   61 manifestes /126, et la source varie sur TROIS valeurs
                        (silu 52, gelu_pytorch_tanh 8, relu2 5) — un « silu »
                        deviné corrompt le MLP d'un modèle gelu ou relu².
    torch_dtype         22 /126, source bfloat16 92 / float16 12.

Ce test EXIGE le comportement cible : quand la source ne déclare NI la clé NI
son alias, le champ vaut `None`, pas le défaut. Il échoue tant que
`load_model_spec` porte encore `or "silu"` / `, "bfloat16"` — c'est voulu :
il garde la décision de forme, il ne la suppose pas.
"""
import json
import os
import tempfile

import pytest

from acvram.engine.config import load_model_spec

# une config minimale, volontairement MUETTE sur les deux champs de classe 3
_MUET = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 2048, "intermediate_size": 5632,
    "num_hidden_layers": 4, "num_attention_heads": 16,
    "num_key_value_heads": 16, "vocab_size": 32000,
    # ni hidden_activation ni hidden_act ni torch_dtype
}


def _spec(cfg):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return load_model_spec(d, "muet")


def test_source_muette_ne_devient_pas_un_defaut_devine():
    spec = _spec(_MUET)
    assert spec.hidden_activation is None, (
        "hidden_activation deviné 'silu' sur une source muette : la moitié du "
        "parc ne dit rien, et la source varie sur silu/gelu/relu² — un défaut "
        "deviné corrompt le MLP. Retirer `or \"silu\"` de load_model_spec et "
        "appliquer le repli au point d'usage (loader.py, act=).")
    assert spec.torch_dtype is None, (
        "torch_dtype deviné 'bfloat16' sur une source muette. Retirer le "
        "défaut de l'extraction ; le repli va au point d'usage (résolution "
        "KVCacheConfig).")


def test_source_qui_declare_est_conservee():
    """Le pendant : une valeur DÉCLARÉE, y compris via l'alias `hidden_act`,
    survit — sinon le remède casserait les 65 manifestes honnêtes."""
    a = _spec({**_MUET, "hidden_act": "gelu_pytorch_tanh", "torch_dtype": "float16"})
    assert a.hidden_activation == "gelu_pytorch_tanh"
    assert a.torch_dtype == "float16"
