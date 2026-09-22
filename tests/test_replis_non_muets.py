"""Pièce 38 (22/09) : deux replis silencieux rendus NON muets, sortie au bit
inchangée. Ce test échoue si le repli ChatML redevient silencieux (l'avertissement
ou le gabarit effectif disparaît). Le repli rotary de gguf.py (rope_freqs illisible
→ 0.25) expose de même `cfg["partial_rotary_source"]` et avertit une fois ; il est
couvert par relecture (chemin gemma, non fabriqué ici)."""
from acvram.server.chat import Tokenizer


def test_repli_chatml_averti_et_lisible(capsys):
    Tokenizer._repli_averti = False
    # gabarit jinja invalide → compilation échoue → repli ChatML
    tok = Tokenizer(backend=None, config={}, template="{% invalide %}{{", template_source="test")
    out = tok.apply_chat_template([{"role": "user", "content": "salut"}], add_generation_prompt=True)
    assert isinstance(out, str) and out                 # le repli produit bien un rendu
    assert tok.gabarit_effectif == "chatml-repli"       # gabarit effectif lisible (ligne de régime)
    err = capsys.readouterr().err
    assert "chatml-repli" in err                        # NON muet : échoue si le repli redevient silencieux


def test_repli_averti_une_seule_fois(capsys):
    Tokenizer._repli_averti = False
    for _ in range(3):
        tok = Tokenizer(backend=None, config={}, template="{% invalide %}{{", template_source="test")
        tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True)
    assert capsys.readouterr().err.count("chatml-repli") == 1   # une fois par processus


def test_gabarit_jinja_valide_pas_de_repli(capsys):
    tok = Tokenizer(backend=None, config={},
                    template="{% for m in messages %}{{ m['content'] }}{% endfor %}", template_source="test")
    tok.apply_chat_template([{"role": "user", "content": "bonjour"}], add_generation_prompt=False)
    assert tok.gabarit_effectif == "jinja"
    assert "chatml-repli" not in capsys.readouterr().err
