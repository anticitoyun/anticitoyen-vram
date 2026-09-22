"""Le gabarit de chat est compile UNE FOIS, pas a chaque requete.

Mesure du 9/09 sur le gabarit de Qwen2.5 (2507 caracteres) :
    from_string (compilation)  3,975 ms
    render seul                0,007 ms   -> facteur 570

Recompiler a chaque requete coutait 3,975 ms, soit 3,7 % des 108 ms hors
forward du TTFT. Ecart de conception releve par Laurine : llama.cpp compile
une fois au chargement (common/chat.cpp:591) et ne reparse jamais ensuite.

Les tests appellent `_render_jinja` DIRECTEMENT : `apply_chat_template` avale
toute exception et retombe sur `_chatml`, donc un rendu casse y passerait
inapercu.
"""
from acvram.server.chat import Tokenizer

GABARIT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
           "{{ m['content'] }}<|im_end|>\n{% endfor %}"
           "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
MSGS = [{"role": "user", "content": "bonjour"}]


def _tok(gabarit=GABARIT):
    return Tokenizer(backend=None, config={}, template=gabarit,
                     template_source="test")


def test_le_template_compile_est_reutilise():
    t = _tok()
    t._render_jinja(MSGS, True)
    premier = t._templates[GABARIT]
    t._render_jinja(MSGS, True)
    assert t._templates[GABARIT] is premier, (
        "le Template a ete recompile : le cache ne sert a rien")


def test_le_cache_est_indexe_par_le_TEXTE_du_gabarit():
    """Un gabarit different ne doit pas se voir servir celui d'un autre.

    Une requete peut fournir son propre `chat_template` ; si le cache etait
    indexe par autre chose que le texte, elle recevrait le mauvais rendu —
    defaut silencieux, et bien pire que 4 ms.
    """
    t = _tok()
    t._render_jinja(MSGS, True)
    t.template = GABARIT.replace("assistant", "robot")
    sortie = t._render_jinja(MSGS, True)
    assert "robot" in sortie, "le cache a servi l'ancien gabarit"
    assert len(t._templates) == 2, "les deux gabarits doivent coexister"


def test_le_rendu_reste_correct():
    s = _tok()._render_jinja(MSGS, True)
    assert "bonjour" in s and s.endswith("<|im_start|>assistant\n")
