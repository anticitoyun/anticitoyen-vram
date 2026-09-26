"""Complétion BRUTE (/v1/completions, plongements) : l'invite reçoit le préfixe
littéral du gabarit de conversation (`[gMASK]<sop>` sur GLM-4.x) quand le
tokeniseur ne pose aucun jeton spécial lui-même — poste7 § 10 (cause de poste3) :
sans ce préfixe GLM n'a aucun puits d'attention et s'effondre en silence. Le
chemin conversation, dont le gabarit rendu contient déjà le préfixe, n'en
reçoit pas un second ; un texte qui le porte déjà non plus ; un gabarit qui
commence par une balise (Llama, Qwen) n'ajoute rien."""
from acvram.server.chat import Tokenizer


class _Encodage:
    def __init__(self, ids): self.ids = ids


class _Backend:
    """Tokeniseur jouet : un jeton par mot ; add_special_tokens pose un BOS
    seulement si le backend en a un (``bos``)."""
    VOCAB = {"[gMASK]": 1, "<sop>": 2, "<s>": 3}

    def __init__(self, bos=None):
        self.bos = bos

    def encode(self, text, add_special_tokens=False):
        mots = text.replace("[gMASK]<sop>", "[gMASK] <sop> ").split()
        ids = [self.VOCAB.get(m, 100 + sum(map(ord, m)) % 1000) for m in mots]
        if add_special_tokens and self.bos is not None:
            ids = [self.bos] + ids
        return _Encodage(ids)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def token_to_id(self, tok):
        return self.VOCAB.get(tok)

    def get_added_tokens_decoder(self):
        return {i: t for t, i in self.VOCAB.items()}          # les seuls jetons spéciaux


GLM = "[gMASK]<sop>\n{%- if tools -%}\n<|system|>{{ tools }}{%- endif -%}{{ messages[0].content }}"
LLAMA = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"


def test_prefixe_litteral_du_gabarit():
    t = Tokenizer(backend=_Backend(), config={}, template=GLM, template_source="test")
    assert t.prefixe_gabarit == "[gMASK]<sop>"
    assert Tokenizer(backend=_Backend(), config={}, template=LLAMA, template_source="t").prefixe_gabarit == ""
    assert Tokenizer(backend=_Backend(), config={}, template=None, template_source="t").prefixe_gabarit == ""


def test_completion_brute_recoit_le_prefixe_glm():
    t = Tokenizer(backend=_Backend(), config={}, template=GLM, template_source="test")
    brut = t.encode_brut("bonjour monde")
    assert brut[:2] == [1, 2] and brut[2:] == t.encode("bonjour monde")
    # déjà présent dans le texte : pas de doublon
    assert t.encode_brut("[gMASK]<sop>bonjour monde") == brut
    # chemin conversation : encode() ne préfixe pas (le gabarit rendu le fait)
    assert t.encode("bonjour monde")[:2] != [1, 2]


def test_bos_du_tokeniseur_suffit():
    """Un tokeniseur qui pose lui-même un BOS (post-traitement) : rien de plus,
    même si le gabarit avait un préfixe littéral."""
    t = Tokenizer(backend=_Backend(bos=3), config={"bos_token": "<s>"}, template=LLAMA, template_source="t")
    assert t.encode_brut("bonjour") == [3] + t.encode("bonjour")
    t2 = Tokenizer(backend=_Backend(bos=3), config={}, template=GLM, template_source="t")
    assert t2.encode_brut("bonjour")[:1] == [3] and 1 not in t2.encode_brut("bonjour")


def test_sans_gabarit_rien_n_est_ajoute():
    t = Tokenizer(backend=_Backend(), config={}, template=None, template_source="t")
    assert t.encode_brut("bonjour") == t.encode("bonjour")


def test_gabarit_qui_commence_par_du_texte_ordinaire_n_injecte_rien():
    """Garde (poste7) : le littéral de tête ne vaut préfixe que s'il s'encode
    entièrement en jetons spéciaux — du texte ordinaire, un espace ou un
    commentaire en tête de gabarit n'ajoutent rien."""
    for gabarit in ("Tu es un assistant.\n{{ messages[0].content }}",
                    "   {# commentaire #}{{ messages[0].content }}",
                    "[gMASK] bonjour {{ messages[0].content }}"):
        t = Tokenizer(backend=_Backend(), config={}, template=gabarit, template_source="t")
        assert t.encode_brut("bonjour monde") == t.encode("bonjour monde"), gabarit


def test_backend_sans_vocabulaire_ajoute_n_injecte_rien():
    class _SansAjoutes(_Backend):
        get_added_tokens_decoder = None
    t = Tokenizer(backend=_SansAjoutes(), config={}, template=GLM, template_source="t")
    assert t.encode_brut("bonjour") == t.encode("bonjour")
