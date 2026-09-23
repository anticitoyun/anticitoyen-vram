# Recherche — écart Triton 9,5 µs vs unified_attention vLLM 5,9 µs (Q18, 23/09)

Rendu complet : `~/Bureau/Vibe/duck-reponses-23-09.md` (hors dépôt) ; synthèse : `~/Bureau/Vibe/qr.md` § Q(18).

Trois modèles (Luna, gpt-oss 120B, Gemma 4 31B, raisonnement) interrogés sur l'écart de noyau ; gpt-oss invente
une version vLLM et une désignation SM inexistantes (à ne pas citer). Vérification au code, vLLM v0.29.0 réel
(`vllm/v1/attention/ops/triton_unified_attention.py`) : les warps/étages ne sont pas figés dans le cas général
(seulement pour `head_size==256` sur B200, hors sujet ici) ; la tuile 16 est conditionnelle au dtype KV, pas une
constante ; le mécanisme absent des trois réponses est le kernel 3D à segments (`kernel_unified_attention` +
`reduce_segments()`, split-K façon flash-decoding) déclenché en decode court-contexte à petit lot — piste la plus
actionnable, à vérifier contre notre noyau avant tout réglage de warps/tuiles.
