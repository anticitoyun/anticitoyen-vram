# Pièce 88 — capture échouée : justesse des hybrides, puis tri des échecs (Océane, 23/09)

instrument : `tests/test_piece88_capture_echouee.py` (à sec : primitives CUDA doublées, échec injecté à la sortie du contexte de capture)
commit : 1ff30a09 (restauration), 4e0026d3 (tri)
régime : à sec, aucune carte (tests lancés carte libre, après la fenêtre p87bis de Manon)
scellé : le test d'injection doit casser sur l'arbre d'avant pour la bonne raison (état ≠ état d'avant capture) ; chaque branche du tri a un test rouge sans le tri
mesuré : avant 1ff30a09, état GDN après capture ratée = 7,0 contre 5,0 (deux pas d'échauffement non restaurés) ; 3/3 tests du tri rouges sur 1ff30a09, 6/6 verts sur 4e0026d3 ; 75 tests ciblés graphes et régime verts
verdict : défaut de JUSTESSE confirmé et corrigé ; tri livré
durée : aucune prise de carte

1. **Justesse** (`acvram/engine/graphs.py`, `_capture`) : l'état récurrent est photographié, puis les deux pas
   d'échauffement le font avancer, puis vient la capture. La restauration n'existait que sur le chemin du succès. Une
   capture ratée laissait donc l'état avancé de deux pas, et le pas eager de repli calculait faux, sur les modèles
   hybrides seulement. Correctif : échauffement et capture dans un `try`, restauration dans le `finally`. Le chemin
   du succès exécute les mêmes instructions, dans le même ordre.
2. **Tri** (`_tri_echec_capture`, `_refus_de_cle`) :
   - transitoire (capture invalidée de l'extérieur) : la clé attend 64 pas puis se retente ; au 3ᵉ échec, elle est
     refusée ;
   - déterministe (OOM, opération non capturable) : refus de cette seule clé pour la vie du serveur ;
   - si le `synchronize` qui suit l'échec lève, le contexte est en erreur : on coupe tout, comme avant.
   Dans les deux premiers cas, les graphes sains restent, le pool partagé est abandonné pour les captures suivantes,
   et chaque pas hors graphe est compté et nommé (`repli_eager`, raisons).
Restes, nommés : (a) aucune preuve sur carte. La restauration ne sert qu'aux hybrides, et aucune prise hybride n'a
été faite. Le chemin du succès est identique par construction, mais REGLES § 3 demande une passe de capture aux godets
{1, 2, 8, 16} sur un hybride (Qwen3-Next ou Kimi-Linear) : c'est à Manon, à la prochaine fenêtre. (b) La classe
« transitoire » se reconnaît au texte du message : une nouvelle formulation du pilote tomberait dans « déterministe ».
C'est le côté sûr (refus d'une clé, jamais tout couper). (c) La ligne de régime dit `graphes=on` alors que des clés
sont refusées ; seuls `repli_eager` et ses raisons le portent.
