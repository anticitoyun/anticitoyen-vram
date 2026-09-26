"""Console web du serveur acvram, servie à la racine.

POURQUOI ELLE EXISTE. Le paquet s'installait sans rien de visible : ni entrée
de menu, ni icône, ni page. `acvram serve` démarrait un serveur qui ne
répondait qu'à des clients OpenAI — donc parfaitement muet pour qui venait
d'installer le `.deb` et voulait vérifier que quelque chose tournait.

CE QU'ELLE MONTRE, ET POURQUOI CES TROIS CHOSES.

* **la répartition** — quelles couches sur quel appareil, dans quel format.
  C'est la question que le projet existe pour résoudre : donner à chaque GPU
  le format que son silicium sait lire. Elle n'était visible nulle part, pas
  même dans un journal.
* **les tâches** — plusieurs invites lancées ensemble, chacune suivie
  séparément. Un serveur d'inférence sert des requêtes concurrentes ; une
  console qui n'en montre qu'une ne montre pas ce qu'il fait.
* **l'état du moteur** — ce que `/metrics` rend, sans rien recalculer. Une
  console qui recalcule finit par diverger de ce qu'elle est censée montrer.

AUCUNE DEPENDANCE. Pas de fichier statique, pas de gabarit, pas de bibliothèque
distante : une seule chaîne, servie telle quelle. Un paquet qui télécharge son
interface au premier lancement ne marche pas hors ligne, et la machine de
mesure n'a pas toujours de réseau.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>acvram — console</title>
<style>
 /* Sombre a base de roses : le fond porte deja une teinte magenta tres
    desaturee, sinon les roses posent sur du gris neutre et paraissent sales.
    Douze nuances nommees, du presque-noir au rose pale. */
 :root{
   /* Sans ceci, <select> et son menu deroulant sont rendus par le theme GTK
      natif de la machine (pas par ce CSS) : sur un theme clair, la liste des
      modeles et des cartes devient blanche sur texte clair, illisible — le
      18/09, signale par l utilisateur. `color-scheme` demande au moteur de
      rendre les CONTROLES DE FORMULAIRE (select, son popup, scrollbars) en
      sombre, sans toucher au reste de la page qui a deja ses couleurs. */
   color-scheme: dark;
   --fond:#120a10; --fond2:#180d15; --surface:#20111b; --surface2:#291624;
   --bord:#3a1f31; --bord-vif:#5c2c47;
   --rose-pale:#fce7f3; --rose-clair:#f9a8d4; --rose:#f472b6;
   --rose-vif:#ec4899; --rose-fonce:#db2777; --rose-profond:#be185d;
   --rose-nuit:#9d174d; --texte:#f7e4ef; --doux:#c9a3b8; --faible:#8d6b7e;
   --ok:#f9a8d4; --alerte:#fb7185;
 }
 *{box-sizing:border-box}
 body{margin:0;background:
        radial-gradient(1200px 600px at 15% -10%, #2a0f21 0%, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, #3a1030 0%, transparent 55%),
        var(--fond);
      color:var(--texte);font:14px/1.55 system-ui,-apple-system,sans-serif;
      min-height:100vh}
 header{padding:16px 22px;border-bottom:1px solid var(--bord);
        display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
        background:linear-gradient(90deg,rgba(190,24,93,.18),transparent 60%)}
 h1{font-size:16px;margin:0;letter-spacing:.2em;text-transform:uppercase;
    background:linear-gradient(90deg,var(--rose-clair),var(--rose-fonce));
    -webkit-background-clip:text;background-clip:text;color:transparent}
 .v{color:var(--doux);font-size:12px}
 .pastille{margin-left:auto;font-size:12px;padding:4px 12px;border-radius:99px;
        border:1px solid var(--bord-vif);color:var(--doux)}
 .ok{color:var(--ok);border-color:var(--rose-vif);
     box-shadow:0 0 12px rgba(236,72,153,.25)}
 .ko{color:var(--alerte);border-color:var(--alerte)}
 main{padding:22px;display:grid;gap:18px;max-width:1180px;margin:0 auto}
 .grille{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
 .carte{background:linear-gradient(160deg,var(--surface2),var(--surface));
        border:1px solid var(--bord);border-radius:12px;padding:13px 15px}
 .k{color:var(--faible);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
 .n{font-size:22px;margin-top:4px;font-variant-numeric:tabular-nums;
    color:var(--rose-clair)}
 .n small{font-size:12px;color:var(--faible);font-weight:400}
 section{background:var(--surface);border:1px solid var(--bord);
         border-radius:12px;padding:16px}
 h2{font-size:12px;margin:0 0 12px;color:var(--rose);text-transform:uppercase;
    letter-spacing:.12em}
 textarea,input,button,select{font:inherit;background:var(--fond2);
    color:var(--texte);border:1px solid var(--bord);border-radius:9px;
    padding:9px 12px}
 textarea{width:100%;min-height:70px;resize:vertical}
 textarea:focus,input:focus{outline:none;border-color:var(--rose-vif);
    box-shadow:0 0 0 3px rgba(236,72,153,.16)}
 .ligne{display:flex;gap:10px;margin-top:11px;align-items:center;flex-wrap:wrap}
 button{cursor:pointer;border-color:var(--rose-fonce);color:var(--rose-clair);
        background:linear-gradient(180deg,rgba(236,72,153,.16),rgba(157,23,77,.10))}
 button:hover:not(:disabled){border-color:var(--rose);color:var(--rose-pale);
        background:linear-gradient(180deg,rgba(236,72,153,.30),rgba(157,23,77,.18))}
 button:disabled{opacity:.4;cursor:default}
 button.second{border-color:var(--bord-vif);color:var(--doux);background:transparent}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th{text-align:left;color:var(--faible);font-size:11px;font-weight:500;
    text-transform:uppercase;letter-spacing:.08em;padding:0 0 6px}
 td{padding:5px 0;border-bottom:1px solid var(--bord)}
 td:not(:first-child){text-align:right;color:var(--doux)}
 .note{color:var(--faible);font-size:12px;margin-top:10px}
 #regime{cursor:pointer;font-family:monospace;padding:2px 20px;margin-top:0;user-select:all}
 code{background:var(--fond2);padding:1px 6px;border-radius:5px;
      border:1px solid var(--bord);color:var(--rose-clair)}
 .barre{height:7px;border-radius:99px;background:var(--fond2);overflow:hidden;
        margin-top:7px;border:1px solid var(--bord)}
 .barre i{display:block;height:100%;
    background:linear-gradient(90deg,var(--rose-nuit),var(--rose-vif),var(--rose-clair))}
 .tache{border:1px solid var(--bord);border-radius:10px;padding:11px 13px;
        margin-top:10px;background:var(--fond2)}
 .tache.encours{border-color:var(--rose-fonce)}
 .tache.fini{border-color:var(--bord-vif)}
 .tache.echec{border-color:var(--alerte)}
 .tete{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
 .tete b{color:var(--rose-clair);font-weight:600;font-size:13px}
 .tete span{color:var(--faible);font-size:11px;margin-left:auto}
 .rep{white-space:pre-wrap;word-break:break-word;margin:8px 0 0;
      color:var(--texte);font-size:13px}
 .puce{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;
       border:1px solid var(--bord-vif);color:var(--doux);margin-right:6px}
 .puce.f{border-color:var(--rose-fonce);color:var(--rose-clair)}
 /* Fond d ambiance : l image est posee SOUS un voile epais, pas derriere une
    transparence legere. Une console dont on ne lit plus les chiffres n est
    plus une console — l agrement ne doit jamais couter la lisibilite. */
 #fond{position:fixed;inset:0;z-index:-2;background-size:cover;
       background-position:center 22%;opacity:0;transition:opacity .5s;
       filter:saturate(1.15)}
 #voile{position:fixed;inset:0;z-index:-1;pointer-events:none;opacity:0;
        transition:opacity .5s;
        background:linear-gradient(180deg,rgba(18,10,16,.90),rgba(18,10,16,.94)),
                   radial-gradient(900px 500px at 50% 0%,rgba(236,72,153,.20),transparent 70%)}
 body.ambiance #fond{opacity:.55} body.ambiance #voile{opacity:1}
 body.ambiance section,body.ambiance .carte{backdrop-filter:blur(3px);
        background-color:rgba(32,17,27,.82)}
 .amb{display:flex;gap:7px}
 .amb button,.amb .bouton{padding:4px 13px;font-size:12px;border-radius:99px}
 .bouton{cursor:pointer;border:1px solid var(--rose-fonce);color:var(--rose-clair);
    text-decoration:none;display:inline-block;line-height:1.5;
    background:linear-gradient(180deg,rgba(236,72,153,.16),rgba(157,23,77,.10))}
 .bouton:hover{border-color:var(--rose);color:var(--rose-pale);
    background:linear-gradient(180deg,rgba(236,72,153,.30),rgba(157,23,77,.18))}
 #langue{padding:4px 10px;font-size:12px;border-radius:99px;
    border-color:var(--bord-vif);color:var(--doux)}
 #langue:focus{outline:none;border-color:var(--rose-vif)}
 .amb button.actif{border-color:var(--rose-clair);color:var(--rose-pale);
        background:linear-gradient(180deg,rgba(249,168,212,.28),rgba(236,72,153,.16))}
 .expl p{margin:0 0 9px;color:var(--doux);font-size:13px}
 .expl b{color:var(--rose-clair);font-weight:600}
 .reglage{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
          margin-top:13px}
 .reglage label{display:flex;flex-direction:column;gap:5px;font-size:11px;
        color:var(--faible);text-transform:uppercase;letter-spacing:.08em}
 .reglage input{text-transform:none;letter-spacing:normal;font-size:13px;
        color:var(--texte)}
 .refus{color:var(--alerte)}
 .parc{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
 .parc label{display:flex;flex-direction:column;gap:5px;font-size:11px;
      color:var(--faible);text-transform:uppercase;letter-spacing:.08em}
 .parc select{font-size:13px;text-transform:none;letter-spacing:normal}
 .p_expl{margin-top:13px;display:grid;gap:9px;
      grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
 .p_expl .f{background:var(--fond2);border:1px solid var(--bord);
      border-radius:9px;padding:9px 11px}
 .p_expl .f b{display:block;color:var(--rose-clair);font-size:15px;
      font-variant-numeric:tabular-nums}
 .p_expl .f span{color:var(--faible);font-size:11px;text-transform:uppercase;
      letter-spacing:.08em}
 .verdict{grid-column:1/-1;padding:9px 12px;border-radius:9px;font-size:13px;
      border:1px solid var(--bord)}
 .verdict.oui{border-color:var(--rose-vif);color:var(--rose-clair);
      background:rgba(236,72,153,.10)}
 .verdict.serre{border-color:var(--rose-nuit);color:var(--rose)}
 .verdict.non{border-color:var(--alerte);color:var(--alerte)}
 pre.cmd{white-space:pre-wrap;word-break:break-all;background:var(--fond2);
      border:1px solid var(--bord);border-radius:9px;padding:11px;margin:11px 0 0;
      color:var(--rose-clair);font-size:12.5px;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
 footer{margin-top:6px;padding-top:14px;border-top:1px solid var(--bord)}
 .pied{display:flex;gap:9px;flex-wrap:wrap;align-items:baseline;
       font-size:12px;color:var(--faible);justify-content:center}
 .pied b{letter-spacing:.22em;font-weight:700;font-size:13px;
    background:linear-gradient(90deg,var(--rose-clair),var(--rose-vif),var(--rose-fonce));
    -webkit-background-clip:text;background-clip:text;color:transparent}
 .pied a{color:var(--rose);text-decoration:none;border-bottom:1px solid transparent}
 .pied a:hover{color:var(--rose-clair);border-bottom-color:var(--rose-fonce)}
 .cartes{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
 .gpu{background:linear-gradient(160deg,var(--surface2),var(--surface));
      border:1px solid var(--bord);border-radius:11px;padding:12px 14px}
 .gpu.inactive{opacity:.62;border-style:dashed}
 .gpu h3{margin:0;font-size:13px;color:var(--rose-clair);font-weight:600}
 .gpu .adr{font-size:11px;color:var(--faible)}
 .gpu .chiffres{display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;
        font-size:12px;color:var(--doux);font-variant-numeric:tabular-nums}
 .gpu .chiffres b{color:var(--rose-clair);font-weight:600}
 /* capteurs : une carte par puce, une ligne par lecture, jauge coloree par
    rapport au seuil de la puce quand elle en donne un. */
 .capteurs{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
 .puce{background:var(--carte);border:1px solid var(--bord);border-radius:12px;padding:12px 14px}
 .puce h3{margin:0 0 8px;font-size:14px;color:var(--rose-clair)}
 .lect{display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;
   font-size:12px;padding:3px 0;border-bottom:1px dashed var(--bord)}
 .lect:last-child{border-bottom:none}
 .lect .e{color:var(--texte-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .lect .v{font-variant-numeric:tabular-nums;color:var(--rose-clair);font-weight:600;text-align:right}
 .lect .j{grid-column:1/3;height:4px;background:var(--fond);border-radius:2px;overflow:hidden}
 .lect .j i{display:block;height:100%;background:var(--rose-nuit)}
 .lect.tiede .j i{background:var(--rose-vif)} .lect.chaud .j i{background:var(--alerte)}
 .lect.chaud .v{color:var(--alerte)}
 .bridage{display:inline-block;margin:2px 4px 0 0;padding:1px 8px;border-radius:99px;
   font-size:11px;border:1px solid var(--alerte);color:var(--alerte)}
 .systeme{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin-bottom:10px}
 .systeme span b{color:var(--rose-clair)}
 #capteurs_filtre{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
 #capteurs_filtre label{font-size:12px;color:var(--texte-2);cursor:pointer}
 .mot{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:9px 0;
      border-bottom:1px solid var(--bord)}
 .mot:last-child{border-bottom:none}
 .mot .nom{color:var(--rose-clair);font-weight:600;min-width:96px}
 .mot .cmd{color:var(--faible);font-size:11px;flex:1 1 260px;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .mot .ch{color:var(--doux);font-size:12px;font-variant-numeric:tabular-nums}
 .mot button{padding:3px 11px;font-size:11px;border-radius:99px}
 .mot .perm{border-color:var(--rose-nuit);color:var(--rose)}
 .mot.soi{opacity:.7}
 .mot.intrus{background:rgba(251,113,133,.08);border-radius:8px;padding-left:8px}
 .mot.intrus .nom{color:var(--alerte)}
 .arret{border-color:var(--alerte)!important;color:var(--alerte)!important}
 .verrou{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:6px 0}
 .verrou .pastille{margin-left:0}
</style></head><body>
<div id="fond"></div><div id="voile"></div>
<header>
  <h1>acvram</h1><span class="v" id="version"></span>
  <span class="v" id="modele"></span>
  <span class="amb" id="amb">
    <button data-cle="zen"  title="Ambiance zen — touche Z">zen</button>
    <button data-cle="cool" title="Ambiance cool — touche C">cool</button>
    <a id="galerie" class="bouton" href="/galerie" target="_blank" rel="noopener"
       title="Ouvrir la galerie — touche G">galerie</a>
    <select id="langue" title="Langue de l'interface">
      <option value="fr">Français</option><option value="en">English</option>
      <option value="de">Deutsch</option><option value="es">Español</option>
      <option value="eo">Esperanto</option>
    </select>
  </span>
  <span class="pastille" id="etat">…</span>
</header>
<div id="regime" class="note" style="display:none" title="cliquer pour copier"></div>
<main>
  <div class="grille">
    <div class="carte"><div class="k" data-t="decodage">décodage</div>
      <div class="n" id="d_tok">—<small> j/s</small></div></div>
    <div class="carte"><div class="k" data-t="prefill">prefill</div>
      <div class="n" id="p_tok">—<small> j/s</small></div></div>
    <div class="carte"><div class="k" data-t="encours">en cours / en file</div>
      <div class="n" id="rw">—</div></div>
    <div class="carte"><div class="k" data-t="kvlibre">cache KV libre</div>
      <div class="n" id="kv">—</div><div class="barre"><i id="kvb" style="width:0"></i></div></div>
  </div>

  <section>
    <h2 data-t="h_cartes">cartes graphiques</h2>
    <div id="cartes" class="cartes"></div>
    <div class="note" id="cartes_note"></div>
  </section>

  <section>
    <h2 data-t="h_verrou">verrou de la carte</h2>
    <div id="verrou"></div>
    <div class="note" data-t="n_verrou">« libre » vaut « pas de verrou pose sur cette carte » — pas une lecture instantanee : un verrou peut se prendre l'instant d'apres.</div>
  </section>

  <section>
    <h2 data-t="h_moteurs">moteurs sur les cartes</h2>
    <div id="moteurs"></div>
    <div class="note" id="moteurs_note"></div>
  </section>

  <section>
    <h2 data-t="h_capteurs">capteurs et températures</h2>
    <div class="systeme" id="systeme"></div>
    <div id="capteurs_filtre">
      <label><input type="checkbox" id="f_temp" checked> <span data-t="f_temp">températures</span></label>
      <label><input type="checkbox" id="f_fan" checked> <span data-t="f_fan">ventilateurs</span></label>
      <label><input type="checkbox" id="f_in"> <span data-t="f_in">tensions</span></label>
      <label><input type="checkbox" id="f_power" checked> <span data-t="f_power">puissances</span></label>
      <label><input type="checkbox" id="f_curr"> <span data-t="f_curr">courants</span></label>
    </div>
    <div id="capteurs" class="capteurs"></div>
    <div class="note" id="capteurs_note"></div>
  </section>

  <section>
    <h2 data-t="h_parc">choisir un modèle par carte</h2>
    <div class="parc">
      <label><span data-t="l_carte">carte</span>
        <select id="p_carte"></select></label>
      <label><span data-t="l_modele">modèle compatible</span>
        <select id="p_modele"></select></label>
    </div>
    <div id="p_expl" class="p_expl"></div>
    <div class="ligne">
      <button id="p_copier" class="second" data-t="b_copier">Copier la commande</button>
      <span class="v" id="p_etat"></span>
    </div>
    <pre id="p_cmd" class="cmd"></pre>
    <div class="note" id="p_note"></div>
  </section>

  <section>
    <h2 data-t="h_repart">répartition du travail</h2>
    <table id="repart"><thead><tr><th data-t="c_appareil">appareil</th><th data-t="c_couches">couches</th>
      <th data-t="c_formats">formats</th><th data-t="c_attn">attention</th><th data-t="c_mlp">MLP</th><th data-t="c_hote">en RAM hôte</th>
      </tr></thead><tbody></tbody></table>
    <div class="note" id="repart_note"></div>
  </section>

  <section>
    <h2 data-t="h_taches">tâches</h2>
    <textarea id="invites" placeholder="Une invite par ligne — elles partent toutes ensemble."
>Explique NVFP4 en une phrase.
Qu'est-ce que l'attention paginée ?
Donne trois usages d'un cache KV quantifié.</textarea>
    <div class="ligne">
      <button id="envoyer" data-t="b_distribuer">Distribuer</button>
      <button id="vider" class="second" data-t="b_vider">Vider</button>
      <label class="v"><span data-t="l_jetons">jetons</span> <input id="max" type="number" value="128"
             min="1" max="4096" style="width:88px"></label>
      <span class="v" id="resume"></span>
    </div>
    <div id="taches"></div>
    <div class="note">Chaque ligne devient une requête <code>/v1/chat/completions</code>,
      envoyée en même temps que les autres. Le moteur les met en lot lui-même :
      la colonne « en cours / en file » ci-dessus montre ce qu'il en fait.</div>
  </section>

  <section>
    <h2 data-t="h_regl">explications &amp; réglages</h2>
    <div class="expl" id="expl"></div>
    <div class="reglage">
      <label><span data-t="l_zen">fond « zen »</span><input id="r_zen" type="text" spellcheck="false"
             placeholder="/chemin/vers/une/image.jpg"></label>
      <label><span data-t="l_cool">fond « cool »</span><input id="r_cool" type="text" spellcheck="false"
             placeholder="/chemin/vers/une/autre.png"></label>
    </div>
    <div class="ligne">
      <button id="enregistrer" data-t="b_enreg">Enregistrer</button>
      <span class="v" id="r_etat"></span>
    </div>
    <div class="note" id="r_note"></div>
  </section>

  <section>
    <h2 data-t="h_moteur">moteur</h2>
    <div id="tronque" class="note" style="display:none"></div>
    <div id="energie" class="note"></div>
    <table id="details"><tbody></tbody></table>
    <div class="note">Pour brancher un client :
      <code>OPENAI_BASE_URL</code> ou <code>ANTHROPIC_BASE_URL</code> sur
      l'adresse de cette page. Ambiances : touches <code>Z</code> et
      <code>C</code>, ou les deux boutons en haut à droite ; un second appui
      retire le fond. Galerie : touche <code>G</code> ou le bouton
      « galerie », qui ouvre une fenêtre séparée.</div>
  </section>
  <footer>
    <div class="pied">
      <span><b>ANTICITOYEN</b></span>
      <span>·</span>
      <span>acvram — serveur d'inférence LLM pour GPU hétérogènes</span>
      <span>·</span>
      <span>GNU <a href="https://www.gnu.org/licenses/gpl-3.0.html"
            target="_blank" rel="noopener">GPL-3.0-or-later</a></span>
      <span>·</span>
      <a href="https://outils.nuages.noho.st/gitlab/anticitoyen/anticitoyen-vram"
         target="_blank" rel="noopener">code source sur GitLab</a>
    </div>
  </footer>
</main>
<script>
const $ = i => document.getElementById(i);
const nb = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toLocaleString('fr-FR', {maximumFractionDigits: d});
const gio = o => (o && o > 0) ? nb(o / 1073741824, 2) + ' Gio' : '—';

let modele = null;

async function rafraichir() {
  try {
    const m = await (await fetch('/metrics')).json();
    const e = m.engine || {};
    $('etat').textContent = 'en ligne';
    $('etat').className = 'pastille ok';
    $('d_tok').innerHTML = nb(e.decode_tok_s) + '<small> j/s</small>';
    $('p_tok').innerHTML = nb(e.prefill_tok_s) + '<small> j/s</small>';
    $('rw').textContent = (e.running ?? '—') + ' / ' + (e.waiting ?? '—');
    $('kv').textContent = nb(e.kv_blocks_free, 0) + ' / ' + nb(e.kv_blocks_total, 0);
    if (e.kv_blocks_total > 0)
      $('kvb').style.width = (100 * e.kv_blocks_free / e.kv_blocks_total) + '%';
    if (m.version) $('version').textContent = 'v' + m.version;

    // Ajout n°2 (poste7-gui-ajouts-18-09) : un compteur > 0 noye dans le
    // tableau generique ci-dessous n'a jamais ete vu — « toutes les
    // cellules b > 8 faussees jusqu'au 17/09 sans une ligne d'erreur ».
    // Le ratio kv_max_tokens / kv_planned_seqs donne le budget REEL par
    // sequence planifiee, pour lire un compteur > 0 sans deviner s'il est
    // structurel (budget sous-dimensionne) ou accidentel.
    const tronq = $('tronque');
    if ((e.sequences_tronquees_budget || 0) > 0) {
      tronq.style.display = '';
      tronq.className = 'note ko';
      tronq.textContent = e.sequences_tronquees_budget + ' séquence(s) tronquée(s) par '
        + 'budget KV épuisé (hors max_tokens demandé) — budget : '
        + nb(m.kv_tokens_par_sequence_planifiee, 0) + ' jetons/séquence planifiée ('
        + nb(m.kv_max_tokens, 0) + ' / ' + nb(m.kv_planned_seqs, 0) + ' séquences).';
    } else {
      tronq.style.display = 'none';
    }

    // Ajout n°3 (poste7-gui-ajouts-18-09 § 3) : J/jeton, integre cote serveur
    // sur la fenetre glissante entre deux appels a /metrics (compteur NVML
    // monotone, pas une moyenne de puissances) — a vide "—", jamais 0.
    const nrj = m.energie || {};
    const plafonds = (nrj.cartes || [])
      .map(c => nb(c.horloge_sm, 0) + ' MHz / ' + nb(c.watts_plafond, 0) + ' W')
      .join(', ');
    $('energie').textContent = 'énergie : ' + nb(nrj.j_par_jeton_10s, 3)
      + ' J/jeton (' + nb(nrj.jetons_fenetre, 0) + ' jetons / ' + nb(nrj.fenetre_s, 0) + ' s)'
      + (plafonds ? ' — horloge SM / plafond : ' + plafonds : '');

    // Ajout n°4 (poste7-gui-ajouts-18-09 § 4) : la meme ligne "[régime] ..."
    // qu'un JSON de mesure — copiable, pour qu'un rapport puisse la coller
    // telle quelle plutot que de retaper les variables ACVRAM_* actives.
    if (m.regime_ligne) {
      $('regime').style.display = '';
      $('regime').textContent = m.regime_ligne;
    }

    // Le tableau montre ce que /metrics rend, sans trier ni interpreter :
    // une console qui choisit ce qu'elle affiche cache ce qu'elle omet.
    const t = $('details').tBodies[0];
    t.innerHTML = '';
    for (const [k, v] of Object.entries(e)) {
      if (typeof v === 'object') continue;
      const tr = t.insertRow();
      tr.insertCell().textContent = k;
      tr.insertCell().textContent = typeof v === 'number' ? nb(v, 3) : String(v);
    }
  } catch (err) {
    $('etat').textContent = 'injoignable';
    $('etat').className = 'pastille ko';
  }
}

// -- cartes graphiques ----------------------------------------------------
// Toutes les cartes de la machine, pas seulement celle qui sert. Une console
// qui ne montrerait que la carte utilisee laisserait croire que la machine
// n en a qu une — et c est precisement le sujet du projet : plusieurs GPU
// heterogenes, chacun avec le format que son silicium sait lire.
async function charger_cartes() {
  try {
    const r = await (await fetch('/materiel')).json();
    const box = $('cartes');
    box.innerHTML = '';
    for (const c of r.cartes || []) {
      const el = document.createElement('div');
      el.className = 'gpu' + (c.pilotee ? '' : ' inactive');
      let corps = '<h3>' + c.nom + '</h3><div class="adr">' + c.adresse
                + ' · ' + (c.fabricant || '') + (c.pilotee ? '' : ' · non pilotée') + '</div>';
      if (c.pilotee) {
        const pct = c.mio_total ? (100 * c.mio_pris / c.mio_total) : 0;
        corps += '<div class="barre"><i style="width:' + pct.toFixed(1) + '%"></i></div>'
          + '<div class="chiffres">'
          + '<span><b>' + nb(c.mio_pris / 1024, 1) + '</b> / ' + nb(c.mio_total / 1024, 1) + ' Gio</span>'
          + '<span><b>' + nb(c.occupation, 0) + '</b> %</span>'
          + '<span><b>' + nb(c.temperature, 0) + '</b> °C</span>'
          + '<span><b>' + nb(c.watts, 0) + '</b> / ' + nb(c.watts_max, 0) + ' W</span>'
          + '<span>PCIe x' + (c.pcie_largeur || '?') + '</span>'
          + '</div>';
      } else {
        corps += '<div class="chiffres"><span>présente, pas de compute</span></div>';
      }
      el.innerHTML = corps;
      box.appendChild(el);
    }
    $('cartes_note').textContent =
      (r.cartes || []).length + ' carte(s) détectée(s) — nvidia-smi pour l\\'état, '
      + 'lspci pour l\\'inventaire complet'
      + (r.servie ? ' · ce serveur sert CUDA_VISIBLE_DEVICES=' + r.servie : '');
  } catch (e) {
    $('cartes_note').textContent = 'inventaire indisponible : ' + e.message;
  }
}

// -- verrou de la carte -----------------------------------------------------
// « Une annonce a une minute de retard, une fenetre GUI non » (poste7-gui-
// ajouts-18-09) : le seul incident du circuit qui coute plus qu'une seule
// mesure (six manches le 10/09, rejoue le 14 et le 15) est celui que la
// console peut voir avant qu'une session ne l'annonce.
async function charger_verrou() {
  try {
    const r = await (await fetch('/verrou')).json();
    const box = $('verrou');
    box.innerHTML = '';
    if (!(r.verrous || []).length) {
      box.innerHTML = '<div class="verrou"><span class="pastille ok">libre</span></div>';
      return;
    }
    for (const v of r.verrous) {
      const el = document.createElement('div');
      el.className = 'verrou';
      if (v.tenu) {
        const duree = v.depuis_secondes > 3600
          ? Math.floor(v.depuis_secondes / 3600) + ' h'
          : Math.floor(v.depuis_secondes / 60) + ' min ' + (v.depuis_secondes % 60) + ' s';
        el.innerHTML =
          '<span class="pastille">carte ' + (v.carte ?? '?') + '</span>'
          + '<span class="ch">' + v.nom + ' (' + v.type + ')</span>'
          + '<span class="ch">pid ' + v.pid + '</span>'
          + '<span class="ch">depuis ' + duree + '</span>';
      } else {
        el.innerHTML =
          '<span class="pastille ok">carte ' + (v.carte ?? '?') + ' libre</span>';
      }
      box.appendChild(el);
    }
  } catch (e) {
    $('verrou').innerHTML = '<div class="note">verrou indisponible : ' + e.message + '</div>';
  }
}

// -- moteurs voisins ------------------------------------------------------
// Une machine de mesure porte souvent plusieurs serveurs d inference. Un
// voisin qui calcule pendant qu on mesure FABRIQUE le resultat — c est arrive
// trois fois dans une seule soiree. La console les montre, et permet de les
// arreter, ce qui est destructif et se traite comme tel : deux clics, et les
// services permanents demandent une confirmation de plus.
let a_confirmer = null;

async function charger_moteurs() {
  try {
    const r = await (await fetch('/moteurs')).json();
    const box = $('moteurs');
    box.innerHTML = '';
    for (const m of r.moteurs || []) {
      const el = document.createElement('div');
      // « intrus » = ni port permanent ni verrou de sa carte (legitime,
      // calcule cote serveur) — un rouge que la console ne trie ni ne juge,
      // elle affiche le champ tel que /moteurs le rend.
      el.className = 'mot' + (m.moi ? ' soi' : '') + (!m.moi && !m.legitime ? ' intrus' : '');
      const duree = m.secondes === null ? ''
        : (m.secondes > 3600 ? Math.floor(m.secondes / 3600) + ' h'
           : Math.floor(m.secondes / 60) + ' min');
      el.innerHTML =
        '<span class="nom">' + m.moteur + '</span>'
        + '<span class="ch">pid ' + m.pid + '</span>'
        + '<span class="ch">' + nb(m.mio / 1024, 1) + ' Gio</span>'
        + '<span class="ch">carte ' + (m.carte ?? '?') + '</span>'
        + (duree ? '<span class="ch">' + duree + '</span>' : '')
        + (m.ports.length ? '<span class="ch">:' + m.ports.join(' :') + '</span>' : '')
        + '<span class="cmd" title="' + m.commande.replace(/"/g, '&quot;') + '">'
          + m.commande + '</span>';
      if (m.moi) {
        el.innerHTML += '<span class="ch">ce serveur</span>';
      } else {
        const b = document.createElement('button');
        b.textContent = m.permanent ? 'permanent' : 'arrêter';
        if (m.permanent) b.className = 'perm';
        b.title = m.permanent
          ? 'service permanent (' + m.ports.join(', ') + ') — l\\'arrêter est une panne, pas un nettoyage'
          : 'envoie SIGTERM';
        b.onclick = () => demander_arret(b, m);
        el.appendChild(b);
      }
      box.appendChild(el);
    }
    if (!(r.moteurs || []).length)
      box.innerHTML = '<div class="note">aucun processus sur les cartes</div>';
    $('moteurs_note').textContent =
      'ports protégés : ' + (r.ports_permanents || []).join(', ')
      + ' — un service permanent demande une seconde confirmation.';
  } catch (e) {
    $('moteurs_note').textContent = 'inventaire des moteurs indisponible : ' + e.message;
  }
}

async function demander_arret(bouton, m) {
  // DEUX CLICS, toujours. Le premier arme, le second execute — et le bouton
  // dit ce qui va se passer entre les deux. Un arret a un clic sur une liste
  // qui se rafraichit toutes les deux secondes tuerait le mauvais processus
  // le jour ou la ligne bouge.
  if (a_confirmer !== m.pid) {
    a_confirmer = m.pid;
    bouton.classList.add('arret');
    bouton.textContent = m.permanent
      ? 'confirmer (PERMANENT)' : 'confirmer';
    setTimeout(() => {
      if (a_confirmer === m.pid) { a_confirmer = null; charger_moteurs(); }
    }, 6000);
    return;
  }
  a_confirmer = null;
  bouton.disabled = true; bouton.textContent = '…';
  try {
    const r = await fetch('/moteurs/arreter', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pid: m.pid, force: m.permanent}),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || ('HTTP ' + r.status));
    // On ne dit pas « arrete » : un moteur met plusieurs secondes a rendre sa
    // VRAM, et l affirmer serait une conclusion non verifiee. La liste se
    // relit, et c est elle qui dira s il est parti.
    $('moteurs_note').textContent =
      'SIGTERM envoyé à ' + j.moteur + ' (pid ' + j.pid + ') — la liste dira s\\'il part.';
  } catch (err) {
    $('moteurs_note').textContent = 'refus : ' + err.message;
  } finally {
    setTimeout(charger_moteurs, 1200);
  }
}

// -- parc de modeles ------------------------------------------------------
// CE QUE CETTE SECTION NE FAIT PAS : elle ne change pas le modele servi. Un
// moteur charge ses poids au demarrage ; pretendre en changer a chaud serait
// un mensonge d interface. Elle prepare la COMMANDE, et dit honnetement ce
// qu'il faut lancer — ce qui est aussi ce que fait le menu du systeme.
let PARC = null;

const gg = x => nb(x, 2) + ' Gio';   // deja en Gio, contrairement a gio()

async function charger_parc() {
  try {
    PARC = await (await fetch('/parc')).json();
  } catch (e) { $('p_note').textContent = 'parc indisponible : ' + e.message; return; }
  const sc = $('p_carte');
  sc.innerHTML = '';
  for (const c of PARC.cartes || []) {
    const o = document.createElement('option');
    o.value = c.index;
    o.textContent = '[' + c.index + '] ' + c.nom
      + ' — ' + gg((c.mio_total - c.mio_pris) / 1024) + ' libres sur '
      + gg(c.mio_total / 1024);
    sc.appendChild(o);
  }
  // La 3080 Ti d abord si elle existe : c est la carte pour laquelle la
  // question se pose, la 5090 prenant a peu pres tout.
  if ((PARC.cartes || []).length > 1) sc.selectedIndex = 1;
  remplir_modeles();
}

function remplir_modeles() {
  if (!PARC) return;
  const idx = Number($('p_carte').value);
  const c = (PARC.cartes || []).find(x => x.index === idx);
  const sm = $('p_modele');
  const garde = sm.value;
  sm.innerHTML = '';
  const compat = new Set(c ? c.compatibles : []);
  const libres = new Set(c ? c.compatibles_maintenant : []);
  let n = 0;
  for (const m of PARC.modeles || []) {
    if (!compat.has(m.nom)) continue;
    n++;
    const o = document.createElement('option');
    o.value = m.nom;
    // Le marqueur dit s'il tient MAINTENANT, avec ce qui est deja pris. Un
    // modele qui tient sur une carte vide et pas sur la carte actuelle n est
    // pas le meme choix, et la liste doit le montrer sans qu'on calcule.
    o.textContent = (libres.has(m.nom) ? '● ' : '○ ') + m.nom
      + ' — ' + gg(m.gio) + ' · ' + m.genres.join('+');
    sm.appendChild(o);
  }
  if (garde && compat.has(garde)) sm.value = garde;
  $('p_note').textContent = n
    ? n + ' modèle(s) tiennent sur cette carte · ● tient maintenant, '
      + '○ tient sur la carte vide · marge ×' + PARC.marge
      + ' au-dessus des poids (contexte, activations, arène) · parc : ' + PARC.dossier
    : 'aucun modèle du parc ne tient sur cette carte · parc : ' + PARC.dossier;
  decrire();
}

function decrire() {
  if (!PARC) return;
  const idx = Number($('p_carte').value);
  const c = (PARC.cartes || []).find(x => x.index === idx);
  const m = (PARC.modeles || []).find(x => x.nom === $('p_modele').value);
  const box = $('p_expl');
  if (!m || !c) { box.innerHTML = ''; $('p_cmd').textContent = ''; return; }
  const libre = (c.mio_total - c.mio_pris) / 1024, total = c.mio_total / 1024;
  const tient_maintenant = m.gio_requis <= libre;
  const marge = total - m.gio_requis;
  const f = (v, k) => '<div class="f"><b>' + v + '</b><span>' + k + '</span></div>';
  let verdict, cls;
  if (tient_maintenant) {
    verdict = 'Tient maintenant : ' + gg(m.gio_requis) + ' requis, '
            + gg(libre) + ' libres.'; cls = 'oui';
  } else if (m.gio_requis <= total) {
    verdict = 'Tient sur la carte vide (' + gg(m.gio_requis) + ' requis sur '
            + gg(total) + '), mais ' + gg(libre)
            + " seulement sont libres — liberez la carte avant."; cls = 'serre';
  } else {
    verdict = 'Ne tient pas : ' + gg(m.gio_requis) + ' requis pour '
            + gg(total) + '.'; cls = 'non';
  }
  box.innerHTML =
      f(gg(m.gio), 'poids sur disque')
    + f(gg(m.gio_requis), 'requis avec marge')
    + f(m.params_total + ' G', 'paramètres')
    + f((m.params_actifs || m.params_total) + ' G', 'actifs par jeton')
    + f(m.genres.join(' + '), 'architecture')
    + f(m.formats.join(' '), 'formats')
    + f(m.couches, 'couches')
    + f(nb(m.contexte_max, 0), 'contexte max')
    + '<div class="verdict ' + cls + '">' + verdict + '</div>';
  // La commande porte la carte ET le verrou : les deux ont manque ce soir,
  // et chacun a coute une manche de mesure a quelqu'un.
  $('p_cmd').textContent =
    'ACVRAM_VERROU=/tmp/acvram-carte-' + idx + '.lock ACVRAM_NOM=serveur-' + idx + ' \\\n'
    + '  outils/carte.sh env CUDA_VISIBLE_DEVICES=' + idx + ' \\\n'
    + '  acvram serve "' + m.chemin + '" --port ' + (8100 + idx)
    + ' --max-model-len 4096';
}

$('p_carte').onchange = remplir_modeles;
$('p_modele').onchange = decrire;
$('p_copier').onclick = async () => {
  try {
    await navigator.clipboard.writeText($('p_cmd').textContent);
    $('p_etat').textContent = 'copiée';
  } catch (e) {
    // Le presse-papiers exige un contexte sur : sur http:// il est refuse.
    // Le dire plutot que de laisser croire que la copie a eu lieu.
    $('p_etat').textContent = 'copie refusée par le navigateur — sélectionnez le texte';
  }
  setTimeout(() => { $('p_etat').textContent = ''; }, 4000);
};

async function charger_repartition() {
  try {
    const r = await (await fetch('/repartition')).json();
    const t = $('repart').tBodies[0];
    t.innerHTML = '';
    for (const [dev, d] of Object.entries(r.appareils || {})) {
      const tr = t.insertRow();
      tr.insertCell().textContent = dev;
      tr.insertCell().textContent = d.couches + (d.moe ? ' (' + d.moe + ' MoE)' : '');
      const c = tr.insertCell();
      c.innerHTML = Object.entries(d.formats)
        .map(([f, n]) => '<span class="puce f">' + f + ' ×' + n + '</span>').join('');
      c.style.textAlign = 'right';
      tr.insertCell().textContent = gio(d.attn_octets);
      tr.insertCell().textContent = gio(d.mlp_octets);
      tr.insertCell().textContent = d.mlp_hote ? gio(d.mlp_hote) : '—';
    }
    const av = (r.avertissements || []).length
      ? ' · ' + r.avertissements.length + ' avertissement(s) du planificateur' : '';
    $('repart_note').textContent =
      'plongement sur ' + r.embed + ' · tête de sortie sur ' + r.lm_head
      + ' · cache KV ' + nb(r.kv_octets_par_jeton, 0) + ' o/jeton, '
      + nb(r.kv_jetons_max, 0) + ' jetons au plus'
      + ' · estimation du plan : ' + nb(r.estimation_decode_j_s) + ' j/s en décodage'
      + av;
  } catch (err) {
    $('repart_note').textContent = 'répartition indisponible : ' + err.message;
  }
}

async function charger_modele() {
  try {
    const d = await (await fetch('/v1/models')).json();
    modele = d.data?.[0]?.id ?? null;
    if (modele) $('modele').textContent = modele;
  } catch (err) { /* la pastille dit deja que le serveur ne repond pas */ }
}

function carte_tache(i, invite) {
  const el = document.createElement('div');
  el.className = 'tache encours';
  el.innerHTML = '<div class="tete"><b>#' + (i + 1) + '</b>'
    + '<span class="etat">en cours…</span></div>'
    + '<div class="v"></div><div class="rep"></div>';
  el.querySelector('.v').textContent = invite;
  return el;
}

$('vider').onclick = () => { $('taches').innerHTML = ''; $('resume').textContent = ''; };

$('envoyer').onclick = async () => {
  const lignes = $('invites').value.split('\\n').map(s => s.trim()).filter(Boolean);
  if (!lignes.length) return;
  const b = $('envoyer');
  b.disabled = true;
  $('taches').innerHTML = '';
  const cartes = lignes.map((inv, i) => {
    const c = carte_tache(i, inv);
    $('taches').appendChild(c);
    return c;
  });
  const t0 = performance.now();
  let finis = 0, jetons = 0;
  $('resume').textContent = '0 / ' + lignes.length;

  // TOUTES lancees d un coup, sans attendre les precedentes : c est le point
  // de cette section. Les lancer en serie montrerait un debit sequentiel et
  // cacherait exactement ce que le moteur sait faire.
  await Promise.all(lignes.map(async (inv, i) => {
    const el = cartes[i], t1 = performance.now();
    try {
      const r = await fetch('/v1/chat/completions', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          model: modele, max_tokens: Number($('max').value) || 128,
          messages: [{role: 'user', content: inv}],
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error?.message || ('HTTP ' + r.status));
      const u = j.usage || {}, s = (performance.now() - t1) / 1000;
      jetons += u.completion_tokens || 0;
      el.className = 'tache fini';
      el.querySelector('.rep').textContent =
        j.choices?.[0]?.message?.content ?? JSON.stringify(j, null, 2);
      el.querySelector('.etat').textContent =
        s.toFixed(2) + ' s · ' + (u.completion_tokens ?? '?') + ' jetons';
    } catch (err) {
      el.className = 'tache echec';
      el.querySelector('.etat').textContent = 'échec';
      el.querySelector('.rep').textContent = err.message;
    } finally {
      finis++;
      $('resume').textContent = finis + ' / ' + lignes.length;
    }
  }));

  // Le debit affiche ici est celui du CLIENT : il inclut le reseau et
  // l attente en file. Il est donc TOUJOURS inferieur a celui du moteur, et
  // les deux sont montres pour que l ecart soit visible plutot que suppose.
  const s = (performance.now() - t0) / 1000;
  $('resume').textContent = lignes.length + ' tâches · ' + s.toFixed(2) + ' s · '
    + jetons + ' jetons · ' + nb(jetons / s) + ' j/s vus du client';
  b.disabled = false;
};

// -- ambiances ------------------------------------------------------------
// Deux fonds, deux raccourcis. L etat est garde dans le navigateur : une
// console qu on rouvre vingt fois par jour ne doit pas redemander son reglage.
// Un CLIC SUR L AMBIANCE ACTIVE la retire — sinon il n y a aucun moyen de
// revenir au fond nu, et un reglage sans retour arriere est un piege.
let ambiance = null;

function poser_ambiance(cle) {
  ambiance = (ambiance === cle) ? null : cle;
  document.body.classList.toggle('ambiance', ambiance !== null);
  if (ambiance) $('fond').style.backgroundImage = 'url(/fond/' + ambiance + ')';
  for (const b of document.querySelectorAll('.amb button'))
    b.classList.toggle('actif', b.dataset.cle === ambiance);
  try { localStorage.setItem('acvram_ambiance', ambiance || ''); } catch (e) {}
}

async function init_ambiances(rejeu) {
  let dispo = {};
  try { dispo = await (await fetch('/fonds')).json(); } catch (e) {}
  for (const b of document.querySelectorAll('.amb button')) {
    const c = b.dataset.cle;
    // Un bouton actif qui ne montre rien serait pire que pas de bouton : la
    // page dirait que le fond existe alors que le fichier est introuvable.
    if (!dispo[c]) {
      b.disabled = true;
      b.title = 'aucun fond « ' + c + ' » : poser '
                + (c === 'zen' ? 'ACVRAM_FOND_ZEN' : 'ACVRAM_FOND_COOL')
                + ' sur un fichier image, ou le régler dans la console';
      continue;
    }
    b.onclick = () => poser_ambiance(c);
  }
  if (rejeu) {
    // Apres un changement de chemin, l image du fond courant doit etre
    // rechargee : le navigateur garderait l ancienne sous la meme URL.
    if (ambiance && dispo[ambiance])
      $('fond').style.backgroundImage = 'url(/fond/' + ambiance + '?t=' + Date.now() + ')';
    return;
  }
  let garde = '';
  try { garde = localStorage.getItem('acvram_ambiance') || ''; } catch (e) {}
  if (garde && dispo[garde]) poser_ambiance(garde);
}

document.addEventListener('keydown', ev => {
  if (ev.target.matches('input,textarea') || ev.ctrlKey || ev.altKey || ev.metaKey) return;
  const k = ev.key.toLowerCase();
  if (k === 'z' || k === 'c') {
    const b = document.querySelector('.amb button[data-cle="' + (k === 'z' ? 'zen' : 'cool') + '"]');
    if (b && !b.disabled) { poser_ambiance(b.dataset.cle); ev.preventDefault(); }
  }
});

// -- reglages -------------------------------------------------------------
function montrer_reglages(r) {
  $('r_zen').value = r.fonds?.zen?.chemin || '';
  $('r_cool').value = r.fonds?.cool?.chemin || '';
  const etats = ['zen', 'cool'].map(c => {
    const f = r.fonds?.[c];
    if (!f || !f.chemin) return c + ' : non réglé';
    return c + (f.servable ? ' : servable' : ' : INTROUVABLE');
  });
  $('r_note').textContent = etats.join(' · ')
    + (r.fichier ? ' · gardé dans ' + r.fichier : '')
    + (r.extensions ? ' · formats servis : ' + r.extensions.join(' ') : '');
}

$('enregistrer').onclick = async () => {
  const b = $('enregistrer');
  b.disabled = true; $('r_etat').textContent = '…';
  try {
    const r = await (await fetch('/reglages', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({zen: $('r_zen').value, cool: $('r_cool').value}),
    })).json();
    const refus = Object.entries(r.refus || {});
    // Un reglage refuse doit le DIRE avec sa raison : « enregistre » sur un
    // chemin rejete serait le meme mensonge que le bouton actif qui ne montre
    // rien.
    $('r_etat').innerHTML = refus.length
      ? '<span class="refus">' + refus.map(([c, m]) => c + ' — ' + m).join(' ; ') + '</span>'
      : (r.garde ? 'enregistré' : 'appliqué, mais NON gardé sur disque');
    montrer_reglages(r);
    await init_ambiances(true);
  } catch (err) {
    $('r_etat').innerHTML = '<span class="refus">échec : ' + err.message + '</span>';
  } finally { b.disabled = false; }
};

async function charger_reglages() {
  try { montrer_reglages(await (await fetch('/reglages')).json()); }
  catch (e) { $('r_note').textContent = 'réglages indisponibles'; }
}

// LA GALERIE EST UN LIEN, PAS UN window.open.
//
// La premiere version appelait window.open('/galerie', nom, 'width=1280,...').
// Un window.open AVEC DIMENSIONS est traite comme une pop-up par Chromium et
// bloque en silence : le clic ne faisait rien, aucune erreur, aucune fenetre.
// Un lien <a target="_blank"> clique par l utilisateur n est jamais bloque —
// c est le meme geste, et le navigateur le lit comme une navigation et non
// comme une fenetre imposee.
//
// La touche G declenche le clic sur le lien plutot que d ouvrir elle-meme :
// le geste reste celui de l utilisateur, donc la meme regle s applique.
document.addEventListener('keydown', ev => {
  if (ev.target.matches('input,textarea') || ev.ctrlKey || ev.altKey || ev.metaKey) return;
  if (ev.key.toLowerCase() === 'g') { $('galerie').click(); ev.preventDefault(); }
});


// -- traduction -----------------------------------------------------------
// Cinq langues, un seul dictionnaire. Les textes VIVENT ICI et non dans le
// HTML : une page qui porterait cinq versions de chaque paragraphe serait
// cinq fois plus lourde, et quatre versions sur cinq seraient invisibles a
// qui la relit.
//
// Ce qui n est PAS traduit, et c est deliberе : les noms de champs de
// /metrics, les formats (nvfp4, int4_awq), les noms de moteurs et les unites.
// Ce sont des identifiants — les traduire romprait le lien avec le code, et
// c est la meme regle que dans le depot : le francais partout SAUF ce qui
// constitue un contrat externe.
const T = {
 fr:{h_parc:'choisir un modèle par carte',l_carte:'carte',l_modele:'modèle compatible',b_copier:'Copier la commande',decodage:'décodage',prefill:'prefill',encours:'en cours / en file',
  kvlibre:'cache KV libre',h_cartes:'cartes graphiques',
  h_verrou:'verrou de la carte',n_verrou:'« libre » vaut « pas de verrou posé sur cette carte » — pas une lecture instantanée : un verrou peut se prendre l\\'instant d\\'après.',
  h_moteurs:'moteurs sur les cartes',h_capteurs:'capteurs et températures',f_temp:'températures',f_fan:'ventilateurs',f_in:'tensions',f_power:'puissances',f_curr:'courants',h_repart:'répartition du travail',
  h_taches:'tâches',h_regl:'explications & réglages',h_moteur:'moteur',
  b_distribuer:'Distribuer',b_vider:'Vider',b_enreg:'Enregistrer',
  c_appareil:'appareil',c_couches:'couches',c_formats:'formats',
  c_attn:'attention',c_mlp:'MLP',c_hote:'en RAM hôte',l_jetons:'jetons',
  l_zen:'fond « zen »',l_cool:'fond « cool »',
  ph_invites:'Une invite par ligne — elles partent toutes ensemble.',
  e_repart:'<b>Répartition</b> — où le travail se fait. Chaque étage reçoit le format que son silicium sait lire : <code>nvfp4</code> sur Blackwell, <code>int4_awq</code> sur Ampere.',
  e_cartes:'<b>Cartes graphiques</b> — toutes celles de la machine. <code>nvidia-smi</code> donne l\\'état des cartes NVIDIA, <code>lspci</code> complète l\\'inventaire : un iGPU apparaît comme « présente, pas de compute ».',
  e_moteurs:'<b>Moteurs sur les cartes</b> — qui d\\'autre occupe la VRAM. Un voisin qui calcule pendant une mesure <i>fabrique</i> le résultat. L\\'arrêt demande deux clics ; les services permanents une confirmation de plus.',
  e_taches:'<b>Tâches</b> — une invite par ligne, toutes envoyées <i>en même temps</i>. Le débit affiché en bas est celui <i>vu du client</i> : il inclut l\\'attente en file.',
  e_kv:'<b>Cache KV</b> — les blocs libres sur le total. Quand la barre descend, les requêtes attendent au lieu d\\'entrer.',
  e_amb:'<b>Ambiances</b> — touches <code>Z</code> et <code>C</code>, <code>G</code> pour la galerie. Les images sont servies, jamais copiées.'},
 en:{h_parc:'pick a model per card',l_carte:'card',l_modele:'compatible model',b_copier:'Copy the command',decodage:'decode',prefill:'prefill',encours:'running / queued',
  kvlibre:'free KV cache',h_cartes:'graphics cards',
  h_verrou:'card lock',n_verrou:'"free" means "no lock set on this card" — not an instantaneous read: a lock can be taken the moment after.',
  h_moteurs:'engines on the cards',h_capteurs:'sensors and temperatures',f_temp:'temperatures',f_fan:'fans',f_in:'voltages',f_power:'power',f_curr:'currents',h_repart:'work placement',
  h_taches:'tasks',h_regl:'explanations & settings',h_moteur:'engine',
  b_distribuer:'Dispatch',b_vider:'Clear',b_enreg:'Save',
  c_appareil:'device',c_couches:'layers',c_formats:'formats',
  c_attn:'attention',c_mlp:'MLP',c_hote:'in host RAM',l_jetons:'tokens',
  l_zen:'“zen” backdrop',l_cool:'“cool” backdrop',
  ph_invites:'One prompt per line — they all go out together.',
  e_repart:'<b>Placement</b> — where the work happens. Each tier gets the format its silicon can read: <code>nvfp4</code> on Blackwell, <code>int4_awq</code> on Ampere.',
  e_cartes:'<b>Graphics cards</b> — every card in the machine. <code>nvidia-smi</code> reports NVIDIA state, <code>lspci</code> completes the inventory: an iGPU shows as “present, no compute”.',
  e_moteurs:'<b>Engines on the cards</b> — who else holds VRAM. A neighbour computing during a measurement <i>manufactures</i> the result. Stopping takes two clicks; permanent services one more.',
  e_taches:'<b>Tasks</b> — one prompt per line, all sent <i>at once</i>. The rate shown below is the <i>client-side</i> one: it includes queue wait.',
  e_kv:'<b>KV cache</b> — free blocks out of the total. When the bar drops, requests wait instead of entering.',
  e_amb:'<b>Backdrops</b> — keys <code>Z</code> and <code>C</code>, <code>G</code> for the gallery. Images are served, never copied.'},
 de:{h_parc:'Modell je Karte wählen',l_carte:'Karte',l_modele:'kompatibles Modell',b_copier:'Befehl kopieren',decodage:'Dekodierung',prefill:'Prefill',encours:'laufend / wartend',
  kvlibre:'freier KV-Cache',h_cartes:'Grafikkarten',
  h_verrou:'Kartensperre',n_verrou:'„frei" bedeutet „keine Sperre auf dieser Karte" — keine Momentaufnahme: eine Sperre kann im nächsten Moment gesetzt werden.',
  h_moteurs:'Engines auf den Karten',h_capteurs:'Sensoren und Temperaturen',f_temp:'Temperaturen',f_fan:'Lüfter',f_in:'Spannungen',f_power:'Leistung',f_curr:'Ströme',h_repart:'Arbeitsverteilung',
  h_taches:'Aufgaben',h_regl:'Erklärungen & Einstellungen',h_moteur:'Engine',
  b_distribuer:'Verteilen',b_vider:'Leeren',b_enreg:'Speichern',
  c_appareil:'Gerät',c_couches:'Schichten',c_formats:'Formate',
  c_attn:'Attention',c_mlp:'MLP',c_hote:'im Host-RAM',l_jetons:'Token',
  l_zen:'„zen“-Hintergrund',l_cool:'„cool“-Hintergrund',
  ph_invites:'Ein Prompt pro Zeile — sie gehen alle gleichzeitig raus.',
  e_repart:'<b>Verteilung</b> — wo gerechnet wird. Jede Stufe bekommt das Format, das ihr Silizium lesen kann: <code>nvfp4</code> auf Blackwell, <code>int4_awq</code> auf Ampere.',
  e_cartes:'<b>Grafikkarten</b> — alle Karten der Maschine. <code>nvidia-smi</code> liefert den Zustand der NVIDIA-Karten, <code>lspci</code> vervollständigt die Liste: eine iGPU erscheint als „vorhanden, kein Compute“.',
  e_moteurs:'<b>Engines auf den Karten</b> — wer sonst VRAM belegt. Ein Nachbar, der während einer Messung rechnet, <i>erzeugt</i> das Ergebnis. Stoppen: zwei Klicks; permanente Dienste eine Bestätigung mehr.',
  e_taches:'<b>Aufgaben</b> — ein Prompt pro Zeile, alle <i>gleichzeitig</i> gesendet. Die unten gezeigte Rate ist die <i>clientseitige</i>: sie enthält die Wartezeit.',
  e_kv:'<b>KV-Cache</b> — freie Blöcke von insgesamt. Sinkt der Balken, warten Anfragen, statt einzutreten.',
  e_amb:'<b>Hintergründe</b> — Tasten <code>Z</code> und <code>C</code>, <code>G</code> für die Galerie. Bilder werden ausgeliefert, nie kopiert.'},
 es:{h_parc:'elegir un modelo por tarjeta',l_carte:'tarjeta',l_modele:'modelo compatible',b_copier:'Copiar el comando',decodage:'decodificación',prefill:'prefill',encours:'en curso / en cola',
  kvlibre:'caché KV libre',h_cartes:'tarjetas gráficas',
  h_verrou:'bloqueo de la tarjeta',n_verrou:'«libre» significa «sin bloqueo en esta tarjeta» — no es una lectura instantánea: un bloqueo puede tomarse al instante siguiente.',
  h_moteurs:'motores en las tarjetas',h_capteurs:'sensores y temperaturas',f_temp:'temperaturas',f_fan:'ventiladores',f_in:'tensiones',f_power:'potencias',f_curr:'corrientes',h_repart:'reparto del trabajo',
  h_taches:'tareas',h_regl:'explicaciones y ajustes',h_moteur:'motor',
  b_distribuer:'Distribuir',b_vider:'Vaciar',b_enreg:'Guardar',
  c_appareil:'dispositivo',c_couches:'capas',c_formats:'formatos',
  c_attn:'atención',c_mlp:'MLP',c_hote:'en RAM del host',l_jetons:'tokens',
  l_zen:'fondo «zen»',l_cool:'fondo «cool»',
  ph_invites:'Un prompt por línea — salen todos a la vez.',
  e_repart:'<b>Reparto</b> — dónde se hace el trabajo. Cada nivel recibe el formato que su silicio sabe leer: <code>nvfp4</code> en Blackwell, <code>int4_awq</code> en Ampere.',
  e_cartes:'<b>Tarjetas gráficas</b> — todas las de la máquina. <code>nvidia-smi</code> da el estado de las NVIDIA, <code>lspci</code> completa el inventario: una iGPU aparece como «presente, sin cómputo».',
  e_moteurs:'<b>Motores en las tarjetas</b> — quién más ocupa la VRAM. Un vecino que calcula durante una medición <i>fabrica</i> el resultado. Detener: dos clics; los servicios permanentes, una confirmación más.',
  e_taches:'<b>Tareas</b> — un prompt por línea, enviados <i>a la vez</i>. La tasa mostrada abajo es la <i>del cliente</i>: incluye la espera en cola.',
  e_kv:'<b>Caché KV</b> — bloques libres sobre el total. Cuando la barra baja, las peticiones esperan en vez de entrar.',
  e_amb:'<b>Ambientes</b> — teclas <code>Z</code> y <code>C</code>, <code>G</code> para la galería. Las imágenes se sirven, nunca se copian.'},
 eo:{h_parc:'elekti modelon laŭ karto',l_carte:'karto',l_modele:'kongrua modelo',b_copier:'Kopii la komandon',decodage:'malkodado',prefill:'antaŭplenigo',encours:'kurantaj / atendantaj',
  kvlibre:'libera KV-kaŝmemoro',h_cartes:'grafikaj kartoj',
  h_verrou:'ŝloso de la karto',n_verrou:'"libera" signifas "neniu ŝloso sur ĉi tiu karto" — ne tuja legado: ŝloso povas esti prenita la sekvan momenton.',
  h_moteurs:'motoroj sur la kartoj',h_capteurs:'sensiloj kaj temperaturoj',f_temp:'temperaturoj',f_fan:'ventoliloj',f_in:'tensioj',f_power:'potencoj',f_curr:'kurentoj',h_repart:'disdivido de la laboro',
  h_taches:'taskoj',h_regl:'klarigoj kaj agordoj',h_moteur:'motoro',
  b_distribuer:'Disdoni',b_vider:'Malplenigi',b_enreg:'Konservi',
  c_appareil:'aparato',c_couches:'tavoloj',c_formats:'formatoj',
  c_attn:'atento',c_mlp:'MLP',c_hote:'en gastiga RAM',l_jetons:'ĵetonoj',
  l_zen:'fono « zen »',l_cool:'fono « cool »',
  ph_invites:'Unu instrukcio po linio — ĉiuj ekiras kune.',
  e_repart:'<b>Disdivido</b> — kie la laboro okazas. Ĉiu etaĝo ricevas la formaton, kiun ĝia silicio scipovas legi: <code>nvfp4</code> sur Blackwell, <code>int4_awq</code> sur Ampere.',
  e_cartes:'<b>Grafikaj kartoj</b> — ĉiuj de la maŝino. <code>nvidia-smi</code> donas la staton de la NVIDIA-kartoj, <code>lspci</code> kompletigas la inventaron: iGPU aperas kiel « ĉeesta, sen komputado ».',
  e_moteurs:'<b>Motoroj sur la kartoj</b> — kiu alia okupas la VRAM. Najbaro kiu kalkulas dum mezurado <i>fabrikas</i> la rezulton. Halti: du klakoj; permanentaj servoj, unu konfirmo plu.',
  e_taches:'<b>Taskoj</b> — unu instrukcio po linio, ĉiuj senditaj <i>samtempe</i>. La malsupre montrata rapido estas tiu <i>de la kliento</i>: ĝi enhavas la atendon en vico.',
  e_kv:'<b>KV-kaŝmemoro</b> — liberaj blokoj el la tuto. Kiam la strio malsupreniras, petoj atendas anstataŭ eniri.',
  e_amb:'<b>Etosoj</b> — klavoj <code>Z</code> kaj <code>C</code>, <code>G</code> por la galerio. La bildoj estas servataj, neniam kopiataj.'},
};
let LANG = 'fr';
const tr = c => (T[LANG] && T[LANG][c]) || T.fr[c] || c;

function traduire(code) {
  LANG = T[code] ? code : 'fr';
  document.documentElement.lang = LANG;
  for (const el of document.querySelectorAll('[data-t]')) el.textContent = tr(el.dataset.t);
  $('invites').placeholder = tr('ph_invites');
  $('expl').innerHTML = ['e_repart','e_cartes','e_moteurs','e_taches','e_kv','e_amb']
    .map(c => '<p>' + tr(c) + '</p>').join('');
  $('langue').value = LANG;
  try { localStorage.setItem('acvram_langue', LANG); } catch (e) {}
  // Les listes deja peintes portent des libelles traduits : on les repeint,
  // sinon la moitie de la page change de langue et l autre non.
  charger_cartes(); charger_verrou(); charger_moteurs(); charger_repartition(); remplir_modeles();
}

$('langue').onchange = e => traduire(e.target.value);

let lang0 = 'fr';
try { lang0 = localStorage.getItem('acvram_langue') || (navigator.language || 'fr').slice(0, 2); } catch (e) {}
traduire(lang0);

charger_reglages();
charger_parc();
// -- capteurs --------------------------------------------------------------
// Tout ce que la machine expose : hwmon (temperatures, ventilateurs, tensions,
// puissances), les cartes NVIDIA avec leurs RAISONS DE BRIDAGE, et le systeme.
// La jauge se colore par rapport au seuil que la puce donne (max, crit) ; sans
// seuil, elle reste neutre — on n invente pas un seuil.
function classe_seuil(l) {
  if (l.famille === 'temp') {
    const ref = l.crit || l.max;
    if (ref) { if (l.valeur >= ref) return 'chaud'; if (l.valeur >= ref * 0.8) return 'tiede'; }
    else { if (l.valeur >= 85) return 'chaud'; if (l.valeur >= 70) return 'tiede'; }
  }
  if (l.alarme) return 'chaud';
  return '';
}
function largeur_jauge(l) {
  if (l.famille === 'temp') { const ref = l.crit || l.max || 100; return Math.min(100, 100 * l.valeur / ref); }
  if (l.famille === 'fan')  return Math.min(100, 100 * l.valeur / 2000);
  if (l.famille === 'power') return Math.min(100, 100 * l.valeur / (l.max || 1000));
  return 0;
}
function familles_actives() {
  return ['temp','fan','in','power','curr'].filter(f => $('f_' + f).checked);
}
function bloc_gpu(c) {
  const el = document.createElement('div'); el.className = 'puce';
  const t = c.temp_gpu || 0, cl = t >= 85 ? 'chaud' : (t >= 70 ? 'tiede' : '');
  let h = '<h3>GPU ' + c.index + ' · ' + c.nom + '</h3>';
  h += '<div class="lect ' + cl + '"><span class="e">température GPU</span><span class="v">' + nb(c.temp_gpu,0) + ' °C</span>'
     + '<span class="j"><i style="width:' + Math.min(100, t) + '%"></i></span></div>';
  if (c.temp_mem != null) h += '<div class="lect"><span class="e">température mémoire</span><span class="v">' + nb(c.temp_mem,0) + ' °C</span></div>';
  h += '<div class="lect"><span class="e">ventilateur</span><span class="v">' + nb(c.ventilateur_pct,0) + ' %</span></div>';
  h += '<div class="lect"><span class="e">puissance</span><span class="v">' + nb(c.watts,0) + ' / ' + nb(c.watts_max,0) + ' W</span>'
     + '<span class="j"><i style="width:' + (c.watts_max ? 100*c.watts/c.watts_max : 0).toFixed(0) + '%"></i></span></div>';
  h += '<div class="lect"><span class="e">horloge SM</span><span class="v">' + nb(c.horloge_sm,0) + ' / ' + nb(c.horloge_sm_max,0) + ' MHz</span></div>';
  h += '<div class="lect"><span class="e">horloge mémoire</span><span class="v">' + nb(c.horloge_mem,0) + ' MHz</span></div>';
  h += '<div class="lect"><span class="e">occupation calcul / mémoire</span><span class="v">' + nb(c.occupation,0) + ' % / ' + nb(c.occupation_mem,0) + ' %</span></div>';
  h += '<div class="lect"><span class="e">VRAM</span><span class="v">' + nb(c.mio_pris/1024,1) + ' / ' + nb(c.mio_total/1024,1) + ' Gio</span></div>';
  h += '<div class="lect"><span class="e">PCIe</span><span class="v">gen ' + nb(c.pcie_gen,0) + ' x' + nb(c.pcie_largeur,0) + '</span></div>';
  if (c.bridages && c.bridages.length)
    h += '<div>' + c.bridages.map(b => '<span class="bridage">' + b + '</span>').join('') + '</div>';
  el.innerHTML = h; return el;
}
async function charger_capteurs() {
  try {
    const r = await (await fetch('/capteurs')).json();
    const box = $('capteurs'); box.innerHTML = '';
    for (const c of r.nvidia || []) box.appendChild(bloc_gpu(c));
    const fam = familles_actives();
    let n = 0;
    for (const p of r.hwmon || []) {
      const lect = p.lectures.filter(l => fam.includes(l.famille));
      if (!lect.length) continue;
      const el = document.createElement('div'); el.className = 'puce';
      let h = '<h3>' + p.nom + '</h3>';
      for (const l of lect) {
        n++;
        const cl = classe_seuil(l);
        const seuil = l.crit ? ' · crit ' + nb(l.crit,0) : (l.max ? ' · max ' + nb(l.max,0) : '');
        h += '<div class="lect ' + cl + '"><span class="e" title="' + l.id + '">' + l.etiquette + '</span>'
           + '<span class="v">' + nb(l.valeur, l.famille === 'in' ? 3 : (l.famille === 'fan' ? 0 : 1)) + ' ' + l.unite + '</span>';
        if (l.famille !== 'in' && l.famille !== 'curr')
          h += '<span class="j" title="' + seuil.replace(' · ','') + '"><i style="width:' + largeur_jauge(l).toFixed(0) + '%"></i></span>';
        h += '</div>';
      }
      el.innerHTML = h; box.appendChild(el);
    }
    const s = r.systeme || {};
    if (s.disponible) {
      const j = Math.floor(s.depuis / 86400), hh = Math.floor((s.depuis % 86400) / 3600);
      $('systeme').innerHTML =
        '<span>CPU <b>' + nb(s.cpu_pct,0) + ' %</b> · ' + nb(s.cpu_mhz,0) + ' MHz</span>'
      + '<span>charge <b>' + s.charge_1_5_15.join(' / ') + '</b></span>'
      + '<span>RAM <b>' + nb(s.ram_gio_prise,1) + '</b> / ' + nb(s.ram_gio_total,1) + ' Gio</span>'
      + '<span>swap <b>' + nb(s.swap_gio_pris,1) + '</b> / ' + nb(s.swap_gio_total,1) + ' Gio</span>'
      + '<span>processus <b>' + s.processus + '</b></span>'
      + '<span>allumé depuis <b>' + j + ' j ' + hh + ' h</b></span>'
      + (s.disques || []).filter(d => d.gio_total > 50).map(d =>
          '<span title="' + d.dev + '">' + d.montage + ' <b>' + nb(d.pct,0) + ' %</b> de ' + nb(d.gio_total/1024,1) + ' Tio</span>').join('');
    }
    $('capteurs_note').textContent = (r.hwmon || []).length + ' puce(s) hwmon, ' + n + ' lecture(s) affichée(s), '
      + (r.nvidia || []).length + ' GPU NVIDIA — /sys/class/hwmon, nvidia-smi, psutil';
  } catch (e) { $('capteurs_note').textContent = 'capteurs indisponibles : ' + e.message; }
}
for (const f of ['temp','fan','in','power','curr']) $('f_' + f).addEventListener('change', charger_capteurs);
charger_capteurs();
setInterval(charger_capteurs, 2000);

charger_cartes();
charger_verrou();
charger_moteurs();
setInterval(charger_cartes, 2000);
// « affiche < 5 s » (poste7-gui-ajouts-18-09) : un verrou pris par un temoin
// doit apparaitre vite, plus vite que le rythme des moteurs (3 s).
setInterval(charger_verrou, 2000);
setInterval(() => { if (a_confirmer === null) charger_moteurs(); }, 3000);
$('regime').addEventListener('click', () => {
  navigator.clipboard?.writeText($('regime').textContent).catch(() => {});
});
init_ambiances();
charger_modele();
charger_repartition();
rafraichir();
setInterval(rafraichir, 1500);
</script></body></html>
"""


# La galerie est une PAGE A PART, pas une fenetre modale de la console : elle
# s'ouvre dans une autre fenetre du navigateur, donc elle doit porter son
# propre style. Un modal aurait recouvert les chiffres qu'on regarde, et
# c'est precisement ce qu'on ne veut pas d'un agrement.
GALERIE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>acvram — galerie</title>
<style>
 :root{--fond:#120a10;--bord:#3a1f31;--rose-clair:#f9a8d4;--rose-vif:#ec4899;
       --doux:#c9a3b8;--faible:#8d6b7e;--texte:#f7e4ef}
 *{box-sizing:border-box}
 body{margin:0;min-height:100vh;color:var(--texte);
      font:14px/1.55 system-ui,sans-serif;
      background:radial-gradient(1100px 600px at 20% -10%,#2a0f21,transparent 60%),
                 radial-gradient(900px 500px at 100% 10%,#3a1030,transparent 55%),
                 var(--fond)}
 header{padding:14px 20px;border-bottom:1px solid var(--bord);display:flex;
        gap:14px;align-items:baseline;flex-wrap:wrap}
 h1{margin:0;font-size:15px;letter-spacing:.2em;text-transform:uppercase;
    background:linear-gradient(90deg,var(--rose-clair),#db2777);
    -webkit-background-clip:text;background-clip:text;color:transparent}
 .v{color:var(--faible);font-size:12px}
 main{padding:18px;display:grid;gap:14px;
      grid-template-columns:repeat(auto-fill,minmax(230px,1fr));max-width:1800px;
      margin:0 auto}
 main.duo{grid-template-columns:repeat(auto-fit,minmax(420px,1fr));max-width:1400px}
 h2.sec{grid-column:1/-1;margin:14px 0 0;font-size:12px;color:var(--rose-clair);
        text-transform:uppercase;letter-spacing:.12em}
 figure{margin:0;border:1px solid var(--bord);border-radius:14px;overflow:hidden;
        background:#1a0d15}
 figure img{display:block;width:100%;height:auto;cursor:zoom-in;
      aspect-ratio:1/1;object-fit:cover;background:#1a0d15}
 figure.grande img{aspect-ratio:auto;object-fit:contain}
 figcaption{padding:9px 14px;font-size:12px;color:var(--doux);
            border-top:1px solid var(--bord);display:flex;gap:10px;
            align-items:baseline}
 figcaption b{color:var(--rose-clair);text-transform:uppercase;
              letter-spacing:.1em;font-size:11px}
 figcaption span{margin-left:auto;color:var(--faible);font-size:11px}
 .vide{padding:26px;border:1px dashed var(--bord);border-radius:14px;
       color:var(--faible);grid-column:1/-1}
 .plein{position:fixed;inset:0;background:rgba(10,5,9,.96);display:none;
        align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
 .plein img{max-width:96vw;max-height:94vh;border-radius:10px;
            box-shadow:0 0 60px rgba(236,72,153,.28)}
 .plein.on{display:flex}
 /* UNE SORTIE VISIBLE, PAS SEULEMENT UNE TOUCHE. Le plein ecran se fermait
    par Echap ou par un clic sur le fond — deux gestes qu il faut CONNAITRE.
    Qui ne les connait pas se retrouve enferme dans une image sans rien a
    cliquer : un ecran sans porte visible est un ecran sans porte. */
 .fermer{position:fixed;top:16px;right:18px;z-index:10;cursor:pointer;
    width:42px;height:42px;border-radius:50%;font-size:18px;line-height:1;
    background:rgba(32,17,27,.9);border:1px solid var(--rose-vif);
    color:var(--rose-clair)}
 .fermer:hover{background:var(--rose-vif);color:#150a11}
 .retour{color:var(--rose-clair);text-decoration:none;font-size:12px;
    border:1px solid var(--bord);border-radius:99px;padding:4px 12px;
    background:rgba(236,72,153,.12)}
 .retour:hover{border-color:var(--rose-vif);color:var(--rose-pale)}
 .pied{display:flex;gap:9px;flex-wrap:wrap;justify-content:center;padding:16px;
       font-size:12px;color:var(--faible);border-top:1px solid var(--bord)}
 .pied b{letter-spacing:.22em;font-weight:700;font-size:13px;
    background:linear-gradient(90deg,var(--rose-clair),var(--rose-vif));
    -webkit-background-clip:text;background-clip:text;color:transparent}
 .pied a{color:var(--rose-vif);text-decoration:none}
 .pied a:hover{color:var(--rose-clair)}
</style></head><body>
<header><a class="retour" href="/">← console</a>
  <h1>acvram — galerie</h1>
  <span class="v" id="compte"></span>
  <span class="v">clic pour agrandir · <b>Échap</b> ferme · <b>←</b> <b>→</b> défilent</span>
</header>
<main id="g"></main>
<footer class="pied"><span><b>ANTICITOYEN</b></span><span>·</span>
  <span>GNU <a href="https://www.gnu.org/licenses/gpl-3.0.html"
        target="_blank" rel="noopener">GPL-3.0-or-later</a></span><span>·</span>
  <a href="https://outils.nuages.noho.st/gitlab/anticitoyen/anticitoyen-vram"
     target="_blank" rel="noopener">GitLab</a></footer>
<div class="plein" id="plein">
  <button class="fermer" id="fermer" title="Fermer — Échap">✕</button>
  <img id="pleinimg" alt=""></div>
<script>
const NOMS = {zen: 'zen', cool: 'cool'};
const g = document.getElementById('g');
const plein = document.getElementById('plein'), pimg = document.getElementById('pleinimg');
let sources = [], courant = -1;

function ajouter(src, titre, sous, grande) {
  const f = document.createElement('figure');
  if (grande) f.className = 'grande';
  // LE CHARGEMENT EST PARESSEUX. 633 images de 585 Mio chargees d un coup
  // saturent la memoire du navigateur et le rendent inutilisable — la
  // galerie serait « ouverte » et illisible. loading=lazy laisse le
  // navigateur ne chercher que ce qui approche de l ecran.
  f.innerHTML = '<img loading="lazy" decoding="async" src="' + src
    + '" alt="' + titre + '" data-i="' + sources.length + '">'
    + '<figcaption><b>' + titre + '</b><span>' + (sous || '') + '</span></figcaption>';
  g.appendChild(f);
  sources.push(src);
}

function montrer(i) {
  if (i < 0 || i >= sources.length) return;
  courant = i; pimg.src = sources[i]; plein.classList.add('on');
}

(async () => {
  let dispo = {}, photos = {photos: [], n: 0, dossier: ''};
  try { dispo = await (await fetch('/fonds')).json(); } catch (e) {}
  try { photos = await (await fetch('/photos')).json(); } catch (e) {}

  const titres = document.createElement('h2');
  titres.className = 'sec'; titres.textContent = 'ambiances';
  let ambiances = 0;
  for (const [cle, titre] of Object.entries(NOMS)) {
    if (!dispo[cle]) continue;
    if (!ambiances++) g.appendChild(titres);
    // L horodatage evite que le navigateur reserve l ancienne image apres un
    // changement de chemin dans les reglages : meme URL, autre fichier.
    ajouter('/fond/' + cle + '?t=' + Date.now(), cle,
            'touche ' + (cle === 'zen' ? 'Z' : 'C') + ' dans la console', true);
  }
  if (ambiances) g.classList.add('duo');

  if (photos.n) {
    const h = document.createElement('h2');
    h.className = 'sec';
    h.textContent = 'photos — ' + photos.n + ' dans ' + photos.dossier;
    g.appendChild(h);
    g.classList.remove('duo');
    photos.photos.forEach((p, i) => {
      const kio = Math.round(p.octets / 1024);
      ajouter('/photo/' + i, p.nom.replace(/[.][^.]+$/, ''),
              kio > 1024 ? (kio / 1024).toFixed(1) + ' Mio' : kio + ' Kio', false);
    });
  }
  document.getElementById('compte').textContent =
    sources.length + ' image(s)' + (photos.n ? ' · ' + photos.n + ' photo(s)' : '');

  if (!sources.length) {
    g.innerHTML = '<div class="vide">Aucune image. Réglez les fonds dans '
      + '« explications &amp; réglages » de la console, ou posez '
      + '<code>ACVRAM_GALERIE_DIR</code> sur un dossier de photos.</div>';
  }
})();

g.addEventListener('click', ev => {
  if (ev.target.tagName !== 'IMG') return;
  montrer(Number(ev.target.dataset.i));
});
// Le clic sur le fond ferme, mais PAS le clic sur l image elle-meme : sinon
// on ne peut pas la regarder sans la faire disparaitre par megarde.
plein.onclick = ev => { if (ev.target !== pimg) plein.classList.remove('on'); };
document.getElementById('fermer').onclick = () => plein.classList.remove('on');
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') plein.classList.remove('on');
  else if (plein.classList.contains('on')) {
    // Les fleches ne servent QUE quand le plein ecran est ouvert : sinon
    // elles voleraient le defilement de la grille.
    if (ev.key === 'ArrowRight') { montrer(courant + 1); ev.preventDefault(); }
    if (ev.key === 'ArrowLeft')  { montrer(courant - 1); ev.preventDefault(); }
  }
});
</script></body></html>
"""
