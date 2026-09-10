"""Console web du serveur acvram, servie à la racine.

POURQUOI ELLE EXISTE. Le paquet s'installait sans rien de visible : ni entrée
de menu, ni icône, ni page. `acvram serve` démarrait un serveur qui ne
répondait qu'à des clients OpenAI — donc parfaitement muet pour qui venait
d'installer le `.deb` et voulait vérifier que quelque chose tournait.

CE QU'ELLE N'EST PAS. Ce n'est pas une interface de conversation destinée à
remplacer un vrai client : elle sert à VOIR l'état du moteur et à s'assurer
en trois secondes qu'il répond. Le vrai usage reste un client OpenAI ou
Anthropic branché sur le port — voir `docs/BRANCHER-UN-CLIENT.md`.

AUCUNE DEPENDANCE. Pas de fichier statique, pas de gabarit, pas de bibliothèque
distante : une seule chaîne, servie telle quelle. Un paquet qui télécharge sa
propre interface au premier lancement est un paquet qui ne marche pas hors
ligne, et la machine de mesure n'a pas toujours de réseau.
"""

from __future__ import annotations

# La page se rafraîchit toute seule ; les nombres qu'elle montre sont ceux de
# /metrics, c'est-à-dire exactement ceux que le moteur publie — aucun calcul
# n'est refait ici. Une console qui recalcule finit par diverger de ce qu'elle
# est censée montrer.
PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>acvram — console</title>
<style>
 :root{--f:#0f1115;--c:#161a22;--b:#232936;--t:#e6e9ef;--g:#8b94a7;--a:#5eead4;--r:#f87171}
 *{box-sizing:border-box}
 body{margin:0;background:var(--f);color:var(--t);font:14px/1.5 system-ui,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--b);display:flex;
        align-items:baseline;gap:14px;flex-wrap:wrap}
 h1{font-size:15px;margin:0;letter-spacing:.14em;text-transform:uppercase}
 .v{color:var(--g);font-size:12px}
 .pastille{margin-left:auto;font-size:12px;padding:3px 10px;border-radius:99px;
        border:1px solid var(--b)}
 .ok{color:var(--a);border-color:var(--a)} .ko{color:var(--r);border-color:var(--r)}
 main{padding:20px;display:grid;gap:16px;max-width:1100px}
 .grille{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
 .carte{background:var(--c);border:1px solid var(--b);border-radius:10px;padding:12px 14px}
 .k{color:var(--g);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
 .n{font-size:20px;margin-top:3px;font-variant-numeric:tabular-nums}
 .n small{font-size:12px;color:var(--g);font-weight:400}
 section{background:var(--c);border:1px solid var(--b);border-radius:10px;padding:14px}
 h2{font-size:12px;margin:0 0 10px;color:var(--g);text-transform:uppercase;
    letter-spacing:.08em}
 textarea,input,button{font:inherit;background:var(--f);color:var(--t);
    border:1px solid var(--b);border-radius:8px;padding:9px 11px}
 textarea{width:100%;min-height:74px;resize:vertical}
 .ligne{display:flex;gap:10px;margin-top:10px;align-items:center;flex-wrap:wrap}
 button{cursor:pointer;border-color:var(--a);color:var(--a);background:transparent}
 button:disabled{opacity:.45;cursor:default}
 pre{white-space:pre-wrap;word-break:break-word;background:var(--f);
     border:1px solid var(--b);border-radius:8px;padding:11px;margin:10px 0 0;
     min-height:60px}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 td{padding:3px 0;border-bottom:1px solid var(--b)}
 td:last-child{text-align:right;color:var(--g)}
 .note{color:var(--g);font-size:12px;margin-top:8px}
 code{background:var(--f);padding:1px 5px;border-radius:4px;border:1px solid var(--b)}
</style></head><body>
<header>
  <h1>acvram</h1><span class="v" id="version"></span>
  <span class="v" id="modele"></span>
  <span class="pastille" id="etat">…</span>
</header>
<main>
  <div class="grille">
    <div class="carte"><div class="k">décodage</div>
      <div class="n" id="d_tok">—<small> j/s</small></div></div>
    <div class="carte"><div class="k">prefill</div>
      <div class="n" id="p_tok">—<small> j/s</small></div></div>
    <div class="carte"><div class="k">en cours / en file</div>
      <div class="n" id="rw">—</div></div>
    <div class="carte"><div class="k">blocs KV libres</div>
      <div class="n" id="kv">—</div></div>
  </div>

  <section>
    <h2>essai</h2>
    <textarea id="invite" placeholder="Écrivez une invite, puis Envoyer…">Explique en une phrase ce qu'est la quantification NVFP4.</textarea>
    <div class="ligne">
      <button id="envoyer">Envoyer</button>
      <label class="v">jetons <input id="max" type="number" value="128"
             min="1" max="4096" style="width:88px"></label>
      <span class="v" id="chrono"></span>
    </div>
    <pre id="sortie"></pre>
    <div class="note">Cet essai passe par <code>/v1/chat/completions</code>,
      la même route que n'importe quel client OpenAI.</div>
  </section>

  <section>
    <h2>moteur</h2>
    <table id="details"></table>
    <div class="note">Pour brancher un client :
      <code>OPENAI_BASE_URL</code> ou <code>ANTHROPIC_BASE_URL</code> sur
      l'adresse de cette page. Voir <code>docs/BRANCHER-UN-CLIENT.md</code>.</div>
  </section>
</main>
<script>
const $ = i => document.getElementById(i);
const nb = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toLocaleString('fr-FR', {maximumFractionDigits: d});

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
    if (m.version) $('version').textContent = 'v' + m.version;

    // Le tableau montre ce que /metrics rend, sans trier ni interpreter :
    // une console qui choisit ce qu'elle affiche cache ce qu'elle omet.
    const t = $('details');
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

async function charger_modele() {
  try {
    const d = await (await fetch('/v1/models')).json();
    modele = d.data?.[0]?.id ?? null;
    if (modele) $('modele').textContent = modele;
  } catch (err) { /* la pastille dira deja que le serveur ne repond pas */ }
}

$('envoyer').onclick = async () => {
  const b = $('envoyer'), out = $('sortie');
  b.disabled = true; out.textContent = ''; $('chrono').textContent = '…';
  const t0 = performance.now();
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: modele, max_tokens: Number($('max').value) || 128,
        messages: [{role: 'user', content: $('invite').value}],
      }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message || ('HTTP ' + r.status));
    out.textContent = j.choices?.[0]?.message?.content ?? JSON.stringify(j, null, 2);
    const u = j.usage || {};
    const s = (performance.now() - t0) / 1000;
    // Le debit affiche ici est celui du CLIENT : il inclut le reseau et
    // l attente en file, donc il est toujours inferieur a celui du moteur.
    // Les deux sont montres pour que l ecart soit visible plutot que suppose.
    $('chrono').textContent = s.toFixed(2) + ' s · '
      + (u.completion_tokens ?? '?') + ' jetons · '
      + (u.completion_tokens ? nb(u.completion_tokens / s) + ' j/s vus du client' : '');
  } catch (err) {
    out.textContent = 'échec : ' + err.message;
    $('chrono').textContent = '';
  } finally { b.disabled = false; }
};

charger_modele();
rafraichir();
setInterval(rafraichir, 2000);
</script></body></html>
"""
