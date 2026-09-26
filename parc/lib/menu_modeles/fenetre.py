"""Commun claude-modeles / kimi-modeles — la fenêtre GTK4.

La classe Fenetre et l'habillage sont identiques aux noms près entre les deux
menus ; les ~116 lignes de divergence (nom affiché, icône, couleurs d'accent,
bouton web, « Ouvrir dans … »/« Précharger », refus tabby/yals) sont pilotées par
un PROFIL (dict) passé au constructeur. Chaque lanceur ne construit que ce profil
puis lance l'Application.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk, Pango  # noqa: E402

from .config import (PARC, MAISON, KIMI_DIR, TSV_DIR, CONFIG, NOTES, GGUF_TSV,  # noqa: E402,F401
                     VLLM_TSV, ACVRAM_TSV, VISION_TSV, SECRETS, BIN,
                     DOSSIERS_LANCEMENT, DOSSIER_LANCEMENT_DEFAUT, ANSI, GUI_TEST,
                     MOTEURS, ORDRE_MOTEUR, FICHE_ABSENTE, NOTE_MOTS, REFUS_RANG)
from .moteur import (Moteur, secrets, cle_moteur, modele_servi, vram, pid_du_port)  # noqa: E402,F401
from .parc import (Modele, lire_tsv, charger_parc, rang_qualite, rang_refus, ecrire_note)  # noqa: E402,F401


# ─── fenêtre ──────────────────────────────────────────────────────────────────
class Fenetre(Adw.ApplicationWindow):
    def __init__(self, app, profil):
        self.profil = profil
        super().__init__(application=app, title=self.profil["titre_fenetre"],
                         default_width=1340, default_height=800)
        self.parc = []
        self.dossier_lancement = DOSSIER_LANCEMENT_DEFAUT   # cwd de « Ouvrir dans Claude »
        self.proc = None                 # lanceur en cours
        self.proc_tail = None            # suivi de journal associé
        self.filtre_moteur = None
        self.filtre_texte = ""
        self.filtre_sans_censure = False
        self.filtre_outils = False
        self.filtre_vedettes = False          # ≈Opus / ≈Fable (poste7-menus-vedettes-19-09 § 4)
        self.filtre_code_os = False           # code android/linux
        self.tri_multi = []              # [(idx colonne, descendant)], ordre = priorité
        self.etat = {}                   # provider → identifiant servi
        self._dossiers_deja_proposes = False  # une seule proposition auto par session (pièce 84)
        self._construire()
        self._recharger()
        self.sonder()
        GLib.timeout_add_seconds(6, self._sonde_periodique)

    # ---- construction -------------------------------------------------------
    def _construire(self):
        self.toasts = Adw.ToastOverlay()
        self.set_content(self.toasts)
        vue = Adw.ToolbarView()
        self.toasts.set_child(vue)

        entete = Adw.HeaderBar()
        vue.add_top_bar(entete)

        self.recherche = Gtk.SearchEntry(placeholder_text="Filtrer : code, créatif, nsfw, vision…",
                                         width_chars=32)
        self.recherche.connect("search-changed", self._sur_recherche)
        self.recherche.connect("stop-search", lambda e: e.set_text(""))
        entete.set_title_widget(self.recherche)

        b_maj = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Recharger (Ctrl+R)")
        b_maj.connect("clicked", lambda *_: (self._recharger(), self.sonder()))
        entete.pack_start(b_maj)

        self.b_dossiers_modeles = Gtk.Button(label="Dossiers des modèles…",
                                             tooltip_text="Choisir où sont les modèles (acvram, GGUF, HF) ; "
                                                          "balayés automatiquement pour remplir cette liste")
        self.b_dossiers_modeles.connect("clicked", lambda *_: self.choisir_dossiers_modeles())
        entete.pack_start(self.b_dossiers_modeles)
        self.b_rebalayer = Gtk.Button(icon_name="folder-symbolic",
                                      tooltip_text="Rebalayer les dossiers de modèles déjà choisis")
        self.b_rebalayer.connect("clicked", lambda *_: self.rebalayer())
        entete.pack_start(self.b_rebalayer)

        menu = Gio.Menu()
        menu.append("Journal du moteur actif", "win.journal")
        menu.append("Arrêter le moteur actif", "win.arreter")
        menu.append("Libérer la VRAM", "win.liberer")
        menu.append("Copier la commande", "win.copier")
        menu.append("Soutenir : buymeacoffee.com/anticitoyen", "win.soutenir")
        b_menu = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        entete.pack_end(b_menu)

        self.banniere = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12,
                                margin_start=12, margin_end=12, margin_top=8, margin_bottom=8)
        self.lbl_etat = Gtk.Label(xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE)
        self.lbl_vram = Gtk.Label(xalign=1)
        self.lbl_vram.add_css_class("dim-label")
        self.barre_vram = Gtk.LevelBar(min_value=0, max_value=1, width_request=140,
                                       valign=Gtk.Align.CENTER)
        self.banniere.append(self.lbl_etat)
        self.banniere.append(self.barre_vram)
        self.banniere.append(self.lbl_vram)
        vue.add_top_bar(self.banniere)

        # filtres par moteur
        barre = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
                        margin_start=12, margin_end=12, margin_bottom=6)
        self.boutons_moteur = {}
        premier = None
        for cle, libelle in [(None, "Tous"), ("acvram", "acvram"), ("vllm", "vLLM"),
                             ("llamacpp", "llama.cpp"), ("tabby", "TabbyAPI"),
                             ("yals", "YALS"), ("rapide", "Appoint")]:
            b = Gtk.ToggleButton(label=libelle)
            if premier is None:
                premier = b
                b.set_active(True)
            else:
                b.set_group(premier)
            b.connect("toggled", self._sur_moteur, cle)
            self.boutons_moteur[cle] = b
            barre.append(b)
        barre.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        self.b_censure = Gtk.ToggleButton(label="Sans censure",
                                          tooltip_text="Refus mesurés nuls ou rares")
        self.b_censure.connect("toggled", self._sur_drapeaux)
        self.b_outils = Gtk.ToggleButton(label="Outils OK",
                                         tooltip_text="Appels d'outils validés (agent-ok)")
        self.b_outils.connect("toggled", self._sur_drapeaux)
        barre.append(self.b_censure)
        barre.append(self.b_outils)
        # puces vedettes (poste7-menus-vedettes-19-09 § 4) : premier mot-clé d'usage de la fiche
        self.b_vedettes = Gtk.ToggleButton(label="≈ Opus/Fable",
                                           tooltip_text="Lignées ≈Opus / ≈Fable (fiche d'usage commençant par ≈)")
        self.b_vedettes.connect("toggled", self._sur_drapeaux)
        self.b_code_os = Gtk.ToggleButton(label="code Android/Linux",
                                          tooltip_text="Fiche d'usage commençant par « code android/linux »")
        self.b_code_os.connect("toggled", self._sur_drapeaux)
        barre.append(self.b_vedettes)
        barre.append(self.b_code_os)
        barre.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        self.b_tri = Gtk.MenuButton(
            label="Tri…",
            tooltip_text="Trier sur plusieurs colonnes à la fois — cocher dans l'ordre "
                         "voulu, la première case cochée est le critère principal")
        self.pop_tri = Gtk.Popover()
        self.b_tri.set_popover(self.pop_tri)
        self.pop_tri.connect("show", lambda *_: self._construire_popover_tri())
        barre.append(self.b_tri)
        self.lbl_compte = Gtk.Label(xalign=1, hexpand=True)
        self.lbl_compte.add_css_class("dim-label")
        barre.append(self.lbl_compte)
        vue.add_top_bar(barre)

        # liste + détail
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, position=880,
                               shrink_start_child=False, shrink_end_child=False,
                               resize_end_child=False)
        vue.set_content(self.paned)
        self.paned.set_start_child(self._construire_liste())
        self.paned.set_end_child(self._construire_detail())

        # console
        self.console = Gtk.TextView(editable=False, monospace=True, cursor_visible=False,
                                    wrap_mode=Gtk.WrapMode.WORD_CHAR,
                                    left_margin=8, right_margin=8, top_margin=6)
        defiler = Gtk.ScrolledWindow(child=self.console, min_content_height=170,
                                     vexpand=False)
        tampon_console = self.console.get_buffer()
        self.fin_console = tampon_console.create_mark("fin", tampon_console.get_end_iter(), False)
        self.reveleur = Gtk.Revealer(child=defiler, transition_type=Gtk.RevealerTransitionType.SLIDE_UP)
        vue.add_bottom_bar(self.reveleur)

        # actions du bas
        bas = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, margin_start=12,
                      margin_end=12, margin_top=8, margin_bottom=10)
        dossier_menu = Gio.Menu()
        for d in DOSSIERS_LANCEMENT:
            dossier_menu.append(str(d), f"win.dossier('{d}')")
        dossier_menu.append("Choisir…", "win.dossier_autre")
        self.b_dossier = Gtk.MenuButton(label=f"Dossier : {self.dossier_lancement.name}",
                                        menu_model=dossier_menu)
        self.b_dossier.set_tooltip_text(f"Répertoire de lancement de « Ouvrir dans {self.profil['nom']} »")
        self.b_ouvrir = Gtk.Button(label=f"Ouvrir dans {self.profil['nom']}")
        self.b_ouvrir.add_css_class("suggested-action")
        self.b_ouvrir.connect("clicked", lambda *_: self.ouvrir_kimi())
        self.b_web = Gtk.Button(label=self.profil["web_label"])
        self.b_web.set_tooltip_text(self.profil["web_tooltip"])
        self.b_web.connect("clicked", lambda *_: self.ouvrir_web())
        self.b_openwebui = Gtk.Button(label="Ouvrir Open WebUI")
        self.b_openwebui.set_tooltip_text("Démarre Open WebUI (port 3000) si besoin et ouvre le navigateur ; "
                                         "mêmes moteurs locaux (acvram 8090, routeur 8790), images par ComfyUI")
        self.b_openwebui.connect("clicked", lambda *_: self.ouvrir_openwebui())
        self.b_precharger = Gtk.Button(label="Précharger")
        self.b_precharger.set_tooltip_text(f"Démarrer le serveur sans ouvrir {self.profil['nom']}")
        self.b_precharger.connect("clicked", lambda *_: self.precharger())
        self.b_stop = Gtk.Button(label="Interrompre")
        self.b_stop.add_css_class("destructive-action")
        self.b_stop.set_sensitive(False)
        self.b_stop.connect("clicked", lambda *_: self.interrompre())
        bancs = Gio.Menu()
        bancs.append("Banc d'outils (12 épreuves)", "win.banc_outils")
        bancs.append("Banc de refus (5 fictions)", "win.banc_refus")
        b_bancs = Gtk.MenuButton(label="Mesurer", menu_model=bancs)
        self.b_console = Gtk.ToggleButton(label="Console")
        self.b_console.connect("toggled", lambda b: self.reveleur.set_reveal_child(b.get_active()))
        bas.append(self.b_dossier)
        bas.append(self.b_ouvrir)
        bas.append(self.b_web)
        bas.append(self.b_openwebui)
        bas.append(self.b_precharger)
        bas.append(b_bancs)
        bas.append(self.b_stop)
        bas.append(Gtk.Box(hexpand=True))
        bas.append(self.b_console)
        vue.add_bottom_bar(bas)

        for nom, fn in [("journal", self.voir_journal), ("arreter", self.arreter_moteur),
                        ("liberer", self.liberer_vram), ("copier", self.copier_commande),
                        ("soutenir", lambda: Gtk.show_uri(self, "https://buymeacoffee.com/anticitoyen", Gdk.CURRENT_TIME)),
                        ("banc_outils", lambda: self.banc("outils")),
                        ("banc_refus", lambda: self.banc("refus"))]:
            a = Gio.SimpleAction.new(nom, None)
            a.connect("activate", lambda *_a, f=fn: f())
            self.add_action(a)

        a_dossier = Gio.SimpleAction.new("dossier", GLib.VariantType.new("s"))
        a_dossier.connect("activate", lambda a, v: self._choisir_dossier_lancement(v.get_string()))
        self.add_action(a_dossier)
        a_dossier_autre = Gio.SimpleAction.new("dossier_autre", None)
        a_dossier_autre.connect("activate", lambda *_a: self._choisir_dossier_lancement_dialogue())
        self.add_action(a_dossier_autre)

        raccourcis = Gtk.ShortcutController()
        for touche, action in [("<Control>f", lambda *_: self.recherche.grab_focus()),
                               ("<Control>r", lambda *_: (self._recharger(), self.sonder())),
                               ("<Control>l", lambda *_: self.b_console.set_active(
                                   not self.b_console.get_active())),
                               ("Escape", lambda *_: self.recherche.set_text(""))]:
            raccourcis.add_shortcut(Gtk.Shortcut(
                trigger=Gtk.ShortcutTrigger.parse_string(touche),
                action=Gtk.CallbackAction.new(lambda *_a, f=action: (f(), True)[1])))
        self.add_controller(raccourcis)

    def _colonne(self, titre, rendu, tri=None, fixe=None, expand=False):
        fabrique = Gtk.SignalListItemFactory()

        def setup(_f, item):
            lbl = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
            item.set_child(lbl)

        def bind(_f, item):
            rendu(item.get_child(), item.get_item())

        fabrique.connect("setup", setup)
        fabrique.connect("bind", bind)
        col = Gtk.ColumnViewColumn(title=titre, factory=fabrique, expand=expand,
                                   resizable=True)
        if fixe:
            col.set_fixed_width(fixe)
        if tri:
            col.set_sorter(Gtk.CustomSorter.new(
                lambda a, b, _u=None, f=tri: (f(a) > f(b)) - (f(a) < f(b))))
        return col

    def _construire_liste(self):
        self.store = Gio.ListStore(item_type=Modele)
        self.filtre = Gtk.CustomFilter.new(self._passe_filtre)
        filtree = Gtk.FilterListModel(model=self.store, filter=self.filtre)
        self.vue_liste = Gtk.ColumnView(show_row_separators=True, single_click_activate=False)
        self.sorter_natif = self.vue_liste.get_sorter()   # un seul critère, clic sur l'en-tête
        self.modele_trie = Gtk.SortListModel(model=filtree, sorter=self.sorter_natif)
        triee = self.modele_trie
        self.selection = Gtk.SingleSelection(model=triee, autoselect=True)
        self.selection.connect("notify::selected-item", lambda *_: self._sur_selection())
        self.vue_liste.set_model(self.selection)
        self.vue_liste.connect("activate", lambda *_: self.ouvrir_kimi())

        def rendu_alias(lbl, m):
            point = '<span foreground="#33d17a">●</span> ' if m.charge else "   "
            lbl.set_markup(point + GLib.markup_escape_text(m.alias))
            lbl.set_tooltip_text(m.nom)

        def rendu_qual(lbl, m):
            lbl.set_text(m.qual or "?")
            lbl.remove_css_class("dim-label")
            if not m.qual or m.qual == "?":
                lbl.add_css_class("dim-label")

        def rendu_refus(lbl, m):
            r = rang_refus(m.refus)
            couleur = "#33d17a" if r == 0 else "#e5a50a" if r <= 2 else "#e01b24" if r <= 4 else None
            txt = GLib.markup_escape_text(m.refus or "?")
            lbl.set_markup(f'<span foreground="{couleur}">{txt}</span>' if couleur else txt)

        def rendu_usage(lbl, m):
            lbl.set_text((m.usage or "").split(" · ")[0])
            lbl.set_tooltip_text(m.usage or None)

        def rendu_caps(lbl, m):
            marques = []
            if "thinking" in m.capacites:
                marques.append("réflexion")
            if "image_in" in m.capacites or "vision" in m.capacites:
                marques.append("voit images+vidéos" if "video_in" in m.capacites
                               else "voit images")
            # la génération d'images/vidéos passe par generer-media (ComfyUI) :
            # tout modèle dont les appels d'outils fonctionnent sait la piloter
            if m.outils_etat == "ok":
                marques.append("génère images+vidéos")
            elif m.outils_etat == "non":
                marques.append("sans outils")
            lbl.set_text(" · ".join(marques))
            lbl.add_css_class("dim-label")

        cols = [
            self._colonne("Alias", rendu_alias, lambda m: m.alias.lower(), 232),
            self._colonne("Moteur", lambda l, m: l.set_text(MOTEURS[m.provider].nom),
                          lambda m: ORDRE_MOTEUR.get(m.provider, 9), 104),
            self._colonne("Qualité", rendu_qual, rang_qualite_de := (lambda m: rang_qualite(m.qual)), 88),
            self._colonne("tok/s (* avant 20/09)", lambda l, m: l.set_text(str(m.tps or "?")),
                          lambda m: m.debit, 62),
            self._colonne("Refus", rendu_refus, lambda m: rang_refus(m.refus), 88),
            self._colonne("Usage", rendu_usage, lambda m: (m.usage or "").lower(), 112),
            self._colonne("Contexte", lambda l, m: l.set_text(f"{m.ctx:,}".replace(",", " ")),
                          lambda m: m.ctx, 96),
            self._colonne("Capacités", rendu_caps, None, expand=True),
        ]
        for c in cols:
            self.vue_liste.append_column(c)
        self.vue_liste.sort_by_column(cols[0], Gtk.SortType.ASCENDING)
        # Tri multi-colonnes (bouton « Tri… ») : les colonnes sans clé (None,
        # ex. Capacités) n'ont pas d'ordre naturel, donc absentes du popover.
        self.colonnes_tri = [(titre, cle) for titre, cle in
                             zip(("Alias", "Moteur", "Qualité", "tok/s", "Refus",
                                  "Usage", "Contexte"),
                                 (lambda m: m.alias.lower(),
                                  lambda m: ORDRE_MOTEUR.get(m.provider, 9),
                                  rang_qualite_de,
                                  lambda m: m.debit,
                                  lambda m: rang_refus(m.refus),
                                  lambda m: (m.usage or "").lower(),
                                  lambda m: m.ctx))]
        return Gtk.ScrolledWindow(child=self.vue_liste, hexpand=True, vexpand=True)

    def _construire_detail(self):
        boite = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12,
                        margin_start=14, margin_end=14, margin_top=14, margin_bottom=14)
        self.d_titre = Gtk.Label(xalign=0, wrap=True)
        self.d_titre.add_css_class("title-3")
        self.d_sous = Gtk.Label(xalign=0, wrap=True, selectable=True)
        self.d_sous.add_css_class("dim-label")
        boite.append(self.d_titre)
        boite.append(self.d_sous)

        groupe = Adw.PreferencesGroup(title="Service")
        self.d_lignes = {}
        for cle, titre in [("moteur", "Moteur"), ("chemin", "Dossier"), ("taille", "Taille"),
                           ("ctxs", "Contexte servi"), ("ctx", "Contexte de l'alias"),
                           ("gabarit", "Gabarit de chat")]:
            r = Adw.ActionRow(title=titre, subtitle="—", subtitle_selectable=True)
            r.set_subtitle_lines(2)
            groupe.add(r)
            self.d_lignes[cle] = r
        boite.append(groupe)

        fiche = Adw.PreferencesGroup(
            title="Fiche",
            description="Ce que montre le menu. Les bancs remplissent refus et tok/s.")
        self.f_qual = Adw.EntryRow(title="Qualité")
        self.f_tps = Adw.EntryRow(title="Débit (tok/s)")
        self.f_refus = Adw.EntryRow(title="Refus")
        self.f_usage = Adw.EntryRow(title="Usage")
        for r in (self.f_qual, self.f_tps, self.f_refus, self.f_usage):
            fiche.add(r)
        b_enr = Gtk.Button(label="Enregistrer la fiche", halign=Gtk.Align.END)
        b_enr.connect("clicked", lambda *_: self.enregistrer_fiche())
        fiche.add(b_enr)
        boite.append(fiche)

        return Gtk.ScrolledWindow(child=boite, width_request=380)

    # ---- filtre et sélection ------------------------------------------------
    def _passe_filtre(self, m):
        if self.filtre_moteur and m.provider != self.filtre_moteur:
            return False
        if self.filtre_sans_censure and rang_refus(m.refus) > 1:
            return False
        if self.filtre_outils and not m.outils_ok:
            return False
        if self.filtre_vedettes and not (m.usage or "").startswith("≈"):
            return False
        if self.filtre_code_os and not (m.usage or "").lower().startswith("code android/linux"):
            return False
        if self.filtre_texte:
            txt = self.filtre_texte
            # « vision »/« voit images » : mot-clé sémantique, pas un substring.
            # Seul un modèle qui SERT des images (vision_status) sort ; le mot
            # présent dans l'alias ou le nom d'un converti texte-seul ne compte
            # pas (décision chef 21/09). Le reste du texte reste substring.
            if "vision" in txt or "voit images" in txt:
                if m.vision_status != "vision":
                    return False
                reste = txt.replace("voit images", " ").replace("vision", " ").strip()
                if reste and reste not in m.texte_recherche():
                    return False
            elif txt not in m.texte_recherche():
                return False
        return True

    def _sur_recherche(self, entree):
        self.filtre_texte = entree.get_text().strip().lower()
        self.filtre.changed(Gtk.FilterChange.DIFFERENT)
        self._compter()

    def _sur_moteur(self, bouton, cle):
        if bouton.get_active():
            self.filtre_moteur = cle
            self.filtre.changed(Gtk.FilterChange.DIFFERENT)
            self._compter()

    def _sur_drapeaux(self, _b):
        self.filtre_sans_censure = self.b_censure.get_active()
        self.filtre_outils = self.b_outils.get_active()
        self.filtre_vedettes = self.b_vedettes.get_active()
        self.filtre_code_os = self.b_code_os.get_active()
        self.filtre.changed(Gtk.FilterChange.DIFFERENT)
        self._compter()

    # ---- tri multi-colonnes ---------------------------------------------
    def _construire_popover_tri(self):
        """Reconstruit le contenu du popover à chaque ouverture : plus
        simple et plus sûr qu'un état de widgets tenu en parallèle de
        `self.tri_multi`, et le nombre de colonnes triables est petit."""
        boite = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4,
                        margin_start=10, margin_end=10, margin_top=10, margin_bottom=10)
        actifs = {idx: desc for idx, desc in self.tri_multi}
        for idx, (titre, _cle) in enumerate(self.colonnes_tri):
            ligne = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            priorite = [i for i, (j, _d) in enumerate(self.tri_multi) if j == idx]
            libelle = titre if not priorite else f"{priorite[0] + 1}. {titre}"
            case = Gtk.CheckButton(label=libelle, active=idx in actifs)
            case.connect("toggled", self._sur_case_tri, idx)
            ligne.append(case)
            sens = Gtk.ToggleButton(label="↓" if actifs.get(idx) else "↑",
                                    active=bool(actifs.get(idx)),
                                    sensitive=idx in actifs,
                                    tooltip_text="Sens du tri pour cette colonne")
            sens.connect("toggled", self._sur_sens_tri, idx)
            ligne.append(sens)
            boite.append(ligne)
        if self.tri_multi:
            boite.append(Gtk.Separator(margin_top=4, margin_bottom=4))
            effacer = Gtk.Button(label="Revenir au tri simple (clic sur une colonne)")
            effacer.connect("clicked", self._effacer_tri_multi)
            boite.append(effacer)
        self.pop_tri.set_child(boite)

    def _sur_case_tri(self, case, idx):
        actif = {j for j, _d in self.tri_multi}
        if case.get_active() and idx not in actif:
            self.tri_multi.append((idx, False))
        elif not case.get_active() and idx in actif:
            self.tri_multi = [(j, d) for j, d in self.tri_multi if j != idx]
        self._appliquer_tri_multi()
        self._construire_popover_tri()   # renumérote les priorités affichées

    def _sur_sens_tri(self, bouton, idx):
        descendant = bouton.get_active()
        bouton.set_label("↓" if descendant else "↑")
        self.tri_multi = [(j, descendant if j == idx else d) for j, d in self.tri_multi]
        self._appliquer_tri_multi()

    def _effacer_tri_multi(self, *_a):
        self.tri_multi = []
        self.modele_trie.set_sorter(self.sorter_natif)
        self.pop_tri.popdown()

    def _appliquer_tri_multi(self):
        if not self.tri_multi:
            self.modele_trie.set_sorter(self.sorter_natif)
            return
        combine = Gtk.MultiSorter()
        for idx, descendant in self.tri_multi:
            _titre, cle = self.colonnes_tri[idx]

            def comparer(a, b, _u=None, f=cle, inverse=descendant):
                r = (f(a) > f(b)) - (f(a) < f(b))
                return -r if inverse else r

            combine.append(Gtk.CustomSorter.new(comparer))
        self.modele_trie.set_sorter(combine)

    def _compter(self):
        vus = self.selection.get_n_items()
        self.lbl_compte.set_text(f"{vus} / {self.store.get_n_items()} modèles")

    def selection_courante(self):
        return self.selection.get_selected_item()

    def _selectionner(self, alias):
        for i in range(self.selection.get_n_items()):
            if self.selection.get_item(i).alias == alias:
                self.selection.set_selected(i)
                return alias
        return None

    def _textes_visibles(self):
        """Tous les textes de Gtk.Label actuellement affichés sous la liste (DFS),
        colonne par colonne dans l'ordre des enfants — sert au crochet `trier:`."""
        acc = []

        def parcourir(w):
            if isinstance(w, Gtk.Label):
                acc.append(w.get_text())
            enfant = w.get_first_child()
            while enfant is not None:
                parcourir(enfant)
                enfant = enfant.get_next_sibling()

        parcourir(self.vue_liste)
        return acc

    def test_jouer(self):
        """ACVRAM_GUI_TEST (voir l'en-tête) : joue l'action, imprime `GUI_TEST {…}`, quitte. Retour = toasts, lignes de
        console, argv qui auraient été lancés, URI qui auraient été ouvertes, sélection, compte du filtre."""
        r = self._test = {"test": GUI_TEST, "toasts": [], "console": [], "spawns": [], "uris": []}
        self.toast = lambda texte: r["toasts"].append(texte)
        dire = self.dire
        self.dire = lambda texte, gras=False: (r["console"].append(texte), dire(texte, gras))
        Gio.SubprocessLauncher.spawnv = lambda lanceur, argv: r["spawns"].append(list(argv))
        Gtk.show_uri = lambda fen, uri, *_: r["uris"].append(str(uri))
        rc = 0
        try:
            genre, _, arg = GUI_TEST.partition(":")
            if genre == "filtre":
                self.recherche.set_text(arg); self._sur_recherche(self.recherche)   # search-changed n'arrive qu'après 150 ms
                r["visibles"], r["total"] = self.selection.get_n_items(), self.store.get_n_items()
            elif genre == "clic":
                nom, _, alias = arg.partition("@")
                if alias:
                    r["selection"] = self._selectionner(alias)
                b = getattr(self, nom, None)
                if not isinstance(b, Gtk.Button):
                    r["erreur"], rc = f"bouton inconnu : {nom}", 2
                elif isinstance(b, Gtk.ToggleButton):
                    b.set_active(not b.get_active()); r["actif"] = b.get_active()
                else:
                    r["sensible"] = b.get_sensitive(); b.emit("clicked")
            elif genre == "trier":
                # Simule un clic sur l'en-tête `arg` (gtk_column_view_sort_by_column
                # est exactement l'appel que fait le gestionnaire de clic interne du
                # ColumnView) ; laisse deux tours de boucle pour que la disposition
                # (et le texte des cellules) se stabilise avant/après, puis imprime
                # et quitte lui-même (retour anticipé : pas le print/quit générique).
                col = next((c for c in self.vue_liste.get_columns() if c.get_title() == arg), None)
                if col is None:
                    r["erreur"], rc = f"colonne inconnue : {arg}", 2
                    print("GUI_TEST " + json.dumps(r, ensure_ascii=False), flush=True)
                    self.get_application().rc_test = rc
                    self.get_application().quit()
                    return
                def apres_tri():
                    r["apres"] = self._textes_visibles()
                    print("GUI_TEST " + json.dumps(r, ensure_ascii=False), flush=True)
                    self.get_application().rc_test = 0
                    self.get_application().quit()
                    return False
                def avant_tri():
                    r["avant"] = self._textes_visibles()
                    self.vue_liste.sort_by_column(col, Gtk.SortType.ASCENDING)
                    GLib.timeout_add(200, apres_tri)
                    return False
                GLib.timeout_add(200, avant_tri)
                return
            else:
                r["erreur"], rc = f"forme inconnue : {GUI_TEST} (clic:<bouton>[@<alias>] | filtre:<texte> | trier:<titre colonne>)", 2
        except Exception as e:                          # le retour d'un gestionnaire qui plante est le plantage lui-même
            r["exception"], rc = f"{type(e).__name__}: {e}", 1
        print("GUI_TEST " + json.dumps(r, ensure_ascii=False), flush=True)
        self.get_application().rc_test = rc
        self.get_application().quit()
        return False

    def _sur_selection(self):
        m = self.selection_courante()
        if m is None:
            return
        self.d_titre.set_text(m.alias)
        self.d_sous.set_text(m.nom)
        self.d_lignes["moteur"].set_subtitle(
            f"{MOTEURS[m.provider].nom} · port {MOTEURS[m.provider].port}")
        self.d_lignes["chemin"].set_subtitle(m.dossier or "chargé par le serveur lui-même")
        self.d_lignes["ctxs"].set_subtitle(
            f"{int(m.ctx_service):,}".replace(",", " ") + " tokens" if m.ctx_service else "—")
        self.d_lignes["ctx"].set_subtitle(f"{m.ctx:,}".replace(",", " ") + " tokens")
        self.d_lignes["gabarit"].set_subtitle(
            os.path.basename(m.gabarit) if m.gabarit else "celui du modèle")
        self.d_lignes["taille"].set_subtitle(m.taille or ("—" if not m.dossier else "…"))
        self.f_qual.set_text(m.qual or "")
        self.f_tps.set_text(str(m.tps or ""))
        self.f_refus.set_text(m.refus or "")
        self.f_usage.set_text(m.usage or "")
        self.b_precharger.set_sensitive(m.lancable and self.proc is None)
        if m.dossier and m.taille is None:
            threading.Thread(target=self._mesurer_taille, args=(m,), daemon=True).start()

    def _mesurer_taille(self, m):
        try:
            s = subprocess.run(["du", "-sh", m.dossier], capture_output=True, text=True,
                               timeout=120)
            t = s.stdout.split("\t")[0].strip() if s.returncode == 0 else "—"
        except Exception:
            t = "—"
        m.taille = t or "—"
        GLib.idle_add(self._maj_taille, m)

    def _maj_taille(self, m):
        if self.selection_courante() is m:
            self.d_lignes["taille"].set_subtitle(m.taille)
        return False

    # ---- état des moteurs ---------------------------------------------------
    def _sonde_periodique(self):
        if self.proc is None:                       # pendant un lancement, on laisse le CPU
            self.sonder()
        return True

    def sonder(self):
        threading.Thread(target=self._sonder_fond, daemon=True).start()

    def _sonder_fond(self):
        coffre = secrets()
        etat = {cle: modele_servi(m, coffre) for cle, m in MOTEURS.items()}
        u, t = vram()
        GLib.idle_add(self._appliquer_etat, etat, u, t)

    def _appliquer_etat(self, etat, u, t):
        self.etat = etat
        actifs = [(MOTEURS[c].nom, v) for c, v in etat.items() if v]
        if actifs:
            self.lbl_etat.set_markup(
                " · ".join(f'<span foreground="#33d17a">●</span> <b>{GLib.markup_escape_text(n)}</b> '
                           f'{GLib.markup_escape_text(os.path.basename(str(v)).removesuffix(".gguf"))}'
                           for n, v in actifs))
        else:
            self.lbl_etat.set_markup('<span alpha="60%">aucun moteur en service</span>')
        if t:
            self.barre_vram.set_value(min(1.0, u / t))
            self.barre_vram.set_visible(True)
            self.lbl_vram.set_text(f"VRAM {(t - u) / 1024:.0f}/{t / 1024:.0f} Go libres")
        else:
            self.barre_vram.set_visible(False)
            self.lbl_vram.set_text("")
        for i in range(self.store.get_n_items()):
            m = self.store.get_item(i)
            servi = etat.get(m.provider)
            avant = m.charge
            m.charge = bool(servi) and self._correspond(m, servi)
            if avant != m.charge:
                self.store.items_changed(i, 1, 1)
        self._compter()
        return False

    @staticmethod
    def _correspond(m, servi):
        s = os.path.basename(str(servi)).removesuffix(".gguf")
        n = os.path.basename(m.nom or "").removesuffix(".gguf")
        if s == n:
            return True
        if m.dossier and s == os.path.basename(m.dossier):
            return True
        return False

    # ---- console et sous-processus -----------------------------------------
    LIGNES_CONSOLE = 4000

    def dire(self, texte, gras=False):
        tampon = self.console.get_buffer()
        tampon.insert(tampon.get_end_iter(), ("\n" if gras else "") + texte + "\n")
        # un chargement suivi par tail -F débite des milliers de lignes : sans
        # cette coupe, le TextView ralentit jusqu'à figer la fenêtre.
        trop = tampon.get_line_count() - self.LIGNES_CONSOLE
        if trop > 0:
            tampon.delete(tampon.get_start_iter(), tampon.get_iter_at_line(trop)[1])
        GLib.idle_add(self._defiler_bas)

    def _defiler_bas(self):
        # une seule marque, déplacée : en créer une par ligne les accumulait
        # toutes dans le tampon (20 000 lignes = 16 s au lieu de 0,04 s).
        tampon = self.console.get_buffer()
        tampon.move_mark(self.fin_console, tampon.get_end_iter())
        self.console.scroll_to_mark(self.fin_console, 0, False, 0, 0)
        return False

    def ouvrir_openwebui(self):
        """Open WebUI local (lanceur « openwebui » du dossier bin de parc.toml) : le lanceur démarre le serveur
        s'il dort puis ouvre le navigateur ; rien n'est dupliqué ici."""
        lanceur = BIN / "openwebui"
        if not lanceur.exists():
            self.toast(f"Lanceur absent : {lanceur}")
            return
        try:
            Gio.SubprocessLauncher.new(Gio.SubprocessFlags.NONE).spawnv([str(lanceur), "open"])
        except GLib.Error as e:
            self.toast(f"Open WebUI : {e.message}")
            return
        self.toast("Open WebUI : démarrage si besoin, le navigateur s'ouvre sur le port 3000")

    def toast(self, texte):
        self.toasts.add_toast(Adw.Toast(title=texte, timeout=4))

    def _occupe(self, oui):
        self.b_stop.set_sensitive(oui)
        self.b_precharger.set_sensitive(not oui and bool(self.selection_courante()
                                                         and self.selection_courante().lancable))
        self.b_ouvrir.set_sensitive(not oui)

    def executer(self, argv, titre, journal=None, fini=None, env=None):
        """Lance un outil du parc et affiche sa sortie en direct dans la console.
        Les lanceurs écrivent l'essentiel dans leur journal, pas sur leur sortie :
        sans le tail associé, la console resterait muette pendant tout le chargement."""
        if self.proc is not None:
            self.toast("Une opération est déjà en cours.")
            return
        self.b_console.set_active(True)
        self.dire(f"$ {' '.join(argv)}", gras=True)
        if GUI_TEST:                                   # à sec : la commande est le retour, rien n'est lancé
            self._test["spawns"].append(list(argv)); self._test["env"] = dict(env or {})
            return
        try:
            lanceur = Gio.SubprocessLauncher.new(
                Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_MERGE)
            lanceur.setenv("PATH", f"{BIN}:{os.environ.get('PATH', '')}", True)
            for k, v in (env or {}).items():
                lanceur.setenv(k, v, True)
            proc = lanceur.spawnv(argv)
        except GLib.Error as e:
            self.dire(f"impossible de lancer : {e.message}")
            self.toast("Lancement impossible")
            return
        self.proc = proc
        self._occupe(True)
        self._lire(Gio.DataInputStream.new(proc.get_stdout_pipe()))
        if journal and Path(journal).exists():
            try:
                self.proc_tail = Gio.SubprocessLauncher.new(
                    Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_SILENCE
                ).spawnv(["tail", "-n", "0", "-F", str(journal)])
                self._lire(Gio.DataInputStream.new(self.proc_tail.get_stdout_pipe()))
            except GLib.Error:
                self.proc_tail = None
        proc.wait_async(None, self._fin_proc, (titre, fini))

    def _lire(self, flux):
        """Lit en octets, pas en UTF-8 validé : un journal de serveur contient
        parfois un octet illégal (tail -F relit un fichier coupé au milieu d'un
        caractère). Avec read_line_finish_utf8, cette seule ligne arrêtait la
        lecture pour de bon — console muette, puis tuyau plein et lanceur figé."""
        def suite(source, res):
            try:
                donnees, _ = source.read_line_finish(res)
            except GLib.Error:
                donnees = b""                       # ligne perdue, pas le flux
            if donnees is None:                     # fin réelle du flux
                return
            texte = ANSI.sub("", donnees.decode("utf-8", "replace")).rstrip()
            if texte:
                self.dire(texte)
                if getattr(self, "capture", None) is not None:
                    self.capture.append(texte)
            source.read_line_async(GLib.PRIORITY_DEFAULT, None, suite)

        flux.read_line_async(GLib.PRIORITY_DEFAULT, None, suite)

    def _fin_proc(self, proc, res, donnees):
        titre, rappel = donnees
        try:
            proc.wait_finish(res)
            # après « Interrompre » (SIGKILL), get_exit_status assène un
            # G_CRITICAL et renvoie 1 : le fils n'est pas sorti par exit().
            if proc.get_if_exited():
                code = proc.get_exit_status()
            else:
                code = -proc.get_term_sig()
        except GLib.Error:
            code = -1
        if self.proc_tail is not None:
            self.proc_tail.force_exit()
            self.proc_tail = None
        self.proc = None
        self._occupe(False)
        etat = ("terminé" if code == 0 else
                "interrompu" if code < 0 else f"échec (code {code})")
        self.dire(f"— {titre} : {etat}")
        self.toast(f"{titre} : {etat}" + (" — voir la console" if code > 0 else ""))
        self.sonder()
        if rappel:
            rappel(code)

    def interrompre(self):
        if self.proc is None:
            return
        self.dire("— interruption demandée")
        self.proc.force_exit()

    # ---- actions ------------------------------------------------------------
    def _choisir_dossier_lancement(self, chemin):
        self.dossier_lancement = Path(chemin)
        self.b_dossier.set_label(f"Dossier : {self.dossier_lancement.name}")

    def _choisir_dossier_lancement_dialogue(self):
        dialogue = Gtk.FileDialog(initial_folder=Gio.File.new_for_path(str(self.dossier_lancement)))
        def fini(d, resultat):
            try:
                dossier = d.select_folder_finish(resultat)
            except GLib.Error:
                return
            if dossier:
                self._choisir_dossier_lancement(dossier.get_path())
        dialogue.select_folder(self, None, fini)

    def ouvrir_kimi(self):
        m = self.selection_courante()
        if m is None:
            return
        if self.profil["refuse_tabby_yals"] and m.provider in ("tabby", "yals"):
            self.toast(f"{MOTEURS[m.provider].nom} n'expose pas l'API Anthropic : "
                       "choisir un alias llamacpp-*, rapide-*, vllm-* ou acvram-*.")
            return
        term = self._terminal()
        if term is None:
            self.toast("Aucun terminal graphique trouvé")
            return
        # sous-shell : le CLI finit par exec, sans les parenthèses la pause
        # de fin ne serait jamais atteinte et la fenêtre se fermerait d'un coup.
        cmd = (f'cd {GLib.shell_quote(str(self.dossier_lancement))} && '
               f'({BIN}/{self.commande_claude(m)}); echo; read -rp "Entrée pour fermer… "')
        argv = term + ["bash", "-lc", cmd]
        try:
            Gio.SubprocessLauncher.new(Gio.SubprocessFlags.NONE).spawnv(argv)
            self.toast(f"{self.profil['nom']} ouvert sur {m.alias}")
        except GLib.Error as e:
            self.toast(f"Terminal : {e.message}")

    def ouvrir_web(self):
        # le bouton web fait ComfyUI (claude) ou l'interface kimi web (kimi),
        # selon le profil ; les deux implémentations vivent ici.
        if self.profil["web"] == "kimi_web":
            self.ouvrir_kimi_web()
        else:
            self.ouvrir_comfyui()

    def ouvrir_comfyui(self):
        comfy_url = self.profil["comfy_url"]
        comfy_start = self.profil["comfy_start"]

        def vivant():
            try:
                urllib.request.urlopen(f"{comfy_url}/system_stats", timeout=1)
                return True
            except Exception:
                return False

        if vivant():
            Gtk.show_uri(self, comfy_url, Gdk.CURRENT_TIME)
            self.toast("ComfyUIClaude ouvert")
            return
        if comfy_start is None or not comfy_start.exists():
            self.toast("ComfyUI non configuré (extras.comfy_start de parc.toml)" if comfy_start is None
                       else f"Script absent : {comfy_start}")
            return
        # ACVRAM_VERROU (même variable que carte.sh et surveillance-groupe.sh) : un test pose un verrou factice
        # ailleurs que sur le vrai — jxm 24/09 : le test du clic écrivait puis effaçait le vrai .qui en pleine mesure
        verrou = Path(os.environ.get("ACVRAM_VERROU", "/tmp/acvram-carte-0.lock") + ".qui")
        if verrou.exists() and verrou.read_text().strip():
            self.toast("ComfyUI : une mesure est en cours (verrou acvram)")
            return
        try:
            # le script démarre ComfyUIClaude (GPU 1), attend le port 8188 puis ouvre le navigateur
            Gio.SubprocessLauncher.new(Gio.SubprocessFlags.NONE).spawnv(["bash", str(comfy_start)])
        except GLib.Error as e:
            self.toast(f"ComfyUI : {e.message}")
            return
        self.toast("Démarrage de ComfyUIClaude… le navigateur s'ouvrira tout seul")

    def ouvrir_kimi_web(self):
        m = self.selection_courante()
        alias = m.alias if m else None
        port_web = self.profil["port_web"]
        kimi_bin = self.profil["kimi_bin"]

        def vivant():
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port_web}/api/v1/healthz", timeout=1)
                return True
            except Exception:
                return False

        def naviguer():
            jeton = ""
            try:
                jeton = (KIMI_DIR / "server.token").read_text().strip()
            except OSError:
                pass
            url = f"http://127.0.0.1:{port_web}/" + (f"#token={jeton}" if jeton else "")
            Gtk.show_uri(self, url, Gdk.CURRENT_TIME)
            self.toast(f"kimi web ouvert — choisir {alias} dans l'interface"
                       if alias else "kimi web ouvert")

        if vivant():
            naviguer()
            return
        try:
            lanceur = Gio.SubprocessLauncher.new(Gio.SubprocessFlags.NONE)
            bureau_kimi = DOSSIER_LANCEMENT_DEFAUT
            if bureau_kimi.is_dir():
                lanceur.set_cwd(str(bureau_kimi))
            lanceur.spawnv([str(kimi_bin), "web",
                            "--no-open", "--port", str(port_web)])
        except GLib.Error as e:
            self.toast(f"kimi web : {e.message}")
            return
        self.toast("Démarrage de kimi web…")

        def attendre():
            for _ in range(40):
                if vivant():
                    GLib.idle_add(naviguer)
                    return
                time.sleep(0.5)
            GLib.idle_add(self.toast, "kimi web n'a pas démarré")

        threading.Thread(target=attendre, daemon=True).start()

    def _terminal(self):
        for nom, args in [("gnome-terminal", ["--title", self.profil["terminal_titre"], "--"]), ("kgx", ["--"]),
                          ("ptyxis", ["--"]), ("konsole", ["-e"]), ("xfce4-terminal", ["-x"]),
                          ("alacritty", ["-e"]), ("kitty", ["--"]), ("xterm", ["-e"])]:
            chemin = shutil.which(nom)
            if chemin:
                return [chemin] + args
        return None

    def precharger(self):
        m = self.selection_courante()
        if m is None:
            return
        if self.profil["refuse_tabby_yals"]:
            if m.provider == "tabby":
                self.toast("TabbyAPI charge son modèle lui-même, rien à précharger ici — "
                           "et non utilisable avec Claude ensuite (API Anthropic absente).")
                return
            if m.provider == "yals":
                self.toast("YALS se lance par kimi-yals (API OpenAI seule) — "
                           "non utilisable avec Claude (API Anthropic absente).")
                return
        elif m.provider in ("tabby", "yals"):
            self.toast(f"{MOTEURS[m.provider].nom} charge son modèle lui-même : "
                       f"ouvrez-le dans {self.profil['nom']}.")
            return
        mot = MOTEURS[m.provider]
        env = {}
        if m.provider == "rapide":
            argv = [str(BIN / "llamacpp-appoint")]
        elif not m.dossier:
            self.toast(f"Aucun chemin connu pour {m.alias}")
            return
        elif m.provider == "acvram":
            argv = [str(BIN / "acvram-serveur"), m.alias]
        elif m.provider == "llamacpp":
            argv = [str(BIN / "llamacpp-serveur"), m.dossier, str(m.ctx_service or m.ctx)]
            # sans GABARIT, kimi-linear est servi avec le gabarit du GGUF et
            # n'appelle plus aucun outil : le menu texte, lui, le transmet.
            if m.gabarit:
                env["GABARIT"] = m.gabarit
        else:
            argv = [str(BIN / "vllm-serveur"), m.dossier, str(m.ctx_service or m.ctx * 2)]
        self.executer(argv, f"chargement de {m.alias}",
                      journal=mot.journal if mot.journal else None, env=env)

    def banc(self, lequel):
        m = self.selection_courante()
        if m is None:
            return
        servi = self.etat.get(m.provider)
        if not (servi and self._correspond(m, servi)):
            self.toast(f"{m.alias} n'est pas chargé : préchargez-le d'abord.")
            return
        port = MOTEURS[m.provider].port
        outil = "banc-outils" if lequel == "outils" else "banc-refus"
        self.capture = []
        self.executer([str(BIN / outil), "--port", str(port)],
                      f"banc {lequel} sur {m.alias}",
                      fini=lambda code, mm=m, l=lequel: self._fiche_depuis_banc(mm, l, code))

    def _fiche_depuis_banc(self, m, lequel, code):
        """Reporte le résultat d'un banc dans la fiche, sans ressaisie :
        banc-outils → tok/s et « outils n/N » dans l'usage ; banc-refus → colonne refus."""
        lignes = self.capture or []
        self.capture = None
        if code != 0:
            return
        maj = []
        for l in lignes:
            if lequel == "outils":
                r = re.search(r"→ (\d+)/(\d+) appels corrects · (\d+) tok/s", l)
                if r:
                    m.tps = r.group(3)
                    mention = f"outils {r.group(1)}/{r.group(2)}"
                    if re.search(r"outils \d+/\d+", m.usage):
                        m.usage = re.sub(r"outils \d+/\d+", mention, m.usage)
                    else:
                        m.usage = (m.usage.strip() + " · " + mention).strip(" ·") if m.usage.strip() not in ("", "?") else mention
                    maj += [f"{m.tps} tok/s", mention]
            else:
                r = re.search(r"\t(\d+)/(\d+) refus\t(\S.*)$", l)
                if r:
                    m.refus = r.group(3).strip()
                    maj.append(f"refus {r.group(1)}/{r.group(2)} → {m.refus}")
        if not maj:
            self.toast("Résultat non reconnu — reportez-le à la main dans la fiche")
            return
        try:
            ecrire_note(m.alias, m.refus, m.tps, m.qual, m.usage)
        except OSError as e:
            self.toast(f"Écriture impossible : {e}")
            return
        for i in range(self.store.get_n_items()):
            if self.store.get_item(i) is m:
                self.store.items_changed(i, 1, 1)
                break
        if self.selection_courante() is m:
            self._sur_selection()
        self.toast(f"Fiche de {m.alias} mise à jour : {', '.join(maj)}")

    def enregistrer_fiche(self):
        m = self.selection_courante()
        if m is None:
            return
        m.qual = self.f_qual.get_text().strip()
        m.tps = self.f_tps.get_text().strip()
        m.refus = self.f_refus.get_text().strip()
        m.usage = self.f_usage.get_text().strip()
        try:
            ecrire_note(m.alias, m.refus, m.tps, m.qual, m.usage)
        except OSError as e:
            self.toast(f"Écriture impossible : {e}")
            return
        for i in range(self.store.get_n_items()):
            if self.store.get_item(i) is m:
                self.store.items_changed(i, 1, 1)
                break
        self.toast(f"Fiche de {m.alias} enregistrée")

    def voir_journal(self):
        # le moteur du modèle choisi d'abord : l'appoint tourne en permanence et
        # masquerait toujours le journal du gros moteur, le seul intéressant.
        cible = None
        m = self.selection_courante()
        if m is not None:
            cible = MOTEURS[m.provider].journal
        if cible is None or not cible.exists():
            for cle in ("acvram", "llamacpp", "vllm", "rapide", "tabby", "yals"):
                if self.etat.get(cle) and MOTEURS[cle].journal:
                    cible = MOTEURS[cle].journal
                    break
        if cible is None or not cible.exists():
            self.toast("Aucun journal disponible")
            return
        self.b_console.set_active(True)
        self.dire(f"— {cible} (200 dernières lignes)", gras=True)
        try:
            lignes = cible.read_text(errors="replace").splitlines()[-200:]
        except OSError as e:
            lignes = [f"illisible : {e}"]
        for ligne in lignes:
            self.dire(ANSI.sub("", ligne))

    def arreter_moteur(self):
        vivants = [(cle, MOTEURS[cle]) for cle, v in self.etat.items() if v]
        if not vivants:
            self.toast("Aucun moteur en service")
            return
        noms = ", ".join(m.nom for _, m in vivants)
        d = Adw.AlertDialog(heading="Arrêter le moteur ?",
                            body=f"{noms} sera arrêté et sa VRAM libérée. "
                                 "Une session Claude en cours perdra son serveur.")
        d.add_response("non", "Annuler")
        d.add_response("oui", "Arrêter")
        d.set_response_appearance("oui", Adw.ResponseAppearance.DESTRUCTIVE)
        d.connect("response", lambda _d, r: self._arreter_vraiment(vivants) if r == "oui" else None)
        d.present(self)

    def _arreter_vraiment(self, vivants):
        self.b_console.set_active(True)
        for _, mot in vivants:
            pid = pid_du_port(mot.port)
            if pid is None:
                self.dire(f"— {mot.nom} : port {mot.port} sans propriétaire identifiable")
                continue
            self.dire(f"— arrêt de {mot.nom} (pid {pid})")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as e:
                self.dire(f"  échec : {e}")
        GLib.timeout_add_seconds(4, lambda: (self.sonder(), False)[1])

    def liberer_vram(self):
        outil = BIN / "liberer-vram"
        if not outil.exists():
            self.toast("liberer-vram absent")
            return
        self.executer([str(outil)], "libération de la VRAM")

    def commande_claude(self, m):
        """La commande « Ouvrir dans … » — une seule construction pour le bouton, le presse-papier et
        l'affichage (poste7 21/09 : deux constructeurs = une ligne qui peut mentir)."""
        return f"{self.profil['cli']} {GLib.shell_quote(m.alias)}"

    def copier_commande(self):
        m = self.selection_courante()
        if m is None:
            return
        cmd = self.commande_claude(m)
        Gdk.Display.get_default().get_clipboard().set(cmd)
        self.toast(f"Copié : {cmd}")

    # ---- rechargement -------------------------------------------------------
    def _recharger(self):
        try:
            parc = charger_parc()
        except RuntimeError as e:
            self.toast(str(e))
            return
        garde = self.selection_courante()
        alias_garde = garde.alias if garde else None
        self.parc = parc
        self.store.remove_all()
        for m in parc:
            self.store.append(m)
        self._compter()
        cible = 0
        if alias_garde:
            for i in range(self.selection.get_n_items()):
                if self.selection.get_item(i).alias == alias_garde:
                    cible = i
                    break
        if self.selection.get_n_items():
            self.selection.set_selected(cible)
        self._sur_selection()
        # Premier lancement (acvram-parc installé, aucun modèle balayé) : la
        # liste est vide sans que ce soit une faute (pièce 84) — proposer le
        # choix des dossiers directement, une seule fois par session, jamais
        # sous GUI_TEST (Gtk.FileDialog est modal et ne rend jamais sous test).
        if not parc and not self._dossiers_deja_proposes and not GUI_TEST:
            self._dossiers_deja_proposes = True
            GLib.idle_add(self.choisir_dossiers_modeles)

    # ---- dossiers de modèles (pièce 84) --------------------------------------
    def choisir_dossiers_modeles(self):
        dialogue = Gtk.FileDialog(title="Dossiers de modèles (acvram, GGUF, HF)")
        def fini(d, resultat):
            try:
                dossiers = d.select_multiple_folders_finish(resultat)
            except GLib.Error:
                return False
            chemins = [dossiers.get_item(i).get_path() for i in range(dossiers.get_n_items())]
            if chemins:
                self._balayer_dossiers(chemins)
            return False
        dialogue.select_multiple_folders(self, None, fini)

    def rebalayer(self):
        self._balayer_dossiers([])

    def _balayer_dossiers(self, dossiers):
        """Lance `parc-installer --auto --sans-balayage` (le même outil que l'installation
        manuelle, pas une réécriture) restreint aux dossiers déjà connus + ceux donnés :
        aucun balayage automatique de tous les disques depuis un clic de menu. Persiste les
        racines dans parc.toml (parc-installer le fait déjà), régénère les TSV et
        ~/.kimi-code/config.toml, puis recharge la liste."""
        outil = shutil.which("parc-installer") or "/usr/bin/parc-installer"
        argv = [outil, "--auto", "--sans-balayage", "--sans-minuteur"]
        for d in dossiers:
            argv += ["--racine", d]
        self.executer(argv, "balayage des dossiers de modèles",
                      fini=lambda code: self._recharger())


def appliquer_habillage(accent_bg, accent_color):
    """Les couleurs nommées accent_* pilotent boutons suggérés, sélections,
    interrupteurs et barres de niveau — une seule définition suffit. Les deux
    accents (violet claude, rose kimi) viennent du profil du lanceur."""
    css = f"""
    @define-color accent_bg_color {accent_bg};
    @define-color accent_color {accent_color};
    columnview row:selected, columnview row:selected:hover {{
      background: alpha({accent_bg}, 0.42);
    }}
    columnview row:hover {{ background: alpha({accent_bg}, 0.12); }}
    levelbar block.low, levelbar block.high, levelbar block.full {{
      background: {accent_bg}; border-color: {accent_bg};
    }}
    """
    prov = Gtk.CssProvider()
    prov.load_from_string(css)
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


