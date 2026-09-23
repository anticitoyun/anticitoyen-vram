//! Le moteur : chargement des poids sur la carte, préfill et décodage glouton b=1.
//!
//! La SÉQUENCE de noyaux d'un pas n'est pas reconstituée en lisant le Python (branches
//! d'environnement, norme fusionnée, résidu différé : `couches.py:422-462`) : elle est RELEVÉE sur le
//! moteur servi (prise 1, `scratchpad/poste5-rust-p1-23-09/`) puis transcrite ici famille par famille,
//! chacune avec son test au bit contre un vidage Python. Tant qu'elle ne l'est pas, `generer` refuse.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::manifeste::Manifeste;
use crate::noyaux::Noyaux;
use crate::poids::Poids;
use crate::tokeniseur::Tokeniseur;
use crate::{erreur, Resultat};

pub struct Moteur {
    pub manifeste: Manifeste,
    pub tokeniseur: Tokeniseur,
    pub noyaux: Noyaux,
    pub dossier: PathBuf,
    _ctx: Arc<CudaContext>,
    _flux: Arc<CudaStream>,
    /// nom physique → octets sur la carte (le format est lu au manifeste par le tenseur logique)
    carte: HashMap<String, CudaSlice<u8>>,
    pub octets_carte: usize,
}

impl Moteur {
    pub fn charger(dossier: &Path, so: &Path) -> Resultat<Self> {
        let manifeste = Manifeste::lire(dossier)?;
        let tokeniseur = Tokeniseur::charger(dossier)?;
        let ctx = CudaContext::new(0).map_err(|e| erreur!("contexte CUDA : {e:?}"))?;
        let flux = ctx.default_stream();
        let noyaux = Noyaux::charger(&ctx, so)?;
        let poids = Poids::ouvrir(dossier)?;
        let mut carte = HashMap::new();
        let mut octets_carte = 0usize;
        for nom in poids.noms().map(str::to_string).collect::<Vec<_>>() {
            let v = poids.vue(&nom)?;
            let d = flux.clone_htod(v.octets).map_err(|e| erreur!("{nom} vers la carte : {e:?}"))?;
            octets_carte += v.octets.len();
            carte.insert(nom, d);
        }
        flux.synchronize().map_err(|e| erreur!("{e:?}"))?;
        Ok(Self { manifeste, tokeniseur, noyaux, dossier: dossier.to_path_buf(), _ctx: ctx, _flux: flux, carte, octets_carte })
    }

    pub fn tenseurs_carte(&self) -> usize {
        self.carte.len()
    }

    /// Décodage glouton de `ids` : rend les ids générés (sans l'invite), arrêt sur EOS ou `max`.
    pub fn generer(&self, _ids: &[u32], _max: usize) -> Resultat<Vec<u32>> {
        Err(erreur!(
            "séquence de noyaux non transcrite : relevé de la prise 1 à porter (scratchpad/poste5-rust-p1-23-09)"
        ))
    }

    pub fn est_fin(&self, id: u32) -> bool {
        self.manifeste.model.eos_token_id.contains(&id)
    }
}

/// Glouton : premier indice du maximum, comme `torch.argmax` (sampler.py:150). Un NaN gagne,
/// comme dans ATen (le max de torch propage NaN) — il vaut mieux le voir que le masquer.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    let mut meilleur: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        match meilleur {
            None => meilleur = Some((i, v)),
            Some((_, m)) if m.is_nan() => {}
            Some((_, m)) if v.is_nan() || v > m => meilleur = Some((i, v)),
            _ => {}
        }
    }
    meilleur.map(|(i, _)| i)
}
