//! Poids : les `acvram-0000N.safetensors` projetés en mémoire, un index nom physique → fichier.
//! Rien n'est copié à l'ouverture ; `octets` rend la tranche du fichier projeté.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::{erreur, Resultat};

pub struct Poids {
    projections: Vec<Mmap>,
    /// nom physique → indice dans `projections`
    index: HashMap<String, usize>,
}

pub struct VueTenseur<'a> {
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
    pub octets: &'a [u8],
}

impl Poids {
    pub fn ouvrir(dossier: &Path) -> Resultat<Self> {
        let mut fichiers: Vec<_> = std::fs::read_dir(dossier)
            .map_err(|e| erreur!("dossier illisible {} : {e}", dossier.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        fichiers.sort();
        if fichiers.is_empty() {
            return Err(erreur!("aucun .safetensors dans {}", dossier.display()));
        }
        let mut projections = Vec::with_capacity(fichiers.len());
        let mut index = HashMap::new();
        for (i, f) in fichiers.iter().enumerate() {
            let fh = File::open(f).map_err(|e| erreur!("{} : {e}", f.display()))?;
            // SAFETY : fichier de poids en lecture seule, jamais réécrit pendant la vie du moteur.
            let mm = unsafe { Mmap::map(&fh) }.map_err(|e| erreur!("mmap {} : {e}", f.display()))?;
            let st = SafeTensors::deserialize(&mm).map_err(|e| erreur!("{} : {e}", f.display()))?;
            for nom in st.names() {
                if index.insert(nom.to_string(), i).is_some() {
                    return Err(erreur!("tenseur {nom} présent dans deux fichiers"));
                }
            }
            projections.push(mm);
        }
        Ok(Self { projections, index })
    }

    pub fn noms(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    pub fn vue(&self, nom: &str) -> Resultat<VueTenseur<'_>> {
        let i = *self.index.get(nom).ok_or_else(|| erreur!("tenseur physique absent : {nom}"))?;
        let st = SafeTensors::deserialize(&self.projections[i]).map_err(|e| erreur!("{e}"))?;
        let t = st.tensor(nom).map_err(|e| erreur!("{nom} : {e}"))?;
        Ok(VueTenseur { dtype: t.dtype(), shape: t.shape().to_vec(), octets: t.data() })
    }
}
