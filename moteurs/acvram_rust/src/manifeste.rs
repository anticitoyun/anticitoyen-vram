//! Lecture de `acvram_manifest.json` : spécification du modèle et format de chaque tenseur logique.
//! Seuls les champs dont le moteur a besoin sont lus ; les autres (plan, options) restent au Python.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::{erreur, Resultat};

#[derive(Debug, Clone, Deserialize)]
pub struct SpecModele {
    pub name: String,
    pub architecture: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub eos_token_id: Vec<u32>,
}

/// Un tenseur logique (`model.layers.0.mlp.up_proj.weight`) et les tenseurs physiques qui le
/// portent (`.qweight`, `.block_scale`, …) selon son format.
#[derive(Debug, Clone, Deserialize)]
pub struct EntreeTenseur {
    pub format: String,
    pub shape: Vec<usize>,
    pub keys: Vec<String>,
    #[serde(default)]
    pub promoted_from: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifeste {
    pub acvram_version: serde_json::Value,
    pub model: SpecModele,
    pub tensors: HashMap<String, EntreeTenseur>,
    pub weight_map: HashMap<String, String>,
}

impl Manifeste {
    pub fn lire(dossier: &Path) -> Resultat<Self> {
        let chemin = dossier.join("acvram_manifest.json");
        let texte = std::fs::read_to_string(&chemin)
            .map_err(|e| erreur!("manifeste illisible {} : {e}", chemin.display()))?;
        let m: Manifeste = serde_json::from_str(&texte)
            .map_err(|e| erreur!("manifeste invalide {} : {e}", chemin.display()))?;
        m.verifier_perimetre()?;
        Ok(m)
    }

    /// L'étape 1 ne couvre qu'un modèle DENSE de type llama/Qwen3 ; tout le reste est refusé ici,
    /// nommément, plutôt que servi à moitié.
    fn verifier_perimetre(&self) -> Resultat<()> {
        let s = &self.model;
        if s.num_experts != 0 {
            return Err(erreur!("{} : MoE ({} experts) hors du périmètre de l'étape 1", s.name, s.num_experts));
        }
        if s.architecture != "llama" {
            return Err(erreur!("{} : architecture {:?} hors du périmètre de l'étape 1", s.name, s.architecture));
        }
        Ok(())
    }

    pub fn tenseur(&self, nom: &str) -> Resultat<&EntreeTenseur> {
        self.tensors.get(nom).ok_or_else(|| erreur!("tenseur absent du manifeste : {nom}"))
    }

    /// Formats présents, avec leur nombre de tenseurs logiques — imprimé au chargement.
    pub fn formats(&self) -> Vec<(String, usize)> {
        let mut n: HashMap<&str, usize> = HashMap::new();
        for e in self.tensors.values() {
            *n.entry(e.format.as_str()).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = n.into_iter().map(|(k, c)| (k.to_string(), c)).collect();
        v.sort();
        v
    }
}
