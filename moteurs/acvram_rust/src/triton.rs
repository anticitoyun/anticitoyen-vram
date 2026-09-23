//! Noyaux Triton compilés par le moteur servi, lancés depuis Rust sans Triton.
//!
//! Le vidage (`outils/vidage_tables.py`) livre chaque variante RÉELLEMENT compilée : cubin, noms des
//! paramètres (signature, ordre conservé), warps, mémoire partagée, et la CLÉ de cache Triton 3.8 —
//! `[('*bf16', 'D'), ('i32', ''), ('fp32', None), ('constexpr', 4), …]` — qui dit, paramètre par paramètre,
//! le type, la divisibilité par 16 supposée à la compilation (`'D'`) et la valeur de chaque constexpr (y compris
//! les entiers spécialisés à 1, absents des arguments). Une variante se choisit sur cette clé, jamais par le seul
//! nom, et le nombre de paramètres du cubin se vérifie au chargement : un écart d'ABI est un refus.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::Path;

use cudarc::driver::sys;

use crate::{erreur, Resultat};

/// Valeur d'un argument, typée comme la signature Triton l'exige.
#[derive(Debug, Clone, Copy)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    I64(i64),
    F32(f32),
}

impl Arg {
    fn divisible_16(&self) -> bool {
        match *self {
            Arg::Ptr(p) => p % 16 == 0,
            Arg::I32(v) => v % 16 == 0,
            Arg::I64(v) => v % 16 == 0,
            Arg::F32(_) => false,
        }
    }
    fn entier(&self) -> Option<i64> {
        match *self {
            Arg::I32(v) => Some(v as i64),
            Arg::I64(v) => Some(v),
            _ => None,
        }
    }
}

/// Un paramètre de la clé de cache.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// argument passé au lancement : type Triton, divisibilité par 16 supposée (None pour un flottant)
    Execution { ty: String, div16: Option<bool> },
    /// valeur fixée à la compilation
    Constante(i64),
}

/// Lit la clé `[('*bf16', 'D'), ('i32', ''), ('fp32', None), ('constexpr', True), …]{…}`.
pub fn lire_cle(cle: &str) -> Resultat<Vec<Param>> {
    let corps = cle
        .strip_prefix("[(")
        .and_then(|s| s.split(")]").next())
        .ok_or_else(|| erreur!("clé Triton illisible : {}", &cle[..cle.len().min(60)]))?;
    let mut v = Vec::new();
    for item in corps.split("), (") {
        let (ty, spec) = item.split_once(", ").ok_or_else(|| erreur!("clé Triton : élément {item:?}"))?;
        let ty = ty.trim_matches('\'').to_string();
        let spec = spec.trim();
        if ty == "constexpr" {
            let val = match spec {
                "True" => 1,
                "False" => 0,
                s => s.parse::<i64>().map_err(|_| erreur!("constexpr non entier : {s}"))?,
            };
            v.push(Param::Constante(val));
        } else {
            let div16 = match spec {
                "None" => None,
                "'D'" => Some(true),
                "''" => Some(false),
                s => return Err(erreur!("spécialisation inconnue {s} pour {ty}")),
            };
            v.push(Param::Execution { ty, div16 });
        }
    }
    Ok(v)
}

pub struct Variante {
    pub hash: String,
    fonction: sys::CUfunction,
    pub num_warps: u32,
    pub shared: u32,
    noms: Vec<String>,
    params: Vec<Param>,
    /// arguments de brouillon ajoutés en fin par Triton (global_scratch, profile_scratch), passés nuls
    brouillons: usize,
}

// SAFETY : un `CUfunction` est un handle de processus, valide dans tout fil où le contexte est lié ; le moteur
// ne lance que sous son verrou (serveur.rs), jamais en concurrence.
unsafe impl Send for Variante {}
unsafe impl Sync for Variante {}

pub struct NoyauTriton {
    pub nom: String,
    pub variantes: Vec<Variante>,
}

impl NoyauTriton {
    pub fn charger(dossier: &Path, fonction: &str) -> Resultat<Self> {
        let texte = std::fs::read_to_string(dossier.join("tables.json")).map_err(|e| erreur!("tables.json : {e}"))?;
        let j: serde_json::Value = serde_json::from_str(&texte).map_err(|e| erreur!("tables.json : {e}"))?;
        let mut variantes = Vec::new();
        let mut nom = String::new();
        for v in j["triton"].as_array().ok_or_else(|| erreur!("tables.json : pas de liste triton"))? {
            if v["fonction"] != fonction {
                continue;
            }
            nom = v["nom"].as_str().unwrap_or_default().to_string();
            let hash = v["hash"].as_str().unwrap_or_default().to_string();
            let cubin = std::fs::read(dossier.join("triton").join(format!("{hash}.cubin")))
                .map_err(|e| erreur!("cubin {hash} : {e}"))?;
            let noms: Vec<String> = v["signature"]
                .as_object()
                .ok_or_else(|| erreur!("{hash} : signature absente"))?
                .keys()
                .cloned()
                .collect();
            let params = lire_cle(v["cle"].as_str().unwrap_or_default())?;
            if params.len() != noms.len() {
                return Err(erreur!("{nom} {hash} : clé de {} paramètres, signature de {}", params.len(), noms.len()));
            }
            let fonction_cu = charger_cubin(&cubin, &nom)?;
            let execution = params.iter().filter(|p| matches!(p, Param::Execution { .. })).count();
            let n = nombre_de_parametres(fonction_cu);
            let brouillons = n.checked_sub(execution).filter(|b| *b <= 2).ok_or_else(|| {
                erreur!("{nom} {hash} : {n} paramètres dans le cubin, {execution} arguments d'exécution")
            })?;
            variantes.push(Variante {
                hash,
                fonction: fonction_cu,
                num_warps: v["num_warps"].as_u64().unwrap_or(4) as u32,
                shared: v["shared"].as_u64().unwrap_or(0) as u32,
                noms,
                params,
                brouillons,
            });
        }
        if variantes.is_empty() {
            return Err(erreur!("aucune variante de {fonction} dans le vidage"));
        }
        Ok(Self { nom, variantes })
    }

    /// Lance l'unique variante compatible : chaque constexpr égal à `constantes` (ou, pour un paramètre absent
    /// de `constantes`, à la valeur entière de `args` — cas des entiers spécialisés à 1), chaque argument
    /// d'exécution de la même divisibilité par 16 qu'à la compilation.
    pub fn lancer(&self, flux: sys::CUstream, grille: (u32, u32, u32), args: &HashMap<&str, Arg>,
                  constantes: &HashMap<&str, i64>) -> Resultat<&str> {
        let compatible = |v: &Variante| -> Resultat<bool> {
            for (nom, p) in v.noms.iter().zip(&v.params) {
                match p {
                    Param::Constante(c) => {
                        let attendu = match constantes.get(nom.as_str()) {
                            Some(&x) => Some(x),
                            None => args.get(nom.as_str()).and_then(Arg::entier),
                        };
                        if attendu != Some(*c) {
                            return Ok(false);
                        }
                    }
                    Param::Execution { div16, .. } => {
                        let a = args.get(nom.as_str()).ok_or_else(|| erreur!("{} : argument {nom} manquant", self.nom))?;
                        if let Some(d) = div16 {
                            if *d != a.divisible_16() {
                                return Ok(false);
                            }
                        }
                    }
                }
            }
            Ok(true)
        };
        let mut choix = None;
        for v in &self.variantes {
            if compatible(v)? {
                if choix.is_some() {
                    return Err(erreur!("{} : deux variantes compatibles, choix ambigu", self.nom));
                }
                choix = Some(v);
            }
        }
        let v = choix.ok_or_else(|| erreur!("{} : aucune variante compilée ne correspond à ces arguments", self.nom))?;
        let mut valeurs: Vec<[u8; 8]> = Vec::new();
        for (nom, p) in v.noms.iter().zip(&v.params) {
            let Param::Execution { ty, .. } = p else { continue };
            let mut b = [0u8; 8];
            match (args[nom.as_str()], ty.as_str()) {
                (Arg::Ptr(x), t) if t.starts_with('*') => b.copy_from_slice(&x.to_le_bytes()),
                (Arg::I32(x), "i32") => b[..4].copy_from_slice(&x.to_le_bytes()),
                (Arg::I64(x), "i64") => b.copy_from_slice(&x.to_le_bytes()),
                (Arg::F32(x), "fp32") => b[..4].copy_from_slice(&x.to_le_bytes()),
                (a, t) => return Err(erreur!("{} : {nom} vaut {a:?}, type Triton {t}", self.nom)),
            }
            valeurs.push(b);
        }
        valeurs.extend(std::iter::repeat_n([0u8; 8], v.brouillons));
        let mut ptrs: Vec<*mut c_void> = valeurs.iter_mut().map(|b| b.as_mut_ptr() as *mut c_void).collect();
        // SAFETY : fonction chargée de ce cubin, paramètres comptés au chargement, grille/bloc/partagée du vidage.
        let r = unsafe {
            sys::cuLaunchKernel(v.fonction, grille.0, grille.1, grille.2, 32 * v.num_warps, 1, 1, v.shared,
                                flux, ptrs.as_mut_ptr(), std::ptr::null_mut())
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("{} {} : {r:?}", self.nom, v.hash));
        }
        Ok(&v.hash)
    }
}

fn charger_cubin(cubin: &[u8], nom: &str) -> Resultat<sys::CUfunction> {
    // SAFETY : contexte courant lié par l'appelant (le moteur lie son contexte au fil avant de charger).
    unsafe {
        let mut m: sys::CUmodule = std::ptr::null_mut();
        let r = sys::cuModuleLoadData(&mut m, cubin.as_ptr() as *const c_void);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("cubin {nom} : {r:?}"));
        }
        let c = CString::new(nom).unwrap();
        let mut f: sys::CUfunction = std::ptr::null_mut();
        let r = sys::cuModuleGetFunction(&mut f, m, c.as_ptr());
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("fonction {nom} : {r:?}"));
        }
        Ok(f)
    }
}

fn nombre_de_parametres(f: sys::CUfunction) -> usize {
    let mut n = 0;
    loop {
        let (mut off, mut taille) = (0usize, 0usize);
        // SAFETY : interrogation en lecture seule d'une fonction chargée.
        let r = unsafe { sys::cuFuncGetParamInfo(f, n, &mut off, &mut taille) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return n;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{lire_cle, Param};

    #[test]
    fn cle_de_cache_lue() {
        let c = "[('*bf16', 'D'), ('i32', ''), ('fp32', None), ('constexpr', 1), ('constexpr', 16), ('constexpr', True)]{'num_warps': 8}";
        let p = lire_cle(c).unwrap();
        assert_eq!(p[0], Param::Execution { ty: "*bf16".into(), div16: Some(true) });
        assert_eq!(p[1], Param::Execution { ty: "i32".into(), div16: Some(false) });
        assert_eq!(p[2], Param::Execution { ty: "fp32".into(), div16: None });
        assert_eq!(&p[3..], &[Param::Constante(1), Param::Constante(16), Param::Constante(1)]);
        assert!(lire_cle("{}").is_err());
    }
}
