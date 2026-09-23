//! Noyaux Triton compilés par le moteur servi, lancés depuis Rust sans Triton.
//!
//! Le vidage (`outils/vidage_tables.py`) livre chaque variante RÉELLEMENT compilée : cubin, signature
//! (nom → type), constexpr (dont les entiers spécialisés à 1, absents des arguments), attributs de
//! divisibilité par 16, warps, mémoire partagée. Une variante se choisit sur ces clés — jamais par le seul
//! nom — et son nombre de paramètres se vérifie au chargement (`cuFuncGetParamInfo`) : un écart d'ABI est un
//! refus, pas un lancement.

use std::collections::{BTreeSet, HashMap};
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
    fn vaut_un(&self) -> bool {
        matches!(*self, Arg::I32(1) | Arg::I64(1))
    }
}

pub struct Variante {
    pub hash: String,
    fonction: sys::CUfunction,
    pub num_warps: u32,
    pub shared: u32,
    /// noms des paramètres dans l'ordre de la signature, et leur type Triton
    signature: Vec<(String, String)>,
    /// indices de la signature fixés à la compilation (constexpr déclarés ou entiers spécialisés)
    constexprs: HashMap<usize, serde_json::Value>,
    div16: BTreeSet<usize>,
    /// arguments de brouillon ajoutés en fin par Triton (global_scratch, profile_scratch)
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

fn indices_divisibilite(attrs: &str) -> BTreeSet<usize> {
    // forme `{(0,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], ...}`
    let mut s = BTreeSet::new();
    for morceau in attrs.split('(').skip(1) {
        if let Some((idx, reste)) = morceau.split_once(",)") {
            if reste.contains("tt.divisibility") && reste.split(']').next().is_some_and(|r| r.contains("16")) {
                if let Ok(i) = idx.trim().parse::<usize>() {
                    s.insert(i);
                }
            }
        }
    }
    s
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
            let signature: Vec<(String, String)> = v["signature"]
                .as_object()
                .ok_or_else(|| erreur!("{hash} : signature absente"))?
                .iter()
                .map(|(k, t)| (k.clone(), t.as_str().unwrap_or_default().to_string()))
                .collect();
            let constexprs: HashMap<usize, serde_json::Value> = v["constexprs"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, val)| {
                            let i = k.trim_matches(|c| c == '(' || c == ')' || c == ',').trim().parse::<usize>().ok()?;
                            Some((i, val.clone()))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let div16 = indices_divisibilite(v["attrs"].as_str().unwrap_or_default());
            let brouillons = ["global_scratch_size", "profile_scratch_size"]
                .iter()
                .filter(|k| !v[**k].is_null())
                .count();
            let fonction_cu = charger_cubin(&cubin, &nom)?;
            let attendus = signature.iter().enumerate().filter(|(i, (_, t))| t != "constexpr" && !constexprs.contains_key(i)).count() + brouillons;
            let n = nombre_de_parametres(fonction_cu);
            if n != attendus {
                return Err(erreur!("{nom} {hash} : {n} paramètres dans le cubin, {attendus} attendus d'après la signature"));
            }
            variantes.push(Variante {
                hash,
                fonction: fonction_cu,
                num_warps: v["num_warps"].as_u64().unwrap_or(4) as u32,
                shared: v["shared"].as_u64().unwrap_or(0) as u32,
                signature,
                constexprs,
                div16,
                brouillons,
            });
        }
        if variantes.is_empty() {
            return Err(erreur!("aucune variante de {fonction} dans le vidage"));
        }
        Ok(Self { nom, variantes })
    }

    /// Lance la variante dont les constexpr et la divisibilité correspondent exactement à `args`
    /// (nom de paramètre → valeur ; `ct` = valeur attendue des constexpr nommés).
    pub fn lancer(&self, flux: sys::CUstream, grille: (u32, u32, u32), args: &HashMap<&str, Arg>,
                  constantes: &HashMap<&str, i64>) -> Resultat<&str> {
        let mut choix = None;
        'v: for v in &self.variantes {
            for (i, (nom, t)) in v.signature.iter().enumerate() {
                if t == "constexpr" {
                    if let Some(&c) = constantes.get(nom.as_str()) {
                        if v.constexprs.get(&i).and_then(|x| x.as_i64()) != Some(c) {
                            continue 'v;
                        }
                    }
                    continue;
                }
                let a = args.get(nom.as_str()).ok_or_else(|| erreur!("{} : argument {nom} manquant", self.nom))?;
                if v.constexprs.contains_key(&i) != a.vaut_un() || v.div16.contains(&i) != a.divisible_16() {
                    continue 'v;
                }
            }
            choix = Some(v);
            break;
        }
        let v = choix.ok_or_else(|| erreur!("{} : aucune variante compilée ne correspond à ces arguments", self.nom))?;
        let mut valeurs: Vec<[u8; 8]> = Vec::new();
        for (i, (nom, t)) in v.signature.iter().enumerate() {
            if t == "constexpr" || v.constexprs.contains_key(&i) {
                continue;
            }
            let mut b = [0u8; 8];
            match (args[nom.as_str()], t.as_str()) {
                (Arg::Ptr(p), t) if t.starts_with('*') => b.copy_from_slice(&p.to_le_bytes()),
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
    use super::indices_divisibilite;

    #[test]
    fn divisibilite_lue_dans_les_attributs() {
        let a = "{(0,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}";
        assert_eq!(indices_divisibilite(a).into_iter().collect::<Vec<_>>(), vec![0, 3, 12]);
        assert!(indices_divisibilite("{}").is_empty());
    }
}
