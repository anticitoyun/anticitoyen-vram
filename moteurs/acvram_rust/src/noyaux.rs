//! Nos noyaux, pris dans l'extension que le moteur Python SERT (`~/.cache/acvram/kernels-<hash>/acvram_kernels.so`).
//!
//! Pourquoi pas une recompilation : `acvram_kernels.cu` inclut `torch/extension.h` et ses lanceurs
//! prennent des `torch::Tensor` — aucun lien C sans libtorch, et une recompilation hors de la chaîne
//! Python ne garantirait pas le même SASS. Le fatbin de la section `.nv_fatbin` est chargé tel quel
//! par le pilote (`cuModuleLoadData` accepte un fatbin) : les deux moteurs exécutent le même binaire.
//!
//! Les noyaux vivent dans un espace de noms ANONYME dont le nom mutilé change à chaque compilation
//! (`N50_GLOBAL__N__c2b095cd_17_acvram_kernels_cu_487d8e4e25paged_attn…`) : on les indexe par une clé
//! où ce segment est remplacé par `@`, et une clé qui désigne deux symboles est un refus.

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{sys, CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use object::{Object, ObjectSection};
use sha2::{Digest, Sha256};

use crate::{erreur, Resultat};

const MAGIC_FATBIN: u32 = 0xBA55_ED50;

/// Clé stable d'un symbole mutilé : le segment `<n>_GLOBAL__N__…` (longueur préfixée) devient `@`.
/// Un symbole sans espace anonyme est rendu tel quel.
pub fn cle_stable(mutile: &str) -> String {
    let Some(p) = mutile.find("_GLOBAL__N__") else {
        return mutile.to_string();
    };
    let debut_nombre = mutile[..p].rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    let Ok(longueur) = mutile[debut_nombre..p].parse::<usize>() else {
        return mutile.to_string();
    };
    let fin = p + longueur;
    if debut_nombre == p || fin > mutile.len() {
        return mutile.to_string();
    }
    format!("{}@{}", &mutile[..debut_nombre], &mutile[fin..])
}

/// La section `.nv_fatbin` de l'extension, vérifiée (magie, une seule image, taille cohérente).
pub fn fatbin_de(so: &Path) -> Resultat<Vec<u8>> {
    let donnees = std::fs::read(so).map_err(|e| erreur!("{} : {e}", so.display()))?;
    let obj = object::File::parse(&*donnees).map_err(|e| erreur!("{} : ELF illisible : {e}", so.display()))?;
    let section = obj
        .section_by_name(".nv_fatbin")
        .ok_or_else(|| erreur!("{} : pas de section .nv_fatbin", so.display()))?;
    let octets = section.data().map_err(|e| erreur!("{e}"))?;
    if octets.len() < 16 {
        return Err(erreur!("fatbin tronqué ({} o)", octets.len()));
    }
    let magie = u32::from_le_bytes(octets[0..4].try_into().unwrap());
    let entete = u16::from_le_bytes(octets[6..8].try_into().unwrap()) as usize;
    let charge = u64::from_le_bytes(octets[8..16].try_into().unwrap()) as usize;
    if magie != MAGIC_FATBIN {
        return Err(erreur!("fatbin : magie {magie:#x}, attendu {MAGIC_FATBIN:#x}"));
    }
    // Une seule image dans la section : c'est ce que produit la chaîne torch (vérifié 23/09 : 16 + 14 454 008 o).
    if entete + charge > octets.len() {
        return Err(erreur!("fatbin : {entete} + {charge} o annoncés, section de {} o", octets.len()));
    }
    Ok(octets[..entete + charge].to_vec())
}

pub struct Noyaux {
    module: Arc<CudaModule>,
    /// clé stable → nom mutilé exact
    noms: HashMap<String, String>,
    pub sha256_fatbin: String,
}

impl Noyaux {
    pub fn charger(ctx: &Arc<CudaContext>, so: &Path) -> Resultat<Self> {
        let fatbin = fatbin_de(so)?;
        let sha256_fatbin = format!("{:x}", Sha256::digest(&fatbin));
        let noms_mutiles = enumerer(ctx, &fatbin)?;
        let mut noms = HashMap::new();
        let mut ambigus = Vec::new();
        for n in noms_mutiles {
            let cle = cle_stable(&n);
            if let Some(ancien) = noms.insert(cle.clone(), n.clone()) {
                ambigus.push(format!("{cle} ({ancien} / {n})"));
            }
        }
        if !ambigus.is_empty() {
            return Err(erreur!("{} clés ambiguës, premier : {}", ambigus.len(), ambigus[0]));
        }
        let module = ctx
            .load_module(Ptx::from_binary(fatbin))
            .map_err(|e| erreur!("chargement du fatbin : {e:?}"))?;
        Ok(Self { module, noms, sha256_fatbin })
    }

    pub fn nombre(&self) -> usize {
        self.noms.len()
    }

    /// Les clés qui contiennent `motif` — pour construire la table d'appel d'après le relevé.
    pub fn cles(&self, motif: &str) -> Vec<&str> {
        let mut v: Vec<&str> = self.noms.keys().map(|s| s.as_str()).filter(|k| k.contains(motif)).collect();
        v.sort();
        v
    }

    pub fn fonction(&self, cle: &str) -> Resultat<CudaFunction> {
        let nom = self.noms.get(cle).ok_or_else(|| erreur!("noyau absent du fatbin servi : {cle}"))?;
        self.module.load_function(nom).map_err(|e| erreur!("{nom} : {e:?}"))
    }
}

/// Noms des fonctions du fatbin, par l'API pilote (CUDA ≥ 12.4) : un module temporaire, énuméré puis
/// déchargé. cudarc ne rend pas le `CUmodule` qu'il charge, d'où ce second chargement au démarrage.
fn enumerer(ctx: &Arc<CudaContext>, fatbin: &[u8]) -> Resultat<Vec<String>> {
    ctx.bind_to_thread().map_err(|e| erreur!("{e:?}"))?;
    let verifier = |r: sys::CUresult, quoi: &str| -> Resultat<()> {
        if r == sys::CUresult::CUDA_SUCCESS { Ok(()) } else { Err(erreur!("{quoi} : {r:?}")) }
    };
    // SAFETY : appels pilote sur un contexte lié au fil ; le module est déchargé avant de rendre.
    unsafe {
        let mut m: sys::CUmodule = std::ptr::null_mut();
        verifier(sys::cuModuleLoadData(&mut m, fatbin.as_ptr() as *const _), "cuModuleLoadData")?;
        let mut n: std::ffi::c_uint = 0;
        let r = sys::cuModuleGetFunctionCount(&mut n, m);
        let mut fonctions = vec![std::ptr::null_mut(); n as usize];
        let r2 = if r == sys::CUresult::CUDA_SUCCESS {
            sys::cuModuleEnumerateFunctions(fonctions.as_mut_ptr(), n, m)
        } else {
            r
        };
        let mut noms = Vec::with_capacity(n as usize);
        let mut r3 = sys::CUresult::CUDA_SUCCESS;
        if r2 == sys::CUresult::CUDA_SUCCESS {
            for f in &fonctions {
                let mut p: *const std::ffi::c_char = std::ptr::null();
                r3 = sys::cuFuncGetName(&mut p, *f);
                if r3 != sys::CUresult::CUDA_SUCCESS {
                    break;
                }
                noms.push(CStr::from_ptr(p).to_string_lossy().into_owned());
            }
        }
        sys::cuModuleUnload(m);
        verifier(r, "cuModuleGetFunctionCount")?;
        verifier(r2, "cuModuleEnumerateFunctions")?;
        verifier(r3, "cuFuncGetName")?;
        Ok(noms)
    }
}
