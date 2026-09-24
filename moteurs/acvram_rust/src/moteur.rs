//! Le moteur : poids sur la carte dans la disposition servie, cache KV int8 paginé, et le pas de décodage
//! b=1 dans l'ordre exact du relevé (`decodage.rs`). Option A du chef : le préfill vient du moteur Python
//! (KV et premier jeton vidés, `outils/vidage_reference.py`) ; `decoder_injecte` est la porte de l'étape 1.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
use safetensors::SafeTensors;

use crate::argmax::Argmax;
use crate::chargement::Chargeur;
use crate::decodage::{nblk_du_pas, tranches, Couche, GateUp, Lineaire, Qkv};
use crate::lanceurs::{self as l, cles, Ptr, NUL};
use crate::manifeste::Manifeste;
use crate::noyaux::Noyaux;
use crate::poids::Poids;
use crate::tokeniseur::Tokeniseur;
use crate::triton::{Arg, NoyauTriton};
use crate::{erreur, Resultat};

const BLOC: u32 = 16;
const NREP_TUILE: u32 = 16;
const TRANCHES_MAX: u32 = 32;

struct Fonctions {
    nvfp4: CudaFunction,
    int8: CudaFunction,
    int8_f32: CudaFunction,
    rmsnorm: CudaFunction,
    rope: CudaFunction,
    kv: CudaFunction,
    swiglu: CudaFunction,
    swiglu2: CudaFunction,
}

struct Kv {
    kc: Ptr,
    vc: Ptr,
    ks: Ptr,
    vs: Ptr,
}

/// Tampons du pas, adresses fixes pour toute la vie du moteur.
struct Tampons {
    x: [Ptr; 2],
    delta: Ptr,
    h: Ptr,
    qkv: Ptr,
    attn: Ptr,
    a: Ptr,
    gu: Ptr,
    g: Ptr,
    u: Ptr,
    act: Ptr,
    logits: Ptr,
    part: Ptr,
    pm: Ptr,
    pl: Ptr,
    cnt: Ptr,
    tables: Ptr,
    lens: Ptr,
    pos: Ptr,
    slot: Ptr,
    jeton: Ptr,
}

pub struct Moteur {
    pub manifeste: Manifeste,
    pub tokeniseur: Tokeniseur,
    pub noyaux: Noyaux,
    pub dossier: PathBuf,
    ctx: Arc<CudaContext>,
    flux: Arc<CudaStream>,
    f: Fonctions,
    attention: NoyauTriton,
    argmax_carte: Argmax,
    couches: Vec<Couche>,
    embed: Ptr,
    norme_finale: Ptr,
    tete: Lineaire,
    cos: Ptr,
    sin: Ptr,
    rope_d: u32,
    kv: Vec<Kv>,
    t: Tampons,
    sms: u32,
    blocs: u32,
    _tenus: Vec<CudaSlice<u8>>,
    pub octets_carte: usize,
    pub empreintes: Vec<(String, String, String)>,
    /// hash de la variante Triton lancée → nombre de lancements (au journal de la porte)
    pub variantes_lancees: HashMap<String, u64>,
    /// porte au bit : sha256 des logits fp32 de chaque pas (None : pas de journal)
    pub journal_logits: Option<Vec<String>>,
}

fn envoyer(flux: &Arc<CudaStream>, tenus: &mut Vec<CudaSlice<u8>>, b: &[u8]) -> Resultat<Ptr> {
    let d = flux.clone_htod(b).map_err(|e| erreur!("{e:?}"))?;
    let p = ptr_de(&d, flux);
    tenus.push(d);
    Ok(p)
}

fn zeros(flux: &Arc<CudaStream>, tenus: &mut Vec<CudaSlice<u8>>, n: usize) -> Resultat<Ptr> {
    let z = flux.alloc_zeros::<u8>(n).map_err(|e| erreur!("{e:?}"))?;
    let p = ptr_de(&z, flux);
    tenus.push(z);
    Ok(p)
}

fn ptr_de(s: &CudaSlice<u8>, flux: &CudaStream) -> Ptr {
    let (p, _g) = s.device_ptr(flux);
    p
}

impl Moteur {
    /// `vidage` : dossier du vidage Python (tête liée int8, tables RoPE, cubins Triton).
    pub fn charger(dossier: &Path, so: &Path, vidage: &Path) -> Resultat<Self> {
        let manifeste = Manifeste::lire(dossier)?;
        let tokeniseur = Tokeniseur::charger(dossier)?;
        let ctx = CudaContext::new(0).map_err(|e| erreur!("contexte CUDA : {e:?}"))?;
        // Le moteur tient ses tampons lui-même : pas de suivi d'événements par tranche sur le chemin chaud.
        unsafe { ctx.disable_event_tracking() };
        let flux = ctx.default_stream();
        let noyaux = Noyaux::charger(&ctx, so)?;
        let fonction = |c: &str| noyaux.fonction(c);
        let f = Fonctions {
            nvfp4: fonction(cles::NVFP4_GEMV)?,
            int8: fonction(cles::INT8_GEMV_BF16)?,
            int8_f32: fonction(cles::INT8_GEMV_F32)?,
            rmsnorm: fonction(cles::RMSNORM)?,
            rope: fonction(cles::ROPE)?,
            kv: fonction(cles::KV_WRITE_INT8)?,
            swiglu: fonction(cles::SWIGLU)?,
            swiglu2: fonction(cles::SWIGLU2)?,
        };
        ctx.bind_to_thread().map_err(|e| erreur!("{e:?}"))?;
        let attention = NoyauTriton::charger(vidage, "_partiel_reduit_kernel")?;
        let argmax_carte = Argmax::charger(&ctx)?;
        let sms = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| erreur!("{e:?}"))? as u32;

        let poids = Poids::ouvrir(dossier)?;
        let mut ch = Chargeur::nouveau(flux.clone(), &manifeste, &poids);
        let spec = manifeste.model.clone();
        let couches = (0..spec.num_layers).map(|i| ch.couche(i)).collect::<Resultat<Vec<_>>>()?;
        let embed = ch.bf16("model.embed_tokens.weight")?;
        let norme_finale = ch.bf16("model.norm.weight")?;

        // tête liée int8, quantifiée au chargement par le Python (loader.py:1087-1089) : prise du vidage
        let tete_octets = std::fs::read(vidage.join("tete.safetensors")).map_err(|e| erreur!("tete.safetensors : {e}"))?;
        let st = SafeTensors::deserialize(&tete_octets).map_err(|e| erreur!("tete.safetensors : {e}"))?;
        let partie = |n: &str| st.tensor(n).map_err(|e| erreur!("tête {n} : {e}"));
        let (tq, tsc, tze) = (partie("qweight")?, partie("scales")?, partie("zeros")?);
        let (m, k) = (tq.shape()[0] as u32, tq.shape()[1] as u32);
        let group = k / tsc.shape()[1] as u32;
        let mut tenus = Vec::new();
                let tete = Lineaire::Int8 { qw: envoyer(&flux, &mut tenus, tq.data())?, scales: envoyer(&flux, &mut tenus, tsc.data())?, zeros: envoyer(&flux, &mut tenus, tze.data())?, group, m, k };

        let rope_octets = std::fs::read(vidage.join("rope.safetensors")).map_err(|e| erreur!("rope.safetensors : {e}"))?;
        let rst = SafeTensors::deserialize(&rope_octets).map_err(|e| erreur!("{e}"))?;
        let cos_t = rst.tensor("cos32").map_err(|e| erreur!("{e}"))?;
        let rope_d = *cos_t.shape().last().unwrap() as u32;
        let lignes_rope = cos_t.shape()[0] as u32;
        let cos = envoyer(&flux, &mut tenus, cos_t.data())?;
        let sin = envoyer(&flux, &mut tenus, rst.tensor("sin32").map_err(|e| erreur!("{e}"))?.data())?;

        // cache KV int8 paginé [NB, 16, HKV, D] + échelles fp16 [NB, 16, HKV] ; blocs en identité
        let (hkv, d) = (spec.num_key_value_heads as u32, spec.head_dim as u32);
        let blocs = lignes_rope.div_ceil(BLOC) + 1;
        let lignes = (blocs * BLOC * hkv) as usize;
        let mut kv = Vec::new();
        for _ in 0..spec.num_layers {
            kv.push(Kv { kc: zeros(&flux, &mut tenus, lignes * d as usize)?, vc: zeros(&flux, &mut tenus, lignes * d as usize)?, ks: zeros(&flux, &mut tenus, lignes * 2)?, vs: zeros(&flux, &mut tenus, lignes * 2)? });
        }
        let (hsz, isz) = (spec.hidden_size, spec.intermediate_size);
        let (hq, qkv_n) = (spec.num_attention_heads, (spec.num_attention_heads + 2 * spec.num_key_value_heads) * spec.head_dim);
        let parts = (hkv * TRANCHES_MAX * NREP_TUILE) as usize;
        let t = Tampons {
            x: [zeros(&flux, &mut tenus, hsz * 2)?, zeros(&flux, &mut tenus, hsz * 2)?],
            delta: zeros(&flux, &mut tenus, hsz * 2)?,
            h: zeros(&flux, &mut tenus, hsz * 2)?,
            qkv: zeros(&flux, &mut tenus, qkv_n * 2)?,
            attn: zeros(&flux, &mut tenus, hq * spec.head_dim * 2)?,
            a: zeros(&flux, &mut tenus, hsz * 2)?,
            gu: zeros(&flux, &mut tenus, 2 * isz * 2)?,
            g: zeros(&flux, &mut tenus, isz * 2)?,
            u: zeros(&flux, &mut tenus, isz * 2)?,
            act: zeros(&flux, &mut tenus, isz * 2)?,
            logits: zeros(&flux, &mut tenus, m as usize * 4)?,
            part: zeros(&flux, &mut tenus, parts * d as usize * 4)?,
            pm: zeros(&flux, &mut tenus, parts * 4)?,
            pl: zeros(&flux, &mut tenus, parts * 4)?,
            cnt: zeros(&flux, &mut tenus, hkv as usize * 4)?,
            tables: zeros(&flux, &mut tenus, blocs.next_power_of_two().max(8) as usize * 8)?,
            lens: zeros(&flux, &mut tenus, 8)?,
            pos: zeros(&flux, &mut tenus, 8)?,
            slot: zeros(&flux, &mut tenus, 8)?,
            jeton: zeros(&flux, &mut tenus, 8)?,
        };
        tenus.append(&mut ch.tenus);
        let octets_carte = tenus.iter().map(|s| s.len()).sum::<usize>();
        let empreintes = std::mem::take(&mut ch.empreintes);
        drop(ch);
        flux.synchronize().map_err(|e| erreur!("{e:?}"))?;
        Ok(Self {
            manifeste, tokeniseur, noyaux, dossier: dossier.to_path_buf(), ctx, flux, f, attention, argmax_carte, couches, embed,
            norme_finale, tete, cos, sin, rope_d, kv, t, sms, blocs, _tenus: tenus, octets_carte, empreintes,
            variantes_lancees: HashMap::new(),
            journal_logits: None,
        })
    }

    fn gemv(&self, lin: &Lineaire, x: Ptr, y: Ptr, sortie_f32: bool) -> Resultat<()> {
        let s = &self.flux;
        match *lin {
            Lineaire::Nvfp4 { qw, bscale, gscale, gscale_rows, m, k } => {
                l::nvfp4_gemv(&self.f.nvfp4, s, self.sms, qw, bscale, gscale, x, y, m, k, gscale_rows)
            }
            Lineaire::Int8 { qw, scales, zeros, group, m, k } => {
                let f = if sortie_f32 { &self.f.int8_f32 } else { &self.f.int8 };
                l::int8_gemv(f, s, self.sms, qw, scales, zeros, x, y, m, k, group)
            }
        }
    }

    fn h2d<T: cudarc::driver::DeviceRepr>(&self, dst: Ptr, v: &[T]) -> Resultat<()> {
        let octets = std::mem::size_of_val(v);
        // SAFETY : copie synchrone vers un tampon du moteur assez grand (tailles fixées au chargement).
        let r = unsafe { cudarc::driver::sys::cuMemcpyHtoD_v2(dst, v.as_ptr() as *const _, octets) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("copie vers la carte : {r:?}"));
        }
        Ok(())
    }

    fn d2d(&self, dst: Ptr, src: Ptr, octets: usize) -> Resultat<()> {
        // SAFETY : régions du moteur, sans recouvrement.
        let r = unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, octets, self.flux.cu_stream()) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("copie sur la carte : {r:?}"));
        }
        Ok(())
    }

    /// Relit les lignes KV des positions 0..longueur (blocs en identité) : (nom, octets) par couche, dans le format
    /// du vidage Python (`couche{i}.k|v|k_scale|v_scale`) — bisection du préfill.
    pub fn lire_kv(&self, longueur: u32) -> Resultat<Vec<(String, Vec<u8>, Vec<usize>)>> {
        let spec = &self.manifeste.model;
        let (hkv, d) = (spec.num_key_value_heads, spec.head_dim);
        let l = longueur as usize;
        let mut v = Vec::new();
        self.flux.synchronize().map_err(|e| erreur!("{e:?}"))?;
        for (i, c) in self.kv.iter().enumerate() {
            for (nom, src, octets, forme) in [("k", c.kc, l * hkv * d, vec![l, hkv, d]), ("v", c.vc, l * hkv * d, vec![l, hkv, d]),
                                              ("k_scale", c.ks, l * hkv * 2, vec![l, hkv]), ("v_scale", c.vs, l * hkv * 2, vec![l, hkv])] {
                let mut b = vec![0u8; octets];
                // SAFETY : régions du cache du moteur, lues après synchronisation.
                let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(b.as_mut_ptr() as *mut _, src, octets) };
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(erreur!("KV vers l'hôte : {r:?}"));
                }
                v.push((format!("couche{i}.{nom}"), b, forme));
            }
        }
        Ok(v)
    }

    /// Place les lignes KV de l'invite (positions 0..L) vidées par le Python ; blocs en identité, donc la
    /// ligne p est à l'octet p·HKV·D du cache.
    pub fn injecter_kv(&self, fichier: &Path) -> Resultat<u32> {
        let octets = std::fs::read(fichier).map_err(|e| erreur!("{} : {e}", fichier.display()))?;
        let st = SafeTensors::deserialize(&octets).map_err(|e| erreur!("{e}"))?;
        let mut longueur = 0;
        for (i, c) in self.kv.iter().enumerate() {
            for (nom, dst) in [("k", c.kc), ("v", c.vc), ("k_scale", c.ks), ("v_scale", c.vs)] {
                let t = st.tensor(&format!("couche{i}.{nom}")).map_err(|e| erreur!("KV couche {i} {nom} : {e}"))?;
                longueur = t.shape()[0] as u32;
                self.h2d(dst, t.data())?;
            }
        }
        Ok(longueur)
    }

    /// Un pas : le jeton `jeton` à la position `p` ; rend l'argmax des logits.
    pub fn pas(&mut self, jeton: u32, p: u32) -> Resultat<u32> {
        if self.journal_logits.is_some() {
            // porte au bit : logits à l'hôte (journal), ET l'argmax de la carte vérifié contre celui de l'hôte
            let logits = self.pas_logits(jeton, p)?;
            let hote = argmax(&logits).ok_or_else(|| erreur!("logits vides"))? as u32;
            let carte = self.argmax_sur_carte()?;
            if hote != carte {
                return Err(erreur!("argmax carte {carte} ≠ hôte {hote} à la position {p}"));
            }
            return Ok(hote);
        }
        self.calculer(jeton, p)?;
        self.argmax_sur_carte()
    }

    /// Argmax des logits du dernier pas sur la carte ; 4 octets reviennent (copie synchrone).
    fn argmax_sur_carte(&self) -> Resultat<u32> {
        let v = self.manifeste.model.vocab_size.min(self.tete.m() as usize) as u32;
        self.argmax_carte.lancer(&self.flux, self.t.logits, v, self.t.jeton)?;
        let mut j = [0i32; 1];
        // SAFETY : copie synchrone de l'indice (int32) écrit par le noyau, sur le flux par défaut.
        let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(j.as_mut_ptr() as *mut _, self.t.jeton, 4) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS || j[0] < 0 {
            return Err(erreur!("argmax carte : {r:?} indice {}", j[0]));
        }
        Ok(j[0] as u32)
    }

    /// Préfill v1 (scellé `revue/poste5-rust-prefill-kl-scelle-24-09.md`) : les jetons de l'invite passent un par
    /// un dans le pas de décodage prouvé au bit ; rend les logits fp32 du dernier (ceux du premier jeton généré).
    /// Pas au bit du Python (qui préfille par GEMM et attention flash) : jugé par KL.
    pub fn prefill(&mut self, ids: &[u32]) -> Resultat<Vec<f32>> {
        self.prefill_depuis(ids, 0)
    }

    /// Préfill à partir de la position `debut` (les lignes KV 0..debut sont déjà en place, injectées).
    pub fn prefill_depuis(&mut self, ids: &[u32], debut: usize) -> Resultat<Vec<f32>> {
        let mut dernier = Err(erreur!("invite vide"));
        for (p, &id) in ids.iter().enumerate().skip(debut) {
            dernier = Ok(self.pas_logits(id, p as u32).map_err(|e| erreur!("préfill position {p} : {e}"))?);
        }
        dernier
    }

    /// Un pas : logits fp32 du vocabulaire (copiés vers l'hôte).
    pub fn pas_logits(&mut self, jeton: u32, p: u32) -> Resultat<Vec<f32>> {
        self.calculer(jeton, p)?;
        let s = self.flux.clone();
        let v = self.manifeste.model.vocab_size;
        let mut logits = vec![0f32; self.tete.m() as usize];
        s.synchronize().map_err(|e| erreur!("{e:?}"))?;
        // SAFETY : tampon de logits du moteur, taille m·4 octets.
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(logits.as_mut_ptr() as *mut _, self.t.logits, logits.len() * 4)
        };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("logits vers l'hôte : {r:?}"));
        }
        logits.truncate(v.min(logits.len()));
        if let Some(j) = self.journal_logits.as_mut() {
            use sha2::{Digest, Sha256};
            let octets: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
            j.push(format!("{:x}", Sha256::digest(&octets)));
        }
        Ok(logits)
    }

    /// Les lancements d'un pas, jusqu'aux logits fp32 dans leur tampon (aucune synchronisation).
    fn calculer(&mut self, jeton: u32, p: u32) -> Resultat<()> {
        let spec = &self.manifeste.model;
        let (hsz, hq, hkv, d) = (spec.hidden_size as u32, spec.num_attention_heads as u32,
                                 spec.num_key_value_heads as u32, spec.head_dim as u32);
        let isz = spec.intermediate_size as u32;
        let eps = spec.rms_norm_eps as f32;
        if p + 1 >= self.blocs * BLOC {
            return Err(erreur!("position {p} au-delà du cache ({} jetons)", self.blocs * BLOC));
        }
        let nblk = nblk_du_pas(p);
        let (c, chunk) = tranches(nblk, 1, hkv, self.sms);
        let table: Vec<i64> = (0..nblk as i64).collect();
        self.h2d(self.t.tables, &table)?;
        self.h2d(self.t.lens, &[p as i64 + 1])?;
        self.h2d(self.t.pos, &[p as i64])?;
        self.h2d(self.t.slot, &[p as i64])?;
        self.d2d(self.t.x[0], self.embed + jeton as u64 * hsz as u64 * 2, hsz as usize * 2)?;

        let s = self.flux.clone();
        let t = &self.t;
        let (qo, ko, vo) = (0u64, (hq * d) as u64 * 2, ((hq + hkv) * d) as u64 * 2);
        let ld = (hq + 2 * hkv) as i64 * d as i64;
        let mut xi = 0usize;
        let mut ct = 1u32;
        while ct < c {
            ct *= 2;
        }
        let ct = ct.max(2);
        for (i, cou) in self.couches.iter().enumerate() {
            if i == 0 {
                l::rmsnorm(&self.f.rmsnorm, &s, t.x[xi], cou.input_norm, t.h, NUL, NUL, 1.0, hsz, eps)?;
            } else {
                l::rmsnorm(&self.f.rmsnorm, &s, t.delta, cou.input_norm, t.h, t.x[xi], t.x[1 - xi], 1.0, hsz, eps)?;
                xi = 1 - xi;
            }
            match &cou.qkv {
                Qkv::Pile(lin) => self.gemv(lin, t.h, t.qkv, false)?,
                Qkv::Separes(q, k, v) => {
                    self.gemv(q, t.h, t.qkv + qo, false)?;
                    self.gemv(k, t.h, t.qkv + ko, false)?;
                    self.gemv(v, t.h, t.qkv + vo, false)?;
                }
            }
            l::rope(&self.f.rope, &s, t.qkv + qo, t.qkv + ko, self.cos, self.sin, t.pos, cou.q_norm, cou.k_norm,
                    eps, hq, hkv, d, d, self.rope_d, ld, ld)?;
            let kv = &self.kv[i];
            l::kv_write_int8(&self.f.kv, &s, t.qkv + ko, t.qkv + vo, t.slot, kv.kc, kv.vc, kv.ks, kv.vs,
                             hkv, d, BLOC, ld, ld)?;
            let args: HashMap<&str, Arg> = HashMap::from([
                ("q_ptr", Arg::Ptr(t.qkv + qo)), ("kc_ptr", Arg::Ptr(kv.kc)), ("ks_ptr", Arg::Ptr(kv.ks)),
                ("vc_ptr", Arg::Ptr(kv.vc)), ("vs_ptr", Arg::Ptr(kv.vs)), ("tables_ptr", Arg::Ptr(t.tables)),
                ("lens_ptr", Arg::Ptr(t.lens)), ("part_ptr", Arg::Ptr(t.part)), ("pm_ptr", Arg::Ptr(t.pm)),
                ("pl_ptr", Arg::Ptr(t.pl)), ("cnt_ptr", Arg::Ptr(t.cnt)), ("out_ptr", Arg::Ptr(t.attn)),
                ("HQ", Arg::I32(hq as i32)), ("HKV", Arg::I32(hkv as i32)), ("N", Arg::I32(nblk as i32)),
                ("C", Arg::I32(c as i32)), ("chunk", Arg::I32(chunk as i32)),
                ("scale", Arg::F32(echelle_attention(d))), ("window", Arg::I32(1 << 30)),
                ("stride_qb", Arg::I32(ld as i32)), ("stride_qh", Arg::I32(d as i32)),
                ("stride_page", Arg::I32((BLOC * hkv) as i32)), ("stride_tok", Arg::I32(hkv as i32)),
                ("stride_kvh", Arg::I32(1)), ("stride_sp", Arg::I32((BLOC * hkv) as i32)),
                ("stride_st", Arg::I32(hkv as i32)), ("stride_ob", Arg::I32((hq * d) as i32)),
                ("stride_oh", Arg::I32(d as i32)),
            ]);
            let constantes: HashMap<&str, i64> = HashMap::from([
                ("NREP", (hq / hkv) as i64), ("D", d as i64), ("BN", 64), ("PAGE_C", BLOC as i64),
                ("NREP_T", NREP_TUILE as i64), ("CT", ct as i64), ("DEROULE", 1),
            ]);
            let h = self.attention.lancer(s.cu_stream(), (1, hkv, c), &args, &constantes)?.to_string();
            *self.variantes_lancees.entry(h).or_default() += 1;
            self.gemv(&cou.o, t.attn, t.a, false)?;
            l::rmsnorm(&self.f.rmsnorm, &s, t.a, cou.post_norm, t.h, t.x[xi], t.x[1 - xi], 1.0, hsz, eps)?;
            xi = 1 - xi;
            match &cou.gate_up {
                GateUp::Pile(lin) => {
                    self.gemv(lin, t.h, t.gu, false)?;
                    l::swiglu(&self.f.swiglu, &s, t.gu, t.act, isz)?;
                }
                GateUp::Separes(g, u) => {
                    self.gemv(g, t.h, t.g, false)?;
                    self.gemv(u, t.h, t.u, false)?;
                    l::swiglu2(&self.f.swiglu2, &s, t.g, t.u, t.act, isz)?;
                }
            }
            self.gemv(&cou.down, t.act, t.delta, false)?;
        }
        l::rmsnorm(&self.f.rmsnorm, &s, t.delta, self.norme_finale, t.h, t.x[xi], t.x[1 - xi], 1.0, hsz, eps)?;
        self.gemv(&self.tete, t.h, t.logits, true)?;
        Ok(())
    }

    /// Porte de l'étape 1 (option A) : KV de l'invite et premier jeton du Python, puis décodage glouton.
    /// Rend tous les jetons générés, premier compris, comme `seq.output_ids` du Python.
    pub fn decoder_injecte(&mut self, kv: &Path, premier: u32, max: usize) -> Resultat<Vec<u32>> {
        let longueur = self.injecter_kv(kv)?;
        let mut sortie = vec![premier];
        let mut jeton = premier;
        while sortie.len() < max && !self.est_fin(jeton) {
            let p = longueur + sortie.len() as u32 - 1;
            jeton = self.pas(jeton, p)?;
            sortie.push(jeton);
        }
        Ok(sortie)
    }

    /// Génération gloutonne complète : préfill v1 puis décodage ; rend les jetons générés, EOS compris.
    pub fn generer(&mut self, ids: &[u32], max: usize) -> Resultat<Vec<u32>> {
        let mut sortie = Vec::new();
        self.generer_flux(ids, max, false, |j| {
            sortie.push(j);
            true
        })?;
        Ok(sortie)
    }

    /// Génération gloutonne en flux : `rappel(jeton)` à chaque jeton (false = arrêt demandé par le client).
    /// Chemin de service : aucun logit ne revient à l'hôte (préfill sans copie, argmax sur la carte).
    /// `ignore_eos` : l'EOS ne termine pas (sémantique du banc, `banc-llamacpp-16-09.py`).
    pub fn generer_flux(&mut self, ids: &[u32], max: usize, ignore_eos: bool,
                        mut rappel: impl FnMut(u32) -> bool) -> Resultat<usize> {
        if ids.is_empty() || max == 0 {
            return Ok(0);
        }
        // Le serveur appelle depuis un fil `spawn_blocking` quelconque : sans contexte courant, la première copie rend
        // CUDA_ERROR_INVALID_CONTEXT (première prise ABBA du 24/09, bras B entier en erreur ; la porte, elle, tourne
        // sur le fil qui a chargé le moteur).
        self.ctx.bind_to_thread().map_err(|e| erreur!("contexte CUDA du fil : {e:?}"))?;
        for (p, &id) in ids.iter().enumerate() {
            self.calculer(id, p as u32).map_err(|e| erreur!("préfill position {p} : {e}"))?;
        }
        let mut jeton = self.argmax_sur_carte()?;
        let mut n = 1;
        if !rappel(jeton) {
            return Ok(n);
        }
        while n < max && (ignore_eos || !self.est_fin(jeton)) {
            jeton = self.pas(jeton, (ids.len() + n - 1) as u32)?;
            n += 1;
            if !rappel(jeton) {
                break;
            }
        }
        Ok(n)
    }

    pub fn tenseurs_carte(&self) -> usize {
        self._tenus.len()
    }

    pub fn est_fin(&self, id: u32) -> bool {
        self.manifeste.model.eos_token_id.contains(&id)
    }

    pub fn contexte(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

/// `head_dim ** -0.5` (Python, float64) passé au noyau en fp32. Cassure prévue d'avance de la porte au bit
/// (feature `cassure-echelle`, jamais par défaut) : UN ulp fp32 de plus — perturbation minimale d'un seul
/// paramètre, que la porte des ids peut laisser passer et que celle des logits doit attraper.
/// (La première cassure, bascule des tranches un pas plus tôt, était équivalente AU BIT : à la position 127 les
/// deux tranches ajoutées sont vides et pèsent exactement 0 dans la réduction — elle ne cassait rien, 24/09.)
pub fn echelle_attention(d: u32) -> f32 {
    let s = (d as f64).powf(-0.5) as f32;
    #[cfg(feature = "cassure-echelle")]
    let s = f32::from_bits(s.to_bits() + 1);
    s
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
