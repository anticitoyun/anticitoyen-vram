//! Le moteur : poids sur la carte dans la disposition servie, cache KV int8 paginé, et le pas de décodage
//! b=1 dans l'ordre exact du relevé (`decodage.rs`). Option A du chef : le préfill vient du moteur Python
//! (KV et premier jeton vidés, `outils/vidage_reference.py`) ; `decoder_injecte` est la porte de l'étape 1.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
use safetensors::SafeTensors;

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
        };
        tenus.append(&mut ch.tenus);
        let octets_carte = ch.octets + tenus.iter().map(|s| s.len()).sum::<usize>();
        let empreintes = std::mem::take(&mut ch.empreintes);
        drop(ch);
        flux.synchronize().map_err(|e| erreur!("{e:?}"))?;
        Ok(Self {
            manifeste, tokeniseur, noyaux, dossier: dossier.to_path_buf(), ctx, flux, f, attention, couches, embed,
            norme_finale, tete, cos, sin, rope_d, kv, t, sms, blocs, _tenus: tenus, octets_carte, empreintes,
            variantes_lancees: HashMap::new(),
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
                ("scale", Arg::F32((d as f64).powf(-0.5) as f32)), ("window", Arg::I32(1 << 30)),
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
        let v = spec.vocab_size;
        let mut logits = vec![0f32; self.tete.m() as usize];
        s.synchronize().map_err(|e| erreur!("{e:?}"))?;
        // SAFETY : tampon de logits du moteur, taille m·4 octets.
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(logits.as_mut_ptr() as *mut _, t.logits, logits.len() * 4)
        };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(erreur!("logits vers l'hôte : {r:?}"));
        }
        argmax(&logits[..v.min(logits.len())]).map(|i| i as u32).ok_or_else(|| erreur!("logits vides"))
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

    pub fn generer(&self, _ids: &[u32], _max: usize) -> Resultat<Vec<u32>> {
        Err(erreur!("préfill Rust non écrit (option A : la porte de l'étape 1 passe par decoder_injecte)"))
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
