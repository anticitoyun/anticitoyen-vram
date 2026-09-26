//! Lanceurs de NOS noyaux de décodage, recopiés des lanceurs hôtes de `acvram/kernels/acvram_kernels.cu`
//! (grille, bloc, mémoire partagée, ordre et valeur des arguments, y compris les paramètres par défaut du
//! noyau). Toute divergence de forme de lancement change le résultat ou le rend indéfini : chaque fonction
//! cite sa source, et ce qui sort du régime relevé (b=1, splits = 1) est un REFUS, jamais un repli.
//!
//! Les pointeurs sont des `u64` bruts (adresses de périphérique) : le moteur tient ses tampons lui-même et
//! n'utilise pas le suivi d'événements de cudarc sur le chemin chaud.

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::{erreur, Resultat};

pub type Ptr = u64;
pub const NUL: Ptr = 0;

// acvram_kernels.cu:60-62
const WARP: u32 = 32;
const ROWS_PER_BLOCK: u32 = 4;
const WEIGHTS_PER_LOAD: u32 = 16;

/// acvram_kernels.cu:1226 `threads_for`
pub fn threads_for(k: u32) -> u32 {
    let nloads = k / WEIGHTS_PER_LOAD;
    let mut t = 256;
    while t > 64 && t > nloads {
        t >>= 1;
    }
    t
}

/// acvram_kernels.cu:1236 `threads_for_pairs`
pub fn threads_for_pairs(k: u32) -> u32 {
    let npairs = k / (2 * WEIGHTS_PER_LOAD);
    let mut t = 256;
    while t > 64 && t > npairs {
        t >>= 1;
    }
    t
}

/// acvram_kernels.cu:1244 `splits_for` (`sms` = multiProcessorCount)
pub fn splits_for(m: u32, k: u32, sms: u32) -> u32 {
    let row_blocks = m.div_ceil(ROWS_PER_BLOCK);
    let want = sms * 2;
    if row_blocks >= want {
        return 1;
    }
    let nloads = k / WEIGHTS_PER_LOAD;
    let splits = want.div_ceil(row_blocks);
    splits.min((nloads / 256).max(1)).max(1)
}

fn cfg(grid: (u32, u32, u32), bloc: u32, shm: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: grid, block_dim: (bloc, 1, 1), shared_mem_bytes: shm }
}

fn lancer(r: Result<Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>, cudarc::driver::DriverError>, quoi: &str) -> Resultat<()> {
    r.map(|_| ()).map_err(|e| erreur!("lancement {quoi} : {e:?}"))
}

/// `nvfp4_gemv` (acvram_kernels.cu:1316-1368), chemin bf16 → bf16, N = 1 jeton.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemv(f: &CudaFunction, s: &CudaStream, sms: u32, qw: Ptr, bscale: Ptr, gscale: f32,
                  x: Ptr, y: Ptr, m: u32, k: u32, gscale_rows: Ptr) -> Resultat<()> {
    if k % 16 != 0 {
        return Err(erreur!("nvfp4_gemv : K = {k} non divisible par 16"));
    }
    if splits_for(m, k, sms) != 1 {
        return Err(erreur!("nvfp4_gemv : M = {m}, K = {k} demande un découpage de K (chemin fp32 non porté)"));
    }
    let th = threads_for_pairs(k);
    let nwarps = th.div_ceil(WARP);
    let (mi, ki, n, splits) = (m as i32, k as i32, 1i32, 1i32);
    let mut l = s.launch_builder(f);
    l.arg(&qw).arg(&bscale).arg(&gscale).arg(&x).arg(&y).arg(&mi).arg(&ki).arg(&n).arg(&splits).arg(&gscale_rows);
    // SAFETY : arguments dans l'ordre et les types de la signature `nvfp4_gemv_kernel<4,1,bf16,bf16>`.
    lancer(unsafe { l.launch(cfg((m.div_ceil(ROWS_PER_BLOCK), 1, 1), th, ROWS_PER_BLOCK * nwarps * 4)) }, "nvfp4_gemv")
}

/// `int8_gemv` (acvram_kernels.cu:1630-1725), N = 1, sortie bf16 ou fp32 (`sortie_fp32`, tête), sans norme
/// fusionnée : les paramètres par défaut du noyau (res, nw, xout = nul, eps 0, mult 1) sont passés tels quels.
#[allow(clippy::too_many_arguments)]
pub fn int8_gemv(f: &CudaFunction, s: &CudaStream, sms: u32, qw: Ptr, scales: Ptr, zeros: Ptr,
                 x: Ptr, y: Ptr, m: u32, k: u32, group: u32) -> Resultat<()> {
    if k % group != 0 || group % 16 != 0 {
        return Err(erreur!("int8_gemv : K = {k}, groupe = {group} invalides"));
    }
    if splits_for(m, k, sms) != 1 {
        return Err(erreur!("int8_gemv : M = {m}, K = {k} demande un découpage de K (chemin fp32 non porté)"));
    }
    let th = threads_for(k);
    let nwarps = th.div_ceil(WARP);
    let (mi, ki, n, gi, splits) = (m as i32, k as i32, 1i32, group as i32, 1i32);
    let (eps, mult) = (0f32, 1f32);
    let mut l = s.launch_builder(f);
    l.arg(&qw).arg(&scales).arg(&zeros).arg(&x).arg(&y).arg(&mi).arg(&ki).arg(&n).arg(&gi).arg(&splits)
        .arg(&NUL).arg(&NUL).arg(&NUL).arg(&eps).arg(&mult);
    // SAFETY : signature `int8_gemv_kernel<4,1,bf16,YT>` complète, défauts compris (acvram_kernels.cu:569-580).
    lancer(unsafe { l.launch(cfg((m.div_ceil(ROWS_PER_BLOCK), 1, 1), th, ROWS_PER_BLOCK * nwarps * 4)) }, "int8_gemv")
}

/// `rmsnorm_bf16` (acvram_kernels.cu:7318-7352), une ligne. `res` nul : y = norme(x) ; sinon
/// xn = bf16(res + mult·x) puis y = norme(xn).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(f: &CudaFunction, s: &CudaStream, x: Ptr, w: Ptr, y: Ptr, res: Ptr, xn: Ptr,
               mult: f32, h: u32, eps: f32) -> Resultat<()> {
    let th = if h >= 2048 { 1024 } else if h >= 1024 { 512 } else { 256 };
    let mult = if res == NUL { 1f32 } else { mult };
    let hi = h as i32;
    let mut l = s.launch_builder(f);
    l.arg(&x).arg(&w).arg(&y).arg(&res).arg(&xn).arg(&mult).arg(&hi).arg(&eps);
    // SAFETY : signature `rmsnorm_bf16_kernel(x, w, y, res, xn, mult, H, eps)`.
    lancer(unsafe { l.launch(cfg((1, 1, 1), th, 0)) }, "rmsnorm")
}

/// `rope_inplace_pos` (acvram_kernels.cu:4140-4175), un jeton, normes par tête q/k fusionnées.
#[allow(clippy::too_many_arguments)]
pub fn rope(f: &CudaFunction, s: &CudaStream, q: Ptr, k: Ptr, cos: Ptr, sin: Ptr, pos: Ptr,
            wq: Ptr, wk: Ptr, eps: f32, hq: u32, hk: u32, dq: u32, dk: u32, d: u32,
            ldq: i64, ldk: i64) -> Resultat<()> {
    let th = 256u32.min(dq.max(dk).div_ceil(32) * 32);
    let (hqi, hki, dqi, dki, di) = (hq as i32, hk as i32, dq as i32, dk as i32, d as i32);
    let mut l = s.launch_builder(f);
    l.arg(&q).arg(&k).arg(&cos).arg(&sin).arg(&pos).arg(&wq).arg(&wk).arg(&eps)
        .arg(&hqi).arg(&hki).arg(&dqi).arg(&dki).arg(&di).arg(&ldq).arg(&ldk);
    // SAFETY : signature `rope_inplace_kernel` (acvram_kernels.cu:4076-4085).
    lancer(unsafe { l.launch(cfg((1, hq + hk, 1), th, 0)) }, "rope")
}

/// `kv_write_int8` (acvram_kernels.cu:4232-4262), un jeton ; `sk`/`sv` = pas entre jetons (éléments).
#[allow(clippy::too_many_arguments)]
pub fn kv_write_int8(f: &CudaFunction, s: &CudaStream, k: Ptr, v: Ptr, slots: Ptr, kc: Ptr, vc: Ptr,
                     ks: Ptr, vs: Ptr, h: u32, d: u32, bs: u32, sk: i64, sv: i64) -> Resultat<()> {
    let th = 256u32.min(d.div_ceil(32) * 32);
    let (hi, di, bsi) = (h as i32, d as i32, bs as i32);
    let mut l = s.launch_builder(f);
    l.arg(&k).arg(&v).arg(&slots).arg(&kc).arg(&vc).arg(&ks).arg(&vs).arg(&hi).arg(&di).arg(&bsi).arg(&sk).arg(&sv);
    // SAFETY : signature `kv_write_int8_kernel` (acvram_kernels.cu:4190-4194).
    lancer(unsafe { l.launch(cfg((1, h, 1), th, 0)) }, "kv_write_int8")
}

/// `swiglu_bf16` (acvram_kernels.cu:7774-7790) : gate·up empilés [2I] → [I].
pub fn swiglu(f: &CudaFunction, s: &CudaStream, gu: Ptr, y: Ptr, i: u32) -> Resultat<()> {
    let n = i as i64;
    let ii = i as i32;
    let mut l = s.launch_builder(f);
    l.arg(&gu).arg(&y).arg(&ii).arg(&n);
    // SAFETY : signature `swiglu_bf16_kernel(gu, y, I, n)`.
    lancer(unsafe { l.launch(cfg((i.div_ceil(256), 1, 1), 256, 0)) }, "swiglu")
}

/// `swiglu2_bf16` (acvram_kernels.cu:7754-7771) : gate et up séparés.
pub fn swiglu2(f: &CudaFunction, s: &CudaStream, g: Ptr, u: Ptr, y: Ptr, i: u32) -> Resultat<()> {
    let n = i as i64;
    let mut l = s.launch_builder(f);
    l.arg(&g).arg(&u).arg(&y).arg(&n);
    // SAFETY : signature `swiglu2_bf16_kernel(g, u, y, n)`.
    lancer(unsafe { l.launch(cfg((i.div_ceil(256), 1, 1), 256, 0)) }, "swiglu2")
}

/// Clés stables des noyaux ci-dessus (relevées dans le fatbin servi, cf. `noyaux::cle_stable`).
pub mod cles {
    pub const NVFP4_GEMV: &str = "_ZN@17nvfp4_gemv_kernelILi4ELi1E13__nv_bfloat16S1_EEvPKhS3_fPKT1_PT2_iiiiPKf";
    pub const INT8_GEMV_BF16: &str = "_ZN@16int8_gemv_kernelILi4ELi1E13__nv_bfloat16S1_EEvPKhPK6__halfS3_PKT1_PT2_iiiiiPKS1_SD_PS1_ff";
    pub const INT8_GEMV_F32: &str = "_ZN@16int8_gemv_kernelILi4ELi1E13__nv_bfloat16fEEvPKhPK6__halfS3_PKT1_PT2_iiiiiPKS1_SD_PS1_ff";
    pub const RMSNORM: &str = "_Z19rmsnorm_bf16_kernelPK13__nv_bfloat16S1_PS_S1_S2_fif";
    pub const ROPE: &str = "_Z19rope_inplace_kernelP13__nv_bfloat16S0_PKfS2_PKlPKS_S6_fiiiiill";
    pub const KV_WRITE_INT8: &str = "_Z20kv_write_int8_kernelPK13__nv_bfloat16S1_PKlPaS4_P6__halfS6_iiill";
    pub const SWIGLU: &str = "_Z18swiglu_bf16_kernelPK13__nv_bfloat16PS_il";
    pub const SWIGLU2: &str = "_Z19swiglu2_bf16_kernelPK13__nv_bfloat16S1_PS_l";
}
