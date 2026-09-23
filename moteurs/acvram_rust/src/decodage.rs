//! Un pas de décodage b=1, transcrit du chemin servi `ACVRamModel.decode_fixed` → `DecoderLayer.decode_fixed_res`
//! (model.py:321, couches.py:422) et vérifié contre le relevé de la prise 1 (432 lancements/pas) :
//!
//! ```text
//! couche 0 : h = rmsnorm(x)                       couches > 0 : (x, h) = add_norm(x, delta, input_layernorm)
//!   q,k,v  = GEMV (empilés si même format, attention.py:160-164)
//!   rope_inplace(q, k, normes q/k fusionnées) ; kv_write_int8(k, v, slot) ; attention paginée (Triton)
//!   a = GEMV o
//!   (x, h2) = add_norm(x, a, post_attention_layernorm)
//!   delta = down(swiglu(gate·up))  ou  down(swiglu2(gate, up))    (attention.py:591-624)
//! fin : (x, h) = add_norm(x, delta, norm) ; logits fp32 = tête int8 ; argmax
//! ```
//!
//! Règles d'empilement (loader) : int8 de même K et même groupe → lignes concaténées (layers.py
//! `stack_int8_linears`) ; nvfp4 de même K → lignes concaténées + échelle globale PAR LIGNE
//! (`stack_nvfp4_linears`) ; formats mêlés → GEMV séparées.

use crate::lanceurs::Ptr;

/// Un linéaire quantifié sur la carte.
#[derive(Debug, Clone)]
pub enum Lineaire {
    Nvfp4 { qw: Ptr, bscale: Ptr, gscale: f32, gscale_rows: Ptr, m: u32, k: u32 },
    Int8 { qw: Ptr, scales: Ptr, zeros: Ptr, group: u32, m: u32, k: u32 },
}

impl Lineaire {
    pub fn m(&self) -> u32 {
        match self {
            Lineaire::Nvfp4 { m, .. } | Lineaire::Int8 { m, .. } => *m,
        }
    }
    pub fn k(&self) -> u32 {
        match self {
            Lineaire::Nvfp4 { k, .. } | Lineaire::Int8 { k, .. } => *k,
        }
    }
    pub fn format(&self) -> &'static str {
        match self {
            Lineaire::Nvfp4 { .. } => "nvfp4",
            Lineaire::Int8 { .. } => "int8",
        }
    }
}

/// Projections q/k/v : une GEMV empilée (sorties contiguës q|k|v) ou trois.
#[derive(Debug, Clone)]
pub enum Qkv {
    Pile(Lineaire),
    Separes(Lineaire, Lineaire, Lineaire),
}

/// gate/up : empilés (swiglu) ou séparés (swiglu2).
#[derive(Debug, Clone)]
pub enum GateUp {
    Pile(Lineaire),
    Separes(Lineaire, Lineaire),
}

#[derive(Debug, Clone)]
pub struct Couche {
    pub input_norm: Ptr,
    pub post_norm: Ptr,
    pub q_norm: Ptr,
    pub k_norm: Ptr,
    pub qkv: Qkv,
    pub o: Lineaire,
    pub gate_up: GateUp,
    pub down: Lineaire,
}

/// `bucket_blocks` (memory/kvcache.py:40) : puissance de deux ≥ n, au moins 8.
pub fn bucket_blocks(n: u32) -> u32 {
    let mut b = 8;
    while b < n {
        b <<= 1;
    }
    b
}

/// Largeur de la table de blocs vue par l'attention au pas qui écrit la position `p` (0-indexée) :
/// le godet des blocs couvrant p + 1 jetons (graphs.py:645 ; confirmé par le relevé : 2 tranches jusqu'à
/// p = 127, 4 ensuite).
pub fn nblk_du_pas(p: u32) -> u32 {
    bucket_blocks((p + 1).div_ceil(16))
}

/// `_tranches` (kernels/attn_paginee.py:246) : (C, chunk en jetons).
pub fn tranches(n_pages: u32, b: u32, hkv: u32, sms: u32) -> (u32, u32) {
    const TRANCHES_MAX: u32 = 32;
    const PAGES_PAR_TUILE: u32 = 4;
    const PAGE: u32 = 16;
    let voulu = (2 * sms).div_ceil((b * hkv).max(1)).clamp(1, TRANCHES_MAX);
    let mut ppt = PAGES_PAR_TUILE.max(n_pages.div_ceil(voulu));
    ppt = ppt.div_ceil(PAGES_PAR_TUILE) * PAGES_PAR_TUILE;
    (n_pages.div_ceil(ppt), ppt * PAGE)
}

/// Empilement décidé comme le loader : `None` = GEMV séparées.
pub fn empilable(lins: &[&Lineaire]) -> bool {
    let premier = lins[0];
    lins.iter().all(|l| match (premier, l) {
        (Lineaire::Int8 { k: k0, group: g0, .. }, Lineaire::Int8 { k, group, .. }) => k == k0 && group == g0,
        (Lineaire::Nvfp4 { k: k0, .. }, Lineaire::Nvfp4 { k, m, .. }) => k == k0 && m % 4 == 0,
        _ => false,
    })
}
