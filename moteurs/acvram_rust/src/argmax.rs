//! Argmax glouton sur la carte (débit, étape 1 (3)) : un bloc, rend l'indice comme `torch.argmax` (sampler.py:150) —
//! premier indice du maximum, un NaN gagne (premier NaN). Seul l'indice (4 octets) revient à l'hôte, au lieu des
//! 600 Ko de logits. Compilé par NVRTC au chargement ; son accord avec l'argmax hôte se vérifie à chaque pas de la
//! porte au bit (`porte.rs`).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use std::ffi::CString;

use cudarc::nvrtc::{result as nvrtc, sys as nvrtc_sys, Ptx};

use crate::{erreur, Resultat};

const SOURCE: &str = r#"
extern "C" __global__ void argmax_f32(const float *x, int n, int *sortie) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float m = 0.f; int im = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        // règle : NaN gagne ; sinon plus grand ; à égalité l'indice le plus petit (le premier vu par ce fil)
        if (im < 0 || (!(m != m) && ((v != v) || v > m))) { m = v; im = i; }
    }
    sv[threadIdx.x] = m; si[threadIdx.x] = im;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float a = sv[threadIdx.x], b = sv[threadIdx.x + s];
            const int ia = si[threadIdx.x], ib = si[threadIdx.x + s];
            bool prendre_b;
            if (ia < 0) prendre_b = ib >= 0;
            else if (ib < 0) prendre_b = false;
            else if (a != a) prendre_b = (b != b) && ib < ia;
            else if (b != b) prendre_b = true;
            else prendre_b = b > a || (b == a && ib < ia);
            if (prendre_b) { sv[threadIdx.x] = b; si[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) sortie[0] = si[0];
}
"#;

/// SASS sm_120 et non PTX : le PTX de NVRTC 13.4 est refusé par un pilote plus ancien (595.91, CUDA 13.2 :
/// CUDA_ERROR_UNSUPPORTED_PTX_VERSION, première prise release du 24/09) ; un cubin d'une version mineure plus
/// récente se charge dans la même version majeure.
fn cubin_sm120() -> Resultat<Vec<u8>> {
    let src = CString::new(SOURCE).map_err(|e| erreur!("{e}"))?;
    let prog = nvrtc::create_program(&src, None).map_err(|e| erreur!("NVRTC argmax : {e:?}"))?;
    // SAFETY : `prog` créé ci-dessus, détruit une seule fois à la fin, quelle que soit l'issue.
    unsafe {
        let res = nvrtc::compile_program(prog, &["--gpu-architecture=sm_120"])
            .map_err(|e| erreur!("NVRTC argmax : {e:?} {}", String::from_utf8_lossy(
                &nvrtc::get_program_log(prog).unwrap_or_default().iter().map(|&c| c as u8).collect::<Vec<_>>())))
            .and_then(|_| {
                let mut n = 0usize;
                nvrtc_sys::nvrtcGetCUBINSize(prog, &mut n).result().map_err(|e| erreur!("{e:?}"))?;
                let mut cubin = vec![0u8; n];
                nvrtc_sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()).result().map_err(|e| erreur!("{e:?}"))?;
                Ok(cubin)
            });
        let _ = nvrtc::destroy_program(prog);
        res
    }
}

pub struct Argmax {
    f: CudaFunction,
}

impl Argmax {
    pub fn charger(ctx: &Arc<CudaContext>) -> Resultat<Self> {
        let m = ctx.load_module(Ptx::from_binary(cubin_sm120()?)).map_err(|e| erreur!("module argmax : {e:?}"))?;
        Ok(Self { f: m.load_function("argmax_f32").map_err(|e| erreur!("{e:?}"))? })
    }

    /// Lance l'argmax de `x` (n flottants) ; l'indice est écrit en `sortie` (int32 sur la carte).
    pub fn lancer(&self, s: &CudaStream, x: u64, n: u32, sortie: u64) -> Resultat<()> {
        let ni = n as i32;
        let mut l = s.launch_builder(&self.f);
        l.arg(&x).arg(&ni).arg(&sortie);
        // SAFETY : noyau ci-dessus, 1 bloc de 1 024 fils, mémoire partagée statique.
        unsafe { l.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }) }
            .map(|_| ())
            .map_err(|e| erreur!("lancement argmax : {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    /// À sec (NVRTC seul, aucune carte) : le module est un ELF (SASS), jamais un texte PTX que le pilote devrait JIT.
    #[test]
    fn argmax_compile_en_sass() {
        let c = super::cubin_sm120().expect("NVRTC");
        assert_eq!(&c[..4], b"\x7fELF");
    }
}
