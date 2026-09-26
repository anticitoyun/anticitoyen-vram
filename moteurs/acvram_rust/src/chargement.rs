//! Poids vers la carte, dans la disposition que le moteur Python sert : empilements q·k·v et gate·up
//! faits ici, depuis les octets du fichier (concaténation de lignes, jamais de réarrondi), puis envoyés une
//! seule fois. Les empreintes des tenseurs résultants se comparent à `poids.json` du vidage Python.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::decodage::{empilable, Couche, GateUp, Lineaire, Qkv};
use crate::manifeste::Manifeste;
use crate::poids::Poids;
use crate::{erreur, Resultat};

/// Un linéaire lu dans le fichier, avant envoi.
struct LinHote<'a> {
    format: String,
    m: u32,
    k: u32,
    group: u32,
    gscale: f32,
    /// suffixe physique (`qweight`, `scales`, …) → octets
    parties: HashMap<String, &'a [u8]>,
}

impl LinHote<'_> {
    fn comme_lineaire(&self) -> Lineaire {
        // pour `empilable` seulement : pas de pointeurs
        match self.format.as_str() {
            "nvfp4" => Lineaire::Nvfp4 { qw: 0, bscale: 0, gscale: self.gscale, gscale_rows: 0, m: self.m, k: self.k },
            _ => Lineaire::Int8 { qw: 0, scales: 0, zeros: 0, group: self.group, m: self.m, k: self.k },
        }
    }
}

pub struct Chargeur<'a> {
    pub flux: Arc<CudaStream>,
    manifeste: &'a Manifeste,
    poids: &'a Poids,
    /// allocations tenues en vie pour la durée du moteur
    pub tenus: Vec<CudaSlice<u8>>,
    pub octets: usize,
    /// nom (logique ou empilé) → sha256 des octets envoyés, pour la comparaison à `poids.json`
    pub empreintes: Vec<(String, String, String)>,
}

impl<'a> Chargeur<'a> {
    pub fn nouveau(flux: Arc<CudaStream>, manifeste: &'a Manifeste, poids: &'a Poids) -> Self {
        Self { flux, manifeste, poids, tenus: Vec::new(), octets: 0, empreintes: Vec::new() }
    }

    fn envoyer(&mut self, octets: &[u8]) -> Resultat<u64> {
        let d = self.flux.clone_htod(octets).map_err(|e| erreur!("envoi de {} o : {e:?}", octets.len()))?;
        let p = {
            let (p, _garde) = d.device_ptr(&self.flux);
            p
        };
        self.octets += octets.len();
        self.tenus.push(d);
        Ok(p)
    }

    fn noter(&mut self, nom: &str, partie: &str, octets: &[u8]) {
        use sha2::{Digest, Sha256};
        self.empreintes.push((nom.to_string(), partie.to_string(), format!("{:x}", Sha256::digest(octets))));
    }

    /// Un tenseur bf16 simple (normes, plongement).
    pub fn bf16(&mut self, nom: &str) -> Resultat<u64> {
        let e = self.manifeste.tenseur(nom)?;
        if e.format != "bf16" || e.keys.len() != 1 {
            return Err(erreur!("{nom} : bf16 simple attendu, trouvé {} ({} clés)", e.format, e.keys.len()));
        }
        let v = self.poids.vue(&e.keys[0])?;
        self.noter(nom, "poids", v.octets);
        self.envoyer(v.octets)
    }

    fn lire(&self, nom: &str) -> Resultat<LinHote<'a>> {
        let e = self.manifeste.tenseur(nom)?;
        let (m, k) = (e.shape[0] as u32, e.shape[1] as u32);
        let mut parties = HashMap::new();
        let mut gscale = 1f32;
        for cle in &e.keys {
            let suffixe = cle.rsplit('.').next().unwrap_or("").to_string();
            let v = self.poids.vue(cle)?;
            if suffixe == "global_scale" {
                gscale = f32::from_le_bytes(v.octets[..4].try_into().map_err(|_| erreur!("{cle} : f32 attendu"))?);
            }
            parties.insert(suffixe, v.octets);
        }
        let group = match e.format.as_str() {
            "int8" => {
                let sc = self.poids.vue(&format!("{nom}.scales"))?;
                k / sc.shape[1] as u32
            }
            "nvfp4" => 16,
            f => return Err(erreur!("{nom} : format {f} non porté à l'étape 1")),
        };
        Ok(LinHote { format: e.format.clone(), m, k, group, gscale, parties })
    }

    /// Envoie un groupe de linéaires de même entrée : empilé si le loader l'empilerait, sinon séparés.
    fn groupe(&mut self, noms: &[String], nom_pile: &str) -> Resultat<Vec<Lineaire>> {
        let hotes: Vec<LinHote> = noms.iter().map(|n| self.lire(n)).collect::<Resultat<_>>()?;
        let formes: Vec<Lineaire> = hotes.iter().map(|h| h.comme_lineaire()).collect();
        let refs: Vec<&Lineaire> = formes.iter().collect();
        if hotes.len() > 1 && empilable(&refs) {
            return Ok(vec![self.envoyer_lin(nom_pile, &hotes)?]);
        }
        let mut v = Vec::new();
        for (n, h) in noms.iter().zip(hotes.iter()) {
            v.push(self.envoyer_lin(n, std::slice::from_ref(h))?);
        }
        Ok(v)
    }

    /// Concatène les lignes de `hotes` (un seul = pas d'empilement) et envoie.
    fn envoyer_lin(&mut self, nom: &str, hotes: &[LinHote]) -> Resultat<Lineaire> {
        let concat = |partie: &str| -> Resultat<Vec<u8>> {
            let mut v = Vec::new();
            for h in hotes {
                v.extend_from_slice(h.parties.get(partie).ok_or_else(|| erreur!("{nom} : partie {partie} absente"))?);
            }
            Ok(v)
        };
        let m: u32 = hotes.iter().map(|h| h.m).sum();
        let k = hotes[0].k;
        match hotes[0].format.as_str() {
            "nvfp4" => {
                let (qw, bs) = (concat("qweight")?, concat("block_scale")?);
                self.noter(nom, "qweight", &qw);
                self.noter(nom, "block_scale", &bs);
                let gscale_rows = if hotes.len() > 1 {
                    // stack_nvfp4_linears : une échelle globale fp32 par ligne de sortie
                    let lignes: Vec<u8> = hotes
                        .iter()
                        .flat_map(|h| std::iter::repeat_n(h.gscale, h.m as usize))
                        .flat_map(f32::to_le_bytes)
                        .collect();
                    self.noter(nom, "global_scale_rows", &lignes);
                    self.envoyer(&lignes)?
                } else {
                    0
                };
                Ok(Lineaire::Nvfp4 {
                    qw: self.envoyer(&qw)?,
                    bscale: self.envoyer(&bs)?,
                    gscale: hotes[0].gscale,
                    gscale_rows,
                    m,
                    k,
                })
            }
            _ => {
                let (qw, sc, ze) = (concat("qweight")?, concat("scales")?, concat("zeros")?);
                self.noter(nom, "qweight", &qw);
                self.noter(nom, "scales", &sc);
                self.noter(nom, "zeros", &ze);
                Ok(Lineaire::Int8 {
                    qw: self.envoyer(&qw)?,
                    scales: self.envoyer(&sc)?,
                    zeros: self.envoyer(&ze)?,
                    group: hotes[0].group,
                    m,
                    k,
                })
            }
        }
    }

    pub fn couche(&mut self, i: usize) -> Resultat<Couche> {
        let a = format!("model.layers.{i}.self_attn");
        let p = format!("model.layers.{i}.mlp");
        let w = |s: &str| format!("{s}.weight");
        let qkv_noms = [w(&format!("{a}.q_proj")), w(&format!("{a}.k_proj")), w(&format!("{a}.v_proj"))];
        let mut qkv = self.groupe(&qkv_noms, &format!("model.layers.{i}.self_attn.qkv_proj"))?;
        let qkv = if qkv.len() == 1 {
            Qkv::Pile(qkv.remove(0))
        } else {
            let (q, k, v) = (qkv.remove(0), qkv.remove(0), qkv.remove(0));
            Qkv::Separes(q, k, v)
        };
        let gu_noms = [w(&format!("{p}.gate_proj")), w(&format!("{p}.up_proj"))];
        let mut gu = self.groupe(&gu_noms, &format!("model.layers.{i}.mlp.gate_up"))?;
        let gate_up = if gu.len() == 1 { GateUp::Pile(gu.remove(0)) } else { GateUp::Separes(gu.remove(0), gu.remove(0)) };
        let o = self.groupe(&[w(&format!("{a}.o_proj"))], "")?.remove(0);
        let down = self.groupe(&[w(&format!("{p}.down_proj"))], "")?.remove(0);
        Ok(Couche {
            input_norm: self.bf16(&w(&format!("model.layers.{i}.input_layernorm")))?,
            post_norm: self.bf16(&w(&format!("model.layers.{i}.post_attention_layernorm")))?,
            q_norm: self.bf16(&w(&format!("{a}.q_norm")))?,
            k_norm: self.bf16(&w(&format!("{a}.k_norm")))?,
            qkv,
            o,
            gate_up,
            down,
        })
    }
}
