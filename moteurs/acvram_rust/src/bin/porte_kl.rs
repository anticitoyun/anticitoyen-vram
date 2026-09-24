//! Porte (2) de l'étape 1 : préfill Rust jugé par KL (scellé `revue/poste5-rust-prefill-kl-scelle-24-09.md`, écrit
//! avant). Par invite : préfill Rust (jetons un par un dans le pas prouvé au bit), KL du 1er jeton contre les logits
//! Python complets ; puis forçage par l'enseignant sur la sortie Python, KL de chaque pas. KL(P_python ‖ Q_rust),
//! nats, f64. Seuils : 1er jeton argmax égal et KL ≤ 1e-3 ; forçage KL moyenne ≤ 1e-3, max ≤ 1e-2 par invite ;
//! argmax égaux ≥ 99 % des pas sur l'ensemble.
//!
//! `porte-kl <dossier-modèle> <acvram_kernels.so> <dossier-vidage>` — code 0 si tout tient, 1 sinon.

use std::path::PathBuf;

use acvram_rust::moteur::{argmax, Moteur};
use acvram_rust::tokeniseur::Message;
use safetensors::SafeTensors;

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
    let s: f64 = x.iter().map(|&v| ((v as f64) - m).exp()).sum();
    let l = m + s.ln();
    x.iter().map(|&v| v as f64 - l).collect()
}

/// KL(P ‖ Q) en nats, P = référence Python, Q = Rust.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 3 {
        eprintln!("usage : porte-kl <dossier-modèle> <acvram_kernels.so> <dossier-vidage>");
        std::process::exit(64);
    }
    let (dossier, so, vidage) = (PathBuf::from(&a[0]), PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vidage.join("reference.json")).expect("reference.json")).expect("json");
    let invites: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/invites.json")).expect("invites"),
    )
    .expect("json");
    let mut m = Moteur::charger(&dossier, &so, &vidage).unwrap_or_else(|e| {
        eprintln!("chargement : {e}");
        std::process::exit(2)
    });
    let v = m.manifeste.model.vocab_size;
    let (mut tout_tient, mut pas_total, mut pas_egaux) = (true, 0usize, 0usize);
    for (inv, r) in invites.iter().zip(reference["invites"].as_array().expect("invites")) {
        let nom = r["nom"].as_str().unwrap();
        assert_eq!(inv["nom"].as_str(), Some(nom), "ordre des invites");
        let messages: Vec<Message> = serde_json::from_value(inv["messages"].clone()).unwrap();
        let ids = m.tokeniseur.encoder(&m.tokeniseur.rendre(&messages, true).unwrap()).unwrap();
        assert_eq!(ids.len() as u64, r["longueur_invite"].as_u64().unwrap(), "{nom} : longueur d'invite");
        let octets = std::fs::read(vidage.join(format!("logits-{nom}.safetensors"))).expect("logits Python (VIDAGE_LOGITS=1)");
        let st = SafeTensors::deserialize(&octets).expect("safetensors");
        let t = st.tensor("logits").expect("logits");
        let py: Vec<f32> = t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let n_pas = t.shape()[0];
        let ligne = |i: usize| &py[i * v..(i + 1) * v];
        let sortie: Vec<u32> = r["sortie"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();

        let q0 = m.prefill(&ids).expect("préfill");
        let (kl0, egal0) = (kl(ligne(0), &q0), argmax(ligne(0)) == argmax(&q0));
        let mut kls = Vec::new();
        let mut egaux = 0;
        for k in 0..sortie.len().saturating_sub(1).min(n_pas - 1) {
            let q = m.pas_logits(sortie[k], (ids.len() + k) as u32).expect("pas");
            kls.push(kl(ligne(k + 1), &q));
            egaux += (argmax(ligne(k + 1)) == argmax(&q)) as usize;
        }
        let moy = kls.iter().sum::<f64>() / kls.len().max(1) as f64;
        let max = kls.iter().cloned().fold(0f64, f64::max);
        let tient = egal0 && kl0 <= 1e-3 && moy <= 1e-3 && max <= 1e-2;
        tout_tient &= tient;
        pas_total += kls.len();
        pas_egaux += egaux;
        println!("{nom} {} : 1er jeton KL {kl0:.3e} argmax {} ; forçage {} pas KL moy {moy:.3e} max {max:.3e} argmax {egaux}/{}",
                 if tient { "TENU" } else { "FAUX" }, if egal0 { "égal" } else { "DIFFÉRENT" }, kls.len(), kls.len());
    }
    let taux = pas_egaux as f64 / pas_total.max(1) as f64;
    let ok = tout_tient && taux >= 0.99;
    println!("argmax égaux {pas_egaux}/{pas_total} ({:.2} %)", 100.0 * taux);
    println!("PORTE KL PRÉFILL {}", if ok { "TENUE" } else { "FAUSSE" });
    std::process::exit(if ok { 0 } else { 1 });
}
