//! Porte de l'étape 1, option A (scellée : sha des ids 5/5) : pour chaque invite du vidage Python, KV après
//! préfill et premier jeton injectés, décodage glouton Rust, sha256 des ids comparé à la référence.
//! Imprime aussi les empreintes des poids envoyés contre `poids.json` et les variantes Triton lancées.
//!
//! `porte <dossier-modèle> <acvram_kernels.so> <dossier-vidage>` — code 0 si 5/5, 1 sinon.

use std::collections::HashMap;
use std::path::PathBuf;

use acvram_rust::moteur::Moteur;
use sha2::{Digest, Sha256};

fn sha_ids(ids: &[u32]) -> String {
    // json.dumps(list) du Python : « [a, b, c] »
    let texte = format!("[{}]", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
    format!("{:x}", Sha256::digest(texte.as_bytes()))
}

fn normaliser(n: &str) -> String {
    n.trim_start_matches("model.").trim_end_matches(".weight").to_string()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 3 {
        eprintln!("usage : porte <dossier-modèle> <acvram_kernels.so> <dossier-vidage>");
        std::process::exit(64);
    }
    let (dossier, so, vidage) = (PathBuf::from(&a[0]), PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vidage.join("reference.json")).expect("reference.json")).expect("json");
    let mut m = Moteur::charger(&dossier, &so, &vidage).unwrap_or_else(|e| {
        eprintln!("chargement : {e}");
        std::process::exit(2)
    });
    println!("fatbin_sha256 {} noyaux {} octets_carte {}", m.noyaux.sha256_fatbin, m.noyaux.nombre(), m.octets_carte);

    // empreintes des poids contre le Python
    let poids: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vidage.join("poids.json")).expect("poids.json")).expect("json");
    let mut py: HashMap<(String, String), String> = HashMap::new();
    for (nom, e) in poids["quantlinear"].as_object().expect("quantlinear") {
        for (partie, t) in e["tenseurs"].as_object().expect("tenseurs") {
            py.insert((normaliser(nom), partie.clone()), t["sha256"].as_str().unwrap_or_default().to_string());
        }
    }
    for (nom, e) in poids["empreintes"].as_object().expect("empreintes") {
        py.insert((normaliser(nom), "poids".into()), e["sha256"].as_str().unwrap_or_default().to_string());
    }
    let (mut egaux, mut differents, mut absents) = (0, Vec::new(), 0);
    for (nom, partie, sha) in &m.empreintes {
        match py.get(&(normaliser(nom), partie.clone())) {
            Some(s) if s == sha => egaux += 1,
            Some(_) => differents.push(format!("{nom}.{partie}")),
            None => absents += 1,
        }
    }
    println!("empreintes : {egaux} égales, {} différentes, {absents} sans pendant Python", differents.len());
    for d in differents.iter().take(10) {
        println!("  DIFFÉRENTE {d}");
    }

    let mut tenues = 0;
    let mut bits_tenus = 0;
    let invites = reference["invites"].as_array().expect("invites");
    for inv in invites {
        let nom = inv["nom"].as_str().unwrap();
        let premier = inv["premier_jeton"].as_u64().unwrap() as u32;
        let attendu: Vec<u32> = inv["sortie"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        m.journal_logits = Some(Vec::new());
        let t0 = std::time::Instant::now();
        let sortie = match m.decoder_injecte(&vidage.join(format!("kv-{nom}.safetensors")), premier, 128) {
            Ok(s) => s,
            Err(e) => {
                println!("{nom} ERREUR {e}");
                continue;
            }
        };
        let dt = t0.elapsed().as_secs_f64();
        let (sha, sha_ref) = (sha_ids(&sortie), inv["sha256_sortie"].as_str().unwrap().to_string());
        let ok = sha == sha_ref;
        tenues += ok as usize;
        let div = sortie.iter().zip(&attendu).position(|(a, b)| a != b);
        println!("{nom} {} jetons {}/{} sha {} {} (1re divergence : {:?}) {:.2} s",
                 if ok { "TENU" } else { "FAUX" }, sortie.len(), attendu.len(), &sha[..16], &sha_ref[..16], div, dt);
        // porte au bit : logits du pas i (Rust) contre l'entrée i + 1 du journal Python (l'entrée 0 = préfill)
        let rust = m.journal_logits.take().unwrap_or_default();
        let py: Vec<&str> = inv["logits"].as_array().map(|a| a.iter().filter_map(|x| x["sha256"].as_str()).collect()).unwrap_or_default();
        let py_types: Vec<&str> = inv["logits"].as_array().map(|a| a.iter().filter_map(|x| x["dtype"].as_str()).collect()).unwrap_or_default();
        let comparables = rust.len().min(py.len().saturating_sub(1));
        let bit = (0..comparables).position(|i| rust[i] != py[i + 1]);
        let tenu_bit = !py.is_empty() && comparables == rust.len() && bit.is_none();
        bits_tenus += tenu_bit as usize;
        println!("{nom} LOGITS {} : {comparables} pas comparés, 1er pas différent : {:?} (types Python {:?})",
                 if tenu_bit { "AU BIT" } else { "FAUX" }, bit, py_types.first());
    }
    println!("variantes Triton lancées : {:?}", m.variantes_lancees);
    println!("PORTE {tenues}/{}", invites.len());
    println!("PORTE AU BIT (logits) {bits_tenus}/{}", invites.len());
    std::process::exit(if tenues == invites.len() { 0 } else { 1 });
}
