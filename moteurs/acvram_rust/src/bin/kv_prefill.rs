//! Bisection du préfill (issue nommée au scellé KL) : vide le KV écrit par le préfill Rust v1, positions de l'invite,
//! dans le format de `kv-<invite>.safetensors` du vidage Python. `kv-prefill <modèle> <so> <vidage> <sortie>`.

use std::path::PathBuf;

use acvram_rust::moteur::Moteur;
use acvram_rust::tokeniseur::Message;
use safetensors::{serialize_to_file, tensor::TensorView, Dtype};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(a.len(), 4, "usage : kv-prefill <dossier-modèle> <acvram_kernels.so> <dossier-vidage> <sortie>");
    let (dossier, so, vidage, sortie) = (PathBuf::from(&a[0]), PathBuf::from(&a[1]), PathBuf::from(&a[2]), PathBuf::from(&a[3]));
    std::fs::create_dir_all(&sortie).expect("sortie");
    let invites: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/invites.json")).expect("invites"),
    )
    .expect("json");
    let mut m = Moteur::charger(&dossier, &so, &vidage).expect("chargement");
    for inv in &invites {
        let nom = inv["nom"].as_str().unwrap();
        let messages: Vec<Message> = serde_json::from_value(inv["messages"].clone()).unwrap();
        let ids = m.tokeniseur.encoder(&m.tokeniseur.rendre(&messages, true).unwrap()).unwrap();
        // KV_LIGNE0_DE=<dossier> : la ligne 0 (et elle seule, les suivantes sont réécrites) vient d'un vidage Python —
        // pour séparer l'origine de la ligne 0 (préfill Python d'un jeton) du reste du chemin de décodage.
        match std::env::var("KV_LIGNE0_DE") {
            Ok(d) => {
                m.injecter_kv(&PathBuf::from(d).join(format!("kv-{nom}.safetensors"))).expect("injection");
                m.prefill_depuis(&ids, 1).expect("préfill");
            }
            Err(_) => {
                m.prefill(&ids).expect("préfill");
            }
        }
        let kv = m.lire_kv(ids.len() as u32).expect("KV");
        let vues: Vec<(String, TensorView)> = kv
            .iter()
            .map(|(n, b, f)| {
                let dt = if n.ends_with("scale") { Dtype::F16 } else { Dtype::I8 };
                (n.clone(), TensorView::new(dt, f.clone(), b).unwrap())
            })
            .collect();
        serialize_to_file(vues, None, &sortie.join(format!("kv-{nom}.safetensors"))).expect("écriture");
        println!("{nom} {} positions", ids.len());
    }
}
