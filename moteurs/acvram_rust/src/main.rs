//! `acvram-rust serve <dossier-modèle> --noyaux <acvram_kernels.so> --vidage <dossier> [--port 8110] [--served-name nom]`

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use acvram_rust::moteur::Moteur;
use acvram_rust::serveur::{routes, Etat};

fn usage() -> ! {
    eprintln!("usage : acvram-rust serve <dossier-modèle> --noyaux <acvram_kernels.so> --vidage <dossier> [--port N] [--served-name nom]");
    std::process::exit(64)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("serve") || args.len() < 2 {
        usage();
    }
    let dossier = PathBuf::from(&args[1]);
    let (mut so, mut vidage, mut port, mut nom) = (None, None, 8110u16, None);
    let mut i = 2;
    while i < args.len() {
        let valeur = args.get(i + 1).cloned().unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--noyaux" => so = Some(PathBuf::from(valeur)),
            "--vidage" => vidage = Some(PathBuf::from(valeur)),
            "--port" => port = valeur.parse().unwrap_or_else(|_| usage()),
            "--served-name" => nom = Some(valeur),
            _ => usage(),
        }
        i += 2;
    }
    let so = so.unwrap_or_else(|| usage());
    let vidage = vidage.unwrap_or_else(|| usage());
    let moteur = match Moteur::charger(&dossier, &so, &vidage) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("acvram-rust : {e}");
            std::process::exit(1)
        }
    };
    // Ligne de régime, comme le moteur Python : ce qui a été chargé, et d'où.
    eprintln!(
        "acvram-rust : modèle={} formats={:?} tenseurs_carte={} octets_carte={} noyaux={} fatbin_sha256={}",
        moteur.manifeste.model.name, moteur.manifeste.formats(), moteur.tenseurs_carte(),
        moteur.octets_carte, moteur.noyaux.nombre(), moteur.noyaux.sha256_fatbin
    );
    let nom = nom.unwrap_or_else(|| moteur.manifeste.model.name.clone());
    let (decodeur, vocab) = (moteur.tokeniseur.copie(), moteur.manifeste.model.vocab_size);
    let etat = Arc::new(Etat { nom_servi: nom, decodeur, vocab, moteur: Mutex::new(moteur) });
    let ecoute = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap_or_else(|e| {
        eprintln!("acvram-rust : port {port} : {e}");
        std::process::exit(1)
    });
    eprintln!("acvram-rust : écoute sur 127.0.0.1:{port}");
    if let Err(e) = axum::serve(ecoute, routes(etat)).await {
        eprintln!("acvram-rust : {e}");
        std::process::exit(1)
    }
}
