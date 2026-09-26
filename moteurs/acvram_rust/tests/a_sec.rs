//! Tests à sec (aucune carte). Ceux qui lisent le modèle ou l'extension du poste sont sautés, en le
//! disant, quand le fichier n'existe pas (CI sans /mnt ni ~/.cache/acvram).

use std::path::{Path, PathBuf};

use acvram_rust::manifeste::Manifeste;
use acvram_rust::moteur::argmax;
use acvram_rust::noyaux::{cle_stable, fatbin_de};
use acvram_rust::tokeniseur::{Message, Tokeniseur};

const MODELE: &str = "/mnt/AI_GENERATOR/models_acvram/Qwen3-4B-srcgguf-nvfp4";

fn modele() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("ACVRAM_RUST_MODELE").unwrap_or_else(|_| MODELE.into()));
    if p.join("acvram_manifest.json").is_file() {
        Some(p)
    } else {
        eprintln!("SAUTÉ : modèle absent ({})", p.display());
        None
    }
}

#[test]
fn cle_stable_efface_le_seul_espace_anonyme() {
    let a = "_ZN50_GLOBAL__N__c2b095cd_17_acvram_kernels_cu_487d8e4e25paged_attn_partial_kernelILi128EffLb1ELb0EEEvPKT0_i";
    let b = "_ZN50_GLOBAL__N__0badf00d_17_acvram_kernels_cu_12345678925paged_attn_partial_kernelILi128EffLb1ELb0EEEvPKT0_i";
    assert_eq!(cle_stable(a), "_ZN@25paged_attn_partial_kernelILi128EffLb1ELb0EEEvPKT0_i");
    // deux compilations = deux hachages, une seule clé
    assert_eq!(cle_stable(a), cle_stable(b.replace("123456789", "12345678").as_str()));
    // les arguments de gabarit restent : deux variantes ne se confondent pas
    assert_ne!(cle_stable(a), cle_stable(&a.replace("ILi128E", "ILi256E")));
    // sans espace anonyme : inchangé
    assert_eq!(cle_stable("_Z10rms_kernelPf"), "_Z10rms_kernelPf");
    // longueur annoncée hors du nom : inchangé, jamais une coupe arbitraire
    assert_eq!(cle_stable("_ZN99_GLOBAL__N__ab"), "_ZN99_GLOBAL__N__ab");
}

#[test]
fn argmax_premier_indice_du_maximum() {
    assert_eq!(argmax(&[]), None);
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
    assert_eq!(argmax(&[-1.0, f32::NAN, 5.0]), Some(1));
    assert_eq!(argmax(&[f32::NEG_INFINITY, -1e30]), Some(1));
}

#[test]
fn manifeste_du_modele_d_etape_1() {
    let Some(d) = modele() else { return };
    let m = Manifeste::lire(&d).expect("manifeste");
    assert_eq!((m.model.num_layers, m.model.hidden_size, m.model.head_dim), (36, 2560, 128));
    assert!(m.model.tie_word_embeddings);
    let f: Vec<String> = m.formats().into_iter().map(|(f, _)| f).collect();
    assert_eq!(f, ["bf16", "int8", "nvfp4"]);
    for e in m.tensors.values() {
        for k in &e.keys {
            assert!(m.weight_map.contains_key(k), "clé physique sans fichier : {k}");
        }
    }
}

#[test]
fn ids_d_invite_identiques_au_python() {
    let Some(d) = modele() else { return };
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/ids-qwen3-4b.json")).unwrap(),
    )
    .unwrap();
    let invites: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/invites.json")).unwrap(),
    )
    .unwrap();
    let t = Tokeniseur::charger(&d).expect("tokeniseur");
    for (inv, attendu) in invites.iter().zip(fixture["invites"].as_array().unwrap()) {
        let messages: Vec<Message> = serde_json::from_value(inv["messages"].clone()).unwrap();
        let texte = t.rendre(&messages, true).unwrap();
        assert_eq!(texte, attendu["texte"].as_str().unwrap(), "texte rendu, invite {}", inv["nom"]);
        let ids: Vec<u64> = t.encoder(&texte).unwrap().into_iter().map(u64::from).collect();
        let ids_py: Vec<u64> = attendu["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(ids, ids_py, "ids, invite {}", inv["nom"]);
    }
}

#[test]
fn fatbin_de_l_extension_servie() {
    let Some(racine) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/acvram")) else { return };
    let mut sos: Vec<PathBuf> = std::fs::read_dir(&racine)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path().join("acvram_kernels.so")))
        .filter(|p| p.is_file())
        .collect();
    sos.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    let Some(so) = sos.pop() else {
        eprintln!("SAUTÉ : aucune extension compilée sous {}", racine.display());
        return;
    };
    let f = fatbin_de(&so).expect("fatbin");
    assert_eq!(u32::from_le_bytes(f[0..4].try_into().unwrap()), 0xBA55_ED50);
    assert!(f.len() > 1 << 20, "fatbin de {} o : trop petit pour 347 noyaux", f.len());
}

/// Grilles et blocs relevés sur le moteur servi (prise 1, `revue/poste5-rust-p1-releve-23-09.md`) :
/// la réécriture des règles de lancement doit les redonner à l'identique.
#[test]
fn regles_de_lancement_redonnent_le_releve() {
    use acvram_rust::lanceurs::{splits_for, threads_for, threads_for_pairs};
    const SMS: u32 = 170; // RTX 5090
    // (M, K, format, grille relevée, bloc relevé)
    let cas = [
        (19456u32, 2560u32, "nvfp4", 4864u32, 64u32), // gate·up empilés
        (9728, 2560, "nvfp4", 2432, 64),             // gate seul
        (9728, 2560, "int8", 2432, 128),             // up seul
        (2560, 9728, "nvfp4", 640, 256),             // down
        (2560, 9728, "int8", 640, 256),
        (2560, 4096, "nvfp4", 640, 128),             // o
        (6144, 2560, "nvfp4", 1536, 64),             // q·k·v empilés
        (4096, 2560, "nvfp4", 1024, 64),             // q
        (1024, 2560, "int8", 256, 128),              // k ou v
        (151936, 2560, "int8", 37984, 128),          // tête
    ];
    for (m, k, fmt, grille, bloc) in cas {
        assert_eq!(splits_for(m, k, SMS), 1, "{m}×{k}");
        assert_eq!(m.div_ceil(4), grille, "{m}×{k}");
        let t = if fmt == "nvfp4" { threads_for_pairs(k) } else { threads_for(k) };
        assert_eq!(t, bloc, "{fmt} {m}×{k}");
    }
}

/// Tranches de l'attention au fil des pas : le relevé montre 2 tranches jusqu'à la position 127 puis 4
/// (invite « code » : 30 pas à 4 tranches pour 30 + 128 jetons).
#[test]
fn tranches_de_l_attention_suivent_le_releve() {
    use acvram_rust::decodage::{bucket_blocks, nblk_du_pas, tranches};
    assert_eq!((bucket_blocks(1), bucket_blocks(8), bucket_blocks(9), bucket_blocks(33)), (8, 8, 16, 64));
    assert_eq!((nblk_du_pas(127), nblk_du_pas(128)), (8, 16));
    assert_eq!(tranches(8, 1, 8, 170), (2, 64));
    assert_eq!(tranches(16, 1, 8, 170), (4, 64));
    assert_eq!(tranches(128, 1, 8, 170), (32, 64));
    let pas_a_4 = (30u32..30 + 128).filter(|&p| tranches(nblk_du_pas(p), 1, 8, 170).0 == 4).count();
    assert_eq!(pas_a_4, 30);
}
