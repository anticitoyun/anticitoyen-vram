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
