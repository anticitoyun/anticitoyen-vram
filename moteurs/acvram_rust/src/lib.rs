//! acvram_rust : moteur de comparaison d'acvram (contrat `revue/moteurs-rust-mojo-23-09.md`).
//!
//! Question mesurée : le langage HÔTE seul. Les noyaux ne sont ni réécrits ni recompilés : le fatbin
//! sm_120 est pris tel quel dans l'extension que le moteur Python sert (`noyaux`), pour que les deux
//! moteurs exécutent le même SASS et que seule la partie hôte diffère.

pub mod argmax;
pub mod chargement;
pub mod decodage;
pub mod lanceurs;
pub mod manifeste;
pub mod moteur;
pub mod noyaux;
pub mod poids;
pub mod serveur;
pub mod tokeniseur;
pub mod triton;

/// Erreur unique du moteur : un message en français, sans hiérarchie de types — le moteur de
/// comparaison n'a pas d'appelant qui trie les erreurs, il les dit.
#[derive(Debug)]
pub struct Erreur(pub String);

impl std::fmt::Display for Erreur {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Erreur {}

pub type Resultat<T> = Result<T, Erreur>;

#[macro_export]
macro_rules! erreur {
    ($($arg:tt)*) => { $crate::Erreur(format!($($arg)*)) };
}
