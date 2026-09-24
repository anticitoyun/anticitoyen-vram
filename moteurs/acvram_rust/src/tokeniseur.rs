//! Tokeniseur et gabarit de conversation, calqués sur `acvram/server/chat.py` :
//! même bibliothèque (`tokenizers`, même version), gabarit Jinja de `tokenizer_config.json` rendu
//! avec `trim_blocks` et `lstrip_blocks` (chat.py:146), encodage SANS jetons spéciaux (chat.py:41).
//! Toute différence ici change les ids d'invite et rend la comparaison au bit impossible.

use std::path::Path;

use minijinja::{Environment, Error, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{erreur, Resultat};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct Tokeniseur {
    tok: Tokenizer,
    env: Environment<'static>,
    /// variables scalaires de `tokenizer_config.json`, passées au gabarit comme en Python (chat.py:171)
    config: serde_json::Map<String, serde_json::Value>,
}

impl Tokeniseur {
    pub fn charger(dossier: &Path) -> Resultat<Self> {
        let tok = Tokenizer::from_file(dossier.join("tokenizer.json"))
            .map_err(|e| erreur!("tokenizer.json : {e}"))?;
        let texte = std::fs::read_to_string(dossier.join("tokenizer_config.json"))
            .map_err(|e| erreur!("tokenizer_config.json : {e}"))?;
        let config: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&texte).map_err(|e| erreur!("tokenizer_config.json : {e}"))?;
        let gabarit = config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| erreur!("pas de chat_template : le repli ChatML du Python n'est pas porté"))?
            .to_string();
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("conversation", gabarit)
            .map_err(|e| erreur!("gabarit illisible : {e}"))?;
        Ok(Self { tok, env, config })
    }

    pub fn rendre(&self, messages: &[Message], add_generation_prompt: bool) -> Resultat<String> {
        let mut ctx = serde_json::Map::new();
        for (k, v) in &self.config {
            let scalaire = v.is_string() || v.is_number() || v.is_boolean();
            if scalaire && !matches!(k.as_str(), "messages" | "add_generation_prompt" | "bos_token" | "eos_token") {
                ctx.insert(k.clone(), v.clone());
            }
        }
        ctx.insert("messages".into(), serde_json::to_value(messages).map_err(|e| erreur!("{e}"))?);
        ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
        let jeton = |cle: &str| match self.config.get(cle) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Object(o)) => {
                o.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string()
            }
            _ => String::new(),
        };
        ctx.insert("bos_token".into(), jeton("bos_token").into());
        ctx.insert("eos_token".into(), jeton("eos_token").into());
        let t = self.env.get_template("conversation").map_err(|e| erreur!("{e}"))?;
        t.render(Value::from_serialize(&ctx)).map_err(|e| erreur!("rendu du gabarit : {e}"))
    }

    pub fn encoder(&self, texte: &str) -> Resultat<Vec<u32>> {
        let enc = self.tok.encode(texte, false).map_err(|e| erreur!("encodage : {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Copie du tokeniseur pour détokeniser le flux hors du verrou du moteur (serveur).
    pub fn copie(&self) -> Tokenizer {
        self.tok.clone()
    }

    pub fn decoder(&self, ids: &[u32]) -> Resultat<String> {
        self.tok.decode(ids, true).map_err(|e| erreur!("décodage : {e}"))
    }
}
