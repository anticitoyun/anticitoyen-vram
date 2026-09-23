//! Porte d'entrée commune (contrat point 1) : `/v1/models` et `/v1/chat/completions`, API OpenAI,
//! pour que `banc-llamacpp-16-09.py` mesure ce moteur sans adaptation. Non streamé à l'étape 1.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::moteur::Moteur;
use crate::tokeniseur::Message;

pub struct Etat {
    pub nom_servi: String,
    /// b=1 : une requête à la fois, le verrou EST l'ordonnanceur de l'étape 1.
    pub moteur: Mutex<Moteur>,
}

#[derive(Deserialize)]
struct Requete {
    messages: Vec<Message>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
}

pub fn routes(etat: Arc<Etat>) -> Router {
    Router::new()
        .route("/v1/models", get(modeles))
        .route("/v1/chat/completions", post(conversation))
        .with_state(etat)
}

async fn modeles(State(etat): State<Arc<Etat>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": etat.nom_servi, "object": "model", "owned_by": "acvram_rust"}]}))
}

fn refus(code: StatusCode, message: String) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"error": {"message": message, "type": "invalid_request_error"}})))
}

async fn conversation(State(etat): State<Arc<Etat>>, Json(r): Json<Requete>) -> (StatusCode, Json<Value>) {
    if r.stream.unwrap_or(false) {
        return refus(StatusCode::BAD_REQUEST, "stream non porté à l'étape 1".into());
    }
    if r.temperature.is_some_and(|t| t != 0.0) {
        return refus(StatusCode::BAD_REQUEST, "étape 1 : glouton seulement (temperature 0)".into());
    }
    let max = r.max_tokens.unwrap_or(128);
    let e = etat.clone();
    let res = tokio::task::spawn_blocking(move || {
        let m = e.moteur.lock().map_err(|_| "moteur empoisonné".to_string())?;
        let texte = m.tokeniseur.rendre(&r.messages, true).map_err(|x| x.0)?;
        let ids = m.tokeniseur.encoder(&texte).map_err(|x| x.0)?;
        let sortie = m.generer(&ids, max).map_err(|x| x.0)?;
        let fini = sortie.last().is_some_and(|&t| m.est_fin(t));
        let contenu = m.tokeniseur.decoder(&sortie).map_err(|x| x.0)?;
        Ok::<_, String>((ids.len(), sortie.len(), fini, contenu))
    })
    .await;
    match res {
        Ok(Ok((n_invite, n_sortie, fini, contenu))) => {
            let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            (StatusCode::OK, Json(json!({
                "id": format!("chatcmpl-{t}"), "object": "chat.completion", "created": t,
                "model": etat.nom_servi,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": contenu},
                             "finish_reason": if fini { "stop" } else { "length" }}],
                "usage": {"prompt_tokens": n_invite, "completion_tokens": n_sortie,
                          "total_tokens": n_invite + n_sortie}
            })))
        }
        Ok(Err(m)) => refus(StatusCode::INTERNAL_SERVER_ERROR, m),
        Err(e) => refus(StatusCode::INTERNAL_SERVER_ERROR, format!("tâche : {e}")),
    }
}
