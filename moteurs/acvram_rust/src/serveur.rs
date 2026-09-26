//! Porte d'entrée commune (contrat point 1) : `/v1/models` et `/v1/chat/completions`, API OpenAI,
//! pour que `banc-llamacpp-16-09.py` mesure ce moteur sans adaptation. `/v1/chat/completions` non streamé ;
//! `/v1/completions` en flux SSE (étape 3, cellule de débit : le banc horodate chaque fragment).

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::moteur::Moteur;
use crate::tokeniseur::Message;

pub struct Etat {
    pub nom_servi: String,
    /// copie du tokeniseur : le flux se détokenise hors du verrou, le fil de génération ne fait que calculer
    pub decodeur: Tokenizer,
    pub vocab: usize,
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

#[derive(Deserialize, Default)]
struct OptionsFlux {
    #[serde(default)]
    include_usage: bool,
}

/// `/v1/completions` : champs lus par le banc (`completion()` de `banc-llamacpp-16-09.py`).
#[derive(Deserialize)]
struct RequeteTexte {
    /// liste d'ids, ou texte (encodé sans jetons spéciaux, comme le chat)
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<OptionsFlux>,
}

/// Ce que le fil de génération envoie au flux HTTP.
enum Sortie {
    Jeton(u32),
    Fin { n_invite: usize, n: usize, fini: bool },
    Erreur(String),
}

pub fn routes(etat: Arc<Etat>) -> Router {
    Router::new()
        .route("/v1/models", get(modeles))
        .route("/metrics", get(metriques))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(conversation))
        .with_state(etat)
}

/// Le banc lit `cartes` pour poser sa carte NVML (même clé que `Engine.regime()`) : le moteur est sur l'ordinal 0.
async fn metriques() -> Json<Value> {
    Json(json!({"cartes": ["cuda:0"]}))
}

fn maintenant() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn ids_invite(etat: &Etat, prompt: &Value) -> Result<Vec<u32>, String> {
    let ids: Vec<u32> = match prompt {
        Value::String(t) => etat.decodeur.encode(t.as_str(), false).map_err(|e| format!("encodage : {e}"))?
            .get_ids().to_vec(),
        Value::Array(a) => a.iter()
            .map(|v| v.as_u64().filter(|&x| (x as usize) < etat.vocab).map(|x| x as u32))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("prompt : ids entiers < {} attendus", etat.vocab))?,
        _ => return Err("prompt : texte ou liste d'ids".into()),
    };
    if ids.is_empty() {
        return Err("prompt vide".into());
    }
    Ok(ids)
}

async fn completion(State(etat): State<Arc<Etat>>, Json(r): Json<RequeteTexte>) -> Response {
    if r.temperature.is_some_and(|t| t != 0.0) {
        return refus(StatusCode::BAD_REQUEST, "glouton seulement (temperature 0)".into()).into_response();
    }
    let ids = match ids_invite(&etat, &r.prompt) {
        Ok(ids) => ids,
        Err(m) => return refus(StatusCode::BAD_REQUEST, m).into_response(),
    };
    let (max, ignore_eos) = (r.max_tokens.unwrap_or(16), r.ignore_eos);
    let (tx, mut rx) = mpsc::unbounded_channel();
    let e = etat.clone();
    tokio::task::spawn_blocking(move || {
        let n_invite = ids.len();
        let res = e.moteur.lock().map_err(|_| "moteur empoisonné".to_string()).and_then(|mut m| {
            let mut dernier = None;
            // client parti (récepteur fermé) → arrêt au jeton suivant
            let n = m.generer_flux(&ids, max, ignore_eos, |j| {
                dernier = Some(j);
                tx.send(Sortie::Jeton(j)).is_ok()
            }).map_err(|x| x.0)?;
            Ok((n, dernier.is_some_and(|j| m.est_fin(j))))
        });
        let _ = tx.send(match res {
            Ok((n, fini)) => Sortie::Fin { n_invite, n, fini: fini && !ignore_eos },
            Err(m) => Sortie::Erreur(m),
        });
    });
    let (id, cree) = (format!("cmpl-{}", maintenant()), maintenant());
    if !r.stream {
        let (mut sortie, mut fin) = (Vec::new(), None);
        while let Some(m) = rx.recv().await {
            match m {
                Sortie::Jeton(j) => sortie.push(j),
                Sortie::Fin { n_invite, n, fini } => fin = Some((n_invite, n, fini)),
                Sortie::Erreur(m) => return refus(StatusCode::INTERNAL_SERVER_ERROR, m).into_response(),
            }
        }
        let Some((n_invite, n, fini)) = fin else {
            return refus(StatusCode::INTERNAL_SERVER_ERROR, "génération interrompue".into()).into_response();
        };
        let texte = etat.decodeur.decode(&sortie, true).unwrap_or_default();
        return Json(json!({
            "id": id, "object": "text_completion", "created": cree, "model": etat.nom_servi,
            "choices": [{"index": 0, "text": texte, "logprobs": null, "finish_reason": if fini { "stop" } else { "length" }}],
            "usage": {"prompt_tokens": n_invite, "completion_tokens": n, "total_tokens": n_invite + n}
        })).into_response();
    }
    let usage = r.stream_options.unwrap_or_default().include_usage;
    let nom = etat.nom_servi.clone();
    let fragment = move |choix: Value| json!({
        "id": id, "object": "text_completion", "created": cree, "model": nom, "choices": choix
    });
    // Un fragment par jeton (le banc compte les fragments à texte non vide, ou `usage.completion_tokens`).
    // Détokenisation jeton par jeton : un caractère coupé entre deux jetons rend U+FFFD — le texte n'est pas jugé ici.
    let flux = futures_util::stream::unfold((rx, false), move |(mut rx, clos)| {
        let (e, fragment) = (etat.clone(), fragment.clone());
        async move {
            if clos {
                return None;
            }
            let evenements: Vec<Event> = match rx.recv().await {
                Some(Sortie::Jeton(j)) => {
                    let texte = e.decodeur.decode(&[j], true).unwrap_or_default();
                    vec![Event::default().data(fragment(json!([{"index": 0, "text": texte, "logprobs": null, "finish_reason": null}])).to_string())]
                }
                Some(Sortie::Fin { n_invite, n, fini }) => {
                    let mut v = vec![Event::default().data(fragment(json!([{"index": 0, "text": "", "logprobs": null,
                        "finish_reason": if fini { "stop" } else { "length" }}])).to_string())];
                    if usage {
                        let mut f = fragment(json!([]));
                        f["usage"] = json!({"prompt_tokens": n_invite, "completion_tokens": n, "total_tokens": n_invite + n});
                        v.push(Event::default().data(f.to_string()));
                    }
                    v.push(Event::default().data("[DONE]"));
                    return Some((futures_util::stream::iter(v.into_iter().map(Ok::<_, Infallible>)), (rx, true)));
                }
                Some(Sortie::Erreur(m)) => vec![
                    Event::default().data(json!({"error": {"message": m, "type": "server_error"}}).to_string()),
                    Event::default().data("[DONE]"),
                ],
                None => return None,
            };
            let clos = evenements.len() > 1;
            Some((futures_util::stream::iter(evenements.into_iter().map(Ok::<_, Infallible>)), (rx, clos)))
        }
    });
    Sse::new(futures_util::StreamExt::flatten(flux)).into_response()
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
        let mut m = e.moteur.lock().map_err(|_| "moteur empoisonné".to_string())?;
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
