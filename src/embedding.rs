//! Ollama embedding API client.
//!
//! Wraps the Ollama `/api/embed` endpoint with batching support.
//! Mirrors the Python `embedding.py` module.

use serde::{Deserialize, Serialize};

use crate::error::ColibriError;

/// Maximum texts per Ollama embedding batch request.
const EMBED_BATCH_SIZE: usize = 32;

/// Request timeout per batch (seconds).
const EMBED_TIMEOUT_SECS: u64 = 120;

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Embed a list of texts using Ollama's `/api/embed` endpoint.
///
/// Batches requests in groups of [`EMBED_BATCH_SIZE`] to stay within limits.
pub async fn embed_texts(
    texts: &[String],
    model: &str,
    base_url: &str,
) -> Result<Vec<Vec<f32>>, ColibriError> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(EMBED_TIMEOUT_SECS))
        .build()
        .map_err(|e| ColibriError::Embedding(format!("Failed to build HTTP client: {e}")))?;

    let url = format!("{base_url}/api/embed");
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for batch in texts.chunks(EMBED_BATCH_SIZE) {
        let batch_vec: Vec<String> = batch.to_vec();
        let request = EmbedRequest {
            model,
            input: &batch_vec,
        };

        let response = client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ColibriError::Embedding(format!("Ollama request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ColibriError::Embedding(format!(
                "Ollama returned {status}: {body}"
            )));
        }

        let data: EmbedResponse = response.json().await.map_err(|e| {
            ColibriError::Embedding(format!("Failed to parse Ollama response: {e}"))
        })?;

        all_embeddings.extend(data.embeddings);
    }

    Ok(all_embeddings)
}

/// Check if Ollama is reachable by requesting the root endpoint.
pub async fn check_ollama(base_url: &str) -> Result<bool, ColibriError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| ColibriError::Embedding(format!("Failed to build HTTP client: {e}")))?;

    match client.get(base_url).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(_) => Ok(false),
    }
}
