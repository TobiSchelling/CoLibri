//! Shared helpers for CLI integration tests.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

/// Minimal stand-in for Ollama: `GET /` answers 200, `POST /api/embed`
/// returns deterministic 8-dimensional vectors.
pub struct FakeOllama {
    pub url: String,
    embeds: Arc<AtomicUsize>,
}

impl FakeOllama {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake ollama");
        let url = format!("http://{}", listener.local_addr().unwrap());
        let embeds = Arc::new(AtomicUsize::new(0));
        let counter = embeds.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let counter = counter.clone();
                std::thread::spawn(move || {
                    let _ = handle(stream, &counter);
                });
            }
        });
        Self { url, embeds }
    }

    /// Number of `/api/embed` requests served so far.
    pub fn embed_requests(&self) -> usize {
        self.embeds.load(Ordering::SeqCst)
    }
}

fn vector_for(text: &str) -> Vec<f32> {
    let mut v = [0.0f32; 8];
    for (i, b) in text.bytes().enumerate() {
        v[i % 8] += b as f32 / 255.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / norm).collect()
}

fn handle(mut stream: std::net::TcpStream, embeds: &AtomicUsize) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line.is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    let response = if request_line.starts_with("POST /api/embed") {
        embeds.fetch_add(1, Ordering::SeqCst);
        let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        let inputs: Vec<String> = req["input"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let embeddings: Vec<Vec<f32>> = inputs.iter().map(|t| vector_for(t)).collect();
        json!({ "embeddings": embeddings }).to_string()
    } else if request_line.starts_with("GET /api/tags") {
        json!({ "models": [{ "name": "fake:latest" }] }).to_string()
    } else {
        "Ollama is running".to_string()
    };
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.len(),
        response
    )?;
    stream.flush()
}

/// Isolated config + data dir + source folder for one test.
pub struct TestHome {
    root: tempfile::TempDir,
}

pub struct Output {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn assert_ok(&self) {
        assert!(
            self.success,
            "command failed\nstdout:\n{}\nstderr:\n{}",
            self.stdout, self.stderr
        );
    }
}

impl TestHome {
    /// Config with one mirror `notes` over the source dir (markdown + yaml).
    pub fn new(ollama: &FakeOllama) -> Self {
        Self::with_mirrors(ollama, "")
    }

    /// Like `new`, plus extra YAML mirror entries (each starting with `  - `).
    pub fn with_mirrors(ollama: &FakeOllama, extra_mirrors: &str) -> Self {
        let root = tempfile::TempDir::new().unwrap();
        let home = Self { root };
        std::fs::create_dir_all(home.source_dir()).unwrap();
        let config = format!(
            "embedding:\n  endpoint: {}\n  model: fake\nchunking:\n  chunk_size: 400\n  chunk_overlap: 40\nretrieval:\n  similarity_threshold: 0.0\nprune:\n  min_count: 25\nmirrors:\n  - name: notes\n    path: {}\n    include: ['**/*.md', '**/*.yaml']\n{extra_mirrors}",
            ollama.url,
            home.source_dir().display()
        );
        std::fs::write(home.config_path(), config).unwrap();
        home
    }

    pub fn config_path(&self) -> PathBuf {
        self.root.path().join("config.yaml")
    }

    pub fn data_dir(&self) -> PathBuf {
        self.root.path().join("data")
    }

    pub fn source_dir(&self) -> PathBuf {
        self.root.path().join("sources")
    }

    pub fn write_source(&self, rel: &str, text: &str) {
        let path = self.source_dir().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    pub fn colibri(&self, args: &[&str]) -> Output {
        self.colibri_with_stdin(args, "")
    }

    pub fn colibri_with_stdin(&self, args: &[&str], stdin: &str) -> Output {
        let mut child = Command::new(env!("CARGO_BIN_EXE_colibri"))
            .args(args)
            .env("COLIBRI_CONFIG_PATH", self.config_path())
            .env("COLIBRI_HOME", self.data_dir())
            .env("HOME", self.root.path())
            .env_remove("COLIBRI_CONFIG")
            .env_remove("COLIBRI_DATA_DIR")
            .env_remove("OLLAMA_BASE_URL")
            .env_remove("COLIBRI_EMBEDDING_MODEL")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("run colibri");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(stdin.as_bytes())
            .unwrap();
        let out = child.wait_with_output().expect("wait for colibri");
        Output {
            success: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    }

    pub fn path(&self) -> &Path {
        self.root.path()
    }
}

/// (relative path -> (size, mtime)) for every entry under `root`.
pub fn snapshot(root: &Path) -> std::collections::BTreeMap<PathBuf, (u64, std::time::SystemTime)> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                stack.push(path.clone());
            }
            out.insert(path, (meta.len(), meta.modified().unwrap()));
        }
    }
    out
}
