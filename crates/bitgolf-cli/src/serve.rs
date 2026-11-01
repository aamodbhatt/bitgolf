//! Static server for the browser demo: `web/` at `/`, `.bgm` files at `/models/`, and a
//! `/models/index.json` listing them, built on each request.
//!
//! Request paths are matched against a strict character whitelist and never joined with
//! `..`, so nothing outside the two directories can be served.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use tiny_http::{Header, Response, Server};

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        _ => "application/octet-stream",
    }
}

fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Maps a URL path to a file under `web` or `models`, or `None` if it is not servable.
fn route(url_path: &str, web: &Path, models: &Path) -> Option<PathBuf> {
    let path = url_path.split(['?', '#']).next().unwrap_or("/");
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if !segments.iter().all(|s| safe_segment(s)) {
        return None;
    }
    match segments.as_slice() {
        [] => Some(web.join("index.html")),
        ["models", name] if name.ends_with(".bgm") => Some(models.join(name)),
        rest => Some(rest.iter().fold(web.to_path_buf(), |p, s| p.join(s))),
    }
}

fn models_index(models: &Path) -> Result<String> {
    let mut list = Vec::new();
    for entry in std::fs::read_dir(models).with_context(|| format!("listing {}", models.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "bgm") {
            let bytes = std::fs::read(&path)?;
            if let Ok(model) = bitgolf_core::Model::from_bytes(&bytes) {
                list.push(serde_json::json!({
                    "file": path.file_name().and_then(|n| n.to_str()),
                    "name": model.config().name,
                    "params": model.num_params(),
                    "bytes": bytes.len(),
                    "window": model.config().window,
                    "sha256": model.id_hex(),
                }));
            }
        }
    }
    list.sort_by_key(|m| m["bytes"].as_u64());
    Ok(serde_json::to_string(&list)?)
}

/// Serves until the process is stopped.
pub fn serve(addr: &str, web: &Path, models: &Path) -> Result<()> {
    let server = Server::http(addr).map_err(|e| anyhow!("cannot listen on {addr}: {e}"))?;
    eprintln!("bitgolf demo on http://{addr}/  (web: {}, models: {})", web.display(), models.display());
    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let response = if url.split('?').next() == Some("/models/index.json") {
            match models_index(models) {
                Ok(json) => Response::from_data(json.into_bytes()).with_header(header("application/json")),
                Err(e) => Response::from_string(e.to_string()).with_status_code(500),
            }
        } else {
            match route(&url, web, models).and_then(|p| std::fs::read(&p).ok().map(|b| (p, b))) {
                Some((path, bytes)) => Response::from_data(bytes).with_header(header(content_type(&path))),
                None => Response::from_string("not found").with_status_code(404),
            }
        };
        let response = response
            .with_header(Header::from_bytes("Cache-Control", "no-cache").expect("static header is valid"));
        if let Err(e) = request.respond(response) {
            eprintln!("{url}: {e}");
        }
    }
    Ok(())
}

fn header(value: &str) -> Header {
    Header::from_bytes("Content-Type", value).expect("static header is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_only_whitelisted_paths() {
        let (web, models) = (Path::new("web"), Path::new("models"));
        assert_eq!(route("/", web, models), Some(web.join("index.html")));
        assert_eq!(route("/app.js?v=1", web, models), Some(web.join("app.js")));
        assert_eq!(route("/pkg/a_bg.wasm", web, models), Some(web.join("pkg").join("a_bg.wasm")));
        assert_eq!(route("/models/bitgolf-small.bgm", web, models), Some(models.join("bitgolf-small.bgm")));
        assert_eq!(route("/../secret", web, models), None);
        assert_eq!(route("/..%2fsecret", web, models), None);
        assert_eq!(route("/.git/config", web, models), None);
        assert_eq!(route("/a\\b", web, models), None);
    }
}
