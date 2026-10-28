//! Interactive connection setup; never launches a server or consumes piped chat.
use std::io::{self, BufRead, Write};

use reqwest::Url;

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080";
const MANAGED_ENDPOINT: &str = "http://127.0.0.1:8080";

pub fn resolve(
    provided: Option<&str>,
    managed: bool,
    interactive: bool,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> io::Result<String> {
    if let Some(provided) = provided {
        return normalize(provided).map(|url| url.to_string());
    }
    if managed {
        return Ok(MANAGED_ENDPOINT.into());
    }
    if !interactive {
        return Err(invalid(
            "set --server-url or SPINE_LLM_URL, or run interactive setup once, when stdin is not a terminal",
        ));
    }
    loop {
        let endpoint = prompt(input, output, "Model server URL", DEFAULT_ENDPOINT)?;
        match normalize(&endpoint) {
            Ok(url) => return Ok(url.to_string()),
            Err(error) => writeln!(output, "{error}")?,
        }
    }
}

pub fn normalize(value: &str) -> io::Result<Url> {
    if !value.contains("://") {
        return Err(invalid("Enter a valid HTTP(S) endpoint URL including ://."));
    }
    let mut url =
        Url::parse(value.trim()).map_err(|_| invalid("Enter a valid HTTP(S) endpoint URL."))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(invalid(
            "Endpoint must use http:// or https:// and include a host.",
        ));
    }
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "Endpoint must not contain credentials, query parameters or a fragment; use SPINE_LLM_API_KEY for credentials.",
        ));
    }
    if url.port() == Some(0) {
        return Err(invalid("Endpoint port must be from 1 to 65535."));
    }
    let path = url.path().trim_end_matches('/');
    let base = if let Some(base) = path.strip_suffix("/v1/chat/completions") {
        // Keep root /health and /props discovery working, as well as /v1/models.
        // The provider appends /v1/chat/completions to this server prefix.
        base.to_owned()
    } else if path.ends_with("/completions") {
        return Err(invalid(
            "Use a base URL or /v1/chat/completions, not the text-only /v1/completions endpoint.",
        ));
    } else {
        path.to_owned()
    };
    url.set_path(&base);
    Ok(url)
}

fn prompt(
    input: &mut impl BufRead,
    output: &mut impl Write,
    label: &str,
    default: &str,
) -> io::Result<String> {
    write!(output, "{label} [{default}]: ")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "endpoint setup cancelled: end of input",
        ));
    }
    let value = line.trim();
    Ok(if value.is_empty() {
        default.to_owned()
    } else {
        value.to_owned()
    })
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn interact(text: &str) -> (io::Result<String>, String, u64) {
        let mut input = Cursor::new(text);
        let mut output = Vec::new();
        let result = resolve(None, false, true, &mut input, &mut output);
        (result, String::from_utf8(output).unwrap(), input.position())
    }

    #[test]
    fn one_url_prompt_leaves_chat_unconsumed() {
        let (result, output, consumed) = interact("\nhello\n");
        assert_eq!(result.unwrap(), "http://127.0.0.1:8080/");
        assert_eq!(output.matches("Model server URL").count(), 1);
        assert_eq!(consumed, 1);
        assert_eq!(
            interact("https://example.com/v1\n").0.unwrap(),
            "https://example.com/v1"
        );
        assert_eq!(
            interact("http://localhost:9001/v1/chat/completions\n")
                .0
                .unwrap(),
            "http://localhost:9001/"
        );
    }
    #[test]
    fn invalid_answers_retry_and_eof_never_accepts_defaults() {
        let (result, output, _) = interact("bad-url\nhttp://localhost/v1/completions\n\n");
        assert_eq!(result.unwrap(), "http://127.0.0.1:8080/");
        assert!(output.contains("not the text-only"));
        assert_eq!(
            interact("").0.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn explicit_and_managed_modes_do_not_prompt_or_read_input() {
        for (provided, managed, expected) in [
            (
                Some("https://example.com/v1/chat/completions"),
                false,
                "https://example.com/",
            ),
            (None, true, MANAGED_ENDPOINT),
            (
                Some("http://localhost:9001"),
                true,
                "http://localhost:9001/",
            ),
        ] {
            let mut input = Cursor::new("do not consume\n");
            let mut output = Vec::new();
            assert_eq!(
                resolve(provided, managed, false, &mut input, &mut output).unwrap(),
                expected
            );
            assert_eq!(input.position(), 0);
            assert!(output.is_empty());
        }
    }

    #[test]
    fn noninteractive_default_requires_configuration_without_consuming_chat() {
        let mut input = Cursor::new("hello\n");
        let mut output = Vec::new();
        let error = resolve(None, false, false, &mut input, &mut output).unwrap_err();
        assert!(error.to_string().contains("--server-url or SPINE_LLM_URL"));
        assert_eq!(input.position(), 0);
        assert!(output.is_empty());
    }

    #[test]
    fn unsafe_or_incompatible_urls_fail_without_echoing_credentials() {
        // Assemble the deliberately invalid fixture without checking in a
        // credential-bearing URL that the release hygiene scan would reject.
        let credential_url = ["http:", "//", "u:secret", "@example.com"].concat();
        for value in [
            "http:localhost",
            "ftp://example.com",
            &credential_url,
            "http://localhost:0",
            "http://localhost/v1/completions",
            "http://localhost?key=secret",
            "http://localhost#secret",
        ] {
            let error = normalize(value).unwrap_err();
            assert!(!error.to_string().contains("secret"));
        }
    }

    #[tokio::test]
    async fn prompted_endpoint_supports_models_only_and_root_health_only_servers() {
        use axum::{
            Json, Router,
            routing::{get, post},
        };
        use serde_json::{Value, json};
        use spine_runtime::{
            CompletionRequest, LlamaCppConfig, LlamaCppProvider, Message, MessageRole,
            ModelProvider,
        };

        for health_only in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let app = Router::new()
            .route("/v1/chat/completions", post(|Json(body): Json<Value>| async move {
                assert_eq!(body["messages"][0]["content"], "hello");
                Json(json!({"choices": [{"message": {"role": "assistant", "content": "endpoint-ok"}}]}))
            }));
            let app = if health_only {
                app.route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            } else {
                app.route(
                    "/v1/models",
                    get(|| async { Json(json!({"data": [{"id": "test-model"}]})) }),
                )
            };
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let base = interact(&format!("http://127.0.0.1:{port}\n")).0.unwrap();
            let mut config = LlamaCppConfig::new(base);
            config.maximum_retries = 0;
            config.timeout = std::time::Duration::from_secs(5);
            let provider = LlamaCppProvider::new(config).unwrap();
            let result = async {
                provider.health().await?;
                provider
                    .complete(CompletionRequest {
                        progress: None,
                        messages: vec![Message::new(MessageRole::User, "hello")],
                        tools: Vec::new(),
                        allow_tool_calls: false,
                        temperature: None,
                    })
                    .await
            }
            .await;
            server.abort();
            assert_eq!(result.unwrap().content, "endpoint-ok");
        }
    }
}
