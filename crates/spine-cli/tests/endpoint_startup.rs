use std::process::{Command, Stdio};

fn command(directory: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_spine"));
    command
        .env_clear()
        .current_dir(directory)
        .stdin(Stdio::null())
        .args(["chat", "--incognito-mode", "--no-nli"]);
    command
}

#[test]
fn noninteractive_default_exits_before_models_or_heart_creation() {
    let temp = tempfile::tempdir().unwrap();
    let output = command(temp.path()).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--server-url or SPINE_LLM_URL"));
    assert!(output.stdout.is_empty());
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn url_configuration_bypasses_prompt_via_flag_and_environment() {
    let (url, server) = healthy_server().await;
    let temp = tempfile::tempdir().unwrap();
    for use_env in [false, true] {
        let mut child = command(temp.path());
        child
            .arg("--model-dir")
            .arg(temp.path().join("missing-model"));
        if use_env {
            child.env("SPINE_LLM_URL", format!("{url}/v1/chat/completions"));
        } else {
            child.args(["--server-url", &format!("{url}/v1/chat/completions")]);
        }
        let output = child.output().unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("missing-model"),
            "expected model validation, got {error}"
        );
        assert!(!error.contains("stdin is not a terminal"));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Model server URL"));
    }
    server.abort();
}

async fn healthy_server() -> (String, tokio::task::JoinHandle<()>) {
    let app = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, server)
}

#[tokio::test(flavor = "multi_thread")]
async fn saved_connection_is_used_without_prompt_and_failed_startup_preserves_settings() {
    let (url, server) = healthy_server().await;
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("settings.json");
    let heart = temp.path().join("heart.spine");
    let original = serde_json::json!({
        "server_url": url, "heart": heart,
        "model_dir": temp.path().join("missing-model")
    })
    .to_string();
    std::fs::write(&config, &original).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_spine"))
        .env_clear()
        .current_dir(temp.path())
        .stdin(Stdio::null())
        .args(["chat", "--config"])
        .arg(&config)
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing-model"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("Model server URL"));
    assert!(!heart.exists());
    assert_eq!(std::fs::read_to_string(config).unwrap(), original);
    server.abort();
}

#[test]
fn refused_connection_fails_before_creating_heart_models_or_settings() {
    let temp = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let output = Command::new(env!("CARGO_BIN_EXE_spine"))
        .env_clear()
        .current_dir(temp.path())
        .stdin(Stdio::null())
        .args(["chat", "--config"])
        .arg(temp.path().join("settings.json"))
        .arg(temp.path().join("heart.spine"))
        .args(["--server-url", &url])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Cannot connect"));
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn bare_launch_uses_chat_and_no_settings_bypasses_a_broken_profile() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("settings.json");
    std::fs::write(&config, "invalid-json").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_spine"))
        .env_clear()
        .env("SPINE_CONFIG", &config)
        .current_dir(temp.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stderr).contains("Cannot read settings"));
    let output = Command::new(env!("CARGO_BIN_EXE_spine"))
        .env_clear()
        .env("SPINE_CONFIG", &config)
        .current_dir(temp.path())
        .stdin(Stdio::null())
        .args(["chat", "--no-settings"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stderr).contains("--server-url or SPINE_LLM_URL"));
    assert_eq!(std::fs::read_to_string(config).unwrap(), "invalid-json");
}

#[test]
fn llama_environment_paths_alone_cannot_start_a_server() {
    let temp = tempfile::tempdir().unwrap();
    let output = command(temp.path())
        .env("SPINE_LLAMA_MODEL", "model.gguf")
        .env("SPINE_LLAMA_SERVER", "llama-server")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--llamacpp"));
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}
