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

#[test]
fn url_configuration_bypasses_prompt_via_flag_and_environment() {
    let temp = tempfile::tempdir().unwrap();
    for use_env in [false, true] {
        let mut child = command(temp.path());
        child
            .arg("--model-dir")
            .arg(temp.path().join("missing-model"));
        if use_env {
            child.env("SPINE_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions");
        } else {
            child.args(["--server-url", "http://127.0.0.1:8080/v1/chat/completions"]);
        }
        let output = child.output().unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("missing-model"),
            "expected model validation, got {error}"
        );
        assert!(!error.contains("stdin is not a terminal"));
        assert!(output.stdout.is_empty());
    }
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
