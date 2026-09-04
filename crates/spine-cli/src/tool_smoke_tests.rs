use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use axum::{Router, response::Html, routing::get};
use serde_json::{Value, json};
use spine_heart::{
    AgentId, CognitiveConfig, Embedding, HeartConfig, ModelManifest, Result as HeartResult,
    SemanticEncoder, SpineHeart,
};
use spine_runtime::{
    CompletionRequest, Harness, HarnessConfig, MessageRole, ModelProvider, ModelTurn, ToolCall,
    ToolContext, ToolRegistry, ToolResult,
};

use crate::{agent_tools, cognition_tools, partner_tools};

#[derive(Clone)]
struct TinyEncoder {
    manifest: ModelManifest,
}

impl TinyEncoder {
    fn new() -> Self {
        Self {
            manifest: ModelManifest {
                schema: 1,
                model_name: "tool-smoke-encoder".into(),
                artifact_hash: [31; 32],
                tokenizer_hash: [32; 32],
                dimension: 3,
                normalized: true,
                quantization: None,
            },
        }
    }
}

impl SemanticEncoder for TinyEncoder {
    fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    fn encode(&self, text: &str) -> HeartResult<Embedding> {
        let digest = blake3::hash(text.as_bytes());
        Embedding::normalized(
            vec![
                f32::from(digest.as_bytes()[0]) + 1.0,
                f32::from(digest.as_bytes()[1]) + 1.0,
                f32::from(digest.as_bytes()[2]) + 1.0,
            ],
            self.manifest.dimension,
        )
    }
}

#[derive(Default)]
struct FinalProvider;

#[async_trait]
impl ModelProvider for FinalProvider {
    async fn complete(&self, _request: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        Ok(ModelTurn {
            content: "sub-agent completed its bounded task".into(),
            ..ModelTurn::default()
        })
    }
}

struct PendingProvider;

#[async_trait]
impl ModelProvider for PendingProvider {
    async fn complete(&self, _request: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        tokio::time::sleep(Duration::from_secs(60)).await;
        Ok(ModelTurn::default())
    }
}

struct ScriptedProvider {
    turns: Mutex<VecDeque<ModelTurn>>,
}

#[async_trait]
impl ModelProvider for ScriptedProvider {
    async fn complete(&self, _request: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        self.turns
            .lock()
            .expect("scripted provider lock poisoned")
            .pop_front()
            .ok_or_else(|| spine_runtime::RuntimeError::Provider("script exhausted".into()))
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    heart: Arc<SpineHeart>,
    registry: ToolRegistry,
    audit_path: std::path::PathBuf,
}

fn fixture(model_memory_writes: bool, subagents: bool) -> Fixture {
    let directory = tempfile::tempdir().expect("temporary directory");
    let encoder = Arc::new(TinyEncoder::new());
    let created = SpineHeart::create(
        HeartConfig::new(directory.path().join("tool-smoke.spine")),
        "tool-smoke-passphrase",
    )
    .expect("create heart");
    created
        .heart
        .initialize_cognition(
            CognitiveConfig::new(1, encoder.manifest().clone(), 2)
                .expect("valid cognitive configuration"),
        )
        .expect("initialize cognition");
    let heart = Arc::new(created.heart);
    let audit_path = directory.path().join("tool-audit.jsonl");
    let mut registry = ToolRegistry::default();
    cognition_tools::register_cognition_tools(
        &mut registry,
        Arc::clone(&heart),
        Arc::clone(&encoder),
        model_memory_writes,
    )
    .expect("register cognition tools");
    partner_tools::register_action_tools(
        &mut registry,
        Arc::clone(&heart),
        encoder,
        directory.path().to_path_buf(),
        audit_path.clone(),
    )
    .expect("register action tools");
    if subagents {
        let child_registry = registry.clone();
        agent_tools::register_subagent_tools(
            &mut registry,
            Arc::new(FinalProvider),
            child_registry,
            3,
            2,
        )
        .expect("register sub-agent tools");
    }
    Fixture {
        _directory: directory,
        heart,
        registry,
        audit_path,
    }
}

async fn execute(registry: &ToolRegistry, name: &str, arguments: Value) -> ToolResult {
    let tool = Arc::clone(
        registry
            .get(name)
            .unwrap_or_else(|| panic!("registered tool {name}")),
    );
    tool.execute(
        &ToolCall {
            id: format!("smoke-{name}"),
            name: name.into(),
            arguments,
        },
        &ToolContext {
            harness_id: "tool-smoke".into(),
            agent_id: Some(AgentId::new("main").expect("agent id")),
            ..ToolContext::default()
        },
    )
    .await
    .unwrap_or_else(|error| panic!("execute {name}: {error}"))
}

#[test]
fn registration_exposes_the_complete_validated_tool_surface() {
    let fixture = fixture(true, true);
    let names = fixture
        .registry
        .specs()
        .into_iter()
        .map(|spec| {
            assert!(!spec.description.trim().is_empty(), "{}", spec.name);
            assert_eq!(spec.parameters["type"], "object", "{}", spec.name);
            spec.name
        })
        .collect::<BTreeSet<_>>();
    let expected = [
        "cancel_agent",
        "cancel_task",
        "check_results",
        "check_tasks",
        "delegate",
        "fact_aggregate",
        "fact_search",
        "feel",
        "file_list",
        "file_read",
        "file_search",
        "file_write",
        "heart_recall",
        "heart_stats",
        "ingest_documents",
        "maintain_memory",
        "memory_stats",
        "save_memory",
        "search_memory",
        "shell",
        "web_back",
        "web_fetch",
        "web_forward",
        "web_navigate",
        "web_search",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(names, expected);
}

#[tokio::test]
async fn filesystem_shell_browser_and_task_controls_work_together() {
    let fixture = fixture(false, false);
    let written = execute(
        &fixture.registry,
        "file_write",
        json!({"path":"notes/example.md","content":"hello from the tool smoke test"}),
    )
    .await;
    assert!(written.success, "{:?}", written.error);

    let read = execute(
        &fixture.registry,
        "file_read",
        json!({"path":"notes/example.md"}),
    )
    .await;
    assert!(read.success);
    assert!(read.output.contains("hello from the tool smoke test"));

    let listed = execute(
        &fixture.registry,
        "file_list",
        json!({"path":".","recursive":true}),
    )
    .await;
    assert!(listed.success);
    let expected_listed_path = std::path::Path::new("notes")
        .join("example.md")
        .display()
        .to_string();
    assert!(
        listed.output.contains(expected_listed_path.as_str()),
        "unexpected file_list output: {}",
        listed.output
    );

    let searched = execute(
        &fixture.registry,
        "file_search",
        json!({"path":"notes","pattern":"*.md"}),
    )
    .await;
    assert!(searched.success);
    assert!(searched.output.contains("example.md"));
    let missing_search = execute(
        &fixture.registry,
        "file_search",
        json!({"path":"missing","pattern":"*.md"}),
    )
    .await;
    assert!(!missing_search.success);
    assert!(
        missing_search
            .error
            .as_deref()
            .is_some_and(|error| error.contains("search root not found"))
    );

    #[cfg(not(target_os = "windows"))]
    let command = "printf shell-ok";
    #[cfg(target_os = "windows")]
    let command = "Write-Output shell-ok";
    let shell = execute(
        &fixture.registry,
        "shell",
        json!({"command":command,"timeout_s":5}),
    )
    .await;
    assert!(shell.success, "{:?}", shell.error);
    assert!(shell.output.contains("shell-ok"));
    let tasks = execute(&fixture.registry, "check_tasks", json!({})).await;
    assert!(tasks.success);
    assert!(tasks.output.contains("status=completed"));

    let shell_tool = Arc::clone(fixture.registry.get("shell").expect("shell tool"));
    #[cfg(not(target_os = "windows"))]
    let long_command = "sleep 30";
    #[cfg(target_os = "windows")]
    let long_command = "Start-Sleep -Seconds 30";
    let running = tokio::spawn(async move {
        shell_tool
            .execute(
                &ToolCall {
                    id: "long-shell".into(),
                    name: "shell".into(),
                    arguments: json!({"command":long_command,"timeout_s":60}),
                },
                &ToolContext::default(),
            )
            .await
            .expect("long shell execution")
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let running_status = execute(&fixture.registry, "check_tasks", json!({})).await;
    assert!(running_status.output.contains("[task2] status=running"));
    let cancelled = execute(&fixture.registry, "cancel_task", json!({"task_id":"task2"})).await;
    assert!(cancelled.output.contains("Cancelled 1 task"));
    let long_result = tokio::time::timeout(Duration::from_secs(2), running)
        .await
        .expect("cancelled shell stopped promptly")
        .expect("shell task joined");
    assert!(!long_result.success);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind local test server");
    let address = listener.local_addr().expect("local server address");
    let app = Router::new()
        .route(
            "/one",
            get(|| async {
                Html(
                    "<html><title>Page One</title><body>alpha page<a href='/two'>next</a></body></html>",
                )
            }),
        )
        .route(
            "/two",
            get(|| async {
                Html("<html><title>Page Two</title><body>beta page</body></html>")
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve local test pages");
    });
    let first_url = format!("http://{address}/one");
    let second_url = format!("http://{address}/two");
    let fetched = execute(&fixture.registry, "web_fetch", json!({"url":first_url})).await;
    assert!(fetched.success, "{:?}", fetched.error);
    assert!(fetched.output.contains("Page One"));
    assert!(fetched.output.contains(&second_url));
    let navigated = execute(&fixture.registry, "web_navigate", json!({"url":second_url})).await;
    assert!(navigated.success);
    assert!(navigated.output.contains("Page Two"));
    let back = execute(&fixture.registry, "web_back", json!({})).await;
    assert!(back.success);
    assert!(back.output.contains("Page One"));
    let forward = execute(&fixture.registry, "web_forward", json!({})).await;
    assert!(forward.success);
    assert!(forward.output.contains("Page Two"));
    let missing = execute(
        &fixture.registry,
        "web_fetch",
        json!({"url":format!("http://{address}/missing")}),
    )
    .await;
    assert!(!missing.success);
    assert!(
        missing
            .error
            .as_deref()
            .is_some_and(|error| error.contains("404"))
    );
    server.abort();

    let audit = fs::read_to_string(&fixture.audit_path).expect("shell audit");
    assert!(audit.contains("shell-ok"));
}

#[tokio::test]
async fn ingestion_and_cognition_tools_round_trip_real_heart_state() {
    let fixture = fixture(true, false);
    let document = "I am 32 years old. I love cobalt hedgehogs.";
    fs::create_dir_all(fixture._directory.path().join("documents")).expect("document directory");
    fs::write(fixture._directory.path().join("documents/one.md"), document)
        .expect("first document");
    fs::write(fixture._directory.path().join("documents/two.md"), document)
        .expect("duplicate document");

    let ingested = execute(
        &fixture.registry,
        "ingest_documents",
        json!({
            "paths":["documents"],
            "recursive":true,
            "maintain":false,
            "chunk_words":40,
            "overlap_words":0
        }),
    )
    .await;
    assert!(ingested.success, "{:?}", ingested.error);
    assert!(ingested.output.contains("files_discovered=2"));
    assert!(ingested.output.contains("chunks_ingested=1"));
    assert!(ingested.output.contains("skipped=1"));

    let repeated = execute(
        &fixture.registry,
        "ingest_documents",
        json!({"paths":["documents"],"maintain":false}),
    )
    .await;
    assert!(repeated.success);
    assert!(repeated.output.contains("chunks_ingested=0"));
    assert!(repeated.output.contains("skipped=2"));

    for recall_name in ["heart_recall", "search_memory"] {
        let recalled = execute(
            &fixture.registry,
            recall_name,
            json!({"query":"cobalt hedgehogs","top_k":5}),
        )
        .await;
        assert!(recalled.success, "{recall_name}: {:?}", recalled.error);
        assert!(
            recalled.output.contains("cobalt hedgehogs"),
            "{recall_name}"
        );
    }

    let fact = execute(
        &fixture.registry,
        "fact_search",
        json!({"query":"age","top_k":5}),
    )
    .await;
    assert!(fact.success);
    assert!(fact.output.contains("profile.age"));
    assert!(fact.output.contains("32"));
    let latest = execute(
        &fixture.registry,
        "fact_aggregate",
        json!({"slot_prefix":"profile.age","operation":"latest"}),
    )
    .await;
    assert!(latest.success);
    assert!(latest.output.contains("32"));

    let saved = execute(
        &fixture.registry,
        "save_memory",
        json!({"text":"the model explicitly recorded a smoke-test memory"}),
    )
    .await;
    assert!(saved.success, "{:?}", saved.error);
    assert!(saved.output.contains("Saved unverified model memory"));
    let feeling = execute(
        &fixture.registry,
        "feel",
        json!({"context":"tool reliability"}),
    )
    .await;
    assert!(feeling.success);
    assert!(!feeling.output.contains("\"available\":false"));

    let heart_stats = execute(&fixture.registry, "heart_stats", json!({})).await;
    let memory_stats = execute(&fixture.registry, "memory_stats", json!({})).await;
    assert!(heart_stats.success && memory_stats.success);
    assert_eq!(heart_stats.output, memory_stats.output);
    assert_eq!(
        serde_json::from_str::<Value>(&heart_stats.output)
            .expect("stats JSON")
            .get("events")
            .and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(fixture.heart.stats().expect("heart stats").events, 2);

    let maintained = execute(
        &fixture.registry,
        "maintain_memory",
        json!({"maximum_rounds":1}),
    )
    .await;
    assert!(maintained.success);
    let maintenance: Value = serde_json::from_str(&maintained.output).expect("maintenance JSON");
    assert!(maintenance.get("merges").is_some());
}

#[tokio::test]
async fn subagent_tools_start_report_finish_and_cancel() {
    let mut registry = ToolRegistry::default();
    agent_tools::register_subagent_tools(
        &mut registry,
        Arc::new(FinalProvider),
        ToolRegistry::default(),
        3,
        2,
    )
    .expect("register final sub-agent tools");
    let delegated = execute(
        &registry,
        "delegate",
        json!({"task":"return a deterministic report","check_in_every":1,"max_ticks":2}),
    )
    .await;
    assert!(delegated.success);
    assert!(delegated.output.contains("Sub-agent sa1 started"));
    let mut reports = String::new();
    for _ in 0..50 {
        let result = execute(&registry, "check_results", json!({})).await;
        reports.push_str(&result.output);
        if reports.contains("[sa1 | FINAL]") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        reports.contains("sub-agent completed its bounded task"),
        "{reports}"
    );
    let mut cleaned_up = false;
    for _ in 0..4 {
        let result = execute(&registry, "check_results", json!({})).await;
        if result.output == "(no sub-agents running)" {
            cleaned_up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(cleaned_up);

    let mut pending_registry = ToolRegistry::default();
    agent_tools::register_subagent_tools(
        &mut pending_registry,
        Arc::new(PendingProvider),
        ToolRegistry::default(),
        3,
        2,
    )
    .expect("register pending sub-agent tools");
    let delegated = execute(
        &pending_registry,
        "delegate",
        json!({"task":"wait until cancelled","max_ticks":2}),
    )
    .await;
    assert!(delegated.output.contains("sa1"));
    let cancelled = execute(&pending_registry, "cancel_agent", json!({"agent_id":"sa1"})).await;
    assert!(cancelled.success);
    assert_eq!(cancelled.output, "Cancelled sub-agent sa1");
}

#[tokio::test]
async fn harness_blocks_a_real_destructive_shell_call_by_default() {
    let fixture = fixture(false, false);
    let victim = fixture._directory.path().join("victim.txt");
    fs::write(&victim, "keep me").expect("victim file");
    let provider = Arc::new(ScriptedProvider {
        turns: Mutex::new(VecDeque::from([
            ModelTurn {
                tool_calls: vec![ToolCall {
                    id: "destructive".into(),
                    name: "shell".into(),
                    arguments: json!({"command":"rm victim.txt"}),
                }],
                ..ModelTurn::default()
            },
            ModelTurn {
                content: "The destructive operation was blocked.".into(),
                ..ModelTurn::default()
            },
        ])),
    });
    let harness = Harness::new(provider, fixture.registry, HarnessConfig::default())
        .expect("construct harness");
    let outcome = harness
        .run("test system", "try a destructive operation")
        .await
        .expect("harness run");
    assert!(victim.exists());
    assert_eq!(outcome.completed_tool_calls, 1);
    assert!(outcome.messages.iter().any(|message| {
        message.role == MessageRole::Tool
            && message
                .content
                .contains("destructive tool call blocked by this harness")
    }));
}
