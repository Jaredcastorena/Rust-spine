#![forbid(unsafe_code)]

mod agent_tools;
#[cfg(test)]
mod checkpoint_recovery_tests;
mod cognition_tools;
mod document_ingest;
mod grounding;
mod longmem;
mod onboarding;
mod partner_tools;
mod resilience;
mod terminal_input;
#[cfg(test)]
mod tool_smoke_tests;
mod web_server;

use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    io::{self, IsTerminal, Write},
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use spine_heart::{
    AgentId, CognitiveConfig, Content, ContextLeaf, Embedding, EventId, EventKind, HeartConfig,
    InteractionInput, KeySource, ParticipantRole, Provenance, SemanticEncoder, SignedEvent,
    SpineHeart, ThreadId, ToolExchange,
};
use spine_models::{MiniLmAssets, MiniLmEncoder};
use spine_runtime::{
    Harness, HarnessCheckpoint, HarnessConfig, HarnessEvent, HarnessPolicy, LlamaCppConfig,
    LlamaCppProvider, Message, MessageRole, ModulationConfig, ModulationInput, RunOutcome,
    ToolCall, ToolRegistry,
};

use resilience::{CircuitBreaker, ResilienceChannel};

#[derive(Parser)]
#[command(
    name = "spine",
    version,
    about = "Single-binary encrypted AI partner for the terminal and browser"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a new encrypted heart.
    Create {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
    },
    /// Show encrypted-heart storage statistics.
    Stats {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
    },
    /// Save a named encrypted snapshot.
    Snapshot {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
        #[arg(long)]
        label: Option<String>,
    },
    /// Initialize native cognitive state for an existing heart.
    CognitionInit {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        thymos_channels: usize,
    },
    /// Store one message in native cognitive memory.
    Remember {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "main")]
        agent: String,
        #[arg(long, default_value = "default")]
        thread: String,
        text: String,
    },
    /// Search native cognitive memory.
    Recall {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        #[arg(long)]
        show_events: bool,
        #[arg(long, default_value_t = 4)]
        max_events_per_node: usize,
        query: String,
    },
    /// Import a LongMemEval dataset into an empty heart.
    LongMemIngest {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value_t = NonZeroUsize::new(32).expect("nonzero"))]
        embedding_batch_size: NonZeroUsize,
    },
    /// Talk to Spine in the terminal or embedded browser.
    Chat {
        #[arg(
            value_name = "HEART",
            env = "SPINE_HEART_PATH",
            hide_env_values = true,
            help = "Encrypted heart path (defaults to the platform data directory)"
        )]
        path: Option<PathBuf>,
        #[arg(
            long,
            visible_alias = "test-mode",
            help = "Use a temporary heart that is deleted on exit"
        )]
        incognito_mode: bool,
        #[arg(
            long = "web",
            visible_alias = "web-server",
            help = "Serve the embedded browser interface"
        )]
        web_server: bool,
        #[arg(
            long,
            default_value = "127.0.0.1",
            requires = "web_server",
            help = "Address for the embedded web interface"
        )]
        web_host: String,
        #[arg(
            long,
            default_value_t = 8_088,
            requires = "web_server",
            help = "Port for the embedded web interface"
        )]
        web_port: u16,
        #[arg(
            long,
            requires = "web_server",
            help = "Allow a non-loopback web bind (HTTP token authentication is not TLS)"
        )]
        allow_remote_web: bool,
        #[arg(
            long,
            env = "SPINE_HEART_PASSPHRASE",
            hide_env_values = true,
            help = "Heart passphrase (prefer the environment or prompt)"
        )]
        passphrase: Option<String>,
        #[arg(
            long,
            env = "SPINE_MINILM_DIR",
            hide_env_values = true,
            help = "MiniLM snapshot directory"
        )]
        model_dir: Option<PathBuf>,
        #[arg(
            long,
            env = "SPINE_LLAMA_MODEL",
            hide_env_values = true,
            requires = "llama_server_bin",
            help = "GGUF model to serve in a managed local llama.cpp process"
        )]
        llama_model: Option<PathBuf>,
        #[arg(
            long,
            env = "SPINE_LLAMA_SERVER",
            hide_env_values = true,
            requires = "llama_model",
            help = "llama-server executable to start and stop with this session"
        )]
        llama_server_bin: Option<PathBuf>,
        #[arg(
            long,
            default_value_t = -1,
            allow_hyphen_values = true,
            help = "GPU layers for a managed llama-server (-1 means all)"
        )]
        gpu_layers: i32,
        #[arg(
            long,
            env = "SPINE_LLM_URL",
            hide_env_values = true,
            default_value = "http://127.0.0.1:8080",
            help = "OpenAI-compatible server base URL"
        )]
        server_url: String,
        #[arg(long, env = "SPINE_LLM_API_KEY", hide = true, hide_env_values = true)]
        api_key: Option<String>,
        #[arg(
            long,
            env = "SPINE_LLM_MODEL",
            hide_env_values = true,
            help = "Provider model name when the endpoint requires one"
        )]
        server_model: Option<String>,
        #[arg(
            long,
            env = "SPINE_REASONING_EFFORT",
            hide_env_values = true,
            help = "Optional provider reasoning-effort value"
        )]
        reasoning_effort: Option<String>,
        #[arg(long, default_value = "main")]
        agent: String,
        #[arg(long, default_value = "interactive")]
        thread: String,
        #[arg(long, help = "Optional positive ceiling for tool rounds")]
        max_tool_rounds: Option<NonZeroU64>,
        #[arg(long, default_value_t = 8_192)]
        max_tokens: i64,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 7_200)]
        timeout_seconds: u64,
        #[arg(long, default_value_t = 2)]
        provider_retries: u32,
        #[arg(
            long,
            env = "SPINE_MAX_CONTEXT_TOKENS",
            hide_env_values = true,
            help = "Provider context-window ceiling"
        )]
        max_context_tokens: Option<usize>,
        #[arg(
            long,
            env = "SPINE_NLI_DIR",
            hide_env_values = true,
            help = "Optional local NLI snapshot directory"
        )]
        nli_model_dir: Option<PathBuf>,
        #[arg(
            long,
            conflicts_with = "nli_model_dir",
            help = "Disable answer grounding explicitly"
        )]
        no_nli: bool,
        #[arg(long, help = "Permit the model to request unverified memory writes")]
        allow_model_memory_writes: bool,
        #[arg(long, help = "Skip the first-heart getting-to-know-you conversation")]
        skip_onboarding: bool,
        #[arg(
            long,
            default_value_t = 12,
            help = "Conversation turns retained in the live provider context"
        )]
        max_history_turns: usize,
        #[arg(
            long,
            default_value_t = 64_000,
            help = "Character budget retained in the live provider context"
        )]
        max_history_chars: usize,
    },
    /// Execute one non-interactive harness task.
    HarnessRun {
        path: PathBuf,
        #[arg(long, help = "Heart passphrase (prefer the environment or prompt)")]
        passphrase: Option<String>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(
            long,
            env = "SPINE_LLM_URL",
            hide_env_values = true,
            default_value = "http://127.0.0.1:8080"
        )]
        server_url: String,
        #[arg(long, env = "SPINE_LLM_API_KEY", hide = true, hide_env_values = true)]
        api_key: Option<String>,
        #[arg(long, env = "SPINE_LLM_MODEL", hide_env_values = true)]
        server_model: Option<String>,
        #[arg(
            long,
            env = "SPINE_REASONING_EFFORT",
            hide_env_values = true,
            help = "Optional provider reasoning-effort value"
        )]
        reasoning_effort: Option<String>,
        #[arg(long, default_value = "main")]
        agent: String,
        #[arg(long, default_value = "harness")]
        thread: String,
        #[arg(long)]
        max_tool_rounds: Option<NonZeroU64>,
        #[arg(long, default_value_t = 2_048)]
        max_tokens: i64,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 7_200)]
        timeout_seconds: u64,
        #[arg(long, default_value_t = 2)]
        provider_retries: u32,
        #[arg(long, env = "SPINE_MAX_CONTEXT_TOKENS", hide_env_values = true)]
        max_context_tokens: Option<usize>,
        task: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Create { path, passphrase } => {
            let passphrase = resolve_heart_passphrase(passphrase, true)?;
            let created = SpineHeart::create(HeartConfig::new(&path), &passphrase)?;
            println!("created {}", created.heart.path().display());
            println!("recovery phrase: {}", created.recovery_phrase.expose());
        }
        Command::Stats { path, passphrase } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart =
                SpineHeart::open(HeartConfig::new(&path), KeySource::Passphrase(passphrase))?;
            let stats = heart.stats()?;
            println!(
                "events={} blobs={} snapshots={} tombstones={}",
                stats.events, stats.blobs, stats.snapshots, stats.tombstones
            );
        }
        Command::Snapshot {
            path,
            passphrase,
            label,
        } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart =
                SpineHeart::open(HeartConfig::new(&path), KeySource::Passphrase(passphrase))?;
            println!("{}", heart.snapshot(label)?);
        }
        Command::CognitionInit {
            path,
            passphrase,
            model_dir,
            thymos_channels,
        } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart =
                SpineHeart::open(HeartConfig::new(&path), KeySource::Passphrase(passphrase))?;
            let encoder = MiniLmEncoder::load(MiniLmAssets::from_directory(model_dir), 256)?;
            heart.initialize_cognition(CognitiveConfig::new(
                1,
                encoder.manifest().clone(),
                thymos_channels,
            )?)?;
            println!("initialized cognitive projection generation 1");
        }
        Command::Remember {
            path,
            passphrase,
            model_dir,
            agent,
            thread,
            text,
        } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart =
                SpineHeart::open(HeartConfig::new(&path), KeySource::Passphrase(passphrase))?;
            let encoder = MiniLmEncoder::load(MiniLmAssets::from_directory(model_dir), 256)?;
            let interaction = InteractionInput {
                agent_id: AgentId::new(agent)?,
                thread_id: ThreadId::new(thread)?,
                role: ParticipantRole::User,
                kind: EventKind::Message,
                content: Content::Inline(text.clone()),
                causal_parents: Vec::new(),
                provenance: Provenance::default(),
                tool: None,
                attachments: Vec::new(),
                outcome: None,
            };
            let (commit, memory) = heart.commit_embedded(interaction, encoder.encode(&text)?)?;
            println!("event={} node={}", commit.event.id, memory.node_id);
        }
        Command::Recall {
            path,
            passphrase,
            model_dir,
            top_k,
            show_events,
            max_events_per_node,
            query,
        } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart =
                SpineHeart::open(HeartConfig::new(&path), KeySource::Passphrase(passphrase))?;
            let encoder = MiniLmEncoder::load(MiniLmAssets::from_directory(model_dir), 256)?;
            let embedding = encoder.encode(&query)?;
            if show_events {
                for (rank, memory) in heart
                    .recall_memories(&embedding, f64::MAX, top_k, max_events_per_node)?
                    .into_iter()
                    .enumerate()
                {
                    println!(
                        "rank={} node={} score={:.6} semantic={:.6} confidence={:.6} tension={}",
                        rank + 1,
                        memory.hit.node_id,
                        memory.hit.score,
                        memory.hit.semantic_score,
                        memory.hit.confidence,
                        memory.hit.tensioned
                    );
                    for event in memory.events {
                        if let Content::Inline(text) = event.body.interaction.content {
                            println!(
                                "event={} source={} text={}",
                                event.id,
                                event
                                    .body
                                    .interaction
                                    .provenance
                                    .source_uri
                                    .as_deref()
                                    .unwrap_or("-"),
                                serde_json::to_string(&text)?
                            );
                        }
                    }
                }
            } else {
                for (rank, hit) in heart
                    .recall(&embedding, f64::MAX, top_k)?
                    .into_iter()
                    .enumerate()
                {
                    println!(
                        "rank={} node={} score={:.6} semantic={:.6} confidence={:.6} tension={}",
                        rank + 1,
                        hit.node_id,
                        hit.score,
                        hit.semantic_score,
                        hit.confidence,
                        hit.tensioned
                    );
                }
            }
        }
        Command::LongMemIngest {
            path,
            passphrase,
            model_dir,
            dataset,
            embedding_batch_size,
        } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart =
                SpineHeart::open(HeartConfig::new(&path), KeySource::Passphrase(passphrase))?;
            let stats = heart.stats()?;
            if stats.events != 0 {
                return Err(format!(
                    "LongMemEval ingestion requires an empty heart; found {} events",
                    stats.events
                )
                .into());
            }
            let encoder = MiniLmEncoder::load(MiniLmAssets::from_directory(model_dir), 256)?;
            let state = heart
                .cognition()?
                .ok_or("cognitive projection is not initialized")?;
            if !state.is_current(&heart.sync_frontier()?.devices) {
                return Err("cognitive projection is stale".into());
            }
            if &state.config.model != encoder.manifest() {
                return Err("embedder does not match the initialized cognitive projection".into());
            }

            let started = Instant::now();
            let corpus = longmem::load(&dataset)?;
            println!(
                "loaded questions={} unique_sessions={} chunks={}",
                corpus.question_count,
                corpus.session_count,
                corpus.chunks.len()
            );
            let mut embeddings = Vec::with_capacity(corpus.chunks.len());
            for (batch_index, batch) in corpus.chunks.chunks(embedding_batch_size.get()).enumerate()
            {
                let texts = batch
                    .iter()
                    .map(|chunk| chunk.text.clone())
                    .collect::<Vec<_>>();
                embeddings.extend(encoder.encode_batch(&texts)?);
                let embedded = embeddings.len();
                if batch_index == 0 || embedded == corpus.chunks.len() || embedded % 512 == 0 {
                    println!("embedded {embedded}/{}", corpus.chunks.len());
                }
            }
            if embeddings.len() != corpus.chunks.len() {
                return Err("embedder returned the wrong number of vectors".into());
            }

            let agent_id = AgentId::new("longmemeval")?;
            let mut items = Vec::with_capacity(corpus.chunks.len());
            for (chunk, embedding) in corpus.chunks.into_iter().zip(embeddings) {
                let mut metadata = BTreeMap::new();
                metadata.insert("dataset".into(), "LongMemEval".into());
                metadata.insert("session_id".into(), chunk.session_id.clone());
                metadata.insert("session_index".into(), chunk.session_index.to_string());
                metadata.insert("date".into(), chunk.date.clone());
                metadata.insert("session_time".into(), chunk.date.replace('/', "-"));
                metadata.insert("chunk_index".into(), chunk.chunk_index.to_string());
                metadata.insert("has_answer".into(), chunk.has_answer.to_string());
                let source_uri = format!("longmemeval://session/{}", chunk.session_id);
                items.push((
                    InteractionInput {
                        agent_id: agent_id.clone(),
                        thread_id: ThreadId::new(chunk.session_id)?,
                        role: ParticipantRole::User,
                        kind: EventKind::Message,
                        content: Content::Inline(chunk.text),
                        causal_parents: Vec::new(),
                        provenance: Provenance {
                            provider: Some("LongMemEval".into()),
                            source_uri: Some(source_uri),
                            metadata,
                            ..Provenance::default()
                        },
                        tool: None,
                        attachments: Vec::new(),
                        outcome: None,
                    },
                    embedding,
                ));
            }
            println!("committing and projecting {} chunks", items.len());
            let receipts = heart.commit_embedded_batch(items)?;
            let state = heart
                .cognition()?
                .ok_or("cognitive projection disappeared")?;
            let maximum_node_events = state
                .dcmdb
                .nodes
                .values()
                .map(|node| node.event_ids.len())
                .max()
                .unwrap_or_default();
            println!(
                "ingested={} events={} active_nodes={} absorbed_nodes={} max_events_per_node={} elapsed_seconds={:.3}",
                receipts.len(),
                heart.stats()?.events,
                state.dcmdb.nodes.len(),
                state.dcmdb.absorbed.len(),
                maximum_node_events,
                started.elapsed().as_secs_f64()
            );
        }
        Command::Chat {
            path,
            incognito_mode,
            web_server: enable_web_server,
            web_host,
            web_port,
            allow_remote_web,
            passphrase,
            model_dir,
            llama_model,
            llama_server_bin,
            gpu_layers,
            server_url,
            api_key,
            server_model,
            reasoning_effort,
            agent,
            thread,
            max_tool_rounds,
            max_tokens,
            temperature,
            timeout_seconds,
            provider_retries,
            max_context_tokens,
            nli_model_dir,
            no_nli,
            allow_model_memory_writes,
            skip_onboarding,
            max_history_turns,
            max_history_chars,
        } => {
            let heart_target = ChatHeartTarget::resolve(path, incognito_mode)?;
            let path = &heart_target.path;
            let model_dir = resolve_required_model_directory(model_dir)?;
            let nli_model_dir = if no_nli {
                None
            } else {
                resolve_optional_nli_directory(nli_model_dir)?
            };
            let passphrase = if incognito_mode {
                ephemeral_passphrase()?
            } else {
                resolve_heart_passphrase(passphrase, !path.exists())?
            };
            let encoder = Arc::new(MiniLmEncoder::load(
                MiniLmAssets::from_directory(model_dir),
                256,
            )?);
            let (heart, created_new_heart) = if path.exists() {
                (
                    SpineHeart::open(
                        HeartConfig::new(path),
                        KeySource::Passphrase(passphrase.clone()),
                    )?,
                    false,
                )
            } else {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let created = SpineHeart::create(HeartConfig::new(path), &passphrase)?;
                created.heart.initialize_cognition(CognitiveConfig::new(
                    1,
                    encoder.manifest().clone(),
                    8,
                )?)?;
                if incognito_mode {
                    println!("Incognito mode: temporary encrypted heart; nothing will persist");
                } else {
                    println!("created new encrypted heart: {}", path.display());
                    println!("recovery phrase: {}", created.recovery_phrase.expose());
                }
                (created.heart, true)
            };
            if catch_up_stale_cognition(&heart, encoder.as_ref())? {
                eprintln!("[caught up stale cognitive projection from the canonical event log]");
            }
            if heart.upgrade_fact_projection()? {
                eprintln!("[upgraded typed facts from the canonical event log]");
            }
            let heart = Arc::new(heart);
            let heart_was_empty = heart.stats()?.events == 0;
            let agent_id = AgentId::new(agent)?;
            let thread_id = ThreadId::new(thread)?;
            let resolved_api_key = api_key.filter(|value| !value.is_empty()).or_else(|| {
                std::env::var("SPINE_LLM_API_KEY")
                    .ok()
                    .filter(|value| !value.is_empty())
            });
            let mut managed_server = match (llama_server_bin, llama_model) {
                (Some(binary), Some(model)) => Some(ManagedLlamaServer::start(
                    &binary,
                    &model,
                    &server_url,
                    gpu_layers,
                    max_context_tokens.unwrap_or(115_968),
                    resolved_api_key.as_deref(),
                    path.with_extension("llama-server.log"),
                )?),
                (None, None) => None,
                _ => {
                    return Err("--llama-model and --llama-server-bin must be used together".into());
                }
            };
            let mut provider_config = LlamaCppConfig::new(server_url);
            provider_config.api_key = resolved_api_key;
            provider_config.model = server_model;
            provider_config.reasoning_effort = reasoning_effort;
            provider_config.max_tokens = max_tokens;
            provider_config.temperature = temperature;
            provider_config.timeout = Duration::from_secs(timeout_seconds);
            provider_config.maximum_retries = provider_retries;
            provider_config.max_context_tokens = max_context_tokens;
            let provider = LlamaCppProvider::new(provider_config)?;
            if let Some(server) = managed_server.as_mut() {
                wait_for_managed_server(&provider, server, Duration::from_secs(120)).await?;
            } else {
                provider.health().await?;
            }
            report_provider_settings(&provider);
            let provider = Arc::new(provider);
            let grounding = nli_model_dir
                .map(grounding::GroundingGate::load)
                .transpose()?;

            let mut registry = ToolRegistry::default();
            cognition_tools::register_cognition_tools(
                &mut registry,
                Arc::clone(&heart),
                Arc::clone(&encoder),
                allow_model_memory_writes,
            )?;
            let running_tasks = partner_tools::register_action_tools(
                &mut registry,
                Arc::clone(&heart),
                Arc::clone(&encoder),
                std::env::current_dir()?,
                path.with_extension("action-audit.jsonl"),
            )?;
            let cognitive_config = heart
                .cognition()?
                .ok_or("heart has no cognitive projection")?
                .config;
            let child_registry = registry.clone();
            agent_tools::register_subagent_tools(
                &mut registry,
                provider.clone(),
                child_registry,
                cognitive_config.model.dimension,
                cognitive_config.thymos_channels,
            )?;
            let mut harness = Harness::new(
                provider.clone(),
                registry,
                HarnessConfig {
                    max_tool_rounds,
                    ..HarnessConfig::default()
                },
            )?
            .with_agent_id(agent_id.clone());

            let (line_sender, mut line_receiver) = tokio::sync::mpsc::unbounded_channel();
            let web_server = if enable_web_server {
                Some(
                    web_server::start(
                        web_server::WebBind {
                            host: &web_host,
                            port: web_port,
                            allow_remote: allow_remote_web,
                        },
                        line_sender.clone(),
                        web_heart_label(path, incognito_mode),
                        incognito_mode,
                        grounding.is_some(),
                        harness.registry().len(),
                    )
                    .await?,
                )
            } else {
                None
            };
            let web_ui = web_server.as_ref().map(web_server::WebServer::ui);
            if let Some(server) = &web_server {
                println!("Spine web UI: {}", server.access_url);
            }

            let status = Arc::new(TerminalStatus::new());
            let event_status = Arc::clone(&status);
            let event_web = web_ui.clone();
            let mut events = harness.subscribe();
            let event_task = tokio::spawn(async move {
                while let Ok(event) = events.recv().await {
                    match event {
                        HarnessEvent::ToolStarted { id, name } => {
                            event_status.show(tool_activity(&name));
                            if let Some(web) = &event_web {
                                web.activity(tool_activity(&name));
                                web.tool_started(&id, &name);
                            }
                        }
                        HarnessEvent::ToolCompleted { id, success, .. } => {
                            event_status.tool_completed();
                            if let Some(web) = &event_web {
                                web.tool_completed(&id, success);
                            }
                        }
                        HarnessEvent::GuidanceInjected { .. } => {
                            event_status.show("Applying guidance");
                            if let Some(web) = &event_web {
                                web.activity("Applying guidance");
                            }
                        }
                        HarnessEvent::GracefulStopBoundary => {
                            event_status.show("Stopping safely");
                            if let Some(web) = &event_web {
                                web.activity("Stopping safely");
                            }
                        }
                        HarnessEvent::ModelTurnCompleted => {
                            event_status.show("Thinking");
                            if let Some(web) = &event_web {
                                web.activity("Thinking");
                            }
                        }
                    }
                }
            });

            terminal_input::spawn(line_sender.clone());

            let onboarding_state = onboarding::OnboardingState::inspect(&heart.events_canonical()?);
            let should_onboard =
                created_new_heart || heart_was_empty || onboarding_state.in_progress();
            let mut interaction_profile = onboarding_state.profile.clone();
            let mut quit_after_onboarding = false;
            if should_onboard {
                if skip_onboarding {
                    onboarding::record_skipped(
                        &heart,
                        encoder.as_ref(),
                        &agent_id,
                        &thread_id,
                        "first conversation skipped by operator flag",
                    )?;
                    let message = "No problem—we can learn how to work together as we go.";
                    println!("spine> {message}");
                    if let Some(web) = &web_ui {
                        web.finish_onboarding(message, true);
                    }
                } else {
                    let result = run_first_conversation(
                        provider.as_ref(),
                        &heart,
                        encoder.as_ref(),
                        &agent_id,
                        &thread_id,
                        onboarding_state,
                        &mut line_receiver,
                        web_ui.as_ref(),
                        status.as_ref(),
                    )
                    .await?;
                    interaction_profile = result.profile.or(interaction_profile);
                    quit_after_onboarding = result.quit;
                }
            }
            let partner_system_prompt = format!(
                "{PARTNER_SYSTEM_PROMPT}{}",
                onboarding::profile_context(interaction_profile.as_ref())
            );

            let restored_checkpoint =
                discover_persisted_checkpoint(&heart.events_canonical()?, &agent_id, &thread_id);
            let mut checkpoint = match restored_checkpoint {
                CheckpointDiscovery::Available(resumable) => Some(resumable),
                CheckpointDiscovery::None => None,
                CheckpointDiscovery::Rejected(reason) => {
                    eprintln!("[persisted checkpoint ignored: {reason}]");
                    if let Some(web) = &web_ui {
                        web.notice(format!("Persisted checkpoint ignored: {reason}"));
                    }
                    None
                }
            };
            if let Some(web) = &web_ui {
                web.set_checkpoint_available(checkpoint.is_some());
            }

            if !quit_after_onboarding {
                println!(
                    "Spine ready: Rust heart={} events={} tools={} grounding={} (/tasks, /circuit, /stop, /interrupt, /resume, /quit)",
                    path.display(),
                    heart.stats()?.events,
                    harness.registry().len(),
                    if grounding.is_some() {
                        "NLI"
                    } else {
                        "disabled"
                    },
                );
                if checkpoint.is_some() {
                    println!("[resumable checkpoint restored; use /resume to continue]");
                }
            }
            let mut history = Vec::<Message>::new();
            let mut completed_turns = 0_u64;
            let mut circuit_breaker = CircuitBreaker::default();
            if !quit_after_onboarding {
                loop {
                    print!("you> ");
                    io::stdout().flush()?;
                    let Some(line) = line_receiver.recv().await else {
                        println!();
                        break;
                    };
                    let task = line?;
                    let task = task.trim();
                    if matches!(task, "/quit" | "/exit") {
                        break;
                    }
                    if task.is_empty() {
                        continue;
                    }
                    if matches!(task, "reset" | "reset cb" | "reset circuit breaker") {
                        circuit_breaker.reset_all();
                        println!("[circuit breakers reset]");
                        if let Some(web) = &web_ui {
                            web.complete_command("Circuit breakers reset", checkpoint.is_some());
                        }
                        continue;
                    }
                    if let Some(channel) =
                        task.strip_prefix("/reset ").and_then(|name| {
                            match name.trim().to_ascii_lowercase().as_str() {
                                "llm" => Some(ResilienceChannel::Llm),
                                "thymos" => Some(ResilienceChannel::Thymos),
                                "dcmdb" | "memory" => Some(ResilienceChannel::Dcmdb),
                                _ => None,
                            }
                        })
                    {
                        circuit_breaker.reset(channel);
                        println!(
                            "[circuit breaker reset: {}]",
                            task.trim_start_matches("/reset ")
                        );
                        if let Some(web) = &web_ui {
                            web.complete_command(
                                format!(
                                    "Circuit breaker reset: {}",
                                    task.trim_start_matches("/reset ")
                                ),
                                checkpoint.is_some(),
                            );
                        }
                        continue;
                    }
                    if task == "/circuit" {
                        let summary = circuit_breaker.status_summary();
                        println!("{summary}");
                        if let Some(web) = &web_ui {
                            web.complete_command(summary, checkpoint.is_some());
                        }
                        continue;
                    }
                    if task == "/tasks" {
                        let tasks = running_tasks.format();
                        println!("{tasks}");
                        if let Some(web) = &web_ui {
                            web.complete_command(tasks, checkpoint.is_some());
                        }
                        continue;
                    }
                    if let Some(task_id) = task.strip_prefix("/cancel-task ") {
                        let result = running_tasks.cancel(task_id.trim());
                        println!("{result}");
                        if let Some(web) = &web_ui {
                            web.complete_command(result, checkpoint.is_some());
                        }
                        continue;
                    }
                    let is_resume = task == "/resume";
                    if is_resume && checkpoint.is_none() {
                        println!("[no resumable checkpoint]");
                        if let Some(web) = &web_ui {
                            web.notice("No resumable checkpoint");
                            web.complete("", false, false);
                        }
                        continue;
                    }
                    if !circuit_breaker.available(ResilienceChannel::Llm, Instant::now()) {
                        let notice = format!(
                            "LLM circuit is open; retry after cooldown or use /reset llm ({})",
                            circuit_breaker.status_summary()
                        );
                        eprintln!("[{notice}]");
                        if let Some(web) = &web_ui {
                            web.complete_command(notice, checkpoint.is_some());
                        }
                        continue;
                    }
                    if !circuit_breaker.available(ResilienceChannel::Dcmdb, Instant::now()) {
                        let notice = format!(
                            "memory circuit is open; retry after cooldown or use /reset dcmdb ({})",
                            circuit_breaker.status_summary()
                        );
                        eprintln!("[{notice}]");
                        if let Some(web) = &web_ui {
                            web.complete_command(notice, checkpoint.is_some());
                        }
                        continue;
                    }
                    if !is_resume {
                        if let Some(web) = &web_ui {
                            web.begin_turn(task);
                        }
                        status.begin("Checking memory");
                        let task_embedding = match encoder.encode(task) {
                            Ok(embedding) => embedding,
                            Err(error) => {
                                circuit_breaker
                                    .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                let notice = format!(
                                    "task embedding unavailable: {error}; turn was not sent"
                                );
                                eprintln!("[{notice}]");
                                if let Some(web) = &web_ui {
                                    web.fail(&notice);
                                }
                                status.end();
                                continue;
                            }
                        };
                        let triangle_context = if circuit_breaker
                            .allow(ResilienceChannel::Dcmdb, Instant::now())
                        {
                            match cognition_tools::rehydrate_triangle_context(
                                &heart,
                                &task_embedding,
                            ) {
                                Ok(context) => {
                                    circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                                    context
                                }
                                Err(error) => {
                                    circuit_breaker
                                        .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                    eprintln!("[triangle context unavailable: {error}]");
                                    "[]".into()
                                }
                            }
                        } else {
                            "[]".into()
                        };
                        if !circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
                            let notice = "memory circuit opened during recall; turn was not sent";
                            eprintln!("[{notice}]");
                            if let Some(web) = &web_ui {
                                web.fail(notice);
                            }
                            status.end();
                            continue;
                        }
                        let user_commit = match commit_text(
                            &heart,
                            &encoder,
                            &agent_id,
                            &thread_id,
                            ParticipantRole::User,
                            EventKind::Message,
                            task,
                            None,
                        ) {
                            Ok(commit) => {
                                circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                                commit
                            }
                            Err(error) => {
                                circuit_breaker
                                    .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                let recovered = catch_up_stale_cognition(&heart, encoder.as_ref())
                                    .unwrap_or(false);
                                let notice = format!(
                                    "memory write failed: {error}; projection_caught_up={recovered}; turn was not sent",
                                );
                                eprintln!("[{notice}]");
                                if let Some(web) = &web_ui {
                                    web.fail(&notice);
                                }
                                status.end();
                                continue;
                            }
                        };
                        let modulation_config = ModulationConfig::default();
                        let preliminary_policy = modulation_config.compute(ModulationInput {
                            surprise: user_commit.1.trajectory.surprise,
                            valence: user_commit.1.feeling.valence,
                            arousal: user_commit.1.feeling.arousal,
                            risk: 0.0,
                            tensions: 0,
                            base_temperature: temperature,
                            configured_tool_rounds: max_tool_rounds,
                        });
                        let mut automatic_recall = resilient_automatic_recall(
                            &heart,
                            &task_embedding,
                            task,
                            preliminary_policy.recall_top_k,
                            user_commit.1.event_id,
                            preliminary_policy.recall_expansion_depth(),
                            &mut circuit_breaker,
                        );
                        let risk_estimate = if circuit_breaker
                            .allow(ResilienceChannel::Thymos, Instant::now())
                        {
                            match heart.predict_risk(
                                &agent_id,
                                &task_embedding,
                                &automatic_recall.risk_stats(),
                            ) {
                                Ok(risk) if risk.is_finite() => {
                                    circuit_breaker.record_success(ResilienceChannel::Thymos);
                                    Some(risk)
                                }
                                result => {
                                    circuit_breaker
                                        .record_failure(ResilienceChannel::Thymos, Instant::now());
                                    eprintln!("[host risk estimate unavailable: {result:?}]");
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        let risk = risk_estimate.unwrap_or(1.0);
                        let risk_context = risk_estimate.map_or_else(
                            || "unavailable; conservative high-risk policy enforced".into(),
                            |risk| format!("{risk:.3}"),
                        );
                        let policy = modulation_config.compute(ModulationInput {
                            surprise: user_commit.1.trajectory.surprise,
                            valence: user_commit.1.feeling.valence,
                            arousal: user_commit.1.feeling.arousal,
                            risk,
                            tensions: automatic_recall.tension_count(),
                            base_temperature: temperature,
                            configured_tool_rounds: max_tool_rounds,
                        });
                        if policy.recall_top_k > preliminary_policy.recall_top_k
                            || policy.recall_expansion_depth()
                                > preliminary_policy.recall_expansion_depth()
                        {
                            automatic_recall = resilient_automatic_recall(
                                &heart,
                                &task_embedding,
                                task,
                                policy.recall_top_k,
                                user_commit.1.event_id,
                                policy.recall_expansion_depth(),
                                &mut circuit_breaker,
                            );
                        }
                        let retrieval_stats = automatic_recall.risk_stats();
                        let recalled = automatic_recall.context;
                        let feeling = serde_json::to_string(&user_commit.1.feeling)
                            .unwrap_or_else(|_| "unavailable".into());
                        let harness_policy = HarnessPolicy {
                            temperature: Some(policy.provider_temperature),
                            max_action_calls: NonZeroUsize::new(policy.max_actions),
                            max_tool_rounds: policy.max_tool_rounds,
                        };
                        harness.set_tool_metadata(turn_introspection_metadata(
                            &user_commit.1,
                            &policy,
                            temperature,
                            grounding.is_some(),
                            risk_estimate,
                        )?)?;
                        let system_prompt = format!(
                            "{partner_system_prompt}\n\nCurrent committed Thymos proprioception: {feeling}\nCommitted trajectory surprise: {:.3}. Host risk estimate for this memory region: {risk_context}. Host policy is enforced at recall_k={}, temperature={:.3}, action_budget={}, and coverage_threshold={:.3}. At higher or unavailable risk, deepen recall and avoid unsupported certainty.\n\nAutomatically recalled canonical evidence for this turn (it may be irrelevant; verify before using):\n{recalled}\n\nBudgeted triangle-context rehydration:\n{triangle_context}\n\nRunning/recent host tasks:\n{}",
                            user_commit.1.trajectory.surprise,
                            policy.recall_top_k,
                            policy.provider_temperature,
                            policy.max_actions,
                            policy.coverage_threshold,
                            running_tasks.format()
                        );
                        let persist_start = history.len() + 2;
                        let mut turn_leaves = vec![ContextLeaf {
                            node_id: user_commit.1.node_id,
                            chronology: user_commit.0.event.body.device_sequence,
                        }];
                        let run = Box::pin(harness.run_with_history_policy(
                            system_prompt,
                            &history,
                            task,
                            harness_policy,
                        ))
                            as std::pin::Pin<Box<dyn Future<Output = _>>>;
                        assert!(
                            circuit_breaker.allow(ResilienceChannel::Llm, Instant::now()),
                            "an available LLM circuit accepts its pending call"
                        );
                        status.show("Thinking");
                        let controlled = run_with_operator_controls(
                            &harness,
                            run,
                            &mut line_receiver,
                            status.as_ref(),
                        )
                        .await;
                        let outcome = match controlled.outcome {
                            Ok(Some(outcome)) => {
                                circuit_breaker.record_success(ResilienceChannel::Llm);
                                Some(outcome)
                            }
                            Ok(None) => {
                                circuit_breaker
                                    .record_cancelled(ResilienceChannel::Llm, Instant::now());
                                None
                            }
                            Err(error) => {
                                circuit_breaker
                                    .record_failure(ResilienceChannel::Llm, Instant::now());
                                commit_control_text(
                                    &heart,
                                    &encoder,
                                    &agent_id,
                                    &thread_id,
                                    &format!("active turn failed: {error}"),
                                    "provider_or_harness_error",
                                )?;
                                status.end();
                                if let Some(web) = &web_ui {
                                    web.fail(&error.to_string());
                                }
                                eprintln!("[turn failed: {error}]");
                                if controlled.quit_after {
                                    break;
                                }
                                continue;
                            }
                        };
                        let Some(mut outcome) = outcome else {
                            commit_control_text(
                                &heart,
                                &encoder,
                                &agent_id,
                                &thread_id,
                                "operator interrupted the active turn",
                                "interrupted",
                            )?;
                            status.end();
                            if let Some(web) = &web_ui {
                                web.complete("", true, checkpoint.is_some());
                            }
                            println!("[active turn interrupted]");
                            if controlled.quit_after {
                                break;
                            }
                            continue;
                        };
                        if circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
                            match persist_harness_messages(
                                &heart,
                                &encoder,
                                &agent_id,
                                &thread_id,
                                &outcome.messages[persist_start.min(outcome.messages.len())..],
                            ) {
                                Ok(leaves) => {
                                    circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                                    turn_leaves.extend(leaves);
                                }
                                Err(error) => {
                                    circuit_breaker
                                        .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                    let recovered =
                                        catch_up_stale_cognition(&heart, encoder.as_ref())
                                            .unwrap_or(false);
                                    eprintln!(
                                        "[turn-message persistence incomplete: {error}; projection_caught_up={recovered}]"
                                    );
                                }
                            }
                        } else {
                            eprintln!("[turn-message persistence deferred: DCMDB circuit is open]");
                        }
                        let mut quit_after = controlled.quit_after;
                        if let Some(gate) = &grounding
                            && !outcome.response.trim().is_empty()
                            && !outcome.stopped_gracefully
                        {
                            status.show("Verifying answer");
                            if let Some(web) = &web_ui {
                                web.activity("Verifying answer");
                            }
                            let evidence = grounding::evidence_from_recall_and_messages(
                                &recalled,
                                &outcome.messages,
                            );
                            let decision = gate.verify(
                                &outcome.response,
                                &evidence,
                                policy.coverage_threshold,
                            );
                            if let Err(error) = &decision {
                                commit_control_text(
                                    &heart,
                                    &encoder,
                                    &agent_id,
                                    &thread_id,
                                    &format!("grounding verification failed: {error}"),
                                    "grounding_verifier_error",
                                )?;
                                status.show("Verifier unavailable; using answer");
                                if let Some(web) = &web_ui {
                                    web.notice("Grounding verifier unavailable; using answer");
                                }
                            }
                            if let Ok(decision) = decision {
                                if let Some(tension) = decision.risk_target()
                                    && circuit_breaker
                                        .allow(ResilienceChannel::Thymos, Instant::now())
                                {
                                    match heart.update_risk(
                                        &agent_id,
                                        &task_embedding,
                                        &retrieval_stats,
                                        tension,
                                    ) {
                                        Ok(_) => circuit_breaker
                                            .record_success(ResilienceChannel::Thymos),
                                        Err(error) => {
                                            circuit_breaker.record_failure(
                                                ResilienceChannel::Thymos,
                                                Instant::now(),
                                            );
                                            eprintln!("[risk update unavailable: {error}]");
                                        }
                                    }
                                }
                                if decision.needs_repair {
                                    status.show("Repairing answer");
                                    if let Some(web) = &web_ui {
                                        web.activity("Repairing answer");
                                    }
                                    let repair_start = outcome.messages.len() + 1;
                                    let repair_task = format!(
                                        "[HOST GROUNDING REPAIR] The draft's factual coverage was {:.3} and contradiction risk was {:.3}. Re-check the supplied evidence and tool results. Use more recall/tools if needed, correct unsupported claims, and abstain explicitly where evidence remains insufficient. Return the corrected final answer.",
                                        decision.report.coverage, decision.report.contradiction
                                    );
                                    assert!(
                                        circuit_breaker
                                            .allow(ResilienceChannel::Llm, Instant::now()),
                                        "a successful draft leaves the LLM circuit available for repair"
                                    );
                                    let repair = Box::pin(harness.repair(&outcome, repair_task))
                                        as std::pin::Pin<Box<dyn Future<Output = _>>>;
                                    let repaired = run_with_operator_controls(
                                        &harness,
                                        repair,
                                        &mut line_receiver,
                                        status.as_ref(),
                                    )
                                    .await;
                                    quit_after |= repaired.quit_after;
                                    match repaired.outcome {
                                        Ok(Some(mut repaired_outcome)) => {
                                            circuit_breaker.record_success(ResilienceChannel::Llm);
                                            let repaired_evidence =
                                                grounding::evidence_from_recall_and_messages(
                                                    &recalled,
                                                    &repaired_outcome.messages,
                                                );
                                            let still_unverified = gate
                                                .verify(
                                                    &repaired_outcome.response,
                                                    &repaired_evidence,
                                                    policy.coverage_threshold,
                                                )
                                                .map_or(true, |decision| decision.needs_repair);
                                            if still_unverified {
                                                grounding::append_terminal_caveat(
                                                    &mut repaired_outcome,
                                                );
                                                status
                                                    .show("Answer caveated after grounding repair");
                                                if let Some(web) = &web_ui {
                                                    web.notice(concat!(
                                                    "Some claims remained unverified after repair; ",
                                                    "a terminal caveat was added"
                                                ));
                                                }
                                            }
                                            if circuit_breaker
                                                .allow(ResilienceChannel::Dcmdb, Instant::now())
                                            {
                                                match persist_harness_messages(
                                                    &heart,
                                                    &encoder,
                                                    &agent_id,
                                                    &thread_id,
                                                    &repaired_outcome.messages[repair_start
                                                        .min(repaired_outcome.messages.len())..],
                                                ) {
                                                    Ok(leaves) => {
                                                        circuit_breaker.record_success(
                                                            ResilienceChannel::Dcmdb,
                                                        );
                                                        turn_leaves.extend(leaves);
                                                    }
                                                    Err(error) => {
                                                        circuit_breaker.record_failure(
                                                            ResilienceChannel::Dcmdb,
                                                            Instant::now(),
                                                        );
                                                        let recovered = catch_up_stale_cognition(
                                                            &heart,
                                                            encoder.as_ref(),
                                                        )
                                                        .unwrap_or(false);
                                                        eprintln!(
                                                            "[repair-message persistence incomplete: {error}; projection_caught_up={recovered}]"
                                                        );
                                                    }
                                                }
                                            } else {
                                                eprintln!(
                                                    "[repair-message persistence deferred: DCMDB circuit is open]"
                                                );
                                            }
                                            outcome = repaired_outcome;
                                        }
                                        Ok(None) => {
                                            circuit_breaker.record_cancelled(
                                                ResilienceChannel::Llm,
                                                Instant::now(),
                                            );
                                            commit_control_text(
                                                &heart,
                                                &encoder,
                                                &agent_id,
                                                &thread_id,
                                                "operator interrupted the grounding repair",
                                                "interrupted",
                                            )?;
                                            status.show("Repair interrupted; using draft");
                                            if let Some(web) = &web_ui {
                                                web.notice("Repair interrupted; using draft");
                                            }
                                        }
                                        Err(error) => {
                                            circuit_breaker.record_failure(
                                                ResilienceChannel::Llm,
                                                Instant::now(),
                                            );
                                            commit_control_text(
                                                &heart,
                                                &encoder,
                                                &agent_id,
                                                &thread_id,
                                                &format!("grounding repair failed: {error}"),
                                                "grounding_repair_error",
                                            )?;
                                            status.show("Repair unavailable; using draft");
                                            if let Some(web) = &web_ui {
                                                web.notice("Repair unavailable; using draft");
                                            }
                                        }
                                    }
                                } else if decision.claim_count == 0 {
                                    status.show("Answer ready");
                                } else {
                                    status.show("Answer verified");
                                }
                            }
                        }
                        if !turn_leaves.is_empty() {
                            status.show("Saving context");
                            if let Some(web) = &web_ui {
                                web.activity("Saving context");
                            }
                            if circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
                                if let Err(error) = heart.compact_context(turn_leaves, 6) {
                                    circuit_breaker
                                        .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                    eprintln!("[context compaction deferred: {error}]");
                                } else {
                                    circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                                }
                            } else {
                                eprintln!("[context compaction deferred: DCMDB circuit is open]");
                            }
                        }
                        completed_turns = completed_turns.saturating_add(1);
                        if completed_turns.is_multiple_of(10) {
                            status.show("Maintaining memory");
                            if let Some(web) = &web_ui {
                                web.activity("Maintaining memory");
                            }
                            if circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
                                if let Err(error) = heart.maintain_cognition(4) {
                                    circuit_breaker
                                        .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                    eprintln!("[memory maintenance deferred: {error}]");
                                } else {
                                    circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                                }
                            } else {
                                eprintln!("[memory maintenance deferred: DCMDB circuit is open]");
                            }
                        }
                        if let Err(error) = checkpoint_from_outcome(
                            &heart,
                            encoder.as_ref(),
                            &agent_id,
                            &thread_id,
                            &outcome,
                            &mut checkpoint,
                            &mut circuit_breaker,
                        ) {
                            checkpoint = None;
                            outcome.checkpoint = None;
                            let notice = format!(
                                "checkpoint persistence could not be confirmed: {error}; do not rely on /resume until restart"
                            );
                            eprintln!("[{notice}]");
                            if let Some(web) = &web_ui {
                                web.notice(notice);
                            }
                        }
                        history.push(Message::new(MessageRole::User, task));
                        status.end();
                        if let Some(web) = &web_ui {
                            web.capture_messages(&outcome.messages);
                            web.complete(
                                &outcome.response,
                                outcome.stopped_gracefully,
                                outcome.checkpoint.is_some(),
                            );
                        }
                        finish_visible_turn(
                            &outcome,
                            &mut history,
                            max_history_turns,
                            max_history_chars,
                        );
                        if quit_after {
                            break;
                        }
                        continue;
                    }
                    let persist_start;
                    let run = if is_resume {
                        let resumable = match prepare_checkpoint_resume(
                            &heart,
                            encoder.as_ref(),
                            &agent_id,
                            &thread_id,
                            &mut checkpoint,
                            &mut circuit_breaker,
                        ) {
                            Ok(resumable) => resumable,
                            Err(error) => {
                                let notice = format!("Checkpoint resume was not started: {error}");
                                eprintln!("[{notice}]");
                                if let Some(web) = &web_ui {
                                    web.complete_command(notice, checkpoint.is_some());
                                }
                                continue;
                            }
                        };
                        persist_start = resumable.messages.len() + 1;
                        // A different turn may have replaced the host snapshot since
                        // this checkpoint was saved. Only its effective policy is known.
                        harness.set_tool_metadata(BTreeMap::from([(
                            "spine_modulation".into(),
                            serde_json::to_string(&harness.policy_for_checkpoint(&resumable))?,
                        )]))?;
                        assert!(
                            circuit_breaker.allow(ResilienceChannel::Llm, Instant::now()),
                            "an available LLM circuit accepts its pending resume"
                        );
                        Box::pin(harness.resume(resumable))
                            as std::pin::Pin<Box<dyn Future<Output = _>>>
                    } else {
                        unreachable!("new turns are handled above")
                    };
                    if let Some(web) = &web_ui {
                        web.begin_resume();
                    }
                    status.begin("Resuming");
                    let controlled = run_with_operator_controls(
                        &harness,
                        run,
                        &mut line_receiver,
                        status.as_ref(),
                    )
                    .await;
                    let outcome = match controlled.outcome {
                        Ok(Some(outcome)) => {
                            circuit_breaker.record_success(ResilienceChannel::Llm);
                            Some(outcome)
                        }
                        Ok(None) => {
                            circuit_breaker
                                .record_cancelled(ResilienceChannel::Llm, Instant::now());
                            None
                        }
                        Err(error) => {
                            circuit_breaker.record_failure(ResilienceChannel::Llm, Instant::now());
                            commit_control_text(
                                &heart,
                                &encoder,
                                &agent_id,
                                &thread_id,
                                &format!("resumed turn failed: {error}"),
                                "provider_or_harness_error",
                            )?;
                            status.end();
                            if let Some(web) = &web_ui {
                                web.fail(&error.to_string());
                            }
                            eprintln!("[resumed turn failed: {error}]");
                            if controlled.quit_after {
                                break;
                            }
                            continue;
                        }
                    };
                    let Some(mut outcome) = outcome else {
                        commit_control_text(
                            &heart,
                            &encoder,
                            &agent_id,
                            &thread_id,
                            "operator interrupted the active turn",
                            "interrupted",
                        )?;
                        status.end();
                        if let Some(web) = &web_ui {
                            web.complete("", true, checkpoint.is_some());
                        }
                        println!("[active turn interrupted]");
                        if controlled.quit_after {
                            break;
                        }
                        continue;
                    };
                    let leaves = if circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now())
                    {
                        match persist_harness_messages(
                            &heart,
                            &encoder,
                            &agent_id,
                            &thread_id,
                            &outcome.messages[persist_start.min(outcome.messages.len())..],
                        ) {
                            Ok(leaves) => {
                                circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                                leaves
                            }
                            Err(error) => {
                                circuit_breaker
                                    .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                let recovered = catch_up_stale_cognition(&heart, encoder.as_ref())
                                    .unwrap_or(false);
                                eprintln!(
                                    "[resumed-message persistence incomplete: {error}; projection_caught_up={recovered}]"
                                );
                                Vec::new()
                            }
                        }
                    } else {
                        eprintln!("[resumed-message persistence deferred: DCMDB circuit is open]");
                        Vec::new()
                    };
                    if !leaves.is_empty() {
                        status.show("Saving context");
                        if circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
                            if let Err(error) = heart.compact_context(leaves, 6) {
                                circuit_breaker
                                    .record_failure(ResilienceChannel::Dcmdb, Instant::now());
                                eprintln!("[context compaction deferred: {error}]");
                            } else {
                                circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                            }
                        } else {
                            eprintln!("[context compaction deferred: DCMDB circuit is open]");
                        }
                    }
                    if let Err(error) = checkpoint_from_outcome(
                        &heart,
                        encoder.as_ref(),
                        &agent_id,
                        &thread_id,
                        &outcome,
                        &mut checkpoint,
                        &mut circuit_breaker,
                    ) {
                        checkpoint = None;
                        outcome.checkpoint = None;
                        let notice = format!(
                            "checkpoint persistence could not be confirmed: {error}; do not rely on /resume until restart"
                        );
                        eprintln!("[{notice}]");
                        if let Some(web) = &web_ui {
                            web.notice(notice);
                        }
                    }
                    status.end();
                    if let Some(web) = &web_ui {
                        web.capture_messages(&outcome.messages);
                        web.complete(
                            &outcome.response,
                            outcome.stopped_gracefully,
                            outcome.checkpoint.is_some(),
                        );
                    }
                    finish_visible_turn(
                        &outcome,
                        &mut history,
                        max_history_turns,
                        max_history_chars,
                    );
                    if controlled.quit_after {
                        break;
                    }
                }
            }
            event_task.abort();
            if incognito_mode {
                println!("Incognito session ended; temporary heart discarded");
            } else {
                let snapshot = heart.snapshot(Some("interactive-exit".into()))?;
                println!("saved encrypted heart snapshot={snapshot}");
            }
            if let Some(server) = web_server {
                server.shutdown().await;
            }
        }
        Command::HarnessRun {
            path,
            passphrase,
            model_dir,
            server_url,
            api_key,
            server_model,
            reasoning_effort,
            agent,
            thread,
            max_tool_rounds,
            max_tokens,
            temperature,
            timeout_seconds,
            provider_retries,
            max_context_tokens,
            task,
        } => {
            let passphrase = resolve_heart_passphrase(passphrase, false)?;
            let heart = Arc::new(SpineHeart::open(
                HeartConfig::new(&path),
                KeySource::Passphrase(passphrase),
            )?);
            let encoder = Arc::new(MiniLmEncoder::load(
                MiniLmAssets::from_directory(model_dir),
                256,
            )?);
            catch_up_stale_cognition(&heart, encoder.as_ref())?;
            heart.upgrade_fact_projection()?;
            let agent_id = AgentId::new(agent)?;
            let thread_id = ThreadId::new(thread)?;
            let mut provider_config = LlamaCppConfig::new(server_url);
            provider_config.api_key = api_key.filter(|value| !value.is_empty()).or_else(|| {
                std::env::var("SPINE_LLM_API_KEY")
                    .ok()
                    .filter(|value| !value.is_empty())
            });
            provider_config.model = server_model;
            provider_config.reasoning_effort = reasoning_effort;
            provider_config.max_tokens = max_tokens;
            provider_config.temperature = temperature;
            provider_config.timeout = Duration::from_secs(timeout_seconds);
            provider_config.maximum_retries = provider_retries;
            provider_config.max_context_tokens = max_context_tokens;
            let provider = Arc::new(LlamaCppProvider::new(provider_config)?);
            provider.health().await?;
            report_provider_settings(&provider);
            let onboarding_state = onboarding::OnboardingState::inspect(&heart.events_canonical()?);
            let partner_system_prompt = format!(
                "You are a long-running Spine partner operating against an encrypted heart. Use the supplied heart tools whenever the task requests stored facts or store state. Treat tool output as authoritative, do not fabricate results, and give a concise final answer after completing the requested checks.{}",
                onboarding::profile_context(onboarding_state.profile.as_ref())
            );

            let mut registry = ToolRegistry::default();
            cognition_tools::register_cognition_tools(
                &mut registry,
                Arc::clone(&heart),
                Arc::clone(&encoder),
                false,
            )?;
            let _running_tasks = partner_tools::register_action_tools(
                &mut registry,
                Arc::clone(&heart),
                Arc::clone(&encoder),
                std::env::current_dir()?,
                path.with_extension("action-audit.jsonl"),
            )?;
            let cognitive_config = heart
                .cognition()?
                .ok_or("heart has no cognitive projection")?
                .config;
            let child_registry = registry.clone();
            agent_tools::register_subagent_tools(
                &mut registry,
                provider.clone(),
                child_registry,
                cognitive_config.model.dimension,
                cognitive_config.thymos_channels,
            )?;
            let mut harness = Harness::new(
                provider,
                registry,
                HarnessConfig {
                    max_tool_rounds,
                    ..HarnessConfig::default()
                },
            )?
            .with_agent_id(agent_id.clone());

            let task_embedding = encoder.encode(&task)?;
            let triangle_context =
                cognition_tools::rehydrate_triangle_context(&heart, &task_embedding)?;
            let user_commit = commit_text(
                &heart,
                &encoder,
                &agent_id,
                &thread_id,
                ParticipantRole::User,
                EventKind::Message,
                &task,
                None,
            )?;
            let modulation_config = ModulationConfig::default();
            let preliminary_policy = modulation_config.compute(ModulationInput {
                surprise: user_commit.1.trajectory.surprise,
                valence: user_commit.1.feeling.valence,
                arousal: user_commit.1.feeling.arousal,
                risk: 0.0,
                tensions: 0,
                base_temperature: temperature,
                configured_tool_rounds: max_tool_rounds,
            });
            let mut automatic_recall = cognition_tools::automatic_recall_context(
                &heart,
                &task_embedding,
                &task,
                preliminary_policy.recall_top_k,
                user_commit.1.event_id,
                preliminary_policy.recall_expansion_depth(),
            )?;
            let risk =
                heart.predict_risk(&agent_id, &task_embedding, &automatic_recall.risk_stats())?;
            let policy = modulation_config.compute(ModulationInput {
                surprise: user_commit.1.trajectory.surprise,
                valence: user_commit.1.feeling.valence,
                arousal: user_commit.1.feeling.arousal,
                risk,
                tensions: automatic_recall.tension_count(),
                base_temperature: temperature,
                configured_tool_rounds: max_tool_rounds,
            });
            if policy.recall_top_k > preliminary_policy.recall_top_k
                || policy.recall_expansion_depth() > preliminary_policy.recall_expansion_depth()
            {
                automatic_recall = cognition_tools::automatic_recall_context(
                    &heart,
                    &task_embedding,
                    &task,
                    policy.recall_top_k,
                    user_commit.1.event_id,
                    policy.recall_expansion_depth(),
                )?;
            }
            harness.set_tool_metadata(turn_introspection_metadata(
                &user_commit.1,
                &policy,
                temperature,
                false,
                Some(risk),
            )?)?;
            let system_prompt = format!(
                "{partner_system_prompt}\n\nCurrent committed Thymos proprioception: {}\nCommitted trajectory surprise: {:.3}. Host risk estimate: {risk:.3}. Host policy is enforced at recall_k={}, temperature={:.3}, and action_budget={}.\n\nAutomatically recalled canonical evidence:\n{}\n\nBudgeted triangle-context rehydration:\n{triangle_context}",
                serde_json::to_string(&user_commit.1.feeling)
                    .unwrap_or_else(|_| "unavailable".into()),
                user_commit.1.trajectory.surprise,
                policy.recall_top_k,
                policy.provider_temperature,
                policy.max_actions,
                automatic_recall.context,
            );
            let outcome = harness
                .run_with_history_policy(
                    system_prompt,
                    &[],
                    &task,
                    HarnessPolicy {
                        temperature: Some(policy.provider_temperature),
                        max_action_calls: NonZeroUsize::new(policy.max_actions),
                        max_tool_rounds: policy.max_tool_rounds,
                    },
                )
                .await?;
            persist_harness_messages(
                &heart,
                &encoder,
                &agent_id,
                &thread_id,
                &outcome.messages[2..],
            )?;
            println!("{}", outcome.response);
            println!(
                "harness={} tool_calls={} tool_rounds={} prompt_tokens={} completion_tokens={}",
                harness.id(),
                outcome.completed_tool_calls,
                outcome.completed_tool_rounds,
                outcome.usage.prompt,
                outcome.usage.completion
            );
        }
    }
    Ok(())
}

const PARTNER_SYSTEM_PROMPT: &str = "You are Spine, a long-running partner backed by an encrypted portable heart. Preserve continuity with the supplied conversation history. Use heart_recall or fact tools whenever a request may depend on older conversations or stored facts. Use action tools to complete requested work, treat tool output as authoritative, never fabricate tool results or memories, and continue testing until the requested outcome is genuinely handled. Destructive actions remain host-gated. Give a clear final response after completing any needed tool calls.";

struct ManagedLlamaServer {
    child: std::process::Child,
    log_path: PathBuf,
}

impl ManagedLlamaServer {
    #[allow(clippy::too_many_arguments)]
    fn start(
        binary: &std::path::Path,
        model: &std::path::Path,
        server_url: &str,
        gpu_layers: i32,
        context_tokens: usize,
        api_key: Option<&str>,
        log_path: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if !binary.is_file() {
            return Err(format!("llama-server binary not found: {}", binary.display()).into());
        }
        if !model.is_file() {
            return Err(format!("GGUF model not found: {}", model.display()).into());
        }
        let endpoint = reqwest::Url::parse(server_url)?;
        let host = endpoint
            .host_str()
            .ok_or("managed llama-server URL requires a host")?;
        let port = endpoint
            .port_or_known_default()
            .ok_or("managed llama-server URL requires a port")?;
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        let error_log = log.try_clone()?;
        let mut command = std::process::Command::new(binary);
        command.args(llama_server_args(
            model,
            host,
            port,
            gpu_layers,
            context_tokens,
        ));
        command
            .env_remove("SPINE_HEART_PASSPHRASE")
            .env_remove("SPINE_LLM_API_KEY");
        if let Some(api_key) = api_key.filter(|value| !value.is_empty()) {
            command.env("LLAMA_API_KEY", api_key);
        }
        let child = command
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(error_log))
            .spawn()?;
        println!(
            "starting managed llama-server model={} log={}",
            model.display(),
            log_path.display()
        );
        Ok(Self { child, log_path })
    }
}

impl Drop for ManagedLlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn llama_server_args(
    model: &std::path::Path,
    host: &str,
    port: u16,
    gpu_layers: i32,
    context_tokens: usize,
) -> Vec<std::ffi::OsString> {
    vec![
        "-m".into(),
        model.as_os_str().to_owned(),
        "-ngl".into(),
        gpu_layers.to_string().into(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string().into(),
        "-c".into(),
        context_tokens.max(256).to_string().into(),
        "--jinja".into(),
    ]
}

async fn wait_for_managed_server(
    provider: &LlamaCppProvider,
    server: &mut ManagedLlamaServer,
    timeout: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        if provider.health().await.is_ok() {
            println!("managed llama-server ready");
            return Ok(());
        }
        if let Some(status) = server.child.try_wait()? {
            return Err(format!(
                "llama-server exited with {status}; inspect {}",
                server.log_path.display()
            )
            .into());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "llama-server did not become ready; inspect {}",
                server.log_path.display()
            )
            .into());
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn report_provider_settings(provider: &LlamaCppProvider) {
    let model = provider.model().map(|model| {
        let model = model
            .rsplit('/')
            .next()
            .unwrap_or(model)
            .rsplit('\\')
            .next()
            .unwrap_or(model);
        model
            .chars()
            .filter(|character| !character.is_control())
            .take(120)
            .collect::<String>()
    });
    match (model.as_deref(), provider.context_tokens()) {
        (Some(model), Some(tokens)) => {
            println!("provider: model={model} context={tokens} tokens")
        }
        (Some(model), None) => println!("provider: model={model} context=not advertised"),
        (None, Some(tokens)) => println!("provider: model=server-default context={tokens} tokens"),
        (None, None) => println!("provider: model=server-default context=not advertised"),
    }
    if provider
        .context_tokens()
        .is_some_and(|tokens| tokens < 8_192)
    {
        eprintln!(
            "warning: the provider's runtime context is below 8192 tokens; onboarding can fit, but full tool turns may exhaust it (16384 or more recommended)"
        );
    }
}

struct ChatHeartTarget {
    path: PathBuf,
    _temporary: Option<tempfile::TempDir>,
}

fn web_heart_label(path: &std::path::Path, incognito: bool) -> String {
    if incognito {
        return "temporary incognito heart".into();
    }
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "persistent heart".into())
}

fn resolve_heart_passphrase(provided: Option<String>, creating: bool) -> io::Result<String> {
    if let Some(value) = provided.filter(|value| !value.is_empty()).or_else(|| {
        std::env::var("SPINE_HEART_PASSPHRASE")
            .ok()
            .filter(|value| !value.is_empty())
    }) {
        return Ok(value);
    }
    if !io::stdin().is_terminal() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "set SPINE_HEART_PASSPHRASE or run from a terminal to enter it securely",
        ));
    }
    let prompt = if creating {
        "Choose heart passphrase: "
    } else {
        "Heart passphrase: "
    };
    let value = rpassword::prompt_password(prompt)?;
    if value.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "heart passphrase cannot be empty",
        ));
    }
    if creating {
        let confirmation = rpassword::prompt_password("Confirm heart passphrase: ")?;
        if value != confirmation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "heart passphrases did not match",
            ));
        }
    }
    Ok(value)
}

fn default_heart_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SPINE_HEART_PATH").filter(|value| !value.is_empty()) {
        return Some(expand_home(PathBuf::from(path)));
    }
    if cfg!(target_os = "windows") {
        if let Some(local) = std::env::var_os("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            return Some(local.join("Spine/default.spine"));
        }
        return user_home_directory().map(|home| home.join("AppData/Local/Spine/default.spine"));
    }
    if cfg!(target_os = "macos") {
        return user_home_directory()
            .map(|home| home.join("Library/Application Support/Spine/default.spine"));
    }
    if let Some(data) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(data).join("spine/default.spine"));
    }
    user_home_directory().map(|home| home.join(".local/share/spine/default.spine"))
}

fn user_home_directory() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn expand_home(path: PathBuf) -> PathBuf {
    let value = path.to_string_lossy();
    if value == "~" {
        return user_home_directory().unwrap_or(path);
    }
    if let Some(suffix) = value.strip_prefix("~/")
        && let Some(home) = user_home_directory()
    {
        return home.join(suffix);
    }
    path
}

fn resolve_required_model_directory(explicit: Option<PathBuf>) -> io::Result<PathBuf> {
    let configured = explicit.or_else(|| {
        std::env::var_os("SPINE_MINILM_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    if let Some(path) = configured {
        return validate_model_directory(expand_home(path), "MiniLM");
    }
    cached_model_directories("models--sentence-transformers--all-MiniLM-L6-v2")
        .into_iter()
        .find(|path| model_assets_present(path))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "MiniLM assets not found; pass --model-dir or set SPINE_MINILM_DIR",
            )
        })
}

fn resolve_optional_nli_directory(explicit: Option<PathBuf>) -> io::Result<Option<PathBuf>> {
    let configured = explicit.or_else(|| {
        std::env::var_os("SPINE_NLI_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    if let Some(path) = configured {
        return validate_model_directory(expand_home(path), "NLI").map(Some);
    }
    Ok(
        cached_model_directories("models--cross-encoder--nli-MiniLM2-L6-H768")
            .into_iter()
            .find(|path| model_assets_present(path)),
    )
}

fn validate_model_directory(path: PathBuf, label: &str) -> io::Result<PathBuf> {
    let missing_assets_error = || {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{label} model directory {} must contain config.json, tokenizer.json, and model.safetensors",
                path.display()
            ),
        )
    };
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        let asset = path.join(name);
        match std::fs::metadata(&asset) {
            Ok(metadata) if metadata.is_file() => {}
            Ok(_) => return Err(missing_assets_error()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(missing_assets_error());
            }
            Err(error) => {
                return Err(io::Error::new(
                    error.kind(),
                    format!(
                        "cannot read {label} model asset {}: {error}",
                        asset.display()
                    ),
                ));
            }
        }
        if let Err(error) = std::fs::File::open(&asset) {
            if error.kind() == io::ErrorKind::NotFound {
                return Err(missing_assets_error());
            }
            return Err(io::Error::new(
                error.kind(),
                format!(
                    "cannot read {label} model asset {}: {error}",
                    asset.display()
                ),
            ));
        }
    }
    Ok(path)
}

fn model_assets_present(path: &std::path::Path) -> bool {
    ["config.json", "tokenizer.json", "model.safetensors"]
        .into_iter()
        .all(|name| path.join(name).is_file())
}

fn cached_model_directories(repository_cache_name: &str) -> Vec<PathBuf> {
    let mut hubs = Vec::new();
    if let Some(path) = std::env::var_os("HF_HUB_CACHE").filter(|value| !value.is_empty()) {
        hubs.push(expand_home(PathBuf::from(path)));
    }
    if let Some(path) = std::env::var_os("HF_HOME").filter(|value| !value.is_empty()) {
        hubs.push(expand_home(PathBuf::from(path)).join("hub"));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        hubs.push(PathBuf::from(path).join("huggingface/hub"));
    }
    if let Some(home) = user_home_directory() {
        hubs.push(home.join(".cache/huggingface/hub"));
    }
    let mut seen_hubs = BTreeSet::new();
    hubs.retain(|path| seen_hubs.insert(path.clone()));

    let mut candidates = Vec::new();
    for hub in hubs {
        let snapshots = hub.join(repository_cache_name).join("snapshots");
        if let Ok(entries) = std::fs::read_dir(snapshots) {
            let mut cached = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>();
            cached.sort();
            cached.reverse();
            candidates.extend(cached);
        }
    }
    candidates
}

impl ChatHeartTarget {
    fn resolve(path: Option<PathBuf>, incognito_mode: bool) -> io::Result<Self> {
        if incognito_mode {
            let temporary = tempfile::Builder::new()
                .prefix("rust-spine-incognito-")
                .tempdir()?;
            let path = temporary.path().join("incognito.spine");
            return Ok(Self {
                path,
                _temporary: Some(temporary),
            });
        }
        let path = path.or_else(default_heart_path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "persistent chat requires a heart path or a platform home/data directory",
            )
        })?;
        Ok(Self {
            path: expand_home(path),
            _temporary: None,
        })
    }
}

fn ephemeral_passphrase() -> Result<String, getrandom::Error> {
    let mut random = [0_u8; 32];
    getrandom::fill(&mut random)?;
    Ok(hex::encode(random))
}

struct FirstConversationResult {
    profile: Option<onboarding::InteractionProfile>,
    quit: bool,
}

#[allow(clippy::too_many_arguments)]
async fn run_first_conversation(
    provider: &dyn spine_runtime::ModelProvider,
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    state: onboarding::OnboardingState,
    lines: &mut tokio::sync::mpsc::UnboundedReceiver<io::Result<String>>,
    web: Option<&web_server::WebUi>,
    status: &TerminalStatus,
) -> Result<FirstConversationResult, Box<dyn std::error::Error>> {
    let waiting_for_answer = state.waiting_for_answer();
    let mut transcript = state.transcript;
    let mut answer_count = state.answers;
    let mut pending_question = waiting_for_answer.then(|| {
        transcript
            .last()
            .expect("waiting question exists")
            .content
            .clone()
    });

    loop {
        let question = if let Some(question) = pending_question.take() {
            question
        } else {
            status.begin("Getting acquainted");
            if let Some(web) = web {
                web.begin_onboarding_model_turn();
            }
            let turn = match onboarding::generate_turn(provider, &transcript, answer_count).await {
                Ok(turn) => turn,
                Err(error) => {
                    status.end();
                    onboarding::record_pending(
                        heart,
                        encoder,
                        agent_id,
                        thread_id,
                        &format!("first conversation paused after provider error: {error}"),
                    )?;
                    eprintln!(
                        "[first conversation paused: {error}; we can learn each other while working]"
                    );
                    if let Some(web) = web {
                        web.pause_onboarding(
                            "The first conversation is paused; you can start working normally.",
                        );
                    }
                    return Ok(FirstConversationResult {
                        profile: None,
                        quit: false,
                    });
                }
            };
            status.end();
            if turn.complete {
                let profile = turn
                    .profile
                    .ok_or("completed first conversation did not include a profile")?;
                onboarding::record_profile_and_closing(
                    heart,
                    encoder,
                    agent_id,
                    thread_id,
                    &profile,
                    &turn.reply,
                )?;
                println!("spine> {}", turn.reply);
                if let Some(web) = web {
                    web.finish_onboarding(&turn.reply, true);
                }
                return Ok(FirstConversationResult {
                    profile: Some(profile),
                    quit: false,
                });
            }
            onboarding::record_question(heart, encoder, agent_id, thread_id, &turn.reply)?;
            transcript.push(Message::new(MessageRole::Assistant, &turn.reply));
            turn.reply
        };

        status.end();
        println!("spine> {question}");
        if let Some(web) = web {
            web.finish_onboarding(&question, false);
        }

        loop {
            print!("you> ");
            io::stdout().flush()?;
            let Some(line) = lines.recv().await else {
                println!();
                if let Some(web) = web {
                    web.pause_onboarding("First conversation paused until the next launch.");
                }
                return Ok(FirstConversationResult {
                    profile: None,
                    quit: true,
                });
            };
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("[operator input error: {error}]");
                    continue;
                }
            };
            let answer = line.trim();
            if answer.is_empty() {
                continue;
            }
            if matches!(answer, "/quit" | "/exit") {
                if let Some(web) = web {
                    web.pause_onboarding("First conversation paused until the next launch.");
                }
                return Ok(FirstConversationResult {
                    profile: None,
                    quit: true,
                });
            }
            if answer.eq_ignore_ascii_case("/skip") || answer.eq_ignore_ascii_case("skip") {
                onboarding::record_skipped(
                    heart,
                    encoder,
                    agent_id,
                    thread_id,
                    "person chose to learn each other naturally while working",
                )?;
                let message = "Absolutely—we can learn each other naturally while we work.";
                println!("spine> {message}");
                if let Some(web) = web {
                    web.finish_onboarding(message, true);
                }
                return Ok(FirstConversationResult {
                    profile: None,
                    quit: false,
                });
            }
            if answer == "/tasks" {
                let message = "No tasks are running yet; reply naturally, or type /skip.";
                println!("[{message}]");
                if let Some(web) = web {
                    web.notice(message);
                }
                continue;
            }
            if let Some(web) = web {
                web.begin_onboarding_answer(answer);
            }
            onboarding::record_answer(heart, encoder, agent_id, thread_id, answer)?;
            transcript.push(Message::new(MessageRole::User, answer));
            answer_count = answer_count.saturating_add(1);
            break;
        }
    }
}

struct TerminalStatus {
    enabled: bool,
    active: AtomicBool,
    completed_tools: AtomicU64,
    output: Mutex<()>,
}

impl TerminalStatus {
    fn new() -> Self {
        Self {
            enabled: io::stdout().is_terminal(),
            active: AtomicBool::new(false),
            completed_tools: AtomicU64::new(0),
            output: Mutex::new(()),
        }
    }

    fn begin(&self, message: &str) {
        self.completed_tools.store(0, Ordering::Release);
        self.active.store(true, Ordering::Release);
        self.show(message);
    }

    fn show(&self, message: &str) {
        if !self.enabled || !self.active.load(Ordering::Acquire) {
            return;
        }
        let _guard = self.output.lock().expect("terminal status lock poisoned");
        let completed = self.completed_tools.load(Ordering::Acquire);
        let suffix = if completed == 0 {
            String::new()
        } else {
            format!(
                " · {completed} tool{}",
                if completed == 1 { "" } else { "s" }
            )
        };
        print!("\r\x1b[2K\x1b[90m  {message}{suffix}\x1b[0m");
        let _ = io::stdout().flush();
    }

    fn tool_completed(&self) {
        self.completed_tools.fetch_add(1, Ordering::AcqRel);
        self.show("Thinking");
    }

    fn end(&self) {
        self.active.store(false, Ordering::Release);
        if !self.enabled {
            return;
        }
        let _guard = self.output.lock().expect("terminal status lock poisoned");
        print!("\r\x1b[2K");
        let _ = io::stdout().flush();
    }
}

fn tool_activity(name: &str) -> &'static str {
    match name {
        "heart_recall" | "search_memory" | "fact_search" | "fact_aggregate" => "Checking memory",
        "heart_stats" | "memory_stats" | "feel" => "Inspecting heart",
        "shell" => "Running command",
        "file_read" | "file_list" | "file_search" => "Reading workspace",
        "file_write" => "Updating workspace",
        "web_fetch" | "web_search" | "web_navigate" | "web_back" | "web_forward" => "Browsing",
        "ingest_documents" => "Ingesting documents",
        "delegate" => "Starting sub-agent",
        "check_results" => "Checking sub-agent",
        "cancel_agent" | "cancel_task" => "Stopping task",
        "check_tasks" => "Checking tasks",
        "maintain_memory" => "Maintaining memory",
        "save_memory" => "Saving memory",
        _ => "Working",
    }
}

struct ControlledRun {
    outcome: spine_runtime::Result<Option<RunOutcome>>,
    quit_after: bool,
}

async fn run_with_operator_controls<'a>(
    harness: &Harness,
    mut run: std::pin::Pin<Box<dyn Future<Output = spine_runtime::Result<RunOutcome>> + 'a>>,
    lines: &mut tokio::sync::mpsc::UnboundedReceiver<io::Result<String>>,
    status: &TerminalStatus,
) -> ControlledRun {
    let controls = harness.controls();
    let mut quit_after = false;
    loop {
        tokio::select! {
            outcome = &mut run => {
                return ControlledRun { outcome: outcome.map(Some), quit_after };
            }
            line = lines.recv() => {
                let Some(line) = line else {
                    controls.request_graceful_stop();
                    quit_after = true;
                    continue;
                };
                let line = match line {
                    Ok(line) => line,
                    Err(error) => {
                        eprintln!("[operator input error: {error}]");
                        continue;
                    }
                };
                let line = line.trim();
                match line {
                    "/interrupt" => {
                        return ControlledRun {
                            outcome: Ok(None),
                            quit_after: false,
                        };
                    }
                    "/stop" => {
                        controls.request_graceful_stop();
                        status.show("Stopping safely");
                    }
                    "/quit" | "/exit" => {
                        controls.request_graceful_stop();
                        quit_after = true;
                        status.show("Stopping safely, then exiting");
                    }
                    "" => {}
                    guidance => {
                        controls.queue_guidance(guidance);
                        status.show("Guidance queued");
                    }
                }
            }
            signal = tokio::signal::ctrl_c() => {
                if signal.is_ok() {
                    controls.request_graceful_stop();
                    status.show("Stopping safely");
                }
            }
        }
    }
}

fn trim_history(history: &mut Vec<Message>, maximum_turns: usize, maximum_chars: usize) {
    let maximum_messages = maximum_turns.max(1).saturating_mul(2);
    while history.len() > maximum_messages
        || history
            .iter()
            .map(|message| message.content.chars().count())
            .sum::<usize>()
            > maximum_chars.max(1)
    {
        let remove = history.len().min(2);
        history.drain(..remove);
    }
}

fn finish_visible_turn(
    outcome: &RunOutcome,
    history: &mut Vec<Message>,
    maximum_turns: usize,
    maximum_chars: usize,
) {
    if !outcome.response.trim().is_empty() {
        history.push(Message::new(MessageRole::Assistant, &outcome.response));
        trim_history(history, maximum_turns, maximum_chars);
        println!("spine> {}", outcome.response);
    } else if outcome.stopped_gracefully {
        if outcome.checkpoint.is_some() {
            println!("[stopped safely; use /resume to continue]");
        } else {
            println!("[stopped; no confirmed resumable checkpoint]");
        }
    }
}

const CHECKPOINT_CONSUMED_RECORD_TYPE: &str = "harness_checkpoint_consumed";
const CHECKPOINT_CONSUMED_OUTCOME: &str = "harness_checkpoint_consumed";

#[derive(Clone, Debug, PartialEq)]
struct PersistedHarnessCheckpoint {
    checkpoint: HarnessCheckpoint,
    event_id: EventId,
}

#[derive(Debug, PartialEq)]
enum CheckpointDiscovery {
    Available(PersistedHarnessCheckpoint),
    None,
    Rejected(String),
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointConsumption {
    schema: u32,
    checkpoint_event_id: String,
}

fn discover_persisted_checkpoint(
    events: &[SignedEvent],
    agent_id: &AgentId,
    thread_id: &ThreadId,
) -> CheckpointDiscovery {
    let mut consumed = BTreeSet::new();
    for event in events.iter().rev() {
        let interaction = &event.body.interaction;
        if &interaction.agent_id != agent_id || &interaction.thread_id != thread_id {
            continue;
        }
        match interaction
            .provenance
            .metadata
            .get("record_type")
            .map(String::as_str)
        {
            Some(CHECKPOINT_CONSUMED_RECORD_TYPE) => {
                let checkpoint_id = match consumed_checkpoint_id(interaction) {
                    Ok(id) => id,
                    Err(reason) => return CheckpointDiscovery::Rejected(reason),
                };
                consumed.insert(checkpoint_id);
            }
            Some(HarnessCheckpoint::RECORD_TYPE) => {
                if consumed.contains(&event.id) {
                    return CheckpointDiscovery::None;
                }
                return match HarnessCheckpoint::from_interaction(interaction) {
                    Ok(checkpoint) => CheckpointDiscovery::Available(PersistedHarnessCheckpoint {
                        checkpoint,
                        event_id: event.id,
                    }),
                    Err(error) => CheckpointDiscovery::Rejected(format!(
                        "checkpoint {} failed validation: {error}",
                        event.id
                    )),
                };
            }
            _ => {}
        }
    }
    CheckpointDiscovery::None
}

fn consumed_checkpoint_id(interaction: &InteractionInput) -> Result<EventId, String> {
    if interaction.role != ParticipantRole::Operator
        || interaction.kind != EventKind::Control
        || interaction.outcome.as_deref() != Some(CHECKPOINT_CONSUMED_OUTCOME)
        || interaction.tool.is_some()
        || !interaction.attachments.is_empty()
    {
        return Err("checkpoint consumption record has an invalid envelope".into());
    }
    let metadata_id = interaction
        .provenance
        .metadata
        .get("checkpoint_event_id")
        .ok_or_else(|| "checkpoint consumption record is missing its event id".to_owned())?;
    let Content::Inline(content) = &interaction.content else {
        return Err("checkpoint consumption record is not inline JSON".into());
    };
    let record: CheckpointConsumption = serde_json::from_str(content)
        .map_err(|error| format!("invalid checkpoint consumption JSON: {error}"))?;
    if record.schema != 1 || record.checkpoint_event_id != *metadata_id {
        return Err("checkpoint consumption metadata does not match its payload".into());
    }
    metadata_id
        .parse()
        .map_err(|_| "checkpoint consumption event id is invalid".into())
}

fn checkpoint_consumption_interaction(
    resumable: &PersistedHarnessCheckpoint,
    agent_id: AgentId,
    thread_id: ThreadId,
) -> spine_runtime::Result<InteractionInput> {
    let checkpoint_event_id = resumable.event_id.to_string();
    let record = CheckpointConsumption {
        schema: 1,
        checkpoint_event_id: checkpoint_event_id.clone(),
    };
    let mut metadata = BTreeMap::new();
    metadata.insert("record_type".into(), CHECKPOINT_CONSUMED_RECORD_TYPE.into());
    metadata.insert("checkpoint_event_id".into(), checkpoint_event_id);
    Ok(InteractionInput {
        agent_id,
        thread_id,
        role: ParticipantRole::Operator,
        kind: EventKind::Control,
        content: Content::Inline(serde_json::to_string(&record)?),
        causal_parents: Vec::new(),
        provenance: Provenance {
            metadata,
            ..Provenance::default()
        },
        tool: None,
        attachments: Vec::new(),
        outcome: Some(CHECKPOINT_CONSUMED_OUTCOME.into()),
    })
}

fn mark_checkpoint_consumed(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    resumable: &PersistedHarnessCheckpoint,
) -> Result<(), Box<dyn std::error::Error>> {
    let interaction =
        checkpoint_consumption_interaction(resumable, agent_id.clone(), thread_id.clone())?;
    let Content::Inline(text) = &interaction.content else {
        unreachable!("checkpoint consumption records are always inline")
    };
    let embedding = encoder.encode(text)?;
    heart.commit_embedded(interaction, embedding)?;
    Ok(())
}

/// Consume the exact persisted checkpoint before any resumed work can start.
/// A failed projection write may already have committed its canonical marker.
fn prepare_checkpoint_resume(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    checkpoint: &mut Option<PersistedHarnessCheckpoint>,
    circuit_breaker: &mut CircuitBreaker,
) -> Result<HarnessCheckpoint, String> {
    let resumable = checkpoint.take().ok_or("no resumable checkpoint")?;
    if !circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
        *checkpoint = Some(resumable);
        return Err("memory circuit is open; checkpoint remains available after recovery".into());
    }
    match mark_checkpoint_consumed(heart, encoder, agent_id, thread_id, &resumable) {
        Ok(()) => {
            circuit_breaker.record_success(ResilienceChannel::Dcmdb);
            Ok(resumable.checkpoint)
        }
        Err(error) => {
            circuit_breaker.record_failure(ResilienceChannel::Dcmdb, Instant::now());
            let recovery = catch_up_stale_cognition(heart, encoder);
            if let Err(recovery_error) = &recovery {
                circuit_breaker.record_failure(ResilienceChannel::Dcmdb, Instant::now());
                eprintln!("[checkpoint projection recovery unavailable: {recovery_error}]");
            }
            let consumed = heart
                .events_canonical()
                .map_err(|error| error.to_string())
                .and_then(|events| {
                    exact_checkpoint_consumed(&events, agent_id, thread_id, resumable.event_id)
                });
            match consumed {
                Ok(true) => {
                    if recovery.is_ok() {
                        circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                    }
                    eprintln!(
                        "[checkpoint consumption confirmed in canonical events after write error; resuming once]"
                    );
                    Ok(resumable.checkpoint)
                }
                Ok(false) => {
                    *checkpoint = Some(resumable);
                    Err(format!(
                        "{error}; no consumption marker was committed; checkpoint remains available"
                    ))
                }
                Err(canonical_error) => Err(format!(
                    "{error}; checkpoint consumption is indeterminate ({canonical_error}); local resume disabled"
                )),
            }
        }
    }
}

fn exact_checkpoint_consumed(
    events: &[SignedEvent],
    agent_id: &AgentId,
    thread_id: &ThreadId,
    checkpoint_id: EventId,
) -> Result<bool, String> {
    let mut consumed = false;
    for event in events {
        let interaction = &event.body.interaction;
        if &interaction.agent_id == agent_id
            && &interaction.thread_id == thread_id
            && interaction
                .provenance
                .metadata
                .get("record_type")
                .map(String::as_str)
                == Some(CHECKPOINT_CONSUMED_RECORD_TYPE)
        {
            // Validate every candidate: a malformed record makes absence uncertain.
            consumed |= consumed_checkpoint_id(interaction)? == checkpoint_id;
        }
    }
    Ok(consumed)
}

fn checkpoint_from_outcome(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    outcome: &RunOutcome,
    checkpoint: &mut Option<PersistedHarnessCheckpoint>,
    circuit_breaker: &mut CircuitBreaker,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(resumable) = outcome.checkpoint.clone() {
        let interaction = resumable.to_interaction(agent_id.clone(), thread_id.clone())?;
        let text = match &interaction.content {
            Content::Inline(text) => text.clone(),
            Content::ColdBlob(_) | Content::Redacted => String::new(),
        };
        if !circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
            return Err("checkpoint was not saved: DCMDB circuit is open".into());
        }
        let commit = (|| -> Result<EventId, Box<dyn std::error::Error>> {
            let (receipt, _) = heart.commit_embedded(interaction, encoder.encode(&text)?)?;
            Ok(receipt.event.id)
        })();
        let event_id = match commit {
            Ok(id) => id,
            Err(error) => {
                circuit_breaker.record_failure(ResilienceChannel::Dcmdb, Instant::now());
                return Err(error);
            }
        };
        circuit_breaker.record_success(ResilienceChannel::Dcmdb);
        *checkpoint = Some(PersistedHarnessCheckpoint {
            checkpoint: resumable,
            event_id,
        });
    }
    Ok(())
}

fn commit_control_text(
    heart: &SpineHeart,
    encoder: &MiniLmEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    text: &str,
    outcome: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let commit = (|| -> Result<(), Box<dyn std::error::Error>> {
        heart.commit_embedded(
            InteractionInput {
                agent_id: agent_id.clone(),
                thread_id: thread_id.clone(),
                role: ParticipantRole::Operator,
                kind: EventKind::Control,
                content: Content::Inline(text.to_owned()),
                causal_parents: Vec::new(),
                provenance: Provenance::default(),
                tool: None,
                attachments: Vec::new(),
                outcome: Some(outcome.into()),
            },
            encoder.encode(text)?,
        )?;
        Ok(())
    })();
    if let Err(error) = commit {
        let recovered = catch_up_stale_cognition(heart, encoder).unwrap_or(false);
        eprintln!(
            "[control-event persistence unavailable: {error}; projection_caught_up={recovered}]"
        );
    }
    Ok(())
}

fn resilient_automatic_recall(
    heart: &SpineHeart,
    query: &Embedding,
    task: &str,
    top_k: usize,
    excluded_event: EventId,
    expansion_depth: usize,
    circuit_breaker: &mut CircuitBreaker,
) -> cognition_tools::AutomaticRecall {
    if circuit_breaker.allow(ResilienceChannel::Dcmdb, Instant::now()) {
        match cognition_tools::automatic_recall_context(
            heart,
            query,
            task,
            top_k,
            excluded_event,
            expansion_depth,
        ) {
            Ok(recall) => {
                circuit_breaker.record_success(ResilienceChannel::Dcmdb);
                return recall;
            }
            Err(error) => {
                circuit_breaker.record_failure(ResilienceChannel::Dcmdb, Instant::now());
                eprintln!("[memory recall unavailable: {error}]");
            }
        }
    } else {
        eprintln!("[memory recall skipped: DCMDB circuit is open]");
    }
    let layout = heart
        .cognition()
        .ok()
        .flatten()
        .map_or(6, |state| state.config.retrieval_stat_dimensions);
    cognition_tools::AutomaticRecall::empty_for_layout(layout)
}

fn turn_introspection_metadata(
    receipt: &spine_heart::MemoryReceipt,
    policy: &spine_runtime::HostModulation,
    base_temperature: f32,
    nli_enabled: bool,
    risk_estimate: Option<f32>,
) -> Result<BTreeMap<String, String>, serde_json::Error> {
    let mut modulation = serde_json::to_value(policy)?;
    modulation["reason_temp_modifier"] =
        serde_json::json!((base_temperature - policy.provider_temperature).max(0.0));
    Ok(BTreeMap::from([
        (
            "spine_trajectory".into(),
            serde_json::to_string(&receipt.trajectory)?,
        ),
        (
            "spine_modulation".into(),
            serde_json::to_string(&modulation)?,
        ),
        (
            "spine_risk_policy".into(),
            serde_json::json!({
                "risk_estimate": risk_estimate,
                "available": risk_estimate.is_some(),
                "effective_risk": policy.risk,
                "coverage_threshold": policy.coverage_threshold,
                "expansion_probability": policy.expansion_probability,
                "nli_enabled": nli_enabled,
                "risk_enabled": true,
            })
            .to_string(),
        ),
    ]))
}

fn catch_up_stale_cognition(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
) -> Result<bool, Box<dyn std::error::Error>> {
    if heart.cognition_is_current()? {
        return Ok(false);
    }
    heart.catch_up_cognition(encoder)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn commit_text(
    heart: &SpineHeart,
    encoder: &MiniLmEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    role: ParticipantRole,
    kind: EventKind,
    text: &str,
    tool: Option<ToolExchange>,
) -> Result<(spine_heart::CommitReceipt, spine_heart::MemoryReceipt), Box<dyn std::error::Error>> {
    let embedding = encoder.encode(text)?;
    let mut metadata = BTreeMap::new();
    if role == ParticipantRole::User {
        metadata.insert(
            "thymos_learning_multiplier".into(),
            cognition_tools::reflection_multiplier(heart, agent_id, &embedding)?.to_string(),
        );
    }
    Ok(heart.commit_embedded(
        InteractionInput {
            agent_id: agent_id.clone(),
            thread_id: thread_id.clone(),
            role,
            kind,
            content: Content::Inline(text.to_owned()),
            causal_parents: Vec::new(),
            provenance: Provenance {
                provider: Some("llama.cpp".into()),
                metadata,
                ..Provenance::default()
            },
            tool,
            attachments: Vec::new(),
            outcome: None,
        },
        embedding,
    )?)
}

fn persist_harness_messages(
    heart: &SpineHeart,
    encoder: &MiniLmEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    messages: &[Message],
) -> Result<Vec<ContextLeaf>, Box<dyn std::error::Error>> {
    let mut calls = BTreeMap::<String, ToolCall>::new();
    let mut leaves = Vec::new();
    for message in messages {
        match message.role {
            MessageRole::System | MessageRole::User => {}
            MessageRole::Assistant => {
                if !message.content.trim().is_empty() {
                    let commit = commit_text(
                        heart,
                        encoder,
                        agent_id,
                        thread_id,
                        ParticipantRole::Assistant,
                        EventKind::Message,
                        &message.content,
                        None,
                    )?;
                    leaves.push(ContextLeaf {
                        node_id: commit.1.node_id,
                        chronology: commit.0.event.body.device_sequence,
                    });
                }
                for call in &message.tool_calls {
                    calls.insert(call.id.clone(), call.clone());
                    let arguments = serde_json::to_string(&call.arguments)?;
                    let commit = commit_text(
                        heart,
                        encoder,
                        agent_id,
                        thread_id,
                        ParticipantRole::Assistant,
                        EventKind::ToolCall,
                        &format!("{}({arguments})", call.name),
                        Some(ToolExchange {
                            operation_id: call.id.clone(),
                            tool_name: call.name.clone(),
                            arguments: Content::Inline(arguments),
                            result: None,
                            succeeded: None,
                            background: false,
                        }),
                    )?;
                    leaves.push(ContextLeaf {
                        node_id: commit.1.node_id,
                        chronology: commit.0.event.body.device_sequence,
                    });
                }
            }
            MessageRole::Tool => {
                let operation_id = message.tool_call_id.clone().unwrap_or_default();
                let call = calls.get(&operation_id);
                let commit = commit_text(
                    heart,
                    encoder,
                    agent_id,
                    thread_id,
                    ParticipantRole::Tool,
                    EventKind::ToolResult,
                    &message.content,
                    Some(ToolExchange {
                        operation_id,
                        tool_name: call.map_or_else(|| "unknown".into(), |call| call.name.clone()),
                        arguments: Content::Inline(
                            call.map_or_else(|| "{}".into(), |call| call.arguments.to_string()),
                        ),
                        result: Some(Content::Inline(message.content.clone())),
                        succeeded: None,
                        background: false,
                    }),
                )?;
                leaves.push(ContextLeaf {
                    node_id: commit.1.node_id,
                    chronology: commit.0.event.body.device_sequence,
                });
            }
        }
    }
    Ok(leaves)
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn host_introspection_exposes_committed_signals_and_honest_unknown_risk() {
        let receipt = spine_heart::MemoryReceipt {
            event_id: EventId::from_bytes([1; 32]),
            node_id: spine_heart::NodeId::from_bytes([2; 32]),
            feeling: spine_heart::FeelingVector {
                raw: vec![0.0; 2],
                activated: vec![0.0; 2],
                valence: 0.0,
                arousal: 0.0,
                dominant_channel: 0,
                dominant_label: None,
                input_norm: 1.0,
            },
            trajectory: spine_heart::TrajectoryStep {
                surprise: 1.2,
                speed: 0.4,
                heading_norm: 0.8,
            },
        };
        let policy = ModulationConfig::default().compute(ModulationInput {
            surprise: receipt.trajectory.surprise,
            valence: 0.0,
            arousal: 0.0,
            risk: 1.0,
            tensions: 0,
            base_temperature: 0.7,
            configured_tool_rounds: None,
        });
        let values = turn_introspection_metadata(&receipt, &policy, 0.7, true, None).unwrap();
        let trajectory: serde_json::Value =
            serde_json::from_str(&values["spine_trajectory"]).unwrap();
        let modulation: serde_json::Value =
            serde_json::from_str(&values["spine_modulation"]).unwrap();
        let risk: serde_json::Value = serde_json::from_str(&values["spine_risk_policy"]).unwrap();
        assert!(
            (trajectory["surprise"].as_f64().unwrap() - f64::from(receipt.trajectory.surprise))
                .abs()
                < 1e-6
        );
        assert_eq!(modulation["max_actions"], 1);
        assert!(modulation["max_tool_rounds"].is_null());
        assert_eq!(risk["available"], false);
        assert!(risk["risk_estimate"].is_null());
        assert_eq!(risk["effective_risk"], 1.0);
        assert_eq!(risk["nli_enabled"], true);
    }

    fn checkpoint(harness_id: &str, task: &str) -> HarnessCheckpoint {
        HarnessCheckpoint {
            schema: 1,
            harness_id: harness_id.into(),
            messages: vec![
                Message::new(MessageRole::System, "system"),
                Message::new(MessageRole::User, task),
                Message::new(MessageRole::Assistant, "paused safely"),
            ],
            completed_tool_calls: 0,
            completed_tool_rounds: 0,
            pending_task: task.into(),
            host_plan: None,
            completed_action_calls: 0,
            policy: Some(spine_runtime::HarnessPolicy::default()),
        }
    }

    fn test_event(seed: u8, interaction: InteractionInput) -> SignedEvent {
        SignedEvent {
            id: EventId::from_bytes([seed; 32]),
            body: spine_heart::EventBody {
                schema: 1,
                device_id: spine_heart::DeviceId::from_bytes([9; 32]),
                authorization_epoch: 0,
                device_sequence: u64::from(seed),
                timestamp: spine_heart::HybridTimestamp {
                    wall_millis: u64::from(seed),
                    counter: 0,
                },
                interaction,
            },
            signer_public_key: [0; 32],
            signature: Vec::new(),
        }
    }

    fn ids() -> (AgentId, ThreadId) {
        (
            AgentId::new("main").unwrap(),
            ThreadId::new("interactive").unwrap(),
        )
    }

    fn ordinary_interaction(
        agent_id: AgentId,
        thread_id: ThreadId,
        text: &str,
    ) -> InteractionInput {
        InteractionInput {
            agent_id,
            thread_id,
            role: ParticipantRole::User,
            kind: EventKind::Message,
            content: Content::Inline(text.into()),
            causal_parents: Vec::new(),
            provenance: Provenance::default(),
            tool: None,
            attachments: Vec::new(),
            outcome: None,
        }
    }

    fn incognito_args(flag: &str) -> Vec<&str> {
        vec!["spine", "chat", flag, "--model-dir", "models"]
    }

    #[test]
    fn persisted_checkpoint_is_discovered_after_heart_reopen() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("checkpoint.spine");
        let (agent_id, thread_id) = ids();
        let created = SpineHeart::create(HeartConfig::new(&path), "secret").unwrap();
        let checkpoint = checkpoint("harness-restart", "inspect the repository");
        let receipt = created
            .heart
            .commit_interaction(
                checkpoint
                    .to_interaction(agent_id.clone(), thread_id.clone())
                    .unwrap(),
            )
            .unwrap();
        drop(created.heart);

        let reopened = SpineHeart::open(
            HeartConfig::new(path),
            KeySource::Passphrase("secret".into()),
        )
        .unwrap();
        assert_eq!(
            discover_persisted_checkpoint(
                &reopened.events_canonical().unwrap(),
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::Available(PersistedHarnessCheckpoint {
                checkpoint,
                event_id: receipt.event.id,
            })
        );
    }

    #[test]
    fn consumed_checkpoint_stays_unavailable_after_heart_reopen() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("consumed-checkpoint.spine");
        let (agent_id, thread_id) = ids();
        let created = SpineHeart::create(HeartConfig::new(&path), "secret").unwrap();
        let checkpoint = checkpoint("harness-restart", "inspect the repository");
        let receipt = created
            .heart
            .commit_interaction(
                checkpoint
                    .to_interaction(agent_id.clone(), thread_id.clone())
                    .unwrap(),
            )
            .unwrap();
        let persisted = PersistedHarnessCheckpoint {
            checkpoint,
            event_id: receipt.event.id,
        };
        created
            .heart
            .commit_interaction(
                checkpoint_consumption_interaction(&persisted, agent_id.clone(), thread_id.clone())
                    .unwrap(),
            )
            .unwrap();
        drop(created.heart);

        let reopened = SpineHeart::open(
            HeartConfig::new(path),
            KeySource::Passphrase("secret".into()),
        )
        .unwrap();
        assert_eq!(
            discover_persisted_checkpoint(
                &reopened.events_canonical().unwrap(),
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::None
        );
    }

    #[test]
    fn consumed_checkpoint_is_stale_but_a_newer_checkpoint_is_available() {
        let (agent_id, thread_id) = ids();
        let first = checkpoint("harness-one", "first task");
        let first_event = test_event(
            1,
            first
                .to_interaction(agent_id.clone(), thread_id.clone())
                .unwrap(),
        );
        let persisted = PersistedHarnessCheckpoint {
            checkpoint: first,
            event_id: first_event.id,
        };
        let consumed = test_event(
            2,
            checkpoint_consumption_interaction(&persisted, agent_id.clone(), thread_id.clone())
                .unwrap(),
        );
        assert_eq!(
            discover_persisted_checkpoint(
                &[first_event.clone(), consumed.clone()],
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::None
        );

        let second = checkpoint("harness-one", "second task");
        let second_event = test_event(
            3,
            second
                .to_interaction(agent_id.clone(), thread_id.clone())
                .unwrap(),
        );
        assert_eq!(
            discover_persisted_checkpoint(
                &[first_event, consumed, second_event.clone()],
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::Available(PersistedHarnessCheckpoint {
                checkpoint: second,
                event_id: second_event.id,
            })
        );
    }

    #[test]
    fn normal_later_events_do_not_invalidate_an_open_checkpoint() {
        let (agent_id, thread_id) = ids();
        let checkpoint = checkpoint("harness-open", "paused task");
        let checkpoint_event = test_event(
            1,
            checkpoint
                .to_interaction(agent_id.clone(), thread_id.clone())
                .unwrap(),
        );
        let later = test_event(
            2,
            ordinary_interaction(agent_id.clone(), thread_id.clone(), "unrelated later turn"),
        );
        assert_eq!(
            discover_persisted_checkpoint(
                &[checkpoint_event.clone(), later],
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::Available(PersistedHarnessCheckpoint {
                checkpoint,
                event_id: checkpoint_event.id,
            })
        );
    }

    #[test]
    fn malformed_newest_checkpoint_and_consumption_records_fail_closed() {
        let (agent_id, thread_id) = ids();
        let valid = checkpoint("harness-valid", "valid task");
        let valid_event = test_event(
            1,
            valid
                .to_interaction(agent_id.clone(), thread_id.clone())
                .unwrap(),
        );
        let mut malformed_checkpoint = checkpoint("harness-malformed", "new task")
            .to_interaction(agent_id.clone(), thread_id.clone())
            .unwrap();
        malformed_checkpoint.content = Content::Inline("{not-json".into());
        assert!(matches!(
            discover_persisted_checkpoint(
                &[valid_event.clone(), test_event(2, malformed_checkpoint)],
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::Rejected(_)
        ));

        let persisted = PersistedHarnessCheckpoint {
            checkpoint: valid,
            event_id: valid_event.id,
        };
        let mut malformed_consumption =
            checkpoint_consumption_interaction(&persisted, agent_id.clone(), thread_id.clone())
                .unwrap();
        malformed_consumption.content = Content::Inline("{}".into());
        assert!(matches!(
            discover_persisted_checkpoint(
                &[valid_event, test_event(3, malformed_consumption)],
                &agent_id,
                &thread_id,
            ),
            CheckpointDiscovery::Rejected(_)
        ));
    }

    #[test]
    fn checkpoint_discovery_is_scoped_to_the_selected_agent_and_thread() {
        let (agent_id, thread_id) = ids();
        let event = test_event(
            1,
            checkpoint("other", "other task")
                .to_interaction(
                    AgentId::new("other-agent").unwrap(),
                    ThreadId::new("other-thread").unwrap(),
                )
                .unwrap(),
        );
        assert_eq!(
            discover_persisted_checkpoint(&[event], &agent_id, &thread_id),
            CheckpointDiscovery::None
        );
    }

    #[test]
    fn incognito_mode_does_not_require_a_heart_path() {
        let cli = Cli::try_parse_from(incognito_args("--incognito-mode")).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                path: None,
                incognito_mode: true,
                ..
            }
        ));
    }

    #[test]
    fn test_mode_is_a_visible_incognito_alias() {
        let cli = Cli::try_parse_from(incognito_args("--test-mode")).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                path: None,
                incognito_mode: true,
                ..
            }
        ));
    }

    #[test]
    fn persistent_chat_accepts_the_documented_default_heart_path() {
        let cli = Cli::try_parse_from(["spine", "chat"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                path: None,
                incognito_mode: false,
                ..
            }
        ));
    }

    #[test]
    fn first_conversation_has_an_explicit_automation_escape_hatch() {
        let cli = Cli::try_parse_from(["spine", "chat", "--skip-onboarding"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                skip_onboarding: true,
                ..
            }
        ));
    }

    #[test]
    fn provider_reasoning_effort_is_optional_and_configurable() {
        let cli = Cli::try_parse_from(["spine", "chat", "--reasoning-effort", "high"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                reasoning_effort: Some(value),
                ..
            } if value == "high"
        ));
        let cli = Cli::try_parse_from(["spine", "chat"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                reasoning_effort: None,
                ..
            }
        ));
    }

    #[test]
    fn web_mode_uses_portable_local_defaults() {
        let cli = Cli::try_parse_from(["spine", "chat", "--web"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Chat {
                web_server: true,
                web_host,
                web_port: 8_088,
                allow_remote_web: false,
                server_url,
                ..
            } if web_host == "127.0.0.1" && server_url == "http://127.0.0.1:8080"
        ));
        assert!(Cli::try_parse_from(["spine", "chat", "--allow-remote-web"]).is_err());
    }

    #[test]
    fn maintenance_commands_can_prompt_for_the_passphrase() {
        let cli = Cli::try_parse_from(["spine", "stats", "heart.spine"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Stats {
                passphrase: None,
                ..
            }
        ));
    }

    #[test]
    fn browser_state_does_not_expose_the_heart_directory() {
        let label = web_heart_label(
            std::path::Path::new("/private/developer/location/default.spine"),
            false,
        );
        assert_eq!(label, "default.spine");
        assert!(!label.contains("developer"));
    }

    #[test]
    fn managed_server_options_are_explicitly_paired() {
        assert!(
            Cli::try_parse_from([
                "spine",
                "chat",
                "heart.spine",
                "--model-dir",
                "models",
                "--llama-model",
                "model.gguf",
            ])
            .is_err()
        );
    }

    #[test]
    fn managed_server_arguments_enable_jinja_and_context_without_server_tools() {
        let arguments = llama_server_args(
            std::path::Path::new("model.gguf"),
            "127.0.0.1",
            9123,
            -1,
            32_768,
        )
        .into_iter()
        .map(|value| value.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
        assert_eq!(
            arguments[arguments.iter().position(|item| item == "--port").unwrap() + 1],
            "9123"
        );
        assert_eq!(
            arguments[arguments.iter().position(|item| item == "-c").unwrap() + 1],
            "32768"
        );
        assert!(arguments.iter().any(|item| item == "--jinja"));
        assert!(!arguments.iter().any(|item| item == "--tools"));
        assert!(!arguments.iter().any(|item| item == "secret"));
    }

    #[test]
    fn explicit_minilm_directory_is_validated_without_python() {
        let temporary = tempfile::tempdir().unwrap();
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(temporary.path().join(name), b"").unwrap();
        }
        assert_eq!(
            resolve_required_model_directory(Some(temporary.path().to_owned())).unwrap(),
            temporary.path()
        );
    }

    #[test]
    fn invalid_explicit_model_directory_is_not_silently_replaced() {
        let temporary = tempfile::tempdir().unwrap();
        let error =
            resolve_required_model_directory(Some(temporary.path().to_owned())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("model.safetensors"));
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_explicit_model_directory_preserves_permission_error() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("minilm");
        std::fs::create_dir(&model).unwrap();
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(model.join(name), b"").unwrap();
        }
        std::fs::set_permissions(&model, std::fs::Permissions::from_mode(0o000)).unwrap();

        let result = resolve_required_model_directory(Some(model.clone()));
        std::fs::set_permissions(&model, std::fs::Permissions::from_mode(0o700)).unwrap();
        if result.is_ok() {
            // Privileged test runners can bypass Unix mode bits.
            return;
        }
        let error = result.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("config.json"));
    }

    #[test]
    fn temporary_heart_directory_is_removed_with_its_guard() {
        let target = ChatHeartTarget::resolve(None, true).unwrap();
        let directory = target.path.parent().unwrap().to_owned();
        std::fs::write(&target.path, b"temporary-heart-marker").unwrap();
        assert!(target.path.exists());
        drop(target);
        assert!(!directory.exists());
    }

    #[test]
    fn ephemeral_passphrases_are_random_and_nonempty() {
        let first = ephemeral_passphrase().unwrap();
        let second = ephemeral_passphrase().unwrap();
        assert_eq!(first.len(), 64);
        assert_ne!(first, second);
    }
}
