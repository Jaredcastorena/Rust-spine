#![forbid(unsafe_code)]

mod agent_tools;
mod cognition_tools;
mod grounding;
mod longmem;
mod onboarding;
mod partner_tools;
#[cfg(test)]
mod tool_smoke_tests;
mod web_server;

use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    io::{self, BufRead, IsTerminal, Write},
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use spine_heart::{
    AgentId, CognitiveConfig, Content, ContextLeaf, EventKind, HeartConfig, InteractionInput,
    KeySource, ParticipantRole, Provenance, SemanticEncoder, SpineHeart, ThreadId, ToolExchange,
};
use spine_models::{MiniLmAssets, MiniLmEncoder};
use spine_runtime::{
    Harness, HarnessCheckpoint, HarnessConfig, HarnessEvent, LlamaCppConfig, LlamaCppProvider,
    Message, MessageRole, RunOutcome, ToolCall, ToolRegistry,
};

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
                metadata.insert("date".into(), chunk.date);
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
            let harness = Harness::new(
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

            let stdin_sender = line_sender.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    if stdin_sender.send(line).is_err() {
                        break;
                    }
                }
            });

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

            if !quit_after_onboarding {
                println!(
                    "Spine ready: Rust heart={} events={} tools={} grounding={} (/tasks, /stop, /interrupt, /resume, /quit)",
                    path.display(),
                    heart.stats()?.events,
                    harness.registry().len(),
                    if grounding.is_some() {
                        "NLI"
                    } else {
                        "disabled"
                    },
                );
            }
            let mut history = Vec::<Message>::new();
            let mut checkpoint = None::<HarnessCheckpoint>;
            let mut completed_turns = 0_u64;
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
                    if task == "/tasks" {
                        let tasks = running_tasks.format();
                        println!("{tasks}");
                        if let Some(web) = &web_ui {
                            web.notice(tasks);
                        }
                        continue;
                    }
                    if let Some(task_id) = task.strip_prefix("/cancel-task ") {
                        let result = running_tasks.cancel(task_id.trim());
                        println!("{result}");
                        if let Some(web) = &web_ui {
                            web.notice(result);
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
                    if !is_resume {
                        if let Some(web) = &web_ui {
                            web.begin_turn(task);
                        }
                        status.begin("Checking memory");
                        let recalled = if heart.stats()?.events == 0 {
                            "[]".into()
                        } else {
                            cognition_tools::recall_context(&heart, encoder.as_ref(), task, 5)?
                        };
                        let recalled_count =
                            serde_json::from_str::<Vec<serde_json::Value>>(&recalled)
                                .map_or(0, |items| items.len());
                        let retrieval_stats = [
                            (recalled_count as f32 / 10.0).min(1.0),
                            if recalled_count == 0 { 1.0 } else { 0.0 },
                            0.0,
                            0.0,
                        ];
                        let task_embedding = encoder.encode(task)?;
                        let triangle_context =
                            cognition_tools::rehydrate_triangle_context(&heart, &task_embedding)?;
                        let risk =
                            heart.predict_risk(&agent_id, &task_embedding, &retrieval_stats)?;
                        let user_commit = commit_text(
                            &heart,
                            &encoder,
                            &agent_id,
                            &thread_id,
                            ParticipantRole::User,
                            EventKind::Message,
                            task,
                            None,
                        )?;
                        let feeling = heart.feel(&agent_id, &task_embedding)?.map_or_else(
                            || "unavailable".into(),
                            |value| {
                                serde_json::to_string(&value)
                                    .unwrap_or_else(|_| "unavailable".into())
                            },
                        );
                        let system_prompt = format!(
                            "{partner_system_prompt}\n\nCurrent Thymos proprioception: {feeling}\nHost risk estimate for this memory region: {risk:.3}. At higher risk, deepen recall and avoid unsupported certainty.\n\nAutomatically recalled canonical evidence for this turn (it may be irrelevant; verify before using):\n{recalled}\n\nBudgeted triangle-context rehydration:\n{triangle_context}\n\nRunning/recent host tasks:\n{}",
                            running_tasks.format()
                        );
                        let persist_start = history.len() + 2;
                        let mut turn_leaves = vec![ContextLeaf {
                            node_id: user_commit.1.node_id,
                            chronology: user_commit.0.event.body.device_sequence,
                        }];
                        let run = Box::pin(harness.run_with_history(system_prompt, &history, task))
                            as std::pin::Pin<Box<dyn Future<Output = _>>>;
                        status.show("Thinking");
                        let controlled = run_with_operator_controls(
                            &harness,
                            run,
                            &mut line_receiver,
                            status.as_ref(),
                        )
                        .await;
                        let outcome = match controlled.outcome {
                            Ok(outcome) => outcome,
                            Err(error) => {
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
                        turn_leaves.extend(persist_harness_messages(
                            &heart,
                            &encoder,
                            &agent_id,
                            &thread_id,
                            &outcome.messages[persist_start.min(outcome.messages.len())..],
                        )?);
                        let mut quit_after = controlled.quit_after;
                        if let Some(gate) = &grounding
                            && !outcome.response.trim().is_empty()
                        {
                            status.show("Verifying answer");
                            if let Some(web) = &web_ui {
                                web.activity("Verifying answer");
                            }
                            let evidence = grounding::evidence_from_recall_and_messages(
                                &recalled,
                                &outcome.messages,
                            );
                            let decision = gate.verify(&outcome.response, &evidence);
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
                                let tension = (0.6 * (1.0 - decision.report.coverage)
                                    + 0.4 * decision.report.contradiction)
                                    .clamp(0.0, 1.0);
                                heart.update_risk(
                                    &agent_id,
                                    &task_embedding,
                                    &retrieval_stats,
                                    tension,
                                )?;
                                if decision.needs_repair {
                                    status.show("Repairing answer");
                                    if let Some(web) = &web_ui {
                                        web.activity("Repairing answer");
                                    }
                                    let repair_history = outcome.messages[1..].to_vec();
                                    let repair_start = repair_history.len() + 2;
                                    let repair_task = format!(
                                        "[HOST GROUNDING REPAIR] The draft's factual coverage was {:.3} and contradiction risk was {:.3}. Re-check the supplied evidence and tool results. Use more recall/tools if needed, correct unsupported claims, and abstain explicitly where evidence remains insufficient. Return the corrected final answer.",
                                        decision.report.coverage, decision.report.contradiction
                                    );
                                    let repair = Box::pin(harness.run_with_history(
                                        partner_system_prompt.clone(),
                                        &repair_history,
                                        repair_task,
                                    ))
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
                                            let repaired_evidence =
                                                grounding::evidence_from_recall_and_messages(
                                                    &recalled,
                                                    &repaired_outcome.messages,
                                                );
                                            let still_unverified = gate
                                                .verify(
                                                    &repaired_outcome.response,
                                                    &repaired_evidence,
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
                                            turn_leaves.extend(persist_harness_messages(
                                                &heart,
                                                &encoder,
                                                &agent_id,
                                                &thread_id,
                                                &repaired_outcome.messages[repair_start
                                                    .min(repaired_outcome.messages.len())..],
                                            )?);
                                            outcome = repaired_outcome;
                                        }
                                        Ok(None) => {
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
                            heart.compact_context(turn_leaves, 6)?;
                        }
                        completed_turns = completed_turns.saturating_add(1);
                        if completed_turns.is_multiple_of(10) {
                            status.show("Maintaining memory");
                            if let Some(web) = &web_ui {
                                web.activity("Maintaining memory");
                            }
                            heart.maintain_cognition(4)?;
                        }
                        checkpoint_from_outcome(
                            &heart,
                            &encoder,
                            &agent_id,
                            &thread_id,
                            &outcome,
                            &mut checkpoint,
                        )?;
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
                        let resumable = checkpoint.take().expect("checkpoint exists");
                        persist_start = resumable.messages.len() + 1;
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
                        Ok(outcome) => outcome,
                        Err(error) => {
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
                    let Some(outcome) = outcome else {
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
                    let leaves = persist_harness_messages(
                        &heart,
                        &encoder,
                        &agent_id,
                        &thread_id,
                        &outcome.messages[persist_start.min(outcome.messages.len())..],
                    )?;
                    if !leaves.is_empty() {
                        status.show("Saving context");
                        heart.compact_context(leaves, 6)?;
                    }
                    checkpoint_from_outcome(
                        &heart,
                        &encoder,
                        &agent_id,
                        &thread_id,
                        &outcome,
                        &mut checkpoint,
                    )?;
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
            let agent_id = AgentId::new(agent)?;
            let thread_id = ThreadId::new(thread)?;
            let mut provider_config = LlamaCppConfig::new(server_url);
            provider_config.api_key = api_key.filter(|value| !value.is_empty()).or_else(|| {
                std::env::var("SPINE_LLM_API_KEY")
                    .ok()
                    .filter(|value| !value.is_empty())
            });
            provider_config.model = server_model;
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
            let harness = Harness::new(
                provider,
                registry,
                HarnessConfig {
                    max_tool_rounds,
                    ..HarnessConfig::default()
                },
            )?
            .with_agent_id(agent_id.clone());

            commit_text(
                &heart,
                &encoder,
                &agent_id,
                &thread_id,
                ParticipantRole::User,
                EventKind::Message,
                &task,
                None,
            )?;
            let outcome = harness.run(partner_system_prompt, &task).await?;
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
        println!("[stopped safely; use /resume to continue]");
    }
}

fn checkpoint_from_outcome(
    heart: &SpineHeart,
    encoder: &MiniLmEncoder,
    agent_id: &AgentId,
    thread_id: &ThreadId,
    outcome: &RunOutcome,
    checkpoint: &mut Option<HarnessCheckpoint>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(resumable) = outcome.checkpoint.clone() {
        let interaction = resumable.to_interaction(agent_id.clone(), thread_id.clone())?;
        let text = match &interaction.content {
            Content::Inline(text) => text.clone(),
            Content::ColdBlob(_) | Content::Redacted => String::new(),
        };
        heart.commit_embedded(interaction, encoder.encode(&text)?)?;
        *checkpoint = Some(resumable);
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
                ..Provenance::default()
            },
            tool,
            attachments: Vec::new(),
            outcome: None,
        },
        encoder.encode(text)?,
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

    fn incognito_args(flag: &str) -> Vec<&str> {
        vec!["spine", "chat", flag, "--model-dir", "models"]
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
