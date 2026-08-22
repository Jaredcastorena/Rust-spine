#![forbid(unsafe_code)]

mod agent_tools;
mod cognition_tools;
mod grounding;
mod longmem;
mod partner_tools;

use std::{
    collections::BTreeMap,
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
#[command(name = "spine", version, about = "Portable encrypted Spine heart")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Create {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
    },
    Stats {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
    },
    Snapshot {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
        #[arg(long)]
        label: Option<String>,
    },
    CognitionInit {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        thymos_channels: usize,
    },
    Remember {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "main")]
        agent: String,
        #[arg(long, default_value = "default")]
        thread: String,
        text: String,
    },
    Recall {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
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
    LongMemIngest {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value_t = NonZeroUsize::new(32).expect("nonzero"))]
        embedding_batch_size: NonZeroUsize,
    },
    Chat {
        path: PathBuf,
        #[arg(long)]
        passphrase: Option<String>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:9001")]
        server_url: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        server_model: Option<String>,
        #[arg(long, default_value = "main")]
        agent: String,
        #[arg(long, default_value = "interactive")]
        thread: String,
        #[arg(long)]
        max_tool_rounds: Option<NonZeroU64>,
        #[arg(long, default_value_t = 8_192)]
        max_tokens: i64,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 7_200)]
        timeout_seconds: u64,
        #[arg(long, default_value_t = 2)]
        provider_retries: u32,
        #[arg(long)]
        max_context_tokens: Option<usize>,
        #[arg(long)]
        nli_model_dir: Option<PathBuf>,
        #[arg(long)]
        allow_model_memory_writes: bool,
        #[arg(long, default_value_t = 12)]
        max_history_turns: usize,
        #[arg(long, default_value_t = 64_000)]
        max_history_chars: usize,
    },
    HarnessRun {
        path: PathBuf,
        #[arg(long)]
        passphrase: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:9001")]
        server_url: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
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
        #[arg(long)]
        max_context_tokens: Option<usize>,
        task: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Create { path, passphrase } => {
            let created = SpineHeart::create(HeartConfig::new(&path), &passphrase)?;
            println!("created {}", created.heart.path().display());
            println!("recovery phrase: {}", created.recovery_phrase.expose());
        }
        Command::Stats { path, passphrase } => {
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
            nli_model_dir,
            allow_model_memory_writes,
            max_history_turns,
            max_history_chars,
        } => {
            let passphrase = passphrase
                .or_else(|| std::env::var("SPINE_HEART_PASSPHRASE").ok())
                .filter(|value| !value.is_empty())
                .ok_or("set SPINE_HEART_PASSPHRASE or pass --passphrase")?;
            let encoder = Arc::new(MiniLmEncoder::load(
                MiniLmAssets::from_directory(model_dir),
                256,
            )?);
            let heart = if path.exists() {
                SpineHeart::open(
                    HeartConfig::new(&path),
                    KeySource::Passphrase(passphrase.clone()),
                )?
            } else {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let created = SpineHeart::create(HeartConfig::new(&path), &passphrase)?;
                created.heart.initialize_cognition(CognitiveConfig::new(
                    1,
                    encoder.manifest().clone(),
                    8,
                )?)?;
                println!("created new encrypted heart: {}", path.display());
                println!("recovery phrase: {}", created.recovery_phrase.expose());
                created.heart
            };
            let heart = Arc::new(heart);
            let agent_id = AgentId::new(agent)?;
            let thread_id = ThreadId::new(thread)?;
            let mut provider_config = LlamaCppConfig::new(server_url);
            provider_config.api_key = api_key.or_else(|| std::env::var("SPINE_LLM_API_KEY").ok());
            provider_config.model = server_model;
            provider_config.max_tokens = max_tokens;
            provider_config.temperature = temperature;
            provider_config.timeout = Duration::from_secs(timeout_seconds);
            provider_config.maximum_retries = provider_retries;
            provider_config.max_context_tokens = max_context_tokens;
            let provider = Arc::new(LlamaCppProvider::new(provider_config)?);
            provider.health().await?;
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
                provider,
                registry,
                HarnessConfig {
                    max_tool_rounds,
                    ..HarnessConfig::default()
                },
            )?
            .with_agent_id(agent_id.clone());

            let status = Arc::new(TerminalStatus::new());
            let event_status = Arc::clone(&status);
            let mut events = harness.subscribe();
            let event_task = tokio::spawn(async move {
                while let Ok(event) = events.recv().await {
                    match event {
                        HarnessEvent::ToolStarted { name, .. } => {
                            event_status.show(tool_activity(&name));
                        }
                        HarnessEvent::ToolCompleted { .. } => {
                            event_status.tool_completed();
                        }
                        HarnessEvent::GuidanceInjected { .. } => {
                            event_status.show("Applying guidance");
                        }
                        HarnessEvent::GracefulStopBoundary => {
                            event_status.show("Stopping safely");
                        }
                        HarnessEvent::ModelTurnCompleted => {
                            event_status.show("Thinking");
                        }
                    }
                }
            });

            let (line_sender, mut line_receiver) = tokio::sync::mpsc::unbounded_channel();
            thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    if line_sender.send(line).is_err() {
                        break;
                    }
                }
            });

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
            let mut history = Vec::<Message>::new();
            let mut checkpoint = None::<HarnessCheckpoint>;
            let mut completed_turns = 0_u64;
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
                    println!("{}", running_tasks.format());
                    continue;
                }
                if let Some(task_id) = task.strip_prefix("/cancel-task ") {
                    println!("{}", running_tasks.cancel(task_id.trim()));
                    continue;
                }
                let is_resume = task == "/resume";
                if is_resume && checkpoint.is_none() {
                    println!("[no resumable checkpoint]");
                    continue;
                }
                if !is_resume {
                    status.begin("Checking memory");
                    let recalled = if heart.stats()?.events == 0 {
                        "[]".into()
                    } else {
                        cognition_tools::recall_context(&heart, &encoder, task, 5)?
                    };
                    let recalled_count = serde_json::from_str::<Vec<serde_json::Value>>(&recalled)
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
                    let risk = heart.predict_risk(&agent_id, &task_embedding, &retrieval_stats)?;
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
                            serde_json::to_string(&value).unwrap_or_else(|_| "unavailable".into())
                        },
                    );
                    let system_prompt = format!(
                        "{PARTNER_SYSTEM_PROMPT}\n\nCurrent Thymos proprioception: {feeling}\nHost risk estimate for this memory region: {risk:.3}. At higher risk, deepen recall and avoid unsupported certainty.\n\nAutomatically recalled canonical evidence for this turn (it may be irrelevant; verify before using):\n{recalled}\n\nBudgeted triangle-context rehydration:\n{triangle_context}\n\nRunning/recent host tasks:\n{}",
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
                        let evidence = grounding::evidence_from_recall_and_messages(
                            &recalled,
                            &outcome.messages,
                        );
                        let decision = gate.verify(&outcome.response, &evidence)?;
                        let tension = (0.6 * (1.0 - decision.report.coverage)
                            + 0.4 * decision.report.contradiction)
                            .clamp(0.0, 1.0);
                        heart.update_risk(&agent_id, &task_embedding, &retrieval_stats, tension)?;
                        if decision.needs_repair {
                            status.show("Repairing answer");
                            let repair_history = outcome.messages[1..].to_vec();
                            let repair_start = repair_history.len() + 2;
                            let repair_task = format!(
                                "[HOST GROUNDING REPAIR] The draft's factual coverage was {:.3} and contradiction risk was {:.3}. Re-check the supplied evidence and tool results. Use more recall/tools if needed, correct unsupported claims, and abstain explicitly where evidence remains insufficient. Return the corrected final answer.",
                                decision.report.coverage, decision.report.contradiction
                            );
                            let repair = Box::pin(harness.run_with_history(
                                PARTNER_SYSTEM_PROMPT,
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
                                Ok(Some(repaired_outcome)) => {
                                    turn_leaves.extend(persist_harness_messages(
                                        &heart,
                                        &encoder,
                                        &agent_id,
                                        &thread_id,
                                        &repaired_outcome.messages
                                            [repair_start.min(repaired_outcome.messages.len())..],
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
                                }
                            }
                        } else if decision.claim_count == 0 {
                            status.show("Answer ready");
                        } else {
                            status.show("Answer verified");
                        }
                    }
                    if !turn_leaves.is_empty() {
                        status.show("Saving context");
                        heart.compact_context(turn_leaves, 6)?;
                    }
                    completed_turns = completed_turns.saturating_add(1);
                    if completed_turns.is_multiple_of(10) {
                        status.show("Maintaining memory");
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
                status.begin("Resuming");
                let controlled =
                    run_with_operator_controls(&harness, run, &mut line_receiver, status.as_ref())
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
                finish_visible_turn(&outcome, &mut history, max_history_turns, max_history_chars);
                if controlled.quit_after {
                    break;
                }
            }
            event_task.abort();
            let snapshot = heart.snapshot(Some("interactive-exit".into()))?;
            println!("saved encrypted heart snapshot={snapshot}");
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
            provider_config.api_key = api_key.or_else(|| std::env::var("SPINE_LLM_API_KEY").ok());
            provider_config.model = server_model;
            provider_config.max_tokens = max_tokens;
            provider_config.temperature = temperature;
            provider_config.timeout = Duration::from_secs(timeout_seconds);
            provider_config.maximum_retries = provider_retries;
            provider_config.max_context_tokens = max_context_tokens;
            let provider = Arc::new(LlamaCppProvider::new(provider_config)?);
            provider.health().await?;

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
            let outcome = harness
                .run(
                    "You are a long-running Spine partner operating against an encrypted heart. Use the supplied heart tools whenever the task requests stored facts or store state. Treat tool output as authoritative, do not fabricate results, and give a concise final answer after completing the requested checks.",
                    &task,
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
