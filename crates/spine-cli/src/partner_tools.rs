use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use regex::Regex;
use spine_heart::{
    AgentId, Content, EventKind, InteractionInput, ParticipantRole, Provenance, SemanticEncoder,
    SpineHeart, ThreadId,
};
use spine_runtime::{
    RuntimeError, Tool, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk,
    ToolSpec,
};
use tokio::{process::Command, time::timeout};
use walkdir::WalkDir;

const MAX_FILE_READ_CHARS: usize = 50_000;
const MAX_WEB_CHARS: usize = 12_000;
const MAX_WEB_BODY_BYTES: usize = 2 * 1024 * 1024;

pub(crate) fn register_action_tools<E: SemanticEncoder + 'static>(
    registry: &mut ToolRegistry,
    heart: Arc<SpineHeart>,
    encoder: Arc<E>,
    cwd: PathBuf,
    audit_path: PathBuf,
) -> spine_runtime::Result<Arc<RunningTaskManager>> {
    let encoder: Arc<dyn SemanticEncoder> = encoder;
    registry.register(FileReadTool { cwd: cwd.clone() })?;
    registry.register(FileWriteTool { cwd: cwd.clone() })?;
    registry.register(FileListTool { cwd: cwd.clone() })?;
    registry.register(FileSearchTool { cwd: cwd.clone() })?;
    let running_tasks = Arc::new(RunningTaskManager::default());
    registry.register(ShellTool {
        cwd: cwd.clone(),
        audit_path,
        default_timeout: Duration::from_secs(120),
        running_tasks: Arc::clone(&running_tasks),
    })?;
    registry.register(RunningTasksTool {
        manager: Arc::clone(&running_tasks),
    })?;
    registry.register(CancelTaskTool {
        manager: Arc::clone(&running_tasks),
    })?;
    let browser = Arc::new(
        WebBrowser::new().map_err(|error| RuntimeError::InvalidConfig(error.to_string()))?,
    );
    registry.register(WebFetchTool {
        browser: Arc::clone(&browser),
    })?;
    registry.register(WebNavigateTool {
        browser: Arc::clone(&browser),
    })?;
    registry.register(WebSearchTool {
        browser: Arc::clone(&browser),
    })?;
    registry.register(WebBackTool {
        browser: Arc::clone(&browser),
    })?;
    registry.register(WebForwardTool { browser })?;
    registry.register(DocumentIngestTool {
        heart,
        encoder,
        cwd,
        running_tasks: Arc::clone(&running_tasks),
    })?;
    Ok(running_tasks)
}

fn resolve(cwd: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

fn string_arg<'a>(call: &'a ToolCall, name: &str) -> Option<&'a str> {
    call.arguments.get(name).and_then(|value| value.as_str())
}

struct FileReadTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for FileReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "file_read".into(),
            description: "Read a UTF-8 text file, returning its path, size, and contents.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{"path":{"type":"string"}},
                "required":["path"],
                "additionalProperties":false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(path) = string_arg(call, "path").filter(|value| !value.trim().is_empty()) else {
            return Ok(ToolResult::failure("path is required"));
        };
        let path = resolve(&self.cwd, path);
        let result: Result<String, String> = (|| {
            if path.is_dir() {
                return Err(format!("path is a directory: {}", path.display()));
            }
            let bytes = fs::read(&path).map_err(|error| error.to_string())?;
            let text = String::from_utf8_lossy(&bytes);
            let truncated: String = text.chars().take(MAX_FILE_READ_CHARS).collect();
            Ok(format!(
                "File: {}\nSize: {} chars, {} lines\n\n=== Content ===\n{}{}",
                path.display(),
                text.chars().count(),
                text.lines().count().max(1),
                truncated,
                if text.chars().count() > MAX_FILE_READ_CHARS {
                    "\n[truncated]"
                } else {
                    ""
                }
            ))
        })();
        Ok(match result {
            Ok(output) => ToolResult::success(output),
            Err(error) => ToolResult::failure(error),
        })
    }
}

struct FileWriteTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for FileWriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "file_write".into(),
            description: "Write UTF-8 text to a file, creating parent directories when needed."
                .into(),
            category: ToolCategory::Action,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "content":{"type":"string"}
                },
                "required":["path","content"],
                "additionalProperties":false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(path) = string_arg(call, "path").filter(|value| !value.trim().is_empty()) else {
            return Ok(ToolResult::failure("path is required"));
        };
        let Some(content) = string_arg(call, "content") else {
            return Ok(ToolResult::failure("content must be a string"));
        };
        let path = resolve(&self.cwd, path);
        let result: Result<String, String> = (|| {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            fs::write(&path, content).map_err(|error| error.to_string())?;
            Ok(format!(
                "Written: {}\nSize: {} chars, {} lines",
                path.display(),
                content.chars().count(),
                content.lines().count().max(1)
            ))
        })();
        Ok(match result {
            Ok(output) => ToolResult::success(output),
            Err(error) => ToolResult::failure(error),
        })
    }
}

struct FileListTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for FileListTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "file_list".into(),
            description: "List a directory, optionally recursively.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "recursive":{"type":"boolean"},
                    "max_items":{"type":"integer","minimum":1,"maximum":10000}
                },
                "additionalProperties":false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let path = resolve(&self.cwd, string_arg(call, "path").unwrap_or("."));
        let recursive = call
            .arguments
            .get("recursive")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let maximum = call
            .arguments
            .get("max_items")
            .and_then(|value| value.as_u64())
            .unwrap_or(100)
            .clamp(1, 10_000) as usize;
        if !path.exists() {
            return Ok(ToolResult::failure(format!(
                "path not found: {}",
                path.display()
            )));
        }
        if path.is_file() {
            return Ok(ToolResult::success(format!("{} (file)", path.display())));
        }
        let mut items = Vec::new();
        if recursive {
            for entry in WalkDir::new(&path)
                .min_depth(1)
                .into_iter()
                .filter_map(Result::ok)
            {
                let relative = entry.path().strip_prefix(&path).unwrap_or(entry.path());
                items.push(format!(
                    "{}{}",
                    relative.display(),
                    if entry.file_type().is_dir() { "/" } else { "" }
                ));
                if items.len() >= maximum {
                    break;
                }
            }
        } else {
            match fs::read_dir(&path) {
                Ok(entries) => {
                    for entry in entries.filter_map(Result::ok) {
                        items.push(format!(
                            "{}{}",
                            entry.file_name().to_string_lossy(),
                            if entry.path().is_dir() { "/" } else { "" }
                        ));
                    }
                    items.sort();
                    items.truncate(maximum);
                }
                Err(error) => return Ok(ToolResult::failure(error.to_string())),
            }
        }
        Ok(ToolResult::success(format!(
            "Directory: {}\nItems: {}\n\n{}",
            path.display(),
            items.len(),
            items.join("\n")
        )))
    }
}

struct FileSearchTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for FileSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "file_search".into(),
            description: "Find files whose names match a glob pattern.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string"},
                    "path":{"type":"string"},
                    "max_results":{"type":"integer","minimum":1,"maximum":10000}
                },
                "required":["pattern"],
                "additionalProperties":false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(pattern) = string_arg(call, "pattern").filter(|value| !value.is_empty()) else {
            return Ok(ToolResult::failure("pattern is required"));
        };
        let root = resolve(&self.cwd, string_arg(call, "path").unwrap_or("."));
        if !root.exists() {
            return Ok(ToolResult::failure(format!(
                "search root not found: {}",
                root.display()
            )));
        }
        let maximum = call
            .arguments
            .get("max_results")
            .and_then(|value| value.as_u64())
            .unwrap_or(50)
            .clamp(1, 10_000) as usize;
        let matcher = match glob::Pattern::new(pattern) {
            Ok(matcher) => matcher,
            Err(error) => return Ok(ToolResult::failure(error.to_string())),
        };
        let results: Vec<_> = WalkDir::new(&root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| {
                matcher.matches_path(entry.path())
                    || matcher.matches(entry.file_name().to_string_lossy().as_ref())
            })
            .take(maximum)
            .map(|entry| entry.path().display().to_string())
            .collect();
        Ok(ToolResult::success(format!(
            "Matches: {}\n{}",
            results.len(),
            results.join("\n")
        )))
    }
}

#[derive(Default)]
pub(crate) struct RunningTaskManager {
    next: AtomicU64,
    tasks: Mutex<BTreeMap<String, RunningTaskRecord>>,
}

struct RunningTaskRecord {
    command: String,
    started: Instant,
    status: &'static str,
    result: Option<ToolResult>,
    abort: Option<tokio::task::AbortHandle>,
    cancellable: bool,
}

impl RunningTaskManager {
    fn begin(&self, command: &str, cancellable: bool) -> String {
        let id = format!(
            "task{}",
            self.next.fetch_add(1, AtomicOrdering::Relaxed) + 1
        );
        self.tasks
            .lock()
            .expect("running-task lock poisoned")
            .insert(
                id.clone(),
                RunningTaskRecord {
                    command: command.into(),
                    started: Instant::now(),
                    status: "running",
                    result: None,
                    abort: None,
                    cancellable,
                },
            );
        id
    }

    fn attach(&self, id: &str, abort: tokio::task::AbortHandle) {
        if let Some(task) = self
            .tasks
            .lock()
            .expect("running-task lock poisoned")
            .get_mut(id)
        {
            task.abort = Some(abort);
        }
    }

    fn finish(&self, id: &str, result: ToolResult) {
        if let Some(task) = self
            .tasks
            .lock()
            .expect("running-task lock poisoned")
            .get_mut(id)
        {
            task.status = if result.success {
                "completed"
            } else {
                "failed"
            };
            task.result = Some(result);
            task.abort = None;
        }
    }

    pub(crate) fn format(&self) -> String {
        let tasks = self.tasks.lock().expect("running-task lock poisoned");
        if tasks.is_empty() {
            return "(no running or recent tasks)".into();
        }
        tasks
            .iter()
            .map(|(id, task)| {
                let result = task.result.as_ref().map_or_else(String::new, |result| {
                    let text = result.error.as_ref().unwrap_or(&result.output);
                    format!("\n{}", text.chars().take(2_000).collect::<String>())
                });
                format!(
                    "[{id}] status={} cancellable={} elapsed={:.1}s command={}{}",
                    task.status,
                    task.cancellable,
                    task.started.elapsed().as_secs_f64(),
                    task.command,
                    result
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub(crate) fn cancel(&self, id: &str) -> String {
        let mut tasks = self.tasks.lock().expect("running-task lock poisoned");
        let ids: Vec<_> = if id.is_empty() || id == "all" {
            tasks
                .iter()
                .filter(|(_, task)| task.status == "running" && task.cancellable)
                .map(|(id, _)| id.clone())
                .collect()
        } else if tasks
            .get(id)
            .is_some_and(|task| task.status == "running" && task.cancellable)
        {
            vec![id.to_owned()]
        } else {
            Vec::new()
        };
        for id in &ids {
            if let Some(task) = tasks.get_mut(id) {
                if let Some(abort) = task.abort.take() {
                    abort.abort();
                }
                task.status = "cancelled";
                task.result = Some(ToolResult::failure("task cancelled by operator"));
            }
        }
        if ids.is_empty() {
            format!("(no active task matching {id:?})")
        } else {
            format!("Cancelled {} task(s): {}", ids.len(), ids.join(", "))
        }
    }
}

struct RunningTasksTool {
    manager: Arc<RunningTaskManager>,
}

#[async_trait]
impl Tool for RunningTasksTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "check_tasks".into(),
            description: "Show running and recently completed host shell tasks.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: serde_json::json!({"type":"object","properties":{},"additionalProperties":false}),
        }
    }

    async fn execute(
        &self,
        _call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        Ok(ToolResult::success(self.manager.format()))
    }
}

struct CancelTaskTool {
    manager: Arc<RunningTaskManager>,
}

#[async_trait]
impl Tool for CancelTaskTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "cancel_task".into(),
            description: "Cancel a running shell task by id, or all running shell tasks.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{"task_id":{"type":"string"}},
                "required":["task_id"],
                "additionalProperties":false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let id = string_arg(call, "task_id").unwrap_or("").trim();
        Ok(ToolResult::success(self.manager.cancel(id)))
    }
}

#[derive(Clone)]
struct ShellTool {
    cwd: PathBuf,
    audit_path: PathBuf,
    default_timeout: Duration,
    running_tasks: Arc<RunningTaskManager>,
}

impl ShellTool {
    fn destructive(command: &str) -> bool {
        let command = command.to_ascii_lowercase();
        [
            "rm ",
            "rm\t",
            "rm -rf",
            "rm -fr",
            "rmdir",
            "pkill -9",
            "kill -9",
            "killall -9",
            "mkfs",
            "wipefs",
            "git reset --hard",
            "git clean -f",
            "truncate ",
            "shred ",
            "chmod 000",
            "crontab -r",
            "shutdown",
            "reboot",
            "poweroff",
            "drop database",
            "drop table",
            "> /dev/sd",
            "dd if=",
            ":(){",
            "remove-item",
            "clear-disk",
            "format-volume",
            "initialize-disk",
            "remove-partition",
            "stop-computer",
            "restart-computer",
            "diskpart",
            "del /",
            "erase /",
        ]
        .iter()
        .any(|needle| command.contains(needle))
    }

    fn audit(&self, command: &str, status: &str, detail: &str, elapsed: f64) {
        if let Some(parent) = self.audit_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let record = serde_json::json!({
            "schema":1,
            "unix_millis":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or_default(),
            "command":command,
            "status":status,
            "detail":detail,
            "elapsed_seconds":elapsed,
        });
        if let Ok(mut file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_path)
        {
            use std::io::Write as _;
            let _ = writeln!(file, "{record}");
        }
    }

    fn shell_process(&self, command: &str) -> Command {
        #[cfg(target_os = "windows")]
        let mut process = {
            let mut process = Command::new("powershell.exe");
            process
                .args(["-NoLogo", "-NoProfile", "-NonInteractive", "-Command"])
                .arg(command);
            process
        };
        #[cfg(not(target_os = "windows"))]
        let mut process = {
            let shell = if Path::new("/bin/bash").is_file() {
                "/bin/bash"
            } else if Path::new("/bin/sh").is_file() {
                "/bin/sh"
            } else {
                "sh"
            };
            let mut process = Command::new(shell);
            process.arg("-lc").arg(command);
            process
        };
        process
            .current_dir(&self.cwd)
            .env_remove("SPINE_HEART_PASSPHRASE")
            .env_remove("SPINE_LLM_API_KEY")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        process
    }

    async fn run_command(&self, command: &str, duration: Duration, unlimited: bool) -> ToolResult {
        let started = Instant::now();
        let child = match self.shell_process(command).spawn() {
            Ok(child) => child,
            Err(error) => return ToolResult::failure(error.to_string()),
        };
        let output = if unlimited {
            child
                .wait_with_output()
                .await
                .map_err(|error| error.to_string())
        } else {
            match timeout(duration, child.wait_with_output()).await {
                Ok(output) => output.map_err(|error| error.to_string()),
                Err(_) => {
                    self.audit(
                        command,
                        "timed_out",
                        "timeout",
                        started.elapsed().as_secs_f64(),
                    );
                    return ToolResult::failure(format!(
                        "command timed out after {:.1} seconds",
                        duration.as_secs_f64()
                    ));
                }
            }
        };
        let output = match output {
            Ok(output) => output,
            Err(error) => {
                self.audit(command, "failed", &error, started.elapsed().as_secs_f64());
                return ToolResult::failure(error);
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let text = format!(
            "exit_code={}\nstdout:\n{}\nstderr:\n{}",
            output.status.code().unwrap_or(-1),
            stdout,
            stderr
        );
        let status = if output.status.success() {
            "completed"
        } else {
            "failed"
        };
        self.audit(command, status, &text, started.elapsed().as_secs_f64());
        if output.status.success() {
            ToolResult::success(text)
        } else {
            ToolResult::failure(text)
        }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell".into(),
            description: "Run a shell command with timeout, captured output, and host audit."
                .into(),
            category: ToolCategory::Action,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{
                    "command":{"type":"string"},
                    "timeout_s":{"type":"number","minimum":0,"maximum":86400}
                },
                "required":["command"],
                "additionalProperties":false
            }),
        }
    }

    fn risk_for_call(&self, call: &ToolCall) -> ToolRisk {
        if string_arg(call, "command").is_some_and(Self::destructive) {
            ToolRisk::Destructive
        } else {
            ToolRisk::Mutating
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(command) = string_arg(call, "command").filter(|value| !value.trim().is_empty())
        else {
            return Ok(ToolResult::failure("command is required"));
        };
        let requested = call
            .arguments
            .get("timeout_s")
            .and_then(|value| value.as_f64());
        let duration = requested
            .filter(|seconds| *seconds > 0.0)
            .map(Duration::from_secs_f64)
            .unwrap_or(self.default_timeout);
        let unlimited = requested == Some(0.0);
        let task_id = self.running_tasks.begin(command, true);
        let runner = self.clone();
        let command = command.to_owned();
        let manager = Arc::clone(&self.running_tasks);
        let completed_id = task_id.clone();
        let handle = tokio::spawn(async move {
            let result = runner.run_command(&command, duration, unlimited).await;
            manager.finish(&completed_id, result.clone());
            result
        });
        self.running_tasks.attach(&task_id, handle.abort_handle());
        match handle.await {
            Ok(result) => Ok(result),
            Err(error) if error.is_cancelled() => Ok(ToolResult::failure(format!(
                "shell task {task_id} was cancelled"
            ))),
            Err(error) => Ok(ToolResult::failure(format!(
                "shell task {task_id} failed to join: {error}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
struct WebPage {
    url: String,
    title: String,
    text: String,
    links: Vec<(String, String)>,
}

struct BrowserState {
    history: Vec<WebPage>,
    cursor: Option<usize>,
}

struct WebBrowser {
    client: reqwest::Client,
    search_endpoint: reqwest::Url,
    state: Mutex<BrowserState>,
}

impl WebBrowser {
    fn new() -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: reqwest::Client::builder()
                .user_agent("Spine/0.1 text browser")
                .timeout(Duration::from_secs(30))
                .build()?,
            search_endpoint: reqwest::Url::parse("https://html.duckduckgo.com/html/")
                .expect("static search URL is valid"),
            state: Mutex::new(BrowserState {
                history: Vec::new(),
                cursor: None,
            }),
        })
    }

    async fn fetch(&self, value: &str, record: bool) -> Result<WebPage, String> {
        let url = if value.starts_with("http://") || value.starts_with("https://") {
            value.to_owned()
        } else {
            format!("https://{value}")
        };
        let mut response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|error| error.to_string())?;
        let status = response.status();
        let final_url = response.url().to_string();
        if !status.is_success() {
            return Err(format!("HTTP {status} for {final_url}"));
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
            if body.len().saturating_add(chunk.len()) > MAX_WEB_BODY_BYTES {
                return Err(format!(
                    "response body exceeds {} bytes for {final_url}",
                    MAX_WEB_BODY_BYTES
                ));
            }
            body.extend_from_slice(&chunk);
        }
        let body = String::from_utf8_lossy(&body);
        let title_re = Regex::new(r"(?is)<title[^>]*>(.*?)</title>").expect("valid regex");
        let link_re =
            Regex::new(r#"(?is)<a[^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>"#).expect("valid regex");
        let tag_re = Regex::new(r"(?is)<[^>]+>").expect("valid regex");
        let space_re = Regex::new(r"[ \t\r\f\v]+").expect("valid regex");
        let title = title_re
            .captures(&body)
            .and_then(|capture| capture.get(1))
            .map(|value| tag_re.replace_all(value.as_str(), " ").trim().to_owned())
            .unwrap_or_default();
        let mut links = Vec::new();
        for capture in link_re.captures_iter(&body).take(100) {
            let href = capture
                .get(1)
                .map(|value| value.as_str())
                .unwrap_or_default();
            let label = capture
                .get(2)
                .map(|value| value.as_str())
                .unwrap_or_default();
            let label = decode_html_entities(tag_re.replace_all(label, " ").trim());
            let href = reqwest::Url::parse(&final_url)
                .ok()
                .and_then(|base| base.join(href).ok())
                .map_or_else(|| href.to_owned(), |url| url.to_string());
            links.push((label, href));
        }
        let without_scripts =
            Regex::new(r"(?is)<(?:script|style|noscript)[^>]*>.*?</(?:script|style|noscript)>")
                .expect("valid regex")
                .replace_all(&body, " ");
        let text = tag_re.replace_all(&without_scripts, "\n");
        let text = space_re.replace_all(&text, " ");
        let text = decode_html_entities(
            &text
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .take(2_000)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let page = WebPage {
            url: final_url,
            title: if title.is_empty() {
                format!("HTTP {status}")
            } else {
                decode_html_entities(&title)
            },
            text: text.chars().take(MAX_WEB_CHARS).collect(),
            links,
        };
        if record {
            let mut state = self.state.lock().expect("browser lock poisoned");
            if let Some(cursor) = state.cursor {
                state.history.truncate(cursor + 1);
            }
            state.history.push(page.clone());
            state.cursor = Some(state.history.len() - 1);
        }
        Ok(page)
    }

    fn move_history(&self, offset: isize) -> Option<WebPage> {
        let mut state = self.state.lock().expect("browser lock poisoned");
        let cursor = state.cursor?;
        let next = cursor.checked_add_signed(offset)?;
        if next >= state.history.len() {
            return None;
        }
        state.cursor = Some(next);
        state.history.get(next).cloned()
    }
}

fn decode_html_entities(value: &str) -> String {
    value
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn page_text(page: &WebPage) -> String {
    let links = page
        .links
        .iter()
        .enumerate()
        .map(|(index, (label, href))| format!("[{}] {}: {}", index + 1, label, href))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "URL: {}\nTitle: {}\n\n=== Content ===\n{}\n\n=== Links ===\n{}",
        page.url, page.title, page.text, links
    )
}

struct WebFetchTool {
    browser: Arc<WebBrowser>,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        web_spec(
            "web_fetch",
            "Fetch a URL and return rendered text and links.",
            true,
        )
    }
    async fn execute(&self, call: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        let Some(url) = string_arg(call, "url") else {
            return Ok(ToolResult::failure("url is required"));
        };
        Ok(match self.browser.fetch(url, true).await {
            Ok(page) => ToolResult::success(page_text(&page)),
            Err(error) => ToolResult::failure(error),
        })
    }
}

struct WebSearchTool {
    browser: Arc<WebBrowser>,
}

struct WebNavigateTool {
    browser: Arc<WebBrowser>,
}

#[async_trait]
impl Tool for WebNavigateTool {
    fn spec(&self) -> ToolSpec {
        web_spec(
            "web_navigate",
            "Navigate to a URL in the shared browser session.",
            true,
        )
    }
    async fn execute(&self, call: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        let Some(url) = string_arg(call, "url") else {
            return Ok(ToolResult::failure("url is required"));
        };
        Ok(match self.browser.fetch(url, true).await {
            Ok(page) => ToolResult::success(page_text(&page)),
            Err(error) => ToolResult::failure(error),
        })
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        web_spec(
            "web_search",
            "Search the web and return rendered results and links.",
            false,
        )
    }
    async fn execute(&self, call: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        let Some(query) = string_arg(call, "query") else {
            return Ok(ToolResult::failure("query is required"));
        };
        let mut url = self.browser.search_endpoint.clone();
        url.query_pairs_mut().append_pair("q", query);
        Ok(match self.browser.fetch(url.as_str(), true).await {
            Ok(page) => ToolResult::success(page_text(&page)),
            Err(error) => ToolResult::failure(error),
        })
    }
}

fn web_spec(name: &str, description: &str, url: bool) -> ToolSpec {
    let key = if url { "url" } else { "query" };
    ToolSpec {
        name: name.into(),
        description: description.into(),
        category: ToolCategory::Internal,
        risk: ToolRisk::ReadOnly,
        parameters: serde_json::json!({
            "type":"object",
            "properties":{key:{"type":"string"}},
            "required":[key],
            "additionalProperties":false
        }),
    }
}

struct WebBackTool {
    browser: Arc<WebBrowser>,
}

#[async_trait]
impl Tool for WebBackTool {
    fn spec(&self) -> ToolSpec {
        empty_spec("web_back", "Go back one page in browser history.")
    }
    async fn execute(&self, _: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        Ok(self.browser.move_history(-1).map_or_else(
            || ToolResult::failure("no previous page"),
            |page| ToolResult::success(page_text(&page)),
        ))
    }
}

struct WebForwardTool {
    browser: Arc<WebBrowser>,
}

#[async_trait]
impl Tool for WebForwardTool {
    fn spec(&self) -> ToolSpec {
        empty_spec("web_forward", "Go forward one page in browser history.")
    }
    async fn execute(&self, _: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        Ok(self.browser.move_history(1).map_or_else(
            || ToolResult::failure("no next page"),
            |page| ToolResult::success(page_text(&page)),
        ))
    }
}

fn empty_spec(name: &str, description: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: description.into(),
        category: ToolCategory::Internal,
        risk: ToolRisk::ReadOnly,
        parameters: serde_json::json!({
            "type":"object","properties":{},"additionalProperties":false
        }),
    }
}

#[derive(Clone)]
struct DocumentIngestTool {
    heart: Arc<SpineHeart>,
    encoder: Arc<dyn SemanticEncoder>,
    cwd: PathBuf,
    running_tasks: Arc<RunningTaskManager>,
}

impl DocumentIngestTool {
    fn spec_value() -> ToolSpec {
        ToolSpec {
            name: "ingest_documents".into(),
            description: "Bulk-ingest text files, directories, or glob patterns into the encrypted heart with exact source provenance and deduplication.".into(),
            category: ToolCategory::Action,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({
                "type":"object",
                "properties":{
                    "paths":{"type":"array","items":{"type":"string"}},
                    "recursive":{"type":"boolean"},
                    "force":{"type":"boolean"},
                    "maintain":{"type":"boolean"},
                    "max_file_mb":{"type":"number","minimum":0.001,"maximum":1024},
                    "extensions":{"type":"array","items":{"type":"string"}},
                    "chunk_words":{"type":"integer","minimum":40,"maximum":10000},
                    "overlap_words":{"type":"integer","minimum":0,"maximum":5000}
                },
                "required":["paths"],
                "additionalProperties":false
            }),
        }
    }

    fn execute_sync(&self, call: &ToolCall) -> spine_runtime::Result<ToolResult> {
        let paths: Vec<&str> = call
            .arguments
            .get("paths")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        if paths.is_empty() {
            return Ok(ToolResult::failure(
                "paths must contain at least one string",
            ));
        }
        let recursive = call
            .arguments
            .get("recursive")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        let force = call
            .arguments
            .get("force")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let maintain = call
            .arguments
            .get("maintain")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        let maximum_bytes = (call
            .arguments
            .get("max_file_mb")
            .and_then(|value| value.as_f64())
            .unwrap_or(25.0)
            * 1_048_576.0) as u64;
        let chunk_words = call
            .arguments
            .get("chunk_words")
            .and_then(|value| value.as_u64())
            .unwrap_or(250)
            .clamp(40, 10_000) as usize;
        let overlap = call
            .arguments
            .get("overlap_words")
            .and_then(|value| value.as_u64())
            .unwrap_or(30) as usize;
        if overlap >= chunk_words {
            return Ok(ToolResult::failure(
                "overlap_words must be smaller than chunk_words",
            ));
        }
        let extensions: BTreeSet<String> = call
            .arguments
            .get("extensions")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str())
                    .map(|value| value.trim_start_matches('.').to_ascii_lowercase())
                    .collect()
            })
            .unwrap_or_else(default_extensions);
        let files = match discover_files(&self.cwd, &paths, recursive, &extensions) {
            Ok(files) => files,
            Err(error) => return Ok(ToolResult::failure(error)),
        };
        let mut seen_hashes: BTreeSet<String> = if force {
            BTreeSet::new()
        } else {
            self.heart
                .events_canonical()?
                .into_iter()
                .filter_map(|event| {
                    event
                        .body
                        .interaction
                        .provenance
                        .metadata
                        .get("document_chunk_hash")
                        .cloned()
                })
                .collect()
        };
        let mut pending = Vec::new();
        let mut skipped = 0_usize;
        for path in &files {
            let metadata = match fs::metadata(path) {
                Ok(metadata) => metadata,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            if metadata.len() > maximum_bytes {
                skipped += 1;
                continue;
            }
            let text = match fs::read_to_string(path) {
                Ok(text) if !text.contains('\0') => text,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
                Ok(_) => {
                    skipped += 1;
                    continue;
                }
            };
            for (index, chunk) in chunk_words_preserving(&text, chunk_words, overlap)
                .into_iter()
                .enumerate()
            {
                let hash = blake3::hash(chunk.as_bytes()).to_hex().to_string();
                if !seen_hashes.insert(hash.clone()) {
                    skipped += 1;
                    continue;
                }
                pending.push((path.clone(), index, hash, chunk));
            }
        }
        let mut items = Vec::with_capacity(pending.len());
        for batch in pending.chunks(32) {
            let texts: Vec<String> = batch.iter().map(|item| item.3.clone()).collect();
            let embeddings = self.encoder.encode_batch(&texts)?;
            for ((path, index, hash, text), embedding) in batch.iter().zip(embeddings) {
                let source = format!("file://{}", path.display());
                let mut metadata = BTreeMap::new();
                metadata.insert("document_chunk_hash".into(), hash.clone());
                metadata.insert("chunk_index".into(), index.to_string());
                let thread = format!(
                    "document-{}",
                    &blake3::hash(path.as_os_str().as_encoded_bytes()).to_hex()[..16]
                );
                items.push((
                    InteractionInput {
                        agent_id: AgentId::new("document-ingest")?,
                        thread_id: ThreadId::new(thread)?,
                        role: ParticipantRole::User,
                        kind: EventKind::Message,
                        content: Content::Inline(format!(
                            "[document: {}] [chunk: {}]\n{}",
                            path.display(),
                            index,
                            text
                        )),
                        causal_parents: Vec::new(),
                        provenance: Provenance {
                            provider: Some("spine-document-ingest".into()),
                            source_uri: Some(source),
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
        }
        let imported = items.len();
        self.heart.commit_embedded_batch(items)?;
        let maintenance = if maintain && imported > 0 {
            Some(self.heart.maintain_cognition(4)?)
        } else {
            None
        };
        Ok(ToolResult::success(format!(
            "files_discovered={} chunks_ingested={} skipped={} maintenance={}",
            files.len(),
            imported,
            skipped,
            maintenance.map_or_else(
                || "disabled".into(),
                |report| format!(
                    "merges:{},pruned:{},walks:{}",
                    report.merges, report.pruned, report.walks_completed
                )
            )
        )))
    }
}

#[async_trait]
impl Tool for DocumentIngestTool {
    fn spec(&self) -> ToolSpec {
        Self::spec_value()
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let task_id = self.running_tasks.begin("document ingestion", false);
        let runner = self.clone();
        let call = call.clone();
        let manager = Arc::clone(&self.running_tasks);
        let completed_id = task_id.clone();
        let worker = tokio::spawn(async move {
            let result = tokio::task::spawn_blocking(move || runner.execute_sync(&call))
                .await
                .map_err(|error| RuntimeError::Tool(error.to_string()))
                .and_then(|result| result)
                .unwrap_or_else(|error| ToolResult::failure(error.to_string()));
            manager.finish(&completed_id, result.clone());
            result
        });
        self.running_tasks.attach(&task_id, worker.abort_handle());
        match worker.await {
            Ok(result) => Ok(result),
            Err(error) if error.is_cancelled() => Ok(ToolResult::failure(format!(
                "document ingestion task {task_id} was cancelled"
            ))),
            Err(error) => Ok(ToolResult::failure(format!(
                "document ingestion task {task_id} failed to join: {error}"
            ))),
        }
    }
}

fn default_extensions() -> BTreeSet<String> {
    [
        "txt", "md", "markdown", "rst", "py", "rs", "js", "ts", "tsx", "jsx", "json", "jsonl",
        "toml", "yaml", "yml", "csv", "tsv", "xml", "html", "htm", "log", "css", "sh", "sql", "c",
        "h", "cpp", "hpp", "java", "go", "rb", "php", "swift", "kt", "tex",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn discover_files(
    cwd: &Path,
    values: &[&str],
    recursive: bool,
    extensions: &BTreeSet<String>,
) -> Result<Vec<PathBuf>, String> {
    let mut files = BTreeSet::new();
    for value in values {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        let resolved = resolve(cwd, value);
        if value.contains(['*', '?', '[']) {
            let pattern = resolved.to_string_lossy();
            for path in glob::glob(&pattern)
                .map_err(|error| error.to_string())?
                .flatten()
            {
                collect_path(&path, recursive, extensions, &mut files);
            }
        } else {
            collect_path(&resolved, recursive, extensions, &mut files);
        }
    }
    Ok(files.into_iter().collect())
}

fn collect_path(
    path: &Path,
    recursive: bool,
    extensions: &BTreeSet<String>,
    files: &mut BTreeSet<PathBuf>,
) {
    if path.is_file() {
        if accepted(path, extensions) {
            files.insert(path.to_path_buf());
        }
    } else if path.is_dir() {
        let walker = if recursive {
            WalkDir::new(path)
        } else {
            WalkDir::new(path).max_depth(1)
        };
        for entry in walker.into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() && accepted(entry.path(), extensions) {
                files.insert(entry.path().to_path_buf());
            }
        }
    }
}

fn accepted(path: &Path, extensions: &BTreeSet<String>) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| extensions.contains(&value.to_ascii_lowercase()))
}

fn chunk_words_preserving(text: &str, target: usize, overlap: usize) -> Vec<String> {
    let matcher = Regex::new(r"\S+").expect("static word regex");
    let words: Vec<_> = matcher.find_iter(text).collect();
    if words.is_empty() {
        return Vec::new();
    }
    let step = target - overlap;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + target).min(words.len());
        let byte_start = if start == 0 { 0 } else { words[start].start() };
        let byte_end = if end == words.len() {
            text.len()
        } else {
            words[end].start()
        };
        let chunk = &text[byte_start..byte_end];
        if !chunk.trim().is_empty() {
            chunks.push(chunk.to_owned());
        }
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, response::Html, routing::get};

    #[test]
    fn destructive_shell_calls_are_classified_dynamically() {
        assert!(ShellTool::destructive("rm -rf /tmp/example"));
        assert!(ShellTool::destructive("git reset --hard HEAD~1"));
        assert!(ShellTool::destructive(
            "Remove-Item -Recurse -Force C:\\example"
        ));
        assert!(ShellTool::destructive("Clear-Disk -Number 1"));
        assert!(!ShellTool::destructive("rg -n hello ."));
    }

    #[test]
    fn shell_children_do_not_inherit_spine_secrets() {
        let tool = ShellTool {
            cwd: std::env::current_dir().expect("cwd"),
            audit_path: std::env::temp_dir().join("unused-shell-audit.jsonl"),
            default_timeout: Duration::from_secs(1),
            running_tasks: Arc::new(RunningTaskManager::default()),
        };
        let process = tool.shell_process("true");
        #[cfg(target_os = "windows")]
        assert_eq!(
            process.as_std().get_program(),
            std::ffi::OsStr::new("powershell.exe")
        );
        #[cfg(not(target_os = "windows"))]
        assert!(
            ["/bin/bash", "/bin/sh", "sh"]
                .into_iter()
                .any(|shell| process.as_std().get_program() == std::ffi::OsStr::new(shell))
        );
        let environment = process.as_std().get_envs().collect::<Vec<_>>();
        for secret in ["SPINE_HEART_PASSPHRASE", "SPINE_LLM_API_KEY"] {
            assert!(
                environment.iter().any(|(name, value)| {
                    *name == std::ffi::OsStr::new(secret) && value.is_none()
                })
            );
        }
    }

    #[test]
    fn document_chunks_overlap_without_losing_words() {
        let chunks = chunk_words_preserving("one two three four five six", 4, 2);
        assert_eq!(chunks, ["one two three four ", "three four five six"]);
        let source = "  one\n two\tthree  four ";
        assert_eq!(chunk_words_preserving(source, 2, 0).concat(), source);
    }

    #[test]
    fn empty_document_path_does_not_expand_to_the_working_directory() {
        let temporary = tempfile::tempdir().unwrap();
        fs::write(temporary.path().join("should-not-be-found.txt"), "payload").unwrap();
        let files =
            discover_files(temporary.path(), &["", "   "], true, &default_extensions()).unwrap();
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn interrupted_shell_call_keeps_host_task_alive() {
        let manager = Arc::new(RunningTaskManager::default());
        let audit =
            std::env::temp_dir().join(format!("spine-shell-audit-{}.jsonl", std::process::id()));
        let tool = ShellTool {
            cwd: std::env::current_dir().expect("cwd"),
            audit_path: audit.clone(),
            default_timeout: Duration::from_secs(2),
            running_tasks: Arc::clone(&manager),
        };
        #[cfg(target_os = "windows")]
        let command = "Start-Sleep -Milliseconds 500; Write-Output survived";
        #[cfg(not(target_os = "windows"))]
        let command = "sleep 0.5; echo survived";
        let call = ToolCall {
            id: "test".into(),
            name: "shell".into(),
            arguments: serde_json::json!({"command": command, "timeout_s": 0}),
        };
        let outer = tokio::spawn(async move { tool.execute(&call, &ToolContext::default()).await });
        tokio::time::timeout(Duration::from_secs(3), async {
            while !manager.format().contains("status=running") {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("shell task did not start");
        outer.abort();
        let outer_error = outer.await.expect_err("outer shell call was not interrupted");
        assert!(outer_error.is_cancelled(), "{outer_error}");
        let status = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let status = manager.format();
                if status.contains("status=completed") && status.contains("survived") {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("detached shell task did not finish: {}", manager.format()));
        assert!(status.contains("status=completed"), "{status}");
        assert!(status.contains("survived"), "{status}");
        let _ = fs::remove_file(audit);
    }

    #[tokio::test]
    async fn web_search_uses_the_browser_pipeline_and_large_pages_fail_boundedly() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/search",
                get(|| async {
                    Html(
                        "<html><title>Search Results</title><body>deterministic result</body></html>"
                            .to_owned(),
                    )
                }),
            )
            .route(
                "/large",
                get(|| async { Html("x".repeat(MAX_WEB_BODY_BYTES + 1)) }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut browser = WebBrowser::new().unwrap();
        browser.search_endpoint = reqwest::Url::parse(&format!("http://{address}/search")).unwrap();
        let browser = Arc::new(browser);
        let search = WebSearchTool {
            browser: Arc::clone(&browser),
        }
        .execute(
            &ToolCall {
                id: "search".into(),
                name: "web_search".into(),
                arguments: serde_json::json!({"query":"spine tools"}),
            },
            &ToolContext::default(),
        )
        .await
        .unwrap();
        assert!(search.success, "{:?}", search.error);
        assert!(search.output.contains("Search Results"));
        assert!(search.output.contains("deterministic result"));

        let oversized = browser
            .fetch(&format!("http://{address}/large"), false)
            .await
            .unwrap_err();
        assert!(oversized.contains("response body exceeds"), "{oversized}");
        server.abort();
    }
}
