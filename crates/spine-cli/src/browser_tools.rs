//! Optional W3C WebDriver tools. The driver and browser are supplied by the user.
use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::{Client, Method, Url};
use serde_json::{Value, json};
use spine_runtime::{
    RuntimeError, Tool, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk,
    ToolSpec,
};
use tokio::sync::Mutex;

const ELEMENT: &str = "element-6066-11e4-a52e-4f735466cecf";
const MAX_RESPONSE_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_CHARS: usize = 12_000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const SNAPSHOT_ATTEMPTS: usize = 10;
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(3);
const SNAPSHOT_RETRY_DELAY: Duration = Duration::from_millis(75);

// Only fixed scripts are sent to WebDriver; user values are separate JSON arguments.
const SNAPSHOT_SCRIPT: &str = r#"
if (document.readyState === 'loading') return {ready:false};
const root = arguments[0] || document.body || document.documentElement;
if (!root) return {ready:false};
function selector(e) {
  if (e.id && document.querySelectorAll('#' + CSS.escape(e.id)).length === 1)
    return '#' + CSS.escape(e.id);
  const parts = [];
  for (; e && e.nodeType === 1; e = e.parentElement) {
    const tag = e.localName;
    let n = 1;
    for (let p = e.previousElementSibling; p; p = p.previousElementSibling)
      if (p.localName === tag) n++;
    parts.unshift(tag + ':nth-of-type(' + n + ')');
  }
  return parts.join(' > ');
}
const pattern = 'a[href],button,input:not([type=hidden]),textarea,select,[role=button]';
const nodes = root ? [...(root.matches(pattern) ? [root] : []), ...root.querySelectorAll(pattern)] : [];
const controls = [];
let omitted = 0;
for (const e of nodes) {
  if (!e.getClientRects().length || getComputedStyle(e).visibility === 'hidden') continue;
  if (controls.length >= 40) { omitted++; continue; }
  const css = selector(e);
  if (css.length > 1024) { omitted++; continue; }
  controls.push({selector: css, tag: e.localName, type: (e.getAttribute('type') || '').slice(0,40),
    label: (e.getAttribute('aria-label') || (e.labels && [...e.labels].map(l => l.innerText).join(' ')) || e.innerText || e.getAttribute('placeholder') || e.getAttribute('name') || '').slice(0,160),
    disabled: e.matches(':disabled'),
    ...(e.localName === 'select' ? {options: [...e.options].slice(0,20).map(o => ({value: o.value.slice(0,160), label: o.text.slice(0,160), selected:o.selected})), options_omitted: Math.max(0,e.options.length-20)} : {})});
}
const url = new URL(location.href); url.username = ''; url.password = '';
return {url: url.href.slice(0,2048), title: document.title.slice(0,512),
  text: (root ? root.innerText || '' : '').slice(0,6000), controls, controls_omitted: omitted};
"#;

const SELECT_SCRIPT: &str = r#"
const e = arguments[0], value = arguments[1];
if (e.localName !== 'select') return {error:'not_select'};
if (e.matches(':disabled')) return {error:'disabled'};
const options = [...e.options].filter(o => o.value === value);
if (options.length !== 1) return {error:'ambiguous_option'};
const option = options[0];
if (option.disabled || (option.parentElement.localName === 'optgroup' && option.parentElement.disabled)) return {error:'disabled'};
for (const o of e.options) o.selected = o === option;
e.dispatchEvent(new Event('input', {bubbles:true}));
e.dispatchEvent(new Event('change', {bubbles:true}));
return {ok:true};
"#;

const SUBMIT_SCRIPT: &str = r#"
const e = arguments[0];
const form = e.localName === 'form' ? e : e.form;
if (!form) return {error:'not_form'};
const submitter = (e.localName === 'button' && e.type === 'submit') || (e.localName === 'input' && (e.type === 'submit' || e.type === 'image'));
if (e.matches(':disabled')) return {error:'disabled'};
if (!form.noValidate && !(submitter && e.formNoValidate) && !HTMLFormElement.prototype.checkValidity.call(form)) return {error:'invalid_form'};
HTMLFormElement.prototype.requestSubmit.call(form, ...(submitter ? [e] : []));
return {ok:true};
"#;

pub(crate) fn register_browser_tools(registry: &mut ToolRegistry) -> spine_runtime::Result<()> {
    let endpoint = match std::env::var("SPINE_WEBDRIVER_URL") {
        Ok(endpoint) => endpoint,
        Err(std::env::VarError::NotPresent) => return Ok(()),
        Err(_) => {
            return Err(RuntimeError::InvalidConfig(
                "SPINE_WEBDRIVER_URL must be UTF-8".into(),
            ));
        }
    };
    let capabilities = std::env::var("SPINE_WEBDRIVER_CAPABILITIES")
        .map(Some)
        .or_else(|error| match error {
            std::env::VarError::NotPresent => Ok(None),
            _ => Err(RuntimeError::InvalidConfig(
                "SPINE_WEBDRIVER_CAPABILITIES must be UTF-8".into(),
            )),
        })?;
    register_with_endpoint(registry, Some(&endpoint), capabilities.as_deref())
}

fn register_with_endpoint(
    registry: &mut ToolRegistry,
    endpoint: Option<&str>,
    capabilities: Option<&str>,
) -> spine_runtime::Result<()> {
    let Some(endpoint) = endpoint else {
        return Ok(());
    };
    let browser = Arc::new(BrowserPool {
        template: Browser::new(endpoint, REQUEST_TIMEOUT, capabilities)?,
        sessions: Mutex::new(BTreeMap::new()),
    });
    for action in [
        Action::Open,
        Action::Snapshot,
        Action::Click,
        Action::Type,
        Action::Select,
        Action::Submit,
        Action::Close,
    ] {
        registry.register(BrowserTool {
            browser: Arc::clone(&browser),
            action,
        })?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Action {
    Open,
    Snapshot,
    Click,
    Type,
    Select,
    Submit,
    Close,
}

impl Action {
    fn name(self) -> &'static str {
        match self {
            Self::Open => "browser_open",
            Self::Snapshot => "browser_snapshot",
            Self::Click => "browser_click",
            Self::Type => "browser_type",
            Self::Select => "browser_select",
            Self::Submit => "browser_submit",
            Self::Close => "browser_close",
        }
    }
}

struct BrowserTool {
    browser: Arc<BrowserPool>,
    action: Action,
}

#[async_trait]
impl Tool for BrowserTool {
    fn spec(&self) -> ToolSpec {
        let (description, properties, required) = match self.action {
            Action::Open => (
                "Open an HTTP(S) URL in the optional JavaScript-capable browser. Creates a separate session on first use; returns page text and CSS controls.",
                json!({"url":{"type":"string"}}),
                vec!["url"],
            ),
            Action::Snapshot => (
                "Read current rendered page text and CSS controls. Optional selector scopes the snapshot to exactly one element. Repeat after asynchronous page updates; input values are omitted.",
                json!({"selector":{"type":"string"}}),
                vec![],
            ),
            Action::Click => (
                "Click exactly one element matched by CSS selector, then return the current page. Clicking can submit forms or change remote state.",
                json!({"selector":{"type":"string"}}),
                vec!["selector"],
            ),
            Action::Type => (
                "Replace text in exactly one CSS-matched editable element (clear defaults true); set clear false to append. Returns the current page without echoing input values.",
                json!({"selector":{"type":"string"},"text":{"type":"string","maxLength":12000},"clear":{"type":"boolean"}}),
                vec!["selector", "text"],
            ),
            Action::Select => (
                "Choose one option by its exact value in exactly one CSS-matched select; emits input/change events and returns the current page.",
                json!({"selector":{"type":"string"},"value":{"type":"string"}}),
                vec!["selector", "value"],
            ),
            Action::Submit => (
                "Submit the form matched by CSS selector, or the form associated with one matched control. Honors validation and submit handlers; returns the current page. Can change remote state.",
                json!({"selector":{"type":"string"}}),
                vec!["selector"],
            ),
            Action::Close => (
                "Close the separate WebDriver browser session and clear local session state. Call when finished.",
                json!({}),
                vec![],
            ),
        };
        ToolSpec {
            name: self.action.name().into(),
            description: description.into(),
            category: if self.action == Action::Snapshot {
                ToolCategory::Internal
            } else {
                ToolCategory::Action
            },
            risk: if self.action == Action::Snapshot {
                ToolRisk::ReadOnly
            } else {
                ToolRisk::Mutating
            },
            parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let result = self
            .browser
            .execute(self.action, &call.arguments, &context.harness_id)
            .await;
        Ok(match result {
            Ok(output) => {
                let limit = context
                    .metadata
                    .get("max_tool_result_chars")
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(MAX_OUTPUT_CHARS)
                    .min(MAX_OUTPUT_CHARS);
                ToolResult::success(bounded(&output, limit))
            }
            Err(error) => ToolResult::failure(error.message),
        })
    }
}

#[derive(Default)]
struct Session {
    id: Option<String>,
    interrupted: bool,
}

struct Browser {
    client: Client,
    endpoint: Url,
    capabilities: Value,
    session: Mutex<Session>,
}

struct BrowserPool {
    template: Browser,
    sessions: Mutex<BTreeMap<String, Arc<Browser>>>,
}

impl BrowserPool {
    async fn execute(
        &self,
        action: Action,
        args: &Value,
        harness_id: &str,
    ) -> DriverResult<String> {
        if harness_id.len() > 1024 {
            return Err(DriverError::local("Browser harness ID is too long"));
        }
        // Registry clones share the pool, but each harness gets a separate session.
        // Holding this lock also makes closing/removing entries atomic with tool use.
        let mut sessions = self.sessions.lock().await;
        if !sessions.contains_key(harness_id) {
            if action == Action::Close {
                return Ok("Browser session closed.".into());
            }
            if action != Action::Open {
                return Err(DriverError::local(
                    "No browser session; use browser_open first",
                ));
            }
            if sessions.len() >= 32 {
                return Err(DriverError::local(
                    "Browser session limit reached; close an existing harness session first",
                ));
            }
            sessions.insert(
                harness_id.into(),
                Arc::new(Browser {
                    client: self.template.client.clone(),
                    endpoint: self.template.endpoint.clone(),
                    capabilities: self.template.capabilities.clone(),
                    session: Mutex::new(Session::default()),
                }),
            );
        }
        let browser = Arc::clone(&sessions[harness_id]);
        let result = browser.execute(action, args).await;
        if browser.session.lock().await.id.is_none() {
            sessions.remove(harness_id);
        }
        result
    }
}

#[derive(Debug)]
struct DriverError {
    message: String,
    reset: bool,
    session_gone: bool,
}

impl DriverError {
    fn local(message: &str) -> Self {
        Self {
            message: message.into(),
            reset: false,
            session_gone: false,
        }
    }
    fn protocol(message: &str) -> Self {
        Self {
            message: message.into(),
            reset: true,
            session_gone: false,
        }
    }
}

type DriverResult<T> = Result<T, DriverError>;

impl Browser {
    fn new(
        endpoint: &str,
        timeout: Duration,
        capabilities: Option<&str>,
    ) -> spine_runtime::Result<Self> {
        let endpoint = Url::parse(endpoint).ok().filter(|url| {
            matches!(url.scheme(), "http" | "https") && url.host_str().is_some()
                && url.username().is_empty() && url.password().is_none() && url.query().is_none() && url.fragment().is_none()
        }).ok_or_else(|| RuntimeError::InvalidConfig("SPINE_WEBDRIVER_URL must be an HTTP(S) service URL without credentials, query, or fragment".into()))?;
        let mut capabilities = match capabilities {
            Some(value) if value.len() <= 8192 => serde_json::from_str::<Value>(value)
                .ok()
                .filter(Value::is_object),
            Some(_) => None,
            None => Some(json!({})),
        }
        .ok_or_else(|| {
            RuntimeError::InvalidConfig(
                "SPINE_WEBDRIVER_CAPABILITIES must be a JSON object of at most 8192 bytes".into(),
            )
        })?;
        // User capabilities can choose a browser and headless mode, but cannot remove
        // the finite timeouts or cause dialogs to be silently accepted.
        capabilities["timeouts"] = json!({"implicit":0,"pageLoad":15000,"script":5000});
        capabilities["pageLoadStrategy"] = json!("normal");
        capabilities["unhandledPromptBehavior"] = json!("dismiss and notify");
        let client = Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout.min(Duration::from_secs(5)))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| RuntimeError::InvalidConfig("could not create WebDriver client".into()))?;
        Ok(Self {
            client,
            endpoint,
            capabilities,
            session: Mutex::new(Session::default()),
        })
    }

    async fn request(
        &self,
        method: Method,
        path: &[&str],
        body: Option<Value>,
    ) -> DriverResult<Value> {
        let mut url = self.endpoint.clone();
        url.path_segments_mut()
            .map_err(|_| DriverError::protocol("invalid WebDriver service path"))?
            .pop_if_empty()
            .extend(path.iter().copied());
        let mut request = self.client.request(method, url);
        if let Some(body) = body {
            request = request.json(&body);
        }
        // Never display reqwest or remote error messages: either may contain secrets.
        let transport_error = |error: reqwest::Error| {
            DriverError::protocol(if error.is_timeout() {
                "WebDriver request timed out; session state is uncertain"
            } else {
                "WebDriver connection failed"
            })
        };
        let mut response = request.send().await.map_err(transport_error)?;
        let status = response.status();
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
            if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                return Err(DriverError::protocol(
                    "WebDriver response exceeded the size limit",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let envelope: Value = serde_json::from_slice(&bytes)
            .map_err(|_| DriverError::protocol("WebDriver returned invalid JSON"))?;
        let value = envelope
            .get("value")
            .ok_or_else(|| DriverError::protocol("WebDriver response has no value"))?;
        if !status.is_success() {
            return Err(match value.get("error").and_then(Value::as_str) {
                Some("invalid selector") => {
                    DriverError::local("WebDriver rejected the CSS selector")
                }
                Some("stale element reference") => DriverError::local(
                    "Element changed; take a new snapshot and retry the selector",
                ),
                Some(
                    "element not interactable"
                    | "element click intercepted"
                    | "invalid element state",
                ) => DriverError::local(
                    "Element is not currently interactable; inspect a new snapshot",
                ),
                Some("invalid session id") => DriverError {
                    message:
                        "Browser session is no longer available; use browser_open to start again"
                            .into(),
                    reset: true,
                    session_gone: true,
                },
                Some("no such window") => DriverError::protocol(
                    "Browser window is no longer available; use browser_open to start again",
                ),
                Some("timeout" | "script timeout") => DriverError::protocol(
                    "WebDriver operation timed out; session state is uncertain",
                ),
                _ => DriverError::protocol("WebDriver rejected the operation"),
            });
        }
        Ok(value.clone())
    }

    async fn close(&self, session: &mut Session) -> DriverResult<()> {
        if let Some(id) = session.id.as_deref() {
            let result = self.request(Method::DELETE, &["session", id], None).await;
            match result {
                Ok(_) => session.id = None,
                Err(error) if error.session_gone => session.id = None,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    async fn execute(&self, action: Action, args: &Value) -> DriverResult<String> {
        let required = |key| {
            args.get(key).and_then(Value::as_str).ok_or_else(|| {
                DriverError::local("Required browser argument is missing or is not a string")
            })
        };
        let selector = if matches!(
            action,
            Action::Click | Action::Type | Action::Select | Action::Submit
        ) || args.get("selector").is_some()
        {
            let value = required("selector")?;
            if value.trim().is_empty() || value.len() > 4096 {
                return Err(DriverError::local("selector must contain 1 to 4096 bytes"));
            }
            Some(value)
        } else {
            None
        };
        let target = if action == Action::Open {
            Some(
                Url::parse(required("url")?)
                    .ok()
                    .filter(|url| {
                        matches!(url.scheme(), "http" | "https")
                            && url.host_str().is_some()
                            && url.username().is_empty()
                            && url.password().is_none()
                    })
                    .ok_or_else(|| {
                        DriverError::local("url must be HTTP(S) without embedded credentials")
                    })?,
            )
        } else {
            None
        };
        if action == Action::Type
            && (required("text")?.chars().count() > 12000
                || args.get("clear").is_some_and(|v| !v.is_boolean()))
        {
            return Err(DriverError::local(
                "text is limited to 12000 characters and clear must be boolean",
            ));
        }
        if action == Action::Select && required("value")?.len() > 4096 {
            return Err(DriverError::local("option value is limited to 4096 bytes"));
        }

        // Serialize the whole action and observation, so concurrent tools cannot change
        // the page between element lookup and interaction. A dropped future leaves this
        // marker set and forces cleanup before the next operation uses the session.
        let mut session = self.session.lock().await;
        if session.interrupted {
            self.close(&mut session).await?;
            session.interrupted = false;
        }
        session.interrupted = true;
        let result = self
            .perform(action, args, selector, target.as_ref(), &mut session)
            .await;
        if result.as_ref().is_err_and(|e| e.reset) {
            if action != Action::Close {
                let _ = self.close(&mut session).await;
            }
            session.interrupted = session.id.is_some();
        } else {
            session.interrupted = false;
        }
        result
    }

    async fn perform(
        &self,
        action: Action,
        args: &Value,
        selector: Option<&str>,
        target: Option<&Url>,
        session: &mut Session,
    ) -> DriverResult<String> {
        if action == Action::Close {
            self.close(session).await?;
            return Ok("Browser session closed.".into());
        }
        if session.id.is_none() && action == Action::Open {
            let value = self
                .request(
                    Method::POST,
                    &["session"],
                    Some(json!({"capabilities":{"alwaysMatch":self.capabilities}})),
                )
                .await?;
            let id = value
                .get("sessionId")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty() && id.len() <= 512)
                .ok_or_else(|| {
                    DriverError::protocol("WebDriver did not return a valid session ID")
                })?;
            session.id = Some(id.to_owned());
        }
        let id = session
            .id
            .as_deref()
            .ok_or_else(|| DriverError::local("No browser session; use browser_open first"))?;
        if let Some(target) = target {
            self.request(
                Method::POST,
                &["session", id, "url"],
                Some(json!({"url":target.as_str()})),
            )
            .await?;
        }
        let element = if let Some(selector) = selector {
            let value = self
                .request(
                    Method::POST,
                    &["session", id, "elements"],
                    Some(json!({"using":"css selector","value":selector})),
                )
                .await?;
            let matches = value.as_array().ok_or_else(|| {
                DriverError::protocol("WebDriver returned invalid element matches")
            })?;
            if matches.len() != 1 {
                return Err(DriverError::local(if matches.is_empty() {
                    "CSS selector matched no elements; inspect a new snapshot"
                } else {
                    "CSS selector matched multiple elements; use a more specific selector"
                }));
            }
            let element_id = matches[0]
                .get(ELEMENT)
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty() && id.len() <= 512)
                .ok_or_else(|| {
                    DriverError::protocol("WebDriver returned an invalid element reference")
                })?;
            Some(element_id.to_owned())
        } else {
            None
        };
        if let Some(element_id) = element.as_deref() {
            match action {
                Action::Click => {
                    self.request(
                        Method::POST,
                        &["session", id, "element", element_id, "click"],
                        Some(json!({})),
                    )
                    .await?;
                }
                Action::Type => {
                    let input_type = self
                        .request(
                            Method::GET,
                            &["session", id, "element", element_id, "property", "type"],
                            None,
                        )
                        .await?;
                    if input_type.as_str() == Some("file") {
                        return Err(DriverError::local(
                            "File upload inputs are not supported by browser_type",
                        ));
                    }
                    if args.get("clear").and_then(Value::as_bool).unwrap_or(true) {
                        self.request(
                            Method::POST,
                            &["session", id, "element", element_id, "clear"],
                            Some(json!({})),
                        )
                        .await?;
                    }
                    self.request(
                        Method::POST,
                        &["session", id, "element", element_id, "value"],
                        Some(json!({"text":args["text"]})),
                    )
                    .await?;
                }
                Action::Select | Action::Submit => {
                    let script = if action == Action::Select {
                        SELECT_SCRIPT
                    } else {
                        SUBMIT_SCRIPT
                    };
                    let value = self
                        .script(id, script, json!([{ELEMENT:element_id}, args.get("value")]))
                        .await?;
                    if value.get("ok") != Some(&Value::Bool(true)) {
                        return Err(DriverError::local(
                            match value.get("error").and_then(Value::as_str) {
                                Some("not_select") => "Matched element is not a select",
                                Some("ambiguous_option") => {
                                    "Option value must match exactly one option"
                                }
                                Some("disabled") => "Matched control or option is disabled",
                                Some("not_form") => "Matched element has no associated form",
                                Some("invalid_form") => {
                                    "Form validation failed; inspect and complete the required fields"
                                }
                                _ => "Browser form operation failed",
                            },
                        ));
                    }
                }
                _ => {}
            }
        }
        let root = if action == Action::Snapshot {
            element.map(|id| json!({ELEMENT:id})).unwrap_or(Value::Null)
        } else {
            Value::Null
        };
        let snapshot = self.observe(id, root).await?;
        let mut output = format!(
            "URL: {}\nTitle: {}\n\nPage text:\n{}\n\nControls (CSS selectors):\n",
            snapshot["url"].as_str().unwrap_or(""),
            snapshot["title"].as_str().unwrap_or(""),
            snapshot["text"].as_str().unwrap_or("")
        );
        if let Some(controls) = snapshot["controls"].as_array() {
            for control in controls {
                output.push_str(&control.to_string());
                output.push('\n');
            }
        }
        if snapshot["controls_omitted"].as_u64().unwrap_or(0) > 0 {
            output.push_str(
                "More controls omitted; use browser_snapshot with a selector to narrow the page.\n",
            );
        }
        Ok(output)
    }

    async fn observe(&self, id: &str, root: Value) -> DriverResult<Value> {
        let not_ready = || {
            DriverError::local(
                "Page observation did not become ready; the preceding action may have completed. Use browser_snapshot before repeating an action",
            )
        };
        // A native form navigation can unload Firefox's script context between
        // submission and observation, producing a successful null script result.
        // Retry only this fixed read; never replay the preceding mutation.
        tokio::time::timeout(SNAPSHOT_TIMEOUT, async {
            for attempt in 0..SNAPSHOT_ATTEMPTS {
                let snapshot = self.script(id, SNAPSHOT_SCRIPT, json!([root])).await?;
                if snapshot.is_null() || snapshot.get("ready") == Some(&Value::Bool(false)) {
                    if attempt + 1 < SNAPSHOT_ATTEMPTS {
                        tokio::time::sleep(SNAPSHOT_RETRY_DELAY).await;
                        continue;
                    }
                    return Err(not_ready());
                }
                if !snapshot["url"].is_string()
                    || !snapshot["title"].is_string()
                    || !snapshot["text"].is_string()
                    || !snapshot["controls"].is_array()
                {
                    return Err(DriverError::protocol(
                        "WebDriver returned an invalid snapshot",
                    ));
                }
                return Ok(snapshot);
            }
            Err(not_ready())
        })
        .await
        .unwrap_or_else(|_| Err(not_ready()))
    }

    async fn script(&self, id: &str, script: &str, args: Value) -> DriverResult<Value> {
        self.request(
            Method::POST,
            &["session", id, "execute", "sync"],
            Some(json!({"script":script,"args":args})),
        )
        .await
    }
}

fn bounded(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.into();
    }
    let suffix = "\n[truncated; narrow browser_snapshot with selector]";
    if limit < suffix.len() {
        return value.chars().take(limit).collect();
    }
    value.chars().take(limit - suffix.len()).collect::<String>() + suffix
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::State,
        http::{Request, StatusCode},
        response::IntoResponse,
        routing::any,
    };
    use std::{collections::VecDeque, sync::Mutex as StdMutex};
    use tokio::{sync::Notify, task::JoinHandle};

    fn credential_url(base: &str) -> String {
        let mut url = Url::parse(base).unwrap();
        url.set_username("user").unwrap();
        url.set_password(Some("secret")).unwrap();
        url.to_string()
    }

    #[derive(Clone, Debug)]
    struct Recorded {
        method: Method,
        path: String,
        body: Value,
    }
    struct Reply {
        status: StatusCode,
        body: String,
        delay: Duration,
    }
    impl Reply {
        fn value(value: Value) -> Self {
            Self {
                status: StatusCode::OK,
                body: json!({"value":value}).to_string(),
                delay: Duration::ZERO,
            }
        }
        fn error(code: &str) -> Self {
            Self { status: StatusCode::INTERNAL_SERVER_ERROR, body: json!({"value":{"error":code,"message":"secret-password must never appear","stacktrace":"secret-stack"}}).to_string(), delay: Duration::ZERO }
        }
    }
    #[derive(Clone)]
    struct FixtureState {
        replies: Arc<StdMutex<VecDeque<Reply>>>,
        recorded: Arc<StdMutex<Vec<Recorded>>>,
        delayed: Arc<Notify>,
    }
    struct Fixture {
        url: String,
        state: FixtureState,
        task: JoinHandle<()>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.task.abort();
        }
    }
    impl Fixture {
        async fn start(replies: Vec<Reply>) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/wd/hub", listener.local_addr().unwrap());
            let state = FixtureState {
                replies: Arc::new(StdMutex::new(replies.into())),
                recorded: Arc::default(),
                delayed: Arc::default(),
            };
            async fn respond(
                State(state): State<FixtureState>,
                request: Request<Body>,
            ) -> impl IntoResponse {
                let (parts, body) = request.into_parts();
                let body = to_bytes(body, 512 * 1024).await.unwrap();
                state.recorded.lock().unwrap().push(Recorded {
                    method: parts.method,
                    path: parts.uri.path().into(),
                    body: if body.is_empty() {
                        Value::Null
                    } else {
                        serde_json::from_slice(&body).unwrap()
                    },
                });
                let reply = state
                    .replies
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("unexpected WebDriver request");
                if !reply.delay.is_zero() {
                    state.delayed.notify_one();
                    tokio::time::sleep(reply.delay).await;
                }
                (
                    reply.status,
                    [("content-type", "application/json")],
                    reply.body,
                )
            }
            let app = Router::new()
                .fallback(any(respond))
                .with_state(state.clone());
            let task = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            Self { url, state, task }
        }
        fn browser(&self, timeout: Duration) -> Arc<Browser> {
            Arc::new(Browser::new(&self.url, timeout, None).unwrap())
        }
        fn recorded(&self) -> Vec<Recorded> {
            self.state.recorded.lock().unwrap().clone()
        }
        fn exhausted(&self) {
            assert!(
                self.state.replies.lock().unwrap().is_empty(),
                "unused fixture replies"
            );
        }
    }
    fn null() -> Reply {
        Reply::value(Value::Null)
    }
    fn session() -> Reply {
        Reply::value(json!({"sessionId":"fresh-session","capabilities":{}}))
    }
    fn snapshot() -> Reply {
        Reply::value(
            json!({"url":"https://example.test/form","title":"Rendered form","text":"JavaScript rendered content","controls":[{"selector":"#name","tag":"input","label":"Name"}],"controls_omitted":0}),
        )
    }
    fn element() -> Reply {
        Reply::value(json!([{ELEMENT:"element/id"}]))
    }
    async fn open(browser: &Browser) -> String {
        browser
            .execute(Action::Open, &json!({"url":"https://example.test/form"}))
            .await
            .unwrap()
    }

    #[test]
    fn optional_registration_and_risk_are_explicit() {
        let mut registry = ToolRegistry::default();
        register_with_endpoint(&mut registry, None, None).unwrap();
        assert!(registry.is_empty());
        register_with_endpoint(&mut registry, Some("http://127.0.0.1:4444/wd/hub"), None).unwrap();
        assert_eq!(registry.len(), 7);
        for spec in registry.specs() {
            let snapshot = spec.name == "browser_snapshot";
            assert_eq!(
                spec.risk,
                if snapshot {
                    ToolRisk::ReadOnly
                } else {
                    ToolRisk::Mutating
                }
            );
            assert_eq!(
                spec.category,
                if snapshot {
                    ToolCategory::Internal
                } else {
                    ToolCategory::Action
                }
            );
        }
        let credential_endpoint = credential_url("http://localhost");
        for endpoint in [
            "",
            "file:///tmp/driver",
            credential_endpoint.as_str(),
            "http://localhost?token=secret",
        ] {
            let error = Browser::new(endpoint, REQUEST_TIMEOUT, None)
                .err()
                .unwrap()
                .to_string();
            assert!(!error.contains("secret"));
        }
        for capabilities in ["[]", "{secret-invalid-json", "null"] {
            let error = Browser::new("http://localhost:4444", REQUEST_TIMEOUT, Some(capabilities))
                .err()
                .unwrap()
                .to_string();
            assert!(!error.contains("secret"));
        }
        let browser = Browser::new("http://localhost:4444", REQUEST_TIMEOUT, Some(r#"{"browserName":"firefox","moz:firefoxOptions":{"args":["-headless"]},"timeouts":{"script":null}}"#)).unwrap();
        assert_eq!(browser.capabilities["browserName"], "firefox");
        assert_eq!(browser.capabilities["timeouts"]["script"], 5000);
        assert_eq!(
            browser.capabilities["unhandledPromptBehavior"],
            "dismiss and notify"
        );
    }

    #[tokio::test]
    async fn protocol_open_type_select_submit_click_and_close() {
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            element(),
            Reply::value(json!("text")),
            null(),
            null(),
            snapshot(),
            element(),
            Reply::value(json!({"ok":true})),
            snapshot(),
            element(),
            Reply::value(json!({"ok":true})),
            snapshot(),
            element(),
            null(),
            snapshot(),
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        assert!(open(&browser).await.contains("JavaScript rendered content"));
        let text = "quoted \" text \\ newline\n日本語";
        browser
            .execute(Action::Type, &json!({"selector":"#name","text":text}))
            .await
            .unwrap();
        browser
            .execute(
                Action::Select,
                &json!({"selector":"#choice","value":"a'\"\\b"}),
            )
            .await
            .unwrap();
        browser
            .execute(Action::Submit, &json!({"selector":"#form"}))
            .await
            .unwrap();
        browser
            .execute(Action::Click, &json!({"selector":"#next"}))
            .await
            .unwrap();
        browser.execute(Action::Close, &json!({})).await.unwrap();
        assert!(browser.session.lock().await.id.is_none());
        let requests = fixture.recorded();
        assert_eq!(requests.len(), 18);
        assert_eq!(requests[0].path, "/wd/hub/session");
        assert_eq!(
            requests[0].body["capabilities"]["alwaysMatch"]["timeouts"]["implicit"],
            0
        );
        assert_eq!(requests[1].body, json!({"url":"https://example.test/form"}));
        assert_eq!(
            requests[3].body,
            json!({"using":"css selector","value":"#name"})
        );
        assert_eq!(
            requests[5].path,
            "/wd/hub/session/fresh-session/element/element%2Fid/clear"
        );
        assert_eq!(requests[4].method, Method::GET);
        assert!(requests[4].path.ends_with("/property/type"));
        assert_eq!(requests[6].body, json!({"text":text}));
        assert_eq!(requests[9].body["script"], SELECT_SCRIPT);
        assert_eq!(
            requests[9].body["args"],
            json!([{ELEMENT:"element/id"},"a'\"\\b"])
        );
        assert_eq!(requests[12].body["script"], SUBMIT_SCRIPT);
        assert!(requests[15].path.ends_with("/click"));
        assert_eq!(requests[17].method, Method::DELETE);
        fixture.exhausted();
    }

    #[tokio::test]
    async fn selectors_are_unambiguous_and_form_errors_preserve_session() {
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            Reply::value(json!([])),
            Reply::value(json!([{ELEMENT:"one"},{ELEMENT:"two"}])),
            element(),
            Reply::value(json!({"error":"invalid_form"})),
            element(),
            Reply::error("stale element reference"),
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        assert!(
            browser
                .execute(Action::Click, &json!({"selector":".missing"}))
                .await
                .unwrap_err()
                .message
                .contains("no elements")
        );
        assert!(
            browser
                .execute(Action::Click, &json!({"selector":"button"}))
                .await
                .unwrap_err()
                .message
                .contains("multiple elements")
        );
        assert!(
            browser
                .execute(Action::Submit, &json!({"selector":"#form"}))
                .await
                .unwrap_err()
                .message
                .contains("validation failed")
        );
        let error = browser
            .execute(Action::Click, &json!({"selector":"#stale"}))
            .await
            .unwrap_err();
        assert!(error.message.contains("Element changed"));
        assert!(!error.message.contains("secret"));
        assert!(browser.session.lock().await.id.is_some());
        browser.execute(Action::Close, &json!({})).await.unwrap();
        fixture.exhausted();
    }

    #[tokio::test]
    async fn dead_session_is_cleared_and_open_can_recover() {
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            Reply::error("invalid session id"),
            null(),
            session(),
            null(),
            snapshot(),
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        let error = browser
            .execute(Action::Snapshot, &json!({}))
            .await
            .unwrap_err();
        assert!(error.message.contains("no longer available"));
        assert!(browser.session.lock().await.id.is_none());
        open(&browser).await;
        browser.execute(Action::Close, &json!({})).await.unwrap();
        let requests = fixture.recorded();
        assert_eq!(requests[4].method, Method::DELETE);
        assert_eq!(requests[5].path, "/wd/hub/session");
        fixture.exhausted();
    }

    #[tokio::test]
    async fn timeout_size_and_malformed_response_fail_closed() {
        let mut delayed = snapshot();
        delayed.delay = Duration::from_millis(200);
        let malformed = Reply {
            status: StatusCode::OK,
            body: "secret-invalid-json".into(),
            delay: Duration::ZERO,
        };
        let oversized = Reply::value(json!("x".repeat(MAX_RESPONSE_BYTES)));
        for (reply, expected) in [
            (delayed, "timed out"),
            (malformed, "invalid JSON"),
            (oversized, "size limit"),
        ] {
            let fixture = Fixture::start(vec![session(), null(), snapshot(), reply, null()]).await;
            let browser = fixture.browser(Duration::from_millis(80));
            open(&browser).await;
            let error = browser
                .execute(Action::Snapshot, &json!({}))
                .await
                .unwrap_err();
            assert!(error.message.contains(expected), "{}", error.message);
            assert!(!error.message.contains("secret"));
            assert!(browser.session.lock().await.id.is_none());
            assert_eq!(fixture.recorded().last().unwrap().method, Method::DELETE);
            fixture.exhausted();
        }
    }

    #[tokio::test]
    async fn cancelled_action_is_cleaned_before_next_use() {
        let mut delayed = null();
        delayed.delay = Duration::from_secs(5);
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            element(),
            delayed,
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        let pending = Arc::clone(&browser);
        let task = tokio::spawn(async move {
            pending
                .execute(Action::Click, &json!({"selector":"#pending"}))
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), fixture.state.delayed.notified())
            .await
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let error = browser
            .execute(Action::Snapshot, &json!({}))
            .await
            .unwrap_err();
        assert!(error.message.contains("use browser_open"));
        assert_eq!(fixture.recorded().last().unwrap().method, Method::DELETE);
        fixture.exhausted();
    }

    #[tokio::test]
    async fn scoped_snapshot_and_output_budget() {
        let mut long = snapshot();
        long.body = json!({"value":{"url":"https://example.test/","title":"Test","text":"語".repeat(13000),"controls":[]}}).to_string();
        let fixture =
            Fixture::start(vec![session(), null(), snapshot(), element(), long, null()]).await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        let tool = BrowserTool {
            browser: Arc::new(BrowserPool {
                template: Browser::new(&fixture.url, REQUEST_TIMEOUT, None).unwrap(),
                sessions: Mutex::new([("".into(), Arc::clone(&browser))].into()),
            }),
            action: Action::Snapshot,
        };
        let context = ToolContext {
            metadata: [("max_tool_result_chars".into(), "1000".into())].into(),
            ..Default::default()
        };
        let result = tool
            .execute(
                &ToolCall {
                    id: "snapshot".into(),
                    name: "browser_snapshot".into(),
                    arguments: json!({"selector":"#section"}),
                },
                &context,
            )
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output.chars().count(), 1000);
        assert!(result.output.contains("truncated"));
        assert_eq!(
            fixture.recorded()[4].body["args"],
            json!([{ELEMENT:"element/id"}])
        );
        browser.execute(Action::Close, &json!({})).await.unwrap();
        fixture.exhausted();
    }

    #[tokio::test]
    async fn validation_and_no_session_do_not_contact_driver() {
        let fixture = Fixture::start(vec![]).await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        for (action, args) in [
            (Action::Open, json!({"url":"file:///etc/passwd"})),
            (
                Action::Open,
                json!({"url":credential_url("https://example.test")}),
            ),
            (
                Action::Type,
                json!({"selector":"#name","text":"ok","clear":"yes"}),
            ),
            (Action::Click, json!({"selector":" "})),
            (Action::Snapshot, json!({})),
        ] {
            assert!(browser.execute(action, &args).await.is_err());
        }
        assert!(fixture.recorded().is_empty());
    }

    #[tokio::test]
    async fn typing_cannot_upload_driver_files() {
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            element(),
            Reply::value(json!("file")),
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        let error = browser
            .execute(
                Action::Type,
                &json!({"selector":"#file","text":"/private/file"}),
            )
            .await
            .unwrap_err();
        assert!(error.message.contains("File upload"));
        browser.execute(Action::Close, &json!({})).await.unwrap();
        assert!(
            !fixture
                .recorded()
                .iter()
                .any(|r| r.path.ends_with("/value") || r.path.ends_with("/clear"))
        );
        fixture.exhausted();
    }

    #[tokio::test]
    async fn navigation_retries_only_observation_after_one_submission() {
        let landed = Reply::value(
            json!({"url":"https://example.test/received","title":"Received","text":"Submission received","controls":[]}),
        );
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            element(),
            Reply::value(json!({"ok":true})),
            null(),
            Reply::value(json!({"ready":false})),
            landed,
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        let output = browser
            .execute(Action::Submit, &json!({"selector":"#form"}))
            .await
            .unwrap();
        assert!(output.contains("Submission received"));
        assert!(output.contains("https://example.test/received"));
        browser.execute(Action::Close, &json!({})).await.unwrap();
        let requests = fixture.recorded();
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.body.get("script") == Some(&json!(SUBMIT_SCRIPT)))
                .count(),
            1
        );
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.body.get("script") == Some(&json!(SNAPSHOT_SCRIPT)))
                .count(),
            4
        );
        assert!(
            requests[5..8]
                .iter()
                .all(|r| r.body.get("script") == Some(&json!(SNAPSHOT_SCRIPT)))
        );
        fixture.exhausted();
    }

    #[tokio::test]
    async fn unavailable_observation_is_bounded_and_does_not_replay_action() {
        let mut replies = vec![session(), null(), snapshot(), element(), null()];
        replies.extend((0..SNAPSHOT_ATTEMPTS).map(|_| null()));
        replies.push(snapshot());
        replies.push(null());
        let fixture = Fixture::start(replies).await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        let error = tokio::time::timeout(
            Duration::from_secs(2),
            browser.execute(Action::Click, &json!({"selector":"#next"})),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(error.message.contains("may have completed"));
        assert!(error.message.contains("browser_snapshot"));
        assert!(browser.session.lock().await.id.is_some());
        browser.execute(Action::Snapshot, &json!({})).await.unwrap();
        browser.execute(Action::Close, &json!({})).await.unwrap();
        assert_eq!(
            fixture
                .recorded()
                .iter()
                .filter(|r| r.path.ends_with("/click"))
                .count(),
            1
        );
        fixture.exhausted();
    }

    #[tokio::test]
    async fn malformed_snapshot_is_not_retried() {
        for value in [json!({}), json!([])] {
            let fixture = Fixture::start(vec![
                session(),
                null(),
                snapshot(),
                Reply::value(value),
                null(),
            ])
            .await;
            let browser = fixture.browser(REQUEST_TIMEOUT);
            open(&browser).await;
            assert!(
                browser
                    .execute(Action::Snapshot, &json!({}))
                    .await
                    .unwrap_err()
                    .message
                    .contains("invalid snapshot")
            );
            assert_eq!(fixture.recorded().len(), 5);
            assert!(browser.session.lock().await.id.is_none());
            fixture.exhausted();
        }
    }

    #[tokio::test]
    async fn failed_cleanup_is_retried_before_creating_another_session() {
        let fixture = Fixture::start(vec![
            session(),
            null(),
            snapshot(),
            Reply::error("unknown error"),
            Reply::error("unknown error"),
            null(),
            session(),
            null(),
            snapshot(),
            null(),
        ])
        .await;
        let browser = fixture.browser(REQUEST_TIMEOUT);
        open(&browser).await;
        assert!(browser.execute(Action::Close, &json!({})).await.is_err());
        assert!(browser.session.lock().await.id.is_some());
        assert!(
            browser
                .execute(Action::Open, &json!({"url":"https://example.test/"}))
                .await
                .is_err()
        );
        assert_eq!(fixture.recorded().last().unwrap().method, Method::DELETE);
        open(&browser).await;
        browser.execute(Action::Close, &json!({})).await.unwrap();
        let requests = fixture.recorded();
        assert!(requests[3..6].iter().all(|r| r.method == Method::DELETE));
        assert_eq!(requests[6].path, "/wd/hub/session");
        fixture.exhausted();
    }

    #[tokio::test]
    async fn cloned_registry_sessions_are_isolated_by_harness() {
        let page = |url: &str| {
            Reply::value(json!({"url":url,"title":"Isolated","text":"page","controls":[]}))
        };
        let fixture = Fixture::start(vec![
            Reply::value(json!({"sessionId":"parent"})),
            null(),
            page("https://parent.test/"),
            Reply::value(json!({"sessionId":"child"})),
            null(),
            page("https://child.test/"),
            page("https://parent.test/"),
            page("https://child.test/"),
            null(),
            null(),
        ])
        .await;
        let mut registry = ToolRegistry::default();
        register_with_endpoint(&mut registry, Some(&fixture.url), None).unwrap();
        let child_registry = registry.clone();
        let parent = ToolContext {
            harness_id: "parent-harness".into(),
            ..Default::default()
        };
        let child = ToolContext {
            harness_id: "child-harness".into(),
            ..Default::default()
        };
        let call = |name: &str, arguments| ToolCall {
            id: name.into(),
            name: name.into(),
            arguments,
        };
        registry
            .get("browser_open")
            .unwrap()
            .execute(
                &call("browser_open", json!({"url":"https://parent.test/"})),
                &parent,
            )
            .await
            .unwrap();
        let no_session = child_registry
            .get("browser_snapshot")
            .unwrap()
            .execute(&call("browser_snapshot", json!({})), &child)
            .await
            .unwrap();
        assert!(!no_session.success);
        child_registry
            .get("browser_open")
            .unwrap()
            .execute(
                &call("browser_open", json!({"url":"https://child.test/"})),
                &child,
            )
            .await
            .unwrap();
        for (context, expected) in [
            (&parent, "https://parent.test/"),
            (&child, "https://child.test/"),
        ] {
            let result = registry
                .get("browser_snapshot")
                .unwrap()
                .execute(&call("browser_snapshot", json!({})), context)
                .await
                .unwrap();
            assert!(result.success);
            assert!(result.output.contains(expected));
        }
        for context in [&parent, &child] {
            registry
                .get("browser_close")
                .unwrap()
                .execute(&call("browser_close", json!({})), context)
                .await
                .unwrap();
        }
        let requests = fixture.recorded();
        assert_eq!(requests[6].path, "/wd/hub/session/parent/execute/sync");
        assert_eq!(requests[7].path, "/wd/hub/session/child/execute/sync");
        assert_eq!(requests[8].path, "/wd/hub/session/parent");
        assert_eq!(requests[9].path, "/wd/hub/session/child");
        fixture.exhausted();
    }

    // Explicit opt-in only: supplied WebDriver plus a local page, with no public network.
    #[tokio::test]
    #[ignore = "requires SPINE_WEBDRIVER_TEST_URL pointing to a user-supplied W3C driver"]
    async fn real_browser_javascript_and_form_roundtrip() {
        async fn step(
            browser: &Browser,
            action: Action,
            args: Value,
            phase: &str,
        ) -> DriverResult<String> {
            browser.execute(action, &args).await.map_err(|mut error| {
                error.message = format!("{phase}: {}", error.message);
                error
            })
        }
        let endpoint =
            std::env::var("SPINE_WEBDRIVER_TEST_URL").expect("set SPINE_WEBDRIVER_TEST_URL");
        let caps = std::env::var("SPINE_WEBDRIVER_CAPABILITIES").ok();
        let browser = Browser::new(&endpoint, REQUEST_TIMEOUT, caps.as_deref()).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let page = format!("http://{}", listener.local_addr().unwrap());
        let html = r#"<!doctype html><title>Form fixture</title><main id="app"></main><script>
        document.querySelector('#app').innerHTML = '<h1>Rendered by JavaScript</h1><form id="form"><label>Name<input required id="name" name="name"></label><label>Choice<select id="choice" name="choice"><option value="a">Alpha</option><option value="b">Beta</option></select></label><button id="send">Send</button></form><p id="result"></p><form id="navigate" action="/landing"><input name="query" value="fixture"><button>Navigate</button></form>';
        document.querySelector('#form').addEventListener('submit', e => { e.preventDefault(); document.title = 'Submitted'; document.querySelector('#result').textContent = 'Submitted ' + document.querySelector('#name').value + ' / ' + document.querySelector('#choice').value; });
        </script>"#;
        let app = Router::new()
            .route(
                "/",
                axum::routing::get(move || async move { axum::response::Html(html) }),
            )
            .route(
                "/landing",
                axum::routing::get(|| async {
                    axum::response::Html(
                        "<title>Navigation complete</title><h1>Navigation form received</h1>",
                    )
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let result = async {
            let output = step(
                &browser,
                Action::Open,
                json!({"url":page}),
                "open rendered page",
            )
            .await?;
            assert!(output.contains("Rendered by JavaScript"));
            assert!(output.contains("#name"));
            step(
                &browser,
                Action::Type,
                json!({"selector":"#name","text":"Ada"}),
                "type name",
            )
            .await?;
            step(
                &browser,
                Action::Select,
                json!({"selector":"#choice","value":"b"}),
                "select option",
            )
            .await?;
            let output = step(
                &browser,
                Action::Submit,
                json!({"selector":"#form"}),
                "submit handler form",
            )
            .await?;
            assert!(output.contains("Submitted Ada / b"), "{output}");
            assert!(output.contains("Title: Submitted"));
            let output = step(
                &browser,
                Action::Click,
                json!({"selector":"#send"}),
                "click handler submitter",
            )
            .await?;
            assert!(output.contains("Submitted Ada / b"));
            let output = step(
                &browser,
                Action::Submit,
                json!({"selector":"#navigate"}),
                "submit navigation form",
            )
            .await?;
            assert!(output.contains("Navigation form received"), "{output}");
            assert!(output.contains("/landing?query=fixture"), "{output}");
            DriverResult::Ok(())
        }
        .await;
        let closed = browser.execute(Action::Close, &json!({})).await;
        server.abort();
        result.unwrap();
        closed.unwrap();
    }
}
