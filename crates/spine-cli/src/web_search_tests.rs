use super::*;
use axum::{
    Router,
    extract::{Path as RequestPath, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect},
    routing::get,
};

type Attempts = Arc<Mutex<Vec<String>>>;

struct SearchFixture {
    base: String,
    attempts: Attempts,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for SearchFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl SearchFixture {
    async fn start() -> Self {
        let attempts: Attempts = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/{page}",
                get(|RequestPath(page): RequestPath<String>, State(attempts): State<Attempts>| async move {
                    attempts.lock().unwrap().push(page.clone());
                    let html = match page.as_str() {
                        "js" => "<title>Google Search</title><script>location.href='/real-results'</script>Please click here if you are not redirected within a few seconds.<a href='/enablejs'>Retry with JavaScript</a><a href='https://support.google.com/websearch'>Help with Search</a>",
                        "consent" => "<title>Before you continue</title><form action='/consent'><button>Accept all</button></form><a href='https://policies.google.com/privacy'>Privacy Policy</a>",
                        "captcha" => "<title>Search</title><form id='challenge-form' action='/anomaly.js'><input name='g-recaptcha-response'></form><a href='/privacy'>Privacy Policy</a>",
                        "challenge" => "<title>Just a moment...</title>Checking your browser.<a href='https://www.cloudflare.com/'>Cloudflare</a>",
                        "script" => "<title>Search</title><script>const html = \"<a href='https://example.com/article'>Fake result</a>\";</script><noscript>Enable JavaScript</noscript>",
                        "empty" => "<title>Search</title><a href='/'>Home</a><form><input name='q'></form>",
                        "redirect" => return Redirect::temporary("/enablejs").into_response(),
                        // Even a link that looks usable must not make the final challenge URL valid.
                        "enablejs" => "<title>Search</title><a href='https://example.com/article'>Browser documentation</a>",
                        "good" => "<title>Real search results</title><a href='/evidence'>Rust language result</a>",
                        "topics" => "<title>JavaScript CAPTCHA consent search results</title><a href='https://example.com/captcha'>CAPTCHA and JavaScript guide</a><p>Please enable JavaScript to use our demo.</p><a href='https://example.com/consent'>Consent forms explained</a>",
                        "no-results" => "<title>Search</title><div class='no-results'>No results found for <b>some query</b></div><a href='/settings'>Settings</a>",
                        "before" => "<title>Before</title>Earlier page",
                        "after" => "<title>After</title>Later page",
                        _ => return (StatusCode::SERVICE_UNAVAILABLE, "engine unavailable").into_response(),
                    };
                    Html(html).into_response()
                }),
            )
            .with_state(Arc::clone(&attempts));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            base,
            attempts,
            server,
        }
    }

    fn browser(&self, paths: &[&str]) -> Arc<WebBrowser> {
        let mut browser = WebBrowser::new().unwrap();
        browser.search_endpoints = paths
            .iter()
            .map(|path| reqwest::Url::parse(&format!("{}/{path}", self.base)).unwrap())
            .collect();
        Arc::new(browser)
    }

    fn paths(&self) -> Vec<String> {
        self.attempts.lock().unwrap().clone()
    }
}

async fn search(browser: &Arc<WebBrowser>) -> ToolResult {
    WebSearchTool {
        browser: Arc::clone(browser),
    }
    .execute(
        &ToolCall {
            id: "search".into(),
            name: "web_search".into(),
            arguments: serde_json::json!({"query":"JavaScript CAPTCHA consent"}),
        },
        &ToolContext::default(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn web_search_http_200_walls_fall_back_and_only_record_results() {
    let fixture = SearchFixture::start().await;
    let browser = fixture.browser(&["js", "consent", "captcha", "challenge", "redirect", "good"]);
    let result = search(&browser).await;
    assert!(result.success, "{:?}", result.error);
    assert!(
        result.output.contains("Real search results"),
        "{}",
        result.output
    );
    assert_eq!(
        fixture.paths(),
        [
            "js",
            "consent",
            "captcha",
            "challenge",
            "redirect",
            "enablejs",
            "good"
        ]
    );
    let state = browser.state.lock().unwrap();
    assert_eq!(state.history.len(), 1);
    assert_eq!(state.history[0].title, "Real search results");
    assert_eq!(state.cursor, Some(0));
}

#[tokio::test]
async fn web_search_all_walls_fail_without_changing_existing_history() {
    let fixture = SearchFixture::start().await;
    let browser = fixture.browser(&["js", "consent", "captcha", "challenge", "script", "empty"]);
    browser
        .fetch(&format!("{}/before", fixture.base), true)
        .await
        .unwrap();
    browser
        .fetch(&format!("{}/after", fixture.base), true)
        .await
        .unwrap();
    assert_eq!(browser.move_history(-1).unwrap().title, "Before");
    fixture.attempts.lock().unwrap().clear();
    let result = search(&browser).await;
    assert!(!result.success, "{}", result.output);
    assert!(result.output.is_empty());
    let error = result.error.unwrap();
    assert!(error.contains("all configured engines"), "{error}");
    assert!(error.contains("no usable search results"), "{error}");
    assert_eq!(
        fixture.paths(),
        ["js", "consent", "captcha", "challenge", "script", "empty"]
    );
    {
        let state = browser.state.lock().unwrap();
        assert_eq!(state.cursor, Some(0));
        assert_eq!(state.history.len(), 2);
    }
    assert_eq!(browser.move_history(1).unwrap().title, "After");
}

#[tokio::test]
async fn web_search_topic_keywords_and_explicit_no_results_are_valid() {
    let fixture = SearchFixture::start().await;
    for (endpoint, expected) in [
        ("topics", "CAPTCHA and JavaScript guide"),
        ("no-results", "No results found for"),
    ] {
        fixture.attempts.lock().unwrap().clear();
        let browser = fixture.browser(&[endpoint, "good"]);
        let result = search(&browser).await;
        assert!(result.success, "{:?}", result.error);
        assert!(result.output.contains(expected));
        assert_eq!(fixture.paths(), [endpoint]);
        assert_eq!(browser.state.lock().unwrap().history.len(), 1);
    }
}

#[test]
fn web_search_provider_redirects_are_results_but_boilerplate_is_not() {
    let results = vec![(
        "JavaScript and CAPTCHA tutorial".into(),
        "https://html.duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fcaptcha".into(),
    )];
    assert_eq!(
        search_page_error(
            "",
            "https://html.duckduckgo.com/html/?q=captcha",
            "CAPTCHA",
            "CAPTCHA",
            &results
        ),
        None
    );
    let boilerplate = vec![(
        "Google Search help".into(),
        "https://support.google.com/websearch".into(),
    )];
    assert!(
        search_page_error(
            "",
            "https://www.google.com/search?q=captcha",
            "Google Search",
            "Search help",
            &boilerplate
        )
        .is_some()
    );
    let provider_documentation = vec![(
        "reCAPTCHA documentation".into(),
        "https://developers.google.com/recaptcha/docs/display".into(),
    )];
    assert_eq!(
        search_page_error(
            "",
            "https://www.google.com/search?q=captcha",
            "CAPTCHA results",
            "reCAPTCHA documentation",
            &provider_documentation
        ),
        None
    );
    let wrapped_provider_documentation = vec![(
        "Google Docs help for JavaScript".into(),
        "https://www.google.com/url?q=https%3A%2F%2Fdocs.google.com%2Fdocument%2Fd%2Fexample"
            .into(),
    )];
    assert_eq!(
        search_page_error(
            "",
            "https://www.google.com/search?q=JavaScript",
            "JavaScript results",
            "Google Docs help for JavaScript",
            &wrapped_provider_documentation
        ),
        None
    );
}
