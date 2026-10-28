//! First-run assets and deliberately non-secret launch preferences.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
const REPOSITORY: &str = "sentence-transformers/all-MiniLM-L6-v2";
const NLI_REVISION: &str = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d";
const NLI_REPOSITORY: &str = "cross-encoder/nli-MiniLM2-L6-H768";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub version: u32,
    pub server_url: Option<String>,
    pub server_model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub max_context_tokens: Option<usize>,
    pub heart: Option<PathBuf>,
    pub model_dir: Option<PathBuf>,
    pub workspace: Option<PathBuf>,
    pub web: bool,
    pub web_port: u16,
    pub no_nli: bool,
    pub nli_model_dir: Option<PathBuf>,
    pub skip_onboarding: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: 1,
            server_url: None,
            server_model: None,
            reasoning_effort: None,
            max_context_tokens: None,
            heart: None,
            model_dir: None,
            workspace: None,
            web: false,
            web_port: 8088,
            no_nli: false,
            nli_model_dir: None,
            skip_onboarding: false,
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => return Err(error),
        };
        let mut bytes = Vec::new();
        file.take(65_537).read_to_end(&mut bytes)?;
        if bytes.len() > 65_536 {
            return Err(invalid(
                "Settings file is too large; use --no-settings to bypass it.",
            ));
        }
        let settings: Self = serde_json::from_slice(&bytes).map_err(|_| {
            invalid("Cannot read settings; repair the JSON file or use --no-settings to bypass it.")
        })?;
        settings.validate()?;
        Ok(settings)
    }

    fn validate(&self) -> io::Result<()> {
        if self.version != 1 {
            return Err(invalid(
                "Unsupported settings version; use --no-settings to bypass it.",
            ));
        }
        if let Some(url) = &self.server_url {
            crate::endpoint_prompt::normalize(url)?;
        }
        if self.max_context_tokens == Some(0) {
            return Err(invalid("Saved context size must be positive."));
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        self.validate()?;
        if Self::load(path).ok().as_ref() == Some(self) && path.is_file() {
            return Ok(());
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        private_directory(parent)?;
        // NamedTempFile uses restrictive permissions and replaces atomically on
        // Unix and Windows. Never truncate the last working configuration.
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        serde_json::to_writer_pretty(&mut temporary, self)?;
        temporary.write_all(b"\n")?;
        temporary.as_file().sync_all()?;
        temporary.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

pub fn config_path() -> io::Result<PathBuf> {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return Ok(data_directory()?.join("settings.json"));
    }
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| crate::user_home_directory().map(|home| home.join(".config")))
        .ok_or_else(|| invalid("Set --config or SPINE_CONFIG to a settings file location."))?;
    Ok(base.join("spine/settings.json"))
}

fn data_directory() -> io::Result<PathBuf> {
    let home = crate::user_home_directory();
    let directory = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.map(|home| home.join("AppData/Local")))
            .map(|path| path.join("Spine"))
    } else if cfg!(target_os = "macos") {
        home.map(|home| home.join("Library/Application Support/Spine"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.map(|home| home.join(".local/share")))
            .map(|path| path.join("spine"))
    };
    directory
        .ok_or_else(|| invalid("Cannot locate app data; set SPINE_MINILM_DIR to a local model."))
}

pub fn minilm_directory() -> io::Result<PathBuf> {
    Ok(data_directory()?
        .join("models/all-MiniLM-L6-v2")
        .join(REVISION))
}

pub fn nli_directory() -> io::Result<PathBuf> {
    Ok(data_directory()?
        .join("models/nli-MiniLM2-L6-H768")
        .join(NLI_REVISION))
}

pub fn absolute(path: PathBuf) -> io::Result<PathBuf> {
    std::path::absolute(crate::expand_home(path))
}

pub fn choose<T>(matches: &clap::ArgMatches, name: &str, current: T, saved: T) -> T {
    if matches
        .value_source(name)
        .is_some_and(|source| source != clap::parser::ValueSource::DefaultValue)
    {
        current
    } else {
        saved
    }
}

fn private_directory(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub async fn validate_connection(provider: &spine_runtime::LlamaCppProvider) -> io::Result<()> {
    match tokio::time::timeout(Duration::from_secs(20), provider.health()).await {
        Ok(Ok(())) => Ok(()),
        // Do not echo provider response bodies: they may contain credentials.
        Ok(Err(_)) => Err(io::Error::other(
            "Cannot connect to the model server. Check that it is running and SPINE_LLM_API_KEY is set if required. Use --setup or --server-url to change the address. Existing settings were not changed.",
        )),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Model server check timed out after 20 seconds. Check the address or use --setup. Existing settings were not changed.",
        )),
    }
}

struct Asset<'a> {
    name: &'a str,
    size: u64,
    sha256: &'a str,
}

// SHA-256 of the exact public files at REVISION, independently checked with hf.
const ASSETS: &[Asset<'static>] = &[
    Asset {
        name: "config.json",
        size: 612,
        sha256: "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    },
    Asset {
        name: "tokenizer.json",
        size: 466_247,
        sha256: "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    },
    Asset {
        name: "model.safetensors",
        size: 90_868_376,
        sha256: "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    },
];

const NLI_ASSETS: &[Asset<'static>] = &[
    Asset {
        name: "config.json",
        size: 875,
        sha256: "8b0e41caff7567c0f53e6983f35591c3dec59507c9173ab125c5823394fb57f3",
    },
    Asset {
        name: "tokenizer.json",
        size: 1_356_048,
        sha256: "82139106e603ee4e1d5bc99d056ccbed5a92bc24848b1b5a7137c26e00d0dbf6",
    },
    Asset {
        name: "model.safetensors",
        size: 328_499_560,
        sha256: "ec9c5b6a08b60d7a01b465c7ed44b496244dfd1bed3274c95875a74cea89ab35",
    },
];

pub async fn ensure_minilm(directory: &Path, offline: bool) -> io::Result<()> {
    ensure_model(directory, offline, REPOSITORY, REVISION, ASSETS).await
}

pub async fn ensure_nli(directory: &Path, offline: bool) -> io::Result<()> {
    ensure_model(directory, offline, NLI_REPOSITORY, NLI_REVISION, NLI_ASSETS).await
}

async fn ensure_model(
    directory: &Path,
    offline: bool,
    repository: &str,
    revision: &str,
    assets: &[Asset<'_>],
) -> io::Result<()> {
    let client = reqwest::Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(network_error)?;
    let base = format!("https://huggingface.co/{repository}/resolve/{revision}");
    ensure_assets(directory, &base, assets, &client, offline).await
}

fn verified(path: &Path, asset: &Asset<'_>) -> io::Result<bool> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if file.metadata()?.len() != asset.size {
        return Ok(false);
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65_536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()) == asset.sha256)
}

async fn ensure_assets(
    directory: &Path,
    base: &str,
    assets: &[Asset<'_>],
    client: &reqwest::Client,
    offline: bool,
) -> io::Result<()> {
    for asset in assets {
        let destination = directory.join(asset.name);
        if verified(&destination, asset)? {
            continue;
        }
        if offline {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "A required model is missing or damaged. Run without --no-download, supply local --model-dir/--nli-model-dir assets, or use --no-nli to disable the answer checker.",
            ));
        }
        private_directory(directory)?;
        eprintln!(
            "Installing local model: {} ({:.1} MB)",
            asset.name,
            asset.size as f64 / 1_000_000.0
        );
        let mut response = client
            .get(format!("{base}/{}", asset.name))
            .send()
            .await
            .map_err(network_error)?
            .error_for_status()
            .map_err(network_error)?;
        if response
            .content_length()
            .is_some_and(|size| size != asset.size)
        {
            return Err(invalid(
                "Model download has an unexpected size; rerun setup to retry.",
            ));
        }
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        let mut hasher = Sha256::new();
        let mut received = 0_u64;
        let mut progress = 0_u64;
        while let Some(chunk) = response.chunk().await.map_err(network_error)? {
            received += chunk.len() as u64;
            if received > asset.size {
                return Err(invalid("Model download exceeded its expected size."));
            }
            hasher.update(&chunk);
            temporary.write_all(&chunk)?;
            if received / 10_000_000 > progress {
                progress = received / 10_000_000;
                eprintln!("Model download: {}%", received * 100 / asset.size);
            }
        }
        if received != asset.size || hex::encode(hasher.finalize()) != asset.sha256 {
            return Err(invalid(
                "Model checksum verification failed; rerun setup to retry.",
            ));
        }
        temporary.as_file().sync_all()?;
        temporary
            .persist(destination)
            .map_err(|error| error.error)?;
    }
    Ok(())
}

fn network_error(error: reqwest::Error) -> io::Error {
    io::Error::other(format!(
        "Could not download model: {}. Rerun to retry; completed files are reused.",
        error.without_url()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::get};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn settings_round_trip_and_atomic_replacement_are_private() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config/settings.json");
        let mut settings = Settings {
            server_url: Some("https://example.com/v1".into()),
            web: true,
            ..Settings::default()
        };
        settings.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), settings);
        settings.web = false;
        settings.save(&path).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), settings);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(path.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn invalid_settings_and_credentials_are_rejected_without_echo_or_overwrite() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let good = Settings::default();
        good.save(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let bad = Settings {
            server_url: Some("https://example.com/v1?key=secret".into()),
            ..good
        };
        assert!(bad.save(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        for value in [
            r#"{"api_key":"secret"}"#,
            r#"{"version":99}"#,
            "not-json",
            r#"{"max_context_tokens":0}"#,
        ] {
            fs::write(&path, value).unwrap();
            let error = Settings::load(&path).unwrap_err();
            assert!(!error.to_string().contains("secret"));
            assert_eq!(fs::read_to_string(&path).unwrap(), value);
        }
    }

    async fn serve(body: &'static str) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let count = Arc::new(AtomicUsize::new(0));
        let calls = count.clone();
        let app = Router::new().route(
            "/asset",
            get(move || {
                calls.fetch_add(1, Ordering::SeqCst);
                async move { body }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (base, count, server)
    }

    #[tokio::test]
    async fn download_verifies_reuses_and_repairs_without_accepting_partial_files() {
        let (base, calls, server) = serve("model").await;
        let temp = tempfile::tempdir().unwrap();
        let hash = hex::encode(Sha256::digest(b"model"));
        let assets = [Asset {
            name: "asset",
            size: 5,
            sha256: &hash,
        }];
        let client = reqwest::Client::new();
        ensure_assets(temp.path(), &base, &assets, &client, true)
            .await
            .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        ensure_assets(temp.path(), &base, &assets, &client, false)
            .await
            .unwrap();
        ensure_assets(temp.path(), &base, &assets, &client, true)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        fs::write(temp.path().join("asset"), b"wrong").unwrap();
        ensure_assets(temp.path(), &base, &assets, &client, true)
            .await
            .unwrap_err();
        ensure_assets(temp.path(), &base, &assets, &client, false)
            .await
            .unwrap();
        assert_eq!(fs::read(temp.path().join("asset")).unwrap(), b"model");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[tokio::test]
    async fn failed_download_preserves_prior_file_and_cleans_temporary_files() {
        let (base, _, server) = serve("wrong").await;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("asset");
        fs::write(&path, b"prior").unwrap();
        let hash = hex::encode(Sha256::digest(b"model"));
        for size in [5, 4, 6] {
            let assets = [Asset {
                name: "asset",
                size,
                sha256: &hash,
            }];
            ensure_assets(temp.path(), &base, &assets, &reqwest::Client::new(), false)
                .await
                .unwrap_err();
            assert_eq!(fs::read(&path).unwrap(), b"prior");
            assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
        }
        server.abort();
    }

    #[tokio::test]
    async fn connection_failure_never_saves_settings_or_echoes_server_body() {
        let app =
            Router::new().fallback(|| async { (axum::http::StatusCode::UNAUTHORIZED, "secret") });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let before = Settings {
            server_url: Some("http://localhost:8080".into()),
            ..Settings::default()
        };
        before.save(&path).unwrap();
        let provider =
            spine_runtime::LlamaCppProvider::new(spine_runtime::LlamaCppConfig::new(url)).unwrap();
        let error = validate_connection(&provider).await.unwrap_err();
        assert!(!error.to_string().contains("secret"));
        assert_eq!(Settings::load(&path).unwrap(), before);
        server.abort();
    }

    #[tokio::test]
    async fn public_health_does_not_mask_rejected_api_credentials() {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/v1/models",
                get(|| async { axum::http::StatusCode::UNAUTHORIZED }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // Explicit model/context must not bypass the credential check either.
        let mut config = spine_runtime::LlamaCppConfig::new(url);
        config.model = Some("test".into());
        config.max_context_tokens = Some(16384);
        let provider = spine_runtime::LlamaCppProvider::new(config).unwrap();
        assert!(validate_connection(&provider).await.is_err());
        server.abort();
    }

    #[test]
    fn command_line_preferences_override_saved_defaults() {
        use clap::CommandFactory;
        let matches = crate::Cli::command()
            .try_get_matches_from(["spine", "chat", "--web", "--web-port", "9000"])
            .unwrap();
        let chat = matches.subcommand_matches("chat").unwrap();
        assert!(choose(chat, "web_server", true, false));
        assert_eq!(choose(chat, "web_port", 9000, 8088), 9000);
        let matches = crate::Cli::command()
            .try_get_matches_from(["spine", "chat"])
            .unwrap();
        let chat = matches.subcommand_matches("chat").unwrap();
        assert!(choose(chat, "web_server", false, true));
        assert_eq!(choose(chat, "web_port", 8088, 9000), 9000);
    }
}
