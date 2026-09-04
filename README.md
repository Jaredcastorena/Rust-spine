# Spine

Spine is a self-hosted personal AI partner. Its memory lives in an encrypted
local **heart**, it can use tools, and it works in either a terminal or a
browser. Spine itself is one executable: there is no database to manage and no
Python, Node.js, npm, or separate web server to install.

## First-time setup

Spine does not include a chat model. A working setup has three required pieces:

| Piece | What it does |
| --- | --- |
| **Spine** | Runs the partner, tools, terminal interface, and browser interface. |
| **MiniLM** | Finds relevant memories. It is small and is **not** the chat model. |
| **Model server and chat model** | Generate replies through an OpenAI-compatible API. |

An optional fourth model, **NLI**, checks how well answers are supported. Leave
it off during the first setup with `--no-nli`; chat and memory still work.

Follow these steps in order for a native installation. If you prefer a
container, complete steps 2 and 3 first, then use the
[Docker installation](#docker-installation).

### 1. Install Spine

#### Build from source

This route works now and requires Git, Rust, and your platform's normal native
build tools.

1. Install [Git](https://git-scm.com/install/) and
   [Rust with rustup](https://rust-lang.org/tools/install/).

   On Linux, macOS, or WSL, the official rustup command is:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

   On Windows, run the installer from the Rust page and accept the offered
   [Visual Studio C++ prerequisites](https://rust-lang.github.io/rustup/installation/windows-msvc.html).
   Debian/Ubuntu users can install the native compiler with
   `sudo apt install build-essential`; macOS users can run
   `xcode-select --install`.

2. Open a new terminal and verify the tools are available:

   ```bash
   git --version
   rustc --version
   cargo --version
   ```

3. Download this repository and enter its directory:

   ```bash
   git clone https://github.com/Jaredcastorena/Rust-spine.git
   cd Rust-spine
   ```

4. Build and install the `spine` executable, then verify it:

   ```bash
   cargo install --locked --path crates/spine-cli
   spine --version
   ```

The repository pins Rust 1.98.0, and rustup selects or downloads that version
automatically. Cargo normally installs executables in `~/.cargo/bin` on Linux
and macOS or `%USERPROFILE%\.cargo\bin` on Windows. If `spine` is not found,
open a new terminal and make sure that directory is on `PATH`.

Rust is needed only when building from source.

#### Prebuilt archive (when available)

Tagged release archives and their checksum files will appear on the
[GitHub Releases page](https://github.com/Jaredcastorena/Rust-spine/releases).
If that page has no archive for your system, use the source-build or Docker
route.

| System | Archive name contains |
| --- | --- |
| Linux, Intel/AMD 64-bit | `x86_64-unknown-linux-gnu` |
| Linux, ARM 64-bit | `aarch64-unknown-linux-gnu` |
| macOS, Intel | `x86_64-apple-darwin` |
| macOS, Apple silicon | `aarch64-apple-darwin` |
| Windows, Intel/AMD 64-bit | `x86_64-pc-windows-msvc` |

The Linux archives target glibc-based distributions, not Alpine/musl.

1. Download the archive for your system and the file beside it ending in
   `.sha256`, and place just those two files in a new folder.
2. Verify the download. A successful Linux or macOS check says `OK`; the
   PowerShell check returns `True`.

   Linux:

   ```bash
   sha256sum --check spine-*.tar.gz.sha256
   ```

   macOS:

   ```bash
   shasum -a 256 --check spine-*.tar.gz.sha256
   ```

   Windows PowerShell:

   ```powershell
   $archive = (Get-ChildItem .\spine-*.zip).FullName
   $expected = ((Get-Content ($archive + '.sha256') -Raw) -split '\s+')[0]
   (Get-FileHash $archive -Algorithm SHA256).Hash -ieq $expected
   ```

   Stop if the checksum does not match.

3. Extract the archive and check the executable:

   Linux or macOS:

   ```bash
   tar -xzf spine-*.tar.gz
   cd spine-v*/
   ./spine --version
   ```

   Windows PowerShell:

   ```powershell
   Expand-Archive .\spine-*.zip -DestinationPath .\spine
   .\spine\spine.exe --version
   ```

4. Run the executable from that directory or move it into a directory on
   `PATH`. The terminal and browser interfaces are both inside the same file.

### 2. Download MiniLM

1. Outside the `Rust-spine` checkout, create a folder named
   `all-MiniLM-L6-v2`. Keep model weights out of the repository.
2. Download these three files into that folder without renaming them:

   - [`config.json`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json?download=true)
   - [`tokenizer.json`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json?download=true)
   - [`model.safetensors`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors?download=true)

   These links pin one model revision so all three files stay consistent.

3. Confirm the files are directly inside the folder:

   ```text
   all-MiniLM-L6-v2/
   ├── config.json
   ├── model.safetensors
   └── tokenizer.json
   ```

4. Copy the folder's absolute path. You will use it as `--model-dir` in step 4.
   Despite the option name, this is the MiniLM folder, not the chat-model path.

If Hugging Face tooling already downloaded this model into its standard cache,
Spine finds it automatically and you can omit `--model-dir`.

### 3. Start a model server

If an OpenAI-compatible model server is already running at
`http://127.0.0.1:8080`, continue to step 4.

For a local setup, Spine works with
[`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
from [llama.cpp](https://github.com/ggml-org/llama.cpp#quick-start). Install
llama.cpp, choose a GGUF chat model that fits your hardware, and start it in a
separate terminal:

```bash
llama-server -m "/absolute/path/to/model.gguf" --host 127.0.0.1 --port 8080 -c 16384 --jinja
```

On Windows, use `llama-server.exe` and replace the example with a Windows path.

Leave that terminal running. When the model finishes loading, verify the server
from another terminal:

Linux or macOS:

```bash
curl --fail http://127.0.0.1:8080/health
```

Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

llama.cpp is the tested server path. Other providers must accept Spine's Chat
Completions request fields at `/v1/chat/completions` and expose at least one
usable startup endpoint at `/health`, `/props`, or `/v1/models`.

Spine asks the server which model and context size it is actually running, so
most people do not need to enter either value by hand. For another address, see
[Use a model server at another address](#use-a-model-server-at-another-address).

### 4. Start Spine

Launch Spine from the directory you want its relative file and shell tools to
use as their workspace. Replace the example MiniLM path with the absolute path
from step 2:

```bash
spine chat --no-nli --model-dir "/absolute/path/to/all-MiniLM-L6-v2"
```

These examples assume `spine` is on `PATH`. For an extracted prebuilt archive,
use `./spine` on Linux or macOS, or the path to `spine.exe` on Windows.
Windows paths such as `C:\Models\all-MiniLM-L6-v2` work when placed in quotes.

If Spine found MiniLM in the Hugging Face cache, the shorter command is enough:

```bash
spine chat --no-nli
```

Spine asks for a passphrase, creates an encrypted heart, shows a recovery phrase
to save offline, and begins a short 5–10 question introduction. The answers are
soft working preferences, not a hard-coded personality. Type `/skip` at any time
to begin chatting normally.

The heart and its large-blob sidecars are encrypted on your machine. Requests
to the model server can include conversation text, selected memories, and tool
results; use a local server if you want those requests to remain local. When
Spine runs a shell command, the command and its captured output are written to
a separate, unencrypted `*.action-audit.jsonl` file, so treat that file as
sensitive.

Model weights stay outside Spine. The verified Linux amd64 executable is
**13.3 MB**. A minimal container setup is about **113.7 MB** for Spine plus
MiniLM; your chat model is additional. See
[Honest size numbers](#honest-size-numbers) for the full measurements.

## Docker installation

This route requires [Docker](https://docs.docker.com/get-started/get-docker/)
and, if you are cloning the repository, [Git](https://git-scm.com/install/).
It does not require Rust on the host. It still needs the MiniLM folder from step
2 and a reachable model server from step 3. Model weights and credentials stay
outside the image so they are not silently copied into an image layer.

The multiline commands below use Bash. In PowerShell, enter each command on one
line or replace each trailing `\` with PowerShell's backtick continuation.

1. If you do not already have this checkout, download it:

   ```bash
   git clone https://github.com/Jaredcastorena/Rust-spine.git
   cd Rust-spine
   ```

2. Build the image:

   ```bash
   docker build -t spine:local .
   ```

3. Run the command for your platform. Replace the MiniLM source path with the
   absolute path to the folder that directly contains the three files from
   step 2.

   Linux, with the model server from step 3 on the same machine:

   ```bash
   docker run --rm -it --name spine \
     --network host \
     --mount source=spine-data,target=/data \
     --mount "type=bind,src=/absolute/path/to/all-MiniLM-L6-v2,dst=/models/minilm,readonly" \
     -e SPINE_MINILM_DIR=/models/minilm \
     -e SPINE_LLM_URL=http://127.0.0.1:8080 \
     spine:local
   ```

   macOS or Windows with Docker Desktop:

   ```bash
   docker run --rm -it --name spine \
     --mount source=spine-data,target=/data \
     --mount "type=bind,src=/absolute/path/to/all-MiniLM-L6-v2,dst=/models/minilm,readonly" \
     -e SPINE_MINILM_DIR=/models/minilm \
     spine:local
   ```

Spine now prompts for the heart passphrase and starts terminal chat without the
optional NLI model. The notes below explain storage, networking, Hugging Face
cache mounts, and browser mode.

#### Container details

The named `spine-data` volume retains the encrypted heart while the image and
container remain disposable. The example mounts a standalone MiniLM directory
containing real files. If you instead mount a normal Hugging Face cache, mount
the complete `models--sentence-transformers--all-MiniLM-L6-v2` directory that
contains both `blobs/` and `snapshots/`; mounting only a `snapshots/<hash>`
directory breaks its relative symlinks into `blobs/`.

The heart stays under `/data`, while relative file paths and shell commands
start at `/workspace`. This separation is organizational, not a security
boundary: an absolute path can still reach `/data`. To work on host files,
bind-mount one explicitly selected directory at `/workspace` and ensure UID
10001 can write it. Do not mount a home directory or a broader filesystem tree.

The Docker Desktop example uses its built-in `host.docker.internal` name to
reach port 8080 on the host. The Linux example uses host networking so the
loopback-only server from step 3 remains reachable without exposing it to the
LAN. Host networking also lets the container reach other host-loopback
services, so use it only on a trusted machine. Override `SPINE_LLM_URL` (and
pass `SPINE_LLM_API_KEY` with `-e` when needed) for another endpoint.

If you choose bridge networking on native Linux instead, add
`--add-host=host.docker.internal:host-gateway`. The host provider must then
listen on an address reachable from Docker's bridge, not only on `127.0.0.1`.

Automatic model and context discovery works the same way in the container as
it does for the native executable. The details and manual overrides are under
[Use a model server at another address](#use-a-model-server-at-another-address).

Some Docker Desktop installations do not route bridge traffic into the host's
Tailscale interface. For a trusted host endpoint, use `--network host` when that
Docker installation supports it. Also set
`-e SPINE_LLM_URL=http://127.0.0.1:8080`; the default
`host.docker.internal` name is no longer needed in host-network mode.

Docker reports `healthy` only while the actual Spine process is PID 1. From a
second terminal, check it with:

```bash
docker inspect --format '{{.State.Health.Status}}' spine
```

For the browser interface on Docker Desktop, publish only to local loopback and
replace the image's default command:

```bash
docker run --rm -it --name spine-web \
  --mount source=spine-data,target=/data \
  --mount "type=bind,src=/absolute/path/to/all-MiniLM-L6-v2,dst=/models/minilm,readonly" \
  -e SPINE_MINILM_DIR=/models/minilm \
  -p 127.0.0.1:8088:8088 spine:local \
  chat --no-nli --web --web-host 0.0.0.0 --allow-remote-web
```

On Linux with the host-network command above, replace its final `spine:local`
with `spine:local chat --no-nli --web`. The default loopback bind is already
correct, so do not add `--allow-remote-web`.

## Honest size numbers

These measurements came from the verified Linux amd64 reference build on
2026-08-29. Docker reports decimal units here (1 MB = 1,000,000 bytes).

| Item | Measured size | Where it lives |
| --- | ---: | --- |
| Linux amd64 release executable | 13.3 MB | The native `spine` program |
| Docker build context | 1.06 MB | Sent to the builder |
| Runnable Spine image | 22.1 MB | Docker image storage |
| Gzip-compressed `docker save` archive | 9.35 MB | Transfer estimate, not a registry guarantee |
| Required MiniLM snapshot | 91.6 MB | Read-only host mount, outside the image |
| Optional NLI snapshot | 331.1 MB | Read-only host mount, outside the image |
| Rust builder stage | 1.08 GB | Build-only Docker storage |
| Reusable Cargo registry and target caches | 927.9 MB | BuildKit cache |

The minimal running footprint is therefore about 113.7 MB for the image plus
MiniLM, before Docker metadata and writable heart data. GGUF/LLM weights are
additional and entirely model-dependent; they are not copied into the image.
Encrypted hearts start small and grow with conversations, memories, and blobs.

A first Docker build needs about 2.01 GB for the measured builder and Cargo
caches in addition to the final image, so allow at least 2.1 GB free. Docker
retains those caches to make later builds faster. The compressed archive number
was measured with `docker save spine:local | gzip -9`; a registry may compress
layers differently.

Repeated development rebuilds can retain multiple cache generations. After the
full reference verification sequence, Docker reported 5.9 GB of reclaimable
BuildKit cache on the development host. That cache is not part of the runnable
image or user data and can be managed with Docker's normal cache controls.

## Everyday use

### Start or reopen your heart

If MiniLM is in the Hugging Face cache or `SPINE_MINILM_DIR` is set, run:

```bash
spine chat --no-nli
```

Otherwise, repeat its directory explicitly:

```bash
spine chat --no-nli --model-dir "/absolute/path/to/all-MiniLM-L6-v2"
```

Spine prompts for the heart passphrase without echoing it. On a new heart it
asks you to confirm the passphrase and prints a recovery phrase; save that
phrase offline. Later launches reopen the same heart. A wrong passphrase stops
with an error and never overwrites the existing heart.

Unless you choose another location, the heart lives in the normal app-data
folder for your operating system. MiniLM and NLI are found automatically when
they are already in the Hugging Face cache.

### The first conversation

An empty heart begins with a short, model-led introduction. Spine asks one
question at a time, and each answer helps it choose the next question. There is
no fixed script. It stops after 5–10 answers, once it has a useful sense of:

- the tone and amount of detail you prefer;
- how proactive it should be;
- how you want disagreement and uncertainty handled;
- your working boundaries; and
- the kinds of goals that matter to you.

Those questions and answers become the first encrypted memories in DCMDb. The
resulting interaction profile is only a set of soft defaults. It does not grant
extra permissions, and your newest explicit request always wins.

Type `/skip` to learn each other naturally while working. Type `/quit` to pause
the introduction and continue it next time. For scripts, use
`--skip-onboarding`. Spine does not ask for credentials, addresses, legal
identity, or demographic profiling.

### Controls while chatting

| What you enter | What Spine does |
| --- | --- |
| Ordinary text while a tool is running | Queues your guidance for the next completed-tool boundary. |
| `/stop` | Saves a checkpoint and stops gracefully. |
| `/interrupt` | Stops immediately. |
| `/resume` | Continues a saved checkpoint. |
| `/tasks` | Shows work managed by the host. |
| `/quit` | Exits at a safe boundary. |

### Browser interface

Add `--web` to the normal chat command:

```bash
spine chat --no-nli --web --model-dir "/absolute/path/to/all-MiniLM-L6-v2"
```

As with terminal chat, omit `--model-dir` when Spine can find MiniLM
automatically.

Spine prints the complete private URL to open. The interface listens only on
`127.0.0.1:8088` by default and uses a new browser token for every process. The
terminal and browser share the same conversation, memories, tools, tasks, and
stop controls. Do not share the private URL because it contains that token.

Binding the web interface to another machine requires an explicit address and
`--allow-remote-web`. That mode is plain HTTP. Keep it on a trusted private
network or put it behind a TLS reverse proxy; do not expose it directly to the
Internet.

### Temporary test session

To try onboarding or tools without changing your normal heart:

```bash
spine chat --incognito-mode --no-nli
```

No heart path or passphrase is needed. Spine makes a temporary, randomly keyed
heart and deletes it, its encrypted blobs, and its action audit after a normal
exit. `--test-mode` is an alias for the same behavior. MiniLM is still required;
add `--model-dir` when it is not in the Hugging Face cache.

## Settings and optional features

Most people can leave these alone. The table gives the exact environment and
command-line names for each setting.

| Environment variable | Command-line form | What it changes |
| --- | --- | --- |
| `SPINE_HEART_PATH` | positional `HEART` argument | Location of the encrypted heart. |
| `SPINE_HEART_PASSPHRASE` | `--passphrase` | Heart passphrase. Let Spine prompt when practical. |
| `SPINE_MINILM_DIR` | `--model-dir` | Location of the required MiniLM snapshot. |
| `SPINE_NLI_DIR` | `--nli-model-dir` | Location of the optional NLI snapshot. |
| `SPINE_LLM_URL` | `--server-url` | Address of the OpenAI-compatible model server. |
| `SPINE_LLM_MODEL` | `--server-model` | Model name, when the server cannot report it. |
| `SPINE_LLM_API_KEY` | `--api-key` (hidden; environment preferred) | Model-server credential, when one is required. |
| `SPINE_MAX_CONTEXT_TOKENS` | `--max-context-tokens` | Context limit, when the server cannot report it. |
| `SPINE_LLAMA_SERVER` | `--llama-server-bin` | Local `llama-server` executable for Spine to manage. |
| `SPINE_LLAMA_MODEL` | `--llama-model` | Local GGUF model for that managed server. |

Never put a literal passphrase, recovery phrase, or API key in a command you
might save or share.

### Use a model server at another address

Set the URL and start chat normally. If a credential is required, load it into
`PROVIDER_API_KEY` with your shell's secret prompt or a secret manager first:

Linux or macOS:

```bash
SPINE_MINILM_DIR="/absolute/path/to/all-MiniLM-L6-v2" \
SPINE_LLM_URL=https://provider.example/v1 \
SPINE_LLM_API_KEY="$PROVIDER_API_KEY" \
  spine chat --no-nli
```

Windows PowerShell:

```powershell
$env:SPINE_MINILM_DIR = 'C:\absolute\path\to\all-MiniLM-L6-v2'
$env:SPINE_LLM_URL = 'https://provider.example/v1'
$env:SPINE_LLM_API_KEY = $env:PROVIDER_API_KEY
spine chat --no-nli
```

Omit `SPINE_MINILM_DIR` when MiniLM is already in the Hugging Face cache.

At startup Spine checks `/props` and `/v1/models`. When the server advertises
its active model ID and runtime context size, Spine adopts both automatically.
Use `SPINE_LLM_MODEL` or `SPINE_MAX_CONTEXT_TOKENS` only when the server omits
that information or you intentionally want an override.

On a multi-model endpoint, set `SPINE_LLM_MODEL` explicitly. Otherwise Spine
uses the model advertised by `/v1/models` (normally its first entry) before
falling back to `/props`.

The runtime value matters more than the model's theoretical maximum. A model
may support 262,144 tokens but report only 4,096 when the server allocated a 4K
context. Onboarding fits at 4K, but 16K or more is a practical starting point
for normal tool use.

### Enable the optional answer checker

Outside the `Rust-spine` checkout, create a folder named
`nli-MiniLM2-L6-H768`, then download
[`config.json`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/resolve/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d/config.json?download=true),
[`tokenizer.json`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/resolve/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d/tokenizer.json?download=true),
and
[`model.safetensors`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/resolve/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d/model.safetensors?download=true)
directly into it. Start Spine without `--no-nli` and point to that folder:

```bash
spine chat --model-dir "/absolute/path/to/all-MiniLM-L6-v2" --nli-model-dir "/absolute/path/to/nli-MiniLM2-L6-H768"
```

Omit `--model-dir` when the required MiniLM model is already in the Hugging
Face cache.

### Let Spine manage llama.cpp

Spine can start and stop a local `llama-server` with the chat session:

```bash
spine chat --no-nli --model-dir "/absolute/path/to/all-MiniLM-L6-v2" --llama-server-bin "/absolute/path/to/llama-server" --llama-model "/absolute/path/to/model.gguf" --max-context-tokens 16384
```

Server output is written beside the heart. `--max-tool-rounds` is optional;
tool rounds are unlimited by default. Omit `--model-dir` when Spine finds
MiniLM automatically.

**Managed-server security:** this mode currently starts llama.cpp with its
built-in filesystem and shell tools enabled. Those server-side tools are
separate from Spine's action audit and host gates. Keep the server on loopback,
use only a trusted local model, and use the separate-server setup in step 3 if
you do not want that additional tool surface.

## Developer reference

### What is in this repository

- `spine-heart`: encrypted storage, identity, DCMDb memory, Thymos state,
  typed facts, learned risk, snapshots, sync, and context compaction.
- `spine-models`: native Candle loaders for MiniLM and the optional NLI model.
- `spine-runtime`: the model provider, tool harness, plan cursor, live controls,
  checkpoints, and temporary subagents.
- `spine-cli`: terminal and browser chat, memory commands, action tools,
  ingestion, and grounding.
- `fuzz`: source-only libFuzzer targets. Generated corpora and artifacts are
  excluded.

Model weights, hearts, logs, build output, and credentials are deliberately not
part of the repository. This Rust project does not modify or depend on the
legacy Python DCMDb runtime.

### Full runtime checklist

- A running OpenAI-compatible chat-completions server, or `llama-server` plus a
  local GGUF model for Spine to manage. The default server address is
  `http://127.0.0.1:8080`.
- A local `sentence-transformers/all-MiniLM-L6-v2` snapshot containing
  `config.json`, `tokenizer.json`, and `model.safetensors`.
- Optionally, a local `cross-encoder/nli-MiniLM2-L6-H768` snapshot containing
  the same three files.

If no heart path is supplied, Linux uses
`$XDG_DATA_HOME/spine/default.spine` or
`$HOME/.local/share/spine/default.spine`. macOS uses the normal Application
Support location, and Windows uses Local AppData. Model discovery honors
`HF_HOME` and `HF_HUB_CACHE`.

### Build and test

Rust 1.98.0 is needed only for source builds. The pinned toolchain includes
`rustfmt` and `clippy` through rustup.

```bash
cargo build --release --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --all -- --check
```

The resulting executable is `target/release/spine`.

The harness accepts native OpenAI tool calls and the reference text forms used
by existing Spine integrations: canonical, legacy, unfenced, truncated,
bare-JSON, and Laguna calls. Multi-step plans advance only after tool evidence;
a response that merely promises more work is not treated as complete.

The single-binary browser approach and visual direction are inspired by
llama.cpp's MIT-licensed `llama-ui`. Spine's HTML, CSS, JavaScript, and host API
are implemented specifically for this harness.

### Legacy Python memory boundary

This standalone baseline reads the new encrypted Rust heart only. The legacy
Python DCMDb may run as a separate backup, but it is not a transparent fallback
for `heart_recall`. Old memories must be imported or exposed through a
deliberate dual-read bridge before they participate in the native partner's
recall.

## Files and privacy

For a heart at `default.spine`, runtime sidecars use the same base location:

- `default.spine`: encrypted redb heart.
- `default.spine.blobs/`: encrypted large-blob ciphertext.
- `default.action-audit.jsonl`: unencrypted host action audit containing shell
  commands and their captured output.
- `default.llama-server.log`: managed llama.cpp output, when `--llama-model` is used.

These paths are ignored by Git. Treat the action audit and server log as
sensitive even though they are not part of the encrypted heart. Never commit
heart files, recovery phrases, passphrases, provider keys, audits, or logs.

See [PROJECT_STATE.md](PROJECT_STATE.md) for the accepted baseline and known integration boundary.

## License and attribution

Spine is available under either the MIT License or Apache License 2.0. See
[LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The generated
[THIRD_PARTY_LICENSES.html](THIRD_PARTY_LICENSES.html) carries the crate-level
license texts and attributions for dependencies incorporated into the binary.
