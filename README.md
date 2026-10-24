# Spine

Spine is a self-hosted personal AI partner. Its memory lives in an encrypted
local **heart**, it can use tools, and it works in either a terminal or a
browser. Spine itself is one executable: there is no database to manage and no
Python, Node.js, npm, or separate web server to install.

## First-time setup

Spine v0.3.1 includes automatic memory-model setup and remembers successful
launch settings. Download the package for your platform, open Spine, connect your
model service once, and choose a heart passphrase.

Spine connects to an existing OpenAI-compatible chat model service. It installs
its small memory model, MiniLM, automatically when needed. MiniLM is not the chat
model. The optional NLI answer checker can be configured later.

### 1. Install Spine

Download an archive and its checksum from the
[GitHub Releases page](https://github.com/Jaredcastorena/Rust-spine/releases).
No Rust, Git, Python, or Node.js is needed to run a prebuilt archive.
The Windows executable includes its C runtime, so no separate Visual C++
Redistributable installer is needed.

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

### 2. Open Spine

From the directory you want Spine to work in, run:

```bash
spine
```

For an extracted archive use `./spine` on Linux/macOS, or `.\spine.exe` in
PowerShell. `spine chat` does the same thing. On the first launch there is one
connection question:

```text
Model server URL [http://127.0.0.1:8080]:
```

Press Enter for that address, or paste your server URL including its port.
Spine checks the connection (with a 20-second deadline), discovers the model and
context size when advertised, then downloads about **91 MB** of MiniLM assets
from Hugging Face if they are not already available. Downloads use a pinned
revision and SHA-256 checks, show progress, and reuse completed files on retry.
No model download prompt or manual file placement is needed.

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

Model weights stay outside Spine. Historical size measurements are available below; they predate v0.3.0 and its
updated dependencies. MiniLM is about 91 MB; your chat model is additional. See
[Honest size numbers](#honest-size-numbers) for the full measurements.

### Next launches and saved settings

Run `spine` again. A successful normal chat startup remembers the connection,
explicit provider model/context/reasoning overrides, heart and MiniLM locations,
workspace, terminal/browser preference and browser port, NLI selection, and
`--skip-onboarding`. Subsequent launches reuse these settings and prompt only
for the heart passphrase. File and shell tools keep using the saved workspace,
even if you open Spine from another directory; use `--workspace` to change it.

Command-line options and their environment variables override saved settings.
Changing the server URL clears inherited model/context/reasoning overrides so
settings from one provider do not silently carry over to another. Model and
context values discovered from the server are checked anew on each launch.
Settings are written only after connection, model loading, heart unlock, and
optional web startup succeed. A failed setup keeps the last working settings.

Useful changes:

```bash
spine chat --setup                       # Ask for a different connection URL
spine chat --web                         # Remember browser mode
spine chat --terminal                    # Switch back to terminal mode
spine chat --workspace "/path/to/project"
spine chat --skip-onboarding             # Go straight to chat
spine chat --no-download                 # Never download MiniLM
spine chat --no-settings                 # Ignore saved settings and do not save
```

Settings are a private-permission JSON file at:

| System | Default file |
| --- | --- |
| Linux | `$XDG_CONFIG_HOME/spine/settings.json`, or `~/.config/spine/settings.json` |
| macOS | `~/Library/Application Support/Spine/settings.json` |
| Windows | `%LOCALAPPDATA%\Spine\settings.json` |

Use `--config PATH` or `SPINE_CONFIG` for a separate profile. The file contains
nonsecret preferences only: API keys, passphrases, recovery phrases, remote web
binding, and elevated tool permissions are never saved. For a service requiring
credentials, supply `SPINE_LLM_API_KEY` through your normal secret manager or
environment on each launch. Incognito, managed llama.cpp, and `--no-settings`
runs neither read nor write this file. Other chat tuning flags remain per-run.

MiniLM is stored under the platform's Spine app-data directory in
`models/all-MiniLM-L6-v2/<revision>/`, separate from the executable and heart.
Explicit `--model-dir`/`SPINE_MINILM_DIR` and existing Hugging Face cache snapshots
are reused. An invalid explicit path produces an error instead of silently
replacing your model. Missing or damaged files in Spine's managed model directory
are downloaded again unless `--no-download` is set. For an offline machine,
copy the model files in advance using the instructions below.

### Connect a model server

The chat model service must already be running. Spine does not install or start
one unless you explicitly select the managed llama.cpp mode. A base URL ending
in `/v1` or a full `/v1/chat/completions` URL is accepted. Setup checks service
readiness and rejects authentication failures from `/v1/models`; it does not
send a billable test conversation or guarantee every provider's tool support.

For unattended use, supply `--server-url`/`SPINE_LLM_URL` or an already validated
settings file. Setup never consumes redirected chat messages as answers.

For an optional local setup, Spine works with
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

### Offline or custom MiniLM installation

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

4. Supply the folder's absolute path with `--model-dir` when starting chat.
   Despite the option name, this is the MiniLM folder, not the chat-model path.

If Hugging Face tooling already downloaded this model into its standard cache,
Spine finds it automatically and you can omit `--model-dir`.

## Docker installation

This route requires [Docker](https://docs.docker.com/get-started/get-docker/)
and, if you are cloning the repository, [Git](https://git-scm.com/install/).
It does not require Rust on the host. Use a MiniLM folder from [offline installation](#offline-or-custom-minilm-installation)
and a reachable [model server](#connect-a-model-server). Model weights and credentials stay
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
   [offline installation](#offline-or-custom-minilm-installation).

   The container runs as UID and GID `10001`. On Linux, that user must be able
   to traverse the mounted folder and read all three model files. If the folder
   was created with restrictive permissions, fix only that folder and those
   files before starting the container:

   ```bash
   chmod a+rx "/absolute/path/to/all-MiniLM-L6-v2"
   chmod a+r "/absolute/path/to/all-MiniLM-L6-v2"/{config.json,tokenizer.json,model.safetensors}
   ```

   Linux, with the model server described above on the same machine:

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
containing real files. To use a normal Hugging Face cache instead, omit
`-e SPINE_MINILM_DIR=/models/minilm` and replace the MiniLM mount with:

```bash
--mount "type=bind,src=/absolute/path/to/models--sentence-transformers--all-MiniLM-L6-v2,dst=/models/models--sentence-transformers--all-MiniLM-L6-v2,readonly"
```

Mount the complete repository directory containing both `blobs/` and
`snapshots/`. Mounting only a `snapshots/<hash>` directory breaks its relative
symlinks into `blobs/`. The image's `HF_HUB_CACHE=/models` setting discovers the
mounted snapshot automatically.

The heart stays under `/data`, while relative file paths and shell commands
start at `/workspace`. This separation is organizational, not a security
boundary: an absolute path can still reach `/data`. To work on host files,
bind-mount one explicitly selected directory at `/workspace` and ensure UID
10001 can write it. Do not mount a home directory or a broader filesystem tree.

The Docker Desktop example uses its built-in `host.docker.internal` name to
reach port 8080 on the host. The Linux example uses host networking so the
loopback-only server described above remains reachable without exposing it to the
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

These historical measurements predate v0.3.0. They came from the verified Linux
amd64 reference build on
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

```bash
spine
```

Spine reuses your successful setup and prompts for the passphrase without echoing
it. A wrong passphrase never overwrites the existing heart or saved preferences.
Use `spine chat --setup` to enter a new connection, or `--workspace PATH` to
change the directory used by tools. The heart lives in the normal app-data
folder unless you selected another location.

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
| `/circuit` | Shows LLM, Thymos, and DCMDb circuit-breaker state. |
| `/reset llm`, `/reset thymos`, `/reset dcmdb` | Resets one recovered subsystem circuit. |
| `reset`, `reset cb`, `reset circuit breaker` | Resets all subsystem circuits. |
| `/quit` | Exits at a safe boundary. |

Terminal lines that arrive in one paste burst are submitted together as a
single multi-line prompt or guidance message. Individually typed commands and
guidance keep their normal one-line behavior.

Graceful-stop checkpoints are stored in the encrypted heart. Reopening the
same agent and thread restores the newest unconsumed checkpoint for `/resume`;
once a resume begins, that exact checkpoint is durably marked consumed so it
cannot be replayed after another restart.

Transient recall, Thymos, or provider failures degrade the affected operation
without discarding the encrypted canonical event log. If a prior write left the
cognitive projection stale, the next chat startup verifies it and applies only
the missing canonical suffix before accepting a turn.

### Browser interface

Add `--web` to the normal chat command:

```bash
spine chat --web
```

Spine reuses the saved connection and memory model. Browser mode is remembered;
use `spine chat --terminal` to switch back.

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
exit. `--test-mode` is an alias for the same behavior. MiniLM is still required and is automatically downloaded if needed. Incognito
mode can reuse/install model files but does not read or change saved launch settings.
Supply `--server-url` for a non-default service.

## Settings and optional features

Most people can leave these alone. The table gives the exact environment and
command-line names for each setting.

| Environment variable | Command-line form | What it changes |
| --- | --- | --- |
| `SPINE_CONFIG` | `--config` | Separate launch-settings profile. |
| `SPINE_HEART_PATH` | positional `HEART` argument | Location of the encrypted heart. |
| `SPINE_HEART_PASSPHRASE` | `--passphrase` | Heart passphrase. Let Spine prompt when practical. |
| `SPINE_MINILM_DIR` | `--model-dir` | Location of the required MiniLM snapshot. |
| `SPINE_NLI_DIR` | `--nli-model-dir` | Location of the optional NLI snapshot. |
| `SPINE_LLM_URL` | `--server-url` | Base URL or full chat-completions endpoint; skips endpoint/port prompts. |
| `SPINE_LLM_MODEL` | `--server-model` | Model name, when the server cannot report it. |
| `SPINE_LLM_API_KEY` | `--api-key` (hidden; environment preferred) | Model-server credential, when one is required. |
| `SPINE_REASONING_EFFORT` | `--reasoning-effort` | Optional provider-specific reasoning effort; omitted by default for broad compatibility. |
| `SPINE_MAX_CONTEXT_TOKENS` | `--max-context-tokens` | Context limit, when the server cannot report it. |
| — | `--llamacpp` | Explicitly enable managed local llama.cpp; requires its model and executable options. |
| `SPINE_LLAMA_SERVER` | `--llama-server-bin` | Local executable; requires `--llamacpp`. |
| `SPINE_LLAMA_MODEL` | `--llama-model` | Local GGUF model; requires `--llamacpp`. |

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

Omit `SPINE_MINILM_DIR` to reuse saved/cached MiniLM or install it automatically.

At startup Spine checks `/props` and `/v1/models`. When the server advertises
its active model ID and runtime context size, Spine adopts both automatically.
Use `SPINE_LLM_MODEL` or `SPINE_MAX_CONTEXT_TOKENS` only when the server omits
that information or you intentionally want an override.

On a multi-model endpoint, set `SPINE_LLM_MODEL` explicitly. Otherwise Spine
uses the model advertised by `/v1/models` (normally its first entry) before
falling back to `/props`.

Spine omits `reasoning_effort` unless `SPINE_REASONING_EFFORT` or
`--reasoning-effort` is set, because that optional field is not accepted by
every OpenAI-compatible provider.

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

Only with `--llamacpp`, Spine can start and stop a local `llama-server` with the
chat session. This opt-in skips the endpoint prompts and defaults to port 8080;
use `--server-url` to override it:

```bash
spine chat --llamacpp --no-nli --model-dir "/absolute/path/to/all-MiniLM-L6-v2" --llama-server-bin "/absolute/path/to/llama-server" --llama-model "/absolute/path/to/model.gguf" --max-context-tokens 16384
```

Server output is written beside the heart. `--max-tool-rounds` is optional;
tool rounds are unlimited by default. Omit `--model-dir` when Spine finds
MiniLM automatically. Managed mode enables Jinja function calling without
enabling llama.cpp's separate built-in host tools.

After each user message is committed, the host uses its Thymos trajectory,
recalled-memory tensions, and learned risk to widen automatic recall, cool the
provider temperature, cap external actions, and raise the answer-grounding
coverage requirement. Surprise also scales reflection learning through a
per-observation multiplier stored in the signed event. Risk expands exact
evidence from absorbed DCMDb nodes to a bounded depth of one to three levels.
A positive `--max-tool-rounds` ceiling can expand under risk; omitting it remains
unlimited. Checkpoints retain their explicitly saved effective policy, including
unlimited rounds. Legacy checkpoints with no saved policy inherit the current
harness ceiling, counting rounds already completed. Triangle-context rehydration
keeps its fixed bounded topology.

Risk training uses Python's normalized affect and all six retrieval statistics:
top score, score margin, mean confidence, tension fraction, hierarchy depth, and
corpus size. Existing released hearts upgrade atomically on open. Their learned
raw-affect and count/empty-evidence contributions remain in separate compatibility
segments, while the new oracle coordinates start with zero weights. Unknown or
inconsistent layouts are rejected without resetting learned state.

## Build from source

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

The repository pins Rust 1.99.1, and rustup selects or downloads that version
automatically. Cargo normally installs executables in `~/.cargo/bin` on Linux
and macOS or `%USERPROFILE%\.cargo\bin` on Windows. If `spine` is not found,
open a new terminal and make sure that directory is on `PATH`.

Rust is needed only when building from source.


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

Rust 1.99.1 is needed only for source builds. The pinned toolchain includes
`rustfmt` and `clippy` through rustup.

```bash
cargo build --release --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt -p spine-cli -p spine-heart -p spine-models -p spine-runtime -- --check
```

The resulting executable is `target/release/spine`.

The harness accepts native OpenAI tool calls and the reference text forms used
by existing Spine integrations: canonical, legacy, unfenced, truncated,
bare-JSON, and Laguna calls. Multi-step plans advance only after tool evidence;
a response that merely promises more work is not treated as complete.

The file reader returns bounded pages with Unicode character offsets and a
`next_offset` cursor, so the harness can continue reading large files without
losing text at its output limit.

The Candle ZIP 9 compatibility backport and its source-verification procedure
are documented in [vendor/README.md](vendor/README.md).

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
