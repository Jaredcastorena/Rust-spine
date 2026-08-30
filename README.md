# Spine

Spine is a personal AI partner that you run yourself. It remembers your work in
one encrypted file, can use tools, and works in either a terminal or a browser.
The app itself is one executable: there is no database to manage and no Python,
Node.js, npm, or separate web server to install.

## Quick start

If `spine` is already installed, start a first conversation with:

```bash
spine chat --no-nli
```

That one command is enough when:

- an OpenAI-compatible model server is running at `http://127.0.0.1:8080`; and
- MiniLM is already in the normal Hugging Face cache.

If you do not have MiniLM yet, download `config.json`, `tokenizer.json`, and
`model.safetensors` from the
[official MiniLM model repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
into one folder. Point Spine to that folder as part of the same command:

```bash
spine chat --no-nli \
  --model-dir /absolute/path/to/all-MiniLM-L6-v2/snapshot
```

Spine will ask for a passphrase, create an encrypted **heart**, show a recovery
phrase to save offline, and begin a short 5–10 question introduction. The
answers help it learn how you like to work; they are soft preferences, not a
hard-coded personality. You can type `/skip` and start chatting normally at any
time.

Spine also asks the model server which model and context size it is actually
running, so most people do not need to enter either value by hand. If your
server is not at the default address, see
[Use a model server at another address](#use-a-model-server-at-another-address).

### The four terms worth knowing

| Term | Plain-English meaning |
| --- | --- |
| **Heart** | Your encrypted local memory file. |
| **Model server** | The program that runs the chat model. Spine works with OpenAI-compatible servers such as llama.cpp. |
| **MiniLM** | A small local model Spine uses to find relevant memories. The measured snapshot is 91.6 MB. |
| **NLI** | An optional 331.1 MB model that checks how well answers are supported. `--no-nli` leaves it off. |

The heart stays encrypted on your machine. Conversation text is sent to the
model server you choose; use a local server if you want that traffic to remain
local too.

Model weights are deliberately kept outside the app. The verified Linux amd64
executable is **13.3 MB**. A minimal container setup is about **113.7 MB** for
Spine plus MiniLM; your chat model is additional and its size depends on the
model you choose. Full measurements are listed under
[Honest size numbers](#honest-size-numbers).

## Install

Choose whichever route fits your machine. Each route takes no more than two
steps.

### Prebuilt executable

1. Download the release archive for your platform and verify it with the
   adjacent `.sha256` file.
2. Put `spine` (or `spine.exe`) somewhere on `PATH`, then use the quick-start
   command above.

The terminal app and browser interface are both compiled into that one file.

### Build from this checkout

1. Install the pinned Rust toolchain through [rustup](https://rustup.rs/) if it
   is not already installed.
2. Run:

   ```bash
   cargo install --locked --path crates/spine-cli
   ```

Rust is needed only for this source-build route.

### Container (two commands)

This path uses an existing Docker installation; it does not install Docker or
another container service. A running OpenAI-compatible endpoint and a local
MiniLM snapshot are still runtime prerequisites. They stay outside the image so
model weights and credentials are not silently copied into an image layer.

1. Build the image from this checkout:

   ```bash
   docker build -t spine:local .
   ```

2. Run it, replacing the MiniLM source path with the absolute Hugging Face
   model-repository cache directory on the host (the directory containing both
   `blobs/` and `snapshots/`):

   ```bash
   docker run --rm -it --name spine \
     --add-host=host.docker.internal:host-gateway \
     --mount source=spine-data,target=/data \
     --mount type=bind,src=/absolute/path/to/models--sentence-transformers--all-MiniLM-L6-v2,dst=/models/models--sentence-transformers--all-MiniLM-L6-v2,readonly \
     spine:local
   ```

That is the complete container setup. If those defaults fit your machine, you
can stop here. The notes below explain the command for people who need to change
its storage, networking, or browser settings.

#### Container details

The named `spine-data` volume retains the encrypted heart while the image and
container remain disposable. Spine prompts for the heart passphrase, and the
default command starts terminal chat without the optional NLI model. The image
auto-discovers the mounted snapshot through `HF_HUB_CACHE`; mounting only a
typical Hugging Face `snapshots/<hash>` directory would break its relative
symlinks into `blobs/`. A standalone directory containing real files instead
of symlinks can be mounted at `/models/minilm` and selected with
`-e SPINE_MINILM_DIR=/models/minilm`.

The heart stays under `/data`, while model-requested file and command tools are
rooted separately at `/workspace`; this prevents ordinary workspace tools from
seeing or modifying heart files. To work on host files, additionally bind-mount
one explicitly selected directory at `/workspace` and ensure UID 10001 can
write it. Do not mount a home directory or a broader filesystem tree.

The image expects the provider at `http://host.docker.internal:8080`; override
`SPINE_LLM_URL` (and pass `SPINE_LLM_API_KEY` with `-e` when needed) for another
endpoint. On native Linux, a host provider must listen on an address reachable
from Docker's bridge, not only the host loopback interface.

Automatic model and context discovery works the same way in the container as
it does for the native executable. The details and manual overrides are under
[Use a model server at another address](#use-a-model-server-at-another-address).

Some Docker Desktop installations do not route bridge traffic into the host's
Tailscale interface. For a trusted host endpoint, replace
`--add-host=host.docker.internal:host-gateway` with `--network host` when that
Docker installation has host networking enabled.

Docker reports `healthy` only while the actual Spine process is PID 1. From a
second terminal, check it with:

```bash
docker inspect --format '{{.State.Health.Status}}' spine
```

For the browser interface, publish only to local loopback and replace the
image's default command:

```bash
docker run --rm -it --name spine-web \
  --add-host=host.docker.internal:host-gateway \
  --mount source=spine-data,target=/data \
  --mount type=bind,src=/absolute/path/to/models--sentence-transformers--all-MiniLM-L6-v2,dst=/models/models--sentence-transformers--all-MiniLM-L6-v2,readonly \
  -p 127.0.0.1:8088:8088 spine:local \
  chat --no-nli --web --web-host 0.0.0.0 --allow-remote-web
```

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

```bash
spine chat --no-nli
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
spine chat --no-nli --web
```

Spine prints the complete private URL to open. The interface listens only on
`127.0.0.1:8088` by default and uses a new browser token for every process. The
terminal and browser share the same conversation, memories, tools, tasks, and
stop controls.

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
exit. `--test-mode` is an alias for the same behavior.

## Settings and optional features

Most people can leave these alone. Every setting below can be supplied through
the environment or its matching command-line option.

| Environment variable | What it changes |
| --- | --- |
| `SPINE_HEART_PATH` | Location of the encrypted heart. |
| `SPINE_HEART_PASSPHRASE` | Heart passphrase. Let Spine prompt when practical. |
| `SPINE_MINILM_DIR` | Location of the required MiniLM snapshot. |
| `SPINE_NLI_DIR` | Location of the optional NLI snapshot. |
| `SPINE_LLM_URL` | Address of the OpenAI-compatible model server. |
| `SPINE_LLM_MODEL` | Model name, when the server cannot report it. |
| `SPINE_LLM_API_KEY` | Model-server credential, when one is required. |
| `SPINE_MAX_CONTEXT_TOKENS` | Context limit, when the server cannot report it. |
| `SPINE_LLAMA_SERVER` | Local `llama-server` executable for Spine to manage. |
| `SPINE_LLAMA_MODEL` | Local GGUF model for that managed server. |

Never put a literal passphrase, recovery phrase, or API key in a command you
might save or share.

### Use a model server at another address

Set the URL and start chat normally. If a credential is required, load it into
`PROVIDER_API_KEY` with your shell's secret prompt or a secret manager first:

```bash
SPINE_LLM_URL=https://provider.example/v1 \
SPINE_LLM_API_KEY="$PROVIDER_API_KEY" \
  spine chat --no-nli
```

At startup Spine checks `/props` and `/v1/models`. When the server advertises
its active model ID and runtime context size, Spine adopts both automatically.
Use `SPINE_LLM_MODEL` or `SPINE_MAX_CONTEXT_TOKENS` only when the server omits
that information or you intentionally want an override.

The runtime value matters more than the model's theoretical maximum. A model
may support 262,144 tokens but report only 4,096 when the server allocated a 4K
context. Onboarding fits at 4K, but 16K or more is a practical starting point
for normal tool use.

### Enable the optional answer checker

Download the files from the
[official NLI model repository](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768),
point Spine to that folder, and leave off `--no-nli`:

```bash
spine chat \
  --nli-model-dir /path/to/nli-MiniLM2-L6-H768/snapshot
```

### Let Spine manage llama.cpp

Spine can start and stop a local `llama-server` with the chat session:

```bash
spine chat --no-nli \
  --llama-server-bin /path/to/llama-server \
  --llama-model /path/to/model.gguf \
  --max-context-tokens 115968
```

Server output is written beside the heart. `--max-tool-rounds` is optional;
tool rounds are unlimited by default.

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
- `default.action-audit.jsonl`: host action audit log.
- `default.llama-server.log`: managed llama.cpp output, when `--llama-model` is used.

These paths are ignored by Git. Never commit heart files, recovery phrases, passphrases, or provider keys.

See [PROJECT_STATE.md](PROJECT_STATE.md) for the accepted baseline and known integration boundary.

## License and attribution

Spine is available under either the MIT License or Apache License 2.0. See
[LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The generated
[THIRD_PARTY_LICENSES.html](THIRD_PARTY_LICENSES.html) carries the crate-level
license texts and attributions for dependencies incorporated into the binary.
