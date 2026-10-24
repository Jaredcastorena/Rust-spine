FROM rust:1.99.1-alpine3.24@sha256:b1d5dd5a22a09404d3e798c0cf051b689077360374eb505fa5bb93c479693d87 AS builder

ENV RUSTUP_TOOLCHAIN=1.99.1
WORKDIR /build
RUN apk add --no-cache g++ cmake perl
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/build/target,sharing=locked \
    cargo build --locked --release --package spine-cli \
    && install -D -m 0755 target/release/spine /out/spine

FROM alpine:3.24@sha256:be2656061b536c1883c4034790e21d0a09c5aaaf51968b89368a3c0371ea2c9a AS runtime

ARG APP_VERSION=0.3.1
LABEL org.opencontainers.image.title="Spine" \
      org.opencontainers.image.description="Standalone native Spine harness" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

RUN apk add --no-cache ca-certificates \
    && install -d -o 10001 -g 10001 /data /workspace \
    && install -d /usr/share/doc/spine

COPY --from=builder /out/spine /usr/local/bin/spine
COPY --from=builder /build/LICENSE-APACHE /build/LICENSE-MIT \
    /build/THIRD_PARTY_NOTICES.md /build/THIRD_PARTY_LICENSES.html \
    /usr/share/doc/spine/

ENV HOME=/data \
    XDG_DATA_HOME=/data \
    SPINE_HEART_PATH=/data/default.spine \
    HF_HUB_CACHE=/models \
    SPINE_LLM_URL=http://host.docker.internal:8080

WORKDIR /workspace
VOLUME ["/data"]
EXPOSE 8088
STOPSIGNAL SIGINT

# This checks the real long-running process, not just whether a new binary can start.
HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
    CMD test "$(readlink /proc/1/exe)" = "/usr/local/bin/spine" || exit 1

USER 10001:10001
ENTRYPOINT ["/usr/local/bin/spine"]
CMD ["chat", "--no-nli"]
