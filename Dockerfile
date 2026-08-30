FROM rust:1.98.0-alpine3.22@sha256:2e452153b2dc6bed8ef123c4801cfd3f637e7c092846f974ecdce5e64af3120c AS builder

ENV RUSTUP_TOOLCHAIN=1.98.0
WORKDIR /build
RUN apk add --no-cache g++
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/build/target,sharing=locked \
    cargo build --locked --release --package spine-cli \
    && install -D -m 0755 target/release/spine /out/spine

FROM alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce AS runtime

ARG APP_VERSION=0.1.0
LABEL org.opencontainers.image.title="Spine" \
      org.opencontainers.image.description="Standalone native Spine harness" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

RUN install -d -o 10001 -g 10001 /data /workspace \
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
