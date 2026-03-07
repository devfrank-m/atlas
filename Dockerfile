FROM rust:1.94-slim AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:trixie-slim
COPY --from=builder /app/target/release/atlas /usr/local/bin/atlas
CMD ["atlas"]
