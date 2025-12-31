# AuroraDB Dockerfile
# Multi-stage build for the AuroraDB Rust library

FROM rustlang/rust:nightly-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files for dependency caching
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Copy any additional files
COPY README.md ./

# Build the library with default features (includes CLI)
RUN cargo build --release --features cli

# Create data directory
RUN mkdir -p /app/data

# Set environment variables for the database
ENV RUST_LOG=info
ENV AURORADB_DATA_DIR=/app/data

# Default command - open a shell for development/interaction
CMD ["/bin/bash"]
