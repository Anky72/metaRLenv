FROM python:3.11-slim

# HF Spaces runs containers as UID 1000 — create the user first
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies — fallback to --no-frozen if lockfile is stale
RUN uv sync --no-install-project || uv sync --no-frozen --no-install-project

# Copy the full project
COPY . .

# Remove any accidentally committed secrets
RUN rm -f .env _env

# Final dependency sync (installs the project itself)
RUN uv sync || uv sync --no-frozen

# Fix line endings and make entrypoint executable
RUN sed -i 's/\r//' /app/Entrypoint.sh && chmod +x /app/Entrypoint.sh

# Give the non-root user ownership of /app
RUN chown -R 1000:1000 /app

# Switch to non-root user (required by HF Spaces)
USER 1000

# HF Spaces always maps to port 7860
EXPOSE 7860

# Default: run the web server on port 7860
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]