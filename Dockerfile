FROM python:3.12-slim-bookworm

# Install build dependencies
RUN apt-get update && apt-get install -y \
    bash \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library
# Source: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the application code
COPY . .

# Ensure scripts are executable
RUN chmod +x train.sh test.sh

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# Run training and prediction by default
CMD ["bash", "-lc", "./train.sh && ./test.sh"]