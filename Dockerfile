FROM python:3.11-slim AS base

# Set up environment
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Poetry using pip
ENV POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_DEFAULT_TIMEOUT=300

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy poetry configuration files
COPY pyproject.toml ./
COPY poetry.lock ./

FROM base AS development

# Install development dependencies
RUN poetry install --no-interaction --no-ansi

# Copy project files
COPY . .

FROM base AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    poetry config installer.max-workers 4 && \
    poetry install --without dev --no-interaction --no-ansi --no-root || true && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .


RUN pip install -e .

# Create data directory with appropriate permissions
RUN mkdir -p /app/data && \
    chmod 777 /app/data

CMD ["python", "-m", "mloprec"]
