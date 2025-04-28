# Project Setup

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker and Docker Compose (for containerization)
- Git

## Installation

### Installation with Docker

```bash
# Clone the repository
git clone https://github.com/Mihail-Olegovich/MLOps-RecSys.git
cd MLOps-RecSys

# Build and run with Docker Compose
docker-compose up dev
```

## Project Structure

```
MLOps-RecSys/
├── .github/           # GitHub Actions configurations
├── docs/              # Project documentation
├── mloprec/           # Main project code
├── tests/             # Tests
├── .pre-commit-config.yaml
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```
