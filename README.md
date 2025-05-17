# MLOps-RecSys

## GitHub Flow

We follow the GitHub Flow model for this project, which consists of the following steps:

1. Create a branch: Branch off from `main` with a descriptive name (e.g., `feature/add-login`).
2. Make changes: Commit logical, atomic changes with clear messages.
3. Open a Pull Request: Push your branch and open a PR against `main`.
4. Collaborate and review: Request reviews, discuss changes, and address feedback.
5. Ensure CI passes: Wait for automated tests and lint checks to succeed.
6. Merge the Pull Request: Once approved, merge into `main` (using merge commit or squash merge).
7. Deploy: Deploy the updated `main` branch to the production environment.
8. Clean up: Delete the feature branch locally and remotely after merging.

This workflow helps keep the codebase stable and simplifies collaboration.

## Code Quality and Linting

This project uses several tools to ensure code quality:

- **Black**: For code formatting
- **Ruff**: For fast Python linting (replaces Flake8, isort, and others)
- **MyPy**: For static type checking
- **Pre-commit**: For running checks before commits

### Setting up the development environment

1. Install dependencies:
   ```bash
   make install
   ```

2. Run formatters:
   ```bash
   make format
   ```

3. Run linters:
   ```bash
   make lint
   ```

4. Run all pre-commit hooks:
   ```bash
   make pre-commit
   ```

Pre-commit hooks will automatically run on each commit to ensure code quality.

## Development Environment

### Setting up with Poetry

This project uses Poetry for dependency management. To set up the development environment:

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Running with Docker

You can also use Docker to run the project:

1. Build the development image:
   ```bash
   docker build --target development -t mloprec-dev .
   ```

2. Run the container:
   ```bash
   docker run -it --rm mloprec-dev
   ```

For production:
```bash
docker build --target production -t mloprec .
docker run -it --rm mloprec
```

## CI/CD Pipeline

Our CI/CD pipeline automatically performs the following steps:

1. **Linting & Type Checking**: Runs Black, Ruff, and MyPy
2. **Testing**: Runs unit tests with pytest
3. **Package Building**: Builds the Python package using Poetry
4. **Docker Image Building**: Builds and publishes the Docker image to GitHub Container Registry
5. **Documentation**: Deploys documentation to GitHub Pages

The pipeline is triggered on:
- Pull requests to `main`
- Pushes to `main`
- Tag pushes (for releases)

## Data Version Control with DVC

This project uses Data Version Control (DVC) to manage and version datasets and models. DVC helps track changes to data files without storing them in Git.

### Getting Started with DVC

1. Install DVC (if not already installed):
   ```bash
   pip install dvc
   # For S3 support
   pip install 'dvc[s3]'
   ```

2. Initialize DVC in the project (already done):
   ```bash
   dvc init
   ```

### Working with Data

1. Add data files to DVC:
   ```bash
   dvc add data/cat_features.csv data/event.csv data/fclickstream.csv
   ```

2. Push data to remote storage:
   ```bash
   dvc push
   ```

3. Pull data from remote storage:
   ```bash
   dvc pull
   ```

Always commit the generated `.dvc` files to Git to track the versions of your datasets and models while keeping the large files themselves outside of the Git repository.
