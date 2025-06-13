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

## Apache Airflow

### Installation and Setup

To install and run Apache Airflow with project integration:

1. Add Airflow to your project dependencies (already done in pyproject.toml):
   ```bash
   poetry add apache-airflow
   ```

2. Update the lock file:
   ```bash
   poetry lock
   ```

3. Launch Airflow using Docker Compose:
   ```bash
   docker-compose -f docker-compose-airflow.yml up -d
   ```

4. Initialize database and create admin user:
   ```bash
   docker exec -it mlops-recsys-airflow-1 airflow db init
   docker exec -it mlops-recsys-airflow-1 airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
   ```

5. Access the Airflow UI at http://localhost:8080
   - Username: admin
   - Password: admin

6. To stop Airflow:
   ```bash
   docker-compose -f docker-compose-airflow.yml down
   ```

### Project Structure for Airflow

- `docker-compose-airflow.yml` - Docker Compose configuration for Airflow
- `airflow.Dockerfile` - Dockerfile specifically for Airflow
- `requirements-airflow.txt` - Dependencies for the Airflow container
- `dags/` - Directory for Airflow DAG files
- `logs/` - Directory for logs
- `plugins/` - Directory for plugins
- `data/` - Directory for data shared between the project and Airflow

### Creating DAGs
To create new DAGs, simply add new Python files to the `dags/` directory.

### Working with DAGs

1. List available DAGs:
   ```bash
   docker exec -it mlops-recsys-airflow-1 airflow dags list
   ```

2. Trigger a DAG run:
   ```bash
   docker exec -it mlops-recsys-airflow-1 airflow dags trigger mloprec_train_pipeline
   ```

3. Unpause a DAG:
   ```bash
   docker exec -it mlops-recsys-airflow-1 airflow dags unpause mloprec_train_pipeline
   ```

### Integration with MLOps-RecSys Project

The Airflow setup is integrated with the MLOps-RecSys project in the following ways:

1. The Airflow container has access to the entire project code
2. Required dependencies are installed in the Airflow container
3. DAGs can import and use functions from the `mloprec` package
4. It shares the data directory with the main project

### Architecture

The Airflow setup consists of the following components:

1. PostgreSQL database for metadata storage
2. Airflow webserver for UI access
3. Airflow scheduler for running DAGs
4. Shared volumes for DAGs, logs, and data

The containers communicate via a dedicated Docker network.

### MLOps Pipelines with Airflow and DVC

The project includes a complete ML pipeline for training and evaluating recommendation models using Airflow and DVC:

1. **DVC Integration**:
   - The project uses Data Version Control (DVC) to track data and models
   - Integration ensures that all artifacts are properly versioned
   - The pipeline automatically tracks input data, models, and evaluation results

2. **Recommendation System Pipeline**:
   - `recsys_training_pipeline`: A DAG that runs the entire ML lifecycle
   - Pipeline steps include data preparation, model training, evaluation, and versioning
   - All files are tracked with DVC and pushed to remote storage

3. **Setting up DVC in Airflow**:
   - `dvc_init`: A DAG for initializing DVC and configuring remote storage
   - Run this DAG first to set up DVC in the Airflow environment
   - This only needs to be run once for initial setup

4. **Running the Pipeline**:
   ```bash
   # Initialize DVC for the first time
   docker exec -it mlops-recsys-airflow-1 airflow dags trigger dvc_init

   # Run the full recommendation system pipeline
   docker exec -it mlops-recsys-airflow-1 airflow dags trigger recsys_training_pipeline
   ```

5. **Viewing Results**:
   - Check pipeline status in the Airflow UI at http://localhost:8080
   - Model evaluation results are stored in `models/als_evaluation_results.txt`
   - Both data and models are versioned with DVC

## Experiment Tracking with ClearML

This project uses ClearML for experiment tracking, model management, and MLOps automation. ClearML helps track, compare, and visualize machine learning experiments.

### Setting up ClearML

1. Install ClearML:
   ```bash
   pip install clearml
   # or with Poetry
   poetry add clearml
   ```

2. Configure ClearML credentials:
   ```bash
   clearml-init
   ```

   Alternatively, you can use the existing configuration file (`clearml.conf`) located in your home directory.

### Using ClearML for Experiment Tracking

1. Import ClearML in your training scripts:
   ```python
   from clearml import Task

   # Initialize a new task (experiment)
   task = Task.init(project_name="MLOps-RecSys", task_name="ALS Model Training")
   ```

2. Log parameters, metrics, and artifacts:
   ```python
   # Log hyperparameters
   task.connect({'learning_rate': 0.01, 'batch_size': 64})

   # Log metrics
   task.logger.report_scalar(title='Metrics', series='Recall@K', value=0.85, iteration=0)

   # Log artifacts
   task.upload_artifact('model', artifact_object='path/to/model.pkl')
   ```

3. Compare experiments in the ClearML Web UI:
   - Access your ClearML Web UI at https://app.clear.ml/
   - Navigate to your project to view all experiments
   - Compare metrics, parameters, and artifacts between experiments
