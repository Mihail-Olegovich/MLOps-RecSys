"""DAG для тренировки и оценки рекомендательной системы с интеграцией DVC."""

import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from recsys_dvc_helpers import push_to_dvc_remote, track_file_with_dvc

# Путь к проекту
PROJECT_PATH = "/app"
MODELS_DIR = os.path.join(PROJECT_PATH, "models")
DATA_DIR = os.path.join(PROJECT_PATH, "data")

default_args = {
    "owner": "mloprec",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def prepare_directories(**kwargs: dict[str, Any]) -> bool:
    """Создает необходимые директории для моделей и результатов."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Models directory created: {MODELS_DIR}")
    print(f"Data directory created: {DATA_DIR}")
    return True


def track_files_with_dvc(**kwargs: dict[str, Any]) -> bool:
    """Добавляет файлы под контроль DVC."""
    files_to_track = kwargs.get(
        "files",
        [
            "data/train.csv",
            "data/eval.csv",
            "models/als_model.pkl",
            "models/als_evaluation_results.txt",
        ],
    )

    for file_path in files_to_track:
        full_path = os.path.join(PROJECT_PATH, file_path)
        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} does not exist, skipping")
            continue

        success, message = track_file_with_dvc(PROJECT_PATH, file_path)
        print(message)
        if not success:
            print(f"Warning: Failed to track {file_path}: {message}")

    return True


def push_to_remote(**kwargs: dict[str, Any]) -> bool:
    """Отправляет данные в удаленное хранилище DVC."""
    success, message = push_to_dvc_remote(PROJECT_PATH)
    print(message)
    if not success:
        print(f"Warning: Failed to push to remote: {message}")

    return True


with DAG(
    "recsys_training_pipeline",
    default_args=default_args,
    description="Тренировка и оценка рекомендательной системы с интеграцией DVC",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["recsys", "als", "dvc"],
) as dag:
    # Задача 0: Подготовка окружения
    prepare_env = PythonOperator(
        task_id="prepare_environment",
        python_callable=prepare_directories,
    )

    # Задача 1: Подготовка данных
    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command=f"cd {PROJECT_PATH} && python -m mloprec.scripts.data_prep",
    )

    # Задача 2: Трекинг данных с DVC
    track_data = PythonOperator(
        task_id="track_data_with_dvc",
        python_callable=track_files_with_dvc,
        op_kwargs={"files": ["data/train.csv", "data/eval.csv"]},
    )

    # Задача 3: Тренировка ALS модели
    train_als_model = BashOperator(
        task_id="train_als_model",
        bash_command=f"cd {PROJECT_PATH} && python -m mloprec.scripts.train_als",
    )

    # Задача 4: Трекинг модели с DVC
    track_model = PythonOperator(
        task_id="track_model_with_dvc",
        python_callable=track_files_with_dvc,
        op_kwargs={"files": ["models/als_model.pkl"]},
    )

    # Задача 5: Оценка модели
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_PATH} && python -m mloprec.scripts.als_eval",
    )

    # Задача 6: Трекинг результатов оценки с DVC
    track_evaluation = PythonOperator(
        task_id="track_evaluation_with_dvc",
        python_callable=track_files_with_dvc,
        op_kwargs={"files": ["models/als_evaluation_results.txt"]},
    )

    # Задача 7: Отправка всех данных в удаленное хранилище DVC
    push_dvc = PythonOperator(
        task_id="push_to_dvc_remote",
        python_callable=push_to_remote,
    )

    (
        prepare_env
        >> data_preparation
        >> track_data
        >> train_als_model
        >> track_model
        >> evaluate_model
        >> track_evaluation
        >> push_dvc
    )
