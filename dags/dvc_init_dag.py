"""DAG для инициализации и настройки DVC."""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from recsys_dvc_helpers import check_dvc_initialized, initialize_dvc, setup_dvc_remote

# Путь к проекту
PROJECT_PATH = "/app"
# Настройки DVC remote
DVC_REMOTE_NAME = "myremote"
# Предполагаем, что DVC использует локальное хранилище для демонстрации
# В продакшене это может быть S3, GCS, SSH и т.д.
DVC_REMOTE_URL = "/app/dvc-store"

default_args = {
    "owner": "mloprec",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def init_dvc_task(**kwargs: dict[str, Any]) -> str:
    """Инициализирует DVC в проекте."""
    if check_dvc_initialized(PROJECT_PATH):
        print("DVC already initialized")
        return "DVC already initialized"

    success, message = initialize_dvc(PROJECT_PATH)
    if not success:
        raise Exception(message)

    print(message)
    return str(message)


def setup_dvc_remote_task(**kwargs: dict[str, Any]) -> str:
    """Настраивает удаленное хранилище DVC."""
    import os

    # Создаем директорию для локального хранилища, если она не существует
    remote_dir = DVC_REMOTE_URL
    if not os.path.exists(remote_dir):
        os.makedirs(remote_dir)

    success, message = setup_dvc_remote(PROJECT_PATH, DVC_REMOTE_NAME, DVC_REMOTE_URL)
    if not success:
        raise Exception(message)

    print(message)
    return str(message)


with DAG(
    "dvc_init",
    default_args=default_args,
    description="Инициализация и настройка DVC",
    schedule_interval=None,  # Запускается только вручную
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["dvc", "setup"],
) as dag:
    # Задача 1: Инициализация DVC
    init_dvc = PythonOperator(
        task_id="init_dvc",
        python_callable=init_dvc_task,
    )

    # Задача 2: Настройка DVC remote
    setup_remote = PythonOperator(
        task_id="setup_dvc_remote",
        python_callable=setup_dvc_remote_task,
    )

    init_dvc >> setup_remote
