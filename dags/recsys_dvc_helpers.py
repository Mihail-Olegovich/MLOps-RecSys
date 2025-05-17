"""Вспомогательные функции для работы с DVC в Airflow DAG."""

import os
import subprocess


def run_command(command: str, cwd: str | None = None) -> tuple[int, str, str]:
    """
    Запускает команду и возвращает код возврата, stdout и stderr.

    Args:
        command: Команда для выполнения
        cwd: Рабочая директория (опционально)

    Returns:
        Кортеж из (код возврата, stdout, stderr)
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def check_dvc_initialized(project_path: str) -> bool:
    """
    Проверяет, инициализирован ли DVC в проекте.

    Args:
        project_path: Путь к проекту

    Returns:
        True если DVC инициализирован, иначе False
    """
    return os.path.exists(os.path.join(project_path, ".dvc"))


def initialize_dvc(project_path: str) -> tuple[bool, str]:
    """
    Инициализирует DVC в проекте, если он еще не инициализирован.

    Args:
        project_path: Путь к проекту

    Returns:
        Кортеж из (успех, сообщение)
    """
    if check_dvc_initialized(project_path):
        return True, "DVC already initialized"

    returncode, stdout, stderr = run_command("dvc init", cwd=project_path)
    if returncode != 0:
        return False, f"Failed to initialize DVC: {stderr}"
    return True, "DVC initialized successfully"


def setup_dvc_remote(
    project_path: str, remote_name: str, remote_url: str
) -> tuple[bool, str]:
    """
    Настраивает удаленное хранилище DVC.

    Args:
        project_path: Путь к проекту
        remote_name: Имя удаленного хранилища
        remote_url: URL удаленного хранилища

    Returns:
        Кортеж из (успех, сообщение)
    """
    # Добавление удаленного хранилища
    cmd_add = f"dvc remote add {remote_name} {remote_url}"
    returncode, stdout, stderr = run_command(cmd_add, cwd=project_path)
    if returncode != 0:
        # Если хранилище уже существует, попробуем его изменить
        cmd_modify = f"dvc remote modify {remote_name} url {remote_url}"
        returncode, stdout, stderr = run_command(cmd_modify, cwd=project_path)
        if returncode != 0:
            return False, f"Failed to setup DVC remote: {stderr}"

    # Установка хранилища по умолчанию
    cmd_default = f"dvc remote default {remote_name}"
    returncode, stdout, stderr = run_command(cmd_default, cwd=project_path)
    if returncode != 0:
        return False, f"Failed to set default DVC remote: {stderr}"

    return True, "DVC remote setup successfully"


def track_file_with_dvc(project_path: str, file_path: str) -> tuple[bool, str]:
    """
    Добавляет файл под контроль DVC.

    Args:
        project_path: Путь к проекту
        file_path: Относительный путь к файлу от корня проекта

    Returns:
        Кортеж из (успех, сообщение)
    """
    abs_file_path = os.path.join(project_path, file_path)
    if not os.path.exists(abs_file_path):
        return False, f"File {abs_file_path} does not exist"

    returncode, stdout, stderr = run_command(f"dvc add {file_path}", cwd=project_path)
    if returncode != 0:
        return False, f"Failed to add file to DVC: {stderr}"
    return True, f"File {file_path} added to DVC successfully"


def push_to_dvc_remote(project_path: str) -> tuple[bool, str]:
    """
    Отправляет данные в удаленное хранилище DVC.

    Args:
        project_path: Путь к проекту

    Returns:
        Кортеж из (успех, сообщение)
    """
    returncode, stdout, stderr = run_command("dvc push", cwd=project_path)
    if returncode != 0:
        return False, f"Failed to push data to DVC remote: {stderr}"
    return True, "Data pushed to DVC remote successfully"
