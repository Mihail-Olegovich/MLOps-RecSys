"""Compare performance of different models using ClearML artifacts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task

from mloprec.tracking import init_task, log_artifact


def get_model_metrics(
    project_name: str = "MLOps-RecSys",
) -> dict[str, dict[str, float]]:
    """
    Retrieve metrics from ClearML tasks.

    Args:
        project_name: Name of the ClearML project

    Returns:
        Dictionary mapping model names to their metrics
    """
    # Get all completed testing tasks from the project
    tasks = Task.get_tasks(
        project_name=project_name,
        task_filter={
            "type": ["testing"],
            "status": ["completed", "closed"],
        },
    )

    model_metrics = {}

    for task in tasks:
        # Get model name from task name
        if "ALS with Features" in task.name:
            model_name = "ALS with Features"
        elif "ALS without Features" in task.name:
            model_name = "ALS without Features"
        elif "LightFM" in task.name:
            model_name = "LightFM without Features"
        else:
            continue

        # Get metrics from task
        metrics = {}
        last_metrics = task.get_last_scalar_metrics()

        if not last_metrics:
            task.get_logger().report_text(
                f"Нет метрик для задачи {task.id} ({task.name})"
            )
            continue

        for _, series_dict in last_metrics.items():
            for series_name, scalar_dict in series_dict.items():
                if isinstance(scalar_dict, dict):
                    for _, value_data in scalar_dict.items():
                        if isinstance(value_data, tuple):
                            metrics[series_name] = value_data[1]
                        else:
                            metrics[series_name] = value_data
                else:
                    metrics[series_name] = scalar_dict

        model_metrics[model_name] = metrics

    return model_metrics


def generate_comparison_plots(
    metrics: dict[str, dict[str, float]], output_dir: str = "models"
) -> tuple[str | None, str | None, str | None]:
    """
    Generate plots comparing model performance.

    Args:
        metrics: Dictionary mapping model names to their metrics
        output_dir: Directory to save plots

    Returns:
        Tuple of paths to generated plot, table and report files
    """
    if not metrics:
        print("Нет метрик для сравнения моделей.")
        return None, None, None

    # Create bar chart of Recall@40
    plt.figure(figsize=(10, 6))
    models = list(metrics.keys())
    recalls = [metrics[model].get("Recall@40", 0) for model in models]

    # Generate colors
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 0.8, len(models)))

    bars = plt.bar(models, recalls, color=colors)
    plt.title("Recall@40 Comparison", fontsize=16)
    plt.ylabel("Recall@40", fontsize=14)
    plt.ylim(0, max(recalls) * 1.2)

    # Add value labels on bars
    for bar, recall in zip(bars, recalls, strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{recall:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Create comparison table
    model_data = []
    for model_name, model_metrics in metrics.items():
        model_data.append(
            {
                "Model": model_name,
                "Recall@40": model_metrics.get("Recall@40", 0),
            }
        )

    df_comparison = pd.DataFrame(model_data)

    # Save comparison table
    table_path = os.path.join(output_dir, "model_comparison.csv")
    df_comparison.to_csv(table_path, index=False)

    # Generate report with findings
    report = "# Model Comparison Report\n\n"
    report += "## Performance Metrics\n\n"
    report += "| Model | Recall@40 |\n"
    report += "|-------|----------|\n"

    for model_name, model_metrics in metrics.items():
        report += f"| {model_name} | {model_metrics.get('Recall@40', 0):.4f} |\n"

    report += "\n## Analysis\n\n"

    # Find best model
    best_model = max(metrics.items(), key=lambda x: x[1].get("Recall@40", 0))[0]
    report += (
        f"The best performing model is **{best_model}** "
        f"with a Recall@40 of {metrics[best_model].get('Recall@40', 0):.4f}.\n\n"
    )

    # Compare ALS with and without features
    if "ALS with Features" in metrics and "ALS without Features" in metrics:
        als_with = metrics["ALS with Features"].get("Recall@40", 0)
        als_without = metrics["ALS without Features"].get("Recall@40", 0)
        diff = als_with - als_without

        if diff > 0:
            report += (
                f"Using item features with ALS improves Recall@40 by "
                f"{diff:.4f} ({diff/als_without*100:.1f}%).\n\n"
            )
        else:
            report += (
                f"Using item features with ALS does not improve Recall@40 "
                f"(difference: {diff:.4f}).\n\n"
            )

    # Compare ALS without features and LightFM
    if "ALS without Features" in metrics and "LightFM without Features" in metrics:
        als = metrics["ALS without Features"].get("Recall@40", 0)
        lightfm = metrics["LightFM without Features"].get("Recall@40", 0)
        diff = als - lightfm

        if diff > 0:
            report += (
                f"ALS without features outperforms LightFM without features by "
                f"{diff:.4f} ({diff/lightfm*100:.1f}%).\n\n"
            )
        else:
            report += (
                f"LightFM without features outperforms ALS without features by "
                f"{abs(diff):.4f} ({abs(diff)/als*100:.1f}%).\n\n"
            )

    # Save report
    report_path = os.path.join(output_dir, "model_comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return plot_path, table_path, report_path


def main() -> None:
    """Compare model performance and generate report."""
    # Initialize ClearML task for model comparison
    task = init_task(
        task_name="Model Comparison Analysis",
        task_type="data_processing",
    )

    # Get metrics from ClearML
    metrics = get_model_metrics()

    if not metrics:
        task.get_logger().report_text(
            "Не удалось получить метрики для сравнения моделей. "
            "Убедитесь, что задачи оценки были успешно выполнены."
        )
        task.close()
        return

    # Generate comparison plots and report
    plot_path, table_path, report_path = generate_comparison_plots(metrics)

    if plot_path and table_path and report_path:
        # Log artifacts to ClearML
        log_artifact(task, "comparison_plot", plot_path)
        log_artifact(task, "comparison_table", table_path)
        log_artifact(task, "comparison_report", report_path)

        # Также добавим текстовый отчет в лог
        with open(report_path) as f:
            report_content = f.read()
        task.get_logger().report_text(report_content)

    # Complete the task
    task.close()


if __name__ == "__main__":
    main()
