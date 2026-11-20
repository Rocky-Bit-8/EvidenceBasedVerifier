from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DEFAULT_CRITERION_DIR = ROOT / "inputs" / "criterionANDjudgment"
DEFAULT_LABEL_DIR = ROOT / "inputs" / "frameAnomalyLabels"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
LABELS = ("anomaly", "success")


@dataclass
class TaskData:
    task: str
    criterion_ids: list[str]
    feature_names: list[str]
    x: np.ndarray
    y: np.ndarray
    frame_indices: np.ndarray
    frame_episode_ids: np.ndarray
    episode_ids: list[str]
    episode_splits: list[str]
    episode_rows: dict[str, np.ndarray]


class CriterionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def index_payloads(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = payload["task"]
        if task in indexed:
            raise ValueError(f"Duplicate task {task} in {directory}")
        indexed[task] = (path, payload)
    return indexed


def encode_frame_judgments(
    episode: dict[str, Any], criterion_ids: list[str], frame_count: int
) -> np.ndarray:
    by_id = {item["id"]: item for item in episode["criterions"]}
    channels: list[np.ndarray] = []
    for criterion_id in criterion_ids:
        if criterion_id not in by_id:
            raise ValueError(f"Missing criterion {criterion_id}")
        values = np.asarray(by_id[criterion_id]["judgment"], dtype=np.int8)
        if len(values) != frame_count:
            raise ValueError(f"Criterion {criterion_id} frame count mismatch")
        if not np.isin(values, [-1, 0, 1]).all():
            raise ValueError(f"Criterion {criterion_id} contains labels outside -1/0/1")
        channels.append(np.where(values >= 0, values, 0).astype(np.float32))
        channels.append((values >= 0).astype(np.float32))
    return np.stack(channels, axis=1)


def load_task(
    criterion_payload: dict[str, Any], label_payload: dict[str, Any]
) -> TaskData:
    if criterion_payload["task"] != label_payload["task"]:
        raise ValueError("Criterion and frame-label task mismatch")
    if label_payload.get("label_mapping") != {"anomaly": 0, "success": 1}:
        raise ValueError(f"Unexpected label mapping for {criterion_payload['task']}")
    criterion_episodes = criterion_payload["episodes"]
    label_episodes = label_payload["episodes"]
    if set(criterion_episodes) != set(label_episodes):
        raise ValueError(f"Episode IDs do not match for {criterion_payload['task']}")
    first = next(iter(criterion_episodes.values()))
    criterion_ids = [item["id"] for item in first["criterions"]]
    feature_names = [
        f"{criterion_id}.{kind}"
        for criterion_id in criterion_ids
        for kind in ("state", "observable")
    ]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    frame_index_parts: list[np.ndarray] = []
    frame_episode_parts: list[np.ndarray] = []
    episode_ids: list[str] = []
    episode_splits: list[str] = []
    episode_rows: dict[str, np.ndarray] = {}
    offset = 0
    for episode_id, episode in criterion_episodes.items():
        label_episode = label_episodes[episode_id]
        frame_count = int(episode["frame_count"])
        if frame_count != int(label_episode["frame_count"]):
            raise ValueError(f"Frame count mismatch for {episode_id}")
        if episode["split"] != label_episode["split"]:
            raise ValueError(f"Split mismatch for {episode_id}")
        labels = np.asarray(label_episode["frame_labels"], dtype=np.int64)
        if len(labels) != frame_count or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Invalid frame labels for {episode_id}")
        x_parts.append(encode_frame_judgments(episode, criterion_ids, frame_count))
        y_parts.append(labels)
        frame_index_parts.append(np.arange(frame_count, dtype=np.int64))
        frame_episode_parts.append(np.full(frame_count, episode_id, dtype=object))
        episode_ids.append(episode_id)
        episode_splits.append(episode["split"])
        episode_rows[episode_id] = np.arange(offset, offset + frame_count, dtype=np.int64)
        offset += frame_count
    return TaskData(
        task=criterion_payload["task"],
        criterion_ids=criterion_ids,
        feature_names=feature_names,
        x=np.concatenate(x_parts),
        y=np.concatenate(y_parts),
        frame_indices=np.concatenate(frame_index_parts),
        frame_episode_ids=np.concatenate(frame_episode_parts),
        episode_ids=episode_ids,
        episode_splits=episode_splits,
        episode_rows=episode_rows,
    )


def episode_order(episode_ids: list[str], seed: int) -> list[str]:
    order = episode_ids.copy()
    random.Random(seed).shuffle(order)
    return order


def rows_for_episodes(data: TaskData, episode_ids: list[str]) -> np.ndarray:
    if not episode_ids:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([data.episode_rows[episode_id] for episode_id in episode_ids])


def class_weights(y: np.ndarray, class_ids: list[int]) -> torch.Tensor:
    counts = Counter(int(value) for value in y)
    total = len(y)
    return torch.tensor(
        [total / (len(class_ids) * counts[class_id]) for class_id in class_ids],
        dtype=torch.float32,
    )


def fit_model(
    x: np.ndarray,
    y_global: np.ndarray,
    class_ids: list[int],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> CriterionMLP:
    set_seed(seed)
    local_by_global = {global_id: local_id for local_id, global_id in enumerate(class_ids)}
    y = np.asarray([local_by_global[int(value)] for value in y_global], dtype=np.int64)
    model = CriterionMLP(x.shape[1], hidden_dim, len(class_ids))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, list(range(len(class_ids)))))
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    return model


def predict(
    model: CriterionMLP, x: np.ndarray, class_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    local_predictions = probabilities.argmax(axis=1)
    predictions = np.asarray([class_ids[index] for index in local_predictions], dtype=np.int64)
    global_probabilities = np.zeros((len(x), len(LABELS)), dtype=np.float32)
    for local_id, global_id in enumerate(class_ids):
        global_probabilities[:, global_id] = probabilities[:, local_id]
    return predictions, global_probabilities


def metrics(y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[int]) -> dict[str, Any]:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id in class_ids:
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[LABELS[class_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[class_id, :].sum()),
        }
    return {
        "frame_count": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": {"label_order": list(LABELS), "values": matrix.tolist()},
    }


def evaluate_episodes(
    model: CriterionMLP,
    data: TaskData,
    episode_ids: list[str],
    class_ids: list[int],
    include_predictions: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not episode_ids:
        return None, []
    rows = rows_for_episodes(data, episode_ids)
    predictions, probabilities = predict(model, data.x[rows], class_ids)
    report = metrics(data.y[rows], predictions, class_ids)
    episode_accuracies = []
    offset = 0
    for episode_id in episode_ids:
        size = len(data.episode_rows[episode_id])
        episode_accuracies.append(float(np.mean(data.y[rows[offset:offset + size]] == predictions[offset:offset + size])))
        offset += size
    report["episode_count"] = len(episode_ids)
    report["episode_macro_accuracy"] = float(np.mean(episode_accuracies))
    output_rows: list[dict[str, Any]] = []
    if include_predictions:
        split_by_episode = dict(zip(data.episode_ids, data.episode_splits))
        for position, row in enumerate(rows):
            output_rows.append({
                "episode_id": str(data.frame_episode_ids[row]),
                "frame_index": int(data.frame_indices[row]),
                "split": split_by_episode[str(data.frame_episode_ids[row])],
                "truth": LABELS[int(data.y[row])],
                "prediction": LABELS[int(predictions[position])],
                "probabilities": {
                    label: float(probabilities[position, label_id])
                    for label_id, label in enumerate(LABELS)
                },
            })
    return report, output_rows


def summarize_metric_reports(reports: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    available = [report for report in reports if report is not None]
    if not available:
        return None
    summary: dict[str, Any] = {"repeat_count": len(available)}
    for metric in ("accuracy", "macro_f1", "episode_macro_accuracy"):
        values = np.asarray([float(report[metric]) for report in available], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    return summary


def train_task(
    data: TaskData,
    train_steps: list[int],
    repeats: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    output_dir: Path,
    model_tag: str = "frame",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_to_episodes = {
        split: [episode_id for episode_id, value in zip(data.episode_ids, data.episode_splits) if value == split]
        for split in ("train", "val", "test")
    }
    class_ids = sorted(set(int(value) for value in data.y))
    if class_ids != [0, 1]:
        raise ValueError(f"{data.task} does not contain both frame classes")
    steps = sorted({min(step, len(split_to_episodes["train"])) for step in train_steps if step >= 1})
    if len(split_to_episodes["train"]) not in steps:
        steps.append(len(split_to_episodes["train"]))
    safe_task = data.task.replace("/", "__")
    repeat_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        repeat_id = repeat_index + 1
        repeat_seed = seed + repeat_index * 1000
        order = episode_order(split_to_episodes["train"], repeat_seed)
        rounds: list[dict[str, Any]] = []
        final_model: CriterionMLP | None = None
        for round_id, size in enumerate(steps, start=1):
            acquired = order[:size]
            remaining = order[size:]
            acquired_rows = rows_for_episodes(data, acquired)
            model_seed = repeat_seed + round_id
            model = fit_model(
                data.x[acquired_rows], data.y[acquired_rows], class_ids,
                hidden_dim, epochs, learning_rate, model_seed,
            )
            remaining_metrics, _ = evaluate_episodes(model, data, remaining, class_ids)
            val_metrics, _ = evaluate_episodes(model, data, split_to_episodes["val"], class_ids)
            rounds.append({
                "round": round_id,
                "train_episode_count": size,
                "train_frame_count": int(len(acquired_rows)),
                "sampled_episode_ids": acquired,
                "remaining_episode_ids": remaining,
                "model_seed": model_seed,
                "remaining_train_backtest": remaining_metrics,
                "validation": val_metrics,
            })
            final_model = model
        assert final_model is not None
        final_val, val_predictions = evaluate_episodes(
            final_model, data, split_to_episodes["val"], class_ids, True
        )
        final_test, test_predictions = evaluate_episodes(
            final_model, data, split_to_episodes["test"], class_ids, True
        )
        for evaluation, rows in (("validation", val_predictions), ("test", test_predictions)):
            for row in rows:
                row.update({
                    "task": data.task,
                    "repeat": repeat_id,
                    "repeat_seed": repeat_seed,
                    "evaluation": evaluation,
                })
                prediction_rows.append(row)
        model_name = f"{safe_task}_{model_tag}_model_repeat_{repeat_id:02d}.pt"
        torch.save({
            "state_dict": final_model.state_dict(),
            "input_dim": data.x.shape[1],
            "hidden_dim": hidden_dim,
            "class_ids": class_ids,
            "labels": LABELS,
            "criterion_ids": data.criterion_ids,
            "feature_names": data.feature_names,
            "input_features": model_tag,
            "supervision_level": "frame",
            "repeat": repeat_id,
            "repeat_seed": repeat_seed,
            "sampled_train_episode_ids": order,
        }, output_dir / model_name)
        repeat_reports.append({
            "repeat": repeat_id,
            "repeat_seed": repeat_seed,
            "episode_order": order,
            "rounds": rounds,
            "final_validation": final_val,
            "final_test": final_test,
            "model_path": model_name,
        })
    step_summaries = []
    for round_index, size in enumerate(steps):
        step_summaries.append({
            "train_episode_count": size,
            "remaining_train_backtest": summarize_metric_reports([
                report["rounds"][round_index]["remaining_train_backtest"]
                for report in repeat_reports
            ]),
            "validation": summarize_metric_reports([
                report["rounds"][round_index]["validation"]
                for report in repeat_reports
            ]),
        })
    return {
        "task": data.task,
        "episode_count": len(data.episode_ids),
        "frame_count": int(len(data.y)),
        "criterion_ids": data.criterion_ids,
        "feature_count": int(data.x.shape[1]),
        "available_classes": [LABELS[index] for index in class_ids],
        "split_episode_counts": dict(Counter(data.episode_splits)),
        "split_frame_counts": {
            split: int(len(rows_for_episodes(data, episodes)))
            for split, episodes in split_to_episodes.items()
        },
        "repeat_count": repeats,
        "repeats": repeat_reports,
        "step_summaries": step_summaries,
        "final_validation_summary": summarize_metric_reports([
            report["final_validation"] for report in repeat_reports
        ]),
        "final_test_summary": summarize_metric_reports([
            report["final_test"] for report in repeat_reports
        ]),
    }, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling frame-level Criterion-to-Outcome experiment")
    parser.add_argument("--criterion-dir", type=Path, default=DEFAULT_CRITERION_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-steps", default="10,20,30,40,50")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to estimate sampling variability")
    train_steps = [int(value) for value in args.train_steps.split(",") if value]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    criterion_payloads = index_payloads(args.criterion_dir)
    label_payloads = index_payloads(args.label_dir)
    if set(criterion_payloads) != set(label_payloads):
        raise ValueError("Criterion and frame-label task sets do not match")
    task_reports = []
    predictions: list[dict[str, Any]] = []
    for task in sorted(criterion_payloads):
        criterion_path, criterion_payload = criterion_payloads[task]
        label_path, label_payload = label_payloads[task]
        data = load_task(criterion_payload, label_payload)
        report, rows = train_task(
            data, train_steps, args.repeats, args.hidden_dim, args.epochs,
            args.learning_rate, args.seed, args.output_dir,
        )
        report["criterion_input_file"] = criterion_path.name
        report["label_input_file"] = label_path.name
        task_reports.append(report)
        predictions.extend(rows)
    available = sorted({label for report in task_reports for label in report["available_classes"]})
    result = {
        "experiment": "CRIA repeated frame-level Criterion-label black-box MLP",
        "base_seed": args.seed,
        "repeat_count": args.repeats,
        "repeat_seeds": [args.seed + index * 1000 for index in range(args.repeats)],
        "feature_encoding": {
            "unit": "frame",
            "channels_per_criterion": ["binary_state", "observable_mask"],
            "unknown_encoding": {"state": 0, "observable": 0},
        },
        "rolling_protocol": {
            "train_steps_are_episode_counts": train_steps,
            "repeated_sampling": "each repeat independently shuffles official train episodes with its recorded seed",
            "same_size_sets": "different repeats use different episode subsets for the same train size",
            "training": "all frames from acquired episodes",
            "backtest": "all frames from official train episodes not yet acquired",
            "leakage_control": "an episode is never split between acquired and remaining sets",
            "validation": "official validation episodes are never acquired",
            "final_test": "official test episodes are evaluated only by the full-train model",
        },
        "data_and_supervision": {
            "criterion_source": str(args.criterion_dir),
            "label_source": str(args.label_dir),
            "features": "frame-aligned criterion judgments only",
            "supervision": "frame_label only",
        },
        "label_semantics": {
            "supported": available,
            "mapping": {"0": "anomaly", "1": "success"},
        },
        "tasks": task_reports,
    }
    (args.output_dir / "frame_rolling_backtest_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "frame_predictions.json").write_text(
        json.dumps({"predictions": predictions}, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "tasks": len(task_reports),
        "available_classes": available,
        "result": str(args.output_dir / "frame_rolling_backtest_results.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
