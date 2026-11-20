from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blackbox_experiment import (
    DEFAULT_CRITERION_DIR,
    DEFAULT_LABEL_DIR,
    DEFAULT_OUTPUT_DIR,
    TaskData,
    index_payloads,
    load_task,
    train_task,
)

HISTORY_CHANNELS = (
    "state",
    "observable",
    "yes_ratio",
    "yes_ever",
    "since_last_yes_norm",
    "unknown_ratio",
)


def causal_history_features(
    judgments: np.ndarray, window: int | None
) -> np.ndarray:
    """Build per-frame features from judgments up to and including time t only.

    judgments: (T, K) int8 array with values in {-1, 0, 1}.
    window: None means the full history [0, t]; an int N means [t-N+1, t].
    No frame labels are ever touched here.
    """
    frames, k = judgments.shape
    idx = np.arange(frames, dtype=np.int64)
    if window is None:
        start = np.zeros(frames, dtype=np.int64)
    else:
        if window < 1:
            raise ValueError("--window must be 'full' or an integer >= 1")
        start = np.maximum(0, idx - (window - 1))
    lengths = (idx - start + 1).astype(np.float32)

    yes = (judgments == 1).astype(np.float32)
    unknown = (judgments == -1).astype(np.float32)

    def prefix_ratio(indicator: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(indicator, axis=0)
        before = np.zeros_like(cumulative)
        before[1:] = cumulative[:-1]
        count = cumulative - before[start]
        return count / lengths[:, None]

    yes_ratio = prefix_ratio(yes)
    unknown_ratio = prefix_ratio(unknown)
    yes_ever = (yes_ratio > 0).astype(np.float32)

    last_yes = np.where(judgments == 1, idx[:, None], -np.int64(1))
    last_yes = np.maximum.accumulate(last_yes, axis=0)
    has_yes = last_yes >= start[:, None]
    since_last_yes = np.where(
        has_yes,
        (idx[:, None] - last_yes) / lengths[:, None],
        np.ones((frames, k), dtype=np.float32),
    )

    current = judgments.astype(np.float32)
    state = np.where(current >= 0, current, 0.0)
    observable = (current >= 0).astype(np.float32)

    channels = [state, observable, yes_ratio, yes_ever, since_last_yes, unknown_ratio]
    return np.stack(channels, axis=2).reshape(frames, k * len(channels)).astype(np.float32)


def build_history_task(
    data: TaskData, criterion_payload: dict[str, Any], window: int | None
) -> TaskData:
    """Replace the per-frame encodings with causal history features.

    Episode ordering matches load_task exactly (same payload iteration order),
    so y / frame rows / episode rows are reused unchanged.
    """
    k = len(data.criterion_ids)
    feature_names = [
        f"{criterion_id}.{kind}"
        for criterion_id in data.criterion_ids
        for kind in HISTORY_CHANNELS
    ]
    x_parts: list[np.ndarray] = []
    for episode_id, episode in criterion_payload["episodes"].items():
        frame_count = int(episode["frame_count"])
        by_id = {item["id"]: item for item in episode["criterions"]}
        sequences = [
            np.asarray(by_id[criterion_id]["judgment"], dtype=np.int8)
            for criterion_id in data.criterion_ids
        ]
        judgments = np.stack(sequences, axis=1)
        if judgments.shape != (frame_count, k):
            raise ValueError(f"Judgment matrix mismatch for {episode_id}")
        x_parts.append(causal_history_features(judgments, window))
    return TaskData(
        task=data.task,
        criterion_ids=data.criterion_ids,
        feature_names=feature_names,
        x=np.concatenate(x_parts),
        y=data.y,
        frame_indices=data.frame_indices,
        frame_episode_ids=data.frame_episode_ids,
        episode_ids=data.episode_ids,
        episode_splits=data.episode_splits,
        episode_rows=data.episode_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frame-level Criterion-to-Outcome experiment with causal history features"
    )
    parser.add_argument("--criterion-dir", type=Path, default=DEFAULT_CRITERION_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-steps", default="10,20,30,40,50")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--windows",
        default="10,30,full",
        help="Comma-separated causal windows; 'full' means [0, t], integer N means [t-N+1, t]",
    )
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to estimate sampling variability")
    window_specs = [value.strip().lower() for value in args.windows.split(",") if value.strip()]
    if not window_specs or len(window_specs) != len(set(window_specs)):
        raise ValueError("--windows must contain unique values")
    windows: list[tuple[str, int | None]] = []
    for spec in window_specs:
        window = None if spec == "full" else int(spec)
        if window is not None and window < 1:
            raise ValueError("history windows must be 'full' or integers >= 1")
        windows.append((spec, window))
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
        base_data = load_task(criterion_payload, label_payload)
        window_reports = []
        for window_name, window in windows:
            history_data = build_history_task(base_data, criterion_payload, window)
            report, rows = train_task(
                history_data,
                train_steps,
                args.repeats,
                args.hidden_dim,
                args.epochs,
                args.learning_rate,
                args.seed,
                args.output_dir,
                model_tag=f"history_w{window_name}",
            )
            report["history_window"] = window_name
            report["history_range"] = "[0, t]" if window is None else f"[t-{window - 1}, t]"
            window_reports.append(report)
            for row in rows:
                row["history_window"] = window_name
                predictions.append(row)
        task_reports.append({
            "task": task,
            "criterion_input_file": criterion_path.name,
            "label_input_file": label_path.name,
            "criterion_ids": base_data.criterion_ids,
            "windows": window_reports,
        })

    available = sorted({
        label
        for task_report in task_reports
        for report in task_report["windows"]
        for label in report["available_classes"]
    })
    result = {
        "experiment": "CRIA repeated frame-level Criterion-label black-box MLP with causal history features",
        "base_seed": args.seed,
        "repeat_count": args.repeats,
        "repeat_seeds": [args.seed + index * 1000 for index in range(args.repeats)],
        "history_encoding": {
            "windows": {
                name: "full history [0, t]" if window is None else f"[t-{window - 1}, t]"
                for name, window in windows
            },
            "channels_per_criterion": list(HISTORY_CHANNELS),
            "channel_definitions": {
                "state": "judgment at t, unknown mapped to 0",
                "observable": "1 if judgment at t is known (0/1), 0 if unknown (-1)",
                "yes_ratio": "fraction of Yes frames in the causal window",
                "yes_ever": "1 if any Yes occurred in the causal window",
                "since_last_yes_norm": "(t - last Yes index) / window length, 1.0 if no Yes yet",
                "unknown_ratio": "fraction of Unknown frames in the causal window",
            },
            "feature_count_per_criterion": len(HISTORY_CHANNELS),
        },
        "rolling_protocol": {
            "identical_to_baseline": "episode-level splits, repeated seeded sampling, no episode is split between train and backtest",
            "causality": "features at time t only use judgments in [0, t] or [t-W+1, t]; future frames are never read",
            "leakage_control": "frame_label / episode_label and any label-derived statistic never enter the features; labels are used only as supervision y",
        },
        "data_and_supervision": {
            "criterion_source": str(args.criterion_dir),
            "label_source": str(args.label_dir),
            "features": "causal history aggregates of criterion judgments only",
            "supervision": "frame_label only",
        },
        "label_semantics": {
            "supported": available,
            "mapping": {"0": "anomaly", "1": "success"},
        },
        "tasks": task_reports,
    }
    result_path = args.output_dir / "history_window_backtest_results.json"
    prediction_path = args.output_dir / "history_window_predictions.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_path.write_text(
        json.dumps({"predictions": predictions}, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "tasks": len(task_reports),
        "windows": window_specs,
        "features_per_criterion": len(HISTORY_CHANNELS),
        "result": str(result_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
