"""Phase 1 command line runner and result writer."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml

from safu.config import load_config
from safu.data import dirichlet_partition, load_data
from safu.federated import FederatedServer, evaluate
from safu.model import build_model
from safu.reproducibility import git_commit, seed_everything


class JsonFormatter(logging.Formatter):
    """One JSON object per log line for machine-readable experiment records."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }, ensure_ascii=False)


def _result_dir(root: Path, name: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / name / f"{timestamp}_seed{seed}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    return path


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed_everything(config.seed, config.deterministic)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                          "cpu" if config.device == "auto" else config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = _result_dir(Path(config.results_root), config.name, config.seed)
    file_handler = logging.FileHandler(output / "logs.txt")
    file_handler.setFormatter(JsonFormatter())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("safu")
    started = time.perf_counter()
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    logger.info("Loading dataset %s on %s", config.dataset.name, device)
    data = load_data(config.dataset, config.seed)
    partitions = dirichlet_partition(
        data.train_labels, config.federated.num_clients, config.federated.non_iid_alpha,
        config.seed, config.federated.min_client_samples,
    )
    (output / "partitions.json").write_text(
        json.dumps({str(client_id): part.tolist() for client_id, part in enumerate(partitions)}) + "\n",
        encoding="utf-8",
    )
    model = build_model(config.model, data.num_classes)
    server = FederatedServer(model, data.train, data.test, partitions,
                             config.federated, device, config.seed)
    rows: list[dict] = []
    baseline = evaluate(server.model, data.test, config.federated.batch_size, device)
    logger.info("Initial test accuracy %.4f", baseline["test_accuracy"])
    metrics_path = output / "metrics.csv"
    for round_number in range(1, config.federated.global_rounds + 1):
        row = server.run_round(round_number)
        rows.append(row)
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Round %d: loss=%.4f accuracy=%.4f bytes=%d",
                    round_number, row["train_loss"], row["test_accuracy"], row["communication_bytes"])
        if config.save_each_round:
            torch.save(server.model.state_dict(), output / "checkpoints" / f"round_{round_number:04d}.pt")
    torch.save(server.model.state_dict(), output / "checkpoints" / "final.pt")
    summary = {
        "status": "completed",
        "phase": 1,
        "experiment_name": config.name,
        "seed": config.seed,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "dataset": config.dataset.name,
        "num_train_samples": len(data.train),
        "num_test_samples": len(data.test),
        "client_sample_counts": [len(part) for part in partitions],
        "initial_test_accuracy": baseline["test_accuracy"],
        "final_test_accuracy": rows[-1]["test_accuracy"],
        "total_communication_bytes": sum(row["communication_bytes"] for row in rows),
        "duration_seconds": time.perf_counter() - started,
        "synthetic_smoke_test": config.dataset.name == "synthetic",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Results saved to %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAFU phase 1 federated baseline")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
