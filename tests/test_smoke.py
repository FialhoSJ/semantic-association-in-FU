import json
from pathlib import Path

import yaml

from safu.cli import run


def test_complete_phase1_run(tmp_path):
    config = yaml.safe_load(Path("configs/experiments/phase1_smoke.yaml").read_text())
    config["results_root"] = str(tmp_path)
    config["federated"]["global_rounds"] = 1
    config["dataset"]["synthetic_train_size"] = 48
    config["dataset"]["synthetic_test_size"] = 24
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = run(config_path)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["num_train_samples"] == 48
    assert summary["total_communication_bytes"] > 0
    assert (output / "metrics.csv").exists()
    partitions = json.loads((output / "partitions.json").read_text())
    assert sorted(index for values in partitions.values() for index in values) == list(range(48))
    assert (output / "checkpoints" / "final.pt").exists()
