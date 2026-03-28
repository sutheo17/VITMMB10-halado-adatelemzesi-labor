from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from roboflow import Roboflow


@dataclass(frozen=True)
class RoboflowConfig:
    workspace: str = "mohamed-uob"
    project_name: str = "denim"
    version: int = 1
    export_format: str = "coco"
    api_key: Optional[str] = None

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not key:
            raise ValueError(
                "Roboflow API key not found. Set ROBOFLOW_API_KEY or pass api_key explicitly."
            )
        return key


def download_roboflow_coco(config: RoboflowConfig) -> tuple[Path, Path, dict[str, Any]]:
    rf = Roboflow(api_key=config.resolved_api_key())
    project = rf.workspace(config.workspace).project(config.project_name)
    dataset = project.version(config.version).download(config.export_format)

    dataset_root = Path(dataset.location)
    json_files = sorted(dataset_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No COCO json file found under: {dataset_root}")

    json_path = json_files[0]
    coco_data = json.loads(json_path.read_text(encoding="utf-8"))
    return dataset_root, json_path.parent, coco_data
