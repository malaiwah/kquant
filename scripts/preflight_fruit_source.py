#!/usr/bin/env python3
"""Authenticate and inventory one pinned Fruit source checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.fruit_source import (
    FRUIT_MODEL_SPECS,
    fruit_model_spec,
    preflight_fruit_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        choices=tuple(FRUIT_MODEL_SPECS),
        default="annealed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = fruit_model_spec(args.variant)
    evidence = preflight_fruit_checkpoint(
        args.checkpoint,
        spec,
        spec.checkpoint_sha256,
    )
    print(
        json.dumps(
            evidence,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
