from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch


def numeric_sort_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Could not find a numeric id in filename: {path.name}")
    return tuple(int(number) for number in numbers)


def load_pt(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise TypeError(f"{path} is {type(obj).__name__}, expected a dict of tensors.")
    return obj


def concat_dicts(files: list[Path]) -> dict[str, Any]:
    if not files:
        raise ValueError("No input files found.")

    buckets: dict[str, list[Any]] = {}
    expected_keys: set[str] | None = None

    for path in files:
        data = load_pt(path)
        keys = set(data.keys())

        if expected_keys is None:
            expected_keys = keys
            for key in data:
                buckets[key] = []
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(f"Key mismatch in {path.name}. Missing={missing}, extra={extra}")

        print(f"Loading {path.name}")
        for key, value in data.items():
            buckets[key].append(value)

    output: dict[str, Any] = {}

    for key, values in buckets.items():
        first = values[0]

        if torch.is_tensor(first):
            for file_path, value in zip(files, values):
                if not torch.is_tensor(value):
                    raise TypeError(f"Key {key!r} is tensor in first file but not in {file_path.name}.")
                if value.dim() != first.dim():
                    raise ValueError(
                        f"Rank mismatch for key {key!r} in {file_path.name}: "
                        f"{value.dim()} vs {first.dim()}"
                    )
                if first.dim() > 0 and value.shape[1:] != first.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for key {key!r} in {file_path.name}: "
                        f"{tuple(value.shape)} vs expected (*, {tuple(first.shape[1:])})"
                    )

            output[key] = torch.cat(values, dim=0) if first.dim() > 0 else torch.stack(values, dim=0)
        else:
            output[key] = values

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate ordered map_*_ranked.pt files into one dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Folder containing the ranked .pt files. Defaults to the current folder.",
    )
    parser.add_argument(
        "--pattern",
        default="map_*_ranked.pt",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt path. Defaults to <input-dir>/ranked_concat.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output or input_dir / "ranked_concat.pt"
    output_path = output_path.resolve()

    files = sorted(input_dir.glob(args.pattern), key=numeric_sort_key)
    files = [path for path in files if path.resolve() != output_path]

    print("Input order:")
    for path in files:
        print(f"  {path.name}")

    dataset = concat_dicts(files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\nSaved: {output_path}")
    for key, value in dataset.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value).__name__}, len={len(value) if hasattr(value, '__len__') else 'n/a'}")


if __name__ == "__main__":
    main()
