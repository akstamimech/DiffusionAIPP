"""
Concatenates DataCollector_3D_randomstart_CMAESregularized.py's per-map chunks
(map_{selected_map:03d}_ranked_3d.pt, written by save_dataset_chunk, living in
CMAES_beamsearch_dataset_3d_randomstart_multimodal_chunks/) for the first
MAP_COUNT maps into one dataset file. Uses the same field-agnostic
torch.cat-along-dim-0 logic as the collector's own consolidate_chunks, so it
doesn't need to import the collector module (no mpi4py/cma/sklearn
dependency), just torch. Lives in scripts/ itself, not the chunks subfolder -
CHUNK_DIR points at the subfolder explicitly.

Only concatenates whichever of those maps' chunks are actually present;
missing maps are reported, not treated as an error, since a partial
collection run is a normal thing to want to consolidate early.
"""
import re
from pathlib import Path

import torch

START_MAP = 0
MAP_COUNT = 40  # first 40 maps: START_MAP .. START_MAP+MAP_COUNT-1
SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_DIR = SCRIPT_DIR / "CMAES_beamsearch_dataset_3d_randomstart_multimodal_chunks"
OUTPUT_PATH = SCRIPT_DIR / f"CMAES_beamsearch_dataset_3d_maps_{START_MAP:03d}-{START_MAP + MAP_COUNT - 1:03d}.pt"
DELETE_CHUNKS_AFTER = False  # True -> remove the consolidated per-map chunk files afterward,
                             # same as the collector's own consolidate_chunks default. Left off
                             # by default here since this is meant to be run standalone/early,
                             # possibly against a still-in-progress collection job.

CHUNK_NAME_RE = re.compile(r"^map_(\d{3})_ranked_3d\.pt$")


def discover_chunk_paths():
    wanted_ids = set(range(START_MAP, START_MAP + MAP_COUNT))
    found = {}
    for path in CHUNK_DIR.glob("map_*_ranked_3d.pt"):
        match = CHUNK_NAME_RE.match(path.name)
        if not match:
            continue
        map_id = int(match.group(1))
        if map_id in wanted_ids:
            found[map_id] = path

    missing = sorted(wanted_ids - found.keys())
    if missing:
        print(f"Missing chunks for {len(missing)} map(s) (not collected yet, or a different "
              f"MAPTYPE/naming): {missing}")

    return [found[map_id] for map_id in sorted(found)]


def main():
    chunk_paths = discover_chunk_paths()
    if not chunk_paths:
        raise SystemExit(f"No chunks found for maps {START_MAP}..{START_MAP + MAP_COUNT - 1} in {CHUNK_DIR}")

    print(f"Consolidating {len(chunk_paths)} chunk(s): {[p.name for p in chunk_paths]}")

    first_payload = torch.load(chunk_paths[0], map_location="cpu")
    final_payload = {key: [] for key in first_payload}

    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu")
        for key, value in payload.items():
            final_payload[key].append(value)

    final_payload = {
        key: torch.cat(values, dim=0)
        for key, values in final_payload.items()
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, OUTPUT_PATH)
    print(f"Saved consolidated dataset to {OUTPUT_PATH}")
    print(f"Samples: {len(final_payload['map_id'])}")

    if DELETE_CHUNKS_AFTER:
        for path in chunk_paths:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                print(f"Could not delete locked chunk {path}; leaving it on disk.")


if __name__ == "__main__":
    main()
