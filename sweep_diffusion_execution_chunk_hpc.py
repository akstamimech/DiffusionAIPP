import os

from hpc_sweep_common import PlannerConfig, _array_task_id, _run_one, _write_summary


DEFAULT_CHUNKS = (10, 15, 20, 25, 30, 35, 40)


def _parse_chunks():
    raw = os.environ.get("EXECUTION_CHUNK_VALUES")
    if not raw:
        return list(DEFAULT_CHUNKS)
    chunks = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            chunks.append(int(item))
    if not chunks:
        raise SystemExit("EXECUTION_CHUNK_VALUES did not contain any integer chunks")
    return chunks


def _chunk_grid():
    map_id = int(os.environ.get("CHUNK_SWEEP_MAP_ID", os.environ.get("SELECTED_MAP", "95")))
    repeats = int(os.environ.get("REPEATS_PER_CHUNK", "4"))
    chunks = _parse_chunks()
    return [(map_id, execution_chunk, repeat) for execution_chunk in chunks for repeat in range(repeats)]


def _task_subset(items):
    task_id = _array_task_id()
    if task_id is None:
        return items
    index = int(task_id)
    if index < 0 or index >= len(items):
        raise SystemExit(f"Task index {index} is outside 0..{len(items) - 1}")
    return [items[index]]


def main():
    task_id = _array_task_id()
    rows = []
    eta = os.environ.get("ETA", "0.0")

    for map_id, execution_chunk, repeat in _task_subset(_chunk_grid()):
        os.environ["EXECUTION_CHUNK"] = str(execution_chunk)
        variant = f"diffusion_chunk{execution_chunk}"
        print(
            f"[diffusion execution_chunk sweep] map={map_id} "
            f"execution_chunk={execution_chunk} repeat={repeat} eta={eta}",
            flush=True,
        )
        row = _run_one(
            PlannerConfig(
                planner=variant,
                script_name="Diffusionplanner_singlemap.py",
                output_prefix="diffusion",
                extra_env={
                    "ETA": eta,
                    "CHUNK_SWEEP_MAP_ID": str(map_id),
                    "CHUNK_SWEEP_EXECUTION_CHUNK": str(execution_chunk),
                },
            ),
            map_id,
            repeat,
        )
        row["planner"] = "diffusion"
        row["planner_variant"] = variant
        row["sweep"] = "execution_chunk"
        row["execution_chunk"] = execution_chunk
        row["chunk_repeat"] = repeat
        row["eta"] = eta
        rows.append(row)
        _write_summary("diffusion_execution_chunk", rows, task_id=task_id)
        if row["return_code"] != 0 and os.environ.get("STOP_ON_FAILURE", "0") == "1":
            raise SystemExit(row["return_code"])

    summary_path = _write_summary("diffusion_execution_chunk", rows, task_id=task_id)
    print(f"[diffusion execution_chunk sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
