# tracin_ghost_tool

Minimal modular TracIn pipeline with exact `faiss.IndexFlatIP` search.

## Run interactively (container)
```bash
bash run_container.sh config.yaml
```

## Submit Slurm job (generic template)
```bash
bash submit_slurm.sh config.yaml tracin-ghost
```

- Argument 1: config path (relative to repo root)
- Argument 2: Slurm job name

## Sync artifacts back from scratch
Sync a single job:
```bash
bash sync_from_scratch.sh 1064576
```

Sync all jobs:
```bash
bash sync_from_scratch.sh all
```

## Where files go
- Scratch runtime outputs: `/scratch/leihong/tracin_ghost_tool_results/<job_id>/outputs`
- Scratch runtime logs: `/scratch/leihong/tracin_ghost_tool_results/<job_id>/logs`
- Scratch slurm logs: `/scratch/leihong/slurm_logs/`
- Synced workspace outputs: `outputs/runs/<job_id>/`
- Synced workspace logs: `logs/runs/<job_id>/` and `logs/slurm/`

## Current reusable scripts
- `run_container.sh`: local container run
- `slurm_run_container.sh`: batch runner (internal template)
- `submit_slurm.sh`: one-command sbatch submit
- `sync_from_scratch.sh`: one-command artifact sync
