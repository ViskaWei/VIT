# Minimal ViT Launcher

This repo now keeps only the pieces required to train, evaluate, search for a learning rate, and run W&B sweeps. Everything routes through `launch.sh`.

## Commands

| Command | What it does | Defaults |
|---------|--------------|----------|
| `./launch.sh run` | Train using `scripts/run.py` | config: `configs/exp/att_clp/baseline.yaml`, W&B on |
| `./launch.sh test` | Evaluate a checkpoint via `scripts/test.py` | expects `--ckpt best|last|/path.ckpt` |
| `./launch.sh lr` | Two-stage LR + scheduler sweep via `src/opt/parallel_sweep.py` | config: `configs/config.yaml` |
| `./launch.sh sweep` | Create a W&B sweep and launch local agents | requires `-c sweep.yaml` |

### Common flags

- `-c, --config PATH` — override the config file (sweep mode expects a sweep YAML).
- `-g, --gpu VALUE` — for `run`/`test` supply a count (e.g. `-g 2`). For `lr`/`sweep` supply GPU ids (e.g. `-g 0,1,2,3`).
- `-w, --wandb {0,1}` — toggle W&B logging for `run`/`test`.
- `--save` — enable checkpoint saving during `run`.
- `--ckpt PATH` — checkpoint path (or `best`/`last`) for `test`.
- `--dry-run` — preview LR sweep without launching jobs.
- `-e/--entity`, `-p/--project`, `--count` — W&B sweep options.

## Configuration

- `configs/exp/att_clp/baseline.yaml` — fast baseline for sanity checks.
- `configs/config.yaml` — consolidated master config with every tunable field documented.

## W&B Sweep flow

```
# Create sweep + launch agents on GPUs 0 and 1
./launch.sh sweep -c configs/sweep.yaml -e my-entity -p my-project -g 0,1
```

The script creates the sweep via the W&B CLI, prints the sweep ID, and spawns one agent per GPU.

## Learning Rate search

```
./launch.sh lr -c configs/config.yaml -g 0,1,2,3
```

This runs a 7-value LR sweep, grabs the best LR, then compares schedulers using that LR. Results live in `opt_runs/sweep/`.

## Testing

```
./launch.sh test --ckpt checkpoints/latest.ckpt
```

Runs `scripts/test.py`, which simply loads the config, attaches the datamodule, and calls Lightning's `Trainer.test`.

## Run

```
./launch.sh run --save -w 1
```

Trains with the baseline config (or the file provided via `-c`).

---

Everything else (old helper scripts, cached results, etc.) has been removed to keep the repo focused on these four entry points.
