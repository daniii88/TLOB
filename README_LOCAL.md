# lob_model (local project wrapper)

This project is created at:

- `/home/daniii/code/lob_model`

Base model code:

- upstream TLOB repository cloned into this folder
- source: `https://github.com/LeonardoBerti00/TLOB`

Additional local training scaffold (in the style of your TFT pipeline):

- `lob_pipeline.py` profile-driven runner (`train` / `backtest`)
- `profiles/btc_tlob_fast.json` default starter profile
- `Makefile` shortcuts

## Quick start

```bash
cd /home/daniii/code/lob_model
make setup
make train
```

## Run with custom overrides

```bash
cd /home/daniii/code/lob_model
python3 lob_pipeline.py train \
  --profile profiles/btc_tlob_fast.json \
  --set experiment.max_epochs=30 \
  --set experiment.seed=7
```

Select a custom dataset folder (example for ENGINE):

```bash
cd /home/daniii/code/lob_model
python3 main.py +model=tlob +dataset=engine hydra.job.chdir=False \
  experiment.is_wandb=False experiment.is_data_preprocessed=True \
  dataset.data_path=/absolute/path/to/ENGINE
```

## Notes

- The runner delegates to upstream `main.py` (Hydra overrides).
- Profile values are intentionally conservative starter defaults.
- You can add more profiles under `profiles/*.json`.
- BTC training now auto-builds `data/BTC/{train,val,test}.npy` if they are missing, even when `experiment.is_data_preprocessed=true`.

## Train from dydx-highrisk-engine shadow export

1. Enable compact decision export in engine `.env`:

```bash
DYDX_SHADOW_ENTRY_SIGNAL_EXPORT_JSONL=reports/entry_signal_events.jsonl
DYDX_SHADOW_ENTRY_SIGNAL_EXPORT_APPEND=1
DYDX_SHADOW_WRITE_REPORT=0
```

2. Run shadow mode to collect events.

3. Build `ENGINE` dataset (from snapshot folder):

```bash
cd /home/daniii/code/lob_model
make build-engine-dataset ENGINE_EXPORT=/code/dydx-highrisk-engine/reports/entry_signal_snapshots
```

The dataset builder accepts a single `.jsonl`, a `.jsonl.gz`, or a folder with many snapshot files.  
It also supports CPU parallel scan via `--workers` (in `make`: `ENGINE_WORKERS=<n>`).
This uses strict quality filtering by default and writes:

- `data/ENGINE/train.npy`
- `data/ENGINE/val.npy`
- `data/ENGINE/test.npy`
- `data/ENGINE/quality_report.json`

4. Train TLOB on ENGINE dataset:

```bash
cd /home/daniii/code/lob_model
make train-engine
```

Optional: inspect usable-data quality report

```bash
cd /home/daniii/code/lob_model
make analyze-engine-dataset
```
