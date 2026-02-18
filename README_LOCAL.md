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

## ENGINE/TLOB Recall Optimization (h100)

Use profile:

- `profiles/engine_tlob_h100_recall.json`

### 6-run matrix (3 alpha variants x 2 training variants)

This runs:

- `alpha_mult` in `{0.50, 0.35, 0.20}`
- Variant A: `weighted_ce` + no sampler
- Variant B: `weighted_ce` + `WeightedRandomSampler` (`pow=1.0`)

Command:

```bash
cd /home/daniii/code/lob_model
make engine-recall-matrix ENGINE_EXPORT=/code/dydx-highrisk-engine/reports/entry_signal_snapshots
```

Artifacts:

- Matrix results: `artifacts/engine_h100_recall_matrix.json`
- Per-run metrics: `data/checkpoints/TLOB/<dir_ckpt>/metrics_summary.json`

### Winner selection rule

- Primary sort: highest `val_event_recall`
- Guard: `val_event_precision >= 0.20`
- Tiebreakers: higher `val_event_f1`, then lower `val_loss`
- Fallback guard once to `0.15` if no run passes `0.20`
- Additional logged diagnostics per run:
  - `val_predicted_event_rate`, `val_event_pr_auc`
  - `val_precision_top_0_1pct`, `val_precision_top_0_2pct`
  - threshold sweep pick on event score (`0.20..0.95`) for `experiment.target_event_rate`

Manual winner selection:

```bash
cd /home/daniii/code/lob_model
make engine-select-winner
```

### Hourly batch rebuild + retrain

Run one cycle:

```bash
cd /home/daniii/code/lob_model
make engine-hourly-once ENGINE_EXPORT=/code/dydx-highrisk-engine/reports/entry_signal_snapshots
```

Run continuously (every hour):

```bash
cd /home/daniii/code/lob_model
python3 scripts/hourly_engine_retrain.py \
  --input /code/dydx-highrisk-engine/reports/entry_signal_snapshots \
  --winner-json artifacts/engine_h100_recall_matrix.json
```

Retention policy in batch script:

- Keep latest run
- Keep top 3 by `val_event_recall`
- Prune older runs for prefix `ENGINE_live_h100*`

### Smoke test for 3 training modes

Runs one short training per mode on synthetic ENGINE splits:

```bash
cd /home/daniii/code/lob_model
make engine-smoke-modes
```
