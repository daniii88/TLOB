PYTHON ?= python3
PROFILE ?= profiles/btc_tlob_fast.json
ENGINE_PROFILE ?= profiles/engine_tlob_entry_v1.json
ENGINE_RECALL_PROFILE ?= profiles/engine_tlob_h100_recall.json
ENGINE_EXPORT ?= /code/dydx-highrisk-engine/reports/entry_signal_snapshots
ENGINE_WORKERS ?= 8

.PHONY: setup train backtest build-engine-dataset train-engine analyze-engine-dataset engine-recall-matrix engine-select-winner engine-hourly-once engine-smoke-modes

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	$(PYTHON) lob_pipeline.py train --profile $(PROFILE)

backtest:
	$(PYTHON) lob_pipeline.py backtest --profile $(PROFILE)

build-engine-dataset:
	$(PYTHON) scripts/build_engine_dataset.py --input $(ENGINE_EXPORT) --output-dir data/ENGINE --normalize --strict-quality --workers $(ENGINE_WORKERS)

analyze-engine-dataset:
	$(PYTHON) -c "import json; print(json.dumps(json.load(open('data/ENGINE/quality_report.json')), indent=2))"

train-engine:
	$(PYTHON) lob_pipeline.py train --profile $(ENGINE_PROFILE)

engine-recall-matrix:
	$(PYTHON) scripts/run_engine_recall_matrix.py --input $(ENGINE_EXPORT) --workers $(ENGINE_WORKERS)

engine-select-winner:
	$(PYTHON) scripts/select_engine_recall_winner.py --metrics-glob "data/checkpoints/TLOB/ENGINE_h100_*/metrics_summary.json"

engine-hourly-once:
	$(PYTHON) scripts/hourly_engine_retrain.py --input $(ENGINE_EXPORT) --once --workers $(ENGINE_WORKERS)

engine-smoke-modes:
	$(PYTHON) scripts/smoke_engine_training_modes.py
