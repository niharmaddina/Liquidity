# Liquidity: common commands. Uses the Python on PATH; set PYTHON=path/to/venv/bin/python to override.
PYTHON ?= python

.PHONY: install test evaluate evaluate-fixed report train serve

install:        ## backend requirements plus pytest
	$(PYTHON) -m pip install -r backend/requirements-dev.txt

test:           ## unit tests (no network)
	$(PYTHON) -m pytest backend/tests -q

evaluate:       ## walk-forward evaluation, fixed and refit-every-63-days variants; writes results/ (about 30 min on a laptop CPU)
	cd backend && $(PYTHON) evaluate.py

evaluate-fixed: ## fixed-model variant only (about 4 min)
	cd backend && $(PYTHON) evaluate.py --refit-every 0

report:         ## rebuild results/walk_forward.{csv,md} and the figure from results/predictions.csv without retraining
	cd backend && $(PYTHON) evaluate.py --report-only

train:          ## retrain on AAPL; overwrites backend/lnn_ode_model.pth and backend/scalers.joblib
	cd backend && $(PYTHON) train_model.py

serve:          ## FastAPI on http://localhost:8000
	cd backend && $(PYTHON) -m uvicorn main:app --reload
