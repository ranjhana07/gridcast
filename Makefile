PYTHON ?= python
RAW ?= data/raw/LD2011_2014.txt

.PHONY: install data benchmark dispatch retrain forecast serve test lint docker-up clean

install:
	$(PYTHON) -m pip install -e ".[models,deep,serve,dev]"

data:
	$(PYTHON) -m gridcast.cli build-data --raw $(RAW)

benchmark:
	$(PYTHON) -m gridcast.cli benchmark --folds 4 --fold-days 21 --stride 6

dispatch:
	$(PYTHON) -m gridcast.cli dispatch-eval --model lightgbm --model lstm_quantile --reference naive_168

retrain:
	$(PYTHON) -m gridcast.cli retrain

forecast:
	$(PYTHON) -m gridcast.cli forecast

serve:
	$(PYTHON) -m gridcast.cli serve

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall -q gridcast tests
	node --check dashboard/app.js

docker-up:
	docker compose up --build

clean:
	rm -rf mlruns artifacts .pytest_cache
