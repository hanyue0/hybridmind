.PHONY: install demo eval-local eval-cloud eval-auto eval-all test clean

PY ?= python
VENV ?= .venv
BIN := $(VENV)/Scripts

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
	$(BIN)/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

data:
	$(BIN)/python scripts/gen_eval_set.py

demo:
	$(BIN)/python -m uvicorn app.web.main:app --host 127.0.0.1 --port 8000

eval-local:
	$(BIN)/python -m app.eval.run_eval --mode local

eval-cloud:
	$(BIN)/python -m app.eval.run_eval --mode cloud

eval-auto:
	$(BIN)/python -m app.eval.run_eval --mode auto

eval-all: eval-local eval-cloud eval-auto

test:
	$(BIN)/python -m unittest discover -s tests -t . -v

clean:
	rm -rf $(VENV) app/__pycache__ app/*/__pycache__ data/results/*.json
