STUDENT_ID ?= CHANGE_ME
PY ?= python

.PHONY: install smoke first20 test test-fast check analysis clean-partial

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

smoke:
	@if [ "$(STUDENT_ID)" = "CHANGE_ME" ]; then echo "usage: make smoke STUDENT_ID=<your id>"; exit 1; fi
	$(PY) smoke_test.py --student-id $(STUDENT_ID)

first20:
	@if [ "$(STUDENT_ID)" = "CHANGE_ME" ]; then echo "usage: make first20 STUDENT_ID=<your id>"; exit 1; fi
	$(PY) -m harness check-first20 --student-id $(STUDENT_ID)

test:
	$(PY) -m pytest -q tests

test-fast:
	$(PY) -m pytest -q tests --no-model

check:
	$(PY) validate_predictions.py --dir predictions
	$(PY) check_results.py

analysis:
	$(PY) analysis.py

clean-partial:
	rm -f predictions/*.partial
