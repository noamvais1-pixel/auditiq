VENV := backend/.venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup ingest api web dev test eval check clean demo

help:
	@echo "make setup   התקנת תלויות (Python + Node)"
	@echo "make demo    התקנה, אינדוקס והרצה — הכל בפקודה אחת"
	@echo "make api     הרצת שרת ה-API על פורט 8000"
	@echo "make web     הרצת הממשק על פורט 5173"
	@echo "make test    הרצת מבחני היחידה"
	@echo "make eval    הרצת חבילת ההערכה"
	@echo "make check   test + eval + typecheck — מה שרץ ב-CI"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r backend/requirements.txt

setup: $(VENV)
	cd frontend && npm install

ingest: $(VENV)
	cd backend && ../$(PY) -m app.ingest

api: $(VENV)
	cd backend && ../$(VENV)/bin/uvicorn app.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test: $(VENV)
	cd backend && ../$(PY) -m pytest tests -q

eval: $(VENV)
	cd backend && ../$(PY) evals/run_evals.py

check: test eval
	cd frontend && npm run typecheck

demo: setup ingest
	@echo ""
	@echo "  הכל מוכן. הרץ/י בשני טרמינלים:"
	@echo "    make api"
	@echo "    make web"
	@echo ""

clean:
	rm -rf $(VENV) frontend/node_modules frontend/dist backend/data/index.json
