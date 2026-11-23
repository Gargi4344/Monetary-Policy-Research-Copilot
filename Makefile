# Monetary Policy Research Copilot. No Docker needed.
# Needs: uv, pnpm (Node 22+), Ollama, GNU make (Windows: Git Bash or `winget install ezwinports.make`).
#
# Fresh clone:  make setup && make models && make data && make dev   -> http://localhost:3000

UV ?= uv
BE := cd backend &&
FE := cd frontend &&
CP := $(BE) $(UV) run copilot

.PHONY: help setup models data scrape index tone eval results dev api web test lint format fixtures clean

help:          ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:         ## Install Python and web dependencies, create .env, install pre-commit hooks
	@test -f .env || cp .env.example .env
	$(BE) $(UV) sync
	$(FE) pnpm install
	$(BE) $(UV) run pre-commit install || true

models:        ## Pull the local LLM into Ollama (~4.7 GB)
	ollama pull qwen2.5:7b

data:          ## Market data + scrape + chunk + index (~15 minutes; cached after the first run)
	$(CP) market
	$(CP) scrape
	$(CP) chunk
	$(CP) index

tone:          ## Label announcement sentences, build the tone index, test next-decision prediction
	$(CP) tone

eval:          ## 30-question evaluation + guardrail suite (writes data/eval/latest.json)
	$(CP) eval

results:       ## Regenerate RESULTS.md and the README results and highlights from this run
	$(CP) results

api:           ## FastAPI on :8000 with reload
	$(BE) $(UV) run uvicorn copilot.api.main:app --reload --port 8000

web:           ## Next.js on :3000
	$(FE) pnpm dev

dev:           ## API (:8000) and web app (:3000) together
	@echo "API http://localhost:8000/docs   Web http://localhost:3000   (Ctrl+C stops both)"
	$(BE) $(UV) run uvicorn copilot.api.main:app --port 8000 & \
	cd frontend && pnpm dev

test:          ## pytest (fake LLM, fixtures; no network or model) and the web lint/build
	$(BE) $(UV) run pytest -q
	$(FE) pnpm lint

fixtures:      ## The whole pipeline on committed fixtures (what CI runs)
	$(BE) $(UV) run python ../scripts/pipeline_fixtures.py

lint:          ## ruff + eslint
	$(BE) $(UV) run ruff check . && $(UV) run ruff format --check .
	$(FE) pnpm lint

format:        ## Auto-format Python
	$(BE) $(UV) run ruff check --fix . && $(UV) run ruff format .

clean:         ## Delete downloaded data, indexes and results (keeps fixtures)
	rm -rf data
