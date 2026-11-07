# Model Selection & Orchestration Guidelines

This repository supports three major LLM providers interchangeably for autonomous financial reasoning and SEC filings analysis:

1. **Google Gemini** (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash`, `gemini-2.5-pro`)
2. **Anthropic Claude** (`claude-3-5-sonnet-latest`, `claude-3-7-sonnet-latest`, `claude-3-5-haiku-latest`)
3. **OpenAI GPT** (`gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`)

---

## 1. Selecting Models in Antigravity (`agy`)

When working with this project inside Antigravity:

* **Switch Model at Runtime**: Type `/model` in the chat to open the interactive model picker, or pass `--model` when starting `agy`:
  ```bash
  agy --model gemini-2.5-pro
  # or
  agy --dangerously-skip-permissions
  ```
* **Provider API Keys**: Stored in `/Users/aadipatel/AgentLens/.env`:
  * `GEMINI_API_KEY`: Active (Google AI Studio)
  * `ANTHROPIC_API_KEY`: For Claude models
  * `OPENAI_API_KEY`: For GPT-4o / o1 / o3-mini models
* **Subagent Dispatching**:
  * Fast scanning & initial code search: Use `flash_lite` or `flash`.
  * Deep reasoning & multi-step financial architecture: Use `pro` or `inherit`.

---

## 2. Selecting Models in AgentLens CLI

Run any query with your model of choice:
```bash
# Run with Gemini (Default)
uv run python scripts/run_query.py --provider gemini --model gemini-3.5-flash-lite "Compare NVDA and IBM"

# Run with Claude
uv run python scripts/run_query.py --provider anthropic --model claude-3-5-sonnet-latest "Compare NVDA and IBM"

# Run with OpenAI
uv run python scripts/run_query.py --provider openai --model gpt-4o-mini "Compare NVDA and IBM"

# Run Model Arena (Executes all 3 concurrently and shows side-by-side scorecard!)
uv run python scripts/run_query.py --arena "For fiscal year 2023, which company had higher total revenue: NVIDIA or IBM?"
```

---

## 3. Selecting Models in the Visual UI (`http://localhost:8000`)

The local visual dashboard at `http://localhost:8000` features a **Model Selector dropdown**:
* **✨ Gemini 3.5 Flash-Lite**: Fast, free-tier execution.
* **🧠 Claude 3.5 Sonnet**: Deep financial synthesis.
* **⚡ OpenAI GPT-4o-mini**: Precision structured analysis.
* **⚔️ Model Arena**: Executes all 3 models in parallel, displaying cross-model consensus, latency rankings, and overlaid datasets!
