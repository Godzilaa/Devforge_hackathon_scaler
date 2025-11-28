# Devforge Hackathon — CodeIDE AI Agent

This repository contains a lightweight web-based development environment (CodeIDE) augmented with AI-assisted code analysis and automated patch generation using a local Ollama runtime. It includes:

- A Next.js frontend with Monaco editor (`app/`) — file explorer, tabs, terminal, and an AI Logger UI.
- A FastAPI backend (`api/`) that executes code in a sandbox, analyzes errors, and generates patches via Ollama.
- AI patching flow that can automatically detect logic issues (tree traversals, Fibonacci memoization) and propose fixes.

This README explains installation, recommended models, features, and how to run the system locally on Windows (PowerShell) as well as useful troubleshooting tips.

**Important:** The repository previously included a `Dockerfile` under `api/`, but it has been removed per project preferences.

---

## Table of Contents

- Prerequisites
- Recommended Ollama Models
- Install & Run (Backend)
- Install & Run (Frontend)
- Development Workflow (Run / Fix / Apply Patch)
- API Reference (important endpoints)
- Heuristics & Fallbacks
- Testing
- Troubleshooting
- Contributing

---

## Prerequisites

On Windows (PowerShell):

- Node.js (16+ recommended) and `npm` or `pnpm` for frontend.
- Python 3.10+ and `pip` for backend dependencies.
- Ollama (local LLM runtime) installed and accessible from PATH. See: https://ollama.ai/docs

Ensure `ollama` is available in your PATH and you can run `ollama serve` from a PowerShell terminal.

## Recommended Ollama Models

The backend uses a local Ollama model to generate code fixes. You may choose any of the following (recommended):

- `qwen2.5-coder:1.5b` — another capable code model and used as a default in some configurations.

Notes:

- Models can be pulled with `ollama pull <model>` (may require disk space and time).
- The backend will attempt to `ollama pull` the configured model if it is missing and the Ollama CLI is present.
- If the configured model is not available, the backend will try to fall back to any local model reported by Ollama.

## Install & Run — Backend (API)

1. Open PowerShell and install Python dependencies:

```powershell
cd .\api
python -m pip install -r requirements.txt
```

2. (Optional) Pull your preferred Ollama model:

```powershell
# or
ollama pull qwen2.5-coder:1.5b
```

3. Start Ollama (in a separate terminal):

```powershell
ollama serve
```

4. Start the FastAPI backend:

```powershell
cd .\api
python -m uvicorn main:app --reload --port 8000
```

The backend exposes REST endpoints at `http://localhost:8000` (the UI will call these).

## Install & Run — Frontend (Next.js)

1. Install frontend dependencies and run the dev server:

```powershell
cd ..\  # repo root
npm install
npm run dev
# or using pnpm: pnpm install && pnpm dev
```

2. Open the app in your browser at `http://localhost:3000`.

## Development Workflow

UI Overview:

- File Explorer: create/open files.
- Tabs: edit files in Monaco editor.
- Run button: executes the active file via `POST /api/execute` (sandboxed) and prints stdout/stderr to the built-in Terminal.
- Fix (AI) button: sends file contents to the AI continuous fixer (`/api/fix-continuously`) which streams logs and patch proposals.
- Patch Review Dialog: When the AI proposes a patch, the UI shows an interactive review dialog with:
	- A short explanation of the fix.
	- A preview of execution output when the patched code is run in the sandbox.
	- A compact "Changed Lines Preview" that shows the exact original lines and their replacement.
	- An Apply button that opens a small confirmation modal listing the edited lines; confirming applies the patch and runs it in the sandbox.

Server-side flow summary:

1. The backend executes the provided code in an isolated temporary directory using `SandboxExecutor.execute_code()`.
2. It parses stderr for syntax/traceback errors and also runs additional heuristic detectors for logic errors (e.g., tree traversal output mismatch and Fibonacci memoization issues).
3. If errors are deemed auto-fixable, the system attempts to generate a patch using Ollama. If Ollama is unavailable or fails, lightweight heuristic fixes are attempted.
4. The backend returns structured `FilePatch` objects which include `line_edits`, `original_content`, and `patched_content`. The frontend uses these to show precise changed-line previews.

## API Reference (important endpoints)

- `POST /api/execute` — Execute code in sandbox. Request body: `{ code, language, file_path }`. Response includes `stdout`, `stderr`, `exit_code`, `execution_time`, `success`.
- `POST /api/analyze` — Execute + analyze + (optionally) return patches and analysis.
- `POST /api/analyze-stream` — Streamed analyze endpoint returning NLJSON events for progress.
- `POST /api/fix-continuously` — Streams logs and one-or-more `patch` events as patches are produced. The frontend pauses on each patch for user review.
- `POST /api/session/{session_id}` & `GET /api/session/{session_id}` — Session lifecycle (info & apply may be supported).

Refer to `api/main.py` for full model of returned JSON objects (`ExecutionResult`, `FilePatch`, etc.).

## Heuristics & Fallbacks

- Ollama resilience: `OllamaClient` in `api/main.py` attempts to detect when the local Ollama runner crashes, will try to restart it (on Windows it calls `ollama serve`), and will retry generation with reduced prompt sizes and settings.
- Heuristic fixes: For common and well-known issues (preorder traversal append location, Fibonacci memoization errors, mutable default dicts), the server includes deterministic string-based fixes when LLM assistance is unavailable.

## Tests

There are helper scripts under `scripts/` (if present) for unit tests such as tree traversal and fibonacci examples. You can run the backend's python checks with:

```powershell
cd .\api
python -m py_compile main.py
```

Add or run test scripts as you need for validation.

## Troubleshooting

- Ollama not responding: ensure `ollama serve` is running and accessible at `http://localhost:11434`. Restart it and re-run the backend.
- Model pull failures: `ollama pull <model>` may fail due to disk space or network. Use a smaller local model if needed.
- Ports in use: the API runs on port `8000` and frontend on `3000`. Kill conflicting processes or change ports.

## Security & Sandbox Notes

- Code execution runs locally inside temporary directories created by the backend. It runs the language runtime (Python/Node) directly — treat this as untrusted execution. Do not run this server in production without additional sandboxing (containers, time/resource limits, seccomp, user restrictions).

## Contributing

If you plan to add features or refine the AI fixer:

1. Improve detectors in `api/main.py` (e.g., robust AST-based checks rather than regex heuristics).
2. Improve patch generation prompts and error handling in `AICodeFixer`.
3. Add tests to `scripts/` and CI validation to ensure the LLM fallback heuristics and streaming endpoints behave as expected.

---

If you want, I can:

- Run the frontend typecheck and fix TypeScript `implicit-any` warnings.
- Start the backend and frontend here and exercise a sample file run + patch flow.
- Reintroduce a Docker setup if you prefer containerized development (I removed the `Dockerfile` per your request but can provide a new multi-stage Dockerfile if needed).

Thank you — let me know which next step you want me to take.
