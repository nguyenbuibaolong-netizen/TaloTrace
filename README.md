# Chemistry Video Request Service

A compact FastAPI prototype that accepts one of three chemistry questions, creates an asynchronous generation job, and returns a validated narrated MP4. The service uses constrained AI lesson planning, deterministic Pillow diagrams, scene-by-scene speech, and system FFmpeg composition.

## Setup

Requirements:

- Python 3.11+
- `ffmpeg` and `ffprobe` installed and available on `PATH`
- An OpenAI API key

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
python -m pip install fastapi uvicorn openai pillow pydantic pytest httpx
```

Set the API key before starting the service:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

```bash
export OPENAI_API_KEY="your-api-key"
```

Run one application worker from the repository root:

```bash
uvicorn app.main:app --workers 1
```

## API

- `POST /videos` creates a job and returns `202 Accepted` with a `Location` header.
- `GET /videos` lists jobs newest-first.
- `GET /videos/{job_id}` returns job status, stage, errors, artifact URL, and completed metrics.
- `GET /videos/{job_id}/artifact` streams the MP4 after completion; unavailable artifacts return `409`.

Example request:

```bash
curl -i -X POST http://127.0.0.1:8000/videos \
  -H "Content-Type: application/json" \
  -d '{"query":"How does the pH scale work?"}'
```

Poll the URL in the response's `Location` header until `status` is `completed`, then request the returned `artifact_url`. If generation fails, the job remains observable as `failed` with a sanitized error and the stage where processing stopped.

Supported challenge queries:

- `How does the pH scale work?`
- `Why do atoms form covalent bonds?`
- `What is the difference between ionic and covalent bonding?`

Explicit normalized aliases are also accepted; arbitrary chemistry questions are not.

## Tests and samples

Run the complete test suite with:

```bash
python -m pytest -q
```

Generated submission samples are in [`samples/`](samples/):

- [`ph-scale.mp4`](samples/ph-scale.mp4)
- [`covalent-bonds.mp4`](samples/covalent-bonds.mp4)
- [`ionic-vs-covalent.mp4`](samples/ionic-vs-covalent.mp4)
- [`manifest.json`](samples/manifest.json) contains measured planner, narration, duration, size, and validation data.

## Cost efficiency

The LLM plans a short structured lesson but does not generate images or video. Pillow renders deterministic visuals locally, and FFmpeg performs local composition. Speech is requested once per scene. Actual planner tokens, TTS characters, request counts, and artifact metrics are recorded so cost can be calculated against current provider pricing; no fixed per-video price is assumed.

## Limitations

- Job state is in memory and is lost when the process restarts.
- FastAPI in-process background execution is prototype-only and assumes one Uvicorn worker.
- Only the three challenge concepts above are supported.
- The artifact store is local filesystem storage.
- Technical ffprobe and decode validation verifies media integrity, not semantic correctness or teaching quality.
- TTS and lesson planning require network access and may fail due to provider availability.
- Deterministic visuals were chosen over generative video for cost, reproducibility, and reliability.
