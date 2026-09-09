# Architecture

## Request and lifecycle flow

```text
FastAPI -> JobService -> GenerationPipeline -> LocalArtifactStore
              |
              +-> locked InMemoryJobRepository
```

FastAPI validates the request shape and exposes create, list, status, and artifact endpoints. `JobService` is the sole lifecycle orchestrator: it resolves the concept before creating a `queued` job, runs generation in a FastAPI background task, advances explicit processing stages, and records either `completed` or `failed`. A small semaphore serializes generation. The repository owns job records and protects reads and writes with one lock.

This is deliberately a prototype: state is held in memory and is lost on restart, while in-process background execution is unsuitable for durable production workloads. It is intended for a single Uvicorn worker.

## Trusted content and planning

`ConceptRegistry` normalizes an explicit set of queries and aliases. It resolves only three supported concepts: pH scale, covalent bonds, and ionic versus covalent bonding. Each trusted `ConceptSpec` defines required chemistry facts, allowed visual primitives, educational constraints, and a complete canonical `LessonPlan`.

The planner asks `gpt-5.6-luna` for a structured 3–5 scene lesson. Model output is untrusted: strict schema checks reject extra fields, unknown fact IDs, disallowed primitives, invalid text bounds, and missing required-fact coverage. Invalid output receives one retry with validation feedback; a second invalid result or provider failure uses the concept's canonical fallback. Planning request counts, retries, fallback use, and available token usage are captured.

## Media pipeline

For each planned scene, the pipeline:

1. Renders one deterministic 1280x720 PNG with Pillow from the selected visual primitive.
2. Requests spoken narration from `gpt-4o-mini-tts`, retrying a failed scene at most once.
3. Measures narration duration with ffprobe.
4. Uses a direct FFmpeg subprocess to encode a consistent H.264/AAC scene MP4.

FFmpeg concatenates the scene segments into a temporary final MP4. `ArtifactValidator` then requires a non-trivial file, video and audio streams, 1280x720 dimensions, and positive duration from ffprobe, followed by a complete FFmpeg decode smoke check. Only after both checks pass does `LocalArtifactStore` atomically promote the temporary file with `os.replace`. The pipeline then returns, allowing `JobService` to mark the job completed. A validation or promotion failure cannot expose an artifact as completed.

Deterministic diagrams were chosen over generative video for predictable chemistry visuals, repeatability, lower cost, and a stronger reliability floor. Technical media validation confirms encoding and stream integrity, but it does not guarantee semantic correctness, narration quality, synchronization quality, or educational effectiveness; the submission samples were therefore also inspected manually.

## Failures and observability

Jobs expose their current processing stage. Any planner, renderer, TTS, FFmpeg, probe, decode, or promotion exception moves the running job to `failed` with a stable error code, sanitized learner-safe message, and failed stage. Completed jobs include measured provider usage, scene timing, total timing, artifact size and duration, and validation flags. Detailed provider and subprocess errors are not returned through the public API.

## Production evolution

A production version would replace the in-memory repository and background task with durable database records and an external queue/worker, move artifacts to object storage, add authenticated access and retention policies, enforce provider/time budgets, and add structured logs, tracing, retries with backoff, idempotency, and operational monitoring. Broader topic support would require reviewed `ConceptSpec` content and renderer primitives rather than accepting arbitrary LLM-generated facts or visuals.
