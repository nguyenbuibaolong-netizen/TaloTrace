# Growtrics Chemistry Video Service — Final Implementation Plan

## Goal

Build the smallest reliable FastAPI backend that accepts one of three supported chemistry questions, processes it as an asynchronous video-generation job, and exposes a validated MP4 artifact.

The prototype prioritizes a clear job lifecycle, constrained but meaningful AI planning, deterministic educational visuals, observable failures, and measured provider usage over production infrastructure.

## Architecture

```text
FastAPI
   │
JobService ─── locked InMemoryJobRepository
   │
GenerationPipeline
   ├── ConceptRegistry → ConceptSpec
   ├── constrained LLM LessonPlan
   ├── Pillow scene renderer
   ├── per-scene gpt-4o-mini-tts
   └── per-scene FFmpeg composition
   │
ArtifactValidator ─── ffprobe + FFmpeg decode check
   │
LocalArtifactStore ─── atomic promotion
```

Required boundaries:

- `JobRepository` stores only job metadata and uses one small lock for atomic reads and updates.
- `JobService` owns lifecycle transitions and an optional single-generation semaphore.
- `GenerationPipeline` produces a temporary MP4 but never marks jobs complete or exposes final paths.
- `ArtifactStore` owns final artifact naming, atomic promotion, and retrieval.
- No queue framework, state-machine framework, event system, transaction abstraction, provider registry, MoviePy, or `imageio-ffmpeg`.

## API and Job Lifecycle

- `POST /videos`
  - Accept `{ "query": string }`.
  - Resolve the query through `ConceptRegistry`.
  - Return `422` for unsupported input without creating a job.
  - Create a `QUEUED` job and return `202` with a `Location` header.
- `GET /videos`
  - Return job summaries newest-first.
- `GET /videos/{job_id}`
  - Return the job or `404`.
- `GET /videos/{job_id}/artifact`
  - Return inline `video/mp4` only after completion.
  - Return `409` before completion or after failure.
  - Return `404` for an unknown job.

Job data includes ID, original query, canonical concept ID, status, current stage, timestamps, artifact URL, sanitized failure details, and optional completed-generation metrics.

```text
QUEUED
 → RUNNING/planning
 → RUNNING/rendering
 → RUNNING/narrating
 → RUNNING/composing
 → RUNNING/validating
 → COMPLETED

Any running stage → FAILED
```

Run generation as an in-process FastAPI background task. Use the repository lock for atomic updates and, if needed, one semaphore to prevent overlapping TTS/FFmpeg work. Jobs are intentionally non-durable, and the prototype runs with one Uvicorn worker.

## ConceptRegistry and Structured Planning

`content.py` defines a `ConceptRegistry` as the only entry point for supported concepts.

Normalization:

- Apply Unicode NFKC normalization and case-folding.
- Trim leading/trailing whitespace and collapse internal whitespace.
- Ignore terminal `?`, `.`, and `!`.
- Do not use fuzzy matching or let the LLM classify arbitrary queries.

Each concept declares explicit aliases, including its required challenge question and concise equivalents:

- `ph-scale`: “How does the pH scale work?”, “pH scale”, “how pH works”.
- `covalent-bonds`: “Why do atoms form covalent bonds?”, “covalent bonds”, “why atoms share electrons”.
- `ionic-vs-covalent`: “What is the difference between ionic and covalent bonding?”, “ionic vs covalent bonding”, “ionic and covalent bond differences”.

Unknown normalized input returns `422` with the three canonical supported questions.

Each registry entry contains a trusted `ConceptSpec`:

- Canonical concept ID and query.
- Explicit aliases.
- Required chemistry facts with stable IDs.
- Allowed visual primitives.
- Educational constraints and misconceptions to avoid.
- Complete canonical fallback `LessonPlan`.

The LLM may produce a structured `LessonPlan` with 3–5 ordered scenes containing trusted fact IDs, an allowed visual primitive, narration, concise labels, and emphasis references. It controls scene ordering, narration, labels, and emphasis, but cannot produce coordinates, rendering code, media prompts, URLs, arbitrary facts, unsupported visual types, or extra schema fields.

Planning flow:

1. Call `gpt-5.6-luna` with structured output and `none` or `low` reasoning.
2. Validate schema, fact coverage, primitive allow-list, scene count, text limits, educational constraints, and topic-specific chemistry invariants.
3. Retry exactly once with validation feedback.
4. Use the complete canonical `LessonPlan` after a second invalid response or unrecoverable planner failure.
5. Record both attempts and fallback use in generation metrics.

## Rendering, Per-Scene Composition, and Artifacts

For every scene:

1. Render a deterministic 1280×720 PNG with Pillow.
2. Generate that scene’s narration using the verified `gpt-4o-mini-tts` SDK path.
3. Inspect the audio duration with ffprobe.
4. Invoke system FFmpeg with an argument array and `shell=False` to create a scene MP4:
   - Loop the still image for the measured audio duration.
   - Encode consistent H.264 video, AAC audio, fixed frame rate, and `yuv420p`.
   - Apply only subtle fade-in/fade-out treatment.
   - Use bounded subprocess timeouts and capture diagnostic stderr.
5. Encode all scene segments with identical stream settings.
6. Concatenate the segments with FFmpeg into one temporary MP4.

Validate the combined artifact with ffprobe JSON, requiring video and audio streams, 1280×720 dimensions, positive plausible duration, parseable metadata, and a non-empty file. Then run an FFmpeg decode smoke check.

Only after composition, ffprobe validation, decode validation, and atomic promotion through `os.replace` may `JobService` mark the job `COMPLETED`. Use system FFmpeg and ffprobe exclusively; do not introduce an alternate media stack.

## Visual Quality

- Use a consistent color-blind-conscious palette with stable semantic colors across scenes.
- Maintain at least 64 px safe margins and high text/background contrast.
- Use no more than two font families, with approximately 48 px or larger titles and 32 px or larger explanatory text.
- Limit each scene to one teaching objective, one dominant diagram, and no more than three short labels.
- Measure text wrapping and bounds so text never clips or overlaps diagrams.
- Prefer a title → diagram → key-takeaway hierarchy.
- Use gradients, electron pairs, charges, arrows, side-by-side comparisons, and concise recap cards instead of paragraphs.
- Apply emphasis through renderer-controlled color, weight, outline, or scale; the LLM never controls raw styling values.
- Avoid decorative animation, stock media, generative images, excessive transitions, and dense equations.
- Manually inspect final samples for correctness, readability, pacing, synchronization, and usefulness without narration.

## Usage Metrics and Cost Explanation

Capture actual `GenerationMetrics`:

- Planner model, reasoning setting, request/retry counts, fallback use, input tokens, and output tokens.
- TTS model, request/retry counts, input characters, available token usage, and generated audio seconds.
- Scene count and individual scene durations.
- FFmpeg and total generation wall time.
- Final artifact duration and byte size.
- Validation and decode-check results.

Persist metrics with the in-memory job and include them in completed job responses. Copy measured metrics for the three committed samples into `samples/manifest.json`.

Do not hard-code or promise a per-video price. The README explains how measured usage can be multiplied by a dated pricing snapshot, distinguishing provider cost from local CPU and storage.

## File Structure

```text
TaloTrace/
├── app/
│   ├── main.py
│   ├── models.py
│   ├── jobs.py
│   ├── generation.py
│   ├── content.py
│   └── artifacts.py
├── tests/
│   ├── test_api.py
│   └── test_generation.py
├── samples/
│   ├── manifest.json
│   ├── ph-scale.mp4
│   ├── covalent-bonds.mp4
│   └── ionic-vs-covalent.mp4
├── README.md
├── ARCHITECTURE.md
├── PLAN.md
└── .gitignore
```

## Tests

Must-have tests:

1. `POST /videos` returns `202`, including when using a registered alias.
2. Unsupported queries return `422`.
3. Unknown jobs return `404`.
4. Artifact requests before completion return `409`.
5. A successful lifecycle reaches `COMPLETED`.
6. Generator failure reaches `FAILED` with an observable sanitized error.
7. Invalid planner output triggers exactly one retry.
8. Repeated invalid output activates the canonical fallback.
9. A job cannot become `COMPLETED` before validation and atomic promotion succeed.

Use fake planner, TTS, FFmpeg, and probe boundaries in automated tests. Deeper cleanup, simultaneous-request, timeout, and corrupted-intermediate tests are optional if time is constrained.

## Implementation Order

1. Update this plan, verify FFmpeg/ffprobe, and define models.
2. Implement `ConceptRegistry`, normalization, aliases, `ConceptSpec` data, and canonical plans.
3. Implement the locked repository, lifecycle service, API routes, and basic API tests.
4. Implement structured planning, validation, retry/fallback behavior, usage capture, and planner tests.
5. Implement Pillow visuals, per-scene TTS, per-scene FFmpeg composition, and concatenation.
6. Implement ffprobe validation, decode checking, atomic promotion, and completion-ordering tests.
7. Run all three concepts end-to-end, inspect the results, and commit the best artifacts with actual manifest metrics.
8. Finish README and architecture documentation.

If time becomes constrained, preserve the must-have tests, canonical fallbacks, artifact validation, and visual legibility. Deeper cleanup and unusual concurrency tests are optional.
