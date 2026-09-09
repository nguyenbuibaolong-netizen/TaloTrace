"""Lesson planning and media generation boundary.

OpenAI, Pillow, TTS, and FFmpeg integration are intentionally deferred.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import UUID

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from app.artifacts import ArtifactValidator, LocalArtifactStore
from app.models import ConceptSpec, GenerationMetrics, GenerationStage
from app.models import LessonPlan, LessonScene, VisualPrimitive


StageReporter = Callable[[GenerationStage], None]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Output returned after a generator has finished its full responsibility."""

    artifact_url: str | None = None
    metrics: GenerationMetrics | None = None


@dataclass(frozen=True, slots=True)
class PlanningResult:
    plan: LessonPlan
    metrics: GenerationMetrics


class LessonPlanValidationError(ValueError):
    """Raised when a schema-valid lesson plan violates its ConceptSpec."""


def validate_lesson_plan(plan: LessonPlan, concept: ConceptSpec) -> None:
    """Validate concept-specific facts, coverage, and visual primitives."""

    valid_fact_ids = {fact.id for fact in concept.required_facts}
    used_fact_ids = {
        fact_id for scene in plan.scenes for fact_id in scene.fact_ids
    }
    unknown_fact_ids = used_fact_ids - valid_fact_ids
    if unknown_fact_ids:
        raise LessonPlanValidationError(
            f"unknown fact IDs: {', '.join(sorted(unknown_fact_ids))}"
        )

    missing_fact_ids = valid_fact_ids - used_fact_ids
    if missing_fact_ids:
        raise LessonPlanValidationError(
            f"missing required fact IDs: {', '.join(sorted(missing_fact_ids))}"
        )

    unsupported_primitives = {
        scene.visual_primitive for scene in plan.scenes
    } - concept.allowed_visual_primitives
    if unsupported_primitives:
        names = sorted(primitive.value for primitive in unsupported_primitives)
        raise LessonPlanValidationError(
            f"unsupported visual primitives: {', '.join(names)}"
        )


class OpenAILessonPlanner:
    """Create a constrained lesson plan with one retry and canonical fallback."""

    MODEL = "gpt-5.6-luna"
    REASONING_EFFORT = "low"
    MAX_REQUESTS = 2

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def plan(self, concept: ConceptSpec) -> PlanningResult:
        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        validation_feedback: str | None = None

        while requests < self.MAX_REQUESTS:
            requests += 1
            try:
                if self._client is None:
                    self._client = OpenAI()
                response = self._client.responses.parse(
                    model=self.MODEL,
                    reasoning={"effort": self.REASONING_EFFORT},
                    instructions=self._instructions(),
                    input=self._prompt(concept, validation_feedback),
                    text_format=LessonPlan,
                    max_output_tokens=2_000,
                    store=False,
                    timeout=30.0,
                )
                input_tokens = self._add_usage(
                    input_tokens, getattr(response.usage, "input_tokens", None)
                    if response.usage is not None
                    else None
                )
                output_tokens = self._add_usage(
                    output_tokens, getattr(response.usage, "output_tokens", None)
                    if response.usage is not None
                    else None
                )
                parsed = LessonPlan.model_validate(response.output_parsed)
                validate_lesson_plan(parsed, concept)
            except (ValidationError, LessonPlanValidationError, ValueError) as exc:
                validation_feedback = self._feedback(exc)
                continue
            except Exception:
                validation_feedback = "The provider request failed. Produce a valid plan."
                continue

            return PlanningResult(
                plan=parsed,
                metrics=self._metrics(
                    requests=requests,
                    fallback_used=False,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        return PlanningResult(
            plan=concept.canonical_plan.model_copy(deep=True),
            metrics=self._metrics(
                requests=requests,
                fallback_used=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    @staticmethod
    def _instructions() -> str:
        return (
            "Create a short educational chemistry lesson plan using only the supplied "
            "facts and visual primitives. Do not add facts, coordinates, rendering code, "
            "media prompts, URLs, or unsupported fields. Keep labels concise and use "
            "emphasis only for labels present in the same scene."
        )

    @staticmethod
    def _prompt(concept: ConceptSpec, feedback: str | None) -> str:
        trusted_context = {
            "concept_id": concept.concept_id.value,
            "learner_query": concept.canonical_query,
            "required_facts": [fact.model_dump() for fact in concept.required_facts],
            "allowed_visual_primitives": sorted(
                primitive.value for primitive in concept.allowed_visual_primitives
            ),
            "educational_constraints": list(concept.educational_constraints),
        }
        prompt = "Trusted lesson constraints:\n" + json.dumps(
            trusted_context, ensure_ascii=False, indent=2
        )
        if feedback:
            prompt += "\n\nThe previous result was invalid. Correct these issues:\n" + feedback
        return prompt

    @staticmethod
    def _feedback(error: Exception) -> str:
        return str(error)[:1_500]

    @staticmethod
    def _add_usage(total: int | None, value: int | None) -> int | None:
        if value is None:
            return total
        return (total or 0) + value

    def _metrics(
        self,
        *,
        requests: int,
        fallback_used: bool,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> GenerationMetrics:
        return GenerationMetrics(
            planner_model=self.MODEL,
            planner_reasoning=self.REASONING_EFFORT,
            planner_requests=requests,
            planner_retries=max(0, requests - 1),
            planner_input_tokens=input_tokens,
            planner_output_tokens=output_tokens,
            fallback_used=fallback_used,
        )


class PillowSceneRenderer:
    WIDTH = 1280
    HEIGHT = 720
    BACKGROUND = "#0B1220"
    PANEL = "#172033"
    TEXT = "#F8FAFC"
    MUTED = "#B8C4D8"
    ACCENT = "#4FD1C5"
    EMPHASIS = "#FBBF24"
    ACID = "#FB7185"
    BASE = "#60A5FA"

    def render(self, plan: LessonPlan, output_dir: Path) -> list[Path]:
        output: list[Path] = []
        for index, scene in enumerate(plan.scenes):
            image = Image.new("RGB", (self.WIDTH, self.HEIGHT), self.BACKGROUND)
            draw = ImageDraw.Draw(image)
            self._header(draw, plan.title, index, len(plan.scenes))
            self._draw_primitive(draw, scene)
            self._labels(draw, scene)
            path = output_dir / f"scene_{index:02d}.png"
            image.save(path, format="PNG")
            output.append(path)
        return output

    def _header(
        self, draw: ImageDraw.ImageDraw, title: str, index: int, total: int
    ) -> None:
        self._wrapped_text(
            draw,
            title,
            (64, 32, 1110, 118),
            self._font(44, bold=True),
            self.TEXT,
            max_lines=2,
        )
        draw.text(
            (1175, 54),
            f"{index + 1}/{total}",
            font=self._font(26, bold=True),
            fill=self.MUTED,
            anchor="mm",
        )
        draw.line((64, 132, 1216, 132), fill="#334155", width=2)

    def _draw_primitive(
        self, draw: ImageDraw.ImageDraw, scene: LessonScene
    ) -> None:
        primitive = scene.visual_primitive
        if primitive is VisualPrimitive.PH_SCALE:
            self._ph_scale(draw)
        elif primitive is VisualPrimitive.PH_EXAMPLES:
            self._ph_examples(draw)
        elif primitive is VisualPrimitive.VALENCE_SHELL:
            self._valence_shell(draw)
        elif primitive is VisualPrimitive.ELECTRON_SHARING:
            self._electron_sharing(draw)
        elif primitive is VisualPrimitive.ELECTRON_TRANSFER:
            self._electron_transfer(draw)
        elif primitive is VisualPrimitive.BOND_COMPARISON:
            self._bond_comparison(draw)
        elif primitive is VisualPrimitive.SUMMARY_CARD:
            self._summary(draw, scene)
        else:
            self._title_card(draw, scene)

    def _title_card(self, draw: ImageDraw.ImageDraw, scene: LessonScene) -> None:
        draw.ellipse((490, 190, 650, 350), fill="#2563EB", outline=self.TEXT, width=4)
        draw.ellipse((630, 290, 790, 450), fill="#7C3AED", outline=self.TEXT, width=4)
        draw.line((625, 315, 655, 325), fill=self.ACCENT, width=12)
        draw.text(
            (640, 495),
            scene.labels[0] if scene.labels else "Chemistry, visualized",
            font=self._fit_font(draw, scene.labels[0] if scene.labels else "Chemistry, visualized", 620, 42),
            fill=self.TEXT,
            anchor="mm",
        )

    def _ph_scale(self, draw: ImageDraw.ImageDraw) -> None:
        start_x, end_x, y = 105, 1175, 355
        step = (end_x - start_x) / 14
        for value in range(15):
            x = start_x + value * step
            color = self.ACID if value < 7 else self.EMPHASIS if value == 7 else self.BASE
            draw.line((x, y - 28, x, y + 28), fill=color, width=5)
            draw.text(
                (x, y + 52), str(value), font=self._font(24, bold=True), fill=self.TEXT, anchor="mm"
            )
        draw.line((start_x, y, end_x, y), fill=self.MUTED, width=5)
        self._category(draw, (105, 185, 540, 265), "ACIDIC", self.ACID)
        self._category(draw, (555, 185, 725, 265), "NEUTRAL", self.EMPHASIS)
        self._category(draw, (740, 185, 1175, 265), "BASIC", self.BASE)
        draw.ellipse((start_x + 7 * step - 24, y - 24, start_x + 7 * step + 24, y + 24), fill=self.EMPHASIS)
        draw.text((640, 500), "Each step = 10× change", font=self._font(38, bold=True), fill=self.ACCENT, anchor="mm")

    def _ph_examples(self, draw: ImageDraw.ImageDraw) -> None:
        examples = (
            ("Lemon juice", "≈ pH 2", self.ACID),
            ("Pure water", "pH 7", self.EMPHASIS),
            ("Soapy water", "≈ pH 10", self.BASE),
        )
        for index, (name, value, color) in enumerate(examples):
            left = 100 + index * 390
            draw.rounded_rectangle((left, 190, left + 340, 500), radius=28, fill=self.PANEL, outline=color, width=5)
            draw.ellipse((left + 105, 225, left + 235, 355), fill=color)
            draw.text((left + 170, 400), name, font=self._font(30, bold=True), fill=self.TEXT, anchor="mm")
            draw.text((left + 170, 455), value, font=self._font(32, bold=True), fill=color, anchor="mm")

    def _valence_shell(self, draw: ImageDraw.ImageDraw) -> None:
        for center_x, symbol in ((430, "H"), (850, "H")):
            draw.ellipse((center_x - 145, 205, center_x + 145, 495), outline="#64748B", width=4)
            draw.ellipse((center_x - 66, 284, center_x + 66, 416), fill="#2563EB", outline=self.TEXT, width=4)
            draw.text((center_x, 350), symbol, font=self._font(56, bold=True), fill=self.TEXT, anchor="mm")
            electron_x = center_x + (135 if center_x < 640 else -135)
            draw.ellipse((electron_x - 14, 336, electron_x + 14, 364), fill=self.EMPHASIS)
        draw.text((640, 180), "Valence electrons sit in the outer region", font=self._font(34, bold=True), fill=self.ACCENT, anchor="mm")

    def _electron_sharing(self, draw: ImageDraw.ImageDraw) -> None:
        for center_x, symbol in ((455, "H"), (825, "H")):
            draw.ellipse((center_x - 115, 245, center_x + 115, 475), fill="#1D4ED8", outline=self.TEXT, width=4)
            draw.text((center_x, 360), symbol, font=self._font(60, bold=True), fill=self.TEXT, anchor="mm")
        draw.line((570, 360, 710, 360), fill=self.ACCENT, width=8)
        for x in (620, 660):
            draw.ellipse((x - 14, 346, x + 14, 374), fill=self.EMPHASIS, outline=self.TEXT, width=2)
        draw.text((640, 205), "Shared electron pair", font=self._font(40, bold=True), fill=self.EMPHASIS, anchor="mm")
        draw.text((640, 515), "Both nuclei attract the shared pair", font=self._font(32, bold=True), fill=self.ACCENT, anchor="mm")

    def _electron_transfer(self, draw: ImageDraw.ImageDraw) -> None:
        self._ion(draw, 350, "Na", "+", "#2563EB")
        self._ion(draw, 930, "Cl", "−", "#7C3AED")
        draw.line((500, 320, 770, 320), fill=self.EMPHASIS, width=8)
        draw.polygon(((770, 320), (730, 292), (730, 348)), fill=self.EMPHASIS)
        draw.ellipse((620, 290, 650, 320), fill=self.EMPHASIS)
        draw.text((635, 250), "electron transfer", font=self._font(34, bold=True), fill=self.EMPHASIS, anchor="mm")
        draw.text((640, 505), "Opposite charges attract", font=self._font(38, bold=True), fill=self.ACCENT, anchor="mm")

    def _bond_comparison(self, draw: ImageDraw.ImageDraw) -> None:
        draw.rounded_rectangle((75, 170, 610, 555), radius=28, fill=self.PANEL, outline=self.ACID, width=4)
        draw.rounded_rectangle((670, 170, 1205, 555), radius=28, fill=self.PANEL, outline=self.BASE, width=4)
        draw.text((342, 215), "IONIC", font=self._font(38, bold=True), fill=self.ACID, anchor="mm")
        draw.text((938, 215), "COVALENT", font=self._font(38, bold=True), fill=self.BASE, anchor="mm")
        draw.text((342, 300), "e⁻  →", font=self._font(64, bold=True), fill=self.EMPHASIS, anchor="mm")
        draw.text((938, 300), "•  •", font=self._font(64, bold=True), fill=self.EMPHASIS, anchor="mm")
        draw.text((342, 390), "transfer", font=self._font(38, bold=True), fill=self.TEXT, anchor="mm")
        draw.text((938, 390), "sharing", font=self._font(38, bold=True), fill=self.TEXT, anchor="mm")
        draw.text((342, 470), "+     −", font=self._font(50, bold=True), fill=self.ACCENT, anchor="mm")
        draw.text((938, 470), "atom — atom", font=self._font(36, bold=True), fill=self.ACCENT, anchor="mm")

    def _summary(self, draw: ImageDraw.ImageDraw, scene: LessonScene) -> None:
        labels = scene.labels or ("Observe", "Connect", "Remember")
        for index, label in enumerate(labels):
            y = 195 + index * 120
            color = self.EMPHASIS if label in scene.emphasis else self.ACCENT
            draw.ellipse((150, y, 210, y + 60), fill=color)
            draw.text((180, y + 30), "✓", font=self._font(32, bold=True), fill=self.BACKGROUND, anchor="mm")
            draw.text((245, y + 30), label, font=self._fit_font(draw, label, 850, 38), fill=self.TEXT, anchor="lm")

    def _labels(self, draw: ImageDraw.ImageDraw, scene: LessonScene) -> None:
        if not scene.labels or scene.visual_primitive in {VisualPrimitive.SUMMARY_CARD, VisualPrimitive.TITLE_CARD}:
            return
        count = len(scene.labels)
        width = min(340, (1100 - (count - 1) * 24) // count)
        total = count * width + (count - 1) * 24
        start = (self.WIDTH - total) // 2
        for index, label in enumerate(scene.labels):
            left = start + index * (width + 24)
            emphasized = label in scene.emphasis
            draw.rounded_rectangle(
                (left, 600, left + width, 660),
                radius=20,
                fill="#2A354A" if not emphasized else "#4B3B12",
                outline=self.EMPHASIS if emphasized else "#475569",
                width=3,
            )
            draw.text(
                (left + width / 2, 630),
                label,
                font=self._fit_font(draw, label, width - 28, 27),
                fill=self.TEXT,
                anchor="mm",
            )

    def _category(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, color: str) -> None:
        draw.rounded_rectangle(box, radius=22, fill=self.PANEL, outline=color, width=4)
        draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font=self._font(30, bold=True), fill=color, anchor="mm")

    def _ion(self, draw: ImageDraw.ImageDraw, center_x: int, symbol: str, charge: str, color: str) -> None:
        draw.ellipse((center_x - 115, 245, center_x + 115, 475), fill=color, outline=self.TEXT, width=4)
        draw.text((center_x, 360), symbol, font=self._font(58, bold=True), fill=self.TEXT, anchor="mm")
        draw.text((center_x + 90, 265), charge, font=self._font(44, bold=True), fill=self.EMPHASIS, anchor="mm")

    @staticmethod
    def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        names = ("C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf")
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _fit_font(self, draw: ImageDraw.ImageDraw, text: str, max_width: float, start_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        size = start_size
        while size > 18:
            font = self._font(size, bold=True)
            if draw.textlength(text, font=font) <= max_width:
                return font
            size -= 2
        return self._font(18, bold=True)

    @staticmethod
    def _wrapped_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        box: tuple[int, int, int, int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        fill: str,
        *,
        max_lines: int,
    ) -> None:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= box[2] - box[0]:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        lines = lines[:max_lines]
        line_height = (box[3] - box[1]) / max(1, len(lines))
        for index, line in enumerate(lines):
            draw.text((box[0], box[1] + index * line_height), line, font=font, fill=fill)


@dataclass(frozen=True, slots=True)
class NarrationResult:
    requests: int
    retries: int
    input_characters: int


class TTSGenerationError(RuntimeError):
    pass


class OpenAITTS:
    MODEL = "gpt-4o-mini-tts"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def synthesize(self, narration: str, output_path: Path) -> NarrationResult:
        for attempt in range(2):
            try:
                if self._client is None:
                    self._client = OpenAI()
                with self._client.audio.speech.with_streaming_response.create(
                    model=self.MODEL,
                    voice="coral",
                    input=narration,
                    instructions="Speak clearly and warmly like a concise science teacher.",
                    response_format="mp3",
                    timeout=45.0,
                ) as response:
                    response.stream_to_file(output_path)
                if not output_path.is_file() or output_path.stat().st_size < 1_000:
                    raise TTSGenerationError("TTS returned an empty audio artifact")
                return NarrationResult(
                    requests=attempt + 1,
                    retries=attempt,
                    input_characters=len(narration),
                )
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                if attempt == 1:
                    raise TTSGenerationError(
                        "narration generation failed after one retry"
                    ) from exc
        raise AssertionError("unreachable")


class MediaCompositionError(RuntimeError):
    pass


class FFmpegComposer:
    def __init__(self, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    def audio_duration(self, audio_path: Path) -> float:
        result = self._run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            timeout=30.0,
        )
        try:
            duration = float(result.stdout.strip())
        except ValueError as exc:
            raise MediaCompositionError("ffprobe returned an invalid audio duration") from exc
        if duration <= 0:
            raise MediaCompositionError("narration duration must be positive")
        return duration

    def compose_scene(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        duration: float,
    ) -> None:
        fade_out = max(0.0, duration - 0.25)
        self._run(
            [
                self._ffmpeg,
                "-y",
                "-nostdin",
                "-loop",
                "1",
                "-framerate",
                "30",
                "-i",
                str(image_path),
                "-i",
                str(audio_path),
                "-t",
                f"{duration:.3f}",
                "-vf",
                f"scale=1280:720,fade=t=in:st=0:d=0.2,fade=t=out:st={fade_out:.3f}:d=0.25,format=yuv420p",
                "-af",
                f"afade=t=in:st=0:d=0.15,afade=t=out:st={fade_out:.3f}:d=0.25",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(output_path),
            ],
            timeout=120.0,
        )

    def concatenate(self, segments: list[Path], output_path: Path) -> None:
        manifest = output_path.with_suffix(".txt")
        lines = []
        for segment in segments:
            escaped = segment.resolve().as_posix().replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        manifest.write_text("\n".join(lines), encoding="utf-8")
        self._run(
            [
                self._ffmpeg,
                "-y",
                "-nostdin",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            timeout=120.0,
        )

    @staticmethod
    def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                shell=False,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            detail = stderr.strip()[-1_000:]
            raise MediaCompositionError(
                f"media command failed{': ' + detail if detail else ''}"
            ) from exc


class GenerationPipeline:
    def __init__(
        self,
        planner: OpenAILessonPlanner,
        renderer: PillowSceneRenderer,
        narrator: OpenAITTS,
        composer: FFmpegComposer,
        validator: ArtifactValidator,
        artifacts: LocalArtifactStore,
    ) -> None:
        self._planner = planner
        self._renderer = renderer
        self._narrator = narrator
        self._composer = composer
        self._validator = validator
        self._artifacts = artifacts

    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        started = time.perf_counter()
        report_stage(GenerationStage.PLANNING)
        planning = self._planner.plan(concept)
        work_dir = self._artifacts.create_work_dir(job_id)
        try:
            report_stage(GenerationStage.RENDERING)
            images = self._renderer.render(planning.plan, work_dir)

            report_stage(GenerationStage.NARRATING)
            audio_paths: list[Path] = []
            durations: list[float] = []
            tts_requests = 0
            tts_retries = 0
            tts_characters = 0
            for index, scene in enumerate(planning.plan.scenes):
                audio_path = work_dir / f"scene_{index:02d}.mp3"
                narration = self._narrator.synthesize(scene.narration, audio_path)
                duration = self._composer.audio_duration(audio_path)
                audio_paths.append(audio_path)
                durations.append(duration)
                tts_requests += narration.requests
                tts_retries += narration.retries
                tts_characters += narration.input_characters

            report_stage(GenerationStage.COMPOSING)
            compose_started = time.perf_counter()
            segments: list[Path] = []
            for index, (image_path, audio_path, duration) in enumerate(
                zip(images, audio_paths, durations, strict=True)
            ):
                segment = work_dir / f"scene_{index:02d}.mp4"
                self._composer.compose_scene(image_path, audio_path, segment, duration)
                segments.append(segment)
            temporary_final = work_dir / "final.tmp.mp4"
            self._composer.concatenate(segments, temporary_final)
            ffmpeg_seconds = time.perf_counter() - compose_started

            report_stage(GenerationStage.VALIDATING)
            metadata = self._validator.validate(temporary_final)
            metrics = planning.metrics.model_copy(
                update={
                    "tts_model": OpenAITTS.MODEL,
                    "tts_requests": tts_requests,
                    "tts_retries": tts_retries,
                    "tts_input_characters": tts_characters,
                    "audio_seconds": sum(durations),
                    "scene_durations": tuple(durations),
                    "ffmpeg_wall_seconds": ffmpeg_seconds,
                    "total_wall_seconds": time.perf_counter() - started,
                    "artifact_duration_seconds": metadata.duration_seconds,
                    "artifact_bytes": metadata.size_bytes,
                    "probe_validated": True,
                    "decode_validated": True,
                }
            )
            result = GenerationResult(
                artifact_url=f"/videos/{job_id}/artifact",
                metrics=metrics,
            )
            self._artifacts.promote(temporary_final, job_id)
            return result
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class VideoGenerator(Protocol):
    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        """Generate a video, reporting meaningful processing stage changes."""


class UnavailableVideoGenerator:
    """Honest default until the real generation slice is implemented."""

    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        del job_id, concept, report_stage
        raise RuntimeError("video generation is not implemented")
