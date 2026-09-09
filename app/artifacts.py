"""Local artifact validation, atomic promotion, and retrieval."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


class ArtifactValidationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    duration_seconds: float
    width: int
    height: int
    size_bytes: int


class LocalArtifactStore:
    def __init__(self, root: Path | str = "artifacts") -> None:
        self.root = Path(root)
        self._work_root = self.root / "tmp"
        self._final_root = self.root / "videos"

    def create_work_dir(self, job_id: UUID) -> Path:
        self._work_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=self._work_root))

    def final_path(self, job_id: UUID) -> Path:
        return self._final_root / f"{job_id}.mp4"

    def promote(self, temporary_path: Path, job_id: UUID) -> Path:
        self._final_root.mkdir(parents=True, exist_ok=True)
        destination = self.final_path(job_id)
        os.replace(temporary_path, destination)
        return destination

    def get(self, job_id: UUID) -> Path | None:
        path = self.final_path(job_id)
        return path if path.is_file() else None


class ArtifactValidator:
    MINIMUM_SIZE_BYTES = 10_000

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._timeout_seconds = timeout_seconds

    def validate(self, path: Path) -> ArtifactMetadata:
        if not path.is_file():
            raise ArtifactValidationError("composed artifact does not exist")
        size = path.stat().st_size
        if size < self.MINIMUM_SIZE_BYTES:
            raise ArtifactValidationError("composed artifact is unexpectedly small")

        probe = self._run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        try:
            payload = json.loads(probe.stdout)
            streams = payload["streams"]
            duration = float(payload["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError("ffprobe returned invalid metadata") from exc

        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"), None
        )
        if video_stream is None:
            raise ArtifactValidationError("artifact has no video stream")
        if audio_stream is None:
            raise ArtifactValidationError("artifact has no audio stream")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        if (width, height) != (1280, 720):
            raise ArtifactValidationError(
                f"artifact dimensions are {width}x{height}, expected 1280x720"
            )
        if duration <= 0:
            raise ArtifactValidationError("artifact duration must be positive")

        self._run(
            [
                self._ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ]
        )
        return ArtifactMetadata(
            duration_seconds=duration,
            width=width,
            height=height,
            size_bytes=size,
        )

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                shell=False,
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            detail = stderr.strip()[-1_000:]
            raise ArtifactValidationError(
                f"media validation command failed{': ' + detail if detail else ''}"
            ) from exc
