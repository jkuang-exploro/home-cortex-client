"""Optional local frame readers. Neither backend is selected unless configured.

Apple Vision answers person, animal, and a short object list through a small
helper process. OpenCV HOG answers person only. Hardware acceleration stays in
the trace, not in the semantic contract. Face identity is not requested.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .analyze import AnalysisDeferred, AnalysisFailure, FrameRead, map_labels
from .config import ChangeDetectionConfig
from .semantic import SemanticStatus
from .sources import _load_cv2


VISION_FRAME_SWIFT = r"""import AppKit
import Foundation
import Vision

struct Box: Codable {
    let confidence: Double
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct Label: Codable {
    let identifier: String
    let confidence: Double
}

struct Report: Codable {
    let elapsed_ms: Double
    let humans: [Box]
    let animals: [Label]
    let classifications: [Label]
}

guard CommandLine.arguments.count == 2 else {
    fputs("usage: vision-frame <jpeg>\n", stderr)
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let data: Data
do {
    data = try Data(contentsOf: url)
} catch {
    fputs("image_unreadable\n", stderr)
    exit(3)
}
guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fputs("image_unreadable\n", stderr)
    exit(3)
}

let humans = VNDetectHumanRectanglesRequest()
let animals = VNRecognizeAnimalsRequest()
let classifications = VNClassifyImageRequest()
let handler = VNImageRequestHandler(cgImage: image, options: [:])
let started = Date()
do {
    try handler.perform([humans, animals, classifications])
} catch {
    fputs("vision_failed\n", stderr)
    exit(4)
}
let elapsed = Date().timeIntervalSince(started) * 1000

let humanBoxes = (humans.results ?? []).map { item in
    Box(
        confidence: Double(item.confidence),
        x: Double(item.boundingBox.origin.x),
        y: Double(item.boundingBox.origin.y),
        width: Double(item.boundingBox.size.width),
        height: Double(item.boundingBox.size.height)
    )
}
let animalLabels = (animals.results ?? []).flatMap { item in
    item.labels.map { label in
        Label(identifier: label.identifier, confidence: Double(label.confidence))
    }
}
let topLabels = (classifications.results ?? [])
    .filter { $0.confidence >= 0.05 }
    .prefix(8)
    .map { Label(identifier: $0.identifier, confidence: Double($0.confidence)) }

let report = Report(
    elapsed_ms: elapsed,
    humans: humanBoxes,
    animals: animalLabels,
    classifications: Array(topLabels)
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
let body = try encoder.encode(report)
FileHandle.standardOutput.write(body)
FileHandle.standardOutput.write(Data([0x0A]))
"""

HELPER_SHA = hashlib.sha256(VISION_FRAME_SWIFT.encode("utf-8")).hexdigest()


def open_model(name: str) -> VisionBackend | HogBackend:
    if name == "vision":
        return VisionBackend()
    if name == "hog":
        return HogBackend()
    raise ValueError("analyzer must be vision or hog")


def jpeg_luma_grid(jpeg: bytes) -> bytes | None:
    """32×24 luma for peak-frame selection. Missing OpenCV skips the peak frame."""
    try:
        cv2 = _load_cv2()
    except RuntimeError:
        return None
    import numpy as np

    try:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8).copy(), cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None
    if image is None:
        return None
    small = cv2.resize(
        image,
        (ChangeDetectionConfig.grid_width, ChangeDetectionConfig.grid_height),
        interpolation=cv2.INTER_AREA,
    )
    raw = small.tobytes()
    expected = ChangeDetectionConfig.grid_width * ChangeDetectionConfig.grid_height
    if len(raw) != expected:
        return None
    return raw


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def hog_confidence(weight: float) -> float:
    """Map an unbounded HOG weight into model confidence in [0, 1]."""
    if not math.isfinite(weight) or weight <= 0:
        return 0.0
    return 1.0 - math.exp(-weight)


class VisionBackend:
    """Apple Vision human, animal, and image classifiers via a compiled helper."""

    model_id = "apple-vision"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (Path.home() / "Library" / "Caches" / "home-cortex-client")
        self._binary: Path | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._compile_failed = False

    @property
    def model_version(self) -> str:
        macos = platform.mac_ver()[0] or "unknown"
        return f"macos-{macos}+helper-{HELPER_SHA[:12]}"

    def available(self) -> bool:
        if self._compile_failed or platform.system() != "Darwin":
            return False
        binary = self._compiled_binary()
        if binary.is_file() and os.access(binary, os.X_OK):
            return True
        return shutil.which("swiftc") is not None

    def close(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def read_frame(self, jpeg: bytes) -> FrameRead:
        binary = self._ensure_binary()
        fd, path = tempfile.mkstemp(prefix="vision-frame-", suffix=".jpg")
        os.close(fd)
        Path(path).write_bytes(jpeg)
        proc: subprocess.Popen[bytes] | None = None
        try:
            try:
                proc = subprocess.Popen(
                    [str(binary), path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as error:
                raise AnalysisFailure(SemanticStatus.FAILED, "model_error") from error
            with self._lock:
                self._proc = proc
            try:
                os.setpriority(os.PRIO_PROCESS, proc.pid, 10)
            except OSError:
                pass
            try:
                stdout, _stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired as error:
                proc.kill()
                proc.communicate()
                raise AnalysisFailure(SemanticStatus.FAILED, "model_error") from error
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            Path(path).unlink(missing_ok=True)
        if proc is None:
            raise AnalysisFailure(SemanticStatus.FAILED, "model_error")
        if proc.returncode is not None and proc.returncode < 0:
            raise AnalysisDeferred()
        if proc.returncode == 3:
            raise AnalysisFailure(SemanticStatus.FAILED, "corrupt_evidence")
        if proc.returncode != 0:
            raise AnalysisFailure(SemanticStatus.FAILED, "model_error")
        try:
            body = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise AnalysisFailure(SemanticStatus.FAILED, "model_error") from error
        pairs: list[tuple[str, float]] = []
        for item in body.get("humans") or []:
            if isinstance(item, dict) and "confidence" in item:
                pairs.append(("person", float(item["confidence"])))
        for item in body.get("animals") or []:
            if isinstance(item, dict) and "identifier" in item:
                pairs.append((str(item["identifier"]), float(item["confidence"])))
        for item in body.get("classifications") or []:
            if isinstance(item, dict) and "identifier" in item:
                pairs.append((str(item["identifier"]), float(item["confidence"])))
        elapsed = body.get("elapsed_ms")
        return FrameRead(
            categories=map_labels(pairs),
            raw_labels=tuple((name, score) for name, score in pairs),
            elapsed_ms=float(elapsed) if isinstance(elapsed, (int, float)) else None,
        )

    def _compiled_binary(self) -> Path:
        return self.cache_dir / f"vision-frame-{HELPER_SHA[:16]}"

    def _ensure_binary(self) -> Path:
        binary = self._compiled_binary()
        if binary.is_file() and os.access(binary, os.X_OK):
            self._binary = binary
            return binary
        if platform.system() != "Darwin" or shutil.which("swiftc") is None:
            raise AnalysisFailure(SemanticStatus.UNAVAILABLE, "model_unavailable")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        source = self.cache_dir / f"vision-frame-{HELPER_SHA[:16]}.swift"
        source.write_text(VISION_FRAME_SWIFT, encoding="utf-8")
        completed = subprocess.run(
            [
                "swiftc", "-O",
                "-framework", "Vision",
                "-framework", "AppKit",
                "-framework", "CoreGraphics",
                "-o", str(binary),
                str(source),
            ],
            capture_output=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0 or not binary.is_file():
            self._compile_failed = True
            raise AnalysisFailure(SemanticStatus.UNAVAILABLE, "model_unavailable")
        self._binary = binary
        return binary


class HogBackend:
    """OpenCV's default people detector. It has no animal or object head."""

    model_id = "opencv-hog-people"

    def __init__(self) -> None:
        self._hog = None

    @property
    def model_version(self) -> str:
        try:
            cv2 = _load_cv2()
        except RuntimeError:
            return "opencv-missing"
        return f"opencv-{cv2.__version__}-default-people"

    def available(self) -> bool:
        try:
            self._detector()
        except RuntimeError:
            return False
        return True

    def close(self) -> None:
        return None

    def read_frame(self, jpeg: bytes) -> FrameRead:
        try:
            cv2 = _load_cv2()
        except RuntimeError as error:
            raise AnalysisFailure(SemanticStatus.UNAVAILABLE, "model_unavailable") from error
        import numpy as np

        started = time.perf_counter()
        try:
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8).copy(), cv2.IMREAD_GRAYSCALE)
        except cv2.error as error:
            raise AnalysisFailure(SemanticStatus.FAILED, "corrupt_evidence") from error
        if image is None:
            raise AnalysisFailure(SemanticStatus.FAILED, "corrupt_evidence")
        if image.shape[0] < 128 or image.shape[1] < 64:
            return FrameRead(categories=(), elapsed_ms=_elapsed_ms(started))
        hog = self._detector()
        try:
            found, weights = hog.detectMultiScale(
                image, winStride=(8, 8), padding=(8, 8), scale=1.05,
            )
        except cv2.error as error:
            raise AnalysisFailure(SemanticStatus.FAILED, "model_error") from error
        scores = [hog_confidence(float(weight)) for weight in np.array(weights).reshape(-1)]
        if len(found) and not scores:
            scores = [0.5]
        elapsed = _elapsed_ms(started)
        if not scores:
            return FrameRead(categories=(), raw_labels=(), elapsed_ms=elapsed)
        confidence = max(scores)
        return FrameRead(
            categories=(("person", confidence),),
            raw_labels=tuple(("person", score) for score in scores),
            elapsed_ms=elapsed,
        )

    def _detector(self):
        if self._hog is not None:
            return self._hog
        cv2 = _load_cv2()
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hog = hog
        return hog
