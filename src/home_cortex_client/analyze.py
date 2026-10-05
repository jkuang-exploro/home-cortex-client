"""Local semantic filter for an existing Stage 2 candidate clip.

The analyzer samples a few frames, asks a pluggable model narrow questions,
and stores a ``SemanticCandidate``. It does not open a camera, upload bytes,
or delete a clip when inference fails. The worker then applies the canonical
promotion policy and writes that decision beside the analysis.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .detect import change_score
from .evidence import EvidenceFailure, decode_clip
from .policy import (
    PolicyConfig,
    decide,
    parse_policy_decision,
    policy_decision_as_mapping,
    with_promotion,
)
from .semantic import (
    CATEGORY_LABELS,
    ActivityObservation,
    AnalysisProvenance,
    CategoryObservation,
    PromotionDecision,
    SemanticCandidate,
    SemanticContractError,
    SemanticFingerprint,
    SemanticStatus,
    VisualEventCandidate,
    fingerprint_for,
    parse_semantic_candidate,
    semantic_candidate_as_mapping,
)


# Person and animal misses are costlier here than an extra retained candidate.
PERSON_CONFIDENCE_FLOOR = 0.25
ANIMAL_CONFIDENCE_FLOOR = 0.25
OBJECT_CONFIDENCE_FLOOR = 0.50
_KEY = re.compile(r"[0-9a-f]{32}\Z")

# Apple's scene classifier emits these on dark or empty rooms. They are not
# household objects, so they never become V1 categories.
SCENE_LABELS = frozenset({
    "outdoor", "indoor", "interior", "exterior", "sky", "night_sky", "night",
    "day", "room", "wall", "ceiling", "floor", "landscape", "nature", "cloud",
    "sun", "moon", "celestial_body", "material", "textile", "color", "darkness",
    "light", "shadow", "empty", "background", "building", "architecture",
    "road", "water", "tree", "plant", "grass", "flower", "ground", "horizon",
})
CATEGORY_ALIASES = {
    "person": "person",
    "human": "person",
    "people": "person",
    "pedestrian": "person",
    "animal": "animal",
    "cat": "animal",
    "dog": "animal",
    "bird": "animal",
    "bottle": "food_or_drink",
    "cup": "food_or_drink",
    "drink": "food_or_drink",
    "food": "food_or_drink",
    "chair": "furniture",
    "couch": "furniture",
    "sofa": "furniture",
    "table": "furniture",
    "bed": "furniture",
    "desk": "furniture",
    "door": "door",
    "screen": "screen",
    "monitor": "screen",
    "television": "screen",
    "tv": "screen",
    "laptop": "screen",
    "phone": "screen",
    "cellphone": "screen",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
    "package": "package",
    "box": "package",
    "suitcase": "package",
}


class AnalysisFailure(Exception):
    """Inference or evidence checks failed. The Stage 2 clip stays in place."""

    def __init__(self, status: SemanticStatus, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(code)


class AnalysisDeferred(Exception):
    """An interactive observation holds the analyzer. The candidate stays queued."""


@dataclass(frozen=True)
class FrameRead:
    """One sampled frame after labels have been mapped onto the V1 vocabulary."""

    categories: tuple[tuple[str, float], ...]
    raw_labels: tuple[tuple[str, float], ...] = ()
    elapsed_ms: float | None = None


class FrameModel(Protocol):
    model_id: str
    model_version: str

    def available(self) -> bool:
        """Return whether this process can run the model."""

    def read_frame(self, jpeg: bytes) -> FrameRead:
        """Read one JPEG. Raise ``AnalysisFailure`` when the frame or model fails."""

    def close(self) -> None:
        """Release an in-flight helper, if this model has one."""


def normalize_label(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def map_labels(pairs: list[tuple[str, float]] | tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
    """Collapse detector labels onto V1. Named people and scene words are dropped."""
    best: dict[str, float] = {}
    for raw, confidence in pairs:
        if not isinstance(raw, str) or isinstance(confidence, bool):
            continue
        if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
            continue
        token = normalize_label(raw)
        if not token or token in SCENE_LABELS:
            continue
        label = CATEGORY_ALIASES.get(token)
        if label is None:
            continue
        score = min(1.0, max(0.0, float(confidence)))
        if score > best.get(label, -1.0):
            best[label] = score
    return tuple(sorted(best.items()))


def sample_plan(
    frame_count: int, peak_index: int | None = None,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Pick the first, peak-motion, middle, and last frames, in time order."""
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("clip has no frames")
    chosen = {0, frame_count - 1, frame_count // 2}
    if peak_index is not None:
        if type(peak_index) is not int or not 0 <= peak_index < frame_count:
            raise ValueError("peak index is outside the clip")
        chosen.add(peak_index)
    plan: list[tuple[int, tuple[str, ...]]] = []
    for index in sorted(chosen):
        roles: list[str] = []
        if index == 0:
            roles.append("start")
        if peak_index is not None and index == peak_index:
            roles.append("peak")
        if index == frame_count // 2:
            roles.append("middle")
        if index == frame_count - 1:
            roles.append("end")
        plan.append((index, tuple(roles)))
    return tuple(plan)


def peak_index_from_grids(grids: list[bytes] | tuple[bytes, ...]) -> int | None:
    """Index of the later frame in the strongest consecutive change. Ties keep the earlier pair."""
    if len(grids) < 2:
        return None
    best_score = -1.0
    best_index: int | None = None
    for index in range(1, len(grids)):
        score = change_score(grids[index - 1], grids[index])
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def opaque_hash(
    model_id: str,
    model_version: str,
    observations: tuple[CategoryObservation, ...],
    activity: ActivityObservation | None,
) -> str:
    body = {
        "activity": None if activity is None else activity.label,
        "categories": [item.label for item in observations],
        "model_id": model_id,
        "model_version": model_version,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def analyze_candidate(
    record: Mapping[str, object],
    payload: bytes,
    manifest: Mapping[str, object] | None,
    model: FrameModel,
    *,
    clock: Callable[[], datetime],
    luma: Callable[[bytes], bytes | None] | None = None,
    hold: threading.Event | None = None,
    stop: threading.Event | None = None,
) -> tuple[SemanticCandidate, dict[str, object]]:
    """Analyze one local clip. Raises instead of returning a failed candidate."""
    source = VisualEventCandidate.from_stage2(record)
    if record.get("status") != "local":
        raise AnalysisFailure(SemanticStatus.FAILED, "evidence_unavailable")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record.get("media_sha256"):
        raise AnalysisFailure(SemanticStatus.FAILED, "hash_mismatch")
    if manifest is not None and (
        manifest.get("sha256") != digest or manifest.get("evidence_id") != record.get("evidence_id")
    ):
        raise AnalysisFailure(SemanticStatus.FAILED, "hash_mismatch")
    try:
        frames = decode_clip(payload)
    except EvidenceFailure as error:
        raise AnalysisFailure(SemanticStatus.FAILED, "corrupt_evidence") from error
    if not model.available():
        raise AnalysisFailure(SemanticStatus.UNAVAILABLE, "model_unavailable")
    decoder = luma if luma is not None else _jpeg_luma
    grids: list[bytes] | None = []
    for _sequence, _millis, jpeg in frames:
        grid = decoder(jpeg)
        if grid is None or grids is None:
            grids = None
            continue
        grids.append(grid)
    peak = None
    if grids is not None:
        try:
            peak = peak_index_from_grids(grids)
        except ValueError:
            peak = None
    plan = sample_plan(len(frames), peak)
    started = time.perf_counter()
    reads: list[FrameRead] = []
    sampled: list[dict[str, object]] = []
    for index, roles in plan:
        _wait_for_observation(hold, stop)
        sequence, _millis, jpeg = frames[index]
        read = model.read_frame(jpeg)
        reads.append(read)
        sampled.append({
            "index": index,
            "roles": list(roles),
            "sequence": sequence,
            "elapsed_ms": read.elapsed_ms,
            "raw_labels": [[label, score] for label, score in read.raw_labels],
        })
    observations, activity = _aggregate(reads, record)
    provenance = AnalysisProvenance(model.model_id, model.model_version)
    fingerprint = SemanticFingerprint(
        fingerprint_for(observations, activity).categories,
        None if activity is None else activity.label,
        opaque_hash(provenance.model_id, provenance.model_version, observations, activity),
    )
    candidate = SemanticCandidate(
        source=source,
        analyzed_at=_analyzed_at(clock, source.clip_end),
        semantic_status=SemanticStatus.SUCCESS,
        analysis=provenance,
        observations=observations,
        activity=activity,
        semantic_fingerprint=fingerprint,
    )
    trace = {
        "sampler": "start-peak-middle-end",
        "frames": sampled,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "model_id": provenance.model_id,
        "model_version": provenance.model_version,
        "accelerator": "not_recorded",
    }
    return candidate, trace


class SemanticStore:
    """Semantic JSON stored beside candidate clips. This store never deletes clips."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def directory(self, candidate_id: str) -> Path:
        key = candidate_id.split(":", 1)[1] if isinstance(candidate_id, str) and ":" in candidate_id else ""
        if not _KEY.fullmatch(key):
            raise ValueError("candidate_id is invalid")
        return self.root / key

    def read_status(self, candidate_id: str) -> str | None:
        path = self.directory(candidate_id) / "semantic.json"
        if not path.is_file():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        status = body.get("semantic_status") if isinstance(body, dict) else None
        if status in {"SUCCESS", "FAILED", "UNAVAILABLE"}:
            return str(status)
        return None

    def load_mapping(self, candidate_id: str) -> dict[str, object] | None:
        path = self.directory(candidate_id) / "semantic.json"
        if not path.is_file():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            parse_semantic_candidate(body)
        except (OSError, json.JSONDecodeError, SemanticContractError, TypeError):
            return None
        return body

    def list_results(self) -> list[dict[str, object]]:
        if not self.root.exists():
            return []
        rows: list[dict[str, object]] = []
        for child in sorted(path for path in self.root.iterdir() if path.is_dir()):
            mapping = None
            try:
                mapping = self.load_mapping("candidate:" + child.name)
            except ValueError:
                continue
            if mapping is None:
                continue
            fingerprint = mapping.get("semantic_fingerprint")
            categories = None
            activity = None
            if isinstance(fingerprint, dict):
                categories = fingerprint.get("categories")
                activity = fingerprint.get("activity")
            promotion_decision, promotion_reason = self._policy_fields(child, mapping)
            rows.append({
                "candidate_id": mapping["source"]["candidate_id"] if isinstance(mapping.get("source"), dict) else None,
                "semantic_status": mapping.get("semantic_status"),
                "failure_code": mapping.get("failure_code"),
                "categories": categories,
                "activity": activity,
                "promotion_decision": promotion_decision,
                "promotion_reason": promotion_reason,
            })
        return rows

    def load_candidate(self, candidate_id: str) -> SemanticCandidate | None:
        body = self.load_mapping(candidate_id)
        if body is None:
            return None
        return parse_semantic_candidate(body)

    def history(self) -> list[SemanticCandidate]:
        """Local semantic records only. Household memory is not consulted."""
        if not self.root.exists():
            return []
        found: list[SemanticCandidate] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                parsed = self.load_candidate("candidate:" + child.name)
            except ValueError:
                continue
            if parsed is not None:
                found.append(parsed)
        found.sort(key=lambda item: (item.analyzed_at, item.candidate_id))
        return found

    def save(
        self,
        candidate: SemanticCandidate,
        trace: Mapping[str, object],
        policy: Mapping[str, object] | None = None,
    ) -> Path:
        directory = self.directory(candidate.candidate_id)
        body = json.dumps(semantic_candidate_as_mapping(candidate), indent=2, sort_keys=True) + "\n"
        traced = json.dumps(trace, indent=2, sort_keys=True) + "\n"
        policy_body = None
        if policy is not None:
            parsed = parse_policy_decision(policy)
            if parsed.candidate_id != candidate.candidate_id or parsed.evidence_id != candidate.evidence_id:
                raise SemanticContractError("promotion metadata must name the same evidence")
            policy_body = json.dumps(policy_decision_as_mapping(parsed), indent=2, sort_keys=True) + "\n"
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_text(directory / "semantic.json", body)
            _atomic_text(directory / "trace.json", traced)
            if policy_body is not None:
                _atomic_text(directory / "policy.json", policy_body)
        return directory

    def _policy_fields(self, directory: Path, mapping: Mapping[str, object]) -> tuple[str | None, str | None]:
        path = directory / "policy.json"
        if not path.is_file():
            return None, None
        source = mapping.get("source")
        candidate_id = source.get("candidate_id") if isinstance(source, Mapping) else None
        try:
            parsed = parse_policy_decision(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, SemanticContractError, TypeError, ValueError):
            return None, None
        if parsed.candidate_id != candidate_id:
            return None, None
        return parsed.decision.value, parsed.reason


class SemanticWorker:
    """Background consumer of local candidates. One clip at a time, never on the capture thread."""

    def __init__(
        self,
        candidates: object,
        model: FrameModel,
        *,
        root: Path,
        hold: threading.Event,
        clock: Callable[[], datetime],
        policy: PolicyConfig | None = None,
        override: str | None = None,
        on_promoted: Callable[..., None] | None = None,
    ) -> None:
        self.candidates = candidates
        self.model = model
        self.store = SemanticStore(root)
        self._hold = hold
        self._clock = clock
        self._policy = policy if policy is not None else PolicyConfig()
        if not isinstance(self._policy, PolicyConfig):
            raise TypeError("policy configuration must be canonical")
        if override not in {None, "force_promote", "force_retain"}:
            raise ValueError("promotion override must be force_promote, force_retain, or unset")
        self._override = override
        self._on_promoted = on_promoted
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unreadable: set[str] = set()

    def set_promoted_listener(self, listener: Callable[..., None] | None) -> None:
        """Receive each ``PROMOTE`` decision after it is stored. Other decisions are not sent."""
        self._on_promoted = listener

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="semantic-analyzer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.model.close()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def run_available(self) -> int:
        """Write results for queued local clips. Already settled results are left alone."""
        written = 0
        records = self.candidates.local_records()
        for record in records:
            if self._stop.is_set() or not self._pause_for_observation():
                break
            candidate_id = str(record.get("candidate_id"))
            if candidate_id in self._unreadable:
                continue
            try:
                status = self.store.read_status(candidate_id)
            except ValueError:
                self._unreadable.add(candidate_id)
                continue
            if status in {"SUCCESS", "FAILED"}:
                continue
            if status == "UNAVAILABLE" and not self.model.available():
                continue
            try:
                inspected = self.candidates.inspect(candidate_id)
                if not inspected.get("evidence_available"):
                    raise AnalysisFailure(SemanticStatus.FAILED, "evidence_unavailable")
                payload = self.candidates.media(candidate_id)
                result, trace = analyze_candidate(
                    record,
                    payload,
                    inspected.get("manifest") if isinstance(inspected.get("manifest"), Mapping) else None,
                    self.model,
                    clock=self._clock,
                    hold=self._hold,
                    stop=self._stop,
                )
            except AnalysisDeferred:
                break
            except SemanticContractError:
                self._unreadable.add(candidate_id)
                continue
            except AnalysisFailure as error:
                result, trace = _failed_result(record, self.model, error, self._clock)
                if result is None:
                    self._unreadable.add(candidate_id)
                    continue
            except Exception:
                failure = AnalysisFailure(SemanticStatus.FAILED, "model_error")
                result, trace = _failed_result(record, self.model, failure, self._clock)
                if result is None:
                    self._unreadable.add(candidate_id)
                    continue
            self._store_result(result, trace)
            written += 1
        return written

    def _store_result(self, result: SemanticCandidate, trace: Mapping[str, object]) -> None:
        decision = decide(
            result,
            self.store.history(),
            self._policy,
            decided_at=_decision_stamp(self._clock),
            override=self._override,
        )
        saved = with_promotion(result, decision)
        self.store.save(saved, trace, policy_decision_as_mapping(decision))
        if self._on_promoted is None or decision.decision != PromotionDecision.PROMOTE:
            return
        try:
            inspected = self.candidates.inspect(saved.candidate_id)
            manifest = inspected.get("manifest")
            if inspected.get("hash_ok") and isinstance(manifest, Mapping):
                self._on_promoted(decision, saved, manifest, self.candidates.media(saved.candidate_id))
        except Exception:
            return

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_available()
            except Exception:
                pass
            if self._stop.wait(1.0):
                return

    def _pause_for_observation(self) -> bool:
        while self._hold.is_set():
            if self._stop.wait(0.05):
                return False
        return not self._stop.is_set()


def _failed_result(
    record: Mapping[str, object],
    model: FrameModel,
    error: AnalysisFailure,
    clock: Callable[[], datetime],
) -> tuple[SemanticCandidate, dict[str, object]] | tuple[None, dict[str, object]]:
    try:
        source = VisualEventCandidate.from_stage2(record)
        provenance = _provenance(model)
        candidate = SemanticCandidate(
            source=source,
            analyzed_at=_analyzed_at(clock, source.clip_end),
            semantic_status=error.status,
            analysis=provenance,
            failure_code=error.code,
        )
    except (SemanticContractError, TypeError, ValueError):
        return None, {}
    return candidate, {"failure_code": error.code, "model_id": None if provenance is None else provenance.model_id}


def _provenance(model: FrameModel) -> AnalysisProvenance | None:
    try:
        return AnalysisProvenance(model.model_id, model.model_version)
    except (TypeError, ValueError, SemanticContractError):
        return None


def _aggregate(
    reads: list[FrameRead], record: Mapping[str, object],
) -> tuple[tuple[CategoryObservation, ...], ActivityObservation | None]:
    scored = [_frame_scores(read) for read in reads]
    best: dict[str, float] = {}
    for frame in scored:
        for label, confidence in frame.items():
            if confidence > best.get(label, -1.0):
                best[label] = confidence
    observations = tuple(CategoryObservation(label, best[label]) for label in sorted(best))
    person = [frame.get("person") for frame in scored]
    present = [score for score in person if score is not None]
    activity: ActivityObservation | None
    if present:
        if person[0] is None and person[-1] is not None:
            activity = ActivityObservation("person_entered", person[-1])
        elif person[0] is not None and person[-1] is None:
            activity = ActivityObservation("person_left", person[0])
        else:
            activity = ActivityObservation("person_present", max(present))
    else:
        sets = [frozenset(frame) for frame in scored]
        changed = any(item != sets[0] for item in sets) and any(sets)
        if changed:
            unstable = set().union(*sets) - set.intersection(*sets) if sets else set()
            confidence = max(
                frame[label] for frame in scored for label in frame if label in unstable
            )
            activity = ActivityObservation("object_activity", confidence)
        elif _large_scene(record):
            activity = ActivityObservation("large_scene_change", _scene_confidence(record))
        else:
            activity = None
    return observations, activity


def _frame_scores(read: FrameRead) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, confidence in read.categories:
        if label not in CATEGORY_LABELS or isinstance(confidence, bool):
            continue
        if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
            continue
        score = min(1.0, max(0.0, float(confidence)))
        if score < _floor(label):
            continue
        if score > scores.get(label, -1.0):
            scores[label] = score
    return scores


def _floor(label: str) -> float:
    if label == "person":
        return PERSON_CONFIDENCE_FLOOR
    if label == "animal":
        return ANIMAL_CONFIDENCE_FLOOR
    return OBJECT_CONFIDENCE_FLOOR


def _large_scene(record: Mapping[str, object]) -> bool:
    if record.get("trigger_type") == "scene_change":
        return True
    score = record.get("scene_change_score")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and float(score) >= 0.45


def _scene_confidence(record: Mapping[str, object]) -> float:
    for name in ("scene_change_score", "peak_score"):
        score = record.get(name)
        if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)):
            return min(1.0, max(0.0, float(score)))
    return 1.0


def _decision_stamp(clock: Callable[[], datetime]) -> str:
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.astimezone()
    return now.isoformat(timespec="milliseconds")


def _analyzed_at(clock: Callable[[], datetime], clip_end: str) -> str:
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.astimezone()
    ended = datetime.fromisoformat(clip_end)
    if now < ended:
        now = ended
    return now.isoformat(timespec="milliseconds")


def _wait_for_observation(hold: threading.Event | None, stop: threading.Event | None) -> None:
    if hold is None:
        return
    while hold.is_set():
        if stop is not None and stop.wait(0.05):
            raise AnalysisDeferred()
        if stop is None:
            time.sleep(0.05)


def _jpeg_luma(jpeg: bytes) -> bytes | None:
    from .perception import jpeg_luma_grid

    return jpeg_luma_grid(jpeg)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
