"""Portable, CUDA-free validation harness for the full Speech Pipeline v0.

This module wires the four production stages together exactly as they run on
ai-core and asserts the end-to-end properties the milestone requires. It is the
single source of truth for "what a successful ``text -> TTS -> forced
alignment -> audio postprocess -> caption compiler -> WAV + SRT`` run looks
like".

It is intentionally pure:

- it never imports ``torch``, ``qwen_tts``, ``qwen_asr`` or ``soundfile``;
- it never references the immutable ``tests/fixtures`` evidence or any of its
  exact durations/timestamps -- model output is checked structurally, never
  against historical values;
- it drives only the documented production interfaces
  (:meth:`media_pipeline.runtimes.qwen_tts.Qwen3CustomVoiceTTS.synthesize`,
  :meth:`media_pipeline.runtimes.qwen_aligner.Qwen3ForcedAlignment.align`,
  :func:`media_pipeline.postprocess.postprocess_speech`,
  :func:`media_pipeline.captions.compile_srt`).

Two layers are kept separate so the logic is unit-testable on CPU with fakes:

- :func:`run_chain` performs the mechanical steps and captures every artifact;
- :func:`validate_e2e` asserts every required property against those artifacts.

The two model stages are passed in as callables so the same harness drives the
CPU unit test (fake engines) and the real ai-core run (the production runtime
adapters) without any change to the assertions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from media_pipeline.captions import (
    DEFAULT_PAUSE_THRESHOLD,
    AlignedToken,
    Caption,
    CaptionError,
    build_captions,
    compile_srt,
    format_timestamp,
    load_alignment,
)
from media_pipeline.postprocess import (
    TrimPlan,
    postprocess_speech,
    read_wav,
)

__all__ = [
    "E2EArtifacts",
    "E2EAssertionError",
    "RunChainHooks",
    "align_callable",
    "read_srt",
    "run_chain",
    "snapshot_wav",
    "validate_e2e",
    "wav_callable",
]

#: Tolerance for coarse float-second comparisons that are not frame-exact
#: (durations, gaps). Frame-exact invariants use the WAV grid directly.
_TIMING_TOL = 1e-6

#: Requested Chinese narration instruction, reviewed integration input.
_INSTRUCT = (
    "自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。"
)


class E2EAssertionError(AssertionError):
    """A single end-to-end property failed.

    Message names the property so the audit record states exactly which
    invariant a real run did not satisfy.
    """


# --- stage contracts accepted by the harness --------------------------------


class wav_callable(Protocol):
    """Anything usable as the TTS stage: ``synthesize(request, path) -> artifact``."""

    def __call__(self, request: object, output_wav_path: object) -> Any:  # noqa: D401
        ...


class align_callable(Protocol):
    """Anything usable as the Alignment stage: ``align(request, path) -> artifact``."""

    def __call__(self, request: object, output_alignment_path: object) -> Any:  # noqa: D401
        ...


@dataclass
class RunChainHooks:
    """Optional injection points between pipeline stages.

    The CPU unit test uses these to prove that the harness detects a ``raw.wav``
    that changes after synthesis and an alignment that does not match the
    request text. They never exist on a real run.
    """

    after_tts: Callable[[], None] | None = None
    after_align: Callable[[], None] | None = None


# --- captured artifacts -----------------------------------------------------


@dataclass
class E2EArtifacts:
    """Everything a validated run produced, plus the data needed to assert on it.

    ``raw_wav_snapshot``/``raw_wav_digest`` capture the fresh TTS WAV *immediately
    after synthesis* so downstream processing can be proven not to have altered
    it. ``original_tokens`` are the validated tokens read from the alignment that
    Alignment consumed; ``adjusted_tokens`` are what the Caption Compiler
    consumes. Model output is never compared against historical values here.
    """

    run_dir: Path
    text: str
    speaker: str = "Uncle_Fu"

    raw_wav_path: Path | None = None
    raw_wav_frames: int | None = None
    raw_wav_sample_rate: int | None = None
    raw_wav_snapshot: bytes | None = None
    raw_wav_digest: str | None = None

    alignment_path: Path | None = None
    token_count: int | None = None
    alignment_sample_rate: int | None = None
    alignment_frames: int | None = None

    final_wav_path: Path | None = None
    final_wav_frames: int | None = None
    final_wav_sample_rate: int | None = None

    adjusted_alignment_path: Path | None = None
    trim: TrimPlan | None = None
    original_tokens: list[AlignedToken] | None = None
    adjusted_tokens: list[AlignedToken] | None = None

    captions: list[Caption] | None = None
    srt_path: Path | None = None
    srt_text: str | None = None

    extra: dict = field(default_factory=dict)


# --- raw WAV capture --------------------------------------------------------


def snapshot_wav(path: str | Path) -> tuple[bytes, str]:
    """Read a WAV's raw bytes and return ``(bytes, sha256 hex digest)``."""

    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def assert_wav_unchanged(
    snapshot: tuple[bytes, str], path: str | Path
) -> None:
    """Fail unless ``path`` is byte-identical to the captured ``snapshot``."""

    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != snapshot[1]:
        raise E2EAssertionError(
            f"{path} changed after synthesis (downstream processing must not "
            f"alter the input WAV)"
        )


# --- chain ------------------------------------------------------------------


def run_chain(
    *,
    run_dir: str | Path,
    text: str,
    language: str = "Chinese",
    speaker: str = "Uncle_Fu",
    wav_engine: wav_callable,
    align_engine: align_callable,
    postprocess_kwargs: dict | None = None,
    hooks: RunChainHooks | None = None,
) -> E2EArtifacts:
    """Run the full production chain in one fresh run directory.

    ``text -> TTS -> raw.wav -> alignment -> alignment.raw.json -> postprocess
    -> final.wav + adjusted.json -> caption compiler -> final.srt``.

    Only production interfaces are used. ``raw.wav`` is snapshotted immediately
    after synthesis; the harness verifies below that it is unchanged after
    postprocess. The two model engines are plain callables so the same call
    drives the CPU unit test (fakes) and the real run (production adapters).

    ``hooks.after_tts`` / ``hooks.after_align`` run between stages on demand
    (CPU unit tests only) to prove the harness detects tampering and mismatch.
    """

    hooks = hooks or RunChainHooks()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_wav_path = run_dir / "raw.wav"
    alignment_path = run_dir / "alignment.raw.json"
    final_wav_path = run_dir / "final.wav"
    adjusted_alignment_path = run_dir / "adjusted.json"
    srt_path = run_dir / "final.srt"

    # --- text -> Production TTS -> raw.wav ---------------------------------
    from media_pipeline import CustomVoiceRequest

    request = CustomVoiceRequest(
        text=text,
        language=language,
        speaker=speaker,
        instruct=_INSTRUCT,
    )
    tts_artifact = wav_engine(request, raw_wav_path)
    raw_wav_frames = int(tts_artifact.frames)
    raw_wav_sample_rate = int(tts_artifact.sample_rate)

    # Capture the fresh TTS WAV before anything downstream reads it.
    raw_wav_snapshot, raw_wav_digest = snapshot_wav(raw_wav_path)

    if hooks.after_tts is not None:
        # Unit-test injection point: prove the harness detects a raw.wav that
        # changes after synthesis but before anything downstream reads it.
        hooks.after_tts()

    # --- Production Alignment consumes that exact WAV ----------------------
    from media_pipeline.alignment import AlignmentRequest

    align_request = AlignmentRequest(
        wav_path=raw_wav_path,
        text=text,
        language=language,
    )
    align_artifact = align_engine(align_request, alignment_path)
    token_count = int(align_artifact.token_count)
    alignment_sample_rate = int(align_artifact.sample_rate)
    alignment_frames = int(align_artifact.frames)

    # The aligned WAV is the fresh TTS WAV, unchanged.
    assert_wav_unchanged((raw_wav_snapshot, raw_wav_digest), raw_wav_path)

    if hooks.after_align is not None:
        # Unit-test injection point: prove the harness rejects an alignment JSON
        # that no longer matches the request text (Caption Compiler mismatch).
        hooks.after_align()

    # --- Audio Postprocess: raw.wav -> final.wav + adjusted alignment ------
    trimmed = postprocess_speech(
        raw_wav_path,
        alignment_path,
        final_wav_path,
        adjusted_alignment_path,
        **(postprocess_kwargs or {}),
    )

    # --- proof the input WAV was not touched by postprocess ----------------
    # The snapshot above was taken immediately after synthesis; if Alignment or
    # Audio Postprocess altered raw.wav, the byte check below fails. A unit test
    # injects a raw.wav mutation before postprocess (via ``after_align``) and
    # relies on this check to catch it, proving the baseline was captured early.
    assert_wav_unchanged((raw_wav_snapshot, raw_wav_digest), raw_wav_path)

    # --- Caption Compiler: adjusted alignment -> final.srt -----------------
    original_tokens = load_alignment(alignment_path)
    adjusted_tokens = load_alignment(adjusted_alignment_path)
    captions = build_captions(text, adjusted_tokens)
    srt_text = compile_srt(text, adjusted_tokens)
    srt_path.write_text(srt_text, encoding="utf-8", newline="\n")

    final_wav_frames = int(read_wav(final_wav_path)[2])
    final_wav_sample_rate = int(read_wav(final_wav_path)[1])

    return E2EArtifacts(
        run_dir=run_dir,
        text=text,
        speaker=speaker,
        raw_wav_path=raw_wav_path,
        raw_wav_frames=raw_wav_frames,
        raw_wav_sample_rate=raw_wav_sample_rate,
        raw_wav_snapshot=raw_wav_snapshot,
        raw_wav_digest=raw_wav_digest,
        alignment_path=alignment_path,
        token_count=token_count,
        alignment_sample_rate=alignment_sample_rate,
        alignment_frames=alignment_frames,
        final_wav_path=final_wav_path,
        final_wav_frames=final_wav_frames,
        final_wav_sample_rate=final_wav_sample_rate,
        adjusted_alignment_path=adjusted_alignment_path,
        trim=trimmed,
        original_tokens=original_tokens,
        adjusted_tokens=adjusted_tokens,
        captions=captions,
        srt_path=srt_path,
        srt_text=srt_text,
    )


# --- validation of the required E2E properties ------------------------------


def validate_e2e(artifacts: E2EArtifacts) -> None:
    """Assert every end-to-end property the milestone requires.

    Raises :class:`E2EAssertionError` on the first failed property, with a
    message naming it. Model output is checked structurally -- never against
    the immutable fixtures or their exact durations/timestamps.
    """

    _fresh_tts_wav_valid(artifacts)
    _alignment_consumes_exact_wav(artifacts)
    _raw_wav_unchanged(artifacts)
    _alignment_validation_succeeds(artifacts)
    _postprocess_frame_timing_invariants(artifacts)
    _adjusted_alignment_preserved(artifacts)
    _caption_compiler_consumes(artifacts)
    _reconstructed_text_matches(artifacts)
    _srt_valid_and_ordered(artifacts)
    _srt_timing_consistent_with_final_wav(artifacts)


def _fresh_tts_wav_valid(artifacts: E2EArtifacts) -> None:
    """Fresh TTS WAV is a valid, non-empty, mono 16-bit PCM WAV."""

    if artifacts.raw_wav_path is None:
        raise E2EAssertionError("fresh TTS WAV was not produced")
    # read_wav raises on anything that is not a valid non-empty mono 16-bit
    # PCM WAV, so a successful read is the whole assertion.
    try:
        samples, sample_rate, frames = read_wav(artifacts.raw_wav_path)
    except Exception as exc:  # noqa: BLE001 - format failure is the failure
        raise E2EAssertionError(f"fresh TTS WAV is not valid mono 16-bit PCM: {exc}") from exc

    if not artifacts.raw_wav_frames or artifacts.raw_wav_frames != frames:
        raise E2EAssertionError(
            f"fresh TTS WAV frame count mismatch: reported={artifacts.raw_wav_frames}, "
            f"read_back={frames}"
        )
    if not artifacts.raw_wav_sample_rate or artifacts.raw_wav_sample_rate != sample_rate:
        raise E2EAssertionError(
            f"fresh TTS WAV sample rate mismatch: reported={artifacts.raw_wav_sample_rate}, "
            f"read_back={sample_rate}"
        )
    if frames <= 0 or len(samples) != frames:
        raise E2EAssertionError(
            f"fresh TTS WAV is empty or malformed: frames={frames}, samples={len(samples)}"
        )


def _alignment_consumes_exact_wav(artifacts: E2EArtifacts) -> None:
    """Alignment consumed that exact fresh WAV (authoritative sample-rate/frames)."""

    if artifacts.alignment_sample_rate != artifacts.raw_wav_sample_rate:
        raise E2EAssertionError(
            "Alignment was not run against the fresh TTS WAV: "
            f"alignment sample_rate={artifacts.alignment_sample_rate} "
            f"!= TTS sample_rate={artifacts.raw_wav_sample_rate}"
        )
    if artifacts.alignment_frames != artifacts.raw_wav_frames:
        raise E2EAssertionError(
            "Alignment was not run against the fresh TTS WAV: "
            f"alignment frames={artifacts.alignment_frames} "
            f"!= TTS frames={artifacts.raw_wav_frames}"
        )
    if not artifacts.token_count or artifacts.token_count <= 0:
        raise E2EAssertionError(
            f"production alignment produced no effective tokens: "
            f"token_count={artifacts.token_count}"
        )


def _raw_wav_unchanged(artifacts: E2EArtifacts) -> None:
    """The original raw.wav is byte-identical after downstream processing."""

    if artifacts.raw_wav_snapshot is None or artifacts.raw_wav_digest is None:
        raise E2EAssertionError("raw.wav was never snapshotted after synthesis")
    assert_wav_unchanged((artifacts.raw_wav_snapshot, artifacts.raw_wav_digest), artifacts.raw_wav_path)


def _alignment_validation_succeeds(artifacts: E2EArtifacts) -> None:
    """Production alignment validation succeeds on the consumed WAV."""

    if artifacts.original_tokens is None or not artifacts.original_tokens:
        raise E2EAssertionError("alignment has no effective tokens after loading")
    # Re-validate the *raw* alignment records against the exact WAV the
    # alignment was computed on. This is the same gate the Production Alignment
    # runtime applies before writing; a stale/blank/overflowing alignment could
    # never reach the caption stage here. We re-read the raw records (not the
    # already-normalized AlignedTokens) so the full gate runs on real data.
    from media_pipeline.alignment import validate_alignment

    try:
        with open(artifacts.alignment_path, encoding="utf-8") as handle:
            raw_records = json.load(handle)
        revalidated = validate_alignment(
            raw_records,
            sample_rate=artifacts.raw_wav_sample_rate,
            frames=artifacts.raw_wav_frames,
        )
    except Exception as exc:  # noqa: BLE001
        raise E2EAssertionError(f"production alignment validation failed: {exc}") from exc

    if len(revalidated) != artifacts.token_count:
        raise E2EAssertionError(
            f"alignment token count inconsistent: reported={artifacts.token_count}, "
            f"revalidated={len(revalidated)}"
        )


def _postprocess_frame_timing_invariants(artifacts: E2EArtifacts) -> None:
    """Postprocess frame/timing invariants hold on the frame grid."""

    trim = artifacts.trim
    if trim is None:
        raise E2EAssertionError("Audio Postprocess did not return a TrimPlan")

    if trim.frame_rate != artifacts.raw_wav_sample_rate:
        raise E2EAssertionError(
            "trim frame rate does not match the WAV: "
            f"{trim.frame_rate} != {artifacts.raw_wav_sample_rate}"
        )
    if trim.frames != artifacts.raw_wav_frames:
        raise E2EAssertionError(
            "trim total frames do not match the original WAV: "
            f"{trim.frames} != {artifacts.raw_wav_frames}"
        )
    if artifacts.final_wav_sample_rate != artifacts.raw_wav_sample_rate:
        raise E2EAssertionError(
            "final WAV sample rate changed under postprocess: "
            f"{artifacts.final_wav_sample_rate} != {artifacts.raw_wav_sample_rate}"
        )
    if artifacts.final_wav_frames is None:
        raise E2EAssertionError("final WAV frame count not captured")
    if artifacts.final_wav_frames != trim.kept_frames:
        raise E2EAssertionError(
            f"final WAV frame count != kept frames: "
            f"{artifacts.final_wav_frames} != {trim.kept_frames}"
        )
    if trim.kept_frames <= 0:
        raise E2EAssertionError("postprocess kept zero frames")
    if trim.trim_before < 0 or trim.trim_after < 0:
        raise E2EAssertionError("trim before/after must be non-negative")


def _adjusted_alignment_preserved(artifacts: E2EArtifacts) -> None:
    """Adjusted alignment preserves token text/order/count/durations/gaps.

    Under the documented timing shift every adjusted token equals the
    corresponding original token shifted by the same quantized offset, so
    per-token durations and the gaps between tokens are unchanged.
    """

    original = artifacts.original_tokens
    adjusted = artifacts.adjusted_tokens
    if original is None or adjusted is None:
        raise E2EAssertionError("adjusted alignment tokens were not captured")
    if len(original) != len(adjusted):
        raise E2EAssertionError(
            f"adjusted alignment token count changed: {len(original)} -> {len(adjusted)}"
        )
    if [token.text for token in original] != [token.text for token in adjusted]:
        raise E2EAssertionError(
            "adjusted alignment changed token text or order"
        )

    offset = artifacts.trim.start_frame / artifacts.trim.frame_rate

    # Production Postprocess shifts every timestamp by *minus* the quantized
    # front trim offset (new = old - offset). The sign is fixed by the contract,
    # so the shift is checked as signed equality, not magnitude: an alignment
    # shifted by *+*offset (the wrong direction) must fail here.
    expected_shift = -offset

    for index, (old, new) in enumerate(zip(original, adjusted)):
        duration_old = old.end - old.start
        duration_new = new.end - new.start
        if abs(duration_new - duration_old) > _TIMING_TOL:
            raise E2EAssertionError(
                f"adjusted token {index} duration changed: {duration_old} -> {duration_new}"
            )
        # A uniform signed shift by the quantized offset preserves every
        # per-token duration and every gap between tokens.
        start_shift = new.start - old.start
        end_shift = new.end - old.end
        if abs(start_shift - expected_shift) > _TIMING_TOL:
            raise E2EAssertionError(
                f"adjusted token {index} start shifted by {start_shift}, "
                f"expected the signed quantized offset {expected_shift}"
            )
        if abs(end_shift - expected_shift) > _TIMING_TOL:
            raise E2EAssertionError(
                f"adjusted token {index} end shifted by {end_shift}, "
                f"expected the signed quantized offset {expected_shift}"
            )
        if abs(start_shift - end_shift) > _TIMING_TOL:
            raise E2EAssertionError(
                f"adjusted token {index} start/end shifted inconsistently: "
                f"{start_shift} / {end_shift}"
            )

    # The gap between consecutive tokens is preserved by a uniform shift.
    for index in range(len(original) - 1):
        gap_old = original[index + 1].start - original[index].end
        gap_new = adjusted[index + 1].start - adjusted[index].end
        if abs(gap_new - gap_old) > _TIMING_TOL:
            raise E2EAssertionError(
                f"adjusted alignment gap {index} changed: {gap_old} -> {gap_new}"
            )


def _caption_compiler_consumes(artifacts: E2EArtifacts) -> None:
    """The Caption Compiler consumes the adjusted alignment without error."""

    if artifacts.captions is None:
        raise E2EAssertionError("Caption Compiler produced no captions")
    if not artifacts.captions:
        raise E2EAssertionError("Caption Compiler produced zero captions")


def _reconstructed_text_matches(artifacts: E2EArtifacts) -> None:
    """Reconstructed caption text matches the source under whitespace semantics."""

    captions = artifacts.captions
    assert captions is not None
    joined = "".join(caption.text for caption in captions)
    if "".join(joined.split()) != "".join(artifacts.text.split()):
        raise E2EAssertionError(
            "reconstructed caption text does not match the source text"
        )


def _srt_valid_and_ordered(artifacts: E2EArtifacts) -> None:
    """The resulting SRT is non-empty, ordered and valid.

    This checks the SRT independently of the production renderer: it parses the
    document with :func:`read_srt` and re-derives the expected millisecond
    timestamp string for each caption through an *independent* ROUND_HALF_UP
    oracle (:func:`_srt_timestamp_ms`) rather than by re-running the production
    ``render_srt``. That way a regression in the production millisecond rounding
    still surfaces here instead of cancelling out against an identical re-render.
    Production contract only forbids true overlap: captions may touch
    (``next.start == previous.end``), so touching is allowed and only a real
    ``start < previous_end`` fails.
    """

    srt = artifacts.srt_text
    if not srt or not srt.strip():
        raise E2EAssertionError("SRT is empty")

    blocks = read_srt(srt)
    captions = artifacts.captions
    assert captions is not None
    if len(blocks) != len(captions):
        raise E2EAssertionError(
            f"SRT block count {len(blocks)} != caption count {len(captions)}"
        )

    previous_end = -1.0
    for index, (block, caption) in enumerate(zip(blocks, captions)):
        if block["index"] != index + 1:
            raise E2EAssertionError(f"SRT block {index} has a non-sequential index")
        if not (block["start"] < block["end"]):
            raise E2EAssertionError(f"SRT block {index} has start >= end")
        # Only true overlap fails; touching boundaries are allowed.
        if block["start"] < previous_end - _TIMING_TOL:
            raise E2EAssertionError(
                f"SRT block {index} overlaps the previous block"
            )
        previous_end = block["end"]

        # The SRT text is the caption text verbatim, and its timestamps are the
        # production millisecond ROUND_HALF_UP rendering, checked here through an
        # independent oracle (see _srt_timestamp_ms) rather than by re-running
        # the production render_srt, so a rounding regression still surfaces here.
        if block["text"] != caption.text:
            raise E2EAssertionError(
                f"SRT block {index} text {block['text']!r} != caption text "
                f"{caption.text!r}"
            )
        expected_start = _srt_timestamp_ms(caption.start)
        expected_end = _srt_timestamp_ms(caption.end)
        if block["start_str"] != expected_start:
            raise E2EAssertionError(
                f"SRT block {index} start timestamp {block['start_str']} != "
                f"production rounding {expected_start}"
            )
        if block["end_str"] != expected_end:
            raise E2EAssertionError(
                f"SRT block {index} end timestamp {block['end_str']} != "
                f"production rounding {expected_end}"
            )


def _srt_timing_consistent_with_final_wav(artifacts: E2EArtifacts) -> None:
    """The *emitted* SRT millisecond timestamps fall inside the final WAV.

    The check uses the actual millisecond timestamps parsed out of the produced
    SRT document, not the float ``caption.end``: ``ROUND_HALF_UP`` rendering can
    push an emitted timestamp past the float boundary (for example
    ``caption.end = 1.2346`` -> SRT ``1.235``), so a float-only check could
    accept an SRT whose emitted end exceeds the final WAV. Touching the very end
    of the WAV is allowed, since the last caption has no following block.
    """

    captions = artifacts.captions
    if captions is None:
        raise E2EAssertionError("no captions to check timing against")
    if artifacts.final_wav_sample_rate is None or artifacts.final_wav_frames is None:
        raise E2EAssertionError("final WAV duration is unknown")
    final_duration = artifacts.final_wav_frames / artifacts.final_wav_sample_rate

    blocks = read_srt(artifacts.srt_text)
    for index, (block, caption) in enumerate(zip(blocks, captions)):
        if block["start"] < 0 or block["end"] < 0:
            raise E2EAssertionError(f"SRT block {index} has negative timing")
        # The *emitted* millisecond end (not the float caption.end) must fall
        # inside the final WAV; touching the end is allowed.
        if block["end"] > final_duration + _TIMING_TOL:
            raise E2EAssertionError(
                f"SRT block {index} ends at {block['end']}, past the final WAV "
                f"duration {final_duration}"
            )
        if block["start"] > final_duration + _TIMING_TOL:
            raise E2EAssertionError(
                f"SRT block {index} starts at {block['start']}, past the final WAV "
                f"duration {final_duration}"
            )


def _srt_timestamp_ms(seconds: float) -> str:
    """Render ``seconds`` as an SRT timestamp string (independent oracle).

    This reproduces the production millisecond ``ROUND_HALF_UP`` semantics from
    :func:`media_pipeline.captions.format_timestamp` as a stand-alone helper so
    the E2E check does not depend on re-running the production renderer.
    """

    total_ms = int(
        (Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


_SRT_BLOCK = re.compile(
    r"(?P<index>\d+)\s*\n"
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(?P<text>[^\n]*)"   # the whole caption payload line, captured raw
    r"\n\n?"              # text-line terminator + optional blank-line separator
    r"(?=\d+\s*\n|\Z)",  # next block index, or the end of the document
)


def read_srt(document: str) -> list[dict]:
    """Parse a production SRT document into blocks.

    Each block carries ``index``/``start``/``end`` (timestamps parsed both back
    to seconds, for timing checks against the WAV, and kept as their original
    ``HH:MM:SS,mmm`` string, for checking the production rounding) and ``text``.

    The whole document is enforced to conform: the first block must start at the
    beginning (rejecting a garbage prefix), consecutive blocks must be exactly
    contiguous (rejecting content between them), and the last block must reach
    the end (rejecting trailing garbage). This proves the *entire* document is
    the production SRT structure, not merely that some valid blocks appear in it.

    The caption ``text`` is returned *verbatim* -- it is never ``.strip()``-ed,
    so any leading/trailing whitespace the renderer emits on the payload line is
    preserved and can be rejected by an exact comparison downstream.
    """

    def to_seconds(value: str) -> float:
        time_part, millis = value.split(",")
        hours, minutes, secs = (int(part) for part in time_part.split(":"))
        return hours * 3600 + minutes * 60 + secs + int(millis) / 1000.0

    matches = list(_SRT_BLOCK.finditer(document))
    if not matches:
        raise CaptionError(f"SRT document has no valid block: {document!r}")
    if matches[0].start() != 0:
        raise CaptionError("SRT document has content before the first block")
    for previous, match in zip(matches[:-1], matches[1:]):
        if match.start() != previous.end():
            raise CaptionError("SRT document has content between blocks")
    if matches[-1].end() != len(document):
        raise CaptionError("SRT document has trailing content after the last block")

    blocks: list[dict] = []
    for match in matches:
        blocks.append(
            {
                "index": int(match.group("index")),
                "start": to_seconds(match.group("start")),
                "end": to_seconds(match.group("end")),
                "start_str": match.group("start"),
                "end_str": match.group("end"),
                "text": match.group("text"),
            }
        )
    return blocks
