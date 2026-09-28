"""Qwen3 ForcedAligner runtime adapter (v0).

This is the ai-core GPU adapter for the forced-alignment stage. It wraps the
verified ``experiments/test_align.py`` smoke test into a production-shaped
engine:

- the model loads once and the engine can align many requests,
- each request is one already-segmented utterance,
- the output is written as the existing ``{"text", "start", "end"}`` JSON that
  Audio Postprocess and the Caption Compiler consume, with the model's
  timestamp precision preserved.

The heavy imports (``torch``, ``qwen_asr``) happen lazily inside
:meth:`Qwen3ForcedAlignment._load_aligner`, so merely importing this module (or
the portable :mod:`media_pipeline.alignment` contracts) does not require torch,
CUDA, or the model runtime.

v0 is deliberately narrow: Qwen3 ForcedAligner only, ``torch.bfloat16``, and
``device="cuda:0"`` (configurable). There is no dtype abstraction, CPU fallback,
batching, long-text splitting, ASR, language detection, or provider abstraction.

The model is loaded lazily on first construction through the overridable
:meth:`Qwen3ForcedAlignment._load_aligner` seam. Loading once at construction
keeps the engine reusable across many :meth:`align` calls, and the seam keeps
the mapping/validation logic unit-testable on CPU without the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..alignment import (
    AlignmentArtifact,
    AlignmentRequest,
    AlignmentRequestError,
    AlignmentRuntimeError,
    validate_alignment,
    validate_request,
    validate_wav,
)
from ..captions import check_alignment_matches_text

__all__ = ["Qwen3ForcedAlignment"]


class Qwen3ForcedAlignment:
    """Qwen3 ForcedAligner engine: loads once, aligns many requests.

    ``model_path`` and ``device`` are call-site arguments; neither a model path
    nor any host path is hard-coded. The model is loaded once by
    :meth:`_load_aligner` and reused for every :meth:`align` call.
    """

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0") -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._aligner = self._load_aligner()

    def _load_aligner(self):
        """Load the Qwen3 ForcedAligner model once and reuse it.

        ``torch`` and ``qwen_asr`` are imported lazily here only -- never at
        module import time -- so merely importing the portable surface does not
        require them. Missing dependencies, model-loading, and initialization
        failures all cross the runtime boundary, so they are reported as
        :class:`AlignmentRuntimeError` with chaining.
        """

        try:
            import torch  # lazy: only when an engine is actually created
            from qwen_asr import Qwen3ForcedAligner  # lazy
        except Exception as exc:  # pragma: no cover - runtime specific
            raise AlignmentRuntimeError(
                f"failed to import Qwen3 ForcedAligner runtime: {exc}"
            ) from exc

        try:
            return Qwen3ForcedAligner.from_pretrained(
                self._model_path,
                device_map=self._device,
                dtype=torch.bfloat16,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise AlignmentRuntimeError(
                f"failed to load Qwen3 ForcedAligner model {self._model_path}: {exc}"
            ) from exc

    def align(
        self,
        request: AlignmentRequest,
        output_alignment_path: str | Path,
    ) -> AlignmentArtifact:
        """Align one utterance and write its alignment JSON.

        Mirrors the verified smoke test: ``validate_request`` and
        ``validate_wav`` gate bad input (raising :class:`AlignmentRequestError`),
        the loaded engine aligns the *original* WAV path with the request's
        ``text`` / ``language``, the runtime must return exactly one result set
        (raising :class:`AlignmentRuntimeError` otherwise), the single result is
        mapped in order into the existing public alignment representation without
        sorting or rounding, ``validate_alignment`` gates malformed output
        (raising :class:`AlignmentError`), the aligned text is matched against the
        original request text (raising :class:`AlignmentMismatchError` otherwise),
        and only then is the JSON written so a failed validation never creates a
        bogus target file.
        """

        # Never overwrite the input WAV with an alignment artifact.
        output_path = Path(output_alignment_path)
        if output_path.resolve() == Path(request.wav_path).resolve():
            raise AlignmentRequestError(
                "alignment output path must not overwrite the input WAV"
            )

        # Portable validation first: invalid request or input WAV never reaches
        # the model and never writes any file.
        validate_request(request)
        sample_rate, frames = validate_wav(request.wav_path)

        aligner = self._aligner
        try:
            # Runtime boundary: qwen_asr accepts a string filesystem path, so
            # convert the public Path here and only here.
            results = aligner.align(
                audio=str(request.wav_path),
                text=request.text,
                language=request.language,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise AlignmentRuntimeError(
                f"failed to align utterance: {exc}"
            ) from exc

        # A Production Alignment request represents exactly one already-segmented
        # utterance, so the runtime must return exactly one result set. Reject an
        # unexpected result count up front: a second set would otherwise be
        # silently discarded and the first silently consumed.
        try:
            result_count = len(results)
        except TypeError as exc:  # pragma: no cover - runtime specific
            raise AlignmentRuntimeError(
                f"runtime returned malformed alignment: {results!r}"
            ) from exc
        if result_count != 1:
            raise AlignmentRuntimeError(
                f"runtime returned {result_count} result set(s); expected exactly 1"
            )

        # Map the single result set in order. Do not sort, round, repair, add,
        # or remove: this preserves the model's exact timestamp precision and
        # content for the downstream stages.
        try:
            raw = [
                {"text": item.text, "start": item.start_time, "end": item.end_time}
                for item in results[0]
            ]
        except (IndexError, TypeError, AttributeError) as exc:  # pragma: no cover - runtime specific
            raise AlignmentRuntimeError(
                f"runtime returned malformed alignment: {results!r}"
            ) from exc

        # Validate the complete result before writing anything.
        tokens = validate_alignment(raw, sample_rate=sample_rate, frames=frames)

        # Match the aligned token text against the original utterance text before
        # writing. Reuses the Caption Compiler's character-level matching so a
        # stale alignment (missing, extra, or incorrect aligned text) fails with
        # AlignmentMismatchError instead of silently overwriting the target.
        check_alignment_matches_text(request.text, tokens)

        # Write the original mapped runtime records, unmodified. Blank records
        # are rejected above and invalid timestamps are rejected above, so every
        # written record is valid and the record count matches token_count. The
        # model's exact content and timestamp precision are preserved rather than
        # reconstructed from AlignedToken objects.
        try:
            with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(raw, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError:
            raise

        return AlignmentArtifact(
            alignment_path=output_path,
            sample_rate=sample_rate,
            frames=frames,
            token_count=len(tokens),
        )
