# Qwen 0.1.1 generation-control audit

Scope: `qwen-tts 0.1.1`, `Qwen3-TTS-12Hz-1.7B-Base`, normal ICL with the
accepted reusable prompt. This is evidence for the [Layer 1 experiment](qwen-clone-prosody-v0.md),
not a production sampling API or an Issue #8 solution.

## Source identity

The published [PyPI 0.1.1 source archive](https://pypi.org/project/qwen-tts/0.1.1/#files)
was downloaded for read-only inspection, without installing packages. Its
SHA256 matched PyPI metadata:
`afba5fa235806a6883f46a389e67540b46f8a55da457216bf1d7342903814780`.
The two audited files were byte-identical to both upstream commit `6cafe55`
and repository-supported commit `022e286b98fbec7e1e916cb940cdf532cd9f488e`:

| File | Released SHA256 |
| --- | --- |
| `qwen_tts/inference/qwen3_tts_model.py` | `e1da450732857c1f5fe3e36ebab85db2f6dc6a48caaff6973f463384e30275e4` |
| `qwen_tts/core/models/modeling_qwen3_tts.py` | `25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0` |

Authoritative immutable references: [wrapper](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py),
[core model](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py).
The runner records installed source hashes, versions and local checkpoint hashes;
version labels alone cannot identify a locally patched installation.

## Supported controls

`generate_voice_clone` formally exposes text/language, reference audio/text,
reference-mode flags, reusable prompt, `non_streaming_mode`, and `**kwargs`.
None of the following ten controls is an explicit high-level signature argument.
Each is a **documented named kwarg**, explicitly merged by `_merge_generate_kwargs`
(wrapper lines 287–356) and explicitly accepted by the core `generate` signature
(core lines 2022–2066):

| Control | Meaning / receiving path | Wrapper fallback |
| --- | --- | ---: |
| `do_sample` | Primary talker sampling switch | `True` |
| `temperature` | Primary sampler temperature | `0.9` |
| `top_k` | Primary top-k sampling | `50` |
| `top_p` | Primary top-p sampling | `1.0` |
| `repetition_penalty` | Primary talker repetition penalty | `1.05` |
| `subtalker_dosample` | Code predictor sampling switch | `True` |
| `subtalker_temperature` | Code predictor temperature | `0.9` |
| `subtalker_top_k` | Code predictor top-k sampling | `50` |
| `subtalker_top_p` | Code predictor top-p sampling | `1.0` |
| `max_new_tokens` | Maximum generated codec tokens | `2048` |

Explicit non-`None` kwargs take precedence over loaded checkpoint defaults,
which take precedence over the wrapper fallback. The loader actually reads
`generation_config.json` (core lines 1922–1936); wrapper comments call it
`generate_config.json`. Core direct-call `max_new_tokens=4096` is bypassed by
the wrapper's resolved value. The [official Base checkpoint config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/generation_config.json)
observed during this audit uses the table's values except `max_new_tokens=8192`.
That URL is mutable: real local file hashes and effective controls govern each run.
The runner refuses a default condition whose actual sampling switches are off.

Subtalker controls reach `code_predictor.generate` (core lines 1671–1680).
The wrapper calls them tokenizer-v2 controls; the inspected 12Hz Base path uses
the predictor and these switches. Sampling-off diagnostics set only
`do_sample=False, subtalker_dosample=False`. No temperature/top-k/top-p override
is needed. This removes these samplers; it does not guarantee identical GPU
output across environments or seeds. Waveform/PCM hashes and repeat metrics
allow empirical repeatability checks later.

## RNG, batches and context

No formal or documented Qwen `seed` or `torch.Generator` argument exists here.
The wrapper broadly documents forwarding generic Transformers kwargs, but core
`generate` constructs a fixed `talker_kwargs` dictionary; it does not forward
arbitrary kwargs to the talker. `seed`/`generator` are unsupported by this path,
not justified by a generic Hugging Face API. The experiment instead seeds
process-global Python, NumPy and PyTorch RNGs before model load. This is an
external experiment procedure, not a Qwen per-call control. GPU nondeterminism
and topology-dependent RNG consumption remain possible; paired seeds do not
make S3/B3/L1 outputs equivalent draws.

List-of-text cloning is formally exposed and documented. Wrapper lines 556–610
broadcast one reusable prompt across texts, tokenize each target separately,
then invoke the core once. Core lines 2086–2292 build separate per-item prompts,
pad the batch and invoke `self.talker.generate` once. B3 therefore uses one true
three-item invocation, with separate samples. No preceding batch item's target
text/audio is inserted into the next item's conditioning. Padding, EOS behavior,
batched numerical execution and RNG consumption can differ from separate calls;
this is not evidence of shared discourse context.

Neither inspected public path offers `previous_text`, `previous_audio`,
`previous_request_id` or reusable continuation state. Normal ICL inserts the
original reference transcript/codes and speaker embedding, not earlier generated
target units. Internal generation cache is local to one invocation. L1 preserves
one target generation stream across the combined text; S3 restarts it each time.
Public `non_streaming_mode=False` simulates streaming text input and is not a
cross-request continuation API. The experiment leaves that default unchanged.

## Implementation boundary

The accepted `Qwen3VoiceCloneTTS.synthesize` cannot pass controls or lists.
The experiment uses it directly for default separate/continuous calls. A small
experiment-owned bridge accesses its pinned model and prompt reconstruction for
greedy/batch calls, retaining its asset compatibility checks and PCM conversion.
This is an implementation detail with CPU regression coverage, confined to
`experiments/`; no production seam, schema, provider abstraction or behavior
change is needed. No unsupported generation control is sent.

This audit inspected source and release bytes. It did not run a real ai-core
experiment, evaluate generated prosody or establish a root cause.
