# First podcast-video workflow assessment

Status: evaluation plan requested by the owner, **not a new implementation
milestone**. Speech Pipeline v0 is closed. Use a real episode to decide which
small gaps need project code under the [admission rule](../ROADMAP.md).

## Intended workflow

Research and source notes -> topic discussion -> approved episode outline ->
chapter scripts and HTML visual pages -> narration WAV + SRT -> rendered visual
assets -> FFmpeg or NLE assembly -> reviewed final video.

The workflow is feasible with existing authoring and media tools plus the
verified speech components. The repository does not yet offer a single
arbitrary-manuscript-to-episode operation.

## Current capability and remaining work

| Step | Available now | Remaining work for the pilot |
| --- | --- | --- |
| Research, topic, outline | External research / writing tools and human review | Choose topic, audience, source material and chapter structure; retain source links |
| Narration script | External writing tools | Separate spoken text from headings, citations and stage directions; review pronunciations and approve segment boundaries |
| Voice selection | `CustomVoiceRequest(text, language, speaker, instruct)` and runtime speaker/language discovery | Select a supported voice and audition actual script material; only Chinese / Uncle_Fu has the recorded E2E acceptance |
| WAV + SRT per utterance | Existing production APIs and verified chain | Supply real text / voice through a small production-specific call site; the validation driver fixes its text and speaker |
| Whole-episode speech | No long-script splitting or episode assembly | First try manually approved short segments and existing assembly tools; only implement a small helper if that exposes a reproducible gap |
| HTML visual pages | External HTML authoring and browser rendering | Produce PNGs for static pages or video/frame sequences for animations; HTML itself is not an encoded video asset |
| Chapter timing | Per-utterance audio and alignment timing | Record chapter/segment start and end from final audio frame counts, including inserted pauses |
| Final video | External FFmpeg or chosen NLE | Combine rendered pages/clips, narration and subtitles; review timing, loudness, resolution and final export |

## Voice and script boundaries

`speaker` selects a model-supported preset; `instruct` is a natural-language
delivery instruction such as calm, clear narration at a moderate pace. It is
not a promise of an exact speaking rate, pitch, emotion strength, or duration.
The adapter exposes no numeric speed/pitch controls and no voice cloning or
VoiceDesign. Those require separate demonstrated needs, not a v0 redesign.

Start with one narrator. A full chapter may still need several short utterances;
no supported maximum manuscript length has been established by the short E2E
case. Review numbers, abbreviations, names and pronunciations in the spoken
script before synthesis. The current caption compiler preserves that script;
alternate spoken/display text is not an existing feature.

## Smallest useful pilot

1. Choose one topic, one finishing tool (FFmpeg or one NLE), and one agent host.
   A proposed first trial is a 3-5 minute single-narrator video with three
   chapters and static visual pages. These sizes are trial choices, not model
   limits or product defaults.
2. Approve the outline and script, then manually segment it into short, complete
   utterances. Keep stable chapter/segment names so one paragraph can be revised
   without rewriting the entire episode.
3. Audition the selected speaker on a real paragraph, then run the existing
   speech stages for each accepted segment. A narrowly scoped call site may
   pass script text, speaker and instruction into those APIs; do not repurpose
   or overwrite the immutable smoke fixture to supply user input.
4. Use the produced audio durations to place each visual page. Maintain a small
   editing sheet: chapter, segment, script file, visual file, WAV, SRT, start,
   end and inter-segment pause. It can be a plain table, not a new schema or
   workflow engine. Generate visual content from the outline, but finalize its
   timing only after audio exists.
5. Assemble with the chosen tool. For a standalone episode WAV + SRT, concatenate
   compatible audio and shift each segment's captions by its cumulative final
   audio start. Renumber subtitle cues. Never concatenate SRT text files without
   shifting their timestamps.
6. Revise one paragraph, regenerate that segment, recalculate later offsets,
   and export again. Accept the trial only after full listening and visual /
   subtitle synchronization review of the final encoded video.

For a common audio sample rate `R`, segment `i` begins at
`sum(previous final segment frames + inserted silence frames) / R`. Carry the
integer frame total until SRT millisecond rendering, rather than accumulating
rounded subtitle durations. Different audio formats need normalization before
assembly. Crossfades and speed changes alter timing; avoid them in the first
speech assembly, or account for their exact effect before placing subtitles.

## HTML pages and finishing

HTML can supply slide-like pages without creating a `.pptx`. A browser must
render them: static pages become image assets; animated pages become captured
clips or rendered frame sequences. Fix viewport, fonts, loaded resources and
animation start state. Use narration/chapter timing to determine asset duration.
Browser capture belongs to the production workflow, not the speech module.

FFmpeg provides established concatenation, overlay, subtitle rendering and
loudness filters. Its subtitle burn-in filter requires the appropriate build
support. A chosen NLE can instead handle visual placement and final editing;
verify its actual WAV/SRT import behavior during the pilot. Do not build NLE
automation before trying its native capabilities.

Audio Postprocess v0 performs trim/padding/fades, not loudness normalization,
music mixing or mastering. Those remain finishing tasks. Any edit that changes
audio duration or ordering must be reflected in subtitle and visual timing.

References: [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html),
[FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html),
[Qwen3-TTS upstream](https://github.com/QwenLM/Qwen3-TTS).

## Decision after the pilot

The first likely friction is supplying real script/voice parameters; the next
is repeatable segment assembly with correct subtitle offsets. These are
candidate small improvements, not approved scope. Measure them in the real
trial, reuse existing tools first, and stop once the demonstrated gap is closed.

No HTTP, MCP, ASR, Vision, generic orchestration, job queue, environment
consolidation, HTML renderer, or NLE integration is proposed for this repository.
