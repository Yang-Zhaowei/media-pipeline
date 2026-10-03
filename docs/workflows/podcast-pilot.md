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
| Speech invocation | `media-pipeline speech validate` and `speech render` ([CLI contract](../dev/render-entry-point.md#command-line)) | Author `speech.json`, review its performance-unit plan, and supply the existing runtime configuration |
| Whole-episode speech | Existing renderer assembles caller-approved units into WAV + SRT + timeline | Choose and approve unit boundaries; evaluate long-form production stability under Issue #8 |
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
   `media-pipeline speech validate speech.json` command. Review the plan, then
   use `media-pipeline speech render speech.json --output <new_run_dir>` with
   the existing runtime configuration. The caller authors the units and
   instructions; the CLI preserves them.
4. Use the produced audio durations to place each visual page. Maintain a small
   editing sheet: chapter, segment, script file, visual file, WAV, SRT, start,
   end and inter-segment pause. It can be a plain table, not a new schema or
   workflow engine. Generate visual content from the outline, but finalize its
   timing only after audio exists.
5. Use the renderer's `final/final.wav`, `final/final.srt`, and
   `final/timeline.json` to assemble the video with the chosen finishing tool.
   The renderer already assembles speech and shifts captions by integer-frame
   offsets; use the final timeline to place visuals.
6. Revise one paragraph in the authored script, validate and render into a new
   run directory, update visual timing from the new timeline, and export again.
   Accept the trial only after full listening and visual /
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

Script invocation and speech assembly are available through the existing CLI
and renderer. Evaluate their use in the real trial, including long-form
stability and revision handling. Any remaining reproducible gap must pass the
feature admission rule; reuse existing tools first and stop once it is closed.

No HTTP, MCP, ASR, Vision, generic orchestration, job queue, environment
consolidation, HTML renderer, or NLE integration is proposed for this repository.
