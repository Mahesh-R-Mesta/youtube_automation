## Plan: History Storytelling Pipeline

Pivot the pipeline from news explainer videos to **history storytelling videos with AI-animated images**. Instead of scraping live trends, the pipeline picks an interesting historical story, researches it deeply, writes a cinematic narrative script, generates Flux AI scene images styled as "historical artwork / documentary", and publishes to YouTube.

**Key changes vs current news pipeline:**
- `trend_agent` → `history_agent` (Wikipedia + Gemini to pick & research topics)
- `script_agent` → reuse core but with new history-focused prompts
- `visual_agent` → reuse with new art-direction prompts (oil painting / engraving / documentary style)
- Orchestrator flow updated: `pick_topic → research → write_script → voiceover → images → video → upload`
- New `ANIMATED_IMAGE_EFFECT` in video_editor: Ken Burns pan/zoom on each still to create illusion of motion
- YouTube category changes to `27` (Education) and tags/SEO tuned for history audience

---

**Steps**

### Phase 1 — New prompts (`config/prompts.py`)
1. Add `HISTORY_TOPIC_PICKER_PROMPT` — given an optional era/theme, Gemini picks one fascinating, under-covered true history story. Returns `{topic, era, region, hook}`.
2. Add `HISTORY_RESEARCHER_PROMPT` — given topic, Gemini writes a deep-research brief (key figures, timeline, surprising facts, 3-5 pivotal moments). Returns structured JSON.
3. Add `HISTORY_SCRIPTWRITER_PROMPT` — like current `SCRIPTWRITER_PROMPT` but: epic documentary tone (think Ken Burns), `|SCENE_N|` markers, 6 scenes × 50 words, opens with a dramatic hook, ends with a reflection/lesson. No CTA until scene 6.
4. Add `HISTORY_IMAGE_PROMPT_GENERATOR_PROMPT` — like current `IMAGE_PROMPT_GENERATOR_PROMPT` but: each prompt styles the scene as **one of**: oil painting, detailed engraving, sepia photograph, watercolour illustration, or cinematic documentary still — whichever fits the era/mood. No modern people.

*Depends on nothing — can be done immediately.*

### Phase 2 — `agents/history_agent.py` (NEW FILE)
5. `pick_history_topic(era: str | None, theme: str | None) → dict` — calls `HISTORY_TOPIC_PICKER_PROMPT`, returns `{topic, era, region, hook}`. Falls back to a built-in list of 20 curated topics if Gemini fails.
6. `research_topic(topic: dict) → dict` — calls `HISTORY_RESEARCHER_PROMPT`, returns structured research brief `{topic, era, region, key_figures, timeline_events, pivotal_moments, surprising_fact}`.

*Depends on Phase 1 prompts.*

### Phase 3 — Update `agents/script_agent.py`
7. Add `HistoryScript` dataclass (extends/replaces `VideoScript`) — same fields plus: `era: str`, `region: str`, `research_brief: dict`.
8. Add `write_history_script(research_brief: dict) → str` — uses `HISTORY_SCRIPTWRITER_PROMPT`. Injects key figures, dates, pivotal moments from brief.
9. Add `generate_history_video_script(era, theme) → HistoryScript` — new top-level function: `pick_history_topic → research_topic → write_history_script → generate_seo_metadata → generate_history_image_prompts`. Keep existing `generate_video_script()` untouched.

*Depends on Phases 1 & 2.*

### Phase 4 — Update `config/settings.py`
10. Add `youtube_history_category_id: str = "27"` (Education).
11. Add `history_era: Optional[str] = None` — optional CLI/env override (e.g. `"Ancient Rome"`).
12. Add `history_theme: Optional[str] = None` — optional topical focus (e.g. `"forgotten women"`).

*Parallel with Phases 1–3.*

### Phase 5 — Update `pipeline/orchestrator.py`
13. Add `HistoryProductionState(BaseModel)` — same as `VideoProductionState` plus `era`, `region`, `research_brief` fields.
14. Add `HistoryStorytellerFlow(Flow[HistoryProductionState])` — new flow class:
    - `init_run` (reuse from base)
    - `pick_and_research` → `history_agent.*`
    - `write_script` → `script_agent.generate_history_video_script`
    - `create_voiceover` → `voice_agent.generate_voiceover` (unchanged)
    - `fetch_visuals` → `visual_agent.generate_scene_images` (unchanged, new prompts handle art style)
    - `compile_video` → `video_editor.assemble_video` (unchanged)
    - `upload_to_youtube` → category `27` instead of `25`
15. Keep `YouTubeAutomationFlow` intact (news pipeline still usable).

*Depends on Phases 2 & 3.*

### Phase 6 — Animated image effect in `pipeline/video_editor.py`
16. Add `_ken_burns_clip(image_path, duration, direction) → ImageClip` — uses MoviePy v2 `resize`/crop transforms to slowly zoom in (1.0× → 1.12×) or pan left/right over the clip duration. Direction cycles: zoom-in, zoom-out, pan-left, pan-right.
17. Update `assemble_video` to accept `animated: bool = True` — when True, pass each scene through `_ken_burns_clip` instead of static `ImageClip`.

*Parallel with Phases 3–5.*

### Phase 7 — Update `main.py`
18. Add `--mode` flag: `news` (default, existing) | `history`.
19. Add `--era` and `--theme` optional flags.
20. When `--mode history`, instantiate `HistoryStorytellerFlow` and pass era/theme.

*Depends on Phase 5.*

### Phase 8 — Tests
21. `tests/test_history_agent.py` — mock Gemini for `pick_history_topic` and `research_topic`, assert returned dicts have required keys.
22. `tests/test_script_agent.py` — add `TestWriteHistoryScript` and `TestGenerateHistoryVideoScript` classes (mock `_call_gemini`).
23. `tests/test_video_editor.py` (NEW) — test `_ken_burns_clip` produces expected clip duration, and `assemble_video(animated=True)` completes without errors (mock audio).

*Depends on Phases 2, 3, 6.*

---

**Relevant files**

- `config/prompts.py` — add 4 new prompt constants; keep existing 4 intact
- `config/settings.py` — add `youtube_history_category_id`, `history_era`, `history_theme`
- `agents/history_agent.py` — NEW; public API: `pick_history_topic()`, `research_topic()`
- `agents/script_agent.py` — add `HistoryScript` dataclass + 3 new functions; do NOT touch `generate_video_script()`
- `pipeline/orchestrator.py` — add `HistoryProductionState` + `HistoryStorytellerFlow`; keep `YouTubeAutomationFlow`
- `pipeline/video_editor.py` — add `_ken_burns_clip()`, update `assemble_video(animated=True)`
- `main.py` — add `--mode`, `--era`, `--theme` flags
- `tests/test_history_agent.py` — NEW
- `tests/test_script_agent.py` — extend with history tests
- `tests/test_video_editor.py` — NEW

---

**Verification**

1. Run `pytest tests/ -q` — all existing 50 tests still pass (no regressions)
2. Run `pytest tests/test_history_agent.py tests/test_script_agent.py -v` — new tests pass
3. Run `python main.py --run-now --mode history --era "Ancient Rome"` — verify output/ directory gets audio + 6 images + compiled MP4
4. Inspect the MP4: check Ken Burns motion is visible, voice narration is synced, subtitles render correctly
5. Run `python main.py --run-now --mode news` — confirm news pipeline still works unmodified

---

**Decisions**

- **Research source**: Gemini only (no Wikipedia API calls) — Gemini 2.5 Flash has strong historical knowledge baked in; adding a Wikipedia fetch would be a nice-to-have but adds complexity
- **Backwards compatibility**: `YouTubeAutomationFlow` and `generate_video_script()` preserved unchanged — the `story_teller` branch only **adds** new code paths
- **Image style**: handled entirely via prompt engineering on `HISTORY_IMAGE_PROMPT_GENERATOR_PROMPT` — no new API needed; Pixazo Flux handles painting/engraving styles well
- **Ken Burns**: pure MoviePy v2 transform — no FFmpeg subprocess, no extra dependency
- **Category**: History → Education (`27`), not News & Politics (`25`)
- **OUT of scope**: voice cloning for a "narrator character", multi-language support, shorts format
