"""
All LLM prompt templates for the pipeline.
Kept here so they can be tuned without touching agent code.
"""

# ── Topic Selection ──────────────────────────────────────────────────────────
TOPIC_SELECTOR_PROMPT = """\
You are a YouTube content strategist. Given the following list of trending news headlines, \
select the SINGLE best topic for a 3-5 minute YouTube news explainer video.

Choose based on:
1. Mass appeal — broad audience interest across age groups
2. Clear narrative potential — the story can be explained and visualized
3. NOT overly political, partisan, or polarizing
4. NOT pure breaking news that will be outdated within hours — prefer story-driven topics

Trending Headlines:
{headlines}

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
  "selected_headline": "the exact headline you chose",
  "explanation": "2-sentence reason why this is the best YouTube topic"
}}
"""

# ── Script Writing ───────────────────────────────────────────────────────────
SCRIPTWRITER_PROMPT = """\
You are an expert YouTube scriptwriter for a news explainer channel with 1 million subscribers. \
Write an engaging, conversational script for a 3-4 minute narrated video.

Topic: {topic}

Language instruction: {language_directive}

Requirements:
- Open with a compelling hook in the first 15 seconds that makes viewers want to keep watching
- Use a neutral, professional news anchor tone — informative but conversational
- Divide into exactly 6 scenes using |SCENE_1|, |SCENE_2|, |SCENE_3|, |SCENE_4|, |SCENE_5|, |SCENE_6| as markers
- Each scene: 40-55 words of narration (30-40 seconds when spoken at a natural pace)
- Close scene 6 with a clear call-to-action: ask viewers to like, subscribe, and comment with their opinion
- Use active voice, short sentences, and no jargon
- Total word count: 380-450 words
- NO politically charged language, NO unverified speculation presented as fact

Output ONLY the script text with scene markers. Example format:
|SCENE_1|
Hook text here...

|SCENE_2|
Context text here...
"""

# ── SEO Metadata ─────────────────────────────────────────────────────────────
SEO_OPTIMIZER_PROMPT = """\
You are a YouTube SEO expert. Based on the topic and script excerpt below, \
create fully optimized YouTube video metadata.

Language instruction: {language_directive}

Topic: {topic}
Script excerpt (first 200 words):
{script_excerpt}

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
  "title": "Engaging SEO title — max 100 characters, include 2-3 high-search keywords, no clickbait",
  "description": "Write 3 paragraphs:\\n1. Compelling 2-sentence summary with main keywords.\\n2. Bullet-style key points covered (start each with •).\\n3. Call to action + subscribe request.\\nAdd 6-8 relevant hashtags on the final line (e.g. #WorldNews #BreakingNews #News).",
  "tags": ["tag1", "tag2", "tag3", "...", "exactly 15 tags total, mix broad and specific"],
  "thumbnail_text": "3-5 ALL CAPS bold words for the thumbnail — must create curiosity or urgency"
}}
"""

# ── Scene Image Prompt Generation ────────────────────────────────────────────
IMAGE_PROMPT_GENERATOR_PROMPT = """\
You are a cinematic AI image director. Given the video script below, write ONE \
Flux AI image generation prompt for EACH of the 6 scenes.

Script:
{script}

Rules for each prompt:
- Write a rich, detailed description of a SINGLE photorealistic scene (50–80 words)
- Start with the main subject and setting, then add lighting, mood, camera angle, and style
- Use cinematic language: "golden hour lighting", "aerial drone shot", "shallow depth of field", \
"photorealistic 8K", "cinematic wide angle", "dramatic chiaroscuro"
- Match the mood of the narration — tense scenes get dramatic lighting, hopeful scenes get \
bright warm tones
- DO NOT include any real person's likeness, recognizable political figures, or copyrighted logos
- DO NOT use words like "news anchor", "reporter", or "journalist" (triggers safety filters)
- Use abstract or environmental representations of news topics \
(e.g. "a glowing globe surrounded by digital data streams" for global tech news)
- Landscape orientation is required: wide, cinematic framing

Respond with ONLY a JSON array of exactly 6 strings (no markdown fences, no extra text), \
in scene order:
["prompt for scene 1", "prompt for scene 2", "prompt for scene 3", "prompt for scene 4", "prompt for scene 5", "prompt for scene 6"]
"""

# ── History Storytelling Prompts ──────────────────────────────────────────────

HISTORY_TOPIC_PICKER_PROMPT = """\
You are a historian and YouTube content strategist. Pick ONE fascinating, \
under-represented true history story that would captivate a general audience.

{era_filter}{theme_filter}\
Requirements:
- Must be a real, documented historical event or figure
- Prefer stories that are surprising, little-known, or have a dramatic turning point
- Avoid extremely well-known stories (no Battle of Waterloo, no Moon Landing)
- Must have strong visual storytelling potential (places, objects, dramatic moments)
- Avoid graphic violence or content that would be demonetized

Respond with ONLY a JSON object (no markdown fences):
{{
  "topic": "concise name of the historical story or figure",
  "era": "time period (e.g. Ancient Rome, Medieval Europe, 19th Century)",
  "region": "geographic region (e.g. Ottoman Empire, West Africa, East Asia)",
  "hook": "one dramatic sentence that opens the video and grabs the viewer immediately"
}}
"""

HISTORY_RESEARCHER_PROMPT = """\
You are an expert historian. Write a detailed research brief for the following \
historical topic, drawing on your knowledge to provide accurate, engaging facts.

Topic: {topic}
Era: {era}
Region: {region}

Provide a thorough brief covering:
- Key figures involved (name, role, why they matter)
- Chronological timeline of 4-6 pivotal events with approximate dates
- 2-3 surprising or little-known facts most people don't know
- The turning point — the single moment that changed everything
- The lasting legacy or lesson for today

Respond with ONLY a JSON object (no markdown fences):
{{
  "topic": "{topic}",
  "era": "{era}",
  "region": "{region}",
  "key_figures": [{{"name": "...", "role": "...", "significance": "..."}}],
  "timeline": [{{"date": "...", "event": "...", "impact": "..."}}],
  "surprising_facts": ["fact1", "fact2", "fact3"],
  "turning_point": "The single most dramatic moment and why it changed everything",
  "legacy": "Why this story matters today in 2-3 sentences"
}}
"""

HISTORY_SCRIPTWRITER_PROMPT = """\
You are a master documentary scriptwriter in the tradition of Ken Burns. \
Write an emotionally gripping, factually grounded narration script for a \
5-minute YouTube history video.

Topic: {topic}
Era: {era}
Region: {region}

Research brief:
{research_brief}

Opening hook: {hook}

Language instruction: {language_directive}

Requirements:
- Open scene 1 with the exact hook sentence provided, then expand
- Epic, intimate documentary tone — like a narrator speaking directly to the viewer
- Divide into exactly 6 scenes using |SCENE_1|...|SCENE_6| markers
- Each scene: 50-65 words of narration
- Scenes 1-2: Hook + historical context, set the stage
- Scenes 3-4: Rising tension, key events, key figures
- Scene 5: The turning point / climax
- Scene 6: Aftermath, legacy, reflection. Close with: "If you found this story \
remarkable, subscribe for more hidden histories that changed the world."
- Active voice, vivid sensory language, short punchy sentences mixed with longer rhythm
- NO anachronisms, NO speculation presented as fact

Output ONLY the script text with scene markers.
"""

HISTORY_IMAGE_PROMPT_GENERATOR_PROMPT = """\
You are an AI art director specialising in historical documentaries. \
Given the history video script and era below, write ONE image generation \
prompt for EACH of the 6 scenes.

Era: {era}
Region: {region}
Script:
{script}

Rules:
- Each prompt must evoke the historical period's authentic visual language
- Choose the MOST appropriate style for the era from:
    * Ancient/Classical: "detailed oil painting, Renaissance style, dramatic chiaroscuro"
    * Medieval: "illuminated manuscript style, hand-painted, rich jewel tones, gold leaf detail"
    * Early Modern (1400-1700): "Dutch Golden Age oil painting, Rembrandt lighting"
    * 18th-19th Century: "photorealistic oil painting, John Singer Sargent style, warm palette"
    * Early Photography Era (1840-1920): "sepia-toned daguerreotype photograph, aged paper texture"
    * 20th Century: "cinematic documentary still, archival photograph style, grainy film"
- Describe the scene setting, atmosphere, and composition (50-70 words per prompt)
- NO real identifiable person faces — use silhouettes, backs, or symbolic representations
- NO modern objects visible in historical scenes
- Landscape/wide establishing shots preferred; close-up still-life for detail scenes
- Append "landscape orientation, 16:9 aspect ratio, high detail" to every prompt

Respond with ONLY a JSON array of exactly 6 strings (no markdown fences):
["prompt for scene 1", "prompt for scene 2", ..., "prompt for scene 6"]
"""
