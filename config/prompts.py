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
