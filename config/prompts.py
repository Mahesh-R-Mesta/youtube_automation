"""
All LLM prompt templates for the spiritual content pipeline.
Kept here so they can be tuned without touching agent code.

Supported scriptures:
  bhagavad_gita  — chapters, slokas, Gita philosophy
  upanishads     — Isha, Kena, Katha, Mundaka, Mandukya, and others
  atharva_veda   — mantras, hymns, healing & cosmic knowledge
  mahabharata    — stories, characters, moral dilemmas from the epic
  puranas        — mythology, deity stories from Vishnu, Shiva, Devi Bhagavata

Content modes:
  sloka    — explain a specific verse/mantra with its Sanskrit text
  story    — narrate a scripture-based story
  teaching — explain a spiritual concept or philosophy

Image styles:
  tanjore     — traditional South Indian Tanjore painting (gold leaf, vivid jewel tones)
  vedic       — ancient manuscript / palm leaf aesthetic (sepia, ochre, ink illustration)
  cosmic      — celestial / divine light (nebula, starfield, radiant aura)
  minimalist  — modern spiritual minimalism (soft pastels, clean geometry, lotus motifs)
"""

# ── 1. Scripture & Topic Auto-Picker ────────────────────────────────────────
SCRIPTURE_PICKER_PROMPT = """\
You are an expert in Indian scriptures and spiritual content creation for Instagram.
Your task is to choose ONE compelling spiritual topic suitable for a short Instagram Reel.

{scripture_filter}\
Available scriptures: Bhagavad Gita, Upanishads, Atharva Veda, Mahabharata stories, Puranas.
Available modes: sloka (explain a verse), story (scripture narrative), teaching (philosophy concept).

Selection criteria:
- Must have universal appeal — wisdom that resonates across cultures and ages
- Must be visually rich — the content should inspire beautiful imagery
- Prefer topics that are profound yet accessible to someone with no prior spiritual knowledge
- Mix of lesser-known gems and timeless classics

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
  "scripture": "one of: bhagavad_gita | upanishads | atharva_veda | mahabharata | puranas",
  "mode": "one of: sloka | story | teaching",
  "topic": "concise topic name, e.g. 'Bhagavad Gita 2:47 — Nishkama Karma' or 'The Birth of Ganesha'",
  "scripture_ref": "specific reference, e.g. 'Chapter 2, Verse 47' or 'Vishnu Purana, Book 1'",
  "hook": "one captivating sentence that opens the Reel and immediately grabs attention",
  "selection_reason": "brief reason why this topic is ideal for Instagram right now"
}}
"""

# ── 2. Sanskrit Text Extractor ───────────────────────────────────────────────
SANSKRIT_EXTRACTOR_PROMPT = """\
You are a Sanskrit scholar. For the given spiritual topic, provide the primary Sanskrit \
text and its transliteration.

Topic: {topic}
Scripture reference: {scripture_ref}
Mode: {mode}

For sloka mode: provide the actual verse in Sanskrit.
For story/teaching mode: provide a relevant Sanskrit title, key term, or opening invocation.

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
  "sanskrit_devanagari": "the Sanskrit text in Devanagari script (Unicode)",
  "transliteration": "IAST or readable Roman transliteration",
  "word_by_word": "brief word-by-word meaning (one line)",
  "english_meaning": "clear 1-2 sentence English translation or summary",
  "context": "2-3 sentences of scriptural context — who spoke it, to whom, and why"
}}
"""

# ── 3. Sloka Explainer Script ────────────────────────────────────────────────
SLOKA_EXPLAINER_PROMPT = """\
You are a spiritual storyteller who makes ancient Indian wisdom accessible to modern \
audiences. Write a narration script for an Instagram Reel explaining a Sanskrit sloka.

Topic: {topic}
Scripture: {scripture_ref}
Sanskrit: {sanskrit_devanagari}
Meaning: {english_meaning}
Context: {context}
Hook: {hook}

Requirements:
- Open with the exact hook sentence provided, then expand naturally
- Speak in a warm, conversational Indian English tone — like a wise friend sharing wisdom
- Divide into exactly 6 scenes using |SCENE_1| through |SCENE_6| markers
- Each scene: 30–40 words of narration (suitable for ≤90 second total Reel)
- Scene 1: Hook + introduce the verse dramatically
- Scene 2: What the Sanskrit words mean literally
- Scene 3–4: Deep teaching — what it truly means for daily life
- Scene 5: A relatable modern-day example that brings the teaching alive
- Scene 6: Closing reflection. End with: "Jai Shri Krishna. Like and follow for daily wisdom."
- Use active voice, vivid imagery, natural cadence — no jargon
- Total word count: 200–240 words

Output ONLY the script with scene markers. No titles, no extra commentary.
"""

# ── 4. Story Narrator Script ─────────────────────────────────────────────────
STORY_NARRATOR_PROMPT = """\
You are a master storyteller of Indian scriptures. Write a narration script for an \
Instagram Reel retelling a story from the scriptures.

Topic: {topic}
Scripture: {scripture_ref}
Context: {context}
Hook: {hook}

Requirements:
- Open with the exact hook sentence provided
- Use a cinematic, immersive storytelling voice — as though narrating an epic film
- Divide into exactly 6 scenes using |SCENE_1| through |SCENE_6| markers
- Each scene: 30–40 words (suitable for ≤90 second total Reel)
- Scene 1: Hook — set the stage dramatically
- Scene 2: Introduce the main characters and setting
- Scene 3–4: Heart of the story — conflict, divine intervention, or pivotal moment
- Scene 5: Resolution — what happened and why it matters
- Scene 6: Spiritual lesson. End with: "Jai Shri Krishna. Like and follow for daily wisdom."
- Rich sensory language, short punchy sentences, build tension
- Total word count: 200–240 words

Output ONLY the script with scene markers. No titles, no extra commentary.
"""

# ── 5. Teaching Explainer Script ─────────────────────────────────────────────
TEACHING_EXPLAINER_PROMPT = """\
You are a renowned spiritual teacher making India's ancient philosophy accessible to \
modern seekers worldwide. Write a narration script for an Instagram Reel on a spiritual concept.

Topic: {topic}
Scripture: {scripture_ref}
Sanskrit key term: {sanskrit_devanagari} ({transliteration})
Context: {context}
Hook: {hook}

Requirements:
- Open with the exact hook sentence provided
- Warm, clear, insightful tone — like a wise teacher speaking to a modern audience
- Divide into exactly 6 scenes using |SCENE_1| through |SCENE_6| markers
- Each scene: 30–40 words (suitable for ≤90 second total Reel)
- Scene 1: Hook — pose a profound question or observation
- Scene 2: What the ancient scriptures say about this concept
- Scene 3–4: Break it down — layers of meaning, practical wisdom
- Scene 5: How to apply this teaching in daily modern life
- Scene 6: Inspiring close. End with: "Jai Shri Krishna. Like and follow for daily wisdom."
- Explain Sanskrit terms naturally within the narration
- Total word count: 200–240 words

Output ONLY the script with scene markers. No titles, no extra commentary.
"""

# ── 6. Instagram Metadata Generator ─────────────────────────────────────────
SPIRITUAL_METADATA_PROMPT = """\
You are an Instagram growth expert specialising in spiritual content. \
Generate metadata for a spiritual Reel.

Topic: {topic}
Scripture reference: {scripture_ref}
Mode: {mode}
Script excerpt (first 120 words):
{script_excerpt}

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
  "title": "Short title for the Reel cover — max 60 characters, compelling and clear",
  "caption": "Engaging Instagram caption (max 2200 chars). Include:\\n1. Thought-provoking opening (2 sentences).\\n2. The spiritual insight in 3-4 sentences.\\n3. Call to action: ask to save for daily reflection and share with loved ones.\\n4. Sanskrit blessing close, e.g. 'Namaste 🙏' or 'Jai Shri Krishna 🕉️'.\\nTotal: 150-250 words.",
  "hashtags": ["30 hashtags: mix scripture-specific (#BhagavadGita #Upanishads), spiritual (#AncientWisdom #VedicKnowledge #HinduPhilosophy), yogic (#Dharma #Karma #Moksha #Vedanta), and engagement tags (#DailyWisdom #SpiritualQuotes #InnerPeace #Spirituality #IndianMythology #SanatanaDharma)"]
}}
"""

# ── 7. Spiritual Image Prompt Generator ──────────────────────────────────────
SPIRITUAL_IMAGE_PROMPT = """\
You are an AI art director specialising in Indian spiritual visual art. \
Write ONE Pixazo Flux image generation prompt for EACH of the 6 scenes.

Image style: {image_style}
Scripture: {scripture}
Script:
{script}

Style guide for "{image_style}":
{style_description}

Rules for each prompt:
- Rich, visually detailed description (50–70 words)
- Start with the main subject and divine setting, then add lighting, color palette, and style
- Apply the specified art style consistently across all 6 prompts
- Portrait orientation: tall, vertical compositions (9:16 aspect) — reference "portrait 9:16 vertical"
- Append "no text, no letters, no watermarks" at the end of every prompt
- Evoke sacred, mystical atmosphere appropriate to Indian spiritual themes
- No real people's likenesses; no violence or disturbing imagery

Respond with ONLY a JSON array of exactly 6 strings (no markdown fences, no extra text):
["prompt for scene 1", "prompt for scene 2", "prompt for scene 3", \
"prompt for scene 4", "prompt for scene 5", "prompt for scene 6"]
"""

# ── Style descriptions injected into SPIRITUAL_IMAGE_PROMPT ─────────────────
IMAGE_STYLE_DESCRIPTIONS = {
    "tanjore": (
        "Traditional South Indian Tanjore (Thanjavur) painting style: "
        "rich jewel tones (deep red, cobalt blue, forest green), intricate gold leaf embellishments, "
        "frontal deity portraits, detailed gold borders and jewelry ornamentation, "
        "flat decorative composition with textured fabrics and raised gesso relief, "
        "luminous warm glow, traditional Hindu iconography"
    ),
    "vedic": (
        "Ancient Vedic manuscript and palm leaf text aesthetic: "
        "aged parchment and palm leaf texture background, hand-drawn ink illustration style, "
        "ochre, sepia, and burnt umber warm tones, sacred geometry borders and decorative ink elements, "
        "folk art quality with expressive linework, soft oil lamp or candlelight illumination"
    ),
    "cosmic": (
        "Celestial divine cosmic artwork: "
        "deep space nebula background (indigo, violet, gold), radiant divine light from center, "
        "ethereal floating sacred figures or symbols in cosmic space, starfield and galaxy elements, "
        "golden aura and halo effects, glowing ethereal particles, "
        "8K digital art quality, cinematic cosmic perspective"
    ),
    "minimalist": (
        "Modern minimalist spiritual illustration: "
        "soft pastel palette (pale gold, ivory, dusty rose, sage green), "
        "clean geometric and mandala-inspired composition, subtle lotus or Om motifs, "
        "generous negative space, contemporary sacred art aesthetic, "
        "flat style with delicate linework, calming and meditative atmosphere"
    ),
}
