"""
app.py — Ami on Hugging Face Spaces (Docker + Flask + SSE streaming)
Port 7860 requis par HF Spaces Docker.


"""

import os, re, json, time, random, hashlib, traceback
from collections import deque
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import httpx

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL         = "gemma-4-26b-a4b-it"
AI_STUDIO_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
DATA_DIR      = Path(__file__).parent / "data"

# ── Gemma calls ───────────────────────────────────────────────────────
def _api_key():
    key = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_AI_STUDIO_KEY secret is not set.")
    return key

def _post(payload: dict) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{AI_STUDIO_URL}?key={_api_key()}", json=payload)
    if not resp.is_success:
        raise RuntimeError(f"AI Studio {resp.status_code}: {resp.text[:300]}")
    return resp.json()

def _parse(response: dict) -> str:
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = next((p.get("text","") for p in parts if p.get("text") and not p.get("thought")), "")
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text).strip()
    return text

def gemma_call(prompt: str, max_tokens=400, temperature=0.75,
               sys_instr: str = None) -> str:
    """Appel AI Studio. Si sys_instr est fourni, on l'essaie en systemInstruction.
    Si l'API retourne 500 (non supporté), on retombe sur un seul user message."""
    gen_cfg = {
        "maxOutputTokens": max_tokens,
        "temperature":     min(temperature, 1.0),
        "topP":            0.95,
        "topK":            64,
    }
    if sys_instr:
        payload_with_sys = {
            "contents":          [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig":  gen_cfg,
            "systemInstruction": {"parts": [{"text": sys_instr}]},
        }
        try:
            return _parse(_post(payload_with_sys))
        except RuntimeError as e:
            if "500" not in str(e) and "400" not in str(e):
                raise
            # Fallback : concaténer system + user en un seul message
    payload = {
        "contents": [{"role": "user", "parts": [{"text":
            (sys_instr + "\n\n" + prompt) if sys_instr else prompt
        }]}],
        "generationConfig": gen_cfg,
    }
    return _parse(_post(payload))

def gemma_vision_call(image_b64: str, prompt: str, max_tokens=300, temperature=0.6) -> str:
    mime = "image/jpeg"
    raw  = image_b64
    if "," in image_b64 and image_b64.startswith("data:"):
        header, raw = image_b64.split(",", 1)
        if "image/png"  in header: mime = "image/png"
        if "image/webp" in header: mime = "image/webp"
        if "image/gif"  in header: mime = "image/gif"
    return _parse(_post({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": mime, "data": raw}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": min(temperature, 1.0),
            "topP": 0.95,
            "topK": 64,
        },
    }))

def extract_json(text: str):
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# ── Catalogue ─────────────────────────────────────────────────────────
_state = None

def get_state():
    global _state
    if _state: return _state
    cat  = json.loads((DATA_DIR / "ami_catalogue_updated.json").read_text("utf-8"))
    anec = json.loads((DATA_DIR / "ami_anecdotes.json").read_text("utf-8"))
    seen, all_items = set(), []
    for key in ["films", "series", "novels"]:
        for item in cat.get(key, []):
            if item.get("id") and item["id"] not in seen:
                seen.add(item["id"]); all_items.append(item)
    for item in anec.get("anecdotes", []):
        if item.get("id") and item["id"] not in seen:
            seen.add(item["id"]); all_items.append(item)
    items_by_id = {i["id"]: i for i in all_items}
    all_papers  = cat.get("meta", {}).get("papers", [])
    counts      = {}
    for i in all_items:
        if i.get("mechanism"): counts[i["mechanism"]] = counts.get(i["mechanism"], 0) + 1

    # ── Filtre ≥10 items : on ne propose à Gemma que les mécanismes avec
    # un catalogue suffisant pour éviter l'effet "même item ressort toujours"
    # (ex: 1 seul item social_surrogate = Friends à chaque reco lonely).
    MIN_ITEMS_PER_MECHANISM = 10
    papers = [p for p in all_papers if counts.get(p["id"], 0) >= MIN_ITEMS_PER_MECHANISM]
    excluded = [p["id"] for p in all_papers if counts.get(p["id"], 0) < MIN_ITEMS_PER_MECHANISM]
    if excluded:
        print(f"[startup] mechanisms excluded (<{MIN_ITEMS_PER_MECHANISM} items): "
              f"{', '.join(f'{m}({counts.get(m,0)})' for m in excluded)}", flush=True)
    print(f"[startup] active mechanisms: "
          f"{', '.join(p['id'] + '(' + str(counts.get(p['id'],0)) + ')' for p in papers)}",
          flush=True)

    def tag(n):
        if n >= 50: return "RICH catalogue"
        if n >= 20: return "good catalogue"
        if n >= 10: return "limited catalogue"
        return "limited"
    mechanism_block = "\n".join(
        f'- "{p["id"]}": {p["mechanism"]} ({tag(counts.get(p["id"],0))}) — {p.get("description","")[:150]}'
        for p in papers
    )
    moods = [m["key"] for m in cat.get("meta", {}).get("moods", [])] or [
        "overwhelmed","anxious","sad","angry","exhausted","lonely","empty","lost",
        "guilty","ashamed","discouraged","bored","nostalgic","grieving","melancholic",
        "cynical","doubtful","confused","restless","self_angry","envious",
        "embarrassed","self_disgust","meaningless",
    ]
    # System instruction partagée entre psy et librarian
    sys_instr = f"""You are Ami's AI engine — a science-backed well-being app.
You will be called for two tasks: psychological decoding OR recommendation writing.

=== PSYCHOLOGICAL MECHANISMS (use these exact IDs) ===
{mechanism_block}

=== AVAILABLE MOOD KEYS ===
{", ".join(moods)}

=== HOW TO READ IMAGES ===
- MEME: the meme's tone IS the user's tone. Irony counts.
- PHOTO: atmosphere reveals mood. Lighting, framing, absences matter.
- SCREENSHOT: the situation is what they're processing (breakup, layoff...).
- TEXT QUOTE: THE QUOTE'S MEANING IS THE SIGNAL. Read like a literature student. Identify author if visible.
- ARTWORK: emotional weight, not art-historical context.

=== RECOMMENDATION WRITING RULES ===
personal_intro: max 12 words, speak to "you", warm and immediate like a friend,
NEVER summarize the work. If tone is "tender": no humor, no irony.
Always address the person as "you" with warmth and compassion. Never write "user".
thesis: 2-3 poetic compassionate sentences grounded in the primary mechanism's science,
address user as "you". If tone is "tender": quiet and gentle, no levity.
Always return ONLY valid JSON, no backticks, no extra text."""

    _state = dict(all_items=all_items, items_by_id=items_by_id, papers=papers,
                  mechanism_block=mechanism_block, mood_list_str=", ".join(moods),
                  sys_instr=sys_instr)
    return _state




# ── Safety ────────────────────────────────────────────────────────────
CRISIS_RE = [
    r"\b(kill|killing)\s+myself\b", r"\bend(ing)?\s+(my|it)\s+(life|all)\b",
    r"\bi\s+want\s+to\s+die\b", r"\bsuicid(e|al)\b",
    r"\bbetter\s+off\s+(dead|without\s+me|gone)\b", r"\bself[\s-]?harm\b",
    r"\bnever\s+wake\s+up\b",
]
DISTRESS_RE = [
    r"\bcan'?t\s+(go\s+on|do\s+this|take\s+(it|this))\b", r"\bhopeless\b",
    r"\bworthless\b", r"\bi\s+hate\s+(myself|my\s+life)\b",
    r"\bdeep(ly)?\s+depress(ed|ion)\b",
]
CRISIS_RESPONSE = {
    "crisis": True, "blocked": True,
    "message": "What you're carrying sounds incredibly heavy. Please reach out to someone trained for moments like this.",
    "resources": [
        {"region": "France",        "name": "3114",              "detail": "Numéro national de prévention du suicide — 24/7, gratuit"},
        {"region": "United States", "name": "988",               "detail": "Suicide & Crisis Lifeline — call or text"},
        {"region": "United Kingdom","name": "116 123",           "detail": "Samaritans — free, 24/7"},
        {"region": "International", "name": "findahelpline.com", "detail": "Find a crisis line in your country"},
    ],
    "items": [], "thesis": "", "thematic_data": [], "psych_profile": None,
}

def crisis_payload():
    return CRISIS_RESPONSE

def classify_safety(text, moods):
    if any(re.search(p, text, re.I) for p in CRISIS_RE):
        return {"level": "crisis", "method": "regex"}
    severe = {"grieving","self_disgust","self_angry","ashamed","guilty","meaningless"}
    distress_hit = any(re.search(p, text, re.I) for p in DISTRESS_RE)
    if not text:
        return {"level": "high_distress" if len(severe & set(moods)) >= 2 else "normal", "method": "mood-rule"}
    if not distress_hit and not (severe & set(moods)) and len(text) < 15:
        return {"level": "normal", "method": "regex"}
    return {"level": "high_distress" if distress_hit else "normal", "method": "regex"}

# ── Psychologist ──────────────────────────────────────────────────────
PSYCHOLOGIST_PROMPT = """You are the psychological engine of Ami, a science-backed well-being app.

The user has described how they feel. Decode their emotional state precisely
and choose the most fitting psychological mechanism from those listed in your context.

=== USER INPUT ===
Free text: {free_text}
Selected mood buttons: {selected_moods}
Birth year: {birth_year}

=== TASK ===
Analyze the user's input carefully — especially the free text, which often contains
nuances not captured by the mood buttons.

Return a JSON with:
- "emotion_core": a precise 3-6 word description of the user's actual emotional state
- "primary_mechanism": the single most fitting mechanism id (from your context)
- "secondary_mechanism": a second mechanism id for variety (must be different)
- "active_moods": list of 2-4 most relevant mood keys from your context
- "intensity": "low", "medium", or "high"
- "nostalgia_relevant": true if the user's birth year nostalgia window is relevant
- "nostalgia_decade": the cultural decade (e.g. "1990s") if nostalgia_relevant, else null
- "humour_relevant": true if humour is relevant
- "humour_word": the word we can joke about if humour_relevant, else null
- "surprise_ok": true if the user seems open to a counter-intuitive recommendation
- "include_anecdote": true if a short historical anecdote would help them right now; false if it would feel dissonant (e.g. grief, acute distress)
- "decoder_note": one sentence (max 15 words) explaining what you understood — address the person as "you", with warmth and compassion, never say "user"

Respond ONLY with valid JSON, no backticks:
{{"emotion_core":"...","primary_mechanism":"...","secondary_mechanism":"...","active_moods":[...],"intensity":"...","nostalgia_relevant":false,"nostalgia_decade":null,"humour_relevant":false,"humour_word":null,"surprise_ok":false,"include_anecdote":true,"decoder_note":"..."}}
""".strip()

PSYCHOLOGIST_MULTIMODAL_PROMPT = """You are the psychological engine of Ami, a science-backed well-being app.
The user shared an image AND optionally some text. Use the image reading guide from your context.
Read the IMAGE and the TEXT together as one signal.

=== USER INPUT ===
Free text: {free_text}
Selected mood buttons: {selected_moods}
Birth year: {birth_year}

(An image is attached. Read its emotional subtext using the image guide in your context.)

=== TASK ===
Analyze image + text together. The image is often the dominant emotional signal.

Return a JSON with:
- "image_read": one short sentence — what the image MEANS to share (not what it shows)
- "is_meme": true|false
- "is_quote": true if the image is primarily a text quote/passage
- "quote_author": the author if identifiable, else null
- "emotion_core": a precise 3-6 word description of the user's emotional state
- "primary_mechanism": the single most fitting mechanism id
- "secondary_mechanism": a second mechanism id (must be different)
- "active_moods": list of 2-4 most relevant mood keys
- "intensity": "low" | "medium" | "high"
- "nostalgia_relevant": true if birth-year nostalgia is relevant
- "nostalgia_decade": cultural decade if relevant, else null
- "humour_relevant": true if humour is relevant
- "humour_word": the word we can joke about if humour_relevant, else null
- "surprise_ok": true if open to counter-intuitive picks
- "include_anecdote": true if a historical anecdote would help; false if dissonant
- "decoder_note": one sentence (max 15 words) reflecting what you understood — address the person as "you" with warmth, never say "user". Mention author if visible.

Respond ONLY with valid JSON, no backticks:
{{"image_read":"...","is_meme":false,"is_quote":false,"quote_author":null,"emotion_core":"...","primary_mechanism":"...","secondary_mechanism":"...","active_moods":[...],"intensity":"...","nostalgia_relevant":false,"nostalgia_decade":null,"humour_relevant":false,"humour_word":null,"surprise_ok":false,"include_anecdote":true,"decoder_note":"..."}}
""".strip()


def _fallback(selected_moods):
    primary = "transport"
    if any(m in selected_moods for m in ["nostalgic","grieving","melancholic"]): primary = "nostalgia"
    elif any(m in selected_moods for m in ["lonely","exhausted"]): primary = "social_surrogate"
    elif any(m in selected_moods for m in ["bored","lost","empty"]): primary = "self_expansion"
    elif any(m in selected_moods for m in ["sad","grieving"]): primary = "benign_masochism"
    return {
        "emotion_core":        " / ".join(selected_moods[:2]) if selected_moods else "undefined",
        "primary_mechanism":   primary,
        "secondary_mechanism": "elevation",
        "active_moods":        selected_moods[:3],
        "intensity":           "medium",
        "nostalgia_relevant":  False,
        "nostalgia_decade":    None,
        "surprise_ok":         False,
        "include_anecdote":    True,
        "decoder_note":        "I sensed something difficult. Here are works that may help.",
    }

def run_psychologist(free_text, selected_moods, birth_year, mechanism_block, mood_list_str, papers=None, sys_instr=None):
    nostalgia_window = ""
    if birth_year:
        nostalgia_window = f"born {birth_year}, nostalgia window {birth_year+8}–{birth_year+16}"
    # Avec le filtre ≥10 items dans get_state(), mechanism_block contient déjà
    # uniquement les mécanismes utilisables (7 max). Pas besoin de re-filtrer.
    prompt = PSYCHOLOGIST_PROMPT.format(
        free_text       = free_text or "(no free text provided)",
        selected_moods  = ", ".join(selected_moods) if selected_moods else "(none selected)",
        birth_year      = nostalgia_window or "(not provided)",
        mechanism_block = mechanism_block,
        mood_list       = mood_list_str,
    )
    raw = gemma_call(prompt, max_tokens=180, temperature=0.70, sys_instr=sys_instr)
    return extract_json(raw) or _fallback(selected_moods)

def run_psychologist_multimodal(free_text, selected_moods, birth_year, image_b64, mechanism_block, mood_list_str, papers=None, sys_instr=None):
    nostalgia_window = ""
    if birth_year:
        nostalgia_window = f"born {birth_year}, nostalgia window {birth_year+8}–{birth_year+16}"
    # mechanism_block est déjà filtré à 7 mécas dans get_state()
    prompt = PSYCHOLOGIST_MULTIMODAL_PROMPT.format(
        free_text       = free_text or "(no free text — image only)",
        selected_moods  = ", ".join(selected_moods) if selected_moods else "(none selected)",
        birth_year      = nostalgia_window or "(not provided)",
        mechanism_block = mechanism_block,
        mood_list       = mood_list_str,
    )
    raw = gemma_vision_call(image_b64, prompt, max_tokens=200, temperature=0.65)
    data = extract_json(raw)
    if not data:
        fb = _fallback(selected_moods)
        fb["image_read"] = "(image received, could not decode)"
        return fb
    data.setdefault("image_read",          "")
    data.setdefault("is_meme",             False)
    data.setdefault("secondary_mechanism", "elevation")
    data.setdefault("active_moods",        selected_moods[:3])
    data.setdefault("intensity",           "medium")
    data.setdefault("nostalgia_relevant",  False)
    data.setdefault("nostalgia_decade",    None)
    data.setdefault("humour_relevant",     False)
    data.setdefault("humour_word",         None)
    data.setdefault("surprise_ok",         False)
    data.setdefault("decoder_note",        "")
    return data

# ── Librarian ─────────────────────────────────────────────────────────
_recent = deque(maxlen=12)
_seed   = None

def set_request_context(nonce=None):
    global _seed
    bucket = int(time.time() // (24*3600))
    _seed  = int(hashlib.sha256(f"{bucket}::{nonce or ''}".encode()).hexdigest()[:8], 16)

def _valid_decades_for_birth_year(birth_year):
    """
    Returns the set of decades (e.g. {"2000s", "2010s"}) covered by the user's
    nostalgia window [birth_year + 8, birth_year + 16].

    Example: born 2000 → window 2008–2016 → {"2000s", "2010s"}
    Example: born 1985 → window 1993–2001 → {"1990s", "2000s"}

    Returns an empty set if birth_year is missing or invalid.
    """
    try:
        by = int(birth_year)
    except (TypeError, ValueError):
        return set()
    if by < 1900 or by > 2025:
        return set()
    start_year = by + 8
    end_year   = by + 16
    decades = set()
    for y in range(start_year, end_year + 1):
        decades.add(f"{(y // 10) * 10}s")  # 2008 → "2000s", 2016 → "2010s"
    return decades

def filter_catalogue_for_profile(psych_profile, all_items, max_items=15, nonce=None, pool_multiplier=3):
    primary   = psych_profile.get("primary_mechanism", "")
    secondary = psych_profile.get("secondary_mechanism", "")
    moods     = set(psych_profile.get("active_moods", []))
    decade    = psych_profile.get("nostalgia_decade")
    tone      = psych_profile.get("tone", "warm")
    require_anecdote = bool(psych_profile.get("include_anecdote", False))

    # ── Filtre anecdotes ──
    # Si birth_year fourni → on calcule les décennies valides et on retire toutes
    #   les anecdotes hors fenêtre (ex: anecdote 1985 pour user né 2000 → exclue).
    # Si birth_year absent → on désactive complètement l'inclusion d'anecdotes
    #   (mieux vaut 4 films cohérents qu'une anecdote au hasard).
    birth_year     = psych_profile.get("_birth_year")
    valid_decades  = _valid_decades_for_birth_year(birth_year)

    if valid_decades:
        # On filtre les anecdotes : seules celles dont `decade` matche restent dans le pool
        all_items = [
            i for i in all_items
            if i.get("type") != "anecdote" or i.get("decade") in valid_decades
        ]
    else:
        # Sans birth_year, on retire toutes les anecdotes du catalogue éligible
        all_items = [i for i in all_items if i.get("type") != "anecdote"]
        require_anecdote = False  # on force la désactivation

    seed = _seed if _seed is not None else int(time.time())
    rng  = random.Random(seed)
    recent = set(_recent)

    def base_score(item):
        s = 0.0
        if item.get("mechanism") == primary:                          s += 10
        if item.get("mechanism") == secondary:                        s += 5
        if moods & set(item.get("moods", [])):                       s += 3
        # Bonus décennie : champ `decade` (anecdotes) ou `nostalgia_decades` (films)
        item_decades = set(item.get("nostalgia_decades", []))
        if item.get("decade"):
            item_decades.add(item.get("decade"))
        if decade and decade in item_decades:                        s += 4
        if item.get("surprise_factor", 1) >= 2:                      s += 1
        if tone == "tender" and item.get("surprise_factor", 1) >= 2: s -= 3
        if item.get("id") in recent:                                  s -= 6
        return s

    def jittered(item):
        return base_score(item) + rng.gauss(0, 1.5)

    pool_size = max_items * pool_multiplier
    pool = sorted(all_items, key=jittered, reverse=True)[:pool_size]

    if require_anecdote:
        anecdotes_in_pool = sum(1 for i in pool if i.get("type") == "anecdote")
        if anecdotes_in_pool < 2:
            anecdote_pool = [i for i in all_items if i.get("type") == "anecdote"]
            def anecdote_score(a):
                s = 0.0
                if moods & set(a.get("moods", [])): s += 5
                # Anecdotes use `decade` (singular string), films use `nostalgia_decades`
                a_decades = set(a.get("nostalgia_decades", []))
                if a.get("decade"):
                    a_decades.add(a.get("decade"))
                if decade and decade in a_decades: s += 4
                if a.get("id") in recent: s -= 6
                s += rng.gauss(0, 1.0)
                return s
            best_anecdotes = sorted(anecdote_pool, key=anecdote_score, reverse=True)
            existing_ids = {i["id"] for i in pool}
            injected = [a for a in best_anecdotes if a["id"] not in existing_ids][:2]
            pool = injected + pool[:max(0, pool_size - len(injected))]

    TEMP = 2.5
    weights = [max(0.01, 2 ** (base_score(i) / TEMP)) for i in pool]
    limits  = {"film": 5, "series": 4, "novel": 3, "anecdote": 1}
    counts  = {k: 0 for k in limits}
    result, used = [], set()

    attempts = 0
    while len(result) < max_items and attempts < pool_size * 4:
        attempts += 1
        if not pool: break
        pick = rng.choices(pool, weights=weights, k=1)[0]
        if pick["id"] in used: continue
        t = pick.get("type", "film")
        if counts.get(t, 0) >= limits.get(t, 4): continue
        result.append(pick)
        used.add(pick["id"])
        counts[t] = counts.get(t, 0) + 1

    if require_anecdote and not any(i.get("type") == "anecdote" for i in result):
        anecdotes_in_pool = [i for i in pool if i.get("type") == "anecdote" and i["id"] not in used]
        if anecdotes_in_pool:
            best_anecdote = max(anecdotes_in_pool, key=base_score)
            removable = [i for i in result if i.get("type") != "anecdote" and i.get("mechanism") != primary]
            if not removable:
                removable = [i for i in result if i.get("type") != "anecdote"]
            if removable:
                worst = min(removable, key=base_score)
                result = [i for i in result if i["id"] != worst["id"]]
                result.append(best_anecdote)

    return result

def make_compact_catalogue(items, max_items=15):
    """Format ultra-compact : id|type|title(year)|mechanism — 40% moins de tokens."""
    lines = []
    for item in items[:max_items]:
        mech  = (item.get("mechanism") or "")[:20]
        year  = item.get("year", "")
        title = (item.get("title") or "")[:28]
        t     = (item.get("type") or "film")[0]  # f/s/n/a
        lines.append(f'{item["id"]}|{t}|{title}({year})|{mech}')
    return "\n".join(lines)

LIBRARIAN_PROMPT = """You are the recommendation engine of Ami, a science-backed well-being recommendation app.

=== PSYCHOLOGICAL PROFILE ===
Emotion: {emotion_core}
Primary mechanism: {primary_mechanism} — {primary_desc}
Secondary mechanism: {secondary_mechanism}
Moods: {active_moods}
Intensity: {intensity}
Nostalgia decade: {nostalgia_decade}
Humour : {humour_relevant} — {humour_word}
Open to surprise: {surprise_ok}
Note: {decoder_note}
Tone: {tone}

=== CATALOGUE (id | type | title | mechanism | moods) ===
{compact_catalogue}

=== TASK ===
Pick exactly 4 items from the catalogue above.
Rules:
- 2+ items matching primary mechanism
- 1 item matching secondary mechanism
- Vary types: pick {anecdote_rule}, then balance films/series/novels
- Always start with a movie.
- If nostalgia_relevant, include 1 item from that decade
- If surprise_ok, include 1 counter-intuitive pick

For each item write "personal_intro": a SHORT HOOK (max 12 words) displayed just before the title on the recommendation card. Speak directly to "you", make it feel like a friend saying "hey, look at this" — warm, immediate, never explanatory.
- if tone is "tender" → no humor, no irony, no surprise. Soft, quiet, kind only.
- do NOT summarize the work (the catalogue already provides description + why); just create the personal bridge between the user's state and this specific item.
Write a "thesis" (2-3 poetic and compassionate sentences grounded in the primary mechanism's science (it should help better understand the primary mechanism), including joke or playful advice if permited, mention the author of the quote if provided, address user as "you", connect the birth year to the release year if helpful).
- If tone is "tender" → tender and quiet, no levity.

Return a JSON object with exactly:
- "items": array of 4 objects each with "id" and "personal_intro"
- "thesis": string (2-3 sentences)
""".strip()

def run_librarian(psych_profile: dict, all_items: list, items_by_id: dict, papers: list, sys_instr: str = None) -> dict:
    """
    Version du notebook Kaggle : UN SEUL appel Gemma qui fait sélection + intros + thesis.
    Plus rapide en moyenne et 2x moins de risque de timeout vs split en 2 appels.
    """
    filtered = filter_catalogue_for_profile(
        psych_profile, all_items, max_items=15, nonce=psych_profile.get("_nonce"))
    compact_catalogue = make_compact_catalogue(filtered, max_items=15)

    primary      = psych_profile.get("primary_mechanism", "transport")
    secondary    = psych_profile.get("secondary_mechanism", "elevation")
    tone         = psych_profile.get("tone", "warm")
    primary_desc = next((p.get("description", "")[:100] for p in papers if p["id"] == primary), "")

    if psych_profile.get("include_anecdote", False):
        anecdote_rule = "1 anecdote (REQUIRED — already present in the catalogue above)"
    else:
        anecdote_rule = "no anecdote needed (focus on films/series/novels)"

    prompt = LIBRARIAN_PROMPT.format(
        emotion_core        = psych_profile.get("emotion_core", ""),
        primary_mechanism   = primary,
        primary_desc        = primary_desc,
        secondary_mechanism = secondary,
        active_moods        = ", ".join(psych_profile.get("active_moods", [])),
        intensity           = psych_profile.get("intensity", "medium"),
        nostalgia_decade    = psych_profile.get("nostalgia_decade") or "n/a",
        surprise_ok         = psych_profile.get("surprise_ok", False),
        humour_relevant     = psych_profile.get("humour_relevant", False),
        humour_word         = psych_profile.get("humour_word", "") or "",
        tone                = tone,
        decoder_note        = psych_profile.get("decoder_note", ""),
        compact_catalogue   = compact_catalogue,
        anecdote_rule       = anecdote_rule,
    )

    raw = gemma_call(prompt, max_tokens=350, temperature=0.78, sys_instr=sys_instr)
    data = extract_json(raw)

    if not data or not data.get("items"):
        return {"items": [], "thesis": "Ami has chosen these works with care.",
                "thematic_data": [], "error": "librarian_parse_failed"}

    # Enrichir items avec les métadonnées complètes du catalogue
    enriched = []
    for chosen in data.get("items", [])[:4]:
        item_id = chosen.get("id", "")
        meta = items_by_id.get(item_id, {})
        if not meta:
            for k in items_by_id:
                if item_id in k or k in item_id:
                    meta = items_by_id[k]
                    break
        if not meta:
            continue

        item_type = meta.get("type", "")
        if item_type == "anecdote":
            description  = meta.get("anecdote", "") or meta.get("overview", "")
            # Static why for anecdotes — nostalgia as the underlying mechanism
            artistic_why = (
                "Revisiting a shared cultural memory is a gentle way to feel "
                "continuous with your past self and less alone in the present."
            )
            # Use the nostalgia paper citation (Sedikides & Wildschut, 2018)
            nostalgia_info = next((p for p in papers if p.get("id") == "nostalgia"), None)
            item_paper = (nostalgia_info.get("paper") if nostalgia_info
                          else "Sedikides & Wildschut, 2018 — Finding Meaning in Nostalgia. Review of General Psychology.")
        else:
            # Prefer the Gemma-rewritten overview when available, fallback to original overview
            description  = meta.get("overview_rewritten") or meta.get("overview", "")
            artistic_why = meta.get("why", "")
            item_paper   = meta.get("paper", "")

        enriched.append({
            **meta,
            "personal_intro": chosen.get("personal_intro", "") or chosen.get("personal_why", ""),
            "description":    description,
            "why":             artistic_why,
            "paper":          item_paper,
        })

    # Cooldown : mémoriser les items recommandés pour la prochaine requête
    for i in enriched:
        _recent.append(i.get("id"))

    return {
        "items":         enriched,
        "thesis":        data.get("thesis", ""),
        "thematic_data": data.get("thematic_data", []),
    }


# ── AJOUT v2 : Fallback ────────────────────────
# Utilisé UNIQUEMENT si run_librarian lève une exception après tous ses retries.


FALLBACK_THESIS = {
    "transport": {
        "warm":   "When the world presses too tightly, story is a doorway. These works don't fix what hurts — they let you step sideways into another life for a while, and that pause is what the mind needs to settle.",
        "tender": "Sometimes we need to be carried somewhere else, gently. These works open a quiet door. You don't have to do anything but follow.",
    },
    "nostalgia": {
        "warm":   "Memory is not the past — it's the soft architecture we return to when the present feels uncertain. These works tune you back to a time when the world felt knowable, and that recognition is itself a form of comfort.",
        "tender": "Memory holds you when nothing else can. These works are pieces of a time that knew you. Let them sit beside you.",
    },
    "elevation": {
        "warm":   "Witnessing moral beauty — a small act of grace, an unexpected kindness — releases something neurologists call elevation. These works are not about heroes; they are about the quiet decency that proves the world is still worth your tenderness.",
        "tender": "Tenderness still exists. These works are small reminders of it, gently offered.",
    },
    "self_expansion": {
        "warm":   "When we feel stuck, we are usually too close to ourselves. These works enlarge the frame — new lives, new questions, new rooms in your own mind. The self that emerges on the other side is wider than the one that began.",
        "tender": "These works open quiet windows onto other lives. No urgency — just a wider horizon, when you're ready.",
    },
    "social_surrogate": {
        "warm":   "The mind does not always distinguish real company from imagined company — and that is a gift, not a deficit. These works will keep you company tonight, the way a friend would, without asking anything in return.",
        "tender": "These works will sit with you, the way a quiet friend would. Nothing required of you.",
    },
    "benign_masochism": {
        "warm":   "Sadness met head-on, in the safety of fiction, releases what we cannot release alone. These works will not lie to you — they will let you feel what is already there, and that honesty is the beginning of relief.",
        "tender": "These works don't look away from what hurts. Sometimes that gentle witness is exactly what the heart needs.",
    },
    "awe": {
        "warm":   "Awe shrinks the self in the most healing way — what felt enormous becomes part of something larger still. These works open the ceiling of your worry and let the sky in.",
        "tender": "These works invite a quiet largeness. Nothing demanded — just space to breathe, gently.",
    },
    "incongruence": {
        "warm":   "Humour, when it is honest, is a release valve the brain installs against absurdity. These works find the angle where pain becomes laughable — not to dismiss it, but to make it carryable.",
        "tender": "These works are gentle company. They don't force lightness, but they let it in when ready.",
    },
}

def _get_fallback_thesis(primary_mechanism: str, tone: str = "warm") -> str:
    entry = FALLBACK_THESIS.get(primary_mechanism) or FALLBACK_THESIS["transport"]
    return entry.get(tone) or entry.get("warm") or ""

def _make_fallback_intro(item: dict, tone: str = "warm") -> str:
    if tone == "tender":
        templates = [
            "A quiet companion for tonight.",
            "Sit with this one, gently.",
            "Let this one find you, slowly.",
            "Something soft, offered without urgency.",
        ]
    else:
        templates = [
            "This one met us where you are.",
            "You'll feel seen by this.",
            "A gentle bridge for right now.",
            "For the part of you that needs this.",
        ]
    idx = abs(hash(item.get("id", item.get("title", "")))) % len(templates)
    return templates[idx]

def run_librarian_fallback(psych_profile: dict, all_items: list, items_by_id: dict, papers: list = None) -> dict:
    """
    Fallback complet SANS LLM. Utilisé UNIQUEMENT si run_librarian lève une exception.
    Garantit qu'on a toujours 4 items à servir, même si AI Studio est down.
    """
    primary = psych_profile.get("primary_mechanism", "transport")
    tone    = psych_profile.get("tone", "warm")

    filtered = filter_catalogue_for_profile(
        psych_profile, all_items, max_items=15, nonce=psych_profile.get("_nonce"))

    # Sélectionner 4 items, garantir un film en premier
    selected = []
    films = [i for i in filtered if i.get("type") == "film"]
    if films:
        selected.append(films[0])
    for item in filtered:
        if item not in selected and len(selected) < 4:
            selected.append(item)

    if not selected:
        return {"items": [], "thesis": "Ami has chosen these works with care.",
                "thematic_data": [], "degraded": True}

    enriched = []
    for meta in selected[:4]:
        is_anec = meta.get("type") == "anecdote"
        if is_anec:
            description  = meta.get("anecdote", "") or meta.get("overview", "")
            artistic_why = (
                "Revisiting a shared cultural memory is a gentle way to feel "
                "continuous with your past self and less alone in the present."
            )
            nostalgia_info = next((p for p in (papers or []) if p.get("id") == "nostalgia"), None)
            item_paper = (nostalgia_info.get("paper") if nostalgia_info
                          else "Sedikides & Wildschut, 2018 — Finding Meaning in Nostalgia. Review of General Psychology.")
        else:
            description  = meta.get("overview_rewritten") or meta.get("overview", "")
            artistic_why = meta.get("why", "")
            item_paper   = meta.get("paper", "")

        enriched.append({
            **meta,
            "personal_intro": _make_fallback_intro(meta, tone),
            "description":    description,
            "why":            artistic_why,
            "paper":          item_paper,
        })

    for i in enriched:
        _recent.append(i.get("id"))

    return {
        "items":         enriched,
        "thesis":        _get_fallback_thesis(primary, tone),
        "thematic_data": [],
        "degraded":      True,
    }


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","model":MODEL})

@app.route("/api/recommend", methods=["POST","OPTIONS"])
def recommend():
    if request.method == "OPTIONS":
        return "", 204

    body           = request.get_json(silent=True) or {}
    free_text      = (body.get("free_text") or "").strip()
    selected_moods = body.get("selected_moods") or []
    birth_year     = body.get("birth_year")
    bypass_safety  = bool(body.get("continue_anyway", False))
    nonce          = body.get("nonce")
    image_b64      = body.get("image_b64")

    if not free_text and not selected_moods and not image_b64:
        return jsonify({"error": "Provide free_text, selected_moods, or image_b64"}), 400

    def generate():
        import threading
        rid = int(time.time() * 1000) % 100000
        t0_total = time.time()
        try:
            if not free_text and not selected_moods and not image_b64:
                yield sse("error", {"error": "Provide free_text, selected_moods, or image_b64"})
                return

            state = get_state()
            set_request_context(nonce=nonce)

            # ── Step 1 · Vision + Psychologist ────────────────────────
            if image_b64:
                yield sse("status", {"step":"vision_psy", "msg":"Reading your image and decoding emotion...", "elapsed":0})
                result_holder = {}
                def _run_vision_psy():
                    try:
                        result_holder["data"] = run_psychologist_multimodal(
                            free_text, selected_moods, birth_year, image_b64,
                            state["mechanism_block"], state["mood_list_str"])
                    except Exception as e:
                        result_holder["error"] = e
                th = threading.Thread(target=_run_vision_psy, daemon=True)
                th.start()
                i = 0
                while th.is_alive():
                    th.join(timeout=15.0)
                    if th.is_alive():
                        yield sse("status", {"step":"working", "msg":"Still reading your image...", "elapsed": round(time.time()-t0_total,1)})
                        i += 1
                if "error" in result_holder: raise result_holder["error"]
                psych_profile = result_holder["data"]

                enriched_text = (free_text+" " if free_text else "") + f"[image: {psych_profile.get('image_read','')}]"
                image_summary = {
                    "scene":          psych_profile.get("image_read",""),
                    "is_meme":        psych_profile.get("is_meme",False),
                    "emotional_tone": psych_profile.get("emotion_core",""),
                    "likely_mood":    (psych_profile.get("active_moods") or ["undefined"])[0],
                    "subtext":        psych_profile.get("decoder_note",""),
                }

                yield sse("status", {"step":"vision_psy_done", "msg":f"Decoded: {psych_profile.get('emotion_core','')}", "elapsed":round(time.time()-t0_total,1)})
                yield sse("vision_insight", {
                    "input_type":    "image+text" if free_text else "image",
                    "image_read":    psych_profile.get("image_read",""),
                    "understanding": psych_profile.get("image_read","") or psych_profile.get("decoder_note",""),
                    "emotion_core":  psych_profile.get("emotion_core",""),
                    "is_meme":       psych_profile.get("is_meme",False),
                    "is_quote":      psych_profile.get("is_quote",False),
                    "quote_author":  psych_profile.get("quote_author"),
                    "active_moods":  psych_profile.get("active_moods",[]),
                    "intensity":     psych_profile.get("intensity","medium"),
                    "decoder_note":  psych_profile.get("decoder_note",""),
                })
            else:
                image_summary = None
                enriched_text = free_text
                psych_profile = None

            # ── Step 2 · Safety ───────────────────────────────────────
            yield sse("status", {"step":"safety", "msg":"Checking safety signals...", "elapsed":round(time.time()-t0_total,1)})
            safety = classify_safety(enriched_text, selected_moods)


            if safety["level"] == "crisis" and not bypass_safety:
                yield sse("crisis", {**crisis_payload(), "safety": safety})
                return

            # ── Step 3 · Text-only psychologist ───────────────────────
            if psych_profile is None:
                yield sse("status", {"step":"psy", "msg":"Decoding your emotional state...", "elapsed":round(time.time()-t0_total,1)})
                result_holder = {}
                def _run_psy():
                    try:
                        result_holder["data"] = run_psychologist(
                            enriched_text, selected_moods, birth_year,
                            state["mechanism_block"], state["mood_list_str"],
                            papers=state["papers"], sys_instr=state.get("sys_instr"))
                    except Exception as e:
                        result_holder["error"] = e
                th = threading.Thread(target=_run_psy, daemon=True)
                th.start()
                i = 0
                while th.is_alive():
                    th.join(timeout=15.0)
                    if th.is_alive():
                        yield sse("status", {"step":"working", "msg":"Still decoding your emotional state...", "elapsed":round(time.time()-t0_total,1)})
                        i += 1
                if "error" in result_holder: raise result_holder["error"]
                psych_profile = result_holder["data"]


            # Tone tweak based on safety
            if safety["level"] in ("high_distress", "crisis"):
                psych_profile["surprise_ok"] = False
                psych_profile["tone"]        = "tender"
            else:
                psych_profile.setdefault("tone", "warm")
            psych_profile["_nonce"] = nonce
            psych_profile["_birth_year"] = birth_year

            yield sse("status", {"step":"psy_done", "msg":psych_profile.get("decoder_note","Understood."), "elapsed":round(time.time()-t0_total,1)})

            if not image_b64:
                yield sse("vision_insight", {
                    "input_type":    "text",
                    "image_read":    "",
                    "understanding": psych_profile.get("decoder_note","") or (free_text[:120] if free_text else ""),
                    "emotion_core":  psych_profile.get("emotion_core",""),
                    "is_meme":       False,
                    "is_quote":      False,
                    "quote_author":  None,
                    "active_moods":  psych_profile.get("active_moods",[]),
                    "intensity":     psych_profile.get("intensity","medium"),
                    "decoder_note":  psych_profile.get("decoder_note",""),
                })

            # ── Step 4 · Librarian ────────────────────────────────────
            yield sse("status", {"step":"librarian", "msg":"Curating works for you...", "elapsed":round(time.time()-t0_total,1)})
            result_holder = {}
            def _run_lib():
                try:
                    result_holder["data"] = run_librarian(
                        psych_profile, state["all_items"], state["items_by_id"], state["papers"],
                        sys_instr=state.get("sys_instr"))
                except Exception as e:
                    result_holder["error"] = e
            th = threading.Thread(target=_run_lib, daemon=True)
            th.start()
            heartbeat_msgs = [
                "Still working — Gemma is comparing your state to validated mechanisms...",
                "Selecting items from the catalogue...",
                "Writing personal explanations...",
                "Almost there...",
            ]
            i = 0
            while th.is_alive():
                th.join(timeout=15.0)
                if th.is_alive():
                    msg = heartbeat_msgs[min(i, len(heartbeat_msgs)-1)]
                    yield sse("status", {"step":"librarian_progress", "msg":msg, "elapsed":round(time.time()-t0_total,1)})
                    i += 1

            # ── AJOUT v2 : si librarian a planté, on bascule en fallback dégradé ──
            # Au lieu de raise (qui faisait apparaître une erreur côté front), on construit
            # une réponse complète sans LLM via filter_catalogue + templates pré-écrits.
            if "error" in result_holder:
                print(f"[req {rid}] librarian failed ({result_holder['error']}) — using non-LLM fallback", flush=True)
                result = run_librarian_fallback(
                    psych_profile, state["all_items"], state["items_by_id"], state["papers"])
            else:
                result = result_holder["data"]

            n_items = len(result.get("items",[]))


            yield sse("status", {"step":"librarian_done", "msg":f"Found {n_items} works for you.", "elapsed":round(time.time()-t0_total,1)})

            # ── Step 5 · Stream items one by one ──────────────────────
            for idx, item in enumerate(result.get("items",[])):
                yield sse("item", {"index":idx, "total":n_items, "item":item})
                time.sleep(0.4)

            # ── Step 6 · Final payload ────────────────────────────────
            t_total = time.time() - t0_total
            print(f"[req] done {round(time.time()-t0_total,1)}s", flush=True)
            yield sse("done", {
                "thesis":        result.get("thesis",""),
                "thematic_data": [],
                "psych_profile": {
                    "emotion_core":      psych_profile.get("emotion_core"),
                    "primary_mechanism": psych_profile.get("primary_mechanism"),
                    "decoder_note":      psych_profile.get("decoder_note"),
                    "intensity":         psych_profile.get("intensity"),
                    "surprise_ok":       psych_profile.get("surprise_ok"),
                    "tone":              psych_profile.get("tone","warm"),
                },
                "image_summary":  image_summary,
                "enriched_text":  enriched_text,
                "safety":         safety,
                "agentic":        False,
                "tool_trace":     [],
                "warning":        result.get("error"),
                "timings":        {"total_s": round(t_total,1)},
                "degraded":       bool(result.get("degraded")),
            })


        except Exception as e:
            print(f"[req] error: {type(e).__name__}", flush=True)
            yield sse("error", {"error": str(e)})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control":     "no-cache",
        "X-Accel-Buffering": "no",
        "Connection":        "keep-alive",
    })



# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False, threaded=True)