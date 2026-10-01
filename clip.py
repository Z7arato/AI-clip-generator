import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import av  # pip install av
import cv2  # pip install opencv-python
import yt_dlp
from faster_whisper import WhisperModel  # pip install faster-whisper
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

# ==========================================
# MONKEY-PATCH FOR PYAV (Fixes metadata_errors TypeError)
# ==========================================
_original_av_open = av.open

def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)

av.open = _patched_av_open


# ==========================================
# CONFIGURATION
# ==========================================
YOUTUBE_URL = "https://www.youtube.com/watch?v=Xyo-mst-qPE"
OUTPUT_DIR = Path("./youtube_clips")

# Whisper transcription. Caption accuracy depends mostly on this model.
#   "large-v3"        -> most accurate, any language, slow on CPU
#   "distil-large-v3" -> close to large-v3 accuracy, ~6x faster, ENGLISH ONLY
#   "base"            -> fast but makes lots of mistakes (the old setting)
WHISPER_MODEL_SIZE = "distil-large-v3"
WHISPER_LANGUAGE = "en"     # None = auto-detect (then use "large-v3")
WHISPER_PROMPT = None       # e.g. "Guests: Jordan Peterson, Lex Fridman. Terms: Anthropic, LLM"
# If YouTube still returns 403, let yt-dlp use your logged-in browser cookies:
# "chrome", "firefox", "edge", ...
COOKIES_FROM_BROWSER = None

# Video download: only fetched AFTER clips are chosen, capped to save time/disk.
VIDEO_MAX_HEIGHT = 720      # 720 is plenty for Shorts/Reels; 1080 roughly doubles the size
KEEP_SOURCE = False         # delete the downloaded source video when finished

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_CLIP_SECONDS = 30   # hard limit for spoken content
LEAD = 0.15             # breathing room before first word
TAIL = 0.30             # breathing room after last word
OVERLAP_PAD = 0.50      # extra overlap padding (s) added to start & end boundaries

PAUSE_GAP = 0.35        # silence (s) that counts as a sentence break
MAX_SENTENCE_WORDS = 30

# Beautiful Modern Captions Configuration
CAPTIONS = True
CAPTION_WORDS_PER_LINE = 2        # Show 2-3 words at a time for high pacing
CAPTION_FONT = "Impact"           # Bold high-impact font standard for viral reels
HIGHLIGHT_COLOR = "&H0000FFFF&"   # Vibrant Yellow text color for active word highlight
MAX_SUBTITLE_DURATION = 2.2       # Hard cap (seconds) to prevent subtitles from hanging on screen

SCORE_WEIGHTS = {"hook": 0.35, "story": 0.20, "value": 0.20, "engagement": 0.25}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ViralClip(BaseModel):
    title: str
    start: float
    end: float
    hook_category: str
    hook: int = Field(ge=1, le=10)
    story: int = Field(ge=1, le=10)
    value: int = Field(ge=1, le=10)
    engagement: int = Field(ge=1, le=10)
    loop_score: int = Field(ge=1, le=10)
    reason: str


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not extract a video ID from: {url}")


# ==========================================
# PROFANITY & N-WORD CENSOR FILTER
# ==========================================
PROFANITY_PATTERNS = [
    r"\bn[i1!l*]+g[e3r*]+\b",  # matches variants of the n-word
    r"\bf[u*]+ck\b",
    r"\bs[h*]+it\b",
    r"\bc[u*]+nt\b",
    r"\bb[i*]+tch\b"
]
def censor_text(text: str) -> str:
    # Direct replacement for stubborn words/phrases Whisper frequently transcribes
    bad_words_map = {
        "nigga": "n****",
        "nigger": "n*****",
        "fuck": "f****",
        "shit": "s***",
        # Add any other specific words from your video here
    }
    
    words_in_text = text.split()
    censored_words = []
    for w in words_in_text:
        clean_w = w.lower().strip(".,!?")
        if clean_w in bad_words_map:
            # Preserve original punctuation if possible or just swap
            censored_words.append(bad_words_map[clean_w])
        else:
            censored_words.append(w)
            
    return " ".join(censored_words)

# ==========================================
# TRANSCRIPT (faster-whisper base model)
# ==========================================
def _base_ydl_opts() -> dict:
    opts = {
        "overwrites": True,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
        "no_warnings": True,
        "quiet": True,
    }
    if COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
    return opts


def _pick_device():
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_youtube_words(url: str):
    """Audio-only download (~10-20 MB) -> 16 kHz mono wav -> faster-whisper."""
    device, compute_type = _pick_device()
    print("[1/3] Downloading audio only...")

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = _base_ydl_opts() | {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmpdir) / "raw.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        raw = next(Path(tmpdir).glob("raw.*"))
        audio_path = Path(tmpdir) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
             "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
            check=True,
        )

        print(f"   Transcribing with faster-whisper '{WHISPER_MODEL_SIZE}' on {device}...")
        model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type,
                             cpu_threads=os.cpu_count() or 4)
        segments, info = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=WHISPER_PROMPT,
        )
        print(f"-> Language: {info.language} (p={info.language_probability:.2f})")

        timed_words = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    word_str = html.unescape(w.word).strip()
                    if word_str:
                        timed_words.append({
                            "w": censor_text(word_str),
                            "s": round(w.start, 3),
                            "e": round(w.end, 3),
                        })

    if not timed_words:
        raise ValueError("Transcript was empty.")

    print(f"-> Extracted {len(timed_words)} words with word-level timestamps.")
    return timed_words


def download_source_video(url: str, video_id: str) -> Path:
    """Download the video once, capped at VIDEO_MAX_HEIGHT, using yt-dlp's own downloader.
    (download_ranges makes yt-dlp hand the googlevideo URL to ffmpeg, which YouTube
    rejects with 403.) Called only after clips have been chosen."""
    target = OUTPUT_DIR / f"source_{video_id}.mp4"
    if target.exists() and target.stat().st_size > 0:
        print(f"[3/3] Reusing existing download: {target.name}")
        return target

    print(f"[3/3] Downloading video (max {VIDEO_MAX_HEIGHT}p)...")
    h = VIDEO_MAX_HEIGHT
    ydl_opts = _base_ydl_opts() | {
        "format": f"bv*[height<={h}]+ba/b[height<={h}]/b",
        "format_sort": ["vcodec:h264", "acodec:aac"],
        "merge_output_format": "mp4",
        "outtmpl": str(OUTPUT_DIR / f"source_{video_id}.%(ext)s"),
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not target.exists():
        raise FileNotFoundError(f"Expected {target} after download but it is missing.")
    return target


def _est_duration(word: str) -> float:
    return 0.25 + 0.04 * len(word)


def build_sentences(words):
    sentences, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["s"] - (w["s"] + _est_duration(w["w"]))) if nxt else 99
        if re.search(r"[.!?]['\")\]]*$", w["w"]) or gap >= PAUSE_GAP or len(cur) >= MAX_SENTENCE_WORDS:
            last = cur[-1]
            sentences.append({
                "s": round(cur[0]["s"], 2),
                "e": round(min(last["e"], last["s"] + _est_duration(last["w"])), 2),
                "t": " ".join(x["w"] for x in cur),
            })
            cur = []
    return sentences


def snap_to_sentences(start: float, end: float, sentences):
    limit = MAX_CLIP_SECONDS - LEAD - TAIL
    starts = [s["s"] for s in sentences]
    i = min(range(len(sentences)), key=lambda k: abs(starts[k] - start))
    new_start = starts[i]
    candidates = [k for k in range(i, len(sentences)) if sentences[k]["e"] - new_start <= limit]
    if candidates:
        j = min(candidates, key=lambda k: abs(sentences[k]["e"] - end))
        new_end = sentences[j]["e"]
    else:
        new_end = new_start + limit
    return new_start, new_end


# ==========================================
# GEMINI ANALYZER (HIGH-VIRALITY STRATEGY)
# ==========================================
def generate_with_retry(client, prompt: str, attempts_per_model: int = 4):
    for model in GEMINI_MODELS:
        for attempt in range(1, attempts_per_model + 1):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
            except errors.APIError as e:
                if e.code == 404:
                    print(f"-> {model} not available for this account, skipping.")
                    break
                if e.code not in (429, 500, 503, 504):
                    raise
                wait = 3 * 2 ** (attempt - 1)
                print(f"  {model}: {e.code} {str(e.message)[:120]}")
                print(f"  retry {attempt}/{attempts_per_model} in {wait}s...")
                time.sleep(wait)
        else:
            print(f"-> {model} stayed overloaded, trying next model...")
    raise RuntimeError("No Gemini model was available right now. Try again in a few minutes.")


def weighted_score(clip: dict) -> float:
    return round(sum(clip[k] * w for k, w in SCORE_WEIGHTS.items()), 1)


def analyze_viral_clips(sentences: list):
    print("[2/3] Analyzing transcript with high-virality AI prompt...")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    compact = [{"s": x["s"], "e": x["e"], "t": x["t"]} for x in sentences]
    prompt = f"""
You are an elite short-form algorithmic growth hacker for TikTok, Instagram Reels, and YouTube Shorts.
Analyze the transcript below and isolate up to 10 clips optimized specifically to maximize 100% completion rate and replay loops.

TRANSCRIPT:
{json.dumps(compact, separators=(",", ":"))}

STRICT SELECTION CRITERIA FOR MAXIMUM VIRALITY:
1. DURATION: Must be between 20 and 30 seconds max. Never pick clips under 18 seconds or over 30 seconds.
2. THE 3-SECOND INSTANT HOOK: The selected start sentence MUST be provocative, emotionally charged, highly counterintuitive, or drop the viewer directly into the middle of an intense argument/story. Absolutely reject slow introductions, background explanations, or greetings.
3. OPEN LOOP RETENTION: The segment must build up a strong narrative arc or cliffhanger resolution that forces users to watch till the absolute last second.
4. BOUNDARIES: Start precisely at sentence start "s" and end at sentence end "e".
5. REPLAYABILITY: Prioritize segments where the ending ties seamlessly back into the beginning thought.

SCORING MATRIX (1 to 10):
- hook: Power of the opening 3 seconds to stop scrolling
- story: Cohesiveness and satisfying payoff within 30 seconds
- value: Emotional resonance, shock value, or practical insight
- engagement: Likelihood of prompting comments, arguments, or shares
- loop_score: Seamless circular looping potential

Return ONLY a valid JSON object containing a "clips" list structured as follows:

{{
  "clips": [
    {{
      "title": "short_snake_case_title",
      "start": 120.50,
      "end": 148.20,
      "hook_category": "Contradiction",
      "hook": 10,
      "story": 9,
      "value": 9,
      "engagement": 10,
      "loop_score": 9,
      "reason": "Why this specific hook halts the scroll immediately."
    }}
  ]
}}
"""

    response = generate_with_retry(client, prompt)

    parsed = json.loads(response.text)
    raw_clips = parsed.get("clips", []) if isinstance(parsed, dict) else parsed

    validated = []
    for item in raw_clips:
        try:
            clip = ViralClip(**item).model_dump()
        except ValidationError as e:
            print(f"-> Skipping a malformed clip: {str(e).splitlines()[0]}")
            continue
        if clip["end"] <= clip["start"]:
            print(f"-> Skipping '{clip['title']}': end is not after start.")
            continue

        new_start, new_end = snap_to_sentences(clip["start"], clip["end"], sentences)
        clip["start"], clip["end"] = round(new_start, 2), round(new_end, 2)
        clip["score"] = weighted_score(clip)
        validated.append(clip)

    validated.sort(key=lambda c: c["score"], reverse=True)
    print(f"-> Selected {len(validated)} candidate clip(s), best first:")
    for c in validated:
        print(f"   {c['score']:>4}  {c['title']} [{c['hook_category']}] "
              f"(hook {c['hook']}, story {c['story']}, value {c['value']}, engagement {c['engagement']}, loop {c['loop_score']})")
    return validated


# ==========================================
# BEAUTIFUL SUBTITLES (Hormozi Style ASS)
# ==========================================
def ass_time(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def write_ass(words, width: int, height: int, path: Path):
    fontsize = int(height * 0.055)     # Slightly larger for high impact
    margin_v = int(height * 0.15)      # Positioned nicely in the lower-middle safe zone
    outline = max(3, fontsize // 12)   # Thick professional text border

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        # Style definition: White main color, Black outline, solid drop shadow, center bottom alignment (Alignment 2)
        f"Style: Cap,{CAPTION_FONT},{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},2,5,40,40,{margin_v},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )

    def clean(word):
        word = re.sub(r"^>>", "", word).strip()
        return word.replace("{", "").replace("}", "").upper()

    events = []
    chunks = [words[i:i + CAPTION_WORDS_PER_LINE] for i in range(0, len(words), CAPTION_WORDS_PER_LINE)]
    
    for ci, chunk in enumerate(chunks):
        next_chunk_start = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else None
        
        for wi, active in enumerate(chunk):
            start = active["s"]
            if wi + 1 < len(chunk):
                end = chunk[wi + 1]["s"]
            else:
                end = active["e"]
                if next_chunk_start is not None:
                    end = min(max(end, start + 0.1), next_chunk_start)
            
            # CRITICAL FIX: Enforce max duration cap so subtitles never hang on screen
            if end - start > MAX_SUBTITLE_DURATION:
                end = start + MAX_SUBTITLE_DURATION

            if end <= start:
                continue

            pieces = []
            for j, w in enumerate(chunk):
                cleaned = clean(w['w'])
                if not cleaned:
                    continue
                if j == wi:
                    # Pop animation effect with custom highlight color
                    pieces.append(
                        f"{{\\c{HIGHLIGHT_COLOR}\\fscx112\\fscy112\\t(0,70,\\fscx100\\fscy100)}}"
                        f"{cleaned}{{\\r}}"
                    )
                else:
                    pieces.append(cleaned)
            if pieces:
                events.append(
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{' '.join(pieces)}"
                )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


# ==========================================
# VIDEO: Direct trimming & caption burning with overlap padding
# ==========================================
def render_clip(source_video: Path, clip: dict, words, out_path: Path):
    """Cut the clip from the already-downloaded source with local ffmpeg (no network)."""
    clip_start = max(0.0, clip["start"] - LEAD - OVERLAP_PAD)
    clip_end = clip["end"] + TAIL + OVERLAP_PAD
    duration = clip_end - clip_start

    cap = cv2.VideoCapture(str(source_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    words_rel = [
        {"w": w["w"], "s": w["s"] - clip_start, "e": w["e"] - clip_start}
        for w in words
        if clip_start - 0.01 <= w["s"] < clip_end
    ]

    ass_path = Path("temp_subs.ass")
    use_captions = CAPTIONS and bool(words_rel)
    if use_captions:
        write_ass(words_rel, width, height, ass_path)

    # -ss before -i = fast seek; with re-encoding it is still frame-accurate,
    # and output timestamps restart at 0 so the subtitle times line up.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{clip_start:.3f}", "-i", str(source_video), "-t", f"{duration:.3f}"]
    if use_captions:
        cmd += ["-vf", f"subtitles={ass_path.name}"]

    # Audio normalization filter targeting -14 LUFS standard
    cmd += ["-af", "loudnorm=I=-14:LRA=11:TP=-1.5"]

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out_path)
    ]

    subprocess.run(cmd, check=True)
    ass_path.unlink(missing_ok=True)


def main():
    video_id = extract_video_id(YOUTUBE_URL)

    words = get_youtube_words(YOUTUBE_URL)
    sentences = build_sentences(words)
    print(f"-> Grouped into {len(sentences)} sentences.")
    clips = analyze_viral_clips(sentences)
    if not clips:
        print("No usable clips found.")
        return

    source_video = download_source_video(YOUTUBE_URL, video_id)
    try:
        print("   Rendering clips with captions & normalized audio...")
        for idx, clip in enumerate(clips[:10], 1):
            safe_title = re.sub(r"[^\w\-]+", "_", clip.get("title", f"clip_{idx}")).strip("_")
            out_file = OUTPUT_DIR / f"{idx:02d}_{clip['score']}_{safe_title}.mp4"
            print(f"  Rendering [{out_file.name}] ({clip['start']}s -> {clip['end']}s)...")
            render_clip(source_video, clip, words, out_file)
    finally:
        if not KEEP_SOURCE:
            source_video.unlink(missing_ok=True)

    print(f"\nDone! Clips saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
