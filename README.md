# YouTube Clip Generator

Turn a long YouTube video into short, captioned clips for TikTok, Reels and Shorts.

The script transcribes the video, asks Gemini to pick the most "viral" 20-30 second moments, then cuts each one with animated word-by-word captions and normalized audio.

## How it works

1. **Transcribe**: downloads the audio only, then runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) locally to get word-level timestamps.
2. **Pick clips**: sends the sentence-grouped transcript to Gemini, which returns up to 10 scored clips (hook, story, value, engagement, loop score).
3. **Snap to sentences**: clip boundaries are moved to real sentence starts and ends so clips don't cut mid-word.
4. **Render**: downloads the video once (capped at 720p by default), cuts each clip with ffmpeg, burns in Hormozi-style captions, normalizes audio to -14 LUFS, then deletes the source video.

Clips are saved to `youtube_clips/`, named by rank and score, e.g. `01_9.1_the_big_reveal.mp4`.

## Requirements

- Python 3.10 or newer
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
- A [Gemini API key](https://aistudio.google.com/apikey)
- About 3 GB of RAM free and 1.5 GB of disk for the Whisper model. An NVIDIA GPU is optional; without one, transcription runs on CPU.

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Set your API key as an environment variable (never put it in the code):

```powershell
# Windows PowerShell (current session)
$env:GEMINI_API_KEY = "your-key-here"
```

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"
```

## Usage

1. Open `clip.py` and set `YOUTUBE_URL` near the top.
2. Run:

```bash
python clip.py
```

The first run downloads the Whisper model (about 1.5 GB) and caches it, so later runs skip that.

## Configuration

All settings are at the top of `clip.py`.

| Setting | Default | What it does |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `distil-large-v3` | Transcription model. Use `large-v3` for non-English, `small.en` or `base.en` on slow machines. |
| `WHISPER_LANGUAGE` | `"en"` | Set to `None` to auto-detect (pair with `large-v3`). |
| `WHISPER_PROMPT` | `None` | List names and jargon here to fix spelling mistakes in captions. |
| `VIDEO_MAX_HEIGHT` | `720` | Max video resolution to download. Use `1080` for sharper clips. |
| `KEEP_SOURCE` | `False` | Keep the downloaded source video after rendering. |
| `COOKIES_FROM_BROWSER` | `None` | e.g. `"chrome"`, to use your browser's cookies if YouTube blocks downloads. |
| `MAX_CLIP_SECONDS` | `30` | Hard limit on clip length. |
| `CAPTIONS` | `True` | Burn captions into the clips. |
| `CAPTION_WORDS_PER_LINE` | `2` | Words shown at once. |
| `CAPTION_FONT` | `Impact` | Caption font (must be installed on your system). |
| `GEMINI_MODELS` | list | Models tried in order if one is unavailable or overloaded. Edit it to match models your account can use. |

## Troubleshooting

**`403 Forbidden` when downloading**
- Update yt-dlp: `pip install -U "yt-dlp[default]"`.
- Set `COOKIES_FROM_BROWSER = "chrome"` (or `"firefox"`, `"edge"`).
- Install a JavaScript runtime such as [Deno](https://deno.com), which recent yt-dlp versions need for YouTube.

**Captions have wrong words**
- Use a bigger Whisper model (`distil-large-v3` or `large-v3`).
- Add names and terms to `WHISPER_PROMPT`.
- Background music and crosstalk lower accuracy for any model.

**Transcription is slow or runs out of memory**
- Switch to `small.en` or `base.en`.

**Caption font looks wrong**
- `Impact` must be installed on your system. Change `CAPTION_FONT` to any installed bold font.

## Notes

- Only clip content you own or have permission to reuse. Downloading videos may be against YouTube's Terms of Service.
- Whisper and Gemini can both make mistakes. Check clips before you post them.
