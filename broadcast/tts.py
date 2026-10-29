"""Script text -> Edge TTS speech + per-frame mouth timeline for render.mjs.

Usage: uv run --with edge-tts --with websockets --with pypinyin --with imageio-ffmpeg --with numpy \
         python broadcast/tts.py <script.txt> <out-dir> [--engine edge|doubao] [--voice ...] [--rate +0%] [--fps 30]

Writes <out-dir>/speech.wav (with silence before and after) and <out-dir>/timeline.json:
  { fps, duration, title, frames: [[level 0..1, mouthForm or null], ...], subtitles: [{start, end, text}] }
The script text is sent to the speech service (Microsoft Edge TTS, or Volcengine Doubao with
broadcast/.env); nothing else leaves the machine.
"""
import argparse
import asyncio
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # volc_tts.py; python -I leaves the script folder out
sys.stdout.reconfigure(errors="replace")  # a redirected console may not encode Chinese

import edge_tts
import imageio_ffmpeg
import numpy as np
from pypinyin import Style, pinyin

LEAD, TAIL = 0.6, 1.2          # seconds of idle before and after the speech
RATE = 16000                   # analysis sample rate
# mouthForm per vowel, matching VOWELS in packages/runtime/src/engine/kana.js
FORMS = {"a": 0.0, "i": -0.85, "u": 0.9, "e": -0.4, "o": 0.6}
SENTENCE_END = "。！？!?；;\n"
PUNCT = re.compile(r"[\s，。、！？!?；;：:,.“”\"'‘’（）()《》【】\-—…·]")
# Expression marks in the script, e.g. "[开心]今天是个好日子"; make.mjs knows what each key does.
TAGS = {
    "happy": ("开心", "高兴", "happy"), "surprised": ("惊讶", "吃惊", "surprised"), "sad": ("难过", "伤心", "sad"),
    "angry": ("生气", "愤怒", "angry"), "relaxed": ("平静", "放松", "relaxed"), "shy": ("害羞", "shy"),
    "halfLidded": ("无语", "眯眼", "halfLidded"), "wink": ("眨眼", "wink"), "neutral": ("正常", "默认", "neutral"),
}
TAG_OF = {alias.lower(): key for key, aliases in TAGS.items() for alias in aliases}
# Doubao 2.0 voice instructions (context_texts) for the same marks; neutral speaks plainly.
TONES = {
    "happy": "用开心、愉快的语气说", "surprised": "用惊讶的语气说", "sad": "用难过、低落的语气说",
    "angry": "用生气、不满的语气说", "relaxed": "用平静、放松的语气说", "shy": "用害羞、轻柔的语气说",
    "halfLidded": "用无奈的语气说", "wink": "用俏皮的语气说",
}
TAG_MARK = re.compile(r"[\[【]([^\]】]{1,12})[\]】]")
ANTICIPATE = 0.15             # seconds an expression starts before its first word


def vowel_of(final: str) -> str:
    """Main mouth shape of a pinyin final (no tone), e.g. 'iao' -> a, 'ong' -> o."""
    f = final.replace("v", "u").replace("ü", "u")
    if "a" in f:
        return "a"
    if "o" in f:
        return "o"
    if f in ("u", "un", "iu", "ue") or (f.startswith("u") and "e" not in f):
        return "u"
    if "e" in f:
        return "e"
    return "i"


def char_vowels(text: str) -> list[str]:
    """One vowel per spoken unit of a boundary word: Chinese characters by pinyin, other runs as 'a'."""
    out = []
    for token in re.findall(r"[一-鿿]|[A-Za-z]+|\d", text):
        if re.match(r"[一-鿿]", token):
            final = pinyin(token, style=Style.FINALS, strict=False)[0][0]
            out.append(vowel_of(final or "a"))
        else:
            out.extend("a" * max(1, len(token) // 2))
    return out or ["a"]


def split_script(raw: str) -> tuple[str, str]:
    """A first line starting with '#' is the title; the rest is spoken."""
    lines = raw.strip().splitlines()
    title = ""
    if lines and lines[0].lstrip().startswith("#"):
        title = lines.pop(0).lstrip("# ").strip()
    return title, "\n".join(line.strip() for line in lines if line.strip())


def take_tags(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Remove expression marks; return the clean text and (spoken characters before the mark, key)."""
    clean, tags, last = "", [], 0
    for m in TAG_MARK.finditer(text):
        clean += text[last:m.start()]
        last = m.end()
        key = TAG_OF.get(m.group(1).strip().lower())
        if key is None:
            print(f"unknown expression mark [{m.group(1)}] ignored; known: {', '.join(a[0] for a in TAGS.values())}")
            continue
        tags.append((len(PUNCT.sub("", clean)), key))
    return clean + text[last:], tags


def tone_segments(text: str) -> list[tuple[str, str | None]]:
    """Split the marked script into (clean text, voice instruction) runs, one per expression mark."""
    parts = TAG_MARK.split(text)          # [text, mark, text, mark, text, ...]
    segments, tone = [(parts[0], None)], None
    for mark, chunk in zip(parts[1::2], parts[2::2]):
        key = TAG_OF.get(mark.strip().lower())
        if key is not None:
            tone = TONES.get(key)
        segments.append((chunk, tone))
    # Merge neighbours with the same tone, so each session gets as much text as possible.
    merged = []
    for chunk, tone in segments:
        if merged and merged[-1][1] == tone:
            merged[-1] = (merged[-1][0] + chunk, tone)
        elif PUNCT.sub("", chunk):
            merged.append((chunk, tone))
    return merged


def tag_times(tags: list[tuple[int, str]], words: list[dict], end: float) -> list[dict]:
    """Time of each mark: where its next spoken character falls within the boundary words."""
    out, w, before = [], 0, 0
    for pos, key in tags:
        while w < len(words) and before + len(PUNCT.sub("", words[w]["text"])) <= pos:
            before += len(PUNCT.sub("", words[w]["text"]))
            w += 1
        if w < len(words):
            n = max(1, len(PUNCT.sub("", words[w]["text"])))
            t = words[w]["start"] + words[w]["dur"] * (pos - before) / n
        else:
            t = end
        out.append({"time": round(max(0.0, t + LEAD - ANTICIPATE), 3), "name": key})
    return out


def subtitle_units(text: str, limit: int = 16) -> list[str]:
    """Clauses for on-screen subtitles: split at any pause, then rejoin short clauses of one sentence."""
    units = []
    for sentence in re.findall(rf"[^{SENTENCE_END}]+[{SENTENCE_END}]?", text):
        current = ""
        for clause in re.findall(r"[^，,、：:]+[，,、：:]?", sentence.strip()):
            if current and len(PUNCT.sub("", current + clause)) > limit:
                units.append(current)
                current = ""
            current += clause
        if current.strip():
            units.append(current)
    return [u.strip() for u in units if PUNCT.sub("", u)]


def synthesize_doubao(segments: list[tuple[str, str | None]], voice: str | None, rate: str, out: Path) -> tuple[Path, list[dict]]:
    import volc_tts
    env = volc_tts.load_env(Path(__file__).resolve().parent / ".env")
    if voice:
        env["TTS_SPEAKER"] = voice
    speech_rate = int(round(float(rate.strip().rstrip("%") or 0)))
    pcm, words = asyncio.run(volc_tts.synthesize(segments, env, speech_rate=max(-50, min(100, speech_rate))))
    raw = out / "speech-raw.wav"
    with wave.open(str(raw), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(volc_tts.SAMPLE_RATE)
        f.writeframes(pcm)
    return raw, words


async def synthesize(text: str, voice: str, rate: str, mp3: Path) -> list[dict]:
    words = []
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    with mp3.open("wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({"start": chunk["offset"] / 1e7, "dur": chunk["duration"] / 1e7, "text": chunk["text"]})
    return words


def subtitle_spans(parts: list[str], words: list[dict], end: float) -> list[dict]:
    """Match sentences to boundary words by counting non-punctuation characters."""
    spans, w = [], 0
    for text in parts:
        need = len(PUNCT.sub("", text))
        start = words[w]["start"] if w < len(words) else end
        got, last = 0, w
        while w < len(words) and got < need:
            got += len(PUNCT.sub("", words[w]["text"]))
            last, w = w, w + 1
        stop = words[last]["start"] + words[last]["dur"] if last < len(words) else end
        spans.append({"start": round(start + LEAD, 3), "end": round(stop + LEAD, 3), "text": text.rstrip("。！？!?；;，,、：:")})
    # Keep each subtitle until the next one starts, so the screen never flashes empty mid-speech.
    for a, b in zip(spans, spans[1:]):
        a["end"] = max(a["end"], b["start"])
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--engine", choices=["edge", "doubao"], default="edge")
    ap.add_argument("--voice", default=None, help="Edge voice, or Doubao speaker ID (default: TTS_SPEAKER in .env)")
    ap.add_argument("--rate", default="+0%")
    ap.add_argument("--no-tone", action="store_true", help="Doubao: expression marks change only the face, not the voice")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--reuse", action="store_true", help="use the speech already in <out-dir> and only rebuild the timeline")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    title, marked = split_script(Path(args.script).read_text(encoding="utf-8"))
    text, tags = take_tags(marked)
    segments = tone_segments(marked)
    if not PUNCT.sub("", text):
        raise SystemExit(f"{args.script}: no text to speak")
    saved = out / "speech.json"
    if args.reuse and saved.exists():
        kept = json.loads(saved.read_text(encoding="utf-8"))
        raw, words = out / kept["raw"], kept["words"]
        print(f"reusing {raw.name}")
    elif args.engine == "doubao":
        if args.no_tone:
            segments = [(text, None)]
        raw, words = synthesize_doubao(segments, args.voice, args.rate, out)
        print("voice tones: " + " | ".join(f"{tone or '默认'}: {len(PUNCT.sub('', t))} chars" for t, tone in segments))
    else:
        raw = out / "speech.mp3"
        words = asyncio.run(synthesize(text, args.voice or "zh-CN-XiaoxiaoNeural", args.rate, raw))
    if not words:
        raise SystemExit(f"{args.engine}: the speech service returned no timings")
    saved.write_text(json.dumps({"raw": raw.name, "words": words}, ensure_ascii=False), encoding="utf-8")

    # Pad with silence and decode to mono PCM for the loudness envelope.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    wav = out / "speech.wav"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(raw), "-af",
                    f"adelay={int(LEAD * 1000)}:all=1,apad=pad_dur={TAIL}", "-ac", "1", "-ar", "44100", str(wav)], check=True)
    pcm = subprocess.run([ffmpeg, "-v", "error", "-i", str(wav), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
                         check=True, capture_output=True).stdout
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    duration = len(samples) / RATE

    fps = args.fps
    hop = RATE / fps
    window = int(RATE * 0.04)
    count = int(np.ceil(duration * fps))
    rms = np.zeros(count, dtype=np.float32)
    for i in range(count):
        centre = int(i * hop)
        seg = samples[max(0, centre - window // 2): centre + window // 2]
        rms[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    peak = float(np.percentile(rms[rms > 0.005], 95)) if np.any(rms > 0.005) else 1.0
    level = np.clip((rms - 0.006) / max(peak - 0.006, 1e-4), 0, 1)
    # Calm the envelope: rise within about two frames, fall over about three, so the mouth
    # follows syllables without twitching on every loudness ripple.
    for i in range(1, count):
        k = 0.6 if level[i] > level[i - 1] else 0.4
        level[i] = level[i - 1] + (level[i] - level[i - 1]) * k

    # Mouth shape: each boundary word's duration is shared evenly by its characters' vowels.
    # A character shorter than MIN_SHAPE keeps the shape before it, so quick syllables do not flicker.
    MIN_SHAPE = 0.1
    form, previous = [None] * count, None
    for word in words:
        vowels = char_vowels(word["text"])
        step = word["dur"] / len(vowels)
        for k, v in enumerate(vowels):
            a = int((LEAD + word["start"] + k * step) * fps)
            b = int(np.ceil((LEAD + word["start"] + (k + 1) * step) * fps))
            shape = FORMS[v] if step >= MIN_SHAPE or previous is None else previous
            for i in range(max(0, a), min(count, b)):
                form[i] = shape
            previous = shape
    # The last character of a sentence often sounds on after its timestamp ends:
    # keep its shape while the voice lasts (up to 0.6 s), so the mouth does not change shape mid-sound.
    held = 0
    for i in range(1, count):
        if form[i] is None and form[i - 1] is not None and level[i] > 0.1 and held < int(0.6 * fps):
            form[i] = form[i - 1]
            held += 1
        elif form[i] is not None:
            held = 0

    timeline = {
        "fps": fps,
        "duration": round(duration, 3),
        "title": title,
        "engine": args.engine,
        "voice": args.voice,
        "frames": [[round(float(l), 3), f] for l, f in zip(level, form)],
        "subtitles": subtitle_spans(subtitle_units(text), words, duration - LEAD),
        "expressions": tag_times(tags, words, duration - LEAD),
    }
    (out / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    print(f"{len(words)} words, {len(tags)} expression marks, {duration:.1f} s, {count} frames -> {out}")


if __name__ == "__main__":
    main()
