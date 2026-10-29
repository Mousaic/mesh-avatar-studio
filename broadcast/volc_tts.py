"""Volcengine Doubao speech synthesis 2.0 over the bidirectional WebSocket API (v3 binary protocol).

synthesize() returns 16-bit mono PCM and per-character timestamps. Settings come from broadcast/.env:
  TTS_API_KEY        required, from the console's API Key page
  TTS_SPEAKER        optional voice ID (a seed-tts-2.0 voice), default zh_female_vv_uranus_bigtts
  TTS_RESOURCE_ID    optional, default seed-tts-2.0
  TTS_URL            optional, default wss://openspeech.bytedance.com/api/v3/tts/bidirection
"""
import json
import os
import struct
import uuid
from pathlib import Path

import websockets

URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
SAMPLE_RATE = 24000

# message types (high nibble of byte 1) and the "carries an event number" flag
FULL_CLIENT, FULL_SERVER, AUDIO_SERVER, ERROR = 0b0001, 0b1001, 0b1011, 0b1111
WITH_EVENT, POS_SEQ, NEG_SEQ = 0b0100, 0b0001, 0b0011
# events
START_CONNECTION, FINISH_CONNECTION = 1, 2
CONNECTION_STARTED, CONNECTION_FAILED, CONNECTION_FINISHED = 50, 51, 52
START_SESSION, CANCEL_SESSION, FINISH_SESSION = 100, 101, 102
SESSION_STARTED, SESSION_CANCELED, SESSION_FINISHED, SESSION_FAILED = 150, 151, 152, 153
TASK_REQUEST = 200
SENTENCE_START, SENTENCE_END, TTS_RESPONSE, TTS_SUBTITLE = 350, 351, 352, 364
CONNECTION_EVENTS = {START_CONNECTION, FINISH_CONNECTION, CONNECTION_STARTED, CONNECTION_FAILED, CONNECTION_FINISHED}


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("TTS_")}}


def frame(event: int, payload: dict, session_id: str | None = None) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode()
    out = bytes([0x11, (FULL_CLIENT << 4) | WITH_EVENT, 0x10, 0x00]) + struct.pack(">i", event)
    if session_id is not None:
        sid = session_id.encode()
        out += struct.pack(">I", len(sid)) + sid
    return out + struct.pack(">I", len(body)) + body


def parse(data: bytes) -> dict:
    header_size = (data[0] & 0x0F) * 4
    kind, flags = data[1] >> 4, data[1] & 0x0F
    pos = header_size
    msg = {"type": kind, "event": None}

    def u32():
        nonlocal pos
        value = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        return value

    if flags in (POS_SEQ, NEG_SEQ):
        msg["sequence"] = struct.unpack(">i", data[pos:pos + 4])[0]
        pos += 4
    if kind == ERROR:
        msg["error_code"] = u32()
    if flags == WITH_EVENT:
        msg["event"] = struct.unpack(">i", data[pos:pos + 4])[0]
        pos += 4
        if msg["event"] not in CONNECTION_EVENTS:
            n = u32()
            msg["session_id"] = data[pos:pos + n].decode(errors="replace")
            pos += n
        if msg["event"] in (CONNECTION_STARTED, CONNECTION_FAILED, CONNECTION_FINISHED):
            n = u32()
            pos += n                      # connect id
    n = u32()
    payload = data[pos:pos + n]
    msg["payload"] = payload
    if kind != AUDIO_SERVER and payload[:1] in (b"{", b"["):
        msg["json"] = json.loads(payload.decode("utf-8"))
    return msg


async def synthesize(segments: list[tuple[str, str | None]], env: dict, speech_rate: int = 0) -> tuple[bytes, list[dict]]:
    """Speak (text, voice instruction or None) segments one after another on one connection.

    -> (PCM s16le mono 24 kHz, [{start, dur, text}] per character, times in seconds from the start).
    Each segment is its own session, because a voice instruction (context_texts) applies to a session.
    """
    key = env.get("TTS_API_KEY")
    if not key:
        raise SystemExit("TTS_API_KEY is missing: put it in broadcast/.env")
    headers = {
        "X-Api-Key": key,
        "X-Api-Resource-Id": env.get("TTS_RESOURCE_ID", "seed-tts-2.0"),
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    base = {
        "speaker": env.get("TTS_SPEAKER", "zh_female_vv_uranus_bigtts"),
        "audio_params": {"format": "pcm", "sample_rate": SAMPLE_RATE, "speech_rate": speech_rate, "enable_subtitle": True},
        # one conversation for all segments, so the voice carries on from one segment to the next
        "section_id": str(uuid.uuid4()),
    }
    pcm = bytearray()
    words = []

    async with websockets.connect(env.get("TTS_URL", URL), additional_headers=headers, max_size=64 * 1024 * 1024) as ws:
        async def expect(event: int) -> dict:
            msg = parse(await ws.recv())
            if msg["event"] != event:
                raise SystemExit(f"Doubao TTS: expected event {event}, got {describe(msg)}")
            return msg

        await ws.send(frame(START_CONNECTION, {}))
        await expect(CONNECTION_STARTED)
        for text, context in segments:
            if not text.strip():
                continue
            req = dict(base)
            if context:
                req["additions"] = json.dumps({"context_texts": [context]}, ensure_ascii=False)
            session = str(uuid.uuid4())
            offset = len(pcm) / 2 / SAMPLE_RATE      # segment timestamps start at zero
            sentence_offset, first, seen = offset, len(words), set()
            await ws.send(frame(START_SESSION, {"event": START_SESSION, "req_params": req}, session))
            await expect(SESSION_STARTED)
            await ws.send(frame(TASK_REQUEST, {"event": TASK_REQUEST, "req_params": {**req, "text": text}}, session))
            await ws.send(frame(FINISH_SESSION, {}, session))
            while True:
                msg = parse(await ws.recv())
                if msg["type"] == AUDIO_SERVER:
                    pcm.extend(msg["payload"])
                elif msg["type"] == ERROR or msg["event"] in (SESSION_FAILED, CONNECTION_FAILED, SESSION_CANCELED):
                    raise SystemExit(f"Doubao TTS failed: {describe(msg)}")
                elif msg["event"] == SENTENCE_START:
                    sentence_offset = len(pcm) / 2 / SAMPLE_RATE
                elif msg["event"] == SESSION_FINISHED:
                    break
                if msg["event"] != TTS_SUBTITLE:
                    continue
                # Per-character timestamps, counted from the start of the session's audio. Should a
                # later sentence restart at zero, move it after the audio received before that sentence.
                for w in (msg.get("json") or {}).get("words") or []:
                    start, end = float(w.get("startTime", 0)) + offset, float(w.get("endTime", 0)) + offset
                    if len(words) > first and start < words[-1]["start"] - 0.05:
                        start, end = start - offset + sentence_offset, end - offset + sentence_offset
                    k = (round(start, 3), w.get("word", ""))
                    if k in seen:
                        continue
                    seen.add(k)
                    words.append({"start": start, "dur": max(0.0, end - start), "text": w.get("word", "")})
        await ws.send(frame(FINISH_CONNECTION, {}))
    words.sort(key=lambda w: w["start"])
    return bytes(pcm), words


def describe(msg: dict) -> str:
    body = msg.get("json") if "json" in msg else msg.get("payload", b"")[:300]
    return f"type={msg['type']} event={msg['event']} code={msg.get('error_code')} body={body}"
