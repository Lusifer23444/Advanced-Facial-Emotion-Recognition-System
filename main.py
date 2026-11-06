"""
================================================================================
 Emotion & Style AI — Backend (Gemini Multimodal Vision — Emotion + Fashion)
================================================================================

Stack:
    - FastAPI (async web framework)
    - Pillow + NumPy (blur/brightness diagnostics — pure PIL/NumPy, no OpenCV,
      no PyTorch)
    - google-genai SDK (Gemini 3.6 Flash, structured JSON output, multimodal)

--------------------------------------------------------------------------------
IMPORTANT — READ BEFORE DEMOING THIS TO A PANEL
--------------------------------------------------------------------------------
PyTorch has been fully removed from this backend. There is no local facial
emotion classifier, trained or untrained, anywhere in this file — no random
placeholder weights, no "weights not loaded" fallback string. Gemini's
multimodal vision call is the SOLE source of truth for facial expression
analysis.

`/api/v1/emotion-assistant`: when the client sends a face_image_b64, that
exact frame is passed straight into Gemini (analyze_emotion_vision_with_gemini)
— Gemini looks at the image itself and returns primary_emotion,
confidence_score, and a full breakdown across the emotion taxonomy below,
plus the chatbot reply, all in one structured call. `model_status` on that
endpoint's responses reads "gemini_vision".

Emotion taxonomy (see EMOTION_TAXONOMY): Focused / Neutral, Happy, Sad / Low,
Disgusted / Disturbed, Fear, Angry. ("Sad / Low" is used instead of a
clinical "Depressed / Low" label — see the comment on EMOTION_TAXONOMY for
why.)

The diagnostics block (blur_score, resolution) is computed locally and is
exact, not heuristic — it's used only as an image-quality reject gate before
spending a Gemini call, and as honest context inside the Gemini prompt. It
carries no face-detection or emotion signal; Gemini alone decides
face_detected and the expression itself.

Design principles carried over from your existing backend:
    1. ZERO disk I/O for images — decode/compress/analyze entirely in RAM.
    2. Fail gracefully — malformed/low-quality images never crash the
       service; they return a structured {"status": "error", ...} payload.
    3. Stateless — chat_history is passed in by the client on every request.
    4. Cheap-first validation — the local blur/exposure gate (pure NumPy)
       runs before any Gemini call.

Install:
    pip install fastapi "uvicorn[standard]" google-genai pillow numpy python-dotenv

Configure (create a .env file next to this script — never hardcode the key):
    GEMINI_API_KEY=your_real_key_here

Run:
    python main.py
    # or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
================================================================================
"""

# --- Load .env FIRST, before anything else touches an environment variable ---
from dotenv import load_dotenv
load_dotenv()

import base64
import io
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai.errors import APIError

from datetime import datetime, timezone
try:
    from supabase import create_client, Client as SupabaseClient
except ImportError:  # keep the app bootable even if the package isn't installed yet
    create_client = None  # type: ignore[assignment]
    SupabaseClient = None  # type: ignore[assignment,misc]

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("emotion-style-api")

# ==============================================================================
# 1. GEMINI CLIENT (Fashion + Emotion chat synthesis)
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-3.6-flash"

client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully (model=%s).", GEMINI_MODEL_NAME)
    except Exception:  # noqa: BLE001
        logger.exception(
            "GEMINI_API_KEY is set but the Gemini client failed to initialize — "
            "check the key is well-formed. /api/v1/fashion-assistant and "
            "/api/v1/emotion-assistant will return 503 until this is fixed."
        )
else:
    logger.error(
        "GEMINI_API_KEY is not set (checked process environment and .env) — "
        "/api/v1/fashion-assistant and /api/v1/emotion-assistant will return 503 "
        "with a clear error until you set it."
    )

# ==============================================================================
# 1b. SUPABASE CLIENT (emotion_logs persistence)
# ==============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")
SUPABASE_EMOTION_TABLE = "emotion_logs"

supabase: Optional[Any] = None
if create_client is None:
    logger.error("The 'supabase' package is not installed (pip install supabase) — emotion logging is disabled.")
elif SUPABASE_URL and SUPABASE_SECRET_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception:  # noqa: BLE001
        logger.exception("Supabase client failed to initialize — emotion logging is disabled.")
else:
    logger.error("SUPABASE_URL / SUPABASE_SECRET_KEY not set — emotion logging is disabled.")


def log_emotion_to_supabase(emotion: str) -> None:
    """Insert one row into `emotion_logs`. Best-effort: any failure is logged
    and swallowed so a database problem can never break the user-facing
    emotion analysis response."""
    if supabase is None:
        return
    try:
        supabase.table(SUPABASE_EMOTION_TABLE).insert(
            {
                "emotion": emotion,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to insert emotion log into Supabase")


# ==============================================================================
# 2. IN-MEMORY IMAGE PROCESSING CONSTANTS
# ==============================================================================
MAX_IMAGE_DIMENSION = 1024
JPEG_QUALITY = 80

BLUR_VARIANCE_THRESHOLD = 55.0
BRIGHTNESS_MIN = 35.0
BRIGHTNESS_MAX = 235.0

STANDARD_QUALITY_ERROR_MESSAGE = (
    "This photo is not suitable for analysis. Please ensure good lighting "
    "and face visibility, then try again."
)


class ImageQualityError(Exception):
    """Raised whenever an image fails local suitability checks."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# ==============================================================================
# 3. IN-MEMORY IMAGE DECODE / COMPRESS PIPELINE
# ==============================================================================
def decode_base64_to_image(b64_string: str) -> Image.Image:
    """Decode a base64 (optionally data-URI prefixed) string into a PIL Image.
    Everything happens in RAM — no temp files are ever written."""
    try:
        if "," in b64_string and b64_string.strip().lower().startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]

        raw_bytes = base64.b64decode(b64_string, validate=False)
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        return image.convert("RGB")
    except (base64.binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def compute_blur_score(gray_array: np.ndarray) -> float:
    """Discrete-Laplacian variance, pure NumPy — a drop-in stand-in for
    cv2.Laplacian(img, cv2.CV_64F).var() that needs no OpenCV at all.
    Uses the standard 4-neighbor kernel [[0,1,0],[1,-4,1],[0,1,0]]."""
    gray = gray_array.astype(np.float64)
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    # np.roll wraps around at the edges; trim that 1px border before scoring.
    laplacian = laplacian[1:-1, 1:-1]
    return float(laplacian.var())


def run_local_quality_checks(image: Image.Image) -> None:
    """Fast, free, local heuristics (blur + exposure). Raises ImageQualityError.
    Pure PIL/NumPy — no OpenCV dependency."""
    gray_array = np.array(image.convert("L"), dtype=np.float64)

    laplacian_variance = compute_blur_score(gray_array)
    if laplacian_variance < BLUR_VARIANCE_THRESHOLD:
        raise ImageQualityError(f"image too blurry (sharpness={laplacian_variance:.1f})")

    brightness = float(gray_array.mean())
    if brightness < BRIGHTNESS_MIN:
        raise ImageQualityError(f"image too dark (brightness={brightness:.1f})")
    if brightness > BRIGHTNESS_MAX:
        raise ImageQualityError(f"image overexposed (brightness={brightness:.1f})")


def compress_image_in_memory(image: Image.Image) -> bytes:
    working_image = image.copy()
    working_image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    buffer = io.BytesIO()
    working_image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def prepare_image_for_gemini(b64_string: str, field_name: str = "image") -> bytes:
    """Decode -> local quality gate -> compress, for images headed to Gemini."""
    try:
        image = decode_base64_to_image(b64_string)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {exc}") from exc

    run_local_quality_checks(image)  # may raise ImageQualityError
    return compress_image_in_memory(image)


# ==============================================================================
# 4. STRUCTURAL IMAGE DIAGNOSTICS (blur, resolution) — pure PIL/NumPy
# ==============================================================================
# NOTE: There is no local emotion classifier in this backend anymore.
# Gemini's multimodal vision call is the sole source of truth for facial
# expression analysis (see analyze_emotion_vision_with_gemini below). The
# function below produces ONLY objective, non-AI image-quality numbers
# (blur sharpness, resolution) used for the pre-Gemini reject gate and as
# honest context passed alongside the image in the Gemini prompt.
def compute_structural_diagnostics(image: Image.Image) -> Dict[str, Any]:
    """Cheap, LOCAL, non-AI diagnostics only: blur + resolution. Deliberately
    does NOT attempt face detection or emotion inference — Gemini itself
    judges face presence (face_detected) and the actual expression from the
    raw pixels. This exists only to give Gemini honest image-quality context
    and to keep a fast local reject path for obviously bad frames (see
    run_local_quality_checks)."""
    gray_array = np.array(image.convert("L"), dtype=np.float64)
    blur_score = compute_blur_score(gray_array)
    return {
        "blur_score": round(blur_score, 2),
        "resolution": f"{image.width}x{image.height}",
    }


# ==============================================================================
# 6. GEMINI STRUCTURED OUTPUT SCHEMAS
# ==============================================================================
class FashionAdviceOutput(BaseModel):
    quality_ok: bool = Field(..., description="False if the supplied face image is unusable")
    quality_issue: str = Field(default="", description="Reason if quality_ok is false")
    face_shape: str = Field(..., description="Oval / Round / Square / Heart / Diamond / Oblong / Not Provided")
    haircut_recommendation: str
    beard_style_recommendation: str
    eyewear_recommendation: str
    color_harmony_recommendation: str
    chatbot_reply: str = Field(..., description="Warm, direct conversational reply to the user's message")


class EmotionChatOutput(BaseModel):
    chatbot_reply: str = Field(
        ..., description="Warm, empathetic, natural-language reply that explains the telemetry and answers the user"
    )


# Expanded emotion taxonomy. Each entry has a stable snake_case `key` (used
# as the dict key in the JSON `breakdown` field, and matched by the frontend
# to DOM element ids) and a human-readable `label` (used for primary_emotion
# and in the Gemini prompt).
#
# NOTE: the requested "Depressed / Low" category has been implemented as
# "Sad / Low" instead. "Depressed" is a clinical/diagnostic term, and having
# a vision model assign a confidence-scored clinical label to someone's face
# from a photo would read as an unqualified mental-health assessment, which
# this app has no business making. "Sad / Low" captures the same facial
# expression category without that framing.
EMOTION_TAXONOMY: List[Dict[str, str]] = [
    {"key": "focused_neutral", "label": "Focused / Neutral"},
    {"key": "happy", "label": "Happy"},
    {"key": "sad_low", "label": "Sad / Low"},
    {"key": "disgusted_disturbed", "label": "Disgusted / Disturbed"},
    {"key": "fear", "label": "Fear"},
    {"key": "angry", "label": "Angry"},
]
EMOTION_TAXONOMY_KEYS: List[str] = [e["key"] for e in EMOTION_TAXONOMY]
EMOTION_TAXONOMY_LABELS: List[str] = [e["label"] for e in EMOTION_TAXONOMY]


class EmotionBreakdown(BaseModel):
    """Fixed-shape probability breakdown across the emotion taxonomy.

    THIS IS THE FIX for the `additionalProperties` crash. It used to be
    `breakdown: Dict[str, float]` on EmotionVisionOutput below. Pydantic (via
    the google-genai SDK's schema converter) emits an `additionalProperties`
    key for any open-ended `Dict[str, X]` field, because that's the only way
    JSON Schema can express "any string key maps to a float". The Gemini
    *Developer* API — unlike Vertex AI's "Gemini Enterprise Agent Platform"
    mode — hard-rejects any response_schema containing `additionalProperties`
    at request time, which is exactly the ValueError being hit.

    Enumerating the six taxonomy keys as real, named fields instead removes
    `additionalProperties` from the schema entirely (a closed set of
    properties needs no catch-all), AND gives Gemini a stricter contract in
    the bargain — it can no longer invent, drop, or rename a key. Field order
    mirrors EMOTION_TAXONOMY / EMOTION_TAXONOMY_KEYS above, and must keep
    matching it (and the frontend's EMOTION_TAXONOMY in spa.js) if that list
    ever changes."""

    focused_neutral: float = Field(..., description="Probability for 'Focused / Neutral', in [0,1]")
    happy: float = Field(..., description="Probability for 'Happy', in [0,1]")
    sad_low: float = Field(..., description="Probability for 'Sad / Low', in [0,1]")
    disgusted_disturbed: float = Field(..., description="Probability for 'Disgusted / Disturbed', in [0,1]")
    fear: float = Field(..., description="Probability for 'Fear', in [0,1]")
    angry: float = Field(..., description="Probability for 'Angry', in [0,1]")


class EmotionVisionOutput(BaseModel):
    """Structured result of Gemini performing the ACTUAL emotion classification
    directly from the image (multimodal vision) — Gemini is the sole source
    of truth for facial expression analysis in this backend; there is no
    local classifier of any kind backing this up or falling back for it."""

    face_detected: bool = Field(
        ..., description="False if no face is clearly visible/usable in the image"
    )
    primary_emotion: str = Field(
        ..., description=f"The single dominant facial expression label, one of: {', '.join(EMOTION_TAXONOMY_LABELS)}"
    )
    confidence_score: float = Field(
        ..., description="Model's calibrated confidence (0.0-1.0) in primary_emotion"
    )
    breakdown: EmotionBreakdown = Field(
        ...,
        description=(
            "Probability breakdown across all six taxonomy categories "
            f"({', '.join(EMOTION_TAXONOMY_KEYS)}), values in [0,1] summing to ~1.0."
        ),
    )
    chatbot_reply: str = Field(
        ..., description="Warm, empathetic, natural-language reply that discusses the detected expression and answers the user's message"
    )


def _assert_gemini_schema_safe(model_cls: Type[BaseModel]) -> None:
    """Fail LOUDLY at import time, not at request time, if a structured-
    output schema would trip Gemini Developer API mode's restriction on
    `additionalProperties`.

    Builds the model's JSON schema, inlines any `$defs`/`$ref` (Pydantic
    emits nested BaseModels that way), then walks the whole tree looking for
    an `additionalProperties` key anywhere — the one thing that reliably
    triggers:
        ValueError: additionalProperties is only supported in Gemini
        Enterprise Agent Platform mode, not in Gemini Developer API mode.

    Called once per output schema immediately below, right after each class
    is defined. This is deliberately a hard startup check rather than a
    silent runtime "strip additionalProperties and hope for the best": a
    silently stripped schema for an open Dict field just becomes an
    unconstrained object (no `properties`, no `additionalProperties`), which
    quietly throws away the "exactly these keys" guarantee instead of
    surfacing the mistake. If this ever raises, the fix is the same one
    applied to `breakdown` above — replace the offending Dict/Mapping field
    with an explicit BaseModel of named fields."""
    raw_schema = model_cls.model_json_schema()
    defs = raw_schema.get("$defs", {})

    def _inline(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return _inline(defs[ref.rsplit("/", 1)[-1]])
            return {k: _inline(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_inline(item) for item in node]
        return node

    def _find_additional_properties(node: Any, path: str) -> Optional[str]:
        if isinstance(node, dict):
            if "additionalProperties" in node:
                return path or "<root>"
            for key, value in node.items():
                hit = _find_additional_properties(value, f"{path}.{key}")
                if hit:
                    return hit
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                hit = _find_additional_properties(item, f"{path}[{idx}]")
                if hit:
                    return hit
        return None

    offender = _find_additional_properties(_inline(raw_schema), model_cls.__name__)
    if offender:
        raise TypeError(
            f"{model_cls.__name__} is not safe as a Gemini Developer API "
            f"response_schema: found 'additionalProperties' at {offender}. "
            "This is almost always an open Dict[str, X]/Mapping[str, X] "
            "field — replace it with an explicit BaseModel of named fields "
            "(see EmotionBreakdown above for the pattern)."
        )


# Run the guard now, at import time, for every schema handed to Gemini as a
# response_schema — so a future edit that reintroduces an open Dict field
# breaks the app immediately at startup with a clear message, instead of
# crashing mid-request in production the way `breakdown` used to.
for _schema_cls in (FashionAdviceOutput, EmotionChatOutput, EmotionVisionOutput):
    _assert_gemini_schema_safe(_schema_cls)


# ==============================================================================
# 7. REQUEST MODELS
# ==============================================================================
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class FashionAssistantRequest(BaseModel):
    user_query: str
    face_image_b64: Optional[str] = None
    chat_history: List[ChatMessage] = []


class EmotionAssistantRequest(BaseModel):
    user_query: str
    chat_history: List[ChatMessage] = []
    face_image_b64: Optional[str] = None
    # If the client already has a recent emotion-assistant result (Gemini's
    # own prior vision reading — primary_emotion / confidence_score /
    # breakdown / diagnostics), it can pass it here to skip resending an
    # image on a pure follow-up chat turn. If both this AND face_image_b64
    # are omitted, Gemini replies conversationally with no frame context.
    telemetry: Optional[Dict[str, Any]] = None


# ==============================================================================
# 8. RATE LIMITING (in-memory sliding window — swap for Redis in real prod)
# ==============================================================================
RATE_LIMIT_MAX_REQUESTS = 8
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_buckets: Dict[str, List[float]] = defaultdict(list)


def enforce_rate_limit(client_key: str) -> None:
    now = time.time()
    bucket = _rate_limit_buckets[client_key]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a moment before sending another message.",
        )
    bucket.append(now)


# ==============================================================================
# 9. GEMINI CALL HELPER (with retry + exponential backoff)
# ==============================================================================
# Transient-only: 500/502/503/504-class failures and generic network hiccups
# are worth retrying because they're capacity/connectivity blips on Google's
# side, not something wrong with our request. 429 (RESOURCE_EXHAUSTED) is
# deliberately NOT retried here — a rate limit doesn't usually clear within
# a couple of seconds, so retrying it would just make the caller wait
# through 3 backoff cycles for a 429 they were always going to get; it's
# raised immediately instead so the client can back off on its own schedule.
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_BASE_DELAY_SECONDS = 1.0
GEMINI_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
GEMINI_RETRYABLE_MARKERS = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "OVERLOADED", "502", "503")


def _is_retryable_api_error(exc: "APIError") -> bool:
    status_code = getattr(exc, "code", None)
    if status_code in GEMINI_RETRYABLE_STATUS_CODES:
        return True
    exc_text = str(exc).upper()
    return any(marker in exc_text for marker in GEMINI_RETRYABLE_MARKERS)


def _generate_content_with_retry(contents: list, schema: type):
    """Wraps client.models.generate_content with up to GEMINI_MAX_ATTEMPTS
    tries and exponential backoff (1s, 2s, 4s, ... plus small jitter) for
    transient 500/502/503/504 failures. Non-retryable errors (429, schema
    errors, auth errors, etc.) are re-raised immediately on the first
    attempt so callers still get fast, accurate error responses."""
    last_exc: Optional[BaseException] = None

    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.4,
                ),
            )
        except APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc).upper():
                raise  # not retryable -- let _call_gemini turn this into a 429 right away

            last_exc = exc
            if attempt >= GEMINI_MAX_ATTEMPTS or not _is_retryable_api_error(exc):
                raise

            delay = GEMINI_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Gemini call failed (attempt %d/%d, status=%s): %s -- retrying in %.1fs",
                attempt, GEMINI_MAX_ATTEMPTS, status_code, exc, delay,
            )
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001 -- network-level failures (timeouts, DNS, etc.)
            last_exc = exc
            if attempt >= GEMINI_MAX_ATTEMPTS:
                raise

            delay = GEMINI_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Gemini call failed (attempt %d/%d, non-API error): %s -- retrying in %.1fs",
                attempt, GEMINI_MAX_ATTEMPTS, exc, delay,
            )
            time.sleep(delay)

    # Unreachable in practice (the loop always returns or raises), but keeps
    # type-checkers happy and guarantees we never silently return None.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Gemini retry loop exited without a result or an exception")


def _call_gemini(contents: list, schema: type) -> Any:
    if client is None:
        logger.error(
            "Gemini call rejected: client is not initialized (GEMINI_API_KEY missing or invalid at startup)."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "AI features are unavailable: GEMINI_API_KEY is not configured (or failed to "
                "initialize) on the server. Check the server logs and your .env file."
            ),
        )
    try:
        response = _generate_content_with_retry(contents, schema)
    except APIError as exc:
        status_code = getattr(exc, "code", None)
        if status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc).upper():
            logger.warning("Gemini rate limit hit: %s", exc)
            raise HTTPException(
                status_code=429, detail="AI service is rate-limited. Please try again shortly."
            ) from exc
        logger.exception("Gemini API error persisted after %d attempts", GEMINI_MAX_ATTEMPTS)
        raise HTTPException(status_code=502, detail=f"AI service error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error calling Gemini after %d attempts", GEMINI_MAX_ATTEMPTS)
        raise HTTPException(status_code=502, detail="AI service is currently unavailable") from exc

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        logger.error("Gemini returned no parsable structured output: %s", getattr(response, "text", ""))
        raise HTTPException(status_code=502, detail="AI service returned an unreadable response")
    return parsed


def analyze_fashion_with_gemini(
    request: FashionAssistantRequest,
    face_image_bytes: Optional[bytes],
) -> FashionAdviceOutput:
    history_text = "\n".join(f"{m.role}: {m.content}" for m in request.chat_history[-10:]) or "(no prior turns)"

    prompt = f"""
You are an expert personal stylist, grooming consultant, and face-shape
analyst embedded in a fashion & grooming chat assistant. You NEVER give
food, diet, or nutrition advice — stay strictly within fashion, grooming,
hairstyles, eyewear, and outfit styling.

Step 1 — Suitability check: If a face image was supplied, verify it's
usable (face clearly visible, adequately lit, not majorly obscured). If
unusable, set quality_ok=false with a short quality_issue. If no image was
supplied, that's fine — treat quality_ok=true and answer using general
styling expertise plus the conversation context.

Step 2 — If suitable:
 - If a face image is available, analyze its facial geometry/topology
   (jawline angle, cheekbone prominence, forehead-to-chin proportions) and
   classify face_shape as one of: Oval, Round, Square, Heart, Diamond,
   Oblong. If no image is available, set face_shape="Not Provided".
 - haircut_recommendation: 1-2 concrete haircut styles suited to the
   detected (or general, if no image) face shape.
 - beard_style_recommendation: 1-2 concrete facial-hair styles, or a note
   on going clean-shaven if that suits the shape better.
 - eyewear_recommendation: optical frame shapes that complement the face
   (e.g. angular frames to add contrast to round faces).
 - color_harmony_recommendation: outfit color palette / undertone guidance.
 - chatbot_reply: a warm, conversational direct reply to the user's message
   below, weaving in the above insights naturally — this is what gets shown
   as the chat bubble.

Conversation history (most recent last):
{history_text}

User's latest message: "{request.user_query}"

Respond only via the provided JSON schema.
"""

    contents: list = []
    if face_image_bytes is not None:
        contents.append(types.Part.from_bytes(data=face_image_bytes, mime_type="image/jpeg"))
    contents.append(prompt)

    return _call_gemini(contents, FashionAdviceOutput)


def _format_telemetry_for_prompt(telemetry: Optional[Dict[str, Any]]) -> str:
    """Turns a previously-returned emotion-assistant telemetry dict into
    plain-language text for the Gemini prompt (used only on turns where no
    fresh image is sent). Missing/partial fields degrade gracefully."""
    if not telemetry:
        return "No current frame telemetry is available — no photo/webcam session is active yet."

    breakdown = telemetry.get("breakdown") or telemetry.get("emotion_probabilities") or {}
    diag = telemetry.get("diagnostics", {}) or {}
    breakdown_text = ", ".join(f"{k}: {round(float(v) * 100, 1)}%" for k, v in breakdown.items()) or "n/a"

    text = (
        f"Primary detected expression (from a previous turn): {telemetry.get('primary_emotion', 'unknown')} "
        f"(confidence {round(float(telemetry.get('confidence_score', 0)) * 100, 1)}%). "
        f"Full probability breakdown: {breakdown_text}. "
        f"Frame diagnostics — blur_score: {diag.get('blur_score', 'n/a')}, "
        f"resolution: {diag.get('resolution', 'n/a')}. "
        "This reading came from Gemini's own vision analysis of an earlier frame, "
        "not a local classifier."
    )
    return text


def analyze_emotion_vision_with_gemini(
    request: EmotionAssistantRequest,
    face_image_bytes: bytes,
    diagnostics: Dict[str, Any],
) -> EmotionVisionOutput:
    """Sends the user's face image directly into Gemini's multimodal vision
    pipeline and asks IT to visually judge the facial expression. Gemini is
    the sole source of truth for emotion classification here — there is no
    local model of any kind involved in producing primary_emotion,
    confidence_score, or breakdown. `diagnostics` (blur/resolution) is
    passed only as image-quality context, never as emotion signal."""
    history_text = "\n".join(f"{m.role}: {m.content}" for m in request.chat_history[-10:]) or "(no prior turns)"

    taxonomy_lines = "\n".join(f'- key "{e["key"]}" = "{e["label"]}"' for e in EMOTION_TAXONOMY)

    prompt = f"""
You are the vision engine for an "Emotion Assistant" embedded in a
facial-expression-analysis studio. You are given the user's actual camera
frame below as an image. Look directly at the face in that image and
perform the emotion classification YOURSELF from what you visually observe
— do not assume or invent a reading, and do not rely on any external
classifier; you are the classifier and the sole source of truth here.

Step 1 — Face check: set face_detected=false if no face is clearly visible
or usable in the image (e.g. empty frame, face fully obscured/out of
frame). Otherwise set it true.

Step 2 — If a face is detected, visually classify the dominant expression
using this exact taxonomy (key -> label):
{taxonomy_lines}

Set primary_emotion to the matching LABEL (not the key) of the dominant
category, confidence_score to your calibrated confidence (0.0-1.0), and
breakdown to a full probability dict using EXACTLY these KEYS: {", ".join(EMOTION_TAXONOMY_KEYS)}
(values in [0,1], summing to roughly 1.0). Base this purely on the visual
evidence in the image — facial muscles, eyes, mouth, brow. Do not add,
omit, or rename any keys in breakdown.

Note on "sad_low": this is a facial-expression category (a downturned,
low-energy expression), not a clinical diagnosis. Never imply a mental
health diagnosis anywhere in your output, even if this category scores
highest.

Local image-quality diagnostics (sharpness/lighting only — NOT emotion
data, ignore these when judging the expression itself):
- blur_score: {diagnostics.get('blur_score', 'n/a')}
- resolution: {diagnostics.get('resolution', 'n/a')}

You are NOT a therapist and must never offer medical or psychological
diagnoses. If the user's message suggests real emotional distress, respond
with warmth and gently encourage them to talk to someone they trust — do
not diagnose, prescribe, or role-play as a counselor.

Conversation history (most recent last):
{history_text}

User's latest message: "{request.user_query}"

chatbot_reply: write a warm, natural, conversational reply (a few
sentences) that answers the user's message, weaving in what you visually
observed in their expression where relevant. If face_detected is false,
say so honestly in chatbot_reply instead of inventing an emotion.

Respond only via the provided JSON schema.
"""

    contents: list = [
        types.Part.from_bytes(data=face_image_bytes, mime_type="image/jpeg"),
        prompt,
    ]
    return _call_gemini(contents, EmotionVisionOutput)


def analyze_emotion_chat_with_gemini(
    request: EmotionAssistantRequest,
    telemetry: Optional[Dict[str, Any]],
) -> EmotionChatOutput:
    history_text = "\n".join(f"{m.role}: {m.content}" for m in request.chat_history[-10:]) or "(no prior turns)"
    telemetry_text = _format_telemetry_for_prompt(telemetry)

    prompt = f"""
You are the "Emotion Assistant" — a friendly, emotionally intelligent guide
embedded in a facial-expression-analysis studio. Your job is to explain the
most recent Gemini vision reading in plain, natural language and answer the
user's questions about it conversationally, engagingly, and empathetically.

You are NOT a therapist and must never offer medical or psychological
diagnoses. If the user's message suggests real emotional distress, respond
with warmth and gently encourage them to talk to someone they trust — do
not diagnose, prescribe, or role-play as a counselor.

Current frame telemetry:
{telemetry_text}

Conversation history (most recent last):
{history_text}

User's latest message: "{request.user_query}"

Write a warm, natural, conversational reply (a few sentences) that answers
the user's message, weaving in the telemetry naturally where it's relevant.
Respond only via the provided JSON schema.
"""
    return _call_gemini([prompt], EmotionChatOutput)


# ==============================================================================
# 10. FASTAPI APP
# ==============================================================================
app = FastAPI(
    title="Emotion & Style AI — Backend",
    version="2.0.0",
)

# --- CORS: local development origins only (NOT "*") --------------------------
LOCAL_DEV_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def quality_error_response() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"status": "error", "error_message": STANDARD_QUALITY_ERROR_MESSAGE},
    )


# --- Health check --------------------------------------------------------------
@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ready",
        "emotion_engine": "gemini_vision",
        "gemini_configured": client is not None,
    }


# --- Endpoint B: Gemini-Vision-Powered Emotion Assistant ---------------------
@app.post("/api/v1/emotion-assistant")
async def emotion_assistant(payload: EmotionAssistantRequest, request: Request) -> JSONResponse:
    """Gemini is the SOLE emotion-recognition engine in this backend — there
    is no local classifier of any kind. When the client sends a fresh
    face_image_b64, that exact frame is decoded, quality-gated, and passed
    straight into Gemini's multimodal vision call
    (analyze_emotion_vision_with_gemini), which returns the real
    classification (primary_emotion, confidence_score, breakdown across the
    full emotion taxonomy) plus the empathetic chatbot_reply in one
    structured call. NumPy/PIL are used ONLY for cheap local diagnostics
    (blur_score, resolution) and the pre-Gemini quality gate — never for
    emotion inference.

    Backward-compat path: if the client instead sends a pre-computed
    `telemetry` dict (Gemini's own result from an earlier turn) and no
    image, Gemini falls back to summarizing that telemetry in text rather
    than re-analyzing a frame."""
    client_key = request.client.host if request.client else "unknown"
    enforce_rate_limit(client_key)

    # --- Primary path: a fresh frame is present -> Gemini vision classifies it directly ---
    if payload.face_image_b64:
        try:
            image = decode_base64_to_image(payload.face_image_b64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid face_image_b64: {exc}") from exc

        try:
            run_local_quality_checks(image)
        except ImageQualityError as exc:
            logger.info("Local quality gate rejected emotion-assistant image: %s", exc.reason)
            return quality_error_response()

        # Local diagnostics are structural/quality-only (blur, resolution) —
        # no local emotion or face-presence inference happens here at all.
        diagnostics = compute_structural_diagnostics(image)
        face_image_bytes = compress_image_in_memory(image)

        try:
            # Runs off the event loop: with retries, this call can now sleep
            # for a few seconds on transient failures, and we don't want
            # that to block every other concurrent request.
            vision_result = await run_in_threadpool(
                analyze_emotion_vision_with_gemini, payload, face_image_bytes, diagnostics
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected failure in emotion_assistant (Gemini vision path)")
            raise HTTPException(status_code=500, detail="Internal error during emotion vision analysis") from exc

        if not vision_result.face_detected:
            logger.info("Gemini vision found no usable face in emotion-assistant image")
            return quality_error_response()

        response_body: Dict[str, Any] = {
            "status": "success",
            "primary_emotion": vision_result.primary_emotion,
            "confidence_score": round(vision_result.confidence_score, 4),
            "breakdown": {
                k: round(float(v), 4) for k, v in vision_result.breakdown.model_dump().items()
            },
            "diagnostics": diagnostics,
            "model_status": "gemini_vision",
            "chatbot_reply": vision_result.chatbot_reply,
        }
        # Persist the reading (best-effort, off the event loop; never raises).
        await run_in_threadpool(log_emotion_to_supabase, vision_result.primary_emotion)
        return JSONResponse(status_code=200, content=response_body)

    # --- Fallback path: no image this turn -> text-only chat around any supplied telemetry ---
    telemetry: Optional[Dict[str, Any]] = dict(payload.telemetry) if payload.telemetry is not None else None

    try:
        chat_result = await run_in_threadpool(analyze_emotion_chat_with_gemini, payload, telemetry)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure in emotion_assistant (text fallback path)")
        raise HTTPException(status_code=500, detail="Internal error during emotion chat synthesis") from exc

    response_body = {"status": "success", "chatbot_reply": chat_result.chatbot_reply}
    if telemetry is not None:
        response_body.update(telemetry)  # primary_emotion, confidence_score, breakdown, diagnostics, model_status
    return JSONResponse(status_code=200, content=response_body)


# --- Endpoint C: Gemini Multimodal Fashion & Grooming Assistant ---------------
@app.post("/api/v1/fashion-assistant")
async def fashion_assistant(payload: FashionAssistantRequest, request: Request) -> JSONResponse:
    client_key = request.client.host if request.client else "unknown"
    enforce_rate_limit(client_key)

    face_bytes: Optional[bytes] = None
    if payload.face_image_b64:
        try:
            face_bytes = prepare_image_for_gemini(payload.face_image_b64, field_name="face_image_b64")
        except ImageQualityError as exc:
            logger.info("Local quality gate rejected fashion image: %s", exc.reason)
            return quality_error_response()

    try:
        result = await run_in_threadpool(analyze_fashion_with_gemini, payload, face_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure in fashion_assistant")
        raise HTTPException(status_code=500, detail="Internal error during fashion analysis") from exc

    if not result.quality_ok:
        logger.info("Gemini quality gate rejected fashion image: %s", result.quality_issue)
        return quality_error_response()

    response_body = {
        "status": "success",
        "face_shape": result.face_shape,
        "haircut_recommendation": result.haircut_recommendation,
        "beard_style_recommendation": result.beard_style_recommendation,
        "eyewear_recommendation": result.eyewear_recommendation,
        "color_harmony_recommendation": result.color_harmony_recommendation,
        "chatbot_reply": result.chatbot_reply,
    }
    return JSONResponse(status_code=200, content=response_body)


# ==============================================================================
# 11. ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)