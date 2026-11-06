/* ============================================================================
   Emotion & Style AI — Live Backend Integration
   Loaded AFTER the inline <script> in spa.html, so the function declarations
   below (handleImageUpload, handleChatSubmitF3, handleChatSubmitF4,
   resetTelemetryView, resetScannerView) OVERRIDE the versions defined inline.
   Everything else (compressImage, processImageSrc, triggerErrorToast, router,
   etc.) is reused as-is from the inline script — this file does not redeclare
   them.

   Layout of this file:
     1. Config + shared state
     2. Error buffer zone  (ApiError, describeError, notifyError, guards)
     3. API service layer  (apiRequest, normalizers, Emotion/Fashion services)
     4. DOM-safe helpers
     5. Camera (toggleCamera + track cleanup)
     6. Emotion panel + F3 chat / image upload
     7. F4 chat
     8. Global safety nets
   ============================================================================ */

/* ============================================================================
   1. CONFIG + SHARED STATE
   ============================================================================ */
const API_BASE_URL = 'http://127.0.0.1:8000';

const API_CONFIG = {
  baseUrl: API_BASE_URL,
  timeoutMs: 45000,
  endpoints: {
    emotion: '/api/v1/emotion-assistant',
    fashion: '/api/v1/fashion-assistant',
  },
};

let emotionChatHistory = [];
let fashionChatHistory = [];

// In-memory fallback so the active photo still works if localStorage is
// full or blocked (private mode, quota exceeded, etc.).
let activeFaceFallback = null;

/* Taxonomy below MUST match backend EMOTION_TAXONOMY in main.py exactly
   (same keys, same labels). `key` is used to look up DOM elements
   (emotion-<key>-val / emotion-<key>-bar); `label` is what the backend
   sends back as `primary_emotion`. */
const EMOTION_TAXONOMY = [
  { key: 'focused_neutral', label: 'Focused / Neutral', emoji: '😐' },
  { key: 'happy', label: 'Happy', emoji: '😄' },
  { key: 'sad_low', label: 'Sad / Low', emoji: '😔' },
  { key: 'disgusted_disturbed', label: 'Disgusted / Disturbed', emoji: '🤢' },
  { key: 'fear', label: 'Fear', emoji: '😨' },
  { key: 'angry', label: 'Angry', emoji: '😠' },
];

/* ============================================================================
   2. ERROR BUFFER ZONE
   Everything that can fail ends up here, and the user sees triggerErrorToast()
   instead of a dead UI or a silent console error.
   ============================================================================ */
class ApiError extends Error {
  /**
   * @param {'network'|'timeout'|'rate_limit'|'http'|'parse'|'contract'} kind
   */
  constructor(kind, message, extra = {}) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = extra.status || null;
  }
}

/** Calls the page's triggerErrorToast, never throwing if it is missing/broken. */
function notifyError(title, desc) {
  try {
    if (typeof triggerErrorToast === 'function') {
      triggerErrorToast(title, desc);
      return;
    }
  } catch (toastErr) {
    console.error('triggerErrorToast failed:', toastErr);
  }
  console.error(`[${title}] ${desc}`);
}

/** Maps any thrown value to a { title, desc } pair suitable for a toast. */
function describeError(err, fallbackTitle = 'Something Went Wrong') {
  if (err instanceof ApiError) {
    const titles = {
      network: 'Backend Unreachable',
      timeout: 'Request Timed Out',
      rate_limit: 'Too Many Requests',
      http: 'Service Error',
      parse: 'Unreadable Response',
      contract: 'Unexpected Response',
    };
    return { title: titles[err.kind] || fallbackTitle, desc: err.message };
  }
  const msg = err && err.message ? err.message : 'An unexpected error occurred.';
  return { title: fallbackTitle, desc: msg };
}

/**
 * Runs `fn`, and on any failure shows a toast instead of letting the error
 * escape. Use for event handlers wired from inline HTML attributes.
 */
async function runGuarded(fn, fallbackTitle) {
  try {
    return await fn();
  } catch (err) {
    console.error(`[${fallbackTitle}]`, err);
    const { title, desc } = describeError(err, fallbackTitle);
    notifyError(title, desc);
    return undefined;
  }
}

/* ============================================================================
   3. API SERVICE LAYER
   The only place that knows about fetch(), URLs and payload shapes.
   ============================================================================ */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.innerText = str == null ? '' : String(str);
  return div.innerHTML;
}

async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, toNumber(value, 0)));
}

function asString(value, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function extractErrorDetail(body, status) {
  if (body && typeof body === 'object') {
    const d = body.detail;
    if (typeof d === 'string' && d) return d;
    if (Array.isArray(d) && d.length) {
      return d.map((item) => (item && item.msg ? item.msg : String(item))).join('; ');
    }
    if (typeof body.error_message === 'string' && body.error_message) return body.error_message;
  }
  return `Service returned HTTP ${status}`;
}

/** Keeps only well-formed {role, content} turns, capped at the last 10. */
function sanitizeHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter((m) => m && typeof m.role === 'string' && typeof m.content === 'string')
    .slice(-10);
}

/**
 * POSTs JSON to the backend with a timeout and normalizes every failure mode
 * into an ApiError. Resolves only with a parsed JSON object.
 */
async function apiRequest(path, body, { timeoutMs } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs || API_CONFIG.timeoutMs);

  let response;
  try {
    response = await fetch(`${API_CONFIG.baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      throw new ApiError('timeout', 'The AI service took too long to respond. Please try again.');
    }
    throw new ApiError(
      'network',
      `Could not reach the backend at ${API_CONFIG.baseUrl}. Check that the server is running.`
    );
  } finally {
    clearTimeout(timer);
  }

  const data = await safeJson(response);

  if (response.status === 429) {
    throw new ApiError(
      'rate_limit',
      "I'm getting a lot of requests right now — please wait a few seconds and try again.",
      { status: 429 }
    );
  }
  if (!response.ok) {
    throw new ApiError('http', extractErrorDetail(data, response.status), { status: response.status });
  }
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    throw new ApiError('parse', 'The server returned a response that could not be read.');
  }
  return data;
}

/**
 * Coerces any emotion-assistant response into a predictable shape. Missing or
 * malformed fields get safe defaults, so later UI code never sees undefined.
 * Keeps the backend's field names on purpose.
 */
function normalizeEmotionResponse(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ApiError('contract', 'The emotion service returned an unexpected response.');
  }

  if (raw.status === 'error') {
    return {
      status: 'error',
      error_message: asString(raw.error_message, 'That frame was not suitable for analysis.'),
      chatbot_reply: '',
      hasVision: false,
    };
  }

  const reply = asString(raw.chatbot_reply).trim();
  const primary = asString(raw.primary_emotion).trim();
  const rawBreakdown = raw.breakdown && typeof raw.breakdown === 'object' ? raw.breakdown : null;
  const hasVision = !!(primary && rawBreakdown);

  if (!reply && !hasVision) {
    throw new ApiError('contract', 'The emotion service response had no reply or analysis in it.');
  }

  const breakdown = {};
  EMOTION_TAXONOMY.forEach(({ key }) => {
    breakdown[key] = clamp01(rawBreakdown ? rawBreakdown[key] : 0);
  });

  return {
    status: 'success',
    chatbot_reply: reply,
    primary_emotion: primary,
    confidence_score: clamp01(raw.confidence_score),
    breakdown,
    hasVision,
  };
}

function normalizeFashionResponse(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ApiError('contract', 'The fashion service returned an unexpected response.');
  }

  if (raw.status === 'error') {
    return {
      status: 'error',
      error_message: asString(raw.error_message, 'That photo was not suitable for style analysis.'),
    };
  }

  const reply = asString(raw.chatbot_reply).trim();
  if (!reply) {
    throw new ApiError('contract', 'The fashion service response had no reply in it.');
  }

  return {
    status: 'success',
    chatbot_reply: reply,
    face_shape: asString(raw.face_shape),
    haircut_recommendation: asString(raw.haircut_recommendation),
    beard_style_recommendation: asString(raw.beard_style_recommendation),
    eyewear_recommendation: asString(raw.eyewear_recommendation),
    color_harmony_recommendation: asString(raw.color_harmony_recommendation),
  };
}

const EmotionService = {
  async analyze({ userQuery, image, history }) {
    const raw = await apiRequest(API_CONFIG.endpoints.emotion, {
      user_query: asString(userQuery),
      face_image_b64: image || null,
      chat_history: sanitizeHistory(history),
    });
    return normalizeEmotionResponse(raw);
  },
};

const FashionService = {
  async advise({ userQuery, image, history }) {
    const raw = await apiRequest(API_CONFIG.endpoints.fashion, {
      user_query: asString(userQuery),
      face_image_b64: image || null,
      chat_history: sanitizeHistory(history),
    });
    return normalizeFashionResponse(raw);
  },
};

/* ============================================================================
   4. DOM-SAFE HELPERS
   Every lookup tolerates a missing element instead of throwing.
   ============================================================================ */
function getEl(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const el = getEl(id);
  if (el) el.innerText = text;
}

function setBarWidth(id, pct) {
  const el = getEl(id);
  if (el) el.style.width = `${pct}%`;
}

function getActiveFaceImageB64() {
  try {
    const stored = localStorage.getItem('activeFaceSession'); // data:image/jpeg;base64,...
    if (stored && stored.startsWith('data:image')) return stored;
  } catch (err) {
    console.warn('localStorage unavailable:', err);
  }
  return activeFaceFallback;
}

function storeActiveFaceImage(dataUrl) {
  activeFaceFallback = dataUrl;
  try {
    localStorage.setItem('activeFaceSession', dataUrl);
  } catch (err) {
    console.warn('Could not persist the photo to localStorage; using in-memory copy.', err);
  }
}

function clearActiveFaceImage() {
  activeFaceFallback = null;
  try {
    localStorage.removeItem('activeFaceSession');
  } catch {
    /* ignore */
  }
}

function appendUserBubble(container, text) {
  const div = document.createElement('div');
  div.className = 'flex items-start justify-end space-x-3 max-w-[88%] ml-auto';
  div.innerHTML = `<div class="btn-neon text-white p-3.5 rounded-2xl rounded-tr-none shadow-md font-medium">${escapeHtml(text)}</div>`;
  container.appendChild(div);
}

function appendAssistantBubble(container, avatarLabel, initialText) {
  const id = `bubble-${avatarLabel}-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
  const div = document.createElement('div');
  div.className = 'flex items-start space-x-3 max-w-[88%]';
  div.innerHTML = `
    <div class="w-6 h-6 rounded-lg bg-purple-950 border border-purple-500/30 flex items-center justify-center text-[10px] font-bold text-purple-300 shrink-0">${avatarLabel}</div>
    <div id="${id}" class="bg-[#05050a] border border-white/10 p-3.5 rounded-2xl rounded-tl-none text-slate-200 leading-relaxed shadow-md">${escapeHtml(initialText)}</div>`;
  container.appendChild(div);
  return id;
}

function replaceBubble(id, content, isHtml = false) {
  const el = getEl(id);
  if (!el) return;
  if (isHtml) {
    el.innerHTML = content;
  } else {
    el.innerText = content;
  }
}

/** Loads a File into a data URL, rejecting (not hanging) on failure. */
function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('The image file could not be read.'));
    reader.onabort = () => reject(new Error('Reading the image file was cancelled.'));
    reader.readAsDataURL(file);
  });
}

/** compressImage() has no failure path inline, so guard it with a timeout. */
function compressImageSafe(dataUrl, maxWidth = 800, quality = 0.8) {
  return Promise.race([
    compressImage(dataUrl, maxWidth, quality),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('The image could not be processed. Try a different file.')), 15000)
    ),
  ]);
}

/* ============================================================================
   5. CAMERA — toggleCamera() + track cleanup
   Prefers a DroidCam device when one is present, otherwise uses the default
   physical webcam. Needs a secure context (http://localhost or
   http://127.0.0.1 is fine; opening the file via file:// may be blocked).
   ============================================================================ */
let cameraStream = null;
let cameraStarting = false;

function isCameraActive() {
  return !!(cameraStream && cameraStream.getTracks().some((t) => t.readyState === 'live'));
}

async function openCameraStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const e = new Error('Camera access is not supported here. Use a modern browser over http://localhost.');
    e.name = 'UnsupportedError';
    throw e;
  }

  const baseVideo = { width: { ideal: 1280 }, height: { ideal: 720 } };

  // Step 1: ask for any camera. Device labels (needed to spot DroidCam) are
  // only exposed by the browser after permission has been granted.
  let stream = await navigator.mediaDevices.getUserMedia({ video: baseVideo, audio: false });

  // Step 2: if a DroidCam device exists and we are not already on it, switch.
  // The first stream is only released after the second one succeeds, so a
  // failure here still leaves the user with a working default camera.
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const droid = devices.find((d) => d.kind === 'videoinput' && /droidcam/i.test(d.label || ''));
    const track = stream.getVideoTracks()[0];
    const currentId = track && track.getSettings ? track.getSettings().deviceId : null;

    if (droid && droid.deviceId && droid.deviceId !== currentId) {
      const droidStream = await navigator.mediaDevices.getUserMedia({
        video: { ...baseVideo, deviceId: { exact: droid.deviceId } },
        audio: false,
      });
      stream.getTracks().forEach((t) => t.stop());
      stream = droidStream;
    }
  } catch (err) {
    console.warn('DroidCam selection failed, keeping the default camera.', err);
  }

  return stream;
}

function cameraErrorToMessage(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera permission was denied. Allow camera access in your browser and try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera was found. Connect a webcam or start DroidCam, then try again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The camera is busy or unavailable. Close other apps using it and try again.';
    case 'OverconstrainedError':
      return 'The selected camera does not support the requested settings.';
    case 'SecurityError':
      return 'The browser blocked camera access. Open the app via http://localhost or http://127.0.0.1.';
    case 'UnsupportedError':
      return err.message;
    default:
      return (err && err.message) || 'Could not start the camera.';
  }
}

function showCameraUI(video) {
  const preview = getEl('uploaded-image-preview-f3');
  const placeholder = getEl('camera-placeholder-f3');
  const overlay = getEl('live-overlay-f3');

  if (preview) preview.classList.add('hidden');
  if (placeholder) placeholder.classList.add('hidden');
  video.classList.remove('hidden');
  if (overlay) {
    overlay.classList.remove('hidden');
    overlay.classList.add('flex');
  }
  setText('status-tag-text-f3', 'LIVE TELEMETRY');
}

/**
 * Stops every track and detaches the stream from the <video>.
 * restoreUI=true puts the placeholder back (unless an uploaded photo is on
 * screen); restoreUI=false leaves the surrounding UI alone.
 */
function stopCameraStream({ restoreUI = true } = {}) {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => {
      try {
        track.removeEventListener('ended', handleCameraTrackEnded);
        track.stop();
      } catch (err) {
        console.warn('Failed to stop a camera track:', err);
      }
    });
    cameraStream = null;
  }

  const video = getEl('webcam-feed');
  if (video) {
    try {
      video.pause();
    } catch {
      /* ignore */
    }
    video.srcObject = null;
    video.classList.add('hidden');
  }

  if (!restoreUI) return;

  const preview = getEl('uploaded-image-preview-f3');
  const placeholder = getEl('camera-placeholder-f3');
  const overlay = getEl('live-overlay-f3');
  const photoVisible = preview && !preview.classList.contains('hidden');

  if (!photoVisible) {
    if (placeholder) placeholder.classList.remove('hidden');
    if (overlay) {
      overlay.classList.add('hidden');
      overlay.classList.remove('flex');
    }
  }
}

function handleCameraTrackEnded() {
  // Fired when the device is unplugged or DroidCam disconnects.
  if (!cameraStream) return;
  stopCameraStream({ restoreUI: true });
  notifyError('Camera Disconnected', 'The camera stream ended. Reconnect the device and start it again.');
}

/** Grabs the current webcam frame as a compressed JPEG data URL, or null. */
function captureLiveFrameB64(maxWidth = 800, quality = 0.8) {
  try {
    const video = getEl('webcam-feed');
    if (!video || !isCameraActive() || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      return null;
    }
    const scale = Math.min(1, maxWidth / Math.max(video.videoWidth, video.videoHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', quality);
  } catch (err) {
    console.warn('Could not capture a webcam frame:', err);
    return null;
  }
}

/** Wired from the camera button in spa.html: onclick="toggleCamera()". */
async function toggleCamera() {
  if (cameraStarting) return;

  if (isCameraActive()) {
    stopCameraStream({ restoreUI: true });
    return;
  }

  cameraStarting = true;
  try {
    const video = getEl('webcam-feed');
    if (!video) throw new Error('The webcam element (#webcam-feed) was not found on the page.');

    const stream = await openCameraStream();
    cameraStream = stream;
    stream.getVideoTracks().forEach((t) => t.addEventListener('ended', handleCameraTrackEnded));

    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    try {
      await video.play();
    } catch (playErr) {
      // AbortError just means a new load interrupted play(); autoplay covers it.
      if (playErr && playErr.name !== 'AbortError') console.warn('video.play() failed:', playErr);
    }

    showCameraUI(video);
  } catch (err) {
    console.error('toggleCamera failed:', err);
    stopCameraStream({ restoreUI: true });
    notifyError('Camera Unavailable', cameraErrorToMessage(err));
  } finally {
    cameraStarting = false;
  }
}

/* ----------------------------------------------------------------------
   OVERRIDE: resetTelemetryView / resetScannerView
   Same behaviour as the inline versions (clear photo, reload), but the
   camera tracks are stopped first so the webcam light actually turns off.
   ---------------------------------------------------------------------- */
function resetTelemetryView() {
  stopCameraStream({ restoreUI: false });
  clearActiveFaceImage();
  window.location.reload();
}

function resetScannerView() {
  stopCameraStream({ restoreUI: false });
  clearActiveFaceImage();
  window.location.reload();
}

/* ============================================================================
   6. EMOTION PANEL (F3) + F3 CHAT + IMAGE UPLOAD
   ============================================================================ */
function emotionMetaFromLabel(label) {
  return EMOTION_TAXONOMY.find((entry) => entry.label === label) || null;
}

function updateEmotionPanel(data) {
  // Only touch the panel when the response actually carried a vision reading.
  if (!data || data.status !== 'success' || !data.hasVision) return;

  const meta = emotionMetaFromLabel(data.primary_emotion);
  setText('emotion-primary-label', data.primary_emotion);
  setText('emotion-primary-emoji', meta ? meta.emoji : '🙂');
  setText('emotion-confidence', `${(data.confidence_score * 100).toFixed(1)}%`);

  const breakdown = data.breakdown || {};
  EMOTION_TAXONOMY.forEach(({ key }) => {
    const pct = (toNumber(breakdown[key]) * 100).toFixed(1);
    setText(`emotion-${key}-val`, `${pct}%`);
    setBarWidth(`emotion-${key}-bar`, pct);
  });
}

/**
 * Sends a base64/data-URI face image + a text query to the emotion endpoint,
 * syncs the result into the F3 metrics panel and returns the normalized
 * result. Throws ApiError on failure.
 */
async function analyzeEmotionFrame(base64Image, userQuery, chatHistory) {
  const data = await EmotionService.analyze({
    userQuery,
    image: base64Image,
    history: chatHistory,
  });
  updateEmotionPanel(data);
  return data;
}

/* ----------------------------------------------------------------------
   OVERRIDE: handleImageUpload
   Same compress/store/preview flow as the mock, plus an immediate live
   analysis the moment a new photo lands.
   ---------------------------------------------------------------------- */
async function handleImageUpload(e) {
  const input = e && e.target;
  await runGuarded(async () => {
    const file = input && input.files && input.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      notifyError('Unsupported File Format', 'Please upload a valid image file (JPEG, PNG, WEBP).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      notifyError('File Too Large', 'Please upload an image smaller than 10MB.');
      return;
    }

    const dataUrl = await readFileAsDataUrl(file);
    const compressedImage = await compressImageSafe(dataUrl, 800, 0.8);

    // A photo replaces the live feed, so release the camera first.
    stopCameraStream({ restoreUI: false });
    storeActiveFaceImage(compressedImage);
    processImageSrc(compressedImage, 'AI GEOMETRY SCANNER ACTIVE');

    try {
      const result = await analyzeEmotionFrame(
        compressedImage,
        'Give a quick initial read of my current expression.',
        emotionChatHistory
      );
      if (result.status === 'error') {
        notifyError('Photo Not Suitable', result.error_message || 'Please retake the photo.');
      }
    } catch (err) {
      console.error('Initial emotion analysis failed:', err);
      const { title, desc } = describeError(err, 'Emotion Analysis Failed');
      notifyError(title, desc);
    }
  }, 'Image Upload Failed');

  // Allow re-selecting the same file later.
  try {
    if (input) input.value = '';
  } catch {
    /* ignore */
  }
}

/* ----------------------------------------------------------------------
   OVERRIDE: handleChatSubmitF3 (Emotion Assistant)
   When the webcam is live, the current frame is captured and sent with the
   message; otherwise the stored uploaded photo is used.
   ---------------------------------------------------------------------- */
function handleChatSubmitF3(e) {
  if (e && typeof e.preventDefault === 'function') e.preventDefault();
  return runGuarded(() => submitEmotionChat(), 'Emotion Assistant Failed');
}

async function submitEmotionChat() {
  const input = getEl('chat-input-f3');
  const container = getEl('chat-messages-f3');
  if (!input || !container) throw new Error('The emotion chat panel is missing from the page.');

  const text = input.value.trim();
  if (!text) return;

  appendUserBubble(container, text);
  input.value = '';
  container.scrollTop = container.scrollHeight;

  const activeImage = captureLiveFrameB64() || getActiveFaceImageB64();
  const typingId = appendAssistantBubble(container, 'EA', 'Evaluating query against active frame metrics…');
  container.scrollTop = container.scrollHeight;

  try {
    const data = await EmotionService.analyze({
      userQuery: text,
      image: activeImage,
      history: emotionChatHistory,
    });

    if (data.status === 'error') {
      replaceBubble(typingId, data.error_message || 'That frame was not suitable for analysis.');
      return;
    }

    updateEmotionPanel(data);
    const reply = data.chatbot_reply || 'I could not put together a reply this time — please try again.';
    replaceBubble(typingId, reply);

    if (data.chatbot_reply) {
      emotionChatHistory.push({ role: 'user', content: text });
      emotionChatHistory.push({ role: 'assistant', content: data.chatbot_reply });
    }
  } catch (err) {
    console.error('Emotion chat failed:', err);
    if (err instanceof ApiError && err.kind === 'rate_limit') {
      replaceBubble(typingId, err.message);
    } else {
      replaceBubble(typingId, 'Sorry, the emotion assistant is unavailable right now.');
      const { title, desc } = describeError(err, 'Emotion Assistant Failed');
      notifyError(title, desc);
    }
  } finally {
    container.scrollTop = container.scrollHeight;
  }
}

/* ============================================================================
   7. F4 CHAT (Fashion & Grooming Assistant)
   Renders the conversational reply plus a compact recommendation summary and
   syncs the detected face shape into the F4 panel.
   ============================================================================ */
function handleChatSubmitF4(e) {
  if (e && typeof e.preventDefault === 'function') e.preventDefault();
  return runGuarded(() => submitFashionChat(), 'Fashion Assistant Failed');
}

async function submitFashionChat() {
  const input = getEl('chat-input-f4');
  const container = getEl('chat-messages-f4');
  if (!input || !container) throw new Error('The fashion chat panel is missing from the page.');

  const text = input.value.trim();
  if (!text) return;

  appendUserBubble(container, text);
  input.value = '';
  container.scrollTop = container.scrollHeight;

  const activeImage = getActiveFaceImageB64();
  const typingId = appendAssistantBubble(container, 'FA', 'Consulting Gemini…');
  container.scrollTop = container.scrollHeight;

  try {
    const data = await FashionService.advise({
      userQuery: text,
      image: activeImage,
      history: fashionChatHistory,
    });

    if (data.status === 'error') {
      replaceBubble(typingId, data.error_message || 'That photo was not suitable for style analysis.');
      return;
    }

    const row = (label, value) =>
      `<div><span class="text-purple-300 font-semibold">${label}:</span> ${escapeHtml(value || '—')}</div>`;

    const replyHtml =
      escapeHtml(data.chatbot_reply) +
      `
      <div class="mt-2 pt-2 border-t border-white/10 space-y-1 text-[11px] text-slate-400">
        ${row('Face Shape', data.face_shape)}
        ${row('Haircut', data.haircut_recommendation)}
        ${row('Beard', data.beard_style_recommendation)}
        ${row('Eyewear', data.eyewear_recommendation)}
        ${row('Color Harmony', data.color_harmony_recommendation)}
      </div>`;
    replaceBubble(typingId, replyHtml, true);

    fashionChatHistory.push({ role: 'user', content: text });
    fashionChatHistory.push({ role: 'assistant', content: data.chatbot_reply });

    if (data.face_shape) {
      const label = getEl('face-shape-label');
      if (label) {
        label.innerText = data.face_shape;
        label.className = 'text-white font-bold text-xl';
      }
      setText('face-shape-sub', '(via Gemini)');
    }
  } catch (err) {
    console.error('Fashion chat failed:', err);
    if (err instanceof ApiError && err.kind === 'rate_limit') {
      replaceBubble(typingId, err.message);
    } else {
      replaceBubble(typingId, 'Sorry, the fashion assistant is unavailable right now.');
      const { title, desc } = describeError(err, 'Fashion Assistant Failed');
      notifyError(title, desc);
    }
  } finally {
    container.scrollTop = container.scrollHeight;
  }
}

/* ============================================================================
   8. GLOBAL SAFETY NETS
   ============================================================================ */
// Release the camera when the tab closes or the user navigates away from F3.
window.addEventListener('pagehide', () => stopCameraStream({ restoreUI: false }));
window.addEventListener('hashchange', () => {
  if (window.location.hash !== '#f3' && cameraStream) stopCameraStream({ restoreUI: true });
});

// Anything that slips past the guards above still becomes a toast rather than
// a silent failure. Throttled so a repeating error cannot spam the user.
let lastGlobalToastAt = 0;
function throttledGlobalToast(title, desc) {
  const now = Date.now();
  if (now - lastGlobalToastAt < 4000) return;
  lastGlobalToastAt = now;
  notifyError(title, desc);
}

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  const { title, desc } = describeError(event.reason, 'Unexpected Error');
  throttledGlobalToast(title, desc);
});

window.addEventListener('error', (event) => {
  if (!event.error) return; // ignore resource-load noise
  console.error('Uncaught error:', event.error);
  throttledGlobalToast('Unexpected Error', event.error.message || 'Something went wrong in the page.');
});
