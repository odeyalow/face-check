import { createCameraController } from "./modules/cameras.js";
import { createObservation } from "./modules/observation.js";

const video = document.getElementById("video");
const streamCanvas = document.getElementById("stream");
const canvas = document.getElementById("overlay");
const moodEl = document.getElementById("mood");
const nameEl = document.getElementById("name");
const confEl = document.getElementById("conf");
const fpsEl = document.getElementById("fps");
const logsEl = document.getElementById("logs");
const loadingStatusEl = document.getElementById("loadingStatus");
const mediaWrapEl = document.getElementById("mediaWrap");
const rtspNameInput = document.getElementById("rtspName");
const rtspInput = document.getElementById("rtspUrl");
const addRtspBtn = document.getElementById("addRtsp");
const rtspSelect = document.getElementById("rtspSelect");
const rtspEmptyEl = document.getElementById("rtspEmpty");
const connectRtspBtn = document.getElementById("connectRtsp");
const useWebcamBtn = document.getElementById("useWebcam");
const deleteRtspBtn = document.getElementById("deleteRtsp");
const observationToggleBtn = document.getElementById("observationToggle");
const observationResultsEl = document.getElementById("observationResults");
const observationDurationEl = document.getElementById("observationDuration");
const observationTopMoodEl = document.getElementById("observationTopMood");
const observationPeopleSelectEl = document.getElementById("observationPeopleSelect");
const observationPeopleEl = document.getElementById("observationPeople");
const clearObservationBtn = document.getElementById("clearObservation");
const clearLogsBtn = document.getElementById("clearLogs");

const MODEL_URL = "/models";

let sourceEl = video;
let faceMatcher = null;
let recognitionReady = false;
let lastName = "--";
let lastMood = "--";
let lastLoggedName = null;
let lastLoggedMood = null;
let pendingName = null;
let pendingMood = null;
let pendingSince = 0;
let knownFacesLoading = false;
let knownFacesLastAttempt = 0;
const faceSourceCanvas = document.createElement("canvas");
let loopStarted = false;

function moodLabel(key) {
  const map = {
    neutral: "Нейтральный",
    happy: "Счастливый",
    sad: "Грустный",
    angry: "Злой"
  };
  return map[key] || key;
}

const observation = createObservation({
  logsEl,
  observationResultsEl,
  observationDurationEl,
  observationTopMoodEl,
  observationPeopleEl,
  observationPeopleSelectEl,
  observationToggleBtn,
  clearObservationBtn,
  moodLabel
});

const cameraController = createCameraController({
  video,
  streamCanvas,
  loadingStatusEl,
  mediaWrapEl,
  rtspNameInput,
  rtspInput,
  addRtspBtn,
  rtspSelect,
  rtspEmptyEl,
  connectRtspBtn,
  deleteRtspBtn,
  useWebcamBtn,
  onSourceChange: (el) => {
    sourceEl = el;
  }
});

function bestExpression(expressions) {
  let bestKey = "neutral";
  let bestVal = 0;

  for (const [k, v] of Object.entries(expressions)) {
    if (v > bestVal) {
      bestVal = v;
      bestKey = k;
    }
  }

  const allowed = ["neutral", "happy", "sad", "angry"];
  if (!allowed.includes(bestKey)) bestKey = "neutral";

  return { bestKey, bestVal };
}

async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
  try {
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    recognitionReady = true;
  } catch (_err) {
    recognitionReady = false;
  }
}

function labelFromFilename(filename) {
  const base = filename.replace(/\.[^/.]+$/, "");
  return base.replace(/[-_]\d+$/, "");
}

async function loadKnownFaces() {
  if (!recognitionReady) return;
  try {
    const res = await fetch("/known/images.json");
    if (!res.ok) return;
    const images = await res.json();
    if (!Array.isArray(images) || images.length === 0) return;

    const labelMap = new Map();
    images.forEach((file) => {
      const label = labelFromFilename(file);
      if (!label) return;
      if (!labelMap.has(label)) labelMap.set(label, []);
      labelMap.get(label).push(`/known/${encodeURIComponent(file)}`);
    });

    const labeledDescriptors = [];
    for (const [label, urls] of labelMap.entries()) {
      const descriptors = [];
      for (const url of urls) {
        const img = await faceapi.fetchImage(url);
        const detection = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (detection) descriptors.push(detection.descriptor);
      }
      if (descriptors.length) {
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
      }
    }

    if (labeledDescriptors.length) {
      faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
    }
  } catch (_err) {
    console.warn("Failed to load known faces.");
  }
}

function ensureKnownFaces() {
  if (!recognitionReady || faceMatcher || knownFacesLoading) return;
  const now = Date.now();
  if (now - knownFacesLastAttempt < 5000) return;
  knownFacesLastAttempt = now;
  knownFacesLoading = true;
  loadKnownFaces().finally(() => {
    knownFacesLoading = false;
  });
}

let lastTs = performance.now();
let frames = 0;

function syncOverlaySize() {
  const width = sourceEl === video ? video.videoWidth : (sourceEl.width || sourceEl.clientWidth);
  const height = sourceEl === video ? video.videoHeight : (sourceEl.height || sourceEl.clientHeight);

  if (!width || !height) return false;
  if (sourceEl === streamCanvas && (!streamCanvas.width || !streamCanvas.height)) {
    streamCanvas.width = width;
    streamCanvas.height = height;
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return true;
}

function getDetectionSource() {
  if (sourceEl === video) return video;
  const width = sourceEl.width || sourceEl.clientWidth || canvas.width;
  const height = sourceEl.height || sourceEl.clientHeight || canvas.height;
  if (!width || !height) return null;
  if (faceSourceCanvas.width !== width || faceSourceCanvas.height !== height) {
    faceSourceCanvas.width = width;
    faceSourceCanvas.height = height;
  }
  const ctx = faceSourceCanvas.getContext("2d");
  ctx.drawImage(sourceEl, 0, 0, width, height);
  return faceSourceCanvas;
}

async function loop() {
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

  try {
    frames++;
    const now = performance.now();
    if (now - lastTs >= 1000) {
      if (fpsEl) fpsEl.textContent = String(frames);
      frames = 0;
      lastTs = now;
    }

    ensureKnownFaces();
    if (!syncOverlaySize()) return;

    const detectionSource = getDetectionSource();
    if (!detectionSource) return;

    let detection = faceapi.detectSingleFace(detectionSource, opts);
    if (recognitionReady) {
      detection = detection.withFaceLandmarks().withFaceDescriptor();
    }
    const result = await detection.withFaceExpressions();

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result) {
      const resized = faceapi.resizeResults(result, { width: canvas.width, height: canvas.height });
      faceapi.draw.drawDetections(canvas, resized);

      const { bestKey, bestVal } = bestExpression(result.expressions);
      const nowPerf = performance.now();
      if (moodEl) moodEl.textContent = moodLabel(bestKey);
      if (confEl) confEl.textContent = (bestVal * 100).toFixed(1) + "%";
      lastMood = bestKey;
      let currentName = null;
      if (recognitionReady && faceMatcher && nameEl) {
        const match = faceMatcher.findBestMatch(result.descriptor);
        if (match.label && match.label !== "unknown") {
          nameEl.textContent = match.label;
          lastName = match.label;
          currentName = match.label;
        } else {
          nameEl.textContent = "--";
          lastName = "--";
        }
      } else if (nameEl) {
        nameEl.textContent = "--";
        lastName = "--";
      }

      if (currentName) {
        if (pendingName !== currentName || pendingMood !== bestKey) {
          pendingName = currentName;
          pendingMood = bestKey;
          pendingSince = nowPerf;
        } else if (nowPerf - pendingSince >= 3000) {
          if (currentName !== lastLoggedName || bestKey !== lastLoggedMood) {
            if (observation.isActive()) observation.addLogLine(currentName, bestKey);
            lastLoggedName = currentName;
            lastLoggedMood = bestKey;
          }
        }
        observation.updateTiming(currentName, bestKey, nowPerf);
      } else {
        lastLoggedName = null;
        lastLoggedMood = null;
        pendingName = null;
        pendingMood = null;
        pendingSince = 0;
        observation.flushTiming(nowPerf);
      }
    } else {
      if (moodEl) moodEl.textContent = "--";
      if (confEl) confEl.textContent = "--";
      if (nameEl) nameEl.textContent = "--";
      lastMood = "--";
      lastName = "--";
      lastLoggedName = null;
      lastLoggedMood = null;
      pendingName = null;
      pendingMood = null;
      pendingSince = 0;
      observation.flushTiming(performance.now());
    }
  } catch (err) {
    console.error("Detection loop error:", err);
  } finally {
    setTimeout(loop, 120);
  }
}

(async () => {
  await loadModels();
  await loadKnownFaces();
  try {
    await cameraController.startWebcam();
  } catch (_err) {
    // Ignore webcam errors (insecure context or no device).
  }
  if (!loopStarted) {
    loopStarted = true;
    loop();
  }
})();

if (clearLogsBtn) {
  clearLogsBtn.addEventListener("click", () => {
    if (logsEl) logsEl.textContent = "";
  });
}
