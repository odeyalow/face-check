import { createCameraController } from "./modules/cameras.js";
import { createObservation } from "./modules/observation.js";

const video = document.getElementById("video");
const streamCanvas = document.getElementById("stream");
const canvas = document.getElementById("overlay");
const namesEl = document.getElementById("names");
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
const pendingByName = new Map();
const lastLoggedMoodByName = new Map();
let knownFacesLoading = false;
let knownFacesLastAttempt = 0;
const faceSourceCanvas = document.createElement("canvas");
let loopStarted = false;
const stableWindow = 6;
const stableMinHits = 3;
const stableKeepMs = 1500;
const recentLabelFrames = [];
const lastSeenByLabel = new Map();
const lastMoodByLabel = new Map();
const stableOrder = [];

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
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 }))
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (detection) descriptors.push(detection.descriptor);
      }
      if (descriptors.length) {
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
      }
    }

    if (labeledDescriptors.length) {
      faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.7);
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
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 });

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

    let detection = faceapi.detectAllFaces(detectionSource, opts);
    if (recognitionReady) {
      detection = detection.withFaceLandmarks().withFaceDescriptors();
    }
    const results = await detection.withFaceExpressions();

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results && results.length > 0) {
      const resized = faceapi.resizeResults(results, { width: canvas.width, height: canvas.height });
      faceapi.draw.drawDetections(canvas, resized);

      const nowPerf = performance.now();
      const personMoods = new Map();
      for (const result of results) {
        if (!result.expressions) continue;
        const { bestKey } = bestExpression(result.expressions);
        if (recognitionReady && faceMatcher && result.descriptor) {
          const match = faceMatcher.findBestMatch(result.descriptor);
          if (match.label && match.label !== "unknown") {
            if (!personMoods.has(match.label)) personMoods.set(match.label, bestKey);
          }
        }
      }

      const detectedNames = Array.from(personMoods.keys());
      const uniqueFrameLabels = Array.from(new Set(detectedNames));
      recentLabelFrames.push(uniqueFrameLabels);
      if (recentLabelFrames.length > stableWindow) recentLabelFrames.shift();

      for (const label of uniqueFrameLabels) {
        lastSeenByLabel.set(label, nowPerf);
        lastMoodByLabel.set(label, personMoods.get(label));
      }

      const counts = new Map();
      for (const frameLabels of recentLabelFrames) {
        for (const label of frameLabels) {
          counts.set(label, (counts.get(label) || 0) + 1);
        }
      }

      const stableLabels = [];
      for (const [label, count] of counts.entries()) {
        const lastSeen = lastSeenByLabel.get(label) || 0;
        const recentlySeen = nowPerf - lastSeen <= stableKeepMs;
        if (count >= stableMinHits || recentlySeen) stableLabels.push(label);
      }
      for (const label of stableLabels) {
        if (!stableOrder.includes(label)) stableOrder.push(label);
      }
      for (let i = stableOrder.length - 1; i >= 0; i -= 1) {
        if (!stableLabels.includes(stableOrder[i])) stableOrder.splice(i, 1);
      }
      const orderedLabels = stableOrder.slice();

      if (namesEl) {
        if (orderedLabels.length === 0) {
          namesEl.textContent = "--";
        } else {
          namesEl.innerHTML = "";
          for (const name of orderedLabels) {
            const line = document.createElement("div");
            const moodKey = lastMoodByLabel.get(name);
            line.textContent = moodKey ? `${name}: ${moodLabel(moodKey)}` : `${name}: --`;
            namesEl.appendChild(line);
          }
        }
      }

      let currentName = null;
      let bestKey = null;
      if (orderedLabels.length > 0) {
        currentName = orderedLabels[0];
        bestKey = lastMoodByLabel.get(currentName) || null;
      }
      lastName = currentName || "--";
      lastMood = bestKey || "--";
      const currentMoodMap = new Map();
      for (const label of orderedLabels) {
        const moodKey = lastMoodByLabel.get(label);
        if (moodKey) currentMoodMap.set(label, moodKey);
      }

      if (observation.isActive()) {
        for (const [name, moodKey] of currentMoodMap.entries()) {
          const pending = pendingByName.get(name);
          if (!pending || pending.mood !== moodKey) {
            pendingByName.set(name, { mood: moodKey, since: nowPerf });
          } else if (nowPerf - pending.since >= 3000) {
            const lastLogged = lastLoggedMoodByName.get(name);
            if (lastLogged !== moodKey) {
              observation.addLogLine(name, moodKey);
              lastLoggedMoodByName.set(name, moodKey);
            }
          }
        }
        for (const name of pendingByName.keys()) {
          if (!currentMoodMap.has(name)) {
            pendingByName.delete(name);
            lastLoggedMoodByName.delete(name);
          }
        }
      } else {
        pendingByName.clear();
        lastLoggedMoodByName.clear();
      }

      observation.updateFrame(currentMoodMap, nowPerf);
    } else {
      recentLabelFrames.push([]);
      if (recentLabelFrames.length > stableWindow) recentLabelFrames.shift();
      if (namesEl) namesEl.textContent = "--";
      lastMood = "--";
      lastName = "--";
      pendingByName.clear();
      lastLoggedMoodByName.clear();
      observation.updateFrame(new Map(), performance.now());
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
