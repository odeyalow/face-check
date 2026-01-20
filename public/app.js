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
const connectRtspBtn = document.getElementById("connectRtsp");
const useWebcamBtn = document.getElementById("useWebcam");
const clearLogsBtn = document.getElementById("clearLogs");
const deleteRtspBtn = document.getElementById("deleteRtsp");

const MODEL_URL = "/models";

let webcamStream = null;
let player = null;
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

function stopWebcam() {
  if (!webcamStream) return;
  webcamStream.getTracks().forEach((track) => track.stop());
  webcamStream = null;
  video.srcObject = null;
}

function setLoadingStatus(isLoading) {
  if (!loadingStatusEl) return;
  loadingStatusEl.classList.toggle("hidden", !isLoading);
  if (mediaWrapEl) mediaWrapEl.classList.toggle("hidden", isLoading);
}

async function startWebcam() {
  if (player?.destroy) player.destroy();
  player = null;
  stopWebcam();
  setLoadingStatus(true);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false
  });
  webcamStream = stream;
  video.srcObject = stream;

  await new Promise((res) => (video.onloadedmetadata = res));
  video.play();

  video.classList.remove("hidden");
  streamCanvas.classList.add("hidden");
  sourceEl = video;
  setLoadingStatus(false);
}

async function startRtsp(rtspUrl) {
  if (!rtspUrl || !window.loadPlayer) return;

  stopWebcam();
  if (player?.destroy) player.destroy();
  setLoadingStatus(true);

  const wsProto = location.protocol === "https:" ? "wss://" : "ws://";
  const wsUrl = `${wsProto}${location.host}/api/stream?url=${encodeURIComponent(rtspUrl)}`;

  video.classList.add("hidden");
  streamCanvas.classList.remove("hidden");
  sourceEl = streamCanvas;

  player = await window.loadPlayer({
    url: wsUrl,
    canvas: streamCanvas,
    audio: false,
    disableGl: true
  });

  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (streamCanvas.width > 0 && streamCanvas.height > 0) break;
    await new Promise((res) => setTimeout(res, 100));
  }

  if (!streamCanvas.width || !streamCanvas.height) {
    streamCanvas.width = 640;
    streamCanvas.height = 480;
  }
  setLoadingStatus(false);
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
      if (moodEl) moodEl.textContent = bestKey;
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
            addLogLine(currentName, bestKey);
            lastLoggedName = currentName;
            lastLoggedMood = bestKey;
          }
        }
      } else {
        lastLoggedName = null;
        lastLoggedMood = null;
        pendingName = null;
        pendingMood = null;
        pendingSince = 0;
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
    await startWebcam();
  } catch (_err) {
    // Ignore webcam errors (insecure context or no device).
  }
  if (!loopStarted) {
    loopStarted = true;
    loop();
  }
})();

function formatTime(date) {
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function addLogLine(name, mood) {
  if (!logsEl) return;
  const time = formatTime(new Date());
  const line = `${name} ${mood} ${time}`;
  const div = document.createElement("div");
  div.textContent = line;
  logsEl.prepend(div);
}

function loadRtspList() {
  try {
    const raw = localStorage.getItem("rtspUrls");
    const list = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(list)) return [];
    if (list.length === 0) return [];
    if (typeof list[0] === "string") {
      return list.map((url, idx) => ({
        url,
        name: idx === 0 ? "Без названия" : `Без названия ${idx}`
      }));
    }
    return list.filter((item) => item && typeof item.url === "string");
  } catch (_err) {
    return [];
  }
}

function saveRtspList(list) {
  localStorage.setItem("rtspUrls", JSON.stringify(list));
}

function normalizeName(name) {
  return name ? name.trim() : "";
}

function generateDefaultName(existingNames) {
  const base = "Без названия";
  if (!existingNames.has(base)) return base;
  let i = 1;
  while (existingNames.has(`${base} ${i}`)) i++;
  return `${base} ${i}`;
}

function syncRtspSelect(list, selectedUrl) {
  if (!rtspSelect) return;
  rtspSelect.innerHTML = "";
  const emptyOpt = document.createElement("option");
  emptyOpt.value = "";
  emptyOpt.textContent = "-- Выберите камеру --";
  rtspSelect.appendChild(emptyOpt);
  list.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.url;
    opt.textContent = item.name;
    opt.title = item.url;
    rtspSelect.appendChild(opt);
  });
  if (selectedUrl) rtspSelect.value = selectedUrl;
}

const savedRtspUrl = localStorage.getItem("rtspUrl");
const rtspList = loadRtspList();
if (savedRtspUrl && rtspInput) rtspInput.value = savedRtspUrl;
if (savedRtspUrl && rtspNameInput) {
  const savedItem = rtspList.find((item) => item.url === savedRtspUrl);
  if (savedItem) rtspNameInput.value = savedItem.name;
}
syncRtspSelect(rtspList, savedRtspUrl || "");

if (addRtspBtn) {
  addRtspBtn.addEventListener("click", () => {
    const rtspUrl = rtspInput?.value.trim();
    if (!rtspUrl) return;
    const nameRaw = normalizeName(rtspNameInput?.value);
    const list = loadRtspList();
    const existing = list.find((item) => item.url === rtspUrl);
    if (existing) {
      if (nameRaw) existing.name = nameRaw;
    } else {
      const existingNames = new Set(list.map((item) => item.name));
      const name = nameRaw || generateDefaultName(existingNames);
      list.push({ url: rtspUrl, name });
    }
    saveRtspList(list);
    localStorage.setItem("rtspUrl", rtspUrl);
    if (rtspNameInput) {
      const current = list.find((item) => item.url === rtspUrl);
      rtspNameInput.value = current ? current.name : "";
    }
    syncRtspSelect(list, rtspUrl);
  });
}

if (rtspSelect) {
  rtspSelect.addEventListener("change", async () => {
    const rtspUrl = rtspSelect.value;
    if (rtspInput) rtspInput.value = rtspUrl;
    if (rtspNameInput) {
      const item = loadRtspList().find((entry) => entry.url === rtspUrl);
      rtspNameInput.value = item ? item.name : "";
    }
    if (!rtspUrl) return;
    localStorage.setItem("rtspUrl", rtspUrl);
    await startRtsp(rtspUrl);
    if (!loopStarted) {
      loopStarted = true;
      loop();
    }
  });
}

if (connectRtspBtn) {
  connectRtspBtn.addEventListener("click", async () => {
    const rtspUrl = (rtspSelect?.value || rtspInput?.value || "").trim();
    if (!rtspUrl) return;
    const nameRaw = normalizeName(rtspNameInput?.value);
    localStorage.setItem("rtspUrl", rtspUrl);
    if (rtspInput) rtspInput.value = rtspUrl;
    const list = loadRtspList();
    const existing = list.find((item) => item.url === rtspUrl);
    if (existing) {
      if (nameRaw) existing.name = nameRaw;
    } else if (rtspSelect) {
      const existingNames = new Set(list.map((item) => item.name));
      const name = nameRaw || generateDefaultName(existingNames);
      list.push({ url: rtspUrl, name });
    }
    if (rtspSelect) {
      saveRtspList(list);
      syncRtspSelect(list, rtspUrl);
    }
    if (rtspNameInput) {
      const current = list.find((item) => item.url === rtspUrl);
      rtspNameInput.value = current ? current.name : "";
    }
    await startRtsp(rtspUrl);
    if (!loopStarted) {
      loopStarted = true;
      loop();
    }
  });
}

if (useWebcamBtn) {
  useWebcamBtn.addEventListener("click", async () => {
    await startWebcam();
    if (!loopStarted) {
      loopStarted = true;
      loop();
    }
  });
}

if (clearLogsBtn) {
  clearLogsBtn.addEventListener("click", () => {
    if (logsEl) logsEl.textContent = "";
  });
}

if (deleteRtspBtn) {
  deleteRtspBtn.addEventListener("click", () => {
    const rtspUrl = rtspSelect?.value || "";
    if (!rtspUrl) return;
    const list = loadRtspList();
    const item = list.find((entry) => entry.url === rtspUrl);
    const label = item ? item.name : rtspUrl;
    const ok = window.confirm(`Вы точно хотите удалить камеру "${label}"?`);
    if (!ok) return;
    const nextList = list.filter((entry) => entry.url !== rtspUrl);
    saveRtspList(nextList);
    syncRtspSelect(nextList, "");
    if (rtspInput) rtspInput.value = "";
    if (rtspNameInput) rtspNameInput.value = "";
    if (localStorage.getItem("rtspUrl") === rtspUrl) {
      localStorage.removeItem("rtspUrl");
    }
  });
}

