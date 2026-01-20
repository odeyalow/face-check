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
const observationToggleBtn = document.getElementById("observationToggle");
const observationResultsEl = document.getElementById("observationResults");
const observationDurationEl = document.getElementById("observationDuration");
const observationTopMoodEl = document.getElementById("observationTopMood");
const observationPeopleSelectEl = document.getElementById("observationPeopleSelect");
const observationPeopleEl = document.getElementById("observationPeople");
const clearObservationBtn = document.getElementById("clearObservation");

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

let observationActive = false;
let observationStart = 0;
let observationPeople = new Map();
let observationCurrentName = null;
let observationCurrentMood = null;
let observationCurrentAt = 0;
let observationDurations = new Map();

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

function setLoadingStatus(isLoading, message = "Загрузка камеры...", isError = false) {
  if (!loadingStatusEl) return;
  loadingStatusEl.classList.toggle("hidden", !isLoading);
  loadingStatusEl.textContent = message;
  loadingStatusEl.classList.toggle("error", isError);
  if (mediaWrapEl) mediaWrapEl.classList.toggle("hidden", isLoading);
}

function syncAddButtonState() {
  if (!addRtspBtn) return;
  const hasUrl = Boolean(rtspInput && rtspInput.value.trim());
  addRtspBtn.disabled = !hasUrl;
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

  const timeoutMs = 12000;
  await Promise.race([
    new Promise((res) => (video.onloadedmetadata = res)),
    new Promise((_, rej) => setTimeout(() => rej(new Error("webcam timeout")), timeoutMs))
  ]);
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

  const timeoutMs = 12000;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (streamCanvas.width > 0 && streamCanvas.height > 0) break;
    await new Promise((res) => setTimeout(res, 100));
  }

  if (!streamCanvas.width || !streamCanvas.height) {
    setLoadingStatus(true, "Не удалось загрузить поток", true);
    if (mediaWrapEl) mediaWrapEl.classList.add("hidden");
    return;
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
            if (observationActive) {
              addLogLine(currentName, bestKey);
            }
            lastLoggedName = currentName;
            lastLoggedMood = bestKey;
          }
        }
        if (observationActive) {
          updateObservationTiming(currentName, bestKey, nowPerf);
        }
      } else {
        lastLoggedName = null;
        lastLoggedMood = null;
        pendingName = null;
        pendingMood = null;
        pendingSince = 0;
        if (observationActive) {
          flushObservationTiming(nowPerf);
        }
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
      if (observationActive) {
        flushObservationTiming(performance.now());
      }
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

function formatDurationLabel(ms) {
  if (!Number.isFinite(ms)) return "";
  const total = Math.max(0, Math.round(ms / 1000));
  const hh = Math.floor(total / 3600);
  const mm = Math.floor((total % 3600) / 60);
  const ss = total % 60;
  if (hh > 0) return `${hh} ч ${String(mm).padStart(2, "0")} мин ${String(ss).padStart(2, "0")} сек`;
  if (mm > 0) return `${mm} мин ${String(ss).padStart(2, "0")} сек`;
  return `${ss} сек`;
}

function addDuration(name, mood, deltaMs) {
  if (!observationDurations.has(name)) observationDurations.set(name, new Map());
  const moodMap = observationDurations.get(name);
  moodMap.set(mood, (moodMap.get(mood) || 0) + deltaMs);
}

function updateObservationTiming(name, mood, nowPerf) {
  if (!observationCurrentName) {
    observationCurrentName = name;
    observationCurrentMood = mood;
    observationCurrentAt = nowPerf;
    return;
  }

  const delta = nowPerf - observationCurrentAt;
  if (delta > 0) {
    addDuration(observationCurrentName, observationCurrentMood, delta);
  }

  if (observationCurrentName !== name || observationCurrentMood !== mood) {
    observationCurrentName = name;
    observationCurrentMood = mood;
  }
  observationCurrentAt = nowPerf;
}

function flushObservationTiming(nowPerf) {
  if (!observationCurrentName || !observationCurrentMood || !observationCurrentAt) return;
  const delta = nowPerf - observationCurrentAt;
  if (delta > 0) {
    addDuration(observationCurrentName, observationCurrentMood, delta);
  }
  observationCurrentName = null;
  observationCurrentMood = null;
  observationCurrentAt = 0;
}

function topMoodFromMap(map) {
  let bestMood = null;
  let bestMs = null;
  for (const [mood, ms] of map.entries()) {
    if (bestMs === null || ms > bestMs) {
      bestMs = ms;
      bestMood = mood;
    }
  }
  return { bestMood, bestMs };
}

function buildOverallMoodTotals() {
  const totals = new Map();
  for (const moods of observationDurations.values()) {
    for (const [mood, ms] of moods.entries()) {
      totals.set(mood, (totals.get(mood) || 0) + ms);
    }
  }
  return totals;
}

function topMoodFromMap(map) {
  let bestMood = null;
  let bestCount = -1;
  for (const [mood, count] of map.entries()) {
    if (count > bestCount) {
      bestCount = count;
      bestMood = mood;
    }
  }
  return { bestMood, bestCount };
}

function renderObservationResults() {
  if (!observationResultsEl) return;
  observationResultsEl.classList.remove("hidden");
  const durationText = observationStart ? formatDurationLabel(Date.now() - observationStart) : "--";
  if (observationDurationEl) observationDurationEl.textContent = `Длительность: ${durationText}`;

  const overallTotals = buildOverallMoodTotals();
  const overallTop = topMoodFromMap(overallTotals);
  if (observationTopMoodEl) {
    if (overallTop.bestMood) {
      const label = formatDurationLabel(overallTop.bestMs);
      observationTopMoodEl.textContent = label
        ? `Самая частая эмоция за наблюдение: ${overallTop.bestMood} - ${label}`
        : `Самая частая эмоция за наблюдение: ${overallTop.bestMood}`;
    } else {
      observationTopMoodEl.textContent = "Самая частая эмоция за наблюдение: --";
    }
  }

  if (observationPeopleEl) {
    observationPeopleEl.innerHTML = "";
  }

  if (observationDurations.size === 0) {
    if (observationPeopleEl) observationPeopleEl.textContent = "Люди: --";
    if (observationPeopleSelectEl) observationPeopleSelectEl.classList.add("hidden");
    return;
  }

  if (observationPeopleSelectEl) {
    observationPeopleSelectEl.innerHTML = "";
    const allOpt = document.createElement("option");
    allOpt.value = "__all__";
    allOpt.textContent = "Все люди";
    observationPeopleSelectEl.appendChild(allOpt);
  }

  for (const [name, moods] of observationDurations.entries()) {
    const personTop = topMoodFromMap(moods);
    if (observationPeopleSelectEl) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      observationPeopleSelectEl.appendChild(opt);
    }

    const personBlock = document.createElement("div");
    personBlock.className = "person-card";
    personBlock.dataset.person = name;
    const title = document.createElement("div");
    title.className = "person-title";
    title.textContent = name;
    personBlock.appendChild(title);
    if (observationPeopleEl) observationPeopleEl.appendChild(personBlock);

    for (const [mood, ms] of moods.entries()) {
      const line = document.createElement("div");
      line.textContent = `${mood} - ${formatDurationLabel(ms)}`;
      personBlock.appendChild(line);
    }

    if (personTop.bestMood) {
      const topLine = document.createElement("div");
      const label = formatDurationLabel(personTop.bestMs);
      topLine.textContent = label
        ? `Самая частая эмоция у ${name}: ${personTop.bestMood} - ${label}`
        : `Самая частая эмоция у ${name}: ${personTop.bestMood}`;
      personBlock.appendChild(topLine);
    }
  }

  if (observationPeopleSelectEl) {
    observationPeopleSelectEl.classList.toggle("hidden", observationPeopleSelectEl.options.length <= 2);
    observationPeopleSelectEl.value = "__all__";
  }
  applyObservationFilter();
}

function startObservation() {
  observationActive = true;
  observationStart = Date.now();
  observationPeople = new Map();
  observationCurrentName = null;
  observationCurrentMood = null;
  observationCurrentAt = 0;
  observationDurations = new Map();
  lastLoggedName = null;
  lastLoggedMood = null;
  pendingName = null;
  pendingMood = null;
  pendingSince = 0;
  if (observationResultsEl) observationResultsEl.classList.add("hidden");
  if (observationToggleBtn) observationToggleBtn.textContent = "Остановить наблюдение";
}

function stopObservation() {
  observationActive = false;
  flushObservationTiming(performance.now());
  renderObservationResults();
  if (observationToggleBtn) observationToggleBtn.textContent = "Начать наблюдение";
}

function applyObservationFilter() {
  if (!observationPeopleEl) return;
  const selected = observationPeopleSelectEl?.value || "__all__";
  const blocks = observationPeopleEl.querySelectorAll("[data-person]");
  blocks.forEach((block) => {
    block.classList.toggle("hidden", selected !== "__all__" && block.dataset.person !== selected);
  });
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
    if (rtspInput) rtspInput.value = "";
    if (rtspNameInput) rtspNameInput.value = "";
    syncAddButtonState();
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

if (rtspInput) {
  rtspInput.addEventListener("input", syncAddButtonState);
}

syncAddButtonState();

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

if (observationToggleBtn) {
  observationToggleBtn.addEventListener("click", () => {
    if (observationActive) {
      stopObservation();
    } else {
      startObservation();
    }
  });
}

if (clearObservationBtn) {
  clearObservationBtn.addEventListener("click", () => {
    if (observationResultsEl) observationResultsEl.classList.add("hidden");
    if (observationDurationEl) observationDurationEl.textContent = "Длительность: --";
    if (observationTopMoodEl) observationTopMoodEl.textContent = "Самая частая эмоция за наблюдение: --";
    if (observationPeopleEl) observationPeopleEl.textContent = "Люди: --";
    if (observationPeopleSelectEl) observationPeopleSelectEl.classList.add("hidden");
    observationPeople = new Map();
    observationCurrentName = null;
    observationCurrentMood = null;
    observationCurrentAt = 0;
    observationDurations = new Map();
  });
}

if (observationPeopleSelectEl) {
  observationPeopleSelectEl.addEventListener("change", applyObservationFilter);
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
