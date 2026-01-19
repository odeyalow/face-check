const video = document.getElementById("video");
const streamCanvas = document.getElementById("stream");
const canvas = document.getElementById("overlay");
const moodEl = document.getElementById("mood");
const nameEl = document.getElementById("name");
const confEl = document.getElementById("conf");
const fpsEl = document.getElementById("fps");
const logsEl = document.getElementById("logs");
const rtspInput = document.getElementById("rtspUrl");
const connectRtspBtn = document.getElementById("connectRtsp");
const useWebcamBtn = document.getElementById("useWebcam");

const MODEL_URL = "/models";

let webcamStream = null;
let player = null;
let sourceEl = video;
let faceMatcher = null;
let recognitionReady = false;
let lastName = "--";
let lastMood = "--";

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

async function startWebcam() {
  if (player?.destroy) player.destroy();
  player = null;
  stopWebcam();

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
}

async function startRtsp(rtspUrl) {
  if (!rtspUrl || !window.loadPlayer) return;

  stopWebcam();
  if (player?.destroy) player.destroy();

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
    const res = await fetch("/api/known");
    if (!res.ok) return;
    const { files } = await res.json();
    if (!Array.isArray(files) || files.length === 0) return;

    const labelMap = new Map();
    files.forEach((file) => {
      const label = labelFromFilename(file);
      if (!label) return;
      if (!labelMap.has(label)) labelMap.set(label, []);
      labelMap.get(label).push(`/known/${file}`);
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
    // ignore loading errors; webcam/RTSP should still work
  }
}

let lastTs = performance.now();
let frames = 0;

function syncOverlaySize() {
  const width = sourceEl === video ? video.videoWidth : sourceEl.width;
  const height = sourceEl === video ? video.videoHeight : sourceEl.height;

  if (!width || !height) return false;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return true;
}

async function loop() {
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

  if (!syncOverlaySize()) {
    setTimeout(loop, 120);
    return;
  }

  let detection = faceapi.detectSingleFace(sourceEl, opts);
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
    moodEl.textContent = bestKey;
    confEl.textContent = (bestVal * 100).toFixed(1) + "%";
    lastMood = bestKey;
    if (recognitionReady && faceMatcher && nameEl) {
      const match = faceMatcher.findBestMatch(result.descriptor);
      nameEl.textContent = match.label;
      lastName = match.label;
    } else if (nameEl) {
      nameEl.textContent = "--";
      lastName = "--";
    }
  } else {
    moodEl.textContent = "--";
    confEl.textContent = "--";
    if (nameEl) nameEl.textContent = "--";
    lastMood = "--";
    lastName = "--";
  }

  frames++;
  const now = performance.now();
  if (now - lastTs >= 1000) {
    fpsEl.textContent = String(frames);
    frames = 0;
    lastTs = now;
  }

  setTimeout(loop, 120);
}

(async () => {
  await loadModels();
  await loadKnownFaces();
  await startWebcam();
  loop();
})();

function formatTime(date) {
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function addLogLine() {
  if (!logsEl) return;
  const time = formatTime(new Date());
  const line = `${lastName} ${lastMood} ${time}`;
  const div = document.createElement("div");
  div.textContent = line;
  logsEl.prepend(div);
}

setInterval(addLogLine, 5000);

const savedRtspUrl = localStorage.getItem("rtspUrl");
if (savedRtspUrl && rtspInput) rtspInput.value = savedRtspUrl;

if (connectRtspBtn) {
  connectRtspBtn.addEventListener("click", async () => {
    const rtspUrl = rtspInput.value.trim();
    if (!rtspUrl) return;
    localStorage.setItem("rtspUrl", rtspUrl);
    await startRtsp(rtspUrl);
  });
}

if (useWebcamBtn) {
  useWebcamBtn.addEventListener("click", async () => {
    await startWebcam();
  });
}
