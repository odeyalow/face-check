export function createCameraController({
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
  onSourceChange
}) {
  let webcamStream = null;
  let player = null;
  let activeSource = null;
  let emptyStateActive = false;

  function setLoadingStatus(isLoading, message = "Загрузка камеры...", isError = false) {
    if (!loadingStatusEl) return;
    loadingStatusEl.classList.toggle("hidden", !isLoading);
    loadingStatusEl.textContent = message;
    loadingStatusEl.classList.toggle("error", isError);
    if (mediaWrapEl) mediaWrapEl.classList.toggle("hidden", isLoading);
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
    if (onSourceChange) onSourceChange(video);
    activeSource = video;
    setLoadingStatus(false);
  }

  async function startRtsp(rtspUrl) {
    if (!rtspUrl || !window.loadPlayer) {
      setLoadingStatus(true, "Не удалось подключиться к RTSP", true);
      return;
    }

    stopWebcam();
    if (player?.destroy) player.destroy();
    setLoadingStatus(true);

    const wsProto = location.protocol === "https:" ? "wss://" : "ws://";
    const wsUrl = `${wsProto}${location.host}/api/stream?url=${encodeURIComponent(rtspUrl)}`;

    video.classList.add("hidden");
    streamCanvas.classList.remove("hidden");
    if (onSourceChange) onSourceChange(streamCanvas);

    const playerTimeoutMs = 8000;
    const playerPromise = window.loadPlayer({
      url: wsUrl,
      canvas: streamCanvas,
      audio: false,
      disableGl: true
    });
    player = await Promise.race([
      playerPromise,
      new Promise((_, rej) => setTimeout(() => rej(new Error("player timeout")), playerTimeoutMs))
    ]);

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
    activeSource = streamCanvas;
    setLoadingStatus(false);
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
    if (rtspEmptyEl) {
      rtspEmptyEl.classList.toggle("hidden", list.length > 0);
    }
    if (list.length === 0 && !activeSource) {
      emptyStateActive = true;
      setLoadingStatus(true, "Нет добавленных камер");
    } else if (emptyStateActive && list.length > 0) {
      emptyStateActive = false;
      setLoadingStatus(false);
    }
  }

  function syncAddButtonState() {
    if (!addRtspBtn) return;
    const hasUrl = Boolean(rtspInput && rtspInput.value.trim());
    addRtspBtn.disabled = !hasUrl;
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
    });
  }

  if (rtspInput) rtspInput.addEventListener("input", syncAddButtonState);
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
    });
  }

  if (useWebcamBtn) {
    useWebcamBtn.addEventListener("click", async () => {
      await startWebcam();
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
      if (rtspSelect) rtspSelect.value = "";
      if (rtspInput) rtspInput.value = "";
      if (rtspNameInput) rtspNameInput.value = "";
      if (localStorage.getItem("rtspUrl") === rtspUrl) {
        localStorage.removeItem("rtspUrl");
      }
      if (nextList.length === 0) {
        activeSource = null;
        emptyStateActive = true;
        setLoadingStatus(true, "Нет добавленных камер");
      }
    });
  }

  return {
    startWebcam,
    startRtsp
  };
}
