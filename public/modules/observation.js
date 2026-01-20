export function createObservation({
  logsEl,
  observationResultsEl,
  observationDurationEl,
  observationTopMoodEl,
  observationPeopleEl,
  observationPeopleSelectEl,
  observationToggleBtn,
  clearObservationBtn,
  moodLabel = (value) => value
}) {
  let active = false;
  let startTs = 0;
  let currentName = null;
  let currentMood = null;
  let currentAt = 0;
  let durations = new Map();

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
    if (!durations.has(name)) durations.set(name, new Map());
    const moodMap = durations.get(name);
    moodMap.set(mood, (moodMap.get(mood) || 0) + deltaMs);
  }

  function updateTiming(name, mood, nowPerf) {
    if (!active) return;
    if (!currentName) {
      currentName = name;
      currentMood = mood;
      currentAt = nowPerf;
      return;
    }

    const delta = nowPerf - currentAt;
    if (delta > 0) addDuration(currentName, currentMood, delta);

    if (currentName !== name || currentMood !== mood) {
      currentName = name;
      currentMood = mood;
    }
    currentAt = nowPerf;
  }

  function flushTiming(nowPerf) {
    if (!active) return;
    if (!currentName || !currentMood || !currentAt) return;
    const delta = nowPerf - currentAt;
    if (delta > 0) addDuration(currentName, currentMood, delta);
    currentName = null;
    currentMood = null;
    currentAt = 0;
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
    for (const moods of durations.values()) {
      for (const [mood, ms] of moods.entries()) {
        totals.set(mood, (totals.get(mood) || 0) + ms);
      }
    }
    return totals;
  }

  function applyFilter() {
    if (!observationPeopleEl) return;
    const selected = observationPeopleSelectEl?.value || "__all__";
    const blocks = observationPeopleEl.querySelectorAll("[data-person]");
    blocks.forEach((block) => {
      block.classList.toggle("hidden", selected !== "__all__" && block.dataset.person !== selected);
    });
  }

  function renderResults() {
    if (!observationResultsEl) return;
    observationResultsEl.classList.remove("hidden");
    const durationText = startTs ? formatDurationLabel(Date.now() - startTs) : "--";
    if (observationDurationEl) observationDurationEl.textContent = `Длительность: ${durationText}`;

    const overallTotals = buildOverallMoodTotals();
    const overallTop = topMoodFromMap(overallTotals);
    if (observationTopMoodEl) {
      if (overallTop.bestMood) {
        const label = formatDurationLabel(overallTop.bestMs);
        const moodText = moodLabel(overallTop.bestMood);
        observationTopMoodEl.textContent = label
          ? `Самая частая эмоция за наблюдение: ${moodText} - ${label}`
          : `Самая частая эмоция за наблюдение: ${moodText}`;
      } else {
        observationTopMoodEl.textContent = "Самая частая эмоция за наблюдение: --";
      }
    }

    if (observationPeopleEl) observationPeopleEl.innerHTML = "";

    if (durations.size === 0) {
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

    for (const [name, moods] of durations.entries()) {
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

      for (const [mood, ms] of moods.entries()) {
        const line = document.createElement("div");
        line.textContent = `${moodLabel(mood)} - ${formatDurationLabel(ms)}`;
        personBlock.appendChild(line);
      }

      if (personTop.bestMood) {
        const topLine = document.createElement("div");
        const label = formatDurationLabel(personTop.bestMs);
        const moodText = moodLabel(personTop.bestMood);
        topLine.textContent = label
          ? `Самая частая эмоция у ${name}: ${moodText} - ${label}`
          : `Самая частая эмоция у ${name}: ${moodText}`;
        personBlock.appendChild(topLine);
      }

      if (observationPeopleEl) observationPeopleEl.appendChild(personBlock);
    }

    if (observationPeopleSelectEl) {
      observationPeopleSelectEl.classList.toggle("hidden", observationPeopleSelectEl.options.length <= 2);
      observationPeopleSelectEl.value = "__all__";
    }
    applyFilter();
  }

  function start() {
    active = true;
    startTs = Date.now();
    currentName = null;
    currentMood = null;
    currentAt = 0;
    durations = new Map();
    if (observationResultsEl) observationResultsEl.classList.add("hidden");
    if (observationToggleBtn) observationToggleBtn.textContent = "Остановить наблюдение";
  }

  function stop() {
    active = false;
    flushTiming(performance.now());
    renderResults();
    if (observationToggleBtn) observationToggleBtn.textContent = "Начать наблюдение";
  }

  function toggle() {
    if (active) stop();
    else start();
  }

  function clearResults() {
    if (observationResultsEl) observationResultsEl.classList.add("hidden");
    if (observationDurationEl) observationDurationEl.textContent = "Длительность: --";
    if (observationTopMoodEl) observationTopMoodEl.textContent = "Самая частая эмоция за наблюдение: --";
    if (observationPeopleEl) observationPeopleEl.textContent = "Люди: --";
    if (observationPeopleSelectEl) observationPeopleSelectEl.classList.add("hidden");
    durations = new Map();
    currentName = null;
    currentMood = null;
    currentAt = 0;
  }

  function addLogLine(name, mood) {
    if (!active || !logsEl) return;
    const time = new Date();
    const hh = String(time.getHours()).padStart(2, "0");
    const mm = String(time.getMinutes()).padStart(2, "0");
    const ss = String(time.getSeconds()).padStart(2, "0");
    const line = `${name} ${moodLabel(mood)} ${hh}:${mm}:${ss}`;
    const div = document.createElement("div");
    div.textContent = line;
    logsEl.prepend(div);
  }

  if (observationToggleBtn) observationToggleBtn.addEventListener("click", toggle);
  if (clearObservationBtn) clearObservationBtn.addEventListener("click", clearResults);
  if (observationPeopleSelectEl) observationPeopleSelectEl.addEventListener("change", applyFilter);

  return {
    isActive: () => active,
    start,
    stop,
    updateTiming,
    flushTiming,
    addLogLine
  };
}
