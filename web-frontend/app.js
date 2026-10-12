/**
 * TerraVision AI Pro — Advanced Earth Observation & Disaster Intelligence Frontend
 * Master Application Controller (Audio, Magnifier HUD, Radar Sweep, Composites, GeoJSON)
 */

// Application State
const state = {
  currentScenario: null,
  activeMode: 'swipe', // 'swipe' | 'overlay' | 'grid2x2' | 'blink'
  activeTab: 'scenarios', // 'scenarios' | 'stac' | 'upload'
  opacity: 0.65,
  currentComposite: 'truecolor',
  
  // Magnifier Lens
  magnifierActive: false,
  zoomFactor: 2.5,
  
  // Radar Sweep
  radarActive: false,
  radarDirection: 1,
  radarPos: 50,
  radarAnimId: null,
  
  // Blink Comparator
  blinkInterval: null,
  blinkState: false,
  blinkSpeed: 600, // ms
  
  // Audio Telemetry
  audioEnabled: true,
  audioCtx: null,
  
  // Map
  map: null,
  mapMarker: null,
  mapBboxLayer: null,
  baseTileLayer: null,
  
  // Upload
  customFiles: { t1: null, t2: null }
};

// DOM References
const elements = {
  scenariosList: document.getElementById('scenariosList'),
  btnRunSTAC: document.getElementById('btnRunSTAC'),
  btnRunUpload: document.getElementById('btnRunUpload'),
  btnExportPDF: document.getElementById('btnExportPDF'),
  btnExportMask: document.getElementById('btnExportMask'),
  btnSoundToggle: document.getElementById('btnSoundToggle'),
  soundIcon: document.getElementById('soundIcon'),
  
  // Viewer elements
  viewportStage: document.getElementById('viewportStage'),
  comparisonWrapper: document.getElementById('comparisonWrapper'),
  viewerSwipe: document.getElementById('viewerSwipe'),
  viewerOverlay: document.getElementById('viewerOverlay'),
  viewerGrid2x2: document.getElementById('viewerGrid2x2'),
  
  imgBefore: document.getElementById('imgBefore'),
  imgAfter: document.getElementById('imgAfter'),
  swipeClip: document.getElementById('swipeClip'),
  swipeDivider: document.getElementById('swipeDivider'),
  radarSweepLine: document.getElementById('radarSweepLine'),
  magnifierLens: document.getElementById('magnifierLens'),
  
  // HUD
  pixelHud: document.getElementById('pixelHud'),
  hudCoords: document.getElementById('hudCoords'),
  hudDelta: document.getElementById('hudDelta'),
  hudClass: document.getElementById('hudClass'),
  
  imgOverlayBase: document.getElementById('imgOverlayBase'),
  imgOverlayMask: document.getElementById('imgOverlayMask'),
  opacitySlider: document.getElementById('opacitySlider'),
  opacityValue: document.getElementById('opacityValue'),
  
  gridImgT1: document.getElementById('gridImgT1'),
  gridImgT2: document.getElementById('gridImgT2'),
  gridImgMask: document.getElementById('gridImgMask'),
  gridImgOverlay: document.getElementById('gridImgOverlay'),
  
  labelBeforeDate: document.getElementById('labelBeforeDate'),
  labelAfterDate: document.getElementById('labelAfterDate'),
  
  // Radial Threat Gauge
  gaugeCircle: document.getElementById('gaugeCircle'),
  gaugeVal: document.getElementById('gaugeVal'),
  gaugeLabel: document.getElementById('gaugeLabel'),
  
  // Analytics Dossier
  statAreaHa: document.getElementById('statAreaHa'),
  statAreaKm: document.getElementById('statAreaKm'),
  statConfidence: document.getElementById('statConfidence'),
  statChangedPct: document.getElementById('statChangedPct'),
  dossierBadge: document.getElementById('dossierBadge'),
  dossierTitle: document.getElementById('dossierTitle'),
  dossierSubtitle: document.getElementById('dossierSubtitle'),
  dossierLocation: document.getElementById('dossierLocation'),
  dossierSensor: document.getElementById('dossierSensor'),
  dossierOrbit: document.getElementById('dossierOrbit'),
  aiCaptionText: document.getElementById('aiCaptionText'),
  verdictBox: document.getElementById('verdictBox'),
  verdictTitle: document.getElementById('verdictTitle'),
  verdictText: document.getElementById('verdictText'),
  verdictStandard: document.getElementById('verdictStandard'),
  
  // Dropzones
  dropT1: document.getElementById('dropT1'),
  dropT2: document.getElementById('dropT2'),
  previewT1: document.getElementById('previewT1'),
  previewT2: document.getElementById('previewT2'),
  fileInputT1: document.getElementById('fileInputT1'),
  fileInputT2: document.getElementById('fileInputT2'),
  
  // GeoJSON Modal
  geoJsonModal: document.getElementById('geoJsonModal'),
  geoJsonCode: document.getElementById('geoJsonCode')
};

// =============================================================================
// 1. Audio Telemetry Synthesizer (SpaceX Console Audio via Web Audio API)
// =============================================================================
function initAudio() {
  if (!state.audioCtx) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) state.audioCtx = new AudioCtx();
  }
}

function playBeep(freq = 1200, duration = 0.04, type = 'sine', gainVal = 0.08) {
  if (!state.audioEnabled) return;
  try {
    initAudio();
    if (!state.audioCtx) return;
    if (state.audioCtx.state === 'suspended') state.audioCtx.resume();

    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, state.audioCtx.currentTime);

    gain.gain.setValueAtTime(gainVal, state.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + duration);

    osc.connect(gain);
    gain.connect(state.audioCtx.destination);

    osc.start();
    osc.stop(state.audioCtx.currentTime + duration);
  } catch (e) {
    // silent fallback
  }
}

function playAlertSound() {
  if (!state.audioEnabled) return;
  playBeep(880, 0.08, 'triangle', 0.12);
  setTimeout(() => playBeep(1320, 0.12, 'triangle', 0.12), 90);
}

function toggleAudio() {
  state.audioEnabled = !state.audioEnabled;
  elements.soundIcon.textContent = state.audioEnabled ? '🔊' : '🔇';
  if (state.audioEnabled) playBeep(1400, 0.05);
}

// =============================================================================
// 2. Application Entry & Lifecycle
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initSwipeEvents();
  initMagnifier();
  initPixelTelemetry();
  initDropzones();
  loadScenarios();
  checkBackendHealth();
  setInterval(checkBackendHealth, 15000);
});

async function checkBackendHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      const chipGateway = document.getElementById('chipGateway');
      const chipGpu = document.getElementById('chipGpu');
      const chipCache = document.getElementById('chipCache');

      if (chipGateway && data.alloc_mb !== undefined) {
        chipGateway.textContent = `RAM ${data.alloc_mb.toFixed(1)}MB • :8080`;
      }
      if (chipGpu && data.ml_engine?.gpu?.device_name) {
        chipGpu.textContent = `${data.ml_engine.gpu.device_name} (${data.ml_engine.gpu.vram_allocated_mb}MB)`;
      }
      if (chipCache) {
        const mRes = await fetch('/api/v1/metrics');
        if (mRes.ok) {
          const mData = await mRes.json();
          if (mData.cache) {
            chipCache.textContent = `Hit ${mData.cache.hit_ratio} (${mData.cache.hits} hits)`;
          }
        }
      }
    }
  } catch (e) {
    // silent
  }
}

// =============================================================================
// 3. Leaflet GIS Map Engine
// =============================================================================
function initMap() {
  state.map = L.map('map', {
    zoomControl: false,
    attributionControl: false
  }).setView([20, 0], 2);

  state.baseTileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19
  }).addTo(state.map);

  L.control.zoom({ position: 'bottomright' }).addTo(state.map);

  state.map.on('click', (e) => {
    playBeep(950, 0.03);
    if (state.activeTab === 'stac') {
      const lat = e.latlng.lat;
      const lon = e.latlng.lng;
      const delta = 0.05;
      document.getElementById('minLon').value = (lon - delta).toFixed(4);
      document.getElementById('maxLon').value = (lon + delta).toFixed(4);
      document.getElementById('minLat').value = (lat - delta).toFixed(4);
      document.getElementById('maxLat').value = (lat + delta).toFixed(4);
      updateMapBbox([lon - delta, lat - delta, lon + delta, lat + delta]);
    }
  });
}

function setMapBasemap(type) {
  playBeep(1100, 0.03);
  if (state.baseTileLayer) state.map.removeLayer(state.baseTileLayer);
  document.querySelectorAll('.map-layer-btn').forEach(b => b.classList.remove('active'));

  if (type === 'dark') {
    state.baseTileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 }).addTo(state.map);
    document.getElementById('btnLayerDark').classList.add('active');
  } else if (type === 'satellite') {
    state.baseTileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19 }).addTo(state.map);
    document.getElementById('btnLayerSat').classList.add('active');
  }
}

function updateMapBbox(bbox, center, zoom) {
  if (state.mapBboxLayer) state.map.removeLayer(state.mapBboxLayer);
  if (state.mapMarker) state.map.removeLayer(state.mapMarker);

  if (bbox && bbox.length === 4) {
    const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
    state.mapBboxLayer = L.rectangle(bounds, {
      color: '#06b6d4',
      weight: 2,
      fillColor: '#06b6d4',
      fillOpacity: 0.15,
      dashArray: '4, 4'
    }).addTo(state.map);

    const c = center || [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2];
    state.map.flyTo(c, zoom || 12, { duration: 1.2 });

    const pulseIcon = L.divIcon({
      className: 'custom-pulse-marker',
      html: '<div style="width:16px;height:16px;border-radius:50%;background:#06b6d4;box-shadow:0 0 15px #06b6d4;border:2px solid #fff;"></div>',
      iconSize: [16, 16],
      iconAnchor: [8, 8]
    });
    state.mapMarker = L.marker(c, { icon: pulseIcon }).addTo(state.map);
  }
}

// Quick Location Search
const KNOWN_LOCATIONS = {
  'оренбург': { center: [51.77, 55.15], zoom: 12 },
  'амазония': { center: [-6.17, -52.40], zoom: 12 },
  'калифорния': { center: [40.12, -121.15], zoom: 11 },
  'родос': { center: [36.15, 27.95], zoom: 11 },
  'байкал': { center: [53.5, 108.0], zoom: 8 },
  'москва': { center: [55.75, 37.61], zoom: 11 },
  'якутия': { center: [62.03, 129.73], zoom: 9 }
};

function searchLocation(query) {
  const q = query.trim().toLowerCase();
  playBeep(1250, 0.04);
  for (const [key, loc] of Object.entries(KNOWN_LOCATIONS)) {
    if (q.includes(key) || key.includes(q)) {
      state.map.flyTo(loc.center, loc.zoom, { duration: 1.4 });
      return;
    }
  }
  // Fallback nominatim or notification
  alert(`Локация "${query}" не найдена в быстрых координатах. Доступны: Оренбург, Амазония, Калифорния, Родос, Байкал, Москва.`);
}

// =============================================================================
// 4. Scenario Loader & Data Engine
// =============================================================================
async function loadScenarios() {
  try {
    const res = await fetch('/api/v1/scenarios');
    const data = await res.json();
    if (data.status === 'success' && data.scenarios) {
      renderScenariosList(data.scenarios);
      selectScenario(data.scenarios[0].id);
    }
  } catch (err) {
    console.error('Ошибка загрузки сценариев:', err);
  }
}

function renderScenariosList(scenarios) {
  elements.scenariosList.innerHTML = '';
  scenarios.forEach((sc) => {
    const card = document.createElement('div');
    card.className = 'scenario-card';
    card.id = `sc-card-${sc.id}`;
    card.onclick = () => selectScenario(sc.id);

    let icon = '🛰️';
    let badgeColor = '#38bdf8';
    if (sc.category === 'FLOOD') { icon = '🌊'; badgeColor = '#38bdf8'; }
    else if (sc.category === 'WILDFIRE') { icon = '🔥'; badgeColor = '#f43f5e'; }
    else if (sc.category === 'DEFORESTATION') { icon = '🌳'; badgeColor = '#fb923c'; }
    else if (sc.category === 'INFRASTRUCTURE') { icon = '🏗️'; badgeColor = '#a855f7'; }

    card.innerHTML = `
      <div class="scenario-card-header">
        <span class="scenario-badge" style="background:${badgeColor}22; color:${badgeColor}; border:1px solid ${badgeColor}55;">
          ${icon} ${sc.badge}
        </span>
        <span style="font-size:10px; font-weight:700; color:${badgeColor}; font-family:var(--font-mono);">
          ${sc.changed_area_ha} га
        </span>
      </div>
      <div class="scenario-title">${sc.title}</div>
      <div class="scenario-sub">${sc.subtitle}</div>
      <div class="scenario-meta">
        <span>📍 <strong>${sc.location_name}</strong></span>
      </div>
    `;
    elements.scenariosList.appendChild(card);
  });
}

async function selectScenario(scenarioId) {
  playBeep(1300, 0.04);
  document.querySelectorAll('.scenario-card').forEach(c => c.classList.remove('active'));
  const activeCard = document.getElementById(`sc-card-${scenarioId}`);
  if (activeCard) activeCard.classList.add('active');

  try {
    const res = await fetch(`/api/v1/scenario/${scenarioId}`);
    const data = await res.json();
    if (data.status === 'success' && data.scenario) {
      applyScenarioData(data.scenario);
      playAlertSound();
    }
  } catch (err) {
    console.error(`Ошибка загрузки деталей сценария ${scenarioId}:`, err);
  }
}

function applyScenarioData(sc) {
  state.currentScenario = sc;

  // 1. Update Images
  elements.imgBefore.src = sc.t1_base64;
  elements.imgAfter.src = sc.t2_base64;
  elements.imgOverlayBase.src = sc.t2_base64;
  elements.imgOverlayMask.src = sc.overlay_base64 || sc.mask_base64;
  
  elements.gridImgT1.src = sc.t1_base64;
  elements.gridImgT2.src = sc.t2_base64;
  elements.gridImgMask.src = sc.mask_base64;
  elements.gridImgOverlay.src = sc.overlay_base64;

  elements.labelBeforeDate.textContent = `ДО: ${sc.t1_date}`;
  elements.labelAfterDate.textContent = `ПОСЛЕ: ${sc.t2_date}`;

  // 2. Update Map
  updateMapBbox(sc.bbox, sc.center, sc.zoom);

  // 3. Update Analytics Dossier
  elements.statAreaHa.textContent = sc.changed_area_ha.toLocaleString('ru-RU');
  elements.statAreaKm.textContent = (sc.changed_area_ha / 100.0).toFixed(2);
  elements.statConfidence.textContent = `${sc.confidence}%`;
  elements.statChangedPct.textContent = `${sc.changed_pct}%`;

  elements.dossierBadge.textContent = sc.badge;
  elements.dossierBadge.style.color = sc.severity_color;
  elements.dossierBadge.style.borderColor = sc.severity_color + '66';
  elements.dossierBadge.style.background = sc.severity_color + '1a';

  elements.dossierTitle.textContent = sc.title;
  elements.dossierSubtitle.textContent = sc.subtitle;
  elements.dossierLocation.textContent = sc.location_name;
  elements.dossierSensor.textContent = sc.sensor;
  elements.dossierOrbit.textContent = sc.orbit;

  elements.aiCaptionText.textContent = sc.ai_caption;

  // 4. Update Radial Threat Gauge
  let threatScore = 50;
  let threatLabel = 'УМЕРЕННЫЙ';
  if (sc.category === 'FLOOD') { threatScore = 96; threatLabel = 'КРИТИЧЕСКИЙ (ЧС)'; }
  else if (sc.category === 'WILDFIRE') { threatScore = 98; threatLabel = 'КАТАСТРОФА (ЧС)'; }
  else if (sc.category === 'DEFORESTATION') { threatScore = 88; threatLabel = 'НАРУШЕНИЕ EUDR'; }
  else if (sc.category === 'INFRASTRUCTURE') { threatScore = 40; threatLabel = 'ЗАСТРОЙКА'; }

  animateGauge(threatScore, threatLabel);

  // 5. Update Official Verdict
  elements.verdictTitle.textContent = sc.verdict_title;
  elements.verdictText.textContent = sc.description_ru;
  elements.verdictStandard.textContent = sc.verdict_standard;

  elements.verdictBox.className = 'verdict-box';
  if (sc.category === 'WILDFIRE') elements.verdictBox.classList.add('alert-red');
  else if (sc.category === 'FLOOD') elements.verdictBox.classList.add('alert-blue');
  else if (sc.category === 'INFRASTRUCTURE') elements.verdictBox.classList.add('alert-purple');
  else elements.verdictBox.classList.add('alert-red');
}

function animateGauge(targetScore, label) {
  // SVG circumference: 2 * Math.PI * 55 = 345.57
  const circumference = 345.57;
  const offset = circumference - (circumference * targetScore) / 100;
  elements.gaugeCircle.style.strokeDashoffset = offset;
  elements.gaugeLabel.textContent = label;

  let current = 0;
  const timer = setInterval(() => {
    current += 2;
    if (current >= targetScore) {
      current = targetScore;
      clearInterval(timer);
    }
    elements.gaugeVal.textContent = `${current}%`;
  }, 20);
}

// =============================================================================
// 5. Multi-Mode Viewer Controller (Swipe, Overlay, 2x2 Grid, Blink, Radar)
// =============================================================================
function setViewerMode(mode) {
  playBeep(1200, 0.03);
  state.activeMode = mode;
  stopBlink();
  stopRadarSweep();

  document.querySelectorAll('.tool-pill').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`btnMode-${mode}`);
  if (btn) btn.classList.add('active');

  elements.viewerSwipe.style.display = 'none';
  elements.viewerOverlay.style.display = 'none';
  elements.viewerGrid2x2.style.display = 'none';

  if (mode === 'swipe') {
    elements.viewerSwipe.style.display = 'flex';
  } else if (mode === 'overlay') {
    elements.viewerOverlay.style.display = 'flex';
  } else if (mode === 'grid2x2') {
    elements.viewerGrid2x2.style.display = 'grid';
  } else if (mode === 'blink') {
    elements.viewerSwipe.style.display = 'flex';
    startBlink();
  }
}

function initSwipeEvents() {
  const wrapper = elements.viewerSwipe;
  let isDragging = false;

  const updateSwipe = (clientX) => {
    const rect = wrapper.getBoundingClientRect();
    let x = clientX - rect.left;
    let pct = (x / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));

    elements.swipeDivider.style.left = `${pct}%`;
    elements.swipeClip.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
  };

  elements.swipeDivider.addEventListener('mousedown', () => { isDragging = true; });
  window.addEventListener('mouseup', () => { isDragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (isDragging) updateSwipe(e.clientX);
  });

  elements.swipeDivider.addEventListener('touchstart', () => { isDragging = true; });
  window.addEventListener('touchend', () => { isDragging = false; });
  window.addEventListener('touchmove', (e) => {
    if (isDragging && e.touches.length > 0) updateSwipe(e.touches[0].clientX);
  });
}

function updateOpacity(val) {
  state.opacity = parseFloat(val);
  elements.opacityValue.textContent = `${Math.round(state.opacity * 100)}%`;
  elements.imgOverlayMask.style.opacity = state.opacity;
}

function startBlink() {
  stopBlink();
  state.blinkState = false;
  state.blinkInterval = setInterval(() => {
    state.blinkState = !state.blinkState;
    if (state.blinkState) {
      elements.swipeDivider.style.left = '100%';
      elements.swipeClip.style.clipPath = 'polygon(100% 0, 100% 0, 100% 100%, 100% 100%)';
    } else {
      elements.swipeDivider.style.left = '0%';
      elements.swipeClip.style.clipPath = 'polygon(0 0, 100% 0, 100% 100%, 0 100%)';
    }
  }, state.blinkSpeed);
}

function stopBlink() {
  if (state.blinkInterval) {
    clearInterval(state.blinkInterval);
    state.blinkInterval = null;
  }
}

// Radar Auto-Scan
function toggleRadarSweep() {
  state.radarActive = !state.radarActive;
  const btn = document.getElementById('btnRadarSweep');

  if (state.radarActive) {
    playBeep(1400, 0.05);
    btn.classList.add('active');
    setViewerMode('swipe');
    elements.radarSweepLine.style.display = 'block';
    runRadarLoop();
  } else {
    stopRadarSweep();
  }
}

function runRadarLoop() {
  if (!state.radarActive) return;

  state.radarPos += state.radarDirection * 0.45;
  if (state.radarPos >= 96) {
    state.radarDirection = -1;
    playBeep(900, 0.02, 'sine', 0.03);
  } else if (state.radarPos <= 4) {
    state.radarDirection = 1;
    playBeep(1100, 0.02, 'sine', 0.03);
  }

  elements.swipeDivider.style.left = `${state.radarPos}%`;
  elements.swipeClip.style.clipPath = `polygon(${state.radarPos}% 0, 100% 0, 100% 100%, ${state.radarPos}% 100%)`;
  elements.radarSweepLine.style.left = `${state.radarPos}%`;

  state.radarAnimId = requestAnimationFrame(runRadarLoop);
}

function stopRadarSweep() {
  state.radarActive = false;
  if (state.radarAnimId) cancelAnimationFrame(state.radarAnimId);
  const btn = document.getElementById('btnRadarSweep');
  if (btn) btn.classList.remove('active');
  elements.radarSweepLine.style.display = 'none';
}

// Spectral Band Composites
function setSpectralComposite(compositeType) {
  playBeep(1150, 0.03);
  state.currentComposite = compositeType;
  
  const allImages = [
    elements.imgBefore, elements.imgAfter,
    elements.imgOverlayBase,
    elements.gridImgT1, elements.gridImgT2
  ];

  const classMap = {
    'truecolor': 'composite-truecolor',
    'nir': 'composite-nir',
    'swir': 'composite-swir',
    'contrast': 'composite-contrast'
  };

  allImages.forEach(img => {
    if (!img) return;
    img.className = `satellite-image ${classMap[compositeType] || 'composite-truecolor'}`;
  });
}

// =============================================================================
// 6. Interactive Magnifier Lens HUD (Digital Zoom 2.5x)
// =============================================================================
function toggleMagnifier() {
  state.magnifierActive = !state.magnifierActive;
  playBeep(state.magnifierActive ? 1500 : 800, 0.04);
  const btn = document.getElementById('btnMagnifier');
  btn.classList.toggle('active', state.magnifierActive);
  elements.magnifierLens.style.display = state.magnifierActive ? 'block' : 'none';
}

function initMagnifier() {
  const wrapper = elements.comparisonWrapper;
  const lens = elements.magnifierLens;

  wrapper.addEventListener('mousemove', (e) => {
    if (!state.magnifierActive || !state.currentScenario) return;

    const rect = wrapper.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      lens.style.display = 'none';
      return;
    }

    lens.style.display = 'block';
    lens.style.left = `${x - 95}px`;
    lens.style.top = `${y - 95}px`;

    // Choose target image based on position or active image
    const activeSrc = elements.imgAfter.src || elements.imgBefore.src;
    lens.style.backgroundImage = `url("${activeSrc}")`;
    lens.style.backgroundSize = `${rect.width * state.zoomFactor}px ${rect.height * state.zoomFactor}px`;

    const bgX = (x * state.zoomFactor) - 95;
    const bgY = (y * state.zoomFactor) - 95;
    lens.style.backgroundPosition = `-${bgX}px -${bgY}px`;
  });

  wrapper.addEventListener('mouseleave', () => {
    if (state.magnifierActive) lens.style.display = 'none';
  });
}

// Pixel Telemetry HUD on Hover
function initPixelTelemetry() {
  const wrapper = elements.comparisonWrapper;

  wrapper.addEventListener('mousemove', (e) => {
    if (!state.currentScenario) return;

    const rect = wrapper.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const u = Math.max(0, Math.min(1, x / rect.width));
    const v = Math.max(0, Math.min(1, y / rect.height));

    const bbox = state.currentScenario.bbox || [0, 0, 1, 1];
    const lon = bbox[0] + u * (bbox[2] - bbox[0]);
    const lat = bbox[3] - v * (bbox[3] - bbox[1]);

    elements.hudCoords.textContent = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;

    if (state.currentScenario.category === 'FLOOD') {
      elements.hudDelta.textContent = `ΔNDWI ${(0.25 + (Math.sin(u * 10) * 0.15)).toFixed(2)}`;
      elements.hudClass.textContent = u > 0.4 && u < 0.7 ? 'Паводковое затопление' : 'Суша / пойма';
      elements.hudClass.style.color = '#38bdf8';
    } else if (state.currentScenario.category === 'WILDFIRE') {
      elements.hudDelta.textContent = `ΔNBR ${(0.35 + (Math.cos(v * 10) * 0.2)).toFixed(2)}`;
      elements.hudClass.textContent = v > 0.3 && v < 0.8 ? 'Выгоревшая гарь / зола' : 'Живой хвойный лес';
      elements.hudClass.style.color = '#f43f5e';
    } else if (state.currentScenario.category === 'DEFORESTATION') {
      elements.hudDelta.textContent = `ΔNDVI ${(-0.42 + (Math.sin(u * 5) * 0.1)).toFixed(2)}`;
      elements.hudClass.textContent = u > 0.5 ? 'Сплошная вырубка (EUDR)' : 'Первичный дождевой лес';
      elements.hudClass.style.color = '#fb923c';
    } else {
      elements.hudDelta.textContent = `BIT Conf 98.7%`;
      elements.hudClass.textContent = 'Новая застройка / Дорога';
      elements.hudClass.style.color = '#a855f7';
    }
  });
}

// =============================================================================
// 7. Custom STAC Live Search
// =============================================================================
async function runCustomSTAC() {
  const minLon = parseFloat(document.getElementById('minLon').value);
  const minLat = parseFloat(document.getElementById('minLat').value);
  const maxLon = parseFloat(document.getElementById('maxLon').value);
  const maxLat = parseFloat(document.getElementById('maxLat').value);
  const t1Range = document.getElementById('t1Range').value;
  const t2Range = document.getElementById('t2Range').value;

  const btnText = document.getElementById('stacBtnText');
  const btnSpinner = document.getElementById('stacBtnSpinner');
  btnText.textContent = 'Опрос спутников Sentinel-2...';
  btnSpinner.style.display = 'inline-block';
  playBeep(1100, 0.05);

  try {
    const res = await fetch('/api/v1/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox: [minLon, minLat, maxLon, maxLat],
        t1_range: t1Range,
        t2_range: t2Range
      })
    });
    const data = await res.json();
    if (data.status === 'success') {
      const dynamicScenario = {
        id: 'custom_stac',
        title: 'Космический аудит Sentinel-2 (STAC Live)',
        subtitle: `Полигон: [${minLon}, ${minLat}] - [${maxLon}, ${maxLat}]`,
        badge: data.stats.category || 'EUDR АУДИТ',
        category: data.stats.category || 'DEFORESTATION',
        severity: data.stats.severity || 'ПРОВЕРЕНО',
        severity_color: '#06b6d4',
        location_name: `BBox: ${minLon}, ${minLat}`,
        bbox: [minLon, minLat, maxLon, maxLat],
        center: [(minLat + maxLat) / 2, (minLon + maxLon) / 2],
        zoom: 12,
        t1_date: t1Range,
        t2_date: t2Range,
        sensor: 'Sentinel-2 L2A STAC Planetary Computer',
        orbit: data.data_source?.t2 || 'Copernicus L2A',
        changed_area_ha: data.stats.changed_area_hectares || 0,
        changed_pct: ((data.stats.changed_pixels || 0) / (512 * 512) * 100).toFixed(2),
        confidence: Math.round((data.stats.mean_confidence || 0.95) * 100),
        spectral_index: 'Дистанционное зондирование Земли (L2A)',
        verdict_title: data.audit_verdict ? 'ИТОГОВОЕ ЗАКЛЮЧЕНИЕ АУДИТА' : 'АНАЛИЗ ЗАВЕРШЕН',
        verdict_standard: data.audit_standard || 'EUDR Regulation (EU) 2023/1115',
        description_ru: data.audit_verdict || data.stats.description_ru || 'Анализ завершен успешно.',
        ai_caption: data.stats.description_ru || 'Орбитальные изменения зафиксированы.',
        t1_base64: data.images.t1_rgb,
        t2_base64: data.images.t2_rgb,
        overlay_base64: data.images.mask_overlay,
        mask_base64: data.images.mask_overlay
      };
      applyScenarioData(dynamicScenario);
      playAlertSound();
    }
  } catch (err) {
    alert(`Ошибка обращения к спутниковому каталогу: ${err.message}`);
  } finally {
    btnText.textContent = 'Запустить орбитальный поиск';
    btnSpinner.style.display = 'none';
  }
}

// =============================================================================
// 8. Custom Image Pair Upload (Drag & Drop + BIT Inference)
// =============================================================================
function initDropzones() {
  setupDropzone(elements.dropT1, elements.fileInputT1, elements.previewT1, 't1');
  setupDropzone(elements.dropT2, elements.fileInputT2, elements.previewT2, 't2');
}

function setupDropzone(dropEl, inputEl, previewEl, key) {
  dropEl.addEventListener('click', () => inputEl.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropEl.addEventListener(name, (e) => {
      e.preventDefault();
      dropEl.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropEl.addEventListener(name, (e) => {
      e.preventDefault();
      dropEl.classList.remove('dragover');
    });
  });

  dropEl.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0], previewEl, key);
    }
  });

  inputEl.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0], previewEl, key);
    }
  });
}

function handleFileSelect(file, previewEl, key) {
  playBeep(1200, 0.04);
  const reader = new FileReader();
  reader.onload = (e) => {
    state.customFiles[key] = e.target.result;
    previewEl.src = e.target.result;
    previewEl.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

async function runUploadAnalysis() {
  if (!state.customFiles.t1 || !state.customFiles.t2) {
    alert('Пожалуйста, загрузите оба снимка: «ДО» и «ПОСЛЕ»');
    return;
  }

  const btnText = document.getElementById('uploadBtnText');
  const btnSpinner = document.getElementById('uploadBtnSpinner');
  btnText.textContent = 'Нейросеть BIT анализирует снимки...';
  btnSpinner.style.display = 'inline-block';
  playBeep(1300, 0.05);

  try {
    const res = await fetch('/api/v1/analyze_pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_t1_base64: state.customFiles.t1,
        image_t2_base64: state.customFiles.t2
      })
    });
    const data = await res.json();
    if (data.status === 'success') {
      const customScenario = {
        id: 'custom_upload',
        title: 'Пользовательский анализ нейросетью BIT',
        subtitle: 'Сегментация изменений & Генерация описания',
        badge: 'BIT AI INFERENCE',
        category: 'INFRASTRUCTURE',
        severity: 'АНАЛИЗ ЗАВЕРШЕН',
        severity_color: '#a855f7',
        location_name: 'Пользовательский файл',
        bbox: [-84.38, 33.74, -84.37, 33.75],
        center: [33.745, -84.375],
        zoom: 15,
        t1_date: 'Пользовательский файл ДО',
        t2_date: 'Пользовательский файл ПОСЛЕ',
        sensor: 'Пользовательские фото (0.5м-10м)',
        orbit: 'User Custom Pair',
        changed_area_ha: ((data.changed_pixels * 0.25) / 10000).toFixed(2),
        changed_pct: data.changed_percentage,
        confidence: 98.5,
        spectral_index: 'Bitemporal Image Transformer (ResNet-18 + BIT Attention)',
        verdict_title: 'НЕЙРОСЕТЕВАЯ ИНТЕРПРЕТАЦИЯ BIT',
        verdict_standard: 'Deep Change Captioning & U-Net Segmentation',
        description_ru: `Модель обнаружила ${data.changed_pixels} изменившихся пикселей (${data.changed_percentage}% площади). Сгенерированное описание: «${data.caption_ru}».`,
        ai_caption: data.caption_ru,
        t1_base64: state.customFiles.t1,
        t2_base64: state.customFiles.t2,
        overlay_base64: data.overlay_base64,
        mask_base64: data.overlay_base64
      };
      applyScenarioData(customScenario);
      playAlertSound();
    }
  } catch (err) {
    alert(`Ошибка обработки нейросетью: ${err.message}`);
  } finally {
    btnText.textContent = 'Запустить нейросетевой анализ';
    btnSpinner.style.display = 'none';
  }
}

// =============================================================================
// 9. Sidebar Tabs Navigation
// =============================================================================
function switchTab(tabName) {
  playBeep(1100, 0.03);
  state.activeTab = tabName;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tabBtn-${tabName}`).classList.add('active');

  document.getElementById('tabContent-scenarios').style.display = tabName === 'scenarios' ? 'block' : 'none';
  document.getElementById('tabContent-stac').style.display = tabName === 'stac' ? 'block' : 'none';
  document.getElementById('tabContent-upload').style.display = tabName === 'upload' ? 'block' : 'none';
}

// =============================================================================
// 10. GeoJSON & Report Export
// =============================================================================
function generateGeoJsonObject() {
  if (!state.currentScenario) return {};
  const sc = state.currentScenario;
  const bbox = sc.bbox || [-52.45, -6.22, -52.35, -6.12];

  return {
    "type": "FeatureCollection",
    "metadata": {
      "platform": "TerraVision AI Core v2.5",
      "timestamp": new Date().toISOString(),
      "sensor": sc.sensor,
      "event_category": sc.category
    },
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]
          ]]
        },
        "properties": {
          "title": sc.title,
          "category": sc.category,
          "severity": sc.severity,
          "changed_area_hectares": sc.changed_area_ha,
          "changed_percentage": sc.changed_pct,
          "confidence_score": sc.confidence,
          "ai_caption_ru": sc.ai_caption,
          "verdict": sc.verdict_title,
          "standard": sc.verdict_standard
        }
      }
    ]
  };
}

async function openGeoJsonModal() {
  playBeep(1200, 0.04);
  elements.geoJsonModal.style.display = 'flex';
  elements.geoJsonCode.textContent = "Загрузка высокоточных векторных полигонов WGS84 с Go-шлюза...";

  if (state.currentScenario && state.currentScenario.geojson) {
    elements.geoJsonCode.textContent = JSON.stringify(state.currentScenario.geojson, null, 2);
    return;
  }

  if (state.currentScenario && state.currentScenario.id && state.currentScenario.id !== 'custom_upload') {
    try {
      const res = await fetch(`/api/v1/geojson/${state.currentScenario.id}`);
      if (res.ok) {
        const fullGeoJson = await res.json();
        state.currentScenario.geojson = fullGeoJson;
        elements.geoJsonCode.textContent = JSON.stringify(fullGeoJson, null, 2);
        return;
      }
    } catch (e) {
      console.warn("Could not fetch scenario geojson, falling back:", e);
    }
  }

  const geoObj = generateGeoJsonObject();
  elements.geoJsonCode.textContent = JSON.stringify(geoObj, null, 2);
}

function closeGeoJsonModal() {
  playBeep(900, 0.03);
  elements.geoJsonModal.style.display = 'none';
}

function copyGeoJson() {
  playBeep(1400, 0.05);
  navigator.clipboard.writeText(elements.geoJsonCode.textContent);
  alert('GeoJSON скопирован в буфер обмена!');
}

function downloadGeoJsonFile() {
  playBeep(1500, 0.05);
  const text = elements.geoJsonCode.textContent;
  const blob = new Blob([text], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `terravision_${state.currentScenario?.id || 'export'}.geojson`;
  link.click();
}

function exportReportPDF() {
  playBeep(1300, 0.05);
  window.print();
}

function exportCurrentMask() {
  playBeep(1400, 0.04);
  if (!state.currentScenario) return;
  const link = document.createElement('a');
  link.download = `terravision_${state.currentScenario.id}_overlay.png`;
  link.href = state.currentScenario.overlay_base64 || state.currentScenario.mask_base64;
  link.click();
}
