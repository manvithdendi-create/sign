/**
 * VisionSign AI - Frontend Controller
 * Interacts with MediaPipe Hands & Python FastAPI Backend
 */

const API_BASE = "/api";

// State variables
let currentTab = "tab-translator";
let isCameraActive = true;
let handsDetector = null;
let cameraInstance = null;
let handWasDetected = false;
let lastFrameTime = performance.now();
let frameCount = 0;

// Quiz state
let quizActive = false;
let currentQuizTarget = null;
let quizScore = 0;
let quizStreak = 0;

// Studio state
let studioRecording = false;
let studioSamples = [];
const TARGET_STUDIO_SAMPLES = 30;

// DOM Elements
const elements = {
  webcam: document.getElementById('webcam'),
  skeletonCanvas: document.getElementById('skeletonCanvas'),
  fpsCounter: document.getElementById('fpsCounter'),
  handDetectPill: document.getElementById('handDetectPill'),
  detectedSign: document.getElementById('detectedSign'),
  confidenceValue: document.getElementById('confidenceValue'),
  confidenceBar: document.getElementById('confidenceBar'),
  sentenceBuffer: document.getElementById('sentenceBuffer'),
  candidateList: document.getElementById('candidateList'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),
  toggleCameraBtn: document.getElementById('toggleCameraBtn'),
  
  // Translator controls
  speakBtn: document.getElementById('speakBtn'),
  spaceBtn: document.getElementById('spaceBtn'),
  backspaceBtn: document.getElementById('backspaceBtn'),
  clearBtn: document.getElementById('clearBtn'),

  // Dictionary
  dictSearch: document.getElementById('dictSearch'),
  dictGrid: document.getElementById('dictGrid'),

  // Quiz
  startQuizBtn: document.getElementById('startQuizBtn'),
  skipQuizBtn: document.getElementById('skipQuizBtn'),
  quizTargetSign: document.getElementById('quizTargetSign'),
  quizSignTips: document.getElementById('quizSignTips'),
  quizScore: document.getElementById('quizScore'),
  quizStreak: document.getElementById('quizStreak'),
  quizStatusBanner: document.getElementById('quizStatusBanner'),
  quizCanvasMirror: document.getElementById('quizCanvasMirror'),
  quizCurrentDetected: document.getElementById('quizCurrentDetected'),

  // Studio
  customLabelInput: document.getElementById('customLabelInput'),
  recordSamplesBtn: document.getElementById('recordSamplesBtn'),
  trainCustomBtn: document.getElementById('trainCustomBtn'),
  recordProgress: document.getElementById('recordProgress'),
  recordCountText: document.getElementById('recordCountText'),
  studioLog: document.getElementById('studioLog')
};

// Canvas context
const canvasCtx = elements.skeletonCanvas.getContext('2d');
const quizCanvasCtx = elements.quizCanvasMirror.getContext('2d');

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  initMediaPipe();
  setupEventListeners();
  fetchHealthCheck();
  loadDictionary();
});

// Navigation Handling
function setupNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
      });
      document.getElementById(tabId).classList.add('active');
      currentTab = tabId;
    });
  });
}

// MediaPipe Hands Initialization
function initMediaPipe() {
  handsDetector = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  handsDetector.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.65,
    minTrackingConfidence: 0.65
  });

  handsDetector.onResults(onHandResults);

  // Initialize camera stream
  cameraInstance = new Camera(elements.webcam, {
    onFrame: async () => {
      if (isCameraActive) {
        await handsDetector.send({ image: elements.webcam });
      }
    },
    width: 640,
    height: 480
  });

  cameraInstance.start().catch(err => {
    console.error("Camera access error:", err);
    elements.handDetectPill.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Camera Access Denied`;
  });
}

// Frame Processing & Landmark Drawing
function onHandResults(results) {
  // Update FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    elements.fpsCounter.textContent = frameCount;
    frameCount = 0;
    lastFrameTime = now;
  }

  // Clear Canvas
  const width = elements.webcam.videoWidth || 640;
  const height = elements.webcam.videoHeight || 480;
  elements.skeletonCanvas.width = width;
  elements.skeletonCanvas.height = height;
  elements.quizCanvasMirror.width = width;
  elements.quizCanvasMirror.height = height;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, width, height);

  quizCanvasCtx.save();
  quizCanvasCtx.clearRect(0, 0, width, height);
  if (results.image) {
    quizCanvasCtx.drawImage(results.image, 0, 0, width, height);
  }

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Draw Skeleton on Main Canvas
    drawSkeleton(canvasCtx, landmarks, width, height);
    drawSkeleton(quizCanvasCtx, landmarks, width, height);

    elements.handDetectPill.innerHTML = `<i class="fa-solid fa-hand"></i> Hand Detected`;
    elements.handDetectPill.style.borderColor = "rgba(16, 185, 129, 0.4)";

    // Send 21 keypoints to Python Backend
    sendPredictionRequest(landmarks);
    handWasDetected = true;

    // If Studio is recording, capture sample
    if (studioRecording && studioSamples.length < TARGET_STUDIO_SAMPLES) {
      studioSamples.push(landmarks);
      updateStudioProgress();
    }
  } else {
    elements.handDetectPill.innerHTML = `<i class="fa-solid fa-hand"></i> Waiting for Hand...`;
    elements.handDetectPill.style.borderColor = "rgba(255, 255, 255, 0.08)";
    if (handWasDetected) {
      resetTranslatorStability();
      handWasDetected = false;
    }
    resetPredictionOverlay();
  }

  canvasCtx.restore();
  quizCanvasCtx.restore();
}

function drawSkeleton(ctx, landmarks, width, height) {
  // Connectors
  if (typeof drawConnectors === 'function') {
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#06b6d4', lineWidth: 3 });
    drawLandmarks(ctx, landmarks, { color: '#10b981', lineWidth: 2, radius: 4 });
  } else {
    // Custom drawing fallback
    ctx.fillStyle = '#10b981';
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * width, lm.y * height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// API Calls to Python Backend
let lastPredictCallTime = 0;
async function sendPredictionRequest(landmarks) {
  // Throttle API calls to every 100ms for smooth performance
  const now = performance.now();
  if (now - lastPredictCallTime < 100) return;
  lastPredictCallTime = now;

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks: landmarks })
    });

    if (!response.ok) return;
    const data = await response.json();

    updateUIWithPrediction(data);

    // If Quiz is active, evaluate quiz logic
    if (quizActive) {
      evaluateQuizFrame(data.prediction, data.confidence);
    }
  } catch (err) {
    console.error("Prediction API error:", err);
    elements.statusDot.className = "status-dot red";
    elements.statusText.textContent = "Backend Connection Error";
  }
}

function updateUIWithPrediction(data) {
  const pred = data.prediction;
  const confPct = Math.round(data.confidence * 100);

  elements.detectedSign.textContent = pred;
  elements.confidenceValue.textContent = `${confPct}%`;
  elements.confidenceBar.style.width = `${confPct}%`;

  // Update sentence buffer
  if (data.sentence_buffer !== undefined) {
    if (data.sentence_buffer.trim() === "") {
      elements.sentenceBuffer.innerHTML = `<span class="placeholder-text">Signed letters will appear here...</span>`;
    } else {
      elements.sentenceBuffer.textContent = data.sentence_buffer;
    }
  }

  // Update Candidate breakdown
  if (data.top_candidates && data.top_candidates.length > 0) {
    elements.candidateList.innerHTML = data.top_candidates.map(c => `
      <div class="candidate-item">
        <span class="candidate-name">${c.label}</span>
        <span class="candidate-score">${Math.round(c.confidence * 100)}%</span>
      </div>
    `).join('');
  }
}

function resetPredictionOverlay() {
  elements.detectedSign.textContent = "--";
  elements.confidenceValue.textContent = "0%";
  elements.confidenceBar.style.width = "0%";
  elements.candidateList.innerHTML = `<div class="candidate-item empty">Waiting for hand detection...</div>`;
}

// Event Listeners Setup
function setupEventListeners() {
  // Toggle camera button
  elements.toggleCameraBtn.addEventListener('click', () => {
    isCameraActive = !isCameraActive;
    elements.toggleCameraBtn.innerHTML = isCameraActive 
      ? `<i class="fa-solid fa-video"></i> Pause Camera`
      : `<i class="fa-solid fa-video-slash"></i> Resume Camera`;
  });

  // Speech Text-to-Speech
  elements.speakBtn.addEventListener('click', speakCurrentSentence);
  elements.spaceBtn.addEventListener('click', () => callTranslatorAPI('space'));
  elements.backspaceBtn.addEventListener('click', () => callTranslatorAPI('backspace'));
  elements.clearBtn.addEventListener('click', () => callTranslatorAPI('clear'));

  // Dictionary Search & Filter
  elements.dictSearch.addEventListener('input', filterDictionary);
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterDictionary();
    });
  });

  // Quiz buttons
  elements.startQuizBtn.addEventListener('click', startQuizSession);
  elements.skipQuizBtn.addEventListener('click', fetchNextQuizQuestion);

  // Studio buttons
  elements.recordSamplesBtn.addEventListener('click', startRecordingStudioSamples);
  elements.trainCustomBtn.addEventListener('click', submitCustomTraining);
}

// Speech Synthesis
function speakCurrentSentence() {
  const text = elements.sentenceBuffer.textContent;
  if (!text || text.includes("Signed letters will appear")) return;

  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel(); // Stop prior audio
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  } else {
    alert("Speech Synthesis not supported in your browser.");
  }
}

async function callTranslatorAPI(action) {
  try {
    const res = await fetch(`${API_BASE}/translator/${action}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok || typeof data.sentence_buffer !== "string") {
      throw new Error(data.detail || `Translator request failed (${res.status})`);
    }
    if (data.sentence_buffer.trim() === "") {
      elements.sentenceBuffer.innerHTML = `<span class="placeholder-text">Signed letters will appear here...</span>`;
    } else {
      elements.sentenceBuffer.textContent = data.sentence_buffer;
    }
  } catch (err) {
    console.error("Translator API call failed:", err);
    elements.statusDot.className = "status-dot red";
    elements.statusText.textContent = "Translator Connection Error";
  }
}

async function resetTranslatorStability() {
  try {
    await fetch(`${API_BASE}/translator/reset`, { method: "POST" });
  } catch (err) {
    console.error("Translator stability reset failed:", err);
  }
}

// Health Check
async function fetchHealthCheck() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === "healthy") {
      elements.statusDot.className = "status-dot green";
      elements.statusText.textContent = `AI Ready (${data.classes_count} Signs Trained)`;
    }
  } catch (err) {
    elements.statusDot.className = "status-dot red";
    elements.statusText.textContent = "Backend Offline";
  }
}

// Dictionary Logic
let fullDictionary = {};
async function loadDictionary() {
  try {
    const res = await fetch(`${API_BASE}/dictionary`);
    const data = await res.json();
    fullDictionary = data.dictionary || {};
    renderDictionaryGrid(fullDictionary);
  } catch (err) {
    console.error("Failed to load ASL dictionary:", err);
  }
}

function renderDictionaryGrid(dict) {
  const keys = Object.keys(dict);
  if (keys.length === 0) {
    elements.dictGrid.innerHTML = `<p style="color: var(--text-muted);">No matching signs found.</p>`;
    return;
  }

  elements.dictGrid.innerHTML = keys.map(key => {
    const item = dict[key];
    return `
      <div class="dict-card glass-inset">
        <div class="dict-card-top">
          <span class="dict-sign-symbol">${key}</span>
          <span class="dict-category-tag">${item.category}</span>
        </div>
        <h4>${item.name}</h4>
        <p>${item.description}</p>
        <div class="dict-tip">
          <i class="fa-solid fa-lightbulb"></i> ${item.tips}
        </div>
      </div>
    `;
  }).join('');
}

function filterDictionary() {
  const searchTerm = elements.dictSearch.value.toLowerCase();
  const activeCategory = document.querySelector('.filter-btn.active').getAttribute('data-category');

  const filtered = {};
  for (const [key, item] of Object.entries(fullDictionary)) {
    const matchesSearch = key.toLowerCase().includes(searchTerm) || 
                          item.name.toLowerCase().includes(searchTerm) ||
                          item.description.toLowerCase().includes(searchTerm);
    const matchesCategory = activeCategory === "ALL" || item.category === activeCategory;

    if (matchesSearch && matchesCategory) {
      filtered[key] = item;
    }
  }
  renderDictionaryGrid(filtered);
}

// Quiz System Logic
async function startQuizSession() {
  quizActive = true;
  quizScore = 0;
  quizStreak = 0;
  elements.quizScore.textContent = "0";
  elements.quizStreak.textContent = "0";
  await fetchNextQuizQuestion();
}

async function fetchNextQuizQuestion() {
  try {
    const res = await fetch(`${API_BASE}/quiz/question`);
    const data = await res.json();
    currentQuizTarget = data.target_sign;
    elements.quizTargetSign.textContent = data.target_sign;
    elements.quizSignTips.textContent = `${data.name}: ${data.tips}`;
    elements.quizStatusBanner.className = "quiz-status-banner";
    elements.quizStatusBanner.innerHTML = `<i class="fa-solid fa-camera"></i> Show the sign for <strong>${data.target_sign}</strong>`;
  } catch (err) {
    console.error("Quiz fetch failed:", err);
  }
}

let lastQuizMatchTime = 0;
function evaluateQuizFrame(prediction, confidence) {
  elements.quizCurrentDetected.textContent = `${prediction} (${Math.round(confidence * 100)}%)`;

  if (!currentQuizTarget || !quizActive) return;

  if (prediction === currentQuizTarget && confidence >= 0.65) {
    const now = performance.now();
    if (now - lastQuizMatchTime > 2000) { // Require 2s between score increments
      lastQuizMatchTime = now;
      quizScore += 10;
      quizStreak += 1;
      elements.quizScore.textContent = quizScore;
      elements.quizStreak.textContent = quizStreak;

      elements.quizStatusBanner.className = "quiz-status-banner success";
      elements.quizStatusBanner.innerHTML = `<i class="fa-solid fa-circle-check"></i> EXCELLENT! Recognized <strong>${prediction}</strong> (+10 pts)`;
      
      // Auto move to next question after 1.2s
      setTimeout(fetchNextQuizQuestion, 1200);
    }
  }
}

// Studio Logic
function startRecordingStudioSamples() {
  const label = elements.customLabelInput.value.trim();
  if (!label) {
    alert("Please enter a gesture name (e.g. WATER, MY_SIGN) first.");
    return;
  }

  studioRecording = true;
  studioSamples = [];
  elements.recordProgress.style.width = "0%";
  elements.recordCountText.textContent = `0 / ${TARGET_STUDIO_SAMPLES} Samples`;
  elements.trainCustomBtn.disabled = true;
  
  logStudioConsole(`[STUDIO] Recording started for sign '${label.toUpperCase()}'. Hold hand gesture in camera feed...`);
}

function updateStudioProgress() {
  const count = studioSamples.length;
  const pct = Math.round((count / TARGET_STUDIO_SAMPLES) * 100);
  elements.recordProgress.style.width = `${pct}%`;
  elements.recordCountText.textContent = `${count} / ${TARGET_STUDIO_SAMPLES} Samples`;

  if (count >= TARGET_STUDIO_SAMPLES) {
    studioRecording = false;
    elements.trainCustomBtn.disabled = false;
    logStudioConsole(`[STUDIO] Successfully recorded ${TARGET_STUDIO_SAMPLES} samples! Click 'Retrain AI Classifier' to complete training.`);
  }
}

async function submitCustomTraining() {
  const label = elements.customLabelInput.value.trim().toUpperCase();
  if (!label || studioSamples.length === 0) return;

  logStudioConsole(`[AI ENGINE] Sending ${studioSamples.length} samples to Python backend for retraining...`);
  
  try {
    const res = await fetch(`${API_BASE}/train/custom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label: label,
        samples: studioSamples
      })
    });

    const data = await res.json();
    if (res.ok) {
      logStudioConsole(`[SUCCESS] ${data.message}`);
      fetchHealthCheck();
      loadDictionary();
      elements.customLabelInput.value = "";
      studioSamples = [];
      elements.trainCustomBtn.disabled = true;
    } else {
      logStudioConsole(`[ERROR] ${data.detail || "Training failed."}`);
    }
  } catch (err) {
    console.error("Studio training error:", err);
    logStudioConsole(`[ERROR] Network error during training.`);
  }
}

function logStudioConsole(msg) {
  elements.studioLog.innerHTML += `\n${msg}`;
  elements.studioLog.scrollTop = elements.studioLog.scrollHeight;
}
